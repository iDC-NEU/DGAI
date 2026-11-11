#!/bin/bash
# Prepare missing Starling partitions and Greator sidecars before experiments.
#
# Starling (mode1 search):
#   ${prefix}_disk.index.starling
#   ${prefix}_partition.bin.starling
#   via starling_partition/scripts/gen_partition.sh
#
# Greator update:
#   ${prefix}_disk.index.tags
#   ${prefix}_disk.index_with_only_nbrs
#   via ensure_greator_index_sidecars in DGAI env.sh
#
# Skip datasets with no base _disk.index (e.g. deep1m).
# Override PREP_DATASETS / SKIP_STARLING / SKIP_GREATOR / FORCE_STARLING.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

L_DISK="${L_DISK:-75}"
PREP_DATASETS="${PREP_DATASETS:-text2img laion msmarco deep1b deep10m deep100m}"
SKIP_STARLING="${SKIP_STARLING:-0}"
SKIP_GREATOR="${SKIP_GREATOR:-0}"
FORCE_STARLING="${FORCE_STARLING:-0}"
# deep1b Starling is expensive; skip by default (override with STARLING_SKIP_DATASETS="")
STARLING_SKIP_DATASETS="${STARLING_SKIP_DATASETS:-deep1b msmarco}"
STARLING_ROOT="${STARLING_ROOT:-$(cd "$DGAI_ROOT/../starling_partition" 2>/dev/null && pwd)}"
GEN_PARTITION="${GEN_PARTITION:-$STARLING_ROOT/scripts/gen_partition.sh}"
PARTITIONER="${PARTITIONER:-$STARLING_ROOT/build/partitioner}"
STARLING_THREADS="${STARLING_THREADS:-64}"
STARLING_GP_TIMES="${STARLING_GP_TIMES:-16}"

echo_exp_banner "prep_missing_indexes" \
    "datasets=[${PREP_DATASETS}] starling_root=${STARLING_ROOT:-MISSING}" \
    "SKIP_STARLING=${SKIP_STARLING} SKIP_GREATOR=${SKIP_GREATOR} FORCE_STARLING=${FORCE_STARLING}" \
    "STARLING_SKIP_DATASETS=[${STARLING_SKIP_DATASETS}]"

need_starling() {
    local prefix="$1"
    if [ "$FORCE_STARLING" = "1" ]; then
        return 0
    fi
    [ ! -f "${prefix}_disk.index.starling" ] || [ ! -f "${prefix}_partition.bin.starling" ]
}

need_greator() {
    local prefix="$1"
    local npts="$2"
    if ! disk_index_tags_valid "${prefix}_disk.index.tags" "$npts"; then
        return 0
    fi
    if ! disk_index_topology_valid "${prefix}_disk.index_with_only_nbrs"; then
        return 0
    fi
    return 1
}

ensure_partitioner() {
    if [ ! -x "$PARTITIONER" ]; then
        echo "=== Building starling partitioner ==="
        mkdir -p "$STARLING_ROOT/build"
        cmake -S "$STARLING_ROOT" -B "$STARLING_ROOT/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "$STARLING_ROOT/build" --target partitioner -j"$(nproc)"
    fi
    if [ ! -x "$PARTITIONER" ]; then
        echo "ERROR: partitioner not found at $PARTITIONER" >&2
        return 1
    fi
}

starling_dataset_skipped() {
    local data_name="$1"
    local s
    for s in $STARLING_SKIP_DATASETS; do
        if [ "$s" = "$data_name" ]; then
            return 0
        fi
    done
    return 1
}

run_starling_for_prefix() {
    local prefix="$1"
    local data_name="$2"

    if starling_dataset_skipped "$data_name"; then
        echo "SKIP starling ${data_name}: in STARLING_SKIP_DATASETS"
        return 0
    fi
    if [ ! -f "${prefix}_disk.index" ]; then
        echo "SKIP starling ${data_name}: missing ${prefix}_disk.index"
        return 0
    fi
    if ! need_starling "$prefix"; then
        echo "OK starling ${data_name}: already present"
        return 0
    fi

    echo "=== Building Starling partition for ${data_name} ==="
    echo "    INDEX_PREFIX=${prefix}"
    ensure_partitioner

    # gen_partition.sh reads INDEX_PREFIX / FORCE / THREADS from env.
    INDEX_PREFIX="$prefix" \
    DATA_TYPE=float \
    FORCE="$FORCE_STARLING" \
    THREADS="$STARLING_THREADS" \
    GP_TIMES="$STARLING_GP_TIMES" \
    PARTITIONER="$PARTITIONER" \
    bash "$GEN_PARTITION"

    if [ ! -f "${prefix}_disk.index.starling" ] || [ ! -f "${prefix}_partition.bin.starling" ]; then
        echo "ERROR: starling outputs missing after gen_partition for ${data_name}" >&2
        return 1
    fi
    echo "OK starling ${data_name}"
}

run_greator_for_prefix() {
    local prefix="$1"
    local data_name="$2"
    local npts="$3"

    if [ ! -f "${prefix}_disk.index" ]; then
        echo "SKIP greator ${data_name}: missing ${prefix}_disk.index"
        return 0
    fi
    if ! need_greator "$prefix" "$npts"; then
        echo "OK greator ${data_name}: tags+topology already present"
        return 0
    fi

    echo "=== Building Greator sidecars for ${data_name} (npts=${npts}) ==="
    # ensure_greator uses GREATOR_ROOT for topology tool
    export GREATOR_ROOT
    ensure_greator_index_sidecars "$prefix" "$npts" || {
        echo "ERROR: greator sidecars failed for ${data_name}" >&2
        return 1
    }
    echo "OK greator ${data_name}"
}

if [ -z "${STARLING_ROOT}" ] || [ ! -f "$GEN_PARTITION" ]; then
    echo "ERROR: starling_partition not found (STARLING_ROOT=${STARLING_ROOT:-})" >&2
    echo "       expected gen_partition at $GEN_PARTITION" >&2
    if [ "$SKIP_STARLING" != "1" ]; then
        exit 1
    fi
fi

for data_name in $PREP_DATASETS; do
    set_dataset_params "$data_name"
    R=32
    Lbuild=$L_DISK
    prefix=$(benchmark_index_prefix)

    echo ""
    echo "########## prep: ${data_name} ##########"
    echo "prefix: ${prefix}"

    if [ ! -f "${prefix}_disk.index" ]; then
        echo "WARN: no base disk index — skipping all prep for ${data_name}"
        continue
    fi

    if [ "$SKIP_GREATOR" != "1" ]; then
        run_greator_for_prefix "$prefix" "$data_name" "$batch_npts" || {
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        }
    fi

    if [ "$SKIP_STARLING" != "1" ]; then
        run_starling_for_prefix "$prefix" "$data_name" || {
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        }
    fi
done

echo ""
echo "=== Done: prep_missing_indexes ==="
