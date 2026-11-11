#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

# node2
# DATA_PATH=/vol2/data/dataset
# BASE_DATA_PATH=/vol1/loujh/dataset
# DISKANN_INDEX_PATH=/vol2/data/diskann_index_yq
# # DISKANN_INDEX_PATH=/vol2/data/testbed/diskann_index_yq

DATA_PATH=/root/data/dataset
BASE_DATA_PATH=/root/data/dataset
DISKANN_INDEX_PATH=/root/data/diskann_index

OUTPUT_PATH=$DISKANN_INDEX_PATH
# Benchmark scratch copies (shadow / mem / merge); built indexes stay under OUTPUT_PATH.
DISKANN_WORK_ROOT=/root/testbed
GREATOR_ROOT="${GREATOR_ROOT:-$(cd "$PROJECT_PATH/../Greator" 2>/dev/null && pwd)}"
# Update rounds per overall_performance run (insert/delete batches before exit).
NUM_BATCHES="${NUM_BATCHES:-3}"

# 用法: set_dataset_params <data_name> [default_pq_dim]
set_dataset_params() {
    local data_name="$1"
    pq_dim="${2:-64}"
    DATASET_DIR="$data_name"
    vec_dim=128

    case "$data_name" in
        tiny5m)
            npts=5000000
            batch_npts=2500000
            ;;
        laion_text|laion_img)
            npts=10000000
            batch_npts=5000000
            ;;
        sift10m)
            npts=10000000
            batch_npts=5000000
            ;;
        deep10m)
            npts=10000000
            batch_npts=5000000
            pq_dim=48
            vec_dim=96
            ;;
        text2img)
            npts=100000000
            batch_npts=50000000
            pq_dim=100
            vec_dim=200
            ;;
        deep1b)
            npts=1000000000
            batch_npts=500000000
            pq_dim=48
            vec_dim=96
            ;;
        deep100m)
            npts=100000000
            batch_npts=50000000
            pq_dim=48
            vec_dim=96
            ;;
        laion)
            npts=100000000
            batch_npts=50000000
            pq_dim=256
            vec_dim=512
            ;;
        msmarco)
            npts=100000000
            batch_npts=50000000
            pq_dim=512
            vec_dim=1024
            ;;
        msmarco10m)
            npts=10000000
            batch_npts=5000000
            pq_dim=512
            vec_dim=1024
            ;;
        snowflake)
            npts=100000000
            batch_npts=50000000
            pq_dim=384
            vec_dim=768
            # DATASET_DIR=snowflake_need_norm
            ;;
        deep1m)
            npts=1000000
            batch_npts=500000
            pq_dim=48
            vec_dim=96
            ;;
        sift1m)
            npts=1000000
            batch_npts=500000
            vec_dim=128
            ;;
        *)
            npts=1000000
            batch_npts=500000
            ;;
    esac
}

set_search_width() {
    local mode="$1"
    case "$mode" in
        2|4|7) search_width=32 ;;
        1) search_width=4 ;;
        *)   search_width=4 ;;
    esac
}

is_valid_fbin() {
    local p="$1"
    [ -f "$p" ] || return 1
    python3 - "$p" <<'PY'
import struct, os, sys
path = sys.argv[1]
size = os.path.getsize(path)
if size < 8:
    sys.exit(1)
with open(path, "rb") as f:
    n, d = struct.unpack("ii", f.read(8))
if n <= 0 or d <= 0:
    sys.exit(1)
if size < 8 + n * d * 4:
    sys.exit(1)
PY
}

resolve_base_fbin() {
    local data_name="$1"
    local batch_npts="$2"
    local dir="${DATASET_DIR:-$data_name}"
    local candidates=(
        "${BASE_DATA_PATH}/${dir}/${data_name}_base.fbin"
        "${BASE_DATA_PATH}/${dir}/${data_name}_base.fbin${batch_npts}"
        "${DATA_PATH}/${dir}/${data_name}_base.fbin"
        "${DATA_PATH}/${dir}/${data_name}_base.fbin${batch_npts}"
    )
    local p
    for p in "${candidates[@]}"; do
        if is_valid_fbin "$p"; then
            echo "$p"
            return 0
        fi
    done
    echo "ERROR: valid base fbin not found for ${data_name} (tried ${candidates[*]})" >&2
    return 1
}

# Resolve batch-sized base for index build; run change_pts only when needed.
prepare_batch_data() {
    local data_type="$1"
    local full_data="$2"
    local batch_npts="$3"
    local change_pts_bin="$4"
    local data

    if [[ "$full_data" == *"${batch_npts}" ]]; then
        data="$full_data"
    else
        data="${full_data}${batch_npts}"
        if [ ! -f "$data" ]; then
            echo "=== Truncating base to ${batch_npts} pts: ${full_data} -> ${data} ===" >&2
            "$change_pts_bin" "$data_type" "$full_data" "$batch_npts"
        fi
    fi

    if ! is_valid_fbin "$data"; then
        echo "ERROR: invalid or truncated fbin for build: $data" >&2
        python3 - "$data" <<'PY' >&2 || true
import struct, os, sys
path = sys.argv[1]
size = os.path.getsize(path) if os.path.exists(path) else 0
if size >= 8:
    with open(path, "rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    print(f"  header: npts={n}, dim={d}, file_size={size} bytes", file=sys.stderr)
else:
    print(f"  file missing or too small ({size} bytes)", file=sys.stderr)
PY
        return 1
    fi
    echo "$data"
}

resolve_query_fbin() {
    local data_name="$1"
    local dir="${DATASET_DIR:-$data_name}"
    local candidates=(
        "${BASE_DATA_PATH}/${dir}/${data_name}_query.fbin"
        "${DATA_PATH}/${dir}/${data_name}_query.fbin"
    )
    local p
    for p in "${candidates[@]}"; do
        if [ -f "$p" ]; then
            echo "$p"
            return 0
        fi
    done
    echo "ERROR: query fbin not found for ${data_name}" >&2
    return 1
}

# Identity tags [0..npts-1] in DiskANN fbin format (8 + npts*4 bytes).
disk_index_tags_valid() {
    local tags_file="$1"
    local npts="$2"
    local expected=$((8 + npts * 4))
    local size

    [ -f "$tags_file" ] || return 1
    size=$(stat -c%s "$tags_file" 2>/dev/null) || return 1
    [ "$size" -eq "$expected" ]
}

# Build tags once on the canonical index prefix; reuse when size matches npts.
ensure_disk_index_tags() {
    local prefix="$1"
    local npts="${2:-$batch_npts}"
    local tags_file="${prefix}_disk.index.tags"

    if [ "${FORCE_REGENERATE_SIDECARS:-0}" != "1" ] && disk_index_tags_valid "$tags_file" "$npts"; then
        echo "=== Reusing tags: ${tags_file} (npts=${npts}) ==="
        return 0
    fi

    rm -f "$tags_file"
    echo "=== Generating tags: ${tags_file} (npts=${npts}) ==="
    python3 - "$tags_file" "$npts" <<'PY'
import struct, sys
path, n = sys.argv[1], int(sys.argv[2])
with open(path, "wb") as f:
    f.write(struct.pack("ii", n, 1))
    for i in range(n):
        f.write(struct.pack("i", i))
print(f"Wrote tags: n={n}, dims=1")
PY
}

# Topology sidecar for Greator merge; rebuild only when missing or older than _disk.index.
disk_index_topology_valid() {
    local topo_out="$1"
    local min_size="${2:-4096}"
    local size

    [ -f "$topo_out" ] || return 1
    size=$(stat -c%s "$topo_out" 2>/dev/null) || return 1
    [ "$size" -ge "$min_size" ]
}

ensure_disk_index_topology() {
    local prefix="$1"
    local disk_index="${prefix}_disk.index"
    local topo_out="${prefix}_disk.index_with_only_nbrs"
    local tool_dir="${GREATOR_ROOT}/scripts/tools"
    local te_bin="${tool_dir}/te"

    if [ ! -f "$disk_index" ]; then
        echo "ERROR: missing ${disk_index} (cannot extract topology)" >&2
        return 1
    fi
    if [ ! -d "$tool_dir" ] || [ ! -f "${tool_dir}/topology_extraction.cpp" ]; then
        echo "ERROR: Greator topology tool not found under ${tool_dir}" >&2
        return 1
    fi

    if [ "${FORCE_REGENERATE_SIDECARS:-0}" != "1" ] && disk_index_topology_valid "$topo_out" && [ "$topo_out" -nt "$disk_index" ]; then
        echo "=== Reusing topology: ${topo_out} ==="
        return 0
    fi

    rm -f "$topo_out"
    echo "=== Generating topology: ${topo_out} ==="
    if [ ! -x "$te_bin" ] || [ "${tool_dir}/topology_extraction.cpp" -nt "$te_bin" ]; then
        g++ -O2 "${tool_dir}/topology_extraction.cpp" -o "$te_bin" || return 1
    fi
    "$te_bin" "$disk_index" "$topo_out" 4096
}

ensure_greator_index_sidecars() {
    local prefix="$1"
    local npts="${2:-$batch_npts}"
    ensure_disk_index_tags "$prefix" "$npts" || return 1
    ensure_disk_index_topology "$prefix" || return 1
}

# op.sh: sidecars must already exist (generated by build_disk_index.sh).
require_greator_index_sidecars() {
    local prefix="$1"
    local npts="${2:-$batch_npts}"
    local tags_file="${prefix}_disk.index.tags"
    local topo_out="${prefix}_disk.index_with_only_nbrs"

    if ! disk_index_tags_valid "$tags_file" "$npts"; then
        echo "ERROR: missing or invalid ${tags_file} (npts=${npts}); run build_disk_index.sh first" >&2
        return 1
    fi
    if ! disk_index_topology_valid "$topo_out"; then
        echo "ERROR: missing or invalid ${topo_out}; run build_disk_index.sh first" >&2
        return 1
    fi
    return 0
}

copy_shadow_index() {
    local prefix="$1"
    local search_mode="$2"
    local work_dir
    work_dir="$(dirname "$prefix")"
    local out="${prefix}_shadow"

    _cp_idx() {
        if [ -f "$1" ]; then
            cp -f "$1" "$2" || {
                echo "ERROR: failed to copy $1 to $2" >&2
                return 1
            }
        else
            echo "ERROR: missing required index file $1" >&2
            return 1
        fi
    }

    mkdir -p "$work_dir" || return 1
    if [ ! -w "$work_dir" ]; then
        echo "ERROR: shadow directory is not writable: ${work_dir}" >&2
        return 1
    fi
    if command -v findmnt >/dev/null 2>&1 && findmnt -T "$work_dir" -n -o OPTIONS | tr ',' '\n' | grep -qx ro; then
        echo "ERROR: shadow filesystem is read-only: ${work_dir}" >&2
        return 1
    fi
    echo "=== Shadow copy (mode=${search_mode}): ${prefix} -> ${out}_* ==="

    _cp_idx "${prefix}_pq_compressed.bin" "${out}_pq_compressed.bin" || return 1
    _cp_idx "${prefix}_pq_pivots.bin" "${out}_pq_pivots.bin" || return 1

    case "$search_mode" in
        4|5|6)
            _cp_idx "${prefix}_disk.graph" "${out}_disk.graph" || return 1
            _cp_idx "${prefix}_disk.vector" "${out}_disk.vector" || return 1
            _cp_idx "${prefix}_disk.index.meta" "${out}_disk.index.meta" || return 1
            _cp_idx "${prefix}_partition.bin.dgai" "${out}_partition.bin.dgai" || return 1
            ;;
        7)
            # STARLING_DECOUPLE_SEARCH: shared vector/meta + starling.decouple graph layout.
            _cp_idx "${prefix}_disk.vector" "${out}_disk.vector" || return 1
            _cp_idx "${prefix}_disk.index.meta" "${out}_disk.index.meta" || return 1
            _cp_idx "${prefix}_disk.graph.starling.decouple" "${out}_disk.graph.starling.decouple" || return 1
            _cp_idx "${prefix}_partition.bin.starling.decouple" "${out}_partition.bin.starling.decouple" || return 1
            ;;
        1)
            # load() peeks nnodes_per_sector from _disk.index, then opens .starling when >1.
            _cp_idx "${prefix}_disk.index" "${out}_disk.index" || return 1
            if [ -f "${prefix}_disk.index.starling" ]; then
                _cp_idx "${prefix}_disk.index.starling" "${out}_disk.index.starling" || return 1
            fi
            _cp_idx "${prefix}_partition.bin.starling" "${out}_partition.bin.starling" || return 1
            ;;
        *)
            _cp_idx "${prefix}_disk.index" "${out}_disk.index" || return 1
            ;;
    esac

    if [ -f "${out}_disk.index.tags" ]; then
        rm -f "${out}_disk.index.tags" || return 1
        echo "Removed ${out}_disk.index.tags (shadow uses equal id mapping)"
    fi

    local f
    for f in "${out}"_*; do
        if [ -f "$f" ]; then
            chmod u+rw "$f" || return 1
        fi
    done

    index_prefix_shadow="$out"
}

# Greator op.sh: require sidecars on canonical index, always cp them to fresh shadow.
copy_greator_shadow_index() {
    local prefix="$1"
    local search_mode="${2:-0}"
    local out="${prefix}_shadow"

    require_greator_index_sidecars "$prefix" || return 1
    rm -rf "${out}_temp" "${out}"temp* || return 1

    copy_shadow_index "$prefix" "$search_mode" || return 1
    out="${index_prefix_shadow}"

    cp -f "${prefix}_disk.index.tags" "${out}_disk.index.tags" || return 1
    cp -f "${prefix}_disk.index_with_only_nbrs" "${out}_disk.index_with_only_nbrs" || return 1

    index_prefix_shadow="$out"
}

index_dir_path() {
    echo "${OUTPUT_PATH}/${data_name}/original_index_${batch_npts}_R${R:-32}_L${Lbuild:-75}_pq${pq_dim}_${metric:-l2}"
}

index_prefix_path() {
    echo "$(index_dir_path)/${data_name}"
}

diskann_work_index_dir() {
    echo "${DISKANN_WORK_ROOT}/${data_name}/original_index_${batch_npts}_R${R:-32}_L${Lbuild:-75}_pq${pq_dim}_${metric:-l2}"
}

diskann_work_prefix() {
    echo "$(diskann_work_index_dir)/${data_name}"
}

# Canonical index on testbed NVMe (read-only search).
benchmark_index_prefix() {
    index_prefix_path
}

truthset_prefix_path() {
    echo "${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_${UPDATE_RATIO:-0.001}"
}

# Supported distance metrics for search/op benchmarks.
validate_metric() {
    case "$1" in
        l2|cosine) return 0 ;;
        *)
            echo "ERROR: unsupported metric '$1' (use l2 or cosine)" >&2
            return 1
            ;;
    esac
}

# Per-round delete+insert count: batch_npts * UPDATE_RATIO (e.g. 5000 for deep10m @ 0.001).
compute_update_size() {
    local ratio="${UPDATE_RATIO:-0.001}"
    awk -v n="$batch_npts" -v r="$ratio" 'BEGIN { printf "%d", int(n * r) }'
}
