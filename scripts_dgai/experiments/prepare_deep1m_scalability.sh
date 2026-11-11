#!/bin/bash
# Prepare deep1m inputs and indexes for the dataset-scalability experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

data_name=deep1m
data_type=float
metric=l2
topk=10000
target_topk=10
ratio="${UPDATE_SCALE_RATIO:-0.001}"
gt_threads="${GT_THREADS:-64}"
index_threads="${INDEX_THREADS:-100}"
force="${FORCE_DEEP1M_PREP:-0}"

set_dataset_params "$data_name"
R=32
Lbuild=$L_DISK

deep_dir="${DATA_PATH}/${data_name}"
mkdir -p "$deep_dir"

source_full="${DATA_PATH}/deep10m/deep10m_base.fbin"
query_source="${DATA_PATH}/deep1b/deep1b_query.fbin"
deep_full="${deep_dir}/deep1m_base.fbin"
deep_batch="${deep_dir}/deep1m_base.fbin${batch_npts}"
deep_query="${deep_dir}/deep1m_query.fbin"
gt_file="${deep_dir}/deep1m_gt_K${topk}"
truthset_prefix="${deep_dir}/deep1m_${batch_npts}_gt_${ratio}"
prefix="$(benchmark_index_prefix)"
index_dir="$(dirname "$prefix")"

echo "=== Preparing deep1m scalability artifacts ==="
echo "data dir:       $deep_dir"
echo "index prefix:   $prefix"
echo "ratio:          $ratio"
echo "gt_threads:     $gt_threads"
echo "index_threads:  $index_threads"
echo "force:          $force"

if [ ! -f "$source_full" ]; then
    echo "ERROR: source base missing: $source_full" >&2
    exit 1
fi
if [ ! -f "$query_source" ]; then
    echo "ERROR: query source missing: $query_source" >&2
    exit 1
fi

if [ "$force" = "1" ] || ! is_valid_fbin "$deep_full"; then
    tmp_src="${deep_dir}/deep1m_base_source.fbin"
    echo "=== Creating full deep1m base: $deep_full ==="
    rm -f "$tmp_src" "${tmp_src}${npts}"
    ln -s "$source_full" "$tmp_src"
    "$DGAI_ROOT/build/tests/utils/change_pts" "$data_type" "$tmp_src" "$npts"
    mv -f "${tmp_src}${npts}" "$deep_full"
    rm -f "$tmp_src"
else
    echo "=== Reusing valid full base: $deep_full ==="
fi

if [ "$force" = "1" ] || ! is_valid_fbin "$deep_batch"; then
    echo "=== Creating initial deep1m batch: $deep_batch ==="
    "$DGAI_ROOT/build/tests/utils/change_pts" "$data_type" "$deep_full" "$batch_npts"
else
    echo "=== Reusing valid batch base: $deep_batch ==="
fi

if [ "$force" = "1" ] || ! is_valid_fbin "$deep_query"; then
    echo "=== Copying query file: $deep_query ==="
    cp -f "$query_source" "$deep_query"
else
    echo "=== Reusing valid query: $deep_query ==="
fi

if [ "$force" = "1" ] || [ ! -s "$gt_file" ] || [ "$(stat -c%s "$gt_file" 2>/dev/null || echo 0)" -lt 100000000 ]; then
    echo "=== Computing deep1m groundtruth K=${topk}: $gt_file ==="
    "$DGAI_ROOT/build/tests/utils/compute_groundtruth" \
        "$data_type" "$metric" "$deep_full" "$deep_query" "$topk" \
        "$gt_file" \
        "null" "null" "null" "null" "null" "$gt_threads"
else
    echo "=== Reusing full groundtruth: $gt_file ==="
fi

if [ "$force" = "1" ] || [ ! -s "${truthset_prefix}/gt_0.bin" ] || [ "$(stat -c%s "${truthset_prefix}/gt_0.bin" 2>/dev/null || echo 0)" -le 8 ]; then
    update_size=$(awk -v n="$batch_npts" -v r="$ratio" 'BEGIN { printf "%d", int(n * r) }')
    echo "=== Generating update truthsets: ${truthset_prefix} (update_size=${update_size}) ==="
    "$DGAI_ROOT/build/tests/utils/gt_update" \
        "$gt_file" "$batch_npts" "$npts" "$update_size" "$target_topk" "$truthset_prefix" 0
else
    echo "=== Reusing update truthsets: $truthset_prefix ==="
fi

mkdir -p "$index_dir"
if [ "$force" = "1" ] || [ ! -s "${prefix}_disk.index" ] || [ ! -s "${prefix}_partition.bin.dgai" ]; then
    echo "=== Building disk index: $prefix ==="
    "$DGAI_ROOT/build/tests/build_disk_index" \
        "$data_type" \
        "$deep_batch" \
        "$prefix" \
        "$R" \
        "$Lbuild" \
        "$pq_dim" \
        300 \
        "$index_threads" \
        "$metric" \
        pq
else
    echo "=== Reusing disk index and DGAI partition: $prefix ==="
fi

base_sample_rate=0.01
full_sample_prefix="${prefix}_SAMPLE_RATE_${base_sample_rate}"
if [ "$force" = "1" ] || [ ! -s "${prefix}_mem.index" ] || [ ! -s "${prefix}_mem.index.data" ]; then
    echo "=== Building nav graph: ${prefix}_mem.index ==="
    "$DGAI_ROOT/build/tests/utils/gen_random_slice" "$data_type" "$deep_batch" "$full_sample_prefix" "$base_sample_rate"
    "$DGAI_ROOT/build/tests/build_memory_index" \
        "$data_type" "${full_sample_prefix}_data.bin" "${full_sample_prefix}_ids.bin" \
        "${prefix}_mem.index" "$R" 64 1.2 "$(nproc)" "$metric"
else
    echo "=== Reusing nav graph: ${prefix}_mem.index ==="
fi

topo_sample_rate=$(python3 -c "R=${R}; d=${vec_dim}; b=4; g=R*4*1.3; print(min(1.0, 0.01*((g+d*b)/g)))")
topo_sample_prefix="${prefix}_SAMPLE_RATE_${topo_sample_rate}"
if [ "$force" = "1" ] || [ ! -s "${prefix}_mem_topo.index" ] || [ ! -s "${prefix}_mem_topo.index.data" ]; then
    echo "=== Building decouple nav graph: ${prefix}_mem_topo.index (sample_rate=${topo_sample_rate}) ==="
    "$DGAI_ROOT/build/tests/utils/gen_random_slice" "$data_type" "$deep_batch" "$topo_sample_prefix" "$topo_sample_rate"
    "$DGAI_ROOT/build/tests/build_memory_index" \
        "$data_type" "${topo_sample_prefix}_data.bin" "${topo_sample_prefix}_ids.bin" \
        "${prefix}_mem_topo.index" "$R" 64 1.2 "$(nproc)" "$metric"
else
    echo "=== Reusing decouple nav graph: ${prefix}_mem_topo.index ==="
fi

if [ "$force" = "1" ] || [ ! -s "${prefix}_disk.graph" ] || [ ! -s "${prefix}_disk.vector" ] || [ ! -s "${prefix}_disk.index.meta" ]; then
    if [ ! -s "${prefix}_partition.bin.dgai" ]; then
        echo "ERROR: missing DGAI partition: ${prefix}_partition.bin.dgai" >&2
        exit 1
    fi
    echo "=== Reordering disk index by DGAI partition ==="
    "$DGAI_ROOT/build/tests/utils/reorder_disk_index" "$data_type" "$prefix"
    echo "=== Splitting reordered disk index into graph/vector ==="
    "$DGAI_ROOT/build/tests/utils/split_disk_index" "$data_type" "$prefix"
else
    echo "=== Reusing split DGAI disk files ==="
fi

echo "=== Preparing Greator/Starling sidecars for deep1m ==="
PREP_DATASETS=deep1m \
SKIP_GREATOR=0 \
SKIP_STARLING=0 \
STARLING_SKIP_DATASETS="" \
CONTINUE_ON_ERROR=0 \
bash "$SCRIPT_DIR/prep_missing_indexes.sh"

echo "=== deep1m scalability artifacts ready ==="
