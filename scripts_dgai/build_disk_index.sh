#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/env.sh"

# ==============================================================
# 构建项目
# ==============================================================
cd "$PROJECT_PATH/build"
cmake .. >/dev/null
make -j"$(nproc)"
cd "$PROJECT_PATH"

# ==============================================================
# 配置
# ==============================================================
data_type=float
metric=cosine
topk=10000
target_topk=10
UPDATE_RATIO=0.001
GT_THREADS=100

# text2img laion deep100m deep1b msmarco deep10m deep1m
for data_name in snowflake; do

    set_dataset_params "$data_name"
    full_data=$(resolve_base_fbin "$data_name" "$batch_npts")
    query=$(resolve_query_fbin "$data_name")

    gt=${DATA_PATH}/${data_name}/${data_name}_gt_K${topk}
    # gt=${BASE_DATA_PATH}/${data_name}/${data_name}_gt_K${topk}

    R=32
    M=300
    Lbuild=75
    nbr_type=pq
    original_index_path=${DISKANN_INDEX_PATH}/$data_name/original_index_${batch_npts}_R${R}_L${Lbuild}_${nbr_type}${pq_dim}_${metric}
    index_path_prefix=${original_index_path}/${data_name}

    # if [ ! -f "$gt" ]; then
        "$PROJECT_PATH"/build/tests/utils/compute_groundtruth "$data_type" "$metric" "$full_data" "$query" "$topk" \
            "$gt" \
            "null" "null" "null" "null" "null" "$GT_THREADS"
    # else
    #     echo "=== Reusing groundtruth: $gt ==="
    # fi

    for update_ratio in "$UPDATE_RATIO"; do
        update_size=$(echo "$batch_npts * $update_ratio" | bc -l | awk '{print int($1)}')
        echo "=== $data_name: update_ratio=$update_ratio, update_size=$update_size ==="

        truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_${update_ratio}
        if [ ! -d "$truthset_prefix" ]; then
            "$PROJECT_PATH"/build/tests/utils/gt_update \
                "$gt" "$batch_npts" "$npts" "$update_size" "$target_topk" "$truthset_prefix" 0
        else
            echo "=== Reusing truthset: $truthset_prefix ==="
        fi
    done

    # exit 0

    data=$(prepare_batch_data "$data_type" "$full_data" "$batch_npts" \
        "$PROJECT_PATH/build/tests/utils/change_pts") || exit 1
    echo "=== build data: $data ==="

    mkdir -p "$original_index_path"

    cmd="$PROJECT_PATH/build/tests/build_disk_index \
        ${data_type} \
        ${data} \
        ${index_path_prefix} \
        ${R} \
        ${Lbuild} \
        ${pq_dim} \
        ${M} \
        100 \
        ${metric} \
        ${nbr_type}"

    echo "BUILD cmd:"
    echo "$cmd"
    eval "$cmd"

    base_sample_rate=0.01
    data_dim=${vec_dim}
    dtype_bytes=4
    rate_scale=$(python3 -c "R=${R}; d=${data_dim}; b=${dtype_bytes}; g=R*4*1.3; print((g+d*b)/g)")
    topo_sample_rate=$(python3 -c "print(min(1.0, ${base_sample_rate}*${rate_scale}))")

    full_sample_prefix=${index_path_prefix}_SAMPLE_RATE_${base_sample_rate}
    echo "=== Building nav graph (full): sample_rate=${base_sample_rate} -> ${index_path_prefix}_mem.index ==="
    "$PROJECT_PATH/build/tests/utils/gen_random_slice" \
        "$data_type" "$data" "$full_sample_prefix" "$base_sample_rate"
    "$PROJECT_PATH/build/tests/build_memory_index" \
        "$data_type" "${full_sample_prefix}_data.bin" "${full_sample_prefix}_ids.bin" \
        "${index_path_prefix}_mem.index" \
        ${R} 64 1.2 $(nproc) "$metric"

    topo_sample_prefix=${index_path_prefix}_SAMPLE_RATE_${topo_sample_rate}
    echo "=== Building nav graph (decouple): sample_rate=${topo_sample_rate} -> ${index_path_prefix}_mem_topo.index ==="
    "$PROJECT_PATH/build/tests/utils/gen_random_slice" \
        "$data_type" "$data" "$topo_sample_prefix" "$topo_sample_rate"
    "$PROJECT_PATH/build/tests/build_memory_index" \
        "$data_type" "${topo_sample_prefix}_data.bin" "${topo_sample_prefix}_ids.bin" \
        "${index_path_prefix}_mem_topo.index" \
        ${R} 64 1.2 $(nproc) "$metric"

    if [ -f "${index_path_prefix}_partition.bin.dgai" ]; then
        echo "=== Reordering disk index by partition mapping ==="
        "$PROJECT_PATH/build/tests/utils/reorder_disk_index" \
            "$data_type" "${index_path_prefix}"

        echo "=== Splitting into graph + vector (_disk.index.dgai + _disk.index) ==="
        "$PROJECT_PATH/build/tests/utils/split_disk_index" \
            "$data_type" "${index_path_prefix}"

        echo "=== Reorder + Split done ==="
    fi

    echo "Cleaning intermediate files in $original_index_path ..."
    rm -f "${index_path_prefix}_part_tmp.index" \
          "${index_path_prefix}_disk.index.dgai" \
          "${index_path_prefix}"*.ivf* \
          "${index_path_prefix}"*.log

    echo "=== Copying index to testbed: ${DISKANN_INDEX_PATH}/${data_name} -> /root/testbed/${data_name} ==="
    mkdir -p "/root/testbed/${data_name}"
    cp -a "${DISKANN_INDEX_PATH}/${data_name}/." "/root/testbed/${data_name}/"

done
