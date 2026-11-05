#!/bin/bash

source "$(dirname "$0")/env.sh"

# ==============================================================
# 构建项目
# ==============================================================
cd "$PROJECT_PATH/build"
cmake ..
make -j
cd "$PROJECT_PATH"

# ==============================================================
# 配置
# ==============================================================
data_type=float
metric=l2
topk=10000
target_topk=100

for data_name in msmarco ; do

    query=${DATA_PATH}/${data_name}/${data_name}_query.fbin

    set_dataset_params "$data_name"
    # full_data=${DATA_PATH}/${data_name}/${data_name}_base.fbin${batch_npts}
    full_data=${BASE_DATA_PATH}/${data_name}/${data_name}_base.fbin${batch_npts}

    gt=${DATA_PATH}/${data_name}/${data_name}_gt_K${topk}
    
    # # b) 从头计算 ground truth
    # "$PROJECT_PATH"/build/tests/utils/compute_groundtruth "$data_type" "$metric" "$full_data" "$query" "$topk" \
    #     "$gt" \
    #     "null" "null" "null" "null" "null"

    # ##==============================================================
    # #2. 生成增量 ground truth（支持多种 update_ratio）
    # ##==============================================================

    # for update_ratio in 0.1; do
    #     update_size=$(echo "$batch_npts * $update_ratio" | bc -l | awk '{print int($1)}')
    #     echo "=== $data_name: update_ratio=$update_ratio, update_size=$update_size ==="

    #     truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_${update_ratio}
    #     "$PROJECT_PATH"/build/tests/utils/gt_update \
    #         "$gt" "$batch_npts" "$npts" "$update_size" "$target_topk" "$truthset_prefix" 0
    # done

    # exit 0

    # ==============================================================
    # 1. 数据预处理（按需取消注释）
    # ==============================================================
    # a) 截断原始数据到前 batch_npts 条
    # "$PROJECT_PATH"/build/tests/utils/change_pts "$data_type" "$full_data" "$batch_npts"
    
    R=32
    M=300
    Lbuild=75
    nbr_type=pq
    max_thread=80
    # ==============================================================
    # 3. 构建 Disk Index
    # ==============================================================
    original_index_path=${DISKANN_INDEX_PATH}/$data_name/original_index_${batch_npts}_R${R}_L${Lbuild}_${nbr_type}${pq_dim}_${metric}_justpq
    mkdir -p "$original_index_path"
    data=${full_data}
    index_path_prefix=${original_index_path}/${data_name}

    cmd="$PROJECT_PATH/build/tests/build_disk_index \
        ${data_type} \
        ${data} \
        ${index_path_prefix} \
        ${R} \
        ${Lbuild} \
        ${pq_dim} \
        ${M} \
        ${max_thread} \
        ${metric} \
        ${nbr_type}"

    echo "BUILD cmd:"
    echo "$cmd"
    eval "$cmd"
done