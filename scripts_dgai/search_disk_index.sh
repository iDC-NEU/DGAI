#!/bin/bash

source "$(dirname "$0")/env.sh"

# ==============================================================
# 构建项目
# ==============================================================
cd "$PROJECT_PATH/build"
export ADDITIONAL_DEFINITIONS=" -DREAD_ONLY_TESTS"
# export ADDITIONAL_DEFINITIONS=" "
cmake ..
make -j
cd "$PROJECT_PATH" 

# ==============================================================
# 配置
# ==============================================================
K=10
num_threads=8
search_mode=0         # 0=beam, 1=page, 2=pipe, 4=decouple, 5=decouple_beam, 6=decouple_page, 7=starling_decouple
mem_L=10
log_path=$PROJECT_PATH/logs
mkdir -p "$log_path"

for search_mode in 0 2 4; do
for data_name in msmarco; do

    set_dataset_params "$data_name"
    set_search_width "$search_mode"

    index_prefix=$(benchmark_index_prefix)
    query_file=${DATA_PATH}/${data_name}/${data_name}_query.fbin
    truthset_bin=${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_0.001/gt_0.bin
    # truthset_bin=/vol2/data/dataset/text2img/gt_K100.bin
    cur_mem_L=$mem_L
    if [ "$search_mode" -ge 4 ] && [ "$search_mode" -le 7 ]; then
        mem_index_path="${index_prefix}_mem_topo.index"
    else
        mem_index_path="${index_prefix}_mem.index"
    fi
    if [ "$cur_mem_L" -ne 0 ] && { [ ! -f "$mem_index_path" ] || [ ! -f "${mem_index_path}.data" ]; }; then
        echo "Warning: memory index missing ($mem_index_path or ${mem_index_path}.data). Set mem_L=0 for this run."
        cur_mem_L=0
    fi

    if [ "$data_name" = "text2img" ] || [ "$data_name" = "deep1m" ] ; then
        L="10 11 12 14 16 18 20 23 26 30 35 40 50 60 70 80 100 120 150 200 250 300 400 500 600 " 
        # L="100 120 150 200 250 300 400 500 600 " 
    elif [ "$data_name" = "laion" ] || [ "$data_name" = "deep100m" ] || [ "$data_name" = "deep10m" ] ; then
        L="10 12 14 16 18 20 23 26 30 35 40 50 60 70 80 100 120 150 200 250 300 400 500 600 800 1000  1500 2000" 
        # L=" 100 1000  " 
    elif [ "$data_name" = "msmarco" ] ; then
        L="10 12 14 16 18 20 23 26 30 35 40 50 60 70 80 100 120 150 200 250 300 400 500 600 800 1000" 
    else
        L="100 150 200 250 300 400 500 600 700 800 900 1000 1100 1300 1500 1700 1900 2000 2500" 
        # L=" 1500 1700 1900 2100 2500" 
    fi
        # L="100 120 150 200 250 300 400 500 600 " 

    log_file="${log_path}/${data_name}_search_searchmode${search_mode}_width${search_width}_mem${cur_mem_L}_thread${num_threads}_$(date +%m%d_%H%M%S).log"
    # L="100 100 100"
    # log_file="_.log"

    cmd="$PROJECT_PATH/build/tests/search_disk_index \
        float \
        ${index_prefix} \
        ${num_threads} \
        ${search_width} \
        ${query_file} \
        ${truthset_bin} \
        ${K} \
        l2 \
        pq \
        ${search_mode} \
        ${cur_mem_L} \
        ${L}"

    echo "CMD:"
    echo "$cmd"
    eval "$cmd" 2>&1 | tee "$log_file"

done
done