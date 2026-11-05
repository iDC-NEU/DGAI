#!/bin/bash

source "$(dirname "$0")/env.sh"

# ==============================================================
# 配置
# ==============================================================
data_type=float
metric=l2
K=10
L_disk=75
UPDATE_RATIO=0.001
NUM_BATCHES="${NUM_BATCHES:-3}"
Lsearch="100"

log_path=$PROJECT_PATH/logs
mkdir -p "$log_path/fig1"

# search_mode: 2=pipe, 7=starling_decouple (no page split)
for data_name in msmarco; do
for search_mode in  4 ; do

    # Mode 4: decouple + REORDER_COMPUTE_PQ (page split enabled).
    # Mode 7: starling.decouple layout + REORDER_COMPUTE_PQ + no page split.
    # IMPORTANT: clear any cached -DREAD_ONLY_TESTS in CMAKE_CXX_FLAGS — that
    # opens graph/vector O_RDONLY and makes insert pwrite/io_uring return EBADF.
    if [ "$search_mode" -eq 4 ]; then
        export ADDITIONAL_DEFINITIONS="-DNO_POLLUTE_ORIGINAL -DREORDER_COMPUTE_PQ"
        echo "Building with REORDER_COMPUTE_PQ (search_mode=${search_mode}, split=ON)"
        cmake_extra=(-DCMAKE_CXX_FLAGS=)
    elif [ "$search_mode" -eq 7 ]; then
        export ADDITIONAL_DEFINITIONS="-DNO_POLLUTE_ORIGINAL -DREORDER_COMPUTE_PQ -DDGAI_DISABLE_GRAPH_SPLIT"
        echo "Building with REORDER_COMPUTE_PQ + DGAI_DISABLE_GRAPH_SPLIT (search_mode=${search_mode}, starling_decouple)"
        cmake_extra=(-DCMAKE_CXX_FLAGS= -DDGAI_DISABLE_GRAPH_SPLIT=ON)
    else
        export ADDITIONAL_DEFINITIONS="-DNO_POLLUTE_ORIGINAL"
        echo "Building without REORDER_COMPUTE_PQ (search_mode=${search_mode})"
        cmake_extra=(-DCMAKE_CXX_FLAGS=)
    fi
    cd "$PROJECT_PATH/build"
    cmake "${cmake_extra[@]}" .. && make -j
    cd "$PROJECT_PATH"

    set_dataset_params "$data_name"
    R=32
    Lbuild=$L_disk

    set_search_width "$search_mode"
    beam_width=$search_width

    canonical_index_prefix=$(benchmark_index_prefix)

    copy_shadow_index "$canonical_index_prefix" "$search_mode"
    index_prefix="$index_prefix_shadow"

    data_file=$(resolve_base_fbin "$data_name" "$batch_npts")
    query_file=$(resolve_query_fbin "$data_name")
    truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_${UPDATE_RATIO}
    update_size=$(compute_update_size)

    if [ ! -f "${truthset_prefix}/gt_0.bin" ]; then
        echo "ERROR: missing GT ${truthset_prefix}/gt_0.bin (UPDATE_RATIO=${UPDATE_RATIO}). Recall would be reported as 0." >&2
        echo "       Expected under ${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_*" >&2
        exit 1
    fi

    log_file="${log_path}/fig1/${data_name}_overall_perf_mode${search_mode}_width${search_width}_${UPDATE_RATIO}_$(date +%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$log_file")"

    cmd="$PROJECT_PATH/build/tests/overall_performance \
        ${data_type} \
        ${data_file} \
        ${L_disk} \
        ${index_prefix} \
        ${query_file} \
        ${truthset_prefix} \
        ${K} \
        ${beam_width} \
        ${update_size} \
        ${NUM_BATCHES} \
        ${search_mode} \
        ${Lsearch}"

    echo "update_size (batch_npts * UPDATE_RATIO): ${update_size}"
    echo "num_batches:    ${NUM_BATCHES}"
    echo "canonical_index: ${canonical_index_prefix}"
    echo "shadow_index:    ${index_prefix}"
    echo "CMD:"
    echo "$cmd"
    eval "$cmd" 2>&1 | tee "$log_file"

done
done
