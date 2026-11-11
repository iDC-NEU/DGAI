#!/bin/bash
# Search topK sweep using large ground-truth files when available.
#
# Defaults:
#   datasets: text2img laion msmarco
#   K:        1 10 100
#   modes:    SEARCH_MODES from common.sh (0 1 2 4)
#   threads:  8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

EXP_NAME="${TOPK_EXP_NAME:-search_topk}"
DATASETS="${TOPK_DATASETS:-text2img laion msmarco}"
TOPK_LIST="${TOPK_LIST:-1 10 100}"
MODES="${TOPK_SEARCH_MODES:-${SEARCH_MODES}}"
NUM_THREADS="${TOPK_SEARCH_THREADS:-8}"
LOG_DIR="$(log_dir_for search "$EXP_NAME")"

validate_truthset_for_topk() {
    local truthset_bin="$1"
    local top_k="$2"
    local header nqueries ngt
    header="$(od -An -tu4 -N8 "$truthset_bin" | tr -s ' ')"
    read -r nqueries ngt <<<"$header"
    # Accept any GT that covers the requested K (e.g. gt_K100.bin for K=1/10/100).
    # Do not require ngt==10000; full-corpus K10000 files often contain IDs outside the 50M index.
    if [ -z "${nqueries:-}" ] || [ -z "${ngt:-}" ] ||
       [ "$nqueries" -ne 10000 ] || [ "$ngt" -lt "$top_k" ]; then
        echo "ERROR: invalid topK truthset header: ${truthset_bin} (${nqueries:-?} x ${ngt:-?}, K=${top_k})" >&2
        return 1
    fi
}

# Reject GTs whose neighbor IDs mostly lie outside the indexed universe [0, index_npts).
# Full-corpus gt_K10000 files (IDs up to ~1e8 / 99999999) otherwise silently cap recall ~40-50%.
_truthset_ids_in_index_universe() {
    local truthset_bin="$1"
    local index_npts="$2"
    python3 - "$truthset_bin" "$index_npts" <<'PY'
import struct
import sys

path, index_npts = sys.argv[1], int(sys.argv[2])
with open(path, "rb") as f:
    nq, dim = struct.unpack("II", f.read(8))
    n_sample = min(nq, 500)
    raw = f.read(n_sample * dim * 4)
ids = memoryview(raw).cast("I")
if not ids:
    sys.exit(1)
oob = sum(1 for x in ids if x >= index_npts) / len(ids)
# Allow tiny noise; reject full-corpus GTs (~50%+ OOB).
sys.exit(0 if oob <= 0.01 else 1)
PY
}

find_truthset_for_k() {
    local data_name="$1"
    local top_k="$2"
    local root f s index_npts
    local -a sizes=() candidates=()

    # Prefer the smallest GT that covers K, on DATA_PATH before BASE_DATA_PATH.
    for s in 1 10 100 1000 10000; do
        if [ "$s" -ge "$top_k" ]; then
            sizes+=("$s")
        fi
    done

    for root in "$DATA_PATH" "$BASE_DATA_PATH"; do
        for s in "${sizes[@]}"; do
            candidates+=(
                "${root}/${data_name}/gt_K${s}.bin"
                "${root}/${data_name}/${data_name}_gt_K${s}"
            )
        done
        # Also allow update-style 50M subset GT used by search_base (dim=10).
        if [ "$top_k" -le 10 ]; then
            candidates+=("${root}/${data_name}/${data_name}_${batch_npts}_gt_0.001/gt_0.bin")
        fi
    done

    index_npts="${batch_npts}"
    for f in "${candidates[@]}"; do
        [ -s "$f" ] || continue
        validate_truthset_for_topk "$f" "$top_k" >/dev/null 2>&1 || continue
        if ! _truthset_ids_in_index_universe "$f" "$index_npts"; then
            echo "WARN: skip GT outside index universe [0,${index_npts}): ${f}" >&2
            continue
        fi
        echo "$f"
        return 0
    done

    return 1
}

filter_l_for_k() {
    local top_k="$1"
    shift
    local l
    for l in "$@"; do
        if [ "$l" -ge "$top_k" ]; then
            printf '%s ' "$l"
        fi
    done
}

run_dgai_search_topk() {
    local data_name="$1"
    local search_mode="$2"
    local top_k="$3"
    local num_threads="$4"
    local log_dir="$5"

    PROJECT_PATH="$DGAI_ROOT"
    set_dataset_params "$data_name"
    R=32
    Lbuild=$L_DISK
    set_search_width "$search_mode"

    if ! index_exists_for "$data_name"; then
        echo "SKIP topK search: ${data_name} K=${top_k} mode=${search_mode} (missing index)"
        return 0
    fi

    local index_prefix query_file truthset_bin cur_mem_L mem_index_path l_raw l_values
    local log_file cmd existing

    index_prefix=$(benchmark_index_prefix)
    query_file="${DATA_PATH}/${data_name}/${data_name}_query.fbin"
    if [ ! -f "$query_file" ]; then
        query_file="$(resolve_query_fbin "$data_name")" || {
            echo "SKIP topK search: ${data_name} (missing query)"
            return 0
        }
    fi

    truthset_bin="$(find_truthset_for_k "$data_name" "$top_k")" || {
        echo "SKIP topK search: ${data_name} K=${top_k} (missing truthset)"
        return 0
    }
    validate_truthset_for_topk "$truthset_bin" "$top_k" || return 2

    cur_mem_L=$MEM_L
    if [ "$search_mode" -ge 4 ] && [ "$search_mode" -le 7 ]; then
        mem_index_path="${index_prefix}_mem_topo.index"
    else
        mem_index_path="${index_prefix}_mem.index"
    fi
    if [ "$cur_mem_L" -ne 0 ] && { [ ! -f "$mem_index_path" ] || [ ! -f "${mem_index_path}.data" ]; }; then
        echo "Warning: memory index missing ($mem_index_path). Set mem_L=0 for this run."
        cur_mem_L=0
    fi

    if [ "$SKIP_EXISTING" = "1" ]; then
        existing="$(_find_done_log search \
            "${log_dir}/${data_name}_K${top_k}_searchmode${search_mode}_width${search_width}_mem*_thread${num_threads}_*.log" || true)"
        if [ -n "${existing:-}" ]; then
            echo "SKIP topK search (exists): ${data_name} K=${top_k} mode=${search_mode} threads=${num_threads}"
            echo "  prior: ${existing}"
            return 0
        fi
    fi

    l_raw="${TOPK_L_VALUES:-$(search_L_for_dataset "$data_name" "$num_threads")}"
    # Keep L >= K; some binaries reject or under-report when L < K.
    l_values="$(filter_l_for_k "$top_k" $l_raw)"
    if [ -z "${l_values// }" ]; then
        l_values="$top_k"
    fi

    mkdir -p "$log_dir"
    log_file="${log_dir}/${data_name}_K${top_k}_searchmode${search_mode}_width${search_width}_mem${cur_mem_L}_thread${num_threads}_$(_exp_ts).log"

    cmd="$DGAI_ROOT/build/tests/search_disk_index \
        float \
        ${index_prefix} \
        ${num_threads} \
        ${search_width} \
        ${query_file} \
        ${truthset_bin} \
        ${top_k} \
        l2 \
        pq \
        ${search_mode} \
        ${cur_mem_L} \
        ${l_values}"

    echo "=== TOPK SEARCH ${data_name} K=${top_k} mode=${search_mode} threads=${num_threads} ==="
    echo "GT:  ${truthset_bin}"
    echo "CMD: $cmd"
    echo "LOG: $log_file"
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "ERROR: topK search failed for ${data_name} K=${top_k} mode=${search_mode}" >&2
        return 1
    fi
    if ! _topk_log_recall_valid "$log_file" "$top_k"; then
        echo "ERROR: invalid or missing Recall@${top_k} in ${log_file}" >&2
        return 2
    fi
    echo "Recall validation passed: ${log_file}"
    return 0
}

echo_exp_banner "$EXP_NAME" \
    "datasets=[${DATASETS}] K=[${TOPK_LIST}] modes=[${MODES}] threads=${NUM_THREADS} log=${LOG_DIR}"

if [ "${TOPK_SKIP_BUILD:-0}" != "1" ]; then
    build_dgai_search
fi

for top_k in $TOPK_LIST; do
    for data_name in $DATASETS; do
        for search_mode in $MODES; do
            if run_dgai_search_topk "$data_name" "$search_mode" "$top_k" "$NUM_THREADS" "$LOG_DIR"; then
                :
            else
                rc=$?
                if [ "$rc" -eq 2 ]; then
                    echo "ERROR: stop topK sweep because Recall validation failed" >&2
                    exit 2
                fi
                [ "$CONTINUE_ON_ERROR" = "1" ] || exit "$rc"
            fi
        done
    done
done

echo "=== Done: ${EXP_NAME} (logs: ${LOG_DIR}) ==="
