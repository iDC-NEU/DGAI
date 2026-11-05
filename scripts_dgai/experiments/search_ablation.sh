#!/bin/bash
# Experiment: search ablation.
#
# Stages:
#   baseline          : decouple beam search + mem index, mode5
#   sadl              : locality layout + page-level search, mode6
#   search_noopt      : DFS (mode4, no trunc) without CACHE_N / USE_PRIORITY_QUEUE
#   search_notrunc    : locality layout + DGAI search algorithm, mode4, large rerank_window
#   dgai              : locality layout + DGAI search algorithm + small rerank_window (=30)
#
# Default datasets: text2img laion msmarco deep1b
# Override:
#   ABLATION_DATASETS="text2img deep1b" bash search_ablation.sh
#   ABLATION_STAGES="baseline dgai" bash search_ablation.sh
#   SKIP_EXISTING=0 bash search_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

EXP_NAME="search_ablation"
DATASETS="${ABLATION_DATASETS:-text2img laion msmarco deep1b}"
STAGES="${ABLATION_STAGES:-baseline sadl search_noopt search_notrunc dgai}"
NUM_THREADS="${ABLATION_SEARCH_THREADS:-8}"
LOG_DIR="${ABLATION_LOG_DIR:-$(log_dir_for search "$EXP_NAME")}"
NO_TRUNC_MAX="${ABLATION_NO_TRUNC_MAX:-100000}"
TRUNC_MAX="${ABLATION_TRUNC_MAX:-30}"
DECOUPLE_SRC="$DGAI_ROOT/src/search/decouple_search.cpp"

ORIG_DECOUPLE="$(mktemp)"
cp "$DECOUPLE_SRC" "$ORIG_DECOUPLE"

cleanup() {
    cp "$ORIG_DECOUPLE" "$DECOUPLE_SRC"
    rm -f "$ORIG_DECOUPLE"
}
trap cleanup EXIT

echo_exp_banner "$EXP_NAME" \
    "datasets=[${DATASETS}] stages=[${STAGES}] threads=${NUM_THREADS} log=${LOG_DIR} (SKIP_EXISTING=${SKIP_EXISTING})"

# enable=1 keeps CACHE_N / USE_PRIORITY_QUEUE; enable=0 comments them out.
set_decouple_micro_opts() {
    local enable="$1"
    python3 - "$DECOUPLE_SRC" "$enable" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
enable = sys.argv[2] == "1"
text = path.read_text()

# Match the pair of defines (optionally already commented).
pattern = re.compile(
    r"^[ \t]*(?://[ \t]*)?#define[ \t]+CACHE_N[ \t]*\n"
    r"[ \t]*(?://[ \t]*)?#define[ \t]+USE_PRIORITY_QUEUE[ \t]*$",
    re.MULTILINE,
)
if enable:
    replacement = "#define CACHE_N\n#define USE_PRIORITY_QUEUE"
else:
    replacement = "// #define CACHE_N\n// #define USE_PRIORITY_QUEUE"

new_text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit(
        "failed to update CACHE_N / USE_PRIORITY_QUEUE defines in decouple_search.cpp"
    )
path.write_text(new_text)
PY
}

stage_mode() {
    case "$1" in
        baseline) echo 5 ;;
        sadl) echo 6 ;;
        search_noopt|search_notrunc|dgai) echo 4 ;;
        *) echo "Unknown ablation stage: $1" >&2; return 1 ;;
    esac
}

stage_max_candidates() {
    case "$1" in
        baseline|sadl|search_noopt|search_notrunc) echo "$NO_TRUNC_MAX" ;;
        dgai) echo "$TRUNC_MAX" ;;
        *) echo "Unknown ablation stage: $1" >&2; return 1 ;;
    esac
}

# 1 = keep micro-opts; 0 = disable CACHE_N / USE_PRIORITY_QUEUE
stage_micro_opts() {
    case "$1" in
        search_noopt) echo 0 ;;
        baseline|sadl|search_notrunc|dgai) echo 1 ;;
        *) echo "Unknown ablation stage: $1" >&2; return 1 ;;
    esac
}

ablation_L_for_dataset() {
    local data_name="$1"
    if [ -n "${ABLATION_L_OVERRIDE:-}" ]; then
        echo "$ABLATION_L_OVERRIDE"
        return 0
    fi
    case "$data_name" in
        text2img|deep1m)
            echo "18 20 23 26 30 35 40 50 60 80 100 120 150 200 300"
            ;;
        laion|deep100m|deep10m)
            echo "30 35 40 50 60 80 100 150 200 300 500 800 1000 1500 2000 2500 3000"
            ;;
        msmarco)
            echo "30 35 40 50 60 80 100 150 200 300 500 800 1000"
            ;;
        *)
            echo "700 800 900 1000 1200 1500 1800 2100 2500"
            ;;
    esac
}

run_search_ablation() {
    local data_name="$1"
    local stage="$2"
    local search_mode="$3"

    PROJECT_PATH="$DGAI_ROOT"
    set_dataset_params "$data_name"
    R=32
    Lbuild=$L_DISK
    set_search_width "$search_mode"

    if ! index_exists_for "$data_name"; then
        echo "SKIP ${stage}: ${data_name} mode=${search_mode} (missing index)"
        return 0
    fi

    local index_prefix query_file truthset_bin cur_mem_L mem_index_path L log_file cmd existing
    index_prefix=$(benchmark_index_prefix)
    query_file="${DATA_PATH}/${data_name}/${data_name}_query.fbin"
    if [ ! -f "$query_file" ]; then
        query_file="$(resolve_query_fbin "$data_name")" || {
            echo "SKIP ${stage}: ${data_name} (missing query)"
            return 0
        }
    fi
    truthset_bin="${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_0.001/gt_0.bin"
    if [ ! -f "$truthset_bin" ]; then
        echo "WARN: truthset bin missing: ${truthset_bin}" >&2
        if ! truthset_exists_for "$data_name" "0.001"; then
            echo "SKIP ${stage}: ${data_name} (missing truthset)"
            return 0
        fi
    fi

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

    mkdir -p "$LOG_DIR"
    if [ "$SKIP_EXISTING" = "1" ]; then
        existing="$(_find_done_log search \
            "${LOG_DIR}/${data_name}_${stage}_searchmode${search_mode}_width${search_width}_mem*_thread${NUM_THREADS}_*.log" || true)"
        if [ -n "${existing:-}" ]; then
            echo "SKIP ${stage} search (exists): ${data_name} mode=${search_mode} threads=${NUM_THREADS}"
            echo "  prior: ${existing}"
            return 0
        fi
    fi

    L="$(ablation_L_for_dataset "$data_name")"
    log_file="${LOG_DIR}/${data_name}_${stage}_searchmode${search_mode}_width${search_width}_mem${cur_mem_L}_thread${NUM_THREADS}_$(_exp_ts).log"

    rerank_window="$(stage_max_candidates "$stage")"

    cmd="$DGAI_ROOT/build/tests/search_disk_index \
        float \
        ${index_prefix} \
        ${NUM_THREADS} \
        ${search_width} \
        ${query_file} \
        ${truthset_bin} \
        ${K} \
        l2 \
        pq \
        ${search_mode} \
        ${cur_mem_L} \
        --rerank-window ${rerank_window} \
        ${L}"

    echo "=== ABLATION ${stage} ${data_name} mode=${search_mode} threads=${NUM_THREADS} rerank_window=${rerank_window} ==="
    echo "CMD: $cmd"
    echo "LOG: $log_file"
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "ERROR: ablation ${stage} failed for ${data_name}" >&2
        if [ "$CONTINUE_ON_ERROR" != "1" ]; then
            return 1
        fi
    fi
    return 0
}

last_micro_opts=""
for data_name in $DATASETS; do
    echo "=== Dataset: ${data_name} ==="
    for stage in $STAGES; do
        search_mode="$(stage_mode "$stage")"
        max_candidates="$(stage_max_candidates "$stage")"
        micro_opts="$(stage_micro_opts "$stage")"

        # rerank window is a CLI arg; only micro-opt macros require a rebuild.
        if [ "$micro_opts" != "$last_micro_opts" ]; then
            echo "=== Configure build for stage=${stage}: micro_opts=${micro_opts} (rerank_window=${max_candidates} via CLI) ==="
            set_decouple_micro_opts "$micro_opts"
            build_dgai_search
            last_micro_opts="$micro_opts"
        else
            echo "=== Stage=${stage}: reuse build (micro_opts=${micro_opts}), rerank_window=${max_candidates} via CLI ==="
        fi

        run_search_ablation "$data_name" "$stage" "$search_mode" || {
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        }
    done
done

echo "=== Done: ${EXP_NAME} (logs: ${LOG_DIR}) ==="
