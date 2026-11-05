#!/bin/bash
# Final supplementary experiment queue. Runs serially to avoid index-shadow and I/O contention.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

RUN_ID="${FINAL_RUN_ID:-$(date +%m%d_%H%M%S)}"
STATE_ROOT="$DGAI_ROOT/exp_logs/final_supplement/${RUN_ID}"
STATUS_FILE="$STATE_ROOT/status.tsv"
MARKER_FILE="$STATE_ROOT/run.started"
FAILURE_REPORT="$STATE_ROOT/external_failures.txt"
MIN_FREE_GB="${MIN_FREE_GB:-150}"

mkdir -p "$STATE_ROOT"
touch "$MARKER_FILE"
printf '%s\n' "$$" >"$STATE_ROOT/master.pid"
printf '%s\n' "$STATE_ROOT" >"$DGAI_ROOT/exp_logs/final_supplement/latest_state"

record_status() {
    printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" | tee -a "$STATUS_FILE"
}

check_resources() {
    local target="${1:-${DISKANN_INDEX_PATH:-${OUTPUT_PATH:-$DGAI_ROOT}}}"
    local avail_kb min_kb mount_options
    avail_kb="$(df -Pk "$target" | awk 'NR==2 {print $4}')"
    min_kb=$((MIN_FREE_GB * 1024 * 1024))
    mount_options="$(findmnt -T "$target" -n -o OPTIONS)"
    if printf '%s' "$mount_options" | tr ',' '\n' | grep -qx ro; then
        echo "ERROR: filesystem is read-only: ${target} (${mount_options})" >&2
        return 1
    fi
    if [ "$avail_kb" -lt "$min_kb" ]; then
        echo "ERROR: only $((avail_kb / 1024 / 1024)) GiB free on ${target}; require ${MIN_FREE_GB} GiB" >&2
        return 1
    fi
}

assert_macro() {
    local file="$1"
    local pattern="$2"
    grep -qE "$pattern" "$file" || {
        echo "ERROR: unexpected thread defaults in ${file}; missing ${pattern}" >&2
        return 1
    }
}

# Prefer 50M-subset GTs that cover K=100. Full-corpus *_gt_K10000 often has IDs outside the index.
find_truthset_for_preflight() {
    local dataset="$1"
    local candidate
    set_dataset_params "$dataset"
    for candidate in \
        "$DATA_PATH/$dataset/gt_K100.bin" \
        "$DATA_PATH/$dataset/gt_K1000.bin" \
        "$DATA_PATH/$dataset/${dataset}_gt_K10000" \
        "$DATA_PATH/$dataset/${dataset}_${batch_npts}_gt_0.001/gt_0.bin" \
        "$BASE_DATA_PATH/$dataset/gt_K100.bin" \
        "$BASE_DATA_PATH/$dataset/${dataset}_gt_K10000"; do
        if [ -s "$candidate" ]; then
            # Reject full-corpus K10000 with IDs >= index size (same check as search_topk.sh).
            if python3 - "$candidate" "$batch_npts" <<'PY'
import struct, sys
path, npts = sys.argv[1], int(sys.argv[2])
with open(path, "rb") as f:
    nq, dim = struct.unpack("II", f.read(8))
    n = min(nq, 500)
    ids = memoryview(f.read(n * dim * 4)).cast("I")
oob = sum(1 for x in ids if x >= npts) / max(len(ids), 1)
sys.exit(0 if oob <= 0.01 else 1)
PY
            then
                echo "$candidate"
                return 0
            fi
        fi
    done
    echo "ERROR: missing in-index GT for ${dataset} (need gt_K100.bin or filtered K10000)" >&2
    return 1
}

preflight() {
    record_status preflight running
    check_resources
    if pgrep -f '(^|/)(overall_performance|search_disk_index)( |$)' >/dev/null; then
        echo "ERROR: another experiment binary is already running" >&2
        return 1
    fi

    assert_macro "$DGAI_OP_CPP" '^#define NUM_INSERT_THREADS 10$'
    assert_macro "$DGAI_OP_CPP" '^#define NUM_MERGE_THREADS 20$'
    assert_macro "$DGAI_OP_CPP" '^#define NUM_DELETE_THREADS 1$'
    assert_macro "$FRESH_OP_CPP" '^#define NUM_INSERT_THREADS 24$'
    assert_macro "$GREATOR_OP_CPP" '^#define NUM_INSERT_THREADS 10$'

    local dataset gt header nqueries ngt
    for dataset in text2img laion msmarco; do
        gt="$(find_truthset_for_preflight "$dataset")"
        header="$(od -An -tu4 -N8 "$gt" | tr -s ' ')"
        read -r nqueries ngt <<<"$header"
        [ "$nqueries" -eq 10000 ] && [ "$ngt" -ge 10 ] || {
            echo "ERROR: invalid search GT for ${dataset}: ${nqueries} x ${ngt} (${gt})" >&2
            return 1
        }
    done
    record_status preflight passed
}

run_required() {
    local name="$1"
    shift
    check_resources
    record_status "$name" running
    if "$@"; then
        record_status "$name" passed
    else
        local rc=$?
        record_status "$name" "failed:${rc}"
        return "$rc"
    fi
}

run_optional() {
    local name="$1"
    shift
    check_resources
    record_status "$name" running
    if "$@"; then
        record_status "$name" completed
    else
        local rc=$?
        record_status "$name" "skipped_after_failure:${rc}"
    fi
    return 0
}

run_topk_smoke() {
    env \
        TOPK_EXP_NAME=search_topk_smoke \
        TOPK_DATASETS=text2img \
        TOPK_LIST='1 10 100' \
        TOPK_SEARCH_MODES=4 \
        TOPK_SEARCH_THREADS=8 \
        TOPK_L_VALUES=100 \
        TOPK_SKIP_BUILD=1 \
        SKIP_EXISTING=1 \
        CONTINUE_ON_ERROR=0 \
        bash "$SCRIPT_DIR/search_topk.sh"
}

run_core_experiments() {
    if [ "${FINAL_RESUME_FROM:-start}" != "search_topk" ]; then
        run_required dgai_update_base env \
            RUN_UPDATE_SYSTEMS=dgai BASELINE_DATASETS='text2img laion msmarco' \
            SKIP_EXISTING=0 CONTINUE_ON_ERROR=0 \
            bash "$SCRIPT_DIR/update_baseline.sh"

        run_required dgai_update_scale_deep100m env \
            RUN_UPDATE_SYSTEMS=dgai SCALE_DATASETS=deep100m \
            SKIP_EXISTING=0 CONTINUE_ON_ERROR=0 \
            bash "$SCRIPT_DIR/update_scalability.sh"

        run_required dgai_update_batch env \
            RUN_UPDATE_SYSTEMS=dgai BATCH_DATASETS=text2img \
            BATCH_SWEEP_RATIOS='0.001 0.005 0.01 0.05' \
            SKIP_EXISTING=0 CONTINUE_ON_ERROR=0 \
            bash "$SCRIPT_DIR/update_batch_size.sh"
    else
        record_status queue "resumed_from:search_topk"
    fi

    run_required search_topk env \
        TOPK_DATASETS='text2img laion msmarco' \
        TOPK_LIST='1 10 100' TOPK_SEARCH_MODES='0 1 2 4' TOPK_SEARCH_THREADS=8 \
        SKIP_EXISTING=1 CONTINUE_ON_ERROR=0 \
        bash "$SCRIPT_DIR/search_topk.sh"

    run_required dgai_update_threads env \
        RUN_UPDATE_SYSTEMS=dgai UPDATE_THREAD_SWEEP='32 16 8 4 2 1' \
        SKIP_EXISTING=0 CONTINUE_ON_ERROR=0 \
        bash "$SCRIPT_DIR/update_threads.sh"

    run_required odin_update_threads_missing env \
        RUN_UPDATE_SYSTEMS=odin UPDATE_THREAD_SWEEP='8 4 2 1' \
        SKIP_EXISTING=1 CONTINUE_ON_ERROR=0 \
        bash "$SCRIPT_DIR/update_threads.sh"
}

run_external_experiments() {
    run_optional fresh_update_batch_missing env \
        RUN_UPDATE_SYSTEMS=fresh BATCH_DATASETS=text2img \
        BATCH_SWEEP_RATIOS='0.005 0.05' SKIP_EXISTING=1 CONTINUE_ON_ERROR=1 \
        bash "$SCRIPT_DIR/update_batch_size.sh"

    run_optional greator_update_batch env \
        RUN_UPDATE_SYSTEMS=greator BATCH_DATASETS=text2img \
        BATCH_SWEEP_RATIOS='0.001 0.005 0.01 0.05' SKIP_EXISTING=1 CONTINUE_ON_ERROR=1 \
        bash "$SCRIPT_DIR/update_batch_size.sh"

    run_optional fresh_update_threads_missing env \
        RUN_UPDATE_SYSTEMS=fresh UPDATE_THREAD_SWEEP='32 8 4 2 1' \
        SKIP_EXISTING=1 CONTINUE_ON_ERROR=1 \
        bash "$SCRIPT_DIR/update_threads.sh"

    run_optional greator_update_threads env \
        RUN_UPDATE_SYSTEMS=greator UPDATE_THREAD_SWEEP='32 16 8 4 2 1' \
        SKIP_EXISTING=1 CONTINUE_ON_ERROR=1 \
        bash "$SCRIPT_DIR/update_threads.sh"
}

summarize_external_failures() {
    : >"$FAILURE_REPORT"
    local root f
    for root in "$FRESH_ROOT/exp_logs/update_batch" "$FRESH_ROOT/exp_logs/update_threads" \
                "$GREATOR_ROOT/exp_logs/update_batch" "$GREATOR_ROOT/exp_logs/update_threads"; do
        [ -d "$root" ] || continue
        while IFS= read -r -d '' f; do
            if ! _log_looks_done "$f" update; then
                printf '\n=== %s ===\n' "$f" >>"$FAILURE_REPORT"
                grep -iE 'ERROR:|Segmentation fault|Aborted|Killed|Bad file descriptor|Assertion .*failed|terminate called|std::bad_alloc|Read-only file system' "$f" \
                    | tail -n 8 >>"$FAILURE_REPORT" || true
                tail -n 12 "$f" >>"$FAILURE_REPORT"
            fi
        done < <(find "$root" -type f -name '*.log' -newer "$MARKER_FILE" -print0)
    done
    if [ -s "$FAILURE_REPORT" ]; then
        record_status external_failure_report "$FAILURE_REPORT"
    else
        printf 'No new Fresh/Greator failures.\n' >"$FAILURE_REPORT"
        record_status external_failure_report clean
    fi
}

refresh_plots() {
    local script
    for script in plot_update_throughput.py plot_scalability.py plot_update_batch_delete_throughput.py plot_io_time.py; do
        if [ -f "$DGAI_ROOT/exp_logs/$script" ]; then
            (cd "$DGAI_ROOT/exp_logs" && python3 "$script") || \
                echo "WARN: plot refresh failed: ${script}" >&2
        fi
    done
}

finish_defaults() {
    restore_all_thread_patches
    build_dgai_update 4
}

record_status queue started
preflight
if [ "${FINAL_PREFLIGHT_ONLY:-0}" = "1" ]; then
    record_status queue preflight_only_completed
    exit 0
fi
run_required topk_recall_smoke run_topk_smoke
if [ "${FINAL_SMOKE_ONLY:-0}" = "1" ]; then
    record_status queue smoke_only_completed
    exit 0
fi
run_core_experiments
run_external_experiments
finish_defaults
summarize_external_failures
refresh_plots
record_status queue completed
echo "=== Final supplementary queue completed: ${STATE_ROOT} ==="
