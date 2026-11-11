#!/bin/bash
# Experiment: update thread scaling (insert = delete/merge = T)
# Datasets: default text2img only (override with THREAD_DATASETS)
# Systems: odin dgai fresh greator
# Thread order: high → low (32…1). Override with UPDATE_THREAD_SWEEP.
# Does not change common.sh UPDATE_THREAD_LIST (other experiments keep 1→32).
# Batch ratio fixed at 0.001
# Flow per thread T: patch -> rebuild systems -> run datasets -> restore

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

EXP_NAME="update_threads"
DATASETS="${THREAD_DATASETS}"
# High → low only for this sweep (avoid mutating UPDATE_THREAD_LIST).
THREADS="${UPDATE_THREAD_SWEEP:-32 16 8 4 2 1}"
RATIO="${UPDATE_THREAD_RATIO:-0.001}"

echo_exp_banner "$EXP_NAME" \
    "datasets=[${DATASETS}] threads=[${THREADS}] ratio=${RATIO} systems=[${RUN_UPDATE_SYSTEMS}] NUM_BATCHES=${NUM_BATCHES}"

n_data=$(echo "$DATASETS" | wc -w)
n_thr=$(echo "$THREADS" | wc -w)
n_sys=$(echo "$RUN_UPDATE_SYSTEMS" | wc -w)
echo "Estimated runs: $((n_data * n_thr * n_sys)) (plus rebuilds per thread)"

trap_restore_threads

for t in $THREADS; do
    echo "========== Thread setting T=${t} =========="
    thread_tag="thread${t}"

    if _exp_system_enabled odin || _exp_system_enabled dgai; then
        patch_dgai_threads "$t"
    fi
    if _exp_system_enabled fresh; then
        patch_fresh_threads "$t"
    fi
    if _exp_system_enabled greator; then
        patch_greator_threads "$t"
    fi

    if _exp_system_enabled odin; then
        if build_dgai_update 0; then
            log_dir="$(log_dir_for odin "$EXP_NAME")"
            for data_name in $DATASETS; do
                run_dgai_update "$data_name" 0 "$RATIO" "$log_dir" "odin" "$thread_tag" || {
                    [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
                }
            done
        else
            echo "ERROR: skip Odin T=${t}; build failed" >&2
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        fi
    fi

    if _exp_system_enabled dgai; then
        if build_dgai_update 4; then
            log_dir="$(log_dir_for dgai "$EXP_NAME")"
            for data_name in $DATASETS; do
                run_dgai_update "$data_name" 4 "$RATIO" "$log_dir" "dgai" "$thread_tag" || {
                    [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
                }
            done
        else
            echo "ERROR: skip DGAI T=${t}; build failed" >&2
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        fi
    fi

    if _exp_system_enabled fresh; then
        if build_fresh; then
            log_dir="$(log_dir_for fresh "$EXP_NAME")"
            for data_name in $DATASETS; do
                run_fresh_update "$data_name" "$RATIO" "$log_dir" "$thread_tag" || {
                    [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
                }
            done
        else
            echo "ERROR: skip Fresh T=${t}; build failed" >&2
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        fi
    fi

    if _exp_system_enabled greator; then
        if build_greator; then
            log_dir="$(log_dir_for greator "$EXP_NAME")"
            for data_name in $DATASETS; do
                run_greator_update "$data_name" "$RATIO" "$log_dir" "$thread_tag" || {
                    [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
                }
            done
        else
            echo "ERROR: skip Greator T=${t}; build failed" >&2
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        fi
    fi

    restore_all_thread_patches
done

clear_thread_trap
restore_all_thread_patches

if [ "${REBUILD_AFTER_THREAD_SWEEP:-1}" = "1" ]; then
    if _exp_system_enabled dgai; then
        build_dgai_update 4
    elif _exp_system_enabled odin; then
        build_dgai_update 0
    fi
    if _exp_system_enabled fresh; then
        build_fresh || { [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1; }
    fi
    if _exp_system_enabled greator; then
        build_greator || { [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1; }
    fi
fi

echo "=== Done: ${EXP_NAME} ==="
