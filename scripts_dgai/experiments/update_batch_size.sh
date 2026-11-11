#!/bin/bash
# Experiment: update batch-size sweep
# Datasets: default text2img only (override with BATCH_DATASETS)
# Systems: odin (DGAI mode0), dgai (mode4), fresh, greator
# Ratios (small → large): 0.001 0.005 0.01 0.05  (BATCH_SWEEP_RATIOS / BATCH_RATIOS)
# NUM_BATCHES from common.sh default 1 for all update experiments.
# Default threads (no patch). Build once per system, then sweep datasets x ratios.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

EXP_NAME="update_batch"
DATASETS="${BATCH_DATASETS}"
# Small → large for this sweep only.
RATIOS="${BATCH_SWEEP_RATIOS:-${BATCH_RATIOS:-0.001 0.005 0.01 0.05}}"

echo_exp_banner "$EXP_NAME" \
    "datasets=[${DATASETS}] ratios=[${RATIOS}] systems=[${RUN_UPDATE_SYSTEMS}] NUM_BATCHES=${NUM_BATCHES}"

n_data=$(echo "$DATASETS" | wc -w)
n_ratio=$(echo "$RATIOS" | wc -w)
n_sys=$(echo "$RUN_UPDATE_SYSTEMS" | wc -w)
echo "Estimated runs: $((n_data * n_ratio * n_sys))"

if _exp_system_enabled odin; then
    build_dgai_update 0
    log_dir="$(log_dir_for odin "$EXP_NAME")"
    for data_name in $DATASETS; do
        for ratio in $RATIOS; do
            run_dgai_update "$data_name" 0 "$ratio" "$log_dir" "odin" || {
                [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
            }
        done
    done
fi

if _exp_system_enabled dgai; then
    build_dgai_update 4
    log_dir="$(log_dir_for dgai "$EXP_NAME")"
    for data_name in $DATASETS; do
        for ratio in $RATIOS; do
            run_dgai_update "$data_name" 4 "$ratio" "$log_dir" "dgai" || {
                [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
            }
        done
    done
fi

if _exp_system_enabled fresh; then
    build_fresh
    log_dir="$(log_dir_for fresh "$EXP_NAME")"
    for data_name in $DATASETS; do
        for ratio in $RATIOS; do
            run_fresh_update "$data_name" "$ratio" "$log_dir" || {
                [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
            }
        done
    done
fi

if _exp_system_enabled greator; then
    build_greator
    log_dir="$(log_dir_for greator "$EXP_NAME")"
    for data_name in $DATASETS; do
        for ratio in $RATIOS; do
            run_greator_update "$data_name" "$ratio" "$log_dir" || {
                [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
            }
        done
    done
fi

echo "=== Done: ${EXP_NAME} ==="
