#!/bin/bash
# Experiment: baseline update with default parameters.
# Datasets: text2img laion msmarco (deep1b is reused from the completed run)
# Systems: odin (mode0), dgai (mode4), fresh, greator
# UPDATE_RATIO=0.001, default insert/merge threads (no patch)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

EXP_NAME="update_base"
DATASETS="${BASELINE_DATASETS}"
RATIO="${BASELINE_UPDATE_RATIO:-0.001}"

echo_exp_banner "update_baseline" \
    "datasets=[${DATASETS}] ratio=${RATIO} systems=[${RUN_UPDATE_SYSTEMS}] NUM_BATCHES=${NUM_BATCHES} (SKIP_EXISTING=${SKIP_EXISTING})"

n_data=$(echo "$DATASETS" | wc -w)
n_sys=$(echo "$RUN_UPDATE_SYSTEMS" | wc -w)
echo "Estimated runs: $((n_data * n_sys))"

if _exp_system_enabled odin; then
    build_dgai_update 0
    log_dir="$(log_dir_for odin "$EXP_NAME")"
    for data_name in $DATASETS; do
        run_dgai_update "$data_name" 0 "$RATIO" "$log_dir" "odin" || {
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        }
    done
fi

if _exp_system_enabled dgai; then
    build_dgai_update 4
    log_dir="$(log_dir_for dgai "$EXP_NAME")"
    for data_name in $DATASETS; do
        run_dgai_update "$data_name" 4 "$RATIO" "$log_dir" "dgai" || {
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        }
    done
fi

if _exp_system_enabled fresh; then
    build_fresh
    log_dir="$(log_dir_for fresh "$EXP_NAME")"
    for data_name in $DATASETS; do
        run_fresh_update "$data_name" "$RATIO" "$log_dir" || {
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        }
    done
fi

if _exp_system_enabled greator; then
    build_greator
    log_dir="$(log_dir_for greator "$EXP_NAME")"
    for data_name in $DATASETS; do
        run_greator_update "$data_name" "$RATIO" "$log_dir" || {
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        }
    done
fi

echo "=== Done: update_baseline ==="
