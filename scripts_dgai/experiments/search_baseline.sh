#!/bin/bash
# Experiment: baseline search with default parameters.
# Datasets: text2img laion msmarco deep1b
# Modes: 0 1 2 4, threads=8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

EXP_NAME="baseline"
DATASETS="${BASELINE_DATASETS}"
MODES="${SEARCH_MODES}"
NUM_THREADS="${BASELINE_SEARCH_THREADS:-8}"
LOG_DIR="$(log_dir_for search "$EXP_NAME")"
GOVECTOR_LOG_DIR="$(log_dir_for govector "$EXP_NAME")"

echo_exp_banner "search_baseline" \
    "datasets=[${DATASETS}] modes=[${MODES}] threads=${NUM_THREADS} systems=[${RUN_SEARCH_SYSTEMS}] dgai_log=${LOG_DIR} govector_log=${GOVECTOR_LOG_DIR} (SKIP_EXISTING=${SKIP_EXISTING})"

n_data=$(echo "$DATASETS" | wc -w)
n_modes=$(echo "$MODES" | wc -w)
estimated=0
if _exp_search_system_enabled dgai; then
    estimated=$((estimated + n_data * n_modes))
fi
if _exp_search_system_enabled govector; then
    estimated=$((estimated + n_data))
fi
echo "Estimated runs: ${estimated}"

if _exp_search_system_enabled dgai; then
    build_dgai_search

    for data_name in $DATASETS; do
        for search_mode in $MODES; do
            run_dgai_search "$data_name" "$search_mode" "$NUM_THREADS" "$LOG_DIR" || {
                [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
            }
        done
    done
fi

if _exp_search_system_enabled govector; then
    build_govector_search

    for data_name in $DATASETS; do
        run_govector_search "$data_name" "$NUM_THREADS" "$GOVECTOR_LOG_DIR" || {
            [ "$CONTINUE_ON_ERROR" = "1" ] || exit 1
        }
    done
fi

echo "=== Done: search_baseline (logs: ${LOG_DIR}, ${GOVECTOR_LOG_DIR}) ==="
