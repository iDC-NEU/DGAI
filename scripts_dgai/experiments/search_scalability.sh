#!/bin/bash
# Experiment: search scalability across deep1m / deep10m / deep100m / deep1b
# Strategies: 0 1 2 4, threads=8. Target recall@98 from L sweep logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

EXP_NAME="search_scalability"
DATASETS="${SCALE_DATASETS}"
MODES="${SEARCH_MODES}"
NUM_THREADS="${SEARCH_SCALE_THREADS:-8}"
LOG_DIR="$(log_dir_for search "$EXP_NAME")"
GOVECTOR_LOG_DIR="$(log_dir_for govector "$EXP_NAME")"

echo_exp_banner "$EXP_NAME" \
    "datasets=[${DATASETS}] modes=[${MODES}] threads=${NUM_THREADS} systems=[${RUN_SEARCH_SYSTEMS}] dgai_log=${LOG_DIR} govector_log=${GOVECTOR_LOG_DIR}"

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
                if [ "$CONTINUE_ON_ERROR" != "1" ]; then
                    exit 1
                fi
            }
        done
    done
fi

if _exp_search_system_enabled govector; then
    build_govector_search

    for data_name in $DATASETS; do
        run_govector_search "$data_name" "$NUM_THREADS" "$GOVECTOR_LOG_DIR" || {
            if [ "$CONTINUE_ON_ERROR" != "1" ]; then
                exit 1
            fi
        }
    done
fi

echo "=== Done: ${EXP_NAME} (logs: ${LOG_DIR}, ${GOVECTOR_LOG_DIR}) ==="
