#!/bin/bash
# Experiment: search thread scaling (default: text2img only)
# Strategies: 0 1 2 4. Target recall@98.
# Thread order: high → low (32…1). Override with SEARCH_THREAD_SWEEP.
# Does not change common.sh SEARCH_THREAD_LIST (other experiments keep 1→32).
# Override datasets with THREAD_DATASETS="text2img laion msmarco".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

# Thread sweep has many runs (esp. thread=1); use sparse L across all thread counts.
# Bracket 90/98 for interpolation; override with SEARCH_L_DENSITY=full if needed.
export SEARCH_L_DENSITY="${SEARCH_L_DENSITY:-sparse}"

EXP_NAME="search_threads"
DATASETS="${THREAD_DATASETS}"
MODES="${SEARCH_MODES}"
# High → low only for this sweep (avoid mutating SEARCH_THREAD_LIST).
THREADS="${SEARCH_THREAD_SWEEP:-32 16 8 4 2 1}"
LOG_DIR="$(log_dir_for search "$EXP_NAME")"
GOVECTOR_LOG_DIR="$(log_dir_for govector "$EXP_NAME")"

echo_exp_banner "$EXP_NAME" \
    "datasets=[${DATASETS}] modes=[${MODES}] threads=[${THREADS}] systems=[${RUN_SEARCH_SYSTEMS}] dgai_log=${LOG_DIR} govector_log=${GOVECTOR_LOG_DIR}"

n_data=$(echo "$DATASETS" | wc -w)
n_modes=$(echo "$MODES" | wc -w)
n_thr=$(echo "$THREADS" | wc -w)
estimated=0
if _exp_search_system_enabled dgai; then
    estimated=$((estimated + n_data * n_modes * n_thr))
fi
if _exp_search_system_enabled govector; then
    estimated=$((estimated + n_data * n_thr))
fi
echo "Estimated runs: ${estimated}"

if _exp_search_system_enabled dgai; then
    build_dgai_search

    for data_name in $DATASETS; do
        for search_mode in $MODES; do
            for num_threads in $THREADS; do
                run_dgai_search "$data_name" "$search_mode" "$num_threads" "$LOG_DIR" || {
                    if [ "$CONTINUE_ON_ERROR" != "1" ]; then
                        exit 1
                    fi
                }
            done
        done
    done
fi

if _exp_search_system_enabled govector; then
    build_govector_search

    for data_name in $DATASETS; do
        for num_threads in $THREADS; do
            run_govector_search "$data_name" "$num_threads" "$GOVECTOR_LOG_DIR" || {
                if [ "$CONTINUE_ON_ERROR" != "1" ]; then
                    exit 1
                fi
            }
        done
    done
fi

echo "=== Done: ${EXP_NAME} (logs: ${LOG_DIR}, ${GOVECTOR_LOG_DIR}) ==="
