#!/bin/bash
# Master driver for DGAI search / update experiments.
#
# Default order (all):
#   prep -> search_scalability -> update_scalability
#        -> search_threads (text2img) -> update_threads (text2img)
#        -> update_batch_size (text2img) -> search_baseline -> update_baseline
#
# Usage:
#   ./run_all.sh                         # prep missing indexes, then all experiments
#   ./run_all.sh prep                    # only Starling + Greator sidecar prep
#   ./run_all.sh search                  # search sweeps (excl. baseline; still runs prep first)
#   ./run_all.sh update                  # update sweeps
#   ./run_all.sh baseline                # search_baseline + update_baseline
#   SKIP_PREP=1 ./run_all.sh             # skip prep step
#   SKIP_EXISTING=0 ./run_all.sh         # force re-run even if logs exist (default SKIP_EXISTING=1)
#   CONTINUE_ON_ERROR=1 ./run_all.sh
#   RUN_UPDATE_SYSTEMS="odin dgai" ./run_all.sh update_batch_size
#   THREAD_DATASETS="text2img laion" ./run_all.sh search_threads
#   BATCH_DATASETS=text2img BATCH_RATIOS="0.001 0.01 0.1" ./run_all.sh update_batch_size

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SKIP_PREP="${SKIP_PREP:-0}"
FAILED=()

run_one() {
    local script="$1"
    local path="${SCRIPT_DIR}/${script}"
    if [ ! -x "$path" ] && [ -f "$path" ]; then
        chmod +x "$path"
    fi
    echo ""
    echo "################################################################"
    echo "# Starting: ${script}"
    echo "################################################################"
    if bash "$path"; then
        echo "# Finished OK: ${script}"
    else
        local rc=$?
        echo "# FAILED (${rc}): ${script}" >&2
        FAILED+=("${script} (rc=${rc})")
        if [ "$CONTINUE_ON_ERROR" != "1" ]; then
            exit "$rc"
        fi
    fi
}

maybe_prep() {
    if [ "$SKIP_PREP" = "1" ]; then
        echo "# SKIP_PREP=1 — not running prep_missing_indexes.sh"
        return 0
    fi
    run_one prep_missing_indexes.sh
}

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(all)
fi

for target in "${TARGETS[@]}"; do
    case "$target" in
        all)
            maybe_prep
            run_one search_scalability.sh
            run_one update_scalability.sh
            run_one search_threads.sh
            run_one update_threads.sh
            run_one update_batch_size.sh
            run_one search_baseline.sh
            run_one update_baseline.sh
            ;;
        prep|prep_missing_indexes)
            run_one prep_missing_indexes.sh
            ;;
        search)
            maybe_prep
            run_one search_scalability.sh
            run_one search_threads.sh
            ;;
        update)
            maybe_prep
            run_one update_scalability.sh
            run_one update_threads.sh
            run_one update_batch_size.sh
            ;;
        baseline|basic)
            maybe_prep
            run_one search_baseline.sh
            run_one update_baseline.sh
            ;;
        search_baseline)
            maybe_prep
            run_one search_baseline.sh
            ;;
        update_baseline)
            maybe_prep
            run_one update_baseline.sh
            ;;
        search_scalability)
            maybe_prep
            run_one search_scalability.sh
            ;;
        search_threads)
            maybe_prep
            run_one search_threads.sh
            ;;
        update_batch_size|update_batch)
            maybe_prep
            run_one update_batch_size.sh
            ;;
        update_threads)
            maybe_prep
            run_one update_threads.sh
            ;;
        update_scalability)
            maybe_prep
            run_one update_scalability.sh
            ;;
        *)
            echo "Unknown target: ${target}" >&2
            echo "Valid: all | prep | search | update | baseline | search_baseline | update_baseline |" >&2
            echo "       search_scalability | search_threads |" >&2
            echo "       update_batch_size | update_threads | update_scalability" >&2
            exit 1
            ;;
    esac
done

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "=== Completed with failures ==="
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
echo "=== All requested experiments finished successfully ==="
