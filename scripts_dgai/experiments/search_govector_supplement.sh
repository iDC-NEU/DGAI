#!/bin/bash
# Run the GoVector-only search supplement:
#   baseline search, scalability search, and thread-scaling search.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_SEARCH_SYSTEMS="${RUN_SEARCH_SYSTEMS:-govector}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

echo "=== GoVector supplement: search baseline ==="
"$SCRIPT_DIR/search_baseline.sh"

echo "=== GoVector supplement: search scalability ==="
export SCALE_DATASETS="${GOVECTOR_SCALE_DATASETS:-deep10m deep100m}"
"$SCRIPT_DIR/search_scalability.sh"

echo "=== GoVector supplement: search threads ==="
"$SCRIPT_DIR/search_threads.sh"

echo "=== Done: GoVector search supplement ==="
