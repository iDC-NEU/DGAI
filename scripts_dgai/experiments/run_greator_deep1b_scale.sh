#!/bin/bash
# Narrow retry for the missing Greator deep1b update-scalability run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
export RUN_UPDATE_SYSTEMS="${RUN_UPDATE_SYSTEMS:-greator}"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

build_greator
run_greator_update deep1b 0.001 "$(log_dir_for greator update_scalability)"
