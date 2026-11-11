#!/bin/bash
# Narrow rerun: only Greator msmarco update_base.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

echo_exp_banner "greator_msmarco_update_base_only" \
    "datasets=[msmarco] systems=[greator] ratio=0.001 log_exp=update_base"

build_greator
run_greator_update msmarco 0.001 "$(log_dir_for greator update_base)"

echo "=== Done: run_greator_msmarco_update_base ==="
