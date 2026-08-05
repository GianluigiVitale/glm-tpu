#!/usr/bin/env bash
# Protected PP16 entry point; the shared harness remains plan-generic.
set -euo pipefail

export GLM_GREENFIELD_REAL_LAYER_PLAN=PP16_LP2
exec "$(dirname "$0")/run_real_one_layer_pp8.sh" "$@"
