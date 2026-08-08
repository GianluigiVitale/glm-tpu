#!/usr/bin/env bash
# Compatibility entry point for the accepted layer-0 DSA comparison.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
export GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=0
exec bash "$WORKTREE/scripts/greenfield/run_capture_legacy_dsa_internals.sh" "$@"
