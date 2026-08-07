#!/usr/bin/env bash
# Capture the exact all-event DSA oracle for the pinned 8K token trajectory.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh" "$@"
