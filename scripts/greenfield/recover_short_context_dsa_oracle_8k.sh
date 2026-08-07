#!/usr/bin/env bash
# Recover and seal the protected 8K all-event DSA capture.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
export GLM_GREENFIELD_SHORT_DSA_RECOVERY_PROFILE=8k
exec bash "$WORKTREE/scripts/greenfield/recover_short_context_dsa_oracle.sh" "$@"
