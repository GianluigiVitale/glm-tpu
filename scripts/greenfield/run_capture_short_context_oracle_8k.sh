#!/usr/bin/env bash
# Seal the pinned accepted run-104/item-506 8K passkey result.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
export GLM_GREENFIELD_SHORT_CONTEXT_ORACLE_PROFILE=8k
exec "$WORKTREE/scripts/greenfield/run_capture_short_context_oracle.sh" "$@"
