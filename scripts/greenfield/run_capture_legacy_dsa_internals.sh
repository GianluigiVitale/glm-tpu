#!/usr/bin/env bash
# Capture one accepted 8K DSA scorer boundary with exact protections.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly LAYER_ID=${GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID:-0}
if [[ ! $LAYER_ID =~ ^[0-9]+$ ]] ||
  ! ((LAYER_ID <= 2 || (LAYER_ID >= 6 && LAYER_ID < 78 && (LAYER_ID - 6) % 4 == 0))); then
  echo "DSA internal layer must be a full-indexer producer" >&2
  exit 2
fi
readonly TAG=${GLM_GREENFIELD_DSA_INTERNALS_TAG:-greenfield_legacy_layer${LAYER_ID}_dsa_internals_$(date -u +%Y%m%dT%H%M%S%NZ)}

export GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=1
export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
export GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG=$TAG
export GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX=gs://driftbench-dsv4-uc/oracles/greenfield/glm52/dsa_internals/8k/$TAG
exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh" "$@"
