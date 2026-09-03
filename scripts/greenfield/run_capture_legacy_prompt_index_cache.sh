#!/usr/bin/env bash
# Capture one full-indexer layer's accepted 8K prompt index cache with the
# existing legacy-oracle protections. GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID
# selects the layer (default 0); the legacy cache slot is derived downstream.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly PROMPT_CACHE_LAYER_ID=${GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID:-0}
[[ $PROMPT_CACHE_LAYER_ID =~ ^[0-9]+$ ]] || {
  echo "GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID must be a nonnegative integer" >&2
  exit 2
}
readonly TAG=${GLM_GREENFIELD_PROMPT_CACHE_TAG:-greenfield_legacy_layer${PROMPT_CACHE_LAYER_ID}_prompt_index_cache_$(date -u +%Y%m%dT%H%M%S%NZ)}

export GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=1
export GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID=$PROMPT_CACHE_LAYER_ID
# oracle (per-host accepted checkout) or layer1_observer_bundle (recreated pod).
export GLM_GREENFIELD_PROMPT_CACHE_RUNTIME=${GLM_GREENFIELD_PROMPT_CACHE_RUNTIME:-oracle}
export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
export GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG=$TAG
export GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX=gs://driftbench-dsv4-uc/oracles/greenfield/glm52/prompt_index_cache/8k/$TAG
exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh" "$@"
