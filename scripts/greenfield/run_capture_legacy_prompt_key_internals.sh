#!/usr/bin/env bash
# Capture accepted prompt-key FP32 producer state at the first cache mismatch.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly TAG=${GLM_GREENFIELD_PROMPT_KEY_INTERNALS_TAG:-greenfield_legacy_layer0_prompt_key_internals_p113_$(date -u +%Y%m%dT%H%M%S%NZ)}

export GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=1
export GLM_GREENFIELD_DSA_INTERNALS_MODE=prompt_key
export GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=0
export GLM_GREENFIELD_DSA_INTERNALS_POSITION=113
export GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=1
export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
export GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG=$TAG
export GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX=gs://driftbench-dsv4-uc/oracles/greenfield/glm52/prompt_key_internals/8k/$TAG
exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh" "$@"
