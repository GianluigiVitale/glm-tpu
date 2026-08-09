#!/usr/bin/env bash
# Capture the accepted layer-0 FP32 row entering prompt-key projection.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly TAG=${GLM_GREENFIELD_PROMPT_PROJECTION_INPUT_TAG:-greenfield_legacy_layer0_prompt_projection_input_p113_$(date -u +%Y%m%dT%H%M%S%NZ)}

export GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=1
export GLM_GREENFIELD_DSA_INTERNALS_MODE=prompt_key_input
export GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=0
export GLM_GREENFIELD_DSA_INTERNALS_POSITION=113
export GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=1
export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
export GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG=$TAG
export GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX=gs://driftbench-dsv4-uc/oracles/greenfield/glm52/prompt_projection_input/8k/$TAG

exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
