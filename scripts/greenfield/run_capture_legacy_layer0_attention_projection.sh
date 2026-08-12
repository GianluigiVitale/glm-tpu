#!/usr/bin/env bash
# Capture accepted layer-0 operands on both sides of W_UV at position 8155.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly TAG=${GLM_GREENFIELD_ATTENTION_PROJECTION_CAPTURE_TAG:-greenfield_legacy_layer0_attention_projection_p8155_$(date -u +%Y%m%dT%H%M%S%NZ)}

export GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=1
export GLM_GREENFIELD_DSA_INTERNALS_MODE=attention_projection
export GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=0
export GLM_GREENFIELD_DSA_INTERNALS_POSITION=8155
export GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=0
export GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE=0
export GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_CAPTURE=0
export GLM_GREENFIELD_MAIN_CACHE_CAPTURE=0
export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
export GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG=$TAG
export GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX=gs://driftbench-dsv4-uc/oracles/greenfield/glm52/attention_projection/8k/$TAG

exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
