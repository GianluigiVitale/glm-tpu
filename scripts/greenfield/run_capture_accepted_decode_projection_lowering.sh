#!/usr/bin/env bash
# Capture the accepted M32 decode projection's final physical collective HLO.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly TAG=${GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_TAG:-greenfield_accepted_decode_projection_lowering_$(date -u +%Y%m%dT%H%M%S%NZ)}
readonly REMOTE_PREFIX=${GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_REMOTE_PREFIX:-gs://driftbench-dsv4-uc/oracles/greenfield/glm52/decode_projection_lowering/8k/$TAG}

export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
export GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG=$TAG
export GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX=$REMOTE_PREFIX
export GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=0
export GLM_GREENFIELD_DSA_INTERNALS_MODE=scorer
export GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=0
export GLM_GREENFIELD_DSA_INTERNALS_POSITION=
export GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=0
export GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE=0
export GLM_GREENFIELD_MAIN_CACHE_CAPTURE=0
export GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_CAPTURE=1

exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
