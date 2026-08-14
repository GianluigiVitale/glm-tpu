#!/usr/bin/env bash
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly TAG=${GLM_GREENFIELD_DENSE_BOUNDARY_CAPTURE_TAG:-greenfield_legacy_layer0_dense_boundary_p8155_$(date -u +%Y%m%dT%H%M%S%NZ)}

export GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG=$TAG
export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
export GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=1
# Residual-only v2 avoids materializing the dense output and perturbing fusion.
export GLM_GREENFIELD_DSA_INTERNALS_MODE=dense_boundary
export GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=0
export GLM_GREENFIELD_DSA_INTERNALS_POSITION=8155
export GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=0
export GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE=0
export GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_CAPTURE=0
export GLM_GREENFIELD_MAIN_CACHE_CAPTURE=0
export GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX=gs://driftbench-dsv4-uc/oracles/greenfield/glm52/dense_boundary/8k/$TAG

exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
