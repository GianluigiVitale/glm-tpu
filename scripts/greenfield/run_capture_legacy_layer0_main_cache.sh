#!/usr/bin/env bash
# Capture only the final-prefill and first-decode legacy layer-0 main cache.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly TAG=${GLM_GREENFIELD_MAIN_CACHE_TAG:-greenfield_legacy_layer0_main_cache_$(date -u +%Y%m%dT%H%M%S%NZ)}

export GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k
export GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=0
export GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=0
export GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE=0
export GLM_GREENFIELD_MAIN_CACHE_CAPTURE=1
export GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG=$TAG
export GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX=gs://driftbench-dsv4-uc/oracles/greenfield/glm52/layer0_main_cache/8k/$TAG

exec bash "$WORKTREE/scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
