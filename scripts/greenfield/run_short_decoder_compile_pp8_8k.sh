#!/usr/bin/env bash
# Protected PP8 short decoder under the sealed 8K token/DSA profile.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
export GLM_GREENFIELD_SHORT_DECODER_PROFILE=8k
exec bash "$WORKTREE/scripts/greenfield/run_short_decoder_compile_pp8.sh" "$@"
