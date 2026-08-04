#!/bin/bash
# Protected metal exactness A/B for GLM_MOE_DECODE_ALL_GATHER. The shared
# harness keeps the accepted psum stack and parent DCP live-attention lever on
# in both arms; only the MoE collective implementation changes.
set -uo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
export EXACT_LEVER=moe_allgather
export EXACT_PIN="${EXACT_PIN:-5967dffa4}"
if [ -n "${MOE_ALLGATHER_EXACT_RESUME_DIR:-}" ]; then
  export EXACT_RESUME_DIR="$MOE_ALLGATHER_EXACT_RESUME_DIR"
fi
exec bash "$SCRIPT_ROOT/scripts/dcp_live_rows_exact.sh"
