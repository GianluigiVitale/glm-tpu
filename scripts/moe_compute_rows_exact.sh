#!/bin/bash
# Protected metal exactness A/B for GLM_MOE_DECODE_COMPUTE_LIVE_ROWS. The
# shared harness fixes the accepted psum stack and DCP live-attention gate on,
# and fixes the performance-rejected MoE all-gather gate off in both arms.
set -uo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
export EXACT_LEVER=moe_compute_rows
export EXACT_PIN="${EXACT_PIN:-b3c25df47}"
if [ -n "${MOE_COMPUTE_ROWS_EXACT_RESUME_DIR:-}" ]; then
  export EXACT_RESUME_DIR="$MOE_COMPUTE_ROWS_EXACT_RESUME_DIR"
fi
exec bash "$SCRIPT_ROOT/scripts/dcp_live_rows_exact.sh"
