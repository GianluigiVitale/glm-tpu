#!/bin/bash
# Protected metal exactness A/B for GLM_DSA_DCP_DECODE_LIVE_SCORE_ROWS. The
# shared harness fixes the accepted psum/fusion and DCP live-attention stack
# on, and fixes the rejected MoE all-gather plus unaccepted compute-row gates
# off in both arms.
set -uo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
export EXACT_LEVER=dcp_score_rows
export EXACT_PIN="${EXACT_PIN:-13bfbca3a}"
if [ -n "${DCP_SCORE_ROWS_EXACT_RESUME_DIR:-}" ]; then
  export EXACT_RESUME_DIR="$DCP_SCORE_ROWS_EXACT_RESUME_DIR"
fi
exec bash "$SCRIPT_ROOT/scripts/dcp_live_rows_exact.sh"
