#!/bin/bash
# chain_night_0731.sh — the zero-gap pipeline (detached; the 07-31 lesson:
# pod work must never wait on the operator). Sequence:
#   1. wait for the running dense capture arm
#   2. LAND oob-manifest-repair (ff-merge -> push -> hardened sync) — the
#      sync happens BETWEEN workloads (never mid-arm); branch is tested
#      71/71 and is a strict descendant of the -next tip
#   3. arm GLM_WK_OOB_GOLDEN in bench_run + new PIN (wk strikes self-repair)
#   4. AIME-2026 n=30
#   5. GPQA-198 @16K
set -u
LOG=~/glm-run/chain_night_0731.log
say() { echo "[night $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for the dense capture arm..."
while pgrep -f "e0_capture_arm[.]sh|dsa_throughput[.]py" >/dev/null 2>&1; do sleep 60; done
say "pod free — landing oob-manifest-repair"

cd ~/tpu-inference || exit 1
git fetch -q origin
if git merge --ff-only origin/oob-manifest-repair >/dev/null 2>&1; then
  git push -q origin glm-5.2-v4-next
  NEWPIN=$(git rev-parse --short=9 HEAD)
  say "landed: -next @ $NEWPIN"
  if TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash ~/glm-tpu/scripts/sync_workers.sh 2>&1 | tail -1 | grep -q "sync VERIFIED"; then
    say "sync VERIFIED 8x @ $NEWPIN"
    OLDPIN=$(grep -m1 "^PIN=" ~/glm-tpu/scripts/bench_run.sh | cut -d= -f2)
    sed -i "s/^PIN=$OLDPIN/PIN=$NEWPIN/" ~/glm-tpu/scripts/bench_run.sh
    grep -q "GLM_WK_OOB_GOLDEN" ~/glm-tpu/scripts/bench_run.sh || sed -i \
      "s|GLM_WK_OOB_DIR='\"\$OOB_DIR\"'|GLM_WK_OOB_DIR='\"\$OOB_DIR\"' GLM_WK_OOB_GOLDEN=/tmp/golden.json|; s|GLM_WK_OOB_DIR=\$OOB_DIR|GLM_WK_OOB_DIR=\$OOB_DIR GLM_WK_OOB_GOLDEN=/tmp/golden.json|" \
      ~/glm-tpu/scripts/bench_run.sh
    cd ~/glm-tpu && git add scripts/bench_run.sh && \
      git commit -qm "bench_run: PIN -> $NEWPIN (manifest-driven wk repair landed); GLM_WK_OOB_GOLDEN armed" && \
      git push -q origin main
    say "bench_run armed with manifest-driven repair"
  else
    say "WARN: sync verify FAILED — benchmarks continue on the PREVIOUS pin"
    cd ~/tpu-inference && git reset --hard "$(git rev-parse origin/glm-5.2-v4-next)" >/dev/null 2>&1 || true
  fi
else
  say "WARN: ff-merge not possible — skipping the land; benchmarks on the current pin"
fi

say "════ AIME-2026 n=30 ════"
MAX_LEN=32768 MAX_SEQS=4 BLOCKS=64 BENCH_TIMEOUT_S=43200 \
  bash ~/glm-tpu/scripts/bench_run.sh aime_2026 30 16384
say "AIME rc=$?"

say "════ GPQA-198 @16K ════"
MAX_LEN=32768 MAX_SEQS=4 BLOCKS=64 BENCH_TIMEOUT_S=64800 \
  bash ~/glm-tpu/scripts/bench_run.sh gpqa_diamond 198 16384
say "GPQA rc=$?"
say "night chain COMPLETE"
