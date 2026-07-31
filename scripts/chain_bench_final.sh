#!/bin/bash
# chain_bench_final.sh — goal-2 close-out: GPQA batched (the 4-seq config
# would take ~37h; the TP-32 collective floor amortizes with batch, so
# 8 seqs ~doubles aggregate tok/s) then the AIME truncation retry (all 11
# misses were 16K-cap truncations; 0 completed-wrong).
set -u
LOG=~/glm-run/chain_bench_final.log
say() { echo "[benchfinal $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
while pgrep -f "run_bench[.]py" >/dev/null 2>&1; do sleep 60; done
gcloud compute tpus tpu-vm ssh db-v4-64-od --zone us-central2-b --worker=all \
  --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' >/dev/null 2>&1
sleep 60
say "════ GPQA-198 @16K (batched: 8 seqs) ════"
MAX_LEN=20480 MAX_SEQS=8 BLOCKS=80 BENCH_TIMEOUT_S=86400 \
  bash ~/glm-tpu/scripts/bench_run.sh gpqa_diamond 198 16384
say "GPQA rc=$?"
say "════ AIME truncation retry (11 ids @ 32K) ════"
MAX_LEN=40960 MAX_SEQS=4 BLOCKS=80 BENCH_TIMEOUT_S=43200 \
  bash ~/glm-tpu/scripts/bench_run.sh aime_2026 30 32768 \
  --ids aime_9,aime_12,aime_13,aime_14,aime_16,aime_17,aime_22,aime_23,aime_27,aime_28,aime_29
say "AIME retry rc=$?"
say "bench chain COMPLETE"
