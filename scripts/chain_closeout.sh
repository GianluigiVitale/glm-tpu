#!/bin/bash
# chain_closeout.sh — the pod close-out: AIME retry (chunked commits — the
# 08-03 00:26 timeout kill lost a finished-in-all-but-drain run to the
# one-big-generate design; --batch-size makes completed chunks durable) →
# MTP M2 → GPQA-49 retry @32K (chunked, 2-day budget).
set -u
LOG=~/glm-run/chain_closeout.log
say() { echo "[closeout $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
while pgrep -f "run_bench[.]py|mtp_m2_check[.]py" >/dev/null 2>&1; do sleep 60; done
gcloud compute tpus tpu-vm ssh db-v4-64-od --zone us-central2-b --worker=all \
  --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' >/dev/null 2>&1
sleep 60
say "════ AIME retry (11 ids @ 32K, batch-size 4) ════"
MAX_LEN=40960 MAX_SEQS=4 BLOCKS=80 BENCH_TIMEOUT_S=86400 \
  bash ~/glm-tpu/scripts/bench_run.sh aime_2026 30 32768 \
  --ids aime_9,aime_12,aime_13,aime_14,aime_16,aime_17,aime_22,aime_23,aime_27,aime_28,aime_29 \
  --batch-size 4
say "AIME retry rc=$?"
say "════ MTP M2 greedy-equivalence (Stage-3) ════"
gcloud compute tpus tpu-vm ssh db-v4-64-od --zone us-central2-b --worker=all \
  --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' >/dev/null 2>&1
sleep 60
EXTRA_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8 GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH=94b746433 GLM_SPEC_K=2 LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"' \
  TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh > ~/glm-run/mtp_m2_launch.log 2>&1
(
  cd ~/glm-tpu/bench || exit 1
  set -a; . ~/glm-tpu/.env; set +a
  env NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
    OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
    DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
    TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_MLA_DCP=1 \
    GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop \
    GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
    GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
    GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 \
    GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8 \
    GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH=94b746433 GLM_SPEC_K=2 \
    setsid --wait nohup ~/vllm-env/bin/python -u mtp_m2_check.py \
    --max-len 8192 --num-gpu-blocks 32 --gmu 0.90 \
    </dev/null > ~/glm-run/mtp_m2_run.log 2>&1
  echo "MTP_EXIT=$?" >> ~/glm-run/mtp_m2_run.log
)
say "MTP M2 done: $(tail -1 ~/glm-run/mtp_m2_run.log)"
say "════ GPQA-49 retry (@32K, batch-size 8) ════"
gcloud compute tpus tpu-vm ssh db-v4-64-od --zone us-central2-b --worker=all \
  --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' >/dev/null 2>&1
sleep 60
GPQA_IDS=$(cat /tmp/claude-2001/-home-gianl/4c2f7561-156a-465d-9c66-89ce7d69a1d4/scratchpad/gpqa_trunc_ids.txt)
MAX_LEN=40960 MAX_SEQS=4 BLOCKS=80 BENCH_TIMEOUT_S=172800 \
  bash ~/glm-tpu/scripts/bench_run.sh gpqa_diamond 198 32768 \
  --ids "$GPQA_IDS" --batch-size 8
say "GPQA retry rc=$?"
say "closeout chain COMPLETE"
