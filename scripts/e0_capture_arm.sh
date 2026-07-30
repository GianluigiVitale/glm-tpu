#!/bin/bash
# e0_capture_arm.sh <sparse|dense> — ONE E0 decode capture of one arm via
# the IN-WORKER tracer (GLM_JAX_TRACE, fc8669c78): the worker brackets 15
# decode steps with jax start/stop_trace and writes the trace locally —
# no server, no client, no timing race (every remote route failed on
# metal four distinct ways; RESEARCH_LOG 07-29/30). Traces pulled from
# all 8 hosts post-run. Refused draws redraw (x3).
set -u
ARM="${1:?sparse|dense}"
ZONE=us-central2-b
POD=db-v4-64-od
PIN=ab57f0791
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
TAG=e0cap_${ARM}_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
mkdir -p "$RUN_DIR"
say() { echo "[e0cap-$ARM $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

TRC='GLM_JAX_TRACE_DIR=/tmp/glm-jaxtrace GLM_FLIGHT_RECORDER=1 GLM_JAX_TRACE_SKIP=4 GLM_JAX_TRACE_STEPS=15'
LIBTPU='LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
if [ "$ARM" = "sparse" ]; then
  RENV='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_EXPECT_CODE_HASH='"$PIN"
  DEXTRA="GLM_STATE_HASH_REF=/tmp/golden.json GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2"
else
  RENV='GLM_MLA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_NAN_CHECK_IGNORE=.self_attn.indexer. GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_STATE_HASH_REF_IGNORE=.self_attn.indexer. GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_EXPECT_CODE_HASH='"$PIN"
  DEXTRA="GLM_STATE_HASH_REF=/tmp/golden.json GLM_STATE_HASH_REF_IGNORE=.self_attn.indexer. GLM_LOAD_NAN_CHECK_IGNORE=.self_attn.indexer. GLM_MLA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop"
fi
DRIVER="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_WK_OOB_DIR=$OOB_DIR GLM_EXPECT_CODE_HASH=$PIN $DEXTRA"
REFUSAL_RE="StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|CodeFingerprintMismatchError"

for try in 1 2 3 4 5 6; do
  # after a crashed try: full ray reset + settle, else the next launch hits
  # the SliceBuilder wedge (leaked-engine landmine)
  if [ "$try" -gt 1 ]; then
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' >/dev/null 2>&1
    sleep 45
  fi
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='rm -rf /tmp/glm-jaxtrace; test -r ~/gcs-models/models/GLM-5.2-FP8/model.safetensors.index.json || { fusermount -u ~/gcs-models 2>/dev/null; sleep 1; gcsfuse --implicit-dirs -o ro driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1; }' >/dev/null 2>&1
  EXTRA_ENVS="$RENV $TRC $LIBTPU" TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch_t${try}.log" 2>&1
  LOG="$RUN_DIR/driver_t${try}.log"
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER setsid --wait nohup ~/vllm-env/bin/python -u dsa_throughput.py \
      --ctxs 262144 --num-seqs 1 --measure-tokens 256 --gmu 0.90 \
      --max-batched-tokens 2048 --num-gpu-blocks 66 --max-len 262400 \
      --note "E0 capture $ARM ($TAG)" </dev/null > "$LOG" 2>&1
    echo "DRIVER_EXIT=$?" >> "$LOG"
  ) &
  W=$!
  wait "$W" 2>/dev/null
  mkdir -p "$RUN_DIR/trace"
  for w in 0 1 2 3 4 5 6 7; do
    gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$w" --recurse \
      "$POD:/tmp/glm-jaxtrace" "$RUN_DIR/trace/w$w" >/dev/null 2>&1 || true
  done
  N=$(find "$RUN_DIR/trace" -name "*.xplane.pb" 2>/dev/null | wc -l)
  if grep -q "DRIVER_EXIT=0" "$LOG" && [ "$N" -gt 0 ]; then
    say "SUCCESS: $N xplane file(s); driver ok"
    gcloud storage cp -r "$RUN_DIR/trace" "gs://driftbench-dsv4-uc/results/$TAG/" >/dev/null 2>&1 || true
    exit 0
  fi
  if grep -aqE "$REFUSAL_RE" "$LOG"; then say "try $try: draw REFUSED — redraw"; continue; fi
  say "try $try: driver_ok=$(grep -c 'DRIVER_EXIT=0' "$LOG") traces=$N — relaunching (fresh server)"
done
say "FAILED in 6 tries"
exit 1
