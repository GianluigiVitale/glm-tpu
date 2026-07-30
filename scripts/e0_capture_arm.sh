#!/bin/bash
# e0_capture_arm.sh <sparse|dense> — ONE E0 decode capture of one arm.
# Lessons baked in (RESEARCH_LOG 07-29/30): the jax profiler server serves
# ONE profile session per lifetime — never pre-test it; one 8s capture per
# engine, timed into pass B's decode window via the PASSA/PASSB markers.
# Refused draws redraw (x3); a failed capture = relaunch the engine (fresh
# server), not a second client attempt against a spent server.
set -u
ARM="${1:?sparse|dense}"
ZONE=us-central2-b
POD=db-v4-64-od
PIN=4647a8fbc
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
TAG=e0cap_${ARM}_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
mkdir -p "$RUN_DIR"
say() { echo "[e0cap-$ARM $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

SRV='USE_JAX_PROFILER_SERVER=1 JAX_PROFILER_SERVER_PORT=9999'
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

for try in 1 2 3; do
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='test -r ~/gcs-models/models/GLM-5.2-FP8/model.safetensors.index.json || { fusermount -u ~/gcs-models 2>/dev/null; sleep 1; gcsfuse --implicit-dirs -o ro driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1; }' >/dev/null 2>&1
  EXTRA_ENVS="$RENV $SRV $LIBTPU" TPU_MIN_TOKEN_BUCKET=32 \
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
  (
    ta=""
    while kill -0 "$W" 2>/dev/null; do
      ta=$(grep -aom1 "PASSA ctx=262144 wall=[0-9.]*" "$LOG" 2>/dev/null | grep -o "[0-9.]*$" || true)
      [ -n "$ta" ] && break; sleep 20
    done
    [ -z "$ta" ] && exit 0
    while kill -0 "$W" 2>/dev/null; do
      grep -aq "PASSB-START" "$LOG" 2>/dev/null && break; sleep 5
    done
    sleep $(printf '%.0f' "$(echo "$ta + 90" | bc)")
    say "single-shot 8s capture (T_A=${ta}s)"
    ~/vllm-env/bin/python -m jax.collect_profile 9999 8000 \
      --log_dir "$RUN_DIR/trace" --no_perfetto_link --host localhost \
      > "$RUN_DIR/capture_t${try}.log" 2>&1 || true
  ) &
  CAP=$!
  wait "$W" 2>/dev/null; wait "$CAP" 2>/dev/null
  N=$(find "$RUN_DIR/trace" -name "*.xplane.pb" 2>/dev/null | wc -l)
  if grep -q "DRIVER_EXIT=0" "$LOG" && [ "$N" -gt 0 ]; then
    say "SUCCESS: $N xplane file(s); driver ok"
    gcloud storage cp -r "$RUN_DIR/trace" "gs://driftbench-dsv4-uc/results/$TAG/" >/dev/null 2>&1 || true
    exit 0
  fi
  if grep -aqE "$REFUSAL_RE" "$LOG"; then say "try $try: draw REFUSED — redraw"; continue; fi
  say "try $try: driver_ok=$(grep -c 'DRIVER_EXIT=0' "$LOG") traces=$N — relaunching (fresh server)"
done
say "FAILED in 3 tries"
exit 1
