#!/bin/bash
# bench_run.sh — benchmark orchestrator on the VALIDATED serving stack (the
# ratified 128K gate config: dcp=4 sparse + the full protection stack
# REF+OOB+NaN+checksum). One benchmark run per invocation, engine drawn
# fresh + health-probed, driver retried on REFUSED draws (a refusal is a
# protection event, not a result). results.db provenance is written by
# run_bench.py itself; this wrapper adds the engine discipline + GCS ckpt.
#
# Usage: bash bench_run.sh <benchmark> <limit> <max_new> [extra run_bench args...]
#   e.g. bash bench_run.sh gsm8k 200 2048
#        bash bench_run.sh aime_2026 30 16384 --max-len 32768
set -u

ZONE=us-central2-b
POD=db-v4-64-od
PIN=94b746433
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
BENCH="${1:?usage: bench_run.sh <benchmark> <limit> <max_new> [extra args]}"
LIMIT="${2:?limit}"
MAX_NEW="${3:?max_new}"
shift 3
TAG=bench_${BENCH}_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
GCS_CKPT=gs://driftbench-dsv4-uc/results/$TAG
MAX_LEN="${MAX_LEN:-8192}"
MAX_SEQS="${MAX_SEQS:-8}"
BLOCKS="${BLOCKS:-32}"
HEALTH_RETRIES="${HEALTH_RETRIES:-8}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-43200}"

# The ratified gate serving config (dcp=4 sparse), protections armed.
RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 \
GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR=$OOB_DIR GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH=$PIN"

mkdir -p "$RUN_DIR"
ALERT_FILE="$RUN_DIR/DISK_ALERT"
say() { echo "[bench-$BENCH $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

# 0) preflight: serialized pod + golden + oob mounts + pin + disks
if pgrep -f "gate_sparse128k[.]sh|stage256k[.]sh|glm_longctx[.]py|dsa_throughput[.]py" >/dev/null 2>&1; then
  say "ABORT: another pod workload is running"; exit 1; fi
if [ -n "${GLM_STATE_HASH_WRITE:-}" ]; then say "ABORT: WRITE mode armed"; exit 1; fi
GOLDEN_OK=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='ls /tmp/golden.json >/dev/null 2>&1 && echo OK' 2>/dev/null | grep -c OK)
[ "${GOLDEN_OK:-0}" -eq 8 ] || { say "ABORT: golden missing on $((8 - ${GOLDEN_OK:-0}))/8"; exit 1; }
ensure_oob_mounts() {
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='mountpoint -q ~/gcs-models || gcsfuse --implicit-dirs -o ro --stat-cache-ttl 1h --type-cache-ttl 1h driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1; test -r '"$OOB_DIR"'/model.safetensors.index.json && echo OOB_OK' \
    2>/dev/null | grep -c OOB_OK
}
OOB_OK=$(ensure_oob_mounts)
[ "${OOB_OK:-0}" -eq 8 ] || { say "ABORT: oob mirror ${OOB_OK:-0}/8"; exit 1; }
HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && git rev-parse --short=9 HEAD' 2>/dev/null | grep -c "$PIN")
[ "${HASHES:-0}" -eq 8 ] || { say "ABORT: ${HASHES:-0}/8 @ $PIN"; exit 1; }
bash ~/glm-tpu/scripts/disk_watchdog.sh check | tee -a "$RUN_DIR/orchestrator.log" || {
  say "ABORT: disk pre-flight failed"; exit 1; }
ALERT_FILE="$ALERT_FILE" INTERVAL_S=120 setsid nohup \
  bash ~/glm-tpu/scripts/disk_watchdog.sh watch </dev/null \
  > "$RUN_DIR/disk_watch.log" 2>&1 &
WATCH_PID=$!
trap 'kill "$WATCH_PID" 2>/dev/null' EXIT

REFUSAL_RE="StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|CodeFingerprintMismatchError"

# engine draw + health probe (mini-needle at the bench geometry; no dumps)
launch_healthy() {
  local try=0
  while [ "$try" -lt "$HEALTH_RETRIES" ]; do
    try=$((try + 1))
    EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
      bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch_t${try}.log" 2>&1
    local oob_ok; oob_ok=$(ensure_oob_mounts)
    [ "${oob_ok:-0}" -eq 8 ] || { say "try $try: oob ${oob_ok:-0}/8 — relaunch"; continue; }
    local nodes; nodes=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
    [ "${nodes:-0}" -eq 8 ] || { say "try $try: ray ${nodes:-0}/8 — relaunch"; continue; }
    local hlog="$RUN_DIR/health_t${try}.log"
    (
      cd ~/glm-tpu/bench || exit 1
      # shellcheck disable=SC1090
      set -a; . ~/glm-tpu/.env; set +a
      # shellcheck disable=SC2086
      env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u glm_longctx.py \
        --lengths 5000 --depths 0.5 --trials 1 --max-seqs 1 --gmu 0.90 \
        --max-batched-tokens 2048 --num-gpu-blocks "$BLOCKS" --max-len "$MAX_LEN" \
        --note "bench health ($TAG)" </dev/null > "$hlog" 2>&1
    )
    local refusal
    refusal=$(grep -aoEm1 "$REFUSAL_RE" "$hlog" || true)
    if [ -n "$refusal" ]; then say "try $try: SICK:refused_${refusal} — relaunch"; continue; fi
    grep -q "manifest VERIFIED" "$hlog" || { say "try $try: SICK:no_verify — relaunch"; continue; }
    grep -q "correct=True" "$hlog" || { say "try $try: SICK:needle — relaunch"; continue; }
    say "try $try: HEALTHY"; return 0
  done
  say "ABORT: no healthy engine in $HEALTH_RETRIES draws"; return 1
}

# --ids in the extra args IS the selection (run_bench refuses --ids+--limit)
SELECTOR="--limit $LIMIT"
case " $* " in *" --ids "*) SELECTOR="" ;; esac

say "════ BENCH $BENCH n=$LIMIT max_new=$MAX_NEW on the gate config (dcp=4 sparse) ════"
launch_healthy || exit 1

for a in 1 2 3 4; do
  LOG="$RUN_DIR/bench_a${a}.log"
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u run_bench.py \
      --benchmark "$BENCH" $SELECTOR --max-new "$MAX_NEW" \
      --max-len "$MAX_LEN" --max-seqs "$MAX_SEQS" --max-batched-tokens 2048 \
      --num-gpu-blocks "$BLOCKS" --gmu 0.90 --protocol greedy \
      --note "$BENCH n=$LIMIT ($TAG)" "$@" </dev/null > "$LOG" 2>&1
    echo "DRIVER_EXIT=$?" >> "$LOG"
  ) &
  W=$!; WAITED=0
  while kill -0 "$W" 2>/dev/null; do
    sleep 60; WAITED=$((WAITED + 60))
    [ -s "$ALERT_FILE" ] && { say "ABORT: DISK ALERT"; pkill -f "$TAG"; exit 1; }
    [ "$WAITED" -ge "$BENCH_TIMEOUT_S" ] && { say "ABORT: timeout"; pkill -f "$TAG"; exit 1; }
  done
  wait "$W" 2>/dev/null
  if grep -q "DRIVER_EXIT=0" "$LOG"; then
    gcloud storage cp "$LOG" "$GCS_CKPT/" >/dev/null 2>&1
    ~/vllm-env/bin/python - <<PY
import sqlite3
src = sqlite3.connect('$HOME/glm-tpu/bench/results.db')
dst = sqlite3.connect('$RUN_DIR/results_ckpt.db')
src.backup(dst); dst.close(); src.close()
PY
    gcloud storage cp "$RUN_DIR/results_ckpt.db" "$GCS_CKPT/" >/dev/null 2>&1
    say "════ BENCH $BENCH COMPLETE (attempt $a) — summary in results.db; log ckpt'd ════"
    exit 0
  fi
  if grep -aqE "$REFUSAL_RE" "$LOG"; then
    say "attempt $a: engine draw REFUSED ($(grep -aoEm1 "$REFUSAL_RE" "$LOG")) — fresh draw"
    launch_healthy || exit 1
    continue
  fi
  say "ABORT: NON-refusal driver failure (see $LOG)"; exit 1
done
say "ABORT: no clean bench run in 4 attempts"; exit 1
