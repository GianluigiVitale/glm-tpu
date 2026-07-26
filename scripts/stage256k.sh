#!/bin/bash
# stage256k.sh — THE 256K STAGE orchestrator (runs ONLY after the 128K gate
# verdict banks; NEVER concurrently with it — TPU access is serialized).
#
# Plan (RESEARCH_LOG 2026-07-18 19:50; PLAN Stage-3 threshold "measurable
# FLOP/throughput gain at >=256K"):
#   A. dcp=8 bring-up (the 07-12 ratified deferral): per-rank KV at 256K/dcp=8
#      == the PROVEN 128K/dcp=4 fit (32K tokens/rank), so capacity is by
#      construction; block granularity 512*8=4096 tok/entry => 66 blocks >=
#      270336 tok >= max_len 262400. Novel geometry => ~40min cold compile,
#      absorbed by the health mini-needle (which compiles prefill+decode).
#   B. SANITY: 32K needle x2 at dcp=8 (sparse-DCP has ZERO metal tokens at
#      dcp=8 — CPU-certified only; a cheap cell catches striping/LSE breakage
#      before a 25-min 256K prefill does).
#   C. 256K SMOKE: mechanism depths 0.0/0.5/0.95/1.0 x1 (NOT a gate — the 256K
#      criterion is THROUGHPUT; passkey smoke guards interpretability).
#   D. THROUGHPUT A/B at IDENTICAL dcp=8: bench/dsa_throughput.py L=262144,
#      sparse arm then dense arm (fresh engine per arm — one variable).
#
# Protections: full integrity stack armed (PWAL/LOAD NaN + GLM_LOAD_CHECKSUM),
# health probe per engine, disk pre-flight + watch, per-stage db notes. Any
# stage failing => STOP (abort discipline — no tuning past a failure).
set -u

ZONE=us-central2-b
POD=db-v4-64-od
PIN=4647a8fbc
# wk-oob repair source (docs/17 §5.5) — gcsfuse ro mirror; preflight + per-launch remount
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
TAG=stage256k_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
GCS_CKPT=gs://driftbench-dsv4-uc/results/$TAG
MAX_LEN=262400
BLOCKS=66
MEASURE_TOKENS="${MEASURE_TOKENS:-64}"
HEALTH_RETRIES="${HEALTH_RETRIES:-8}"
STAGE_TIMEOUT_S="${STAGE_TIMEOUT_S:-14400}"   # 256K prefill ~25min/needle + cold-compile headroom

# ── env blocks: the gate config verbatim EXCEPT GLM_DCP=8 (F4: one source,
# every launch identical). Dense arm strips the three DSA-mode envs only.
SPARSE_RAYLET='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_DCP_CACHE_DUMP=/tmp/dcp_256health GLM_DCP_CACHE_DUMP_LAYERS=2 GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
# Dense verifies against THE golden manifest with CHECK-SIDE scoping
# (GLM_*_IGNORE=.self_attn.indexer.): the indexer is bypassed under
# TPU_DISABLE_DSA_INDEXER, and manifest editing cannot scope a config (the
# state-only rule flips the refusal — measured 07-26). Both ignore envs are
# UNSET on every correctness-gated config; the dense arm is a throughput
# baseline whose behavior is still gated by the health needle.
DENSE_RAYLET='GLM_MLA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_NAN_CHECK_IGNORE=.self_attn.indexer. GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_STATE_HASH_REF_IGNORE=.self_attn.indexer. GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_DCP_CACHE_DUMP=/tmp/dcp_256health GLM_DCP_CACHE_DUMP_LAYERS=2 GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
COMMON_DRIVER="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_MLA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 \
GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR=$OOB_DIR \
GLM_EXPECT_CODE_HASH=$PIN"
SPARSE_DRIVER="$COMMON_DRIVER GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 \
GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2"
DENSE_DRIVER="$COMMON_DRIVER GLM_STATE_HASH_REF=/tmp/golden.json GLM_STATE_HASH_REF_IGNORE=.self_attn.indexer. GLM_LOAD_NAN_CHECK_IGNORE=.self_attn.indexer."  # last env wins

mkdir -p "$RUN_DIR"
ALERT_FILE="$RUN_DIR/DISK_ALERT"
say() { echo "[stage256k $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

FROM_STAGE="A"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      say "DRY RUN — sparse EXTRA_ENVS=$SPARSE_RAYLET"
      say "DRY RUN — dense  EXTRA_ENVS=$DENSE_RAYLET"
      exit 0 ;;
    --from-stage) FROM_STAGE="$2"; shift 2 ;;  # A (default) | C | D | D2
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

# 0) pre-flight: THE GATE MUST NOT BE RUNNING (serialized TPU access)
if pgrep -f "gate_sparse128k[.]sh" >/dev/null 2>&1; then
  say "ABORT: gate_sparse128k.sh is still running — 256K never runs concurrently."; exit 1
fi
if [ -n "${GLM_STATE_HASH_WRITE:-}" ]; then
  say "ABORT: GLM_STATE_HASH_WRITE is set — WRITE mode must NEVER be armed at a stage run"; exit 1
fi
GOLDEN_OK=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='ls /tmp/golden.json >/dev/null 2>&1 && echo OK' 2>/dev/null | grep -c OK)
[ "${GOLDEN_OK:-0}" -eq 8 ] || { say "ABORT: golden manifest missing on $((8 - ${GOLDEN_OK:-0}))/8 hosts"; exit 1; }
ensure_oob_mounts() {
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='mountpoint -q ~/gcs-models || gcsfuse --implicit-dirs -o ro --stat-cache-ttl 1h --type-cache-ttl 1h driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1; test -r '"$OOB_DIR"'/model.safetensors.index.json && echo OOB_OK' \
    2>/dev/null | grep -c OOB_OK
}
OOB_OK=$(ensure_oob_mounts)
[ "${OOB_OK:-0}" -eq 8 ] || { say "ABORT: wk-oob gcsfuse mirror unreadable on $((8 - ${OOB_OK:-0}))/8 hosts ($OOB_DIR)"; exit 1; }
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

# helper: launch an arm's engine + health-probe it (mini-needle + NaN scan;
# the FIRST probe on a fresh geometry also pays the ~40min cold compile).
launch_healthy() {  # $1=RAYLET envs  $2=DRIVER envs  $3=label
  local renv="$1" denv="$2" label="$3" try=0
  while [ "$try" -lt "$HEALTH_RETRIES" ]; do
    try=$((try + 1))
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command='rm -f /tmp/dcp_256health*.npz' >/dev/null 2>&1
    EXTRA_ENVS="$renv" TPU_MIN_TOKEN_BUCKET=32 \
      bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch_${label}_t${try}.log" 2>&1
    # relaunch can drop the fuse mounts — re-ensure the wk-oob repair source
    local oob_ok
    oob_ok=$(ensure_oob_mounts)
    [ "${oob_ok:-0}" -eq 8 ] || { say "$label try $try: oob mirror ${oob_ok:-0}/8 — relaunch"; continue; }
    local nodes
    nodes=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
    [ "${nodes:-0}" -eq 8 ] || { say "$label try $try: ray ${nodes:-0}/8 — relaunch"; continue; }
    local hlog="$RUN_DIR/health_${label}_t${try}.log"
    (
      cd ~/glm-tpu/bench || exit 1
      # shellcheck disable=SC1090
      set -a; . ~/glm-tpu/.env; set +a
      # shellcheck disable=SC2086
      env $denv setsid --wait nohup ~/vllm-env/bin/python -u glm_longctx.py \
        --lengths 5000 --depths 0.5 --trials 1 --max-seqs 1 --gmu 0.90 \
        --max-batched-tokens 2048 --num-gpu-blocks "$BLOCKS" --max-len "$MAX_LEN" \
        --note "256K health ($label $TAG)" </dev/null > "$hlog" 2>&1
    )
    if ! grep -q "manifest VERIFIED" "$hlog"; then say "$label try $try: SICK:manifest_not_verified — relaunch"; continue; fi
    if ! grep -q "correct=True" "$hlog"; then say "$label try $try: SICK:needle — relaunch"; continue; fi
    local scan
    scan=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all --command='
      ~/vllm-env/bin/python - <<PY
import numpy as np, glob, sys
fs = sorted(glob.glob("/tmp/dcp_256health*.npz"))
if not fs: print("NOFILE"); sys.exit(0)
d = np.load(fs[-1], allow_pickle=True)
bad = 0
for k in d.files:
    if k.endswith("__data"):
        u = d[k].view(np.uint16)
        bad += int((((u & 0x7F80) == 0x7F80) & ((u & 0x007F) != 0)).sum())
print("NAN" if bad else "CLEAN", bad)
PY' 2>/dev/null)
    if [ "$(echo "$scan" | grep -c '^CLEAN')" -eq 8 ]; then say "$label try $try: HEALTHY"; return 0; fi
    say "$label try $try: SICK:scan($(echo "$scan" | tr '\n' ' ')) — relaunch"
  done
  say "ABORT: $label — no healthy engine in $HEALTH_RETRIES draws"; return 1
}

# A refused engine draw is a PROTECTION event, not a stage failure: the
# 07-26 D1 abort was a NaN-flavor refusal (PwalNanCheckError) killing the
# whole stage. Drivers spin their own engine (a fresh lottery draw), so
# every driver run needs the same retry the health loop has.
REFUSAL_RE="StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|CodeFingerprintMismatchError"
run_driver_retry() {  # same args as run_driver; retries REFUSED draws x4
  local denv="$1" note="$2" log="$3"; shift 3
  local a alog
  for a in 1 2 3 4; do
    alog="${log%.log}_a${a}.log"
    if run_driver "$denv" "$note" "$alog" "$@"; then
      cp "$alog" "$log" 2>/dev/null || true
      return 0
    fi
    if grep -aqE "$REFUSAL_RE" "$alog"; then
      say "'$note' attempt $a: engine draw REFUSED ($(grep -aoEm1 "$REFUSAL_RE" "$alog")) — fresh draw"
      continue
    fi
    say "'$note' attempt $a: NON-refusal failure — aborting (see $alog)"
    return 1
  done
  say "'$note': no clean draw in 4 attempts — STARVED; land the manifest-driven repair (oob-manifest-repair) + revalidate"
  return 1
}

run_driver() {  # $1=DRIVER envs  $2=note  $3=log  $4...=args
  local denv="$1" note="$2" log="$3"; shift 3
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $denv setsid --wait nohup ~/vllm-env/bin/python -u "$@" \
      --note "$note" </dev/null > "$log" 2>&1
    echo "DRIVER_EXIT=$?" >> "$log"
  ) &
  local w=$! waited=0
  while kill -0 "$w" 2>/dev/null; do
    sleep 60; waited=$((waited + 60))
    [ -s "$ALERT_FILE" ] && { say "ABORT: DISK ALERT during '$note'"; pkill -f "$note"; return 1; }
    [ "$waited" -ge "$STAGE_TIMEOUT_S" ] && { say "ABORT: timeout on '$note'"; pkill -f "$note"; return 1; }
  done
  wait "$w" 2>/dev/null
  grep -q "DRIVER_EXIT=0" "$log" || { say "ABORT: driver failed on '$note' (see $log)"; return 1; }
  return 0
}

ckpt() {  # per-stage GCS checkpoint (db via sqlite backup API + logs)
  ~/vllm-env/bin/python - <<PY
import sqlite3
src = sqlite3.connect('$HOME/glm-tpu/bench/results.db')
dst = sqlite3.connect('$RUN_DIR/results_ckpt.db')
src.backup(dst); dst.close(); src.close()
PY
  gcloud storage cp -r "$RUN_DIR" "$GCS_CKPT/" >/dev/null 2>&1 || true
}

# ── A+B: sparse dcp=8 bring-up + 32K sanity ──
if [ "$FROM_STAGE" = "A" ]; then
say "════ STAGE A/B: sparse dcp=8 engine + 32K sanity x2 ════"
launch_healthy "$SPARSE_RAYLET" "$SPARSE_DRIVER" sparse || exit 1
run_driver_retry "$SPARSE_DRIVER" "256K-stage 32K sanity ($TAG)" "$RUN_DIR/sanity32k.log" \
  glm_longctx.py --lengths 32768 --depths 0.5 --trials 2 --max-seqs 1 --gmu 0.90 \
  --max-batched-tokens 2048 --num-gpu-blocks "$BLOCKS" --max-len "$MAX_LEN" || exit 1
S32=$(grep -c "correct=True" "$RUN_DIR/sanity32k.log" || true)
[ "${S32:-0}" -eq 2 ] || { say "ABORT: 32K sanity ${S32:-0}/2 at dcp=8 — striping/LSE suspect, instrument before 256K"; exit 1; }
say "32K sanity 2/2"; ckpt
fi

# ── C: 256K passkey smoke ──
if [ "$FROM_STAGE" = "A" ] || [ "$FROM_STAGE" = "C" ]; then
[ "$FROM_STAGE" = "C" ] && { launch_healthy "$SPARSE_RAYLET" "$SPARSE_DRIVER" sparse || exit 1; }
say "════ STAGE C: 256K smoke, mechanism depths x1 ════"
run_driver_retry "$SPARSE_DRIVER" "256K smoke ($TAG)" "$RUN_DIR/smoke256k.log" \
  glm_longctx.py --lengths 256000 --depths 0.0,0.5,0.95,1.0 --trials 1 --max-seqs 1 --gmu 0.90 \
  --max-batched-tokens 2048 --num-gpu-blocks "$BLOCKS" --max-len "$MAX_LEN" || exit 1
SM=$(grep -c "correct=True" "$RUN_DIR/smoke256k.log" || true)
say "256K smoke: ${SM:-0}/4 ($(grep -c 'correct=False' "$RUN_DIR/smoke256k.log" || true) miss)"; ckpt
[ "${SM:-0}" -ge 3 ] || { say "ABORT: 256K smoke <3/4 — not interpretable for throughput; instrument first"; exit 1; }
fi

# ── D1: sparse throughput arm ──
if [ "$FROM_STAGE" != "D2" ]; then
if [ "$FROM_STAGE" = "D" ]; then
  say "── resume at STAGE D: bringing up a fresh sparse engine ──"
  launch_healthy "$SPARSE_RAYLET" "$SPARSE_DRIVER" sparse || exit 1
fi
say "════ STAGE D1: sparse decode-throughput A/B arm, L=262144 ════"
run_driver_retry "$SPARSE_DRIVER" "256K A/B sparse ($TAG)" "$RUN_DIR/ab_sparse.log" \
  dsa_throughput.py --ctxs 262144 --num-seqs 1 --measure-tokens "$MEASURE_TOKENS" \
  --gmu 0.90 --max-batched-tokens 2048 --num-gpu-blocks "$BLOCKS" \
  --max-len "$MAX_LEN" || exit 1
ckpt
fi

# ── D2: dense arm (FRESH engine — one variable: the DSA mode) ──
say "════ STAGE D2: dense arm (fresh engine, identical dcp=8) ════"
launch_healthy "$DENSE_RAYLET" "$DENSE_DRIVER" dense || exit 1
run_driver_retry "$DENSE_DRIVER" "256K A/B dense ($TAG)" "$RUN_DIR/ab_dense.log" \
  dsa_throughput.py --ctxs 262144 --num-seqs 1 --measure-tokens "$MEASURE_TOKENS" \
  --gmu 0.90 --max-batched-tokens 2048 --num-gpu-blocks "$BLOCKS" \
  --max-len "$MAX_LEN" || exit 1
ckpt
say "════ STAGE 256K COMPLETE — compare decode_tok_s in ab_sparse.log vs ab_dense.log; bank in RESEARCH_LOG ════"
