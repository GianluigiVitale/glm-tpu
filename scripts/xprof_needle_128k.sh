#!/bin/bash
# xprof_needle_128k.sh — THE 128K MEASUREMENT NEEDLE (RESEARCH_LOG 2026-07-18):
# one needle at the EXACT gate config with PHASED_PROFILING_DIR armed, to close
# the P0.b residual (~8s/chunk unattributed at 128K — the 07-12 trace was 32K)
# BEFORE burning ~14h of gate time. Decision rule (owner-prompted, pre-committed):
# act ONLY on a >=40% single dominator whose fix is the already-written
# GLM_DSA_SCORER=pallas (S2); anything else => launch the gate as-is and queue
# the finding. Measurement, not correctness: the needle result is NOT a gate row.
#
# Profiling notes: PhasedBasedProfiler auto-captures per-phase; PREFILL_ONLY is
# the phase we want. 4 steps (not the 15 default) — the 07-12 verdict came from
# ONE step; 4 bounds trace volume at 128K. Dumps stay armed (gate parity — the
# gate's own RAYLET_ENVS carries GLM_DCP_CACHE_DUMP; host-side, trace-inert/F3).
set -u
ZONE=us-central2-b
POD=db-v4-64-od
RUN_DIR=~/glm-run/xprof128k_$(date -u +%Y%m%dT%H%M%SZ)
PIN=04507ba1d
XPROF_DIR=/tmp/xprof128k
NEEDLE_TIMEOUT_S="${NEEDLE_TIMEOUT_S:-5400}"
MAX_TRIES=3

# EXACT gate env blocks (gate_sparse128k.sh ca0786b) + the profiling extras.
RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_DCP_CACHE_DUMP=/tmp/dcp_gatehealth GLM_DCP_CACHE_DUMP_LAYERS=2 PHASED_PROFILING_DIR='"$XPROF_DIR"' PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=4 GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_EXPECT_CODE_HASH=$PIN"

mkdir -p "$RUN_DIR"
say() { echo "[xprof128k $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && git rev-parse --short=9 HEAD' 2>/dev/null | grep -c "$PIN")
[ "${HASHES:-0}" -eq 8 ] || { say "ABORT: ${HASHES:-0}/8 @ $PIN"; exit 1; }
bash ~/glm-tpu/scripts/disk_watchdog.sh check >> "$RUN_DIR/orchestrator.log" 2>&1 || {
  say "ABORT: disk pre-flight"; exit 1; }

for TRY in $(seq 1 "$MAX_TRIES"); do
  say "── try $TRY: launch + needle ──"
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="rm -rf $XPROF_DIR /tmp/dcp_gatehealth*.npz && mkdir -p $XPROF_DIR" >/dev/null 2>&1
  EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch_try${TRY}.log" 2>&1
  NODES=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
  [ "${NODES:-0}" -eq 8 ] || { say "try $TRY: ray ${NODES:-0}/8 — relaunch"; continue; }
  LOG="$RUN_DIR/needle_try${TRY}.log"
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u glm_longctx.py \
      --lengths 128000 --depths 0.5 --trials 1 --max-seqs 1 --gmu 0.90 \
      --max-batched-tokens 2048 --num-gpu-blocks 68 --max-len 131840 \
      --note "xprof 128K measurement needle" </dev/null > "$LOG" 2>&1
    echo "DRIVER_EXIT=$?" >> "$LOG"
  ) &
  W=$!; waited=0
  while kill -0 "$W" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    [ "$waited" -ge "$NEEDLE_TIMEOUT_S" ] && { pkill -f "xprof 128K measurement" 2>/dev/null; break; }
  done
  wait "$W" 2>/dev/null
  if grep -qE "PwalNanCheckError|LoadNanCheckError" "$LOG"; then
    say "try $TRY: CORRUPT DRAW refused by the armed checks (the ~10% residual) — relaunch"
    continue
  fi
  if grep -q "correct=True" "$LOG"; then
    say "try $TRY: needle CORRECT — collecting the rank-0 (local) trace"
    ls -la "$XPROF_DIR"/*/plugins/profile/*/ >> "$RUN_DIR/orchestrator.log" 2>&1 || true
    TRACE=$(find "$XPROF_DIR" -name "*.xplane.pb" 2>/dev/null | head -5)
    if [ -n "$TRACE" ]; then
      say "TRACES:"; echo "$TRACE" | tee -a "$RUN_DIR/orchestrator.log"
      cp -r "$XPROF_DIR" "$RUN_DIR/trace" 2>/dev/null || true
      say "DONE — trace copied to $RUN_DIR/trace"
      exit 0
    fi
    say "needle correct but NO local xplane.pb — check $XPROF_DIR phase dirs"; exit 2
  fi
  say "try $TRY: needle failed (no correct=True; DRIVER_EXIT=$(grep -oE 'DRIVER_EXIT=[0-9]+' "$LOG" | tail -1))"
done
say "FAILED after $MAX_TRIES tries"
exit 1
