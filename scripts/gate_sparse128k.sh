#!/bin/bash
# gate_sparse128k.sh — THE 128K SPARSE GATE orchestrator (n=77: 7 depths x 11
# trials, mechanism depths FIRST), rebuilt in-repo after the VM loss (the
# original lived only in ~/glm-run and died with the host — never again).
#
# Fixes baked in from the gate2 audit (RESEARCH_LOG 2026-07-13 09:00):
#   F4  the env blocks are defined ONCE and used verbatim for EVERY per-depth
#       launch — there is no relaunch path that can drop an env (gate2's
#       retry path re-launched with a partial environment => false
#       provenance).
#   MISS-ABORT WATCHDOG: needle lines are graded LIVE. First correct=False =>
#       loud ALERT + continue (a miss is a recorded result; the n->130
#       extension decision is the operator's). SECOND miss => ABORT the gate
#       (Wilson-unrecoverable at n<=130 — burning further pod-hours cannot
#       change the verdict; gate2 burned a 10h depth run on a dead engine).
#   DISK: pre-flight + live watch; any alert during a depth run taints it =>
#       ABORT as INFRA (the w-4 lesson: disk-degraded engines mimic model
#       defects).
#   Per-depth GCS checkpoints: results.db + the depth log land in
#       gs://driftbench-dsv4-uc/results/<tag>/ after every depth — the gate
#       survives a VM loss with at most one depth in flight (proven: run
#       165's 0/11 was captured seconds after it landed).
#
# Config = the ratified gate config (results.db run 163-165 env_json,
# verbatim): dcp=4 + segment + headsplit UNSET + flat DSA scatter (default) +
# pageloop dense bake + chunk 2048 + pool 68 + owned/all-v2 + APC off
# (engine-hardcoded) + UNARMED (no dumps — gate-class runs are never armed).
#
# Usage: bash ~/glm-tpu/scripts/gate_sparse128k.sh [--depths "0.0,0.05,..."]
#        [--trials 11] [--dry-run]
set -u

ZONE=us-central2-b
POD=db-v4-64-od
PIN=04507ba1d
TAG=gate128k_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
GCS_CKPT=gs://driftbench-dsv4-uc/results/$TAG
DEPTHS="0.0,0.05,0.95,1.0,0.25,0.5,0.75"   # mechanism cells FIRST
TRIALS=11
DEPTH_TIMEOUT_S="${DEPTH_TIMEOUT_S:-21600}"  # 11 needles ~110min + cold-compile headroom

# ── THE env blocks (single source of truth — F4).
# GLM_WRITE_PROBE=1: the standing startup sentinel (write-probe through the
# real owner-scatters at engine init — refuses to serve a bad-write engine
# BEFORE it burns a 10h depth run).
# GLM_DCP_CACHE_DUMP at LAYERS=2 ONLY (revision of the "gates are UNARMED"
# rule after the 2026-07-17 lottery forensics): the layer-1 indexer k-cache
# (slot 2) is THE buffer the engine-lottery NaN-poisons, host-side dumps are
# trace-inert (F3-safe), and slot-2-only volume is ~4.5MB/step/proc (~3GB/
# host/depth, purged per depth) — nothing like the 230GB all-slot figure.
# The dump feeds the per-depth ENGINE HEALTH PROBE below: a poisoned engine
# is detected in ~2 min and relaunched instead of burning a 2h depth run.
RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_DCP_CACHE_DUMP=/tmp/dcp_gatehealth GLM_DCP_CACHE_DUMP_LAYERS=2 GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_EXPECT_CODE_HASH=$PIN"

DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --depths) DEPTHS="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_DIR"
ALERT_FILE="$RUN_DIR/DISK_ALERT"
say() { echo "[gate128k $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

if (( DRY )); then
  say "DRY RUN — depths $DEPTHS x $TRIALS trials; EXTRA_ENVS=$RAYLET_ENVS"
  exit 0
fi

# 0) pre-flight: 8x pin + disks + background watch
HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && echo "$(hostname) $(git rev-parse --short=9 HEAD)"' \
  2>/dev/null | grep -E "^t1v-")
echo "$HASHES" | tee -a "$RUN_DIR/orchestrator.log"
[ "$(echo "$HASHES" | grep -c "$PIN")" -eq 8 ] || { say "ABORT: workers not 8x @ $PIN"; exit 1; }
bash ~/glm-tpu/scripts/disk_watchdog.sh check | tee -a "$RUN_DIR/orchestrator.log" || {
  say "ABORT: disk pre-flight failed"; exit 1; }
ALERT_FILE="$ALERT_FILE" INTERVAL_S=120 setsid nohup \
  bash ~/glm-tpu/scripts/disk_watchdog.sh watch </dev/null \
  > "$RUN_DIR/disk_watch.log" 2>&1 &
WATCH_PID=$!
trap 'kill "$WATCH_PID" 2>/dev/null' EXIT

# ── ENGINE HEALTH PROBE (the 2026-07-17 lottery mitigation): after a fresh
# launch, serve one ~5K mini-needle (3 chunks — chunk 1 is the ctx<=topk
# dense fallback, chunks 2-3 exercise the SPARSE path that the lottery
# poisons), then NaN-scan the layer-2 (layer-1 indexer k-cache) dumps on all
# 8 hosts. Any NaN ⇒ the engine drew the bad state ⇒ relaunch. bf16 NaN on
# the uint16-bitcast dump = exponent all-ones + mantissa nonzero.
health_check_engine() {
  local hlog="$1"
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='rm -f /tmp/dcp_gatehealth*.npz' >/dev/null 2>&1
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u glm_longctx.py \
      --lengths 5000 --depths 0.5 --trials 1 --max-seqs 1 --gmu 0.90 \
      --max-batched-tokens 2048 --num-gpu-blocks 68 --max-len 131840 \
      --note "gate health-check ($TAG)" </dev/null > "$hlog" 2>&1
  )
  grep -q "correct=True" "$hlog" || { echo "SICK:needle"; return; }
  local scan
  scan=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all --command='
    ~/vllm-env/bin/python - <<PY
import numpy as np, glob, sys
fs = sorted(glob.glob("/tmp/dcp_gatehealth*.npz"))
if not fs: print("NOFILE"); sys.exit(0)
d = np.load(fs[-1], allow_pickle=True)
bad = 0
for k in d.files:
    if k.endswith("__data"):
        u = d[k].view(np.uint16)
        bad += int((((u & 0x7F80) == 0x7F80) & ((u & 0x007F) != 0)).sum())
print("NAN" if bad else "CLEAN", bad)
PY' 2>/dev/null)
  local n_clean n_nofile
  n_clean=$(echo "$scan" | grep -c "^CLEAN" || true)
  n_nofile=$(echo "$scan" | grep -c "^NOFILE" || true)
  if echo "$scan" | grep -q "^NAN"; then echo "SICK:nan_$(echo "$scan" | grep -c '^NAN')hosts"
  elif [ "${n_nofile:-0}" -gt 0 ]; then echo "SICK:dumps_missing_${n_nofile}"
  elif [ "${n_clean:-0}" -eq 8 ]; then echo "HEALTHY"
  else echo "SICK:scan_${n_clean:-0}of8"; fi
}

TOTAL_MISS=0
HEALTH_RETRIES="${HEALTH_RETRIES:-5}"
IFS=',' read -ra DEPTH_ARR <<< "$DEPTHS"
for d in "${DEPTH_ARR[@]}"; do
  LOG="$RUN_DIR/depth_${d}.log"
  say "── depth $d ($TRIALS trials) ──"
  ALERT_BEFORE=$(wc -l < "$ALERT_FILE" 2>/dev/null || echo 0)

  # fresh engine per depth (the gate2 protocol), F4-proof single env source,
  # health-probed: relaunch until the engine draw is clean (lottery mitigation)
  HEALTH=""; TRY=0
  while [ "$TRY" -lt "$HEALTH_RETRIES" ]; do
    TRY=$((TRY + 1))
    EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
      bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch_${d}_try${TRY}.log" 2>&1
    NODES=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
    [ "${NODES:-0}" -eq 8 ] || { say "depth $d try $TRY: ray nodes ${NODES:-0}/8 — relaunching"; continue; }
    HEALTH=$(health_check_engine "$RUN_DIR/health_${d}_try${TRY}.log" | tail -1)
    say "depth $d try $TRY: engine health = $HEALTH"
    [ "$HEALTH" = "HEALTHY" ] && break
  done
  if [ "$HEALTH" != "HEALTHY" ]; then
    say "ABORT depth $d: no healthy engine in $HEALTH_RETRIES draws (last: $HEALTH) — the"
    say "lottery rate is worse than planned; investigate before burning more pod time."
    exit 1
  fi

  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u glm_longctx.py \
      --lengths 128000 --depths "$d" --trials "$TRIALS" --max-seqs 1 --gmu 0.90 \
      --max-batched-tokens 2048 --num-gpu-blocks 68 --max-len 131840 \
      --note "SPARSE 128K GATE depth=$d ($TAG)" \
      </dev/null > "$LOG" 2>&1
    echo "DRIVER_EXIT=$?" >> "$LOG"
  ) &
  WRAPPER=$!

  # live miss-abort watchdog over the streaming needle lines.
  # INFRA taint semantics (review MAJOR): a disk alert or timeout during a
  # depth run makes the depth INFRA — its misses are NOT results, and the
  # whole gate ABORTS (the header doctrine; never launch the next depth on
  # a breached pod).
  WAITED=0; DEPTH_MISS=0; DEPTH_INFRA=""
  while kill -0 "$WRAPPER" 2>/dev/null; do
    sleep 60; WAITED=$((WAITED + 60))
    ALERT_NOW=$(wc -l < "$ALERT_FILE" 2>/dev/null || echo 0)
    if [ "${ALERT_NOW:-0}" -gt "${ALERT_BEFORE:-0}" ]; then
      DEPTH_INFRA="disk_alert"
      say "ABORT depth $d: DISK ALERT during the run — depth is INFRA (its misses are NOT results). Killing."
      pkill -f "SPARSE 128K GATE depth=$d .*$TAG" 2>/dev/null
      break
    fi
    if [ "$WAITED" -ge "$DEPTH_TIMEOUT_S" ]; then
      DEPTH_INFRA="timeout"
      say "ABORT depth $d: timeout ${DEPTH_TIMEOUT_S}s — depth is INFRA. Killing."
      pkill -f "SPARSE 128K GATE depth=$d .*$TAG" 2>/dev/null
      break
    fi
    MISSES=$(grep -c "correct=False" "$LOG" 2>/dev/null || true)
    if [ "${MISSES:-0}" -gt "$DEPTH_MISS" ]; then
      DEPTH_MISS=$MISSES
      say "⚠ depth $d: MISS #$((TOTAL_MISS + DEPTH_MISS)) — $(grep "correct=False" "$LOG" | tail -1)"
      if [ $((TOTAL_MISS + DEPTH_MISS)) -ge 2 ]; then
        say "ABORT: 2nd miss — Wilson-unrecoverable at n<=130. Killing the depth run."
        pkill -f "SPARSE 128K GATE depth=$d .*$TAG" 2>/dev/null
        break
      fi
    fi
  done
  wait "$WRAPPER" 2>/dev/null
  if [ -n "$DEPTH_INFRA" ]; then
    say "════ GATE ABORTED: depth $d INFRA ($DEPTH_INFRA). Tainted misses ($DEPTH_MISS) NOT counted."
    say "Fix the infrastructure, verify disks on ALL 8 hosts, then resume with"
    say "  --depths \"<remaining incl. $d>\" (results.db keeps the clean depths)."
    exit 1
  fi
  TOTAL_MISS=$((TOTAL_MISS + DEPTH_MISS))

  # per-depth GCS checkpoint (results.db via the sqlite backup API — safe
  # against a writer; the run-165 lesson: this checkpoint saved the gate2
  # verdict by seconds)
  ~/vllm-env/bin/python - <<PY
import sqlite3
src = sqlite3.connect('$HOME/glm-tpu/bench/results.db')
dst = sqlite3.connect('$RUN_DIR/results_ckpt.db')
src.backup(dst); dst.close(); src.close()
PY
  gcloud storage cp "$RUN_DIR/results_ckpt.db" "$GCS_CKPT/results-after-d${d}.db" >/dev/null 2>&1
  gcloud storage cp "$LOG" "$GCS_CKPT/" >/dev/null 2>&1
  N_OK=$(grep -c "correct=True" "$LOG" 2>/dev/null || true)
  say "depth $d done: ${N_OK:-0}/$TRIALS correct, $DEPTH_MISS miss — checkpointed to $GCS_CKPT"
  # per-depth dump hygiene: on a MISS the depth's layer-2 dumps are the
  # specimen — archive them; otherwise purge (the disk doctrine).
  if [ "$DEPTH_MISS" -gt 0 ]; then
    SRC_GLOB="/tmp/dcp_gatehealth*.npz" bash ~/glm-tpu/scripts/dump_archiver.sh \
      archive "$TAG/depth_${d}_MISS" --last-step-only >> "$RUN_DIR/orchestrator.log" 2>&1
  else
    SRC_GLOB="/tmp/dcp_gatehealth*.npz" bash ~/glm-tpu/scripts/dump_archiver.sh \
      purge >> "$RUN_DIR/orchestrator.log" 2>&1
  fi

  if [ "$TOTAL_MISS" -ge 2 ]; then
    say "════ GATE ABORTED at 2 misses. Record honestly; investigate before any re-gate. ════"
    exit 1
  fi
done

TOTAL_OK=$(grep -ch "correct=True" "$RUN_DIR"/depth_*.log | paste -sd+ | bc)
say "════ GATE COMPLETE: $TOTAL_OK correct, $TOTAL_MISS miss ════"
if [ "$TOTAL_MISS" -eq 0 ] && [ "$TOTAL_OK" -ge 77 ]; then
  say "77/77 ⇒ Wilson LB ≈95.3% — THE ≥95%@128K SPARSE GATE IS CLOSED."
  say "Record in RESEARCH_LOG + HANDOFF; backup_bundle.sh; then the 256K A/B."
elif [ "$TOTAL_MISS" -eq 1 ]; then
  say "ONE miss: the gate does NOT pass at n=77. Policy: EXTEND to n≈130"
  say "(129/130 recovers LB>95%) — rerun per-depth with --trials topped up;"
  say "NEVER rerun-until-green."
fi
