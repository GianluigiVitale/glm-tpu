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
PIN=845f4ffeb
TAG=gate128k_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
GCS_CKPT=gs://driftbench-dsv4-uc/results/$TAG
DEPTHS="0.0,0.05,0.95,1.0,0.25,0.5,0.75"   # mechanism cells FIRST
TRIALS=11
DEPTH_TIMEOUT_S="${DEPTH_TIMEOUT_S:-21600}"  # 11 needles ~110min + cold-compile headroom

# ── THE env blocks (single source of truth — F4). Identical to
# probe_lottery.sh minus the cache dump (gate-class runs are UNARMED:
# an armed 128K gate writes ~230GB — the 2026-07-11 rule).
# GLM_WRITE_PROBE=1: the standing startup sentinel (write-probe through the
# real owner-scatters at engine init — refuses to serve a bad-write engine
# BEFORE it burns a 10h depth run). Gate engines are fresh experiments, so
# arming the guard here is exactly its purpose. (probe_lottery.sh deliberately
# does NOT arm it: its draws must stay maximally gate2-faithful in init-time
# HBM behavior; the scrambler handles layout variance there.)
RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_EXPECT_CODE_HASH=$PIN"

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

TOTAL_MISS=0
IFS=',' read -ra DEPTH_ARR <<< "$DEPTHS"
for d in "${DEPTH_ARR[@]}"; do
  LOG="$RUN_DIR/depth_${d}.log"
  say "── depth $d ($TRIALS trials) ──"
  ALERT_BEFORE=$(wc -l < "$ALERT_FILE" 2>/dev/null || echo 0)

  # fresh engine per depth (the gate2 protocol), F4-proof single env source
  EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch_${d}.log" 2>&1
  NODES=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
  [ "$NODES" -eq 8 ] || { say "ABORT depth $d: ray nodes $NODES/8 (INFRA)"; exit 1; }

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
