#!/bin/bash
# probe_timeline.sh — THE BORN-vs-CLOBBERED DISCRIMINATOR (RESEARCH_LOG
# 2026-07-17 17:40). Scramble+probe pairs at the gate2 config with
#   * GLM_DCP_CACHE_DUMP_LAYERS=2 ONLY (the poisoned buffer) and ALL
#     per-step dumps KEPT (~4.5MB x 63 steps x 8 procs ≈ 2.3GB/host) — the
#     per-step NaN timeline for page 1 decides: NaN already at chunk-1's own
#     write step = input-borne; clean at chunk 1 then NaN later = runtime
#     CLOBBER (sparse-branch arena aliasing).
#   * GLM_PWAL_NAN_CHECK=1 armed (init-time param-copy scan; a raise at init
#     = the born-bad-copy leg, caught before serving).
# Stops at the FIRST probe MISS (the specimen is the goal, not statistics);
# max 4 pairs. Reuses the probe_lottery protocol (fixed seed = gate2's first
# d=0.95 needle; scrambler for state diversity; INFRA classification).
set -u

ZONE=us-central2-b
POD=db-v4-64-od
RUN_DIR=~/glm-run/probe_timeline_$(date -u +%Y%m%dT%H%M%SZ)
MAX_PAIRS=4
NEEDLE_TIMEOUT_S="${NEEDLE_TIMEOUT_S:-10800}"
SCRAMBLE_TIMEOUT_S="${SCRAMBLE_TIMEOUT_S:-7200}"
PIN=34d2eef37
DUMP_PREFIX=/tmp/dcp_timeline
GCS_RUN=gs://driftbench-dsv4-uc/dumps/$(basename "$RUN_DIR")

RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_PWAL_NAN_CHECK=1 GLM_EXPECT_CODE_HASH='"$PIN"' GLM_DCP_CACHE_DUMP='"$DUMP_PREFIX"' GLM_DCP_CACHE_DUMP_LAYERS=2 LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_EXPECT_CODE_HASH=$PIN"

mkdir -p "$RUN_DIR"
say() { echo "[timeline $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

run_engine_needle() {  # kind tag log timeout extra-args...
  local kind="$1" tag="$2" log="$3" timeout_s="$4"; shift 4
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='rm -f '"$DUMP_PREFIX"'*.npz' >/dev/null 2>&1
  EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "${log%.log}.launch.log" 2>&1
  local nodes
  nodes=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
  if [ "${nodes:-0}" -ne 8 ]; then echo "INFRA:ray_nodes_${nodes:-0}"; return; fi
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u glm_longctx.py \
      --max-seqs 1 --gmu 0.90 --note "$tag" "$@" \
      </dev/null > "$log" 2>&1
    echo "DRIVER_EXIT=$?" >> "$log"
  ) &
  local wrapper=$! waited=0
  while kill -0 "$wrapper" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge "$timeout_s" ]; then
      pkill -f -- "--note $tag" 2>/dev/null
      echo "INFRA:timeout"; wait "$wrapper" 2>/dev/null; return
    fi
  done
  wait "$wrapper" 2>/dev/null
  if grep -q "PwalNanCheckError" "$log"; then echo "PWAL_POSITIVE"; return; fi
  local needle
  needle=$(grep -E "correct=(True|False)" "$log" | head -1 || true)
  if [ -n "$needle" ]; then
    if echo "$needle" | grep -q "correct=True"; then echo "CORRECT"; else echo "MISS"; fi
  elif ! grep -q "DRIVER_EXIT=0" "$log"; then
    echo "INFRA:driver_exit_$(grep -oE 'DRIVER_EXIT=[0-9]+' "$log" | tail -1 | cut -d= -f2)"
  else echo "INFRA:no_needle_line"; fi
}

# pre-flight: 8x pin + disks
HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && echo "$(hostname) $(git rev-parse --short=9 HEAD)"' \
  2>/dev/null | grep -cE "$PIN")
[ "${HASHES:-0}" -eq 8 ] || { say "ABORT: ${HASHES:-0}/8 workers @ $PIN"; exit 1; }
bash ~/glm-tpu/scripts/disk_watchdog.sh check >> "$RUN_DIR/orchestrator.log" 2>&1 || {
  say "ABORT: disk pre-flight failed"; exit 1; }

pair=0
while [ "$pair" -lt "$MAX_PAIRS" ]; do
  pair=$((pair + 1))
  say "── pair $pair: scrambler ──"
  SV=$(run_engine_needle scramble "timeline-scramble s${pair} $(basename "$RUN_DIR")" \
    "$RUN_DIR/scramble${pair}.log" "$SCRAMBLE_TIMEOUT_S" \
    --lengths 32000 --depths 0.5 --trials 1 --seed $((535353 + pair)) \
    --max-batched-tokens 1024 --num-gpu-blocks 70 --max-len 33280 | tail -1)
  say "pair $pair scrambler: $SV"
  # scrambler dumps are noise for the timeline — purge before the probe
  say "── pair $pair: THE PROBE (per-step dumps kept) ──"
  PV=$(run_engine_needle probe "timeline-probe p${pair} $(basename "$RUN_DIR")" \
    "$RUN_DIR/probe${pair}.log" "$NEEDLE_TIMEOUT_S" \
    --lengths 128000 --depths 0.95 --trials 1 \
    --max-batched-tokens 2048 --num-gpu-blocks 68 --max-len 131840 | tail -1)
  say "pair $pair probe: $PV"
  if [ "$PV" = "MISS" ] || [ "$PV" = "PWAL_POSITIVE" ]; then
    say "SPECIMEN CAUGHT ($PV) — archiving ALL per-step dumps"
    # full archive, NO --last-step-only: the timeline needs every step
    SRC_GLOB="$DUMP_PREFIX*.npz" bash ~/glm-tpu/scripts/dump_archiver.sh \
      archive "$(basename "$RUN_DIR")/probe${pair}_${PV}" >> "$RUN_DIR/orchestrator.log" 2>&1
    say "VERDICT PENDING OFFLINE: NaN-timeline over $GCS_RUN/probe${pair}_${PV}/"
    say "  born-NaN at page-1's own write step = input-borne;"
    say "  clean-then-NaN at a later step = runtime clobber (arena aliasing)."
    exit 0
  fi
  SRC_GLOB="$DUMP_PREFIX*.npz" bash ~/glm-tpu/scripts/dump_archiver.sh purge \
    >> "$RUN_DIR/orchestrator.log" 2>&1
done
say "NO SPECIMEN in $MAX_PAIRS pairs (rate fluctuation) — rerun or widen."
exit 1
