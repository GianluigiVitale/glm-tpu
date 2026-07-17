#!/bin/bash
# probe_lottery.sh — THE ENGINE-LOTTERY DISCRIMINATOR (RESEARCH_LOG
# 2026-07-13 15:40): does the "engine-instance lottery" (gate2's d=0.95 0/11
# between two perfect engines) recur on clean disks, or was it disk-pressure
# all along?
#
# Protocol (adversarially reviewed 2026-07-17; the review REDESIGNED it):
# N valid SCRAMBLED draws. Each draw is a PAIR:
#   (a) SCRAMBLER engine (uncounted): a different-content, different-layout
#       program (32K needle, per-draw seed offset, pool 70/chunk 1024) whose
#       only job is to overwrite HBM with FOREIGN-layout bytes. Without it,
#       back-to-back engines serving the IDENTICAL prompt inherit stale≈fresh
#       HBM and the never-written-stripe class is INVISIBLE (the C-vs-D
#       blind spot, RESEARCH_LOG 2026-07-10 17:55/19:40 — the review caught
#       that the un-scrambled design could not answer its own question).
#   (b) THE PROBE (counted): fresh engine, ONE fixed-seed 128K needle at the
#       exact gate2 config (verbatim from results.db run 165 env_json; seed
#       formula reproduces gate2's first d=0.95 needle). Deterministic
#       prefill ⇒ byte-equal post-prefill caches across good engines ⇒ a bad
#       engine byte-diffs against any good one. GLM_DCP_CACHE_DUMP is
#       raylet-baked, HOST-side (traced program UNCHANGED — audit F3), over
#       ALL 21 indexer k-cache slots + the first MLA slot (the review's slot
#       map: 0,1,2 covered ~3% of the buffer family the defect class lived
#       in).
#
# Classification (the postmortem lesson): every probe ends as exactly one of
#   CORRECT / MISS (a recorded result) / INFRA (named cause; does NOT count).
# A verdict in a disk-alert window is INFRA, and — review BLOCKER — its miss
# is NOT counted: the miss counter increments only after the taint check.
#
# STATISTICS (review): N=20 valid draws. 20/20 correct rejects a true ≥1/7
# lottery at 95% ((6/7)^20 = 4.6%); at N=14 the false-negative rate is 11.6%
# — too weak to bet a 13h gate on. One MISS on clean disks ⇒ lottery REAL.
#
# F4: the env blocks are defined ONCE and used verbatim for EVERY launch.
#
# Usage: bash ~/glm-tpu/scripts/probe_lottery.sh [--probes N] [--dry-run]
set -u

ZONE=us-central2-b
POD=db-v4-64-od
RUN_DIR=~/glm-run/probe_lottery_$(date -u +%Y%m%dT%H%M%SZ)
N_PROBES=20
MAX_PROBES=27
NEEDLE_TIMEOUT_S="${NEEDLE_TIMEOUT_S:-10800}"    # probe 1 pays the cold XLA compile
SCRAMBLE_TIMEOUT_S="${SCRAMBLE_TIMEOUT_S:-7200}"
PIN=845f4ffeb
DUMP_PREFIX=/tmp/dcp_probe                        # per-host; archived+purged per probe
GCS_RUN=gs://driftbench-dsv4-uc/dumps/$(basename "$RUN_DIR")
# All 21 DSA indexer k-cache slots + slot 1 (first MLA latent cache) under
# the pinned interleaved registration (review-derived slot map; the dump
# filenames carry slot indices so dcp_cache_diff self-describes).
DUMP_LAYERS="0,1,2,4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94"

# ── THE env blocks (single source of truth — F4). Raylet side mirrors the
# gate2 run-165 env_json + runbook §8 (LIBTPU quoting verified in raylet
# /proc environ); GLM_DCP_SCATTER_IMPL=pageloop MUST STAY (dense-path metal
# verdict; the dense fallback fires INSIDE sparse serving). HEADSPLIT stays
# UNSET (metal-refused composition — and now trace-time-refused in code).
RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_EXPECT_CODE_HASH='"$PIN"' GLM_DCP_CACHE_DUMP='"$DUMP_PREFIX"' GLM_DCP_CACHE_DUMP_LAYERS='"$DUMP_LAYERS"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
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
    --probes) N_PROBES="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_DIR"
MANIFEST="$RUN_DIR/manifest.tsv"
echo -e "probe\tts\tkind\tverdict\treason\tlog\tdump_uri\tduration_s" > "$MANIFEST"
ALERT_FILE="$RUN_DIR/DISK_ALERT"

say() { echo "[probe-lottery $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

if (( DRY )); then
  say "DRY RUN — would run $N_PROBES scrambled draws with:"
  say "EXTRA_ENVS=$RAYLET_ENVS"
  say "probe driver: env \$DRIVER_ENVS ... glm_longctx.py --lengths 128000 --depths 0.95 --trials 1"
  say "scrambler:    env \$DRIVER_ENVS ... glm_longctx.py --lengths 32000 --depths 0.5 --trials 1 --seed <offset> --num-gpu-blocks 70 --max-batched-tokens 1024"
  exit 0
fi

# launch a fresh engine + run one needle; args: <kind> <tag> <log> <timeout> <extra glm_longctx args...>
run_engine_needle() {
  local kind="$1" tag="$2" log="$3" timeout_s="$4"; shift 4
  EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "${log%.log}.launch.log" 2>&1
  local nodes
  nodes=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
  if [ "${nodes:-0}" -ne 8 ]; then echo "INFRA:ray_nodes_${nodes:-0}"; return; fi
  if [ "$kind" = "probe" ]; then
    # Count hosts whose raylet carries the dump env AT LEAST once — a
    # duplicated environ entry (execve permits them; observed live: one
    # host printed 2) is semantically fine, and a single ssh flake gets
    # one retry before the draw is burned as INFRA.
    local envok attempt
    for attempt in 1 2; do
      envok=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
        --command='P=$(pgrep -x raylet | head -1); tr "\0" "\n" < /proc/$P/environ | grep -c "GLM_DCP_CACHE_DUMP='"$DUMP_PREFIX"'"' \
        2>/dev/null | grep -cE "^[1-9]" || true)
      [ "${envok:-0}" -eq 8 ] && break
      sleep 30
    done
    if [ "${envok:-0}" -ne 8 ]; then echo "INFRA:raylet_env_${envok:-0}"; return; fi
  fi
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # setsid --wait: new session (tool-shell-teardown-immune) that still
    # blocks for the real exit code (bare setsid forks+returns — landmine).
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
  local needle
  needle=$(grep -E "correct=(True|False)" "$log" | head -1 || true)
  if [ -n "$needle" ]; then
    if echo "$needle" | grep -q "correct=True"; then echo "CORRECT"; else echo "MISS"; fi
  elif ! grep -q "DRIVER_EXIT=0" "$log"; then
    echo "INFRA:driver_exit_$(grep -oE 'DRIVER_EXIT=[0-9]+' "$log" | tail -1 | cut -d= -f2)"
  else
    echo "INFRA:no_needle_line"
  fi
}

# 0) Pre-flight: 8x pin verification + disk baseline + watch loop.
say "pre-flight: verifying all 8 workers @ $PIN"
HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && echo "$(hostname) $(git rev-parse --short=9 HEAD)"' \
  2>/dev/null | grep -E "^t1v-")
echo "$HASHES" | tee -a "$RUN_DIR/orchestrator.log"
N_OK=$(echo "$HASHES" | grep -c "$PIN" || true)
[ "${N_OK:-0}" -eq 8 ] || { say "ABORT: ${N_OK:-0}/8 workers at $PIN — run sync_workers.sh first."; exit 1; }
bash ~/glm-tpu/scripts/disk_watchdog.sh check | tee -a "$RUN_DIR/orchestrator.log" || {
  say "ABORT: disk pre-flight failed."; exit 1; }
ALERT_FILE="$ALERT_FILE" INTERVAL_S=120 setsid nohup \
  bash ~/glm-tpu/scripts/disk_watchdog.sh watch </dev/null \
  > "$RUN_DIR/disk_watch.log" 2>&1 &
WATCH_PID=$!
say "disk watch running (pid $WATCH_PID, alerts -> $ALERT_FILE)"
cleanup() { kill "$WATCH_PID" 2>/dev/null; }
trap cleanup EXIT

valid=0; probe=0; miss=0; infra=0
while [ "$valid" -lt "$N_PROBES" ] && [ "$probe" -lt "$MAX_PROBES" ]; do
  probe=$((probe + 1))
  T0=$(date +%s)
  TS=$(date -u +%FT%TZ)

  # (a) SCRAMBLER (uncounted; INFRA-tolerant — its only job is foreign HBM).
  # Different content (seed offset), different layout (pool 70, chunk 1024),
  # different length (32K) than the probe program.
  SLOG="$RUN_DIR/scramble${probe}.log"
  say "── draw $probe: scrambler ──"
  SVERDICT=$(run_engine_needle scramble "probe-lottery-scramble s${probe} $(basename "$RUN_DIR")" \
    "$SLOG" "$SCRAMBLE_TIMEOUT_S" \
    --lengths 32000 --depths 0.5 --trials 1 --seed $((424242 + probe)) \
    --max-batched-tokens 1024 --num-gpu-blocks 70 --max-len 33280 | tail -1)
  echo -e "$probe\t$TS\tscramble\t$SVERDICT\t-\t$SLOG\t-\t$(( $(date +%s) - T0 ))" >> "$MANIFEST"
  say "draw $probe scrambler: $SVERDICT"

  # (b) THE PROBE (counted): fixed-seed d=0.95 128K needle, dumps armed.
  T1=$(date +%s)
  LOG="$RUN_DIR/probe${probe}.log"
  ALERT_LINES_BEFORE=$(wc -l < "$ALERT_FILE" 2>/dev/null || echo 0)
  say "── draw $probe: THE PROBE ──"
  RAW=$(run_engine_needle probe "probe-lottery p${probe} $(basename "$RUN_DIR")" \
    "$LOG" "$NEEDLE_TIMEOUT_S" \
    --lengths 128000 --depths 0.95 --trials 1 \
    --max-batched-tokens 2048 --num-gpu-blocks 68 --max-len 131840 | tail -1)
  ALERT_LINES_AFTER=$(wc -l < "$ALERT_FILE" 2>/dev/null || echo 0)

  # taint check FIRST — a verdict in a disk-alert window is INFRA and its
  # miss is NEVER counted (review BLOCKER: the un-ordered version let a
  # disk-tainted miss flip the headline verdict).
  VERDICT="$RAW"; REASON="-"
  case "$RAW" in INFRA:*) VERDICT=INFRA; REASON="${RAW#INFRA:}";; esac
  if [ "${ALERT_LINES_AFTER:-0}" -gt "${ALERT_LINES_BEFORE:-0}" ]; then
    REASON="disk_alert_in_window(was:$RAW)"
    VERDICT=INFRA
  fi
  DUMPS_OK=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='ls '"$DUMP_PREFIX"'*.npz >/dev/null 2>&1 && echo Y || echo N' \
    2>/dev/null | grep -c "^Y" || true)
  if [ "$VERDICT" != "INFRA" ] && [ "${DUMPS_OK:-0}" -ne 8 ]; then
    say "draw $probe: dumps on ${DUMPS_OK:-0}/8 hosts — verdict kept, byte-diff coverage partial"
    REASON="dumps_${DUMPS_OK:-0}/8"
  fi
  case "$VERDICT" in
    CORRECT) valid=$((valid + 1)) ;;
    MISS)    valid=$((valid + 1)); miss=$((miss + 1)) ;;
    *)       infra=$((infra + 1)) ;;
  esac

  # per-probe archive of the post-prefill caches; purge the rest
  SRC_GLOB="$DUMP_PREFIX*.npz" bash ~/glm-tpu/scripts/dump_archiver.sh \
    archive "$(basename "$RUN_DIR")/probe${probe}" --last-step-only \
    >> "$RUN_DIR/orchestrator.log" 2>&1
  DUR=$(( $(date +%s) - T1 ))
  echo -e "$probe\t$TS\tprobe\t$VERDICT\t$REASON\t$LOG\t$GCS_RUN/probe${probe}/\t$DUR" >> "$MANIFEST"
  say "draw $probe: $VERDICT ($REASON) in ${DUR}s [valid $valid/$N_PROBES, miss $miss, infra $infra]"
done

say "════ DISCRIMINATOR COMPLETE: $valid valid scrambled draws, $miss MISS, $infra INFRA ════"
column -t "$MANIFEST" | tee -a "$RUN_DIR/orchestrator.log"
if [ "$miss" -gt 0 ]; then
  say "VERDICT: the lottery is REAL on clean disks (observed $miss/$valid)."
  say "NEXT: byte-diff a MISS probe's archived caches vs a CORRECT probe's"
  say "      ($GCS_RUN/probe<i>/) — deterministic prefill ⇒ byte-equal unless"
  say "      the WRITE side corrupts; then the F3 bisect on the guilty side."
elif [ "$valid" -ge "$N_PROBES" ]; then
  say "VERDICT: $valid/$valid CORRECT on scrambled draws — rejects a true"
  say "≥1/7 lottery at ~95% ((6/7)^20=4.6%). The gate2 failure was most"
  say "likely operational (disk pressure). Ops fixes stand; proceed to the"
  say "F7/F8 cells and RE-GATE — with the disk watchdog + miss-abort armed,"
  say "a residual rare lottery is caught by the gate itself, not hidden."
else
  say "INCONCLUSIVE: probe budget exhausted at $valid/$N_PROBES valid draws — read the INFRA reasons."
fi