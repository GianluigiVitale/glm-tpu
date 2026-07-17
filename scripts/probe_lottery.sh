#!/bin/bash
# probe_lottery.sh — THE 14-PROBE DISCRIMINATOR (RESEARCH_LOG 2026-07-13
# 15:40): does the "engine-instance lottery" (gate2's d=0.95 0/11 between
# two perfect engines) recur on clean disks, or was it disk-pressure all
# along?
#
# Protocol: N sequential FRESH ENGINES (each launch = one lottery draw),
# each serving ONE fixed-seed 128K needle at the exact gate2 config
# (recovered verbatim from results.db run 165's env_json). The per-trial
# seed is deterministic (base_seed 12345 -> glm_longctx build_trial), so
# every probe serves the IDENTICAL prompt: gate2's first d=0.95 needle.
# Deterministic prefill => byte-equal post-prefill caches across good
# engines; a bad engine's cache byte-diffs against any good one (the
# scrambler protocol, adapted). GLM_DCP_CACHE_DUMP is raylet-baked
# (HOST-side dump; the traced program is UNCHANGED — audit F3: never lead
# an attribution with trace-time env changes).
#
# Classification (the postmortem lesson — the old watcher's grep pattern
# mis-classified 14/14 INFRA-FAILs as pending): every probe ends as exactly
# one of
#   CORRECT   the [longctx] needle line says correct=True
#   MISS      the [longctx] needle line says correct=False  (a RESULT, the
#             lottery expressing — not a retry candidate)
#   INFRA     no needle line + a named cause (disk alert / <8 ray nodes /
#             raylet env missing / engine build failure / driver death /
#             dump requested-but-absent / timeout). INFRA probes do NOT
#             count toward N — the loop extends (cap MAX_PROBES).
#
# F4 fix (audit): the env blocks are defined ONCE below and used verbatim
# for EVERY launch and EVERY driver — including extensions/retries. There
# is no second launch path that could drop an env.
#
# Decision rule (printed at the end):
#   any MISS on clean disks  => the lottery is REAL -> byte-diff the bad
#                               engine's archived cache vs a good one; F3
#                               bisect on the guilty (write/read) side.
#   14/14 CORRECT            => the "lottery" was operational (disk
#                               pressure) -> ops fixes only; RE-GATE.
#
# Usage: bash ~/glm-tpu/scripts/probe_lottery.sh [--probes N] [--dry-run]
set -u

ZONE=us-central2-b
POD=db-v4-64-od
RUN_DIR=~/glm-run/probe_lottery_$(date -u +%Y%m%dT%H%M%SZ)
N_PROBES=14
MAX_PROBES=20
NEEDLE_TIMEOUT_S="${NEEDLE_TIMEOUT_S:-10800}"   # probe 1 pays the cold XLA compile
PIN=a98c77c9c
DUMP_PREFIX=/tmp/dcp_probe                       # per-host; archived+purged per probe
GCS_RUN=gs://driftbench-dsv4-uc/dumps/$(basename "$RUN_DIR")

# ── THE env blocks (single source of truth — F4). Raylet side mirrors the
# gate2 run-165 env_json + runbook §8 (LIBTPU quoting verified in raylet
# /proc environ); GLM_DCP_SCATTER_IMPL=pageloop MUST STAY (dense-path metal
# verdict; the dense fallback fires INSIDE sparse serving). HEADSPLIT stays
# UNSET (metal-refused composition). Cache dump raylet-baked (worker-side).
RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_EXPECT_CODE_HASH='"$PIN"' GLM_DCP_CACHE_DUMP='"$DUMP_PREFIX"' GLM_DCP_CACHE_DUMP_LAYERS=0,1,2 LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
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
echo -e "probe\tts\tverdict\treason\tlog\tdump_uri\tduration_s" > "$MANIFEST"
ALERT_FILE="$RUN_DIR/DISK_ALERT"

say() { echo "[probe-lottery $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

if (( DRY )); then
  say "DRY RUN — would launch with:"
  say "EXTRA_ENVS=$RAYLET_ENVS"
  say "driver: env $DRIVER_ENVS ... glm_longctx.py --lengths 128000 --depths 0.95 --trials 1"
  exit 0
fi

# 0) Pre-flight: worker pin verification (read-only; sync is a manual,
#    deliberate act — see sync_workers.sh) + disk baseline + watch loop.
say "pre-flight: verifying all 8 workers @ $PIN"
HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && echo "$(hostname) $(git rev-parse --short=9 HEAD)"' \
  2>/dev/null | grep -E "^t1v-")
echo "$HASHES" | tee -a "$RUN_DIR/orchestrator.log"
N_OK=$(echo "$HASHES" | grep -c "$PIN" || true)
[ "$N_OK" -eq 8 ] || { say "ABORT: $N_OK/8 workers at $PIN — run sync_workers.sh first."; exit 1; }
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
  LOG="$RUN_DIR/probe${probe}.log"
  say "── probe $probe (valid so far: $valid/$N_PROBES) ──"

  # per-probe alert window: note the alert file's current size
  ALERT_LINES_BEFORE=$(wc -l < "$ALERT_FILE" 2>/dev/null || echo 0)

  # 1) fresh engine = fresh lottery draw (launcher does stop-hygiene first)
  EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch${probe}.log" 2>&1
  NODES=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
  if [ "$NODES" -ne 8 ]; then
    say "probe $probe: INFRA (ray nodes $NODES/8)"
    echo -e "$probe\t$TS\tINFRA\tray_nodes_$NODES\t$LOG\t-\t$(( $(date +%s) - T0 ))" >> "$MANIFEST"
    infra=$((infra + 1)); continue
  fi
  # raylet env verification (the EXTRA_ENVS lesson): every host must carry
  # the baked envs — checked on ONE sentinel var + the dump path.
  ENVOK=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='P=$(pgrep -x raylet | head -1); tr "\0" "\n" < /proc/$P/environ | grep -c "GLM_DCP_CACHE_DUMP='"$DUMP_PREFIX"'"' \
    2>/dev/null | grep -c "^1" || true)
  if [ "$ENVOK" -ne 8 ]; then
    say "probe $probe: INFRA (raylet envs present on $ENVOK/8 hosts)"
    echo -e "$probe\t$TS\tINFRA\traylet_env_$ENVOK\t$LOG\t-\t$(( $(date +%s) - T0 ))" >> "$MANIFEST"
    infra=$((infra + 1)); continue
  fi

  # 2) ONE fixed-seed needle (gate2's first d=0.95 needle), setsid'd driver
  # setsid --wait: a new session (immune to tool-shell teardown SIGTERM —
  # the standing driver rule) that the wrapper still BLOCKS on for the real
  # exit code. Bare `setsid` in the foreground forks and returns instantly
  # (the documented landmine) — --wait is load-bearing here.
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u glm_longctx.py \
      --lengths 128000 --depths 0.95 --trials 1 --max-seqs 1 --gmu 0.90 \
      --max-batched-tokens 2048 --num-gpu-blocks 68 --max-len 131840 \
      --note "probe-lottery p${probe} $(basename "$RUN_DIR")" \
      </dev/null > "$LOG" 2>&1
    echo "DRIVER_EXIT=$?" >> "$LOG"
  ) &
  DRIVER_WRAPPER=$!
  SECONDS_WAITED=0
  while kill -0 "$DRIVER_WRAPPER" 2>/dev/null; do
    sleep 30
    SECONDS_WAITED=$((SECONDS_WAITED + 30))
    if [ "$SECONDS_WAITED" -ge "$NEEDLE_TIMEOUT_S" ]; then
      say "probe $probe: timeout after ${NEEDLE_TIMEOUT_S}s — killing driver"
      # Kill the python by its distinctive --note arg (its own session —
      # NEVER pkill this orchestrator's session; the next launch's
      # stop-hygiene clears any engine-side stragglers).
      pkill -f "probe-lottery p${probe} " 2>/dev/null
      break
    fi
  done
  wait "$DRIVER_WRAPPER" 2>/dev/null

  # 3) classify — needle line first, then named INFRA causes.
  #    NEVER grep bare "PASS" (hlo_passes.cc). The [longctx] result lines
  #    carry correct=True/False.
  NEEDLE_LINE=$(grep -E "correct=(True|False)" "$LOG" | head -1 || true)
  ALERT_LINES_AFTER=$(wc -l < "$ALERT_FILE" 2>/dev/null || echo 0)
  DUMPS_OK=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='ls '"$DUMP_PREFIX"'*.npz >/dev/null 2>&1 && echo Y || echo N' \
    2>/dev/null | grep -c "^Y" || true)
  if [ -n "$NEEDLE_LINE" ]; then
    if echo "$NEEDLE_LINE" | grep -q "correct=True"; then VERDICT=CORRECT; REASON=-;
    else VERDICT=MISS; REASON=needle_wrong; miss=$((miss + 1)); fi
    # a verdict on a probe whose window had a disk alert is TAINTED -> INFRA
    if [ "$ALERT_LINES_AFTER" -gt "$ALERT_LINES_BEFORE" ]; then
      VERDICT=INFRA; REASON=disk_alert_in_window
      [ "$REASON" = needle_wrong ] || true
    fi
  elif [ "$ALERT_LINES_AFTER" -gt "$ALERT_LINES_BEFORE" ]; then
    VERDICT=INFRA; REASON=disk_alert
  elif [ "$SECONDS_WAITED" -ge "$NEEDLE_TIMEOUT_S" ]; then
    VERDICT=INFRA; REASON=timeout
  elif ! grep -q "DRIVER_EXIT=0" "$LOG"; then
    VERDICT=INFRA; REASON=driver_exit_$(grep -oE "DRIVER_EXIT=[0-9]+" "$LOG" | tail -1 | cut -d= -f2)
  else
    VERDICT=INFRA; REASON=no_needle_line
  fi
  # dump honesty: a probe whose dumps never landed cannot be byte-diffed
  if [ "$VERDICT" != "INFRA" ] && [ "$DUMPS_OK" -ne 8 ]; then
    say "probe $probe: dumps on $DUMPS_OK/8 hosts (requested-but-absent) — verdict kept, byte-diff coverage partial"
    REASON="${REASON},dumps_${DUMPS_OK}/8"
  fi

  # 4) archive this probe's post-prefill caches (last step per proc), purge the rest
  SRC_GLOB="$DUMP_PREFIX*.npz" bash ~/glm-tpu/scripts/dump_archiver.sh \
    archive "$(basename "$RUN_DIR")/probe${probe}" --last-step-only \
    >> "$RUN_DIR/orchestrator.log" 2>&1
  DUR=$(( $(date +%s) - T0 ))
  echo -e "$probe\t$TS\t$VERDICT\t$REASON\t$LOG\t$GCS_RUN/probe${probe}/\t$DUR" >> "$MANIFEST"
  say "probe $probe: $VERDICT ($REASON) in ${DUR}s"
  if [ "$VERDICT" != "INFRA" ]; then valid=$((valid + 1)); else infra=$((infra + 1)); fi
done

say "════ DISCRIMINATOR COMPLETE: $valid valid draws ($miss MISS), $infra INFRA ════"
column -t "$MANIFEST" | tee -a "$RUN_DIR/orchestrator.log"
if [ "$miss" -gt 0 ]; then
  say "VERDICT: the lottery is REAL on clean disks (rate ~$miss/$valid)."
  say "NEXT: byte-diff a MISS probe's archived caches vs a CORRECT probe's"
  say "      ($GCS_RUN/probe<i>/) — deterministic prefill ⇒ byte-equal unless"
  say "      the WRITE side corrupts; then the F3 bisect on the guilty side."
elif [ "$valid" -ge "$N_PROBES" ]; then
  say "VERDICT: $valid/$valid CORRECT — the 'lottery' was operational (disk"
  say "pressure). Ops fixes stand; proceed to the F7/F8 cells and RE-GATE."
else
  say "INCONCLUSIVE: probe budget exhausted with only $valid valid draws — read the INFRA reasons."
fi
