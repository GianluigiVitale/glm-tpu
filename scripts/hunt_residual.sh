#!/bin/bash
# hunt_residual.sh — THE RESIDUAL DISSECTION HUNT (RESEARCH_LOG 2026-07-18
# 21:30; corpus verdict: pageloop-family H1 vs actor-order H2).
#
# Each draw = ONE engine serving the full LADDER (5K/32K/128K x d=0.5/1.0,
# FIXED seed => identical needle content across engines => byte-diffable), with
# the content dumps (22 slots), BOTH dcp guards, and the full integrity stack
# armed. A sick engine thereby dissects ITSELF: its ladder is the
# length-dependence profile, its dumps the specimen; healthy draws produce the
# matched baseline that has never existed. Design choices (banked):
#   - NO health probe: sick engines must SERVE the ladder, not be relaunched.
#   - NO scrambler: gate3 drew sick engines without one (~1/7); launches
#     differ in HBM history naturally. (probe_lottery keeps the scrambled
#     protocol if content-forcing is needed later.)
#   - Guards raise loud: GLM_DCP_ASSERT_SHARDING trip = the H2 (mesh/rank
#     order) signature; PWAL/LOAD/CHECKSUM raise = load-class (should be ~0
#     post-t2j-fix); ladder MISS with all clean = H1 (pageloop-family read).
#   - Dumps archived EVERY draw (healthy baselines included — the gate3
#     archiver gap), then purged (disk discipline).
# Stop early once >=1 SICK and >=1 HEALTHY draw are banked (the diff pair).
# Offline next step: runner/dcp_cache_diff.py sick-vs-healthy at matched cells.
set -u
ZONE=us-central2-b
POD=db-v4-64-od
PIN=a225d16b4
TAG=hunt_${SCATTER_IMPL:-flat}_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
GCS_DUMPS=gs://driftbench-dsv4-uc/dumps/$TAG
MAX_DRAWS="${MAX_DRAWS:-10}"
LADDER_TIMEOUT_S="${LADDER_TIMEOUT_S:-12600}"  # 32K-at-gate-geometry cold compile can eat >1h on the first serving draw
DUMP_PREFIX=/tmp/dcp_hunt
SCATTER_IMPL="${SCATTER_IMPL:-}"   # empty = flat (default); "barrier" = the donation-breaking probe arm
DUMP_LAYERS="0,1,2,4"  # 22 slots at 128K filled the disk (step files accumulate across 63 chunks); the victim slot (layer-2 = layer-1 indexer k-cache) + neighbors suffice for the diff

RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_DSA_DCP_SCATTER_IMPL='"$SCATTER_IMPL"' GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=0 GLM_CPU_LOAD_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_DCP_ASSERT_SHARDING=1 GLM_DCP_ASSERT_CACHE_SANITY=1 RAY_DEDUP_LOGS=0 GLM_DCP_CACHE_DUMP='"$DUMP_PREFIX"' GLM_DCP_CACHE_DUMP_LAYERS='"$DUMP_LAYERS"' GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 RAY_DEDUP_LOGS=0 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_DSA_DCP_SCATTER_IMPL=$SCATTER_IMPL GLM_PWAL_NAN_CHECK=0 GLM_CPU_LOAD_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 \
GLM_DCP_ASSERT_SHARDING=1 GLM_DCP_ASSERT_CACHE_SANITY=1 \
GLM_EXPECT_CODE_HASH=$PIN"

mkdir -p "$RUN_DIR"
MANIFEST="$RUN_DIR/manifest.tsv"
echo -e "draw\tverdict\tcells\tguard\tdur_s" > "$MANIFEST"
say() { echo "[hunt $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

if pgrep -f "gate_sparse128k.sh\b" >/dev/null 2>&1; then say "ABORT: gate running"; exit 1; fi
HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && git rev-parse --short=9 HEAD' 2>/dev/null | grep -c "$PIN")
[ "${HASHES:-0}" -eq 8 ] || { say "ABORT: ${HASHES:-0}/8 @ $PIN"; exit 1; }
bash ~/glm-tpu/scripts/disk_watchdog.sh check >> "$RUN_DIR/orchestrator.log" 2>&1 || {
  say "ABORT: disk pre-flight"; exit 1; }

N_SICK=0; N_HEALTHY=0
for i in $(seq 1 "$MAX_DRAWS"); do
  T0=$(date +%s)
  LOG="$RUN_DIR/draw${i}.log"
  say "── draw $i (sick $N_SICK / healthy $N_HEALTHY) ──"
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="rm -f ${DUMP_PREFIX}*.npz" >/dev/null 2>&1
  EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch${i}.log" 2>&1
  NODES=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
  [ "${NODES:-0}" -eq 8 ] || { say "draw $i INFRA (nodes ${NODES:-0}/8)"
    echo -e "$i\tINFRA\tray\t-\t$(( $(date +%s) - T0 ))" >> "$MANIFEST"; continue; }
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u glm_longctx.py \
      --lengths 5000,32000,128000 --depths 0.5,1.0 --trials 1 --max-seqs 1 \
      --gmu 0.90 --max-batched-tokens 2048 --num-gpu-blocks 68 --max-len 131840 \
      --note "residual hunt draw $i ($TAG)" </dev/null > "$LOG" 2>&1
    echo "DRIVER_EXIT=$?" >> "$LOG"
  ) &
  W=$!; waited=0
  while kill -0 "$W" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    FREE_G=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    [ "${FREE_G:-99}" -lt 15 ] && { say "draw $i: local disk <15G — killing ladder (INFRA)"; pkill -f "residual hunt draw $i "; break; }
    [ "$waited" -ge "$LADDER_TIMEOUT_S" ] && { pkill -f "residual hunt draw $i "; break; }
  done
  wait "$W" 2>/dev/null
  NC=$(grep -c "correct=True" "$LOG" || true)
  NM=$(grep -c "correct=False" "$LOG" || true)
  GUARD="-"
  grep -qE "DCPShardingRoundTripError|GLM_DCP_ASSERT_SHARDING.*(fail|trip|mismatch)" "$LOG" && GUARD="SHARDING_TRIP"
  grep -qE "CacheSanity|GLM_DCP_ASSERT_CACHE_SANITY.*(fail|stale)" "$LOG" && GUARD="${GUARD}+SANITY"
  grep -qE "PwalNanCheckError|LoadNanCheckError|LoadChecksumError" "$LOG" && GUARD="${GUARD}+LOADCLASS"
  if echo "$GUARD" | grep -q LOADCLASS; then V=LOAD_REFUSED
  elif [ "${NM:-0}" -gt 0 ] || [ "$GUARD" != "-" ]; then V=SICK; N_SICK=$((N_SICK+1))
  elif [ "${NC:-0}" -eq 6 ]; then V=HEALTHY; N_HEALTHY=$((N_HEALTHY+1))
  else V=INFRA; fi
  DUR=$(( $(date +%s) - T0 ))
  echo -e "$i\t$V\t${NC:-0}/6\t$GUARD\t$DUR" >> "$MANIFEST"
  say "draw $i: $V (${NC:-0}/6 correct, ${NM:-0} miss, guard=$GUARD, ${DUR}s)"
  grep -E "correct=(True|False)" "$LOG" | sed 's/^/  /' | tee -a "$RUN_DIR/orchestrator.log"
  # archive EVERY draw's dumps (healthy baselines included — the gate3 gap)
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all --command='
    H=$(hostname); if ls '"$DUMP_PREFIX"'*.npz >/dev/null 2>&1; then
      tar czf /tmp/hunt_d'"$i"'_$H.tar.gz '"$DUMP_PREFIX"'*.npz &&
      gcloud storage cp /tmp/hunt_d'"$i"'_$H.tar.gz '"$GCS_DUMPS"'/draw'"$i"'_'"$V"'/ &&
      rm -f /tmp/hunt_d'"$i"'_$H.tar.gz '"$DUMP_PREFIX"'*.npz; fi' \
    >> "$RUN_DIR/archive${i}.log" 2>&1
  if [ "$N_SICK" -ge 1 ] && [ "$N_HEALTHY" -ge 1 ]; then
    say "════ DIFF PAIR BANKED (sick $N_SICK, healthy $N_HEALTHY in $i draws) — stopping early ════"
    say "next: runner/dcp_cache_diff.py on $GCS_DUMPS draw dumps at matched cells"
    exit 0
  fi
done
say "════ HUNT ENDED: $N_SICK sick / $N_HEALTHY healthy in $MAX_DRAWS draws ════"
[ "$N_SICK" -ge 1 ] && [ "$N_HEALTHY" -ge 1 ] && exit 0 || exit 1
