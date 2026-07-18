#!/bin/bash
# dissect_load.sh — THE STAGE-SPLITTER DISSECTION (RESEARCH_LOG 2026-07-18
# 08:50): init-only draws with PWAL UNARMED (so the LOAD check's reject
# byte-dumps fire), GLM_CPU_LOAD_NAN_CHECK=1 (pre-t2j CPU-side verdict,
# non-raising) + GLM_LOAD_NAN_CHECK=1 (device-side verdict + dumps + raise).
# Stops at the FIRST CORRUPT draw and reports both verdicts:
#   CPU flagged + device flagged  => streamer/CPU stage guilty -> the §COST
#                                    pre-authorized local-disk fallback.
#   CPU clean  + device flagged   => H2D staging / device-side -> byte dumps.
# Collects /tmp/glm_load_reject_*.npz from every host immediately.
set -u
ZONE=us-central2-b
POD=db-v4-64-od
RUN_DIR=~/glm-run/dissect_$(date -u +%Y%m%dT%H%M%SZ)
MAX_DRAWS=6
INIT_TIMEOUT_S="${INIT_TIMEOUT_S:-1800}"
PIN=a225d16b4

RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_PWAL_NAN_CHECK=0 GLM_LOAD_NAN_CHECK=1 GLM_CPU_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 RAY_DEDUP_LOGS=0 GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 RAY_DEDUP_LOGS=0 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_PWAL_NAN_CHECK=0 GLM_LOAD_NAN_CHECK=1 GLM_CPU_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 \
GLM_EXPECT_CODE_HASH=$PIN"

mkdir -p "$RUN_DIR"
say() { echo "[dissect $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && git rev-parse --short=9 HEAD' 2>/dev/null | grep -c "$PIN")
[ "${HASHES:-0}" -eq 8 ] || { say "ABORT: ${HASHES:-0}/8 @ $PIN"; exit 1; }
bash ~/glm-tpu/scripts/disk_watchdog.sh check >> "$RUN_DIR/orchestrator.log" 2>&1 || {
  say "ABORT: disk pre-flight"; exit 1; }

for i in $(seq 1 "$MAX_DRAWS"); do
  T0=$(date +%s)
  LOG="$RUN_DIR/draw${i}.log"
  say "── draw $i ──"
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='rm -f /tmp/glm_load_reject_*.npz' >/dev/null 2>&1
  EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch${i}.log" 2>&1
  NODES=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
  [ "${NODES:-0}" -eq 8 ] || { say "draw $i INFRA (nodes ${NODES:-0}/8)"; continue; }
  (
    cd ~/glm-tpu/bench || exit 1
    # shellcheck disable=SC1090
    set -a; . ~/glm-tpu/.env; set +a
    # shellcheck disable=SC2086
    env $DRIVER_ENVS setsid --wait nohup ~/vllm-env/bin/python -u -c "
import engine
llm = engine.build_llm(engine.DEFAULT_MODEL, max_len=131840, max_seqs=1,
                       gmu=0.90, max_batched_tokens=2048, num_gpu_blocks=68,
                       log_extra='dissect init')
print('INIT_OK')
" </dev/null > "$LOG" 2>&1
    echo "DRIVER_EXIT=$?" >> "$LOG"
  ) &
  W=$!; waited=0
  while kill -0 "$W" 2>/dev/null; do
    sleep 20; waited=$((waited + 20))
    [ "$waited" -ge "$INIT_TIMEOUT_S" ] && { pkill -f "dissect init" 2>/dev/null; break; }
  done
  wait "$W" 2>/dev/null
  CPU_FLAGS=$(grep -c "GLM_CPU_LOAD_NAN_CHECK.*BEFORE t2j" "$LOG" || true)
  DEV=$(grep -cE "LoadNanCheckError|LoadChecksumError|GLM_LOAD_NAN_CHECK.*nonfinite_tensors=[1-9]|GLM_LOAD_CHECKSUM.*diverged" "$LOG" || true)
  DUR=$(( $(date +%s) - T0 ))
  if [ "${DEV:-0}" -gt 0 ]; then
    say "draw $i: CORRUPT in ${DUR}s — CPU-side flags: ${CPU_FLAGS:-0}, device flagged: yes"
    say "── collecting reject dumps + verdict lines ──"
    mkdir -p "$RUN_DIR/rejects"
    for w in 0 1 2 3 4 5 6 7; do
      gcloud compute tpus tpu-vm scp "$POD:/tmp/glm_load_reject_*.npz" \
        "$RUN_DIR/rejects/" --zone "$ZONE" --worker=$w >/dev/null 2>&1 || true
    done
    grep -E "GLM_CPU_LOAD_NAN_CHECK.*(BEFORE t2j|SUMMARY)" "$LOG" | tail -20 > "$RUN_DIR/cpu_verdict.txt"
    grep -E "GLM_LOAD_NAN_CHECK|LoadNanCheckError" "$LOG" | grep -viE "fingerprint" | tail -20 > "$RUN_DIR/dev_verdict.txt"
    if [ "${CPU_FLAGS:-0}" -gt 0 ]; then
      say "VERDICT: CPU-SIDE CORRUPT BEFORE t2j ⇒ streamer/CPU stage guilty ⇒ the"
      say "§COST-pre-authorized LOCAL-DISK fallback is the fix."
    else
      say "VERDICT: CPU CLEAN but device corrupt ⇒ H2D staging/device-side ⇒ the"
      say "reject byte-dumps in $RUN_DIR/rejects decide next."
    fi
    exit 0
  fi
  say "draw $i: CLEAN in ${DUR}s (cpu flags ${CPU_FLAGS:-0})"
done
say "NO CORRUPT DRAW in $MAX_DRAWS — rerun (rate fluctuation)."
exit 1
