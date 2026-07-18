#!/bin/bash
# loader_ab.sh — THE LOADER CONCURRENCY A/B (RESEARCH_LOG 2026-07-18 07:30).
# Question: does RUNAI_STREAMER_CONCURRENCY drive the per-host weight-load
# corruption (whole-row contiguous-range corruption, fp8 bytes → sometimes
# NaN)? N engine INITS per arm (no serving — the stream is the part under
# test), alternating arms to decorrelate temporal drift, with the init-time
# integrity checks armed (GLM_PWAL_NAN_CHECK + GLM_LOAD_NAN_CHECK). Each
# init is classified CORRUPT (any check flagged/raised) / CLEAN / INFRA.
#
# RAY_DEDUP_LOGS=0 everywhere (the 07-18 forensic lesson: dedup destroyed a
# flag's identity). The launcher's own RUNAI_STREAMER_CONCURRENCY=32 is
# overridden by EXTRA_ENVS (later assignment wins within the export).
#
# Sensitivity note: the NaN-class flag is a PROXY (finite corruption is
# invisible); valid for A-vs-B rate comparison under the assumption that
# byte-corruption→NaN probability is arm-independent.
#
# Usage: bash ~/glm-tpu/scripts/loader_ab.sh [--n 8] [--arms "32,8"]
set -u

ZONE=us-central2-b
POD=db-v4-64-od
RUN_DIR=~/glm-run/loader_ab_$(date -u +%Y%m%dT%H%M%SZ)
N_PER_ARM=8
ARMS="32,8"
INIT_TIMEOUT_S="${INIT_TIMEOUT_S:-1800}"
PIN="${PIN:-629c20e84}"   # override after the load-check lands

BASE_RAYLET='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 RAY_DEDUP_LOGS=0 GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 RAY_DEDUP_LOGS=0 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_EXPECT_CODE_HASH=$PIN"

while [ $# -gt 0 ]; do
  case "$1" in
    --n) N_PER_ARM="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_DIR"
MANIFEST="$RUN_DIR/manifest.tsv"
echo -e "idx\tarm\tverdict\tflags\tdur_s" > "$MANIFEST"
say() { echo "[loader-ab $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

# pre-flight
HASHES=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='cd ~/tpu-inference && git rev-parse --short=9 HEAD' 2>/dev/null | grep -c "$PIN")
[ "${HASHES:-0}" -eq 8 ] || { say "ABORT: ${HASHES:-0}/8 @ $PIN"; exit 1; }
bash ~/glm-tpu/scripts/disk_watchdog.sh check >> "$RUN_DIR/orchestrator.log" 2>&1 || {
  say "ABORT: disk pre-flight"; exit 1; }

IFS=',' read -ra ARM_ARR <<< "$ARMS"
idx=0
for i in $(seq 1 "$N_PER_ARM"); do
  for arm in "${ARM_ARR[@]}"; do
    idx=$((idx + 1)); T0=$(date +%s)
    LOG="$RUN_DIR/init_${idx}_c${arm}.log"
    say "── init $idx (concurrency $arm) ──"
    EXTRA_ENVS="$BASE_RAYLET RUNAI_STREAMER_CONCURRENCY=$arm" TPU_MIN_TOKEN_BUCKET=32 \
      bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "${LOG%.log}.launch.log" 2>&1
    NODES=$(~/vllm-env/bin/ray status 2>/dev/null | grep -cE '^ 1 node_' || true)
    if [ "${NODES:-0}" -ne 8 ]; then
      echo -e "$idx\t$arm\tINFRA\tray_${NODES:-0}\t$(( $(date +%s) - T0 ))" >> "$MANIFEST"
      say "init $idx: INFRA (nodes)"; continue
    fi
    # init-only driver: build the engine (streams weights on every host),
    # then exit. Corruption expresses (and, armed, REFUSES) during build.
    (
      cd ~/glm-tpu/bench || exit 1
      # shellcheck disable=SC1090
      set -a; . ~/glm-tpu/.env; set +a
      # shellcheck disable=SC2086
      env $DRIVER_ENVS RUNAI_STREAMER_CONCURRENCY=$arm setsid --wait nohup \
        ~/vllm-env/bin/python -u -c "
import engine
# EXACT gate geometry — its programs are already in every host's warm XLA
# cache (a novel geometry forces a ~30-45min cold compile; init-1 of run
# 002557Z timed out exactly that way). The stream-under-test is geometry-
# independent; the cheap init is the CACHED one.
llm = engine.build_llm(engine.DEFAULT_MODEL, max_len=131840, max_seqs=1,
                       gmu=0.90, max_batched_tokens=2048, num_gpu_blocks=68,
                       log_extra='loader-ab init')
print('INIT_OK')
" </dev/null > "$LOG" 2>&1
      echo "DRIVER_EXIT=$?" >> "$LOG"
    ) &
    W=$!; waited=0
    while kill -0 "$W" 2>/dev/null; do
      sleep 20; waited=$((waited + 20))
      [ "$waited" -ge "$INIT_TIMEOUT_S" ] && { pkill -f "loader-ab init" 2>/dev/null; break; }
    done
    wait "$W" 2>/dev/null
    FLAGS=$(grep -cE "GLM_PWAL_NAN_CHECK.*(NaN|Inf)|GLM_LOAD_NAN_CHECK.*(NaN|Inf|non-finite)" "$LOG" || true)
    if [ "${FLAGS:-0}" -gt 0 ] || grep -qE "PwalNanCheckError|LoadNanCheckError" "$LOG"; then
      V=CORRUPT
    elif grep -q "INIT_OK" "$LOG"; then V=CLEAN
    elif [ "$waited" -ge "$INIT_TIMEOUT_S" ]; then V=INFRA; FLAGS=timeout
    else V=INFRA; FLAGS=driver_$(grep -oE 'DRIVER_EXIT=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
    fi
    echo -e "$idx\t$arm\t$V\t${FLAGS:-0}\t$(( $(date +%s) - T0 ))" >> "$MANIFEST"
    say "init $idx (c=$arm): $V (flags=$FLAGS)"
  done
done

say "════ A/B COMPLETE ════"
column -t "$MANIFEST" | tee -a "$RUN_DIR/orchestrator.log"
for arm in "${ARM_ARR[@]}"; do
  C=$(awk -F'\t' -v a="$arm" '$2==a && $3=="CORRUPT"' "$MANIFEST" | wc -l)
  K=$(awk -F'\t' -v a="$arm" '$2==a && $3=="CLEAN"' "$MANIFEST" | wc -l)
  say "arm c=$arm: CORRUPT $C / CLEAN $K (infra excluded)"
done
say "Interpretation: a large corrupt-rate drop at the lower concurrency ⇒ the"
say "streamer race is the mechanism ⇒ ship the safe setting + keep refusals."
say "No difference ⇒ adapter/dequant audit next, then byte-integrity manifest."
