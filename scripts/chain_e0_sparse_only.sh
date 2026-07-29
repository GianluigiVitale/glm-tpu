#!/bin/bash
# chain_e0_aime.sh — 07-29 night chain #2 (detached): E0 decode capture via
# the jax profiler SERVER (the phase-based profiler cannot see decode steps
# — RESEARCH_LOG 07-29 17:30) for BOTH 256K arms, then AIME-2026 n=30
# overnight (ungated, cached; GPQA is BLOCKED on a valid owner HF token).
# Capture timing: dsa_throughput now prints PASSA wall + PASSB-START
# markers; pass B's decode window opens ~T_A after PASSB-START.
set -u
ZONE=us-central2-b
POD=db-v4-64-od
PIN=4647a8fbc
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
TAG=e0sparse_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
mkdir -p "$RUN_DIR"
say() { echo "[chain2 $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

SRV='USE_JAX_PROFILER_SERVER=1 JAX_PROFILER_SERVER_PORT=9999'
BASE_SPARSE='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_EXPECT_CODE_HASH='"$PIN"
BASE_DENSE='GLM_MLA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_NAN_CHECK_IGNORE=.self_attn.indexer. GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_STATE_HASH_REF_IGNORE=.self_attn.indexer. GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_EXPECT_CODE_HASH='"$PIN"
LIBTPU='LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
DRIVER_COMMON="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_WK_OOB_DIR=$OOB_DIR GLM_EXPECT_CODE_HASH=$PIN"
SPARSE_DRIVER="$DRIVER_COMMON GLM_STATE_HASH_REF=/tmp/golden.json GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2"
DENSE_DRIVER="$DRIVER_COMMON GLM_STATE_HASH_REF=/tmp/golden.json GLM_STATE_HASH_REF_IGNORE=.self_attn.indexer. GLM_LOAD_NAN_CHECK_IGNORE=.self_attn.indexer. GLM_MLA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop"
REFUSAL_RE="StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|CodeFingerprintMismatchError"

say "waiting for the pod..."
while pgrep -f "bench_run[.]sh|run_bench[.]py|dsa_throughput[.]py" >/dev/null 2>&1; do sleep 120; done

profile_arm() {  # $1=label $2=raylet $3=driver
  local label="$1" renv="$2" denv="$3" try
  for try in 1 2 3; do
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command='test -r /home/gianl/gcs-models/models/GLM-5.2-FP8/model.safetensors.index.json || { fusermount -u ~/gcs-models 2>/dev/null; sleep 1; gcsfuse --implicit-dirs -o ro --stat-cache-ttl 1h --type-cache-ttl 1h driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1; }; test -r /home/gianl/gcs-models/models/GLM-5.2-FP8/model.safetensors.index.json && echo OOB_OK' 2>/dev/null | grep -c OOB_OK | xargs -I{} sh -c '[ {} -eq 8 ] || echo MOUNT_INCOMPLETE' >> "$RUN_DIR/orchestrator.log"
    EXTRA_ENVS="$renv $SRV $LIBTPU" TPU_MIN_TOKEN_BUCKET=32 \
      bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch_${label}_t${try}.log" 2>&1
    local log="$RUN_DIR/prof_${label}_t${try}.log"
    (
      cd ~/glm-tpu/bench || exit 1
      # shellcheck disable=SC1090
      set -a; . ~/glm-tpu/.env; set +a
      # shellcheck disable=SC2086
      env $denv setsid --wait nohup ~/vllm-env/bin/python -u dsa_throughput.py \
        --ctxs 262144 --num-seqs 1 --measure-tokens 256 --gmu 0.90 \
        --max-batched-tokens 2048 --num-gpu-blocks 66 --max-len 262400 \
        --note "E0 jaxprof $label ($TAG)" </dev/null > "$log" 2>&1
      echo "DRIVER_EXIT=$?" >> "$log"
    ) &
    local W=$!
    # capture trigger: PASSA gives T_A; PASSB-START + T_A + 90s = decode window
    (
      local ta=""
      while kill -0 "$W" 2>/dev/null; do
        ta=$(grep -aom1 "PASSA ctx=262144 wall=[0-9.]*" "$log" 2>/dev/null | grep -o "[0-9.]*$" || true)
        [ -n "$ta" ] && break; sleep 20
      done
      [ -z "$ta" ] && exit 0
      while kill -0 "$W" 2>/dev/null; do
        grep -aq "PASSB-START ctx=262144" "$log" 2>/dev/null && break; sleep 5
      done
      sleep $(printf '%.0f' "$(echo "$ta + 90" | bc)")
      say "$label: triggering 25s decode capture (T_A=${ta}s)"
      ~/vllm-env/bin/python -m jax.collect_profile 9999 25000 \
        --log_dir "$RUN_DIR/jaxprof_$label" --no_perfetto_link --host localhost \
        >> "$RUN_DIR/capture_$label.log" 2>&1 \
        || { sleep 30; ~/vllm-env/bin/python -m jax.collect_profile 9999 20000 \
             --log_dir "$RUN_DIR/jaxprof_$label" --no_perfetto_link --host localhost \
             >> "$RUN_DIR/capture_$label.log" 2>&1; }
    ) &
    local CAP=$!
    wait "$W" 2>/dev/null
    wait "$CAP" 2>/dev/null
    if grep -q "DRIVER_EXIT=0" "$log"; then
      local n; n=$(find "$RUN_DIR/jaxprof_$label" -name "*.xplane.pb" 2>/dev/null | wc -l)
      say "$label: driver ok; $n xplane files captured"
      gcloud storage cp -r "$RUN_DIR/jaxprof_$label" "gs://driftbench-dsv4-uc/results/$TAG/" >/dev/null 2>&1 || true
      [ "$n" -gt 0 ] && return 0
      say "$label: NO trace captured — retrying arm"
      continue
    fi
    grep -aqE "$REFUSAL_RE" "$log" && { say "$label try $try: draw REFUSED — fresh draw"; continue; }
    say "$label try $try: NON-refusal failure — retrying once (see $log)"; continue
  done
  say "$label: failed in 3 tries"; return 1
}

profile_arm sparse "$BASE_SPARSE" "$SPARSE_DRIVER" || say "WARN: sparse capture failed"





RC=0
say "chain2 finished rc=$RC"
exit "$RC"
