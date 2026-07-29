#!/bin/bash
# chain_xprof_gpqa.sh — the 07-29 evening pod chain, one self-contained task
# (terminal-close-proof): wait for the running bench to finish → E0 decode
# xprof of BOTH 256K arms (docs/18's unanimous gate; decode-only capture
# forced via PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD — the prior
# attempt's prefill_only trap) → launch GPQA-198@16K overnight (owner go
# given 07-29). Serialized: each stage owns the pod alone.
set -u
ZONE=us-central2-b
POD=db-v4-64-od
PIN=4647a8fbc
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
TAG=xprof256k_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=~/glm-run/$TAG
mkdir -p "$RUN_DIR"
say() { echo "[chain $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

PROF='PHASED_PROFILING_DIR=/tmp/glm-prof PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=20 PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD=200000'
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

say "waiting for the running bench to release the pod..."
while pgrep -f "bench_run[.]sh|run_bench[.]py" >/dev/null 2>&1; do sleep 120; done
say "pod free — starting E0 xprof"

profile_arm() {  # $1=label  $2=raylet envs  $3=driver envs
  local label="$1" renv="$2" denv="$3" try
  for try in 1 2 3; do
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command='rm -rf /tmp/glm-prof; mountpoint -q ~/gcs-models || gcsfuse --implicit-dirs -o ro driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1' >/dev/null 2>&1
    EXTRA_ENVS="$renv $PROF $LIBTPU" TPU_MIN_TOKEN_BUCKET=32 \
      bash ~/glm-tpu/scripts/launch_glm_32chip.sh > "$RUN_DIR/launch_${label}_t${try}.log" 2>&1
    local log="$RUN_DIR/prof_${label}_t${try}.log"
    (
      cd ~/glm-tpu/bench || exit 1
      # shellcheck disable=SC1090
      set -a; . ~/glm-tpu/.env; set +a
      # shellcheck disable=SC2086
      env $denv PHASED_PROFILING_DIR=/tmp/glm-prof PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=20 PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD=200000 \
        setsid --wait nohup ~/vllm-env/bin/python -u dsa_throughput.py \
        --ctxs 262144 --num-seqs 1 --measure-tokens 64 --gmu 0.90 \
        --max-batched-tokens 2048 --num-gpu-blocks 66 --max-len 262400 \
        --note "E0 xprof $label ($TAG)" </dev/null > "$log" 2>&1
      echo "DRIVER_EXIT=$?" >> "$log"
    )
    if grep -q "DRIVER_EXIT=0" "$log"; then
      mkdir -p "$RUN_DIR/prof_$label"
      gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker=0 --recurse \
        "$POD:/tmp/glm-prof" "$RUN_DIR/prof_$label/" >/dev/null 2>&1
      local n; n=$(find "$RUN_DIR/prof_$label" -name "*.pb" -o -name "*.xplane*" 2>/dev/null | wc -l)
      say "$label: PROFILED (driver ok; $n trace files pulled from w0)"
      gcloud storage cp -r "$RUN_DIR/prof_$label" "gs://driftbench-dsv4-uc/results/$TAG/" >/dev/null 2>&1
      return 0
    fi
    if grep -aqE "$REFUSAL_RE" "$log"; then
      say "$label try $try: draw REFUSED ($(grep -aoEm1 "$REFUSAL_RE" "$log")) — fresh draw"
      continue
    fi
    say "$label try $try: NON-refusal failure (see $log)"
    return 1
  done
  say "$label: no clean draw in 3 tries"; return 1
}

profile_arm sparse "$BASE_SPARSE" "$SPARSE_DRIVER" || say "WARN: sparse profile FAILED — GPQA proceeds anyway"
profile_arm dense "$BASE_DENSE" "$DENSE_DRIVER" || say "WARN: dense profile FAILED — GPQA proceeds anyway"
say "E0 xprof stage done — launching GPQA-198@16K overnight"

MAX_LEN=32768 MAX_SEQS=4 BLOCKS=64 BENCH_TIMEOUT_S=64800 \
  bash ~/glm-tpu/scripts/bench_run.sh gpqa_diamond 198 16384
RC=$?
say "GPQA chain finished rc=$RC"
exit "$RC"
