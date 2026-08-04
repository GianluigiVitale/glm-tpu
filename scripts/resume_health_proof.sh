#!/bin/bash
# Produce either the fresh protected 5K health evidence required immediately
# before an E0 capture (the default), or the four-cell 128K correctness smoke
# required after accepting a performance lever. One draw per invocation; a
# refused/sick draw exits nonzero with a new provenance directory.
set -uo pipefail

ZONE=us-central2-b
POD=db-v4-64-od
PIN="${E0_PIN:-94b746433}"
LIVE_ROWS_PSUM="${E0_LIVE_ROWS_PSUM:-0}"
MOE_PSUM_FUSION="${E0_MOE_PSUM_FUSION:-0}"
DCP_DECODE_LIVE_ROWS="${E0_DSA_DCP_DECODE_LIVE_ROWS:-0}"
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
PROOF_MODE="${PROOF_MODE:-health5k}"
# Protected proofs have max_num_batched_tokens=2048 and can safely pad every
# non-decode prefill remainder to that already-required bucket. Compile only
# T=32 decode plus T=2048 prefill instead of the seven-size default ladder.
# The driver-side GLM knob is provenance-recorded; TPU_MIN_TOKEN_BUCKET shapes
# the worker base ladder. The post-run log gate below proves the actual set.
PROOF_MIN_TOKEN_BUCKET=2048
PROOF_COMPILATION_SIZES=32
case "$PROOF_MODE" in
  health5k)
    TAG_PREFIX=resume_health
    PROOF_LENGTHS=5000
    PROOF_DEPTHS=0.5
    PROOF_TRIALS=1
    PROOF_NUM_GPU_BLOCKS=32
    PROOF_MAX_LEN=8192
    PROOF_EXPECTED_CELLS=1
    PROOF_NOTE_PREFIX="resume health proof"
    PROOF_LOG_NAME=health.log
    DRIVER_TIMEOUT_DEFAULT=7200
    ;;
  smoke128k)
    TAG_PREFIX=lever_smoke128k
    PROOF_LENGTHS=128000
    PROOF_DEPTHS=0.0,0.05,0.95,1.0
    PROOF_TRIALS=1
    PROOF_NUM_GPU_BLOCKS=68
    PROOF_MAX_LEN=131840
    PROOF_EXPECTED_CELLS=4
    PROOF_NOTE_PREFIX="lever 128K smoke"
    PROOF_LOG_NAME=smoke.log
    DRIVER_TIMEOUT_DEFAULT=14400
    ;;
  *) echo "PROOF_MODE must be health5k or smoke128k" >&2; exit 2 ;;
esac
TAG=${TAG_PREFIX}_$(date -u +%Y%m%dT%H%M%S%NZ)
RUN_DIR=$HOME/glm-run/$TAG
GCS_RUN=gs://driftbench-dsv4-uc/results/$TAG
HARNESS_SHORT=$(git -C "$HOME/glm-tpu" rev-parse --short HEAD) || exit 1
DRIVER_TIMEOUT_S="${DRIVER_TIMEOUT_S:-$DRIVER_TIMEOUT_DEFAULT}"
case "$LIVE_ROWS_PSUM" in
  0|1) ;;
  *) echo "E0_LIVE_ROWS_PSUM must be 0 or 1" >&2; exit 2 ;;
esac
case "$MOE_PSUM_FUSION" in
  0|1) ;;
  *) echo "E0_MOE_PSUM_FUSION must be 0 or 1" >&2; exit 2 ;;
esac
case "$DCP_DECODE_LIVE_ROWS" in
  0|1) ;;
  *) echo "E0_DSA_DCP_DECODE_LIVE_ROWS must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$MOE_PSUM_FUSION" = 1 ] && [ "$LIVE_ROWS_PSUM" != 1 ]; then
  echo "E0_MOE_PSUM_FUSION=1 requires E0_LIVE_ROWS_PSUM=1 for this proof" >&2
  exit 2
fi
if [ "$DCP_DECODE_LIVE_ROWS" = 1 ] &&
    { [ "$LIVE_ROWS_PSUM" != 1 ] || [ "$MOE_PSUM_FUSION" != 1 ]; }; then
  echo "E0_DSA_DCP_DECODE_LIVE_ROWS=1 requires the accepted psum stack" >&2
  exit 2
fi
EXPERIMENT_ENV="GLM_DECODE_LIVE_ROWS_PSUM=$LIVE_ROWS_PSUM GLM_MOE_PSUM_FUSION=$MOE_PSUM_FUSION GLM_DSA_DCP_DECODE_LIVE_ROWS=$DCP_DECODE_LIVE_ROWS"
# Exact PID matcher used by this installed Ray CLI's `ray stop`: import its
# live RAY_PROCESSES corpus and apply the same name-vs-cmdline semantics. The
# enumerator excludes itself/ancestors and the nonce-marked local gcloud
# controller because their command lines carry this code.
# shellcheck disable=SC2016
RAY_ENUM='GLM_CENSUS_CARRIER='"$TAG"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
mkdir -p "$RUN_DIR"
say() { echo "[$PROOF_MODE $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }
say "RUN_DIR=$RUN_DIR"

# Same lease used by e0_capture_arm.sh: health and trace cannot pass their
# censuses concurrently or broad-reset one another.
exec 9>"$HOME/glm-run/.glm_pod_workload.lock"
flock -n 9 || { say "ABORT: another protected pod workflow holds the lock"; exit 1; }

has_8_unique_markers() {
  local file="$1" marker="$2" lines unique
  lines=$(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l)
  unique=$(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l)
  [ "$lines" -eq 8 ] && [ "$unique" -eq 8 ]
}

strict_census() {
  local label="$1"
  local out=$RUN_DIR/census_${label}.txt
  # Literal remote program; brackets prevent the SSH carrier matching itself.
  # shellcheck disable=SC2016
  local cmd='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$RAY_ENUM"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[r]un_bench[.]py|[g]ate_sparse128k[.]sh|[s]tage256k[.]sh|[b]ench_run[.]sh|[e]0_capture_arm[.]sh" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$TAG" gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cmd" > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" CENSUS_OK
}

RAYLET_ENVS='GLM_HEALTH_TAG='"$TAG"' GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
RAYLET_ENVS="$RAYLET_ENVS $EXPERIMENT_ENV"
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=$PROOF_MIN_TOKEN_BUCKET GLM_COMPILATION_SIZES=$PROOF_COMPILATION_SIZES \
GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=4 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 \
GLM_HEALTH_TAG=$TAG GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR=$OOB_DIR \
GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH=$PIN"
DRIVER_ENVS="$DRIVER_ENVS $EXPERIMENT_ENV"

if pgrep -af 'gate_sparse128k[.]sh|stage256k[.]sh|bench_run[.]sh|e0_capture_arm[.]sh|glm_longctx[.]py|dsa_throughput[.]py|run_bench[.]py' \
    > "$RUN_DIR/census_local.txt" 2>&1; then
  say "ABORT: local workload collision"
  exit 1
fi
say "initial zero-work pod census"
strict_census initial || { say "ABORT: shared-pod collision/incomplete census"; exit 1; }
[ -z "${GLM_STATE_HASH_WRITE:-}" ] || { say "ABORT: write mode armed"; exit 1; }
bash "$HOME/glm-tpu/scripts/disk_watchdog.sh" check | tee -a "$RUN_DIR/orchestrator.log" || exit 1

# Golden + clean fork pin on eight unique workers.
# shellcheck disable=SC2016
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='test -r /tmp/golden.json && echo "GOLDEN_OK $(hostname)"; cd ~/tpu-inference || exit; h=$(git rev-parse --short=9 HEAD); d=$(git status --porcelain --untracked-files=no | wc -l); [ "$h" = '"$PIN"' ] && [ "$d" -eq 0 ] && echo "HASH_OK $(hostname)"' \
  > "$RUN_DIR/preflight.txt" 2>&1 || true
if ! has_8_unique_markers "$RUN_DIR/preflight.txt" GOLDEN_OK ||
    ! has_8_unique_markers "$RUN_DIR/preflight.txt" HASH_OK; then
  say "ABORT: golden/pin preflight failed"
  exit 1
fi

printf 'tag=%s\nproof_mode=%s\nlengths=%s\ndepths=%s\ntrials=%s\nnum_gpu_blocks=%s\nmax_len=%s\nmin_token_bucket=%s\ncompilation_sizes=%s\npin=%s\nharness=%s\nraylet_envs=%s\ndriver_envs=%s\ngcs_run=%s\n' \
  "$TAG" "$PROOF_MODE" "$PROOF_LENGTHS" "$PROOF_DEPTHS" "$PROOF_TRIALS" \
  "$PROOF_NUM_GPU_BLOCKS" "$PROOF_MAX_LEN" "$PROOF_MIN_TOKEN_BUCKET" \
  "$PROOF_COMPILATION_SIZES" "$PIN" "$HARNESS_SHORT" "$RAYLET_ENVS" \
  "$DRIVER_ENVS" "$GCS_RUN" > "$RUN_DIR/config.txt" || exit 1

# This is the last command before the broad-reset launcher.
say "immediate pre-launch zero-work pod census"
strict_census prelaunch || { say "ABORT: pre-launch collision"; exit 1; }

LAUNCHED=0
DRIVER_SESSION=""
stop_driver_session() {
  if [[ "$DRIVER_SESSION" =~ ^[0-9]+$ ]] &&
      kill -0 -- "-$DRIVER_SESSION" 2>/dev/null; then
    kill -TERM -- "-$DRIVER_SESSION" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 -- "-$DRIVER_SESSION" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$DRIVER_SESSION" 2>/dev/null || true
  fi
  DRIVER_SESSION=""
}
ownership_census() {
  local label="$1"
  local out=$RUN_DIR/ownership_${label}.txt
  # Every process affected by ray stop must carry this draw's live unique tag.
  # shellcheck disable=SC2016
  local cmd='
owned_env() {
  local f="$1"
  [ -r "$f" ] &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_HEALTH_TAG='"$TAG"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_EXPECT_CODE_HASH='"$PIN"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_DCP=4" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_DSA_MODE=pallas_decode" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_DECODE_LIVE_ROWS_PSUM='"$LIVE_ROWS_PSUM"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_MOE_PSUM_FUSION='"$MOE_PSUM_FUSION"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_DSA_DCP_DECODE_LIVE_ROWS='"$DCP_DECODE_LIVE_ROWS"'"
}
# Ray strips the job env from these two title-rewritten helpers. Admit only
# their exact Linux comm/argv pair and only through a direct exact-owned
# raylet parent; every other env-less process remains foreign.
owned_aux_agent() {
  local p="$1" name arg0 pp
  name=$(cat "/proc/$p/comm" 2>/dev/null) || return 1
  arg0=$(tr "\0" "\n" < "/proc/$p/cmdline" 2>/dev/null | head -n 1) || return 1
  if ! { [ "$name" = "ray::DashboardA" ] && [ "$arg0" = "ray::DashboardAgent" ]; } &&
     ! { [ "$name" = "ray::RuntimeEnv" ] && [ "$arg0" = "ray::RuntimeEnvAgent" ]; }; then
    return 1
  fi
  pp=$(grep "^PPid:" "/proc/$p/status" 2>/dev/null | tr -dc "0-9") || return 1
  [ -n "$pp" ] && [ "$(cat "/proc/$pp/comm" 2>/dev/null)" = "raylet" ] &&
    owned_env "/proc/$pp/environ"
}
tools_ok=1
command -v pgrep >/dev/null 2>&1 || tools_ok=0
command -v fuser >/dev/null 2>&1 || tools_ok=0
sudo -n true >/dev/null 2>&1 || tools_ok=0
ray_pids=$('"$RAY_ENUM"' 2>/dev/null); ray_rc=$?
vllm_pids=$(pgrep -f "VLLM::[E]ngineCore|[R]ayWorkerWrapper" 2>/dev/null || true)
containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?
holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true)
pids=$(printf "%s\n%s\n%s\n" "$ray_pids" "$vllm_pids" "$holders" | tr " " "\n" | grep -E "^[0-9]+$" | sort -un | tr "\n" " ")
bad=""
for p in $pids; do
  if ! owned_env "/proc/$p/environ" && ! owned_aux_agent "$p"; then bad="$bad $p"; fi
done
if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ] || [ -n "$bad" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then
  echo "OWNER_BAD $(hostname) bad=$bad"
elif [ -n "$pids" ]; then
  echo "OWNER_OK $(hostname) state=OWNED pids=$pids"
else
  echo "OWNER_OK $(hostname) state=EMPTY"
fi'
  GLM_CENSUS_CARRIER="$TAG" gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cmd" > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" OWNER_OK
}
stop_owned_ray() {
  local label="$1"
  ownership_census "prestop_${label}" || return 1
  local owned
  owned=$(grep -c ' state=OWNED ' "$RUN_DIR/ownership_prestop_${label}.txt" || true)
  if [ "$owned" -eq 0 ]; then
    say "no Ray-stop candidates after ownership census ($label)"
    strict_census "nostop_${label}"
    return
  fi
  say "stopping positively-owned proof Ray cluster ($label)"
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' \
    >/dev/null 2>&1 || return 1
  sleep 10
  strict_census "poststop_${label}"
}
cleanup() {
  stop_driver_session
  if [[ "${WATCH_PID:-}" =~ ^[0-9]+$ ]]; then
    kill "$WATCH_PID" 2>/dev/null || true
    WATCH_PID=""
  fi
  if [ "$LAUNCHED" -eq 1 ]; then
    if stop_owned_ray exit_cleanup; then
      LAUNCHED=0
    else
      say "REFUSED cleanup: ambiguous/foreign process; health Ray left running"
    fi
  fi
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

ALERT_FILE=$RUN_DIR/DISK_ALERT
ALERT_FILE="$ALERT_FILE" INTERVAL_S=120 setsid nohup \
  bash "$HOME/glm-tpu/scripts/disk_watchdog.sh" watch </dev/null \
  > "$RUN_DIR/disk_watch.log" 2>&1 &
WATCH_PID=$!

say "launch protected DCP4 $PROOF_MODE Ray cluster"
LAUNCHED=1
EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET="$PROOF_MIN_TOKEN_BUCKET" \
  bash "$HOME/glm-tpu/scripts/launch_glm_32chip.sh" \
  > "$RUN_DIR/launch_${PROOF_MODE}.log" 2>&1 || exit 1

# The launcher unmounts OOB; repair must not start until all eight are restored.
# shellcheck disable=SC2016
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command='mountpoint -q ~/gcs-models || gcsfuse --implicit-dirs -o ro --stat-cache-ttl 1h --type-cache-ttl 1h driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1; test -r '"$OOB_DIR"'/model.safetensors.index.json && echo "OOB_OK $(hostname)"' \
  > "$RUN_DIR/oob_postlaunch.txt" 2>&1 || true
has_8_unique_markers "$RUN_DIR/oob_postlaunch.txt" OOB_OK || {
  say "ABORT: post-launch OOB not 8/8"; exit 1; }
[ "$("$HOME/vllm-env/bin/ray" status 2>/dev/null | grep -cE '^ 1 node_' || true)" -eq 8 ] || {
  say "ABORT: Ray not 8/8"; exit 1; }

# Exact live values, not merely variable names, from every raylet.
# shellcheck disable=SC2016
ENV_CMD='P=$(pgrep -x raylet | head -1); f=/tmp/resume_health_env_$$; [ -n "$P" ] && tr "\0" "\n" < /proc/$P/environ > "$f"; if grep -qx "GLM_HEALTH_TAG='"$TAG"'" "$f" && grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" "$f" && grep -qx "GLM_WK_OOB_GOLDEN=/tmp/golden.json" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$PIN"'" "$f" && grep -qx "GLM_DCP=4" "$f" && grep -qx "GLM_DSA_MODE=pallas_decode" "$f" && grep -qx "GLM_DECODE_LIVE_ROWS_PSUM='"$LIVE_ROWS_PSUM"'" "$f" && grep -qx "GLM_MOE_PSUM_FUSION='"$MOE_PSUM_FUSION"'" "$f" && grep -qx "GLM_DSA_DCP_DECODE_LIVE_ROWS='"$DCP_DECODE_LIVE_ROWS"'" "$f"; then echo "HEALTH_ENV_OK $(hostname) GLM_HEALTH_TAG='"$TAG"' GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_STATE_HASH_REF=/tmp/golden.json GLM_EXPECT_CODE_HASH='"$PIN"' GLM_DCP=4 GLM_DSA_MODE=pallas_decode GLM_DECODE_LIVE_ROWS_PSUM='"$LIVE_ROWS_PSUM"' GLM_MOE_PSUM_FUSION='"$MOE_PSUM_FUSION"' GLM_DSA_DCP_DECODE_LIVE_ROWS='"$DCP_DECODE_LIVE_ROWS"'"; else echo "HEALTH_ENV_BAD $(hostname)"; fi; rm -f "$f"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$ENV_CMD" > "$RUN_DIR/health_raylet_envs.txt" 2>&1 || true
has_8_unique_markers "$RUN_DIR/health_raylet_envs.txt" HEALTH_ENV_OK || {
  say "ABORT: exact raylet env census failed"; exit 1; }

LOG=$RUN_DIR/$PROOF_LOG_NAME
PROOF_START_EPOCH=$(date +%s)
say "run exact protected $PROOF_MODE: lengths=$PROOF_LENGTHS depths=$PROOF_DEPTHS trials=$PROOF_TRIALS"
# The single-quoted program is evaluated by the task-owned child bash.
# shellcheck disable=SC2016,SC2086
setsid --wait bash -c '
  cd "$1" || exit 1
  set -a; . "$2"; set +a
  [ -z "${GLM_STATE_HASH_WRITE:-}" ] || exit 91
  unset GLM_STATE_HASH_WRITE
  shift 2
  exec "$@"
' protected-proof "$HOME/glm-tpu/bench" "$HOME/glm-tpu/.env" \
  env $DRIVER_ENVS "$HOME/vllm-env/bin/python" -u glm_longctx.py \
  --lengths "$PROOF_LENGTHS" --depths "$PROOF_DEPTHS" \
  --trials "$PROOF_TRIALS" --max-seqs 1 --gmu 0.90 \
  --max-batched-tokens 2048 --num-gpu-blocks "$PROOF_NUM_GPU_BLOCKS" \
  --max-len "$PROOF_MAX_LEN" --note "$PROOF_NOTE_PREFIX ($TAG)" \
  </dev/null > "$LOG" 2>&1 &
W=$!
DRIVER_SESSION=$W
sleep 1
SID=$(ps -o sid= -p "$W" 2>/dev/null | tr -d ' ')
if kill -0 "$W" 2>/dev/null && [ "$SID" != "$W" ]; then
  say "ABORT: driver session ownership failed"
  exit 1
fi
WAITED=0
while kill -0 "$W" 2>/dev/null; do
  sleep 30
  WAITED=$((WAITED + 30))
  if [ -s "$ALERT_FILE" ]; then
    say "ABORT: disk watcher alert during $PROOF_MODE driver"
    exit 1
  fi
  [ "$WAITED" -lt "$DRIVER_TIMEOUT_S" ] || {
    say "ABORT: $PROOF_MODE driver timeout"; exit 1; }
done
wait "$W" 2>/dev/null
DRIVER_RC=$?
DRIVER_SESSION=""
echo "DRIVER_EXIT=$DRIVER_RC" >> "$LOG"
CORRECT_CELLS=$(grep -cE "\[longctx\] L=$PROOF_LENGTHS .*correct=True" "$LOG" || true)
INCORRECT_CELLS=$(grep -cE "\[longctx\] L=$PROOF_LENGTHS .*correct=False" "$LOG" || true)
if [ "$DRIVER_RC" -ne 0 ] ||
    ! grep -q 'manifest VERIFIED.*repeated 7x across cluster' "$LOG" ||
    [ "$CORRECT_CELLS" -ne "$PROOF_EXPECTED_CELLS" ] ||
    [ "$INCORRECT_CELLS" -ne 0 ]; then
  say "ABORT: $PROOF_MODE driver/manifest/correctness failed (correct=$CORRECT_CELLS expected=$PROOF_EXPECTED_CELLS incorrect=$INCORRECT_CELLS)"
  exit 1
fi

# get_token_paddings logs the base ladder before TPU runner appends
# additional_config.compilation_sizes. Require the exact base, the exact
# driver-side addition, and the resulting backbone precompile shapes. The
# three independent observations prove the final runner table [32, 2048]
# without pretending the pre-append logger emitted it.
sed -E 's/\x1B\[[0-9;]*[mK]//g' "$LOG" |
  grep 'Prepared token paddings:' > "$RUN_DIR/compile_buckets.txt" || {
    say "ABORT: no compiled token-bucket evidence"; exit 1; }
if grep -vF 'Prepared token paddings: [2048]' \
    "$RUN_DIR/compile_buckets.txt" >/dev/null; then
  say "ABORT: unexpected base token-bucket ladder"
  exit 1
fi
if ! grep -Fq "'additional_config': {'compilation_sizes': [32]}" "$LOG"; then
  say "ABORT: decode compilation-size addition absent"
  exit 1
fi
sed -E 's/\x1B\[[0-9;]*[mK]//g' "$LOG" |
  grep "Precompile worker0 backbone --> {'num_tokens':" |
  sed -E "s/.*'num_tokens': ([0-9]+).*/\1/" | sort -nu \
    > "$RUN_DIR/compile_backbone_buckets.txt" || {
      say "ABORT: no backbone compile-bucket evidence"; exit 1; }
if ! printf '32\n2048\n' | diff -u - \
    "$RUN_DIR/compile_backbone_buckets.txt" > \
    "$RUN_DIR/compile_backbone_buckets.diff"; then
  say "ABORT: unexpected final backbone compile-bucket set"
  exit 1
fi

if ! "$HOME/vllm-env/bin/python" - "$HOME/glm-tpu/bench/results.db" \
  "$RUN_DIR/results_ckpt.db" "$RUN_DIR/run_link.json" \
  "$PROOF_NOTE_PREFIX ($TAG)" "$TAG" "$PROOF_MODE" "$PIN" \
  "$HARNESS_SHORT" "$PROOF_LENGTHS" "$PROOF_DEPTHS" "$PROOF_TRIALS" \
  "$PROOF_NUM_GPU_BLOCKS" "$PROOF_MAX_LEN" "$PROOF_START_EPOCH" \
  "$LIVE_ROWS_PSUM" "$MOE_PSUM_FUSION" "$DCP_DECODE_LIVE_ROWS" \
  "$OOB_DIR" <<'PY'
import datetime
import json
import sqlite3
import sys

(
    db, snapshot, out, note, tag, mode, pin, harness, lengths_raw,
    depths_raw, trials_raw, blocks_raw, max_len_raw, started_raw, live_rows,
    moe_fusion, dcp_decode_live_rows, oob_dir,
) = sys.argv[1:]
lengths = [int(x) for x in lengths_raw.split(",")]
depths = [float(x) for x in depths_raw.split(",")]
trials = int(trials_raw)
expected_cells = len(lengths) * len(depths) * trials

src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
src.row_factory = sqlite3.Row
rows = src.execute(
    "SELECT * FROM runs WHERE note = ? ORDER BY run_id", (note,)
).fetchall()
assert len(rows) == 1, f"expected one tagged proof run, got {len(rows)}"
run = dict(rows[0])
env = json.loads(run["env_json"])
assert run["harness_git"] == harness, (run["harness_git"], harness)
assert run["fork_git"] == pin, (run["fork_git"], pin)
assert run["pod"] == "db-v4-64-od", run["pod"]
created = datetime.datetime.fromisoformat(run["created_utc"]).timestamp()
assert created >= float(started_raw) - 5, (created, started_raw)
assert env["attention_path"] == "dsa-sparse:pallas_decode"
assert env["lengths"] == lengths, (env["lengths"], lengths)
assert env["depths"] == depths, (env["depths"], depths)
assert env["trials"] == trials
assert env["num_gpu_blocks"] == int(blocks_raw)
assert env["max_len"] == int(max_len_raw)
assert env["max_batched_tokens"] == 2048 and env["max_seqs"] == 1
os_env = env["os_env"]
expected_os = {
    "GLM_HEALTH_TAG": tag,
    "GLM_EXPECT_CODE_HASH": pin,
    "GLM_MLA_DCP": "1",
    "GLM_DSA_MODE": "pallas_decode",
    "GLM_DSA_DCP": "1",
    "GLM_DCP": "4",
    "GLM_DCP_SCATTER_IMPL": "pageloop",
    "GLM_DSA_SCORER": "xla",
    "GLM_DSA_DCP_PREFILL_ATTN": "segment",
    "GLM_DSA_BT_WIDTH": "owned",
    "GLM_DSA_MERGE_IMPL": "v2",
    "GLM_DSA_OWNED_SEG_IMPL": "v2",
    "GLM_DSA_SEG_GATHER_IMPL": "v2",
    "GLM_DECODE_LIVE_ROWS_PSUM": live_rows,
    "GLM_MOE_PSUM_FUSION": moe_fusion,
    "GLM_DSA_DCP_DECODE_LIVE_ROWS": dcp_decode_live_rows,
    "GLM_COMPILATION_SIZES": "32",
    "GLM_PWAL_NAN_CHECK": "1",
    "GLM_LOAD_NAN_CHECK": "1",
    "GLM_LOAD_CHECKSUM": "1",
    "GLM_STATE_HASH_REF": "/tmp/golden.json",
    "GLM_WK_OOB_DIR": oob_dir,
    "GLM_WK_OOB_GOLDEN": "/tmp/golden.json",
}
assert {k: os_env.get(k) for k in expected_os} == expected_os

items = [dict(r) for r in src.execute(
    "SELECT benchmark,item_id,raw_output,extracted,correct,n_prompt_tokens,"
    "n_gen_tokens,finish_reason FROM items WHERE run_id = ? ORDER BY id",
    (run["run_id"],),
)]
assert len(items) == expected_cells, (len(items), expected_cells)
assert all(r["correct"] == 1 for r in items), items
assert all(r["benchmark"].startswith(f"passkey_L{lengths[0]}_d")
           for r in items), items
# glm_longctx's documented construction contract is at most the requested
# target and within 1%; BPE boundary merges make a fixed 128-token tolerance
# invalid at 128K (the canonical prompt is about 127.36K). Retain a 128-token
# floor for the 5K health proof while scaling correctly for long contexts.
prompt_delta = max(128, (lengths[0] + 99) // 100)
assert all(lengths[0] - prompt_delta <= r["n_prompt_tokens"] <= lengths[0]
           for r in items), items
assert all(r["n_gen_tokens"] > 0 and r["raw_output"] for r in items), items

summary = [dict(r) for r in src.execute(
    "SELECT benchmark,n,metric,value,note FROM summary "
    "WHERE run_id = ? ORDER BY id", (run["run_id"],)
)]
assert len(summary) == expected_cells + 1, summary
assert all(r["metric"] == "acc" and r["value"] == 100.0 for r in summary)

dst = sqlite3.connect(snapshot)
src.backup(dst)
assert dst.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
dst.close()
src.close()
with open(out, "w") as f:
    json.dump({"mode": mode, "run": run, "env": env,
               "items": items, "summary": summary}, f, indent=2)
print("PROOF_RUN_LINK_VALID", mode, run["run_id"], expected_cells)
PY
then
  say "ABORT: $PROOF_MODE results linkage/snapshot failed"
  exit 1
fi
say "$PROOF_MODE compute valid: $CORRECT_CELLS/$PROOF_EXPECTED_CELLS; cleanup proof pending"
stop_owned_ray success_cleanup || { say "ABORT: owned Ray cleanup failed/refused"; exit 1; }
LAUNCHED=0
touch "$RUN_DIR/SUCCESS" || { say "ABORT: could not write SUCCESS marker"; exit 1; }
if [ "$PROOF_MODE" = smoke128k ]; then
  ARCHIVE_LOG_TMP=/tmp/${TAG}_archive.log
  if ! gcloud storage rsync -r "$RUN_DIR" "$GCS_RUN" > "$ARCHIVE_LOG_TMP" 2>&1; then
    say "ABORT: durable smoke archive failed (local evidence retained at $RUN_DIR)"
    exit 1
  fi
  cp "$ARCHIVE_LOG_TMP" "$RUN_DIR/archive.log" || exit 1
  gcloud storage cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/archive.log" \
    "$RUN_DIR/SUCCESS" "$GCS_RUN/" >> "$ARCHIVE_LOG_TMP" 2>&1 || {
      say "ABORT: final smoke archive confirmation failed"; exit 1; }
  say "128K SMOKE PROOF VALID: $RUN_DIR archive=$GCS_RUN"
else
  say "HEALTH PROOF VALID: $RUN_DIR"
fi
