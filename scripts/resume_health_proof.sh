#!/bin/bash
# Produce the fresh protected 5K health evidence required immediately before
# an E0 resume capture. One draw per invocation; a refused/sick draw exits
# nonzero and must be redrawn with a new run directory.
set -uo pipefail

ZONE=us-central2-b
POD=db-v4-64-od
PIN="${E0_PIN:-94b746433}"
LIVE_ROWS_PSUM="${E0_LIVE_ROWS_PSUM:-0}"
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
TAG=resume_health_$(date -u +%Y%m%dT%H%M%S%NZ)
RUN_DIR=$HOME/glm-run/$TAG
DRIVER_TIMEOUT_S="${DRIVER_TIMEOUT_S:-7200}"
case "$LIVE_ROWS_PSUM" in
  0|1) ;;
  *) echo "E0_LIVE_ROWS_PSUM must be 0 or 1" >&2; exit 2 ;;
esac
EXPERIMENT_ENV="GLM_DECODE_LIVE_ROWS_PSUM=$LIVE_ROWS_PSUM"
# Exact PID matcher used by this installed Ray CLI's `ray stop`: import its
# live RAY_PROCESSES corpus and apply the same name-vs-cmdline semantics. The
# enumerator excludes itself/ancestors and the nonce-marked local gcloud
# controller because their command lines carry this code.
# shellcheck disable=SC2016
RAY_ENUM='GLM_CENSUS_CARRIER='"$TAG"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
mkdir -p "$RUN_DIR"
say() { echo "[resume-health $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }
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
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
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

printf 'tag=%s\npin=%s\nraylet_envs=%s\ndriver_envs=%s\n' \
  "$TAG" "$PIN" "$RAYLET_ENVS" "$DRIVER_ENVS" > "$RUN_DIR/config.txt" || exit 1

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
  local cmd='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$RAY_ENUM"' 2>/dev/null); ray_rc=$?; vllm_pids=$(pgrep -f "VLLM::[E]ngineCore|[R]ayWorkerWrapper" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); pids=$(printf "%s\n%s\n%s\n" "$ray_pids" "$vllm_pids" "$holders" | tr " " "\n" | grep -E "^[0-9]+$" | sort -un | tr "\n" " "); bad=""; for p in $pids; do f=/proc/$p/environ; if [ ! -r "$f" ] || ! tr "\0" "\n" < "$f" | grep -qx "GLM_HEALTH_TAG='"$TAG"'" || ! tr "\0" "\n" < "$f" | grep -qx "GLM_EXPECT_CODE_HASH='"$PIN"'" || ! tr "\0" "\n" < "$f" | grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" || ! tr "\0" "\n" < "$f" | grep -qx "GLM_DCP=4" || ! tr "\0" "\n" < "$f" | grep -qx "GLM_DSA_MODE=pallas_decode" || ! tr "\0" "\n" < "$f" | grep -qx "GLM_DECODE_LIVE_ROWS_PSUM='"$LIVE_ROWS_PSUM"'"; then bad="$bad $p"; fi; done; if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ] || [ -n "$bad" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "OWNER_BAD $(hostname) bad=$bad"; elif [ -n "$pids" ]; then echo "OWNER_OK $(hostname) state=OWNED pids=$pids"; else echo "OWNER_OK $(hostname) state=EMPTY"; fi'
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
  say "stopping positively-owned health Ray cluster ($label)"
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' \
    >/dev/null 2>&1 || return 1
  sleep 10
  strict_census "poststop_${label}"
}
cleanup() {
  stop_driver_session
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

say "launch protected DCP4 health Ray cluster"
LAUNCHED=1
EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
  bash "$HOME/glm-tpu/scripts/launch_glm_32chip.sh" \
  > "$RUN_DIR/launch_health.log" 2>&1 || exit 1

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
ENV_CMD='P=$(pgrep -x raylet | head -1); f=/tmp/resume_health_env_$$; [ -n "$P" ] && tr "\0" "\n" < /proc/$P/environ > "$f"; if grep -qx "GLM_HEALTH_TAG='"$TAG"'" "$f" && grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" "$f" && grep -qx "GLM_WK_OOB_GOLDEN=/tmp/golden.json" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$PIN"'" "$f" && grep -qx "GLM_DCP=4" "$f" && grep -qx "GLM_DSA_MODE=pallas_decode" "$f" && grep -qx "GLM_DECODE_LIVE_ROWS_PSUM='"$LIVE_ROWS_PSUM"'" "$f"; then echo "HEALTH_ENV_OK $(hostname) GLM_HEALTH_TAG='"$TAG"' GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_STATE_HASH_REF=/tmp/golden.json GLM_EXPECT_CODE_HASH='"$PIN"' GLM_DCP=4 GLM_DSA_MODE=pallas_decode GLM_DECODE_LIVE_ROWS_PSUM='"$LIVE_ROWS_PSUM"'"; else echo "HEALTH_ENV_BAD $(hostname)"; fi; rm -f "$f"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$ENV_CMD" > "$RUN_DIR/health_raylet_envs.txt" 2>&1 || true
has_8_unique_markers "$RUN_DIR/health_raylet_envs.txt" HEALTH_ENV_OK || {
  say "ABORT: exact raylet env census failed"; exit 1; }

LOG=$RUN_DIR/health.log
say "run exact protected 5K needle"
# The single-quoted program is evaluated by the task-owned child bash.
# shellcheck disable=SC2016,SC2086
setsid --wait bash -c '
  cd "$1" || exit 1
  set -a; . "$2"; set +a
  [ -z "${GLM_STATE_HASH_WRITE:-}" ] || exit 91
  unset GLM_STATE_HASH_WRITE
  shift 2
  exec "$@"
' resume-health "$HOME/glm-tpu/bench" "$HOME/glm-tpu/.env" \
  env $DRIVER_ENVS "$HOME/vllm-env/bin/python" -u glm_longctx.py \
  --lengths 5000 --depths 0.5 --trials 1 --max-seqs 1 --gmu 0.90 \
  --max-batched-tokens 2048 --num-gpu-blocks 32 --max-len 8192 \
  --note "resume health proof ($TAG)" </dev/null > "$LOG" 2>&1 &
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
  [ "$WAITED" -lt "$DRIVER_TIMEOUT_S" ] || {
    say "ABORT: health driver timeout"; exit 1; }
done
wait "$W" 2>/dev/null
DRIVER_RC=$?
DRIVER_SESSION=""
echo "DRIVER_EXIT=$DRIVER_RC" >> "$LOG"
if [ "$DRIVER_RC" -ne 0 ] ||
    ! grep -q 'manifest VERIFIED.*repeated 7x across cluster' "$LOG" ||
    ! grep -q '\[longctx\] L=5000 .*correct=True' "$LOG"; then
  say "ABORT: health driver/manifest/needle failed"
  exit 1
fi

if ! "$HOME/vllm-env/bin/python" - "$HOME/glm-tpu/bench/results.db" \
  "$RUN_DIR/results_ckpt.db" <<'PY'
import sqlite3
import sys

src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
assert dst.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
dst.close()
src.close()
PY
then
  say "ABORT: results.db snapshot/integrity failed"
  exit 1
fi
say "health compute valid; cleanup proof pending"
stop_owned_ray success_cleanup || { say "ABORT: owned Ray cleanup failed/refused"; exit 1; }
LAUNCHED=0
touch "$RUN_DIR/SUCCESS" || { say "ABORT: could not write SUCCESS marker"; exit 1; }
say "HEALTH PROOF VALID: $RUN_DIR"
