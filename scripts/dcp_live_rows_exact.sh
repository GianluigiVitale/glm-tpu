#!/bin/bash
# Protected single-variable TPU A/B for a selected decode lever. The default
# remains GLM_DSA_DCP_DECODE_LIVE_ROWS; scripts/moe_allgather_exact.sh selects
# GLM_MOE_DECODE_ALL_GATHER while keeping the parent live-attention gate fixed.
# Both sides use the accepted live-row-psum + MoE-psum-fusion stack and the
# production DCP8/256K cache geometry. max_batched_tokens=32 deliberately
# compiles only the production pure-decode bucket changed by this lever; dump
# instrumentation makes this a distinct debug HLO anyway, so compiling unused
# prefill buckets through 2048 would buy no E0 cache reuse. Acceptance requires
# exact selected indices/tie order, exact generated tokens/raw output, guarded
# model state, and clean ownership.
set -uo pipefail

ZONE=us-central2-b
POD=db-v4-64-od
EXACT_LEVER="${EXACT_LEVER:-dcp_live_rows}"
case "$EXACT_LEVER" in
  dcp_live_rows)
    DEFAULT_PIN=979f818e0
    TAG_PREFIX=dcp_live_rows_exact
    NOTE_PREFIX="DCP live rows exact"
    CANDIDATE_ARMED_LOG="GLM_DSA_DCP_DECODE_LIVE_ROWS armed"
    ;;
  moe_allgather)
    DEFAULT_PIN=5967dffa4
    TAG_PREFIX=moe_allgather_exact
    NOTE_PREFIX="MoE decode all-gather exact"
    CANDIDATE_ARMED_LOG="GLM_MOE_DECODE_ALL_GATHER armed"
    ;;
  *)
    echo "EXACT_LEVER must be dcp_live_rows or moe_allgather" >&2
    exit 2
    ;;
esac
PIN="${EXACT_PIN:-$DEFAULT_PIN}"
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
RECOVER_OFF=0
RESUME_DIR="${EXACT_RESUME_DIR:-${DCP_EXACT_RESUME_DIR:-}}"
if [ -n "$RESUME_DIR" ]; then
  RUN_DIR=${RESUME_DIR%/}
  case "$RUN_DIR" in
    "$HOME"/glm-run/${TAG_PREFIX}_*) ;;
    *)
      echo "ABORT: resume dir is outside the exact-artifact namespace: $RUN_DIR"
      exit 1
      ;;
  esac
  [ -d "$RUN_DIR" ] || {
    echo "ABORT: resume dir does not exist: $RUN_DIR"
    exit 1
  }
  TAG=${RUN_DIR##*/}
  RECOVER_OFF=1
else
  TAG=${TAG_PREFIX}_$(date -u +%Y%m%dT%H%M%S%NZ)
  RUN_DIR=$HOME/glm-run/$TAG
fi
GCS_RUN=gs://driftbench-dsv4-uc/results/$TAG
HARNESS_SHORT=$(git -C "$HOME/glm-tpu" rev-parse --short HEAD) || exit 1
ORCHESTRATOR_SHORT=$(git -C "$SCRIPT_ROOT" rev-parse --short HEAD) || exit 1
DRIVER_TIMEOUT_S="${DRIVER_TIMEOUT_S:-10800}"
ACTIVE_SIDE=""
DRIVER_SESSION=""
LAUNCHED=0

mkdir -p "$RUN_DIR"
say() {
  echo "[$EXACT_LEVER-exact $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}
say "RUN_DIR=$RUN_DIR"

exec 9>"$HOME/glm-run/.glm_pod_workload.lock"
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the lock"
  exit 1
}

has_8_unique_markers() {
  local file="$1" marker="$2" lines unique
  lines=$(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l)
  unique=$(awk -v marker="$marker" '$1 == marker {print $2}' "$file" |
    sort -u | wc -l)
  [ "$lines" -eq 8 ] && [ "$unique" -eq 8 ]
}

side_gate() {
  case "$1" in
    off) echo 0 ;;
    on) echo 1 ;;
    *) return 1 ;;
  esac
}

dcp_gate_for_side() {
  if [ "$EXACT_LEVER" = dcp_live_rows ]; then
    side_gate "$1"
  else
    echo 1
  fi
}

moe_allgather_gate_for_side() {
  if [ "$EXACT_LEVER" = moe_allgather ]; then
    side_gate "$1"
  else
    echo 0
  fi
}

# Import the exact process-name corpus used by this Ray installation. The
# carrier marker excludes this census's own local/SSH controller processes.
# shellcheck disable=SC2016
RAY_ENUM='GLM_CENSUS_CARRIER='"$TAG"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'

strict_census() {
  local label="$1"
  local out=$RUN_DIR/census_${label}.txt
  # shellcheck disable=SC2016
  local cmd='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$RAY_ENUM"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[r]un_bench[.]py|[g]ate_sparse128k[.]sh|[s]tage256k[.]sh|[b]ench_run[.]sh|[e]0_capture_arm[.]sh" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$TAG" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$cmd" > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" CENSUS_OK
}

pin_census() {
  local suffix="${1:-}"
  local out=$RUN_DIR/pin_census${suffix:+_$suffix}.txt
  # shellcheck disable=SC2016
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='cd ~/tpu-inference || exit 91; h=$(git rev-parse --short=9 HEAD); d=$(git status --porcelain --untracked-files=no | wc -l); if [ "$h" = '"$PIN"' ] && [ "$d" -eq 0 ]; then echo "PIN_OK $(hostname) $h"; else echo "PIN_BAD $(hostname) head=$h dirty=$d"; fi' \
    > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" PIN_OK
}

ensure_oob() {
  local side="$1"
  local out=$RUN_DIR/oob_${side}.txt
  # shellcheck disable=SC2016
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='mountpoint -q ~/gcs-models || gcsfuse --implicit-dirs -o ro --stat-cache-ttl 1h --type-cache-ttl 1h driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1; test -r '"$OOB_DIR"'/model.safetensors.index.json && test -r /tmp/golden.json && echo "OOB_OK $(hostname)"' \
    > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" OOB_OK
}

ownership_census() {
  local side="$1" label="$2"
  local dcp_gate moe_allgather_gate
  dcp_gate=$(dcp_gate_for_side "$side") || return 1
  moe_allgather_gate=$(moe_allgather_gate_for_side "$side") || return 1
  local expected=${TAG}_${side}
  local out=$RUN_DIR/ownership_${side}_${label}.txt
  # shellcheck disable=SC2016
  local cmd='
owned_env() {
  local f="$1"
  [ -r "$f" ] &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_HEALTH_TAG='"$expected"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_EXPECT_CODE_HASH='"$PIN"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_DCP=8" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_DSA_MODE=pallas_decode" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_DECODE_LIVE_ROWS_PSUM=1" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_MOE_PSUM_FUSION=1" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_DSA_DCP_DECODE_LIVE_ROWS='"$dcp_gate"'" &&
    tr "\0" "\n" < "$f" | grep -qx "GLM_MOE_DECODE_ALL_GATHER='"$moe_allgather_gate"'"
}
owned_aux_agent() {
  local p="$1" name arg0 pp
  name=$(cat "/proc/$p/comm" 2>/dev/null) || return 1
  arg0=$(tr "\0" "\n" < "/proc/$p/cmdline" 2>/dev/null | head -n 1) || return 1
  if ! { [ "$name" = "ray::DashboardA" ] && [ "$arg0" = "ray::DashboardAgent" ]; } &&
     ! { [ "$name" = "ray::RuntimeEnv" ] && [ "$arg0" = "ray::RuntimeEnvAgent" ]; } &&
     ! { [ "$name" = "ray::RayWorkerW" ] && [ "$arg0" = "ray::RayWorkerWrapper" ]; } &&
     ! { [ "$name" = "ray::IDLE" ] && [ "$arg0" = "ray::IDLE" ]; }; then
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
  GLM_CENSUS_CARRIER="$TAG" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$cmd" > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" OWNER_OK
}

stop_owned_ray() {
  local side="$1" label="$2"
  ownership_census "$side" "prestop_${label}" || return 1
  local out=$RUN_DIR/ownership_${side}_prestop_${label}.txt owned
  owned=$(grep -c ' state=OWNED ' "$out" || true)
  if [ "$owned" -eq 0 ]; then
    say "no Ray-stop candidates side=$side ($label)"
    strict_census "nostop_${side}_${label}"
    return
  fi
  say "stopping positively-owned Ray cluster side=$side ($label)"
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' \
    >/dev/null 2>&1 || return 1
  sleep 10
  strict_census "poststop_${side}_${label}"
}

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

cleanup() {
  stop_driver_session
  if [[ "${WATCH_PID:-}" =~ ^[0-9]+$ ]]; then
    kill "$WATCH_PID" 2>/dev/null || true
    WATCH_PID=""
  fi
  if [ "$LAUNCHED" -eq 1 ] && [ -n "$ACTIVE_SIDE" ]; then
    if stop_owned_ray "$ACTIVE_SIDE" exit_cleanup; then
      LAUNCHED=0
    else
      say "REFUSED cleanup: ambiguous/foreign process; exactness Ray left running"
    fi
  fi
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

verify_raylet_envs() {
  local side="$1" prefix="$2" dcp_gate moe_allgather_gate
  dcp_gate=$(dcp_gate_for_side "$side") || return 1
  moe_allgather_gate=$(moe_allgather_gate_for_side "$side") || return 1
  local expected=${TAG}_${side}
  local out=$RUN_DIR/raylet_env_${side}.txt
  # shellcheck disable=SC2016
  local cmd='p=$(pgrep -x raylet | head -1); f=/tmp/decode_lever_exact_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "GLM_HEALTH_TAG='"$expected"'" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$PIN"'" "$f" && grep -qx "GLM_MLA_DCP=1" "$f" && grep -qx "GLM_DSA_MODE=pallas_decode" "$f" && grep -qx "GLM_DSA_DCP=1" "$f" && grep -qx "GLM_DCP=8" "$f" && grep -qx "GLM_DCP_SCATTER_IMPL=pageloop" "$f" && grep -qx "GLM_DSA_SCORER=xla" "$f" && grep -qx "GLM_DSA_DCP_PREFILL_ATTN=segment" "$f" && grep -qx "GLM_DSA_BT_WIDTH=owned" "$f" && grep -qx "GLM_DSA_MERGE_IMPL=v2" "$f" && grep -qx "GLM_DSA_OWNED_SEG_IMPL=v2" "$f" && grep -qx "GLM_DSA_SEG_GATHER_IMPL=v2" "$f" && grep -qx "GLM_WRITE_PROBE=1" "$f" && grep -qx "GLM_PWAL_NAN_CHECK=1" "$f" && grep -qx "GLM_LOAD_NAN_CHECK=1" "$f" && grep -qx "GLM_LOAD_CHECKSUM=1" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f" && grep -qx "GLM_DECODE_LIVE_ROWS_PSUM=1" "$f" && grep -qx "GLM_MOE_PSUM_FUSION=1" "$f" && grep -qx "GLM_DSA_DCP_DECODE_LIVE_ROWS='"$dcp_gate"'" "$f" && grep -qx "GLM_MOE_DECODE_ALL_GATHER='"$moe_allgather_gate"'" "$f" && grep -qx "GLM_DSA_DUMP_TOPK='"$prefix"'" "$f" && grep -qx "GLM_DSA_DUMP_TOPK_EVENTS=0" "$f" && grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" "$f" && grep -qx "GLM_WK_OOB_GOLDEN=/tmp/golden.json" "$f"; then echo "RAYLET_ENV_OK $(hostname)"; else echo "RAYLET_ENV_BAD $(hostname)"; fi; rm -f "$f"'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cmd" > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" RAYLET_ENV_OK
}

gather_dumps() {
  local side="$1" prefix="$2"
  local remote_dir=${prefix%/topk.npz}
  local gathered=$RUN_DIR/${side}_gathered
  local flat=$RUN_DIR/${side}_flat
  mkdir -p "$gathered" "$flat"
  local copied=0
  : > "$RUN_DIR/${side}_dump_hosts.txt"
  for worker in 0 1 2 3 4 5 6 7; do
    if gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
        --recurse "$POD:$remote_dir" "$gathered/w$worker" \
        >/dev/null 2>&1; then
      copied=$((copied + 1))
      echo "$worker" >> "$RUN_DIR/${side}_dump_hosts.txt"
    fi
  done
  find "$gathered" -type f -name 'topk.step*.evt*.proc*.npz' \
    -exec cp -n {} "$flat/" \;
  find "$flat" -type f -name '*.npz' -print0 | sort -z |
    xargs -0 -r sha256sum > "$RUN_DIR/${side}_sha256.txt"
  local files
  files=$(wc -l < "$RUN_DIR/${side}_sha256.txt")
  say "side=$side dump gather hosts=$copied files=$files"
  # jax.debug.callback is process-local and can legitimately materialize on
  # only one Ray worker even though the selected indices are replicated.
  # The differ below therefore explicitly uses --allow-missing-procs. Require
  # at least one authenticated callback process plus three aligned events;
  # exact DB token equality supplies the independent cross-arm output proof.
  [ "$copied" -ge 1 ] && [ "$files" -ge 3 ]
}

BASE_RAYLET_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH='"$PIN"' LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false" GLM_DECODE_LIVE_ROWS_PSUM=1 GLM_MOE_PSUM_FUSION=1'
BASE_DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 \
GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla \
GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned \
GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 \
GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 \
GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json \
GLM_WK_OOB_DIR=$OOB_DIR GLM_WK_OOB_GOLDEN=/tmp/golden.json \
GLM_EXPECT_CODE_HASH=$PIN GLM_DECODE_LIVE_ROWS_PSUM=1 GLM_MOE_PSUM_FUSION=1"

run_side() {
  local side="$1"
  local dcp_gate moe_allgather_gate candidate_gate
  dcp_gate=$(dcp_gate_for_side "$side") || return 1
  moe_allgather_gate=$(moe_allgather_gate_for_side "$side") || return 1
  candidate_gate=$(side_gate "$side") || return 1
  local prefix=/tmp/${TAG}_${side}/topk.npz
  local expected=${TAG}_${side}
  local experiment="GLM_HEALTH_TAG=$expected GLM_DSA_DCP_DECODE_LIVE_ROWS=$dcp_gate GLM_MOE_DECODE_ALL_GATHER=$moe_allgather_gate GLM_DSA_DUMP_TOPK=$prefix GLM_DSA_DUMP_TOPK_EVENTS=0"
  local raylet_envs="$BASE_RAYLET_ENVS $experiment"
  local driver_envs="$BASE_DRIVER_ENVS $experiment"
  local log=$RUN_DIR/driver_${side}.log
  local note="$NOTE_PREFIX $side ($TAG)"

  strict_census "prelaunch_${side}" || {
    say "ABORT: dirty pod before side=$side"
    return 1
  }
  say "launch protected production-shape lever=$EXACT_LEVER side=$side candidate_gate=$candidate_gate dcp_gate=$dcp_gate moe_allgather_gate=$moe_allgather_gate"
  ACTIVE_SIDE="$side"
  LAUNCHED=1
  EXTRA_ENVS="$raylet_envs" TPU_MIN_TOKEN_BUCKET=32 \
    bash "$HOME/glm-tpu/scripts/launch_glm_32chip.sh" \
    > "$RUN_DIR/launch_${side}.log" 2>&1 || return 1
  ensure_oob "$side" || {
    say "ABORT: OOB/golden not 8/8 side=$side"
    return 1
  }
  [ "$("$HOME/vllm-env/bin/ray" status 2>/dev/null | grep -cE '^ 1 node_' || true)" -eq 8 ] || {
    say "ABORT: Ray not 8/8 side=$side"
    return 1
  }
  verify_raylet_envs "$side" "$prefix" || {
    say "ABORT: exact raylet env mismatch side=$side"
    return 1
  }

  say "run protected metal exactness side=$side"
  # The single-quoted program is evaluated by the task-owned child bash;
  # driver_envs is intentionally split into environment assignments by env.
  # shellcheck disable=SC2016,SC2086
  setsid --wait bash -c '
    cd "$1" || exit 1
    set -a; . "$2"; set +a
    [ -z "${GLM_STATE_HASH_WRITE:-}" ] || exit 91
    unset GLM_STATE_HASH_WRITE
    export LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"
    shift 2
    exec "$@"
  ' protected-exact "$HOME/glm-tpu/bench" "$HOME/glm-tpu/.env" \
    env $driver_envs "$HOME/vllm-env/bin/python" -u glm_longctx.py \
    --lengths 4096 --depths 0.5 --trials 1 --max-seqs 1 --gmu 0.90 \
    --max-batched-tokens 32 --num-gpu-blocks 66 --max-len 262400 \
    --max-new 2 --note "$note" </dev/null > "$log" 2>&1 &
  local worker=$!
  DRIVER_SESSION=$worker
  sleep 1
  local sid
  sid=$(ps -o sid= -p "$worker" 2>/dev/null | tr -d ' ')
  if kill -0 "$worker" 2>/dev/null && [ "$sid" != "$worker" ]; then
    say "ABORT: driver session ownership failed side=$side"
    return 1
  fi
  local waited=0
  while kill -0 "$worker" 2>/dev/null; do
    sleep 30
    waited=$((waited + 30))
    if [ -s "$ALERT_FILE" ]; then
      say "ABORT: disk watcher alert side=$side"
      return 1
    fi
    [ "$waited" -lt "$DRIVER_TIMEOUT_S" ] || {
      say "ABORT: driver timeout side=$side"
      return 1
    }
  done
  wait "$worker" 2>/dev/null
  local rc=$?
  DRIVER_SESSION=""
  echo "DRIVER_EXIT=$rc" >> "$log"
  [ "$rc" -eq 0 ] || {
    say "ABORT: driver rc=$rc side=$side"
    return 1
  }
  grep -q 'manifest VERIFIED.*repeated 7x across cluster' "$log" || {
    say "ABORT: manifest coverage missing side=$side"
    return 1
  }
  grep -q "GLM_CODE_FINGERPRINT: git=$PIN.*dirty=0" "$log" || {
    say "ABORT: exact fingerprint missing side=$side"
    return 1
  }
  grep -q 'GLM_DECODE_LIVE_ROWS_PSUM armed' "$log" || {
    say "ABORT: accepted live-row psum gate not armed side=$side"
    return 1
  }
  grep -q 'GLM_MOE_PSUM_FUSION armed' "$log" || {
    say "ABORT: accepted MoE psum fusion not armed side=$side"
    return 1
  }
  if [ "$candidate_gate" = 1 ]; then
    grep -q "$CANDIDATE_ARMED_LOG" "$log" || {
      say "ABORT: candidate armed log missing side=$side lever=$EXACT_LEVER"
      return 1
    }
  elif grep -q "$CANDIDATE_ARMED_LOG" "$log"; then
    say "ABORT: candidate unexpectedly armed on control side lever=$EXACT_LEVER"
    return 1
  fi
  gather_dumps "$side" "$prefix" || {
    say "ABORT: dump coverage missing side=$side"
    return 1
  }
  stop_owned_ray "$side" success_cleanup || {
    say "ABORT: owned cleanup failed/refused side=$side"
    return 1
  }
  LAUNCHED=0
  ACTIVE_SIDE=""
  say "side=$side compute/dump/cleanup valid"
}

validate_recovered_off() {
  local log=$RUN_DIR/driver_off.log
  if [ ! -s "$log" ] || ! grep -q '^DRIVER_EXIT=0$' "$log"; then
    say "ABORT: recovered OFF driver evidence is incomplete"
    return 1
  fi
  grep -q 'manifest VERIFIED.*repeated 7x across cluster' "$log" || {
    say "ABORT: recovered OFF manifest coverage missing"
    return 1
  }
  grep -q "GLM_CODE_FINGERPRINT: git=$PIN.*dirty=0" "$log" || {
    say "ABORT: recovered OFF fingerprint mismatch"
    return 1
  }
  grep -q 'GLM_DECODE_LIVE_ROWS_PSUM armed' "$log" || return 1
  grep -q 'GLM_MOE_PSUM_FUSION armed' "$log" || return 1
  [ -s "$RUN_DIR/off_sha256.txt" ] || {
    say "ABORT: recovered OFF dump hashes missing"
    return 1
  }
  sha256sum -c "$RUN_DIR/off_sha256.txt" >/dev/null 2>&1 || {
    say "ABORT: recovered OFF dump hashes failed"
    return 1
  }
  local files
  files=$(wc -l < "$RUN_DIR/off_sha256.txt")
  [ "$files" -ge 3 ] || {
    say "ABORT: recovered OFF dump coverage too small: $files"
    return 1
  }
  say "recovered OFF accepted: driver/state/gates/hashes files=$files"
}

LOCAL_CENSUS_FILE=$RUN_DIR/census_local.txt
[ "$RECOVER_OFF" -eq 0 ] || \
  LOCAL_CENSUS_FILE=$RUN_DIR/census_local_resume.txt
if pgrep -af 'gate_sparse128k[.]sh|stage256k[.]sh|bench_run[.]sh|e0_capture_arm[.]sh|glm_longctx[.]py|dsa_throughput[.]py|run_bench[.]py' \
    > "$LOCAL_CENSUS_FILE" 2>&1; then
  say "ABORT: local workload collision"
  exit 1
fi
[ -z "${GLM_STATE_HASH_WRITE:-}" ] || {
  say "ABORT: write mode armed"
  exit 1
}
say "fresh exact pin and zero-work preflight"
if [ "$RECOVER_OFF" -eq 1 ]; then
  strict_census resume_initial || exit 1
  pin_census resume || exit 1
else
  strict_census initial || exit 1
  pin_census || exit 1
fi
bash "$HOME/glm-tpu/scripts/disk_watchdog.sh" check |
  tee -a "$RUN_DIR/orchestrator.log" || exit 1

if [ "$RECOVER_OFF" -eq 1 ]; then
  grep -qx "tag=$TAG" "$RUN_DIR/config.txt" || exit 1
  grep -qx "pin=$PIN" "$RUN_DIR/config.txt" || exit 1
  grep -qx "harness=$HARNESS_SHORT" "$RUN_DIR/config.txt" || exit 1
  if ! grep -qx "exact_lever=$EXACT_LEVER" "$RUN_DIR/config.txt"; then
    # Artifacts created by the original single-purpose harness predate the
    # selector line; only its historical DCP mode may resume without it.
    [ "$EXACT_LEVER" = dcp_live_rows ] || exit 1
    ! grep -q '^exact_lever=' "$RUN_DIR/config.txt" || exit 1
  fi
  printf 'recovery_orchestrator=%s\nrecovery_utc=%s\n' \
    "$ORCHESTRATOR_SHORT" "$(date -u +%FT%TZ)" >> "$RUN_DIR/config.txt" || exit 1
  cp "$HOME/glm-tpu/scripts/dcp_live_rows_exact.sh" \
    "$RUN_DIR/original_harness.sh" || exit 1
  cp "$SCRIPT_ROOT/scripts/dcp_live_rows_exact.sh" \
    "$RUN_DIR/recovery_harness.sh" || exit 1
else
  printf 'tag=%s\nexact_lever=%s\npin=%s\nharness=%s\norchestrator=%s\nshape=dcp8 blocks66 max_len262400 max_batched_tokens32\nbase_raylet_envs=%s\nbase_driver_envs=%s\ngcs_run=%s\n' \
    "$TAG" "$EXACT_LEVER" "$PIN" "$HARNESS_SHORT" "$ORCHESTRATOR_SHORT" \
    "$BASE_RAYLET_ENVS" "$BASE_DRIVER_ENVS" "$GCS_RUN" \
    > "$RUN_DIR/config.txt" || exit 1
fi

ALERT_FILE=$RUN_DIR/DISK_ALERT
ALERT_FILE="$ALERT_FILE" INTERVAL_S=120 setsid nohup \
  bash "$HOME/glm-tpu/scripts/disk_watchdog.sh" watch </dev/null \
  > "$RUN_DIR/disk_watch.log" 2>&1 &
WATCH_PID=$!

if [ "$RECOVER_OFF" -eq 1 ]; then
  validate_recovered_off || exit 1
else
  run_side off || exit 1
fi
run_side on || exit 1

say "compare selected indices/tie order and exact generated output"
(
  cd "$HOME/tpu-inference" || exit 1
  JAX_PLATFORMS=cpu PYTHONPATH="$HOME/tpu-inference" \
    "$HOME/vllm-env/bin/python" -m tpu_inference.runner.dsa_topk_diff \
    "$RUN_DIR/off_flat/topk" "$RUN_DIR/on_flat/topk" \
    --allow-missing-procs
) > "$RUN_DIR/topk_diff.txt" 2>&1
DIFF_RC=$?
echo "DIFF_EXIT=$DIFF_RC" >> "$RUN_DIR/topk_diff.txt"
[ "$DIFF_RC" -eq 0 ] || exit 1

if ! "$HOME/vllm-env/bin/python" - "$HOME/glm-tpu/bench/results.db" \
    "$RUN_DIR/results_ckpt.db" "$RUN_DIR/token_exactness.json" \
    "$TAG" "$PIN" "$HARNESS_SHORT" "$OOB_DIR" "$EXACT_LEVER" \
    "$NOTE_PREFIX" <<'PY'
import json
import sqlite3
import sys

(
    db, snapshot, out_path, tag, pin, harness, oob_dir, exact_lever,
    note_prefix,
) = sys.argv[1:]
src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
src.row_factory = sqlite3.Row
out = {}
for side in ("off", "on"):
    side_gate = "0" if side == "off" else "1"
    dcp_gate = side_gate if exact_lever == "dcp_live_rows" else "1"
    moe_allgather_gate = side_gate if exact_lever == "moe_allgather" else "0"
    note = f"{note_prefix} {side} ({tag})"
    rows = src.execute("SELECT * FROM runs WHERE note = ?", (note,)).fetchall()
    assert len(rows) == 1, (side, len(rows))
    run = dict(rows[0])
    env = json.loads(run["env_json"])
    os_env = env["os_env"]
    expected = {
        "GLM_HEALTH_TAG": f"{tag}_{side}",
        "GLM_EXPECT_CODE_HASH": pin,
        "GLM_MLA_DCP": "1",
        "GLM_DSA_MODE": "pallas_decode",
        "GLM_DSA_DCP": "1",
        "GLM_DCP": "8",
        "GLM_DCP_SCATTER_IMPL": "pageloop",
        "GLM_DSA_SCORER": "xla",
        "GLM_DSA_DCP_PREFILL_ATTN": "segment",
        "GLM_DSA_BT_WIDTH": "owned",
        "GLM_DSA_MERGE_IMPL": "v2",
        "GLM_DSA_OWNED_SEG_IMPL": "v2",
        "GLM_DSA_SEG_GATHER_IMPL": "v2",
        "GLM_DSA_DCP_DECODE_LIVE_ROWS": dcp_gate,
        "GLM_MOE_DECODE_ALL_GATHER": moe_allgather_gate,
        "GLM_DSA_DUMP_TOPK": f"/tmp/{tag}_{side}/topk.npz",
        "GLM_DSA_DUMP_TOPK_EVENTS": "0",
        "GLM_DECODE_LIVE_ROWS_PSUM": "1",
        "GLM_MOE_PSUM_FUSION": "1",
        "GLM_WRITE_PROBE": "1",
        "GLM_PWAL_NAN_CHECK": "1",
        "GLM_LOAD_NAN_CHECK": "1",
        "GLM_LOAD_CHECKSUM": "1",
        "GLM_STATE_HASH_REF": "/tmp/golden.json",
        "GLM_WK_OOB_DIR": oob_dir,
        "GLM_WK_OOB_GOLDEN": "/tmp/golden.json",
    }
    if (exact_lever == "dcp_live_rows" and
            "GLM_MOE_DECODE_ALL_GATHER" not in os_env):
        # Backward-compatible recovery of a pre-selector DCP exact artifact;
        # the pinned fork did not yet contain this gate, so absence is OFF.
        expected.pop("GLM_MOE_DECODE_ALL_GATHER")
    assert run["fork_git"] == pin, (run["fork_git"], pin)
    assert run["harness_git"] == harness, (run["harness_git"], harness)
    assert env["lengths"] == [4096] and env["depths"] == [0.5]
    assert env["trials"] == 1 and env["max_new"] == 2
    assert env["max_seqs"] == 1 and env["max_batched_tokens"] == 32
    assert env["num_gpu_blocks"] == 66 and env["max_len"] == 262400
    assert {k: os_env.get(k) for k in expected} == expected
    items = [dict(row) for row in src.execute(
        "SELECT gold,raw_output,extracted,correct,n_prompt_tokens,n_gen_tokens,"
        "finish_reason FROM items WHERE run_id = ? ORDER BY id",
        (run["run_id"],),
    )]
    assert len(items) == 1, items
    item = items[0]
    assert item["n_gen_tokens"] == 2 and item["raw_output"], item
    # This is deliberately a two-token exactness probe, not a full-answer
    # correctness benchmark. glm_longctx therefore records correct=0 for the
    # truncated answer. Authenticate that the generated text is the gold
    # prefix, then require the complete item dict to match across OFF/ON.
    raw_prefix = item["raw_output"].strip()
    assert raw_prefix and str(item["gold"]).startswith(raw_prefix), item
    out[side] = {"run": run, "env": env, "item": items[0]}
assert out["off"]["item"] == out["on"]["item"], (
    out["off"]["item"], out["on"]["item"]
)
dst = sqlite3.connect(snapshot)
src.backup(dst)
assert dst.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
dst.close()
src.close()
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print("TOKEN_PREFIX_EXACT", repr(out["off"]["item"]["raw_output"]),
      out["off"]["item"]["n_gen_tokens"])
PY
then
  say "ABORT: exact results linkage/token comparison failed"
  exit 1
fi

evidence_files=(
  "$RUN_DIR"/off_flat/*.npz
  "$RUN_DIR"/on_flat/*.npz
  "$RUN_DIR/topk_diff.txt"
  "$RUN_DIR/token_exactness.json"
)
if [ "$RECOVER_OFF" -eq 1 ]; then
  evidence_files+=("$RUN_DIR/original_harness.sh"
                   "$RUN_DIR/recovery_harness.sh")
fi
sha256sum "${evidence_files[@]}" > "$RUN_DIR/evidence.sha256" || exit 1
touch "$RUN_DIR/SUCCESS" || exit 1
ARCHIVE_LOG_TMP=/tmp/${TAG}_archive.log
if ! gcloud storage rsync -r "$RUN_DIR" "$GCS_RUN" \
    > "$ARCHIVE_LOG_TMP" 2>&1; then
  say "ABORT: durable exactness archive failed (local evidence retained)"
  exit 1
fi
cp "$ARCHIVE_LOG_TMP" "$RUN_DIR/archive.log" || exit 1
gcloud storage cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/archive.log" \
  "$RUN_DIR/SUCCESS" "$GCS_RUN/" >> "$ARCHIVE_LOG_TMP" 2>&1 || {
  say "ABORT: final archive confirmation failed"
  exit 1
}
say "EXACTNESS PASS: selected-set/tie-order + raw output archive=$GCS_RUN"
