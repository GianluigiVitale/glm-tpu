#!/usr/bin/env bash
# Capture and seal one fresh, flat-scatter short-context DSA event oracle.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly HARNESS_REPO=/home/gianl/glm-tpu
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly ORACLE_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly INTERNAL_CAPTURE=${GLM_GREENFIELD_DSA_INTERNALS_CAPTURE:-0}
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  readonly OBSERVER_DEV_REPO=/home/gianl/tpu-inference-greenfield-dsa-internal-observer
  readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-83ff4a357
  readonly OBSERVER_BRANCH=greenfield/legacy-dsa-internal-observer
  readonly OBSERVER_COMMIT_DISTANCE=2
  readonly LEGACY_REPO=$OBSERVER_RUNTIME_REPO
  readonly LEGACY_SOURCE_REPO=$OBSERVER_DEV_REPO
  readonly LEGACY_PIN=83ff4a3576602ca844ea090550139a2ff00b0bb1
else
  readonly LEGACY_REPO=$ORACLE_REPO
  readonly LEGACY_SOURCE_REPO=$ORACLE_REPO
  readonly LEGACY_PIN=$ORACLE_PIN
fi
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly DISK_MIN_FREE_GB=10
readonly DISK_WARN_FREE_GB=15
readonly MODEL_ID=zai-org/GLM-5.2-FP8
readonly REFERENCE_8K_DSA_ORACLE=/home/gianl/glm-run/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle
readonly LAYER0_INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly LAYER0_INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly DISTRIBUTED_Q_A_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_association_20260807T231449677046310Z/distributed_q_a_norm_artifact
readonly DISTRIBUTED_Q_A_MANIFEST_SHA=7518e7eff0487f0dc02cd4b0ff1c3d0fc3ef9ca7c43dcded7d809120e30d8c16
readonly DISTRIBUTED_Q_A_CODE_HASH=ea879a24d196f61e238a22ee5bb393d3b6fa938d
readonly INTERNAL_LAYER=model.layers.0.self_attn.attn

PROFILE=${GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE:-2k}
case "$PROFILE" in
  2k)
    TOKEN_ORACLE_TAG=greenfield_short_context_oracle_20260806T202544155912103Z
    TOKEN_ORACLE_SHA=f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19
    BENCHMARK_LENGTH=2040
    BENCHMARK_DEPTH=0.25
    BENCHMARK_MAX_LEN=2560
    BENCHMARK_MAX_BATCHED_TOKENS=2048
    BENCHMARK_NUM_BLOCKS=8
    EXPECTED_BENCHMARK=passkey_L2040_d0.25
    EXPECTED_PROMPT_TOKENS=2034
    EXPECTED_GENERATED_TOKENS=20
    EXPECTED_SEED=283835
    EXPECTED_GOLD=110391
    FIRST_SOURCE_STEP=2
    FIRST_DECODE_POSITION=2034
    TAG_PREFIX=greenfield_short_context_dsa_oracle
    ;;
  8k)
    TOKEN_ORACLE_TAG=greenfield_short_context_oracle_8k_20260807T172307269147351Z
    TOKEN_ORACLE_SHA=e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2
    BENCHMARK_LENGTH=8192
    BENCHMARK_DEPTH=0.5
    BENCHMARK_MAX_LEN=8704
    BENCHMARK_MAX_BATCHED_TOKENS=2048
    BENCHMARK_NUM_BLOCKS=24
    EXPECTED_BENCHMARK=passkey_L8192_d0.5
    EXPECTED_PROMPT_TOKENS=8155
    EXPECTED_GENERATED_TOKENS=20
    EXPECTED_SEED=1093997
    EXPECTED_GOLD=881446
    # Four 2,048-token prefill chunks precede recurrent decode at 8K.
    FIRST_SOURCE_STEP=5
    FIRST_DECODE_POSITION=8155
    TAG_PREFIX=greenfield_short_context_dsa_oracle_8k
    ;;
  *)
    echo "unsupported short-context DSA oracle profile: $PROFILE" >&2
    exit 2
    ;;
esac
readonly PROFILE TOKEN_ORACLE_TAG TOKEN_ORACLE_SHA BENCHMARK_LENGTH
readonly BENCHMARK_DEPTH BENCHMARK_MAX_LEN BENCHMARK_MAX_BATCHED_TOKENS
readonly BENCHMARK_NUM_BLOCKS EXPECTED_BENCHMARK EXPECTED_PROMPT_TOKENS
readonly EXPECTED_GENERATED_TOKENS EXPECTED_SEED EXPECTED_GOLD
readonly FIRST_SOURCE_STEP FIRST_DECODE_POSITION TAG_PREFIX
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/$PROFILE/$TOKEN_ORACLE_TAG/oracle

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
HARNESS_PIN=$(git -C "$HARNESS_REPO" rev-parse HEAD)
HARNESS_SHORT=$(git -C "$HARNESS_REPO" rev-parse --short HEAD)
LEGACY_SHORT=$(git -C "$LEGACY_SOURCE_REPO" rev-parse --short HEAD)
TAG=${GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG:-${TAG_PREFIX}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
SOURCE_DIR=$RUN_DIR/source_dumps
ORACLE_DIR=$RUN_DIR/oracle
REMOTE_PREFIX=${GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX:-$APPROVED_BUCKET/oracles/greenfield/glm52/short_context_dsa/$PROFILE/$TAG}
DUMP_PREFIX=/tmp/$TAG/topk.npz
INTERNAL_DUMP_PREFIX=/tmp/$TAG/internals.npz
INTERNAL_COMPARISON_DIR=$RUN_DIR/internal_comparison

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing DSA oracle outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing DSA oracle from a dirty greenfield worktree" >&2
  exit 2
}
[[ -z $(git -C "$HARNESS_REPO" status --porcelain --untracked-files=no) ]] || {
  echo "tracked harness files are dirty" >&2
  exit 2
}
[[ $(git -C "$LEGACY_SOURCE_REPO" rev-parse HEAD) == "$LEGACY_PIN" ]] || {
  echo "legacy repository pin changed" >&2
  exit 2
}
[[ -z $(git -C "$LEGACY_SOURCE_REPO" status --porcelain --untracked-files=no) ]] || {
  echo "tracked legacy files are dirty" >&2
  exit 2
}
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  [[ $(git -C "$ORACLE_REPO" rev-parse HEAD) == "$ORACLE_PIN" ]] || {
    echo "accepted legacy oracle pin changed" >&2
    exit 2
  }
  [[ -z $(git -C "$ORACLE_REPO" status --porcelain --untracked-files=no) ]] || {
    echo "accepted legacy oracle worktree is dirty" >&2
    exit 2
  }
  git -C "$LEGACY_SOURCE_REPO" merge-base --is-ancestor \
    "$ORACLE_PIN" "$LEGACY_PIN" || {
      echo "legacy observer does not descend from the accepted oracle" >&2
      exit 2
    }
  [[ $(git -C "$LEGACY_SOURCE_REPO" rev-list --count \
    "$ORACLE_PIN..$LEGACY_PIN") -eq $OBSERVER_COMMIT_DISTANCE ]] || {
      echo "legacy observer commit distance drifted" >&2
      exit 2
    }
fi
[[ -r $RESULTS_DB && -r $TOKEN_ORACLE_DIR/manifest.json ]] || {
  echo "source DB or sealed token oracle is unavailable" >&2
  exit 2
}
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  [[ -r $REFERENCE_8K_DSA_ORACLE/manifest.json &&
     -r $LAYER0_INPUT_DIR/manifest.json &&
     -r $DISTRIBUTED_Q_A_DIR/manifest.json ]] || {
    echo "sealed DSA/internal comparison prerequisites are unavailable" >&2
    exit 2
  }
fi
[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory exists: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR" "$SOURCE_DIR"

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

say() {
  echo "[short-dsa-oracle $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

stop_owned_runtime() {
  # Pre-census proves no pre-existing Ray/vLLM work. Everything matched here
  # was launched under this global lease.
  local command
  # shellcheck disable=SC2016
  command='/home/gianl/vllm-env/bin/ray stop -f >/dev/null 2>&1 || true; sudo pkill -TERM -f "VLLM::[E]ngineCore" >/dev/null 2>&1 || true; sudo pkill -TERM -f "[R]ayWorkerWrapper" >/dev/null 2>&1 || true; sudo pkill -TERM -x raylet >/dev/null 2>&1 || true; sleep 2; sudo pkill -KILL -f "VLLM::[E]ngineCore" >/dev/null 2>&1 || true; sudo pkill -KILL -f "[R]ayWorkerWrapper" >/dev/null 2>&1 || true; sudo pkill -KILL -x raylet >/dev/null 2>&1 || true; sudo rm -f /tmp/libtpu_lockfile; echo STOP_OK $(hostname)'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$RUN_DIR/stop.txt" 2>&1 || return 1
  has_eight_unique_markers "$RUN_DIR/stop.txt" STOP_OK
}

runtime_started=0
post_census_done=0
on_exit() {
  local status=$?
  if [[ $runtime_started -eq 1 ]]; then
    stop_owned_runtime || true
  fi
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR GREENFIELD_PIN=$PIN HARNESS_PIN=$HARNESS_PIN LEGACY_PIN=$LEGACY_PIN"
say "PROFILE=$PROFILE DUMP_PREFIX=$DUMP_PREFIX REMOTE_PREFIX=$REMOTE_PREFIX"
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}
MIN_FREE_GB="$DISK_MIN_FREE_GB" WARN_FREE_GB="$DISK_WARN_FREE_GB" \
  bash "$WORKTREE/scripts/disk_watchdog.sh" check \
  > >(tee "$RUN_DIR/disk_preflight.txt") 2>&1 || {
    say "ABORT: eight-host disk reserve is below ${DISK_MIN_FREE_GB} GiB"
    exit 1
  }

if [[ $INTERNAL_CAPTURE == 1 ]]; then
  # Materialize the two-commit observer in a pin-specific detached worktree;
  # the accepted oracle checkout remains untouched on every host.
  # shellcheck disable=SC2016
  sync_observer='set -e; base='"$ORACLE_REPO"'; dest='"$OBSERVER_RUNTIME_REPO"'; pin='"$LEGACY_PIN"'; oracle='"$ORACLE_PIN"'; branch='"$OBSERVER_BRANCH"'; distance='"$OBSERVER_COMMIT_DISTANCE"'; if git -C "$dest" rev-parse HEAD >/dev/null 2>&1; then :; elif [ -e "$dest" ]; then echo "SYNC_BAD $(hostname) destination_exists"; exit 0; else git -C "$base" fetch origin "$branch" >/dev/null 2>&1 && git -C "$base" worktree add --detach "$dest" "$pin" >/dev/null 2>&1; fi; code=$(git -C "$dest" rev-parse HEAD); dirty=$(git -C "$dest" status --porcelain | wc -l); commits=$(git -C "$dest" rev-list --count "$oracle..$pin"); ancestor=0; git -C "$dest" merge-base --is-ancestor "$oracle" "$pin" && ancestor=1; if [ "$code" = "$pin" ] && [ "$dirty" -eq 0 ] && [ "$commits" -eq "$distance" ] && [ "$ancestor" -eq 1 ]; then echo "SYNC_OK $(hostname)"; else echo "SYNC_BAD $(hostname) code=$code dirty=$dirty commits=$commits ancestor=$ancestor"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$sync_observer" >"$RUN_DIR/sync_observer.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/sync_observer.txt" SYNC_OK || {
    say "ABORT: exact DSA internal observer is unavailable on all hosts"
    exit 1
  }
fi

# All hosts must carry the exact clean legacy tree, golden state file, and
# approved read-only OOB checkpoint mirror used by the PWAL self-healer.
# shellcheck disable=SC2016
prereq='code=$(git -C '"$LEGACY_REPO"' rev-parse HEAD); dirty=$(git -C '"$LEGACY_REPO"' status --porcelain --untracked-files=no | wc -l); mount_source=$(findmnt -T '"$OOB_DIR"' -n -o SOURCE 2>/dev/null); mount_type=$(findmnt -T '"$OOB_DIR"' -n -o FSTYPE 2>/dev/null); if [ "$code" = '"$LEGACY_PIN"' ] && [ "$dirty" -eq 0 ] && [ -r /tmp/golden.json ] && [ -r '"$OOB_DIR"'/model.safetensors.index.json ] && [ "$mount_source" = driftbench-dsv4-uc ] && [ "$mount_type" = fuse.gcsfuse ] && [ ! -e /tmp/'"$TAG"' ]; then echo "PREREQ_OK $(hostname)"; else echo "PREREQ_BAD $(hostname) code=$code dirty=$dirty mount_source=$mount_source mount_type=$mount_type"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$prereq" >"$RUN_DIR/prereq.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/prereq.txt" PREREQ_OK || {
  say "ABORT: exact legacy/golden/OOB/dump prerequisite failed"
  exit 1
}
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  # shellcheck disable=SC2016
  oracle_prereq='code=$(git -C '"$ORACLE_REPO"' rev-parse HEAD); dirty=$(git -C '"$ORACLE_REPO"' status --porcelain --untracked-files=no | wc -l); if [ "$code" = '"$ORACLE_PIN"' ] && [ "$dirty" -eq 0 ]; then echo "ORACLE_OK $(hostname)"; else echo "ORACLE_BAD $(hostname) code=$code dirty=$dirty"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$oracle_prereq" >"$RUN_DIR/oracle_prereq.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/oracle_prereq.txt" ORACLE_OK || {
    say "ABORT: accepted legacy oracle checkout drifted on the fleet"
    exit 1
  }
fi

COMMON_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=1 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_DCP_SCATTER_IMPL=flat GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"' GLM_DSA_DUMP_TOPK_EVENTS=all GLM_DSA_DUMP_TOPK_SKIP_WARMUP=1 GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  COMMON_ENVS="PYTHONPATH=$OBSERVER_RUNTIME_REPO $COMMON_ENVS GLM_DSA_DUMP_INTERNALS=$INTERNAL_DUMP_PREFIX GLM_DSA_DUMP_INTERNALS_LAYER=$INTERNAL_LAYER GLM_DSA_DUMP_INTERNALS_POSITION=$FIRST_DECODE_POSITION GLM_DSA_DUMP_INTERNALS_RUN_TAG=$TAG GLM_DSA_DUMP_INTERNALS_CODE_HASH=$LEGACY_PIN GLM_DSA_DUMP_INTERNALS_ORACLE_PIN=$ORACLE_PIN GLM_DSA_DUMP_INTERNALS_MODEL_ID=$MODEL_ID"
fi
RAYLET_ENVS="$COMMON_ENVS LIBTPU_INIT_ARGS=\"--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false\""
DRIVER_ENVS='NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 RUNAI_STREAMER_CONCURRENCY=32 RUNAI_STREAMER_MEMORY_LIMIT=34359738368 JAX_SHARE_BINARY_BETWEEN_HOSTS=1 JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS=120000 '"$COMMON_ENVS"

say "launching exact protected legacy runtime"
EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
  bash "$HARNESS_REPO/scripts/launch_glm_32chip.sh" \
  >"$RUN_DIR/launch.log" 2>&1
runtime_started=1

# Verify every raylet inherited every source-defining flag.
# shellcheck disable=SC2016
env_check='p=$(pgrep -x raylet | head -1); f=/tmp/dsa_oracle_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "GLM_DCP=1" "$f" && grep -qx "GLM_DCP_SCATTER_IMPL=pageloop" "$f" && grep -qx "GLM_DSA_DCP_SCATTER_IMPL=flat" "$f" && grep -qx "GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"'" "$f" && grep -qx "GLM_DSA_DUMP_TOPK_EVENTS=all" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"'" "$f" && grep -qx "GLM_LOAD_CHECKSUM=1" "$f" && grep -qx "GLM_LOAD_NAN_CHECK=1" "$f" && grep -qx "GLM_PWAL_NAN_CHECK=1" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f" && grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" "$f" && grep -qx "GLM_WK_OOB_GOLDEN=/tmp/golden.json" "$f"; then echo "ENV_OK $(hostname)"; else echo "ENV_BAD $(hostname)"; fi; rm -f "$f"'
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  # shellcheck disable=SC2016
  env_check='p=$(pgrep -x raylet | head -1); f=/tmp/dsa_internal_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "PYTHONPATH='"$OBSERVER_RUNTIME_REPO"'" "$f" && grep -qx "GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS='"$INTERNAL_DUMP_PREFIX"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_LAYER='"$INTERNAL_LAYER"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_POSITION='"$FIRST_DECODE_POSITION"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_RUN_TAG='"$TAG"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_CODE_HASH='"$LEGACY_PIN"'" "$f" && grep -qx "GLM_DSA_DUMP_INTERNALS_ORACLE_PIN='"$ORACLE_PIN"'" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"'" "$f" && grep -qx "GLM_LOAD_CHECKSUM=1" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f"; then echo "ENV_OK $(hostname)"; else echo "ENV_BAD $(hostname)"; fi; rm -f "$f"'
fi
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$env_check" >"$RUN_DIR/raylet_env.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/raylet_env.txt" ENV_OK || {
  say "ABORT: eight-host raylet environment mismatch"
  exit 1
}

say "running one exact $PROFILE raw passkey item"
(
  cd "$HARNESS_REPO/bench"
  set -a
  # shellcheck disable=SC1091
  . "$HARNESS_REPO/.env"
  set +a
  # shellcheck disable=SC2086
  LIBTPU_INIT_ARGS='--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false' \
    env $DRIVER_ENVS setsid --wait /home/gianl/vllm-env/bin/python -u \
    glm_longctx.py --lengths "$BENCHMARK_LENGTH" --depths "$BENCHMARK_DEPTH" --trials 1 \
    --protocol raw --max-new "$EXPECTED_GENERATED_TOKENS" \
    --max-len "$BENCHMARK_MAX_LEN" --max-seqs 1 \
    --max-batched-tokens "$BENCHMARK_MAX_BATCHED_TOKENS" --gmu 0.90 \
    --num-gpu-blocks "$BENCHMARK_NUM_BLOCKS" \
    --seed 12345 --note "fresh flat all-event DSA oracle $TAG" \
    --out-json "$RUN_DIR/legacy_summary.json"
) >"$RUN_DIR/legacy.log" 2>&1
grep -q '\[longctx\].*correct=True' "$RUN_DIR/legacy.log" || {
  say "ABORT: fresh legacy item was not correct"
  exit 1
}

# Ray de-duplicates aggregate driver logs, so validate the load and final state
# from each host's own current-session worker logs. Record any repair messages,
# but do not require a corruption draw: exact final state is the acceptance gate.
# shellcheck disable=SC2016
integrity_check='logs=/tmp/ray/session_latest/logs; checksum=$(grep -Rhs --include="worker-*.out" -E "\[GLM_LOAD_CHECKSUM\].*SUMMARY verified=1882 mismatches=0 skipped=312$" "$logs" 2>/dev/null | tail -1); state=$(grep -Rhs --include="worker-*.out" -E "\[GLM_STATE_HASH\].*manifest VERIFIED leaves=2455 combined=371110325 ref=/tmp/golden.json$" "$logs" 2>/dev/null | tail -1); refusal=$(grep -Rhs --include="worker-*.out" -E "StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|LoadChecksumError|CodeFingerprintMismatchError|GLM_LOAD_CHECKSUM.*diverged|GLM_WK_OOB_GOLDEN.*(no manifest key|does not describe|no checkpoint prefix|still != golden)|GLM_WK_OOB_DIR.*STILL.*all-zero" "$logs" 2>/dev/null | tail -1); repair=$(grep -Rhs --include="worker-*.out" -E "zero-fill repaired at PWAL|manifest-repair at PWAL" "$logs" 2>/dev/null || true); printf "%s\n%s\n" "$checksum" "$state"; [ -z "$repair" ] || printf "%s\n" "$repair"; if [ -n "$checksum" ] && [ -n "$state" ] && [ -z "$refusal" ]; then echo "INTEGRITY_OK $(hostname)"; else [ -z "$refusal" ] || printf "%s\n" "$refusal"; echo "INTEGRITY_BAD $(hostname)"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$integrity_check" >"$RUN_DIR/fleet_integrity.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/fleet_integrity.txt" INTEGRITY_OK || {
  say "ABORT: load/final-state integrity is not exact on all eight hosts"
  exit 1
}
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  # Every JAX process must trace and arm the callback. The decode token row is
  # DCP-sharded, so exactly its owner (process 0 for this sealed one-request
  # workload) observes position 8155 and writes the state artifact. Requiring
  # eight files would require a new full-pod gather in the diagnostic itself.
  # shellcheck disable=SC2016
  internal_integrity='logs=/tmp/ray/session_latest/logs; armed=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DSA_DUMP_INTERNALS] ARMED" "$logs" 2>/dev/null | tail -1); wrote=$(grep -Rhs --include="worker-*.out" --include="worker-*.err" -F "[GLM_DSA_DUMP_INTERNALS] first state file written" "$logs" 2>/dev/null | tail -1); files=$(find /tmp/'"$TAG"' -type f -name "internals.*.position'"$FIRST_DECODE_POSITION"'.proc*.npz" 2>/dev/null | wc -l); owner=$(find /tmp/'"$TAG"' -type f -name "internals.*.position'"$FIRST_DECODE_POSITION"'.proc0.npz" 2>/dev/null | head -1); errors=$(find /tmp/'"$TAG"' -type f -name "*.INTERNAL.ERROR.*" 2>/dev/null | wc -l); printf "%s\n%s\nfiles=%s errors=%s\n" "$armed" "$wrote" "$files" "$errors"; if [ -z "$armed" ] || [ "$errors" -ne 0 ] || [ "$files" -gt 1 ]; then echo "INTERNAL_BAD $(hostname)"; elif [ "$files" -eq 1 ] && [ -n "$wrote" ] && [ -n "$owner" ]; then echo "INTERNAL_OK $(hostname)"; echo "INTERNAL_OWNER $(hostname)"; elif [ "$files" -eq 0 ] && [ -z "$wrote" ]; then echo "INTERNAL_OK $(hostname)"; echo "INTERNAL_NONOWNER $(hostname)"; else echo "INTERNAL_BAD $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$internal_integrity" >"$RUN_DIR/fleet_internal_integrity.txt" 2>&1
  has_eight_unique_markers "$RUN_DIR/fleet_internal_integrity.txt" INTERNAL_OK || {
    say "ABORT: layer-0 DSA internal callback did not arm cleanly fleet-wide"
    exit 1
  }
  [[ $(grep -c '^INTERNAL_OWNER ' "$RUN_DIR/fleet_internal_integrity.txt") -eq 1 && \
      $(grep -c '^INTERNAL_NONOWNER ' "$RUN_DIR/fleet_internal_integrity.txt") -eq 7 ]] || {
    say "ABORT: layer-0 DSA internal owner coverage drifted"
    exit 1
  }
fi

run_id=$(sed -n 's/.*\[longctx\] run_id=\([0-9][0-9]*\).*/\1/p' \
  "$RUN_DIR/legacy.log" | tail -1)
[[ -n $run_id ]] || {
  say "ABORT: legacy run ID is unavailable"
  exit 1
}

say "gathering append-only legacy dump files for DB run $run_id"
for worker in 0 1 2 3 4 5 6 7; do
  if gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
      --recurse "$POD:/tmp/$TAG" "$SOURCE_DIR/w$worker" \
      >/dev/null 2>&1; then
    echo "$worker" >>"$RUN_DIR/dump_hosts.txt"
  fi
done
find "$SOURCE_DIR" -type f -name '*.ERROR*' -print -quit | grep -q . && {
  say "ABORT: armed DSA callback emitted an error sentinel"
  exit 1
}
dump_count=$(find "$SOURCE_DIR" -type f -name 'topk.step*.evt*.proc*.npz' | wc -l)
[[ $dump_count -ge 294 ]] || {
  say "ABORT: incomplete DSA source capture files=$dump_count"
  exit 1
}
internal_count=0
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  internal_count=$(find "$SOURCE_DIR" -type f \
    -name "internals.*.position${FIRST_DECODE_POSITION}.proc*.npz" | wc -l)
  [[ $internal_count -eq 1 ]] || {
    say "ABORT: expected one DCP-owner DSA internal file, found $internal_count"
    exit 1
  }
fi

/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$run_id" \
  "$EXPECTED_PROMPT_TOKENS" "$EXPECTED_GENERATED_TOKENS" \
  >"$RUN_DIR/source_identity.json" <<'PY'
import json
import sqlite3
import sys

connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
run_id = int(sys.argv[2])
row = connection.execute(
    "SELECT r.harness_git,r.fork_git,i.id AS item_row_id,i.correct,"
    "i.n_prompt_tokens,i.n_gen_tokens,i.raw_output FROM runs r JOIN items i "
    "ON i.run_id=r.run_id WHERE r.run_id=? ORDER BY i.id", (run_id,)
).fetchall()
assert len(row) == 1, row
value = dict(row[0])
assert value["correct"] == 1
assert value["n_prompt_tokens"] == int(sys.argv[3])
assert value["n_gen_tokens"] == int(sys.argv[4])
print(json.dumps(value, indent=2, sort_keys=True))
PY
item_row_id=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["item_row_id"])' \
  "$RUN_DIR/source_identity.json")
source_harness=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["harness_git"])' \
  "$RUN_DIR/source_identity.json")
source_fork=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["fork_git"])' \
  "$RUN_DIR/source_identity.json")
[[ $source_harness == "$HARNESS_SHORT" && $source_fork == "$LEGACY_SHORT" ]] || {
  say "ABORT: DB code provenance drifted"
  exit 1
}

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/capture_short_context_dsa_oracle.py" \
  --results-db "$RESULTS_DB" \
  --token-oracle-dir "$TOKEN_ORACLE_DIR" \
  --source-dump-dir "$SOURCE_DIR" \
  --output "$ORACLE_DIR" \
  --expected-code-hash "$PIN" \
  --legacy-repository-pin "$LEGACY_PIN" \
  --token-oracle-manifest-sha256 "$TOKEN_ORACLE_SHA" \
  --run-id "$run_id" \
  --item-row-id "$item_row_id" \
  --expected-harness-git "$source_harness" \
  --expected-fork-git "$source_fork" \
  --expected-benchmark "$EXPECTED_BENCHMARK" \
  --expected-model-uri gs://driftbench-dsv4-uc/models/GLM-5.2-FP8 \
  --expected-prompt-tokens "$EXPECTED_PROMPT_TOKENS" \
  --expected-generated-tokens "$EXPECTED_GENERATED_TOKENS" \
  --expected-seed "$EXPECTED_SEED" \
  --expected-gold "$EXPECTED_GOLD" \
  --expected-oob-dir "$OOB_DIR" \
  --expected-dump-prefix "$DUMP_PREFIX" \
  --expected-process-count 8 \
  --first-source-step "$FIRST_SOURCE_STEP" \
  --decode-step-count 14 \
  --first-decode-position "$FIRST_DECODE_POSITION" \
  --selected-width 2048 >"$RUN_DIR/capture.json"

if [[ $INTERNAL_CAPTURE == 1 ]]; then
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$REFERENCE_8K_DSA_ORACLE" "$ORACLE_DIR" \
    >"$RUN_DIR/dsa_exact_comparison.json" <<'PY'
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.validation import compare_short_context_dsa_oracles

value = compare_short_context_dsa_oracles(Path(sys.argv[1]), Path(sys.argv[2]))
print(json.dumps(value, indent=2, sort_keys=True))
PY
fi

# Snapshot the append-only provenance DB at the exact source row.
/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$RUN_DIR/results_ckpt.db" <<'PY'
import sqlite3
import sys

source = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
destination = sqlite3.connect(sys.argv[2])
source.backup(destination)
assert destination.execute("pragma integrity_check").fetchone()[0] == "ok"
destination.close()
source.close()
PY

stop_owned_runtime
runtime_started=0
if [[ $INTERNAL_CAPTURE == 1 ]]; then
  say "comparing accepted layer-0 scorer state on one local TPU host"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/compare_legacy_layer0_dsa_internals.py" \
    --source-dump-dir "$SOURCE_DIR" \
    --layer0-input-dir "$LAYER0_INPUT_DIR" \
    --distributed-q-a-norm-dir "$DISTRIBUTED_Q_A_DIR" \
    --output "$INTERNAL_COMPARISON_DIR" \
    --run-tag "$TAG" \
    --greenfield-code-hash "$PIN" \
    --legacy-code-hash "$LEGACY_PIN" \
    --oracle-pin "$ORACLE_PIN" \
    --input-manifest-sha256 "$LAYER0_INPUT_MANIFEST_SHA" \
    --q-a-manifest-sha256 "$DISTRIBUTED_Q_A_MANIFEST_SHA" \
    --q-a-code-hash "$DISTRIBUTED_Q_A_CODE_HASH" \
    --model-id "$MODEL_ID" \
    --layer-name "$INTERNAL_LAYER" \
    --position "$FIRST_DECODE_POSITION" \
    --process-count 8 \
    --capture-process-indices 0 >"$RUN_DIR/internal_comparison_summary.json"
fi
strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "freezing fresh DSA-oracle evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
records = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    if (
        not path.is_file()
        or relative == "orchestrator.log"
        or path.name in {"SUCCESS", "evidence_sha256.json"}
    ):
        continue
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    records.append({
        "byte_count": path.stat().st_size,
        "path": relative,
        "sha256": digest.hexdigest(),
    })
(root / "evidence_sha256.json").write_text(
    json.dumps({"files": records}, indent=2, sort_keys=True) + "\n"
)
PY

say "uploading fresh DSA source and compact oracle append-only"
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY' \
  >"$RUN_DIR/remote_objects.json"
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    if (
        not path.is_file()
        or relative == "orchestrator.log"
        or path.name in {"SUCCESS", "remote_objects.json"}
    ):
        continue
    paths.append((path, relative))

def describe(item):
    path, relative = item
    remote = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    if int(remote["size"]) != path.stat().st_size:
        raise SystemExit(f"remote size mismatch: {relative}")
    crc32c = remote.get("crc32c_hash") or remote.get("crc32c")
    if not crc32c:
        raise SystemExit(f"remote CRC32C missing: {relative}")
    return {
        "crc32c": crc32c,
        "generation": remote["generation"],
        "path": relative,
        "size": int(remote["size"]),
    }

with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" \
  "$LEGACY_PIN" "$run_id" "$item_row_id" "$dump_count" \
  "$INTERNAL_CAPTURE" "$internal_count" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
remote = sys.argv[2]
manifest = json.loads((root / "oracle" / "manifest.json").read_text())
lines = {
    "artifact_kind": manifest["artifact_kind"],
    "code_hash": sys.argv[3],
    "legacy_repository_pin": sys.argv[4],
    "manifest_sha256": manifest["manifest_sha256"],
    "source_run_id": sys.argv[5],
    "source_item_row_id": sys.argv[6],
    "source_dump_file_count": sys.argv[7],
    "evidence_sha256": sha256((root / "evidence_sha256.json").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256((root / "remote_objects.json").read_bytes()).hexdigest(),
    "remote_prefix": remote,
}
if sys.argv[8] == "1":
    comparison = json.loads(
        (root / "internal_comparison" / "comparison.json").read_text()
    )
    exact_dsa = json.loads((root / "dsa_exact_comparison.json").read_text())
    if comparison["capture_layout"] != "topology_sharded_live_row_owner":
        raise SystemExit("DSA internal capture layout drifted")
    if comparison["capture_process_indices"] != [0]:
        raise SystemExit("DSA internal owner process drifted")
    lines.update({
        "dsa_internal_capture": "true",
        "dsa_internal_capture_layout": comparison["capture_layout"],
        "dsa_internal_file_count": sys.argv[9],
        "dsa_internal_owner_actual_sha256": comparison[
            "owner_actual_sha256"
        ],
        "dsa_internal_first_divergent_field": comparison[
            "first_divergent_field"
        ] or "none",
        "dsa_event_tensors_exact": str(exact_dsa["exact"]).lower(),
        "accepted_oracle_pin": comparison["oracle_pin"],
    })
(root / "SUCCESS").write_text(
    "".join(f"{key}={value}\n" for key, value in lines.items())
)
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
local_success_sha=$(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}')
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]] || {
  say "ABORT: remote SUCCESS checksum mismatch"
  exit 1
}

manifest_sha=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$ORACLE_DIR/manifest.json")
say "SUCCESS run=$run_id item=$item_row_id files=$dump_count manifest=$manifest_sha"
