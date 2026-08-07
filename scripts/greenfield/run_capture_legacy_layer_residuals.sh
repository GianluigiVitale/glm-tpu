#!/usr/bin/env bash
# Protected, callback-free legacy layer-boundary capture for position 2044.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly HARNESS_REPO=/home/gianl/glm-tpu
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly OBSERVER_DEV_REPO=/home/gianl/tpu-inference-greenfield-residual-observer
readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-residual-cf066ab3e
readonly OBSERVER_BRANCH=greenfield/legacy-residual-observer
readonly OBSERVER_PIN=cf066ab3e29151153d3f930a4c5ecc20ba716a9f
readonly ORACLE_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly MODEL_ID=gs://driftbench-dsv4-uc/models/GLM-5.2-FP8
readonly TOKEN_ORACLE_TAG=greenfield_short_context_oracle_20260806T202544155912103Z
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/2k/$TOKEN_ORACLE_TAG/oracle
readonly TOKEN_ORACLE_SHA=f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19
readonly OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly GREENFIELD_CAPTURE_TAG=greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_downf32_token_oracle_dsa_residual_p2044_trace2_20260807T063203702756771Z
readonly GREENFIELD_CAPTURE_DIR=/home/gianl/glm-run/$GREENFIELD_CAPTURE_TAG/layer_residual_observer
readonly GREENFIELD_NPZ=$GREENFIELD_CAPTURE_DIR/position_2044_boundaries.npz
readonly GREENFIELD_CONTRACT=$GREENFIELD_CAPTURE_DIR/contract.json

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
PIN_SHORT=$(git -C "$WORKTREE" rev-parse --short HEAD)
OBSERVER_SHORT=$(git -C "$OBSERVER_DEV_REPO" rev-parse --short HEAD)
TAG=${GLM_GREENFIELD_LEGACY_RESIDUAL_TAG:-greenfield_legacy_layer_residual_p2044_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
SOURCE_DIR=$RUN_DIR/source_dumps
ORACLE_DIR=$RUN_DIR/logprob_oracle
COMPARISON_DIR=$RUN_DIR/residual_comparison
DUMP_PREFIX=/tmp/$TAG/boundaries.npz
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/layer_residuals/2k/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing residual capture outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing residual capture from a dirty greenfield worktree" >&2
  exit 2
}
[[ $(git -C "$OBSERVER_DEV_REPO" rev-parse HEAD) == "$OBSERVER_PIN" ]] || {
  echo "legacy observer development pin changed" >&2
  exit 2
}
[[ -z $(git -C "$OBSERVER_DEV_REPO" status --porcelain) ]] || {
  echo "legacy observer development worktree is dirty" >&2
  exit 2
}
[[ $(git -C "$OBSERVER_DEV_REPO" rev-parse HEAD^) == "$ORACLE_PIN" ]] || {
  echo "legacy observer is not a one-commit child of the sealed oracle" >&2
  exit 2
}
[[ $(git -C "$ORACLE_REPO" rev-parse HEAD) == "$ORACLE_PIN" ]] || {
  echo "accepted legacy oracle pin changed" >&2
  exit 2
}
[[ -z $(git -C "$HARNESS_REPO" status --porcelain --untracked-files=no) ]] || {
  echo "tracked launcher harness files are dirty" >&2
  exit 2
}
[[ -r $RESULTS_DB && -r $TOKEN_ORACLE_DIR/manifest.json &&
   -r $GREENFIELD_NPZ && -r $GREENFIELD_CONTRACT ]] || {
  echo "source DB or sealed correctness artifact is unavailable" >&2
  exit 2
}
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
  echo "[legacy-residual $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]apture_legacy_short_context_logprobs[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

stop_owned_runtime() {
  local command
  # Pre-census proves every matching process was launched under this lease.
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

say "RUN_DIR=$RUN_DIR GREENFIELD_PIN=$PIN OBSERVER_PIN=$OBSERVER_PIN ORACLE_PIN=$ORACLE_PIN"
say "DUMP_PREFIX=$DUMP_PREFIX REMOTE_PREFIX=$REMOTE_PREFIX"
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}
MIN_FREE_GB=10 WARN_FREE_GB=15 bash "$WORKTREE/scripts/disk_watchdog.sh" check \
  > >(tee "$RUN_DIR/disk_preflight.txt") 2>&1 || {
    say "ABORT: eight-host disk reserve is below 10 GiB"
    exit 1
  }

# Materialize a pin-specific detached worktree on every host.  The accepted
# oracle checkout remains untouched and continues to identify the semantic
# parent.  Existing non-exact destinations fail closed rather than being reset.
# shellcheck disable=SC2016
sync_observer='set -e; base='"$ORACLE_REPO"'; dest='"$OBSERVER_RUNTIME_REPO"'; pin='"$OBSERVER_PIN"'; branch='"$OBSERVER_BRANCH"'; if git -C "$dest" rev-parse HEAD >/dev/null 2>&1; then :; elif [ -e "$dest" ]; then echo "SYNC_BAD $(hostname) destination_exists"; exit 0; else git -C "$base" fetch origin "$branch" >/dev/null 2>&1 && git -C "$base" worktree add --detach "$dest" "$pin" >/dev/null 2>&1; fi; code=$(git -C "$dest" rev-parse HEAD); parent=$(git -C "$dest" rev-parse HEAD^); dirty=$(git -C "$dest" status --porcelain | wc -l); if [ "$code" = "$pin" ] && [ "$parent" = '"$ORACLE_PIN"' ] && [ "$dirty" -eq 0 ]; then echo "SYNC_OK $(hostname)"; else echo "SYNC_BAD $(hostname) code=$code parent=$parent dirty=$dirty"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_observer" >"$RUN_DIR/sync_observer.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync_observer.txt" SYNC_OK || {
  say "ABORT: exact observer worktree is not available on all hosts"
  exit 1
}

# shellcheck disable=SC2016
prereq='base='"$ORACLE_REPO"'; obs='"$OBSERVER_RUNTIME_REPO"'; base_code=$(git -C "$base" rev-parse HEAD); base_dirty=$(git -C "$base" status --porcelain --untracked-files=no | wc -l); obs_code=$(git -C "$obs" rev-parse HEAD); obs_dirty=$(git -C "$obs" status --porcelain | wc -l); mount_source=$(findmnt -T '"$OOB_DIR"' -n -o SOURCE 2>/dev/null); mount_type=$(findmnt -T '"$OOB_DIR"' -n -o FSTYPE 2>/dev/null); if [ "$base_code" = '"$ORACLE_PIN"' ] && [ "$base_dirty" -eq 0 ] && [ "$obs_code" = '"$OBSERVER_PIN"' ] && [ "$obs_dirty" -eq 0 ] && [ -r /tmp/golden.json ] && [ -r '"$OOB_DIR"'/model.safetensors.index.json ] && [ "$mount_source" = driftbench-dsv4-uc ] && [ "$mount_type" = fuse.gcsfuse ] && [ ! -e /tmp/'"$TAG"' ]; then echo "PREREQ_OK $(hostname)"; else echo "PREREQ_BAD $(hostname) base=$base_code/$base_dirty obs=$obs_code/$obs_dirty mount=$mount_source/$mount_type"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$prereq" >"$RUN_DIR/prereq.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/prereq.txt" PREREQ_OK || {
  say "ABORT: exact oracle/observer/golden/OOB prerequisite failed"
  exit 1
}

COMMON_ENVS="PYTHONPATH=$OBSERVER_RUNTIME_REPO GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=1 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_DCP_SCATTER_IMPL=flat GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR=$OOB_DIR GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH=$OBSERVER_SHORT GLM_GREENFIELD_LEGACY_RESIDUAL_OBSERVER=1 GLM_GREENFIELD_LEGACY_RESIDUAL_PATH=$DUMP_PREFIX GLM_GREENFIELD_LEGACY_RESIDUAL_POSITION=2044 GLM_GREENFIELD_LEGACY_RESIDUAL_DECODE_ROWS=32 GLM_GREENFIELD_LEGACY_RESIDUAL_RUN_TAG=$TAG GLM_GREENFIELD_LEGACY_RESIDUAL_CODE_HASH=$OBSERVER_PIN GLM_GREENFIELD_LEGACY_RESIDUAL_ORACLE_PIN=$ORACLE_PIN GLM_GREENFIELD_LEGACY_RESIDUAL_MODEL_ID=$MODEL_ID"
RAYLET_ENVS="$COMMON_ENVS LIBTPU_INIT_ARGS=\"--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false\""
DRIVER_ENVS="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 RUNAI_STREAMER_CONCURRENCY=32 RUNAI_STREAMER_MEMORY_LIMIT=34359738368 JAX_SHARE_BINARY_BETWEEN_HOSTS=1 JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS=120000 $COMMON_ENVS"

say "launching exact observer-only legacy runtime"
EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
  bash "$HARNESS_REPO/scripts/launch_glm_32chip.sh" \
  >"$RUN_DIR/launch.log" 2>&1
runtime_started=1

# shellcheck disable=SC2016
env_check='p=$(pgrep -x raylet | head -1); f=/tmp/legacy_residual_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "PYTHONPATH='"$OBSERVER_RUNTIME_REPO"'" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$OBSERVER_SHORT"'" "$f" && grep -qx "GLM_GREENFIELD_LEGACY_RESIDUAL_OBSERVER=1" "$f" && grep -qx "GLM_GREENFIELD_LEGACY_RESIDUAL_PATH='"$DUMP_PREFIX"'" "$f" && grep -qx "GLM_GREENFIELD_LEGACY_RESIDUAL_POSITION=2044" "$f" && grep -qx "GLM_GREENFIELD_LEGACY_RESIDUAL_CODE_HASH='"$OBSERVER_PIN"'" "$f" && grep -qx "GLM_GREENFIELD_LEGACY_RESIDUAL_ORACLE_PIN='"$ORACLE_PIN"'" "$f" && grep -qx "GLM_LOAD_CHECKSUM=1" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f"; then echo "ENV_OK $(hostname)"; else echo "ENV_BAD $(hostname)"; fi; rm -f "$f"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$env_check" >"$RUN_DIR/raylet_env.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/raylet_env.txt" ENV_OK || {
  say "ABORT: eight-host observer environment mismatch"
  exit 1
}

say "capturing sealed legacy tokens, top-16 logits, and position-2044 boundaries"
(
  cd "$WORKTREE"
  set -a
  # shellcheck disable=SC1091
  . "$HARNESS_REPO/.env"
  set +a
  # shellcheck disable=SC2086
  LIBTPU_INIT_ARGS='--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false' \
    env $DRIVER_ENVS setsid --wait /home/gianl/vllm-env/bin/python -u \
    scripts/greenfield/capture_legacy_short_context_logprobs.py \
    --token-oracle-dir "$TOKEN_ORACLE_DIR" \
    --token-oracle-manifest-sha256 "$TOKEN_ORACLE_SHA" \
    --results-db "$RESULTS_DB" \
    --output "$ORACLE_DIR" \
    --result-json "$RUN_DIR/runner.json" \
    --expected-code-hash "$PIN" \
    --legacy-repository "$OBSERVER_RUNTIME_REPO" \
    --expected-legacy-code-hash "$OBSERVER_PIN" \
    --expected-oracle-base-hash "$ORACLE_PIN" \
    --top-k 16 --step-count 15 --focus-position 2044 \
    --focus-token-id 16345 --focus-token-id 12877
) >"$RUN_DIR/legacy.log" 2>&1

# Validate every host's exact loaded/final state and observer file before use.
# shellcheck disable=SC2016
integrity_check='logs=/tmp/ray/session_latest/logs; checksum=$(grep -Rhs --include="worker-*.out" -E "\[GLM_LOAD_CHECKSUM\].*SUMMARY verified=1882 mismatches=0 skipped=312$" "$logs" 2>/dev/null | tail -1); state=$(grep -Rhs --include="worker-*.out" -E "\[GLM_STATE_HASH\].*manifest VERIFIED leaves=2455 combined=371110325 ref=/tmp/golden.json$" "$logs" 2>/dev/null | tail -1); armed=$(grep -Rhs --include="worker-*.out" -F "[GLM_GREENFIELD_LEGACY_RESIDUAL_OBSERVER] ARMED" "$logs" 2>/dev/null | tail -1); wrote=$(grep -Rhs --include="worker-*.out" -F "[GLM_GREENFIELD_LEGACY_RESIDUAL_OBSERVER] wrote" "$logs" 2>/dev/null | tail -1); files=$(find /tmp/'"$TAG"' -type f -name "boundaries.position2044.proc*.npz" 2>/dev/null | wc -l); refusal=$(grep -Rhs --include="worker-*.out" -E "StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|LoadChecksumError|CodeFingerprintMismatchError|legacy residual.*(drifted|requires|non-finite|already dumped)" "$logs" 2>/dev/null | tail -1); printf "%s\n%s\n%s\n%s\nfiles=%s\n" "$checksum" "$state" "$armed" "$wrote" "$files"; if [ -n "$checksum" ] && [ -n "$state" ] && [ -n "$armed" ] && [ -n "$wrote" ] && [ "$files" -eq 1 ] && [ -z "$refusal" ]; then echo "INTEGRITY_OK $(hostname)"; else [ -z "$refusal" ] || printf "%s\n" "$refusal"; echo "INTEGRITY_BAD $(hostname)"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$integrity_check" >"$RUN_DIR/fleet_integrity.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/fleet_integrity.txt" INTEGRITY_OK || {
  say "ABORT: load/state/observer integrity is not exact on all eight hosts"
  exit 1
}

say "gathering eight append-only process-shard files"
for worker in 0 1 2 3 4 5 6 7; do
  if gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
      --recurse "$POD:/tmp/$TAG" "$SOURCE_DIR/w$worker" \
      >/dev/null 2>&1; then
    echo "$worker" >>"$RUN_DIR/dump_hosts.txt"
  fi
done
[[ $(sort -u "$RUN_DIR/dump_hosts.txt" | wc -l) -eq 8 ]] || {
  say "ABORT: residual dumps were not retrieved from all hosts"
  exit 1
}
dump_count=$(find "$SOURCE_DIR" -type f -name 'boundaries.position2044.proc*.npz' | wc -l)
[[ $dump_count -eq 8 ]] || {
  say "ABORT: expected eight residual dump files, found $dump_count"
  exit 1
}

run_id=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' \
  "$RUN_DIR/runner.json")
item_row_id=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["item_row_id"])' \
  "$RUN_DIR/runner.json")

/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$run_id" "$item_row_id" \
  "$PIN_SHORT" "$OBSERVER_SHORT" <<'PY' >"$RUN_DIR/source_identity.json"
import json
import sqlite3
import sys

connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
rows = connection.execute(
    "SELECT r.harness_git,r.fork_git,i.id AS item_row_id,i.correct,"
    "i.n_prompt_tokens,i.n_gen_tokens FROM runs r JOIN items i "
    "ON i.run_id=r.run_id WHERE r.run_id=? AND i.id=?",
    (int(sys.argv[2]), int(sys.argv[3])),
).fetchall()
assert len(rows) == 1
value = dict(rows[0])
assert value == {
    "harness_git": sys.argv[4],
    "fork_git": sys.argv[5],
    "item_row_id": int(sys.argv[3]),
    "correct": 1,
    "n_prompt_tokens": 2034,
    "n_gen_tokens": 15,
}, value
print(json.dumps(value, indent=2, sort_keys=True))
PY

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
strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "reconstructing the target row and locating the first divergent boundary"
/home/gianl/vllm-env/bin/python \
  scripts/greenfield/compare_legacy_layer_residuals.py \
  --source-dump-dir "$SOURCE_DIR" \
  --greenfield-npz "$GREENFIELD_NPZ" \
  --greenfield-contract "$GREENFIELD_CONTRACT" \
  --output "$COMPARISON_DIR" \
  --position 2044 --run-tag "$TAG" \
  --legacy-code-hash "$OBSERVER_PIN" --oracle-pin "$ORACLE_PIN" \
  --model-id "$MODEL_ID" --process-count 8 \
  >"$RUN_DIR/comparison_summary.json"

/home/gianl/vllm-env/bin/python - "$RUN_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
records = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name in {"SUCCESS", "evidence_sha256.json"}:
        continue
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    records.append({
        "byte_count": path.stat().st_size,
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
    })
(root / "evidence_sha256.json").write_text(
    json.dumps({"files": records}, indent=2, sort_keys=True) + "\n"
)
PY

say "uploading compact residual evidence append-only"
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY' \
  >"$RUN_DIR/remote_objects.json"
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
records = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name in {"SUCCESS", "remote_objects.json"}:
        continue
    relative = path.relative_to(root).as_posix()
    remote = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}",
         "--format=json"], check=True, capture_output=True, text=True,
    ).stdout)
    if int(remote["size"]) != path.stat().st_size:
        raise SystemExit(f"remote size mismatch: {relative}")
    crc32c = remote.get("crc32c_hash") or remote.get("crc32c")
    if not crc32c:
        raise SystemExit(f"remote CRC32C missing: {relative}")
    records.append({
        "crc32c": crc32c,
        "generation": remote["generation"],
        "path": relative,
        "size": int(remote["size"]),
    })
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" \
  "$OBSERVER_PIN" "$ORACLE_PIN" "$run_id" "$item_row_id" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
comparison = json.loads((root / "residual_comparison" / "comparison.json").read_text())
lines = {
    "artifact_kind": comparison["artifact_kind"],
    "code_hash": sys.argv[3],
    "legacy_observer_pin": sys.argv[4],
    "legacy_oracle_pin": sys.argv[5],
    "source_run_id": sys.argv[6],
    "source_item_row_id": sys.argv[7],
    "first_divergent_boundary": comparison["first_divergent_boundary"],
    "legacy_canonical_sha256": comparison["legacy"]["canonical_sha256"],
    "greenfield_canonical_sha256": comparison["greenfield"]["canonical_sha256"],
    "evidence_sha256": sha256((root / "evidence_sha256.json").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256((root / "remote_objects.json").read_bytes()).hexdigest(),
    "remote_prefix": sys.argv[2],
}
(root / "SUCCESS").write_text(
    "".join(f"{key}={value}\n" for key, value in lines.items())
)
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null
local_success_sha=$(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}')
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]] || {
  say "ABORT: remote SUCCESS checksum mismatch"
  exit 1
}

first_boundary=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["first_divergent_boundary"])' \
  "$COMPARISON_DIR/comparison.json")
say "SUCCESS run=$run_id item=$item_row_id first_divergent_boundary=$first_boundary"
