#!/usr/bin/env bash
# Capture and seal one fresh, flat-scatter 2K legacy DSA event oracle.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly HARNESS_REPO=/home/gianl/glm-tpu
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly TOKEN_ORACLE_TAG=greenfield_short_context_oracle_20260806T202544155912103Z
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/2k/$TOKEN_ORACLE_TAG/oracle
readonly TOKEN_ORACLE_SHA=f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19
readonly DISK_MIN_FREE_GB=10
readonly DISK_WARN_FREE_GB=15

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
HARNESS_PIN=$(git -C "$HARNESS_REPO" rev-parse HEAD)
HARNESS_SHORT=$(git -C "$HARNESS_REPO" rev-parse --short HEAD)
LEGACY_SHORT=$(git -C "$LEGACY_REPO" rev-parse --short HEAD)
TAG=${GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG:-greenfield_short_context_dsa_oracle_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
SOURCE_DIR=$RUN_DIR/source_dumps
ORACLE_DIR=$RUN_DIR/oracle
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/short_context_dsa/2k/$TAG
DUMP_PREFIX=/tmp/$TAG/topk.npz

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
[[ $(git -C "$LEGACY_REPO" rev-parse HEAD) == "$LEGACY_PIN" ]] || {
  echo "legacy repository pin changed" >&2
  exit 2
}
[[ -z $(git -C "$LEGACY_REPO" status --porcelain --untracked-files=no) ]] || {
  echo "tracked legacy files are dirty" >&2
  exit 2
}
[[ -r $RESULTS_DB && -r $TOKEN_ORACLE_DIR/manifest.json ]] || {
  echo "source DB or sealed token oracle is unavailable" >&2
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
say "DUMP_PREFIX=$DUMP_PREFIX REMOTE_PREFIX=$REMOTE_PREFIX"
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

# All hosts must carry the exact clean legacy tree and the golden state file.
# shellcheck disable=SC2016
prereq='code=$(git -C /home/gianl/tpu-inference rev-parse HEAD); dirty=$(git -C /home/gianl/tpu-inference status --porcelain --untracked-files=no | wc -l); if [ "$code" = '"$LEGACY_PIN"' ] && [ "$dirty" -eq 0 ] && [ -r /tmp/golden.json ] && [ ! -e /tmp/'"$TAG"' ]; then echo "PREREQ_OK $(hostname)"; else echo "PREREQ_BAD $(hostname) code=$code dirty=$dirty"; fi'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$prereq" >"$RUN_DIR/prereq.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/prereq.txt" PREREQ_OK || {
  say "ABORT: exact legacy/golden/dump prerequisite failed"
  exit 1
}

COMMON_ENVS='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=1 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_DCP_SCATTER_IMPL=flat GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"' GLM_DSA_DUMP_TOPK_EVENTS=all GLM_DSA_DUMP_TOPK_SKIP_WARMUP=1 GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"
RAYLET_ENVS="$COMMON_ENVS LIBTPU_INIT_ARGS=\"--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false\""
DRIVER_ENVS='NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 RUNAI_STREAMER_CONCURRENCY=32 RUNAI_STREAMER_MEMORY_LIMIT=34359738368 JAX_SHARE_BINARY_BETWEEN_HOSTS=1 JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS=120000 '"$COMMON_ENVS"

say "launching exact protected legacy runtime"
EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
  bash "$HARNESS_REPO/scripts/launch_glm_32chip.sh" \
  >"$RUN_DIR/launch.log" 2>&1
runtime_started=1

# Verify every raylet inherited every source-defining flag.
# shellcheck disable=SC2016
env_check='p=$(pgrep -x raylet | head -1); f=/tmp/dsa_oracle_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "GLM_DCP=1" "$f" && grep -qx "GLM_DCP_SCATTER_IMPL=pageloop" "$f" && grep -qx "GLM_DSA_DCP_SCATTER_IMPL=flat" "$f" && grep -qx "GLM_DSA_DUMP_TOPK='"$DUMP_PREFIX"'" "$f" && grep -qx "GLM_DSA_DUMP_TOPK_EVENTS=all" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$LEGACY_SHORT"'" "$f" && grep -qx "GLM_LOAD_CHECKSUM=1" "$f" && grep -qx "GLM_LOAD_NAN_CHECK=1" "$f" && grep -qx "GLM_PWAL_NAN_CHECK=1" "$f" && grep -qx "GLM_STATE_HASH_REF=/tmp/golden.json" "$f"; then echo "ENV_OK $(hostname)"; else echo "ENV_BAD $(hostname)"; fi; rm -f "$f"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$env_check" >"$RUN_DIR/raylet_env.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/raylet_env.txt" ENV_OK || {
  say "ABORT: eight-host raylet environment mismatch"
  exit 1
}

say "running one exact 2K raw passkey item"
(
  cd "$HARNESS_REPO/bench"
  set -a
  # shellcheck disable=SC1091
  . "$HARNESS_REPO/.env"
  set +a
  # shellcheck disable=SC2086
  LIBTPU_INIT_ARGS='--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false' \
    env $DRIVER_ENVS setsid --wait /home/gianl/vllm-env/bin/python -u \
    glm_longctx.py --lengths 2040 --depths 0.25 --trials 1 \
    --protocol raw --max-new 20 --max-len 2560 --max-seqs 1 \
    --max-batched-tokens 2048 --gmu 0.90 --num-gpu-blocks 8 \
    --seed 12345 --note "fresh flat all-event DSA oracle $TAG" \
    --out-json "$RUN_DIR/legacy_summary.json"
) >"$RUN_DIR/legacy.log" 2>&1
grep -q '\[longctx\].*correct=True' "$RUN_DIR/legacy.log" || {
  say "ABORT: fresh legacy item was not correct"
  exit 1
}
grep -q 'GLM_LOAD_CHECKSUM' "$RUN_DIR/legacy.log" || {
  say "ABORT: legacy load checksum evidence is absent"
  exit 1
}

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

/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$run_id" \
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
assert value["n_prompt_tokens"] == 2034
assert value["n_gen_tokens"] == 20
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
  --expected-dump-prefix "$DUMP_PREFIX" >"$RUN_DIR/capture.json"

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
strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

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

say "uploading fresh DSA source and compact oracle append-only"
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

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
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"],
        check=True, capture_output=True, text=True,
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
  "$LEGACY_PIN" "$run_id" "$item_row_id" "$dump_count" <<'PY'
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
