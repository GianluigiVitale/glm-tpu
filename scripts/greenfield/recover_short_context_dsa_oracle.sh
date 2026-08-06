#!/usr/bin/env bash
# Seal the immutable run-480 DSA source after the pad-row sealer correction.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_TAG=greenfield_short_context_dsa_oracle_20260806T221300Z
readonly SOURCE_RUN_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_CAPTURE_PIN=f5e047a63c5bdaff1da96f7af4134948543942ed
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly TOKEN_ORACLE_TAG=greenfield_short_context_oracle_20260806T202544155912103Z
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/2k/$TOKEN_ORACLE_TAG/oracle
readonly TOKEN_ORACLE_SHA=f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19
readonly DUMP_PREFIX=/tmp/$SOURCE_TAG/topk.npz
readonly RUN_ID=480
readonly ITEM_ROW_ID=1763
readonly SOURCE_HARNESS=a4a17ac
readonly SOURCE_FORK=b3c25df47
readonly OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_SHORT_DSA_RECOVERY_TAG:-greenfield_short_context_dsa_oracle_recovery_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
SOURCE_DIR=$RUN_DIR/source_dumps
ORACLE_DIR=$RUN_DIR/oracle
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/short_context_dsa/2k/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing DSA recovery outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing DSA recovery from a dirty worktree" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only recovery directory exists: $RUN_DIR" >&2
  exit 2
}
[[ -r $RESULTS_DB && -r $TOKEN_ORACLE_DIR/manifest.json ]] || {
  echo "source DB or token oracle is unavailable" >&2
  exit 2
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

mkdir -p "$RUN_DIR" "$SOURCE_DIR"
say() {
  echo "[short-dsa-recovery $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

on_exit() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving recovery diagnostics"
  fi
}
trap on_exit EXIT

say "SOURCE=$SOURCE_RUN_DIR SOURCE_CAPTURE_PIN=$SOURCE_CAPTURE_PIN SEALER_PIN=$PIN"
say "RUN_DIR=$RUN_DIR REMOTE_PREFIX=$REMOTE_PREFIX"
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}

has_eight_unique_markers "$SOURCE_RUN_DIR/fleet_integrity.txt" INTEGRITY_OK || {
  say "ABORT: source capture lacks eight exact fleet-integrity proofs"
  exit 1
}
has_eight_unique_markers "$SOURCE_RUN_DIR/census_failure_exit.txt" CENSUS_OK || {
  say "ABORT: source capture lacks eight-host cleanup evidence"
  exit 1
}
grep -q "GREENFIELD_PIN=$SOURCE_CAPTURE_PIN" "$SOURCE_RUN_DIR/orchestrator.log" || {
  say "ABORT: source capture greenfield pin drifted"
  exit 1
}
grep -q '\[longctx\].*correct=True' "$SOURCE_RUN_DIR/legacy.log" || {
  say "ABORT: source capture raw item was not correct"
  exit 1
}
[[ $(find "$SOURCE_RUN_DIR/source_dumps" -type f -name '*.npz' | wc -l) -eq 420 ]] || {
  say "ABORT: immutable source dump inventory drifted"
  exit 1
}

say "hard-linking immutable source evidence without duplicating bytes"
cp -al "$SOURCE_RUN_DIR/source_dumps/." "$SOURCE_DIR/"
mkdir -p "$RUN_DIR/source_capture"
for name in census_pre.txt census_failure_exit.txt disk_preflight.txt \
  fleet_integrity.txt launch.log legacy.log legacy_summary.json \
  legacy_summary.jsonl orchestrator.log prereq.txt raylet_env.txt \
  source_identity.json stop.txt; do
  cp "$SOURCE_RUN_DIR/$name" "$RUN_DIR/source_capture/$name"
done
printf 'source_tag=%s\nsource_capture_pin=%s\nsealer_pin=%s\n' \
  "$SOURCE_TAG" "$SOURCE_CAPTURE_PIN" "$PIN" >"$RUN_DIR/source_recovery.txt"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/capture_short_context_dsa_oracle.py" \
  --results-db "$RESULTS_DB" \
  --token-oracle-dir "$TOKEN_ORACLE_DIR" \
  --source-dump-dir "$SOURCE_DIR" \
  --output "$ORACLE_DIR" \
  --expected-code-hash "$PIN" \
  --source-capture-code-hash "$SOURCE_CAPTURE_PIN" \
  --legacy-repository-pin "$LEGACY_PIN" \
  --token-oracle-manifest-sha256 "$TOKEN_ORACLE_SHA" \
  --run-id "$RUN_ID" \
  --item-row-id "$ITEM_ROW_ID" \
  --expected-harness-git "$SOURCE_HARNESS" \
  --expected-fork-git "$SOURCE_FORK" \
  --expected-oob-dir "$OOB_DIR" \
  --expected-dump-prefix "$DUMP_PREFIX" >"$RUN_DIR/capture.json"

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

strict_census post || {
  say "ABORT: post-seal census is not eight-host zero work"
  exit 1
}

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

say "uploading recovered compact oracle and immutable source evidence"
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
    crc32c = remote.get("crc32c_hash") or remote.get("crc32c")
    if int(remote["size"]) != path.stat().st_size or not crc32c:
        raise SystemExit(f"remote object verification failed: {relative}")
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
  "$SOURCE_CAPTURE_PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "oracle" / "manifest.json").read_text())
values = {
    "artifact_kind": manifest["artifact_kind"],
    "sealer_code_hash": sys.argv[3],
    "source_capture_code_hash": sys.argv[4],
    "manifest_sha256": manifest["manifest_sha256"],
    "source_run_id": manifest["source"]["run_id"],
    "source_item_row_id": manifest["source"]["item_row_id"],
    "source_dump_file_count": len(manifest["source_dump_files"]),
    "evidence_sha256": sha256((root / "evidence_sha256.json").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256((root / "remote_objects.json").read_bytes()).hexdigest(),
    "remote_prefix": sys.argv[2],
}
(root / "SUCCESS").write_text(
    "".join(f"{key}={value}\n" for key, value in values.items())
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
trap - EXIT
say "SUCCESS source_run=$RUN_ID item=$ITEM_ROW_ID manifest=$manifest_sha"
