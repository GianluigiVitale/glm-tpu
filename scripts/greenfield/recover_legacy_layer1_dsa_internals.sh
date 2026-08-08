#!/usr/bin/env bash
# Seal the completed DB500 layer-1 observer capture without reloading the model.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_TAG=greenfield_legacy_layer1_dsa_internals_20260808T081617553654576Z
readonly SOURCE_RUN_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_GREENFIELD_PIN=b435c0c663c52d1352540f2c4b504696cd1d6d01
readonly LEGACY_PIN=83ff4a3576602ca844ea090550139a2ff00b0bb1
readonly ORACLE_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle
readonly TOKEN_ORACLE_SHA=e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2
readonly REFERENCE_DSA_ORACLE=/home/gianl/glm-run/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle
readonly INTERNAL_LAYER=model.layers.1.self_attn.attn
readonly INTERNAL_SOURCE_SHA=eb7a250072d23531707130deaabb6906590c6661e70637fe2cedf8069d35a9da
readonly RUN_ID=500
readonly ITEM_ROW_ID=1783
readonly EXPECTED_DUMP_COUNT=483
readonly EXPECTED_PROMPT_TOKENS=8155
readonly EXPECTED_GENERATED_TOKENS=20
readonly FIRST_SOURCE_STEP=5
readonly FIRST_DECODE_POSITION=8155
readonly SOURCE_DUMP_PREFIX=/tmp/$SOURCE_TAG/topk.npz

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_DSA_INTERNALS_RECOVERY_TAG:-greenfield_legacy_layer1_dsa_internals_recovery_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
SOURCE_DIR=$RUN_DIR/source_dumps
ORACLE_DIR=$RUN_DIR/oracle
CAPTURE_DIR=$RUN_DIR/internal_capture
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/dsa_internals/8k/recovery/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing layer-1 DSA recovery outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing layer-1 DSA recovery from a dirty worktree" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only recovery directory exists: $RUN_DIR" >&2
  exit 2
}
for path in "$RESULTS_DB" "$TOKEN_ORACLE_DIR/manifest.json" \
  "$REFERENCE_DSA_ORACLE/manifest.json" "$SOURCE_RUN_DIR/source_identity.json"; do
  [[ -r $path ]] || {
    echo "required recovery input is unavailable: $path" >&2
    exit 2
  }
done
find "$SOURCE_RUN_DIR/source_dumps" -type l -print -quit | grep -q . && {
  echo "source capture contains a symlink" >&2
  exit 2
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

mkdir -p "$RUN_DIR" "$SOURCE_DIR" "$RUN_DIR/source_capture"
say() {
  echo "[layer1-dsa-recovery $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" |
      sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}" ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving recovery diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "SOURCE=$SOURCE_RUN_DIR SOURCE_PIN=$SOURCE_GREENFIELD_PIN SEALER_PIN=$PIN"
say "RUN_DIR=$RUN_DIR REMOTE_PREFIX=$REMOTE_PREFIX"
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}
has_eight_unique_markers "$SOURCE_RUN_DIR/fleet_integrity.txt" INTEGRITY_OK || {
  say "ABORT: source run lacks exact eight-host state integrity"
  exit 1
}
has_eight_unique_markers "$SOURCE_RUN_DIR/fleet_internal_integrity.txt" INTERNAL_OK || {
  say "ABORT: source observer did not arm cleanly on all eight hosts"
  exit 1
}
[[ $(grep -c '^INTERNAL_OWNER ' "$SOURCE_RUN_DIR/fleet_internal_integrity.txt") -eq 1 &&
   $(grep -c '^INTERNAL_NONOWNER ' "$SOURCE_RUN_DIR/fleet_internal_integrity.txt") -eq 7 ]] || {
  say "ABORT: source owner-process coverage drifted"
  exit 1
}
has_eight_unique_markers "$SOURCE_RUN_DIR/census_failure_exit.txt" CENSUS_OK || {
  say "ABORT: source run lacks authenticated eight-host cleanup"
  exit 1
}
grep -q "GREENFIELD_PIN=$SOURCE_GREENFIELD_PIN" \
  "$SOURCE_RUN_DIR/orchestrator.log" || {
    say "ABORT: source greenfield pin drifted"
    exit 1
  }
grep -q "LEGACY_PIN=$LEGACY_PIN" "$SOURCE_RUN_DIR/orchestrator.log" || {
  say "ABORT: source observer pin drifted"
  exit 1
}
grep -q "INTERNAL_LAYER=$INTERNAL_LAYER" "$SOURCE_RUN_DIR/orchestrator.log" || {
  say "ABORT: source observer layer drifted"
  exit 1
}
grep -q '\[longctx\].*correct=True' "$SOURCE_RUN_DIR/legacy.log" || {
  say "ABORT: source raw item was not correct"
  exit 1
}

topk_count=$(find "$SOURCE_RUN_DIR/source_dumps" -type f \
  -name 'topk.step*.evt*.proc*.npz' | wc -l)
internal_path=$(find "$SOURCE_RUN_DIR/source_dumps" -type f \
  -name 'internals.model_layers_1_self_attn_attn.position8155.proc0.npz')
[[ $topk_count -eq $EXPECTED_DUMP_COUNT && -f $internal_path ]] || {
  say "ABORT: immutable source inventory drifted topk=$topk_count"
  exit 1
}
[[ $(find "$SOURCE_RUN_DIR/source_dumps" -type f \
  -name 'internals.*.position8155.proc*.npz' | wc -l) -eq 1 ]] || {
  say "ABORT: source internal file count drifted"
  exit 1
}
find "$SOURCE_RUN_DIR/source_dumps" -type f -name '*.ERROR*' -print -quit |
  grep -q . && {
    say "ABORT: source capture contains an error sentinel"
    exit 1
  }
[[ $(sha256sum "$internal_path" | awk '{print $1}') == "$INTERNAL_SOURCE_SHA" ]] || {
  say "ABORT: source internal checksum drifted"
  exit 1
}

/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$SOURCE_RUN_DIR/source_identity.json" \
  "$SOURCE_TAG" "$LEGACY_PIN" "$ORACLE_PIN" >"$RUN_DIR/source_identity.json" <<'PY'
import json
import sqlite3
import sys

db_path, source_path, source_tag, observer_pin, oracle_pin = sys.argv[1:]
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
rows = connection.execute(
    "SELECT r.harness_git,r.fork_git,r.env_json,i.id AS item_row_id,"
    "i.correct,i.n_prompt_tokens,i.n_gen_tokens,i.seed,i.gold,i.raw_output "
    "FROM runs r JOIN items i ON i.run_id=r.run_id WHERE r.run_id=500"
).fetchall()
assert len(rows) == 1
value = dict(rows[0])
env = json.loads(value.pop("env_json"))["os_env"]
expected = {
    "harness_git": "a4a17ac",
    "fork_git": "b3c25df47",
    "item_row_id": 1783,
    "correct": 1,
    "n_prompt_tokens": 8155,
    "n_gen_tokens": 20,
    "seed": 1093997,
    "gold": "881446",
    "raw_output": " 881446. Do not forget it. There and back again. The grass is green",
}
assert value == expected, value
expected_env = {
    "GLM_DSA_DUMP_INTERNALS_CODE_HASH": observer_pin,
    "GLM_DSA_DUMP_INTERNALS_LAYER": "model.layers.1.self_attn.attn",
    "GLM_DSA_DUMP_INTERNALS_ORACLE_PIN": oracle_pin,
    "GLM_DSA_DUMP_INTERNALS_POSITION": "8155",
    "GLM_DSA_DUMP_INTERNALS_RUN_TAG": source_tag,
    "GLM_DSA_DUMP_TOPK": f"/tmp/{source_tag}/topk.npz",
    "GLM_DSA_DUMP_TOPK_EVENTS": "all",
    "GLM_EXPECT_CODE_HASH": observer_pin[:9],
}
assert all(env.get(name) == expected_value for name, expected_value in expected_env.items())
source = json.load(open(source_path))
assert source == {name: expected[name] for name in (
    "correct", "fork_git", "harness_git", "item_row_id", "n_gen_tokens",
    "n_prompt_tokens", "raw_output",
)}
print(json.dumps({**value, "observer_env": expected_env}, indent=2, sort_keys=True))
PY

say "hard-linking immutable source evidence without duplicating bytes"
cp -al "$SOURCE_RUN_DIR/source_dumps/." "$SOURCE_DIR/"
for name in census_pre.txt census_failure_exit.txt disk_preflight.txt \
  fleet_integrity.txt fleet_internal_integrity.txt launch.log legacy.log \
  legacy_summary.json legacy_summary.jsonl orchestrator.log oracle_prereq.txt \
  prereq.txt raylet_env.txt source_identity.json stop.txt sync_observer.txt; do
  cp "$SOURCE_RUN_DIR/$name" "$RUN_DIR/source_capture/$name"
done
printf 'source_tag=%s\nsource_greenfield_pin=%s\nsealer_pin=%s\n' \
  "$SOURCE_TAG" "$SOURCE_GREENFIELD_PIN" "$PIN" >"$RUN_DIR/source_recovery.txt"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/capture_short_context_dsa_oracle.py" \
  --results-db "$RESULTS_DB" \
  --token-oracle-dir "$TOKEN_ORACLE_DIR" \
  --source-dump-dir "$SOURCE_DIR" \
  --output "$ORACLE_DIR" \
  --expected-code-hash "$PIN" \
  --source-capture-code-hash "$SOURCE_GREENFIELD_PIN" \
  --legacy-repository-pin "$LEGACY_PIN" \
  --token-oracle-manifest-sha256 "$TOKEN_ORACLE_SHA" \
  --run-id "$RUN_ID" \
  --item-row-id "$ITEM_ROW_ID" \
  --expected-harness-git a4a17ac \
  --expected-fork-git b3c25df47 \
  --expected-benchmark passkey_L8192_d0.5 \
  --expected-model-uri gs://driftbench-dsv4-uc/models/GLM-5.2-FP8 \
  --expected-prompt-tokens "$EXPECTED_PROMPT_TOKENS" \
  --expected-generated-tokens "$EXPECTED_GENERATED_TOKENS" \
  --expected-seed 1093997 \
  --expected-gold 881446 \
  --expected-oob-dir /home/gianl/gcs-models/models/GLM-5.2-FP8 \
  --expected-dump-prefix "$SOURCE_DUMP_PREFIX" \
  --expected-process-count 8 \
  --first-source-step "$FIRST_SOURCE_STEP" \
  --decode-step-count 14 \
  --first-decode-position "$FIRST_DECODE_POSITION" \
  --selected-width 2048 >"$RUN_DIR/capture.json"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$REFERENCE_DSA_ORACLE" "$ORACLE_DIR" >"$RUN_DIR/dsa_exact_comparison.json" <<'PY'
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.validation import compare_short_context_dsa_oracles

value = compare_short_context_dsa_oracles(Path(sys.argv[1]), Path(sys.argv[2]))
if not value["exact"]:
    raise SystemExit("recovered DSA event tensors differ from DB485")
print(json.dumps(value, indent=2, sort_keys=True))
PY

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/inspect_legacy_dsa_internals.py" \
  --source-dump-dir "$SOURCE_DIR" \
  --output "$CAPTURE_DIR" \
  --run-tag "$SOURCE_TAG" \
  --legacy-code-hash "$LEGACY_PIN" \
  --oracle-pin "$ORACLE_PIN" \
  --model-id zai-org/GLM-5.2-FP8 \
  --layer-name "$INTERNAL_LAYER" \
  --position "$FIRST_DECODE_POSITION" \
  --process-count 8 \
  --capture-process-indices 0 >"$RUN_DIR/internal_capture_summary.json"

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
  say "ABORT: post-recovery census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "freezing recovered layer-1 evidence"
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
    if not path.is_file() or relative == "orchestrator.log" or path.name in {
        "SUCCESS", "evidence_sha256.json",
    }:
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

say "uploading recovered evidence append-only"
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
    if not path.is_file() or relative == "orchestrator.log" or path.name in {
        "SUCCESS", "remote_objects.json",
    }:
        continue
    paths.append((path, relative))

def describe(item):
    path, relative = item
    remote = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}",
         "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    crc32c = remote.get("crc32c_hash") or remote.get("crc32c")
    if int(remote["size"]) != path.stat().st_size or not crc32c:
        raise SystemExit(f"remote object verification failed: {relative}")
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

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
oracle = json.loads((root / "oracle" / "manifest.json").read_text())
capture = json.loads((root / "internal_capture" / "capture.json").read_text())
exact_dsa = json.loads((root / "dsa_exact_comparison.json").read_text())
if capture["capture_layout"] != "topology_sharded_live_row_owner":
    raise SystemExit("recovered DSA internal capture layout drifted")
if capture["capture_process_indices"] != [0] or not exact_dsa["exact"]:
    raise SystemExit("recovered DSA owner/event identity drifted")
values = {
    "artifact_kind": "glm52_legacy_dsa_internal_recovery",
    "sealer_code_hash": sys.argv[3],
    "source_capture_code_hash": "b435c0c663c52d1352540f2c4b504696cd1d6d01",
    "legacy_observer_pin": "83ff4a3576602ca844ea090550139a2ff00b0bb1",
    "accepted_oracle_pin": "b3c25df47ac98783912dc658878181ec0a8ae16d",
    "source_run_id": oracle["source"]["run_id"],
    "source_item_row_id": oracle["source"]["item_row_id"],
    "dsa_manifest_sha256": oracle["manifest_sha256"],
    "dsa_event_tensors_exact": "true",
    "dsa_internal_capture_layout": capture["capture_layout"],
    "dsa_internal_file_count": len(capture["process_files"]),
    "dsa_internal_layer_name": capture["layer_name"],
    "dsa_internal_owner_actual_sha256": capture["owner_actual_sha256"],
    "dsa_internal_capture_manifest_sha256": sha256(
        (root / "internal_capture" / "capture.json").read_bytes()
    ).hexdigest(),
    "source_internal_file_sha256": "eb7a250072d23531707130deaabb6906590c6661e70637fe2cedf8069d35a9da",
    "performance_claim": "false",
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
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" |
  sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]] || {
  say "ABORT: remote SUCCESS checksum mismatch"
  exit 1
}

trap - EXIT
say "SUCCESS source_run=$RUN_ID item=$ITEM_ROW_ID capture=$(sed -n 's/^dsa_internal_capture_manifest_sha256=//p' "$RUN_DIR/SUCCESS")"
