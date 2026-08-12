#!/usr/bin/env bash
# Protected four-chip discriminator for layer-0 projection/reduction association.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db

readonly ATTENTION_TAG=greenfield_layer0_attention_arithmetic_20260812T114701365714147Z
readonly ATTENTION_DIR=/home/gianl/gcs-models/results/$ATTENTION_TAG
readonly ATTENTION_TENSOR=$ATTENTION_DIR/attention_arithmetic.npz
readonly ATTENTION_RUNNER=$ATTENTION_DIR/runner.json
readonly ATTENTION_SUCCESS=$ATTENTION_DIR/SUCCESS
readonly ATTENTION_TENSOR_SHA=7d5ebe15dd006a70d77f17d41eb47f25f58c6d6784ee332fbc638f2916581f61
readonly ATTENTION_RUNNER_SHA=7961622c297567a021edf2ab335f0d046db6665e8130e210bbd78919a2d7a4ec
readonly ATTENTION_SUCCESS_SHA=ecc2b873c29ffd9bc6551136e90352b33523f50873827b00ae91279a4db3153c
readonly ATTENTION_REMOTE=$APPROVED_BUCKET/results/$ATTENTION_TAG

readonly INGREDIENT_TAG=greenfield_table_on_layer0_ingredients_p8155_20260812T021718885910564Z
readonly INGREDIENT_DIR=/home/gianl/gcs-models/results/$INGREDIENT_TAG/layer0_ingredients
readonly INGREDIENT_NPZ=$INGREDIENT_DIR/position_8155_ingredients.npz
readonly INGREDIENT_JSON=$INGREDIENT_DIR/contract.json
readonly INGREDIENT_NPZ_SHA=c06fe575f0518e981c3099a8033c1978b01fd0cff843ecbbd7a25cebed8d0e95
readonly INGREDIENT_JSON_SHA=02f0f1c3e22b08c6646167ac5a7ffaf5056932003ee722752f7d40e26e714864
readonly INGREDIENT_REMOTE=$APPROVED_BUCKET/results/$INGREDIENT_TAG/layer0_ingredients

readonly ACCEPTED_TAG=greenfield_legacy_layer0_attention_projection_p8155_20260812T090000000000000Z
readonly ACCEPTED_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/attention_projection/8k/$ACCEPTED_TAG
readonly ACCEPTED_NPZ=$ACCEPTED_ROOT/attention_projection_capture/attention_projection.npz
readonly ACCEPTED_JSON=$ACCEPTED_ROOT/attention_projection_capture/capture.json
readonly ACCEPTED_SUCCESS=$ACCEPTED_ROOT/SUCCESS
readonly ACCEPTED_NPZ_SHA=3a619a0985fbb9ba6e1be9347bbc745c120fa74553ce1de5de830190a29c0a30
readonly ACCEPTED_JSON_SHA=f517b408f4ab6d2fb4d53e1aede696cc74e9d2d08ff70b8aa135291c4dcdaa32
readonly ACCEPTED_SUCCESS_SHA=88691576033fea623414034171441c213472ac26a3d5e7a04388cff434a0a47c
readonly ACCEPTED_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/attention_projection/8k/$ACCEPTED_TAG

readonly LAYER1_TAG=greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z
readonly LAYER1_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/layer1/$LAYER1_TAG
readonly LAYER1_REFERENCE=$LAYER1_ROOT/internals.npz
readonly LAYER1_COMPARISON=$LAYER1_ROOT/comparison.json
readonly LAYER1_SEAL=$LAYER1_ROOT/diagnostic_seal.json
readonly LAYER1_REFERENCE_SHA=79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054
readonly LAYER1_COMPARISON_SHA=1bc43a8e231323c85b6782926182f2999f3014b8adbafa6f4db18b74e3419ad5
readonly LAYER1_SEAL_SHA=283e5e88b7aa836bd237e1b49e6d35ff2039db7639d2e34feccb61ec1fcb10d5
readonly LAYER1_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/dsa_internals/8k/layer1/$LAYER1_TAG

readonly ASSOCIATION_TAG=greenfield_collective_association_20260811T213152133863450Z
readonly ASSOCIATION_ROOT=/home/gianl/gcs-models/results/$ASSOCIATION_TAG
readonly ASSOCIATION_ANALYSIS=$ASSOCIATION_ROOT/association/analysis.json
readonly ASSOCIATION_SUCCESS=$ASSOCIATION_ROOT/SUCCESS
readonly ASSOCIATION_ANALYSIS_SHA=e7e34828365ca3d6cae0052f8d0e2e802143c6ca83810153db3116423f994108
readonly ASSOCIATION_SUCCESS_SHA=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
readonly ASSOCIATION_REMOTE=$APPROVED_BUCKET/results/$ASSOCIATION_TAG

readonly CHECKPOINT_TAG=greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$CHECKPOINT_TAG
readonly CHECKPOINT_SUCCESS_SHA=368ef308c7937258c181cdc42fecddf8a7f7ca7bab04470d39cb622aaa3c5b24
readonly CHECKPOINT_MANIFEST_SHA=de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134
readonly CHECKPOINT_REMOTE=$APPROVED_BUCKET/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$CHECKPOINT_TAG

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_PROJECTION_REDUCTION_TAG:-greenfield_layer0_projection_reduction_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing projection/reduction probe outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing projection/reduction probe from a dirty worktree" >&2
  exit 2
}
[[ -r $RESULTS_DB && ! -e $RUN_DIR ]] || {
  echo "results DB missing or append-only run path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[projection-reduction $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

set +e
remote_prefix_listing=$(gcloud storage ls "$REMOTE_PREFIX/**" 2>&1)
remote_prefix_rc=$?
set -e
printf '%s\n' "$remote_prefix_listing" >"$RUN_DIR/remote_prefix_preflight.txt"
if [[ $remote_prefix_rc -eq 0 ]]; then
  say "ABORT: append-only remote prefix already contains objects"
  exit 1
elif [[ $remote_prefix_rc -ne 1 ]] || ! grep -q "matched no objects" \
  "$RUN_DIR/remote_prefix_preflight.txt"; then
  say "ABORT: remote-prefix vacancy check failed"
  exit 1
fi

require_sha() {
  local path=$1 expected=$2 label=$3
  [[ -r $path && $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: $label prerequisite hash drifted"
    exit 1
  }
}

require_remote_sha() {
  local uri=$1 expected=$2 label=$3
  [[ $(gcloud storage cat "$uri" | sha256sum | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: $label remote hash drifted"
    exit 1
  }
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
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
  local ray_enum
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[p]robe_layer0_projection_reduction[.]py|[p]robe_layer0_attention_arithmetic[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
terminal_success_done=0

rollback_provisional_db() {
  /home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$TAG" "$PIN" \
    >"$RUN_DIR/provisional_db_rollback.txt" <<'PY'
from __future__ import annotations

import json
import sqlite3
import sys

db_path, run_tag, pin = sys.argv[1:]
connection = sqlite3.connect(db_path)
connection.execute("BEGIN IMMEDIATE")
matches = []
for row in connection.execute(
    "SELECT run_id, model, model_revision, env_json, note FROM runs "
    "WHERE model = ? AND model_revision = ?",
    (
        "zai-org/GLM-5.2-FP8:greenfield-layer0-projection-reduction",
        "native-jax-db537-strategy-nd-v1",
    ),
):
    environment = json.loads(row[3])
    if environment.get("greenfield_run_tag") == run_tag:
        matches.append((row, environment))
if not matches:
    connection.rollback()
    print("NO_PROVISIONAL_DB_RUN")
    raise SystemExit(0)
if len(matches) != 1:
    connection.rollback()
    raise SystemExit("refusing ambiguous provisional DB rollback")
(run, environment), = matches
run_id = run[0]
items = connection.execute(
    "SELECT benchmark, item_id, gold, correct, score FROM items WHERE run_id = ?",
    (run_id,),
).fetchall()
summaries = connection.execute(
    "SELECT benchmark, metric, value FROM summary WHERE run_id = ?",
    (run_id,),
).fetchall()
expected_item_prefix = (
        "greenfield_layer0_projection_reduction",
        "position8155",
        "Exact accepted BF16 layer-1 normalized hidden [6144].",
)
item_state_valid = not items or (
    len(items) == 1
    and items[0][0:3] == expected_item_prefix
    and items[0][3] in (0, 1)
    and items[0][4] in (0.0, 1.0)
)
summary_state_valid = not summaries or summaries == [
    ("greenfield_layer0_projection_reduction", "probe_contract_valid", 1.0)
]
if (
    environment.get("greenfield_code_hash") != pin
    or run[4]
    != "Protected layer-0 projection/reduction discriminator; no performance claim."
    or not item_state_valid
    or not summary_state_valid
    or (summaries and not items)
):
    connection.rollback()
    raise SystemExit("refusing non-identical provisional DB rollback")
connection.execute("DELETE FROM summary WHERE run_id = ?", (run_id,))
connection.execute("DELETE FROM items WHERE run_id = ?", (run_id,))
connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
if connection.execute(
    "SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,)
).fetchone()[0]:
    connection.rollback()
    raise SystemExit("provisional DB rollback did not remove run")
connection.commit()
print(f"ROLLED_BACK_PROVISIONAL_DB_RUN={run_id}")
PY
}

on_exit() {
  local status=$?
  if [[ $status -ne 0 && $terminal_success_done -eq 0 ]]; then
    rollback_provisional_db || true
  fi
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN"
require_sha "$ATTENTION_TENSOR" "$ATTENTION_TENSOR_SHA" "DB537 tensor"
require_sha "$ATTENTION_RUNNER" "$ATTENTION_RUNNER_SHA" "DB537 runner"
require_sha "$ATTENTION_SUCCESS" "$ATTENTION_SUCCESS_SHA" "DB537 SUCCESS"
require_sha "$INGREDIENT_NPZ" "$INGREDIENT_NPZ_SHA" "ingredient tensor"
require_sha "$INGREDIENT_JSON" "$INGREDIENT_JSON_SHA" "ingredient contract"
require_sha "$ACCEPTED_NPZ" "$ACCEPTED_NPZ_SHA" "accepted projection"
require_sha "$ACCEPTED_JSON" "$ACCEPTED_JSON_SHA" "accepted capture"
require_sha "$ACCEPTED_SUCCESS" "$ACCEPTED_SUCCESS_SHA" "accepted SUCCESS"
require_sha "$LAYER1_REFERENCE" "$LAYER1_REFERENCE_SHA" "layer-1 reference"
require_sha "$LAYER1_COMPARISON" "$LAYER1_COMPARISON_SHA" "layer-1 comparison"
require_sha "$LAYER1_SEAL" "$LAYER1_SEAL_SHA" "layer-1 diagnostic seal"
require_sha "$ASSOCIATION_ANALYSIS" "$ASSOCIATION_ANALYSIS_SHA" "association analysis"
require_sha "$ASSOCIATION_SUCCESS" "$ASSOCIATION_SUCCESS_SHA" "association SUCCESS"
require_sha "$CHECKPOINT_ROOT/SUCCESS" "$CHECKPOINT_SUCCESS_SHA" "checkpoint SUCCESS"
require_sha "$CHECKPOINT_ROOT/runtime_manifest.json" "$CHECKPOINT_MANIFEST_SHA" "checkpoint manifest"

require_remote_sha "$ATTENTION_REMOTE/attention_arithmetic.npz" "$ATTENTION_TENSOR_SHA" "DB537 tensor"
require_remote_sha "$ATTENTION_REMOTE/runner.json" "$ATTENTION_RUNNER_SHA" "DB537 runner"
require_remote_sha "$ATTENTION_REMOTE/SUCCESS" "$ATTENTION_SUCCESS_SHA" "DB537 SUCCESS"
require_remote_sha "$INGREDIENT_REMOTE/position_8155_ingredients.npz" "$INGREDIENT_NPZ_SHA" "ingredient tensor"
require_remote_sha "$INGREDIENT_REMOTE/contract.json" "$INGREDIENT_JSON_SHA" "ingredient contract"
require_remote_sha "$ACCEPTED_REMOTE/attention_projection_capture/attention_projection.npz" "$ACCEPTED_NPZ_SHA" "accepted projection"
require_remote_sha "$ACCEPTED_REMOTE/attention_projection_capture/capture.json" "$ACCEPTED_JSON_SHA" "accepted capture"
require_remote_sha "$ACCEPTED_REMOTE/SUCCESS" "$ACCEPTED_SUCCESS_SHA" "accepted SUCCESS"
require_remote_sha "$LAYER1_REMOTE/internals.npz" "$LAYER1_REFERENCE_SHA" "layer-1 reference"
require_remote_sha "$LAYER1_REMOTE/comparison.json" "$LAYER1_COMPARISON_SHA" "layer-1 comparison"
require_remote_sha "$LAYER1_REMOTE/diagnostic_seal.json" "$LAYER1_SEAL_SHA" "layer-1 diagnostic seal"
require_remote_sha "$ASSOCIATION_REMOTE/association/analysis.json" "$ASSOCIATION_ANALYSIS_SHA" "association analysis"
require_remote_sha "$ASSOCIATION_REMOTE/SUCCESS" "$ASSOCIATION_SUCCESS_SHA" "association SUCCESS"
require_remote_sha "$CHECKPOINT_REMOTE/SUCCESS" "$CHECKPOINT_SUCCESS_SHA" "checkpoint SUCCESS"
require_remote_sha "$CHECKPOINT_REMOTE/runtime_manifest.json" "$CHECKPOINT_MANIFEST_SHA" "checkpoint manifest"

strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "running four isolated projection/reduction association arms"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    /home/gianl/vllm-env/bin/python \
      scripts/greenfield/probe_layer0_projection_reduction.py \
      --expected-code-hash "$PIN" \
      --attention-arithmetic "$ATTENTION_TENSOR" \
      --attention-arithmetic-sha256 "$ATTENTION_TENSOR_SHA" \
      --attention-runner "$ATTENTION_RUNNER" \
      --attention-runner-sha256 "$ATTENTION_RUNNER_SHA" \
      --accepted-projection "$ACCEPTED_NPZ" \
      --accepted-projection-sha256 "$ACCEPTED_NPZ_SHA" \
      --accepted-projection-capture "$ACCEPTED_JSON" \
      --accepted-projection-capture-sha256 "$ACCEPTED_JSON_SHA" \
      --ingredients "$INGREDIENT_NPZ" \
      --ingredients-sha256 "$INGREDIENT_NPZ_SHA" \
      --ingredients-contract "$INGREDIENT_JSON" \
      --ingredients-contract-sha256 "$INGREDIENT_JSON_SHA" \
      --layer1-reference "$LAYER1_REFERENCE" \
      --layer1-reference-sha256 "$LAYER1_REFERENCE_SHA" \
      --association-analysis "$ASSOCIATION_ANALYSIS" \
      --association-analysis-sha256 "$ASSOCIATION_ANALYSIS_SHA" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA" \
      --output "$RUN_DIR/runner.json" \
      --tensor-output "$RUN_DIR/projection_reduction.npz" \
      --hlo-dir "$RUN_DIR/hlo"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "probe completed in ${elapsed}s"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" "$TAG" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, db_path, repo, elapsed, run_tag = sys.argv[1:]
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
if (
    runner["status"] != "SUCCESS"
    or runner["code_hash"] != pin
    or runner["artifact_kind"] != "glm52_layer0_projection_reduction_probe"
    or len(runner["arms"]) != 4
    or not all(arm["hlo"]["contract"]["passed"] for arm in runner["arms"].values())
    or not all(arm["value_comparison"]["elementwise_exact"] for arm in runner["arms"].values())
):
    raise SystemExit("projection/reduction runner contract failed")

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

connection = pv.connect(db_path)
run_id = pv.start_run(
    connection,
    model="zai-org/GLM-5.2-FP8:greenfield-layer0-projection-reduction",
    revision="native-jax-db537-strategy-nd-v1",
    env={
        "GLM_ENGINE": "greenfield_projection_reduction_probe",
        "greenfield_code_hash": pin,
        "greenfield_run_tag": run_tag,
        "attention_tensor_sha256": runner["source"]["attention_arithmetic_sha256"],
        "association_analysis_sha256": runner["source"]["association_analysis_sha256"],
        "classification": runner["classification"],
    },
    note="Protected layer-0 projection/reduction discriminator; no performance claim.",
    harness_repo=repo,
    fork_repo=None,
)
(run_dir / "provisional_db_run_id.txt").write_text(f"{run_id}\n")
pv.record_item(
    connection,
    run_id,
    benchmark="greenfield_layer0_projection_reduction",
    item_id="position8155",
    prompt="Sealed exact B512 latent and layer-0 residual at first 8K decode row.",
    gold="Exact accepted BF16 layer-1 normalized hidden [6144].",
    raw_output=json.dumps(
        {
            "classification": runner["classification"],
            "exact_arms": runner["exact_arms"],
            "mismatch_counts": {
                name: arm["layer1_comparison"]["mismatch_count"]
                for name, arm in runner["arms"].items()
            },
        },
        sort_keys=True,
    ),
    extracted=",".join(runner["exact_arms"]) or "none",
    correct=bool(runner["exact_arms"]),
    score=float(bool(runner["exact_arms"])),
)
pv.finalize(
    connection,
    run_id,
    benchmark="greenfield_layer0_projection_reduction",
    metric="probe_contract_valid",
    value=1.0,
    note="Diagnostic projection/reduction classification only; no decoder claim.",
)
connection.close()
summary = {
    "artifact_kind": runner["artifact_kind"],
    "classification": runner["classification"],
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "exact_arms": runner["exact_arms"],
    "performance_claim": False,
    "results_db_run_id": run_id,
    "status": "SUCCESS",
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
if sqlite3.connect(run_dir / "results_ckpt.db").execute(
    "PRAGMA integrity_check"
).fetchone()[0] != "ok":
    raise SystemExit("projection/reduction DB snapshot failed integrity check")
print(f"PROJECTION_REDUCTION_VALID db_run={run_id}")
PY

say "freezing and archiving projection/reduction evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum projection_reduction.npz runner.json runner.log summary.json \
    results_ckpt.db census_pre.txt census_post.txt orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" \
  >"$RUN_DIR/remote_objects.json" <<'PY'
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = [
    (path, path.relative_to(root).as_posix())
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.name not in {"SUCCESS", "remote_objects.json"}
]

def local_crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()

def describe(item):
    path, relative = item
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}",
         "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    crc32c = value.get("crc32c_hash") or value.get("crc32c")
    if int(value["size"]) != path.stat().st_size or crc32c != local_crc32c(path):
        raise SystemExit(f"remote object verification failed: {relative}")
    return {
        "crc32c": crc32c,
        "generation": value["generation"],
        "path": relative,
        "size": int(value["size"]),
    }

with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null
local_remote_objects_sha=$(sha256sum "$RUN_DIR/remote_objects.json" | awk '{print $1}')
remote_remote_objects_sha=$(gcloud storage cat "$REMOTE_PREFIX/remote_objects.json" |
  sha256sum | awk '{print $1}')
[[ $local_remote_objects_sha == "$remote_remote_objects_sha" ]] || {
  say "ABORT: remote object ledger checksum mismatch"
  exit 1
}

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
values = {
    "artifact_kind": summary["artifact_kind"],
    "code_hash": sys.argv[3],
    "results_db_run_id": summary["results_db_run_id"],
    "classification": summary["classification"],
    "exact_arms": ",".join(summary["exact_arms"]) or "none",
    "performance_claim": "false",
    "evidence_sha256": sha256((root / "evidence.sha256").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256(
        (root / "remote_objects.json").read_bytes()
    ).hexdigest(),
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
terminal_success_done=1

trap - EXIT
db_run=$(sed -n 's/^results_db_run_id=//p' "$RUN_DIR/SUCCESS")
exact=$(sed -n 's/^exact_arms=//p' "$RUN_DIR/SUCCESS")
say "SUCCESS DB=$db_run exact=$exact"
say "ARCHIVE=$REMOTE_PREFIX"
