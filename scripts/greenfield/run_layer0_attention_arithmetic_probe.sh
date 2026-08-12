#!/usr/bin/env bash
# Protected four-chip discriminator for the first pre-W_UV Gate-D mismatch.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly INGREDIENT_TAG=greenfield_table_on_layer0_ingredients_p8155_20260812T021718885910564Z
readonly INGREDIENT_DIR=/home/gianl/gcs-models/results/$INGREDIENT_TAG/layer0_ingredients
readonly INGREDIENT_NPZ=$INGREDIENT_DIR/position_8155_ingredients.npz
readonly INGREDIENT_JSON=$INGREDIENT_DIR/contract.json
readonly INGREDIENT_NPZ_SHA=c06fe575f0518e981c3099a8033c1978b01fd0cff843ecbbd7a25cebed8d0e95
readonly INGREDIENT_JSON_SHA=02f0f1c3e22b08c6646167ac5a7ffaf5056932003ee722752f7d40e26e714864
readonly ACCEPTED_TAG=greenfield_legacy_layer0_attention_projection_p8155_20260812T090000000000000Z
readonly ACCEPTED_DIR=/home/gianl/glm-run/$ACCEPTED_TAG
readonly ACCEPTED_NPZ=$ACCEPTED_DIR/attention_projection_capture/attention_projection.npz
readonly ACCEPTED_JSON=$ACCEPTED_DIR/attention_projection_capture/capture.json
readonly ACCEPTED_SUCCESS=$ACCEPTED_DIR/SUCCESS
readonly ACCEPTED_NPZ_SHA=3a619a0985fbb9ba6e1be9347bbc745c120fa74553ce1de5de830190a29c0a30
readonly ACCEPTED_JSON_SHA=f517b408f4ab6d2fb4d53e1aede696cc74e9d2d08ff70b8aa135291c4dcdaa32
readonly ACCEPTED_SUCCESS_SHA=88691576033fea623414034171441c213472ac26a3d5e7a04388cff434a0a47c
readonly ACCEPTED_CACHE_TAG=greenfield_legacy_layer0_main_cache_20260811T052303163478417Z
readonly ACCEPTED_CACHE_DIR=/home/gianl/glm-run/$ACCEPTED_CACHE_TAG
readonly ACCEPTED_CACHE_NPZ=$ACCEPTED_CACHE_DIR/layer0_main_cache_comparison/comparison.npz
readonly ACCEPTED_CACHE_JSON=$ACCEPTED_CACHE_DIR/layer0_main_cache_comparison/comparison.json
readonly ACCEPTED_CACHE_SUCCESS=$ACCEPTED_CACHE_DIR/SUCCESS
readonly ACCEPTED_CACHE_NPZ_SHA=a71213370a9f7a96998761af961ab379c91df1f286676f60b6bbab811efc0924
readonly ACCEPTED_CACHE_JSON_SHA=c06f919df5b0b8bac3eb5466e5bd2aa5b2bbc33ec19995e8a052967117e8b8d9
readonly ACCEPTED_CACHE_SUCCESS_SHA=7a46ae6574d1ae65018be9537ad198cdaca26b8e01bb94afbb61785f4ed7cd10
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z
readonly CHECKPOINT_SUCCESS_SHA=368ef308c7937258c181cdc42fecddf8a7f7ca7bab04470d39cb622aaa3c5b24
readonly CHECKPOINT_MANIFEST_SHA=de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134
readonly CHECKPOINT_REMOTE=$APPROVED_BUCKET/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z
readonly INGREDIENT_REMOTE=$APPROVED_BUCKET/results/$INGREDIENT_TAG/layer0_ingredients
readonly ACCEPTED_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/attention_projection/8k/$ACCEPTED_TAG
readonly ACCEPTED_CACHE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/layer0_main_cache/8k/$ACCEPTED_CACHE_TAG
readonly Q_A_TAG=greenfield_layer0_physical_lp4_dsa_query_production_exact_20260810T080508327295662Z
readonly Q_A_DIR=/home/gianl/glm-run/$Q_A_TAG
readonly Q_A_NPZ=$Q_A_DIR/physical_lp4_production_exact.npz
readonly Q_A_SUCCESS=$Q_A_DIR/SUCCESS
readonly Q_A_NPZ_SHA=b371ad77313c085268470c950f231a0cf23c4d17c7c7104b9c791cfc9f49d247
readonly Q_A_SUCCESS_SHA=b3cff36beaff7eaaf71c30349007c8c5acb8c7423ae46fb82c40339061a23b11
readonly Q_A_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/physical_lp4_dsa_query_production_exact/8k/$Q_A_TAG

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_ATTENTION_ARITHMETIC_TAG:-greenfield_layer0_attention_arithmetic_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing attention-arithmetic probe outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing attention-arithmetic probe from a dirty worktree" >&2
  exit 2
}
[[ -r $RESULTS_DB && ! -e $RUN_DIR ]] || {
  echo "results DB missing or append-only run path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[attention-arithmetic $(date -u +%H:%M:%S)] $*" |
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[p]robe_layer0_attention_arithmetic[.]py|[m]icrobench_sparse_attention[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
        "zai-org/GLM-5.2-FP8:greenfield-layer0-attention-arithmetic",
        "native-jax-pregathered-lp4-v1",
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
if (
    environment.get("greenfield_code_hash") != pin
    or run[4]
    != "Protected four-chip pre-W_UV arithmetic discriminator; no performance claim."
    or len(items) != 1
    or items[0][0:3]
    != (
        "greenfield_layer0_attention_arithmetic",
        "position8155",
        "Exact accepted BF16 attended latent [64,512].",
    )
    or items[0][3] not in (0, 1)
    or items[0][4] not in (0.0, 1.0)
    or summaries
    != [("greenfield_layer0_attention_arithmetic", "probe_contract_valid", 1.0)]
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
require_sha "$INGREDIENT_NPZ" "$INGREDIENT_NPZ_SHA" "ingredient tensor"
require_sha "$INGREDIENT_JSON" "$INGREDIENT_JSON_SHA" "ingredient contract"
require_sha "$ACCEPTED_NPZ" "$ACCEPTED_NPZ_SHA" "accepted operand"
require_sha "$ACCEPTED_JSON" "$ACCEPTED_JSON_SHA" "accepted capture"
require_sha "$ACCEPTED_SUCCESS" "$ACCEPTED_SUCCESS_SHA" "accepted SUCCESS"
require_sha "$ACCEPTED_CACHE_NPZ" "$ACCEPTED_CACHE_NPZ_SHA" "accepted main-cache tensor"
require_sha "$ACCEPTED_CACHE_JSON" "$ACCEPTED_CACHE_JSON_SHA" "accepted main-cache manifest"
require_sha "$ACCEPTED_CACHE_SUCCESS" "$ACCEPTED_CACHE_SUCCESS_SHA" "accepted main-cache SUCCESS"
require_sha "$CHECKPOINT_ROOT/SUCCESS" "$CHECKPOINT_SUCCESS_SHA" "checkpoint SUCCESS"
require_sha "$CHECKPOINT_ROOT/runtime_manifest.json" "$CHECKPOINT_MANIFEST_SHA" "checkpoint manifest"
require_sha "$Q_A_NPZ" "$Q_A_NPZ_SHA" "protected q-a reference"
require_sha "$Q_A_SUCCESS" "$Q_A_SUCCESS_SHA" "protected q-a SUCCESS"

[[ $(gcloud storage cat "$INGREDIENT_REMOTE/contract.json" | sha256sum | awk '{print $1}') == "$INGREDIENT_JSON_SHA" ]]
[[ $(gcloud storage cat "$INGREDIENT_REMOTE/position_8155_ingredients.npz" | sha256sum | awk '{print $1}') == "$INGREDIENT_NPZ_SHA" ]]
[[ $(gcloud storage cat "$ACCEPTED_REMOTE/SUCCESS" | sha256sum | awk '{print $1}') == "$ACCEPTED_SUCCESS_SHA" ]]
[[ $(gcloud storage cat "$ACCEPTED_REMOTE/attention_projection_capture/attention_projection.npz" | sha256sum | awk '{print $1}') == "$ACCEPTED_NPZ_SHA" ]]
[[ $(gcloud storage cat "$ACCEPTED_CACHE_REMOTE/SUCCESS" | sha256sum | awk '{print $1}') == "$ACCEPTED_CACHE_SUCCESS_SHA" ]]
[[ $(gcloud storage cat "$ACCEPTED_CACHE_REMOTE/layer0_main_cache_comparison/comparison.json" | sha256sum | awk '{print $1}') == "$ACCEPTED_CACHE_JSON_SHA" ]]
[[ $(gcloud storage cat "$ACCEPTED_CACHE_REMOTE/layer0_main_cache_comparison/comparison.npz" | sha256sum | awk '{print $1}') == "$ACCEPTED_CACHE_NPZ_SHA" ]]
[[ $(gcloud storage cat "$Q_A_REMOTE/SUCCESS" | sha256sum | awk '{print $1}') == "$Q_A_SUCCESS_SHA" ]]
[[ $(gcloud storage cat "$Q_A_REMOTE/physical_lp4_production_exact.npz" | sha256sum | awk '{print $1}') == "$Q_A_NPZ_SHA" ]]
[[ $(gcloud storage cat "$CHECKPOINT_REMOTE/SUCCESS" | sha256sum | awk '{print $1}') == "$CHECKPOINT_SUCCESS_SHA" ]]
[[ $(gcloud storage cat "$CHECKPOINT_REMOTE/runtime_manifest.json" | sha256sum | awk '{print $1}') == "$CHECKPOINT_MANIFEST_SHA" ]]

strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "running five isolated full-segment/block/head arithmetic arms with accepted and greenfield cache inputs"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    /home/gianl/vllm-env/bin/python \
      scripts/greenfield/probe_layer0_attention_arithmetic.py \
      --expected-code-hash "$PIN" \
      --ingredients "$INGREDIENT_NPZ" \
      --ingredients-sha256 "$INGREDIENT_NPZ_SHA" \
      --ingredients-contract "$INGREDIENT_JSON" \
      --ingredients-contract-sha256 "$INGREDIENT_JSON_SHA" \
      --accepted "$ACCEPTED_NPZ" \
      --accepted-sha256 "$ACCEPTED_NPZ_SHA" \
      --accepted-capture "$ACCEPTED_JSON" \
      --accepted-capture-sha256 "$ACCEPTED_JSON_SHA" \
      --accepted-main-cache "$ACCEPTED_CACHE_NPZ" \
      --accepted-main-cache-sha256 "$ACCEPTED_CACHE_NPZ_SHA" \
      --accepted-main-cache-manifest "$ACCEPTED_CACHE_JSON" \
      --accepted-main-cache-manifest-sha256 "$ACCEPTED_CACHE_JSON_SHA" \
      --q-a-reference "$Q_A_NPZ" \
      --q-a-reference-sha256 "$Q_A_NPZ_SHA" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA" \
      --output "$RUN_DIR/runner.json" \
      --tensor-output "$RUN_DIR/attention_arithmetic.npz" \
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
    or runner["artifact_kind"] != "glm52_layer0_attention_arithmetic_probe"
    or len(runner["arms"]) != 5
    or not all(arm["hlo"]["contract"]["passed"] for arm in runner["arms"].values())
):
    raise SystemExit("attention-arithmetic runner contract failed")

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

connection = pv.connect(db_path)
run_id = pv.start_run(
    connection,
    model="zai-org/GLM-5.2-FP8:greenfield-layer0-attention-arithmetic",
    revision="native-jax-pregathered-lp4-v1",
    env={
        "GLM_ENGINE": "greenfield_attention_arithmetic_probe",
        "greenfield_code_hash": pin,
        "accepted_tensor_sha256": runner["source_capture"]["tensor_sha256"],
        "accepted_cache_tensor_sha256": runner["segment"]["accepted_cache"]["tensor_file_sha256"],
        "ingredient_tensor_sha256": runner["ingredients"]["tensor_sha256"],
        "classification": runner["classification"],
        "greenfield_run_tag": run_tag,
    },
    note="Protected four-chip pre-W_UV arithmetic discriminator; no performance claim.",
    harness_repo=repo,
    fork_repo=None,
)
(run_dir / "provisional_db_run_id.txt").write_text(f"{run_id}\n")
pv.record_item(
    connection,
    run_id,
    benchmark="greenfield_layer0_attention_arithmetic",
    item_id="position8155",
    prompt="Sealed layer-0 full selected segment at first 8K decode row.",
    gold="Exact accepted BF16 attended latent [64,512].",
    raw_output=json.dumps(
        {
            "classification": runner["classification"],
            "exact_arms": runner["exact_arms"],
            "greenfield_cache_exact_arms": runner["greenfield_cache_exact_arms"],
            "mismatch_counts": {
                name: arm["accepted_comparison"]["mismatch_count"]
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
    benchmark="greenfield_layer0_attention_arithmetic",
    metric="probe_contract_valid",
    value=1.0,
    note="Diagnostic arithmetic classification only; no decoder/token-rate claim.",
)
connection.close()
summary = {
    "artifact_kind": runner["artifact_kind"],
    "classification": runner["classification"],
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "exact_arms": runner["exact_arms"],
    "greenfield_cache_exact_arms": runner["greenfield_cache_exact_arms"],
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
    raise SystemExit("attention-arithmetic DB snapshot failed integrity check")
print(f"ATTENTION_ARITHMETIC_VALID db_run={run_id}")
PY

say "freezing and archiving attention-arithmetic evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum attention_arithmetic.npz runner.json runner.log summary.json \
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
    if (
        int(value["size"]) != path.stat().st_size
        or crc32c != local_crc32c(path)
    ):
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
    "greenfield_cache_exact_arms": ",".join(summary["greenfield_cache_exact_arms"]) or "none",
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
