#!/usr/bin/env bash
# Protected metadata-only publication of the complete PP16_LP2 checkpoint plan.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly SOURCE_URI=$APPROVED_BUCKET/models/GLM-5.2-FP8
readonly SOURCE_REVISION=gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658
readonly SOURCE_INVENTORY_SHA=a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4
readonly TOPOLOGY_CAPTURE=/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records/topology.rank0.json
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly PLAN_GROUP_HASH=6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21
readonly EXECUTION_PLAN_HASH=079cefe6a646e0c56120c0903c395587bc95bf8dc6d66de0cf9eb029827794c3
readonly PLAN_MANIFEST_SHA=3c3ea07b0f97a84702e7b2ad370b014ea22c9e69bc7d35ad88e23a4ad82ded16

cd "$WORKTREE"
[[ ${GLM_GREENFIELD_PP16_PLAN:-0} == 1 ]] || {
  echo "PP16 checkpoint-plan publication is default-off" >&2
  exit 2
}

PIN=$(git rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_PLAN_TAG:-greenfield_checkpoint_plan_pp16_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_checkpoint_plan_pp16_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid PP16 checkpoint-plan tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
ARTIFACT_DIR=$RUN_DIR/artifact
REMOTE_PREFIX=$APPROVED_BUCKET/checkpoints/greenfield/glm52/plans/PP16_LP2/$TAG
readonly PIN TAG RUN_DIR ARTIFACT_DIR REMOTE_PREFIX

[[ $(git branch --show-current) == "$BRANCH" ]] || {
  echo "refusing PP16 plan outside $BRANCH" >&2
  exit 2
}
[[ -z $(git status --porcelain) ]] || {
  echo "refusing PP16 plan from a dirty worktree" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only PP16 plan run already exists: $RUN_DIR" >&2
  exit 2
}
[[ -r $SOURCE_ROOT/model.safetensors.index.json && -r $TOPOLOGY_CAPTURE ]] || {
  echo "PP16 plan source metadata is unavailable" >&2
  exit 2
}

mkdir -p "$RUN_DIR"
# The operation is CPU/metadata-only. Hold the repository-mirror lease so its
# GCS metadata reads and publication never overlap the five-minute cron sync.
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

say() {
  echo "[checkpoint-plan-pp16 $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

say "PIN=$PIN TAG=$TAG"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]] || {
  say "ABORT: current commit is not pushed to the protected branch"
  exit 1
}

bucket_location=$(gcloud storage buckets describe "$APPROVED_BUCKET" \
  --format='value(location)')
[[ $bucket_location == "$APPROVED_LOCATION" ]] || {
  say "ABORT: approved bucket location is $bucket_location, expected $APPROVED_LOCATION"
  exit 1
}
mount_record=$(findmnt -T "$SOURCE_ROOT" -n -o SOURCE,FSTYPE,OPTIONS)
[[ $mount_record == driftbench-dsv4-uc\ fuse.gcsfuse\ * && $mount_record == *ro* ]] || {
  say "ABORT: source is not the approved read-only bucket mount: $mount_record"
  exit 1
}

gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' \
  >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]] || {
  say "ABORT: append-only remote prefix is occupied"
  exit 1
}

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$TOPOLOGY_CAPTURE" "$RUN_DIR/preflight.json" "$PIN" "$TAG" \
  "$REMOTE_PREFIX" "$bucket_location" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

topology_path, output, pin, tag, remote, location = sys.argv[1:]
raw = Path(topology_path).read_bytes()
record = json.loads(raw)
contract = record["contract"]
expected = {
    "topology_hash": "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559",
    "pp16_lp2_hash": "6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21",
}
if any(contract.get(key) != value for key, value in expected.items()):
    raise SystemExit("replacement-pod topology contract drifted")
value = {
    "approved_bucket": "gs://driftbench-dsv4-uc",
    "bucket_location": location,
    "code_hash": pin,
    "plan_id": "PP16_LP2",
    "remote_prefix": remote,
    "run_tag": tag,
    "source_revision": "gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658",
    "topology_capture_sha256": sha256(raw).hexdigest(),
    **expected,
}
Path(output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

say "building the 141-header PP16 plan; no tensor payload and no TPU access"
PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu \
  /home/gianl/vllm-env/bin/python scripts/greenfield/build_checkpoint_plan.py \
  --plan PP16_LP2 \
  --source-root "$SOURCE_ROOT" \
  --source-uri "$SOURCE_URI" \
  --source-revision "$SOURCE_REVISION" \
  --topology-capture "$TOPOLOGY_CAPTURE" \
  --output-dir "$ARTIFACT_DIR" \
  --expected-code-hash "$PIN" >"$RUN_DIR/builder.log" 2>&1

say "reconciling complete local plan and exact PP16 geometry"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$ARTIFACT_DIR" "$PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.partitioning import SourceInventory, inspect_layout_manifest

root = Path(sys.argv[1]); pin = sys.argv[2]
inventory_raw = json.loads((root / "source_inventory.json").read_text())
inventory = SourceInventory.from_dict(inventory_raw)
plan = json.loads((root / "plan.json").read_text())
layout = inspect_layout_manifest(root / "layout_manifest.json")
summary = json.loads((root / "summary.json").read_text())

expected_inventory = "a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4"
expected_plan = "3c3ea07b0f97a84702e7b2ad370b014ea22c9e69bc7d35ad88e23a4ad82ded16"
expected_execution = "079cefe6a646e0c56120c0903c395587bc95bf8dc6d66de0cf9eb029827794c3"
expected_topology = "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
expected_group = "6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21"
expected_revision = "gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658"
expected_stages = [
    (0, [0, 1], 0, 7), (1, [8, 9], 7, 12),
    (2, [16, 17], 12, 17), (3, [24, 25], 17, 22),
    (4, [26, 27], 22, 26), (5, [2, 3], 26, 30),
    (6, [10, 11], 30, 34), (7, [18, 19], 34, 39),
    (8, [20, 21], 39, 44), (9, [12, 13], 44, 49),
    (10, [4, 5], 49, 54), (11, [28, 29], 54, 59),
    (12, [30, 31], 59, 64), (13, [22, 23], 64, 69),
    (14, [14, 15], 69, 74), (15, [6, 7], 74, 78),
]

if inventory.inventory_sha256 != expected_inventory:
    raise SystemExit("source inventory identity drifted")
if len(inventory.files) != 141 or len(inventory.tensors) != 118629:
    raise SystemExit("source inventory cardinality drifted")
if inventory.source_revision != expected_revision:
    raise SystemExit("source revision drifted")
canonical = json.dumps(plan, allow_nan=False, ensure_ascii=True,
                       separators=(",", ":"), sort_keys=True).encode()
if plan.get("plan_manifest_sha256") != expected_plan:
    raise SystemExit("serialized PP16 plan hash drifted")
without_added_hash = dict(plan); without_added_hash.pop("plan_manifest_sha256")
if sha256(json.dumps(without_added_hash, allow_nan=False, ensure_ascii=True,
                     separators=(",", ":"), sort_keys=True).encode()).hexdigest() != expected_plan:
    raise SystemExit("PP16 plan content does not reproduce its hash")
if plan["execution_plan_sha256"] != expected_execution:
    raise SystemExit("PP16 execution-plan hash drifted")
stages = [(item["stage_id"], item["device_ids"], item["layer_start"],
           item["layer_end_exclusive"]) for item in plan["stages"]]
if stages != expected_stages:
    raise SystemExit("PP16 physical stage/layer assignment drifted")
if plan["index_share_crossings"] != [7, 12, 17, 39, 44, 49, 59, 64, 69]:
    raise SystemExit("PP16 IndexShare crossings drifted")
if not plan["capacity_feasible"] or plan["promotion_memory_proven"]:
    raise SystemExit("PP16 capacity/promotion proof state drifted")

checks = {
    "artifact_kind": "greenfield_checkpoint_plan",
    "base_destination_file_count": 32,
    "capacity_feasible": True,
    "code_hash": pin,
    "index_share_crossings": plan["index_share_crossings"],
    "inventory_sha256": expected_inventory,
    "maximum_accounted_bytes": 26424338752,
    "minimum_free_bytes": 6590074560,
    "mtp_destination_file_count": 2,
    "packed_payload_bytes": 757149950848,
    "plan_group_hash": expected_group,
    "plan_id": "PP16_LP2",
    "plan_manifest_sha256": expected_plan,
    "promotion_memory_proven": False,
    "source_file_count": 141,
    "source_leaf_count": 118629,
    "source_revision": expected_revision,
    "topology_hash": expected_topology,
}
for key, expected in checks.items():
    if summary.get(key) != expected:
        raise SystemExit(f"PP16 summary drifted at {key}")
if (layout["plan_id"] != "PP16_LP2" or layout["code_hash"] != pin or
        layout["topology_hash"] != expected_topology or
        layout["plan_group_hash"] != expected_group or
        layout["plan_manifest_sha256"] != expected_plan or
        layout["source"]["inventory_sha256"] != expected_inventory or
        layout["source"]["revision"] != expected_revision or
        layout["source"]["uri"] != "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"):
    raise SystemExit("PP16 layout identity drifted")
if layout["plan_manifest"] != without_added_hash:
    raise SystemExit("standalone and embedded PP16 plans disagree")
if summary["layout_manifest_sha256"] != layout["manifest_sha256"]:
    raise SystemExit("summary/layout identity drifted")

lines = (root / "evidence.sha256").read_text().splitlines()
expected_names = ["source_inventory.json", "plan.json", "layout_manifest.json", "summary.json"]
if [line.split("  ", 1)[1] for line in lines] != expected_names:
    raise SystemExit("local evidence ledger names drifted")
for line in lines:
    digest, name = line.split("  ", 1)
    if sha256((root / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"local evidence checksum drifted: {name}")
if (root / "SUCCESS").read_text() != f'{layout["manifest_sha256"]}  layout_manifest.json\n':
    raise SystemExit("builder SUCCESS identity drifted")
PY

say "publishing immutable nonterminal plan evidence in $APPROVED_LOCATION"
for name in source_inventory.json plan.json layout_manifest.json summary.json evidence.sha256; do
  gcloud storage cp --no-clobber "$ARTIFACT_DIR/$name" \
    "$REMOTE_PREFIX/$name" >/dev/null
done
gcloud storage cp --no-clobber "$RUN_DIR/preflight.json" \
  "$RUN_DIR/builder.log" "$REMOTE_PREFIX/" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$ARTIFACT_DIR" "$REMOTE_PREFIX" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

import google_crc32c
from google.cloud import storage

run = Path(sys.argv[1]); artifact = Path(sys.argv[2]); remote = sys.argv[3]
local = {
    **{name: artifact / name for name in
       ("source_inventory.json", "plan.json", "layout_manifest.json",
        "summary.json", "evidence.sha256")},
    "preflight.json": run / "preflight.json",
    "builder.log": run / "builder.log",
}
bucket_name, prefix = remote[5:].split("/", 1); prefix = prefix.rstrip("/") + "/"
blobs = {blob.name.removeprefix(prefix): blob for blob in
         storage.Client().list_blobs(bucket_name, prefix=prefix)}
if set(blobs) != set(local):
    raise SystemExit("preterminal remote PP16 object set drifted")
records = []
for name, path in sorted(local.items()):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    crc = base64.b64encode(checksum.digest()).decode("ascii")
    blob = blobs[name]
    if int(blob.size) != path.stat().st_size or blob.crc32c != crc or not blob.generation:
        raise SystemExit(f"remote PP16 object identity drifted: {name}")
    records.append({"crc32c": crc, "generation": int(blob.generation),
                    "name": name, "size": int(blob.size)})
ledger = {"objects": records, "remote_prefix": remote}
canonical = json.dumps(ledger, allow_nan=False, ensure_ascii=True,
                       separators=(",", ":"), sort_keys=True).encode()
ledger["ledger_sha256"] = sha256(canonical).hexdigest()
(run / "remote_objects.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$ARTIFACT_DIR" "$PIN" "$TAG" "$REMOTE_PREFIX" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

import google_crc32c
from google.cloud import storage

run, artifact = Path(sys.argv[1]), Path(sys.argv[2])
pin, tag, remote = sys.argv[3:]
summary = json.loads((artifact / "summary.json").read_text())
ledger_path = run / "remote_objects.json"
ledger = json.loads(ledger_path.read_text())
without_hash = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
canonical = json.dumps(without_hash, allow_nan=False, ensure_ascii=True,
                       separators=(",", ":"), sort_keys=True).encode()
if ledger["ledger_sha256"] != sha256(canonical).hexdigest():
    raise SystemExit("remote ledger self-hash drifted")
bucket_name, prefix = remote[5:].split("/", 1); prefix = prefix.rstrip("/") + "/"
blob = storage.Client().bucket(bucket_name).blob(prefix + "remote_objects.json")
blob.reload()
checksum = google_crc32c.Checksum(ledger_path.read_bytes())
crc = base64.b64encode(checksum.digest()).decode("ascii")
if int(blob.size) != ledger_path.stat().st_size or blob.crc32c != crc or not blob.generation:
    raise SystemExit("published remote ledger identity drifted")
value = {
    "artifact_kind": "greenfield_checkpoint_plan_SUCCESS",
    "capacity_feasible": True,
    "code_hash": pin,
    "execution_plan_sha256": "079cefe6a646e0c56120c0903c395587bc95bf8dc6d66de0cf9eb029827794c3",
    "inventory_sha256": summary["inventory_sha256"],
    "layout_manifest_sha256": summary["layout_manifest_sha256"],
    "minimum_free_bytes": summary["minimum_free_bytes"],
    "performance_claim": False,
    "plan_group_hash": summary["plan_group_hash"],
    "plan_id": "PP16_LP2",
    "plan_manifest_sha256": summary["plan_manifest_sha256"],
    "promotion_memory_proven": False,
    "remote_ledger_crc32c": crc,
    "remote_ledger_generation": int(blob.generation),
    "remote_ledger_sha256": ledger["ledger_sha256"],
    "remote_prefix": remote,
    "run_tag": tag,
    "source_revision": summary["source_revision"],
    "topology_hash": summary["topology_hash"],
}
canonical = json.dumps(value, allow_nan=False, ensure_ascii=True,
                       separators=(",", ":"), sort_keys=True).encode()
value["success_sha256"] = sha256(canonical).hexdigest()
(run / "SUCCESS").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

# The terminal marker is intentionally the final remote write.
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null

say "verifying exact terminal object set and generation/CRC bindings"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

import google_crc32c
from google.cloud import storage

run = Path(sys.argv[1]); remote = sys.argv[2]
ledger = json.loads((run / "remote_objects.json").read_text())
success_path = run / "SUCCESS"; success = json.loads(success_path.read_text())
canonical = lambda value: json.dumps(value, allow_nan=False, ensure_ascii=True,
    separators=(",", ":"), sort_keys=True).encode()
without_success = {key: value for key, value in success.items() if key != "success_sha256"}
if success["success_sha256"] != sha256(canonical(without_success)).hexdigest():
    raise SystemExit("terminal PP16 SUCCESS self-hash drifted")
bucket_name, prefix = remote[5:].split("/", 1); prefix = prefix.rstrip("/") + "/"
blobs = {blob.name.removeprefix(prefix): blob for blob in
         storage.Client().list_blobs(bucket_name, prefix=prefix)}
expected = {item["name"] for item in ledger["objects"]} | {"remote_objects.json", "SUCCESS"}
if set(blobs) != expected:
    raise SystemExit("terminal remote PP16 object set drifted")
for item in ledger["objects"]:
    blob = blobs[item["name"]]
    if (int(blob.size) != item["size"] or blob.crc32c != item["crc32c"] or
            int(blob.generation) != item["generation"]):
        raise SystemExit(f'terminal PP16 payload drifted: {item["name"]}')
for name, path in (("remote_objects.json", run / "remote_objects.json"),
                   ("SUCCESS", success_path)):
    checksum = google_crc32c.Checksum(path.read_bytes())
    crc = base64.b64encode(checksum.digest()).decode("ascii")
    blob = blobs[name]
    if int(blob.size) != path.stat().st_size or blob.crc32c != crc or not blob.generation:
        raise SystemExit(f"terminal PP16 object identity drifted: {name}")
PY

say "COMPLETE layout=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["layout_manifest_sha256"])' "$ARTIFACT_DIR/summary.json")"
say "Metadata-only plan evidence: no TPU, checkpoint payload, DB row, or performance claim."
