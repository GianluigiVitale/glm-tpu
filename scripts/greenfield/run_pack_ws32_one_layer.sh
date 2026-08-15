#!/usr/bin/env bash
# Protected offline WS32 layer-3 derivative. This script never initializes JAX/TPU.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_TAG=greenfield_one_layer_pack_20260805T151828912346032Z
readonly SOURCE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/layer3/PP8_LP4/$SOURCE_TAG
readonly SOURCE_PAYLOAD=$SOURCE_ROOT/packed
readonly SOURCE_URI=$APPROVED_BUCKET/checkpoints/greenfield/glm52/layer3/PP8_LP4/$SOURCE_TAG
readonly SOURCE_MANIFEST_SHA256=68ef82011892456409a194f6fa31697dd1e31d96fe1a3f0069228288f613f938
readonly SOURCE_PACKED_PAYLOAD_BYTES=9716380672
readonly SOURCE_UNIQUE_PAYLOAD_BYTES=9706940416
readonly EXPECTED_PACKED_BYTES=9971249152
readonly TOPOLOGY_CAPTURE=/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z/topology.rank0.json
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly WS32_MESH_HASH=de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_ONE_LAYER_PACK_TAG:-greenfield_ws32_one_layer_pack_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_ws32_one_layer_pack_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid WS32 pack tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/checkpoints/greenfield/glm52/layer3/WS32_2D/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing WS32 pack outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing WS32 pack from a dirty greenfield worktree" >&2
  exit 2
}
[[ -r $SOURCE_ROOT/manifest.json && -r $SOURCE_ROOT/SUCCESS ]] || {
  echo "sealed PP8 one-layer source is unavailable" >&2
  exit 2
}
[[ -r $TOPOLOGY_CAPTURE ]] || {
  echo "protected topology capture is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only WS32 run directory exists: $RUN_DIR" >&2
  exit 2
}
active_account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)')
[[ -n $active_account ]] || {
  echo "gcloud has no active authenticated account" >&2
  exit 2
}
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}
remote_existing=$(gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)')
[[ -z $remote_existing ]] || {
  echo "refusing non-vacant WS32 remote prefix: $REMOTE_PREFIX" >&2
  exit 2
}
available_tmp_bytes=$(df -B1 --output=avail /dev/shm | tail -n 1)
required_tmp_bytes=$(( EXPECTED_PACKED_BYTES + 2147483648 ))
[[ $available_tmp_bytes -ge $required_tmp_bytes ]] || {
  echo "insufficient /dev/shm for bounded WS32 pack: available=$available_tmp_bytes required=$required_tmp_bytes" >&2
  exit 2
}

mkdir "$RUN_DIR"
TMP_ROOT=$(mktemp -d "/dev/shm/${TAG}.XXXXXXXX")
PACK_DIR=$TMP_ROOT/packed
readonly TMP_ROOT PACK_DIR

say() {
  echo "[ws32-one-layer-pack $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

terminal_done=0
success_uploaded=0
cleanup_tmp() {
  if [[ -n ${TMP_ROOT:-} && -d $TMP_ROOT && $TMP_ROOT == /dev/shm/${TAG}.* ]]; then
    rm -rf -- "$TMP_ROOT"
  fi
}
on_exit() {
  local status=$?
  if [[ $status -ne 0 && $terminal_done -eq 0 && $success_uploaded -eq 0 ]]; then
    say "FAILED status=$status; temporary payload will be removed"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
  cleanup_tmp
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN"
say "SOURCE_MANIFEST_SHA256=$SOURCE_MANIFEST_SHA256"
say "WS32_MESH_HASH=$WS32_MESH_HASH"
say "REMOTE_PREFIX=$REMOTE_PREFIX"
say "preflighting sealed source and physical 8x4 ownership"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$SOURCE_ROOT/manifest.json" "$TOPOLOGY_CAPTURE" \
  "$SOURCE_MANIFEST_SHA256" "$SOURCE_PACKED_PAYLOAD_BYTES" \
  "$SOURCE_UNIQUE_PAYLOAD_BYTES" \
  "$TOPOLOGY_HASH" "$WS32_MESH_HASH" <<'PY' >"$RUN_DIR/source_preflight.json"
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.checkpoint.ws32_one_layer import _manifest_hash
from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh
from glm_tpu.greenfield.types import PhysicalTopology

(
    manifest_path,
    topology_path,
    manifest_sha,
    packed_payload_bytes,
    unique_payload_bytes,
    topology_sha,
    mesh_sha,
) = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
if manifest.get("manifest_sha256") != _manifest_hash(manifest):
    raise SystemExit("sealed PP8 source manifest checksum mismatch")
if manifest.get("manifest_sha256") != manifest_sha:
    raise SystemExit("sealed PP8 source manifest identity drifted")
if manifest.get("packed_payload_byte_count") != int(packed_payload_bytes):
    raise SystemExit("sealed PP8 source packed payload bytes drifted")
if manifest.get("source_payload_byte_count") != int(unique_payload_bytes):
    raise SystemExit("sealed PP8 source unique payload bytes drifted")
topology_record = json.loads(Path(topology_path).read_text())
topology = PhysicalTopology.from_dict(topology_record["contract"]["topology"])
if topology.topology_hash != topology_sha:
    raise SystemExit("protected physical topology identity drifted")
mesh = build_ws32_physical_mesh(topology)
if mesh.mesh_hash != mesh_sha:
    raise SystemExit("WS32 physical mesh identity drifted")
print(json.dumps({
    "source_files": [{
        "device_slot": item["device_slot"],
        "file_byte_count": item["file_byte_count"],
        "filename": item["filename"],
        "sha256": item["sha256"],
    } for item in manifest["files"]],
    "source_manifest_sha256": manifest_sha,
    "source_packed_payload_byte_count": int(packed_payload_bytes),
    "source_unique_payload_byte_count": int(unique_payload_bytes),
    "topology_capture_sha256": sha256(Path(topology_path).read_bytes()).hexdigest(),
    "topology_hash": topology.topology_hash,
    "ws32_mesh": mesh.to_dict(),
    "ws32_mesh_hash": mesh.mesh_hash,
}, indent=2, sort_keys=True))
PY

say "packing 32 exact final owners from the sealed PP8 derivative; TPU forbidden"
started=$(date +%s)
(
  cd "$WORKTREE"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    scripts/greenfield/pack_ws32_one_layer.py \
    --source-manifest "$SOURCE_ROOT/manifest.json" \
    --source-payload-dir "$SOURCE_PAYLOAD" \
    --source-artifact-uri "$SOURCE_URI" \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --output "$PACK_DIR" \
    --expected-code-hash "$PIN" \
    --mesh-hash "$WS32_MESH_HASH"
) >"$RUN_DIR/pack.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "pack and full tensor verification completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$PACK_DIR" "$EXPECTED_PACKED_BYTES" "$PIN" \
  "$SOURCE_MANIFEST_SHA256" "$WS32_MESH_HASH" <<'PY' >"$RUN_DIR/inspection.json"
from __future__ import annotations

import gc
from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.checkpoint.ws32_one_layer import (
    inspect_ws32_one_layer,
    load_ws32_one_layer_slot,
)

root, expected_bytes, code_hash, source_sha, mesh_hash = sys.argv[1:]
root = Path(root)
manifest = inspect_ws32_one_layer(root, verify_tensor_hashes=True)
expected = {
    "code_hash": code_hash,
    "files": 32,
    "mesh_hash": mesh_hash,
    "packed_payload_byte_count": int(expected_bytes),
    "source_manifest_sha256": source_sha,
}
observed = {
    "code_hash": manifest["code_hash"],
    "files": len(manifest["files"]),
    "mesh_hash": manifest["mesh_hash"],
    "packed_payload_byte_count": manifest["packed_payload_byte_count"],
    "source_manifest_sha256": manifest["source"]["manifest_sha256"],
}
if observed != expected:
    raise SystemExit(f"WS32 packed identity drifted: {observed} != {expected}")
loads = []
for slot in (0, 31):
    loaded = load_ws32_one_layer_slot(
        root,
        expected_manifest_sha256=manifest["manifest_sha256"],
        device_slot=slot,
    )
    loads.append({
        "device_slot": slot,
        "expert_coordinate": loaded.expert_coordinate,
        "feature_coordinate": loaded.feature_coordinate,
        "file_sha256": loaded.file_sha256,
        "tensor_count": len(loaded.arrays),
        "tensor_schema_sha256": sha256(json.dumps({
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(loaded.arrays.items())
        }, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
    })
    del loaded
    gc.collect()
print(json.dumps({
    **observed,
    "direct_loads": loads,
    "manifest_sha256": manifest["manifest_sha256"],
    "performance_claim": False,
    "tpu_initialized": False,
}, indent=2, sort_keys=True))
PY

say "packing evidence is complete; starting immutable approved-bucket upload"
(
  cd "$RUN_DIR"
  sha256sum source_preflight.json pack.log inspection.json orchestrator.log
  /home/gianl/vllm-env/bin/python - "$PACK_DIR/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
print(f"{manifest['manifest_sha256']}  packed/manifest.json.canonical")
for record in manifest["files"]:
    print(f"{record['sha256']}  packed/{record['filename']}")
PY
) >"$RUN_DIR/evidence.sha256"

gcloud storage cp --no-clobber \
  "$RUN_DIR/source_preflight.json" "$RUN_DIR/pack.log" \
  "$RUN_DIR/inspection.json" "$RUN_DIR/orchestrator.log" \
  "$RUN_DIR/evidence.sha256" "$PACK_DIR/manifest.json" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber "$PACK_DIR"/device_slot_*.safetensors \
  "$REMOTE_PREFIX/packed/" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PACK_DIR" "$REMOTE_PREFIX" <<'PY' >"$RUN_DIR/remote_objects.json"
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

run_dir, pack_dir, remote = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
manifest = json.loads((pack_dir / "manifest.json").read_text())
objects = [
    ("source_preflight.json", run_dir / "source_preflight.json"),
    ("pack.log", run_dir / "pack.log"),
    ("inspection.json", run_dir / "inspection.json"),
    ("orchestrator.log", run_dir / "orchestrator.log"),
    ("evidence.sha256", run_dir / "evidence.sha256"),
    ("manifest.json", pack_dir / "manifest.json"),
] + [
    (f"packed/{item['filename']}", pack_dir / item["filename"])
    for item in manifest["files"]
]

def crc32c(path: Path) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode("ascii")

records = []
for name, path in objects:
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{remote}/{name}", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    local_crc = crc32c(path)
    if int(value["size"]) != path.stat().st_size or value.get("crc32c_hash") != local_crc:
        raise SystemExit(f"remote WS32 object identity drifted for {name}")
    if not value.get("generation"):
        raise SystemExit(f"remote WS32 object has no generation for {name}")
    records.append({
        "crc32c": local_crc,
        "generation": value["generation"],
        "name": name,
        "size": path.stat().st_size,
    })
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY

gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PACK_DIR/manifest.json" "$REMOTE_PREFIX" "$PIN" "$TAG" <<'PY'
from __future__ import annotations

import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

run_dir, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
remote, code_hash, tag = sys.argv[3:]
ledger_path = run_dir / "remote_objects.json"
checksum = google_crc32c.Checksum(ledger_path.read_bytes())
ledger_crc = base64.b64encode(checksum.digest()).decode("ascii")
remote_ledger = json.loads(subprocess.run(
    ["gcloud", "storage", "objects", "describe", f"{remote}/remote_objects.json", "--format=json"],
    check=True, capture_output=True, text=True,
).stdout)
if int(remote_ledger["size"]) != ledger_path.stat().st_size or remote_ledger.get("crc32c_hash") != ledger_crc:
    raise SystemExit("remote WS32 ledger identity drifted")
manifest = json.loads(manifest_path.read_text())
expected_nonterminal = {
    "remote_objects.json", "source_preflight.json", "pack.log",
    "inspection.json", "orchestrator.log", "evidence.sha256", "manifest.json",
} | {f"packed/device_slot_{slot:02d}.safetensors" for slot in range(32)}
listing = subprocess.run(
    ["gcloud", "storage", "objects", "list", f"{remote}/**", "--format=value(name)"],
    check=True, capture_output=True, text=True,
).stdout.splitlines()
bucket_prefix = remote.removeprefix("gs://").split("/", 1)[1].rstrip("/") + "/"
observed_nonterminal = {
    line.removeprefix(bucket_prefix)
    for line in listing
    if line.startswith(bucket_prefix)
}
if observed_nonterminal != expected_nonterminal:
    raise SystemExit(
        "remote WS32 nonterminal object set drifted: "
        f"{sorted(observed_nonterminal ^ expected_nonterminal)}"
    )
success = {
    "artifact_kind": "greenfield_ws32_one_layer_pack_success",
    "code_hash": code_hash,
    "format_version": 1,
    "manifest_sha256": manifest["manifest_sha256"],
    "nonterminal_object_count": len(observed_nonterminal),
    "performance_claim": False,
    "remote_objects_crc32c": ledger_crc,
    "remote_objects_generation": remote_ledger["generation"],
    "remote_objects_sha256": sha256(ledger_path.read_bytes()).hexdigest(),
    "source_manifest_sha256": manifest["source"]["manifest_sha256"],
    "tag": tag,
    "tpu_initialized": False,
    "ws32_mesh_hash": manifest["mesh_hash"],
}
(run_dir / "SUCCESS").write_text(json.dumps(success, indent=2, sort_keys=True) + "\n")
PY

gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
success_uploaded=1

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX" <<'PY'
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

success_path, remote = Path(sys.argv[1]), sys.argv[2]
checksum = google_crc32c.Checksum(success_path.read_bytes())
local_crc = base64.b64encode(checksum.digest()).decode("ascii")
value = json.loads(subprocess.run(
    ["gcloud", "storage", "objects", "describe", f"{remote}/SUCCESS", "--format=json"],
    check=True, capture_output=True, text=True,
).stdout)
if int(value["size"]) != success_path.stat().st_size or value.get("crc32c_hash") != local_crc:
    raise SystemExit("remote WS32 SUCCESS identity drifted")
success = json.loads(success_path.read_text())
if success.get("nonterminal_object_count") != 39:
    raise SystemExit("WS32 SUCCESS nonterminal object count drifted")
print(json.dumps({
    "remote_success_crc32c": local_crc,
    "remote_success_generation": value["generation"],
    "terminal_object_count": success["nonterminal_object_count"] + 1,
}, sort_keys=True))
PY

terminal_done=1
echo "SUCCESS tag=$TAG code=$PIN; protected WS32 checkpoint evidence only, no TPU/performance claim"
