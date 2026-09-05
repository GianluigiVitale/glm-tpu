#!/usr/bin/env bash
# Protected eight-host STREAMING pack of the sealed WS32 final-owner checkpoint into host tmpfs.
#
# The sealed checkpoint greenfield_ws32_runtime_pack_20260815T214050854386790Z (manifest self-hash
# c04f800e..., 786,172,488,192 payload bytes) was deleted from GCS on 2026-08-27. Its manifest and
# SUCCESS survive verbatim in the run lineage. This wrapper re-derives each host's four slots from
# the canonical checkpoint into /dev/shm, reuses the sealed manifest and SUCCESS byte-for-byte, and
# proves byte identity of every slot against the sealed manifest before declaring success. Nothing
# is written to the approved bucket except small host records. Storage stays under the 2 TB ceiling.
set -euo pipefail
[[ ${GLM_GREENFIELD_WS32_SHM_PACK:-0} == 1 ]] || {
  echo "WS32 shm pack is default-off; set GLM_GREENFIELD_WS32_SHM_PACK=1" >&2
  exit 2
}
readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly SOURCE_URI=$APPROVED_BUCKET/models/GLM-5.2-FP8
readonly INVENTORY=/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP8_LP4/greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/source_inventory.json
readonly INVENTORY_SHA=a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4
readonly TOPOLOGY_ROOT=/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly TOPOLOGY_FLEET_HASH=4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301
readonly MESH_HASH=de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88
readonly SEALED_TAG=greenfield_ws32_runtime_pack_20260815T214050854386790Z
readonly SEALED_LINEAGE=/home/gianl/gcs-models/results/$SEALED_TAG
readonly SEALED_MANIFEST=$SEALED_LINEAGE/checkpoint_manifest/manifest.json
readonly SEALED_SUCCESS=$SEALED_LINEAGE/checkpoint_SUCCESS.json
readonly SEALED_MANIFEST_FILE_SHA=88df414301e0d303506163c078484ec14cdb15ff03972789df58960b091a7cbf
readonly SEALED_MANIFEST_SELF_SHA=c04f800edf15651198ab2c5183fff8a8ae9a427609b59cc61ccabdab32f5ee08
readonly SEALED_SUCCESS_FILE_SHA=12703932637a9330f97ae0282b26dd7105d68793b3ae6620e109a10c7b0c7ca2
readonly SEALED_SUCCESS_SELF_SHA=1bfea5bd2dd8b096a3e551d7f96697496d8277bdaaa328ca1c35144feb8f1760
readonly SHM_ROOT=/dev/shm/glm-ws32-runtime/$SEALED_TAG
readonly REQUIRED_SHM_BYTES=100500000000
PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_SHM_PACK_TAG:-greenfield_ws32_runtime_shm_pack_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_ws32_runtime_shm_pack_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid WS32 shm pack tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX
[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || { echo "refusing shm pack from a dirty worktree" >&2; exit 2; }
[[ -r $INVENTORY && -r $SOURCE_ROOT/model.safetensors.index.json ]]
[[ $(sha256sum "$SEALED_MANIFEST" | cut -d' ' -f1) == "$SEALED_MANIFEST_FILE_SHA" ]] || { echo "sealed manifest lineage drifted" >&2; exit 2; }
[[ $(sha256sum "$SEALED_SUCCESS" | cut -d' ' -f1) == "$SEALED_SUCCESS_FILE_SHA" ]] || { echo "sealed SUCCESS lineage drifted" >&2; exit 2; }
/home/gianl/vllm-env/bin/python - "$SEALED_MANIFEST" "$SEALED_SUCCESS" "$SEALED_MANIFEST_SELF_SHA" "$SEALED_SUCCESS_SELF_SHA" "$MESH_HASH" <<'PY'
import hashlib,json,sys
manifest=json.load(open(sys.argv[1])); success=json.load(open(sys.argv[2]))
canonical=lambda v: json.dumps(v,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode()
if manifest["manifest_sha256"]!=sys.argv[3] or success["manifest_sha256"]!=sys.argv[3] or success["success_sha256"]!=sys.argv[4]: raise SystemExit("sealed identity pins drifted")
if hashlib.sha256(canonical({k:v for k,v in success.items() if k!="success_sha256"})).hexdigest()!=sys.argv[4]: raise SystemExit("sealed SUCCESS self-hash drifted")
if manifest["mesh_hash"]!=sys.argv[5] or len(manifest["files"])!=32: raise SystemExit("sealed manifest mesh/files drifted")
PY
[[ ! -e $RUN_DIR ]] || { echo "append-only run directory exists" >&2; exit 2; }
mkdir -p "$RUN_DIR"
say() { echo "[ws32-shm-pack $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }
has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}
strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local command
  # shellcheck disable=SC2016
  command='tools=1; command -v pgrep >/dev/null || tools=0; command -v fuser >/dev/null || tools=0; sudo -n true >/dev/null 2>&1 || tools=0; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[r]un_short_decoder_ws32[.]py|[p]ack_ws32_runtime_checkpoint[.]py|[c]ompile_short_decoder[.]py|[m]icrobench_collectives[.]py" || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [ "$tools" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || { say "ABORT: another protected workflow holds the fleet lease"; exit 1; }
post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then strict_census failure_exit || true; fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving nonterminal evidence"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT
say "RUN_DIR=$RUN_DIR PIN=$PIN SHM_ROOT=$SHM_ROOT sealed_manifest=$SEALED_MANIFEST_SELF_SHA"
if ! gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' >"$RUN_DIR/vacancy_results.txt"; then
  say "ABORT: remote vacancy listing failed"; exit 1
fi
[[ ! -s $RUN_DIR/vacancy_results.txt ]] || { say "ABORT: results prefix is not vacant"; exit 2; }
strict_census pre || { say "ABORT: pre-pack fleet is not authenticated zero-work"; exit 1; }

# Every host: exact code, sealed inputs readable on the mount, tmpfs capacity, target absent.
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; wt='"$WORKTREE"'; branch='"$BRANCH"'; shm='"$SHM_ROOT"'; required='"$REQUIRED_SHM_BYTES"'; manifest='"$SEALED_MANIFEST"'; success='"$SEALED_SUCCESS"'; manifest_sha='"$SEALED_MANIFEST_FILE_SHA"'; success_sha='"$SEALED_SUCCESS_FILE_SHA"'; inventory='"$INVENTORY"'; source_root='"$SOURCE_ROOT"'; topology='"$TOPOLOGY_ROOT"'/topology.rank${idx}.json; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" && $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; else [[ -e "$wt/.git" && -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; for path in "$manifest" "$success" "$inventory" "$source_root/model.safetensors.index.json" "$topology"; do [[ -r $path ]]; done; [[ $(sha256sum "$manifest" | cut -d" " -f1) == "$manifest_sha" && $(sha256sum "$success" | cut -d" " -f1) == "$success_sha" ]]; findmnt -T /dev/shm -n -o FSTYPE | grep -qx tmpfs; [[ ! -e $shm ]]; available=$(df -B1 --output=avail /dev/shm | tail -1); [[ $available -ge $required ]] || { echo "insufficient /dev/shm: $available" >&2; exit 1; }; echo "SYNC_OK $(hostname) $pin"'
sync_rc=0
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1 || sync_rc=$?
if [[ $sync_rc -ne 0 ]] || ! has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK; then
  say "ABORT: exact eight-host sync failed"; exit 1
fi

say "packing each host's four owned slots into tmpfs and proving identity against the sealed manifest"
# shellcheck disable=SC2016
host_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; source_root='"$SOURCE_ROOT"'; source_uri='"$SOURCE_URI"'; inventory='"$INVENTORY"'; inventory_sha='"$INVENTORY_SHA"'; topology_root='"$TOPOLOGY_ROOT"'; topology_sha='"$TOPOLOGY_HASH"'; mesh_sha='"$MESH_HASH"'; pin='"$PIN"'; shm='"$SHM_ROOT"'; manifest='"$SEALED_MANIFEST"'; success='"$SEALED_SUCCESS"'; remote='"$REMOTE_PREFIX"'; capture="$topology_root/topology.rank${idx}.json"; run=/home/gianl/glm-run/$tag/host_pack; mkdir -p "$run"; [[ ! -e $shm && ! -e $shm.identity.json ]]; mkdir -p "$(dirname "$shm")"; mkdir "$shm"; fail(){ trap - ERR; rm -rf -- "$shm" "$shm.identity.json"; echo "SHM_PACK_FAILED $(hostname)" >&2; exit 1; }; trap fail ERR; cd "$wt"; read -r slots mesh observed_inventory observed_topology < <(PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python - "$capture" "$inventory" "$idx" <<'"'"'PY'"'"'
import json,socket,sys
from pathlib import Path
from glm_tpu.greenfield.partitioning.source_inventory import inspect_source_inventory
from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh
from glm_tpu.greenfield.types import PhysicalTopology
x=json.loads(Path(sys.argv[1]).read_text()); rank=int(sys.argv[3])
if x.get("launch_process_id") != rank or x.get("hostname") != socket.gethostname(): raise SystemExit("topology capture/launch host identity drifted")
topology=PhysicalTopology.from_dict(x["contract"]["topology"]); mesh=build_ws32_physical_mesh(topology); slots=tuple(mesh.flattened_device_ids.index(int(device_id)) for device_id in x["local_device_ids"]); print(",".join(map(str,slots)),mesh.mesh_hash,inspect_source_inventory(Path(sys.argv[2])).inventory_sha256,x["contract"]["topology_hash"])
PY
) || fail; [[ "$mesh" == "$mesh_sha" && "$observed_inventory" == "$inventory_sha" && "$observed_topology" == "$topology_sha" ]] || { echo "host source/topology/mesh identity drifted" >&2; fail; }; slot_args=${slots//,/ }; PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python scripts/greenfield/pack_ws32_runtime_checkpoint.py pack-slots --source-inventory "$inventory" --source-root "$source_root" --source-uri "$source_uri" --output "$shm/pack" --expected-code-hash "$pin" --mesh-hash "$mesh_sha" --slots $slot_args >"$run/pack.log" 2>&1 || fail; cp "$shm/pack/slot_records.json" "$run/slot_records.json"; for file in "$shm"/pack/device_slot_*.safetensors; do mv "$file" "$shm/$(basename "$file")"; done; rm -rf -- "$shm/pack"; cp "$manifest" "$shm/manifest.json"; cp "$success" "$shm/SUCCESS"; chmod 0444 "$shm/manifest.json" "$shm/SUCCESS"; PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python - "$shm" "$run/slot_records.json" "$run/identity.json" "$slots" <<'"'"'PY'"'"'
import hashlib,json,os,socket,sys
from pathlib import Path
root=Path(sys.argv[1]); records=json.loads(Path(sys.argv[2]).read_text()); slots=[int(s) for s in sys.argv[4].split(",")]
manifest=json.loads((root/"manifest.json").read_text()); sealed={f["device_slot"]:f for f in manifest["files"]}
present=sorted(p.name for p in root.iterdir())
expected=sorted([f"device_slot_{s:02d}.safetensors" for s in slots]+["manifest.json","SUCCESS"])
if present!=expected: raise SystemExit(f"tmpfs checkpoint membership drifted: {present}")
out=[]
for slot in slots:
    entry=sealed[slot]; path=root/entry["filename"]; h=hashlib.sha256()
    with open(path,"rb") as fh:
        for block in iter(lambda: fh.read(64<<20), b""): h.update(block)
    digest=h.hexdigest(); size=path.stat().st_size
    if size!=entry["file_bytes"] or digest!=entry["sha256"]: raise SystemExit(f"slot {slot} is NOT byte-identical to the sealed checkpoint: {digest} vs {entry['"'"'sha256'"'"']}")
    os.chmod(path,0o444); out.append({"device_slot":slot,"filename":entry["filename"],"file_bytes":size,"sha256":digest})
record=json.dumps({"all_slots_identical":True,"hostname":socket.gethostname(),"manifest_sha256":manifest["manifest_sha256"],"slots":out,"tmpfs_root":str(root)},indent=2,sort_keys=True)+"\n"
Path(sys.argv[3]).write_text(record); marker=Path(str(root)+".identity.json"); marker.write_text(record); marker.chmod(0o444)
PY
[[ -f $shm.identity.json ]] || fail; trap - ERR; if ! gcloud storage cp --recursive --no-clobber "$run" "$remote/host_records/worker${idx}/" >/dev/null; then echo "SHM_PACK_RECORD_UPLOAD_FAILED $(hostname): verified tmpfs root retained; rerun cleanup before repacking" >&2; exit 1; fi; echo "SHM_PACK_OK $(hostname) $slots"'
pack_rc=0
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all --command="$host_command" >"$RUN_DIR/pack.txt" 2>&1 || pack_rc=$?
if [[ $pack_rc -ne 0 ]] || ! has_eight_unique_markers "$RUN_DIR/pack.txt" SHM_PACK_OK; then
  say "ABORT: distributed tmpfs pack did not complete 8/8 with sealed identity"; exit 1
fi
say "recovering host identity records and reconciling all 32 slots against the sealed manifest"
gcloud storage cp --recursive "$REMOTE_PREFIX/host_records" "$RUN_DIR/" >/dev/null
/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$SEALED_MANIFEST" "$RUN_DIR/identity_reconciliation.json" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]); manifest=json.load(open(sys.argv[2])); sealed={f["device_slot"]:f for f in manifest["files"]}
seen={}
for path in sorted(run.glob("host_records/worker*/host_pack/identity.json")):
    record=json.loads(path.read_text())
    for slot in record["slots"]:
        if slot["device_slot"] in seen or slot["sha256"]!=sealed[slot["device_slot"]]["sha256"] or slot["file_bytes"]!=sealed[slot["device_slot"]]["file_bytes"]: raise SystemExit(f"slot reconciliation failed: {slot}")
        seen[slot["device_slot"]]=record["hostname"]
if sorted(seen)!=list(range(32)): raise SystemExit(f"slot coverage drifted: {sorted(seen)}")
Path(sys.argv[3]).write_text(json.dumps({"manifest_sha256":manifest["manifest_sha256"],"slot_hosts":seen,"slots_identical_to_sealed":32},indent=2,sort_keys=True)+"\n")
PY
# Controller copy of manifest/SUCCESS at the same path serves the run wrapper's controller preflight.
[[ ! -e $SHM_ROOT ]] || { say "ABORT: controller tmpfs path exists"; exit 2; }
mkdir -p "$SHM_ROOT" && cp "$SEALED_MANIFEST" "$SHM_ROOT/manifest.json" && cp "$SEALED_SUCCESS" "$SHM_ROOT/SUCCESS"
strict_census post || { say "ABORT: post-pack fleet is not authenticated zero-work"; exit 1; }
post_census_done=1
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/" >/dev/null
say "SHM_PACK_COMPLETE root=$SHM_ROOT manifest=$SEALED_MANIFEST_SELF_SHA success=$SEALED_SUCCESS_SELF_SHA slots=32 identical"
