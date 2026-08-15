#!/usr/bin/env bash
# Protected eight-host offline pack of the complete WS32 final-owner checkpoint.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly SOURCE_URI=$APPROVED_BUCKET/models/GLM-5.2-FP8
readonly INVENTORY=/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP8_LP4/greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/source_inventory.json
readonly INVENTORY_SHA=a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4
readonly TOPOLOGY_RUN=/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly MESH_HASH=de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88
readonly EXPECTED_PAYLOAD_BYTES=786172488192
readonly EXPECTED_SLOT_BYTES=24567890256
readonly EXPECTED_TENSORS_PER_SLOT=2310
readonly EXPECTED_SOURCE_FILES=141

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_RUNTIME_PACK_TAG:-greenfield_ws32_runtime_pack_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_ws32_runtime_pack_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid WS32 runtime tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
CHECKPOINT_URI=$APPROVED_BUCKET/checkpoints/greenfield/glm52/runtime/WS32_2D/$TAG
CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime/WS32_2D/$TAG
MANIFEST_ROOT=$RUN_DIR/checkpoint_manifest
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR CHECKPOINT_URI CHECKPOINT_ROOT MANIFEST_ROOT REMOTE_PREFIX

say() {
  echo "[ws32-runtime-pack $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[p]ack_ws32_runtime_checkpoint[.]py|[r]un_short_decode[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing WS32 pack outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing WS32 pack from a dirty worktree" >&2
  exit 2
}
[[ -r $INVENTORY && -r $SOURCE_ROOT/model.safetensors.index.json ]] || {
  echo "WS32 source inventory/checkpoint is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR && ! -e $CHECKPOINT_ROOT ]] || {
  echo "append-only WS32 runtime destination exists" >&2
  exit 2
}
mkdir "$RUN_DIR"

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
}

post_census_done=0
results_success_uploaded=0
cleanup_manifest_links() {
  local slot path
  for slot in {00..31}; do
    path="$MANIFEST_ROOT/device_slot_${slot}.safetensors"
    [[ ! -L $path ]] || unlink "$path"
  done
}
on_exit() {
  local status=$?
  cleanup_manifest_links
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 && $results_success_uploaded -eq 0 ]]; then
    say "FAILED status=$status; preserving nonterminal evidence"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN"
say "SOURCE_INVENTORY_SHA=$INVENTORY_SHA"
say "MESH_HASH=$MESH_HASH"
say "CHECKPOINT_URI=$CHECKPOINT_URI"

prefixes=("$CHECKPOINT_URI" "$REMOTE_PREFIX")
listings=("$RUN_DIR/vacancy_checkpoint.txt" "$RUN_DIR/vacancy_results.txt")
for index in 0 1; do
  prefix=${prefixes[$index]}
  listing=${listings[$index]}
  if ! gcloud storage objects list "$prefix/**" --format='value(name)' >"$listing"; then
    say "ABORT: remote vacancy listing failed for $prefix"
    exit 1
  fi
  [[ ! -s $listing ]] || {
    say "ABORT: remote prefix is not vacant: $prefix"
    exit 2
  }
done

strict_census pre || {
  say "ABORT: pre-pack fleet is not authenticated zero-work"
  exit 1
}

say "syncing exact reviewed code and immutable source/topology records"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; wt='"$WORKTREE"'; source_root='"$SOURCE_ROOT"'; inventory='"$INVENTORY"'; topology='"$TOPOLOGY_RUN"'; idx=${HOSTNAME##*-w-}; if [[ "$idx" == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin '"$BRANCH"'; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository worktree" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch '"$BRANCH"' "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; capture="$topology/host_records/topology.rank${idx}.json"; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$inventory" ]] && [[ -r "$source_root/model.safetensors.index.json" ]] && [[ -r "$capture" ]] && findmnt -T "$source_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host sync failed"
  exit 1
}

say "packing 32 slots and hashing 141 sources across eight hosts"
# shellcheck disable=SC2016
host_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; source_root='"$SOURCE_ROOT"'; source_uri='"$SOURCE_URI"'; inventory='"$INVENTORY"'; topology='"$TOPOLOGY_RUN"'; topology_sha='"$TOPOLOGY_HASH"'; mesh_sha='"$MESH_HASH"'; inventory_sha='"$INVENTORY_SHA"'; pin='"$PIN"'; checkpoint='"$CHECKPOINT_URI"'; remote='"$REMOTE_PREFIX"'; capture="$topology/host_records/topology.rank${idx}.json"; run=/home/gianl/glm-run/$tag/host_pack; mkdir -p "$run"; tmp=$(mktemp -d "/dev/shm/${tag}.worker${idx}.XXXXXXXX"); cleanup(){ if [[ -n ${tmp:-} && -d $tmp && $tmp == /dev/shm/${tag}.worker${idx}.* ]]; then rm -rf -- "$tmp"; fi; }; trap cleanup EXIT; available=$(df -B1 --output=avail /dev/shm | tail -1); required=100500000000; [[ $available -ge $required ]] || { echo "insufficient /dev/shm: $available" >&2; exit 1; }; cd "$wt"; read -r slots mesh observed_inventory observed_topology < <(PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python - "$capture" "$inventory" <<'"'"'PY'"'"'
import json,sys
from pathlib import Path
from glm_tpu.greenfield.partitioning.source_inventory import inspect_source_inventory
from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh
from glm_tpu.greenfield.types import PhysicalTopology
x=json.loads(Path(sys.argv[1]).read_text()); topology=PhysicalTopology.from_dict(x["contract"]["topology"]); mesh=build_ws32_physical_mesh(topology); slots=tuple(mesh.flattened_device_ids.index(int(device_id)) for device_id in x["local_device_ids"]); print(",".join(map(str,slots)),mesh.mesh_hash,inspect_source_inventory(Path(sys.argv[2])).inventory_sha256,x["contract"]["topology_hash"])
PY
); [[ "$mesh" == "$mesh_sha" && "$observed_inventory" == "$inventory_sha" && "$observed_topology" == "$topology_sha" ]] || { echo "host source/topology/mesh identity drifted" >&2; exit 1; }; slot_args=${slots//,/ }; PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python scripts/greenfield/pack_ws32_runtime_checkpoint.py hash-sources --source-inventory "$inventory" --source-root "$source_root" --output "$run/source_hashes.json" --expected-code-hash "$pin" --shard-index "$idx" --shard-count 8 >"$run/source_hashes.log" 2>&1 & hash_pid=$!; PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python scripts/greenfield/pack_ws32_runtime_checkpoint.py pack-slots --source-inventory "$inventory" --source-root "$source_root" --source-uri "$source_uri" --output "$tmp/packed" --expected-code-hash "$pin" --mesh-hash "$mesh_sha" --slots $slot_args >"$run/pack.log" 2>&1; wait "$hash_pid"; cp "$tmp/packed/slot_records.json" "$run/slot_records.json"; for file in "$tmp"/packed/device_slot_*.safetensors; do gcloud storage cp --no-clobber "$file" "$checkpoint/$(basename "$file")" >/dev/null; done; PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python - "$run/slot_records.json" "$checkpoint" "$run/uploaded_slots.json" <<'"'"'PY'"'"'
import json,sys
from pathlib import Path
from google.cloud import storage
records=json.loads(Path(sys.argv[1]).read_text()); bucket_name,prefix=sys.argv[2][5:].split("/",1); bucket=storage.Client().bucket(bucket_name); out=[]
for record in records["files"]:
 blob=bucket.blob(f"{prefix}/{record['"'"'filename'"'"']}"); blob.reload(timeout=300)
 if int(blob.size)!=record["file_bytes"] or blob.crc32c!=record["crc32c"]: raise SystemExit(f"remote slot mismatch: {record['"'"'filename'"'"']}")
 out.append({"crc32c":blob.crc32c,"file_bytes":int(blob.size),"filename":record["filename"],"generation":int(blob.generation),"sha256":record["sha256"]})
Path(sys.argv[3]).write_text(json.dumps({"files":out},indent=2,sort_keys=True)+"\n")
PY
gcloud storage cp --recursive --no-clobber "$run" "$remote/host_records/worker${idx}/" >/dev/null; echo "PACK_HOST_OK $(hostname) $slots"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$host_command" >"$RUN_DIR/pack.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/pack.txt" PACK_HOST_OK || {
  say "ABORT: distributed WS32 pack did not complete 8/8"
  exit 1
}

say "recovering exact host records and verifying remote object/source CRCs"
gcloud storage cp --recursive "$REMOTE_PREFIX/host_records" "$RUN_DIR/" >/dev/null
mapfile -t slot_records < <(find "$RUN_DIR/host_records" -name slot_records.json -type f | sort)
mapfile -t source_records < <(find "$RUN_DIR/host_records" -name source_hashes.json -type f | sort)
[[ ${#slot_records[@]} -eq 8 && ${#source_records[@]} -eq 8 ]] || {
  say "ABORT: host record cardinality drifted"
  exit 1
}

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$CHECKPOINT_URI" "$SOURCE_URI" "$RUN_DIR/remote_preflight.json" \
  "${slot_records[@]}" -- "${source_records[@]}" <<'PY'
import json,sys
from pathlib import Path
from google.cloud import storage
checkpoint_uri,source_uri,output=sys.argv[1:4]; separator=sys.argv.index("--"); slot_paths=sys.argv[4:separator]; source_paths=sys.argv[separator+1:]
client=storage.Client()
def split(uri): return uri[5:].split("/",1)
cb,cp=split(checkpoint_uri); sb,sp=split(source_uri); checkpoint=client.bucket(cb); source=client.bucket(sb)
slot_records=[json.loads(Path(path).read_text()) for path in slot_paths]; flat_slots=[item for record in slot_records for item in record["files"]]; expected={item["filename"]:item for item in flat_slots}
if len(flat_slots)!=32 or len(expected)!=32: raise SystemExit("checkpoint host records are incomplete or duplicate")
observed={}
for blob in client.list_blobs(cb,prefix=cp.rstrip("/")+"/"):
 name=blob.name.removeprefix(cp.rstrip("/")+"/")
 if "/" in name or not name: raise SystemExit(f"unexpected nested checkpoint object: {blob.name}")
 observed[name]=blob
if set(observed)!=set(expected): raise SystemExit(f"checkpoint object set drifted: {sorted(observed)}")
checkpoint_records=[]
for name,record in sorted(expected.items()):
 blob=observed[name]
 if int(blob.size)!=record["file_bytes"] or blob.crc32c!=record["crc32c"]: raise SystemExit(f"checkpoint CRC mismatch: {name}")
 checkpoint_records.append({"crc32c":blob.crc32c,"file_bytes":int(blob.size),"filename":name,"generation":int(blob.generation),"sha256":record["sha256"]})
flat_sources=[item for record in (json.loads(Path(path).read_text()) for path in source_paths) for item in record["files"]]; source_expected={item["filename"]:item for item in flat_sources}
if len(flat_sources)!=141 or len(source_expected)!=141: raise SystemExit("source hash records are incomplete or duplicate")
source_records=[]
for name,record in sorted(source_expected.items()):
 blob=source.blob(f"{sp.rstrip('/' )}/{name}"); blob.reload(timeout=300)
 if int(blob.size)!=record["file_bytes"] or blob.crc32c!=record["crc32c"]: raise SystemExit(f"source CRC mismatch: {name}")
 source_records.append({"crc32c":blob.crc32c,"file_bytes":int(blob.size),"filename":name,"generation":int(blob.generation),"sha256":record["sha256"]})
Path(output).write_text(json.dumps({"checkpoint_files":checkpoint_records,"source_files":source_records},indent=2,sort_keys=True)+"\n")
PY

say "finalizing manifest only after 32 owners and 141 sources reconcile"
mkdir "$MANIFEST_ROOT"
for slot in {00..31}; do
  payload="$CHECKPOINT_ROOT/device_slot_${slot}.safetensors"
  [[ -f $payload ]] || {
    say "ABORT: mounted WS32 payload is unavailable: $payload"
    exit 1
  }
  ln -s "$payload" "$MANIFEST_ROOT/device_slot_${slot}.safetensors"
done
(
  cd "$WORKTREE"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    scripts/greenfield/pack_ws32_runtime_checkpoint.py finalize \
    --source-inventory "$INVENTORY" --source-root "$SOURCE_ROOT" \
    --source-uri "$SOURCE_URI" --output "$MANIFEST_ROOT" \
    --expected-code-hash "$PIN" --mesh-hash "$MESH_HASH" \
    --slot-records "${slot_records[@]}" \
    --source-records "${source_records[@]}"
) >"$RUN_DIR/finalize.txt" 2>&1
MANIFEST_SHA=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$MANIFEST_ROOT/manifest.json")
cleanup_manifest_links

strict_census post || {
  say "ABORT: post-pack fleet is not authenticated zero-work"
  exit 1
}
post_census_done=1

say "publishing manifest through the create-only GCS API"
gcloud storage cp --no-clobber "$MANIFEST_ROOT/manifest.json" \
  "$CHECKPOINT_URI/manifest.json" >/dev/null
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$CHECKPOINT_URI" "$MANIFEST_ROOT/manifest.json" \
  "$RUN_DIR/remote_preflight.json" "$RUN_DIR/remote_terminal.json" <<'PY'
import base64,json,sys
from hashlib import sha256
from pathlib import Path
import google_crc32c
from google.cloud import storage
uri,manifest_path,preflight_path,output=sys.argv[1:]; bucket_name,prefix=uri[5:].split("/",1); client=storage.Client(); blobs={blob.name.removeprefix(prefix.rstrip("/")+"/"):blob for blob in client.list_blobs(bucket_name,prefix=prefix.rstrip("/") + "/")}
expected={f"device_slot_{slot:02d}.safetensors" for slot in range(32)}|{"manifest.json"}
if set(blobs)!=expected: raise SystemExit(f"terminal checkpoint object set drifted: {sorted(blobs)}")
manifest=json.loads(Path(manifest_path).read_text()); preflight=json.loads(Path(preflight_path).read_text())
manifest_files={item["filename"]:item for item in manifest["files"]}; preflight_files={item["filename"]:item for item in preflight["checkpoint_files"]}
if set(manifest_files)!=expected-{"manifest.json"} or set(preflight_files)!=set(manifest_files): raise SystemExit("terminal checkpoint file ledger drifted")
terminal_files=[]
for name in sorted(manifest_files):
 record=manifest_files[name]; prior=preflight_files[name]; blob=blobs[name]
 if set(prior)!={"crc32c","file_bytes","filename","generation","sha256"} or prior["filename"]!=name or not isinstance(prior["generation"],int) or isinstance(prior["generation"],bool) or prior["generation"]<=0: raise SystemExit(f"preflight slot schema drifted: {name}")
 if prior["file_bytes"]!=record["file_bytes"] or prior["crc32c"]!=record["crc32c"] or prior["sha256"]!=record["sha256"]: raise SystemExit(f"preflight/manifest slot drifted: {name}")
 if int(blob.size)!=prior["file_bytes"] or blob.crc32c!=prior["crc32c"] or int(blob.generation)!=prior["generation"]: raise SystemExit(f"terminal slot identity drifted: {name}")
 terminal_files.append(dict(prior))
raw=Path(manifest_path).read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); expected_crc=base64.b64encode(crc.digest()).decode("ascii"); manifest_blob=blobs["manifest.json"]
if int(manifest_blob.size)!=len(raw) or manifest_blob.crc32c!=expected_crc or not manifest_blob.generation: raise SystemExit("remote manifest bytes drifted")
Path(output).write_text(json.dumps({"checkpoint_files":terminal_files,"manifest":{"crc32c":manifest_blob.crc32c,"file_bytes":int(manifest_blob.size),"generation":int(manifest_blob.generation),"sha256":sha256(raw).hexdigest()},"nonterminal_object_count":33},indent=2,sort_keys=True)+"\n")
PY

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$MANIFEST_ROOT/manifest.json" "$RUN_DIR/remote_preflight.json" \
  "$RUN_DIR/remote_terminal.json" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/checkpoint_SUCCESS.json" "$PIN" "$MESH_HASH" \
  "$INVENTORY_SHA" "$MANIFEST_SHA" "$TOPOLOGY_HASH" \
  "$EXPECTED_PAYLOAD_BYTES" "$EXPECTED_SLOT_BYTES" \
  "$EXPECTED_TENSORS_PER_SLOT" "$EXPECTED_SOURCE_FILES" "$TAG" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
manifest_path,preflight_path,terminal_path,census_path,output,code_hash,mesh_hash,inventory_hash,manifest_hash,topology_hash,expected_payload,expected_slot,expected_tensors,expected_sources,tag=sys.argv[1:]
manifest=json.loads(Path(manifest_path).read_text()); preflight=Path(preflight_path); terminal=Path(terminal_path); census=Path(census_path); preflight_value=json.loads(preflight.read_text()); terminal_value=json.loads(terminal.read_text())
if manifest.get("manifest_sha256")!=manifest_hash or manifest.get("code_hash")!=code_hash or manifest.get("mesh_hash")!=mesh_hash or manifest["source"]["inventory_sha256"]!=inventory_hash: raise SystemExit("terminal WS32 manifest identity drifted")
expected_payload=int(expected_payload); expected_slot=int(expected_slot); expected_tensors=int(expected_tensors); expected_sources=int(expected_sources)
if manifest.get("packed_payload_bytes")!=expected_payload or len(manifest.get("files",[]))!=32 or len(manifest.get("tensor_schema",[]))!=expected_tensors or len(manifest["source"].get("files",[]))!=expected_sources: raise SystemExit("terminal WS32 manifest cardinality/byte contract drifted")
if any(item.get("payload_bytes")!=expected_slot or len(item.get("tensor_sha256",[]))!=expected_tensors for item in manifest["files"]): raise SystemExit("terminal WS32 per-slot contract drifted")
preflight_files={item["filename"]:item for item in preflight_value.get("checkpoint_files",[])}; terminal_files={item["filename"]:item for item in terminal_value.get("checkpoint_files",[])}; manifest_files={item["filename"]:item for item in manifest["files"]}
if len(preflight_files)!=32 or preflight_files!=terminal_files or set(preflight_files)!=set(manifest_files): raise SystemExit("terminal WS32 slot authentication drifted")
for name,record in manifest_files.items():
 sealed=terminal_files[name]
 if sealed.get("file_bytes")!=record["file_bytes"] or sealed.get("crc32c")!=record["crc32c"] or sealed.get("sha256")!=record["sha256"] or not isinstance(sealed.get("generation"),int) or isinstance(sealed.get("generation"),bool) or sealed["generation"]<=0: raise SystemExit(f"terminal WS32 slot record drifted: {name}")
value={"artifact_kind":"greenfield_ws32_runtime_checkpoint_success","code_hash":code_hash,"file_count":32,"format_version":1,"manifest_file_sha256":sha256(Path(manifest_path).read_bytes()).hexdigest(),"manifest_sha256":manifest_hash,"mesh_hash":mesh_hash,"packed_payload_bytes":manifest["packed_payload_bytes"],"performance_claim":False,"post_census_sha256":sha256(census.read_bytes()).hexdigest(),"remote_preflight_sha256":sha256(preflight.read_bytes()).hexdigest(),"remote_terminal_sha256":sha256(terminal.read_bytes()).hexdigest(),"source_file_count":expected_sources,"source_inventory_sha256":inventory_hash,"tag":tag,"topology_hash":topology_hash,"tpu_initialized":False}
without=dict(value); value["success_sha256"]=sha256(json.dumps(without,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode()).hexdigest(); Path(output).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
PY

say "publishing checkpoint SUCCESS last"
gcloud storage cp --no-clobber "$RUN_DIR/checkpoint_SUCCESS.json" \
  "$CHECKPOINT_URI/SUCCESS" >/dev/null
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$CHECKPOINT_URI" "$RUN_DIR/checkpoint_SUCCESS.json" \
  "$RUN_DIR/remote_success.json" <<'PY'
import base64,json,sys
from hashlib import sha256
from pathlib import Path
import google_crc32c
from google.cloud import storage
uri,success_path,output=sys.argv[1:]; bucket_name,prefix=uri[5:].split("/",1); blobs={blob.name.removeprefix(prefix.rstrip("/")+"/"):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix.rstrip("/") + "/")}; expected={f"device_slot_{slot:02d}.safetensors" for slot in range(32)}|{"manifest.json","SUCCESS"}
if set(blobs)!=expected: raise SystemExit(f"sealed checkpoint object set drifted: {sorted(blobs)}")
raw=Path(success_path).read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); expected_crc=base64.b64encode(crc.digest()).decode("ascii"); blob=blobs["SUCCESS"]
if int(blob.size)!=len(raw) or blob.crc32c!=expected_crc: raise SystemExit("remote SUCCESS bytes drifted")
Path(output).write_text(json.dumps({"object_count":34,"success":{"crc32c":blob.crc32c,"file_bytes":int(blob.size),"generation":int(blob.generation),"sha256":sha256(raw).hexdigest()}},indent=2,sort_keys=True)+"\n")
PY
say "sealing protected evidence archive"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
sha256sum "$RUN_DIR/census_pre.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/sync.txt" "$RUN_DIR/pack.txt" "$RUN_DIR/remote_preflight.json" \
  "$RUN_DIR/finalize.txt" "$RUN_DIR/remote_terminal.json" \
  "$RUN_DIR/checkpoint_SUCCESS.json" "$RUN_DIR/remote_success.json" \
  "$RUN_DIR/orchestrator.sealed.log" >"$RUN_DIR/evidence.sha256"

say "uploading and authenticating the nonterminal results archive"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
import sys
from pathlib import Path
from google.cloud import storage
root,uri=Path(sys.argv[1]),sys.argv[2]; bucket_name,prefix=uri[5:].split("/",1); bucket=storage.Client().bucket(bucket_name)
for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
 name=str(path.relative_to(root))
 if name=="host_records" or name.startswith("host_records/"): continue
 bucket.blob(f"{prefix.rstrip('/')}/{name}").upload_from_filename(path,if_generation_match=0,timeout=600)
PY
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" "$RUN_DIR/remote_results.json" <<'PY'
import base64,json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
root,uri,output=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]); bucket_name,prefix=uri[5:].split("/",1); client=storage.Client(); remote={blob.name.removeprefix(prefix.rstrip("/")+"/"):blob for blob in client.list_blobs(bucket_name,prefix=prefix.rstrip("/")+"/")}
local={str(path.relative_to(root)):path for path in root.rglob("*") if path.is_file() and not path.is_symlink() and path!=output}
if set(remote)!=set(local): raise SystemExit(f"nonterminal results object set drifted: {sorted(set(remote)^set(local))}")
records=[]
for name,path in sorted(local.items()):
 raw=path.read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); expected=base64.b64encode(crc.digest()).decode("ascii"); blob=remote[name]
 if int(blob.size)!=len(raw) or blob.crc32c!=expected or not blob.generation: raise SystemExit(f"remote results object drifted: {name}")
 records.append({"crc32c":expected,"file_bytes":len(raw),"generation":int(blob.generation),"name":name})
output.write_text(json.dumps({"objects":records},indent=2,sort_keys=True)+"\n")
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_results.json" \
  "$REMOTE_PREFIX/remote_results.json" >/dev/null
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" "$TAG" "$MANIFEST_SHA" <<'PY'
import base64,json,sys
from hashlib import sha256
from pathlib import Path
import google_crc32c
from google.cloud import storage
root,uri,code_hash,tag,manifest_hash=Path(sys.argv[1]),sys.argv[2],*sys.argv[3:]; bucket_name,prefix=uri[5:].split("/",1); remote={blob.name.removeprefix(prefix.rstrip("/")+"/"):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix.rstrip("/")+"/")}; local={str(path.relative_to(root)):path for path in root.rglob("*") if path.is_file() and not path.is_symlink()}
if set(remote)!=set(local): raise SystemExit(f"sealed nonterminal results object set drifted: {sorted(set(remote)^set(local))}")
ledger=root/"remote_results.json"; crc=google_crc32c.Checksum(); crc.update(ledger.read_bytes()); expected_crc=base64.b64encode(crc.digest()).decode("ascii"); blob=remote["remote_results.json"]
if int(blob.size)!=ledger.stat().st_size or blob.crc32c!=expected_crc or not blob.generation: raise SystemExit("remote results ledger identity drifted")
value={"artifact_kind":"greenfield_ws32_runtime_pack_results_success","code_hash":code_hash,"format_version":1,"manifest_sha256":manifest_hash,"nonterminal_object_count":len(remote),"performance_claim":False,"remote_results_crc32c":expected_crc,"remote_results_generation":int(blob.generation),"remote_results_sha256":sha256(ledger.read_bytes()).hexdigest(),"tag":tag,"tpu_initialized":False}
value["success_sha256"]=sha256(json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True).encode()).hexdigest(); (root/"SUCCESS").write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null
results_success_uploaded=1
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
import base64,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
root,uri=Path(sys.argv[1]),sys.argv[2]; bucket_name,prefix=uri[5:].split("/",1); remote={blob.name.removeprefix(prefix.rstrip("/")+"/"):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix.rstrip("/")+"/")}; local={str(path.relative_to(root)):path for path in root.rglob("*") if path.is_file() and not path.is_symlink()}
if set(remote)!=set(local): raise SystemExit(f"terminal results object set drifted: {sorted(set(remote)^set(local))}")
raw=(root/"SUCCESS").read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); expected=base64.b64encode(crc.digest()).decode("ascii"); blob=remote["SUCCESS"]
if int(blob.size)!=len(raw) or blob.crc32c!=expected or not blob.generation: raise SystemExit("remote results SUCCESS identity drifted")
PY
trap - EXIT
echo "WS32_RUNTIME_PACK_OK $TAG $MANIFEST_SHA"
