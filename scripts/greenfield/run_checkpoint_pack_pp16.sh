#!/usr/bin/env bash
# Protected complete PP16 final-layout pack; CPU/GCS only, no TPU execution.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly SOURCE_REVISION=gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658
readonly SOURCE_INVENTORY_SHA=a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4
readonly PLAN_TAG=greenfield_checkpoint_plan_pp16_20260827T022537742669498Z
readonly PLAN_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP16_LP2/$PLAN_TAG
readonly LAYOUT=$PLAN_ROOT/layout_manifest.json
readonly LAYOUT_FILE_SHA=b6168039ff5eb5dd9da18855f086862d452e9bdc6b4c91d5647a1d94ecee9daa
readonly LAYOUT_MANIFEST_SHA=f97de2d8ec525d984963c522696e09c40b9cb478fb562917f0a42fc2083b15f9
readonly LAYOUT_CODE_HASH=6dc73048fec3b15892e061bbaf1ec5886f22d29c
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly PLAN_GROUP_HASH=6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21
readonly PLAN_MANIFEST_SHA=3c3ea07b0f97a84702e7b2ad370b014ea22c9e69bc7d35ad88e23a4ad82ded16
readonly EXECUTION_PLAN_SHA=079cefe6a646e0c56120c0903c395587bc95bf8dc6d66de0cf9eb029827794c3
readonly PACKED_PAYLOAD_BYTES=757149950848

cd "$WORKTREE"
[[ ${GLM_GREENFIELD_PP16_FULL_PACK:-0} == 1 ]] || {
  echo "PP16 complete checkpoint pack is default-off" >&2
  exit 2
}
readonly RESUME=${GLM_GREENFIELD_PP16_FULL_PACK_RESUME:-0}
[[ $RESUME == 0 || $RESUME == 1 ]] || {
  echo "GLM_GREENFIELD_PP16_FULL_PACK_RESUME must be 0 or 1" >&2
  exit 2
}
PIN=$(git rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_FULL_PACK_TAG:-greenfield_full_pack_pp16_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_full_pack_pp16_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid PP16 full-pack tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
PACK_RUN_DIR=$RUN_DIR/pack
DESTINATION=$APPROVED_BUCKET/checkpoints/greenfield/glm52/packed/PP16_LP2/$TAG
CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP16_LP2/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
DIAGNOSTIC_PREFIX=$APPROVED_BUCKET/diagnostics/greenfield/checkpoint_pack/$TAG
readonly PIN TAG RUN_DIR PACK_RUN_DIR DESTINATION CHECKPOINT_ROOT REMOTE_PREFIX DIAGNOSTIC_PREFIX

[[ $(git branch --show-current) == "$BRANCH" ]]
[[ -z $(git status --porcelain) ]]
[[ -r $SOURCE_ROOT/model.safetensors.index.json && -r $LAYOUT ]]
[[ $(sha256sum "$LAYOUT" | awk '{print $1}') == "$LAYOUT_FILE_SHA" ]]
if [[ $RESUME == 0 ]]; then
  [[ ! -e $RUN_DIR ]] || { echo "append-only run exists: $RUN_DIR" >&2; exit 2; }
  mkdir -p "$RUN_DIR"
else
  [[ -d $PACK_RUN_DIR && -r $RUN_DIR/preflight.json ]] || {
    echo "resume requires the original PP16 run directory" >&2
    exit 2
  }
fi

exec 7>/home/gianl/glm-run/.glm_checkpoint_pack.lock
flock -n 7 || { echo "another checkpoint pack holds the lease" >&2; exit 1; }
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

say() {
  echo "[checkpoint-pack-pp16 $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

on_exit() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; pack is resumable with TAG=$TAG"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$DIAGNOSTIC_PREFIX/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "PIN=$PIN TAG=$TAG RESUME=$RESUME"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]] || {
  say "ABORT: code pin is not pushed"
  exit 1
}
bucket_location=$(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)')
[[ $bucket_location == "$APPROVED_LOCATION" ]] || {
  say "ABORT: bucket is $bucket_location, expected $APPROVED_LOCATION"
  exit 1
}
mount_record=$(findmnt -T "$SOURCE_ROOT" -n -o SOURCE,FSTYPE,OPTIONS)
[[ $mount_record == driftbench-dsv4-uc\ fuse.gcsfuse\ * && $mount_record == *ro* ]] || {
  say "ABORT: source is not the approved read-only mount: $mount_record"
  exit 1
}
gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]] || { say "ABORT: result prefix occupied"; exit 1; }

if [[ $RESUME == 0 ]]; then
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$RUN_DIR/preflight.json" "$PIN" "$TAG" "$DESTINATION" "$bucket_location" <<'PY'
import json,sys
from pathlib import Path
value={'approved_bucket':'gs://driftbench-dsv4-uc','bucket_location':sys.argv[5],
 'code_hash':sys.argv[2],'destination':sys.argv[4],
 'execution_plan_sha256':'079cefe6a646e0c56120c0903c395587bc95bf8dc6d66de0cf9eb029827794c3',
 'layout_manifest_sha256':'f97de2d8ec525d984963c522696e09c40b9cb478fb562917f0a42fc2083b15f9',
 'packed_payload_bytes':757149950848,'performance_claim':False,'plan_id':'PP16_LP2',
 'run_tag':sys.argv[3],'source_inventory_sha256':'a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4',
 'source_revision':'gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658'}
Path(sys.argv[1]).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
fi

say "streaming complete $PACKED_PAYLOAD_BYTES-byte PP16 layout to $APPROVED_LOCATION"
pack_args=(
  --layout-manifest "$LAYOUT"
  --source-root "$SOURCE_ROOT"
  --destination "$DESTINATION"
  --run-dir "$PACK_RUN_DIR"
  --expected-code-hash "$PIN"
)
if [[ $RESUME == 1 ]]; then
  pack_args+=(--resume)
fi
if [[ $RESUME == 1 && -r $PACK_RUN_DIR/packed_manifest.json && \
      -r $PACK_RUN_DIR/evidence.sha256 && -r $CHECKPOINT_ROOT/SUCCESS ]]; then
  if [[ ! -r $PACK_RUN_DIR/SUCCESS ]]; then
    gcloud storage cp "$DESTINATION/SUCCESS" "$PACK_RUN_DIR/SUCCESS" >/dev/null
  fi
  say "checkpoint payload is already complete; resuming independent inspection"
else
  tee_args=("$RUN_DIR/pack.log")
  if [[ $RESUME == 1 ]]; then
    tee_args=(-a "${tee_args[@]}")
  fi
  PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu timeout --signal=TERM --kill-after=60 21600 \
    /home/gianl/vllm-env/bin/python -u scripts/greenfield/pack_checkpoint_streaming.py \
    "${pack_args[@]}" 2>&1 | tee "${tee_args[@]}"
fi

packed_sha=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$PACK_RUN_DIR/packed_manifest.json")
[[ ${#packed_sha} -eq 64 ]]
(cd "$PACK_RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)
[[ $(cat "$PACK_RUN_DIR/SUCCESS") == "$packed_sha  packed_manifest.json" ]]
for _ in $(seq 1 12); do
  [[ -r $CHECKPOINT_ROOT/SUCCESS ]] && break
  sleep 5
done
[[ -r $CHECKPOINT_ROOT/SUCCESS ]] || { say "ABORT: packed mount did not expose SUCCESS"; exit 1; }

say "independently inspecting exact remote object set, generations, CRCs and metadata"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  scripts/greenfield/inspect_packed_checkpoint.py \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output "$RUN_DIR/inspection.json" \
  --expected-code-hash "$PIN" \
  --packed-manifest-sha256 "$packed_sha" \
  --layout-manifest-sha256 "$LAYOUT_MANIFEST_SHA" \
  --source-inventory-sha256 "$SOURCE_INVENTORY_SHA" \
  --source-revision "$SOURCE_REVISION" \
  --topology-sha256 "$TOPOLOGY_HASH" \
  --plan-group-sha256 "$PLAN_GROUP_HASH" \
  --plan-manifest-sha256 "$PLAN_MANIFEST_SHA" \
  --execution-plan-sha256 "$EXECUTION_PLAN_SHA" \
  --layout-code-hash "$LAYOUT_CODE_HASH" \
  --pack-code-hash "$PIN" \
  --destination "$DESTINATION" \
  --plan-id PP16_LP2 >"$RUN_DIR/inspection.log" 2>&1

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$TAG" "$packed_sha" "$DESTINATION" <<'PY'
from hashlib import sha256
import json,sys
from pathlib import Path
run=Path(sys.argv[1]); pin,tag,packed,destination=sys.argv[2:]
inspection=json.loads((run/'inspection.json').read_text())
manifest=json.loads((run/'pack/packed_manifest.json').read_text())
if inspection['status']!='SUCCESS' or manifest['manifest_sha256']!=packed:
 raise SystemExit('PP16 pack inspection identity failed')
if manifest['file_count']!=34 or manifest['packed_payload_bytes']!=757149950848:
 raise SystemExit('PP16 packed cardinality/bytes drifted')
summary={'artifact_kind':'greenfield_full_checkpoint_pack_proof','code_hash':pin,
 'data_file_count':34,'destination':destination,'layout_manifest_sha256':manifest['layout_manifest_sha256'],
 'packed_file_bytes':manifest['packed_file_bytes'],'packed_manifest_sha256':packed,
 'packed_payload_bytes':manifest['packed_payload_bytes'],'performance_claim':False,'plan_id':'PP16_LP2',
 'remote_object_count':inspection['remote_object_count'],'run_tag':tag,'status':'SUCCESS'}
(run/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
for name in ('preflight.json','pack.log','inspection.json','inspection.log','summary.json'):
 path=run/name
 if not path.is_file(): raise SystemExit(f'missing PP16 pack evidence: {name}')
with (run/'evidence.sha256').open('w') as out:
 for name in ('preflight.json','pack.log','inspection.json','inspection.log','summary.json',
              'pack/control.json','pack/packed_manifest.json','pack/evidence.sha256','pack/SUCCESS'):
  path=run/name; out.write(f'{sha256(path.read_bytes()).hexdigest()}  {name}\n')
PY
(cd "$RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)

say "publishing nonterminal pack proof"
for name in preflight.json pack.log inspection.json inspection.log summary.json evidence.sha256; do
  gcloud storage cp --no-clobber "$RUN_DIR/$name" "$REMOTE_PREFIX/$name" >/dev/null
done
for name in control.json packed_manifest.json evidence.sha256 SUCCESS; do
  gcloud storage cp --no-clobber "$PACK_RUN_DIR/$name" "$REMOTE_PREFIX/pack_$name" >/dev/null
done

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
import base64,json,sys
from hashlib import sha256
from pathlib import Path
import google_crc32c
from google.cloud import storage
run=Path(sys.argv[1]); remote=sys.argv[2]
names=['preflight.json','pack.log','inspection.json','inspection.log','summary.json','evidence.sha256',
 'pack_control.json','pack_packed_manifest.json','pack_evidence.sha256','pack_SUCCESS']
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blobs={b.name.removeprefix(prefix):b for b in storage.Client().list_blobs(bucket_name,prefix=prefix)}
if set(blobs)!=set(names): raise SystemExit('PP16 pack preterminal object set drifted')
records=[]
for name in names:
 path=(run/'pack'/name.removeprefix('pack_')) if name.startswith('pack_') else run/name
 crc=base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode()
 blob=blobs[name]
 if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation:
  raise SystemExit(f'PP16 pack proof remote identity drifted: {name}')
 records.append({'crc32c':crc,'generation':int(blob.generation),'name':name,'size':int(blob.size)})
ledger={'objects':records,'remote_prefix':remote}
raw=json.dumps(ledger,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
ledger['ledger_sha256']=sha256(raw).hexdigest()
(run/'remote_objects.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" "$REMOTE_PREFIX/remote_objects.json" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$TAG" "$REMOTE_PREFIX" <<'PY'
import base64,json,sys
from hashlib import sha256
from pathlib import Path
import google_crc32c
from google.cloud import storage
run=Path(sys.argv[1]); pin,tag,remote=sys.argv[2:]
summary=json.loads((run/'summary.json').read_text()); ledger=json.loads((run/'remote_objects.json').read_text())
raw=json.dumps({k:v for k,v in ledger.items() if k!='ledger_sha256'},allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
if ledger['ledger_sha256']!=sha256(raw).hexdigest(): raise SystemExit('PP16 pack ledger self-hash drifted')
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blob=storage.Client().bucket(bucket_name).blob(prefix+'remote_objects.json'); blob.reload()
path=run/'remote_objects.json'; crc=base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode()
if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation:
 raise SystemExit('PP16 pack remote ledger identity drifted')
value={'artifact_kind':'greenfield_full_checkpoint_pack_SUCCESS','code_hash':pin,
 'checkpoint_destination':summary['destination'],'packed_manifest_sha256':summary['packed_manifest_sha256'],
 'performance_claim':False,'remote_ledger_crc32c':crc,'remote_ledger_generation':int(blob.generation),
 'remote_ledger_sha256':ledger['ledger_sha256'],'run_tag':tag}
canonical=lambda v:json.dumps(v,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
value['success_sha256']=sha256(canonical(value)).hexdigest()
(run/'SUCCESS').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
# The terminal marker is intentionally the final remote write.
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
import base64,json,sys
from hashlib import sha256
from pathlib import Path
import google_crc32c
from google.cloud import storage
run=Path(sys.argv[1]); remote=sys.argv[2]; ledger=json.loads((run/'remote_objects.json').read_text()); success=json.loads((run/'SUCCESS').read_text())
canonical=lambda v:json.dumps(v,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
if success['success_sha256']!=sha256(canonical({k:v for k,v in success.items() if k!='success_sha256'})).hexdigest():
 raise SystemExit('PP16 pack SUCCESS self-hash drifted')
if ledger['ledger_sha256']!=sha256(canonical({k:v for k,v in ledger.items() if k!='ledger_sha256'})).hexdigest():
 raise SystemExit('PP16 pack terminal ledger self-hash drifted')
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blobs={b.name.removeprefix(prefix):b for b in storage.Client().list_blobs(bucket_name,prefix=prefix)}
expected={x['name'] for x in ledger['objects']}|{'remote_objects.json','SUCCESS'}
if set(blobs)!=expected: raise SystemExit('PP16 pack terminal object set drifted')
ledger_blob=blobs['remote_objects.json']; ledger_path=run/'remote_objects.json'
ledger_crc=base64.b64encode(google_crc32c.Checksum(ledger_path.read_bytes()).digest()).decode()
if (success['remote_ledger_sha256']!=ledger['ledger_sha256'] or
    success['remote_ledger_crc32c']!=ledger_crc or
    success['remote_ledger_generation']!=int(ledger_blob.generation)):
 raise SystemExit('PP16 pack SUCCESS-to-ledger binding drifted')
for item in ledger['objects']:
 blob=blobs[item['name']]
 if int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']:
  raise SystemExit(f"PP16 pack terminal payload drifted: {item['name']}")
for name in ('remote_objects.json','SUCCESS'):
 path=run/name; crc=base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode(); blob=blobs[name]
 if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation:
  raise SystemExit(f'PP16 pack terminal identity drifted: {name}')
PY
trap - EXIT
echo "[checkpoint-pack-pp16 $(date -u +%H:%M:%S)] COMPLETE manifest=$packed_sha"
