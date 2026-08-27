#!/usr/bin/env bash
# Protected real-byte probe of the complete PP16 checkpoint layout and packer.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly SOURCE_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly PLAN_TAG=greenfield_checkpoint_plan_pp16_20260827T022537742669498Z
readonly PLAN_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP16_LP2/$PLAN_TAG
readonly LAYOUT=$PLAN_ROOT/layout_manifest.json
readonly LAYOUT_FILE_SHA=b6168039ff5eb5dd9da18855f086862d452e9bdc6b4c91d5647a1d94ecee9daa
readonly LAYOUT_MANIFEST_SHA=f97de2d8ec525d984963c522696e09c40b9cb478fb562917f0a42fc2083b15f9
readonly PLAN_SUCCESS_FILE_SHA=5cac6b62101c890df0552a780ca679d664e554a8145e62757de24d67426e2176
readonly SELECTION=$WORKTREE/configs/greenfield-pp16-checkpoint-probe.json

cd "$WORKTREE"
[[ ${GLM_GREENFIELD_PP16_CHECKPOINT_PROBE:-0} == 1 ]] || {
  echo "PP16 real-byte checkpoint probe is default-off" >&2
  exit 2
}
PIN=$(git rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_CHECKPOINT_PROBE_TAG:-greenfield_checkpoint_probe_pp16_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_checkpoint_probe_pp16_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid PP16 checkpoint probe tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
OUTPUT_DIR=$RUN_DIR/probe
REMOTE_PREFIX=$APPROVED_BUCKET/checkpoints/greenfield/glm52/probes/PP16_LP2/$TAG
readonly PIN TAG RUN_DIR OUTPUT_DIR REMOTE_PREFIX

[[ $(git branch --show-current) == "$BRANCH" ]]
[[ -z $(git status --porcelain) ]]
[[ ! -e $RUN_DIR ]]
[[ -r $LAYOUT && -r $PLAN_ROOT/SUCCESS && -r $SELECTION ]]
mkdir -p "$RUN_DIR"
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

say() {
  echo "[checkpoint-probe-pp16 $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

say "PIN=$PIN TAG=$TAG"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == \
  "$APPROVED_LOCATION" ]]
for path in "$SOURCE_ROOT" "$PLAN_ROOT"; do
  mount_record=$(findmnt -T "$path" -n -o SOURCE,FSTYPE,OPTIONS)
  [[ $mount_record == driftbench-dsv4-uc\ fuse.gcsfuse\ * && $mount_record == *ro* ]]
done
[[ $(sha256sum "$LAYOUT" | awk '{print $1}') == "$LAYOUT_FILE_SHA" ]]
[[ $(sha256sum "$PLAN_ROOT/SUCCESS" | awk '{print $1}') == "$PLAN_SUCCESS_FILE_SHA" ]]
gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' \
  >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]]

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/preflight.json" "$PIN" "$TAG" "$REMOTE_PREFIX" "$SELECTION" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

output, pin, tag, remote, selection = sys.argv[1:]
value = {
    "approved_bucket": "gs://driftbench-dsv4-uc",
    "bucket_location": "US-CENTRAL2",
    "code_hash": pin,
    "layout_file_sha256": "b6168039ff5eb5dd9da18855f086862d452e9bdc6b4c91d5647a1d94ecee9daa",
    "layout_manifest_sha256": "f97de2d8ec525d984963c522696e09c40b9cb478fb562917f0a42fc2083b15f9",
    "performance_claim": False,
    "plan_id": "PP16_LP2",
    "remote_prefix": remote,
    "run_tag": tag,
    "selection_file_sha256": sha256(Path(selection).read_bytes()).hexdigest(),
}
Path(output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

say "packing 50.35 MB representative real bytes through both PP16 owners"
PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu \
  /home/gianl/vllm-env/bin/python scripts/greenfield/pack_checkpoint_probe.py \
  --layout-manifest "$LAYOUT" --selection "$SELECTION" \
  --source-root "$SOURCE_ROOT" --output-dir "$OUTPUT_DIR" \
  --expected-code-hash "$PIN" >"$RUN_DIR/pack.log" 2>&1
[[ $(sha256sum "$OUTPUT_DIR/selection.json" | awk '{print $1}') == \
  $(sha256sum "$SELECTION" | awk '{print $1}') ]]

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$OUTPUT_DIR" "$PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]); pin=sys.argv[2]
manifest=json.loads((root/'manifest.json').read_text())
unhashed={key:value for key,value in manifest.items() if key!='manifest_sha256'}
canonical=json.dumps(unhashed,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
if manifest['manifest_sha256']!=sha256(canonical).hexdigest(): raise SystemExit('probe manifest self-hash drifted')
expected={'artifact_kind':'greenfield_checkpoint_pack_probe','code_hash':pin,'file_count':2,
 'layout_file_sha256':'b6168039ff5eb5dd9da18855f086862d452e9bdc6b4c91d5647a1d94ecee9daa',
 'layout_manifest_sha256':'f97de2d8ec525d984963c522696e09c40b9cb478fb562917f0a42fc2083b15f9',
 'packed_payload_bytes':50347904,'plan_id':'PP16_LP2','selected_source_bytes':50345920,'stage_id':0}
for key,value in expected.items():
 if manifest.get(key)!=value: raise SystemExit(f'probe identity drifted at {key}')
if [item['device_slot'] for item in manifest['files']]!=[0,1]: raise SystemExit('probe owner slots drifted')
if {(item['layout'],item['value_class']) for item in manifest['sources']}!={
 ('replicated','parameter'),('replicated','fp8_scale'),('axis_sharded','parameter'),
 ('axis_sharded','fp8_scale'),('expert_identity','parameter'),('expert_identity','fp8_scale')}:
 raise SystemExit('probe layout/value-class coverage drifted')
by_name={item['name']:item for item in manifest['sources']}
if {item['axis'] for item in by_name['model.layers.3.mlp.shared_experts.gate_proj.weight']['destinations']}!={0}:
 raise SystemExit('probe axis-0 coverage drifted')
if {item['axis'] for item in by_name['model.layers.3.mlp.shared_experts.down_proj.weight']['destinations']}!={1}:
 raise SystemExit('probe axis-1 coverage drifted')
if [item['device_slot'] for item in by_name['model.layers.3.mlp.experts.0.down_proj.weight']['destinations']]!=[0] or [item['device_slot'] for item in by_name['model.layers.3.mlp.experts.128.down_proj.weight']['destinations']]!=[1]:
 raise SystemExit('probe expert-owner coverage drifted')
lines=(root/'evidence.sha256').read_text().splitlines()
if len(lines)!=4: raise SystemExit('probe evidence cardinality drifted')
for line in lines:
 digest,name=line.split('  ',1)
 if sha256((root/name).read_bytes()).hexdigest()!=digest: raise SystemExit(f'probe evidence drifted: {name}')
PY

say "publishing nonterminal probe evidence in US-CENTRAL2"
for name in selection.json manifest.json evidence.sha256; do
  gcloud storage cp --no-clobber "$OUTPUT_DIR/$name" "$REMOTE_PREFIX/$name" >/dev/null
done
gcloud storage cp --no-clobber "$RUN_DIR/preflight.json" "$RUN_DIR/pack.log" \
  "$REMOTE_PREFIX/" >/dev/null
while IFS= read -r name; do
  gcloud storage cp --no-clobber "$OUTPUT_DIR/$name" "$REMOTE_PREFIX/$name" >/dev/null
done < <(PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$OUTPUT_DIR/manifest.json" <<'PY'
import json,sys
for item in json.load(open(sys.argv[1]))['files']: print(item['filename'])
PY
)

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$OUTPUT_DIR" "$REMOTE_PREFIX" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys
import google_crc32c
from google.cloud import storage

run=Path(sys.argv[1]); root=Path(sys.argv[2]); remote=sys.argv[3]
manifest=json.loads((root/'manifest.json').read_text())
local={name:root/name for name in ('selection.json','manifest.json','evidence.sha256')}
local.update({'preflight.json':run/'preflight.json','pack.log':run/'pack.log'})
local.update({item['filename']:root/item['filename'] for item in manifest['files']})
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blobs={blob.name.removeprefix(prefix):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix)}
if set(blobs)!=set(local): raise SystemExit('probe preterminal remote set drifted')
records=[]
for name,path in sorted(local.items()):
 checksum=google_crc32c.Checksum(path.read_bytes()); crc=base64.b64encode(checksum.digest()).decode('ascii'); blob=blobs[name]
 if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation: raise SystemExit(f'probe remote identity drifted: {name}')
 records.append({'crc32c':crc,'generation':int(blob.generation),'name':name,'size':int(blob.size)})
ledger={'objects':records,'remote_prefix':remote}
canonical=json.dumps(ledger,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
ledger['ledger_sha256']=sha256(canonical).hexdigest()
(run/'remote_objects.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$OUTPUT_DIR" "$PIN" "$TAG" "$REMOTE_PREFIX" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys
import google_crc32c
from google.cloud import storage

run=Path(sys.argv[1]); root=Path(sys.argv[2]); pin,tag,remote=sys.argv[3:]
manifest=json.loads((root/'manifest.json').read_text()); ledger_path=run/'remote_objects.json'; ledger=json.loads(ledger_path.read_text())
without={k:v for k,v in ledger.items() if k!='ledger_sha256'}
canonical=lambda v:json.dumps(v,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
if ledger['ledger_sha256']!=sha256(canonical(without)).hexdigest(): raise SystemExit('probe ledger self-hash drifted')
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blob=storage.Client().bucket(bucket_name).blob(prefix+'remote_objects.json'); blob.reload()
crc=base64.b64encode(google_crc32c.Checksum(ledger_path.read_bytes()).digest()).decode('ascii')
if int(blob.size)!=ledger_path.stat().st_size or blob.crc32c!=crc or not blob.generation: raise SystemExit('probe ledger remote identity drifted')
value={'artifact_kind':'greenfield_checkpoint_pack_probe_SUCCESS','code_hash':pin,
 'layout_manifest_sha256':manifest['layout_manifest_sha256'],'manifest_sha256':manifest['manifest_sha256'],
 'packed_payload_bytes':manifest['packed_payload_bytes'],'performance_claim':False,'plan_id':'PP16_LP2',
 'remote_ledger_crc32c':crc,'remote_ledger_generation':int(blob.generation),
 'remote_ledger_sha256':ledger['ledger_sha256'],'remote_prefix':remote,'run_tag':tag,
 'selected_source_bytes':manifest['selected_source_bytes']}
value['success_sha256']=sha256(canonical(value)).hexdigest()
(run/'SUCCESS').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null

say "verifying terminal generation/CRC object set"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
import base64,json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
run=Path(sys.argv[1]); remote=sys.argv[2]; ledger=json.loads((run/'remote_objects.json').read_text())
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blobs={blob.name.removeprefix(prefix):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix)}
expected={item['name'] for item in ledger['objects']}|{'remote_objects.json','SUCCESS'}
if set(blobs)!=expected: raise SystemExit('terminal probe object set drifted')
for item in ledger['objects']:
 blob=blobs[item['name']]
 if int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']: raise SystemExit(f"terminal probe payload drifted: {item['name']}")
for name in ('remote_objects.json','SUCCESS'):
 path=run/name; crc=base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode('ascii'); blob=blobs[name]
 if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation: raise SystemExit(f'terminal probe identity drifted: {name}')
PY
say "COMPLETE manifest=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$OUTPUT_DIR/manifest.json")"
say "Bounded pack proof only: no TPU, complete checkpoint, DB row, HBM, or performance claim."
