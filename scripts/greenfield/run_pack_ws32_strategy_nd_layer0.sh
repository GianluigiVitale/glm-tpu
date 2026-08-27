#!/usr/bin/env bash
# Append-only pack and publication of the bounded WS32 layer-0 dense weights.
set -euo pipefail

readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_TAG=greenfield_runtime_feature_qkv_dense_pack_pp8_20260813T152037261372350Z
readonly SOURCE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$SOURCE_TAG
readonly SOURCE_MANIFEST_FILE_SHA=a4d6e2e5fc2408d5e96fded1f8b526979ea0b86202326c9064825da4ad97c9a9
readonly SOURCE_SUCCESS_FILE_SHA=b319816df0571535adec3bb6a40b1556143b4d521c1aa7e5ac3ed8fbcaf07ae1

cd "$WORKTREE"

[[ ${GLM_GREENFIELD_WS32_STRATEGY_ND_PACK:-0} == 1 ]] || {
  echo "WS32 StrategyND layer-0 pack is default-off" >&2
  exit 2
}
PACK_KIND=${GLM_GREENFIELD_WS32_STRATEGY_ND_PACK_KIND:-layer0}
if [[ $PACK_KIND == layer0 ]]; then
  TAG_PREFIX=greenfield_ws32_strategy_nd_layer0_pack
  PACKER=scripts/greenfield/pack_ws32_strategy_nd_layer0.py
  EXPECTED_KIND=greenfield_ws32_strategy_nd_layer0_checkpoint
  EXPECTED_FILES=32
  REMOTE_CLASS=discriminators
elif [[ $PACK_KIND == dense_overlay ]]; then
  TAG_PREFIX=greenfield_ws32_strategy_nd_dense_overlay_pack
  PACKER=scripts/greenfield/pack_ws32_strategy_nd_dense_overlay.py
  EXPECTED_KIND=greenfield_ws32_strategy_nd_dense_overlay
  EXPECTED_FILES=96
  REMOTE_CLASS=overlays
else
  echo "WS32 StrategyND pack kind must be layer0 or dense_overlay" >&2
  exit 2
fi
PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_STRATEGY_ND_PACK_TAG:-${TAG_PREFIX}_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^${TAG_PREFIX}_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid StrategyND checkpoint tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
OUTPUT_ROOT=$RUN_DIR/checkpoint
REMOTE_PREFIX=$APPROVED_BUCKET/checkpoints/greenfield/glm52/$REMOTE_CLASS/WS32_2D/$TAG
readonly PACK_KIND TAG_PREFIX PACKER EXPECTED_KIND EXPECTED_FILES REMOTE_CLASS
readonly PIN TAG RUN_DIR OUTPUT_ROOT REMOTE_PREFIX

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
[[ ! -e $RUN_DIR ]]
mkdir -p "$RUN_DIR"
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

say() {
  echo "[ws32-strategy-nd-pack $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

say "PIN=$PIN source=$SOURCE_TAG"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(sha256sum "$SOURCE_ROOT/runtime_manifest.json" | awk '{print $1}') == \
  "$SOURCE_MANIFEST_FILE_SHA" ]]
[[ $(sha256sum "$SOURCE_ROOT/SUCCESS" | awk '{print $1}') == \
  "$SOURCE_SUCCESS_FILE_SHA" ]]
findmnt -T "$SOURCE_ROOT" -n -o SOURCE,FSTYPE |
  grep -q "driftbench-dsv4-uc fuse.gcsfuse"
gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' \
  >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]]

say "packing $PACK_KIND final-layout tensors"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$PACKER" \
  --source-root "$SOURCE_ROOT" \
  --source-manifest-file-sha256 "$SOURCE_MANIFEST_FILE_SHA" \
  --source-success-file-sha256 "$SOURCE_SUCCESS_FILE_SHA" \
  --expected-code-hash "$PIN" --output-root "$OUTPUT_ROOT" \
  >"$RUN_DIR/pack.log" 2>&1

say "validating local checkpoint before publication"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$OUTPUT_ROOT" "$PIN" "$EXPECTED_KIND" "$EXPECTED_FILES" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]); pin=sys.argv[2]; expected_kind=sys.argv[3]
expected_files=int(sys.argv[4])
manifest=json.loads((root/'manifest.json').read_text())
canonical=lambda value: json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
without_hash={key:value for key,value in manifest.items() if key!='manifest_sha256'}
if manifest['manifest_sha256']!=sha256(canonical(without_hash)).hexdigest():
 raise SystemExit('bounded manifest self-hash drifted')
if manifest['artifact_kind']!=expected_kind or manifest['code_hash']!=pin or manifest['file_count']!=expected_files or len(manifest['files'])!=expected_files:
 raise SystemExit('StrategyND manifest identity drifted')
coverage=set()
for record in manifest['files']:
 path=root/record['filename']
 if sha256(path.read_bytes()).hexdigest()!=record['sha256']:
  raise SystemExit(f"StrategyND file hash drifted: {record['filename']}")
 layer=record.get('layer_id',0); expert=record['expert_coordinate']; feature=record['feature_coordinate']
 coverage.add((layer,expert,feature))
 if record['model_ranks']!=list(range(expert*4,expert*4+4)):
  raise SystemExit('StrategyND model-rank ownership drifted')
layers=range(3) if expected_files==96 else range(1)
if coverage!={(layer,expert,feature) for layer in layers for expert in range(8) for feature in range(4)}:
 raise SystemExit('StrategyND owner coverage drifted')
for layer in layers:
 for expert in range(8):
  records=[item for item in manifest['files'] if item.get('layer_id',0)==layer and item['expert_coordinate']==expert]
  names=sorted(records[0]['tensors'])
  replica_names=[name for name in names if name.endswith(('merged_gate_up.weight_bits_in_out','merged_gate_up.scale_inv_in_out'))]
  if len(replica_names)!=2: raise SystemExit('StrategyND replica tensor names drifted')
  for name in replica_names:
   if len({item['tensors'][name]['sha256'] for item in records})!=1:
    raise SystemExit('feature replicas disagree in StrategyND manifest')
success=json.loads((root/'SUCCESS').read_text())
without_success={key:value for key,value in success.items() if key!='success_sha256'}
if success['success_sha256']!=sha256(canonical(without_success)).hexdigest():
 raise SystemExit('bounded SUCCESS self-hash drifted')
if success['manifest_sha256']!=manifest['manifest_sha256']:
 raise SystemExit('bounded SUCCESS manifest identity drifted')
PY

say "publishing append-only checkpoint payload"
gcloud storage cp --no-clobber "$OUTPUT_ROOT/manifest.json" \
  "$REMOTE_PREFIX/manifest.json" >/dev/null
while IFS= read -r relative; do
  gcloud storage cp --no-clobber "$OUTPUT_ROOT/$relative" \
    "$REMOTE_PREFIX/$relative" >/dev/null
done < <(
  /home/gianl/vllm-env/bin/python - "$OUTPUT_ROOT/manifest.json" <<'PY'
import json,sys
for item in json.load(open(sys.argv[1]))['files']:
 print(item['filename'])
PY
)
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$OUTPUT_ROOT" "$REMOTE_PREFIX" "$RUN_DIR/remote_objects.json" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

import google_crc32c
from google.cloud import storage

root=Path(sys.argv[1]); remote=sys.argv[2]; output=Path(sys.argv[3])
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
manifest=json.loads((root/'manifest.json').read_text())
expected=['manifest.json',*(item['filename'] for item in manifest['files'])]
blobs={blob.name.removeprefix(prefix):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix)}
if set(blobs)!=set(expected): raise SystemExit('pre-SUCCESS remote object set drifted')
records=[]
for name in expected:
 raw_path=root/name; checksum=google_crc32c.Checksum()
 with raw_path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): checksum.update(chunk)
 crc=base64.b64encode(checksum.digest()).decode('ascii'); blob=blobs[name]
 if int(blob.size)!=raw_path.stat().st_size or blob.crc32c!=crc or not blob.generation:
  raise SystemExit(f'remote object identity drifted: {name}')
 records.append({'crc32c':crc,'generation':int(blob.generation),'name':name,'size':int(blob.size)})
ledger={'objects':records,'remote_prefix':remote}
canonical=json.dumps(ledger,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
ledger['ledger_sha256']=sha256(canonical).hexdigest()
output.write_text(json.dumps(ledger,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null
gcloud storage cp --no-clobber "$OUTPUT_ROOT/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null

say "verifying terminal remote object set"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$OUTPUT_ROOT" "$REMOTE_PREFIX" "$RUN_DIR/remote_objects.json" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

import google_crc32c
from google.cloud import storage

root=Path(sys.argv[1]); remote=sys.argv[2]; ledger=json.load(open(sys.argv[3]))
without_hash={key:value for key,value in ledger.items() if key!='ledger_sha256'}
canonical=json.dumps(without_hash,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
if ledger['ledger_sha256']!=sha256(canonical).hexdigest():
 raise SystemExit('remote object ledger self-hash drifted')
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blobs={blob.name.removeprefix(prefix):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix)}
expected={item['name'] for item in ledger['objects']}|{'remote_objects.json','SUCCESS'}
if set(blobs)!=expected: raise SystemExit('terminal remote object set drifted')
for item in ledger['objects']:
 blob=blobs[item['name']]
 if int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']:
  raise SystemExit(f"terminal payload identity drifted: {item['name']}")
for name,local in (('remote_objects.json',Path(sys.argv[3])),('SUCCESS',root/'SUCCESS')):
 checksum=google_crc32c.Checksum()
 with local.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): checksum.update(chunk)
 crc=base64.b64encode(checksum.digest()).decode('ascii'); blob=blobs[name]
 if int(blob.size)!=local.stat().st_size or blob.crc32c!=crc or not blob.generation:
  raise SystemExit(f'terminal {name} identity drifted')
PY
say "COMPLETE $(cat "$RUN_DIR/pack.log")"
