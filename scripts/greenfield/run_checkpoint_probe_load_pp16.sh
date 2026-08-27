#!/usr/bin/env bash
# Protected two-chip device round-trip of the bounded PP16 checkpoint probe.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly PROBE_TAG=greenfield_checkpoint_probe_pp16_20260827T024902158913509Z
readonly PROBE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/probes/PP16_LP2/$PROBE_TAG
readonly PROBE_MANIFEST_SHA=14c36aeb356b9856bd0fa2040ba2b3b2923909cda194afb9d41ab8af011ac4e3
readonly PROBE_MANIFEST_FILE_SHA=f235510cbdfcb1d16d534420cbaacb80283f8bd330c920743cd31443c8470454
readonly PROBE_SUCCESS_FILE_SHA=0e6880d7ff80fc840fea0c15de4030febbd8146c8254d9f1506e463ec0149945
readonly PLAN_TAG=greenfield_checkpoint_plan_pp16_20260827T022537742669498Z
readonly LAYOUT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP16_LP2/$PLAN_TAG/layout_manifest.json
readonly TOPOLOGY=/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records/topology.rank0.json
readonly SOURCE_REVISION=gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658

cd "$WORKTREE"
[[ ${GLM_GREENFIELD_PP16_CHECKPOINT_PROBE_LOAD:-0} == 1 ]] || {
  echo "PP16 checkpoint probe device load is default-off" >&2
  exit 2
}
PIN=$(git rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_CHECKPOINT_PROBE_LOAD_TAG:-greenfield_checkpoint_probe_load_pp16_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_checkpoint_probe_load_pp16_[0-9]{8}T[0-9]{15}Z$ ]]
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX

[[ $(git branch --show-current) == "$BRANCH" ]]
[[ -z $(git status --porcelain) ]]
[[ ! -e $RUN_DIR ]]
[[ -r $PROBE_ROOT/manifest.json && -r $PROBE_ROOT/SUCCESS && -r $LAYOUT && -r $TOPOLOGY ]]
[[ $(sha256sum "$PROBE_ROOT/manifest.json" | awk '{print $1}') == "$PROBE_MANIFEST_FILE_SHA" ]]
[[ $(sha256sum "$PROBE_ROOT/SUCCESS" | awk '{print $1}') == "$PROBE_SUCCESS_FILE_SHA" ]]
mkdir -p "$RUN_DIR"
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || { echo "another protected TPU workflow holds the lease" >&2; exit 1; }
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

say() {
  echo "[checkpoint-probe-load-pp16 $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
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
  local command
  command='generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[l]oad_checkpoint_probe[.]py|[r]un_real_one_layer[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}}" 2>/dev/null); if [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; echo "$generic"; echo "$holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then strict_census failure_exit || true; fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving nonterminal diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "PIN=$PIN TAG=$TAG"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == \
  "$APPROVED_LOCATION" ]]
pod_state=$(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" \
  --format='value(state,health)')
[[ $pod_state == $'READY\tHEALTHY' ]]
gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' \
  >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]]
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/preflight.json" "$PIN" "$TAG" "$REMOTE_PREFIX" <<'PY'
import json,sys
from pathlib import Path
value={'approved_bucket':'gs://driftbench-dsv4-uc','bucket_location':'US-CENTRAL2','code_hash':sys.argv[2],
 'performance_claim':False,'plan_id':'PP16_LP2','probe_manifest_sha256':'14c36aeb356b9856bd0fa2040ba2b3b2923909cda194afb9d41ab8af011ac4e3',
 'remote_prefix':sys.argv[4],'run_tag':sys.argv[3]}
Path(sys.argv[1]).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY

strict_census pre || { say "ABORT: pre-run census is not 8/8 clean"; exit 1; }
say "loading two final owners onto adjacent TPU v4 chips and round-tripping all bytes"
JAX_PLATFORMS=tpu TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$WORKTREE" \
  timeout --signal=TERM --kill-after=30 600 \
  /home/gianl/vllm-env/bin/python scripts/greenfield/load_checkpoint_probe.py \
    --checkpoint-root "$PROBE_ROOT" --layout-manifest "$LAYOUT" \
    --topology-capture "$TOPOLOGY" --destination "$APPROVED_BUCKET/checkpoints/greenfield/glm52/probes/PP16_LP2/$PROBE_TAG" \
    --expected-code-hash "$PIN" --expected-manifest-sha256 "$PROBE_MANIFEST_SHA" \
    --output "$RUN_DIR/runner.json" --state-manifest-output "$RUN_DIR/state.json" \
    >"$RUN_DIR/runner.log" 2>&1
strict_census post || { say "ABORT: post-run census is not 8/8 clean"; exit 1; }
post_census_done=1

say "validating state/HBM and linking the protected correctness record"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$SOURCE_REVISION" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

run=Path(sys.argv[1]); pin,db_path,repo,revision=sys.argv[2:]
runner=json.loads((run/'runner.json').read_text()); state=json.loads((run/'state.json').read_text())
if runner['status']!='SUCCESS' or runner['code_hash']!=pin or runner['performance_claim']:
 raise SystemExit('probe load runner identity failed')
unhashed={k:v for k,v in state.items() if k!='manifest_sha256'}
canonical=json.dumps(unhashed,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
if state['manifest_sha256']!=sha256(canonical).hexdigest() or state['manifest_sha256']!=runner['state_manifest_sha256']:
 raise SystemExit('probe load state identity failed')
load=runner['load']
if load['loaded_payload_bytes']!=50347904 or load['device_roundtrip_bytes']!=50347904 or load['loaded_tensor_count']!=16:
 raise SystemExit('probe load byte/tensor totals failed')
if not load['device_roundtrip_verified'] or any(load[k] for k in ('fp8_device_dequantizations','fp8_host_dequantizations','host_global_concatenations','runtime_checkpoint_reshards')):
 raise SystemExit('probe direct-load contract failed')
if runner['captured_device_ids']!=[0,1] or state['device_ids']!=[0,1] or len(state['files'])!=2:
 raise SystemExit('probe two-owner physical identity failed')
memory=runner['device_memory_after_load']
if len(memory)!=2 or any(item is None for item in memory): raise SystemExit('probe HBM stats missing')
peaks=[item['peak_bytes_in_use'] for item in memory]; largest=[item['largest_free_block_bytes'] for item in memory]
if any(peak<=0 for peak in peaks) or any(value<=0 for value in largest): raise SystemExit('probe HBM stats invalid')
sys.path.insert(0,str(Path(repo)/'bench')); import provenance as pv
conn=pv.connect(db_path)
run_id=pv.start_run(conn,model='zai-org/GLM-5.2-FP8:greenfield-pp16-checkpoint-probe-load',revision=revision,
 env={'GLM_ENGINE':'greenfield','GLM_EXECUTION_PLAN':'PP16_LP2','greenfield_code_hash':pin,
      'probe_manifest_sha256':runner['manifest_sha256'],'topology_hash':runner['topology_hash']},
 note='Bounded complete-layout-derived two-chip raw device-load proof; no performance claim.',harness_repo=repo)
pv.record_item(conn,run_id,benchmark='greenfield_checkpoint_probe_load_pp16',item_id='stage_00',
 prompt='Direct-load both bounded PP16 final owners and round-trip every device byte.',
 gold='Two adjacent owners, exact raw bytes, no dequantization/concat/reshard.',raw_output=json.dumps(runner,sort_keys=True),
 extracted=state['manifest_sha256'],correct=True,score=1.0)
pv.finalize(conn,run_id,benchmark='greenfield_checkpoint_probe_load_pp16',metric='contract_valid',value=1.0,
 note='Correctness/load evidence only; no latency or throughput claim.'); conn.close()
summary={'artifact_kind':'greenfield_checkpoint_probe_load_proof','code_hash':pin,'device_ids':[0,1],
 'device_roundtrip_bytes':50347904,'loaded_tensor_count':16,'manifest_sha256':runner['manifest_sha256'],
 'maximum_peak_hbm_bytes':max(peaks),'minimum_largest_free_block_bytes':min(largest),'performance_claim':False,
 'plan_id':'PP16_LP2','results_db_run_id':run_id,'state_manifest_sha256':state['manifest_sha256'],'status':'SUCCESS'}
(run/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
source=sqlite3.connect(db_path); snapshot=sqlite3.connect(run/'results_ckpt.db'); source.backup(snapshot); snapshot.close(); source.close()
if sqlite3.connect(run/'results_ckpt.db').execute('pragma integrity_check').fetchone()[0]!='ok': raise SystemExit('DB snapshot failed')
PY
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  sha256sum runner.json state.json runner.log summary.json results_ckpt.db \
    preflight.json census_pre.txt census_post.txt orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
(cd "$RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)

say "publishing nonterminal protected load evidence"
for name in runner.json state.json runner.log summary.json results_ckpt.db preflight.json \
  census_pre.txt census_post.txt orchestrator.sealed.log evidence.sha256; do
  gcloud storage cp --no-clobber "$RUN_DIR/$name" "$REMOTE_PREFIX/$name" >/dev/null
done

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys
import google_crc32c
from google.cloud import storage
run=Path(sys.argv[1]); remote=sys.argv[2]
names=['runner.json','state.json','runner.log','summary.json','results_ckpt.db','preflight.json','census_pre.txt','census_post.txt','orchestrator.sealed.log','evidence.sha256']
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blobs={b.name.removeprefix(prefix):b for b in storage.Client().list_blobs(bucket_name,prefix=prefix)}
if set(blobs)!=set(names): raise SystemExit('load preterminal remote set drifted')
records=[]
for name in names:
 path=run/name; crc=base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode('ascii'); blob=blobs[name]
 if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation: raise SystemExit(f'load remote identity drifted: {name}')
 records.append({'crc32c':crc,'generation':int(blob.generation),'name':name,'size':int(blob.size)})
ledger={'objects':records,'remote_prefix':remote}; canonical=json.dumps(ledger,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode(); ledger['ledger_sha256']=sha256(canonical).hexdigest()
(run/'remote_objects.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" "$REMOTE_PREFIX/remote_objects.json" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$TAG" "$REMOTE_PREFIX" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys
import google_crc32c
from google.cloud import storage
run=Path(sys.argv[1]); pin,tag,remote=sys.argv[2:]; summary=json.loads((run/'summary.json').read_text()); ledger_path=run/'remote_objects.json'; ledger=json.loads(ledger_path.read_text())
canonical=lambda v:json.dumps(v,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
without={k:v for k,v in ledger.items() if k!='ledger_sha256'}
if ledger['ledger_sha256']!=sha256(canonical(without)).hexdigest(): raise SystemExit('load ledger self-hash drifted')
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blob=storage.Client().bucket(bucket_name).blob(prefix+'remote_objects.json'); blob.reload(); crc=base64.b64encode(google_crc32c.Checksum(ledger_path.read_bytes()).digest()).decode('ascii')
if int(blob.size)!=ledger_path.stat().st_size or blob.crc32c!=crc or not blob.generation: raise SystemExit('load remote ledger identity drifted')
value={'artifact_kind':'greenfield_checkpoint_probe_load_SUCCESS','code_hash':pin,'performance_claim':False,
 'probe_manifest_sha256':summary['manifest_sha256'],'remote_ledger_crc32c':crc,'remote_ledger_generation':int(blob.generation),
 'remote_ledger_sha256':ledger['ledger_sha256'],'results_db_run_id':summary['results_db_run_id'],'run_tag':tag,
 'state_manifest_sha256':summary['state_manifest_sha256']}; value['success_sha256']=sha256(canonical(value)).hexdigest()
(run/'SUCCESS').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" <<'PY'
import base64
from hashlib import sha256
import json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
run=Path(sys.argv[1]); remote=sys.argv[2]; ledger=json.loads((run/'remote_objects.json').read_text()); success=json.loads((run/'SUCCESS').read_text())
canonical=lambda v:json.dumps(v,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
without={k:v for k,v in success.items() if k!='success_sha256'}
if success['success_sha256']!=sha256(canonical(without)).hexdigest(): raise SystemExit('load SUCCESS self-hash drifted')
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blobs={b.name.removeprefix(prefix):b for b in storage.Client().list_blobs(bucket_name,prefix=prefix)}
expected={item['name'] for item in ledger['objects']}|{'remote_objects.json','SUCCESS'}
if set(blobs)!=expected: raise SystemExit('load terminal remote set drifted')
for item in ledger['objects']:
 blob=blobs[item['name']]
 if int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']: raise SystemExit(f"load terminal payload drifted: {item['name']}")
for name in ('remote_objects.json','SUCCESS'):
 path=run/name; crc=base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode('ascii'); blob=blobs[name]
 if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation: raise SystemExit(f'load terminal identity drifted: {name}')
PY
trap - EXIT
db_id=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' \
  "$RUN_DIR/summary.json")
echo "[checkpoint-probe-load-pp16 $(date -u +%H:%M:%S)] COMPLETE DB=$db_id"
