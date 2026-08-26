#!/usr/bin/env bash
# Protected bounded real layer-0 StrategyND acquisition/numerical proof.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly CHECKPOINT_TAG=greenfield_ws32_strategy_nd_layer0_pack_20260826T232037240198985Z
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/discriminators/WS32_2D/$CHECKPOINT_TAG
readonly CHECKPOINT_MANIFEST_FILE_SHA=0f475ded23edfcc8933a11543bef85c94e0a6ab25dad761905bad47ad4424512
readonly CHECKPOINT_SUCCESS_FILE_SHA=c13045921444739fb4c3218c2f5381280d5f45757b8b3624688c2c210e36840d
readonly SOURCE_TAG=greenfield_layer0_dense_partial_capture_20260813T200736889447458Z
readonly SOURCE_PATH=/home/gianl/gcs-models/results/$SOURCE_TAG/dense_partial_capture.npz
readonly SOURCE_FILE_SHA=f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298
readonly TOPOLOGY_ROOT=/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records
readonly TOPOLOGY_SHA=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly TOPOLOGY_FLEET_SHA=4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301
readonly MESH_SHA=de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88
readonly ZERO_SHA=0000000000000000000000000000000000000000000000000000000000000000

cd "$WORKTREE"
[[ ${GLM_GREENFIELD_WS32_STRATEGY_ND_LAYER0:-0} == 1 ]] || {
  echo "WS32 StrategyND real layer-0 proof is default-off" >&2
  exit 2
}
MODE=${GLM_GREENFIELD_WS32_STRATEGY_ND_LAYER0_MODE:-off}
[[ $MODE == acquire || $MODE == numerical ]] || {
  echo "layer proof mode must be acquire or numerical" >&2
  exit 2
}
if [[ $MODE == acquire ]]; then
  STABLE_PIN=$ZERO_SHA
  OPTIMIZED_PIN=$ZERO_SHA
else
  STABLE_PIN=${GLM_GREENFIELD_WS32_STRATEGY_ND_LAYER0_STABLEHLO_SHA:?set acquired StableHLO SHA}
  OPTIMIZED_PIN=${GLM_GREENFIELD_WS32_STRATEGY_ND_LAYER0_OPTIMIZED_HLO_SHA:?set acquired optimized HLO SHA}
fi
[[ $STABLE_PIN =~ ^[0-9a-f]{64}$ && $OPTIMIZED_PIN =~ ^[0-9a-f]{64}$ ]]

PIN=$(git rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_STRATEGY_ND_LAYER0_TAG:-greenfield_ws32_strategy_nd_layer0_${MODE}_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_ws32_strategy_nd_layer0_${MODE}_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid append-only layer tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
ARCHIVE_DIR=$RUN_DIR/archive
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly MODE STABLE_PIN OPTIMIZED_PIN PIN TAG RUN_DIR ARCHIVE_DIR REMOTE_PREFIX

[[ $(git rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git branch --show-current) == "$BRANCH" ]]
[[ -z $(git status --porcelain) ]]
[[ ! -e $RUN_DIR ]]
mkdir -p "$RUN_DIR/fleet" "$RUN_DIR/fleet_hlo" "$ARCHIVE_DIR"

say() {
  echo "[ws32-strategy-nd-layer0 $(date -u +%H:%M:%S)] $*" |
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
  local command
  # shellcheck disable=SC2016
  command='tools=1; command -v pgrep >/dev/null || tools=0; command -v fuser >/dev/null || tools=0; sudo -n true >/dev/null 2>&1 || tools=0; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[p]robe_ws32_strategy_nd_layer0[.]py|[p]robe_ws32_strategy_nd[.]py|[r]un_short_decoder_ws32[.]py|[c]ompile_short_decoder[.]py" || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [ "$tools" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected workflow holds the fleet lease"
  exit 1
}
post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then strict_census failure_exit || true; fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; append-only diagnostics retained"
    for path in orchestrator.log census_pre.txt census_failure_exit.txt \
      remote_vacancy.txt sync.txt launch.txt; do
      [[ ! -f $RUN_DIR/$path ]] || gcloud storage cp --no-clobber \
        "$RUN_DIR/$path" "$REMOTE_PREFIX/diagnostic_local/$path" \
        >/dev/null 2>&1 || true
    done
  fi
}
trap on_exit EXIT

say "PIN=$PIN mode=$MODE checkpoint_bytes=700728992 source_bytes=862828"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]] || {
  say "ABORT: exact commit is not pushed"
  exit 1
}
[[ $(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" --format='value(state)') == READY ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == US-CENTRAL2 ]]
gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]]
[[ $(sha256sum "$CHECKPOINT_ROOT/manifest.json" | awk '{print $1}') == "$CHECKPOINT_MANIFEST_FILE_SHA" ]]
[[ $(sha256sum "$CHECKPOINT_ROOT/SUCCESS" | awk '{print $1}') == "$CHECKPOINT_SUCCESS_FILE_SHA" ]]
[[ $(sha256sum "$SOURCE_PATH" | awk '{print $1}') == "$SOURCE_FILE_SHA" ]]
for path in "$CHECKPOINT_ROOT" "$SOURCE_PATH" "$TOPOLOGY_ROOT"; do
  findmnt -T "$path" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse"
done
strict_census pre || {
  say "ABORT: pre-run census is not clean"
  exit 1
}

say "synchronizing pushed code and validating same-region mounted evidence"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; wt='"$WORKTREE"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; checkpoint='"$CHECKPOINT_ROOT"'; source='"$SOURCE_PATH"'; topology='"$TOPOLOGY_ROOT"'/topology.rank${idx}.json; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" && $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; elif [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; [[ $(sha256sum "$checkpoint/manifest.json" | awk '\''{print $1}'\'') == '"$CHECKPOINT_MANIFEST_FILE_SHA"' ]]; [[ $(sha256sum "$checkpoint/SUCCESS" | awk '\''{print $1}'\'') == '"$CHECKPOINT_SUCCESS_FILE_SHA"' ]]; [[ $(sha256sum "$source" | awk '\''{print $1}'\'') == '"$SOURCE_FILE_SHA"' ]]; [[ -r $topology ]]; for path in "$checkpoint" "$source" "$topology"; do findmnt -T "$path" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse"; done; echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I | awk '{print \$1}'" 2>/dev/null | tail -1 | tr -d '\r')
[[ -n $coordinator ]]
coordinator="$coordinator:8476"
say "launching bounded eight-host real layer coordinator=$coordinator"
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag; output="$run/runner.rank${idx}.json"; tensor="$run/tensor.rank${idx}.npz"; hlo="$run/hlo"; log="$run/runner.rank${idx}.log"; mkdir -p "$run"; upload(){ local rc=0; [[ ! -f $output ]] || gcloud storage cp --no-clobber "$output" "$remote/host_records/runner.rank${idx}.json" >/dev/null 2>&1 || rc=1; [[ ! -f $tensor ]] || gcloud storage cp --no-clobber "$tensor" "$remote/host_records/tensor.rank${idx}.npz" >/dev/null 2>&1 || rc=1; [[ ! -f $log ]] || gcloud storage cp --no-clobber "$log" "$remote/host_records/runner.rank${idx}.log" >/dev/null 2>&1 || rc=1; [[ ! -f $hlo/layer0.stablehlo.mlir ]] || gcloud storage cp --no-clobber "$hlo/layer0.stablehlo.mlir" "$remote/hlo/layer0.rank${idx}.stablehlo.mlir" >/dev/null 2>&1 || rc=1; [[ ! -f $hlo/layer0.optimized_hlo.txt ]] || gcloud storage cp --no-clobber "$hlo/layer0.optimized_hlo.txt" "$remote/hlo/layer0.rank${idx}.optimized_hlo.txt" >/dev/null 2>&1 || rc=1; return "$rc"; }; trap "upload || true" EXIT; cd "$wt"; env JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_MEM_FRACTION=.20 PYTHONPATH="$wt" GLM_GREENFIELD_RUN_TAG="$tag" timeout --signal=TERM --kill-after=30 900 /home/gianl/vllm-env/bin/python -u scripts/greenfield/probe_ws32_strategy_nd_layer0.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --topology-capture-root '"$TOPOLOGY_ROOT"' --topology-sha256 '"$TOPOLOGY_SHA"' --topology-fleet-sha256 '"$TOPOLOGY_FLEET_SHA"' --mesh-sha256 '"$MESH_SHA"' --expected-code-hash '"$PIN"' --checkpoint-root '"$CHECKPOINT_ROOT"' --source '"$SOURCE_PATH"' --mode '"$MODE"' --expected-stablehlo-sha256 '"$STABLE_PIN"' --expected-optimized-hlo-sha256 '"$OPTIMIZED_PIN"' --output "$output" --tensor-output "$tensor" --hlo-dir "$hlo" >"$log" 2>&1; trap - EXIT; upload; echo "PROBE_OK $(hostname) rank=$idx"'
launch_rc=0
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/launch.txt" 2>&1 || launch_rc=$?
if [[ $launch_rc -ne 0 ]] || ! has_eight_unique_markers "$RUN_DIR/launch.txt" PROBE_OK; then
  say "ABORT: bounded layer fleet did not finish 8/8"
  exit 1
fi

say "materializing all bounded fleet records and graph bytes"
for rank in {0..7}; do
  mkdir -p "$ARCHIVE_DIR/host_records" "$ARCHIVE_DIR/hlo"
  for suffix in json log; do
    gcloud storage cp "$REMOTE_PREFIX/host_records/runner.rank${rank}.${suffix}" \
      "$ARCHIVE_DIR/host_records/runner.rank${rank}.${suffix}" >/dev/null
  done
  if [[ $MODE == numerical ]]; then
    gcloud storage cp "$REMOTE_PREFIX/host_records/tensor.rank${rank}.npz" \
      "$ARCHIVE_DIR/host_records/tensor.rank${rank}.npz" >/dev/null
  fi
  for suffix in stablehlo.mlir optimized_hlo.txt; do
    gcloud storage cp "$REMOTE_PREFIX/hlo/layer0.rank${rank}.${suffix}" \
      "$ARCHIVE_DIR/hlo/layer0.rank${rank}.${suffix}" >/dev/null
  done
  cp "$ARCHIVE_DIR/host_records/runner.rank${rank}.json" "$RUN_DIR/fleet/runner.rank${rank}.json"
done
cp "$ARCHIVE_DIR/hlo/layer0.rank0.stablehlo.mlir" "$RUN_DIR/fleet_hlo/layer0.stablehlo.mlir"
cp "$ARCHIVE_DIR/hlo/layer0.rank0.optimized_hlo.txt" "$RUN_DIR/fleet_hlo/layer0.optimized_hlo.txt"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$TAG" "$MODE" "$PIN" "$STABLE_PIN" "$OPTIMIZED_PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]); tag,mode,pin,stable_pin,optimized_pin=sys.argv[2:]
records=[json.loads((root/'fleet'/f'runner.rank{i}.json').read_text()) for i in range(8)]
if {item['launch_process_id'] for item in records}!=set(range(8)) or {item['jax_process_index'] for item in records}!=set(range(8)):
 raise SystemExit('fleet process identity drifted')
singleton_names=(
 'checkpoint_manifest_sha256','code_hash','compiled_memory','expected_final_bits_sha256',
 'expected_partial_bits_sha256','execution_performed','final_bits_sha256','final_mismatch_count',
 'mesh_sha256','mismatch_count','mode','normalized_bits_sha256','optimized_hlo_sha256',
 'partial_bits_sha256','source_file_sha256','stablehlo_sha256','status',
 'topology_fleet_sha256','topology_sha256',
)
singletons={name:{json.dumps(item[name],sort_keys=True) for item in records} for name in singleton_names}
if any(len(values)!=1 for values in singletons.values()): raise SystemExit(f'fleet records disagree: {singletons}')
first=records[0]
if first['code_hash']!=pin or first['mode']!=mode or any(not item['hlo_contract']['passed'] for item in records):
 raise SystemExit('run identity or HLO contract drifted')
owners=[]
for item in records:
 if len(item['local_checkpoint_records'])!=4 or len(item['device_memory_after'])!=4:
  raise SystemExit('local device record cardinality drifted')
 owners.extend((record['expert_coordinate'],record['feature_coordinate']) for record in item['local_checkpoint_records'])
if set(owners)!={(expert,feature) for expert in range(8) for feature in range(4)} or len(owners)!=32:
 raise SystemExit('fleet checkpoint owner coverage drifted')
if mode=='acquire':
 if first['status']!='HLO_ACQUIRED' or first['execution_performed'] or first['mismatch_count'] is not None or any(item['tensor_file_sha256'] is not None for item in records):
  raise SystemExit('acquisition semantics drifted')
else:
 if stable_pin!=first['stablehlo_sha256'] or optimized_pin!=first['optimized_hlo_sha256']:
  raise SystemExit('numerical graph pin drifted')
 if first['status']!='PASS' or not first['execution_performed'] or first['mismatch_count']!=0 or first['final_mismatch_count']!=0:
  raise SystemExit('numerical layer proof refused')
 if any(not item['tensor_file_sha256'] for item in records):
  raise SystemExit('numerical raw tensor identity is missing')
 if first['partial_bits_sha256']!=first['expected_partial_bits_sha256'] or first['final_bits_sha256']!=first['expected_final_bits_sha256']:
  raise SystemExit('bounded layer tensor identity drifted')
 import numpy as np
 for rank in range(8):
  path=root/'archive'/'host_records'/f'tensor.rank{rank}.npz'
  if sha256(path.read_bytes()).hexdigest()!=records[rank]['tensor_file_sha256']:
   raise SystemExit('fleet raw tensor file identity drifted')
  with np.load(path,allow_pickle=False) as payload:
   if set(payload.files)!={'observed_final_bfloat16_bits','observed_partial_bfloat16_bits'}:
    raise SystemExit('raw tensor fields drifted')
   final=payload['observed_final_bfloat16_bits']; partial=payload['observed_partial_bfloat16_bits']
  if final.shape!=(1,6144) or partial.shape!=(32,1,6144) or final.dtype!=np.uint16 or partial.dtype!=np.uint16:
   raise SystemExit('raw tensor geometry drifted')
  if sha256(final.tobytes()).hexdigest()!=first['final_bits_sha256'] or sha256(partial.tobytes()).hexdigest()!=first['partial_bits_sha256']:
   raise SystemExit('raw tensor hash drifted')
stable_sha=sha256((root/'fleet_hlo'/'layer0.stablehlo.mlir').read_bytes()).hexdigest()
optimized_sha=sha256((root/'fleet_hlo'/'layer0.optimized_hlo.txt').read_bytes()).hexdigest()
if stable_sha!=first['stablehlo_sha256'] or optimized_sha!=first['optimized_hlo_sha256']:
 raise SystemExit('materialized graph identity drifted')
for rank in range(8):
 for suffix,expected in (('stablehlo.mlir',stable_sha),('optimized_hlo.txt',optimized_sha)):
  if sha256((root/'archive'/'hlo'/f'layer0.rank{rank}.{suffix}').read_bytes()).hexdigest()!=expected:
   raise SystemExit('fleet graph bytes drifted')
summary={
 'artifact_kind':'greenfield_ws32_strategy_nd_layer0_summary','checkpoint_manifest_sha256':first['checkpoint_manifest_sha256'],
 'code_hash':pin,'compiled_memory':first['compiled_memory'],'execution_performed':first['execution_performed'],
 'final_bits_sha256':first['final_bits_sha256'],'final_mismatch_count':first['final_mismatch_count'],
 'host_count':8,'maximum_collective_group_size':8,'mesh_sha256':first['mesh_sha256'],
 'mismatch_count':first['mismatch_count'],'mode':mode,'optimized_hlo_sha256':optimized_sha,
 'partial_bits_sha256':first['partial_bits_sha256'],'passed':mode=='numerical','performance_claim':False,
 'results_db_run_id':None,'schema_version':1,'source_file_sha256':first['source_file_sha256'],
 'stablehlo_sha256':stable_sha,'status':first['status'],'tag':tag,
 'topology_fleet_sha256':first['topology_fleet_sha256'],'topology_sha256':first['topology_sha256'],
}
canonical=lambda value: json.dumps(value,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
summary['summary_sha256']=sha256(canonical(summary)).hexdigest()
(root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
marker={'code_hash':pin,'mode':mode,'summary_file_sha256':sha256((root/'summary.json').read_bytes()).hexdigest(),'summary_sha256':summary['summary_sha256'],'tag':tag}
marker['marker_sha256']=sha256(canonical(marker)).hexdigest()
(root/('HLO_ACQUIRED' if mode=='acquire' else 'SUCCESS')).write_text(json.dumps(marker,indent=2,sort_keys=True)+'\n')
PY

strict_census post
post_census_done=1
mkdir -p "$ARCHIVE_DIR/orchestrator"
marker=HLO_ACQUIRED
[[ $MODE == acquire ]] || marker=SUCCESS
for name in orchestrator.log census_pre.txt census_post.txt sync.txt launch.txt summary.json; do
  cp "$RUN_DIR/$name" "$ARCHIVE_DIR/orchestrator/$name"
  gcloud storage cp --no-clobber "$RUN_DIR/$name" "$REMOTE_PREFIX/orchestrator/$name" >/dev/null
done

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$ARCHIVE_DIR" "$REMOTE_PREFIX" "$RUN_DIR/remote_objects.json" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys
import google_crc32c
from google.cloud import storage

root=Path(sys.argv[1]); remote=sys.argv[2]; output=Path(sys.argv[3])
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
expected={path.relative_to(root).as_posix():path for path in root.rglob('*') if path.is_file()}
blobs={blob.name.removeprefix(prefix):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix)}
if set(blobs)!=set(expected): raise SystemExit('preterminal remote object set drifted')
records=[]
for name,path in sorted(expected.items()):
 checksum=google_crc32c.Checksum()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): checksum.update(chunk)
 crc=base64.b64encode(checksum.digest()).decode(); blob=blobs[name]
 if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation:
  raise SystemExit(f'remote object identity drifted: {name}')
 records.append({'crc32c':crc,'generation':int(blob.generation),'name':name,'sha256':sha256(path.read_bytes()).hexdigest(),'size':int(blob.size)})
ledger={'objects':records,'remote_prefix':remote}
canonical=json.dumps(ledger,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
ledger['ledger_sha256']=sha256(canonical).hexdigest()
output.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" "$REMOTE_PREFIX/remote_objects.json" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/$marker" "$REMOTE_PREFIX/$marker" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$REMOTE_PREFIX" "$RUN_DIR/remote_objects.json" "$RUN_DIR/$marker" <<'PY'
import base64,json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
remote=sys.argv[1]; ledger=json.load(open(sys.argv[2])); marker=Path(sys.argv[3])
bucket_name,prefix=remote[5:].split('/',1); prefix=prefix.rstrip('/')+'/'
blobs={blob.name.removeprefix(prefix):blob for blob in storage.Client().list_blobs(bucket_name,prefix=prefix)}
expected={item['name'] for item in ledger['objects']}|{'remote_objects.json',marker.name}
if set(blobs)!=expected: raise SystemExit('terminal remote object set drifted')
for name,path in (('remote_objects.json',Path(sys.argv[2])),(marker.name,marker)):
 checksum=google_crc32c.Checksum(); checksum.update(path.read_bytes())
 crc=base64.b64encode(checksum.digest()).decode(); blob=blobs[name]
 if int(blob.size)!=path.stat().st_size or blob.crc32c!=crc or not blob.generation:
  raise SystemExit(f'terminal object identity drifted: {name}')
PY
say "COMPLETE mode=$MODE summary_sha=$(sha256sum "$RUN_DIR/summary.json" | awk '{print $1}') marker_sha=$(sha256sum "$RUN_DIR/$marker" | awk '{print $1}')"
trap - EXIT
