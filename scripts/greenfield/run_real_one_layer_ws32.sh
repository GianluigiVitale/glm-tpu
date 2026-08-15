#!/usr/bin/env bash
# Protected all-host WS32 layer-3 compiler acquisition (execution stays off).
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly ARTIFACT_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/layer3/WS32_2D/greenfield_ws32_one_layer_pack_20260815T070628458699950Z
readonly ARTIFACT_MANIFEST_SHA=4bf8679de10ebbba055e9d0be991495080388c6355fa449f393c28f4751e1f40
readonly ORACLE_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/layer3/greenfield_one_layer_oracle_20260805T162210370718434Z
readonly ORACLE_MANIFEST_SHA=c63ffa19820d5c2c39865ac8611fb313ffc6ebcd2f893c3507745a356bfdebff
readonly TOPOLOGY_ROOT=/home/gianl/gcs-models/results/greenfield_topology_20260805T125842425591441Z
readonly TOPOLOGY_CAPTURE_ROOT=$TOPOLOGY_ROOT/host_records
readonly TOPOLOGY_SHA=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly TOPOLOGY_FLEET_SHA=50de0729c9e5080c5ddb5ae4f5cd948317c53ce8a8f6c9f3f6064e7afc6515a0
readonly MESH_SHA=de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88
# Intentionally vacant for the one compile-only acquisition. The numerical
# mode is not authorized until the frozen TPU graphs are validated offline.
readonly EXPECTED_STABLEHLO_SHA=$(printf '0%.0s' {1..64})
readonly EXPECTED_OPTIMIZED_HLO_SHA=$(printf '0%.0s' {1..64})

[[ ${GLM_GREENFIELD_WS32_REAL_LAYER:-0} == 1 ]] || {
  echo "WS32 real-layer workflow is default-off; set GLM_GREENFIELD_WS32_REAL_LAYER=1" >&2
  exit 2
}
[[ ${GLM_GREENFIELD_WS32_REAL_LAYER_MODE:-acquire} == acquire ]] || {
  echo "only the reviewed compile-only acquisition is currently authorized" >&2
  exit 2
}

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_REAL_LAYER_TAG:-greenfield_ws32_real_layer_hlo_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
[[ ! -e $RUN_DIR ]]
mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/hlo" "$RUN_DIR/fleet" "$RUN_DIR/fleet_hlo"

say() {
  echo "[ws32-layer $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  local label=$1 out="$RUN_DIR/census_${label}.txt" carrier="${TAG}_${label}"
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[r]un_real_one_layer_ws32[.]py|[r]un_real_one_layer[.]py|[c]ompile_short_decoder[.]py|[m]icrobench_collectives[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; append-only diagnostics retained at $RUN_DIR and $REMOTE_PREFIX"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN artifact=$ARTIFACT_MANIFEST_SHA oracle=$ORACLE_MANIFEST_SHA"
remote_prefix_is_vacant() {
  local output="$RUN_DIR/remote_vacancy.stdout" error="$RUN_DIR/remote_vacancy.stderr" status
  if gcloud storage ls "$REMOTE_PREFIX/**" >"$output" 2>"$error"; then
    [[ ! -s $output ]]
    return
  else
    status=$?
  fi
  [[ $status -eq 1 && ! -s $output ]] &&
    [[ $(grep -c '^ERROR:' "$error") -eq 1 ]] &&
    [[ $(tail -n 1 "$error") == "ERROR: (gcloud.storage.ls) One or more URLs matched no objects." ]]
}
remote_prefix_is_vacant || {
  say "ABORT: remote prefix is not vacant"
  exit 1
}
strict_census pre || {
  say "ABORT: pre-run fleet census is not clean"
  exit 1
}

say "synchronizing code and immutable mounted inputs on all hosts"
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; wt='"$WORKTREE"'; artifact='"$ARTIFACT_ROOT"'; artifact_sha='"$ARTIFACT_MANIFEST_SHA"'; oracle='"$ORACLE_ROOT"'; oracle_sha='"$ORACLE_MANIFEST_SHA"'; topology_root='"$TOPOLOGY_CAPTURE_ROOT"'; topology="$topology_root/topology.rank${idx}.json"; topology_sha='"$TOPOLOGY_SHA"'; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; [[ -r "$artifact/SUCCESS" && -r "$artifact/manifest.json" && -r "$oracle/SUCCESS" && -r "$oracle/manifest.json" && -r "$oracle/oracle.safetensors" && -r "$topology" ]]; findmnt -T "$artifact" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse"; python3 - "$artifact/manifest.json" "$artifact_sha" "$oracle/manifest.json" "$oracle_sha" "$topology" "$topology_sha" "$idx" <<'"'"'PY'"'"'
import hashlib,json,socket,sys
a,ash,o,osh,t,tsh,idx=sys.argv[1:]
assert json.load(open(a))["manifest_sha256"]==ash
assert json.load(open(o))["manifest_sha256"]==osh
d=json.load(open(t)); assert d["contract"]["topology_hash"]==tsh
assert d["launch_process_id"]==int(idx) and d["hostname"]==socket.gethostname()
PY
echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: eight-host synchronization failed"
  exit 1
}

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I | awk '{print \$1}'" 2>/dev/null | tail -1 | tr -d '\r')
[[ -n $coordinator ]]
coordinator="$coordinator:8476"
say "launching compile-only WS32 graph acquisition coordinator=$coordinator"
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; coordinator='"$coordinator"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run/host_records" "$run/hlo"; output="$run/host_records/runner.rank${idx}.json"; pre="$run/host_records/prevalidation.rank${idx}.json"; stable="$run/hlo/layer.rank${idx}.stablehlo.mlir"; optimized="$run/hlo/layer.rank${idx}.optimized_hlo.txt"; log="$run/host_records/runner.rank${idx}.log"; upload() { gcloud storage cp --no-clobber "$output" "$pre" "$log" "$remote/host_records/" >/dev/null 2>&1 || true; gcloud storage cp --no-clobber "$stable" "$optimized" "$remote/hlo/" >/dev/null 2>&1 || true; }; trap upload EXIT; cd "$wt"; GLM_GREENFIELD_RUN_TAG="$tag" JAX_PLATFORMS=tpu PYTHONPATH="$wt" timeout --signal=TERM --kill-after=30 1800 /home/gianl/vllm-env/bin/python -u scripts/greenfield/run_real_one_layer_ws32.py --coordinator-address "$coordinator" --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --artifact-dir '"$ARTIFACT_ROOT"' --oracle-dir '"$ORACLE_ROOT"' --topology-capture-root '"$TOPOLOGY_CAPTURE_ROOT"' --expected-code-hash '"$PIN"' --packed-manifest-sha256 '"$ARTIFACT_MANIFEST_SHA"' --oracle-manifest-sha256 '"$ORACLE_MANIFEST_SHA"' --topology-sha256 '"$TOPOLOGY_SHA"' --topology-fleet-sha256 '"$TOPOLOGY_FLEET_SHA"' --mesh-sha256 '"$MESH_SHA"' --expected-stablehlo-sha256 '"$EXPECTED_STABLEHLO_SHA"' --expected-optimized-hlo-sha256 '"$EXPECTED_OPTIMIZED_HLO_SHA"' --compile-only 1 --output "$output" --prevalidation-output "$pre" --stablehlo-output "$stable" --optimized-hlo-output "$optimized" >"$log" 2>&1; trap - EXIT; upload; echo "WS32_ACQUIRE_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/launch.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/launch.txt" WS32_ACQUIRE_OK || {
  say "ABORT: all-host graph acquisition did not complete"
  exit 1
}

say "downloading and validating immutable fleet graphs"
gcloud storage cp "$REMOTE_PREFIX/host_records/*" "$RUN_DIR/fleet/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/fleet_hlo/" >/dev/null
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" \
  "$ARTIFACT_MANIFEST_SHA" "$ORACLE_MANIFEST_SHA" "$TOPOLOGY_SHA" \
  "$TOPOLOGY_FLEET_SHA" "$MESH_SHA" "$ARTIFACT_ROOT/manifest.json" \
  "$TOPOLOGY_CAPTURE_ROOT" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.benchmarking.ws32_one_layer import validate_ws32_topology_fleet
from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh

root=Path(sys.argv[1]); pin,artifact,oracle,topology,topology_fleet,mesh=sys.argv[2:8]
artifact_manifest=json.loads(Path(sys.argv[8]).read_text())
topology_root=Path(sys.argv[9])
topology_captures=tuple(json.loads((topology_root/f'topology.rank{rank}.json').read_text()) for rank in range(8))
physical_topology,launch_captures,observed_fleet=validate_ws32_topology_fleet(topology_captures,expected_topology_sha256=topology,expected_fleet_sha256=topology_fleet,slice_name='db-v4-64-od')
physical_mesh=build_ws32_physical_mesh(physical_topology)
if artifact_manifest['manifest_sha256'] != artifact or observed_fleet != topology_fleet or physical_mesh.mesh_hash != mesh:
    raise SystemExit('immutable input identities drifted during fleet validation')
files={item['device_slot']:item for item in artifact_manifest['files']}
devices={item['device_id']:item for item in physical_topology.to_dict()['devices']}
slot_by_device={device_id:slot for slot,device_id in enumerate(physical_mesh.flattened_device_ids)}
expected_pre_keys={
    'code_hash','compiled_memory_analysis','compile_seconds',
    'device_memory_after_load','device_memory_before_load','hlo','hostname',
    'jax_devices','jax_process_index','launch_process_id','load_seconds',
    'local_device_slots','mesh_sha256','oracle_manifest_sha256',
    'packed_manifest_sha256','topology_fleet_sha256','topology_sha256',
}
expected_hlo_keys={
    'entry_parameter_count','expert_reduce_count','feature_reduce_count',
    'live_collective_count','live_entry_parameter_count','maximum_group_size',
    'optimized_hlo_collective_count','optimized_hlo_sha256','passed',
    'stablehlo_collective_count','stablehlo_sha256','violations',
}
expected_memory_keys={
    'alias_size_in_bytes','argument_size_in_bytes','generated_code_size_in_bytes',
    'output_size_in_bytes','temp_size_in_bytes',
}
records=[]
for rank in range(8):
    runner=json.loads((root/'fleet'/f'runner.rank{rank}.json').read_text())
    pre=json.loads((root/'fleet'/f'prevalidation.rank{rank}.json').read_text())
    if set(pre) != expected_pre_keys or set(pre['hlo']) != expected_hlo_keys or set(pre['compiled_memory_analysis']) != expected_memory_keys:
        raise SystemExit(f'prevalidation schema mismatch rank {rank}')
    if runner != {**pre,'artifact_kind':'greenfield_ws32_real_layer3_hlo_acquisition','performance_claim':False,'schema_version':1,'status':'HLO_ACQUIRED'}:
        raise SystemExit(f'runner/prevalidation mismatch rank {rank}')
    capture=launch_captures[rank]
    if (runner['code_hash'],runner['hostname'],runner['launch_process_id'],runner['jax_process_index']) != (pin,capture['hostname'],rank,capture['jax_process_index']):
        raise SystemExit(f'fleet identity mismatch rank {rank}')
    if (runner['packed_manifest_sha256'],runner['oracle_manifest_sha256'],runner['topology_sha256'],runner['topology_fleet_sha256'],runner['mesh_sha256']) != (artifact,oracle,topology,topology_fleet,mesh):
        raise SystemExit(f'input provenance mismatch rank {rank}')
    if type(runner['compile_seconds']) is not float or runner['compile_seconds'] <= 0 or type(runner['load_seconds']) is not float or runner['load_seconds'] <= 0:
        raise SystemExit(f'invalid duration record rank {rank}')
    expected_devices=sorted((item for item in devices.values() if item['process_index']==capture['jax_process_index']),key=lambda item:item['local_device_id'])
    if runner['jax_devices'] != expected_devices:
        raise SystemExit(f'JAX/topology device mismatch rank {rank}')
    expected_slots=[]
    for device in expected_devices:
        slot=slot_by_device[device['device_id']]
        file_record=files[slot]
        expected_slots.append({
            'device_id':device['device_id'],'device_slot':slot,
            'expert_coordinate':slot//4,'feature_coordinate':slot%4,
            'file_sha256':file_record['sha256'],
        })
    if runner['local_device_slots'] != expected_slots:
        raise SystemExit(f'direct-loader slot/file binding mismatch rank {rank}')
    for field in ('device_memory_before_load','device_memory_after_load'):
        values=runner[field]
        if len(values)!=4 or any(value is not None and (not isinstance(value,dict) or any(type(number) is not int or number < 0 for number in value.values())) for value in values):
            raise SystemExit(f'invalid {field} rank {rank}')
    if any(value is not None and (type(value) is not int or value < 0) for value in runner['compiled_memory_analysis'].values()):
        raise SystemExit(f'invalid compiled memory analysis rank {rank}')
    hlo=runner['hlo']
    if (hlo['stablehlo_collective_count'],hlo['optimized_hlo_collective_count'],hlo['feature_reduce_count'],hlo['expert_reduce_count'],hlo['maximum_group_size'],hlo['entry_parameter_count'],hlo['live_entry_parameter_count'],hlo['live_collective_count']) != (10,10,9,1,8,15,15,10):
        raise SystemExit(f'HLO structure mismatch rank {rank}')
    if hlo['passed'] is not False:
        raise SystemExit(f'acquisition unexpectedly passed vacant pins rank {rank}')
    if set(hlo['violations']) != {'StableHLO pin is intentionally vacant','optimized HLO pin is intentionally vacant'}:
        raise SystemExit(f'unexpected HLO violations rank {rank}: {hlo["violations"]}')
    stable=root/'fleet_hlo'/f'layer.rank{rank}.stablehlo.mlir'
    optimized=root/'fleet_hlo'/f'layer.rank{rank}.optimized_hlo.txt'
    if sha256(stable.read_bytes()).hexdigest()!=hlo['stablehlo_sha256'] or sha256(optimized.read_bytes()).hexdigest()!=hlo['optimized_hlo_sha256']:
        raise SystemExit(f'HLO file hash mismatch rank {rank}')
    records.append(runner)
stable={r['hlo']['stablehlo_sha256'] for r in records}; optimized={r['hlo']['optimized_hlo_sha256'] for r in records}
if len(stable)!=1 or len(optimized)!=1:
    raise SystemExit('fleet compiler hashes disagree')
summary={'artifact_kind':'greenfield_ws32_real_layer3_hlo_acquisition_summary','code_hash':pin,'fleet_records':8,'mesh_sha256':mesh,'optimized_hlo_sha256':next(iter(optimized)),'oracle_manifest_sha256':oracle,'packed_manifest_sha256':artifact,'performance_claim':False,'stablehlo_sha256':next(iter(stable)),'status':'HLO_ACQUIRED','topology_fleet_sha256':topology_fleet,'topology_sha256':topology}
(root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
PY

strict_census post || {
  say "ABORT: post-run fleet census is not clean"
  exit 1
}
post_census_done=1
sha256sum "$RUN_DIR/summary.json" "$RUN_DIR/census_pre.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/sync.txt" "$RUN_DIR/launch.txt" >"$RUN_DIR/evidence.sha256"
gcloud storage cp --no-clobber "$RUN_DIR/summary.json" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/census_post.txt" "$RUN_DIR/sync.txt" "$RUN_DIR/launch.txt" \
  "$RUN_DIR/evidence.sha256" "$RUN_DIR/orchestrator.log" "$REMOTE_PREFIX/" >/dev/null
say "HLO acquisition complete; no arithmetic, DB row, performance claim, or terminal SUCCESS"
trap - EXIT
