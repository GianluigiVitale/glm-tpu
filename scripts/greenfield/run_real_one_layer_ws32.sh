#!/usr/bin/env bash
# Protected all-host WS32 layer-3 numerical discriminator.
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
readonly EXPECTED_STABLEHLO_SHA=dea1384e0b6548c9ab81b5bafb84aceb3353cce1db7f5e21282742990295c01f
readonly EXPECTED_OPTIMIZED_HLO_SHA=6a6acf94b92a0ad355fc824d3f62d58b562d9bb252e7bbcbafba1bead2f3e157

[[ ${GLM_GREENFIELD_WS32_REAL_LAYER:-0} == 1 ]] || {
  echo "WS32 real-layer workflow is default-off; set GLM_GREENFIELD_WS32_REAL_LAYER=1" >&2
  exit 2
}
[[ ${GLM_GREENFIELD_WS32_REAL_LAYER_MODE:-off} == numerical ]] || {
  echo "only explicit mode=numerical is authorized by the reviewed TPU graph pins" >&2
  exit 2
}

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_REAL_LAYER_TAG:-greenfield_ws32_real_layer_numerical_$(date -u +%Y%m%dT%H%M%S%NZ)}
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
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[r]un_real_one_layer_ws32[.]py|[r]un_real_one_layer[.]py|[c]ompile_short_decoder[.]py|[m]icrobench_collectives[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
terminal_publication_started=0
terminal_success_verified=0
rollback_remote_success() {
  local metadata="$RUN_DIR/success_rollback.describe.json"
  local ownership="$RUN_DIR/success_upload.json" generation=""
  [[ -f $RUN_DIR/SUCCESS && -f $ownership ]] || {
    echo "SUCCESS rollback ownership record is missing" >&2
    return 1
  }
  if gcloud storage objects describe "$REMOTE_PREFIX/SUCCESS" --format=json \
    >"$metadata" 2>"$RUN_DIR/success_rollback.describe.stderr"; then
    generation=$(/home/gianl/vllm-env/bin/python -c \
      'import base64,json,sys; from pathlib import Path; import google_crc32c; path,ownership,metadata,remote=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]; own=json.loads(ownership.read_text()); value=json.loads(metadata.read_text()); checksum=google_crc32c.Checksum(path.read_bytes()); crc=base64.b64encode(checksum.digest()).decode("ascii"); assert isinstance(own,dict) and set(own)=={"crc32c","generation","remote","size"}; assert type(own["generation"]) is str and own["generation"] and type(own["size"]) is int and own["size"]>=0; assert own=={"crc32c":crc,"generation":own["generation"],"remote":remote,"size":path.stat().st_size}; assert str(value.get("generation"))==own["generation"] and int(value["size"])==own["size"] and value.get("crc32c_hash")==crc; print(own["generation"])' \
      "$RUN_DIR/SUCCESS" "$ownership" "$metadata" "$REMOTE_PREFIX/SUCCESS") || generation=""
  fi
  [[ -n $generation ]] || {
    echo "SUCCESS rollback could not authenticate the owned generation" >&2
    return 1
  }
  gcloud storage rm --if-generation-match="$generation" \
    "$REMOTE_PREFIX/SUCCESS" >"$RUN_DIR/success_rollback.rm.stdout" \
    2>"$RUN_DIR/success_rollback.rm.stderr"
}
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    if [[ $terminal_publication_started -eq 1 && $terminal_success_verified -eq 0 ]]; then
      if rollback_remote_success; then
        say "FAILED status=$status after terminal publication; authenticated SUCCESS generation removed and no later remote diagnostic mutation allowed"
      else
        say "FAILED status=$status after terminal publication; SUCCESS ownership could not be authenticated, no object was deleted and no later remote diagnostic mutation is allowed"
      fi
    else
      say "FAILED status=$status; append-only diagnostics retained at $RUN_DIR and $REMOTE_PREFIX"
      gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
    fi
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
# shellcheck disable=SC2016
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
say "launching reviewed WS32 numerical discriminator coordinator=$coordinator"
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; coordinator='"$coordinator"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run/host_records" "$run/hlo"; output="$run/host_records/runner.rank${idx}.json"; pre="$run/host_records/prevalidation.rank${idx}.json"; stable="$run/hlo/layer.rank${idx}.stablehlo.mlir"; optimized="$run/hlo/layer.rank${idx}.optimized_hlo.txt"; log="$run/host_records/runner.rank${idx}.log"; upload() { [[ ! -f $output ]] || gcloud storage cp --no-clobber "$output" "$remote/host_records/" >/dev/null 2>&1 || true; [[ ! -f $pre ]] || gcloud storage cp --no-clobber "$pre" "$remote/host_records/" >/dev/null 2>&1 || true; [[ ! -f $log ]] || gcloud storage cp --no-clobber "$log" "$remote/host_records/" >/dev/null 2>&1 || true; [[ ! -f $stable ]] || gcloud storage cp --no-clobber "$stable" "$remote/hlo/" >/dev/null 2>&1 || true; [[ ! -f $optimized ]] || gcloud storage cp --no-clobber "$optimized" "$remote/hlo/" >/dev/null 2>&1 || true; }; trap upload EXIT; cd "$wt"; GLM_GREENFIELD_RUN_TAG="$tag" JAX_PLATFORMS=tpu PYTHONPATH="$wt" timeout --signal=TERM --kill-after=30 1800 /home/gianl/vllm-env/bin/python -u scripts/greenfield/run_real_one_layer_ws32.py --coordinator-address "$coordinator" --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --artifact-dir '"$ARTIFACT_ROOT"' --oracle-dir '"$ORACLE_ROOT"' --topology-capture-root '"$TOPOLOGY_CAPTURE_ROOT"' --expected-code-hash '"$PIN"' --packed-manifest-sha256 '"$ARTIFACT_MANIFEST_SHA"' --oracle-manifest-sha256 '"$ORACLE_MANIFEST_SHA"' --topology-sha256 '"$TOPOLOGY_SHA"' --topology-fleet-sha256 '"$TOPOLOGY_FLEET_SHA"' --mesh-sha256 '"$MESH_SHA"' --expected-stablehlo-sha256 '"$EXPECTED_STABLEHLO_SHA"' --expected-optimized-hlo-sha256 '"$EXPECTED_OPTIMIZED_HLO_SHA"' --compile-only 0 --warmup 2 --iterations 5 --output "$output" --prevalidation-output "$pre" --stablehlo-output "$stable" --optimized-hlo-output "$optimized" >"$log" 2>&1; trap - EXIT; upload; echo "WS32_LAYER_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/launch.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/launch.txt" WS32_LAYER_OK || {
  say "ABORT: all-host WS32 numerical discriminator did not complete"
  exit 1
}

say "downloading and independently validating numerical fleet evidence"
gcloud storage cp "$REMOTE_PREFIX/host_records/*" "$RUN_DIR/fleet/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/fleet_hlo/" >/dev/null
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" \
  "$ARTIFACT_MANIFEST_SHA" "$ORACLE_MANIFEST_SHA" "$TOPOLOGY_SHA" \
  "$TOPOLOGY_FLEET_SHA" "$MESH_SHA" "$ARTIFACT_ROOT/manifest.json" \
  "$TOPOLOGY_CAPTURE_ROOT" "$ORACLE_ROOT" "$EXPECTED_STABLEHLO_SHA" \
  "$EXPECTED_OPTIMIZED_HLO_SHA" "$TAG" <<'PY'
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

import ml_dtypes
import numpy as np
from safetensors import safe_open
import torch

from glm_tpu.greenfield.benchmarking import (
    REAL_LAYER_OUTPUT_TOLERANCE,
    compare_bounded_tensor,
    latency_distribution,
    validate_ws32_one_layer_hlo,
    validate_ws32_topology_fleet,
)
from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh
from glm_tpu.greenfield.validation import inspect_one_layer_oracle

root=Path(sys.argv[1]); pin,artifact,oracle,topology,topology_fleet,mesh=sys.argv[2:8]
artifact_manifest=json.loads(Path(sys.argv[8]).read_text())
topology_root=Path(sys.argv[9])
oracle_root=Path(sys.argv[10]); stable_pin,optimized_pin,tag=sys.argv[11:14]
topology_captures=tuple(json.loads((topology_root/f'topology.rank{rank}.json').read_text()) for rank in range(8))
physical_topology,launch_captures,observed_fleet=validate_ws32_topology_fleet(topology_captures,expected_topology_sha256=topology,expected_fleet_sha256=topology_fleet,slice_name='db-v4-64-od')
physical_mesh=build_ws32_physical_mesh(physical_topology)
if artifact_manifest['manifest_sha256'] != artifact or observed_fleet != topology_fleet or physical_mesh.mesh_hash != mesh:
    raise SystemExit('immutable input identities drifted during fleet validation')
oracle_manifest=inspect_one_layer_oracle(oracle_root)
if oracle_manifest['manifest_sha256'] != oracle:
    raise SystemExit('oracle manifest identity drifted during fleet validation')
oracle_path=oracle_root/oracle_manifest['file']['filename']
with safe_open(oracle_path,framework='pt',device='cpu') as handle:
    oracle_outputs={}
    for case in ('normal','concentrated'):
        tensor=handle.get_tensor(f'{case}_output')
        if str(tensor.dtype) != 'torch.bfloat16' or list(tensor.shape) != [1,6144]:
            raise SystemExit(f'oracle output metadata drifted for {case}')
        oracle_outputs[case]=np.ascontiguousarray(tensor.view(torch.uint16).numpy())
files={item['device_slot']:item for item in artifact_manifest['files']}
devices={item['device_id']:item for item in physical_topology.to_dict()['devices']}
slot_by_device={device_id:slot for slot,device_id in enumerate(physical_mesh.flattened_device_ids)}
def exact_equal(left,right):
    if type(left) is not type(right):
        return False
    if isinstance(left,dict):
        return set(left)==set(right) and all(exact_equal(left[key],right[key]) for key in left)
    if isinstance(left,list):
        return len(left)==len(right) and all(exact_equal(a,b) for a,b in zip(left,right))
    return left==right

def valid_memory_record(value, *, require_full_reservable_limit):
    keys={
        'bytes_in_use','bytes_limit','bytes_reservable_limit','bytes_reserved',
        'largest_alloc_size','largest_free_block_bytes','num_allocs',
        'peak_bytes_in_use','peak_bytes_reserved',
    }
    return (
        isinstance(value,dict) and set(value)==keys
        and all(type(number) is int and number>=0 for number in value.values())
        and value['bytes_limit']==33014398976
        and 0<value['bytes_reservable_limit']<=value['bytes_limit']
        and (
            not require_full_reservable_limit
            or value['bytes_reservable_limit']==value['bytes_limit']
        )
        and value['bytes_in_use']<=value['peak_bytes_in_use']<=value['bytes_limit']
        and value['bytes_reserved']<=value['peak_bytes_reserved']<=value['bytes_reservable_limit']
        and value['largest_alloc_size']<=value['peak_bytes_in_use']
        and value['largest_free_block_bytes']<=value['bytes_reservable_limit']
    )

expected_pre_keys={
    'code_hash','compiled_memory_analysis','compile_seconds',
    'device_memory_after_load','device_memory_before_load','hlo','hostname',
    'jax_devices','jax_process_index','launch_process_id','load_seconds',
    'local_device_slots','mesh_sha256','oracle_manifest_sha256',
    'packed_manifest_sha256','topology_fleet_sha256','topology_sha256',
}
expected_hlo_keys={
    'bf16_result_reduce_count','entry_parameter_count','expert_reduce_count',
    'f32_operand_reduce_count','f32_result_reduce_count','feature_reduce_count',
    'live_collective_count','live_entry_parameter_count','maximum_group_size',
    'optimized_hlo_collective_count','optimized_hlo_sha256','passed',
    'stablehlo_collective_count','stablehlo_sha256','violations',
}
expected_runner_keys=expected_pre_keys|{
    'artifact_kind','correctness','device_memory_after_execute',
    'performance_claim','schema_version','status','timing',
}
expected_memory_keys={
    'alias_size_in_bytes','argument_size_in_bytes','generated_code_size_in_bytes',
    'output_size_in_bytes','temp_size_in_bytes',
}
expected_compiled_memory={
    'alias_size_in_bytes':0,'argument_size_in_bytes':311753728,
    'generated_code_size_in_bytes':23690240,'output_size_in_bytes':6144,
    'temp_size_in_bytes':34843648,
}
expected_shard_keys={
    'bitwise_mismatch_count','comparison','device_id','device_slot',
    'expert_coordinate','feature_coordinate','observed_bf16_bits',
    'observed_bf16_sha256','oracle_bf16_sha256',
}
expected_timing_keys={
    'iterations','latency','performance_claim','profiler_active','samples_ms','warmup',
}
records=[]
all_slots={case:set() for case in ('normal','concentrated')}
all_samples={case:[] for case in ('normal','concentrated')}
case_mismatches={case:0 for case in ('normal','concentrated')}
case_max_abs={case:0.0 for case in ('normal','concentrated')}
after_execute=[]
for rank in range(8):
    runner=json.loads((root/'fleet'/f'runner.rank{rank}.json').read_text())
    pre=json.loads((root/'fleet'/f'prevalidation.rank{rank}.json').read_text())
    if set(pre)!=expected_pre_keys or set(runner)!=expected_runner_keys or set(pre['hlo'])!=expected_hlo_keys or set(pre['compiled_memory_analysis'])!=expected_memory_keys:
        raise SystemExit(f'prevalidation schema mismatch rank {rank}')
    if not exact_equal({key:runner[key] for key in expected_pre_keys},pre):
        raise SystemExit(f'runner/prevalidation fields mismatch rank {rank}')
    if runner['artifact_kind']!='greenfield_ws32_real_layer3' or runner['performance_claim'] is not False or runner['schema_version']!=1 or runner['status']!='SUCCESS':
        raise SystemExit(f'runner terminal contract mismatch rank {rank}')
    capture=launch_captures[rank]
    if (runner['code_hash'],runner['hostname'],runner['launch_process_id'],runner['jax_process_index']) != (pin,capture['hostname'],rank,capture['jax_process_index']):
        raise SystemExit(f'fleet identity mismatch rank {rank}')
    if (runner['packed_manifest_sha256'],runner['oracle_manifest_sha256'],runner['topology_sha256'],runner['topology_fleet_sha256'],runner['mesh_sha256']) != (artifact,oracle,topology,topology_fleet,mesh):
        raise SystemExit(f'input provenance mismatch rank {rank}')
    if type(runner['compile_seconds']) is not float or not math.isfinite(runner['compile_seconds']) or runner['compile_seconds']<=0 or type(runner['load_seconds']) is not float or not math.isfinite(runner['load_seconds']) or runner['load_seconds']<=0:
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
    for field in ('device_memory_before_load','device_memory_after_load','device_memory_after_execute'):
        values=runner[field]
        require_full_reservable_limit=field!='device_memory_after_execute'
        if not isinstance(values,list) or len(values)!=4 or any(
            not valid_memory_record(
                value,
                require_full_reservable_limit=require_full_reservable_limit,
            )
            for value in values
        ):
            raise SystemExit(f'invalid {field} rank {rank}')
    after_execute.extend(runner['device_memory_after_execute'])
    if not exact_equal(runner['compiled_memory_analysis'],expected_compiled_memory):
        raise SystemExit(f'compiled memory analysis drifted rank {rank}')
    hlo=runner['hlo']
    if (hlo['stablehlo_collective_count'],hlo['optimized_hlo_collective_count'],hlo['feature_reduce_count'],hlo['expert_reduce_count'],hlo['f32_operand_reduce_count'],hlo['bf16_result_reduce_count'],hlo['f32_result_reduce_count'],hlo['maximum_group_size'],hlo['entry_parameter_count'],hlo['live_entry_parameter_count'],hlo['live_collective_count']) != (10,10,9,1,10,10,0,8,15,15,10):
        raise SystemExit(f'HLO structure mismatch rank {rank}')
    stable=root/'fleet_hlo'/f'layer.rank{rank}.stablehlo.mlir'
    optimized=root/'fleet_hlo'/f'layer.rank{rank}.optimized_hlo.txt'
    stable_text=stable.read_text(); optimized_text=optimized.read_text()
    if sha256(stable_text.encode()).hexdigest()!=stable_pin or sha256(optimized_text.encode()).hexdigest()!=optimized_pin:
        raise SystemExit(f'HLO file hash mismatch rank {rank}')
    observed_report=validate_ws32_one_layer_hlo(stable_text,optimized_text,expected_stablehlo_sha256=stable_pin,expected_optimized_hlo_sha256=optimized_pin,expected_optimized_collective_result_dtype='bf16').to_dict()
    if not exact_equal(hlo,observed_report) or hlo['passed'] is not True or hlo['violations']!=[]:
        raise SystemExit(f'HLO report mismatch rank {rank}')
    correctness=runner['correctness']
    if not isinstance(correctness,dict) or set(correctness)!={'normal','concentrated'}:
        raise SystemExit(f'correctness case schema mismatch rank {rank}')
    for case in ('normal','concentrated'):
        result=correctness[case]
        if not isinstance(result,dict) or set(result)!={'case','local_shards','passed'} or result['case']!=case or result['passed'] is not True or not isinstance(result['local_shards'],list) or len(result['local_shards'])!=4:
            raise SystemExit(f'correctness result mismatch rank {rank}:{case}')
        expected_local_slots={item['device_slot'] for item in expected_slots}
        if {item.get('device_slot') for item in result['local_shards']}!=expected_local_slots:
            raise SystemExit(f'correctness slot ownership mismatch rank {rank}:{case}')
        for shard in result['local_shards']:
            if not isinstance(shard,dict) or set(shard)!=expected_shard_keys:
                raise SystemExit(f'correctness shard schema mismatch rank {rank}:{case}')
            slot=shard['device_slot']; device_id=physical_mesh.flattened_device_ids[slot]
            expert_coordinate,feature_coordinate=divmod(slot,4)
            if (shard['device_id'],shard['expert_coordinate'],shard['feature_coordinate'])!=(device_id,expert_coordinate,feature_coordinate):
                raise SystemExit(f'correctness shard identity mismatch rank {rank}:{case}:{slot}')
            bits=shard['observed_bf16_bits']
            if not isinstance(bits,list) or len(bits)!=1536 or any(type(value) is not int or not 0<=value<=65535 for value in bits):
                raise SystemExit(f'observed BF16 payload mismatch rank {rank}:{case}:{slot}')
            observed_bits=np.asarray(bits,dtype=np.uint16).reshape(1,1536)
            start=feature_coordinate*1536
            expected_bits=np.ascontiguousarray(oracle_outputs[case][:,start:start+1536])
            observed_sha=sha256(observed_bits.tobytes()).hexdigest()
            expected_sha=sha256(expected_bits.tobytes()).hexdigest()
            mismatch=int(np.count_nonzero(observed_bits!=expected_bits))
            comparison=compare_bounded_tensor(observed_bits.view(ml_dtypes.bfloat16),expected_bits.view(ml_dtypes.bfloat16),REAL_LAYER_OUTPUT_TOLERANCE)
            if shard['observed_bf16_sha256']!=observed_sha or shard['oracle_bf16_sha256']!=expected_sha or type(shard['bitwise_mismatch_count']) is not int or shard['bitwise_mismatch_count']!=mismatch or not exact_equal(shard['comparison'],comparison) or comparison['passed'] is not True:
                raise SystemExit(f'independent oracle recomputation mismatch rank {rank}:{case}:{slot}')
            if slot in all_slots[case]:
                raise SystemExit(f'duplicate correctness slot {case}:{slot}')
            all_slots[case].add(slot); case_mismatches[case]+=mismatch
            case_max_abs[case]=max(case_max_abs[case],comparison['error']['max_abs'])
    timing=runner['timing']
    if not isinstance(timing,dict) or set(timing)!={'normal','concentrated'}:
        raise SystemExit(f'timing schema mismatch rank {rank}')
    for case in ('normal','concentrated'):
        value=timing[case]
        if not isinstance(value,dict) or set(value)!=expected_timing_keys or value['iterations']!=5 or value['warmup']!=2 or value['performance_claim'] is not False or value['profiler_active'] is not False:
            raise SystemExit(f'timing contract mismatch rank {rank}:{case}')
        samples=value['samples_ms']
        if not isinstance(samples,list) or len(samples)!=5 or any(type(item) is not float or not math.isfinite(item) or item<=0 for item in samples):
            raise SystemExit(f'timing samples mismatch rank {rank}:{case}')
        recomputed=latency_distribution(samples).to_dict()
        if not exact_equal(value['latency'],recomputed):
            raise SystemExit(f'timing distribution mismatch rank {rank}:{case}')
        all_samples[case].extend(samples)
    records.append(runner)
stable={r['hlo']['stablehlo_sha256'] for r in records}; optimized={r['hlo']['optimized_hlo_sha256'] for r in records}
if stable!={stable_pin} or optimized!={optimized_pin} or any(all_slots[case]!=set(range(32)) for case in all_slots):
    raise SystemExit('fleet compiler hashes disagree')
summary={
    'artifact_kind':'greenfield_ws32_real_layer3_summary',
    'code_hash':pin,
    'compiled_memory_analysis':expected_compiled_memory,
    'correctness':{
        case:{
            'bounded_comparison_passed':True,
            'maximum_abs_error':case_max_abs[case],
            'replicated_shard_count':32,
            'total_bitwise_mismatch_count':case_mismatches[case],
        } for case in ('normal','concentrated')
    },
    'diagnostic_only':True,
    'fleet_records':8,
    'hlo':{
        'bf16_result_reduce_count':10,'expert_reduce_count':1,
        'f32_operand_reduce_count':10,'f32_result_reduce_count':0,
        'feature_reduce_count':9,'live_collective_count':10,
        'maximum_group_size':8,'optimized_hlo_sha256':optimized_pin,
        'stablehlo_sha256':stable_pin,
    },
    'measured_memory':{
        'maximum_peak_bytes_in_use_per_chip':max(value['peak_bytes_in_use'] for value in after_execute),
        'minimum_largest_free_block_bytes_per_chip':min(value['largest_free_block_bytes'] for value in after_execute),
        'records':len(after_execute),
    },
    'mesh_sha256':mesh,
    'oracle_manifest_sha256':oracle,
    'packed_manifest_sha256':artifact,
    'performance_claim':False,
    'schema_version':1,
    'status':'SUCCESS',
    'tag':tag,
    'timing_diagnostic':{case:latency_distribution(all_samples[case]).to_dict() for case in ('normal','concentrated')},
    'topology_fleet_sha256':topology_fleet,
    'topology_sha256':topology,
}
payload=json.dumps(summary,allow_nan=False,indent=2,sort_keys=True)+'\n'
temporary=root/f'.summary.json.tmp.{os.getpid()}'
with temporary.open('x') as stream:
    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
temporary.replace(root/'summary.json')
PY

strict_census post || {
  say "ABORT: post-run fleet census is not clean"
  exit 1
}
post_census_done=1
say "numerical evidence validated; sealing append-only archive"
(
  cd "$RUN_DIR"
  sha256sum summary.json census_pre.txt census_post.txt sync.txt launch.txt \
    remote_vacancy.stdout remote_vacancy.stderr orchestrator.log
  sha256sum fleet/* fleet_hlo/*
) >"$RUN_DIR/evidence.sha256"
gcloud storage cp --no-clobber "$RUN_DIR/summary.json" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/census_post.txt" "$RUN_DIR/sync.txt" "$RUN_DIR/launch.txt" \
  "$RUN_DIR/remote_vacancy.stdout" "$RUN_DIR/remote_vacancy.stderr" \
  "$RUN_DIR/evidence.sha256" "$RUN_DIR/orchestrator.log" \
  "$REMOTE_PREFIX/" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" <<'PY' >"$RUN_DIR/remote_objects.json"
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root,remote=Path(sys.argv[1]),sys.argv[2]
objects=[
    ('summary.json',root/'summary.json'),
    ('census_pre.txt',root/'census_pre.txt'),
    ('census_post.txt',root/'census_post.txt'),
    ('sync.txt',root/'sync.txt'),
    ('launch.txt',root/'launch.txt'),
    ('remote_vacancy.stdout',root/'remote_vacancy.stdout'),
    ('remote_vacancy.stderr',root/'remote_vacancy.stderr'),
    ('evidence.sha256',root/'evidence.sha256'),
    ('orchestrator.log',root/'orchestrator.log'),
]
for rank in range(8):
    objects.extend([
        (f'host_records/runner.rank{rank}.json',root/'fleet'/f'runner.rank{rank}.json'),
        (f'host_records/prevalidation.rank{rank}.json',root/'fleet'/f'prevalidation.rank{rank}.json'),
        (f'host_records/runner.rank{rank}.log',root/'fleet'/f'runner.rank{rank}.log'),
        (f'hlo/layer.rank{rank}.stablehlo.mlir',root/'fleet_hlo'/f'layer.rank{rank}.stablehlo.mlir'),
        (f'hlo/layer.rank{rank}.optimized_hlo.txt',root/'fleet_hlo'/f'layer.rank{rank}.optimized_hlo.txt'),
    ])

def crc32c(path):
    checksum=google_crc32c.Checksum()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode('ascii')

records=[]
for name,path in objects:
    value=json.loads(subprocess.run(
        ['gcloud','storage','objects','describe',f'{remote}/{name}','--format=json'],
        check=True,capture_output=True,text=True,
    ).stdout)
    local_crc=crc32c(path)
    if int(value['size'])!=path.stat().st_size or value.get('crc32c_hash')!=local_crc or not value.get('generation'):
        raise SystemExit(f'remote WS32 numerical object identity drifted for {name}')
    records.append({'crc32c':local_crc,'generation':value['generation'],'name':name,'size':path.stat().st_size})
print(json.dumps({'objects':records},indent=2,sort_keys=True))
PY

gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" "$TAG" "$ARTIFACT_MANIFEST_SHA" \
  "$ORACLE_MANIFEST_SHA" "$TOPOLOGY_SHA" "$TOPOLOGY_FLEET_SHA" "$MESH_SHA" \
  "$EXPECTED_STABLEHLO_SHA" "$EXPECTED_OPTIMIZED_HLO_SHA" <<'PY'
from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import google_crc32c

root=Path(sys.argv[1]); remote,pin,tag,artifact,oracle,topology,topology_fleet,mesh,stable,optimized=sys.argv[2:]
summary=json.loads((root/'summary.json').read_text())
ledger_path=root/'remote_objects.json'
checksum=google_crc32c.Checksum(ledger_path.read_bytes())
ledger_crc=base64.b64encode(checksum.digest()).decode('ascii')
remote_ledger=json.loads(subprocess.run(
    ['gcloud','storage','objects','describe',f'{remote}/remote_objects.json','--format=json'],
    check=True,capture_output=True,text=True,
).stdout)
if int(remote_ledger['size'])!=ledger_path.stat().st_size or remote_ledger.get('crc32c_hash')!=ledger_crc or type(remote_ledger.get('generation')) is not str or not remote_ledger['generation']:
    raise SystemExit('remote WS32 numerical ledger identity drifted')
local_objects=[
    ('summary.json',root/'summary.json'),
    ('census_pre.txt',root/'census_pre.txt'),
    ('census_post.txt',root/'census_post.txt'),
    ('sync.txt',root/'sync.txt'),
    ('launch.txt',root/'launch.txt'),
    ('remote_vacancy.stdout',root/'remote_vacancy.stdout'),
    ('remote_vacancy.stderr',root/'remote_vacancy.stderr'),
    ('evidence.sha256',root/'evidence.sha256'),
    ('orchestrator.log',root/'orchestrator.log'),
]
for rank in range(8):
    local_objects.extend([
        (f'host_records/runner.rank{rank}.json',root/'fleet'/f'runner.rank{rank}.json'),
        (f'host_records/prevalidation.rank{rank}.json',root/'fleet'/f'prevalidation.rank{rank}.json'),
        (f'host_records/runner.rank{rank}.log',root/'fleet'/f'runner.rank{rank}.log'),
        (f'hlo/layer.rank{rank}.stablehlo.mlir',root/'fleet_hlo'/f'layer.rank{rank}.stablehlo.mlir'),
        (f'hlo/layer.rank{rank}.optimized_hlo.txt',root/'fleet_hlo'/f'layer.rank{rank}.optimized_hlo.txt'),
    ])
ledger=json.loads(ledger_path.read_text())
if not isinstance(ledger,dict) or set(ledger)!={'objects'} or not isinstance(ledger['objects'],list) or len(ledger['objects'])!=49:
    raise SystemExit('WS32 numerical ledger schema/count drifted')
expected_names=[name for name,_ in local_objects]
if [record.get('name') if isinstance(record,dict) else None for record in ledger['objects']]!=expected_names or len(set(expected_names))!=49:
    raise SystemExit('WS32 numerical ledger names/order drifted')

def crc32c(path):
    checksum=google_crc32c.Checksum()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode('ascii')

for record,(name,path) in zip(ledger['objects'],local_objects):
    if set(record)!={'crc32c','generation','name','size'} or type(record['name']) is not str or type(record['crc32c']) is not str or type(record['generation']) is not str or not record['generation'] or type(record['size']) is not int or record['size']<0:
        raise SystemExit(f'WS32 numerical ledger record schema drifted for {name}')
    local_crc=crc32c(path)
    remote_value=json.loads(subprocess.run(
        ['gcloud','storage','objects','describe',f'{remote}/{name}','--format=json'],
        check=True,capture_output=True,text=True,
    ).stdout)
    generation=remote_value.get('generation')
    expected_record={'crc32c':local_crc,'generation':generation,'name':name,'size':path.stat().st_size}
    if type(generation) is not str or not generation or int(remote_value['size'])!=path.stat().st_size or remote_value.get('crc32c_hash')!=local_crc or record!=expected_record:
        raise SystemExit(f'WS32 numerical ledger/local/remote identity drifted for {name}')
expected_nonterminal=set(expected_names)|{'remote_objects.json'}
listing=subprocess.run(
    ['gcloud','storage','objects','list',f'{remote}/**','--format=value(name)'],
    check=True,capture_output=True,text=True,
).stdout.splitlines()
bucket_prefix=remote.removeprefix('gs://').split('/',1)[1].rstrip('/')+'/'
observed={line.removeprefix(bucket_prefix) for line in listing if line.startswith(bucket_prefix)}
if observed!=expected_nonterminal:
    raise SystemExit(f'remote WS32 numerical object set drifted: {sorted(observed^expected_nonterminal)}')
expected_summary={
    'code_hash':pin,'mesh_sha256':mesh,'oracle_manifest_sha256':oracle,
    'packed_manifest_sha256':artifact,'tag':tag,
    'topology_fleet_sha256':topology_fleet,'topology_sha256':topology,
}
if any(summary.get(key)!=value for key,value in expected_summary.items()) or summary.get('status')!='SUCCESS' or summary.get('performance_claim') is not False or summary.get('diagnostic_only') is not True:
    raise SystemExit('WS32 numerical summary identity drifted before SUCCESS')
if summary['hlo']['stablehlo_sha256']!=stable or summary['hlo']['optimized_hlo_sha256']!=optimized:
    raise SystemExit('WS32 numerical HLO identity drifted before SUCCESS')
success={
    'artifact_kind':'greenfield_ws32_real_layer3_success',
    'code_hash':pin,
    'correctness':summary['correctness'],
    'diagnostic_only':True,
    'format_version':1,
    'measured_memory':summary['measured_memory'],
    'mesh_sha256':mesh,
    'nonterminal_object_count':len(observed),
    'optimized_hlo_sha256':optimized,
    'oracle_manifest_sha256':oracle,
    'packed_manifest_sha256':artifact,
    'performance_claim':False,
    'remote_objects_crc32c':ledger_crc,
    'remote_objects_generation':remote_ledger['generation'],
    'remote_objects_sha256':sha256(ledger_path.read_bytes()).hexdigest(),
    'stablehlo_sha256':stable,
    'tag':tag,
    'topology_fleet_sha256':topology_fleet,
    'topology_sha256':topology,
}
payload=json.dumps(success,indent=2,sort_keys=True)+'\n'
temporary=root/f'.SUCCESS.tmp.{os.getpid()}'
with temporary.open('x') as stream:
    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
temporary.replace(root/'SUCCESS')
PY

terminal_publication_started=1
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" "$RUN_DIR/success_upload.json" <<'PY'
import base64
import json
import os
from pathlib import Path
import sys

from google.cloud import storage
import google_crc32c

path,remote,record_path=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
if not remote.startswith('gs://') or '/' not in remote[5:]:
    raise SystemExit('invalid remote SUCCESS URL')
bucket_name,object_name=remote[5:].split('/',1)
checksum=google_crc32c.Checksum(path.read_bytes())
local_crc=base64.b64encode(checksum.digest()).decode('ascii')
blob=storage.Client().bucket(bucket_name).blob(object_name)
blob.upload_from_filename(str(path),if_generation_match=0,checksum='crc32c',content_type='application/json')
generation=str(blob.generation or '')
if not generation:
    raise SystemExit('SUCCESS upload returned no owned generation')
record={'crc32c':local_crc,'generation':generation,'remote':remote,'size':path.stat().st_size}
payload=json.dumps(record,indent=2,sort_keys=True)+'\n'
temporary=record_path.with_name(f'.{record_path.name}.tmp.{os.getpid()}')
with temporary.open('x') as stream:
    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
temporary.replace(record_path)
blob.reload(if_generation_match=int(generation))
if str(blob.generation)!=generation or int(blob.size)!=path.stat().st_size or blob.crc32c!=local_crc:
    raise SystemExit('owned remote SUCCESS upload metadata drifted')
PY

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/SUCCESS" "$RUN_DIR/success_upload.json" "$REMOTE_PREFIX" <<'PY'
import base64
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

path,ownership_path,remote=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
ownership=json.loads(ownership_path.read_text())
checksum=google_crc32c.Checksum(path.read_bytes())
local_crc=base64.b64encode(checksum.digest()).decode('ascii')
value=json.loads(subprocess.run(
    ['gcloud','storage','objects','describe',f'{remote}/SUCCESS','--format=json'],
    check=True,capture_output=True,text=True,
).stdout)
if not isinstance(ownership,dict) or set(ownership)!={'crc32c','generation','remote','size'} or type(ownership['generation']) is not str or not ownership['generation'] or type(ownership['size']) is not int:
    raise SystemExit('local WS32 numerical SUCCESS ownership drifted')
if ownership!={'crc32c':local_crc,'generation':ownership['generation'],'remote':f'{remote}/SUCCESS','size':path.stat().st_size}:
    raise SystemExit('local WS32 numerical SUCCESS ownership drifted')
if int(value['size'])!=path.stat().st_size or value.get('crc32c_hash')!=local_crc or str(value.get('generation'))!=ownership['generation']:
    raise SystemExit('remote WS32 numerical SUCCESS identity drifted')
success=json.loads(path.read_text())
if success['nonterminal_object_count']!=50:
    raise SystemExit('WS32 numerical terminal object count drifted')
print(json.dumps({'remote_success_crc32c':local_crc,'remote_success_generation':value['generation'],'terminal_object_count':51},sort_keys=True))
PY

terminal_success_verified=1
trap - EXIT
echo "SUCCESS tag=$TAG code=$PIN; protected WS32 layer correctness/HBM diagnostic, no DB or performance claim"
