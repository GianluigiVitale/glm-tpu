#!/usr/bin/env bash
# Protected model-free all-host proof for the WS32 StrategyND dense reducer.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_TAG=greenfield_legacy_layer0_dense_partials_p8155_20260814T100132090917640Z
readonly SOURCE_REMOTE=$APPROVED_BUCKET/oracles/greenfield/glm52/dense_partials/8k/$SOURCE_TAG
readonly SOURCE_NPZ_REMOTE=$SOURCE_REMOTE/dense_partials_capture/dense_partials.npz
readonly SOURCE_NPZ_SHA=e5977248acbe7582db351178b3fc823c6f87db46f8b6143b763319a6b299582c
readonly TOPOLOGY_ROOT=/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records
readonly TOPOLOGY_SHA=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly TOPOLOGY_FLEET_SHA=4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301
readonly MESH_SHA=de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88
readonly ZERO_SHA=0000000000000000000000000000000000000000000000000000000000000000

[[ ${GLM_GREENFIELD_WS32_STRATEGY_ND_PROBE:-0} == 1 ]] || {
  echo "WS32 StrategyND probe is default-off" >&2
  exit 2
}
MODE=${GLM_GREENFIELD_WS32_STRATEGY_ND_MODE:-off}
[[ $MODE == acquire || $MODE == numerical ]] || {
  echo "probe mode must be acquire or numerical" >&2
  exit 2
}
if [[ $MODE == acquire ]]; then
  STABLE_PIN=$ZERO_SHA
  OPTIMIZED_PIN=$ZERO_SHA
else
  STABLE_PIN=${GLM_GREENFIELD_WS32_STRATEGY_ND_STABLEHLO_SHA:?set acquired StableHLO SHA}
  OPTIMIZED_PIN=${GLM_GREENFIELD_WS32_STRATEGY_ND_OPTIMIZED_HLO_SHA:?set acquired optimized HLO SHA}
fi
[[ $STABLE_PIN =~ ^[0-9a-f]{64}$ && $OPTIMIZED_PIN =~ ^[0-9a-f]{64}$ ]] || {
  echo "graph pins must be lowercase SHA-256 values" >&2
  exit 2
}

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_STRATEGY_ND_TAG:-greenfield_ws32_strategy_nd_${MODE}_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_ws32_strategy_nd_${MODE}_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid append-only tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly MODE STABLE_PIN OPTIMIZED_PIN PIN TAG RUN_DIR REMOTE_PREFIX

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
[[ ! -e $RUN_DIR ]]
mkdir -p "$RUN_DIR/fleet" "$RUN_DIR/fleet_hlo"

say() {
  echo "[ws32-strategy-nd $(date -u +%H:%M:%S)] $*" |
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
  command='tools=1; command -v pgrep >/dev/null || tools=0; command -v fuser >/dev/null || tools=0; sudo -n true >/dev/null 2>&1 || tools=0; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[p]robe_ws32_strategy_nd[.]py|[r]un_short_decoder_ws32[.]py|[c]ompile_short_decoder[.]py|[m]icrobench_collectives[.]py" || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [ "$tools" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
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

say "PIN=$PIN mode=$MODE input_bytes=787060"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]] || {
  say "ABORT: exact commit is not pushed"
  exit 1
}
[[ $(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" \
  --format='value(state)') == READY ]] || {
  say "ABORT: existing pod is not READY"
  exit 1
}
gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' \
  >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]] || {
  say "ABORT: remote prefix is not vacant"
  exit 1
}
[[ $(gcloud storage cat "$SOURCE_NPZ_REMOTE" | sha256sum | awk '{print $1}') == \
  "$SOURCE_NPZ_SHA" ]] || {
  say "ABORT: approved DB550 object identity drifted"
  exit 1
}
strict_census pre || {
  say "ABORT: pre-run census is not clean"
  exit 1
}

say "synchronizing exact code and the one sealed 787 KiB input"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; wt='"$WORKTREE"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; topology='"$TOPOLOGY_ROOT"'/topology.rank${idx}.json; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" && $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; elif [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; [[ -r $topology ]]; findmnt -T "$topology" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse"; echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host synchronization failed"
  exit 1
}

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" \
  --worker=0 --command="hostname -I | awk '{print \$1}'" 2>/dev/null |
  tail -1 | tr -d '\r')
[[ -n $coordinator ]]
coordinator="$coordinator:8476"
say "launching bounded eight-host probe coordinator=$coordinator"
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; source_remote='"$SOURCE_NPZ_REMOTE"'; source_sha='"$SOURCE_NPZ_SHA"'; run=/home/gianl/glm-run/$tag; input="$run/source/dense_partials.npz"; output="$run/runner.rank${idx}.json"; hlo="$run/hlo"; log="$run/runner.rank${idx}.log"; mkdir -p "$run/source"; gcloud storage cp "$source_remote" "$input" >/dev/null; [[ $(sha256sum "$input" | awk '\''{print $1}'\'') == "$source_sha" ]]; upload(){ local rc=0; [[ ! -f $output ]] || gcloud storage cp --no-clobber "$output" "$remote/host_records/runner.rank${idx}.json" >/dev/null 2>&1 || rc=1; [[ ! -f $log ]] || gcloud storage cp --no-clobber "$log" "$remote/host_records/runner.rank${idx}.log" >/dev/null 2>&1 || rc=1; [[ ! -f $hlo/strategy_nd.stablehlo.mlir ]] || gcloud storage cp --no-clobber "$hlo/strategy_nd.stablehlo.mlir" "$remote/hlo/strategy_nd.rank${idx}.stablehlo.mlir" >/dev/null 2>&1 || rc=1; [[ ! -f $hlo/strategy_nd.optimized_hlo.txt ]] || gcloud storage cp --no-clobber "$hlo/strategy_nd.optimized_hlo.txt" "$remote/hlo/strategy_nd.rank${idx}.optimized_hlo.txt" >/dev/null 2>&1 || rc=1; return "$rc"; }; trap "upload || true" EXIT; cd "$wt"; env JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_MEM_FRACTION=.20 PYTHONPATH="$wt" GLM_GREENFIELD_RUN_TAG="$tag" timeout --signal=TERM --kill-after=30 600 /home/gianl/vllm-env/bin/python -u scripts/greenfield/probe_ws32_strategy_nd.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --topology-capture-root '"$TOPOLOGY_ROOT"' --topology-sha256 '"$TOPOLOGY_SHA"' --topology-fleet-sha256 '"$TOPOLOGY_FLEET_SHA"' --mesh-sha256 '"$MESH_SHA"' --expected-code-hash '"$PIN"' --input "$input" --input-file-sha256 "$source_sha" --mode '"$MODE"' --expected-stablehlo-sha256 '"$STABLE_PIN"' --expected-optimized-hlo-sha256 '"$OPTIMIZED_PIN"' --output "$output" --hlo-dir "$hlo" >"$log" 2>&1; trap - EXIT; upload; echo "PROBE_OK $(hostname) rank=$idx"'
launch_rc=0
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/launch.txt" 2>&1 || launch_rc=$?
if [[ $launch_rc -ne 0 ]] || \
  ! has_eight_unique_markers "$RUN_DIR/launch.txt" PROBE_OK; then
  say "ABORT: bounded worker fleet did not finish 8/8"
  exit 1
fi

say "materializing and validating the small fleet record"
for rank in {0..7}; do
  gcloud storage cp \
    "$REMOTE_PREFIX/host_records/runner.rank${rank}.json" \
    "$RUN_DIR/fleet/runner.rank${rank}.json" >/dev/null
done
gcloud storage cp "$REMOTE_PREFIX/hlo/strategy_nd.rank0.stablehlo.mlir" \
  "$RUN_DIR/fleet_hlo/strategy_nd.stablehlo.mlir" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/strategy_nd.rank0.optimized_hlo.txt" \
  "$RUN_DIR/fleet_hlo/strategy_nd.optimized_hlo.txt" >/dev/null
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$TAG" "$MODE" "$PIN" "$STABLE_PIN" "$OPTIMIZED_PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]); tag,mode,pin,stable_pin,optimized_pin=sys.argv[2:]
records=[json.loads((root/'fleet'/f'runner.rank{i}.json').read_text()) for i in range(8)]
if {item['launch_process_id'] for item in records} != set(range(8)):
 raise SystemExit('launch-process identities drifted')
if {item['jax_process_index'] for item in records} != set(range(8)):
 raise SystemExit('JAX-process identities drifted')
singleton_names=(
 'code_hash','input_array_sha256','input_file_sha256','mesh_sha256',
 'optimized_hlo_sha256','stablehlo_sha256','topology_fleet_sha256',
 'topology_sha256','status','mode','expected_output_bfloat16_sha256',
 'observed_output_bfloat16_sha256','mismatch_count','execution_performed',
)
singletons={name:{json.dumps(item[name],sort_keys=True) for item in records} for name in singleton_names}
if any(len(values)!=1 for values in singletons.values()):
 raise SystemExit(f'fleet records disagree: {singletons}')
first=records[0]
if first['code_hash']!=pin or first['mode']!=mode:
 raise SystemExit('run identity drifted')
if any(not item['hlo_contract']['passed'] for item in records):
 raise SystemExit('HLO contract refused')
if mode=='acquire':
 if first['status']!='HLO_ACQUIRED' or first['execution_performed'] or any(item['passed'] for item in records):
  raise SystemExit('acquisition semantics drifted')
else:
 if stable_pin!=first['stablehlo_sha256'] or optimized_pin!=first['optimized_hlo_sha256']:
  raise SystemExit('numerical graph pin drifted')
 if first['status']!='PASS' or not first['execution_performed'] or not all(item['passed'] for item in records):
  raise SystemExit('numerical proof refused')
 if first['mismatch_count']!=0 or first['observed_output_bfloat16_sha256']!=first['expected_output_bfloat16_sha256']:
  raise SystemExit('sealed row mismatch')
stable_path=root/'fleet_hlo'/'strategy_nd.stablehlo.mlir'
optimized_path=root/'fleet_hlo'/'strategy_nd.optimized_hlo.txt'
stable_sha=sha256(stable_path.read_bytes()).hexdigest()
optimized_sha=sha256(optimized_path.read_bytes()).hexdigest()
if stable_sha!=first['stablehlo_sha256'] or optimized_sha!=first['optimized_hlo_sha256']:
 raise SystemExit('materialized HLO identity drifted')
summary={
 'artifact_kind':'greenfield_ws32_strategy_nd_model_free_summary',
 'code_hash':pin,
 'execution_performed':first['execution_performed'],
 'host_count':8,
 'input_array_sha256':first['input_array_sha256'],
 'input_file_sha256':first['input_file_sha256'],
 'maximum_collective_group_size':8,
 'mesh_sha256':first['mesh_sha256'],
 'mismatch_count':first['mismatch_count'],
 'mode':mode,
 'optimized_hlo_sha256':optimized_sha,
 'output_bfloat16_sha256':first['observed_output_bfloat16_sha256'],
 'passed':mode=='numerical',
 'performance_claim':False,
 'results_db_run_id':None,
 'schema_version':1,
 'stablehlo_sha256':stable_sha,
 'status':first['status'],
 'tag':tag,
 'topology_fleet_sha256':first['topology_fleet_sha256'],
 'topology_sha256':first['topology_sha256'],
}
canonical=lambda value: json.dumps(value,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
summary['summary_sha256']=sha256(canonical(summary)).hexdigest()
(root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
marker={
 'code_hash':pin,
 'mode':mode,
 'summary_file_sha256':sha256((root/'summary.json').read_bytes()).hexdigest(),
 'summary_sha256':summary['summary_sha256'],
 'tag':tag,
}
marker['marker_sha256']=sha256(canonical(marker)).hexdigest()
name='SUCCESS' if mode=='numerical' else 'HLO_ACQUIRED'
(root/name).write_text(json.dumps(marker,indent=2,sort_keys=True)+'\n')
PY

strict_census post || {
  say "ABORT: post-run census is not clean"
  exit 1
}
post_census_done=1
marker=HLO_ACQUIRED
[[ $MODE == acquire ]] || marker=SUCCESS
for name in orchestrator.log census_pre.txt census_post.txt sync.txt launch.txt \
  summary.json "$marker"; do
  gcloud storage cp --no-clobber "$RUN_DIR/$name" \
    "$REMOTE_PREFIX/orchestrator/$name" >/dev/null
done
local_summary_sha=$(sha256sum "$RUN_DIR/summary.json" | awk '{print $1}')
remote_summary_sha=$(gcloud storage cat \
  "$REMOTE_PREFIX/orchestrator/summary.json" | sha256sum | awk '{print $1}')
local_marker_sha=$(sha256sum "$RUN_DIR/$marker" | awk '{print $1}')
remote_marker_sha=$(gcloud storage cat \
  "$REMOTE_PREFIX/orchestrator/$marker" | sha256sum | awk '{print $1}')
[[ $local_summary_sha == "$remote_summary_sha" && \
  $local_marker_sha == "$remote_marker_sha" ]]
say "COMPLETE mode=$MODE summary_sha=$local_summary_sha marker_sha=$local_marker_sha"
trap - EXIT
