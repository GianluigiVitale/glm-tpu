#!/usr/bin/env bash
# Protected compile-only real-state PP16 feature2 Gate-D acquisition.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly RUNTIME_TAG=greenfield_runtime_feature_qkv_direct_pp16_20260827T164842844148623Z
readonly RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/$RUNTIME_TAG
readonly RUNTIME_REMOTE=$APPROVED_BUCKET/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/$RUNTIME_TAG
readonly RUNTIME_MANIFEST_SHA=b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab
readonly TOKEN_ORACLE_TAG=greenfield_short_context_oracle_8k_20260807T172307269147351Z
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/$TOKEN_ORACLE_TAG/oracle
readonly TOKEN_ORACLE_MANIFEST_SHA=e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2
readonly DSA_ORACLE_TAG=greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z
readonly DSA_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/$DSA_ORACLE_TAG/oracle
readonly DSA_ORACLE_MANIFEST_SHA=f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da
readonly LAYER1_INTERNAL_REFERENCE=/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/layer1/greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z/internals.npz
readonly LAYER1_INTERNAL_REFERENCE_SHA=79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054
readonly DB529_INTERNAL_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/inputs/internal
readonly DB529_INTERNAL_CONTRACT_SHA=9bdab5023b5775b787e15c3c76d542eab921bd7a704602b4502b2305fad03d4c
readonly DB529_INTERNAL_TENSOR_SHA=c2fdeccfdcc81363fe01a566c34bf6c04f2f44b0a18e7b545e76fdf0f0d4560b
readonly FEATURE2_GRAPH_SHA=ab5be45aecf3b0b5d87ad76af8076bc9351823529a08c0eadb414b072b31cb2d
readonly SEALED_MAIN_STABLEHLO_SHA=6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4
readonly SEALED_MAIN_CANONICAL_HLO_SHA=9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5
readonly SEALED_MAIN_CANONICAL_HLO_BYTES=6558627
readonly SEALED_MAIN_CANONICAL_STACK_REFS=14561

[[ ${GLM_GREENFIELD_PP16_FEATURE2_ACQUIRE:-0} == 1 ]] || {
  echo "PP16 feature2 acquisition is default-off" >&2
  exit 2
}
[[ ${GLM_GREENFIELD_PP16_FEATURE2_MODE:-off} == compile_only ]] || {
  echo "set GLM_GREENFIELD_PP16_FEATURE2_MODE=compile_only" >&2
  exit 2
}
FULL_WIDTH_ROUNDED_THEN_SLICE=${GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE:-0}
readonly FULL_WIDTH_ROUNDED_THEN_SLICE
[[ $FULL_WIDTH_ROUNDED_THEN_SLICE == 1 ]] || {
  echo "set GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE=1 for the admitted successor" >&2
  exit 2
}
readonly runner_variant_args=(--full-width-rounded-then-slice)

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_FEATURE2_TAG:-greenfield_pp16_feature2_prefill_acquire_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX

[[ $TAG =~ ^greenfield_pp16_feature2_prefill_acquire_[0-9]{8}T[0-9]{15}Z$ ]]
[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
[[ $(git -C "$WORKTREE" ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == "$APPROVED_LOCATION" ]]
[[ ! -e $RUN_DIR ]]
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[pp16-feature2 $(date -u +%H:%M:%S)] $*" |
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
  local carrier="${TAG}_${label}" ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "[a]cquire_pp16_feature2_prefill[.]py|[c]ompile_short_decoder[.]py|[r]un_short_decoder|[V]LLM::EngineCore|[R]ayWorkerWrapper|[m]icrobench" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; elif [[ -n $ray_pids || -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [[ -z $ray_pids ]] || echo "ray_stop_pids: $ray_pids"; [[ -z $generic ]] || echo "$generic"; [[ -z $holders ]] || echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh \
    "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

verify_failure_diagnostic() {
  local expected relative observed ledger_sha remote_ledger_sha
  ledger_sha=$(sha256sum "$RUN_DIR/diagnostic.evidence.sha256" | awk '{print $1}')
  remote_ledger_sha=$(gcloud storage cat \
    "$REMOTE_PREFIX/diagnostic/diagnostic.evidence.sha256" 2>/dev/null |
    sha256sum | awk '{print $1}')
  [[ $ledger_sha == "$remote_ledger_sha" ]] || return 1
  while read -r expected relative; do
    relative=${relative#\*}
    relative=${relative#./}
    observed=$(gcloud storage cat \
      "$REMOTE_PREFIX/diagnostic/$relative" 2>/dev/null |
      sha256sum | awk '{print $1}')
    [[ $expected == "$observed" ]] || return 1
  done <"$RUN_DIR/diagnostic.evidence.sha256"
  diff -u \
    <(
      {
        echo "$REMOTE_PREFIX/diagnostic/diagnostic.evidence.sha256"
        while read -r _ relative; do
          relative=${relative#\*}
          relative=${relative#./}
          echo "$REMOTE_PREFIX/diagnostic/$relative"
        done <"$RUN_DIR/diagnostic.evidence.sha256"
      } | sort
    ) \
    <(gcloud storage ls --recursive \
      "$REMOTE_PREFIX/diagnostic/**" 2>/dev/null | sort) >/dev/null
}

upload_failure_diagnostic() {
  local original_status=$1 attempt
  printf 'original_status=%s\n' "$original_status" >"$RUN_DIR/failure_status.txt"
  cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
  (
    cd "$RUN_DIR"
    find . -type f ! -name orchestrator.log \
      ! -name diagnostic.evidence.sha256 ! -name HLO_ACQUIRED -print0 |
      sort -z | xargs -0 sha256sum
  ) >"$RUN_DIR/diagnostic.evidence.sha256"
  for attempt in 1 2; do
    if gsutil -m rsync -r \
      -x '(^|/)orchestrator.log$|(^|/)HLO_ACQUIRED$' "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic" >/dev/null 2>&1 &&
      verify_failure_diagnostic; then
      return 0
    fi
  done
  return 1
}

verify_success_evidence() {
  local expected relative observed ledger_sha remote_ledger_sha
  ledger_sha=$(sha256sum "$RUN_DIR/evidence.sha256" | awk '{print $1}')
  remote_ledger_sha=$(gcloud storage cat \
    "$REMOTE_PREFIX/evidence.sha256" 2>/dev/null |
    sha256sum | awk '{print $1}')
  [[ $ledger_sha == "$remote_ledger_sha" ]] || return 1
  while read -r expected relative; do
    relative=${relative#\*}
    relative=${relative#./}
    observed=$(gcloud storage cat "$REMOTE_PREFIX/$relative" 2>/dev/null |
      sha256sum | awk '{print $1}')
    [[ $expected == "$observed" ]] || return 1
  done <"$RUN_DIR/evidence.sha256"
  diff -u \
    <(
      {
        echo "$REMOTE_PREFIX/evidence.sha256"
        while read -r _ relative; do
          relative=${relative#\*}
          relative=${relative#./}
          echo "$REMOTE_PREFIX/$relative"
        done <"$RUN_DIR/evidence.sha256"
      } | sort
    ) \
    <(gcloud storage ls --recursive "$REMOTE_PREFIX/**" 2>/dev/null | sort) \
    >/dev/null
}

post_census_done=0
terminal_written=0
remote_vacant=0
on_exit() {
  local status=$? census_status=0
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || census_status=$?
  fi
  if [[ $status -ne 0 && $terminal_written -eq 0 && $remote_vacant -eq 1 ]]; then
    say "FAILED status=$status; sealing compile-only diagnostic without HLO_ACQUIRED"
    if ! upload_failure_diagnostic "$status"; then
      say "DIAGNOSTIC_UPLOAD_FAILED after two attempts"
      trap - EXIT
      exit 70
    fi
    say "DIAGNOSTIC_UPLOAD_OK verified exact remote hashes and object set"
    if [[ $census_status -ne 0 ]]; then
      say "FAILURE_CENSUS_FAILED status=$census_status"
      trap - EXIT
      exit 71
    fi
  fi
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected TPU workflow holds the global lease"
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN mode=compile_only devices=0,1"
if gcloud storage ls "$REMOTE_PREFIX/**" >"$RUN_DIR/remote_vacancy.txt" 2>&1; then
  say "ABORT: append-only remote prefix already exists"
  exit 2
fi
echo "VACANT $REMOTE_PREFIX" >"$RUN_DIR/remote_vacancy.txt"
remote_vacant=1

say "authenticating the exact selected runtime and accepted event-1 lineage"
PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
  "$RUNTIME_ROOT" "$RUNTIME_REMOTE" "$RUNTIME_MANIFEST_SHA" \
  "$TOKEN_ORACLE_DIR" "$TOKEN_ORACLE_MANIFEST_SHA" \
  "$DSA_ORACLE_DIR" "$DSA_ORACLE_MANIFEST_SHA" \
  "$LAYER1_INTERNAL_REFERENCE" "$LAYER1_INTERNAL_REFERENCE_SHA" \
  "$DB529_INTERNAL_DIR" "$DB529_INTERNAL_CONTRACT_SHA" \
  "$DB529_INTERNAL_TENSOR_SHA" "$FEATURE2_GRAPH_SHA" \
  "$RUN_DIR/source_identity.json" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.benchmarking.pp16_feature2_acquisition import inspect_feature2_event1_lineage
from glm_tpu.greenfield.benchmarking.pp16_feature2_loader import inspect_feature2_selective_plan
from glm_tpu.greenfield.benchmarking.pp16_feature2_prefill import build_feature2_prefill_graph, load_feature2_prefill_inputs
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import derive_feature2_tensor_allowlist, read_feature2_owner_headers

(runtime,remote,runtime_sha,token,token_sha,dsa,dsa_sha,layer1,layer1_sha,
 db529,db529_contract,db529_tensor,graph_sha,output)=sys.argv[1:]
runtime=Path(runtime); token=Path(token); dsa=Path(dsa); layer1=Path(layer1); db529=Path(db529)
manifest=json.loads((runtime/'runtime_manifest.json').read_text())
if manifest.get('manifest_sha256')!=runtime_sha:
    raise SystemExit('feature2 runtime manifest pin drifted')
if sha256(layer1.read_bytes()).hexdigest()!=layer1_sha:
    raise SystemExit('feature2 layer1 reference pin drifted')
lineage=inspect_feature2_event1_lineage(token_oracle_dir=token,dsa_oracle_dir=dsa,layer1_internal_reference=layer1,db529_internal_dir=db529)
if lineage['token_oracle_manifest_sha256']!=token_sha or lineage['dsa_oracle_manifest_sha256']!=dsa_sha or lineage['db529_internal_contract_sha256']!=db529_contract or lineage['db529_internal_tensor_sha256']!=db529_tensor:
    raise SystemExit('feature2 source lineage pin drifted')
graph=build_feature2_prefill_graph(derive_feature2_tensor_allowlist(manifest,read_feature2_owner_headers(runtime,manifest)),load_feature2_prefill_inputs(token))
if graph.graph_sha256!=graph_sha:
    raise SystemExit('feature2 graph pin drifted')
record={'artifact_kind':'greenfield_pp16_feature2_source_identity','runtime_root':str(runtime),'runtime_remote':remote,'runtime_manifest_sha256':runtime_sha,'selective_plan':inspect_feature2_selective_plan(runtime),'event1_target_lineage':lineage,'graph_sha256':graph_sha}
Path(output).write_text(json.dumps(record,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY

strict_census pre || {
  say "ABORT: pre-run census is not authenticated 8/8 zero work"
  exit 1
}

say "synchronizing the exact pushed pin on all eight hosts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; wt='"$WORKTREE"'; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" && $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host code synchronization failed"
  exit 1
}

say "loading 39 authenticated ranges per owner and compiling four graphs; main execution is forbidden"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 \
    PYTHONPATH="$WORKTREE" \
    timeout --signal=TERM --kill-after=60 7200 \
    /home/gianl/vllm-env/bin/python -u \
      scripts/greenfield/acquire_pp16_feature2_prefill.py \
      --expected-code-hash "$PIN" \
      --runtime-root "$RUNTIME_ROOT" \
      --token-oracle-dir "$TOKEN_ORACLE_DIR" \
      --dsa-oracle-dir "$DSA_ORACLE_DIR" \
      --layer1-internal-reference "$LAYER1_INTERNAL_REFERENCE" \
      --db529-internal-dir "$DB529_INTERNAL_DIR" \
      --compile-only 1 \
      "${runner_variant_args[@]}" \
      --output "$RUN_DIR/runner.json" --hlo-dir "$RUN_DIR/hlo"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "compile-only runner completed in ${elapsed}s"

strict_census post || {
  say "ABORT: post-run census is not authenticated 8/8 zero work"
  exit 1
}
post_census_done=1

say "recomputing every HLO/source/load claim without JAX"
PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$TAG" "$REMOTE_PREFIX" "$elapsed" \
  "$FULL_WIDTH_ROUNDED_THEN_SLICE" "$SEALED_MAIN_STABLEHLO_SHA" \
  "$SEALED_MAIN_CANONICAL_HLO_SHA" "$SEALED_MAIN_CANONICAL_HLO_BYTES" \
  "$SEALED_MAIN_CANONICAL_STACK_REFS" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.benchmarking.pp16_feature2_hlo import validate_feature2_sealed_hlo_archive_identity

run=Path(sys.argv[1]); pin,tag,remote,elapsed=sys.argv[2:6]
full_width_rounded_then_slice=bool(int(sys.argv[6]))
expected_stable_sha=sys.argv[7]
expected_canonical_sha=sys.argv[8]
expected_canonical_bytes=int(sys.argv[9])
expected_canonical_stack_refs=int(sys.argv[10])
runner=json.loads((run/'runner.json').read_text())
source=json.loads((run/'source_identity.json').read_text())
if runner.get('status')!='HLO_ACQUIRED' or runner.get('code_hash')!=pin or runner.get('compile_only') is not True or runner.get('sealed_boundary_capture') is not True or runner.get('main_executed') is not False or runner.get('numerical_claim') is not False or runner.get('performance_claim') is not False:
    raise SystemExit('feature2 acquisition claim boundary drifted')
if runner.get('full_width_rounded_then_slice') is not full_width_rounded_then_slice:
    raise SystemExit('feature2 acquisition producer variant drifted')
if runner.get('graph_sha256')!=source.get('graph_sha256') or runner.get('event1_target_lineage')!=source.get('event1_target_lineage') or runner.get('selective_plan')!=source.get('selective_plan'):
    raise SystemExit('feature2 acquisition source lineage drifted')
if runner.get('physical_group')!={'coordinates':[[0,0,0],[1,0,0]],'device_ids':[0,1],'local_device_count_visible':4,'mesh_device_count':2}:
    raise SystemExit('feature2 physical LP2 group drifted')
expected_graphs={'feature2_main','query_fp32','wk_decode_bf16','wk_promote_fp32'}
if set(runner.get('hlo',{}))!=expected_graphs:
    raise SystemExit('feature2 HLO graph set drifted')
expected_terminal_shapes=[[1,2048],[1],[1,2048],[2,1,3072],[2,1,32,256],[1,576],[2,1,6144],[2,1,2048],[2,1,32,128],[2,1,32],[2,16,256,640],[2,16,256,128],[2,16,256,128],[2,2],[1]]
expected_terminal_dtypes=['int32','int32','float32','bfloat16','bfloat16','bfloat16','bfloat16','bfloat16','float32','float32','bfloat16','bfloat16','bfloat16','uint32','bool']
expected_stable_types=['tensor<1x2048xi32>','tensor<1xi32>','tensor<1x2048xf32>','tensor<2x1x3072xbf16>','tensor<2x1x32x256xbf16>','tensor<1x576xbf16>','tensor<2x1x6144xbf16>','tensor<2x1x2048xbf16>','tensor<2x1x32x128xf32>','tensor<2x1x32xf32>','tensor<2x16x256x640xbf16>','tensor<2x16x256x128xbf16>','tensor<2x16x256x128xbf16>','tensor<2x2xui32>','tensor<1xi1>']
expected_optimized_roots=[{'dtype':dtype,'shape':shape} for dtype,shape in [('s32',[1,2048]),('s32',[1]),('f32',[1,2048]),('bf16',[1,1,3072]),('bf16',[1,1,32,256]),('bf16',[1,576]),('bf16',[1,1,6144]),('bf16',[1,1,2048]),('f32',[1,1,32,128]),('f32',[1,1,32]),('bf16',[1,16,256,640]),('bf16',[1,16,256,128]),('bf16',[1,16,256,128]),('u32',[1,2]),('pred',[1])]]
expected_sealed_bindings={'6':'greenfield_pp16_feature2_sealed_normalized_hidden','7':'greenfield_pp16_feature2_sealed_q_a_state','8':'greenfield_pp16_feature2_sealed_dsa_query','9':'greenfield_pp16_feature2_sealed_dsa_head_weights'}
for name,record in runner['hlo'].items():
    for kind,suffix in (('stablehlo','.stablehlo.mlir'),('optimized_hlo','.optimized_hlo.txt')):
        identity=record.get(kind,{})
        path=run/'hlo'/identity.get('filename','')
        if path.parent!=run/'hlo' or not path.name.endswith(suffix) or not path.is_file() or sha256(path.read_bytes()).hexdigest()!=identity.get('sha256'):
            raise SystemExit(f'feature2 {name} {kind} identity drifted')
    if name=='feature2_main':
        for contract in ('jaxpr_contract','optimized_contract','stablehlo_contract','terminal_contract'):
            if record.get(contract,{}).get('passed') is not True:
                raise SystemExit(f'feature2 main {contract} failed')
        stable=record['stablehlo_contract']; optimized=record['optimized_contract']; terminal=record['terminal_contract']
        stable_path=run/'hlo'/record.get('stablehlo',{}).get('filename','')
        optimized_path=run/'hlo'/record.get('optimized_hlo',{}).get('filename','')
        canonical=record.get('execution_canonical_hlo',{})
        canonical_path=run/'hlo'/canonical.get('filename','')
        if stable_path.parent!=run/'hlo' or not stable_path.name.endswith('.stablehlo.mlir') or optimized_path.parent!=run/'hlo' or not optimized_path.name.endswith('.optimized_hlo.txt') or canonical_path.parent!=run/'hlo' or not canonical_path.name.endswith('.execution_canonical_hlo.txt'):
            raise SystemExit('feature2 main HLO archive paths drifted')
        archive_identity=validate_feature2_sealed_hlo_archive_identity(stable_path,optimized_path,canonical_path,expected_stablehlo_sha256=expected_stable_sha,expected_canonical_sha256=expected_canonical_sha,expected_canonical_bytes=expected_canonical_bytes,expected_canonicalizer_version=1,expected_stripped_stack_frame_references=expected_canonical_stack_refs)
        if record.get('stablehlo',{}).get('sha256')!=archive_identity['stablehlo_sha256'] or record.get('optimized_hlo',{}).get('sha256')!=archive_identity['optimized_hlo_sha256']:
            raise SystemExit('feature2 main raw HLO identity records drifted')
        if stable.get('sealed_boundary_capture') is not True or stable.get('stablehlo_sha256')!=archive_identity['stablehlo_sha256'] or stable.get('output_count')!=15 or stable.get('terminal_shapes')!=expected_terminal_shapes or stable.get('terminal_types')!=expected_stable_types:
            raise SystemExit('feature2 main StableHLO sealed terminal drifted')
        expected_canonical={'byte_count':expected_canonical_bytes,'canonicalizer_version':1,'sha256':expected_canonical_sha,'stripped_stack_frame_references':expected_canonical_stack_refs}
        optimized_canonical={key:optimized.get('sealed_canonical_hlo_identity',{}).get(key) for key in expected_canonical}
        runner_canonical={key:canonical.get(key) for key in expected_canonical}
        recomputed_canonical={key:archive_identity['canonical_hlo_identity'].get(key) for key in expected_canonical}
        if optimized.get('sealed_boundary_capture') is not True or optimized.get('output_count')!=15 or optimized.get('root_shapes')!=expected_optimized_roots or optimized.get('sealed_bindings')!=expected_sealed_bindings or optimized_canonical!=expected_canonical:
            raise SystemExit('feature2 main optimized-HLO sealed terminal drifted')
        if runner_canonical!=expected_canonical or recomputed_canonical!=expected_canonical or canonical.get('canonicalizer_code_hash')!=pin:
            raise SystemExit('feature2 main canonical-HLO artifact drifted')
        if terminal.get('sealed_boundary_capture') is not True or terminal.get('output_count')!=15 or terminal.get('terminal_shapes')!=expected_terminal_shapes or terminal.get('terminal_dtypes')!=expected_terminal_dtypes:
            raise SystemExit('feature2 main abstract sealed terminal drifted')
    elif record.get('contract',{}).get('passed') is not True:
        raise SystemExit(f'feature2 materializer {name} contract failed')
state=runner.get('state_manifest',{})
if state.get('plan_id')!='PP16_LP2' or state.get('owner_device_ids')!=[0,1] or state.get('selected_read_count')!=78 or state.get('raw_dense_device_materialization') is not False or state.get('dense_final_layout') is not True:
    raise SystemExit('feature2 selective state contract drifted')
summary={'artifact_kind':'greenfield_pp16_feature2_compile_acquisition_summary','claim_scope':runner['claim_scope'],'code_hash':pin,'elapsed_seconds':int(elapsed),'full_width_rounded_then_slice':full_width_rounded_then_slice,'graph_sha256':runner['graph_sha256'],'hlo_sha256':{name:{kind:record[kind]['sha256'] for kind in ('stablehlo','optimized_hlo')} for name,record in runner['hlo'].items()},'main_executed':False,'numerical_claim':False,'performance_claim':False,'remote_prefix':remote,'run_tag':tag,'sealed_boundary_capture':True,'status':'HLO_ACQUIRED'}
(run/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY

cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log source_identity.json summary.json \
    census_pre.txt census_post.txt sync.txt remote_vacancy.txt \
    orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
(cd "$RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)

say "publishing and byte-verifying the same-region preterminal archive"
gsutil -m rsync -r -x '(^|/)orchestrator.log$' \
  "$RUN_DIR" "$REMOTE_PREFIX" >/dev/null
verify_success_evidence || {
  say "ABORT: preterminal remote hashes or exact object set drifted"
  exit 1
}

PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/summary.json" "$RUN_DIR/evidence.sha256" \
  "$RUN_DIR/HLO_ACQUIRED" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
summary,evidence,output=map(Path,sys.argv[1:])
record={'artifact_kind':'greenfield_pp16_feature2_hlo_acquired','evidence_sha256':sha256(evidence.read_bytes()).hexdigest(),'summary_sha256':sha256(summary.read_bytes()).hexdigest(),'status':'HLO_ACQUIRED'}
raw=json.dumps(record,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
record['marker_self_sha256']=sha256(raw).hexdigest()
output.write_text(json.dumps(record,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/HLO_ACQUIRED" \
  "$REMOTE_PREFIX/HLO_ACQUIRED" >/dev/null
local_marker_sha=$(sha256sum "$RUN_DIR/HLO_ACQUIRED" | awk '{print $1}')
remote_marker_sha=$(gcloud storage cat "$REMOTE_PREFIX/HLO_ACQUIRED" | sha256sum | awk '{print $1}')
[[ $local_marker_sha == "$remote_marker_sha" ]]
terminal_written=1
trap - EXIT
say "HLO_ACQUIRED archive=$REMOTE_PREFIX no_numerical_or_performance_claim=true"
