#!/usr/bin/env bash
# Protected zero-warmup, exactly-once PP16 feature2 numerical discriminator.
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
readonly DB550_BOUNDARY=/home/gianl/gcs-models/results/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/dense_partial_capture.npz
readonly ACQUIRED_MAIN_STABLE_SHA=127bf089f93bc9dd4f1b85576e8e70267a752d2be90dee74525322b5148a955e
readonly ACQUIRED_MAIN_CANONICAL_SHA=fb5aaf025005f3fbb5a3c66e6a719ec3a78fb86d344afcf6288e6e93720310f7
readonly ACQUIRED_MAIN_CANONICAL_BYTES=7870521
readonly ACQUIRED_MAIN_STACK_FRAME_REFERENCES=16170
readonly ACQUIRED_JAX_VERSION=0.10.1
readonly ACQUIRED_JAXLIB_VERSION=0.10.1
readonly ACQUIRED_LIBTPU_VERSION=0.0.41

[[ ${GLM_GREENFIELD_PP16_FEATURE2_NUMERICAL:-0} == 1 ]] || {
  echo "PP16 feature2 numerical discriminator is default-off" >&2
  exit 2
}
[[ ${GLM_GREENFIELD_PP16_FEATURE2_MODE:-off} == execute_once ]] || {
  echo "set GLM_GREENFIELD_PP16_FEATURE2_MODE=execute_once" >&2
  exit 2
}

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_FEATURE2_TAG:-greenfield_pp16_feature2_prefill_numerical_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX

[[ $TAG =~ ^greenfield_pp16_feature2_prefill_numerical_[0-9]{8}T[0-9]{15}Z$ ]]
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

remote_prefix_is_vacant() {
  local output="$RUN_DIR/remote_vacancy.stdout"
  local error="$RUN_DIR/remote_vacancy.stderr" status
  if gcloud storage ls "$REMOTE_PREFIX/**" >"$output" 2>"$error"; then
    [[ ! -s $output && ! -s $error ]]
    return
  else
    status=$?
  fi
  [[ $status -eq 1 && ! -s $output ]] &&
    [[ $(grep -c '^ERROR:' "$error") -eq 1 ]] &&
    [[ $(tail -n 1 "$error") == \
      "ERROR: (gcloud.storage.ls) One or more URLs matched no objects." ]]
}

upload_ledger_no_clobber() {
  local ledger=$1 destination=$2 relative
  while read -r _ relative; do
    relative=${relative#\*}
    relative=${relative#./}
    gcloud storage cp --no-clobber "$RUN_DIR/$relative" \
      "$destination/$relative" >/dev/null
  done <"$ledger"
  gcloud storage cp --no-clobber "$ledger" \
    "$destination/$(basename "$ledger")" >/dev/null
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}" ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "[a]cquire_pp16_feature2_prefill[.]py|[e]xecute_pp16_feature2_prefill[.]py|[c]ompile_short_decoder[.]py|[r]un_short_decoder|[V]LLM::EngineCore|[R]ayWorkerWrapper|[m]icrobench" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; elif [[ -n $ray_pids || -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [[ -z $ray_pids ]] || echo "ray_stop_pids: $ray_pids"; [[ -z $generic ]] || echo "$generic"; [[ -z $holders ]] || echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
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

verify_failure_remote_union() {
  local listing="$RUN_DIR/failure_remote_union.txt" uri relative expected observed
  verify_failure_diagnostic || return 1
  gcloud storage ls --recursive "$REMOTE_PREFIX/**" >"$listing" 2>/dev/null ||
    return 1
  while IFS= read -r uri; do
    [[ -n $uri ]] || continue
    relative=${uri#"$REMOTE_PREFIX/"}
    if [[ $relative == diagnostic/* ]]; then
      continue
    fi
    [[ -f $RUN_DIR/$relative && -f $RUN_DIR/evidence.sha256 ]] || return 1
    if [[ $relative == evidence.sha256 ]]; then
      expected=$(sha256sum "$RUN_DIR/evidence.sha256" | awk '{print $1}')
    else
      expected=$(awk -v target="$relative" \
        '{name=$2; sub(/^\*/,"",name); sub(/^\.\//,"",name); if(name==target){print $1}}' \
        "$RUN_DIR/evidence.sha256")
      [[ -n $expected ]] || return 1
    fi
    observed=$(gcloud storage cat "$uri" 2>/dev/null | sha256sum | awk '{print $1}')
    [[ $expected == "$observed" ]] || return 1
  done <"$listing"
}

upload_failure_diagnostic() {
  local original_status=$1 attempt
  printf 'original_status=%s\n' "$original_status" >"$RUN_DIR/failure_status.txt"
  cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
  (
    cd "$RUN_DIR"
    find . -type f ! -name orchestrator.log \
      ! -name diagnostic.evidence.sha256 ! -name NUMERICAL_EXACT \
      ! -name NUMERICAL_REJECTED -print0 |
      sort -z | xargs -0 sha256sum
  ) >"$RUN_DIR/diagnostic.evidence.sha256"
  for attempt in 1 2; do
    if upload_ledger_no_clobber "$RUN_DIR/diagnostic.evidence.sha256" \
      "$REMOTE_PREFIX/diagnostic" >/dev/null 2>&1 && \
      verify_failure_remote_union; then
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

verify_final_object_set() {
  local terminal=$1
  diff -u \
    <(
      {
        echo "$REMOTE_PREFIX/evidence.sha256"
        echo "$REMOTE_PREFIX/$terminal"
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
terminal_publication_started=0
terminal_remote_name=
remote_vacant=0
rollback_terminal() {
  local receipt="$RUN_DIR/terminal_create.receipt.stderr"
  local current="$RUN_DIR/terminal_rollback.describe.json"
  local error="$RUN_DIR/terminal_rollback.stderr" generation
  [[ -n $terminal_remote_name && -s $receipt ]] || return 1
  gcloud storage objects describe "$REMOTE_PREFIX/$terminal_remote_name" \
    --format=json >"$current" 2>"$error" || return 1
  generation=$(PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$RUN_DIR/$terminal_remote_name" "$receipt" "$current" \
    "$REMOTE_PREFIX/$terminal_remote_name" <<'PY'
import base64
import json
from pathlib import Path
import re
import sys

import google_crc32c

path, receipt, current = map(Path, sys.argv[1:4])
remote = sys.argv[4]
after = json.loads(current.read_text())
checksum = google_crc32c.Checksum(path.read_bytes())
crc = base64.b64encode(checksum.digest()).decode("ascii")
matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
assert matches == [(remote, str(after["generation"]))]
assert path.stat().st_size == int(after["size"])
assert crc == after["crc32c_hash"]
print(str(after["generation"]))
PY
  ) || return 1
  [[ -n $generation ]] || return 1
  gcloud storage rm --if-generation-match="$generation" \
    "$REMOTE_PREFIX/$terminal_remote_name" \
    >"$RUN_DIR/terminal_rollback.stdout" \
    2>>"$RUN_DIR/terminal_rollback.stderr"
}
on_exit() {
  local status=$? census_status=0
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || census_status=$?
  fi
  if [[ $status -ne 0 && $terminal_written -eq 0 && $remote_vacant -eq 1 ]]; then
    if [[ $terminal_publication_started -eq 1 ]]; then
      if ! rollback_terminal; then
        say "TERMINAL_ROLLBACK_FAILED; refusing later remote mutation"
        trap - EXIT
        exit 72
      fi
      say "authenticated run-owned terminal generation rolled back"
    fi
    say "FAILED status=$status; sealing preterminal numerical diagnostic"
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

say "RUN_DIR=$RUN_DIR PIN=$PIN mode=execute_once warmups=0 invocations=1 devices=0,1"
if ! remote_prefix_is_vacant; then
  say "ABORT: remote prefix is occupied or vacancy could not be authenticated"
  exit 2
fi
{
  echo "VACANT $REMOTE_PREFIX"
  cat "$RUN_DIR/remote_vacancy.stdout" "$RUN_DIR/remote_vacancy.stderr"
} >"$RUN_DIR/remote_vacancy.txt"
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

say "loading selected state, compiling four graphs, and executing main exactly once with zero warmups"
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
      scripts/greenfield/execute_pp16_feature2_prefill.py \
      --expected-code-hash "$PIN" \
      --runtime-root "$RUNTIME_ROOT" \
      --token-oracle-dir "$TOKEN_ORACLE_DIR" \
      --dsa-oracle-dir "$DSA_ORACLE_DIR" \
      --layer1-internal-reference "$LAYER1_INTERNAL_REFERENCE" \
      --db529-internal-dir "$DB529_INTERNAL_DIR" \
      --expected-main-stablehlo-sha256 "$ACQUIRED_MAIN_STABLE_SHA" \
      --expected-main-canonical-hlo-sha256 "$ACQUIRED_MAIN_CANONICAL_SHA" \
      --expected-main-canonical-hlo-byte-count "$ACQUIRED_MAIN_CANONICAL_BYTES" \
      --expected-main-stack-frame-reference-count "$ACQUIRED_MAIN_STACK_FRAME_REFERENCES" \
      --expected-jax-version "$ACQUIRED_JAX_VERSION" \
      --expected-jaxlib-version "$ACQUIRED_JAXLIB_VERSION" \
      --expected-libtpu-version "$ACQUIRED_LIBTPU_VERSION" \
      --output "$RUN_DIR/runner.json" --hlo-dir "$RUN_DIR/hlo" \
      --result-npz "$RUN_DIR/result.npz"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "exactly-once runner completed in ${elapsed}s (operational only; not performance evidence)"

strict_census post || {
  say "ABORT: post-run census is not authenticated 8/8 zero work"
  exit 1
}
post_census_done=1

say "recomputing HLO/source/load claims and exact numerical result without JAX"
PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$TAG" "$REMOTE_PREFIX" "$elapsed" \
  "$TOKEN_ORACLE_DIR" "$DSA_ORACLE_DIR" "$LAYER1_INTERNAL_REFERENCE" \
  "$DB529_INTERNAL_DIR" "$DB550_BOUNDARY" \
  "$ACQUIRED_MAIN_STABLE_SHA" "$ACQUIRED_MAIN_CANONICAL_SHA" \
  "$ACQUIRED_MAIN_CANONICAL_BYTES" "$ACQUIRED_MAIN_STACK_FRAME_REFERENCES" \
  "$ACQUIRED_JAX_VERSION" "$ACQUIRED_JAXLIB_VERSION" "$ACQUIRED_LIBTPU_VERSION" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.benchmarking.pp16_feature2_numerical import compare_feature2_numerical_capture

run=Path(sys.argv[1]); pin,tag,remote,elapsed=sys.argv[2:6]
token,dsa,layer1,db529,db550=map(Path,sys.argv[6:11])
stable_pin,canonical_pin=sys.argv[11:13]
canonical_bytes,stack_frame_references=map(int,sys.argv[13:15])
expected_runtime_pins=dict(zip(('jax','jaxlib','libtpu'),sys.argv[15:18],strict=True))
runner=json.loads((run/'runner.json').read_text())
source=json.loads((run/'source_identity.json').read_text())
if runner.get('status')!='NUMERICAL_CAPTURED' or runner.get('code_hash')!=pin or runner.get('compile_only') is not False or runner.get('main_executed') is not True or runner.get('main_execution_count')!=1 or runner.get('numerical_claim') is not False or runner.get('performance_claim') is not False:
    raise SystemExit('feature2 numerical-capture claim boundary drifted')
if runner.get('graph_sha256')!=source.get('graph_sha256') or runner.get('event1_target_lineage')!=source.get('event1_target_lineage') or runner.get('selective_plan')!=source.get('selective_plan'):
    raise SystemExit('feature2 numerical source lineage drifted')
if runner.get('physical_group')!={'coordinates':[[0,0,0],[1,0,0]],'device_ids':[0,1],'local_device_count_visible':4,'mesh_device_count':2}:
    raise SystemExit('feature2 physical LP2 group drifted')
runtime_pins=runner.get('runtime_pins',{})
if {name:runtime_pins.get(name) for name in expected_runtime_pins}!=expected_runtime_pins or runtime_pins.get('backend_platform')!='tpu' or runtime_pins.get('device_kind')!='TPU v4' or not isinstance(runtime_pins.get('platform_version'),str) or not runtime_pins['platform_version']:
    raise SystemExit('feature2 compiler/runtime package pins drifted')
expected_graphs={'feature2_main','query_fp32','wk_decode_bf16','wk_promote_fp32'}
if set(runner.get('hlo',{}))!=expected_graphs:
    raise SystemExit('feature2 HLO graph set drifted')
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
    elif record.get('contract',{}).get('passed') is not True:
        raise SystemExit(f'feature2 materializer {name} contract failed')
main=runner['hlo']['feature2_main']
canonical=main.get('execution_canonical_hlo',{})
if main['stablehlo']['sha256']!=stable_pin or canonical.get('sha256')!=canonical_pin or canonical.get('byte_count')!=canonical_bytes or canonical.get('stripped_stack_frame_references')!=stack_frame_references or canonical.get('canonicalizer_version')!=1 or canonical.get('canonicalizer_code_hash')!=pin:
    raise SystemExit('feature2 numerical executable HLO drifted from acquired graph')
canonical_path=run/'hlo'/canonical.get('filename','')
if canonical_path.parent!=run/'hlo' or canonical_path.name!='feature2_main.execution_canonical_hlo.txt' or not canonical_path.is_file() or canonical_path.stat().st_size!=canonical_bytes or sha256(canonical_path.read_bytes()).hexdigest()!=canonical_pin:
    raise SystemExit('feature2 canonical executable artifact identity drifted')
state=runner.get('state_manifest',{})
if state.get('plan_id')!='PP16_LP2' or state.get('owner_device_ids')!=[0,1] or state.get('selected_read_count')!=78 or state.get('raw_dense_device_materialization') is not False or state.get('dense_final_layout') is not True:
    raise SystemExit('feature2 selective state contract drifted')
memory=runner.get('memory',{})
required_memory={'before_load','after_load','after_compile','after_execute','after_cleanup'}
if set(memory)!=required_memory:
    raise SystemExit('feature2 measured-memory phase set drifted')
for phase in sorted(required_memory):
    records=memory[phase]
    if not isinstance(records,list) or len(records)!=2 or any(not isinstance(record,dict) for record in records):
        raise SystemExit(f'feature2 measured-memory records missing at {phase}')
    required={'bytes_in_use','bytes_limit','largest_free_block_bytes','peak_bytes_in_use'}
    if any(not required.issubset(record) for record in records):
        raise SystemExit(f'feature2 measured-memory fields missing at {phase}')
if any(record['peak_bytes_in_use']>=record['bytes_limit'] or record['largest_free_block_bytes']<8*1024**3 for record in memory['after_execute']):
    raise SystemExit('feature2 numerical execution has unknown/insufficient HBM margin')
if any(record['bytes_in_use']>4*1024**2 for record in memory['after_cleanup']):
    raise SystemExit('feature2 numerical runner did not release device state')
comparison=compare_feature2_numerical_capture(run/'result.npz',token_oracle_dir=token,dsa_oracle_dir=dsa,layer1_internal_reference=layer1,db529_internal_dir=db529,db550_boundary=db550)
if sha256((run/'result.npz').read_bytes()).hexdigest()!=comparison['capture_sha256']:
    raise SystemExit('feature2 numerical capture changed before sealing')
(run/'comparison.json').write_text(json.dumps(comparison,allow_nan=False,indent=2,sort_keys=True)+'\n')
summary={'artifact_kind':'greenfield_pp16_feature2_numerical_summary','claim_scope':comparison['claim_scope'],'code_hash':pin,'elapsed_seconds_operational_only':int(elapsed),'exact':comparison['exact'],'graph_sha256':runner['graph_sha256'],'hlo_sha256':{name:{kind:record[kind]['sha256'] for kind in ('stablehlo','optimized_hlo')} for name,record in runner['hlo'].items()},'main_canonical_hlo':canonical,'main_execution_count':1,'measured_memory':memory,'mismatch_counts':comparison['mismatch_counts'],'numerical_claim':comparison['exact'],'performance_claim':False,'remote_prefix':remote,'run_tag':tag,'runtime_pins':runtime_pins,'status':comparison['status']}
(run/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY

cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log result.npz comparison.json \
    source_identity.json summary.json \
    census_pre.txt census_post.txt sync.txt remote_vacancy.txt \
    orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
(cd "$RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)

say "publishing each preterminal object no-clobber and byte-verifying the archive"
upload_ledger_no_clobber "$RUN_DIR/evidence.sha256" "$REMOTE_PREFIX"
verify_success_evidence || {
  say "ABORT: preterminal remote hashes or exact object set drifted"
  exit 1
}

terminal_status=$(PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu \
  /home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$RUN_DIR/summary.json")
[[ $terminal_status == NUMERICAL_EXACT || $terminal_status == NUMERICAL_REJECTED ]]
terminal_path="$RUN_DIR/$terminal_status"
PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/summary.json" "$RUN_DIR/comparison.json" \
  "$RUN_DIR/evidence.sha256" "$terminal_path" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
summary,comparison,evidence,output=map(Path,sys.argv[1:])
status=json.loads(summary.read_text())['status']
record={'artifact_kind':'greenfield_pp16_feature2_numerical_terminal','comparison_sha256':sha256(comparison.read_bytes()).hexdigest(),'evidence_sha256':sha256(evidence.read_bytes()).hexdigest(),'summary_sha256':sha256(summary.read_bytes()).hexdigest(),'status':status}
raw=json.dumps(record,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
record['marker_self_sha256']=sha256(raw).hexdigest()
output.write_text(json.dumps(record,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY
terminal_remote_name=$terminal_status
terminal_publication_started=1
gcloud storage cp --if-generation-match=0 --print-created-message \
  "$terminal_path" "$REMOTE_PREFIX/$terminal_status" \
  >"$RUN_DIR/terminal_create.stdout" \
  2>"$RUN_DIR/terminal_create.receipt.stderr"
sync -f "$RUN_DIR/terminal_create.receipt.stderr"
gcloud storage objects describe "$REMOTE_PREFIX/$terminal_status" --format=json \
  >"$RUN_DIR/terminal_upload.describe.json"
receipt_generation=$(PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$terminal_path" "$RUN_DIR/terminal_create.receipt.stderr" \
  "$RUN_DIR/terminal_upload.describe.json" \
  "$REMOTE_PREFIX/$terminal_status" <<'PY'
import base64
import json
from pathlib import Path
import re
import sys

import google_crc32c

path, receipt, metadata = map(Path, sys.argv[1:4])
remote = sys.argv[4]
record = json.loads(metadata.read_text())
checksum = google_crc32c.Checksum(path.read_bytes())
crc = base64.b64encode(checksum.digest()).decode("ascii")
matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
assert matches == [(remote, str(record["generation"]))]
assert path.stat().st_size == int(record["size"])
assert crc == record["crc32c_hash"]
print(str(record["generation"]))
PY
)
[[ -n $receipt_generation ]]
local_marker_sha=$(sha256sum "$terminal_path" | awk '{print $1}')
remote_marker_sha=$(gcloud storage cat "$REMOTE_PREFIX/$terminal_status" | sha256sum | awk '{print $1}')
[[ $local_marker_sha == "$remote_marker_sha" ]]
verify_final_object_set "$terminal_status"
terminal_written=1
trap - EXIT
say "$terminal_status archive=$REMOTE_PREFIX no_Gate-D_or_performance_claim=true"
[[ $terminal_status == NUMERICAL_EXACT ]] || exit 3
