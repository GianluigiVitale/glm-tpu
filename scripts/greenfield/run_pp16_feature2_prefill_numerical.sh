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
readonly ACQUIRED_JAX_VERSION=0.10.1
readonly ACQUIRED_JAXLIB_VERSION=0.10.1
readonly ACQUIRED_LIBTPU_VERSION=0.0.41
readonly POSITION113_SEALED_REJECTION=/home/gianl/glm-run/greenfield_pp16_feature2_prefill_numerical_20260829T051119686986506Z/result.npz
readonly POSITION113_SEALED_REJECTION_SHA=e514fc28e9d8c30bc7de9d70f01ae04002666494446e2f4a06a7fe6c66901e65
readonly POSITION113_ORACLE=/home/gianl/gcs-models/oracles/greenfield/glm52/prompt_projection_input/8k/greenfield_legacy_layer0_prompt_projection_input_p113_20260809T050055956585082Z/prompt_projection_input_comparison/prompt_key_internal_comparison.npz
readonly POSITION113_ORACLE_SHA=a2ef16a7a55876099124d0ac4bd139f86c6318b27c0e48fef5d64193ed0a3023
readonly DB518_PROMPT_CACHE_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/inputs/prompt_cache

[[ ${GLM_GREENFIELD_PP16_FEATURE2_NUMERICAL:-0} == 1 ]] || {
  echo "PP16 feature2 numerical discriminator is default-off" >&2
  exit 2
}
[[ ${GLM_GREENFIELD_PP16_FEATURE2_MODE:-off} == execute_once ]] || {
  echo "set GLM_GREENFIELD_PP16_FEATURE2_MODE=execute_once" >&2
  exit 2
}
FULL_WIDTH_ROUNDED_THEN_SLICE=${GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE:-0}
OBSERVE_POSITION_113=${GLM_GREENFIELD_PP16_OBSERVE_POSITION_113:-0}
EXACT_LAYER0_PROMPT_KEYS=${GLM_GREENFIELD_PP16_EXACT_LAYER0_PROMPT_KEYS:-0}
readonly FULL_WIDTH_ROUNDED_THEN_SLICE OBSERVE_POSITION_113 EXACT_LAYER0_PROMPT_KEYS
[[ $FULL_WIDTH_ROUNDED_THEN_SLICE == 1 ]] || {
  echo "set GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE=1 for the admitted successor" >&2
  exit 2
}
case $OBSERVE_POSITION_113 in
  0 | 1) ;;
  *)
    echo "GLM_GREENFIELD_PP16_OBSERVE_POSITION_113 must be exactly 0 or 1" >&2
    exit 2
    ;;
esac
case $EXACT_LAYER0_PROMPT_KEYS in
  0 | 1) ;;
  *)
    echo "GLM_GREENFIELD_PP16_EXACT_LAYER0_PROMPT_KEYS must be exactly 0 or 1" >&2
    exit 2
    ;;
esac
if [[ $EXACT_LAYER0_PROMPT_KEYS == 1 && $OBSERVE_POSITION_113 != 1 ]]; then
  echo "DB518 exact layer-0 keys require position-113 observation" >&2
  exit 2
fi
runner_variant_args=(--full-width-rounded-then-slice)
if [[ $EXACT_LAYER0_PROMPT_KEYS == 1 ]]; then
  ACQUIRED_CODE_HASH=e2a7b4e1d5298d39cb5f0e736808baa96b1c49a3
  ACQUIRED_RUN_TAG=greenfield_pp16_feature2_layer0_db518_acquire_20260829T101159660007493Z
  ACQUIRED_COMPACT_EVIDENCE=$WORKTREE/docs/artifacts/pp16-feature2-layer0-db518-hlo-acquisition.json
  ACQUIRED_COMPACT_EVIDENCE_SHA=52b9e227d124d67cbc908cc02b9a4b47182443dd7482766221b6b5367ededc02
  ACQUIRED_EVIDENCE_LEDGER_SHA=d6cd668388120a1ce29823584079ba4e06463ca9f0d57d8cf4214dbea1a79bfd
  ACQUIRED_RUNNER_SHA=e4900076353f755875f81e9df11c2b2731fe93e38465b433dec0e8b38aca55da
  ACQUIRED_SUMMARY_SHA=f24fc959853b0945aeff03fedb67c95d55890ac63d0ee78c597ce30f7c7e1ef5
  ACQUIRED_TERMINAL_SHA=2a23d911e379107ed984cb9f102432240d07c94e4d80ae96b3f5265e8e952609
  ACQUIRED_TERMINAL_SELF_SHA=c17adbee494c9b69d2c73b9b17e8e0d4dafd1a23d6b92899421c61765ea72c49
  ACQUIRED_MAIN_STABLE_SHA=a79d4823db0fecf8b1bc980b16df283bbe7a497795fa06ff783d3cde8fa98f6b
  ACQUIRED_MAIN_OPTIMIZED_SHA=0b0481d16811c6ad9cb34f73d3b13690f4200f5b45a76e54936ab132cdb9d133
  ACQUIRED_MAIN_CANONICAL_SHA=56b9b88dc081dd5d3ea2219d46be641128cbed6e9759754be7f87578b51b5d8c
  ACQUIRED_MAIN_CANONICAL_BYTES=6863602
  ACQUIRED_MAIN_STACK_FRAME_REFERENCES=15339
  runner_variant_args+=(--observe-position-113 --exact-layer0-prompt-keys)
elif [[ $OBSERVE_POSITION_113 == 1 ]]; then
  ACQUIRED_CODE_HASH=09285f7601ea1ca9cdc3f5447833d1440afb42dd
  ACQUIRED_RUN_TAG=greenfield_pp16_feature2_position113_acquire_20260829T072101902362381Z
  ACQUIRED_COMPACT_EVIDENCE=$WORKTREE/docs/artifacts/pp16-feature2-position113-hlo-acquisition.json
  ACQUIRED_COMPACT_EVIDENCE_SHA=644e59a8b108c8fa0f48b2135713e0084908c59f06215139f81bdc6e1713968d
  ACQUIRED_EVIDENCE_LEDGER_SHA=bbf01f34b20fb4d0c1af66573a5ee01c096abc870563db5f2ffa5bef39142728
  ACQUIRED_RUNNER_SHA=331e7dba5b638ba5560962bc606b7bf3cc9c9069a7ebe418ecda0c9824334bb6
  ACQUIRED_SUMMARY_SHA=043073fc134be9fc2c1ab85d536d70012a6dc8a117d9e3bd2dde01b9f6571d72
  ACQUIRED_TERMINAL_SHA=ddda21877df8568c2504f38eaaa0df5a0538e7f295033c9b3d45c070fe6cd5d4
  ACQUIRED_TERMINAL_SELF_SHA=f5662dbf0ff3ca74efa4e2a701701281ba72eb680b1f5fc8352faf96993f05d4
  ACQUIRED_MAIN_STABLE_SHA=bc2fcc77e84217ee0264e3856309f298e61e70ac0f4e3ecebecf72d290c26035
  ACQUIRED_MAIN_OPTIMIZED_SHA=f1cd8286460c17e40e7a96d3aefc830ecba97338d55aaf3d301f54c4bcb91bfe
  ACQUIRED_MAIN_CANONICAL_SHA=5b5dfacf015c579f41661868486e627bb383322fe8fb5c64597c259e85c4a10e
  ACQUIRED_MAIN_CANONICAL_BYTES=6662190
  ACQUIRED_MAIN_STACK_FRAME_REFERENCES=14781
  runner_variant_args+=(--observe-position-113)
else
  ACQUIRED_CODE_HASH=a2ea1e9439493b0093824d0084bc46c813fc1c33
  ACQUIRED_RUN_TAG=greenfield_pp16_feature2_prefill_acquire_20260829T042559840055981Z
  ACQUIRED_COMPACT_EVIDENCE=$WORKTREE/docs/artifacts/pp16-feature2-sealed-hlo-acquisition.json
  ACQUIRED_COMPACT_EVIDENCE_SHA=9498097422e8bd06e637a6e78360bae8156992777cd5c92294ab9321139025e0
  ACQUIRED_EVIDENCE_LEDGER_SHA=7741bef152843992afc23902e5cab20be52dac55722f1da1ff7f017990449036
  ACQUIRED_RUNNER_SHA=75bdd75f03fa4bba530f7863ae3b5728094745ea2e3884fb4d0ae72e8767c566
  ACQUIRED_SUMMARY_SHA=1dda18f3d007c6859911d29d9b1e526c75e37d748ed3d27f61683026e2218c45
  ACQUIRED_TERMINAL_SHA=b483460ebee19140d5fc30df77fa9851ad5bb080b307c740a0f6baa781b1d271
  ACQUIRED_TERMINAL_SELF_SHA=abec1454910e319be88e72eb8d7e5dbb55b841911990b1f2f76a1795a536fbde
  ACQUIRED_MAIN_STABLE_SHA=6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4
  ACQUIRED_MAIN_OPTIMIZED_SHA=a6307a5f487b0cfcd79712c45ace89753cf0dc54e332b5c24fe9010a36ae3175
  ACQUIRED_MAIN_CANONICAL_SHA=9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5
  ACQUIRED_MAIN_CANONICAL_BYTES=6558627
  ACQUIRED_MAIN_STACK_FRAME_REFERENCES=14561
fi
ACQUIRED_REMOTE_PREFIX=$APPROVED_BUCKET/results/$ACQUIRED_RUN_TAG
readonly ACQUIRED_CODE_HASH ACQUIRED_RUN_TAG ACQUIRED_REMOTE_PREFIX
readonly ACQUIRED_COMPACT_EVIDENCE ACQUIRED_COMPACT_EVIDENCE_SHA
readonly ACQUIRED_EVIDENCE_LEDGER_SHA ACQUIRED_RUNNER_SHA ACQUIRED_SUMMARY_SHA
readonly ACQUIRED_TERMINAL_SHA ACQUIRED_TERMINAL_SELF_SHA
readonly ACQUIRED_MAIN_STABLE_SHA ACQUIRED_MAIN_OPTIMIZED_SHA
readonly ACQUIRED_MAIN_CANONICAL_SHA ACQUIRED_MAIN_CANONICAL_BYTES
readonly ACQUIRED_MAIN_STACK_FRAME_REFERENCES runner_variant_args

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
if [[ $EXACT_LAYER0_PROMPT_KEYS == 1 ]]; then
  TAG=${GLM_GREENFIELD_PP16_FEATURE2_TAG:-greenfield_pp16_feature2_layer0_db518_numerical_$(date -u +%Y%m%dT%H%M%S%NZ)}
elif [[ $OBSERVE_POSITION_113 == 1 ]]; then
  TAG=${GLM_GREENFIELD_PP16_FEATURE2_TAG:-greenfield_pp16_feature2_position113_numerical_$(date -u +%Y%m%dT%H%M%S%NZ)}
else
  TAG=${GLM_GREENFIELD_PP16_FEATURE2_TAG:-greenfield_pp16_feature2_prefill_numerical_$(date -u +%Y%m%dT%H%M%S%NZ)}
fi
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX

if [[ $EXACT_LAYER0_PROMPT_KEYS == 1 ]]; then
  [[ $TAG =~ ^greenfield_pp16_feature2_layer0_db518_numerical_[0-9]{8}T[0-9]{15}Z$ ]]
elif [[ $OBSERVE_POSITION_113 == 1 ]]; then
  [[ $TAG =~ ^greenfield_pp16_feature2_position113_numerical_[0-9]{8}T[0-9]{15}Z$ ]]
else
  [[ $TAG =~ ^greenfield_pp16_feature2_prefill_numerical_[0-9]{8}T[0-9]{15}Z$ ]]
fi
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

verify_acquired_hlo_authorization() {
  local directory="$RUN_DIR/acquired_hlo_authorization" relative expected
  local archive="$RUN_DIR/acquired_hlo_authorization/archive"
  mkdir -p "$archive"
  for relative in evidence.sha256 runner.json summary.json HLO_ACQUIRED; do
    gcloud storage cat "$ACQUIRED_REMOTE_PREFIX/$relative" \
      >"$directory/$relative"
  done
  gcloud storage ls --recursive "$ACQUIRED_REMOTE_PREFIX/**" \
    >"$directory/remote_objects.txt"
  while read -r expected relative; do
    relative=${relative#\*}
    relative=${relative#./}
    case $relative in
      "" | /* | ../* | */../* | */..)
        echo "unsafe acquired ledger path: $relative" >&2
        return 1
        ;;
    esac
    mkdir -p "$archive/$(dirname "$relative")"
    gcloud storage cat "$ACQUIRED_REMOTE_PREFIX/$relative" \
      >"$archive/$relative"
    [[ $(sha256sum "$archive/$relative" | awk '{print $1}') == "$expected" ]] ||
      return 1
  done <"$directory/evidence.sha256"
  PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
    "$directory" "$ACQUIRED_COMPACT_EVIDENCE" \
    "$ACQUIRED_COMPACT_EVIDENCE_SHA" "$ACQUIRED_CODE_HASH" \
    "$ACQUIRED_RUN_TAG" "$ACQUIRED_REMOTE_PREFIX" \
    "$ACQUIRED_EVIDENCE_LEDGER_SHA" "$ACQUIRED_RUNNER_SHA" \
    "$ACQUIRED_SUMMARY_SHA" "$ACQUIRED_TERMINAL_SHA" \
    "$ACQUIRED_TERMINAL_SELF_SHA" "$ACQUIRED_MAIN_STABLE_SHA" \
    "$ACQUIRED_MAIN_OPTIMIZED_SHA" "$ACQUIRED_MAIN_CANONICAL_SHA" \
    "$ACQUIRED_MAIN_CANONICAL_BYTES" \
    "$ACQUIRED_MAIN_STACK_FRAME_REFERENCES" \
    "$OBSERVE_POSITION_113" \
    "$EXACT_LAYER0_PROMPT_KEYS" \
    "$RUN_DIR/acquisition_authorization.json" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

(directory,compact_path,compact_sha,code_hash,run_tag,remote,
 ledger_sha,runner_sha,summary_sha,terminal_sha,terminal_self_sha,
 stable_sha,optimized_sha,canonical_sha,canonical_bytes,stack_refs,observe_p113,
 exact_layer0_prompt_keys,
 output)=sys.argv[1:]
directory=Path(directory); compact_path=Path(compact_path); output=Path(output)
canonical_bytes=int(canonical_bytes); stack_refs=int(stack_refs)
observe_p113=bool(int(observe_p113))
exact_layer0_prompt_keys=bool(int(exact_layer0_prompt_keys))

def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

ledger_path=directory/'evidence.sha256'
runner_path=directory/'runner.json'
summary_path=directory/'summary.json'
terminal_path=directory/'HLO_ACQUIRED'
if digest(compact_path)!=compact_sha or digest(ledger_path)!=ledger_sha or digest(runner_path)!=runner_sha or digest(summary_path)!=summary_sha or digest(terminal_path)!=terminal_sha:
    raise SystemExit('feature2 acquired authorization identity drifted')
compact=json.loads(compact_path.read_text())
runner=json.loads(runner_path.read_text())
summary=json.loads(summary_path.read_text())
terminal=json.loads(terminal_path.read_text())
terminal_without_self={key:terminal[key] for key in ('artifact_kind','evidence_sha256','status','summary_sha256')}
terminal_raw=json.dumps(terminal_without_self,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
if terminal.get('marker_self_sha256')!=terminal_self_sha or sha256(terminal_raw).hexdigest()!=terminal_self_sha or terminal.get('evidence_sha256')!=ledger_sha or terminal.get('summary_sha256')!=summary_sha:
    raise SystemExit('feature2 acquired terminal binding drifted')
if compact.get('status')!='HLO_ACQUIRED' or compact.get('code_hash')!=code_hash or compact.get('run_tag')!=run_tag or compact.get('remote_prefix')!=remote or compact.get('main_executed') is not False or compact.get('numerical_claim') is not False or compact.get('performance_claim') is not False:
    raise SystemExit('feature2 compact acquisition authorization drifted')
if exact_layer0_prompt_keys:
    certificate=compact.get('structural_certificate',{})
    if compact.get('artifact_kind')!='greenfield_pp16_feature2_layer0_db518_hlo_acquisition' or compact.get('remote_object_count')!=20 or compact.get('evidence_ledger_sha256')!=ledger_sha or compact.get('runner_sha256')!=runner_sha or compact.get('summary_sha256')!=summary_sha or compact.get('hlo_acquired_marker_sha256')!=terminal_sha or certificate.get('causal') is not True or certificate.get('scan_count')!=4 or certificate.get('scan_limits')!=[2048,2048,2048,2012] or certificate.get('physical_m64_projection_count')!=16 or certificate.get('cache_write_count')!=4 or certificate.get('zero_cache_initializer_count')!=1 or certificate.get('cache_handoff_count')!=3 or certificate.get('layer0_key_topology_sha256')!='235dfa726f82a152bf0218506cf8d3345520a7673ab7a6b908ef06df821465bd' or certificate.get('position113_branch_topology_sha256')!='92a1b18b523984dd98d2579d25668b6bd582f6f9f23a26d9e4449b29a81ee46d' or certificate.get('rebased_identity_hostile_test_count')!=6:
        raise SystemExit('feature2 DB518 compact acquisition evidence linkage drifted')
elif observe_p113:
    certificate=compact.get('structural_certificate',{})
    if compact.get('artifact_kind')!='greenfield_pp16_feature2_position113_hlo_acquisition' or compact.get('remote_object_count')!=20 or compact.get('evidence_ledger_sha256')!=ledger_sha or compact.get('runner_sha256')!=runner_sha or compact.get('summary_sha256')!=summary_sha or compact.get('hlo_acquired_marker_sha256')!=terminal_sha or certificate.get('causal') is not True or certificate.get('branch_topology_record_count')!=32 or certificate.get('branch_topology_sha256')!='05322ce7ad09db5463e17d4a7e31142edaffe128abe5e9d85d5ee81f5e3eb02b' or certificate.get('rebased_identity_hostile_test_count')!=10:
        raise SystemExit('feature2 p113 compact acquisition evidence linkage drifted')
else:
    if compact.get('review_verdict')!='APPROVE COMMIT AND ONE COMPILE-ONLY REPEAT':
        raise SystemExit('feature2 compact acquisition review drifted')
    if compact.get('evidence',{}).get('remote_object_count_including_terminal')!=20 or compact.get('evidence',{}).get('evidence_ledger_sha256')!=ledger_sha or compact.get('evidence',{}).get('runner_sha256')!=runner_sha or compact.get('evidence',{}).get('summary_sha256')!=summary_sha or compact.get('evidence',{}).get('terminal_file_sha256')!=terminal_sha or compact.get('evidence',{}).get('terminal_marker_self_sha256')!=terminal_self_sha:
        raise SystemExit('feature2 compact acquisition evidence linkage drifted')
for record in (runner,summary):
    if record.get('status')!='HLO_ACQUIRED' or record.get('code_hash')!=code_hash or record.get('main_executed') is not False or record.get('numerical_claim') is not False or record.get('performance_claim') is not False or bool(record.get('observe_position_113',False)) is not observe_p113 or bool(record.get('exact_layer0_prompt_keys',False)) is not exact_layer0_prompt_keys:
        raise SystemExit('feature2 acquired claim boundary drifted')
if runner.get('main_execution_count')!=0 or runner.get('compile_only') is not True or runner.get('full_width_rounded_then_slice') is not True or runner.get('sealed_boundary_capture') is not True or bool(runner.get('observe_position_113',False)) is not observe_p113 or bool(runner.get('exact_layer0_prompt_keys',False)) is not exact_layer0_prompt_keys:
    raise SystemExit('feature2 acquired execution variant drifted')
main=runner.get('hlo',{}).get('feature2_main',{})
canonical=main.get('execution_canonical_hlo',{})
if main.get('stablehlo',{}).get('sha256')!=stable_sha or main.get('optimized_hlo',{}).get('sha256')!=optimized_sha or canonical.get('sha256')!=canonical_sha or canonical.get('byte_count')!=canonical_bytes or canonical.get('canonicalizer_version')!=1 or canonical.get('stripped_stack_frame_references')!=stack_refs or canonical.get('canonicalizer_code_hash')!=code_hash:
    raise SystemExit('feature2 acquired HLO identity drifted')
entries={}
for line in ledger_path.read_text().splitlines():
    expected,relative=line.split(maxsplit=1)
    relative=relative.removeprefix('*').removeprefix('./')
    if relative in entries:
        raise SystemExit('feature2 acquired ledger has duplicate paths')
    if not relative or relative.startswith('/') or '..' in Path(relative).parts:
        raise SystemExit('feature2 acquired ledger path is unsafe')
    entries[relative]=expected
archive=directory/'archive'
observed_archive_files=sorted(
    str(path.relative_to(archive)) for path in archive.rglob('*') if path.is_file()
)
if observed_archive_files!=sorted(entries) or any(digest(archive/name)!=expected for name,expected in entries.items()):
    raise SystemExit('feature2 acquired archive bytes drifted')
required_entries={
    'runner.json':runner_sha,
    'summary.json':summary_sha,
    'hlo/feature2_main.stablehlo.mlir':stable_sha,
    'hlo/feature2_main.optimized_hlo.txt':optimized_sha,
    'hlo/feature2_main.execution_canonical_hlo.txt':canonical_sha,
}
if any(entries.get(name)!=expected for name,expected in required_entries.items()):
    raise SystemExit('feature2 acquired ledger content drifted')
expected_objects=sorted([f'{remote}/evidence.sha256',f'{remote}/HLO_ACQUIRED',*(f'{remote}/{name}' for name in entries)])
observed_objects=sorted(line for line in (directory/'remote_objects.txt').read_text().splitlines() if line)
if len(expected_objects)!=20 or observed_objects!=expected_objects:
    raise SystemExit('feature2 acquired remote object set drifted')
record={
    'artifact_kind':'greenfield_pp16_feature2_acquisition_authorization',
    'acquired_code_hash':code_hash,
    'acquired_compact_evidence_sha256':compact_sha,
    'acquired_evidence_ledger_sha256':ledger_sha,
    'acquired_remote_object_count':len(observed_objects),
    'acquired_remote_prefix':remote,
    'acquired_run_tag':run_tag,
    'acquired_terminal_marker_self_sha256':terminal_self_sha,
    'main_canonical_hlo_byte_count':canonical_bytes,
    'main_canonical_hlo_sha256':canonical_sha,
    'main_optimized_hlo_sha256':optimized_sha,
    'main_stablehlo_sha256':stable_sha,
    'exact_layer0_prompt_keys':exact_layer0_prompt_keys,
    'observe_position_113':observe_p113,
    'passed':True,
}
output.write_text(json.dumps(record,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY
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
crc_values = [
    str(after[key])
    for key in ("crc32c_hash", "crc32c")
    if after.get(key) not in (None, "")
]
matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
if (
    not crc_values
    or len(set(crc_values)) != 1
    or matches != [(remote, str(after.get("generation", "")))]
    or path.stat().st_size != int(after.get("size", -1))
    or crc != crc_values[0]
):
    raise SystemExit("terminal rollback receipt/generation/size/CRC authentication failed")
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
  "$OBSERVE_POSITION_113" \
  "$EXACT_LAYER0_PROMPT_KEYS" \
  "$POSITION113_SEALED_REJECTION" "$POSITION113_SEALED_REJECTION_SHA" \
  "$POSITION113_ORACLE" "$POSITION113_ORACLE_SHA" \
  "$DB518_PROMPT_CACHE_DIR" \
  "$RUN_DIR/source_identity.json" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.benchmarking.pp16_feature2_acquisition import inspect_feature2_event1_lineage
from glm_tpu.greenfield.benchmarking.pp16_feature2_loader import inspect_feature2_selective_plan
from glm_tpu.greenfield.benchmarking.pp16_feature2_prefill import build_feature2_prefill_graph, load_feature2_prefill_inputs
from glm_tpu.greenfield.benchmarking.pp16_feature2_position113 import PP16_FEATURE2_DB518_PROMPT_CACHE_MANIFEST_SHA256, PP16_FEATURE2_DB518_PROMPT_CACHE_SHA256, PP16_FEATURE2_DB518_PROMPT_CACHE_TENSOR_SHA256
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import derive_feature2_tensor_allowlist, read_feature2_owner_headers
from glm_tpu.greenfield.validation.prompt_index_cache import inspect_legacy_prompt_index_cache

(runtime,remote,runtime_sha,token,token_sha,dsa,dsa_sha,layer1,layer1_sha,
 db529,db529_contract,db529_tensor,graph_sha,observe_p113,exact_layer0_prompt_keys,p113_baseline,
 p113_baseline_sha,p113_oracle,p113_oracle_sha,db518_prompt_cache,output)=sys.argv[1:]
runtime=Path(runtime); token=Path(token); dsa=Path(dsa); layer1=Path(layer1); db529=Path(db529)
observe_p113=bool(int(observe_p113)); p113_baseline=Path(p113_baseline); p113_oracle=Path(p113_oracle)
exact_layer0_prompt_keys=bool(int(exact_layer0_prompt_keys))
db518_prompt_cache=Path(db518_prompt_cache)
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
if observe_p113 and (sha256(p113_baseline.read_bytes()).hexdigest()!=p113_baseline_sha or sha256(p113_oracle.read_bytes()).hexdigest()!=p113_oracle_sha):
    raise SystemExit('feature2 p113 numerical source identity drifted')
db518_cache_identity=None
if exact_layer0_prompt_keys:
    cache_manifest,cache_bits=inspect_legacy_prompt_index_cache(db518_prompt_cache,expected_manifest_sha256=PP16_FEATURE2_DB518_PROMPT_CACHE_MANIFEST_SHA256)
    if cache_manifest.get('prompt_index_key_bfloat16_sha256')!=PP16_FEATURE2_DB518_PROMPT_CACHE_SHA256 or cache_manifest.get('tensor_file',{}).get('sha256')!=PP16_FEATURE2_DB518_PROMPT_CACHE_TENSOR_SHA256 or sha256(cache_bits.tobytes()).hexdigest()!=PP16_FEATURE2_DB518_PROMPT_CACHE_SHA256:
        raise SystemExit('feature2 DB518 prompt cache source identity drifted')
    db518_cache_identity={'manifest_sha256':PP16_FEATURE2_DB518_PROMPT_CACHE_MANIFEST_SHA256,'prompt_cache_sha256':PP16_FEATURE2_DB518_PROMPT_CACHE_SHA256,'tensor_file_sha256':PP16_FEATURE2_DB518_PROMPT_CACHE_TENSOR_SHA256}
record={'artifact_kind':'greenfield_pp16_feature2_source_identity','db518_prompt_cache_identity':db518_cache_identity,'runtime_root':str(runtime),'runtime_remote':remote,'runtime_manifest_sha256':runtime_sha,'selective_plan':inspect_feature2_selective_plan(runtime),'event1_target_lineage':lineage,'exact_layer0_prompt_keys':exact_layer0_prompt_keys,'graph_sha256':graph_sha,'observe_position_113':observe_p113,'position113_sealed_rejection_sha256':p113_baseline_sha if observe_p113 else None,'position113_oracle_sha256':p113_oracle_sha if observe_p113 else None}
Path(output).write_text(json.dumps(record,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY

say "authenticating the reviewed sealed-HLO acquisition and exact remote object set"
verify_acquired_hlo_authorization

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
      "${runner_variant_args[@]}" \
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
  "$FULL_WIDTH_ROUNDED_THEN_SLICE" \
  "$ACQUIRED_MAIN_STABLE_SHA" "$ACQUIRED_MAIN_CANONICAL_SHA" \
  "$ACQUIRED_MAIN_CANONICAL_BYTES" "$ACQUIRED_MAIN_STACK_FRAME_REFERENCES" \
  "$ACQUIRED_JAX_VERSION" "$ACQUIRED_JAXLIB_VERSION" "$ACQUIRED_LIBTPU_VERSION" \
  "$OBSERVE_POSITION_113" \
  "$EXACT_LAYER0_PROMPT_KEYS" \
  "$POSITION113_SEALED_REJECTION" "$POSITION113_SEALED_REJECTION_SHA" \
  "$POSITION113_ORACLE" "$POSITION113_ORACLE_SHA" \
  "$DB518_PROMPT_CACHE_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.benchmarking.pp16_feature2_hlo import validate_feature2_sealed_hlo_archive_identity
from glm_tpu.greenfield.benchmarking.pp16_feature2_numerical import compare_feature2_db518_numerical_capture, compare_feature2_full_width_numerical_capture, validate_feature2_in_process_cleanup
from glm_tpu.greenfield.benchmarking.pp16_feature2_position113 import compare_feature2_db518_position113_capture, compare_feature2_position113_capture

run=Path(sys.argv[1]); pin,tag,remote,elapsed=sys.argv[2:6]
token,dsa,layer1,db529,db550=map(Path,sys.argv[6:11])
full_width_rounded_then_slice=bool(int(sys.argv[11]))
stable_pin,canonical_pin=sys.argv[12:14]
canonical_bytes,stack_frame_references=map(int,sys.argv[14:16])
expected_runtime_pins=dict(zip(('jax','jaxlib','libtpu'),sys.argv[16:19],strict=True))
observe_p113=bool(int(sys.argv[19]))
exact_layer0_prompt_keys=bool(int(sys.argv[20]))
p113_baseline=Path(sys.argv[21]); p113_baseline_sha=sys.argv[22]
p113_oracle=Path(sys.argv[23]); p113_oracle_sha=sys.argv[24]
db518_prompt_cache=Path(sys.argv[25])
runner=json.loads((run/'runner.json').read_text())
source=json.loads((run/'source_identity.json').read_text())
authorization=json.loads((run/'acquisition_authorization.json').read_text())
if full_width_rounded_then_slice is not True or authorization.get('passed') is not True or authorization.get('exact_layer0_prompt_keys') is not exact_layer0_prompt_keys:
    raise SystemExit('feature2 numerical authorization drifted')
if runner.get('status')!='NUMERICAL_CAPTURED' or runner.get('code_hash')!=pin or runner.get('compile_only') is not False or runner.get('full_width_rounded_then_slice') is not True or runner.get('sealed_boundary_capture') is not True or runner.get('observe_position_113') is not observe_p113 or runner.get('exact_layer0_prompt_keys') is not exact_layer0_prompt_keys or runner.get('main_executed') is not True or runner.get('main_execution_count')!=1 or runner.get('numerical_claim') is not False or runner.get('performance_claim') is not False:
    raise SystemExit('feature2 numerical-capture claim boundary drifted')
if runner.get('graph_sha256')!=source.get('graph_sha256') or runner.get('event1_target_lineage')!=source.get('event1_target_lineage') or runner.get('selective_plan')!=source.get('selective_plan') or source.get('observe_position_113') is not observe_p113 or source.get('exact_layer0_prompt_keys') is not exact_layer0_prompt_keys:
    raise SystemExit('feature2 numerical source lineage drifted')
expected_db518_cache_identity={'manifest_sha256':'acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab','prompt_cache_sha256':'3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1','tensor_file_sha256':'36303f0638661b4a56d3c9a1d4023a9b39eb19dbfd6d48e0718c29a45c41c07a'} if exact_layer0_prompt_keys else None
if source.get('db518_prompt_cache_identity')!=expected_db518_cache_identity:
    raise SystemExit('feature2 DB518 prompt cache source lineage drifted')
if observe_p113 and (source.get('position113_sealed_rejection_sha256')!=p113_baseline_sha or source.get('position113_oracle_sha256')!=p113_oracle_sha):
    raise SystemExit('feature2 p113 numerical source pins drifted')
if observe_p113 and (sha256(p113_baseline.read_bytes()).hexdigest()!=p113_baseline_sha or sha256(p113_oracle.read_bytes()).hexdigest()!=p113_oracle_sha):
    raise SystemExit('feature2 p113 numerical source bytes changed before classification')
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
stable_path=run/'hlo'/main.get('stablehlo',{}).get('filename','')
optimized_path=run/'hlo'/main.get('optimized_hlo',{}).get('filename','')
canonical_path=run/'hlo'/canonical.get('filename','')
if stable_path.parent!=run/'hlo' or not stable_path.name.endswith('.stablehlo.mlir') or optimized_path.parent!=run/'hlo' or not optimized_path.name.endswith('.optimized_hlo.txt') or canonical_path.parent!=run/'hlo' or canonical_path.name!='feature2_main.execution_canonical_hlo.txt':
    raise SystemExit('feature2 numerical executable HLO archive paths drifted')
archive_identity=validate_feature2_sealed_hlo_archive_identity(stable_path,optimized_path,canonical_path,expected_stablehlo_sha256=stable_pin,expected_canonical_sha256=canonical_pin,expected_canonical_bytes=canonical_bytes,expected_canonicalizer_version=1,expected_stripped_stack_frame_references=stack_frame_references)
if main.get('stablehlo',{}).get('sha256')!=archive_identity['stablehlo_sha256'] or main.get('optimized_hlo',{}).get('sha256')!=archive_identity['optimized_hlo_sha256']:
    raise SystemExit('feature2 numerical raw HLO identity drifted')
expected_terminal_shapes=[[1,2048],[1],[1,2048],[2,1,3072],[2,1,32,256],[1,576],[2,1,6144],[2,1,2048],[2,1,32,128],[2,1,32],[2,16,256,640],[2,16,256,128],[2,16,256,128],[2,2],[1]]
expected_terminal_dtypes=['int32','int32','float32','bfloat16','bfloat16','bfloat16','bfloat16','bfloat16','float32','float32','bfloat16','bfloat16','bfloat16','uint32','bool']
expected_stable_types=['tensor<1x2048xi32>','tensor<1xi32>','tensor<1x2048xf32>','tensor<2x1x3072xbf16>','tensor<2x1x32x256xbf16>','tensor<1x576xbf16>','tensor<2x1x6144xbf16>','tensor<2x1x2048xbf16>','tensor<2x1x32x128xf32>','tensor<2x1x32xf32>','tensor<2x16x256x640xbf16>','tensor<2x16x256x128xbf16>','tensor<2x16x256x128xbf16>','tensor<2x2xui32>','tensor<1xi1>']
expected_optimized_roots=[{'dtype':dtype,'shape':shape} for dtype,shape in [('s32',[1,2048]),('s32',[1]),('f32',[1,2048]),('bf16',[1,1,3072]),('bf16',[1,1,32,256]),('bf16',[1,576]),('bf16',[1,1,6144]),('bf16',[1,1,2048]),('f32',[1,1,32,128]),('f32',[1,1,32]),('bf16',[1,16,256,640]),('bf16',[1,16,256,128]),('bf16',[1,16,256,128]),('u32',[1,2]),('pred',[1])]]
expected_sealed_bindings={'6':'greenfield_pp16_feature2_sealed_normalized_hidden','7':'greenfield_pp16_feature2_sealed_q_a_state','8':'greenfield_pp16_feature2_sealed_dsa_query','9':'greenfield_pp16_feature2_sealed_dsa_head_weights'}
expected_observer_bindings={}
if observe_p113:
    expected_terminal_shapes += [[2,1,6144],[2,1,2048],[2,1,32,128],[2,1,32],[2,1,128],[2,1,2048],[2,1],[2,1,2048],[2,1]]
    expected_terminal_dtypes += ['bfloat16','bfloat16','float32','float32','float32','int32','int32','float32','int32']
    expected_stable_types += ['tensor<2x1x6144xbf16>','tensor<2x1x2048xbf16>','tensor<2x1x32x128xf32>','tensor<2x1x32xf32>','tensor<2x1x128xf32>','tensor<2x1x2048xi32>','tensor<2x1xi32>','tensor<2x1x2048xf32>','tensor<2x1xi32>']
    expected_optimized_roots += [{'dtype':dtype,'shape':shape} for dtype,shape in [('bf16',[1,1,6144]),('bf16',[1,1,2048]),('f32',[1,1,32,128]),('f32',[1,1,32]),('f32',[1,1,128]),('s32',[1,1,2048]),('s32',[1,1]),('f32',[1,1,2048]),('s32',[1,1])]]
    expected_observer_bindings={'15':'greenfield_pp16_feature2_p113_normalized_hidden','16':'greenfield_pp16_feature2_p113_q_a_state','17':'greenfield_pp16_feature2_p113_dsa_query','18':'greenfield_pp16_feature2_p113_dsa_head_weights','19':'greenfield_pp16_feature2_p113_current_key','20':'greenfield_pp16_feature2_p113_selected_positions','21':'greenfield_pp16_feature2_p113_selected_valid_counts','22':'greenfield_pp16_feature2_p113_selected_scores','23':'greenfield_pp16_feature2_p113_observation_count'}
stable=main['stablehlo_contract']; optimized=main['optimized_contract']; terminal=main['terminal_contract']
expected_canonical={'byte_count':canonical_bytes,'canonicalizer_version':1,'sha256':canonical_pin,'stripped_stack_frame_references':stack_frame_references}
expected_output_count=24 if observe_p113 else 15
if stable.get('sealed_boundary_capture') is not True or stable.get('observe_position_113') is not observe_p113 or stable.get('exact_layer0_prompt_keys') is not exact_layer0_prompt_keys or stable.get('stablehlo_sha256')!=archive_identity['stablehlo_sha256'] or stable.get('output_count')!=expected_output_count or stable.get('terminal_shapes')!=expected_terminal_shapes or stable.get('terminal_types')!=expected_stable_types:
    raise SystemExit('feature2 numerical StableHLO sealed terminal drifted')
if optimized.get('sealed_boundary_capture') is not True or optimized.get('observe_position_113') is not observe_p113 or optimized.get('exact_layer0_prompt_keys') is not exact_layer0_prompt_keys or optimized.get('output_count')!=expected_output_count or optimized.get('root_shapes')!=expected_optimized_roots or optimized.get('sealed_bindings')!=expected_sealed_bindings or optimized.get('position113_observer_bindings')!=expected_observer_bindings or optimized.get('position113_observer_root_hints_causal') is not observe_p113 or {key:optimized.get('sealed_canonical_hlo_identity',{}).get(key) for key in expected_canonical}!=expected_canonical:
    raise SystemExit('feature2 numerical optimized-HLO sealed terminal drifted')
if terminal.get('sealed_boundary_capture') is not True or terminal.get('observe_position_113') is not observe_p113 or terminal.get('output_count')!=expected_output_count or terminal.get('terminal_shapes')!=expected_terminal_shapes or terminal.get('terminal_dtypes')!=expected_terminal_dtypes:
    raise SystemExit('feature2 numerical abstract sealed terminal drifted')
if observe_p113:
    lineage=optimized.get('position113_lineage',{})
    expected_p113_topology='92a1b18b523984dd98d2579d25668b6bd582f6f9f23a26d9e4449b29a81ee46d' if exact_layer0_prompt_keys else '05322ce7ad09db5463e17d4a7e31142edaffe128abe5e9d85d5ee81f5e3eb02b'
    if lineage.get('causal') is not True or lineage.get('branch_topology_record_count')!=32 or lineage.get('branch_topology_sha256')!=expected_p113_topology:
        raise SystemExit('feature2 p113 numerical causal lineage drifted')
if exact_layer0_prompt_keys:
    lineage=optimized.get('db518_layer0_key_lineage',{})
    if lineage.get('causal') is not True or lineage.get('scan_count')!=4 or [scan.get('rows') for scan in lineage.get('scans',[])]!=[2048,2048,2048,2012] or lineage.get('physical_m64_projection_count')!=16 or lineage.get('cache_write_count')!=4 or lineage.get('topology_sha256')!='235dfa726f82a152bf0218506cf8d3345520a7673ab7a6b908ef06df821465bd' or sum(scan.get('cache_seed_kind')=='zero' for scan in lineage.get('scans',[]))!=1 or sum(scan.get('cache_seed_kind')=='handoff' for scan in lineage.get('scans',[]))!=3:
        raise SystemExit('feature2 DB518 numerical causal lineage drifted')
if canonical.get('sha256')!=canonical_pin or canonical.get('byte_count')!=canonical_bytes or canonical.get('stripped_stack_frame_references')!=stack_frame_references or canonical.get('canonicalizer_version')!=1 or canonical.get('canonicalizer_code_hash')!=pin or {key:archive_identity['canonical_hlo_identity'].get(key) for key in expected_canonical}!=expected_canonical:
    raise SystemExit('feature2 numerical executable HLO drifted from acquired graph')
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
    required={'bytes_in_use','bytes_limit','largest_free_block_bytes','num_allocs','peak_bytes_in_use'}
    if any(not required.issubset(record) for record in records):
        raise SystemExit(f'feature2 measured-memory fields missing at {phase}')
if any(record['peak_bytes_in_use']>=record['bytes_limit'] or record['largest_free_block_bytes']<8*1024**3 for record in memory['after_execute']):
    raise SystemExit('feature2 numerical execution has unknown/insufficient HBM margin')
generated_code_size=main.get('memory_analysis',{}).get('generated_code_size_in_bytes')
cleanup=validate_feature2_in_process_cleanup(memory['after_cleanup'],generated_code_size_bytes=generated_code_size)
post_census_path=run/'census_post.txt'
post_hosts=[line.split()[1] for line in post_census_path.read_text().splitlines() if line.startswith('CENSUS_OK ')]
if len(post_hosts)!=8 or len(set(post_hosts))!=8:
    raise SystemExit('feature2 post-process cleanup census drifted')
cleanup.update({'post_process_authenticated_zero_work_hosts':8,'post_process_census_sha256':sha256(post_census_path.read_bytes()).hexdigest(),'terminal_cleanup_gate':'authenticated post-process 8/8 zero work'})
expected_mismatch_fields={'carried_bfloat16_bits','contract_valid','event1_positions','event1_scores','event1_valid_counts','layer1_current_key_bfloat16_bits','layer1_normalized_hidden_bfloat16_bits','layer1_q_a_state_bfloat16_bits','layer1_dsa_query_float32','layer1_dsa_head_weights_float32'}
if exact_layer0_prompt_keys:
    full_comparison=compare_feature2_db518_numerical_capture(run/'result.npz',token_oracle_dir=token,dsa_oracle_dir=dsa,layer1_internal_reference=layer1,db529_internal_dir=db529,db550_boundary=db550)
    position113_comparison=compare_feature2_db518_position113_capture(run/'result.npz',sealed_rejection_path=p113_baseline,accepted_prompt_key_path=p113_oracle,accepted_prompt_cache_dir=db518_prompt_cache)
    capture_sha=sha256((run/'result.npz').read_bytes()).hexdigest()
    if capture_sha!=full_comparison.get('capture_sha256') or capture_sha!=position113_comparison.get('capture_npz_sha256') or full_comparison.get('comparison_schema')!='db518_full_width_plus_position113_v1' or full_comparison.get('sealed_boundary_comparisons_required') is not True or set(full_comparison.get('mismatch_counts',{}))!=expected_mismatch_fields or position113_comparison.get('status') not in {'DB518_POSITION113_EXACT','DB518_POSITION113_REJECTED'} or position113_comparison.get('numerical_exactness_claim') is not False or position113_comparison.get('performance_claim') is not False or not all(position113_comparison.get('observer_owner_bitwise_equal',{}).values()):
        raise SystemExit('feature2 DB518 dual numerical comparison contract drifted')
    exact=bool(full_comparison.get('exact') and position113_comparison.get('boundary_exact'))
    comparison={'artifact_kind':'greenfield_pp16_feature2_db518_dual_numerical_comparison','capture_sha256':capture_sha,'claim_scope':'accepted event-1/layer-1/dense/current-key boundaries plus exact DB518 layer-0 position-113 normalized/key/cache boundary; no Gate-D, token-rate, or performance claim','exact':exact,'full_width_comparison':full_comparison,'performance_claim':False,'position113_comparison':position113_comparison,'status':'NUMERICAL_EXACT' if exact else 'NUMERICAL_REJECTED'}
elif observe_p113:
    comparison=compare_feature2_position113_capture(run/'result.npz',sealed_rejection_path=p113_baseline,accepted_prompt_key_path=p113_oracle)
    if sha256((run/'result.npz').read_bytes()).hexdigest()!=comparison['capture_npz_sha256'] or comparison.get('status')!='POSITION113_CAPTURE_CLASSIFIED' or comparison.get('numerical_exactness_claim') is not False or comparison.get('performance_claim') is not False or not all(comparison.get('ordinary_output_bitwise_equal',{}).values()) or not all(comparison.get('observer_owner_bitwise_equal',{}).values()):
        raise SystemExit('feature2 p113 numerical classification contract drifted')
else:
    comparison=compare_feature2_full_width_numerical_capture(run/'result.npz',token_oracle_dir=token,dsa_oracle_dir=dsa,layer1_internal_reference=layer1,db529_internal_dir=db529,db550_boundary=db550)
    if sha256((run/'result.npz').read_bytes()).hexdigest()!=comparison['capture_sha256']:
        raise SystemExit('feature2 numerical capture changed before sealing')
    if comparison.get('comparison_schema')!='full_width_sealed_boundaries_v2' or comparison.get('sealed_boundary_comparisons_required') is not True or set(comparison.get('mismatch_counts',{}))!=expected_mismatch_fields:
        raise SystemExit('feature2 numerical strict comparison contract drifted')
(run/'comparison.json').write_text(json.dumps(comparison,allow_nan=False,indent=2,sort_keys=True)+'\n')
if exact_layer0_prompt_keys:
    summary_kind='greenfield_pp16_feature2_db518_numerical_summary'; summary_claim=comparison['claim_scope']; summary_exact=comparison['exact']; summary_mismatches=comparison['full_width_comparison']['mismatch_counts']; summary_numerical_claim=comparison['exact']
elif observe_p113:
    summary_kind='greenfield_pp16_feature2_position113_numerical_summary'; summary_claim='one position-113 diagnostic capture; no exactness, Gate-D, token-rate, or performance claim'; summary_exact=False; summary_mismatches={}; summary_numerical_claim=False
else:
    summary_kind='greenfield_pp16_feature2_numerical_summary'; summary_claim=comparison['claim_scope']; summary_exact=comparison['exact']; summary_mismatches=comparison['mismatch_counts']; summary_numerical_claim=comparison['exact']
summary={'acquisition_authorization_sha256':sha256((run/'acquisition_authorization.json').read_bytes()).hexdigest(),'artifact_kind':summary_kind,'claim_scope':summary_claim,'cleanup':cleanup,'code_hash':pin,'elapsed_seconds_operational_only':int(elapsed),'exact':summary_exact,'exact_layer0_prompt_keys':exact_layer0_prompt_keys,'full_width_rounded_then_slice':True,'graph_sha256':runner['graph_sha256'],'hlo_sha256':{name:{kind:record[kind]['sha256'] for kind in ('stablehlo','optimized_hlo')} for name,record in runner['hlo'].items()},'main_canonical_hlo':canonical,'main_execution_count':1,'measured_memory':memory,'mismatch_counts':summary_mismatches,'numerical_claim':summary_numerical_claim,'observe_position_113':observe_p113,'performance_claim':False,'remote_prefix':remote,'run_tag':tag,'runtime_pins':runtime_pins,'sealed_boundary_capture':True,'status':comparison['status']}
(run/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2,sort_keys=True)+'\n')
PY

cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  find acquired_hlo_authorization -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log result.npz comparison.json \
    acquisition_authorization.json source_identity.json summary.json \
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
if [[ $EXACT_LAYER0_PROMPT_KEYS == 1 ]]; then
  [[ $terminal_status == NUMERICAL_EXACT || $terminal_status == NUMERICAL_REJECTED ]]
elif [[ $OBSERVE_POSITION_113 == 1 ]]; then
  [[ $terminal_status == POSITION113_CAPTURE_CLASSIFIED ]]
else
  [[ $terminal_status == NUMERICAL_EXACT || $terminal_status == NUMERICAL_REJECTED ]]
fi
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
crc_values = [
    str(record[key])
    for key in ("crc32c_hash", "crc32c")
    if record.get(key) not in (None, "")
]
matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
if (
    not crc_values
    or len(set(crc_values)) != 1
    or matches != [(remote, str(record.get("generation", "")))]
    or path.stat().st_size != int(record.get("size", -1))
    or crc != crc_values[0]
):
    raise SystemExit("terminal receipt/generation/size/CRC authentication failed")
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
if [[ $EXACT_LAYER0_PROMPT_KEYS == 1 || $OBSERVE_POSITION_113 == 0 ]]; then
  [[ $terminal_status == NUMERICAL_EXACT ]] || exit 3
fi
