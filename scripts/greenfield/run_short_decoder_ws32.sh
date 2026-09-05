#!/usr/bin/env bash
# Protected complete WS32 short-context acquisition/numerical workflow.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly RECOVERY_TOOL=$WORKTREE/scripts/greenfield/recover_short_decoder_ws32_acquisition.py
readonly INVENTORY=/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP8_LP4/greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/source_inventory.json
readonly INVENTORY_SHA=a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4
readonly TOPOLOGY_ROOT=/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records
readonly TOPOLOGY_SHA=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly TOPOLOGY_FLEET_SHA=4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301
readonly MESH_SHA=de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88
readonly ZERO_SHA=0000000000000000000000000000000000000000000000000000000000000000

[[ ${GLM_GREENFIELD_WS32_SHORT_DECODER:-0} == 1 ]] || {
  echo "WS32 short decoder is default-off; set GLM_GREENFIELD_WS32_SHORT_DECODER=1" >&2
  exit 2
}
MODE=${GLM_GREENFIELD_WS32_SHORT_DECODER_MODE:-off}
CONTEXT=${GLM_GREENFIELD_WS32_SHORT_DECODER_CONTEXT:-off}
RECOVER=${GLM_GREENFIELD_WS32_SHORT_DECODER_RECOVER:-0}
EXACT_DSA=${GLM_GREENFIELD_WS32_EXACT_DSA:-0}
STRATEGY_ND_DENSE=${GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE:-0}
[[ $MODE == acquire || $MODE == numerical ]] || {
  echo "WS32 mode must be acquire or numerical" >&2
  exit 2
}
[[ $CONTEXT == 2k || $CONTEXT == 8k ]] || {
  echo "WS32 context must be 2k or 8k" >&2
  exit 2
}
[[ $RECOVER == 0 || $RECOVER == 1 ]] || {
  echo "WS32 recovery flag must be 0 or 1" >&2
  exit 2
}
[[ $EXACT_DSA == 0 || $EXACT_DSA == 1 ]] || {
  echo "WS32 exact DSA flag must be 0 or 1" >&2
  exit 2
}
[[ $STRATEGY_ND_DENSE == 0 || $STRATEGY_ND_DENSE == 1 ]] || {
  echo "WS32 StrategyND dense flag must be 0 or 1" >&2
  exit 2
}

readonly DSA_ASSOCIATION_URI=$APPROVED_BUCKET/results/greenfield_ws32_layer0_dsa_association_20260816T101637335765581Z
readonly DSA_ASSOCIATION_SUMMARY_SHA=661142816aa64ec8d085553b427e99f62ab3f1f16b3fc87fc4fc24880d467203
readonly DSA_ASSOCIATION_SUCCESS_SHA=79aba79e24026bc4c1d17aed2ca92055530b8a551ed6e1279a25300d2cb0f52b
if [[ $EXACT_DSA == 0 ]]; then
  DSA_ASSOCIATION_SUMMARY_PIN=0000000000000000000000000000000000000000000000000000000000000000
  DSA_ASSOCIATION_SUCCESS_PIN=0000000000000000000000000000000000000000000000000000000000000000
else
  DSA_ASSOCIATION_SUMMARY_PIN=$DSA_ASSOCIATION_SUMMARY_SHA
  DSA_ASSOCIATION_SUCCESS_PIN=$DSA_ASSOCIATION_SUCCESS_SHA
fi
readonly EXACT_DSA DSA_ASSOCIATION_SUMMARY_PIN DSA_ASSOCIATION_SUCCESS_PIN

# Spec §21.2 first-divergent-event adjudication (default off). When on, the runner and sealer
# bind the committed pre-registered record by path and SHA; only the 8k context has a record.
readonly DSA_ADJUDICATION=${GLM_GREENFIELD_WS32_DSA_ADJUDICATION:-0}
[[ $DSA_ADJUDICATION == 0 || $DSA_ADJUDICATION == 1 ]] || {
  echo "WS32 DSA adjudication flag must be 0 or 1" >&2
  exit 2
}
readonly DSA_ADJUDICATION_RECORD_8K=$WORKTREE/docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json
readonly DSA_ADJUDICATION_RECORD_8K_SHA=4da05468120e3c2e9b82d03931018e0d14eebc5fc28e339381658a04457cd26b
if [[ $DSA_ADJUDICATION == 1 ]]; then
  [[ $CONTEXT == 8k ]] || { echo "WS32 DSA adjudication record exists only for 8k" >&2; exit 2; }
  [[ $(sha256sum "$DSA_ADJUDICATION_RECORD_8K" | awk '{print $1}') == "$DSA_ADJUDICATION_RECORD_8K_SHA" ]] || {
    echo "WS32 DSA adjudication record SHA drifted" >&2; exit 2;
  }
  DSA_ADJUDICATION_CLI=' --dsa-adjudication-record '"$DSA_ADJUDICATION_RECORD_8K"' --dsa-adjudication-sha256 '"$DSA_ADJUDICATION_RECORD_8K_SHA"
else
  DSA_ADJUDICATION_CLI=''
fi
readonly DSA_ADJUDICATION_CLI
# Only after a GATE_D_LESSONS entry for a later-event alarm (spec §21.2); default off.
readonly LATER_EVENT_ALARM_ACK=${GLM_GREENFIELD_WS32_LATER_EVENT_ALARM_ACK:-0}
[[ $LATER_EVENT_ALARM_ACK == 0 || $LATER_EVENT_ALARM_ACK == 1 ]] || {
  echo "WS32 alarm acknowledgement flag must be 0 or 1" >&2
  exit 2
}
# An acknowledgement must be bound to the committed divergence profile (path + SHA) and to the
# pin whose GATE_D_LESSONS entry names the run; the sealer verifies both.
if [[ $LATER_EVENT_ALARM_ACK == 1 ]]; then
  LATER_EVENT_ALARM_PROFILE=$WORKTREE/${GLM_GREENFIELD_WS32_LATER_EVENT_ALARM_PROFILE:?set the committed alarm profile path (repo-relative)}
  LATER_EVENT_ALARM_PROFILE_SHA=${GLM_GREENFIELD_WS32_LATER_EVENT_ALARM_PROFILE_SHA:?set the alarm profile SHA-256}
  [[ -f $LATER_EVENT_ALARM_PROFILE && $(sha256sum "$LATER_EVENT_ALARM_PROFILE" | cut -d' ' -f1) == "$LATER_EVENT_ALARM_PROFILE_SHA" ]] || {
    echo "WS32 alarm profile record drifted" >&2
    exit 2
  }
  LATER_EVENT_ALARM_CLI="--later-event-alarm-profile $LATER_EVENT_ALARM_PROFILE --later-event-alarm-profile-sha256 $LATER_EVENT_ALARM_PROFILE_SHA --later-event-alarm-lessons-pin $(git -C "$WORKTREE" rev-parse HEAD)"
else
  LATER_EVENT_ALARM_CLI=''
fi
readonly LATER_EVENT_ALARM_CLI

if [[ $STRATEGY_ND_DENSE == 1 ]]; then
  STRATEGY_ND_DENSE_OVERLAY_ROOT=${GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_ROOT:?set sealed StrategyND dense overlay root}
  STRATEGY_ND_DENSE_OVERLAY_MANIFEST_SHA=${GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_MANIFEST_SHA:?set overlay manifest SHA}
  STRATEGY_ND_DENSE_OVERLAY_MANIFEST_FILE_SHA=${GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_MANIFEST_FILE_SHA:?set overlay manifest file SHA}
  STRATEGY_ND_DENSE_OVERLAY_SUCCESS_FILE_SHA=${GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_SUCCESS_FILE_SHA:?set overlay SUCCESS file SHA}
  STRATEGY_ND_DENSE_CLI=' --strategy-nd-dense 1 --strategy-nd-dense-overlay-root '"$STRATEGY_ND_DENSE_OVERLAY_ROOT"' --strategy-nd-dense-overlay-manifest-sha256 '"$STRATEGY_ND_DENSE_OVERLAY_MANIFEST_SHA"' --strategy-nd-dense-overlay-manifest-file-sha256 '"$STRATEGY_ND_DENSE_OVERLAY_MANIFEST_FILE_SHA"' --strategy-nd-dense-overlay-success-file-sha256 '"$STRATEGY_ND_DENSE_OVERLAY_SUCCESS_FILE_SHA"
else
  STRATEGY_ND_DENSE_OVERLAY_ROOT=
  STRATEGY_ND_DENSE_OVERLAY_MANIFEST_SHA=$ZERO_SHA
  STRATEGY_ND_DENSE_OVERLAY_MANIFEST_FILE_SHA=$ZERO_SHA
  STRATEGY_ND_DENSE_OVERLAY_SUCCESS_FILE_SHA=$ZERO_SHA
  STRATEGY_ND_DENSE_CLI=' --strategy-nd-dense 0'
fi
readonly STRATEGY_ND_DENSE STRATEGY_ND_DENSE_OVERLAY_ROOT
readonly STRATEGY_ND_DENSE_OVERLAY_MANIFEST_SHA
readonly STRATEGY_ND_DENSE_OVERLAY_MANIFEST_FILE_SHA
readonly STRATEGY_ND_DENSE_OVERLAY_SUCCESS_FILE_SHA STRATEGY_ND_DENSE_CLI

readonly CHECKPOINT_ROOT=${GLM_GREENFIELD_WS32_CHECKPOINT_ROOT:?set sealed checkpoint root}
# Checkpoint transport: gcsfuse (default, sealed object store) or shm (tmpfs copy of the sealed
# WS32 checkpoint built by run_ws32_runtime_checkpoint_shm_pack.sh; manifest/SUCCESS bytes are the
# sealed ones and every host's owned slots are hash-verified by the runner against the manifest).
readonly CHECKPOINT_TRANSPORT=${GLM_GREENFIELD_WS32_CHECKPOINT_TRANSPORT:-gcsfuse}
[[ $CHECKPOINT_TRANSPORT == gcsfuse || $CHECKPOINT_TRANSPORT == shm ]] || {
  echo "WS32 checkpoint transport must be gcsfuse or shm" >&2
  exit 2
}
readonly SHM_CHECKPOINT_ROOT=/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z
readonly SHM_MANIFEST_FILE_SHA=88df414301e0d303506163c078484ec14cdb15ff03972789df58960b091a7cbf
readonly SHM_SUCCESS_FILE_SHA=12703932637a9330f97ae0282b26dd7105d68793b3ae6620e109a10c7b0c7ca2
if [[ $CHECKPOINT_TRANSPORT == shm ]]; then
  [[ $CHECKPOINT_ROOT == "$SHM_CHECKPOINT_ROOT" ]] || { echo "shm transport requires the pinned tmpfs checkpoint root" >&2; exit 2; }
  [[ $(sha256sum "$CHECKPOINT_ROOT/manifest.json" | cut -d' ' -f1) == "$SHM_MANIFEST_FILE_SHA" ]] || { echo "controller tmpfs manifest drifted" >&2; exit 2; }
  [[ $(sha256sum "$CHECKPOINT_ROOT/SUCCESS" | cut -d' ' -f1) == "$SHM_SUCCESS_FILE_SHA" ]] || { echo "controller tmpfs SUCCESS drifted" >&2; exit 2; }
fi
readonly CHECKPOINT_MANIFEST_SHA=${GLM_GREENFIELD_WS32_CHECKPOINT_MANIFEST_SHA:?set checkpoint manifest SHA}
readonly CHECKPOINT_SUCCESS_SHA=${GLM_GREENFIELD_WS32_CHECKPOINT_SUCCESS_SHA:?set checkpoint SUCCESS SHA}
if [[ $CONTEXT == 2k ]]; then
  readonly TOKEN_ORACLE=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/2k/greenfield_short_context_oracle_20260806T202544155912103Z/oracle
  readonly TOKEN_ORACLE_SHA=f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19
  readonly TOKEN_ORACLE_SUCCESS_SHA=07700db5a732f04663f0298625bbdbb68a1c73e63652a3aef27f398993e86eec
  readonly DSA_ORACLE=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/2k/greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z/oracle
  readonly DSA_ORACLE_SHA=71224832652ce61024786d39d43dcbfdc6cde76bf2eff0350531b272f38f4f57
  readonly DSA_ORACLE_SUCCESS_SHA=c091d0b56f712eb2f106ee248f69f599586ade0d5c202cbb5b46b8aa115411b2
else
  readonly TOKEN_ORACLE=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle
  readonly TOKEN_ORACLE_SHA=e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2
  readonly TOKEN_ORACLE_SUCCESS_SHA=38c0aeb6c4833a0256d4e50152b645e85d24a4f00ca7b2b1731db2d892c5b3cc
  readonly DSA_ORACLE=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle
  readonly DSA_ORACLE_SHA=f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da
  readonly DSA_ORACLE_SUCCESS_SHA=0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9
fi
readonly TOKEN_ORACLE_ROOT=${TOKEN_ORACLE%/oracle}
readonly DSA_ORACLE_ROOT=${DSA_ORACLE%/oracle}
# Spec §23.3 Step C: the sealed 8K workload may run at a long-context capacity
# for capacity measurement; the capacity enters the tag and the DB item id.
CONTEXT_CAPACITY=${GLM_GREENFIELD_WS32_CONTEXT_CAPACITY:-8192}
[[ $CONTEXT_CAPACITY =~ ^[0-9]+$ && $CONTEXT_CAPACITY -ge 8192 && $CONTEXT_CAPACITY -le 1048576 && $((CONTEXT_CAPACITY % 512)) -eq 0 ]] || {
  echo "WS32 context capacity must be a multiple of 512 in [8192, 1048576]" >&2
  exit 2
}
readonly CONTEXT_CAPACITY
HOST_MAIN_ROPE_TABLE=${GLM_GREENFIELD_WS32_HOST_MAIN_ROPE_TABLE:-0}
[[ $HOST_MAIN_ROPE_TABLE == 0 || $HOST_MAIN_ROPE_TABLE == 1 ]] || {
  echo "WS32 host main-rotary table flag must be 0 or 1" >&2
  exit 2
}
readonly HOST_MAIN_ROPE_TABLE
ROTARY_DIAGNOSTIC=${GLM_GREENFIELD_WS32_ROTARY_DIAGNOSTIC:-0}
[[ $ROTARY_DIAGNOSTIC == 0 || $ROTARY_DIAGNOSTIC == 1 ]] || {
  echo "WS32 rotary diagnostic flag must be 0 or 1" >&2
  exit 2
}
readonly ROTARY_DIAGNOSTIC
# Spec §23.2: exact prefill runs as fixed chunks plus one tail program.  The
# chunk length is a pinned run input (the equivalence record proves
# C-independence at 2048 and 512); the wall budget bounds the projected
# prefill so a run fails closed long before the worker timeout.
PREFILL_CHUNK=${GLM_GREENFIELD_WS32_PREFILL_CHUNK:-2048}
[[ $PREFILL_CHUNK =~ ^[0-9]+$ && $PREFILL_CHUNK -ge 64 && $PREFILL_CHUNK -le 2048 && $((PREFILL_CHUNK % 64)) -eq 0 ]] || {
  echo "WS32 prefill chunk must be a multiple of 64 in [64, 2048]" >&2
  exit 2
}
readonly PREFILL_CHUNK
readonly PREFILL_BUDGET_SECONDS=3600
# Storage ceiling (goal.md): refuse to launch when a run's evidence could not
# be uploaded in full; at the ceiling the workers' EXIT-trap upload would drop a
# completed run's trace/HLO silently.
readonly STORAGE_CEILING_BYTES=2500000000000
readonly STORAGE_RESERVE_BYTES=6000000000
# A non-default chunk length is part of the run identity (spec §23.3 Step B
# runs C=2048 and C=512): it appears in the tag and is cross-checked by the sealer.
if [[ $PREFILL_CHUNK -eq 2048 ]]; then CHUNK_SUFFIX=; else CHUNK_SUFFIX=_c${PREFILL_CHUNK}; fi
[[ $CONTEXT_CAPACITY -eq 8192 ]] || CHUNK_SUFFIX=${CHUNK_SUFFIX}_cap${CONTEXT_CAPACITY}
[[ $HOST_MAIN_ROPE_TABLE -eq 0 ]] || CHUNK_SUFFIX=${CHUNK_SUFFIX}_hrope
readonly CHUNK_SUFFIX
readonly OBSERVER_STEPS=14
readonly WARMUP=2
readonly ITERATIONS=10
readonly TRACE_STEPS=2
if [[ $MODE == acquire ]]; then
  EXACT_MATERIALIZE_STABLE_SHA=$ZERO_SHA
  EXACT_MATERIALIZE_OPTIMIZED_SHA=$ZERO_SHA
  EXACT_PROMOTE_STABLE_SHA=$ZERO_SHA
  EXACT_PROMOTE_OPTIMIZED_SHA=$ZERO_SHA
  PREFILL_CHUNK_STABLE_SHA=$ZERO_SHA
  PREFILL_CHUNK_OPTIMIZED_SHA=$ZERO_SHA
  PREFILL_TAIL_STABLE_SHA=$ZERO_SHA
  PREFILL_TAIL_OPTIMIZED_SHA=$ZERO_SHA
  OBSERVER_STABLE_SHA=$ZERO_SHA
  OBSERVER_OPTIMIZED_SHA=$ZERO_SHA
  DECODE_STABLE_SHA=$ZERO_SHA
  DECODE_OPTIMIZED_SHA=$ZERO_SHA
  CACHE_PROBE_STABLE_SHA=$ZERO_SHA
  CACHE_PROBE_OPTIMIZED_SHA=$ZERO_SHA
else
  if [[ $EXACT_DSA == 1 ]]; then
    EXACT_MATERIALIZE_STABLE_SHA=${GLM_GREENFIELD_WS32_EXACT_MATERIALIZE_STABLEHLO_SHA:?set acquired exact-materialize StableHLO SHA}
    EXACT_MATERIALIZE_OPTIMIZED_SHA=${GLM_GREENFIELD_WS32_EXACT_MATERIALIZE_OPTIMIZED_HLO_SHA:?set acquired exact-materialize optimized HLO SHA}
    EXACT_PROMOTE_STABLE_SHA=${GLM_GREENFIELD_WS32_EXACT_PROMOTE_STABLEHLO_SHA:?set acquired exact-promote StableHLO SHA}
    EXACT_PROMOTE_OPTIMIZED_SHA=${GLM_GREENFIELD_WS32_EXACT_PROMOTE_OPTIMIZED_HLO_SHA:?set acquired exact-promote optimized HLO SHA}
  else
    EXACT_MATERIALIZE_STABLE_SHA=$ZERO_SHA
    EXACT_MATERIALIZE_OPTIMIZED_SHA=$ZERO_SHA
    EXACT_PROMOTE_STABLE_SHA=$ZERO_SHA
    EXACT_PROMOTE_OPTIMIZED_SHA=$ZERO_SHA
  fi
  PREFILL_CHUNK_STABLE_SHA=${GLM_GREENFIELD_WS32_PREFILL_CHUNK_STABLEHLO_SHA:?set acquired prefill-chunk StableHLO SHA}
  PREFILL_CHUNK_OPTIMIZED_SHA=${GLM_GREENFIELD_WS32_PREFILL_CHUNK_OPTIMIZED_HLO_SHA:?set acquired prefill-chunk optimized HLO SHA}
  PREFILL_TAIL_STABLE_SHA=${GLM_GREENFIELD_WS32_PREFILL_TAIL_STABLEHLO_SHA:?set acquired prefill-tail StableHLO SHA}
  PREFILL_TAIL_OPTIMIZED_SHA=${GLM_GREENFIELD_WS32_PREFILL_TAIL_OPTIMIZED_HLO_SHA:?set acquired prefill-tail optimized HLO SHA}
  OBSERVER_STABLE_SHA=${GLM_GREENFIELD_WS32_OBSERVER_STABLEHLO_SHA:?set acquired observer StableHLO SHA}
  OBSERVER_OPTIMIZED_SHA=${GLM_GREENFIELD_WS32_OBSERVER_OPTIMIZED_HLO_SHA:?set acquired observer optimized HLO SHA}
  DECODE_STABLE_SHA=${GLM_GREENFIELD_WS32_DECODE_STABLEHLO_SHA:?set acquired decode StableHLO SHA}
  DECODE_OPTIMIZED_SHA=${GLM_GREENFIELD_WS32_DECODE_OPTIMIZED_HLO_SHA:?set acquired decode optimized HLO SHA}
  CACHE_PROBE_STABLE_SHA=${GLM_GREENFIELD_WS32_CACHE_PROBE_STABLEHLO_SHA:?set acquired cache-probe StableHLO SHA}
  CACHE_PROBE_OPTIMIZED_SHA=${GLM_GREENFIELD_WS32_CACHE_PROBE_OPTIMIZED_HLO_SHA:?set acquired cache-probe optimized HLO SHA}
fi
readonly PREFILL_CHUNK_STABLE_SHA PREFILL_CHUNK_OPTIMIZED_SHA PREFILL_TAIL_STABLE_SHA PREFILL_TAIL_OPTIMIZED_SHA OBSERVER_STABLE_SHA
readonly EXACT_MATERIALIZE_STABLE_SHA EXACT_MATERIALIZE_OPTIMIZED_SHA
readonly EXACT_PROMOTE_STABLE_SHA EXACT_PROMOTE_OPTIMIZED_SHA
readonly OBSERVER_OPTIMIZED_SHA DECODE_STABLE_SHA DECODE_OPTIMIZED_SHA
readonly CACHE_PROBE_STABLE_SHA CACHE_PROBE_OPTIMIZED_SHA

RECOVERY_PIN=$(git -C "$WORKTREE" rev-parse HEAD)
if [[ $RECOVER == 1 ]]; then
  PIN=${GLM_GREENFIELD_WS32_SOURCE_CODE_HASH:?set the exact code hash used by the completed run}
  TAG=${GLM_GREENFIELD_WS32_SHORT_DECODER_TAG:?set the exact completed run tag}
else
  PIN=$RECOVERY_PIN
  TAG=${GLM_GREENFIELD_WS32_SHORT_DECODER_TAG:-greenfield_ws32_short_decoder_${CONTEXT}_${MODE}${CHUNK_SUFFIX}_$(date -u +%Y%m%dT%H%M%S%NZ)}
fi
[[ $TAG =~ ^greenfield_ws32_short_decoder_${CONTEXT}_${MODE}${CHUNK_SUFFIX}_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "invalid WS32 short-decoder tag: $TAG" >&2
  exit 2
}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN RECOVERY_PIN TAG RUN_DIR REMOTE_PREFIX RECOVER

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
if [[ $RECOVER == 1 ]]; then
  [[ -d $RUN_DIR ]]
  [[ ! -e $RUN_DIR/success_upload.json ]] || {
    echo "a prior terminal SUCCESS verification exists; refusing recovery" >&2
    exit 1
  }
else
  [[ ! -e $RUN_DIR ]]
fi
mkdir -p "$RUN_DIR/fleet" "$RUN_DIR/fleet_hlo" "$RUN_DIR/traces"

say() {
  echo "[ws32-short $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools=1; command -v pgrep >/dev/null || tools=0; command -v fuser >/dev/null || tools=0; sudo -n true >/dev/null 2>&1 || tools=0; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[r]un_short_decoder_ws32[.]py|[c]ompile_short_decoder[.]py|[m]icrobench_collectives[.]py" || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [ "$tools" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected workflow holds the fleet lease"
  exit 1
}
post_census_done=0
db_published=0
archive_upload_started=0
success_upload_started=0
terminal_success_verified=0
success_absent=1
db_rollback_failed=0
rollback_success() {
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX" <<'PY'
import base64,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
from google.api_core.exceptions import NotFound
path=Path(sys.argv[1]); remote=sys.argv[2]; bucket_name,prefix=remote[5:].split('/',1); blob=storage.Client().bucket(bucket_name).blob(prefix.rstrip('/')+'/SUCCESS')
try: blob.reload()
except NotFound: raise SystemExit(0)
raw=path.read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); expected=base64.b64encode(crc.digest()).decode('ascii')
if int(blob.size)!=len(raw) or blob.crc32c!=expected or not blob.generation: raise SystemExit('refusing to remove unowned SUCCESS')
blob.delete(if_generation_match=int(blob.generation))
try: blob.reload()
except NotFound: raise SystemExit(0)
raise SystemExit('remote SUCCESS remains after authenticated delete')
PY
}
rollback_db() {
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$WORKTREE/scripts/greenfield/seal_short_decoder_ws32.py" rollback-db \
    --summary "$RUN_DIR/summary.json" --db-link "$RUN_DIR/db_link.json" \
    --results-db "$RESULTS_DB"
}
rollback_remote_nonterminal() {
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$RUN_DIR" "$REMOTE_PREFIX" "$MODE" <<'PY'
import base64,hashlib,json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
root=Path(sys.argv[1]); remote=sys.argv[2]; mode=sys.argv[3]; bucket_name,prefix=remote[5:].split('/',1)
source=json.loads((root/'source_remote_objects.json').read_text())
canonical=lambda value: json.dumps(value,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
if source.get('ledger_sha256')!=hashlib.sha256(canonical({k:v for k,v in source.items() if k!='ledger_sha256'})).hexdigest() or source.get('remote_prefix')!=remote: raise SystemExit('refusing cleanup with drifted source ledger')
source_names={item['name'] for item in source['objects']}
client=storage.Client(); bucket=client.bucket(bucket_name)
blobs={blob.name.removeprefix(prefix.rstrip('/')+'/'):blob for blob in client.list_blobs(bucket_name,prefix=prefix.rstrip('/')+'/')}
for item in source['objects']:
 blob=blobs.get(item['name'])
 if blob is None or int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']: raise SystemExit(f"refusing cleanup with drifted source object: {item['name']}")
extras=set(blobs)-source_names
if 'SUCCESS' in extras: raise SystemExit('refusing nonterminal cleanup while SUCCESS exists')
for name in sorted(extras):
 if name=='remote_objects.json': path=root/name
 elif name.startswith('orchestrator/') and '/' not in name.removeprefix('orchestrator/'):
  path=root/Path(name).name
 elif mode=='acquire' and name.startswith('diagnostic/') and '/' not in name.removeprefix('diagnostic/'):
  path=root/Path(name).name
 else: raise SystemExit(f'refusing to delete unowned remote object: {name}')
 raw=path.read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); expected=base64.b64encode(crc.digest()).decode('ascii'); blob=blobs[name]
 if int(blob.size)!=len(raw) or blob.crc32c!=expected or not blob.generation: raise SystemExit(f'refusing to delete nonexact remote object: {name}')
 blob.delete(if_generation_match=int(blob.generation))
remaining={blob.name.removeprefix(prefix.rstrip('/')+'/'):blob for blob in client.list_blobs(bucket_name,prefix=prefix.rstrip('/')+'/')}
if set(remaining)!=source_names: raise SystemExit('remote source set was not restored after cleanup')
for item in source['objects']:
 blob=remaining[item['name']]
 if int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']: raise SystemExit(f"source object drifted during cleanup: {item['name']}")
PY
}
rollback_recovery_seed() {
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$RECOVERY_TOOL" rollback \
    --seed "$RUN_DIR/recovery_seed_objects.json" \
    --remote-prefix "$REMOTE_PREFIX"
}
archive_failed_publication() {
  # $1 == with_ledger: only at recovery start, after rollback_remote_nonterminal
  # has proven the remote set equals the stale ledger.  The ledger binds the
  # recovery pin, so a ledger written by a failed attempt is preserved here and
  # regenerated against the same generation-pinned objects.  on_exit never
  # moves it: a later attempt needs it as rollback authority.
  local with_ledger=${1:-}
  local destination
  destination="$RUN_DIR/recovery_failures/$(date -u +%Y%m%dT%H%M%S%NZ)"
  mkdir -p "$destination"
  for name in summary.json validate.log census_post.txt census_recovery_pre.txt \
    materialize.log results.db db_link.json db_publish.log remote_objects.json \
    SUCCESS success_upload.json recovery_seed_objects.json \
    recovery_publish.log; do
    [[ ! -e $RUN_DIR/$name ]] || mv "$RUN_DIR/$name" "$destination/$name"
  done
  if [[ $with_ledger == with_ledger && -e $RUN_DIR/source_remote_objects.json ]]; then
    mv "$RUN_DIR/source_remote_objects.json" "$destination/source_remote_objects.json"
  fi
}
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then (strict_census failure_exit) || true; fi
  if [[ $status -ne 0 && $success_upload_started -eq 1 && $terminal_success_verified -eq 0 ]]; then
    if rollback_success; then
      success_absent=1
    else
      say "ABORT: remote SUCCESS could not be proven absent; retaining DB linkage"
    fi
  fi
  if [[ $status -ne 0 && $archive_upload_started -eq 1 && $terminal_success_verified -eq 0 && $success_absent -eq 1 ]]; then
    rollback_remote_nonterminal || say "ABORT: remote nonterminal objects could not be restored to the source set"
  fi
  if [[ $status -ne 0 && ${RECOVER:-0} == 1 && ! -e $RUN_DIR/source_remote_objects.json && -e $RUN_DIR/recovery_seed_objects.json ]]; then
    rollback_recovery_seed || say "ABORT: recovery seed objects could not be removed"
  fi
  if [[ $status -ne 0 && $db_published -eq 1 && $terminal_success_verified -eq 0 && $success_absent -eq 1 ]]; then
    if rollback_db; then archive_failed_publication; else db_rollback_failed=1; fi
  fi
  # A DB row that could not be rolled back keeps summary.json/db_link.json in
  # place so the next recovery start can retry rollback_db with the link intact.
  if [[ $status -ne 0 && ${RECOVER:-0} == 1 && $terminal_success_verified -eq 0 && $db_rollback_failed -eq 0 ]]; then
    archive_failed_publication
  fi
  if [[ $status -ne 0 && $terminal_success_verified -eq 0 ]]; then
    say "FAILED status=$status; nonterminal diagnostics retained"
    if [[ ${RECOVER:-0} == 0 && ! -e $RUN_DIR/source_remote_objects.json ]]; then
      for path in orchestrator.log remote_vacancy.txt sync.txt launch.txt \
        census_pre.txt census_failure_exit.txt materialize.log validate.log; do
        [[ ! -f $RUN_DIR/$path ]] || gcloud storage cp --no-clobber \
          "$RUN_DIR/$path" "$REMOTE_PREFIX/diagnostic_local/$TAG/$path" \
          >/dev/null 2>&1 || true
      done
    fi
  fi
}
trap on_exit EXIT

if [[ $RECOVER == 1 ]]; then
  if [[ -e $RUN_DIR/SUCCESS ]]; then
    rollback_success
  fi
  if [[ -e $RUN_DIR/source_remote_objects.json ]]; then
    rollback_remote_nonterminal
  elif [[ -e $RUN_DIR/recovery_seed_objects.json ]]; then
    rollback_recovery_seed
  fi
  if [[ -e $RUN_DIR/db_link.json || -e $RUN_DIR/results.db ]]; then
    [[ -e $RUN_DIR/summary.json && -e $RUN_DIR/db_link.json ]]
    rollback_db || { db_rollback_failed=1; exit 1; }
  fi
  archive_failed_publication with_ledger
fi

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RESULTS_DB" "$TAG" <<'PY'
import json,sqlite3,sys
database,tag=sys.argv[1:3]
connection=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
try:
 tables={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 if not {'runs','items','summary'} <= tables: raise SystemExit('protected results DB schema is absent')
 matches=0
 for (raw,) in connection.execute('SELECT env_json FROM runs'):
  try: environment=json.loads(raw)
  except (TypeError,json.JSONDecodeError): continue
  matches += environment.get('run_tag') == tag
 if matches: raise SystemExit('protected results DB already contains this run tag')
finally: connection.close()
PY

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$CHECKPOINT_ROOT" "$CHECKPOINT_MANIFEST_SHA" "$CHECKPOINT_SUCCESS_SHA" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected_manifest,expected_success=sys.argv[2:4]
if any(len(value)!=64 or any(char not in '0123456789abcdef' for char in value) for value in (expected_manifest,expected_success)):
 raise SystemExit('checkpoint identity pins must be lowercase SHA-256 values')
manifest_path=root/'manifest.json'; success_path=root/'SUCCESS'
if not manifest_path.is_file() or not success_path.is_file():
 raise SystemExit('sealed checkpoint manifest/SUCCESS is absent')
manifest=json.loads(manifest_path.read_text()); success=json.loads(success_path.read_text())
canonical=lambda value: json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(',',':'),sort_keys=True).encode()
if manifest.get('manifest_sha256')!=expected_manifest or success.get('manifest_sha256')!=expected_manifest:
 raise SystemExit('checkpoint manifest identity pin drifted')
if success.get('success_sha256')!=expected_success:
 raise SystemExit('checkpoint SUCCESS identity pin drifted')
without_success={key:value for key,value in success.items() if key!='success_sha256'}
if hashlib.sha256(canonical(without_success)).hexdigest()!=expected_success:
 raise SystemExit('checkpoint SUCCESS self-hash drifted')
if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=success.get('manifest_file_sha256'):
 raise SystemExit('checkpoint manifest file hash drifted')
PY

if [[ $EXACT_DSA == 1 ]]; then
  if [[ $RECOVER == 0 ]]; then
    gcloud storage cp --no-clobber "$DSA_ASSOCIATION_URI/summary.json" \
      "$RUN_DIR/exact_dsa_source_summary.json" >/dev/null
    gcloud storage cp --no-clobber "$DSA_ASSOCIATION_URI/SUCCESS" \
      "$RUN_DIR/exact_dsa_source_SUCCESS" >/dev/null
  fi
  printf '%s  %s\n%s  %s\n' \
    "$DSA_ASSOCIATION_SUMMARY_SHA" "$RUN_DIR/exact_dsa_source_summary.json" \
    "$DSA_ASSOCIATION_SUCCESS_SHA" "$RUN_DIR/exact_dsa_source_SUCCESS" \
    | sha256sum -c - >/dev/null
fi
say "PIN=$PIN recovery_pin=$RECOVERY_PIN mode=$MODE context=$CONTEXT recover=$RECOVER exact_dsa=$EXACT_DSA transport=$CHECKPOINT_TRANSPORT"
live_bytes=$(gcloud storage du -s "$APPROVED_BUCKET" 2>/dev/null | awk 'END {print $1}')
[[ $live_bytes =~ ^[0-9]+$ ]] || { say "ABORT: live storage census unavailable"; exit 1; }
say "live storage $live_bytes bytes (ceiling $STORAGE_CEILING_BYTES, reserve $STORAGE_RESERVE_BYTES)"
[[ $((live_bytes + STORAGE_RESERVE_BYTES)) -le $STORAGE_CEILING_BYTES ]] || {
  say "ABORT: live storage plus reserve exceeds the ceiling; reclaim before launching"
  exit 1
}
listing="$RUN_DIR/remote_vacancy.txt"
if [[ $RECOVER == 0 ]]; then
  gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' >"$listing" || {
    say "ABORT: remote vacancy listing failed"
    exit 1
  }
  [[ ! -s $listing ]] || {
    say "ABORT: remote prefix is not vacant"
    exit 1
  }
else
  [[ -f $listing && ! -s $listing ]]
fi
if [[ $RECOVER == 0 ]]; then
  strict_census pre || {
    say "ABORT: pre-run fleet census is not clean"
    exit 1
  }
else
  strict_census recovery_pre || {
    say "ABORT: recovery pre-census is not clean"
    exit 1
  }
fi
if [[ $RECOVER == 1 && $MODE == acquire && ! -e $RUN_DIR/source_remote_objects.json ]]; then
  say "recovering immutable worker prevalidation; no TPU executable will run"
  mkdir -p "$RUN_DIR/recovery_prevalidation"
  for rank in {0..7}; do
    destination="$RUN_DIR/recovery_prevalidation/prevalidation.rank${rank}.json"
    if [[ ! -e $destination ]]; then
      partial="${destination}.partial"
      [[ ! -e $partial ]]
      gcloud compute tpus tpu-vm scp \
        "$POD:/home/gianl/glm-run/$TAG/hlo/prevalidation.json" "$partial" \
        --zone "$ZONE" --worker="$rank" --quiet >/dev/null
      mv "$partial" "$destination"
    fi
  done
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
    "$RECOVERY_TOOL" publish --run-dir "$RUN_DIR" \
    --remote-prefix "$REMOTE_PREFIX" --source-code-hash "$PIN" \
    --recovery-code-hash "$RECOVERY_PIN" \
    >"$RUN_DIR/recovery_publish.log" 2>&1
fi
if [[ $RECOVER == 0 && $MODE == numerical ]]; then
  available_bytes=$(df --output=avail -B1 "$RUN_DIR" | tail -1 | tr -d ' ')
  [[ $available_bytes =~ ^[0-9]+$ && $available_bytes -ge 4294967296 ]] || {
    say "ABORT: less than 4 GiB is available for unique fleet evidence and sealing"
    exit 1
  }
fi

if [[ $RECOVER == 0 ]]; then
say "synchronizing exact code and sealed inputs on all hosts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; wt='"$WORKTREE"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; checkpoint='"$CHECKPOINT_ROOT"'; transport='"$CHECKPOINT_TRANSPORT"'; shm_root='"$SHM_CHECKPOINT_ROOT"'; shm_manifest_sha='"$SHM_MANIFEST_FILE_SHA"'; shm_success_sha='"$SHM_SUCCESS_FILE_SHA"'; overlay_enabled='"$STRATEGY_ND_DENSE"'; overlay='"$STRATEGY_ND_DENSE_OVERLAY_ROOT"'; inventory='"$INVENTORY"'; token='"$TOKEN_ORACLE"'; token_root='"$TOKEN_ORACLE_ROOT"'; token_success_sha='"$TOKEN_ORACLE_SUCCESS_SHA"'; dsa='"$DSA_ORACLE"'; dsa_root='"$DSA_ORACLE_ROOT"'; dsa_success_sha='"$DSA_ORACLE_SUCCESS_SHA"'; topology='"$TOPOLOGY_ROOT"'/topology.rank${idx}.json; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" && $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; elif [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; for path in "$checkpoint/manifest.json" "$checkpoint/SUCCESS" "$inventory" "$token/manifest.json" "$token_root/SUCCESS" "$dsa/manifest.json" "$dsa_root/SUCCESS" "$topology"; do [[ -r $path ]]; done; if [[ $overlay_enabled == 1 ]]; then [[ -r $overlay/manifest.json && -r $overlay/SUCCESS ]]; findmnt -T "$overlay" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse"; fi; token_observed=$(sha256sum "$token_root/SUCCESS"); dsa_observed=$(sha256sum "$dsa_root/SUCCESS"); [[ ${token_observed%% *} == "$token_success_sha" ]]; [[ ${dsa_observed%% *} == "$dsa_success_sha" ]]; if [[ $transport == shm ]]; then findmnt -T "$checkpoint" -n -o FSTYPE | grep -qx tmpfs; [[ $checkpoint == "$shm_root" ]]; [[ $(sha256sum "$checkpoint/manifest.json" | cut -d" " -f1) == "$shm_manifest_sha" && $(sha256sum "$checkpoint/SUCCESS" | cut -d" " -f1) == "$shm_success_sha" ]]; [[ $(ls "$checkpoint"/device_slot_*.safetensors | wc -l) -eq 4 ]]; marker="$checkpoint.identity.json"; [[ -f $marker ]]; /home/gianl/vllm-env/bin/python - "$marker" "$checkpoint" <<'"'"'PY'"'"'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text()); root=Path(sys.argv[2]); manifest=json.loads((root/"manifest.json").read_text())
if m.get("manifest_sha256")!=manifest["manifest_sha256"] or m.get("tmpfs_root")!=str(root) or len(m.get("slots",[]))!=4 or m.get("all_slots_identical") is not True: raise SystemExit("tmpfs identity marker drifted")
for s in m["slots"]:
    p=root/s["filename"]
    if not p.is_file() or p.stat().st_size!=s["file_bytes"]: raise SystemExit(f"tmpfs slot missing: {p}")
PY
else findmnt -T "$checkpoint" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse"; fi; echo "SYNC_OK $(hostname) $pin"'
sync_rc=0
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1 || sync_rc=$?
if [[ $sync_rc -ne 0 ]] || ! has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK; then
  say "ABORT: exact eight-host synchronization failed"
  exit 1
fi

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I | awk '{print \$1}'" 2>/dev/null | tail -1 | tr -d '\r')
[[ -n $coordinator ]]
coordinator="$coordinator:8476"
say "launching complete WS32 worker fleet coordinator=$coordinator"
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag; output="$run/runner.rank${idx}.json"; tensors="$run/runner.rank${idx}.npz"; hlo="$run/hlo"; trace="$run/trace"; log="$run/runner.rank${idx}.log"; mkdir -p "$run"; upload(){ local rc=0; [[ ! -f $output ]] || gcloud storage cp --no-clobber "$output" "$remote/host_records/runner.rank${idx}.json" >/dev/null 2>&1 || rc=1; [[ ! -f $tensors ]] || gcloud storage cp --no-clobber "$tensors" "$remote/host_records/runner.rank${idx}.npz" >/dev/null 2>&1 || rc=1; [[ ! -f $log ]] || gcloud storage cp --no-clobber "$log" "$remote/host_records/runner.rank${idx}.log" >/dev/null 2>&1 || rc=1; upload_shared(){ local src=$1 dst=$2; if gcloud storage cp --no-clobber "$src" "$dst" >/dev/null 2>&1; then return 0; fi; local want have; want=$(gcloud storage hash --skip-md5 --hex "$src" 2>/dev/null | awk "/crc32c/ {print \$NF}"); have=$(gcloud storage objects describe "$dst" --format="value(crc32c_hash)" 2>/dev/null | base64 -d 2>/dev/null | od -An -tx1 | tr -d " \n"); [[ -n $want && -n $have && $want == "$have" ]]; }; for graph in exact_materialize exact_promote prefill_chunk prefill_tail observer decode cache_probe; do for form in stablehlo.mlir optimized_hlo.txt; do [[ -f "$hlo/$graph.$form" ]] || continue; if [[ ! -f "$hlo/$graph.$form.gz" ]]; then gzip -n -9 -c "$hlo/$graph.$form" > "$hlo/$graph.$form.gz.partial" && mv -f "$hlo/$graph.$form.gz.partial" "$hlo/$graph.$form.gz"; fi; upload_shared "$hlo/$graph.$form.gz" "$remote/hlo/${graph}.${form}.gz" || rc=1; done; done; xplane=$(find "$trace" -type f -name "*.xplane.pb" 2>/dev/null | head -1 || true); [[ -z $xplane ]] || gcloud storage cp --no-clobber "$xplane" "$remote/traces/trace.rank${idx}.xplane.pb" >/dev/null 2>&1 || rc=1; return "$rc"; }; trap "upload || true" EXIT; cd "$wt"; env JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" GLM_GREENFIELD_RUN_TAG="$tag" timeout --signal=TERM --kill-after=60 14400 /home/gianl/vllm-env/bin/python -u scripts/greenfield/run_short_decoder_ws32.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --topology-capture-root '"$TOPOLOGY_ROOT"' --checkpoint-root '"$CHECKPOINT_ROOT"' --source-inventory '"$INVENTORY"' --token-oracle-dir '"$TOKEN_ORACLE"' --dsa-oracle-dir '"$DSA_ORACLE"' --expected-code-hash '"$PIN"' --checkpoint-manifest-sha256 '"$CHECKPOINT_MANIFEST_SHA"' --checkpoint-success-sha256 '"$CHECKPOINT_SUCCESS_SHA"' --token-oracle-manifest-sha256 '"$TOKEN_ORACLE_SHA"' --dsa-oracle-manifest-sha256 '"$DSA_ORACLE_SHA"' --token-oracle-success-sha256 '"$TOKEN_ORACLE_SUCCESS_SHA"' --dsa-oracle-success-sha256 '"$DSA_ORACLE_SUCCESS_SHA"' --dsa-association-summary-sha256 '"$DSA_ASSOCIATION_SUMMARY_PIN"' --dsa-association-success-sha256 '"$DSA_ASSOCIATION_SUCCESS_PIN"' --topology-sha256 '"$TOPOLOGY_SHA"' --topology-fleet-sha256 '"$TOPOLOGY_FLEET_SHA"' --mesh-sha256 '"$MESH_SHA"' --expected-exact-materialize-stablehlo-sha256 '"$EXACT_MATERIALIZE_STABLE_SHA"' --expected-exact-materialize-optimized-hlo-sha256 '"$EXACT_MATERIALIZE_OPTIMIZED_SHA"' --expected-exact-promote-stablehlo-sha256 '"$EXACT_PROMOTE_STABLE_SHA"' --expected-exact-promote-optimized-hlo-sha256 '"$EXACT_PROMOTE_OPTIMIZED_SHA"' --expected-prefill-chunk-stablehlo-sha256 '"$PREFILL_CHUNK_STABLE_SHA"' --expected-prefill-chunk-optimized-hlo-sha256 '"$PREFILL_CHUNK_OPTIMIZED_SHA"' --expected-prefill-tail-stablehlo-sha256 '"$PREFILL_TAIL_STABLE_SHA"' --expected-prefill-tail-optimized-hlo-sha256 '"$PREFILL_TAIL_OPTIMIZED_SHA"' --prefill-chunk '"$PREFILL_CHUNK"' --prefill-budget-seconds '"$PREFILL_BUDGET_SECONDS"' --rotary-diagnostic '"$ROTARY_DIAGNOSTIC"' --host-main-rope-table '"$HOST_MAIN_ROPE_TABLE"' --expected-observer-stablehlo-sha256 '"$OBSERVER_STABLE_SHA"' --expected-observer-optimized-hlo-sha256 '"$OBSERVER_OPTIMIZED_SHA"' --expected-decode-stablehlo-sha256 '"$DECODE_STABLE_SHA"' --expected-decode-optimized-hlo-sha256 '"$DECODE_OPTIMIZED_SHA"' --expected-cache-probe-stablehlo-sha256 '"$CACHE_PROBE_STABLE_SHA"' --expected-cache-probe-optimized-hlo-sha256 '"$CACHE_PROBE_OPTIMIZED_SHA"' --context-capacity '"$CONTEXT_CAPACITY"' --compile-only '"$([[ $MODE == acquire ]] && echo 1 || echo 0)"' --exact-dsa '"$EXACT_DSA"' --checkpoint-transport '"$CHECKPOINT_TRANSPORT"''"$STRATEGY_ND_DENSE_CLI"''"$DSA_ADJUDICATION_CLI"' --observer-steps '"$OBSERVER_STEPS"' --warmup '"$WARMUP"' --iterations '"$ITERATIONS"' --trace-steps '"$TRACE_STEPS"' --output "$output" --tensor-output "$tensors" --hlo-dir "$hlo" --trace-dir "$trace" >"$log" 2>&1; trap - EXIT; upload; echo "WS32_SHORT_OK $(hostname) rank=$idx"'
launch_rc=0
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/launch.txt" 2>&1 || launch_rc=$?
if [[ $launch_rc -ne 0 ]] || ! has_eight_unique_markers "$RUN_DIR/launch.txt" WS32_SHORT_OK; then
  say "ABORT: complete WS32 worker fleet did not finish 8/8"
  exit 1
fi
fi

say "materializing generation-pinned all-host evidence with content deduplication"
materialize_args=()
[[ $RECOVER == 0 ]] || materialize_args+=(--allow-failure-diagnostics)
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  -m glm_tpu.greenfield.validation.ws32_evidence \
  --run-dir "$RUN_DIR" --remote-prefix "$REMOTE_PREFIX" --mode "$MODE" \
  --tag "$TAG" --code-hash "$PIN" --recovery-code-hash "$RECOVERY_PIN" \
  --exact-dsa "$EXACT_DSA" \
  --output "$RUN_DIR/source_remote_objects.json" "${materialize_args[@]}" \
  >"$RUN_DIR/materialize.log" 2>&1
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/seal_short_decoder_ws32.py" validate \
  --run-dir "$RUN_DIR" --topology-capture-root "$TOPOLOGY_ROOT" \
  --token-oracle-dir "$TOKEN_ORACLE" --dsa-oracle-dir "$DSA_ORACLE" \
  --mode "$MODE" --context-label "$CONTEXT" --tag "$TAG" --code-hash "$PIN" \
  --checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA" \
  --checkpoint-success-sha256 "$CHECKPOINT_SUCCESS_SHA" \
  --token-oracle-manifest-sha256 "$TOKEN_ORACLE_SHA" \
  --dsa-oracle-manifest-sha256 "$DSA_ORACLE_SHA" \
  --token-oracle-success-sha256 "$TOKEN_ORACLE_SUCCESS_SHA" \
  --dsa-oracle-success-sha256 "$DSA_ORACLE_SUCCESS_SHA" \
  --dsa-association-summary-sha256 "$DSA_ASSOCIATION_SUMMARY_PIN" \
  --dsa-association-success-sha256 "$DSA_ASSOCIATION_SUCCESS_PIN" \
  --topology-sha256 "$TOPOLOGY_SHA" --topology-fleet-sha256 "$TOPOLOGY_FLEET_SHA" \
  --mesh-sha256 "$MESH_SHA" --source-inventory-sha256 "$INVENTORY_SHA" \
  --context-capacity "$CONTEXT_CAPACITY" --observer-steps "$OBSERVER_STEPS" \
  --exact-dsa "$EXACT_DSA" \
  ${DSA_ADJUDICATION_CLI:+$DSA_ADJUDICATION_CLI} \
  --later-event-alarm-acknowledged "$LATER_EVENT_ALARM_ACK" \
  --recovery-code-hash "$RECOVERY_PIN" \
  ${LATER_EVENT_ALARM_CLI:+$LATER_EVENT_ALARM_CLI} \
  --checkpoint-transport "$CHECKPOINT_TRANSPORT" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --rotary-diagnostic "$ROTARY_DIAGNOSTIC" \
  --host-main-rope-table "$HOST_MAIN_ROPE_TABLE" \
  --strategy-nd-dense "$STRATEGY_ND_DENSE" \
  --strategy-nd-dense-overlay-manifest-sha256 "$STRATEGY_ND_DENSE_OVERLAY_MANIFEST_SHA" \
  --strategy-nd-dense-overlay-manifest-file-sha256 "$STRATEGY_ND_DENSE_OVERLAY_MANIFEST_FILE_SHA" \
  --strategy-nd-dense-overlay-success-file-sha256 "$STRATEGY_ND_DENSE_OVERLAY_SUCCESS_FILE_SHA" \
  --warmup "$WARMUP" --iterations "$ITERATIONS" --trace-steps "$TRACE_STEPS" \
  --expected-exact-materialize-stablehlo-sha256 "$EXACT_MATERIALIZE_STABLE_SHA" \
  --expected-exact-materialize-optimized-hlo-sha256 "$EXACT_MATERIALIZE_OPTIMIZED_SHA" \
  --expected-exact-promote-stablehlo-sha256 "$EXACT_PROMOTE_STABLE_SHA" \
  --expected-exact-promote-optimized-hlo-sha256 "$EXACT_PROMOTE_OPTIMIZED_SHA" \
  --expected-prefill-chunk-stablehlo-sha256 "$PREFILL_CHUNK_STABLE_SHA" \
  --expected-prefill-chunk-optimized-hlo-sha256 "$PREFILL_CHUNK_OPTIMIZED_SHA" \
  --expected-prefill-tail-stablehlo-sha256 "$PREFILL_TAIL_STABLE_SHA" \
  --expected-prefill-tail-optimized-hlo-sha256 "$PREFILL_TAIL_OPTIMIZED_SHA" \
  --expected-observer-stablehlo-sha256 "$OBSERVER_STABLE_SHA" \
  --expected-observer-optimized-hlo-sha256 "$OBSERVER_OPTIMIZED_SHA" \
  --expected-decode-stablehlo-sha256 "$DECODE_STABLE_SHA" \
  --expected-decode-optimized-hlo-sha256 "$DECODE_OPTIMIZED_SHA" \
  --expected-cache-probe-stablehlo-sha256 "$CACHE_PROBE_STABLE_SHA" \
  --expected-cache-probe-optimized-hlo-sha256 "$CACHE_PROBE_OPTIMIZED_SHA" \
  --output "$RUN_DIR/summary.json" >"$RUN_DIR/validate.log" 2>&1

strict_census post || {
  say "ABORT: post-run fleet census is not clean"
  exit 1
}
post_census_done=1

if [[ $MODE == acquire ]]; then
  archive_upload_started=1
  acquisition_files=(orchestrator.log summary.json validate.log census_post.txt
    source_remote_objects.json materialize.log)
  [[ $EXACT_DSA == 0 ]] || acquisition_files+=(exact_dsa_source_summary.json
    exact_dsa_source_SUCCESS)
  [[ $RECOVER == 0 ]] || acquisition_files+=(census_recovery_pre.txt)
  [[ ! -e $RUN_DIR/recovery_seed_objects.json ]] || \
    acquisition_files+=(recovery_seed_objects.json recovery_publish.log)
  for name in "${acquisition_files[@]}"; do
    gcloud storage cp --no-clobber "$RUN_DIR/$name" \
      "$REMOTE_PREFIX/diagnostic/$name" >/dev/null
  done
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$RUN_DIR" "$REMOTE_PREFIX" "${acquisition_files[@]}" <<'PY'
import base64,json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
root=Path(sys.argv[1]); remote=sys.argv[2]; outputs=sys.argv[3:]
bucket_name,prefix=remote[5:].split('/',1); client=storage.Client()
blobs={blob.name.removeprefix(prefix.rstrip('/')+'/'):blob for blob in client.list_blobs(bucket_name,prefix=prefix.rstrip('/')+'/')}
source=json.loads((root/'source_remote_objects.json').read_text())
source_by_name={item['name']:item for item in source['objects']}
expected=set(source_by_name)|{f'diagnostic/{name}' for name in outputs}
if set(blobs)!=expected: raise SystemExit('acquisition remote object set drifted')
for name,item in source_by_name.items():
 blob=blobs[name]
 if int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']: raise SystemExit(f'acquisition source object drifted: {name}')
for name in outputs:
 raw=(root/name).read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); expected_crc=base64.b64encode(crc.digest()).decode('ascii'); blob=blobs[f'diagnostic/{name}']
 if int(blob.size)!=len(raw) or blob.crc32c!=expected_crc or not blob.generation: raise SystemExit(f'acquisition diagnostic bytes drifted: {name}')
PY
  say "HLO acquisition complete; pins are in summary.json and no DB/SUCCESS was created"
  trap - EXIT
  exit 0
fi

say "publishing one atomic DB linkage after correctness, trace, HBM and cleanup pass"
db_published=1
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/seal_short_decoder_ws32.py" publish-db \
  --summary "$RUN_DIR/summary.json" --results-db "$RESULTS_DB" \
  --snapshot "$RUN_DIR/results.db" --output "$RUN_DIR/db_link.json" \
  >"$RUN_DIR/db_publish.log"

say "archiving orchestrator evidence and validating the exact remote object set"
archive_upload_started=1
orchestrator_files=(orchestrator.log remote_vacancy.txt sync.txt launch.txt census_pre.txt
  census_post.txt summary.json validate.log results.db db_link.json db_publish.log
  source_remote_objects.json materialize.log)
[[ $EXACT_DSA == 0 ]] || orchestrator_files+=(exact_dsa_source_summary.json
  exact_dsa_source_SUCCESS)
[[ $RECOVER == 0 ]] || orchestrator_files+=(census_recovery_pre.txt)
for name in "${orchestrator_files[@]}"; do
  gcloud storage cp --no-clobber "$RUN_DIR/$name" "$REMOTE_PREFIX/orchestrator/$name" >/dev/null
done
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$REMOTE_PREFIX" "$RUN_DIR/remote_objects.json" <<'PY'
import base64,json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
root=Path(sys.argv[1]); uri=sys.argv[2]; output=Path(sys.argv[3]); bucket_name,prefix=uri[5:].split('/',1); client=storage.Client(); blobs={blob.name.removeprefix(prefix.rstrip('/')+'/'):blob for blob in client.list_blobs(bucket_name,prefix=prefix.rstrip('/')+'/')}; source=json.loads((root/'source_remote_objects.json').read_text())
expected={}
source_by_name={item['name']:item for item in source['objects']}
for name in source_by_name:
 if name.startswith('host_records/'): expected[name]=root/'fleet'/Path(name).name
 elif name.startswith('hlo/'): expected[name]=root/'fleet_hlo'/Path(name).name
 elif name.startswith('traces/'): expected[name]=root/'traces'/Path(name).name
orchestrator=('orchestrator.log','remote_vacancy.txt','sync.txt','launch.txt','census_pre.txt','census_post.txt','summary.json','validate.log','results.db','db_link.json','db_publish.log','source_remote_objects.json','materialize.log')
for name in orchestrator: expected[f'orchestrator/{name}']=root/name
for name in ('exact_dsa_source_summary.json','exact_dsa_source_SUCCESS'):
 if (root/name).exists(): expected[f'orchestrator/{name}']=root/name
recovery_pre=root/'census_recovery_pre.txt'
if recovery_pre.exists(): expected['orchestrator/census_recovery_pre.txt']=recovery_pre
diagnostics={name:item for name,item in source_by_name.items() if name.startswith('diagnostic_local/')}
if set(blobs)!=set(expected)|set(diagnostics): raise SystemExit(f'remote nonterminal object set drifted: missing={sorted((set(expected)|set(diagnostics))-set(blobs))} extra={sorted(set(blobs)-(set(expected)|set(diagnostics)))}')
for name,item in source_by_name.items():
 blob=blobs[name]
 if int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']: raise SystemExit(f'source remote object drifted: {name}')
records=[]
for name,path in sorted(expected.items()):
 raw=path.read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); crc32c=base64.b64encode(crc.digest()).decode('ascii'); blob=blobs[name]
 if int(blob.size)!=len(raw) or blob.crc32c!=crc32c or not blob.generation: raise SystemExit(f'remote bytes drifted: {name}')
 records.append({'crc32c':crc32c,'generation':int(blob.generation),'name':name,'sha256':__import__('hashlib').sha256(raw).hexdigest(),'size':len(raw)})
for name,item in sorted(diagnostics.items()): records.append(item)
value={'artifact_kind':'greenfield_ws32_short_decoder_remote_ledger','objects':records,'remote_prefix':uri}; value['ledger_sha256']=__import__('hashlib').sha256(json.dumps(value,allow_nan=False,separators=(',',':'),sort_keys=True).encode()).hexdigest(); output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/summary.json" "$RUN_DIR/db_link.json" "$RUN_DIR/remote_objects.json" \
  "$RUN_DIR/source_remote_objects.json" \
  "$RUN_DIR/census_post.txt" "$RUN_DIR/SUCCESS" "$TAG" "$PIN" \
  "$RECOVERY_PIN" "$REMOTE_PREFIX" <<'PY'
from hashlib import sha256
import base64,json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
summary_path,db_path,ledger_path,source_path,census_path,output=map(Path,sys.argv[1:7]); tag,pin,recovery_pin,remote=sys.argv[7:11]
summary=json.loads(summary_path.read_text()); db=json.loads(db_path.read_text()); ledger=json.loads(ledger_path.read_text()); source=json.loads(source_path.read_text())
def canonical(value): return json.dumps(value,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
if summary['summary_sha256']!=sha256(canonical({k:v for k,v in summary.items() if k!='summary_sha256'})).hexdigest() or summary['status']!='SUCCESS' or summary['performance_claim'] is not True: raise SystemExit('terminal summary identity drifted')
if db['record_sha256']!=sha256(canonical({k:v for k,v in db.items() if k!='record_sha256'})).hexdigest() or db['summary_sha256']!=summary['summary_sha256']: raise SystemExit('terminal DB link drifted')
if ledger['ledger_sha256']!=sha256(canonical({k:v for k,v in ledger.items() if k!='ledger_sha256'})).hexdigest() or ledger['remote_prefix']!=remote: raise SystemExit('terminal remote ledger drifted')
if source['ledger_sha256']!=sha256(canonical({k:v for k,v in source.items() if k!='ledger_sha256'})).hexdigest() or source['code_hash']!=pin or source['recovery_code_hash']!=recovery_pin: raise SystemExit('terminal source ledger drifted')
markers=[line.split()[1] for line in census_path.read_text().splitlines() if line.startswith('CENSUS_OK ')]
if len(markers)!=8 or len(set(markers))!=8: raise SystemExit('terminal post-census drifted')
bucket_name,prefix=remote[5:].split('/',1); client=storage.Client(); blobs={blob.name.removeprefix(prefix.rstrip('/')+'/'):blob for blob in client.list_blobs(bucket_name,prefix=prefix.rstrip('/')+'/')}; expected={item['name'] for item in ledger['objects']}|{'remote_objects.json'}
if set(blobs)!=expected: raise SystemExit('terminal pre-SUCCESS object set drifted')
for item in ledger['objects']:
 blob=blobs[item['name']]
 if int(blob.size)!=item['size'] or blob.crc32c!=item['crc32c'] or int(blob.generation)!=item['generation']: raise SystemExit(f"terminal remote object drifted: {item['name']}")
raw=ledger_path.read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); crc32c=base64.b64encode(crc.digest()).decode('ascii'); blob=blobs['remote_objects.json']
if int(blob.size)!=len(raw) or blob.crc32c!=crc32c or not blob.generation: raise SystemExit('terminal remote ledger bytes drifted')
value={'artifact_kind':'greenfield_ws32_short_decoder_SUCCESS','code_hash':pin,'context_label':summary['context_label'],'db_record_sha256':db['record_sha256'],'performance_claim':True,'recovery_code_hash':recovery_pin,'remote_ledger_generation':int(blob.generation),'remote_ledger_sha256':ledger['ledger_sha256'],'results_db_run_id':db['results_db_run_id'],'run_tag':tag,'source_ledger_sha256':source['ledger_sha256'],'summary_sha256':summary['summary_sha256'],'topology_sha256':summary['topology_sha256'],'xplane_files':summary['xplane']['n_files'],'xplane_cores':summary['xplane']['n_cores']}; value['success_sha256']=sha256(canonical(value)).hexdigest(); output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
success_upload_started=1
success_absent=0
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX" "$RUN_DIR/success_upload.json" <<'PY'
import base64,json,sys
from pathlib import Path
import google_crc32c
from google.cloud import storage
path=Path(sys.argv[1]); remote=sys.argv[2]; output=Path(sys.argv[3]); bucket_name,prefix=remote[5:].split('/',1); client=storage.Client(); blobs={blob.name.removeprefix(prefix.rstrip('/')+'/'):blob for blob in client.list_blobs(bucket_name,prefix=prefix.rstrip('/')+'/')}; ledger=json.loads((path.parent/'remote_objects.json').read_text()); expected={item['name'] for item in ledger['objects']}|{'remote_objects.json','SUCCESS'}
if set(blobs)!=expected: raise SystemExit('terminal SUCCESS object set drifted')
raw=path.read_bytes(); crc=google_crc32c.Checksum(); crc.update(raw); crc32c=base64.b64encode(crc.digest()).decode('ascii'); blob=blobs['SUCCESS']
if int(blob.size)!=len(raw) or blob.crc32c!=crc32c or not blob.generation: raise SystemExit('remote SUCCESS bytes drifted')
output.write_text(json.dumps({'crc32c':crc32c,'generation':int(blob.generation),'remote':remote+'/SUCCESS','size':len(raw)},indent=2,sort_keys=True)+'\n')
PY
terminal_success_verified=1
trap - EXIT
say "SUCCESS tag=$TAG; protected WS32 $CONTEXT Gate-D record sealed"
