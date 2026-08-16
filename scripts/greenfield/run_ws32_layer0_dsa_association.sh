#!/usr/bin/env bash
# Protected seconds-scale WS32 layer-0 DSA ownership/arithmetic discriminator.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_ROOT=/home/gianl/glm-run/greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z
readonly ASSOCIATION_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly PROMPT_CACHE_MANIFEST_SHA=acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab
readonly INTERNAL_CONTRACT_SHA=9bdab5023b5775b787e15c3c76d542eab921bd7a704602b4502b2305fad03d4c
readonly INTERNAL_TENSOR_SHA=c2fdeccfdcc81363fe01a566c34bf6c04f2f44b0a18e7b545e76fdf0f0d4560b
readonly SOURCE_SUMMARY_SHA=01de9dff6595180c968a7b9a1fac7e8fe1b4773ea79a2b71fb7e0d516348d7c0
readonly SOURCE_DB_RUN_ID=529
readonly SEALER=$WORKTREE/scripts/greenfield/seal_ws32_dsa_association.py

[[ ${GLM_GREENFIELD_WS32_DSA_ASSOCIATION:-0} == 1 ]] || {
  echo "WS32 DSA association workflow is default-off" >&2
  exit 2
}
[[ ${GLM_GREENFIELD_WS32_DSA_ASSOCIATION_MODE:-off} == bounded ]] || {
  echo "set GLM_GREENFIELD_WS32_DSA_ASSOCIATION_MODE=bounded" >&2
  exit 2
}

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_WS32_DSA_ASSOCIATION_TAG:-greenfield_ws32_layer0_dsa_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
git -C "$WORKTREE" fetch -q origin "$BRANCH"
[[ $(git -C "$WORKTREE" rev-parse "origin/$BRANCH") == "$PIN" ]] || {
  echo "reviewed code must be pushed before protected execution" >&2
  exit 2
}
[[ -r $RESULTS_DB && -d $LEGACY_REPO && ! -e $RUN_DIR ]]
mkdir -p "$RUN_DIR/hlo" "$RUN_DIR/inputs/association" \
  "$RUN_DIR/inputs/prompt_cache" "$RUN_DIR/inputs/internal" \
  "$RUN_DIR/inputs/source"

say() {
  echo "[ws32-dsa-association $(date -u +%H:%M:%S)] $*" |
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
  # shellcheck disable=SC2016
  command='tools=1; command -v pgrep >/dev/null 2>&1 || tools=0; command -v fuser >/dev/null 2>&1 || tools=0; sudo -n true >/dev/null 2>&1 || tools=0; generic=$(pgrep -af "[p]robe_ws32_layer0_dsa_association[.]py|[r]un_short_decoder_ws32[.]py|[c]ompile_short_decoder[.]py|[V]LLM::EngineCore|[R]ayWorkerWrapper" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [[ $tools -ne 1 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; elif [[ -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [[ -z $generic ]] || echo "$generic"; [[ -z $holders ]] || echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

common_sealer_args=(
  --run-dir "$RUN_DIR"
  --code-hash "$PIN"
  --tag "$TAG"
  --remote-prefix "$REMOTE_PREFIX"
  --association-manifest-sha256 "$ASSOCIATION_MANIFEST_SHA"
  --prompt-cache-manifest-sha256 "$PROMPT_CACHE_MANIFEST_SHA"
  --internal-contract-sha256 "$INTERNAL_CONTRACT_SHA"
  --internal-tensor-sha256 "$INTERNAL_TENSOR_SHA"
  --source-summary-sha256 "$SOURCE_SUMMARY_SHA"
)

post_census_done=0
terminal_success_verified=0
remote_vacant=0
on_exit() {
  local status=$? rollback_ok=1
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    if [[ -f $RUN_DIR/SUCCESS ]]; then
      if ! PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python "$SEALER" \
        rollback-success "${common_sealer_args[@]}"; then
        rollback_ok=0
      fi
    fi
    if [[ -f $RUN_DIR/validation.json ]]; then
      if ! PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python "$SEALER" \
        rollback-db "${common_sealer_args[@]}" \
        --results-db "$RESULTS_DB" --worktree "$WORKTREE" \
        --legacy-repo "$LEGACY_REPO"; then
        rollback_ok=0
      fi
    fi
    if [[ $terminal_success_verified -eq 0 && $rollback_ok -eq 1 && $remote_vacant -eq 1 ]]; then
      say "FAILED status=$status; diagnostic evidence retained without terminal SUCCESS"
      gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
        "$REMOTE_PREFIX/diagnostic/$TAG/" >/dev/null 2>&1 || true
    elif [[ $rollback_ok -eq 0 ]]; then
      echo "ABORT: authenticated terminal rollback failed; remote evidence was not mutated further" >&2
    fi
  fi
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN bounded_position=8155 context=8156"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python "$SEALER" \
  check-vacant "${common_sealer_args[@]}" >"$RUN_DIR/remote_vacancy.txt"
remote_vacant=1

say "authenticating and copying the sealed DB529 source bundle"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$SOURCE_ROOT/summary.json" "$SOURCE_SUMMARY_SHA" "$RESULTS_DB" \
  "$SOURCE_DB_RUN_ID" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

summary_path=Path(sys.argv[1]); expected=sys.argv[2]; db=Path(sys.argv[3]); run_id=int(sys.argv[4])
if sha256(summary_path.read_bytes()).hexdigest()!=expected:
    raise SystemExit('DB529 source summary hash drifted')
summary=json.loads(summary_path.read_text())
if summary.get('status')!='SUCCESS' or summary.get('results_db_run_id')!=run_id or summary.get('candidate_restored') is not True:
    raise SystemExit('DB529 source summary verdict drifted')
connection=sqlite3.connect(db)
run=connection.execute('SELECT model,model_revision,env_json,pod FROM runs WHERE run_id=?',(run_id,)).fetchone()
item=connection.execute('SELECT benchmark,correct,score,raw_output FROM items WHERE run_id=?',(run_id,)).fetchall()
terminal=connection.execute('SELECT benchmark,n,metric,value,note FROM summary WHERE run_id=?',(run_id,)).fetchall()
connection.close()
if run is None or run[0:2] != ('zai-org/GLM-5.2-FP8:greenfield-layer0-dsa-scorer-association','bounded-wide-highest-vs-default-v1') or run[3]!='db-v4-64-od':
    raise SystemExit('DB529 run provenance drifted')
env=json.loads(run[2]); runner=summary['runner']
if env.get('association_manifest_sha256')!='574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141' or env.get('prompt_cache_manifest_sha256')!='acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab' or env.get('internal_tensor_sha256')!='c2fdeccfdcc81363fe01a566c34bf6c04f2f44b0a18e7b545e76fdf0f0d4560b':
    raise SystemExit('DB529 input provenance drifted')
if len(item)!=1 or item[0][0:3] != ('greenfield_layer0_dsa_scorer_association',1,1.0) or json.loads(item[0][3])!=runner:
    raise SystemExit('DB529 item provenance drifted')
if terminal != [('greenfield_layer0_dsa_scorer_association',1,'default_precision_restored',1.0,'Diagnostic only; no decoder, Gate-D, latency, or token-rate claim.')]:
    raise SystemExit('DB529 summary provenance drifted')
PY
cp "$SOURCE_ROOT/inputs/association/manifest.json" \
  "$SOURCE_ROOT/inputs/association/layer0_dsa_input.safetensors" \
  "$RUN_DIR/inputs/association/"
cp "$SOURCE_ROOT/inputs/prompt_cache/manifest.json" \
  "$SOURCE_ROOT/inputs/prompt_cache/prompt_index_cache.safetensors" \
  "$RUN_DIR/inputs/prompt_cache/"
cp "$SOURCE_ROOT/inputs/internal/contract.json" \
  "$SOURCE_ROOT/inputs/internal/position_8155_internals.npz" \
  "$RUN_DIR/inputs/internal/"
cp "$SOURCE_ROOT/summary.json" "$RUN_DIR/inputs/source/summary.json"

strict_census pre || {
  say "ABORT: pre-run fleet census is not clean"
  exit 1
}

say "synchronizing the exact reviewed pin on all eight hosts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; wt='"$WORKTREE"'; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" && $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host code synchronization failed"
  exit 1
}

say "running one-host four-chip tuple4/tuple8 discriminator"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    timeout --signal=TERM --kill-after=30 600 \
    /home/gianl/vllm-env/bin/python -u \
      scripts/greenfield/probe_ws32_layer0_dsa_association.py \
      --expected-code-hash "$PIN" \
      --association-input-dir "$RUN_DIR/inputs/association" \
      --association-input-manifest-sha256 "$ASSOCIATION_MANIFEST_SHA" \
      --prompt-cache-dir "$RUN_DIR/inputs/prompt_cache" \
      --prompt-cache-manifest-sha256 "$PROMPT_CACHE_MANIFEST_SHA" \
      --internal-observer-dir "$RUN_DIR/inputs/internal" \
      --internal-contract-sha256 "$INTERNAL_CONTRACT_SHA" \
      --internal-tensor-sha256 "$INTERNAL_TENSOR_SHA" \
      --source-summary "$RUN_DIR/inputs/source/summary.json" \
      --source-summary-sha256 "$SOURCE_SUMMARY_SHA" \
      --output "$RUN_DIR/runner.json" --hlo-dir "$RUN_DIR/hlo"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))

say "independently recomputing tensors, classifications, and all HLO contracts"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python "$SEALER" \
  validate "${common_sealer_args[@]}" --output "$RUN_DIR/validation.json"

strict_census post || {
  say "ABORT: post-run fleet census is not clean"
  exit 1
}
post_census_done=1
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"

say "publishing one atomic diagnostic DB row after the clean post-census"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python "$SEALER" \
  publish-db "${common_sealer_args[@]}" --results-db "$RESULTS_DB" \
  --worktree "$WORKTREE" --legacy-repo "$LEGACY_REPO" \
  --elapsed-seconds "$elapsed"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python "$SEALER" \
  publish-archive "${common_sealer_args[@]}"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python "$SEALER" \
  publish-success "${common_sealer_args[@]}"
terminal_success_verified=1
trap - EXIT

restored=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["association_restored"]).lower())' \
  "$RUN_DIR/validation.json")
arms=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(",".join(json.load(open(sys.argv[1]))["exact_arms"]) or "none")' \
  "$RUN_DIR/validation.json")
echo "SUCCESS tag=$TAG association_restored=$restored exact_arms=$arms diagnostic_only=true"
