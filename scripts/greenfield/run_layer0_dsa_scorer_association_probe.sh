#!/usr/bin/env bash
# Protected bounded A/B for the remaining layer-0 DSA scorer association.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly ASSOCIATION_INPUT=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly ASSOCIATION_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly PROMPT_CACHE=/home/gianl/glm-run/greenfield_legacy_layer0_prompt_projection_input_p113_20260809T050055956585082Z/prompt_index_cache
readonly PROMPT_CACHE_MANIFEST_SHA=acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab
readonly SOURCE_RUN=/home/gianl/glm-run/greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_oracle_dsa_dsa_internal_trace2_20260810T145002403468792Z
readonly INTERNAL_CONTRACT_SHA=9bdab5023b5775b787e15c3c76d542eab921bd7a704602b4502b2305fad03d4c
readonly INTERNAL_TENSOR_SHA=c2fdeccfdcc81363fe01a566c34bf6c04f2f44b0a18e7b545e76fdf0f0d4560b
readonly SELECTED_OBSERVATION_SHA=4a1924ff2edb73ba10747ec1c28448884c4a224d4d179f0bcaf5e61a4f2745a5

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_DSA_SCORER_ASSOCIATION_TAG:-greenfield_layer0_dsa_scorer_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing scorer association probe outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing scorer association probe from a dirty worktree" >&2
  exit 2
}
[[ -r $RESULTS_DB && -d $ASSOCIATION_INPUT && -d $PROMPT_CACHE &&
  -d $SOURCE_RUN && ! -e $RUN_DIR ]] || {
  echo "results DB/inputs missing or append-only run path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo" "$RUN_DIR/inputs/association" \
  "$RUN_DIR/inputs/prompt_cache" "$RUN_DIR/inputs/internal" \
  "$RUN_DIR/inputs/selected"
cp "$ASSOCIATION_INPUT/manifest.json" \
  "$ASSOCIATION_INPUT/layer0_dsa_input.safetensors" \
  "$RUN_DIR/inputs/association/"
cp "$PROMPT_CACHE/manifest.json" \
  "$PROMPT_CACHE/prompt_index_cache.safetensors" \
  "$RUN_DIR/inputs/prompt_cache/"
cp "$SOURCE_RUN/dsa_internal_observer/contract.json" \
  "$SOURCE_RUN/dsa_internal_observer/position_8155_internals.npz" \
  "$RUN_DIR/inputs/internal/"
cp "$SOURCE_RUN/dsa_observer/step_00_position_8155.npz" \
  "$RUN_DIR/inputs/selected/"

say() {
  echo "[dsa-scorer-association $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
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
  local ray_enum
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[p]robe_layer0_dsa_scorer_association[.]py|[p]robe_layer0_dsa_association[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN wide=2048 highest-vs-default"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact reviewed pin to all eight hosts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; idx=${HOSTNAME##*-w-}; if [[ "$idx" == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host code sync failed"
  exit 1
}

say "running bounded same-shape highest-versus-default TPU discriminator"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    /home/gianl/vllm-env/bin/python \
      scripts/greenfield/probe_layer0_dsa_scorer_association.py \
      --expected-code-hash "$PIN" \
      --association-input-dir "$RUN_DIR/inputs/association" \
      --association-input-manifest-sha256 "$ASSOCIATION_MANIFEST_SHA" \
      --prompt-cache-dir "$RUN_DIR/inputs/prompt_cache" \
      --prompt-cache-manifest-sha256 "$PROMPT_CACHE_MANIFEST_SHA" \
      --internal-observer-dir "$RUN_DIR/inputs/internal" \
      --internal-contract-sha256 "$INTERNAL_CONTRACT_SHA" \
      --internal-tensor-sha256 "$INTERNAL_TENSOR_SHA" \
      --selected-observation \
        "$RUN_DIR/inputs/selected/step_00_position_8155.npz" \
      --selected-observation-sha256 "$SELECTED_OBSERVATION_SHA" \
      --output "$RUN_DIR/runner.json" \
      --hlo-dir "$RUN_DIR/hlo"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "runner completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, db_path, repo, elapsed = sys.argv[1:]
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
if runner["status"] != "SUCCESS" or runner["code_hash"] != pin:
    raise SystemExit("scorer association runner status/code identity failed")
if runner["profiler_free_timing"] is not False:
    raise SystemExit("scorer association diagnostic claimed performance")
for name in ("current_wide_score", "default_wide_score"):
    if not runner["hlo"][name]["contract"]["passed"]:
        raise SystemExit(f"scorer association HLO failed: {name}")
wide = runner["current_wide_control"]
if not wide["comparison"]["passed"] or not wide["bitwise_exact"]:
    raise SystemExit("current-wide protected control did not reproduce")
candidate = bool(runner["candidate_restored"])

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-layer0-dsa-scorer-association",
    revision="bounded-wide-highest-vs-default-v1",
    env={
        "GLM_ENGINE": "greenfield_layer0_dsa_scorer_association",
        "greenfield_code_hash": pin,
        "association_manifest_sha256": runner["inputs"][
            "association_manifest_sha256"
        ],
        "prompt_cache_manifest_sha256": runner["inputs"][
            "prompt_cache_manifest_sha256"
        ],
        "internal_tensor_sha256": runner["inputs"][
            "internal_tensor_sha256"
        ],
        "current_wide_hlo_sha256": runner["hlo"][
            "current_wide_score"
        ]["sha256"],
        "default_wide_hlo_sha256": runner["hlo"][
            "default_wide_score"
        ]["sha256"],
        "device_kind": runner["device_kind"],
    },
    note="Protected exact-input layer-0 highest-vs-default scorer diagnostic.",
    harness_repo=repo,
    fork_repo=None,
)
pv.record_item(
    conn,
    run_id,
    benchmark="greenfield_layer0_dsa_scorer_association",
    item_id="layer0_position8155_wide_highest_vs_default",
    prompt="Sealed exact query/head/current-key and DB518 prompt key cache.",
    gold="Exact accepted 2,048 positions, scores, and lowest-position tie order.",
    raw_output=json.dumps(runner, sort_keys=True),
    extracted=json.dumps(
        {
            "wide_control_exact": True,
            "default_precision_restored": candidate,
        },
        sort_keys=True,
    ),
    correct=candidate,
    score=float(candidate),
    latency_ms=None,
)
pv.finalize(
    conn,
    run_id,
    benchmark="greenfield_layer0_dsa_scorer_association",
    metric="default_precision_restored",
    value=float(candidate),
    note="Diagnostic only; no decoder, Gate-D, latency, or token-rate claim.",
)
conn.close()

summary = {
    "status": "SUCCESS",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
    "candidate_restored": candidate,
    "runner": runner,
    "claim_scope": runner["claim_scope"],
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
if sqlite3.connect(run_dir / "results_ckpt.db").execute(
    "PRAGMA integrity_check"
).fetchone()[0] != "ok":
    raise SystemExit("results DB snapshot integrity failed")
print(
    f"DSA_SCORER_ASSOCIATION_VALID db_run={run_id} "
    f"default_precision_restored={candidate}"
)
PY

(
  cd "$RUN_DIR"
  find hlo inputs -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log summary.json results_ckpt.db \
    census_pre.txt sync.txt
) >"$RUN_DIR/evidence.sha256"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1
sha256sum "$RUN_DIR/census_post.txt" >>"$RUN_DIR/evidence.sha256"
touch "$RUN_DIR/SUCCESS"

gcloud storage cp --recursive --no-clobber "$RUN_DIR/hlo" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --recursive --no-clobber "$RUN_DIR/inputs" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber \
  "$RUN_DIR/runner.json" "$RUN_DIR/runner.log" "$RUN_DIR/summary.json" \
  "$RUN_DIR/results_ckpt.db" "$RUN_DIR/evidence.sha256" \
  "$RUN_DIR/orchestrator.log" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/sync.txt" "$RUN_DIR/census_post.txt" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ $remote_success == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
say "SUCCESS DB=$(/home/gianl/vllm-env/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' "$RUN_DIR/summary.json")"
say "CANDIDATE_RESTORED=$(/home/gianl/vllm-env/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_restored"])' "$RUN_DIR/summary.json")"
say "ARCHIVE=$REMOTE_PREFIX"
trap - EXIT
