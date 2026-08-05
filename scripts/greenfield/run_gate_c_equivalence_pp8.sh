#!/usr/bin/env bash
# Protected PP8 Gate C dense/DSA/IndexShare correctness, HLO, HBM, and XPlane proof.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly ORACLE_RUN=/home/gianl/glm-run/greenfield_gate_c_oracle_20260805T212801776974822Z
readonly PACK_RUN=/home/gianl/glm-run/greenfield_gate_c_pack_20260805T214609093206269Z
readonly TOPOLOGY_CAPTURE=/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z/topology.rank0.json
readonly PACKED_CODE_HASH=8a50d6a1960581de47b028f3c1203f8806c3809d
readonly PACKED_MANIFEST_SHA=3c5c48dadb1b42ec6c99667b79196477ed978cb555dc57dec9e4af38583f2a8a
readonly LAYOUT_MANIFEST_SHA=cdbea04f226f2bd93f3d5ae006ea14b5119e091e29a36da52d783e9d2284c678
readonly PARENT_LAYOUT_MANIFEST_SHA=aca0eb6d4c498271baf3260dacbe55c9bf935539f42a4b7eaa889947627506a0
readonly ORACLE_MANIFEST_SHA=54262529bd561c57f0833a993e0ac6cdbec20b9e270a0f6f1726d0049d9c4a9f
readonly SOURCE_REVISION=gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly PP8_GROUP_HASH=d5943ab8d7a074677d82f8e823c8bc983847f8df1deefdee1fbda8da98923c14
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly RUN_WORKER=2
readonly TOPOLOGY_RUN=/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z
readonly PACK_REMOTE=gs://driftbench-dsv4-uc/checkpoints/greenfield/glm52/gate_c/PP8_LP4/greenfield_gate_c_pack_20260805T214609093206269Z
readonly ORACLE_REMOTE=gs://driftbench-dsv4-uc/oracles/greenfield/glm52/gate_c/greenfield_gate_c_oracle_20260805T212801776974822Z

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TAG=${GLM_GREENFIELD_GATE_C_TAG:-greenfield_gate_c_pp8_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing Gate C run outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing Gate C run from a dirty greenfield worktree" >&2
  exit 2
}
[[ $ORACLE_PIN == "$LEGACY_PIN" ]] || {
  echo "legacy oracle checkout drifted: $ORACLE_PIN" >&2
  exit 2
}
[[ -f $PACK_RUN/SUCCESS && -f $ORACLE_RUN/SUCCESS ]] || {
  echo "successful Gate C pack/oracle prerequisites are unavailable" >&2
  exit 2
}
[[ -r $TOPOLOGY_CAPTURE && -r $RESULTS_DB ]] || {
  echo "topology capture or provenance DB is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only Gate C run directory already exists: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR"

say() {
  echo "[gate-c-pp8 $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[t]race_pipeline_transport[.]py|[i]nspect_topology[.]py|[r]un_real_one_layer[.]py|[r]un_gate_c_equivalence[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN ORACLE_PIN=$ORACLE_PIN"
say "PACK=$PACKED_MANIFEST_SHA ORACLE=$ORACLE_MANIFEST_SHA"
say "SOURCE_REVISION=$SOURCE_REVISION trace_steps=20"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact pin and immutable Gate C inputs to physical stage-0 worker $RUN_WORKER"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; tag='"$TAG"'; pack='"$PACK_REMOTE"'; oracle='"$ORACLE_REMOTE"'; topology='"$TOPOLOGY_RUN"'; idx=${HOSTNAME##*-w-}; [[ "$idx" == '"$RUN_WORKER"' ]]; if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin '"$BRANCH"'; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch '"$BRANCH"' "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; run=/home/gianl/glm-run/$tag; mkdir -p "$run/input/packed/base_decoder/stage_00" "$run/input/oracle"; gcloud storage cp "$pack/packed/manifest.json" "$pack/packed/layout_manifest.json" "$run/input/packed/" >/dev/null; for slot in 00 01 02 03; do gcloud storage cp "$pack/packed/base_decoder/stage_00/device_slot_${slot}.safetensors" "$run/input/packed/base_decoder/stage_00/" >/dev/null; done; gcloud storage cp "$oracle/manifest.json" "$oracle/oracle.safetensors" "$run/input/oracle/" >/dev/null; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$topology/topology.rank${idx}.json" ]] && [[ -r "$run/input/packed/manifest.json" ]] && [[ -r "$run/input/oracle/manifest.json" ]] && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" \
  --worker="$RUN_WORKER" --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
grep -q '^SYNC_OK ' "$RUN_DIR/sync.txt" || {
  say "ABORT: stage-0 worker pin/input sync failed"
  exit 1
}

say "running direct load and real dense/full-DSA/IndexShare TPU equivalence"
started=$(date +%s)
# shellcheck disable=SC2016
runner_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; [[ "$idx" == '"$RUN_WORKER"' ]]; tag='"$TAG"'; wt='"$WORKTREE"'; topology='"$TOPOLOGY_RUN"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag; log="$run/runner.log"; trap '\''gcloud storage cp --no-clobber "$log" "$run/runner.json" "$remote/diagnostic/" >/dev/null 2>&1 || true'\'' EXIT; cd "$wt"; env JAX_PLATFORMS=tpu TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python -u scripts/greenfield/run_gate_c_equivalence.py --artifact-dir "$run/input/packed" --oracle-dir "$run/input/oracle" --topology-capture "$topology/topology.rank${idx}.json" --expected-code-hash '"$PIN"' --packed-code-hash '"$PACKED_CODE_HASH"' --packed-manifest-sha256 '"$PACKED_MANIFEST_SHA"' --layout-manifest-sha256 '"$LAYOUT_MANIFEST_SHA"' --parent-layout-manifest-sha256 '"$PARENT_LAYOUT_MANIFEST_SHA"' --oracle-manifest-sha256 '"$ORACLE_MANIFEST_SHA"' --source-revision '"$SOURCE_REVISION"' --topology-sha256 '"$TOPOLOGY_HASH"' --plan-group-sha256 '"$PP8_GROUP_HASH"' --output "$run/runner.json" --hlo-dir "$run/hlo" --trace-root "$run/trace" --trace-steps 20 >"$log" 2>&1; gcloud storage cp --recursive --no-clobber "$run/hlo" "$run/trace" "$remote/" >/dev/null; gcloud storage cp --no-clobber "$run/runner.json" "$run/runner.log" "$remote/" >/dev/null; trap - EXIT; echo "GATE_C_UPLOAD_OK $(hostname)"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" \
  --worker="$RUN_WORKER" --command="$runner_command" >"$RUN_DIR/execute.txt" 2>&1
grep -q '^GATE_C_UPLOAD_OK ' "$RUN_DIR/execute.txt" || {
  say "ABORT: stage-0 Gate C execution/upload failed"
  exit 1
}
gcloud storage cp "$REMOTE_PREFIX/runner.json" "$REMOTE_PREFIX/runner.log" \
  "$RUN_DIR/" >/dev/null
gcloud storage cp --recursive "$REMOTE_PREFIX/hlo" "$REMOTE_PREFIX/trace" \
  "$RUN_DIR/" >/dev/null
elapsed=$(( $(date +%s) - started ))
say "runner completed in ${elapsed}s"

say "validating physical trace, linking DB, and building summary"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$ORACLE_PIN" "$RESULTS_DB" "$WORKTREE" \
  "$ORACLE_REPO" "$elapsed" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, oracle_pin, db_path, repo, oracle_repo, elapsed = sys.argv[1:]
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
if runner.get("status") != "SUCCESS" or runner.get("code_hash") != pin:
    raise SystemExit("Gate C runner status/code identity failed")
if runner.get("evidence_class") != "protected_single_host_tpu_v4_gate_c":
    raise SystemExit("Gate C runner is not protected TPU evidence")
if runner.get("performance_claim") is not False:
    raise SystemExit("Gate C runner incorrectly made a performance claim")
if not all(case["passed"] for case in runner["cases"].values()):
    raise SystemExit("Gate C correctness cases did not all pass")
if not all(value["contract"]["passed"] for value in runner["hlo"].values()):
    raise SystemExit("Gate C HLO contract failed")
load = runner["load"]
if (
    load["host_fp8_dequantizations"] != 0
    or load["host_global_concatenations"] != 0
    or load["runtime_checkpoint_reshards"] != 0
    or load["loaded_payload_bytes"] != 502446080
):
    raise SystemExit("Gate C direct-load contract failed")
state = dict(runner["packed_checkpoint"]["state_manifest"])
state_hash = state.pop("manifest_sha256")
computed_state_hash = sha256(
    json.dumps(
        state,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
).hexdigest()
if state_hash != computed_state_hash or state_hash != load["state_manifest_sha256"]:
    raise SystemExit("Gate C loaded-state manifest hash failed")
index_state = runner["index_share_state"]
if (
    index_state["byte_count"] != 8192
    or index_state["dtype"] != "int32"
    or index_state["shape"] != [1, 2048]
    or not index_state["fed_producer_device_result_directly"]
    or not index_state["score_order_preserved"]
):
    raise SystemExit("Gate C compact IndexShare state contract failed")

sys.path.insert(0, str(Path(repo) / "scripts" / "analysis"))
import parse_xplane

module_patterns = {
    "dense": r"jit_dense_step",
    "dsa": r"jit_dsa_step",
    "index_share": r"jit_index_share_step",
}
trace_by_case = {}
for case, pattern in module_patterns.items():
    parsed = parse_xplane.aggregate_fleet(run_dir / "trace", step_module_re=pattern)
    if (
        parsed["n_files"] != 1
        or parsed["n_cores"] != 8
        or len(parsed["hosts"]) != 1
        or parsed["steps_per_core"] != 20
    ):
        raise SystemExit(
            f"Gate C {case} XPlane inventory failed: "
            f"files={parsed['n_files']} cores={parsed['n_cores']} "
            f"hosts={parsed['hosts']} steps={parsed['steps_per_core']}"
        )
    expected_ar = runner["hlo"][case]["contract"]["all_reduce_count"]
    expected_ag = runner["hlo"][case]["contract"]["all_gather_count"]
    observed_ar = parsed["hlo_all_reduce_invocations_per_step"]
    observed_ag = parsed["hlo_all_gather_invocations_per_step"]
    if len(observed_ar) != 160 or set(observed_ar) != {expected_ar}:
        raise SystemExit(
            f"Gate C {case} physical all-reduce count drifted: "
            f"expected={expected_ar} observed={sorted(set(observed_ar))}"
        )
    if len(observed_ag) != 160 or set(observed_ag) != {expected_ag}:
        raise SystemExit(
            f"Gate C {case} physical all-gather count drifted: "
            f"expected={expected_ag} observed={sorted(set(observed_ag))}"
        )
    forbidden = []
    for name, values in parsed["ops"].items():
        if values.get("category") != "collectives":
            continue
        category = (values.get("hlo_category") or "").lower()
        identity = f"{name} {category}".lower()
        if not any(token in identity for token in ("all-reduce", "all-gather")):
            forbidden.append(identity)
    if forbidden:
        raise SystemExit(f"Gate C {case} trace has forbidden collectives: {forbidden}")
    trace_by_case[case] = parsed

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-gate-c-pp8",
    revision=runner["source_revision"],
    env={
        "GLM_ENGINE": "greenfield_gate_c",
        "GLM_EXECUTION_PLAN": "PP8_LP4",
        "greenfield_code_hash": pin,
        "legacy_oracle_code_hash": oracle_pin,
        "layout_manifest_sha256": runner["packed_checkpoint"]["layout_manifest_sha256"],
        "packed_manifest_sha256": runner["packed_checkpoint"]["manifest_sha256"],
        "oracle_manifest_sha256": runner["oracle"]["manifest_sha256"],
        "topology_sha256": runner["topology_sha256"],
    },
    note=(
        "Protected PP8 Gate C dense/full-DSA/IndexShare exact device-score "
        "selection, bounded raw-source oracle, HLO/HBM/XPlane proof."
    ),
    harness_repo=repo,
    fork_repo=oracle_repo,
)
for case in ("dense", "dsa", "index_share"):
    pv.record_item(
        conn,
        run_id,
        benchmark="greenfield_gate_c_pp8",
        item_id=case,
        prompt=f"Execute the real GLM-5.2 Gate C {case} case on one PP8 stage.",
        gold=(
            "Bounded independent-oracle tensors, exact device-score DSA/IndexShare "
            "state, topology-local HLO and physical collectives."
        ),
        raw_output=json.dumps(
            {
                "case": runner["cases"][case],
                "hlo": runner["hlo"][case]["contract"],
            },
            sort_keys=True,
        ),
        extracted="PASS",
        correct=True,
        score=1.0,
    )
pv.finalize(
    conn,
    run_id,
    benchmark="greenfield_gate_c_pp8",
    metric="contract_valid",
    value=1.0,
    note="One-layer correctness/layout evidence only; no token-throughput claim.",
)
conn.close()
summary = {
    "artifact_kind": "greenfield_gate_c_pp8_proof",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "oracle_code_hash": oracle_pin,
    "performance_claim": False,
    "results_db_run_id": run_id,
    "runner": runner,
    "status": "SUCCESS",
    "trace_by_case": trace_by_case,
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
check = sqlite3.connect(run_dir / "results_ckpt.db").execute(
    "PRAGMA integrity_check"
).fetchone()[0]
if check != "ok":
    raise SystemExit(f"results DB snapshot integrity failed: {check}")
print(f"GATE_C_PP8_PROOF_VALID db_run={run_id}")
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "validated; sealing append-only evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo trace -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log summary.json results_ckpt.db \
    census_pre.txt census_post.txt sync.txt execute.txt orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
(cd "$RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)

touch "$RUN_DIR/SUCCESS"
gcloud storage cp --recursive --no-clobber "$RUN_DIR/hlo" "$RUN_DIR/trace" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber \
  "$RUN_DIR/runner.json" "$RUN_DIR/runner.log" "$RUN_DIR/summary.json" \
  "$RUN_DIR/results_ckpt.db" "$RUN_DIR/evidence.sha256" \
  "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log" \
  "$RUN_DIR/census_pre.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/sync.txt" "$RUN_DIR/execute.txt" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ $remote_success == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
db_id=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' \
  "$RUN_DIR/summary.json")
say "SUCCESS DB=$db_id"
say "ARCHIVE=$REMOTE_PREFIX"
trap - EXIT
