#!/usr/bin/env bash
# Protected Gate-A fresh fleet XPlane proof for PP8/PP16 transport.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || exit 2
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing trace from a dirty greenfield worktree" >&2
  exit 2
}
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TAG=${GLM_GREENFIELD_TRANSPORT_TRACE_TAG:-greenfield_transport_trace_$(date -u +%Y%m%dT%H%M%S%NZ)}
WARMUP=${GLM_GREENFIELD_TRANSPORT_TRACE_WARMUP:-50}
TRACE_STEPS=${GLM_GREENFIELD_TRANSPORT_TRACE_STEPS:-20}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 20 && $TRACE_STEPS == 20 ]] || {
  echo "protected transport trace requires warmup>=20 and exactly 20 steps" >&2
  exit 2
}
mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/traces"

say() {
  echo "[transport-trace $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[t]race_pipeline_transport[.]py|[i]nspect_topology[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR and $REMOTE_PREFIX"
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN ORACLE_PIN=$ORACLE_PIN warmup=$WARMUP trace_steps=$TRACE_STEPS"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact greenfield pin to isolated worker worktrees"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; idx=${HOSTNAME##*-w-}; if [[ "$idx" == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host code sync failed"
  exit 1
}

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I" 2>/dev/null | grep -Eo '192\.168\.[0-9]+\.[0-9]+' | head -1)
[[ -n "$coordinator" ]] || {
  say "ABORT: could not resolve worker-0 coordinator address"
  exit 1
}
coordinator="$coordinator:8476"
say "launching fresh eight-host XPlane capture coordinator=$coordinator"

# shellcheck disable=SC2016
capture_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; pin='"$PIN"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run"; cd "$wt"; /home/gianl/vllm-env/bin/python scripts/greenfield/trace_pipeline_transport.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --expected-code-hash "$pin" --output "$run/trace.rank${idx}.json" --trace-root "$run/traces" --warmup '"$WARMUP"' --trace-steps '"$TRACE_STEPS"'; sha256sum "$run/trace.rank${idx}.json" >"$run/trace.rank${idx}.sha256"; gcloud storage cp --no-clobber "$run/trace.rank${idx}.json" "$run/trace.rank${idx}.sha256" "$remote/host_records/" >/dev/null; for plan in pp8_lp4 pp16_lp2; do xplane=$(find "$run/traces/$plan" -type f -name "*.xplane.pb"); [[ $(printf "%s\n" "$xplane" | sed "/^$/d" | wc -l) -eq 1 ]]; gcloud storage cp --no-clobber "$xplane" "$remote/traces/$plan/trace.rank${idx}.xplane.pb" >/dev/null; done; echo "TRACE_UPLOAD_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$capture_command" >"$RUN_DIR/capture.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/capture.txt" TRACE_UPLOAD_OK || {
  say "ABORT: transport trace/upload did not pass on all eight hosts"
  exit 1
}

gcloud storage cp "$REMOTE_PREFIX/host_records/trace.rank*.json" \
  "$RUN_DIR/host_records/" >/dev/null
gcloud storage rsync --recursive "$REMOTE_PREFIX/traces" "$RUN_DIR/traces" >/dev/null
say "validating fresh XPlanes and appending provenance DB rows"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$ORACLE_PIN" \
  "$RESULTS_DB" "$WORKTREE" "$ORACLE_REPO" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, oracle_pin, db_path, repo, oracle_repo = sys.argv[1:]
run_dir = Path(run_dir)
records = [
    json.loads(path.read_text())
    for path in sorted((run_dir / "host_records").glob("*.json"))
]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if len(records) != 8:
    raise SystemExit(f"expected 8 trace records, got {len(records)}")
if {record["jax_process_index"] for record in records} != set(range(8)):
    raise SystemExit("trace records do not cover JAX processes 0..7")
if {record["launch_process_id"] for record in records} != set(range(8)):
    raise SystemExit("trace records do not cover launch ranks 0..7")
if len({record["hostname"] for record in records}) != 8:
    raise SystemExit("trace records do not contain eight distinct hosts")
if {record["code_hash"] for record in records} != {pin}:
    raise SystemExit("trace record carries stale code hash")
if len({record["topology_hash"] for record in records}) != 1:
    raise SystemExit("trace topology hashes differ")

for record in records:
    if [trace["config"]["plan"] for trace in record["traces"]] != [
        "PP8_LP4", "PP16_LP2"
    ]:
        raise SystemExit("trace plan order differs")
    for trace in record["traces"]:
        expected = 8 if trace["config"]["plan"] == "PP8_LP4" else 16
        if (
            trace["trace_steps"] != 20
            or trace["hlo_collective_counts"] != {"collective-permute": expected}
        ):
            raise SystemExit("trace HLO/step contract failed")
        rank = record["launch_process_id"]
        plan = trace["config"]["plan"].lower()
        local = run_dir / "traces" / plan / f"trace.rank{rank}.xplane.pb"
        digest = file_sha256(local)
        if digest != trace["xplane"]["sha256"]:
            raise SystemExit(f"downloaded XPlane hash mismatch: {local}")

sys.path.insert(0, str(Path(repo) / "scripts" / "analysis"))
import parse_xplane

fleet = {}
for plan, expected_hops in (("pp8_lp4", 8), ("pp16_lp2", 16)):
    result = parse_xplane.aggregate_fleet(
        run_dir / "traces" / plan,
        step_module_re=r"jit_transport",
    )
    if (
        result["n_files"] != 8
        or result["n_cores"] != 64
        or len(result["hosts"]) != 8
        or result["steps_per_core"] != 20
    ):
        raise SystemExit(
            f"{plan} fleet trace inventory invalid: "
            f"files={result['n_files']} cores={result['n_cores']} "
            f"hosts={len(result['hosts'])} steps={result['steps_per_core']}"
        )
    forbidden = sum(
        values["invocations_per_step"]
        for values in result["ops"].values()
        if values.get("hlo_category") in {
            "all-gather", "all-reduce", "all-to-all", "reduce-scatter"
        }
    )
    permutes = sum(
        values["invocations_per_step"]
        for values in result["ops"].values()
        if values.get("hlo_category") == "collective-permute"
    )
    if forbidden != 0 or permutes <= 0:
        raise SystemExit(
            f"{plan} physical trace collective contract failed: "
            f"permutes={permutes} forbidden={forbidden} expected_hops={expected_hops}"
        )
    result["observed_collective_permute_invocations_per_step"] = permutes
    result["expected_hops_per_step"] = expected_hops
    fleet[plan] = result

summary = {
    "code_hash": pin,
    "fleet": fleet,
    "mechanism_only": True,
    "oracle_code_hash": oracle_pin,
    "topology_hash": records[0]["topology_hash"],
}

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-transport-trace-only",
    revision=None,
    env={
        "GLM_ENGINE": "greenfield_pipeline_transport_trace",
        "greenfield_code_hash": pin,
        "legacy_oracle_code_hash": oracle_pin,
        "topology_hash": summary["topology_hash"],
    },
    note="Gate-A fresh PP8/PP16 fleet XPlane mechanism proof",
    harness_repo=repo,
    fork_repo=oracle_repo,
)
for plan, result in fleet.items():
    pv.record_item(
        conn,
        run_id,
        benchmark="greenfield_pipeline_transport_trace",
        item_id=plan,
        prompt="Capture twenty full device-resident transport invocations on all hosts.",
        gold="8 fresh XPlanes, 64 cores, exact point-to-point-only transport.",
        raw_output=json.dumps(result, sort_keys=True),
        extracted=str(result["observed_collective_permute_invocations_per_step"]),
        correct=True,
        score=None,
    )
pv.finalize(
    conn,
    run_id,
    benchmark="greenfield_pipeline_transport_trace",
    metric="contract_valid",
    value=1.0,
    note="Trace-contaminated mechanism only; latency claims use DB 415.",
)
conn.close()
summary["results_db_run_id"] = run_id
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
print(f"TRANSPORT_TRACE_PROOF_VALID plans={len(fleet)} db_run={run_id}")
PY

(cd "$RUN_DIR" && find host_records traces -type f -print0 | sort -z | \
  xargs -0 sha256sum >evidence.sha256)
sha256sum "$RUN_DIR/summary.json" "$RUN_DIR/results_ckpt.db" \
  >>"$RUN_DIR/evidence.sha256"
strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/summary.json" "$RUN_DIR/results_ckpt.db" \
  "$RUN_DIR/evidence.sha256" "$RUN_DIR/orchestrator.log" "$RUN_DIR/sync.txt" \
  "$RUN_DIR/capture.txt" "$RUN_DIR/census_pre.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/" >/dev/null
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ $remote_success == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
say "SUCCESS DB=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' "$RUN_DIR/summary.json")"
say "ARCHIVE=$REMOTE_PREFIX"
trap - EXIT
