#!/usr/bin/env bash
# Protected Gate-A PP8/PP16 device-resident transport capture. Model-free.
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
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing benchmark outside $BRANCH" >&2
  exit 2
}
git -C "$WORKTREE" diff --quiet && git -C "$WORKTREE" diff --cached --quiet || {
  echo "refusing benchmark from a dirty greenfield worktree" >&2
  exit 2
}
unexpected_untracked=()
while IFS= read -r -d '' path; do
  case "$path" in
    "docs/DeepSeek-V4-Flash TPU Port — August 2026 Rebase - Obsolescence Audit and Codex Handoff.md" | \
      "docs/deep-research-report.md") ;;
    *) unexpected_untracked+=("$path") ;;
  esac
done < <(git -C "$WORKTREE" ls-files --others --exclude-standard -z)
[[ ${#unexpected_untracked[@]} -eq 0 ]] || {
  echo "refusing benchmark with unexpected untracked files:" >&2
  printf '  %s\n' "${unexpected_untracked[@]}" >&2
  exit 2
}
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TAG=${GLM_GREENFIELD_TRANSPORT_TAG:-greenfield_transport_$(date -u +%Y%m%dT%H%M%S%NZ)}
PLANS=${GLM_GREENFIELD_TRANSPORT_PLANS:-PP8_LP4,PP16_LP2}
KINDS=${GLM_GREENFIELD_TRANSPORT_KINDS:-control,device_resident}
PAIRED_KINDS=${GLM_GREENFIELD_TRANSPORT_PAIRED_KINDS:-control,device_resident,packed_device_resident}
PAYLOADS=${GLM_GREENFIELD_TRANSPORT_PAYLOADS:-bfloat16:1:6144,bfloat16:2:6144,bfloat16:1:2048,int32:1:2048}
WARMUP=${GLM_GREENFIELD_TRANSPORT_WARMUP:-200}
ITERATIONS=${GLM_GREENFIELD_TRANSPORT_ITERATIONS:-2000}
PAIRED_PRODUCTION=${GLM_GREENFIELD_TRANSPORT_PAIRED_PRODUCTION:-0}
PAIRED_ONLY=${GLM_GREENFIELD_TRANSPORT_PAIRED_ONLY:-0}
IFS= read -r ATTEMPT_NONCE </proc/sys/kernel/random/uuid
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $PLANS =~ ^(PP8_LP4|PP16_LP2)(,(PP8_LP4|PP16_LP2))*$ ]] || {
  echo "invalid plans: $PLANS" >&2
  exit 2
}
[[ $KINDS =~ ^(control|device_resident|pallas_remote_copy)(,(control|device_resident|pallas_remote_copy))*$ ]] || {
  echo "invalid kinds: $KINDS" >&2
  exit 2
}
[[ $PAIRED_KINDS =~ ^(control|device_resident|packed_device_resident)(,(control|device_resident|packed_device_resident))*$ ]] || {
  echo "invalid paired kinds: $PAIRED_KINDS" >&2
  exit 2
}
[[ $PAYLOADS =~ ^(bfloat16|float32|int32):[1-9][0-9]*:[1-9][0-9]*(,(bfloat16|float32|int32):[1-9][0-9]*:[1-9][0-9]*)*$ ]] || {
  echo "invalid payloads: $PAYLOADS" >&2
  exit 2
}
[[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 200 ]] || {
  echo "protected run requires warmup>=200" >&2
  exit 2
}
[[ $ITERATIONS =~ ^[0-9]+$ && $ITERATIONS -ge 1000 ]] || {
  echo "protected run requires iterations>=1000" >&2
  exit 2
}
[[ $PAIRED_PRODUCTION =~ ^[01]$ ]] || {
  echo "paired production flag must be 0 or 1" >&2
  exit 2
}
[[ $PAIRED_ONLY =~ ^[01]$ ]] || {
  echo "paired-only flag must be 0 or 1" >&2
  exit 2
}
[[ $ATTEMPT_NONCE =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || {
  echo "failed to generate a protected attempt nonce" >&2
  exit 2
}
if [[ $PAIRED_ONLY == 1 ]]; then
  [[ $PAIRED_PRODUCTION == 1 ]] || {
    echo "paired-only requires paired production" >&2
    exit 2
  }
  [[ ,$PAIRED_KINDS, == *,device_resident,* && \
     ,$PAIRED_KINDS, == *,packed_device_resident,* ]] || {
    echo "paired-only adjudication requires device_resident and packed_device_resident" >&2
    exit 2
  }
  [[ $PLANS == PP8_LP4 ]] || {
    echo "paired-only challenger requires exactly PP8_LP4" >&2
    exit 2
  }
fi

[[ ! -e $RUN_DIR ]] || {
  echo "refusing to reuse existing run directory: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/hlo"

say() {
  echo "[transport $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock -n 8 || {
  say "ABORT: repository mirror sync is active; retry outside its window"
  exit 1
}
existing_remote=$(gcloud storage ls "$REMOTE_PREFIX/**" 2>/dev/null || true)
[[ -z $existing_remote ]] || {
  say "ABORT: refusing to reuse nonempty remote prefix $REMOTE_PREFIX"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[i]nspect_topology[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
terminal_success_done=0

rollback_provisional_db() {
  local run_id_file="$RUN_DIR/results_db_run_id.txt"
  /home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$run_id_file" \
    "$TAG" "$PIN" "$ORACLE_PIN" "$ATTEMPT_NONCE" \
    >"$RUN_DIR/provisional_db_rollback.txt" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import sys

db_path, run_id_path, run_tag, pin, oracle_pin, attempt_nonce = sys.argv[1:]
run_id_file = Path(run_id_path)
try:
    recorded_run_id = (
        int(run_id_file.read_text().strip()) if run_id_file.is_file() else None
    )
except ValueError:
    recorded_run_id = None
connection = sqlite3.connect(db_path)
connection.execute("BEGIN IMMEDIATE")
candidates = []
for row in connection.execute(
    "SELECT run_id, model, model_revision, harness_git, fork_git, env_json, "
    "pod, note FROM runs WHERE model = ?",
    ("zai-org/GLM-5.2-FP8:greenfield-transport-mechanism-only",),
):
    try:
        candidate_environment = json.loads(row[5])
    except (TypeError, json.JSONDecodeError):
        continue
    if (
        candidate_environment.get("greenfield_run_tag") == run_tag
        and candidate_environment.get("greenfield_code_hash") == pin
        and candidate_environment.get("greenfield_attempt_nonce") == attempt_nonce
    ):
        candidates.append((row, candidate_environment))
if not candidates:
    connection.rollback()
    print("NO_PROVISIONAL_DB_RUN")
    raise SystemExit(0)
if len(candidates) != 1:
    connection.rollback()
    raise SystemExit("refusing ambiguous provisional transport DB rollback")
(run, environment), = candidates
run_id = run[0]
if recorded_run_id is not None and recorded_run_id != run_id:
    connection.rollback()
    raise SystemExit("provisional transport DB run-id binding differs")
paired_only = environment.get("paired_only")
expected_environment = {
    "GLM_ENGINE": "greenfield_pipeline_transport",
    "greenfield_code_hash": pin,
    "greenfield_attempt_nonce": attempt_nonce,
    "greenfield_run_tag": run_tag,
    "legacy_oracle_code_hash": oracle_pin,
    "paired_only": paired_only,
    "topology_hash": environment.get("topology_hash"),
}
items = connection.execute(
    "SELECT benchmark, correct FROM items WHERE run_id = ? ORDER BY id",
    (run_id,),
).fetchall()
summaries = connection.execute(
    "SELECT benchmark, metric, value FROM summary WHERE run_id = ? ORDER BY id",
    (run_id,),
).fetchall()
allowed_benchmarks = {
    "greenfield_pipeline_transport",
    "greenfield_paired_pipeline_transport",
}
expected_note = (
    "Gate-D PP8 exact packed transport challenger"
    if paired_only
    else "Gate-A PP8/PP16 device-resident synthetic transport benchmark"
)
expected_summary_benchmark = (
    "greenfield_paired_pipeline_transport"
    if paired_only
    else "greenfield_pipeline_transport"
)
if (
    run[1] != "zai-org/GLM-5.2-FP8:greenfield-transport-mechanism-only"
    or run[2] is not None
    or not run[3]
    or not pin.startswith(run[3])
    or not run[4]
    or not oracle_pin.startswith(run[4])
    or not isinstance(paired_only, bool)
    or environment != expected_environment
    or not re.fullmatch(r"[0-9a-f]{64}", environment["topology_hash"])
    or run[6] != "db-v4-64-od"
    or run[7] != expected_note
    or any(
        benchmark not in allowed_benchmarks or correct != 1
        for benchmark, correct in items
    )
    or summaries not in (
        [],
        [(expected_summary_benchmark, "contract_valid", 1.0)],
    )
):
    connection.rollback()
    raise SystemExit("refusing non-identical provisional transport DB rollback")
connection.execute("DELETE FROM summary WHERE run_id = ?", (run_id,))
connection.execute("DELETE FROM items WHERE run_id = ?", (run_id,))
connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
if connection.execute(
    "SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,)
).fetchone()[0]:
    connection.rollback()
    raise SystemExit("provisional transport DB rollback did not remove run")
connection.commit()
print(f"ROLLED_BACK_PROVISIONAL_DB_RUN={run_id}")
PY
}

on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 && $terminal_success_done -eq 0 ]]; then
    rollback_provisional_db || true
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR and $REMOTE_PREFIX"
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR"
say "PIN=$PIN ORACLE_PIN=$ORACLE_PIN ATTEMPT_NONCE=$ATTEMPT_NONCE"
say "MATRIX plans=$PLANS kinds=$KINDS paired_kinds=$PAIRED_KINDS payloads=$PAYLOADS warmup=$WARMUP iterations=$ITERATIONS paired=$PAIRED_PRODUCTION paired_only=$PAIRED_ONLY"
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
say "launching eight-host device-resident transport coordinator=$coordinator"

# shellcheck disable=SC2016
capture_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; pin='"$PIN"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; paired='"$PAIRED_PRODUCTION"'; paired_only='"$PAIRED_ONLY"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run"; upload_diagnostics() { if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/diagnostic_hlo/" >/dev/null 2>&1 || true; fi; }; trap upload_diagnostics EXIT; cd "$wt"; paired_arg=(); [[ "$paired" == 1 ]] && paired_arg+=(--paired-production); [[ "$paired_only" == 1 ]] && paired_arg+=(--paired-only); GLM_GREENFIELD_RUN_TAG="$tag" /home/gianl/vllm-env/bin/python scripts/greenfield/microbench_pipeline_transport.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --expected-code-hash "$pin" --output "$run/transport.rank${idx}.json" --plans '"$PLANS"' --kinds '"$KINDS"' --paired-kinds '"$PAIRED_KINDS"' --payloads '"$PAYLOADS"' --warmup '"$WARMUP"' --iterations '"$ITERATIONS"' "${paired_arg[@]}"; sha256sum "$run/transport.rank${idx}.json" >"$run/transport.rank${idx}.sha256"; gcloud storage cp --no-clobber "$run/transport.rank${idx}.json" "$run/transport.rank${idx}.sha256" "$remote/host_records/" >/dev/null; if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/hlo/" >/dev/null; fi; trap - EXIT; echo "CAPTURE_UPLOAD_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$capture_command" >"$RUN_DIR/capture.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/capture.txt" CAPTURE_UPLOAD_OK || {
  say "ABORT: transport capture/upload did not pass on all eight hosts"
  exit 1
}

gcloud storage cp "$REMOTE_PREFIX/host_records/transport.rank*.json" \
  "$RUN_DIR/host_records/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/hlo/" >/dev/null
say "validating fleet agreement and appending provenance DB rows"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$ORACLE_PIN" \
  "$RESULTS_DB" "$WORKTREE" "$ORACLE_REPO" "$PAIRED_PRODUCTION" \
  "$PAIRED_ONLY" "$ATTEMPT_NONCE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

(
    run_dir,
    pin,
    oracle_pin,
    db_path,
    repo,
    oracle_repo,
    paired_expected_raw,
    paired_only_raw,
    attempt_nonce,
) = sys.argv[1:]
paired_expected = paired_expected_raw == "1"
paired_only = paired_only_raw == "1"
run_dir = Path(run_dir)
records = [
    json.loads(path.read_text())
    for path in sorted((run_dir / "host_records").glob("*.json"))
]
if len(records) != 8:
    raise SystemExit(f"expected 8 host records, got {len(records)}")
if {record["jax_process_index"] for record in records} != set(range(8)):
    raise SystemExit("host records do not cover JAX process indices 0..7")
if {record["launch_process_id"] for record in records} != set(range(8)):
    raise SystemExit("host records do not cover launch process ids 0..7")
if len({record["hostname"] for record in records}) != 8:
    raise SystemExit("host records do not contain eight distinct hostnames")
if {record["code_hash"] for record in records} != {pin}:
    raise SystemExit("host record carries stale code hash")
if any(
    not record["single_compiled_invocation_per_case"]
    or record["stage_dispatch"] != "device_program_only"
    or record["model_equivalent_compute"]
    for record in records
):
    raise SystemExit("host/Python/Ray dispatch or model-equivalent compute contract failed")
if {record.get("paired_only") for record in records} != {paired_only}:
    raise SystemExit("fleet paired-only provenance differs")
topology_hashes = {record["topology_hash"] for record in records}
if len(topology_hashes) != 1:
    raise SystemExit(f"fleet topology hashes differ: {sorted(topology_hashes)}")
plan_contracts = [
    json.dumps(record["plan_contracts"], sort_keys=True) for record in records
]
if len(set(plan_contracts)) != 1:
    raise SystemExit("fleet plan/lane/pair contracts differ")
case_keys = [
    (
        item["config"]["plan"],
        item["config"]["kind"],
        item["config"]["dtype"],
        item["config"]["rows"],
        item["config"]["width"],
    )
    for item in records[0]["matrix"]
]
if (not paired_only and not case_keys) or len(case_keys) != len(set(case_keys)):
    raise SystemExit("reference host has an unexpected empty or duplicate matrix")

case_summaries = []
for case_index, key in enumerate(case_keys):
    host_items = [record["matrix"][case_index] for record in records]
    if any(
        (
            item["config"]["plan"],
            item["config"]["kind"],
            item["config"]["dtype"],
            item["config"]["rows"],
            item["config"]["width"],
        ) != key
        for item in host_items
    ):
        raise SystemExit(f"fleet matrix order/config differs at {case_index}")
    if len({item["optimized_hlo_sha256"] for item in host_items}) != 1:
        raise SystemExit(f"fleet optimized HLO differs for {key}")
    expected_stages = 8 if key[0] == "PP8_LP4" else 16
    for item in host_items:
        config = item["config"]
        if (
            config["stage_count"] != expected_stages
            or config["warmup_iterations"] < 200
            or config["measured_iterations"] < 1000
        ):
            raise SystemExit(f"unprotected transport config for {key}")
        if item["first_addressable_checksum"] != item["last_addressable_checksum"]:
            raise SystemExit(f"nondeterministic output for {key}")
        if not item["hlo"]["valid"] or item["hlo"]["violations"]:
            raise SystemExit(f"invalid HLO contract for {key}")
        counts = item["hlo"]["collective_counts"]
        expected_counts = (
            {"collective-permute": expected_stages}
            if key[1] == "device_resident"
            else {}
        )
        if counts != expected_counts:
            raise SystemExit(f"physical collective count mismatch for {key}: {counts}")
        pallas = item["pallas_remote_copy"]
        expected_pallas_calls = (
            expected_stages if key[1] == "pallas_remote_copy" else 0
        )
        if (
            not pallas["passed"]
            or pallas["violations"]
            or pallas["kernel_custom_call_count"] != expected_pallas_calls
        ):
            raise SystemExit(f"Pallas remote-copy contract failed for {key}: {pallas}")
        if len(item["latency"]["samples_ms"]) != config["measured_iterations"]:
            raise SystemExit(f"incomplete latency distribution for {key}")
    distributions = {
        name: max(item["latency"][name] for item in host_items)
        for name in ("p50_ms", "p90_ms", "p95_ms", "p99_ms")
    }
    case_summaries.append(
        {
            "case": {
                "dtype": key[2],
                "kind": key[1],
                "plan": key[0],
                "rows": key[3],
                "width": key[4],
            },
            "fleet_maximum_host_latency": distributions,
            "hlo_sha256": host_items[0]["optimized_hlo_sha256"],
            "output_checksum": host_items[0]["first_addressable_checksum"],
            "physical_pairs": host_items[0]["physical_pairs"],
        }
    )

paired_keys = [
    (item["config"]["plan"], item["config"]["kind"])
    for item in records[0]["paired_matrix"]
]
if paired_expected != bool(paired_keys):
    raise SystemExit(
        f"paired production expectation mismatch: expected={paired_expected} "
        f"keys={paired_keys}"
    )
if len(paired_keys) != len(set(paired_keys)):
    raise SystemExit("paired production matrix contains duplicate cases")
paired_case_summaries = []
for case_index, key in enumerate(paired_keys):
    host_items = [record["paired_matrix"][case_index] for record in records]
    if any(
        (item["config"]["plan"], item["config"]["kind"]) != key
        for item in host_items
    ):
        raise SystemExit(f"fleet paired matrix differs at {case_index}")
    if len({item["optimized_hlo_sha256"] for item in host_items}) != 1:
        raise SystemExit(f"fleet paired HLO differs for {key}")
    expected_stages = 8 if key[0] == "PP8_LP4" else 16
    for item in host_items:
        config = item["config"]
        if (
            config["stage_count"] != expected_stages
            or config["hidden_width"] != 6144
            or config["residual_shape"] != [2, 1, 6144]
            or config["metadata_shape"] != [1, 2053]
            or config["metadata_width"] != 2053
            or config["packed_dtype"] != "uint16"
            or config["packed_width"] != 16394
            or config["warmup_iterations"] < 200
            or config["measured_iterations"] < 1000
        ):
            raise SystemExit(f"unprotected paired config for {key}")
        if item["first_addressable_checksum"] != item["last_addressable_checksum"]:
            raise SystemExit(f"nondeterministic paired output for {key}")
        contract = item["hlo"]
        if not contract["passed"] or contract["violations"]:
            raise SystemExit(f"paired HLO failed for {key}: {contract}")
        expected_kernels = expected_stages if key[1] == "pallas_remote_copy" else 0
        expected_collectives = {
            "control": 0,
            "device_resident": 2 * expected_stages,
            "packed_device_resident": expected_stages,
            "pallas_remote_copy": 0,
        }[key[1]]
        if (
            contract["kernel_custom_call_count"] != expected_kernels
            or contract["collective_count"] != expected_collectives
        ):
            raise SystemExit(f"paired mechanism count mismatch for {key}")
        if len(item["latency"]["samples_ms"]) != config["measured_iterations"]:
            raise SystemExit(f"incomplete paired latency distribution for {key}")
    distributions = {
        name: max(item["latency"][name] for item in host_items)
        for name in ("p50_ms", "p90_ms", "p95_ms", "p99_ms")
    }
    paired_case_summaries.append(
        {
            "case": {
                "kind": key[1],
                "metadata_shape": [1, 2053],
                "packed_width": 16394,
                "plan": key[0],
                "residual_shape": [2, 1, 6144],
            },
            "fleet_maximum_host_latency": distributions,
            "hlo_sha256": host_items[0]["optimized_hlo_sha256"],
            "output_checksum": host_items[0]["first_addressable_checksum"],
            "physical_pairs": host_items[0]["physical_pairs"],
        }
    )

for record in records:
    checksums = {
        (item["config"]["plan"], item["config"]["kind"]):
        item["first_addressable_checksum"]
        for item in record["paired_matrix"]
    }
    for plan in {key[0] for key in paired_keys}:
        reference = checksums.get((plan, "device_resident"))
        for candidate_kind in ("packed_device_resident", "pallas_remote_copy"):
            candidate = checksums.get((plan, candidate_kind))
            if reference is not None and candidate is not None and reference != candidate:
                raise SystemExit(
                    f"paired {candidate_kind} output differs from ppermute "
                    f"on {record['hostname']}"
                )

controls = {
    (case["case"]["plan"], case["case"]["dtype"], case["case"]["rows"], case["case"]["width"]):
    case["fleet_maximum_host_latency"]["p50_ms"]
    for case in case_summaries
    if case["case"]["kind"] == "control"
}
for case in case_summaries:
    identity = case["case"]
    key = (identity["plan"], identity["dtype"], identity["rows"], identity["width"])
    case["control_net_p50_ms"] = (
        None
        if identity["kind"] == "control"
        else case["fleet_maximum_host_latency"]["p50_ms"] - controls[key]
    )

paired_controls = {
    case["case"]["plan"]: case["fleet_maximum_host_latency"]["p50_ms"]
    for case in paired_case_summaries
    if case["case"]["kind"] == "control"
}
for case in paired_case_summaries:
    identity = case["case"]
    control_p50 = paired_controls.get(identity["plan"])
    case["control_net_p50_ms"] = (
        None
        if identity["kind"] == "control" or control_p50 is None
        else case["fleet_maximum_host_latency"]["p50_ms"]
        - control_p50
    )

paired_by_key = {
    (case["case"]["plan"], case["case"]["kind"]): case
    for case in paired_case_summaries
}
paired_adjudication = []
for plan in sorted({key[0] for key in paired_keys}):
    reference = paired_by_key.get((plan, "device_resident"))
    packed = paired_by_key.get((plan, "packed_device_resident"))
    if reference is None or packed is None:
        if paired_only:
            raise SystemExit(f"paired-only run lacks packed/reference pair for {plan}")
        continue
    reference_latency = reference["fleet_maximum_host_latency"]
    packed_latency = packed["fleet_maximum_host_latency"]
    p50_reduction_fraction = (
        1.0 - packed_latency["p50_ms"] / reference_latency["p50_ms"]
    )
    p99_non_regression = packed_latency["p99_ms"] <= reference_latency["p99_ms"]
    material_win = p50_reduction_fraction >= 0.20 and p99_non_regression
    paired_adjudication.append(
        {
            "decision": "promote" if material_win else "reject",
            "exact_output": True,
            "hlo_launch_count_packed": 8 if plan == "PP8_LP4" else 16,
            "hlo_launch_count_reference": 16 if plan == "PP8_LP4" else 32,
            "material_win": material_win,
            "minimum_required_p50_reduction_fraction": 0.20,
            "p50_reduction_fraction": p50_reduction_fraction,
            "p99_non_regression": p99_non_regression,
            "plan": plan,
        }
    )

summary = {
    "attempt_nonce": attempt_nonce,
    "cases": case_summaries,
    "code_hash": pin,
    "hostnames": sorted(record["hostname"] for record in records),
    "launch_to_jax_process": {
        str(record["launch_process_id"]): record["jax_process_index"]
        for record in sorted(records, key=lambda item: item["launch_process_id"])
    },
    "mechanism_only": True,
    "oracle_code_hash": oracle_pin,
    "paired_adjudication": paired_adjudication,
    "paired_cases": paired_case_summaries,
    "paired_only": paired_only,
    "plan_contracts": records[0]["plan_contracts"],
    "topology_hash": topology_hashes.pop(),
}

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-transport-mechanism-only",
    revision=None,
    env={
        "GLM_ENGINE": "greenfield_pipeline_transport",
        "greenfield_attempt_nonce": attempt_nonce,
        "greenfield_code_hash": pin,
        "greenfield_run_tag": run_dir.name,
        "legacy_oracle_code_hash": oracle_pin,
        "paired_only": paired_only,
        "topology_hash": summary["topology_hash"],
    },
    note=(
        "Gate-D PP8 exact packed transport challenger"
        if paired_only
        else "Gate-A PP8/PP16 device-resident synthetic transport benchmark"
    ),
    harness_repo=repo,
    fork_repo=oracle_repo,
)
(run_dir / "results_db_run_id.txt").write_text(f"{run_id}\n")
for case in case_summaries:
    identity = case["case"]
    item_id = (
        f"{identity['plan']}:{identity['kind']}:"
        f"{identity['dtype']}:{identity['rows']}x{identity['width']}"
    )
    pv.record_item(
        conn,
        run_id,
        benchmark="greenfield_pipeline_transport",
        item_id=item_id,
        prompt="Measure one protected full device-resident pipeline-stage ring.",
        gold="Exact point-to-point HLO, deterministic checksum, warmed distribution.",
        raw_output=json.dumps(case, sort_keys=True),
        extracted=str(case["fleet_maximum_host_latency"]["p50_ms"]),
        correct=True,
        score=None,
    )
for case in paired_case_summaries:
    identity = case["case"]
    item_id = f"{identity['plan']}:{identity['kind']}:paired-production"
    pv.record_item(
        conn,
        run_id,
        benchmark="greenfield_paired_pipeline_transport",
        item_id=item_id,
        prompt="Measure residual plus compact metadata over one full stage ring.",
        gold="Exact paired output, intended HLO mechanism, warmed distribution.",
        raw_output=json.dumps(case, sort_keys=True),
        extracted=str(case["fleet_maximum_host_latency"]["p50_ms"]),
        correct=True,
        score=None,
    )
pv.finalize(
    conn,
    run_id,
    benchmark=(
        "greenfield_paired_pipeline_transport"
        if paired_only
        else "greenfield_pipeline_transport"
    ),
    metric="contract_valid",
    value=1.0,
    note="Synthetic mechanism only; not model latency or throughput.",
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
print(f"TRANSPORT_PROOF_VALID cases={len(case_summaries)} db_run={run_id}")
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

(cd "$RUN_DIR" && find host_records hlo -type f -print0 | sort -z | \
  xargs -0 sha256sum >evidence.sha256)
sha256sum "$RUN_DIR/summary.json" "$RUN_DIR/results_ckpt.db" \
  "$RUN_DIR/results_db_run_id.txt" "$RUN_DIR/sync.txt" \
  "$RUN_DIR/capture.txt" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/census_post.txt" >>"$RUN_DIR/evidence.sha256"
touch "$RUN_DIR/SUCCESS"
gcloud storage cp --no-clobber "$RUN_DIR/summary.json" "$RUN_DIR/results_ckpt.db" \
  "$RUN_DIR/evidence.sha256" "$RUN_DIR/orchestrator.log" "$RUN_DIR/sync.txt" \
  "$RUN_DIR/capture.txt" "$RUN_DIR/census_pre.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/results_db_run_id.txt" "$REMOTE_PREFIX/" >/dev/null
for artifact in summary.json results_ckpt.db evidence.sha256 orchestrator.log \
  sync.txt capture.txt census_pre.txt census_post.txt results_db_run_id.txt; do
  remote_artifact=$(gcloud storage ls "$REMOTE_PREFIX/$artifact" 2>/dev/null || true)
  [[ $remote_artifact == "$REMOTE_PREFIX/$artifact" ]] || {
    say "ABORT: remote evidence did not verify: $artifact"
    exit 1
  }
done
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" \
  >/dev/null
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ $remote_success == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
terminal_success_done=1
say "SUCCESS cases=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["cases"]))' "$RUN_DIR/summary.json")"
say "ARCHIVE=$REMOTE_PREFIX"
trap - EXIT
