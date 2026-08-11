#!/usr/bin/env bash
# Protected Gate-A dependent collective-chain capture. Model-free; existing v4-64 only.
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
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing benchmark from a dirty greenfield worktree" >&2
  exit 2
}
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
MODE=${GLM_GREENFIELD_COLLECTIVE_MODE:-chain}
ASSOCIATION_TRIALS=${GLM_GREENFIELD_COLLECTIVE_ASSOCIATION_TRIALS:-32}
if [[ $MODE == strategy_nd_fingerprint ]]; then
  default_tag=greenfield_collective_association_$(date -u +%Y%m%dT%H%M%S%NZ)
else
  default_tag=greenfield_collectives_$(date -u +%Y%m%dT%H%M%S%NZ)
fi
TAG=${GLM_GREENFIELD_COLLECTIVE_TAG:-$default_tag}
if [[ $MODE == strategy_nd_fingerprint ]]; then
  default_groups=32
  default_operations=all_reduce
  default_shape=1,6144
else
  default_groups=2,4,8,32
  default_operations=control,all_reduce,reduce_scatter,all_gather,collective_permute,all_to_all,fused_tuple_all_reduce
  default_shape=2,6144
fi
GROUP_SIZES=${GLM_GREENFIELD_COLLECTIVE_GROUPS:-$default_groups}
OPERATIONS=${GLM_GREENFIELD_COLLECTIVE_OPERATIONS:-$default_operations}
SHAPE=${GLM_GREENFIELD_COLLECTIVE_SHAPE:-$default_shape}
DTYPE=${GLM_GREENFIELD_COLLECTIVE_DTYPE:-bfloat16}
CHAIN_LENGTH=${GLM_GREENFIELD_COLLECTIVE_CHAIN_LENGTH:-75}
WARMUP=${GLM_GREENFIELD_COLLECTIVE_WARMUP:-200}
ITERATIONS=${GLM_GREENFIELD_COLLECTIVE_ITERATIONS:-1000}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $MODE =~ ^(chain|strategy_nd_fingerprint)$ ]] || {
  echo "invalid collective mode: $MODE" >&2
  exit 2
}
[[ $ASSOCIATION_TRIALS =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid association trial count: $ASSOCIATION_TRIALS" >&2
  exit 2
}
[[ $GROUP_SIZES =~ ^(2|4|8|32)(,(2|4|8|32))*$ ]] || {
  echo "invalid groups: $GROUP_SIZES" >&2
  exit 2
}
[[ $OPERATIONS =~ ^(control|all_reduce|reduce_scatter|all_gather|collective_permute|all_to_all|fused_tuple_all_reduce)(,(control|all_reduce|reduce_scatter|all_gather|collective_permute|all_to_all|fused_tuple_all_reduce))*$ ]] || {
  echo "invalid operations: $OPERATIONS" >&2
  exit 2
}
[[ $SHAPE =~ ^[1-9][0-9]*,[1-9][0-9]*$ ]] || {
  echo "invalid shape: $SHAPE" >&2
  exit 2
}
[[ $DTYPE =~ ^(bfloat16|float32|int32)$ ]] || {
  echo "invalid dtype: $DTYPE" >&2
  exit 2
}
[[ $CHAIN_LENGTH == 75 && $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 200 ]] || {
  echo "protected run requires chain=75 and warmup>=200" >&2
  exit 2
}
[[ $ITERATIONS =~ ^[0-9]+$ && $ITERATIONS -ge 1000 ]] || {
  echo "protected run requires iterations>=1000" >&2
  exit 2
}
if [[ $MODE == strategy_nd_fingerprint ]] && {
  [[ $GROUP_SIZES != 32 ]] || [[ $OPERATIONS != all_reduce ]] ||
    [[ $SHAPE != 1,6144 ]] || [[ $DTYPE != bfloat16 ]] ||
    [[ $ASSOCIATION_TRIALS != 32 ]]
}; then
  echo "protected StrategyND fingerprint requires groups=32 operation=all_reduce shape=1,6144 dtype=bfloat16 trials=32" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/hlo" "$RUN_DIR/association"

say() {
  echo "[collectives $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[i]nspect_topology[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
say "PIN=$PIN ORACLE_PIN=$ORACLE_PIN"
say "MODE=$MODE groups=$GROUP_SIZES operations=$OPERATIONS shape=$SHAPE dtype=$DTYPE chain=$CHAIN_LENGTH warmup=$WARMUP iterations=$ITERATIONS association_trials=$ASSOCIATION_TRIALS"
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
say "launching eight-host dependent chain coordinator=$coordinator"

# shellcheck disable=SC2016
capture_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; pin='"$PIN"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run"; upload_diagnostics() { if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/diagnostic_hlo/" >/dev/null 2>&1 || true; fi; if compgen -G "$run/association/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/association/* "$remote/diagnostic_association/" >/dev/null 2>&1 || true; fi; }; trap upload_diagnostics EXIT; cd "$wt"; GLM_GREENFIELD_RUN_TAG="$tag" /home/gianl/vllm-env/bin/python scripts/greenfield/microbench_collectives.py --mode '"$MODE"' --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --slice-name '"$POD"' --expected-code-hash "$pin" --output "$run/collective.rank${idx}.json" --groups '"$GROUP_SIZES"' --operations '"$OPERATIONS"' --shape '"$SHAPE"' --dtype '"$DTYPE"' --chain-length '"$CHAIN_LENGTH"' --warmup '"$WARMUP"' --iterations '"$ITERATIONS"' --association-trials '"$ASSOCIATION_TRIALS"'; sha256sum "$run/collective.rank${idx}.json" >"$run/collective.rank${idx}.sha256"; gcloud storage cp --no-clobber "$run/collective.rank${idx}.json" "$run/collective.rank${idx}.sha256" "$remote/host_records/" >/dev/null; if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/hlo/" >/dev/null; fi; if compgen -G "$run/association/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/association/* "$remote/association/" >/dev/null; fi; trap - EXIT; echo "CAPTURE_UPLOAD_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$capture_command" >"$RUN_DIR/capture.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/capture.txt" CAPTURE_UPLOAD_OK || {
  say "ABORT: collective capture/upload did not pass on all eight hosts"
  exit 1
}

gcloud storage cp "$REMOTE_PREFIX/host_records/collective.rank*.json" \
  "$RUN_DIR/host_records/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/hlo/" >/dev/null
if [[ $MODE == strategy_nd_fingerprint ]]; then
  gcloud storage cp "$REMOTE_PREFIX/association/*" "$RUN_DIR/association/" >/dev/null
  /home/gianl/vllm-env/bin/python "$WORKTREE/scripts/greenfield/analyze_collective_association.py" \
    --input-bits "$RUN_DIR/association/input_bits.npy" \
    --output-bits "$RUN_DIR/association/output_bits.npy" \
    --host-record "$RUN_DIR/host_records/collective.rank0.json" \
    --output "$RUN_DIR/association/analysis.json"
fi
say "validating fleet agreement and appending provenance DB rows"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$ORACLE_PIN" \
  "$RESULTS_DB" "$WORKTREE" "$ORACLE_REPO" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

run_dir, pin, oracle_pin, db_path, repo, oracle_repo = sys.argv[1:]
run_dir = Path(run_dir)
records = [
    json.loads(path.read_text())
    for path in sorted((run_dir / "host_records").glob("*.json"))
]
if len(records) != 8:
    raise SystemExit(f"expected 8 host records, got {len(records)}")
if {record["jax_process_index"] for record in records} != set(range(8)):
    raise SystemExit("host records do not cover JAX process indices 0..7 exactly")
if {record["launch_process_id"] for record in records} != set(range(8)):
    raise SystemExit("host records do not cover launch process ids 0..7 exactly")
if len({record["hostname"] for record in records}) != 8:
    raise SystemExit("host records do not contain eight distinct hostnames")
if {record["code_hash"] for record in records} != {pin}:
    raise SystemExit("host record carries stale code hash")
topology_hashes = {record["topology_hash"] for record in records}
if len(topology_hashes) != 1:
    raise SystemExit(f"fleet topology hashes differ: {sorted(topology_hashes)}")
mode_values = {record.get("mode", "chain") for record in records}
if len(mode_values) != 1:
    raise SystemExit(f"fleet record modes differ: {sorted(mode_values)}")
mode = mode_values.pop()
case_summaries = []
association_analysis = None
if mode == "strategy_nd_fingerprint":
    if any(record["matrix"] for record in records):
        raise SystemExit("association fingerprint records must not contain timing cases")
    host_items = [record.get("association_fingerprint") for record in records]
    if any(not isinstance(item, dict) for item in host_items):
        raise SystemExit("fleet association fingerprint record is missing")
    reference = host_items[0]
    stable_fields = (
        "accepted_prompt_projection_hlo_sha256",
        "capture",
        "collective_algorithm",
        "collective_groups",
        "config",
        "diagnostic_only",
        "fleet_hlo_hashes",
        "fleet_input_bits_hashes",
        "fleet_output_bits_hashes",
        "hlo",
        "member_device_ids",
        "optimized_hlo_sha256",
    )
    for field in stable_fields:
        if any(item[field] != reference[field] for item in host_items[1:]):
            raise SystemExit(f"fleet association field differs: {field}")
    if reference["config"] != {"seed": 1196575821, "trials": 32, "width": 6144}:
        raise SystemExit(f"unprotected association config: {reference['config']}")
    if reference["collective_groups"] != [reference["member_device_ids"]]:
        raise SystemExit("association group/member mapping differs")
    if sorted(reference["member_device_ids"]) != list(range(32)):
        raise SystemExit("association member mapping does not cover physical ids 0..31")
    if not reference["diagnostic_only"]:
        raise SystemExit("association fingerprint must be marked diagnostic-only")
    if not reference["hlo"]["valid"] or reference["hlo"]["violations"]:
        raise SystemExit("association fingerprint HLO contract is invalid")
    if reference["hlo"]["collective_counts"] != {"all-reduce": 1}:
        raise SystemExit("association fingerprint does not contain one all-reduce")
    capture = reference["capture"]
    for field in (
        "fleet_hlo_hashes",
        "fleet_input_bits_hashes",
        "fleet_output_bits_hashes",
    ):
        if len(reference[field]) != 8 or len(set(reference[field])) != 1:
            raise SystemExit(f"association {field} lacks eight-host agreement")
    if reference["fleet_hlo_hashes"][0] != reference["optimized_hlo_sha256"]:
        raise SystemExit("fleet HLO hash differs from recorded optimized HLO")
    if reference["fleet_input_bits_hashes"][0] != capture["input_bits_sha256"]:
        raise SystemExit("fleet input hash differs from capture")
    if reference["fleet_output_bits_hashes"][0] != capture["output_bits_sha256"]:
        raise SystemExit("fleet output hash differs from capture")
    replica_hashes = capture["local_replica_output_sha256_by_trial"]
    if len(replica_hashes) != 32 or any(
        len(hashes) != 4 or len(set(hashes)) != 1 for hashes in replica_hashes
    ):
        raise SystemExit("association local replicas do not agree for every trial")
    owners = [
        (record, item)
        for record, item in zip(records, host_items, strict=True)
        if item["artifact_manifest"]
    ]
    if len(owners) != 1 or owners[0][0]["jax_process_index"] != 0:
        raise SystemExit("association raw artifacts require exactly one process-zero owner")
    manifest = json.loads((run_dir / "association" / "manifest.json").read_text())
    if manifest != owners[0][1]["artifact_manifest"]:
        raise SystemExit("retrieved association manifest differs from process-zero record")
    input_bits = np.load(run_dir / "association" / "input_bits.npy", allow_pickle=False)
    output_bits = np.load(run_dir / "association" / "output_bits.npy", allow_pickle=False)
    sys.path.insert(0, repo)
    from glm_tpu.greenfield.benchmarking import array_sha256

    if array_sha256(input_bits) != capture["input_bits_sha256"]:
        raise SystemExit("retrieved association input bits fail raw checksum")
    if array_sha256(output_bits) != capture["output_bits_sha256"]:
        raise SystemExit("retrieved association output bits fail raw checksum")
    association_analysis = json.loads(
        (run_dir / "association" / "analysis.json").read_text()
    )
    if (
        association_analysis["input_bits_sha256"] != capture["input_bits_sha256"]
        or association_analysis["output_bits_sha256"] != capture["output_bits_sha256"]
        or association_analysis["optimized_hlo_sha256"]
        != reference["optimized_hlo_sha256"]
    ):
        raise SystemExit("offline association analysis is not linked to raw capture/HLO")
    case_summaries.append(
        {
            "analysis": association_analysis,
            "case": {
                "dtype": "bfloat16",
                "group_size": 32,
                "kind": "strategy_nd_association_fingerprint",
                "rows": 1,
                "width": 6144,
            },
            "collective_algorithm": reference["collective_algorithm"],
            "collective_groups": reference["collective_groups"],
            "diagnostic_only": True,
            "hlo_sha256": reference["optimized_hlo_sha256"],
            "input_bits_sha256": capture["input_bits_sha256"],
            "output_bits_sha256": capture["output_bits_sha256"],
        }
    )
elif mode == "chain":
    case_keys = [
        (
            item["config"]["kind"],
            item["config"]["group_size"],
            item["config"]["dtype"],
            item["config"]["rows"],
            item["config"]["width"],
        )
        for item in records[0]["matrix"]
    ]
    if not case_keys or len(case_keys) != len(set(case_keys)):
        raise SystemExit("reference host has an empty or duplicate matrix")
    for case_index, key in enumerate(case_keys):
        host_items = [record["matrix"][case_index] for record in records]
        if any(
            (
                item["config"]["kind"],
                item["config"]["group_size"],
                item["config"]["dtype"],
                item["config"]["rows"],
                item["config"]["width"],
            )
            != key
            for item in host_items
        ):
            raise SystemExit(f"fleet matrix order/config differs at {case_index}")
        if len({item["optimized_hlo_sha256"] for item in host_items}) != 1:
            raise SystemExit(f"fleet optimized HLO differs for {key}")
        for item in host_items:
            config = item["config"]
            if (
                config["chain_length"] != 75
                or config["warmup_iterations"] < 200
                or config["measured_iterations"] < 1000
            ):
                raise SystemExit(f"unprotected benchmark config for {key}")
            if item["first_addressable_checksum"] != item["last_addressable_checksum"]:
                raise SystemExit(f"nondeterministic output for {key}")
            if not item["hlo"]["valid"] or item["hlo"]["violations"]:
                raise SystemExit(f"invalid HLO contract for {key}")
            if len(item["latency"]["samples_ms"]) != config["measured_iterations"]:
                raise SystemExit(f"incomplete latency distribution for {key}")
        p50_by_jax_process = {
            str(record["jax_process_index"]): record["matrix"][case_index]["latency"]["p50_ms"]
            for record in records
        }
        case_summaries.append(
            {
                "case": {
                    "dtype": key[2],
                    "group_size": key[1],
                    "kind": key[0],
                    "rows": key[3],
                    "width": key[4],
                },
                "collective_groups": host_items[0]["collective_groups"],
                "hlo_sha256": host_items[0]["optimized_hlo_sha256"],
                "maximum_host_p50_ms": max(p50_by_jax_process.values()),
                "minimum_host_p50_ms": min(p50_by_jax_process.values()),
                "p50_ms_by_jax_process": p50_by_jax_process,
            }
        )
else:
    raise SystemExit(f"unknown fleet record mode: {mode}")

summary = {
    "cases": case_summaries,
    "code_hash": pin,
    "hostnames": sorted(record["hostname"] for record in records),
    "launch_to_jax_process": {
        str(record["launch_process_id"]): record["jax_process_index"]
        for record in sorted(records, key=lambda item: item["launch_process_id"])
    },
    "mechanism_only": True,
    "mode": mode,
    "oracle_code_hash": oracle_pin,
    "topology_hash": topology_hashes.pop(),
}
if association_analysis is not None:
    summary["association_analysis"] = association_analysis

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model=(
        "zai-org/GLM-5.2-FP8:greenfield-strategy-nd-diagnostic-only"
        if mode == "strategy_nd_fingerprint"
        else "zai-org/GLM-5.2-FP8:greenfield-collective-mechanism-only"
    ),
    revision=None,
    env={
        "GLM_ENGINE": (
            "greenfield_strategy_nd_fingerprint"
            if mode == "strategy_nd_fingerprint"
            else "greenfield_collective_chain"
        ),
        "greenfield_code_hash": pin,
        "legacy_oracle_code_hash": oracle_pin,
        "topology_hash": summary["topology_hash"],
    },
    note=(
        "Model-free StrategyND association diagnostic; no latency/throughput claim"
        if mode == "strategy_nd_fingerprint"
        else "Gate-A dependent collective-chain synthetic mechanism benchmark"
    ),
    harness_repo=repo,
    fork_repo=oracle_repo,
)
for case in case_summaries:
    identity = case["case"]
    item_id = (
        f"{identity['kind']}:g{identity['group_size']}:"
        f"{identity['dtype']}:{identity['rows']}x{identity['width']}"
    )
    pv.record_item(
        conn,
        run_id,
        benchmark=(
            "greenfield_strategy_nd_fingerprint"
            if mode == "strategy_nd_fingerprint"
            else "greenfield_collective_chain"
        ),
        item_id=item_id,
        prompt=(
            "Capture raw BF16 association fingerprints from one byte-pinned 32-way StrategyND reduction."
            if mode == "strategy_nd_fingerprint"
            else "Measure one protected 75-operation dependent synthetic collective chain."
        ),
        gold=(
            "Exact StrategyND HLO, eight-host bit agreement, raw input/output integrity, offline replay."
            if mode == "strategy_nd_fingerprint"
            else "Exact optimized HLO, deterministic checksum, complete warmed distribution."
        ),
        raw_output=json.dumps(case, sort_keys=True),
        extracted=str(
            case["analysis"]["union_exact_column_count"]
            if mode == "strategy_nd_fingerprint"
            else case["maximum_host_p50_ms"]
        ),
        correct=True,
        score=None,
    )
pv.finalize(
    conn,
    run_id,
    benchmark=(
        "greenfield_strategy_nd_fingerprint"
        if mode == "strategy_nd_fingerprint"
        else "greenfield_collective_chain"
    ),
    metric="contract_valid",
    value=1.0,
    note="Synthetic diagnostic/mechanism only; not model latency or throughput.",
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
print(f"COLLECTIVE_PROOF_VALID cases={len(case_summaries)} db_run={run_id}")
PY

(cd "$RUN_DIR" && find host_records hlo association -type f -print0 | sort -z | \
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
if [[ $MODE == strategy_nd_fingerprint ]]; then
  gcloud storage cp --no-clobber "$RUN_DIR/association/analysis.json" \
    "$REMOTE_PREFIX/association/" >/dev/null
fi
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ $remote_success == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
say "SUCCESS cases=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["cases"]))' "$RUN_DIR/summary.json")"
say "ARCHIVE=$REMOTE_PREFIX"
trap - EXIT
