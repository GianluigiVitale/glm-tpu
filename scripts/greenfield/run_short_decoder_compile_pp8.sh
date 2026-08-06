#!/usr/bin/env bash
# Protected first real 78-layer / 2K PP8 decoder-body compile and execution.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly WARMUP=${GLM_GREENFIELD_SHORT_DECODER_WARMUP:-2}
readonly ITERATIONS=${GLM_GREENFIELD_SHORT_DECODER_ITERATIONS:-10}
readonly TRACE_STEPS=${GLM_GREENFIELD_SHORT_DECODER_TRACE_STEPS:-0}
readonly SOURCE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP8_LP4/greenfield_full_pack_pp8_20260805T182222755355852Z
readonly SOURCE_MANIFEST_SHA=0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1
readonly SOURCE_RUNTIME_TAG=greenfield_runtime_pack_pp8_20260806T002756318310857Z
readonly SOURCE_RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime/PP8_LP4/$SOURCE_RUNTIME_TAG
readonly SOURCE_RUNTIME_MANIFEST_SHA=fdedaae31fb3c094266272ed48dfe62bb098257a78272b93c14eafbd57e31dec
readonly RUNTIME_KIND=${GLM_GREENFIELD_DECODER_RUNTIME_KIND:-pallas_feature}
case "$RUNTIME_KIND" in
  reference)
    readonly RUNTIME_TAG=$SOURCE_RUNTIME_TAG
    readonly RUNTIME_ROOT=$SOURCE_RUNTIME_ROOT
    readonly RUNTIME_MANIFEST_SHA=$SOURCE_RUNTIME_MANIFEST_SHA
    readonly RUNTIME_LAYOUT_HASH=841a18f6dbbf329243482f97f01d28ca22211d63d323fca859477349cc0abcac
    readonly SPARSE_MOE_BACKEND=reference
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_reference
    ;;
  pallas_feature)
    readonly RUNTIME_TAG=greenfield_runtime_feature_pack_pp8_20260806T064010287072141Z
    readonly RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$RUNTIME_TAG
    readonly RUNTIME_MANIFEST_SHA=54e2f89b1832b994acbf9ef36f5f6ce68c942d9146efc4d7c15360d68b6d9917
    readonly RUNTIME_LAYOUT_HASH=ba21c4ec1500837f17a53047da98a7ed7c3a06782ddffe49d8bd796d4d0d1c9e
    readonly SPARSE_MOE_BACKEND=pallas_feature
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_pallas_feature
    ;;
  pallas_feature_linear)
    readonly RUNTIME_TAG=greenfield_runtime_feature_pack_pp8_20260806T064010287072141Z
    readonly RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$RUNTIME_TAG
    readonly RUNTIME_MANIFEST_SHA=54e2f89b1832b994acbf9ef36f5f6ce68c942d9146efc4d7c15360d68b6d9917
    readonly RUNTIME_LAYOUT_HASH=ba21c4ec1500837f17a53047da98a7ed7c3a06782ddffe49d8bd796d4d0d1c9e
    readonly SPARSE_MOE_BACKEND=pallas_feature
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_pallas_feature_linear
    ;;
  *)
    echo "unsupported decoder runtime kind: $RUNTIME_KIND" >&2
    exit 2
    ;;
esac

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TAG=${GLM_GREENFIELD_SHORT_DECODER_TAG:-greenfield_short_decoder_compile_pp8_${RUNTIME_KIND}_trace${TRACE_STEPS}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing decoder compile outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing decoder compile from a dirty worktree" >&2
  exit 2
}
[[ -f $SOURCE_ROOT/SUCCESS && -f $SOURCE_RUNTIME_ROOT/SUCCESS && -f $RUNTIME_ROOT/SUCCESS ]] || {
  echo "protected source/runtime checkpoint is unavailable" >&2
  exit 2
}
[[ -r $RESULTS_DB ]] || {
  echo "results database is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory exists: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/host_logs" "$RUN_DIR/hlo" "$RUN_DIR/traces"

say() {
  echo "[short-decoder-pp8 $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[r]un_real_one_layer[.]py|[l]oad_full_checkpoint_stage[.]py|[p]ack_runtime_checkpoint[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
    say "FAILED status=$status; preserving diagnostics at $RUN_DIR"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN RUNTIME_KIND=$RUNTIME_KIND WARMUP=$WARMUP ITERATIONS=$ITERATIONS TRACE_STEPS=$TRACE_STEPS"
say "RUNTIME=$RUNTIME_MANIFEST_SHA SOURCE_RUNTIME=$SOURCE_RUNTIME_MANIFEST_SHA SOURCE=$SOURCE_MANIFEST_SHA"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact code and runtime/source artifacts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; source_root='"$SOURCE_ROOT"'; source_runtime_root='"$SOURCE_RUNTIME_ROOT"'; runtime_root='"$RUNTIME_ROOT"'; if [[ ${HOSTNAME##*-w-} == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$source_root/SUCCESS" ]] && [[ -r "$source_runtime_root/SUCCESS" ]] && [[ -r "$runtime_root/SUCCESS" ]] && findmnt -T "$source_runtime_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$runtime_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: eight-host sync/artifact prerequisite failed"
  exit 1
}

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I" 2>/dev/null | grep -Eo '192\.168\.[0-9]+\.[0-9]+' | head -1)
[[ -n $coordinator ]] || {
  say "ABORT: worker-0 coordinator address is unavailable"
  exit 1
}
coordinator="$coordinator:8476"
say "launching real 78-layer 2K load/compile coordinator=$coordinator"
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run/hlo"; output="$run/decoder.rank${idx}.json"; log="$run/decoder.rank${idx}.log"; upload() { gcloud storage cp --no-clobber "$log" "$output" "$remote/host_records/" >/dev/null 2>&1 || true; if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/hlo/" >/dev/null 2>&1 || true; fi; xplane=$(find "$run/trace" -type f -name "*.xplane.pb" 2>/dev/null | head -1 || true); if [[ -n $xplane ]]; then gcloud storage cp --no-clobber "$xplane" "$remote/traces/trace.rank${idx}.xplane.pb" >/dev/null 2>&1 || true; fi; }; trap upload EXIT; cd "$wt"; trace_args=(); if [[ '"$TRACE_STEPS"' -gt 0 ]]; then trace_args=(--trace-root "$run/trace" --trace-steps '"$TRACE_STEPS"'); fi; env JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" GLM_GREENFIELD_RUN_TAG="$tag" timeout --signal=TERM --kill-after=60 10800 /home/gianl/vllm-env/bin/python -u scripts/greenfield/compile_short_decoder.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --expected-code-hash '"$PIN"' --runtime-kind '"$RUNTIME_KIND"' --runtime-root '"$RUNTIME_ROOT"' --runtime-manifest-sha256 '"$RUNTIME_MANIFEST_SHA"' --source-runtime-root '"$SOURCE_RUNTIME_ROOT"' --source-runtime-manifest-sha256 '"$SOURCE_RUNTIME_MANIFEST_SHA"' --source-checkpoint-root '"$SOURCE_ROOT"' --source-packed-manifest-sha256 '"$SOURCE_MANIFEST_SHA"' --context-capacity 2048 --warmup '"$WARMUP"' --iterations '"$ITERATIONS"' "${trace_args[@]}" --output "$output" >"$log" 2>&1; trap - EXIT; upload; echo "DECODER_HOST_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/execute.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/execute.txt" DECODER_HOST_OK || {
  say "ABORT: real decoder load/compile did not pass 8/8"
  exit 1
}

gcloud storage cp "$REMOTE_PREFIX/host_records/decoder.rank*.json" \
  "$RUN_DIR/host_records/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/host_records/decoder.rank*.log" \
  "$RUN_DIR/host_logs/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/hlo/" >/dev/null
if [[ $TRACE_STEPS -gt 0 ]]; then
  gcloud storage cp "$REMOTE_PREFIX/traces/trace.rank*.xplane.pb" \
    "$RUN_DIR/traces/" >/dev/null
fi

say "validating fleet agreement and recording diagnostic DB linkage"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$ORACLE_PIN" \
  "$RESULTS_DB" "$WORKTREE" "$ORACLE_REPO" "$RUNTIME_KIND" \
  "$SPARSE_MOE_BACKEND" "$HLO_BACKEND_CONTRACT" "$RUNTIME_MANIFEST_SHA" \
  "$RUNTIME_LAYOUT_HASH" "$WARMUP" "$ITERATIONS" "$TRACE_STEPS" <<'PY'
from __future__ import annotations

import json
import hashlib
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
    runtime_kind,
    sparse_moe_backend,
    hlo_backend_contract,
    runtime_manifest_sha256,
    runtime_layout_hash,
    warmup,
    iterations,
    trace_steps,
) = sys.argv[1:]
warmup = int(warmup)
iterations = int(iterations)
trace_steps = int(trace_steps)
run_dir = Path(run_dir)
records = [json.loads(path.read_text()) for path in sorted((run_dir / "host_records").glob("*.json"))]
if len(records) != 8:
    raise SystemExit(f"expected eight host records, got {len(records)}")
if {record["launch_process_id"] for record in records} != set(range(8)):
    raise SystemExit("launch process ids do not cover 0..7")
if {record["jax_process_index"] for record in records} != set(range(8)):
    raise SystemExit("JAX process indices do not cover 0..7")
if len({record["hostname"] for record in records}) != 8:
    raise SystemExit("host records are not fleet-distinct")
for field in (
    "code_hash",
    "optimized_hlo_sha256",
    "plan_hash",
    "runtime_layout_hash",
    "runtime_manifest_sha256",
    "schedule_hash",
    "state_layout_hash",
    "topology_hash",
):
    values = {record[field] for record in records}
    if len(values) != 1:
        raise SystemExit(f"fleet field {field} disagrees: {sorted(values)}")
if {record["code_hash"] for record in records} != {pin}:
    raise SystemExit("fleet used stale code")
if {record["runtime_kind"] for record in records} != {runtime_kind}:
    raise SystemExit("fleet runtime kind drifted")
if {record["sparse_moe_backend"] for record in records} != {sparse_moe_backend}:
    raise SystemExit("fleet sparse MoE backend drifted")
expected_linear_backend = (
    "pallas" if runtime_kind == "pallas_feature_linear" else "reference"
)
if {record["linear_backend"] for record in records} != {expected_linear_backend}:
    raise SystemExit("fleet FP8 linear backend drifted")
if {record["runtime_manifest_sha256"] for record in records} != {runtime_manifest_sha256}:
    raise SystemExit("fleet runtime manifest drifted")
if {record["runtime_layout_hash"] for record in records} != {runtime_layout_hash}:
    raise SystemExit("fleet runtime layout drifted")
if any(
    not record["body_only"]
    or not record["transformer_body_timing_only"]
    or record["raw_token_claim"]
    or not record["hlo_contract"]["passed"]
    or not record["metadata_passed"]
    for record in records
):
    raise SystemExit("body/HLO/metadata claim contract failed")
if any(
    record["warmup"] != warmup
    or record["iterations"] != iterations
    or record["profiler_free_body_wall"]["count"] != iterations
    for record in records
):
    raise SystemExit("decoder timing configuration drifted")
if any(record["hlo_contract"]["violations"] for record in records):
    raise SystemExit("decoder HLO has violations")
if any(
    record["hlo_contract"]["backend_contract"] != hlo_backend_contract
    for record in records
):
    raise SystemExit("decoder HLO backend contract drifted")
if runtime_kind in ("pallas_feature", "pallas_feature_linear"):
    expected_kernel_counts = {
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512": 75,
        "greenfield_fp8_block_up_gate_m8_k6144_n512": 75,
        "greenfield_fp8_block_matmul_m8_k512_n6144": 75,
    }
    for record in records:
        feature = record["hlo_contract"]["pallas_feature_contract"]
        if (
            not feature["passed"]
            or feature["kernel_counts"] != expected_kernel_counts
            or feature["expected_kernel_counts"] != expected_kernel_counts
            or feature["forbidden_decoded_expert_overlays"]
        ):
            raise SystemExit("feature-Pallas HLO kernel/overlay contract drifted")
if runtime_kind == "pallas_feature_linear":
    expected_linear_kernel_counts = {
        "greenfield_fp8_block_matmul_m8_k6144_n2048": 78,
        "greenfield_fp8_block_matmul_m8_k2048_n4096": 78,
        "greenfield_fp8_block_matmul_m8_k6144_n640": 78,
        "greenfield_fp8_block_matmul_m8_k4096_n6144": 78,
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512": 78,
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256": 78,
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024": 21,
        "greenfield_fp8_block_matmul_f32_m8_k6144_n128": 21,
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144": 3,
    }
    for record in records:
        linear = record["hlo_contract"]["pallas_stage_linear_contract"]
        if (
            not linear["passed"]
            or linear["kernel_counts"] != expected_linear_kernel_counts
            or linear["expected_kernel_counts"]
            != expected_linear_kernel_counts
            or linear["forbidden_decoded_weight_overlays"]
        ):
            raise SystemExit("stage-linear Pallas HLO contract drifted")
if any(record["load_record"]["runtime_checkpoint_reshards"] != 0 for record in records):
    raise SystemExit("runtime loader performed a checkpoint reshard")
if any(record["load_record"]["host_global_concatenations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a host global concat")
if any(record["load_record"]["fp8_host_dequantizations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a host FP8 dequantization")
if any(record["load_record"]["fp8_device_dequantizations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a device FP8 dequantization")
if any(record["load_record"]["loaded_payload_bytes"] != 104_272_169_728 for record in records):
    raise SystemExit("runtime loader payload bytes per host drifted")
xplane = None
if trace_steps:
    traces = sorted((run_dir / "traces").glob("trace.rank*.xplane.pb"))
    if len(traces) != 8:
        raise SystemExit(f"expected eight decoder XPlanes, got {len(traces)}")
    for record, path in zip(records, traces, strict=True):
        trace = record["trace"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            trace is None
            or trace["steps"] != trace_steps
            or not trace["profiler_started_after_profiler_free_timing"]
            or len(trace["files"]) != 1
            or trace["files"][0]["sha256"] != digest
            or trace["files"][0]["size_bytes"] != path.stat().st_size
        ):
            raise SystemExit("decoder trace record drifted")
    sys.path.insert(0, str(Path(repo) / "scripts" / "analysis"))
    import parse_xplane

    xplane = parse_xplane.aggregate_fleet(
        run_dir / "traces",
        step_module_re=r"jit_mapped",
    )
    if (
        xplane["n_files"] != 8
        or xplane["n_cores"] != 64
        or xplane["steps_per_core"] != trace_steps
    ):
        raise SystemExit("decoder fleet XPlane inventory drifted")
else:
    if any(record["trace"] is not None for record in records):
        raise SystemExit("unrequested decoder trace was captured")

def peak(record):
    values = []
    for stats in record["device_memory_after_execute"]:
        if stats is not None:
            for key in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
                if key in stats:
                    values.append(stats[key])
                    break
    return max(values) if values else None

fleet_p50 = max(record["profiler_free_body_wall"]["p50_ms"] for record in records)
fleet_p99 = max(record["profiler_free_body_wall"]["p99_ms"] for record in records)
peaks = [value for value in map(peak, records) if value is not None]
summary = {
    "artifact_kind": "greenfield_real_78layer_2k_decoder_body_fleet",
    "body_only": True,
    "code_hash": pin,
    "compile_seconds_max": max(record["compile_seconds"] for record in records),
    "context_capacity": 2048,
    "fleet_p50_body_ms": fleet_p50,
    "fleet_p99_body_ms": fleet_p99,
    "hlo_contract": records[0]["hlo_contract"],
    "host_count": 8,
    "iterations": iterations,
    "maximum_peak_hbm_bytes": max(peaks) if peaks else None,
    "optimized_hlo_sha256": records[0]["optimized_hlo_sha256"],
    "plan_hash": records[0]["plan_hash"],
    "raw_token_claim": False,
    "runtime_kind": runtime_kind,
    "runtime_layout_hash": records[0]["runtime_layout_hash"],
    "runtime_manifest_sha256": records[0]["runtime_manifest_sha256"],
    "schedule_hash": records[0]["schedule_hash"],
    "state_layout_hash": records[0]["state_layout_hash"],
    "sparse_moe_backend": sparse_moe_backend,
    "topology_hash": records[0]["topology_hash"],
    "trace_steps": trace_steps,
    "transformer_body_timing_only": True,
    "warmup": warmup,
    "xplane": xplane,
}
(run_dir / "xplane_summary.json").write_text(
    json.dumps(xplane, indent=2, sort_keys=True) + "\n"
)
sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-78layer-2k-body",
    revision=records[0]["runtime_manifest_sha256"],
    env={
        "GLM_ENGINE": "greenfield_pp8_decoder_body",
        "greenfield_runtime_kind": runtime_kind,
        "greenfield_sparse_moe_backend": sparse_moe_backend,
        "greenfield_code_hash": pin,
        "legacy_oracle_code_hash": oracle_pin,
        "hlo_sha256": records[0]["optimized_hlo_sha256"],
        "plan_hash": records[0]["plan_hash"],
        "runtime_layout_hash": records[0]["runtime_layout_hash"],
        "runtime_manifest_sha256": records[0]["runtime_manifest_sha256"],
        "state_layout_hash": records[0]["state_layout_hash"],
    },
    note="Protected real 78-layer 2K transformer-body compile/run; no token or tok/s claim.",
    harness_repo=repo,
    fork_repo=oracle_repo,
)
pv.record_item(
    conn,
    run_id,
    benchmark="greenfield_78layer_2k_body_pp8",
    item_id="body_step",
    prompt="Execute the real 78-layer PP8 decoder body at position zero.",
    gold="Local-only HLO, exact pipeline state, direct runtime load, no raw-token claim.",
    raw_output=json.dumps(summary, sort_keys=True),
    extracted=str(summary["hlo_contract"]["collective_counts"]),
    correct=True,
    score=1.0,
    latency_ms=fleet_p50,
)
pv.finalize(
    conn,
    run_id,
    benchmark="greenfield_78layer_2k_body_pp8",
    metric="contract_valid",
    value=1.0,
    note="Profiler-free body timing is not complete decode latency or tok/s.",
)
conn.close()
summary["results_db_run_id"] = run_id
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
if sqlite3.connect(run_dir / "results_ckpt.db").execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("results DB snapshot integrity failed")
print(f"SHORT_DECODER_BODY_VALID db_run={run_id} p50_ms={fleet_p50:.6f}")
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "sealing and archiving protected diagnostic evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find host_records host_logs hlo traces -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum summary.json xplane_summary.json results_ckpt.db census_pre.txt census_post.txt \
    sync.txt execute.txt orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
DB_RUN=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' \
  "$RUN_DIR/summary.json")
printf 'db_run=%s\n' "$DB_RUN" >"$RUN_DIR/SUCCESS"
gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" --no-clobber >/dev/null
trap - EXIT
echo "SHORT_DECODER_BODY_OK $TAG DB=$DB_RUN"
