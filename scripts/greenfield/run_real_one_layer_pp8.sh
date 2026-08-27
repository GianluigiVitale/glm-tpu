#!/usr/bin/env bash
# Protected real layer-3 PP8/PP16 correctness/HLO/HBM/latency/XPlane capture.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly ORACLE_RUN=/home/gianl/gcs-models/oracles/greenfield/glm52/layer3/greenfield_one_layer_oracle_20260805T162210370718434Z
readonly ORACLE_DIR=$ORACLE_RUN
readonly TOPOLOGY_CAPTURE=/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records/topology.rank0.json
readonly ORACLE_MANIFEST_SHA=c63ffa19820d5c2c39865ac8611fb313ffc6ebcd2f893c3507745a356bfdebff
readonly SOURCE_REVISION=gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658
readonly TOPOLOGY_HASH=294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559
readonly PP8_GROUP_HASH=d5943ab8d7a074677d82f8e823c8bc983847f8df1deefdee1fbda8da98923c14
readonly PP16_GROUP_HASH=6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21
readonly PALLAS_PACK_RUN=/home/gianl/glm-run/greenfield_one_layer_pallas_pack_20260806T041854316280053Z
readonly PALLAS_PACK_MANIFEST_SHA=3da63bd9c2332dd67fc29a1d158e5468e0fdf1db1a9a0e8b7977277b0812e427
readonly PALLAS_SOURCE_MANIFEST_SHA=68ef82011892456409a194f6fa31697dd1e31d96fe1a3f0069228288f613f938
readonly PALLAS_PACK_CODE_HASH=9f42e23272503a3735547c371229757d2c841b3a
readonly PALLAS_FEATURE_PACK_RUN=/home/gianl/glm-run/greenfield_one_layer_pallas_feature_pack_20260806T054520020812918Z
readonly PALLAS_FEATURE_PACK_MANIFEST_SHA=a8b914350ea7b8fd281e425d6eb49eefad2082b48f16ec919c26ccbc7bbb5cc6
readonly PALLAS_FEATURE_SOURCE_MANIFEST_SHA=3da63bd9c2332dd67fc29a1d158e5468e0fdf1db1a9a0e8b7977277b0812e427
readonly PALLAS_FEATURE_PACK_CODE_HASH=5f6bb98c9b0036f393db27fc3857969f54aa171d

PLAN_ID=${GLM_GREENFIELD_REAL_LAYER_PLAN:-PP8_LP4}
KERNEL=${GLM_GREENFIELD_REAL_LAYER_KERNEL:-reference}
FEATURE_OUTPUT_TILE=${GLM_GREENFIELD_FEATURE_OUTPUT_TILE:-128}
FEATURE_FUSE_ROUTE_WEIGHTING=${GLM_GREENFIELD_FEATURE_FUSE_ROUTE_WEIGHTING:-0}
FEATURE_RECONSTRUCT_DOWN_FP32=${GLM_GREENFIELD_FEATURE_RECONSTRUCT_DOWN_FP32:-0}
[[ $FEATURE_OUTPUT_TILE == 128 || $FEATURE_OUTPUT_TILE == 256 ]] || {
  echo "feature output tile must be 128 or 256" >&2
  exit 2
}
[[ $FEATURE_FUSE_ROUTE_WEIGHTING == 0 || $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]] || {
  echo "feature route-weight fusion must be 0 or 1" >&2
  exit 2
}
[[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 0 || $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 ]] || {
  echo "feature FP32 reconstruction must be 0 or 1" >&2
  exit 2
}
case "$PLAN_ID" in
  PP8_LP4)
    PLAN_SLUG=pp8
    PACK_RUN=/home/gianl/glm-run/greenfield_one_layer_pack_20260805T151828912346032Z
    PACK_MANIFEST_SHA=68ef82011892456409a194f6fa31697dd1e31d96fe1a3f0069228288f613f938
    PLAN_GROUP_HASH=$PP8_GROUP_HASH
    TPU_BOUNDS=2,2,1
    TPU_VISIBLE=0,1,2,3
    STAGE_ARGS=()
    EXPECTED_TRACE_CORES=8
    ;;
  PP16_LP2)
    PLAN_SLUG=pp16
    PACK_RUN=${GLM_GREENFIELD_PP16_PACK_RUN:-/home/gianl/glm-run/greenfield_one_layer_pack_pp16_20260805T172003732526347Z}
    PACK_MANIFEST_SHA=${GLM_GREENFIELD_PP16_PACK_MANIFEST_SHA:-385737230d593d6b2daa46911d1ac31973d9b79a7d43c3353cedf6d9779fe454}
    PLAN_GROUP_HASH=$PP16_GROUP_HASH
    # libtpu cannot construct a standalone 2x1x1 slice from local devices
    # 0,1 (duplicate coordinate assignment). Initialize the proven 2x2 host
    # subcube and place the executable/arrays only on adjacent devices 0,1.
    TPU_BOUNDS=2,2,1
    TPU_VISIBLE=0,1,2,3
    STAGE_ARGS=(--stage-id 10)
    EXPECTED_TRACE_CORES=4
    ;;
  *)
    echo "unsupported real-layer plan: $PLAN_ID" >&2
    exit 2
    ;;
esac
case "$KERNEL" in
  reference)
    KERNEL_ARGS=(--kernel reference)
    ;;
  pallas)
    [[ $PLAN_ID == PP8_LP4 ]] || {
      echo "Pallas bounded artifact currently supports protected PP8 only" >&2
      exit 2
    }
    PLAN_SLUG=pp8_pallas
    PACK_RUN=$PALLAS_PACK_RUN
    PACK_MANIFEST_SHA=$PALLAS_PACK_MANIFEST_SHA
    KERNEL_ARGS=(
      --kernel pallas
      --source-packed-manifest-sha256 "$PALLAS_SOURCE_MANIFEST_SHA"
      --packed-code-hash "$PALLAS_PACK_CODE_HASH"
    )
    ;;
  pallas_feature)
    if [[ $PLAN_ID == PP8_LP4 ]]; then
      PLAN_SLUG=pp8_pallas_feature
      PACK_RUN=$PALLAS_FEATURE_PACK_RUN
      PACK_MANIFEST_SHA=$PALLAS_FEATURE_PACK_MANIFEST_SHA
      FEATURE_SOURCE_MANIFEST_SHA=$PALLAS_FEATURE_SOURCE_MANIFEST_SHA
      FEATURE_PACK_CODE_HASH=$PALLAS_FEATURE_PACK_CODE_HASH
    else
      PLAN_SLUG=pp16_pallas_feature
      PACK_RUN=${GLM_GREENFIELD_PP16_PALLAS_FEATURE_PACK_RUN:-}
      PACK_MANIFEST_SHA=${GLM_GREENFIELD_PP16_PALLAS_FEATURE_PACK_MANIFEST_SHA:-}
      FEATURE_SOURCE_MANIFEST_SHA=${GLM_GREENFIELD_PP16_PALLAS_FEATURE_SOURCE_MANIFEST_SHA:-}
      FEATURE_PACK_CODE_HASH=${GLM_GREENFIELD_PP16_PALLAS_FEATURE_PACK_CODE_HASH:-}
      [[ -n $PACK_RUN && -n $PACK_MANIFEST_SHA && \
        -n $FEATURE_SOURCE_MANIFEST_SHA && -n $FEATURE_PACK_CODE_HASH ]] || {
        echo "PP16 Pallas feature run requires exact artifact identities" >&2
        exit 2
      }
    fi
    if [[ $FEATURE_OUTPUT_TILE != 128 ]]; then
      PLAN_SLUG=${PLAN_SLUG}_ot${FEATURE_OUTPUT_TILE}
    fi
    if [[ $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]]; then
      PLAN_SLUG=${PLAN_SLUG}_wsum
    fi
    if [[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 ]]; then
      PLAN_SLUG=${PLAN_SLUG}_downf32
    fi
    KERNEL_ARGS=(
      --kernel pallas_feature
      --source-packed-manifest-sha256 "$FEATURE_SOURCE_MANIFEST_SHA"
      --packed-code-hash "$FEATURE_PACK_CODE_HASH"
      --feature-output-tile "$FEATURE_OUTPUT_TILE"
      --feature-fuse-route-weighting "$FEATURE_FUSE_ROUTE_WEIGHTING"
      --feature-reconstruct-down-fp32 "$FEATURE_RECONSTRUCT_DOWN_FP32"
    )
    ;;
  *)
    echo "unsupported real-layer kernel: $KERNEL" >&2
    exit 2
    ;;
esac
if [[ $KERNEL != pallas_feature && $FEATURE_OUTPUT_TILE != 128 ]]; then
  echo "a non-default feature output tile requires pallas_feature" >&2
  exit 2
fi
if [[ $KERNEL != pallas_feature && $FEATURE_FUSE_ROUTE_WEIGHTING != 0 ]]; then
  echo "feature route-weight fusion requires pallas_feature" >&2
  exit 2
fi
if [[ $KERNEL != pallas_feature && $FEATURE_RECONSTRUCT_DOWN_FP32 != 0 ]]; then
  echo "feature FP32 reconstruction requires pallas_feature" >&2
  exit 2
fi
if [[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 && $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]]; then
  echo "feature FP32 reconstruction is incompatible with route-weight fusion" >&2
  exit 2
fi
readonly PLAN_ID PLAN_SLUG PACK_RUN PACK_MANIFEST_SHA PLAN_GROUP_HASH KERNEL
readonly TPU_BOUNDS TPU_VISIBLE EXPECTED_TRACE_CORES FEATURE_OUTPUT_TILE
readonly FEATURE_FUSE_ROUTE_WEIGHTING
readonly FEATURE_RECONSTRUCT_DOWN_FP32
[[ $KERNEL != pallas_feature ]] || readonly FEATURE_SOURCE_MANIFEST_SHA FEATURE_PACK_CODE_HASH

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TAG=${GLM_GREENFIELD_REAL_LAYER_TAG:-greenfield_real_layer_${PLAN_SLUG}_$(date -u +%Y%m%dT%H%M%S%NZ)}
WARMUP=${GLM_GREENFIELD_REAL_LAYER_WARMUP:-200}
ITERATIONS=${GLM_GREENFIELD_REAL_LAYER_ITERATIONS:-1000}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing real-layer run outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing real-layer run from a dirty greenfield worktree" >&2
  exit 2
}
bucket_location=$(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)')
[[ $bucket_location == US-CENTRAL2 ]] || {
  echo "approved result bucket location drifted: $bucket_location" >&2
  exit 2
}
for protected_input in "$ORACLE_RUN" "$TOPOLOGY_CAPTURE"; do
  findmnt -T "$protected_input" -n -o SOURCE,FSTYPE | grep -q '^driftbench-dsv4-uc fuse.gcsfuse$' || {
    echo "protected input is not on the approved read-only bucket: $protected_input" >&2
    exit 2
  }
done
[[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 200 ]] || {
  echo "protected real-layer run requires warmup>=200" >&2
  exit 2
}
[[ $ITERATIONS =~ ^[0-9]+$ && $ITERATIONS -ge 1000 ]] || {
  echo "protected real-layer run requires iterations>=1000" >&2
  exit 2
}
[[ -f $PACK_RUN/SUCCESS && -f $ORACLE_RUN/SUCCESS ]] || {
  echo "successful pack/oracle prerequisites are unavailable" >&2
  exit 2
}
[[ -r $TOPOLOGY_CAPTURE && -r $RESULTS_DB ]] || {
  echo "topology capture or provenance DB is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory already exists: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[real-layer-$PLAN_SLUG $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[t]race_pipeline_transport[.]py|[i]nspect_topology[.]py|[r]un_real_one_layer[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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
say "PACK=$PACK_MANIFEST_SHA ORACLE=$ORACLE_MANIFEST_SHA"
say "KERNEL=$KERNEL"
say "FEATURE_OUTPUT_TILE=$FEATURE_OUTPUT_TILE"
say "FEATURE_FUSE_ROUTE_WEIGHTING=$FEATURE_FUSE_ROUTE_WEIGHTING"
say "FEATURE_RECONSTRUCT_DOWN_FP32=$FEATURE_RECONSTRUCT_DOWN_FP32"
say "SOURCE_REVISION=$SOURCE_REVISION"
say "warmup=$WARMUP iterations=$ITERATIONS trace_steps=20"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "running direct device-load and exact $PLAN_ID layer"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS="$TPU_BOUNDS" \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES="$TPU_VISIBLE" \
    PYTHONPATH="$WORKTREE" \
    /home/gianl/vllm-env/bin/python scripts/greenfield/run_real_one_layer.py \
      --artifact-dir "$PACK_RUN/packed" \
      --oracle-dir "$ORACLE_DIR" \
      --topology-capture "$TOPOLOGY_CAPTURE" \
      --expected-code-hash "$PIN" \
      --packed-manifest-sha256 "$PACK_MANIFEST_SHA" \
      --oracle-manifest-sha256 "$ORACLE_MANIFEST_SHA" \
      --source-revision "$SOURCE_REVISION" \
      --topology-sha256 "$TOPOLOGY_HASH" \
      --plan-group-sha256 "$PLAN_GROUP_HASH" \
      --plan-id "$PLAN_ID" \
      "${KERNEL_ARGS[@]}" \
      "${STAGE_ARGS[@]}" \
      --output "$RUN_DIR/runner.json" \
      --hlo-output "$RUN_DIR/hlo/layer.optimized_hlo.txt" \
      --trace-root "$RUN_DIR/trace" \
      --warmup "$WARMUP" \
      --iterations "$ITERATIONS" \
      --trace-steps 20
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "runner completed in ${elapsed}s"

say "parsing XPlane, linking DB, and building independent summary"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$ORACLE_PIN" "$RESULTS_DB" "$WORKTREE" \
  "$EXPECTED_TRACE_CORES" \
  "$ORACLE_REPO" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, oracle_pin, db_path, repo, expected_trace_cores, oracle_repo = sys.argv[1:]
expected_trace_cores = int(expected_trace_cores)
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
plan_id = runner.get("plan_id")
plan_slugs = {"PP8_LP4": "pp8", "PP16_LP2": "pp16"}
if plan_id not in plan_slugs:
    raise SystemExit(f"unsupported runner plan identity: {plan_id}")
plan_slug = plan_slugs[plan_id]
kernel = runner.get("kernel")
if kernel not in ("reference", "pallas", "pallas_feature"):
    raise SystemExit(f"unsupported runner kernel identity: {kernel}")
feature_output_tile = runner.get("feature_output_tile")
if feature_output_tile not in (128, 256):
    raise SystemExit(f"unsupported feature output tile: {feature_output_tile}")
if kernel != "pallas_feature" and feature_output_tile != 128:
    raise SystemExit("a non-default feature output tile requires pallas_feature")
feature_fuse_route_weighting = runner.get("feature_fuse_route_weighting")
if not isinstance(feature_fuse_route_weighting, bool):
    raise SystemExit("feature route-weight fusion identity is not boolean")
if kernel != "pallas_feature" and feature_fuse_route_weighting:
    raise SystemExit("feature route-weight fusion requires pallas_feature")
feature_reconstruct_down_fp32 = runner.get("feature_reconstruct_down_fp32")
if not isinstance(feature_reconstruct_down_fp32, bool):
    raise SystemExit("feature FP32 reconstruction identity is not boolean")
if kernel != "pallas_feature" and feature_reconstruct_down_fp32:
    raise SystemExit("feature FP32 reconstruction requires pallas_feature")
if feature_reconstruct_down_fp32 and feature_fuse_route_weighting:
    raise SystemExit("feature FP32 reconstruction is incompatible with fusion")
kernel_suffix = "" if kernel == "reference" else f"_{kernel}"
if kernel == "pallas_feature" and feature_output_tile != 128:
    kernel_suffix += f"_ot{feature_output_tile}"
if feature_fuse_route_weighting:
    kernel_suffix += "_wsum"
if feature_reconstruct_down_fp32:
    kernel_suffix += "_downf32"
benchmark = f"greenfield_real_layer_{plan_slug}{kernel_suffix}"
if runner["status"] != "SUCCESS" or runner["code_hash"] != pin:
    raise SystemExit("runner status/code identity failed")
if not runner["hlo"]["contract"]["passed"]:
    raise SystemExit("optimized HLO contract failed")
if not all(case["passed"] for case in runner["correctness"].values()):
    raise SystemExit("normal/concentrated correctness failed")
if not runner["profiler_free_timing"]:
    raise SystemExit("timing was not profiler-free")

sys.path.insert(0, str(Path(repo) / "scripts" / "analysis"))
import parse_xplane

xplane = parse_xplane.aggregate_fleet(
    run_dir / "trace", step_module_re=r"jit_step_fun_impl"
)
if (
    xplane["n_files"] != 1
    or xplane["n_cores"] != expected_trace_cores
    or len(xplane["hosts"]) != 1
    or xplane["steps_per_core"] != 20
):
    raise SystemExit(
        "one-layer XPlane inventory failed: "
        f"files={xplane['n_files']} cores={xplane['n_cores']} "
        f"hosts={xplane['hosts']} steps={xplane['steps_per_core']}"
    )
collective_ops = {
    name: values
    for name, values in xplane["ops"].items()
    if values.get("category") == "collectives"
}
if not collective_ops:
    raise SystemExit("XPlane contains no physical collective events")
for name, values in collective_ops.items():
    identity = f"{name} {values.get('hlo_category', '')}".lower()
    if "all-reduce" not in identity and "all_reduce" not in identity:
        raise SystemExit(f"XPlane contains a non-all-reduce collective: {identity}")
physical_counts = xplane["hlo_all_reduce_invocations_per_step"]
expected_physical_collectives = 2 if feature_reconstruct_down_fp32 else 1
if (
    len(physical_counts) != expected_trace_cores * 20
    or set(physical_counts) != {expected_physical_collectives}
):
    raise SystemExit(
        "XPlane does not contain the exact physical all-reduce count on every "
        f"selected core/step: count={len(physical_counts)} "
        f"values={sorted(set(physical_counts))}"
    )
xplane["physical_collective_ops"] = collective_ops

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-real-layer3",
    revision=runner["source_revision"],
    env={
        "GLM_ENGINE": benchmark,
        "greenfield_code_hash": pin,
        "legacy_oracle_code_hash": oracle_pin,
        "hlo_sha256": runner["hlo"]["sha256"],
        "kernel": kernel,
        "feature_output_tile": feature_output_tile,
        "feature_fuse_route_weighting": feature_fuse_route_weighting,
        "feature_reconstruct_down_fp32": feature_reconstruct_down_fp32,
        "packed_layout_sha256": runner["packed_checkpoint"].get("layout_sha256"),
        "packed_manifest_sha256": runner["packed_checkpoint"]["manifest_sha256"],
        "source_packed_manifest_sha256": runner["packed_checkpoint"].get("source_manifest_sha256"),
        "oracle_manifest_sha256": runner["oracle"]["manifest_sha256"],
        "plan_group_sha256": runner["plan_group_sha256"],
        "topology_sha256": runner["topology_sha256"],
    },
    note=(
        f"Protected exact real layer-3 {plan_id}/{kernel} "
        f"output_tile={feature_output_tile} "
        f"fuse_route_weighting={feature_fuse_route_weighting} "
        f"reconstruct_down_fp32={feature_reconstruct_down_fp32} "
        "normal/concentrated metal proof"
    ),
    harness_repo=repo,
    fork_repo=oracle_repo,
)
for case in ("normal", "concentrated"):
    correctness = runner["correctness"][case]
    timing = runner["timing"][case]
    pv.record_item(
        conn,
        run_id,
        benchmark=benchmark,
        item_id=case,
        prompt=(
            "Execute one real batch-one GLM-5.2 layer-3 MoE on one "
            f"{plan_id} stage with the {kernel} kernel path and routed "
            f"output tile {feature_output_tile}; fused route weighting "
            f"is {feature_fuse_route_weighting}; FP32 reconstruction is "
            f"{feature_reconstruct_down_fp32}."
        ),
        gold=(
            "Exact routes, bounded BF16 output, and only the declared local "
            "reconstruction/combine all-reduces."
        ),
        raw_output=json.dumps(
            {"correctness": correctness, "timing": timing["latency"]},
            sort_keys=True,
        ),
        extracted=str(correctness["observed_route_indices"]),
        correct=correctness["passed"],
        score=1.0 if correctness["passed"] else 0.0,
        latency_ms=timing["latency"]["p50_ms"],
    )
pv.finalize(
    conn,
    run_id,
    benchmark=benchmark,
    metric="contract_valid",
    value=1.0,
    note="Profiler-free p50 is per one real sparse layer, not token throughput.",
)
conn.close()
summary = {
    "code_hash": pin,
    "elapsed_seconds": None,
    "oracle_code_hash": oracle_pin,
    "results_db_run_id": run_id,
    "runner": runner,
    "xplane": xplane,
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
print(f"REAL_LAYER_{plan_slug.upper()}_PROOF_VALID db_run={run_id}")
PY

/home/gianl/vllm-env/bin/python - "$RUN_DIR/summary.json" "$elapsed" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
value = json.loads(path.read_text())
value["elapsed_seconds"] = int(sys.argv[2])
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

(
  cd "$RUN_DIR"
  find hlo trace -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log summary.json results_ckpt.db \
    census_pre.txt
) >"$RUN_DIR/evidence.sha256"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1
sha256sum "$RUN_DIR/census_post.txt" >>"$RUN_DIR/evidence.sha256"

touch "$RUN_DIR/SUCCESS"
gcloud storage cp --recursive --no-clobber "$RUN_DIR/hlo" "$RUN_DIR/trace" \
  "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber \
  "$RUN_DIR/runner.json" "$RUN_DIR/runner.log" "$RUN_DIR/summary.json" \
  "$RUN_DIR/results_ckpt.db" "$RUN_DIR/evidence.sha256" \
  "$RUN_DIR/orchestrator.log" "$RUN_DIR/census_pre.txt" \
  "$RUN_DIR/census_post.txt" "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
remote_success=$(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true)
[[ $remote_success == "$REMOTE_PREFIX/SUCCESS" ]] || {
  say "ABORT: remote SUCCESS marker did not verify"
  exit 1
}
say "SUCCESS DB=$(/home/gianl/vllm-env/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' "$RUN_DIR/summary.json")"
say "ARCHIVE=$REMOTE_PREFIX"
trap - EXIT
