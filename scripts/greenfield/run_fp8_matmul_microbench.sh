#!/usr/bin/env bash
# Protected production-shaped TPU-v4 proof for greenfield FP8 projection kernels.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
KERNEL=${GLM_GREENFIELD_FP8_MATMUL_KERNEL:-single_up}
WINDOW_ACQUISITION=0
[[ $KERNEL != ws32_prefill_layer_window_acquisition ]] || WINDOW_ACQUISITION=1
[[ $KERNEL != ws32_prefill_window_boundary_acquisition ]] || WINDOW_ACQUISITION=1
[[ $KERNEL != ws32_prefill_completed_window_acquisition ]] || WINDOW_ACQUISITION=1
WINDOW_NUMERICAL=0
[[ $KERNEL != ws32_prefill_layer_window_numerical ]] || WINDOW_NUMERICAL=1
# Same bounded execution family; diagnostic remains non-admission in DB accounting.
[[ $KERNEL != ws32_prefill_window_boundary_diagnostic ]] || WINDOW_NUMERICAL=1
[[ $KERNEL != ws32_prefill_completed_window_numerical ]] || WINDOW_NUMERICAL=1
GROUPED_ADMISSION=0
[[ $KERNEL != ws32_grouped_admission && $KERNEL != ws32_grouped_down_admission && \
   $KERNEL != ws32_prefill_moe_admission && $KERNEL != ws32_prefill_moe_boundary_diagnostic && \
   $KERNEL != ws32_prefill_moe_bounded_admission && $KERNEL != ws32_prefill_layer_admission && \
   $KERNEL != ws32_prefill_router_boundary_diagnostic && $KERNEL != ws32_prefill_prefix_mlp_diagnostic && \
   $KERNEL != ws32_prefill_layer_observed_admission && $KERNEL != ws32_prefill_layer_materialized_admission ]] || GROUPED_ADMISSION=1
BOUNDED_PREFILL=0
[[ $WINDOW_ACQUISITION == 0 && $WINDOW_NUMERICAL == 0 ]] || GROUPED_ADMISSION=1
[[ $KERNEL != ws32_prefill_baseline && $KERNEL != ws32_prefill_moe_scaling_baseline && $GROUPED_ADMISSION != 1 ]] || BOUNDED_PREFILL=1
OUTPUT_TILE=${GLM_GREENFIELD_FP8_OUTPUT_TILE:-128}
SELECTED_CASE=${GLM_GREENFIELD_FP8_SELECTED_CASE:-concentrated_eight}
TAG_STEM=$KERNEL
LAYER=${GLM_GREENFIELD_PREFILL_LAYER:-0}
if [[ $WINDOW_ACQUISITION == 1 || $WINDOW_NUMERICAL == 1 ]]; then
  [[ $LAYER == 6 ]] || exit 2
  TAG_STEM=${KERNEL}_l6
fi
if [[ $KERNEL == ws32_prefill_layer_admission || $KERNEL == ws32_prefill_prefix_mlp_diagnostic || $KERNEL == ws32_prefill_router_boundary_diagnostic || $KERNEL == ws32_prefill_layer_materialized_admission || $KERNEL == ws32_prefill_layer_observed_admission ]]; then
  [[ $LAYER == 0 || $LAYER == 3 ]] || exit 2
  [[ $KERNEL != ws32_prefill_router_boundary_diagnostic || $LAYER == 3 ]] || exit 2
  [[ $KERNEL != ws32_prefill_prefix_mlp_diagnostic || $LAYER == 3 ]] || exit 2
  [[ $KERNEL != ws32_prefill_layer_materialized_admission || $LAYER == 3 ]] || exit 2
  [[ $KERNEL != ws32_prefill_layer_observed_admission || $LAYER == 3 ]] || exit 2
  TAG_STEM=${KERNEL}_l${LAYER}
fi
BASELINE_ROWS=${GLM_GREENFIELD_FP8_BASELINE_ROWS:-8}
[[ $KERNEL != ws32_prefill_baseline ]] || TAG_STEM=${KERNEL}_m${BASELINE_ROWS}
[[ $KERNEL != selected_up_gate && $KERNEL != selected_swiglu_down ]] || \
  TAG_STEM=${KERNEL}_${SELECTED_CASE}
[[ $OUTPUT_TILE == 128 ]] || TAG_STEM=${TAG_STEM}_ot${OUTPUT_TILE}
TAG=${GLM_GREENFIELD_FP8_MATMUL_TAG:-greenfield_fp8_${TAG_STEM}_$(date -u +%Y%m%dT%H%M%S%NZ)}
DEFAULT_WARMUP=200
DEFAULT_ITERATIONS=1000
if [[ $GROUPED_ADMISSION == 1 ]]; then
  DEFAULT_WARMUP=0
  DEFAULT_ITERATIONS=0
elif [[ $KERNEL == ws32_prefill_moe_scaling_baseline ]]; then
  DEFAULT_WARMUP=10
  DEFAULT_ITERATIONS=50
fi
WARMUP=${GLM_GREENFIELD_FP8_MATMUL_WARMUP:-$DEFAULT_WARMUP}
ITERATIONS=${GLM_GREENFIELD_FP8_MATMUL_ITERATIONS:-$DEFAULT_ITERATIONS}
DIAGNOSTIC_REFERENCE=${GLM_GREENFIELD_FP8_DIAGNOSTIC_REFERENCE:-0}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
[[ $TAG =~ ^greenfield_fp8_[a-zA-Z0-9_]+$ ]] || {
  echo "unsafe FP8 run tag" >&2
  exit 2
}
if [[ $KERNEL == ws32_prefill_baseline ]]; then
  [[ $BASELINE_ROWS == 8 || $BASELINE_ROWS == 32 || \
     $BASELINE_ROWS == 128 || $BASELINE_ROWS == 256 ]] || exit 2
  [[ $WARMUP == 200 && $ITERATIONS == 1000 && $DIAGNOSTIC_REFERENCE == 0 ]] || exit 2
fi

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing FP8 kernel run outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing FP8 kernel run from a dirty worktree" >&2
  exit 2
}
[[ $KERNEL == single_up || $KERNEL == single_up_m1 || \
  $KERNEL == rmsnorm_linear || $KERNEL == up_gate || \
  $KERNEL == attention_output || $KERNEL == fused_attention_output || \
  $KERNEL == selected_up_gate || $KERNEL == selected_swiglu_down || \
  $KERNEL == structured_kv_b || $KERNEL == dsa_wq_b || \
  $KERNEL == dsa_wk || $KERNEL == ws32_prefill_baseline || \
  $KERNEL == ws32_grouped_admission || $KERNEL == ws32_grouped_down_admission || \
  $KERNEL == ws32_prefill_moe_admission || $KERNEL == ws32_prefill_moe_boundary_diagnostic || \
  $KERNEL == ws32_prefill_moe_bounded_admission || $KERNEL == ws32_prefill_layer_admission || \
  $KERNEL == ws32_prefill_moe_scaling_baseline || $WINDOW_ACQUISITION == 1 || $WINDOW_NUMERICAL == 1 || \
  $KERNEL == ws32_prefill_router_boundary_diagnostic || $KERNEL == ws32_prefill_prefix_mlp_diagnostic || $KERNEL == ws32_prefill_layer_materialized_admission || $KERNEL == ws32_prefill_layer_observed_admission ]] || {
  echo "FP8 kernel must be single_up, single_up_m1, attention_output," \
    "fused_attention_output, rmsnorm_linear, up_gate, selected_up_gate," \
    "selected_swiglu_down, structured_kv_b, dsa_wq_b, dsa_wk, ws32_prefill_baseline, ws32_grouped_admission, ws32_grouped_down_admission, ws32_prefill_moe_admission, ws32_prefill_moe_boundary_diagnostic, ws32_prefill_moe_bounded_admission, ws32_prefill_layer_admission, ws32_prefill_router_boundary_diagnostic, ws32_prefill_prefix_mlp_diagnostic, ws32_prefill_layer_observed_admission, or ws32_prefill_layer_materialized_admission" >&2
  exit 2
}
[[ $OUTPUT_TILE == 128 || $OUTPUT_TILE == 256 ]] || {
  echo "FP8 output tile must be 128 or 256" >&2
  exit 2
}
[[ $KERNEL == attention_output || $OUTPUT_TILE == 128 ]] || {
  echo "a non-default output tile requires attention_output" >&2
  exit 2
}
[[ $SELECTED_CASE == normal_two || $SELECTED_CASE == concentrated_eight ]] || {
  echo "selected route case must be normal_two or concentrated_eight" >&2
  exit 2
}
if [[ $GROUPED_ADMISSION == 1 ]]; then
  [[ $WARMUP == 0 && $ITERATIONS == 0 && $DIAGNOSTIC_REFERENCE == 0 ]] || exit 2
elif [[ $KERNEL == ws32_prefill_moe_scaling_baseline ]]; then
  [[ $WARMUP == 10 && $ITERATIONS == 50 && $DIAGNOSTIC_REFERENCE == 0 ]] || exit 2
elif [[ $DIAGNOSTIC_REFERENCE == 1 ]]; then
  [[ $KERNEL == single_up_m1 ]] || {
    echo "reference diagnostic requires single_up_m1" >&2
    exit 2
  }
  [[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 1 ]] || exit 2
  [[ $ITERATIONS =~ ^[0-9]+$ && $ITERATIONS -ge 3 ]] || exit 2
else
  [[ $DIAGNOSTIC_REFERENCE == 0 && $KERNEL != single_up_m1 ]] || exit 2
  [[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 200 ]] || exit 2
  [[ $ITERATIONS =~ ^[0-9]+$ && $ITERATIONS -ge 1000 ]] || exit 2
fi
[[ -r $RESULTS_DB && ! -e $RUN_DIR ]] || {
  echo "results DB missing or append-only run path already exists" >&2
  exit 2
}
if [[ $WINDOW_ACQUISITION == 1 || $WINDOW_NUMERICAL == 1 ]]; then
  [[ $TAG == greenfield_fp8_${KERNEL}_l6_* ]] || exit 2
  # Selected-layer graphs/journals and bounded numerical NPZs; no checkpoint copy.
  # This bounded floor is not the full-model4GiB admission floor.
  JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - <<'PY'
import shutil
if shutil.disk_usage('/home/gianl/glm-run').free < 1024**3:
    raise SystemExit('window layer test requires1GiB controller evidence space')
PY
fi
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[fp8-matmul $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

[[ $(git -C "$WORKTREE" rev-parse HEAD) == "$PIN" && \
   -z $(git -C "$WORKTREE" status --porcelain) ]] || exit 2
[[ $(git -C "$WORKTREE" ls-remote origin \
  refs/heads/rewrite/topology-first-decode | awk '{print $1}') == "$PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" \
  --format='value(location)') == "$APPROVED_LOCATION" ]]
if [[ $BOUNDED_PREFILL == 1 ]]; then
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - "$TAG" <<'PY'
from google.cloud import storage
import sys
client = storage.Client()
if next(iter(client.list_blobs('driftbench-dsv4-uc', prefix=f'results/{sys.argv[1]}/', max_results=1)), None):
    raise SystemExit('baseline remote prefix already exists')
PY
fi

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  if [[ $BOUNDED_PREFILL == 1 ]]; then
    local idle_command
    idle_command=$(PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
      -m scripts.greenfield.fp8_baseline_guard census-command) || return 1
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command="$idle_command" >"$RUN_DIR/devices_${label}.txt" 2>&1 || return 1
    PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
      -m scripts.greenfield.fp8_baseline_guard validate-fleet \
      --file "$RUN_DIR/devices_${label}.txt" || return 1
  fi
  local ray_enum
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[t]race_pipeline_transport[.]py|[r]un_real_one_layer[.]py|[m]icrobench_fp8_matmul[.]py|[m]icrobench_structured_kv_b[.]py|[m]icrobench_fused_attention_output[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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

say "RUN_DIR=$RUN_DIR PIN=$PIN kernel=$KERNEL output_tile=$OUTPUT_TILE selected_case=$SELECTED_CASE warmup=$WARMUP iterations=$ITERATIONS"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "compiling/checking GLM $KERNEL on TPU v4 (grouped admission has no timing samples)"
started=$(date +%s)
ROWS=8
[[ $KERNEL != rmsnorm_linear && $KERNEL != single_up_m1 ]] || ROWS=1
CONTRACTION=6144
OUTPUT_WIDTH=2048
if [[ $KERNEL == ws32_prefill_baseline ]]; then
  ROWS=$BASELINE_ROWS
  CONTRACTION=1536
elif [[ $KERNEL == dsa_wq_b ]]; then
  ROWS=1
  CONTRACTION=2048
  OUTPUT_WIDTH=1024
elif [[ $KERNEL == dsa_wk ]]; then
  ROWS=1
  OUTPUT_WIDTH=128
elif [[ $KERNEL == attention_output ]]; then
  ROWS=1
  CONTRACTION=4096
  OUTPUT_WIDTH=6144
fi
(
  cd "$WORKTREE"
  if [[ $WINDOW_ACQUISITION == 1 || $WINDOW_NUMERICAL == 1 || $KERNEL == ws32_prefill_layer_admission || $KERNEL == ws32_prefill_prefix_mlp_diagnostic || $KERNEL == ws32_prefill_router_boundary_diagnostic || $KERNEL == ws32_prefill_layer_materialized_admission || $KERNEL == ws32_prefill_layer_observed_admission ]]; then
    [[ $TAG == greenfield_fp8_${KERNEL}_l${LAYER}_* ]] || exit 2
    JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
      -m scripts.greenfield.ws32_prefill_layer_campaign campaign --tag "$TAG" --pin "$PIN"
    exit 0
  fi
  if [[ $KERNEL == ws32_prefill_moe_admission || $KERNEL == ws32_prefill_moe_boundary_diagnostic || $KERNEL == ws32_prefill_moe_bounded_admission || $KERNEL == ws32_prefill_moe_scaling_baseline ]]; then
    # Distributed worker runtime must not inherit the single-process bounds below.
    JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
      -m scripts.greenfield.ws32_prefill_moe_campaign campaign --tag "$TAG" --pin "$PIN"
    exit 0
  fi
  if [[ $GROUPED_ADMISSION == 1 ]]; then
    DIRECTION=up
    [[ $KERNEL != ws32_grouped_down_admission ]] || DIRECTION=down
    RUNNER=(
      scripts/greenfield/probe_prefill_grouped_fp8.py
      --direction "$DIRECTION"
      --expected-code-hash "$PIN"
      --output "$RUN_DIR/runner.json"
      --hlo-output "$RUN_DIR/hlo/grouped_fp8.optimized_hlo.txt"
    )
  elif [[ $KERNEL == fused_attention_output ]]; then
    RUNNER=(
      scripts/greenfield/microbench_fused_attention_output.py
      --expected-code-hash "$PIN"
      --output "$RUN_DIR/runner.json"
      --hlo-dir "$RUN_DIR/hlo"
      --warmup "$WARMUP"
      --iterations "$ITERATIONS"
    )
  elif [[ $KERNEL == structured_kv_b ]]; then
    RUNNER=(
      scripts/greenfield/microbench_structured_kv_b.py
      --expected-code-hash "$PIN"
      --output "$RUN_DIR/runner.json"
      --hlo-dir "$RUN_DIR/hlo"
      --warmup "$WARMUP"
      --iterations "$ITERATIONS"
    )
  else
    RUNNER=(
      scripts/greenfield/microbench_fp8_matmul.py
      --expected-code-hash "$PIN"
      --output "$RUN_DIR/runner.json"
      --hlo-output "$RUN_DIR/hlo/fp8_matmul.optimized_hlo.txt"
      --kernel "$KERNEL"
      --output-tile "$OUTPUT_TILE"
      --rows "$ROWS"
      --contraction "$CONTRACTION"
      --output-width "$OUTPUT_WIDTH"
      --selected-route-case "$SELECTED_CASE"
      --warmup "$WARMUP"
      --iterations "$ITERATIONS"
    )
  fi
  if [[ $DIAGNOSTIC_REFERENCE == 1 ]]; then
    RUNNER+=(--diagnostic-reference-timing)
  fi
  PYTHON_RUN=(/home/gianl/vllm-env/bin/python)
  if [[ $BOUNDED_PREFILL == 1 ]]; then
    # Bound this single-process mechanism probe, not an hours-long model run.
    PYTHON_RUN=(timeout --kill-after=30s 600s /home/gianl/vllm-env/bin/python)
  fi
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    "${PYTHON_RUN[@]}" "${RUNNER[@]}"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "runner completed in ${elapsed}s"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" "$KERNEL" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, db_path, repo, elapsed, expected_kernel = sys.argv[1:]
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
if runner["status"] != "SUCCESS" or runner["code_hash"] != pin:
    raise SystemExit("runner status/code identity failed")
if runner["kernel"] != expected_kernel:
    raise SystemExit("runner kernel does not match launched kernel")
admission = runner.get("admission_only", False)
if admission != (expected_kernel in ("ws32_grouped_admission", "ws32_grouped_down_admission", "ws32_prefill_moe_admission", "ws32_prefill_moe_bounded_admission", "ws32_prefill_layer_admission", "ws32_prefill_layer_materialized_admission", "ws32_prefill_layer_observed_admission", "ws32_prefill_layer_window_numerical", "ws32_prefill_completed_window_numerical")):
    raise SystemExit("admission classification drifted")
boundary = expected_kernel == "ws32_prefill_moe_boundary_diagnostic"
prefix_mlp = expected_kernel == "ws32_prefill_prefix_mlp_diagnostic"
router_boundary = expected_kernel == "ws32_prefill_router_boundary_diagnostic" or prefix_mlp
observed = expected_kernel == "ws32_prefill_layer_observed_admission"
materialized = expected_kernel == "ws32_prefill_layer_materialized_admission" or observed
window_boundary = expected_kernel == "ws32_prefill_window_boundary_acquisition"
completed_window = expected_kernel == "ws32_prefill_completed_window_acquisition"
window_diagnostic = expected_kernel == "ws32_prefill_window_boundary_diagnostic"
window_acquisition = expected_kernel == "ws32_prefill_layer_window_acquisition" or window_boundary or completed_window
window_acquisition_note = (
    "Layer6 completed B32 prefix, B128/B32 suffixes and two WK programs compiled only; five graphs, zero executable calls. Shared prefix is not independent full-layer proof; exact HLO/numerical memory admission pending."
    if completed_window else
    "Layer6 B128/B32 plus WK decode/promote compiled only; exact HLO and numerical memory admission remain pending. No model or WK execution, correctness or performance claim."
)
window_numerical = expected_kernel == "ws32_prefill_layer_window_numerical"
completed_numerical = expected_kernel == "ws32_prefill_completed_window_numerical"
completed_numerical_note = "Real layer6 completed B32 prefixes with B128 versus four B32 MLP suffixes; 27 model, 2 WK and 30 assembly calls. Synthetic history; shared-prefix DSA/cache agreement is by construction, NOT independent full-layer DSA or original-failure repair. No full-model or performance claim."
window_numerical_note = "Real layer6 B128 versus four completed B32 controls; synthetic history, original per-row/cache bounds and ordered routes; no independent full-score-row DSA, full-model or performance claim."
diagnostic_boundary = boundary or router_boundary or window_acquisition or window_diagnostic
bounded = expected_kernel == "ws32_prefill_moe_bounded_admission"
scaling = expected_kernel == "ws32_prefill_moe_scaling_baseline"
fleet_moe = expected_kernel == "ws32_prefill_moe_admission" or boundary or bounded or scaling
fleet_layer = expected_kernel == "ws32_prefill_layer_admission" or router_boundary or materialized or window_acquisition or window_numerical or window_diagnostic or completed_numerical
untimed = admission or diagnostic_boundary
if fleet_layer:
    from scripts.greenfield.ws32_prefill_layer_campaign import validate_record
    validate_record(runner, pin, diagnostic=router_boundary, materialized=materialized, prefix_mlp=prefix_mlp, observed=observed)
elif scaling:
    from scripts.greenfield.prefill_moe_scaling_evidence import validate_record
    validate_record(runner, pin)
elif fleet_moe:
    from scripts.greenfield.ws32_prefill_moe_campaign import validate_record
    validate_record(runner, pin, boundary=boundary, bounded=bounded)
elif admission:
    from scripts.greenfield.probe_prefill_grouped_fp8 import validate_record
    validate_record(runner)
output_tile = runner.get("output_tile", 128)
if output_tile not in (128, 256):
    raise SystemExit("runner output tile is invalid")
if runner["kernel"] != "attention_output" and output_tile != 128:
    raise SystemExit("non-default output tile requires attention_output")
if not runner["hlo"]["contract"]["passed"]:
    raise SystemExit("Pallas custom-call/full-overlay HLO contract failed")
if not diagnostic_boundary and not runner["comparison"]["passed"]:
    raise SystemExit("Pallas/reference correctness failed")
if not untimed and not runner["profiler_free_timing"]:
    raise SystemExit("kernel wall distribution is not profiler-free")
baseline = runner.get("baseline_only", False)
if baseline != (runner["kernel"] == "ws32_prefill_baseline" or scaling):
    raise SystemExit("baseline classification drifted")
if baseline and not scaling:
    from scripts.greenfield.microbench_fp8_matmul import _validate_shape_contract, _validate_sampling_contract
    m, k = runner['shape']['lhs']
    n = runner['shape']['output'][1]
    _validate_shape_contract(kernel=runner['kernel'], rows=m, contraction=k, output_width=n)
    _validate_sampling_contract(kernel=runner['kernel'], diagnostic_reference_timing=runner['diagnostic_only'],
                                warmup=runner['warmup'], iterations=runner['iterations'])
    if runner['dtype_contract']['output'] != 'float32' or not runner['compiled_memory_estimate']:
        raise SystemExit('baseline dtype/memory evidence missing')
diagnostic_reference = runner.get("reference_diagnostic")
if not diagnostic_boundary and bool(diagnostic_reference) != runner.get("diagnostic_only"):
    raise SystemExit("reference diagnostic identity drifted")
if diagnostic_reference is not None:
    if (
        runner["kernel"] != "single_up_m1"
        or not diagnostic_reference["hlo"]["contract"]["passed"]
    ):
        raise SystemExit("reference diagnostic HLO contract failed")

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model=f"zai-org/GLM-5.2-FP8:greenfield-fp8-{runner['kernel']}-kernel",
    revision="runtime-u8-e4m3fn-block128",
    env={
        "GLM_ENGINE": "greenfield_fp8_matmul",
        "greenfield_code_hash": pin,
        "hlo_sha256": runner["hlo"]["sha256"],
        "device_kind": runner["device_kind"],
        "kernel": runner["kernel"],
        "output_tile": output_tile,
        "selected_route_case": runner["selected_route_case"],
    },
    note=(
        "Protected production-shaped Pallas FP8 projection microbenchmark: "
        + runner["kernel"]
    ),
    harness_repo=repo,
    fork_repo=None,
)
shape_ids = {
    "attention_output": "m1_k4096_n6144",
    "fused_attention_output": "h16_l512_v256_o6144",
    "dsa_wq_b": "m1_k2048_n1024",
    "dsa_wk": "m1_k6144_n128",
    "single_up_m1": "m1_k6144_n2048",
}
if completed_numerical:
    item_id = "layer6_completed_prefix_b128_four_b32_suffix_numerical_59calls_v1"
elif window_diagnostic:
    item_id = "layer6_b128_b32_original_boundary_capture_diagnostic_v1"
elif completed_window:
    item_id = "layer6_completed_b32_prefix_b128_b32_suffix_five_graph_compile_only_v1"
elif window_acquisition:
    item_id = "layer6_b128_b32_actual_boundary_four_graph_compile_only_v1" if window_boundary else "layer6_b128_b32_cap4096_four_graph_compile_only_v1"
elif window_numerical:
    item_id = "layer6_b128_four_b32_boundary_competitive_tail_numerical_v1"
elif router_boundary:
    item_id = "layer3_b17_db585_prefix_completed_mlp_diagnostic_v1" if prefix_mlp else "layer3_b17_router_prefix_same_input_diagnostic_v1"
elif fleet_layer:
    item_id = "complete_layer3_b17_db585_observed_reference_empty_boundary_tail_v3" if observed else ("complete_layer3_b17_materialized_bf16_reference_empty_boundary_tail_v2" if materialized else f"complete_layer{runner['layer']}_b17_raw_reference_empty_boundary_tail_v1")
elif scaling:
    item_id = "real_layer3_equal128_b16_b128_phase_baseline_v1_normal_concentrated"
elif fleet_moe:
    item_id = ("real_layer3_b17_boundaries_v1_normal" if boundary else
               "real_layer3_b17_fp32_route_sum_bounded_v1_normal_concentrated" if bounded else
               "real_layer3_b17_arithmetic_v1_normal_concentrated")
elif admission:
    k = runner['shape']['lhs'][1]
    n = runner['shape']['output'][1]
    item_id = f"arithmetic_v1_m136_k{k}_n{n}_g32_distributed_concentrated_empty"
elif baseline:
    item_id = f"baseline_m{m}_k{k}_n{n}"
elif runner["kernel"] == "structured_kv_b":
    item_id = "h16_p192_l512_v256"
elif runner["kernel"] == "selected_swiglu_down":
    item_id = "m8_k2048_n6144"
else:
    item_id = shape_ids.get(
        runner["kernel"],
        (
            "m1_k6144_n2048"
            if runner["kernel"] == "rmsnorm_linear"
            else "m8_k6144_n2048"
        ),
    )
if runner["selected_route_case"] is not None:
    item_id += "_" + runner["selected_route_case"]
if output_tile != 128:
    item_id += f"_ot{output_tile}"
pv.record_item(
    conn,
    run_id,
    benchmark=f"greenfield_fp8_{runner['kernel']}",
    item_id=item_id,
    prompt=(
        "Raw-U8 E4M3FN 128x128 block-scaled expert projection: "
        + runner["kernel"]
    ),
    gold=completed_numerical_note if completed_numerical else "Original boundary outputs and operands; reproduction or instrumentation perturbation, no numerical or performance admission." if window_diagnostic else window_acquisition_note if window_acquisition else "B128 versus four completed B32 controls; fixed per-row/cache bounds, exact routes and own selected-order/ties; not full-model proof." if window_numerical else "Bounded exact-fallback output and required compact Pallas calls.",
    raw_output=json.dumps(runner, sort_keys=True),
    extracted=str(runner["checksum"]),
    correct=None if diagnostic_boundary else True,
    score=None if diagnostic_boundary else 1.0,
    # Two geometries/two scenarios: never collapse to one ambiguous DB latency.
    latency_ms=None if untimed or scaling else runner["latency"]["p50_ms"],
)
pv.finalize(
    conn,
    run_id,
    benchmark=f"greenfield_fp8_{runner['kernel']}",
    metric="diagnostic_evidence_complete" if diagnostic_boundary else "contract_valid",
    value=1.0,
    note=completed_numerical_note if completed_numerical else "Original boundary outputs and operands; reproduction or instrumentation perturbation, no numerical or performance admission." if window_diagnostic else window_numerical_note if window_numerical else window_acquisition_note if window_acquisition else "Complete layer3 with actual13-output PREnorm and DB585 prefix reproduction; completed MLP, unchanged numerical bounds/interventions; not legacy or model/performance proof." if observed else "DB585 complete scalar prefix fingerprint reproduction followed by completed-input MLP; no full-layer numerical admission or performance claim." if prefix_mlp else "Router prefix reproduction and identical-input arithmetic diagnostic; no numerical admission or performance claim." if router_boundary else "Complete layer3 with completed BF16 scalar-reference MLP input (v2), NOT v1 fused-reference identity; unchanged numerical bounds and exact route/cache interventions; no model/performance claim." if materialized else "Complete batched layer with real weights/synthetic state; bounded raw scalar reference, causal/cache interventions; no legacy/full-model or performance claim." if fleet_layer else "Instrumented real-MoE boundary evidence; no numerical acceptance or performance claim." if boundary else "Standalone kernel microbenchmark; not layer latency or token throughput.",
)
conn.close()

summary = {
    "status": "SUCCESS",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
    "runner": runner,
    "claim_scope": (
        completed_numerical_note if completed_numerical else
        window_acquisition_note if completed_window else
        "Layer6 original-boundary B128/B32 operand captures and original-fingerprint reproduction or perturbation; two WK/five model calls; no numerical admission, cause attribution or performance claim"
        if window_diagnostic else
        "Layer6 instrumented B128/B32 plus WK compiler evidence and owner-output schema; no model or WK execution, original-signature reproduction, numerical admission or performance claim"
        if window_boundary else
        "32-chip real layer6 B128 versus four completed B32 controls; synthetic boundary/competitive/tail history, unchanged per-row bounds and exact routes/cache structure; selected-order/control agreement is not independent full-score-row DSA or full-model/performance proof"
        if window_numerical else
        "Layer6 B128/B32/WK original compiler evidence; no model or WK execution, numerical admission, exact HLO profile or performance claim"
        if window_acquisition else
        "32-chip real layer3 MoE equal128 B16/B128 summed completed-call phase baselines; supplied routes, no full-prefill/TTFT or sustained-throughput claim"
        if scaling else
        "32-chip complete layer3 with actual13-output PREnorm, DB585 prefix reproduction and completed MLP; unchanged bounded cases/interventions, not legacy/full-model/performance proof"
        if observed else
        "32-chip DB585 scalar prefix and completed-input MLP diagnostic; no full-layer numerical admission or performance claim"
        if prefix_mlp else
        "32-chip layer3 router prefix and same-input diagnostic; original ordered-route reproduction only, no original-logit capture, numerical admission or performance claim"
        if router_boundary else
        "32-chip complete layer3; distinct completed-BF16 scalar reference v2, not v1 fused-reference identity; original bounds unchanged; no model/legacy/performance claim"
        if materialized else
        "32-chip complete batched layer, real weights/synthetic state; bounded raw scalar reference, exact cache structure/causal interventions; no legacy/full-model or performance claim"
        if fleet_layer else
        "32-chip real layer3 boundary diagnostic; instrumentation may perturb outputs; no arithmetic acceptance or performance claim"
        if boundary else "32-chip real layer3 FP32 route sum, per-row/aggregate bounded M1 and direct row0 legacy comparison; not bit-exact, no timing or model-performance claim"
        if bounded else "32-chip real layer3 MoE, supplied routes/perturbed activations, exact M1 comparison; no timing or model-performance claim"
        if fleet_moe else "single-chip synthetic grouped projection arithmetic admission; no timing or performance claim"
        if admission else "diagnostic reference-vs-Pallas projection wall; no performance claim"
        if diagnostic_reference is not None
        else "standalone profiler-free kernel wall; no token-rate claim"
    ),
    "performance_claim": False,
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
print(f"FP8_MATMUL_PROOF_VALID db_run={run_id}")
PY

(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log summary.json results_ckpt.db census_pre.txt
) >"$RUN_DIR/evidence.sha256"

sha256sum "$RUN_DIR/census_post.txt" >>"$RUN_DIR/evidence.sha256"
if [[ $BOUNDED_PREFILL == 1 ]]; then
  sha256sum "$RUN_DIR/devices_pre.txt" "$RUN_DIR/devices_post.txt" >>"$RUN_DIR/evidence.sha256"
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" <<'PY'
import json
from hashlib import sha256
from pathlib import Path
import sys
from google.cloud import storage
from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact
root, pin = Path(sys.argv[1]), sys.argv[2]
bucket = storage.Client().bucket('driftbench-dsv4-uc')
receipts = []
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.name in ('SUCCESS', 'archive_receipts.json'):
        continue
    facts = digest_file(path)
    receipts.append(publish_exact(bucket, f'results/{root.name}/{path.relative_to(root)}',
                                  path, facts, compressed=False))
ledger = root / 'archive_receipts.json'
ledger.write_text(json.dumps(receipts, indent=2, sort_keys=True) + '\n')
publish_exact(bucket, f'results/{root.name}/{ledger.name}', ledger, digest_file(ledger), compressed=False)
terminal = root / 'SUCCESS'
runner = json.loads((root/'runner.json').read_text())
terminal.write_text(json.dumps(dict(tag=root.name, code_hash=pin,
    baseline_only=runner['baseline_only'], admission_only=runner.get('admission_only', False),
    boundary_diagnostic=runner.get('boundary_diagnostic', False), performance_claim=False,
    diagnostic_only=runner.get('diagnostic_only', False),
    bounded_admission=runner.get('bounded_admission', False),
    summary_sha256=sha256((root/'summary.json').read_bytes()).hexdigest(),
    archive_receipts_sha256=sha256(ledger.read_bytes()).hexdigest()), sort_keys=True) + '\n')
print(json.dumps(publish_exact(bucket, f'results/{root.name}/SUCCESS', terminal,
                              digest_file(terminal), compressed=False), sort_keys=True))
PY
  say "BOUNDED_PREFILL_SUCCESS ARCHIVE=$REMOTE_PREFIX (not model performance proof)"
  trap - EXIT
  exit 0
fi
touch "$RUN_DIR/SUCCESS"

gcloud storage cp --recursive --no-clobber "$RUN_DIR/hlo" \
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
