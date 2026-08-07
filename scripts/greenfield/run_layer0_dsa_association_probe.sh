#!/usr/bin/env bash
# Protected bounded real-input probe for the sealed 8K layer-0 DSA event.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly DEFAULT_INPUT=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_LAYER0_DSA_TAG:-greenfield_layer0_dsa_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
INPUT_DIR=${GLM_GREENFIELD_LAYER0_DSA_INPUT:-$DEFAULT_INPUT}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing layer-0 DSA probe outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing layer-0 DSA probe from a dirty worktree" >&2
  exit 2
}
[[ -r $RESULTS_DB && -d $INPUT_DIR && ! -e $RUN_DIR ]] || {
  echo "results DB/input missing or append-only run path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo" "$RUN_DIR/input"
cp "$INPUT_DIR/manifest.json" "$RUN_DIR/input/manifest.json"
cp "$INPUT_DIR/layer0_dsa_input.safetensors" \
  "$RUN_DIR/input/layer0_dsa_input.safetensors"

say() {
  echo "[layer0-dsa $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[p]robe_layer0_dsa_association[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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

say "RUN_DIR=$RUN_DIR PIN=$PIN input_manifest=$INPUT_MANIFEST_SHA"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "compiling bounded layer-0 association matrix on one TPU-v4 host"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$WORKTREE" \
    /home/gianl/vllm-env/bin/python \
      scripts/greenfield/probe_layer0_dsa_association.py \
      --expected-code-hash "$PIN" \
      --input-dir "$RUN_DIR/input" \
      --input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
      --output "$RUN_DIR/runner.json" \
      --hlo-dir "$RUN_DIR/hlo"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "runner completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" \
  "$INPUT_MANIFEST_SHA" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, db_path, repo, elapsed, input_sha = sys.argv[1:]
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
if runner["status"] != "SUCCESS" or runner["code_hash"] != pin:
    raise SystemExit("layer-0 DSA runner status/code identity failed")
if runner["input_manifest_sha256"] != input_sha:
    raise SystemExit("layer-0 DSA input identity failed")
for name, record in runner["hlo"]["state"].items():
    if not record["contract"]["passed"]:
        raise SystemExit(f"layer-0 DSA state HLO failed: {name}")
for name in ("legacy_score", "one_row_pagewise_score", "one_row_pallas_score"):
    if not runner["hlo"][name]["contract"]["passed"]:
        raise SystemExit(f"layer-0 DSA scorer HLO failed: {name}")
expected_variants = {
    "legacy_fp32_divsqrt",
    "legacy_fp32_rsqrt",
    "legacy_bf16_divsqrt",
    "greenfield_bf16_rsqrt_legacy_geometry",
    "legacy_fused_qkv_fp32_divsqrt",
    "one_row_pagewise_on_legacy_state",
    "one_row_pallas_on_legacy_state",
    "one_row_pagewise_on_fused_qkv_legacy_state",
    "one_row_pallas_on_fused_qkv_legacy_state",
}
if set(runner["comparisons"]) != expected_variants:
    raise SystemExit("layer-0 DSA comparison matrix is incomplete")
if not all(record["score_all_finite"] for record in runner["comparisons"].values()):
    raise SystemExit("layer-0 DSA comparison emitted non-finite scores")
if runner["profiler_free_timing"] is not False:
    raise SystemExit("diagnostic layer-0 DSA probe must not claim timing")

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model="zai-org/GLM-5.2-FP8:greenfield-layer0-dsa-association",
    revision="bounded-real-layer0-v1",
    env={
        "GLM_ENGINE": "greenfield_layer0_dsa_association",
        "greenfield_code_hash": pin,
        "input_manifest_sha256": input_sha,
        "device_kind": runner["device_kind"],
    },
    note="Protected bounded real layer-0 8K DSA association diagnostic.",
    harness_repo=repo,
    fork_repo=None,
)
pv.record_item(
    conn,
    run_id,
    benchmark="greenfield_layer0_dsa_association",
    item_id="layer0_position8155_event0",
    prompt="Sealed real layer-0 prompt embeddings and DSA tensors.",
    gold="Exact sealed 2,048-position set and lowest-position tie order.",
    raw_output=json.dumps(runner, sort_keys=True),
    extracted=json.dumps(
        {
            name: value["passed"]
            for name, value in runner["comparisons"].items()
        },
        sort_keys=True,
    ),
    correct=bool(runner["legacy_association_restored"]),
    score=float(runner["legacy_association_restored"]),
    latency_ms=None,
)
pv.finalize(
    conn,
    run_id,
    benchmark="greenfield_layer0_dsa_association",
    metric="diagnostic_completed",
    value=1.0,
    note="No decoder, Gate-D, latency, or token-rate claim.",
)
conn.close()

summary = {
    "status": "SUCCESS",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
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
check = sqlite3.connect(run_dir / "results_ckpt.db").execute(
    "PRAGMA integrity_check"
).fetchone()[0]
if check != "ok":
    raise SystemExit(f"results DB snapshot integrity failed: {check}")
print(f"LAYER0_DSA_ASSOCIATION_VALID db_run={run_id}")
PY

(
  cd "$RUN_DIR"
  find hlo input -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log summary.json results_ckpt.db census_pre.txt
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
gcloud storage cp --recursive --no-clobber "$RUN_DIR/input" \
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
