#!/usr/bin/env bash
# Protected production-shaped TPU-v4 proof for fused selected-KV sparse MLA.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_SPARSE_ATTN_TAG:-greenfield_sparse_attention_$(date -u +%Y%m%dT%H%M%S%NZ)}
WARMUP=${GLM_GREENFIELD_SPARSE_ATTN_WARMUP:-200}
ITERATIONS=${GLM_GREENFIELD_SPARSE_ATTN_ITERATIONS:-1000}
GEOMETRY=${GLM_GREENFIELD_SPARSE_ATTN_GEOMETRY:-pp8_lp4_256k}
DIAGNOSTIC_REFERENCE=${GLM_GREENFIELD_SPARSE_ATTN_DIAGNOSTIC_REFERENCE:-0}
[[ $GEOMETRY == pp8_lp4_256k ]] || TAG=${GLM_GREENFIELD_SPARSE_ATTN_TAG:-greenfield_sparse_attention_${GEOMETRY}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing sparse-attention run outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing sparse-attention run from a dirty worktree" >&2
  exit 2
}
[[ $GEOMETRY == pp8_lp4_256k || $GEOMETRY == pp16_lp2_2k ]] || exit 2
if [[ $DIAGNOSTIC_REFERENCE == 1 ]]; then
  [[ $GEOMETRY == pp16_lp2_2k ]] || exit 2
  [[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 1 ]] || exit 2
  [[ $ITERATIONS =~ ^[0-9]+$ && $ITERATIONS -ge 3 ]] || exit 2
else
  [[ $DIAGNOSTIC_REFERENCE == 0 && $GEOMETRY == pp8_lp4_256k ]] || exit 2
  [[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 200 ]] || exit 2
  [[ $ITERATIONS =~ ^[0-9]+$ && $ITERATIONS -ge 1000 ]] || exit 2
fi
[[ -r $RESULTS_DB && ! -e $RUN_DIR ]] || {
  echo "results DB missing or append-only run path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[sparse-attention $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" \
  --format='value(location)') == "$APPROVED_LOCATION" ]]

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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[t]race_pipeline_transport[.]py|[r]un_real_one_layer[.]py|[m]icrobench_fp8_matmul[.]py|[m]icrobench_dsa_score[.]py|[m]icrobench_dsa_topk[.]py|[m]icrobench_sparse_attention[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
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

say "RUN_DIR=$RUN_DIR PIN=$PIN geometry=$GEOMETRY warmup=$WARMUP iterations=$ITERATIONS diagnostic_reference=$DIAGNOSTIC_REFERENCE"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "compiling and timing fused $GEOMETRY selected-KV sparse MLA"
started=$(date +%s)
(
  cd "$WORKTREE"
  RUNNER=(
      /home/gianl/vllm-env/bin/python
      scripts/greenfield/microbench_sparse_attention.py
      --expected-code-hash "$PIN"
      --output "$RUN_DIR/runner.json"
      --hlo-output "$RUN_DIR/hlo/sparse_attention.optimized_hlo.txt"
      --geometry "$GEOMETRY"
      --warmup "$WARMUP"
      --iterations "$ITERATIONS"
    )
    if [[ $DIAGNOSTIC_REFERENCE == 1 ]]; then
      RUNNER+=(--diagnostic-reference-timing)
    fi
    env \
      JAX_PLATFORMS=tpu \
      TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
      TPU_PROCESS_BOUNDS=1,1,1 \
      TPU_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH="$WORKTREE" \
      "${RUNNER[@]}"
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
    raise SystemExit("runner status/code identity failed")
if not runner["hlo"]["contract"]["passed"]:
    raise SystemExit("fused sparse-attention HLO contract failed")
if not runner["comparison"]["passed"]:
    raise SystemExit("fused sparse-attention correctness failed")
if not runner["profiler_free_timing"]:
    raise SystemExit("sparse-attention wall distributions are not profiler-free")
diagnostic_reference = runner.get("reference_latency")
if bool(diagnostic_reference) != runner.get("diagnostic_only"):
    raise SystemExit("sparse-attention diagnostic identity drifted")
if diagnostic_reference is not None:
    if (
        runner.get("geometry") != "pp16_lp2_2k"
        or not runner.get("reference_hlo", {}).get("contract", {}).get("passed")
    ):
        raise SystemExit("sparse-attention reference diagnostic HLO failed")

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
run_id = pv.start_run(
    conn,
    model=(
        "zai-org/GLM-5.2-FP8:greenfield-sparse-attention-kernel:"
        + runner["geometry"]
    ),
    revision="native-jax-pallas-v1",
    env={
        "GLM_ENGINE": "greenfield_sparse_attention",
        "greenfield_code_hash": pin,
        "hlo_sha256": runner["hlo"]["sha256"],
        "device_kind": runner["device_kind"],
        "geometry": runner["geometry"],
        "diagnostic_only": runner["diagnostic_only"],
    },
    note="Protected production-shaped fused selected-KV/sparse-MLA microbenchmark.",
    harness_repo=repo,
    fork_repo=None,
)
for base_item_id, prompt in (
    (
        "balanced_owner512",
        "One owner consumes 512 of 2,048 uniformly distributed selected rows.",
    ),
    (
        "concentrated_owner2048",
        "Worst-case one owner consumes all 2,048 selected rows.",
    ),
):
    item_id = runner["geometry"] + "_" + base_item_id
    pv.record_item(
        conn,
        run_id,
        benchmark="greenfield_sparse_attention",
        item_id=item_id,
        prompt=prompt,
        gold="Reference-correct BF16 partial output/LSE; no selected-KV HBM tensor.",
        raw_output=json.dumps(runner, sort_keys=True),
        extracted=str(runner["checksum"]),
        correct=True,
        score=1.0,
        latency_ms=runner["latency"][
            "balanced_owner0"
            if base_item_id == "balanced_owner512"
            else "concentrated_owner0"
        ]["p50_ms"],
    )
pv.finalize(
    conn,
    run_id,
    benchmark="greenfield_sparse_attention",
    metric="contract_valid",
    value=1.0,
    note="Standalone local-owner partial only; not layer latency or token throughput.",
)
conn.close()

summary = {
    "status": "SUCCESS",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
    "runner": runner,
    "claim_scope": (
        "diagnostic standalone reference/Pallas comparison; no performance, "
        "Gate-D or token-rate claim"
        if runner["diagnostic_only"]
        else "standalone fused local-owner sparse MLA; no token-rate claim"
    ),
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
print(f"SPARSE_ATTENTION_PROOF_VALID db_run={run_id}")
PY

(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
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
