#!/usr/bin/env bash
# Protected one-host TPU matrix for the first divergent layer-0 DSA query.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly CAPTURE_ROOT=/home/gianl/glm-run/greenfield_legacy_layer0_dsa_internals_recovery_20260808T030853085141329Z
readonly CAPTURE_DIR=$CAPTURE_ROOT/internal_comparison
readonly CAPTURE_SUCCESS_SHA=20deb19cc34097d99a4b558b3f0e3cb964716d58dbe91fd825cb4641434ac31c
readonly CAPTURE_COMPARISON_SHA=623818dc50e77530966e1cdfc8a0b310f8ed23733476800e6a8feaa627e8b85e
readonly CAPTURE_TENSORS_SHA=0a724bada77d93ddc524368ac3b6e3c7f70442aba59ed96da5abde3dbe75fa39
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly DISTRIBUTED_Q_A_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_association_20260807T231449677046310Z/distributed_q_a_norm_artifact
readonly DISTRIBUTED_Q_A_MANIFEST_SHA=7518e7eff0487f0dc02cd4b0ff1c3d0fc3ef9ca7c43dcded7d809120e30d8c16
readonly DISTRIBUTED_Q_A_CODE_HASH=ea879a24d196f61e238a22ee5bb393d3b6fa938d
readonly DB502_DIR=/home/gianl/glm-run/greenfield_layer0_q_a_association_20260808T124434046623046Z
readonly DB502_CODE_HASH=c230c11d2c852b52bbbf4b76791bb4dc80c598ba
readonly DB502_RUNNER_SHA=2a77d75d27ae06128084f0d3956b0ea5b2886ae2b3a5ecbe896acc9bd43075c4
readonly DB502_TENSOR_SHA=d9b14bdd47b5def0169b0b25157a8d472bc0017391030b0d7842b794fae8f76e
readonly DB502_SUCCESS_SHA=de2e080d3eae672569acc4dada7eb41501c6f3508ac0e1747291041d95087dab
readonly CURRENT_INTERNAL_NPZ=/home/gianl/glm-run/greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_oracle_dsa_dsa_internal_trace2_20260810T013247766447206Z/dsa_internal_observer/position_8155_internals.npz
readonly CURRENT_INTERNAL_SHA=e1366c58a5eac8d995e6f56a4a3bb582fa6758b25bf19480b67e8f4c54914b50
readonly TARGET=${GLM_GREENFIELD_DSA_ASSOCIATION_TARGET:-query}

[[ $TARGET == query || $TARGET == query_lp4 || \
  $TARGET == query_lp4_q_a_boundary || $TARGET == q_a || \
  $TARGET == qkv_a_production ]] || {
  echo "DSA association target is unknown: $TARGET" >&2
  exit 2
}

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
if [[ $TARGET == q_a ]]; then
  TAG=${GLM_GREENFIELD_DSA_QUERY_ASSOCIATION_TAG:-greenfield_layer0_q_a_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
  REMOTE_KIND=q_a_association
elif [[ $TARGET == qkv_a_production ]]; then
  TAG=${GLM_GREENFIELD_DSA_QUERY_ASSOCIATION_TAG:-greenfield_layer0_qkv_a_production_$(date -u +%Y%m%dT%H%M%S%NZ)}
  REMOTE_KIND=qkv_a_production_association
elif [[ $TARGET == query_lp4 ]]; then
  TAG=${GLM_GREENFIELD_DSA_QUERY_ASSOCIATION_TAG:-greenfield_layer0_physical_lp4_dsa_query_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
  REMOTE_KIND=physical_lp4_dsa_query_association
elif [[ $TARGET == query_lp4_q_a_boundary ]]; then
  TAG=${GLM_GREENFIELD_DSA_QUERY_ASSOCIATION_TAG:-greenfield_layer0_physical_lp4_dsa_q_a_boundary_$(date -u +%Y%m%dT%H%M%S%NZ)}
  REMOTE_KIND=physical_lp4_dsa_q_a_boundary_association
else
  TAG=${GLM_GREENFIELD_DSA_QUERY_ASSOCIATION_TAG:-greenfield_layer0_dsa_query_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
  REMOTE_KIND=dsa_query_association
fi
if [[ $TARGET == query_lp4 || $TARGET == query_lp4_q_a_boundary ]]; then
  [[ -r $CURRENT_INTERNAL_NPZ ]] || {
    echo "current physical LP4 observer is unavailable" >&2
    exit 2
  }
  [[ $(sha256sum "$CURRENT_INTERNAL_NPZ" | awk '{print $1}') == \
    "$CURRENT_INTERNAL_SHA" ]] || {
    echo "current physical LP4 observer identity drifted" >&2
    exit 2
  }
fi
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/$REMOTE_KIND/8k/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing query association outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing query association from a dirty worktree" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only query association directory exists: $RUN_DIR" >&2
  exit 2
}
for path in "$RESULTS_DB" "$CAPTURE_ROOT/SUCCESS" \
  "$CAPTURE_DIR/comparison.json" "$CAPTURE_DIR/internals.npz" \
  "$INPUT_DIR/manifest.json" "$DISTRIBUTED_Q_A_DIR/manifest.json"; do
  [[ -r $path ]] || {
    echo "query association input is unavailable: $path" >&2
    exit 2
  }
done
[[ $(sha256sum "$CAPTURE_ROOT/SUCCESS" | awk '{print $1}') == \
  "$CAPTURE_SUCCESS_SHA" ]] || {
  echo "sealed capture SUCCESS identity drifted" >&2
  exit 2
}
if [[ $TARGET == qkv_a_production ]]; then
  for path in "$DB502_DIR/runner.json" "$DB502_DIR/q_a_candidates.npz" \
    "$DB502_DIR/SUCCESS"; do
    [[ -r $path ]] || {
      echo "production qkv-a DB502 input is unavailable: $path" >&2
      exit 2
    }
  done
  [[ $(sha256sum "$DB502_DIR/runner.json" | awk '{print $1}') == \
    "$DB502_RUNNER_SHA" ]] || {
    echo "sealed DB502 runner identity drifted" >&2
    exit 2
  }
  [[ $(sha256sum "$DB502_DIR/q_a_candidates.npz" | awk '{print $1}') == \
    "$DB502_TENSOR_SHA" ]] || {
    echo "sealed DB502 tensor identity drifted" >&2
    exit 2
  }
  [[ $(sha256sum "$DB502_DIR/SUCCESS" | awk '{print $1}') == \
    "$DB502_SUCCESS_SHA" ]] || {
    echo "sealed DB502 SUCCESS identity drifted" >&2
    exit 2
  }
fi

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

mkdir -p "$RUN_DIR"
say() {
  echo "[dsa-${TARGET}-association $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" |
      sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}" ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py|[p]robe_layer0_dsa_query_association[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
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
    say "FAILED status=$status; preserving query diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "PIN=$PIN TARGET=$TARGET RUN_DIR=$RUN_DIR"
say "CAPTURE=$CAPTURE_ROOT REMOTE_PREFIX=$REMOTE_PREFIX"
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}

started=$(date +%s)
current_args=()
if [[ $TARGET == query_lp4 || $TARGET == query_lp4_q_a_boundary ]]; then
  current_args=(
    --current-internal-npz "$CURRENT_INTERNAL_NPZ"
    --current-internal-sha256 "$CURRENT_INTERNAL_SHA"
  )
fi
env JAX_PLATFORMS=tpu \
  TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH="$WORKTREE" \
  timeout --signal=TERM --kill-after=60 1800 \
  /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/probe_layer0_dsa_query_association.py" \
  --target "$TARGET" \
  --expected-code-hash "$PIN" \
  --capture-dir "$CAPTURE_DIR" \
  --capture-comparison-sha256 "$CAPTURE_COMPARISON_SHA" \
  --capture-tensors-sha256 "$CAPTURE_TENSORS_SHA" \
  --input-dir "$INPUT_DIR" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
  --distributed-q-a-norm-dir "$DISTRIBUTED_Q_A_DIR" \
  --q-a-manifest-sha256 "$DISTRIBUTED_Q_A_MANIFEST_SHA" \
  --q-a-code-hash "$DISTRIBUTED_Q_A_CODE_HASH" \
  --db502-dir "$DB502_DIR" \
  --db502-code-hash "$DB502_CODE_HASH" \
  --db502-runner-sha256 "$DB502_RUNNER_SHA" \
  --db502-tensor-sha256 "$DB502_TENSOR_SHA" \
  "${current_args[@]}" \
  --output "$RUN_DIR/runner.json" \
  --hlo-dir "$RUN_DIR/hlo" >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "query matrix completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" "$TARGET" <<'PY'
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

run_dir, pin, db_path, repo, elapsed, target = sys.argv[1:]
run_dir = Path(run_dir)
runner = json.loads((run_dir / "runner.json").read_text())
if target == "q_a":
    expected_candidates = {
        f"virtual_{projection}_{norm}_m1_n82"
        for projection in ("lax_map_convolution",)
        for norm in (
            "logical_mean", "shard_sum", "left_fold", "topology_tree"
        )
    }
elif target == "qkv_a_production":
    expected_candidates = {
        "production_fused_n82_convolution_shard_sum",
    }
elif target == "query_lp4":
    expected_candidates = {
        "physical_raw_owner_dot_m1_n1024",
        "physical_raw_head_unrolled_m1_n128",
        "physical_predecoded_owner_dot_m1_n1024",
        "physical_predecoded_head_unrolled_m1_n128",
    }
elif target == "query_lp4_q_a_boundary":
    expected_candidates = {
        "physical_fused_q_a_unrounded_default_owner_dot_m1_n1024",
        "physical_fused_q_a_bf16_barrier_default_owner_dot_m1_n1024",
        "physical_fused_q_a_bf16_barrier_highest_owner_dot_m1_n1024",
    }
else:
    expected_candidates = {
        "global_m32_n4096",
        "global_m1_n4096",
        "head_lax_map_m32_n128",
        "head_vmap_m32_n128",
        "head_unrolled_m32_n128",
        "head_fori_m32_n128",
        "head_lax_map_highest_m32_n128",
        "head_lax_map_m1_n128",
        "head_unrolled_m1_n128",
        "lp4_unrolled_m32_n1024",
        "lp4_unrolled_m1_n1024",
        "pallas_global_m1_n4096",
        "pallas_lp4_m1_n1024",
        "pallas_vector_global_m1_n4096",
        "pallas_vector_lp4_m1_n1024",
        "raw_lookup_global_m1_n128_tiles",
        "raw_lookup_lp4_m1_n128_tiles",
        "raw_materialized_global_m1_n4096",
        "raw_materialized_lp4_m1_n1024",
    }
if runner["status"] != "SUCCESS" or runner["code_hash"] != pin:
    raise SystemExit("query association status/code identity failed")
if runner["backend"] != "tpu" or runner["device_count"] != 4:
    raise SystemExit("query association did not use one four-chip TPU host")
if set(runner["candidates"]) != expected_candidates:
    raise SystemExit("query association candidate matrix is incomplete")
if runner["performance_claim"] is not False or not runner["diagnostic_only"]:
    raise SystemExit("query association made a performance claim")
if any(not value["hlo"]["passed"] for value in runner["candidates"].values()):
    raise SystemExit("query association HLO contract failed")
if target in ("q_a", "qkv_a_production"):
    tensor = run_dir / runner["tensor_file"]["filename"]
    if (
        not runner["one_live_row"]
        or runner["virtual_tensor_shards"] != 32
        or runner["local_output_width"] != 82
        or not tensor.is_file()
        or tensor.stat().st_size != runner["tensor_file"]["byte_count"]
        or sha256(tensor.read_bytes()).hexdigest()
        != runner["tensor_file"]["sha256"]
    ):
        raise SystemExit("q-a association tensor/one-row contract failed")
if target == "qkv_a_production":
    candidate = runner["candidates"][
        "production_fused_n82_convolution_shard_sum"
    ]
    hlo = candidate["hlo"]
    if (
        not runner["production_helper"]
        or runner["final_layout_inputs"]["weight_bits_shape"]
        != [32, 6144, 82]
        or runner["final_layout_inputs"]["scale_shape"] != [32, 48, 82]
        or not candidate["comparison"]["elementwise_exact"]
        or not candidate["companion_comparison"]["elementwise_exact"]
        or hlo["convolution_count"] != 1
        or hlo["expected_convolution_count"] != 1
        or hlo["forbidden_operations"]
        or hlo["forbidden_shapes"]
        or not all(hlo["required_shapes"].values())
    ):
        raise SystemExit("production qkv-a arithmetic/HLO contract failed")
if target == "query_lp4":
    tensor = run_dir / runner["tensor_file"]["filename"]
    if (
        runner["artifact_kind"]
        != "glm52_layer0_physical_lp4_dsa_query_association"
        or not runner["one_live_row"]
        or runner["local_parallel_size"] != 4
        or not tensor.is_file()
        or tensor.stat().st_size != runner["tensor_file"]["byte_count"]
        or sha256(tensor.read_bytes()).hexdigest()
        != runner["tensor_file"]["sha256"]
        or any(
            not candidate["hlo"]["passed"]
            for candidate in runner["candidates"].values()
        )
    ):
        raise SystemExit("physical LP4 query arithmetic/HLO contract failed")
if target == "query_lp4_q_a_boundary":
    tensor = run_dir / runner["tensor_file"]["filename"]
    if (
        runner["artifact_kind"]
        != "glm52_layer0_physical_lp4_dsa_q_a_boundary_association"
        or not runner["one_live_row"]
        or runner["local_parallel_size"] != 4
        or set(runner["q_a_exact_candidates"]) != expected_candidates
        or not tensor.is_file()
        or tensor.stat().st_size != runner["tensor_file"]["byte_count"]
        or sha256(tensor.read_bytes()).hexdigest()
        != runner["tensor_file"]["sha256"]
        or any(
            not candidate["hlo"]["passed"]
            for candidate in runner["candidates"].values()
        )
    ):
        raise SystemExit("physical LP4 q-a boundary contract failed")

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

connection = pv.connect(db_path)
run_id = pv.start_run(
    connection,
    model=f"zai-org/GLM-5.2-FP8:greenfield-layer0-{target}-association",
    revision=(
        f"bounded-real-layer0-v3-{target}-association"
        if target == "qkv_a_production"
        else (
            f"bounded-real-layer0-v2-{target}-association"
            if target == "q_a"
            else f"bounded-real-layer0-v1-{target}-association"
        )
    ),
    env={
        "GLM_ENGINE": f"greenfield_layer0_{target}_association",
        "greenfield_code_hash": pin,
        "capture_owner_actual_sha256": runner["capture"][
            "owner_actual_sha256"
        ],
        "input_manifest_sha256": runner["input_manifest_sha256"],
        "q_a_manifest_sha256": runner["q_a_manifest_sha256"],
        "device_kind": runner["device_kind"],
    },
    note=(
        "Protected integrated layer-0 qkv-a production-helper diagnostic."
        if target == "qkv_a_production"
        else f"Protected layer-0 8K DSA {target} association diagnostic."
    ),
    harness_repo=repo,
    fork_repo=None,
)
pv.record_item(
    connection,
    run_id,
    benchmark=f"greenfield_layer0_{target}_association",
    item_id=f"layer0_position8155_{target}",
    prompt=(
        "Sealed accepted layer-0 q-a, DB502 kv-a companion, and source weights."
        if target == "qkv_a_production"
        else f"Sealed accepted layer-0 {target} state and source weights."
    ),
    gold=(
        "Exact accepted q-a and exact sealed DB502 fused kv-a companion."
        if target == "qkv_a_production"
        else f"Elementwise-exact accepted {target} association."
    ),
    raw_output=json.dumps(runner, sort_keys=True),
    extracted=json.dumps(runner["exact_candidates"], sort_keys=True),
    correct=bool(runner["association_restored"]),
    score=float(runner["association_restored"]),
    latency_ms=None,
)
pv.finalize(
    connection,
    run_id,
    benchmark=f"greenfield_layer0_{target}_association",
    metric="diagnostic_completed",
    value=1.0,
    note="No decoder, Gate-D, latency, or token-rate claim.",
)
connection.close()
summary = {
    "status": "SUCCESS",
    "target": target,
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
    "association_restored": runner["association_restored"],
    "exact_candidates": runner["exact_candidates"],
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
    raise SystemExit("query association DB snapshot failed integrity check")
print(f"QUERY_ASSOCIATION_VALID db_run={run_id}")
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "freezing and archiving query-association evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  if [[ -f q_a_candidates.npz ]]; then
    sha256sum q_a_candidates.npz
  fi
  if [[ -f qkv_a_production.npz ]]; then
    sha256sum qkv_a_production.npz
  fi
  if [[ -f physical_lp4_query_candidates.npz ]]; then
    sha256sum physical_lp4_query_candidates.npz
  fi
  if [[ -f physical_lp4_q_a_boundary.npz ]]; then
    sha256sum physical_lp4_q_a_boundary.npz
  fi
  sha256sum runner.json runner.log summary.json results_ckpt.db \
    census_pre.txt census_post.txt orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY' \
  >"$RUN_DIR/remote_objects.json"
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = [
    (path, path.relative_to(root).as_posix())
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.name not in {"SUCCESS", "remote_objects.json"}
]

def describe(item):
    path, relative = item
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}",
         "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    crc32c = value.get("crc32c_hash") or value.get("crc32c")
    if int(value["size"]) != path.stat().st_size or not crc32c:
        raise SystemExit(f"remote object verification failed: {relative}")
    return {
        "crc32c": crc32c,
        "generation": value["generation"],
        "path": relative,
        "size": int(value["size"]),
    }

with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" "$TARGET" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
values = {
    "artifact_kind": (
        "glm52_layer0_q_a_association"
        if sys.argv[4] == "q_a"
        else (
            "glm52_layer0_qkv_a_production_association"
            if sys.argv[4] == "qkv_a_production"
            else (
                "glm52_layer0_physical_lp4_dsa_query_association"
                if sys.argv[4] == "query_lp4"
                else (
                    "glm52_layer0_physical_lp4_dsa_q_a_boundary_association"
                    if sys.argv[4] == "query_lp4_q_a_boundary"
                    else "glm52_layer0_dsa_query_association"
                )
            )
        )
    ),
    "code_hash": sys.argv[3],
    "results_db_run_id": summary["results_db_run_id"],
    "association_restored": str(summary["association_restored"]).lower(),
    "exact_candidates": ",".join(summary["exact_candidates"]) or "none",
    "performance_claim": "false",
    "evidence_sha256": sha256(
        (root / "evidence.sha256").read_bytes()
    ).hexdigest(),
    "remote_objects_sha256": sha256(
        (root / "remote_objects.json").read_bytes()
    ).hexdigest(),
    "remote_prefix": sys.argv[2],
}
(root / "SUCCESS").write_text(
    "".join(f"{key}={value}\n" for key, value in values.items())
)
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
local_success_sha=$(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}')
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" |
  sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]] || {
  say "ABORT: remote SUCCESS checksum mismatch"
  exit 1
}

trap - EXIT
db_run=$(sed -n 's/^results_db_run_id=//p' "$RUN_DIR/SUCCESS")
exact=$(sed -n 's/^exact_candidates=//p' "$RUN_DIR/SUCCESS")
say "SUCCESS DB=$db_run exact=$exact"
