#!/usr/bin/env bash
# Protected one-host discriminator for the accepted FP32 prompt-key projection.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly OBSERVER_REPO=/home/gianl/tpu-inference-greenfield-dsa-internal-observer
readonly OBSERVER_PIN=9c1d6b3b950d5c5dd45bdf885058202517097eba
readonly SOURCE_TAG=greenfield_legacy_layer0_prompt_key_internals_p113_20260809T024848198817596Z
readonly SOURCE_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_key_internals/8k/$SOURCE_TAG
readonly SOURCE_GREENFIELD_PIN=00404a0cb6ce18c71816cc47e16ddbaa5f1a5fa6
readonly SOURCE_HARNESS_PIN=a4a17ac4e90b15f1994bd8b26917ef62daa52660
readonly SOURCE_RUN_ID=513
readonly SOURCE_ITEM_ROW_ID=1798
readonly SOURCE_SUCCESS_SHA=28b3c64e643648257b2d055f6490fc7dbdbf2742bfdf91e2ec48003ee0a4a0f5
readonly SOURCE_REMOTE_OBJECTS_SHA=d4345f53874e85eadae0d63cc86784143a636dafdd135f08dc7b09fd35a828e3
readonly SOURCE_DB_SHA=a56f3a2f921b5e15831377e230b94d772f8fe0367d526e08c50f173b2bac8f77
readonly SOURCE_COMPARISON_FILE_SHA=ef20dea7d952a45f0a836c2bc6e5fb21bd62ee3d4d3b8d21680ff9464b8f4e42
readonly SOURCE_COMPARISON_MANIFEST_SHA=605eeac29ff4278ce4077a194d633d7f9b28c1ed36d94347cefc7920f1b2a04c
readonly SOURCE_CAPTURE_FILE_SHA=6cd64adc8fcc0a6d3dc7bba01bddfb44a852bb08fa7dd44703a12c16e8721159
readonly SOURCE_CAPTURE_TENSOR_SHA=db26efc4a0e5de70f13751dd281908dd25f1eca3d7912a9c7f91cfa7fa45bb64
readonly SOURCE_CAPTURE_MANIFEST_SHA=dd36143747165828c5136ab0948cf87c0fb62a382efbbe231fa6f992f538591f
readonly SOURCE_CACHE_MANIFEST_FILE_SHA=51767ac2ee068811a6a1a8db849d2feb30c9b177aade9a9e007c9ea0c8bbae59
readonly SOURCE_CACHE_TENSOR_SHA=bb5dd2a91c0a30f2c0e460737d5ce680c1e638f6974511e24be0bf05a1d970eb
readonly SOURCE_CACHE_MANIFEST_SHA=b30ddc72481c7e6ece470a8fbb7d0788b5c3ea15cd4c2357fe6f4eec4e7aa0e3
readonly ACCEPTED_CACHE_SHA=3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1
readonly SEALED_BF16_CACHE_SHA=52bf55ed5e9ea74a59551351a21ae830fb39e289e82eaff636fb9ab84d7dcd8a
readonly SOURCE_PRE_CENSUS_SHA=b6e799daaeaeefd02eae09d09690a5cbe1b656dcc632584e224a4e0d261e5b35
readonly SOURCE_POST_CENSUS_SHA=8fb4dc96debeb1eb4bb526aafd893c45669cf773ca3b075f9927ded3635f5659
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly INPUT_MANIFEST_FILE_SHA=bd06714ebfe5177b8466778e2bc33ef262544dced48adcfc5739be37ac6488b9

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_PROMPT_KEY_PROJECTION_TAG:-greenfield_layer0_prompt_key_projection_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_key_projection_association/8k/$TAG
COMPARISON_DIR=$RUN_DIR/comparison

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" &&
  $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing prompt-key projection probe from wrong branch/worktree" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing prompt-key projection probe from a dirty worktree" >&2
  exit 2
}
[[ $(git -C "$LEGACY_REPO" rev-parse HEAD) == "$LEGACY_PIN" &&
  -z $(git -C "$LEGACY_REPO" status --porcelain) ]] || {
  echo "accepted legacy checkout drifted" >&2
  exit 2
}
[[ $(git -C "$OBSERVER_REPO" rev-parse HEAD) == "$OBSERVER_PIN" &&
  -z $(git -C "$OBSERVER_REPO" status --porcelain) ]] || {
  echo "accepted observer checkout drifted" >&2
  exit 2
}
[[ -r $RESULTS_DB && -d $SOURCE_DIR && -d $INPUT_DIR && ! -e $RUN_DIR ]] || {
  echo "projection source/DB missing or append-only path exists" >&2
  exit 2
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

mkdir -p "$RUN_DIR"
say() {
  echo "[prompt-key-projection $(date -u +%H:%M:%S)] $*" |
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py|[c]ompare_accepted_prompt_key_internals[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
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
    say "FAILED status=$status; preserving bounded diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "PIN=$PIN RUN_DIR=$RUN_DIR SOURCE_DB=$SOURCE_RUN_ID/$SOURCE_ITEM_ROW_ID"
available_bytes=$(df -PB1 /home/gianl | awk 'NR == 2 {print $4}')
[[ $available_bytes -ge 8000000000 ]] || {
  say "ABORT: worker0 has less than 8 GB free"
  exit 1
}
printf 'available_bytes=%s\nrequired_bytes=8000000000\n' "$available_bytes" \
  >"$RUN_DIR/disk_preflight.txt"
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}

for contract in \
  "$SOURCE_SUCCESS_SHA $SOURCE_DIR/SUCCESS" \
  "$SOURCE_REMOTE_OBJECTS_SHA $SOURCE_DIR/remote_objects.json" \
  "$SOURCE_DB_SHA $SOURCE_DIR/results_ckpt.db" \
  "$SOURCE_COMPARISON_FILE_SHA $SOURCE_DIR/prompt_key_comparison/comparison.json" \
  "$SOURCE_CAPTURE_FILE_SHA $SOURCE_DIR/prompt_key_comparison/accepted_capture/capture.json" \
  "$SOURCE_CAPTURE_TENSOR_SHA $SOURCE_DIR/prompt_key_comparison/accepted_capture/accepted_prompt_key_states.npz" \
  "$SOURCE_CACHE_MANIFEST_FILE_SHA $SOURCE_DIR/prompt_index_cache/manifest.json" \
  "$SOURCE_CACHE_TENSOR_SHA $SOURCE_DIR/prompt_index_cache/prompt_index_cache.safetensors" \
  "$SOURCE_PRE_CENSUS_SHA $SOURCE_DIR/census_pre.txt" \
  "$SOURCE_POST_CENSUS_SHA $SOURCE_DIR/census_post.txt" \
  "$INPUT_MANIFEST_FILE_SHA $INPUT_DIR/manifest.json"; do
  expected=${contract%% *}
  path=${contract#* }
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: sealed source drifted: $path"
    exit 2
  }
done
has_eight_unique_markers "$SOURCE_DIR/census_pre.txt" CENSUS_OK || exit 2
has_eight_unique_markers "$SOURCE_DIR/census_post.txt" CENSUS_OK || exit 2
remote_source_success_sha=$(gcloud storage cat \
  "$SOURCE_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $remote_source_success_sha == "$SOURCE_SUCCESS_SHA" ]] || {
  say "ABORT: approved DB513 SUCCESS drifted"
  exit 2
}

/home/gianl/vllm-env/bin/python - \
  "$RESULTS_DB" "$SOURCE_DIR" "$RUN_DIR/source_validation.json" \
  "$SOURCE_RUN_ID" "$SOURCE_ITEM_ROW_ID" "$SOURCE_GREENFIELD_PIN" \
  "$SOURCE_HARNESS_PIN" "$OBSERVER_PIN" "$LEGACY_PIN" \
  "$SOURCE_COMPARISON_MANIFEST_SHA" "$SOURCE_CAPTURE_MANIFEST_SHA" \
  "$SOURCE_CACHE_MANIFEST_SHA" "$ACCEPTED_CACHE_SHA" \
  "$SEALED_BF16_CACHE_SHA" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

(db_path, source_path, output_path, run_id, item_id, greenfield_pin,
 harness_pin, observer_pin, oracle_pin, comparison_sha, capture_sha,
 cache_manifest_sha, accepted_cache_sha, candidate_cache_sha) = sys.argv[1:]
run_id, item_id = int(run_id), int(item_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("DB513 source DB integrity failed")
run = connection.execute(
    "SELECT model,harness_git,fork_git,env_json,pod,note FROM runs WHERE run_id=?",
    (run_id,),
).fetchone()
item = connection.execute(
    "SELECT id,benchmark,item_id,correct,score,n_prompt_tokens,n_gen_tokens "
    "FROM items WHERE id=? AND run_id=?", (item_id, run_id),
).fetchone()
if run is None or item != (
    item_id, "passkey_L8192_d0.5", "t0", 1, 1.0, 8155, 20,
):
    raise SystemExit("DB513 source row drifted")
model, harness, fork, env_json, pod, note = run
env = json.loads(env_json)
if (
    model != "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
    or harness != harness_pin[:7]
    or len(fork) < 7
    or not oracle_pin.startswith(fork)
    or pod != "db-v4-64-od"
    or "prompt_key_internals" not in note
    or env["os_env"].get("GLM_DSA_DUMP_INTERNALS_CODE_HASH") != observer_pin
    or env["os_env"].get("GLM_DSA_DUMP_INTERNALS_ORACLE_PIN") != oracle_pin
):
    raise SystemExit("DB513 source provenance drifted")
root = Path(source_path)
success = dict(
    line.split("=", 1)
    for line in (root / "SUCCESS").read_text().splitlines()
    if "=" in line
)
comparison = json.loads(
    (root / "prompt_key_comparison/comparison.json").read_text()
)
capture = json.loads(
    (root / "prompt_key_comparison/accepted_capture/capture.json").read_text()
)
cache = json.loads((root / "prompt_index_cache/manifest.json").read_text())
if (
    success.get("code_hash") != greenfield_pin
    or success.get("legacy_repository_pin") != observer_pin
    or success.get("source_run_id") != str(run_id)
    or success.get("source_item_row_id") != str(item_id)
    or comparison.get("manifest_sha256") != comparison_sha
    or comparison.get("conclusion") != {
        "classification": "projection_association",
        "first_divergent_field": "pre_layer_norm_key",
    }
    or comparison["cache_comparison"]["mismatch_count"] != 45
    or comparison["cache_comparison"]["observed_bfloat16_sha256"]
    != candidate_cache_sha
    or capture.get("manifest_sha256") != capture_sha
    or capture.get("oracle_pin") != oracle_pin
    or capture.get("legacy_code_hash") != observer_pin
    or cache.get("manifest_sha256") != cache_manifest_sha
    or cache.get("prompt_index_key_bfloat16_sha256") != accepted_cache_sha
):
    raise SystemExit("DB513 compact source artifact drifted")
Path(output_path).write_text(json.dumps({
    "status": "SUCCESS",
    "source_run_id": run_id,
    "source_item_row_id": item_id,
    "greenfield_code_hash": greenfield_pin,
    "observer_code_hash": observer_pin,
    "accepted_oracle_pin": oracle_pin,
    "comparison_manifest_sha256": comparison_sha,
    "capture_manifest_sha256": capture_sha,
    "prompt_cache_manifest_sha256": cache_manifest_sha,
}, indent=2, sort_keys=True) + "\n")
connection.close()
PY

say "running source-faithful FP32 projection on one four-chip TPU host"
started=$(date +%s)
env JAX_PLATFORMS=tpu \
  TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH="$WORKTREE" \
  timeout --signal=TERM --kill-after=60 1800 \
  /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/compare_accepted_prompt_key_internals.py" \
  --accepted-capture-dir "$SOURCE_DIR/prompt_key_comparison/accepted_capture" \
  --accepted-capture-manifest-sha256 "$SOURCE_CAPTURE_MANIFEST_SHA" \
  --input-dir "$INPUT_DIR" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
  --prompt-cache-dir "$SOURCE_DIR/prompt_index_cache" \
  --prompt-cache-manifest-sha256 "$SOURCE_CACHE_MANIFEST_SHA" \
  --output "$COMPARISON_DIR" \
  --run-tag "$TAG" \
  --accepted-run-tag "$SOURCE_TAG" \
  --greenfield-code-hash "$PIN" \
  --legacy-code-hash "$OBSERVER_PIN" \
  --oracle-pin "$LEGACY_PIN" \
  --expected-accepted-cache-sha256 "$ACCEPTED_CACHE_SHA" \
  --projection-weight-mode adapted_fp32 \
  >"$RUN_DIR/comparison_summary.json"
elapsed=$(( $(date +%s) - started ))
say "projection discriminator completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" \
  "$SOURCE_RUN_ID" "$SOURCE_ITEM_ROW_ID" \
  "$SOURCE_COMPARISON_MANIFEST_SHA" "$SOURCE_CAPTURE_MANIFEST_SHA" \
  "$SOURCE_CACHE_MANIFEST_SHA" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

(run_path, pin, db_path, repo, elapsed, source_run_id, source_item_id,
 source_comparison_sha, capture_sha, cache_sha) = sys.argv[1:]
root = Path(run_path)
comparison = json.loads((root / "comparison/comparison.json").read_text())
if (
    comparison["status"] != "SUCCESS"
    or comparison["code_hash"] != pin
    or comparison["backend"] != "tpu"
    or comparison["device_count"] != 4
    or comparison["diagnostic_only"] is not True
    or comparison["performance_claim"] is not False
    or comparison["projection_weight_mode"] != "adapted_fp32"
    or comparison["accepted_capture"]["manifest_sha256"] != capture_sha
    or comparison["accepted_cache"]["manifest_sha256"] != cache_sha
):
    raise SystemExit("FP32 projection comparison identity failed")
for name in ("cache", "states"):
    contract = comparison["hlo"][name]["contract"]
    if (
        not contract["passed"]
        or contract["bf16_wk_conversion_count"] != 0
        or contract["convolution_weight_bf16"] is not False
        or contract["convolution_weight_f32"] is not True
        or contract["physical_embedding_gather_count"] != 1
        or contract["gather_coupled_input_rms"] is not True
        or contract["physical_cache_scatter_count"] != 1
        or contract["cache_scatter_update_bf16"] is not True
        or contract["rotary"]["source_literal"] is not True
        or contract["forbidden_operations"]
        or contract["forbidden_shapes"]
        or contract["loop_count"] != 0
    ):
        raise SystemExit(f"FP32 projection {name} HLO contract failed")
state_exact = comparison["state_comparison"]["all_fields_elementwise_exact"]
cache_exact = comparison["cache_comparison"]["elementwise_exact"]
restored = bool(state_exact and cache_exact)

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

connection = pv.connect(db_path)
run_id = pv.start_run(
    connection,
    model="zai-org/GLM-5.2-FP8:greenfield-layer0-prompt-key-projection",
    revision="bounded-real-layer0-prompt-key-fp32-projection-v1",
    env={
        "GLM_ENGINE": "greenfield_layer0_prompt_key_projection_association",
        "greenfield_code_hash": pin,
        "source_run_id": int(source_run_id),
        "source_item_row_id": int(source_item_id),
        "source_comparison_manifest_sha256": source_comparison_sha,
        "accepted_capture_manifest_sha256": capture_sha,
        "prompt_cache_manifest_sha256": cache_sha,
        "comparison_manifest_sha256": comparison["manifest_sha256"],
        "candidate_cache_sha256": comparison["greenfield_cache"][
            "prompt_index_key_bfloat16_sha256"
        ],
        "projection_weight_mode": "adapted_fp32",
        "state_elementwise_exact": state_exact,
        "cache_elementwise_exact": cache_exact,
    },
    note=(
        "Protected bounded source-faithful FP32 prompt-key projection "
        "diagnostic. No decoder, Gate-D, latency, or token-rate claim."
    ),
    harness_repo=repo,
    fork_repo=None,
)
pv.record_item(
    connection,
    run_id,
    benchmark="greenfield_layer0_prompt_key_projection_association",
    item_id="adapted_fp32_m2048_gather_cache_source_rope",
    prompt="Sealed DB513 accepted layer-0 prompt-key producer and cache.",
    gold="Exact producer states at position 113 and exact 8,155-row BF16 cache.",
    raw_output=json.dumps(comparison, sort_keys=True),
    extracted=json.dumps({
        "state": comparison["state_comparison"],
        "cache": comparison["cache_comparison"],
    }, sort_keys=True),
    correct=restored,
    score=float(restored),
    n_prompt_tokens=8155,
    latency_ms=None,
)
item_row_id = connection.execute(
    "SELECT id FROM items WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,)
).fetchone()[0]
pv.finalize(
    connection,
    run_id,
    benchmark="greenfield_layer0_prompt_key_projection_association",
    metric="diagnostic_completed",
    value=1.0,
    note="Exactness is recorded separately; elapsed time is not performance.",
)
connection.close()
summary = {
    "status": "SUCCESS",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
    "results_db_item_row_id": item_row_id,
    "source_run_id": int(source_run_id),
    "source_item_row_id": int(source_item_id),
    "comparison_manifest_sha256": comparison["manifest_sha256"],
    "projection_association_restored": restored,
    "state_elementwise_exact": state_exact,
    "cache_elementwise_exact": cache_exact,
    "candidate_cache_sha256": comparison["greenfield_cache"][
        "prompt_index_key_bfloat16_sha256"
    ],
    "claim_scope": comparison["claim_scope"],
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(root / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
if sqlite3.connect(root / "results_ckpt.db").execute(
    "PRAGMA integrity_check"
).fetchone()[0] != "ok":
    raise SystemExit("projection DB snapshot failed integrity check")
print(f"PROMPT_KEY_PROJECTION_VALID db_run={run_id} restored={restored}")
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "freezing and archiving projection evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find comparison -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum comparison_summary.json summary.json source_validation.json \
    results_ckpt.db disk_preflight.txt census_pre.txt census_post.txt \
    orchestrator.sealed.log
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

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
values = {
    "artifact_kind": "glm52_layer0_prompt_key_projection_association",
    "code_hash": sys.argv[3],
    "results_db_run_id": summary["results_db_run_id"],
    "results_db_item_row_id": summary["results_db_item_row_id"],
    "source_run_id": summary["source_run_id"],
    "source_item_row_id": summary["source_item_row_id"],
    "comparison_manifest_sha256": summary["comparison_manifest_sha256"],
    "projection_association_restored": str(
        summary["projection_association_restored"]
    ).lower(),
    "candidate_cache_sha256": summary["candidate_cache_sha256"],
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
restored=$(sed -n 's/^projection_association_restored=//p' "$RUN_DIR/SUCCESS")
say "SUCCESS DB=$db_run projection_restored=$restored"
