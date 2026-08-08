#!/usr/bin/env bash
# Resume the bounded prompt-cache comparison from an immutable accepted capture.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly SOURCE_TAG=greenfield_legacy_layer0_prompt_index_cache_20260808T193700214955997Z
readonly SOURCE_RUN_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_DUMP_DIR=$SOURCE_RUN_DIR/source_dumps
readonly SOURCE_RUN_ID=505
readonly SOURCE_ITEM_ROW_ID=1788
readonly SOURCE_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache/8k/$SOURCE_TAG/diagnostic_local/$SOURCE_TAG
readonly SOURCE_CAPTURE_SHA=06c2601b303a7d7e6c3dfbd13408f9cefdbdad77bb83c6d5806600d63875f1e6
readonly SOURCE_IDENTITY_SHA=23d949c54f2abeaebac86fceb8bae938c63919856fabe4e514c3a3a1dd2ab379
readonly SOURCE_LEGACY_SUMMARY_SHA=9f166f41eda7855208adea77ce4b7ee407356207316ed7a825330f8404aeb327
readonly SOURCE_PRE_CENSUS_SHA=92ca3d6777af74f0ffea43b2d8bdd572cbdc6a415bb1e11cbf63478fe6f0ceb1
readonly SOURCE_FAILURE_CENSUS_SHA=da5567d17c9fc8265d7451b56dbb1d47af0ecd2f00a153153dcce605a4092cc9
readonly SOURCE_STOP_SHA=e1e0c27b3d87d79909f356fee24936286bfe6d11685056b4cf6eda04a284b916
readonly SOURCE_FLEET_INTEGRITY_SHA=c1c3c24a36f0058a3ccc518b7c44ec1a88aa788c4d04d4e28252a61f26a94d34
readonly SOURCE_CACHE_INTEGRITY_SHA=3a57728a720c221c8ae6c188352657135834137c32a53eb310cd95129a3a3e79
readonly SOURCE_ORACLE_MANIFEST_SHA=0b3489822f0cbd8931e4c9f1dd4732307ebaa4e1cad11d6c08b1072a9f9e63f7
readonly SOURCE_ORACLE_ROW_SHA=a5800577abde853bb08dfbcc8c842a40d294eebd3d16297fcd9b5f929c11ed5f
readonly SOURCE_GLOBAL_CACHE_SHA=c65552a63249d81fd9a9251c55c1254462702554e9ff1767febe35d68b81dad9
readonly SOURCE_BLOCK_TABLE_SHA=eedb3f9250bd5c915ce67ac4041d9d0c4242dd8264de1dd0a5b15bd957bd8a84
readonly SOURCE_LOGICAL_CACHE_SHA=3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly INPUT_MANIFEST_FILE_SHA=bd06714ebfe5177b8466778e2bc33ef262544dced48adcfc5739be37ac6488b9

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_PROMPT_CACHE_COMPARISON_TAG:-greenfield_layer0_prompt_index_cache_comparison_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_comparison/8k/$TAG
PROMPT_CACHE_DIR=$RUN_DIR/prompt_index_cache
COMPARISON_DIR=$RUN_DIR/comparison

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]] || {
  echo "refusing prompt-cache comparison from the wrong worktree" >&2
  exit 2
}
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing prompt-cache comparison outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing prompt-cache comparison from a dirty worktree" >&2
  exit 2
}
[[ $(git -C "$LEGACY_REPO" rev-parse HEAD) == "$LEGACY_PIN" ]] || {
  echo "accepted legacy checkout drifted" >&2
  exit 2
}
[[ -z $(git -C "$LEGACY_REPO" status --porcelain) ]] || {
  echo "accepted legacy checkout is dirty" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only comparison directory exists: $RUN_DIR" >&2
  exit 2
}
for path in "$RESULTS_DB" "$SOURCE_RUN_DIR/capture.json" \
  "$SOURCE_RUN_DIR/source_identity.json" "$SOURCE_RUN_DIR/legacy_summary.json" \
  "$SOURCE_RUN_DIR/census_pre.txt" "$SOURCE_RUN_DIR/census_failure_exit.txt" \
  "$SOURCE_RUN_DIR/stop.txt" "$SOURCE_RUN_DIR/fleet_integrity.txt" \
  "$SOURCE_RUN_DIR/fleet_prompt_cache_integrity.txt" \
  "$SOURCE_RUN_DIR/oracle/manifest.json" \
  "$SOURCE_RUN_DIR/oracle/source_row.json" "$INPUT_DIR/manifest.json"; do
  [[ -r $path ]] || {
    echo "prompt-cache comparison input is unavailable: $path" >&2
    exit 2
  }
done
[[ ! -e $SOURCE_RUN_DIR/SUCCESS ]] || {
  echo "source refusal unexpectedly has final SUCCESS" >&2
  exit 2
}
[[ ! -s $SOURCE_RUN_DIR/prompt_index_cache_capture.json ]] || {
  echo "source refusal unexpectedly sealed a prompt-cache artifact" >&2
  exit 2
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

mkdir -p "$RUN_DIR"
say() {
  echo "[prompt-cache-comparison $(date -u +%H:%M:%S)] $*" |
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py|[p]robe_layer0_prompt_index_cache[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
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

say "PIN=$PIN RUN_DIR=$RUN_DIR"
say "SOURCE=$SOURCE_RUN_DIR SOURCE_DB=$SOURCE_RUN_ID/$SOURCE_ITEM_ROW_ID"
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}

for contract in \
  "$SOURCE_CAPTURE_SHA $SOURCE_RUN_DIR/capture.json" \
  "$SOURCE_IDENTITY_SHA $SOURCE_RUN_DIR/source_identity.json" \
  "$SOURCE_LEGACY_SUMMARY_SHA $SOURCE_RUN_DIR/legacy_summary.json" \
  "$SOURCE_PRE_CENSUS_SHA $SOURCE_RUN_DIR/census_pre.txt" \
  "$SOURCE_FAILURE_CENSUS_SHA $SOURCE_RUN_DIR/census_failure_exit.txt" \
  "$SOURCE_STOP_SHA $SOURCE_RUN_DIR/stop.txt" \
  "$SOURCE_FLEET_INTEGRITY_SHA $SOURCE_RUN_DIR/fleet_integrity.txt" \
  "$SOURCE_CACHE_INTEGRITY_SHA $SOURCE_RUN_DIR/fleet_prompt_cache_integrity.txt" \
  "$SOURCE_ORACLE_MANIFEST_SHA $SOURCE_RUN_DIR/oracle/manifest.json" \
  "$SOURCE_ORACLE_ROW_SHA $SOURCE_RUN_DIR/oracle/source_row.json" \
  "$INPUT_MANIFEST_FILE_SHA $INPUT_DIR/manifest.json"; do
  expected=${contract%% *}
  path=${contract#* }
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: sealed source identity drifted: $path"
    exit 2
  }
done
has_eight_unique_markers "$SOURCE_RUN_DIR/census_pre.txt" CENSUS_OK || {
  say "ABORT: source pre-census is incomplete"
  exit 2
}
has_eight_unique_markers "$SOURCE_RUN_DIR/census_failure_exit.txt" CENSUS_OK || {
  say "ABORT: source failure census is incomplete"
  exit 2
}
has_eight_unique_markers "$SOURCE_RUN_DIR/stop.txt" STOP_OK || {
  say "ABORT: source owned-runtime stop is incomplete"
  exit 2
}

source_count=$(find "$SOURCE_DUMP_DIR" -type f \
  -name 'index_cache.postfwd.step*.proc*.npz' | wc -l)
final_count=$(find "$SOURCE_DUMP_DIR" -type f \
  -name 'index_cache.postfwd.step0004.proc*.npz' | wc -l)
[[ $source_count -eq 32 && $final_count -eq 8 ]] || {
  say "ABORT: source cache coverage drifted total=$source_count final=$final_count"
  exit 2
}
(
  cd "$SOURCE_RUN_DIR"
  find source_dumps -type f \
    -name 'index_cache.postfwd.step*.proc*.npz' -print0 |
    sort -z | xargs -0 sha256sum
) >"$RUN_DIR/source_cache_files.sha256"
grep 'index_cache.postfwd.step0004' "$RUN_DIR/source_cache_files.sha256" \
  >"$RUN_DIR/source_final_files.sha256"

say "verifying eight immutable final snapshots against the approved archive"
while read -r expected relative; do
  remote_sha=$(gcloud storage cat "$SOURCE_REMOTE_PREFIX/$relative" |
    sha256sum | awk '{print $1}')
  [[ $remote_sha == "$expected" ]] || {
    say "ABORT: approved source object drifted: $relative"
    exit 2
  }
  printf '%s  %s\n' "$remote_sha" "$relative"
done <"$RUN_DIR/source_final_files.sha256" \
  >"$RUN_DIR/remote_source_final_files.sha256"
cmp "$RUN_DIR/source_final_files.sha256" \
  "$RUN_DIR/remote_source_final_files.sha256"

/home/gianl/vllm-env/bin/python - \
  "$RESULTS_DB" "$SOURCE_RUN_DIR" "$RUN_DIR/source_validation.json" \
  "$SOURCE_TAG" "$SOURCE_RUN_ID" "$SOURCE_ITEM_ROW_ID" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

db_path, source_path, output_path, tag, run_id, item_id = sys.argv[1:]
run_id = int(run_id)
item_id = int(item_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("source DB integrity failed")
run = connection.execute(
    "SELECT model,harness_git,fork_git,env_json,pod,note FROM runs "
    "WHERE run_id=?", (run_id,)
).fetchone()
if run is None:
    raise SystemExit("source run is absent")
model, harness, fork, env_json, pod, note = run
env = json.loads(env_json)
item = connection.execute(
    "SELECT benchmark,item_id,correct,score,n_prompt_tokens,n_gen_tokens,"
    "raw_output,extracted FROM items WHERE id=? AND run_id=?",
    (item_id, run_id),
).fetchone()
if (
    model != "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
    or harness != "a4a17ac"
    or fork != "b3c25df47"
    or pod != "db-v4-64-od"
    or tag not in note
    or env.get("os_env", {}).get("GLM_GREENFIELD_PROMPT_CACHE_CAPTURE") != "1"
    or env.get("os_env", {}).get("GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG") != tag
):
    raise SystemExit("source run provenance drifted")
if item is None or item[:6] != (
    "passkey_L8192_d0.5", "t0", 1, 1.0, 8155, 20
) or item[7] != "881446" or not item[6].lstrip().startswith("881446"):
    raise SystemExit("source item identity/correctness drifted")
capture = json.loads((Path(source_path) / "capture.json").read_text())
identity = json.loads((Path(source_path) / "source_identity.json").read_text())
if (
    capture != {
        "decode_step_count": 14,
        "event_count": 21,
        "manifest_sha256": "5687ce4b005cd6c51a3ab025f9548db32100538d52ae7c5fe199775a57ec0316",
        "source_dump_file_count": 294,
    }
    or identity["item_row_id"] != item_id
    or identity["correct"] != 1
    or identity["n_prompt_tokens"] != 8155
    or identity["n_gen_tokens"] != 20
):
    raise SystemExit("source capture identity drifted")
value = {
    "status": "SUCCESS",
    "source_capture_failed_closed_after_oracle": True,
    "source_run_id": run_id,
    "source_item_row_id": item_id,
    "source_tag": tag,
    "source_db_integrity": "ok",
    "source_harness_git": harness,
    "source_fork_git": fork,
    "source_item_correct": True,
    "source_prompt_tokens": 8155,
    "source_generated_tokens": 20,
    "source_final_cache_file_count": 8,
    "source_all_cache_file_count": 32,
}
Path(output_path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
connection.close()
PY

say "sealing accepted logical layer-0 prompt index cache"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/capture_legacy_prompt_index_cache.py" \
  --source-dump-dir "$SOURCE_DUMP_DIR" \
  --output "$PROMPT_CACHE_DIR" \
  --expected-code-hash "$PIN" \
  --legacy-repository-pin "$LEGACY_PIN" \
  --run-tag "$TAG" \
  --source-run-id "$SOURCE_RUN_ID" \
  --source-item-row-id "$SOURCE_ITEM_ROW_ID" \
  --layer0-input-dir "$INPUT_DIR" \
  --layer0-input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
  >"$RUN_DIR/capture_summary.json"

prompt_cache_manifest_sha=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$PROMPT_CACHE_DIR/manifest.json")
/home/gianl/vllm-env/bin/python - "$PROMPT_CACHE_DIR/manifest.json" \
  "$SOURCE_GLOBAL_CACHE_SHA" "$SOURCE_BLOCK_TABLE_SHA" \
  "$SOURCE_LOGICAL_CACHE_SHA" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))
layout = value["source_layout"]
if (
    value["prompt_index_key_bfloat16_sha256"] != sys.argv[4]
    or layout["global_cache_bfloat16_sha256"] != sys.argv[2]
    or layout["block_tables_sha256"] != sys.argv[3]
    or layout["global_cache_shape"] != [24, 16, 32, 128]
    or layout["mesh_shape"] != {
        "data": 1, "attn_dp": 1, "attn_dp_expert": 1,
        "expert": 1, "model": 32, "dcp": 1,
    }
    or layout["local_replication_per_process"] != 4
    or layout["physical_replication"] != 32
    or layout["physical_device_count"] != 32
    or value["numerical_contract"]["logical_page_size"] != 512
):
    raise SystemExit("sealed accepted cache layout/content drifted")
PY

say "running production one-row prompt-key scan on one four-chip TPU host"
started=$(date +%s)
env JAX_PLATFORMS=tpu \
  TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH="$WORKTREE" \
  timeout --signal=TERM --kill-after=60 1800 \
  /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/probe_layer0_prompt_index_cache.py" \
  --expected-code-hash "$PIN" \
  --run-tag "$TAG" \
  --input-dir "$INPUT_DIR" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
  --prompt-cache-dir "$PROMPT_CACHE_DIR" \
  --prompt-cache-manifest-sha256 "$prompt_cache_manifest_sha" \
  --output "$COMPARISON_DIR" \
  >"$RUN_DIR/comparison_summary.json"
elapsed=$(( $(date +%s) - started ))
say "production comparison completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$LEGACY_REPO" \
  "$elapsed" "$SOURCE_RUN_ID" "$SOURCE_ITEM_ROW_ID" \
  "$prompt_cache_manifest_sha" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

(
    run_path, pin, db_path, repo, legacy_repo, elapsed,
    source_run_id, source_item_id, cache_manifest_sha,
) = sys.argv[1:]
root = Path(run_path)
comparison = json.loads((root / "comparison" / "comparison.json").read_text())
cache = json.loads((root / "prompt_index_cache" / "manifest.json").read_text())
if (
    comparison["status"] != "SUCCESS"
    or comparison["code_hash"] != pin
    or comparison["backend"] != "tpu"
    or comparison["device_count"] != 4
    or comparison["diagnostic_only"] is not True
    or comparison["performance_claim"] is not False
    or comparison["prompt_cache_manifest_sha256"] != cache_manifest_sha
    or cache["manifest_sha256"] != cache_manifest_sha
    or not comparison["hlo"]["contract"]["passed"]
    or comparison["numerical_contract"]["one_live_row"] is not True
):
    raise SystemExit("production prompt-cache comparison contract failed")
source_run_id = int(source_run_id)
source_item_id = int(source_item_id)
sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

connection = pv.connect(db_path)
run_id = pv.start_run(
    connection,
    model="zai-org/GLM-5.2-FP8:greenfield-layer0-prompt-index-cache",
    revision="bounded-real-layer0-prompt-cache-v1",
    env={
        "GLM_ENGINE": "greenfield_layer0_prompt_index_cache_comparison",
        "greenfield_code_hash": pin,
        "source_run_id": source_run_id,
        "source_item_row_id": source_item_id,
        "source_cache_manifest_sha256": cache_manifest_sha,
        "source_global_cache_bfloat16_sha256": cache["source_layout"][
            "global_cache_bfloat16_sha256"
        ],
        "source_physical_replication": cache["source_layout"][
            "physical_replication"
        ],
        "source_logical_page_size": cache["numerical_contract"][
            "logical_page_size"
        ],
        "comparison_manifest_sha256": comparison["manifest_sha256"],
        "comparison_hlo_sha256": comparison["hlo"][
            "optimized_hlo_sha256"
        ],
        "device_kind": comparison["device_kind"],
    },
    note=(
        "Protected bounded accepted-vs-production layer-0 8K prompt-cache "
        "comparison. No decoder, Gate-D, latency, or token-rate claim."
    ),
    harness_repo=repo,
    fork_repo=legacy_repo,
)
exact = bool(comparison["comparison"]["elementwise_exact"])
pv.record_item(
    connection,
    run_id,
    benchmark="greenfield_layer0_prompt_index_cache_comparison",
    item_id="layer0_prompt_positions_0_8154",
    prompt=(
        "Immutable accepted DB505/item1788 layer-0 BF16 prompt key cache "
        "versus the independent production one-row raw-FP8 scan."
    ),
    gold="Elementwise-exact accepted layer-0 BF16 prompt index cache.",
    raw_output=json.dumps(comparison, sort_keys=True),
    extracted=json.dumps(comparison["comparison"], sort_keys=True),
    correct=exact,
    score=float(exact),
    n_prompt_tokens=8155,
    latency_ms=None,
)
item_row_id = connection.execute(
    "SELECT id FROM items WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,)
).fetchone()[0]
pv.finalize(
    connection,
    run_id,
    benchmark="greenfield_layer0_prompt_index_cache_comparison",
    metric="diagnostic_completed",
    value=1.0,
    note="Exactness is stored on the item; execution time is not performance.",
)
connection.close()
summary = {
    "status": "SUCCESS",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
    "results_db_item_row_id": item_row_id,
    "source_run_id": source_run_id,
    "source_item_row_id": source_item_id,
    "prompt_cache_manifest_sha256": cache_manifest_sha,
    "comparison_manifest_sha256": comparison["manifest_sha256"],
    "elementwise_exact": exact,
    "comparison": comparison["comparison"],
    "chunk_comparisons": comparison["chunk_comparisons"],
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
    raise SystemExit("prompt-cache comparison DB snapshot failed integrity")
print(f"PROMPT_CACHE_COMPARISON_VALID db_run={run_id} exact={exact}")
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "freezing and archiving prompt-cache comparison evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find comparison prompt_index_cache -type f -print0 |
    sort -z | xargs -0 sha256sum
  sha256sum capture_summary.json comparison_summary.json source_validation.json \
    source_cache_files.sha256 source_final_files.sha256 \
    remote_source_final_files.sha256 summary.json results_ckpt.db \
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

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
comparison = summary["comparison"]
values = {
    "artifact_kind": "glm52_layer0_prompt_index_cache_comparison",
    "code_hash": sys.argv[3],
    "source_run_id": summary["source_run_id"],
    "source_item_row_id": summary["source_item_row_id"],
    "results_db_run_id": summary["results_db_run_id"],
    "results_db_item_row_id": summary["results_db_item_row_id"],
    "prompt_cache_manifest_sha256": summary["prompt_cache_manifest_sha256"],
    "comparison_manifest_sha256": summary["comparison_manifest_sha256"],
    "elementwise_exact": str(summary["elementwise_exact"]).lower(),
    "first_mismatch_position": (
        "none" if comparison["first_mismatch_position"] is None
        else comparison["first_mismatch_position"]
    ),
    "mismatch_count": comparison["mismatch_count"],
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
exact=$(sed -n 's/^elementwise_exact=//p' "$RUN_DIR/SUCCESS")
first=$(sed -n 's/^first_mismatch_position=//p' "$RUN_DIR/SUCCESS")
say "SUCCESS DB=$db_run exact=$exact first_mismatch=$first"
