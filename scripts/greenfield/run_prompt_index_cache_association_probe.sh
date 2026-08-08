#!/usr/bin/env bash
# Resume DB506 and isolate prompt-key projection/chunk versus norm association.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly SOURCE_TAG=greenfield_layer0_prompt_index_cache_comparison_20260808T210743348875130Z
readonly SOURCE_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_comparison/8k/$SOURCE_TAG
readonly SOURCE_CODE_HASH=cee8bda49ee1f2de83820078e2e343403d1ea0ac
readonly SOURCE_RUN_ID=506
readonly SOURCE_ITEM_ROW_ID=1789
readonly SOURCE_SUMMARY_FILE_SHA=0bd4bd9387f67650fe1584a96e27500ad34701e43e38a8fb033645068eed6bed
readonly SOURCE_COMPARISON_FILE_SHA=147133f668457bbff274875b0972e890107b7357d44abae1932841651a8ca14b
readonly SOURCE_CACHE_MANIFEST_FILE_SHA=9a8b894a38acd86d55d0909cf14b98caa3581de6be10d9fee0219f63c66b5798
readonly SOURCE_CACHE_TENSOR_FILE_SHA=bb5dd2a91c0a30f2c0e460737d5ce680c1e638f6974511e24be0bf05a1d970eb
readonly SOURCE_OBSERVED_TENSOR_FILE_SHA=4c927f20607460747ef5796c0fcb21529c7586a7c32d39c1521d484c56602832
readonly SOURCE_DB_SNAPSHOT_SHA=b0cc8d128df3d3dd0dbbeaebee0d813646d8856c6400aeb77e01417220e56d7e
readonly SOURCE_SUCCESS_SHA=f3b1f327e6e8ef529a85bd7bf1affa532a0efc5c159d74602939f7eceadd264c
readonly SOURCE_REMOTE_OBJECTS_SHA=d3cbc8e527e12dcbe3cfed36bf5214efdac0d0978b2e54ac0d9928eb9dbfa90f
readonly SOURCE_EVIDENCE_SHA=a22e1dfa32541d41fd3a9b4fcfa2f5af46e4ba76bdf5adc28e5e9ffb11c111f6
readonly SOURCE_PRE_CENSUS_SHA=1e54d7ce2fc353fbdec3e554d86c15298b3fc67a891120c41d91f82f52754700
readonly SOURCE_POST_CENSUS_SHA=6cdbd5e6583d3aa81e853af2c9b042079cd77b0834796c09c2152d4d24ced13f
readonly SOURCE_CACHE_MANIFEST_SHA=d869f6cf038e708541fa2e5633812ae00accb4fa3f31ea063d6ea60f10b9a86f
readonly SOURCE_COMPARISON_MANIFEST_SHA=b1822e71e12cf316e895538caa1dc4680672531c0378bfb7256d0314d6a83151
readonly SOURCE_LOGICAL_CACHE_SHA=3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1
readonly SOURCE_OBSERVED_CACHE_SHA=2f48fc061ccb181500fbb137fcb5d781c73df218fc9394fc06768792531fecbe
readonly MATRIX_TAG=greenfield_layer0_prompt_index_cache_association_20260808T214925579370178Z
readonly MATRIX_DIR=/home/gianl/glm-run/$MATRIX_TAG
readonly MATRIX_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_association/8k/$MATRIX_TAG
readonly MATRIX_CODE_HASH=31a23b8852031bdc3879f0e15acc02ae01fe09fe
readonly MATRIX_RUN_ID=507
readonly MATRIX_ASSOCIATION_FILE_SHA=ee0973a0cefd505c8c1d5a02853819216a08c9c3a0b6242a50fb480b076adc04
readonly MATRIX_SUMMARY_FILE_SHA=dc41b1c64c69acea7ef8b39c131f56d40a99ad684ca4fc1aeade4d4e23cee59f
readonly MATRIX_SUCCESS_SHA=6ce5298992ef665b89b0cd2dab4c2dcb5263cafca50c72110d3a65a1b7a9c227
readonly MATRIX_RESULTS_DB_SHA=b3fb207ba5508e96724b1e8104e16b3465bf0f8d9c7a26296b20d3ee62234b47
readonly MATRIX_EVIDENCE_SHA=affe842436b89686a6231ba83c78554329f70aae85212a1b17ccd5fd5bf3bcaf
readonly MATRIX_REMOTE_OBJECTS_SHA=7787dccdfda2fded013ab1dba9263e2b550ad465101adc3d5e2418d6e89ecbc0
readonly MATRIX_PRE_CENSUS_SHA=3d44069a5c345c4837719327209102f88ce47d6a25cbe8000482a5537a7df9ee
readonly MATRIX_POST_CENSUS_SHA=bb9747afc094835fd8ca3cee0cfd5f9b240c592ad11a0fd1fab66f3128eb51fd
readonly MATRIX_ASSOCIATION_MANIFEST_SHA=7216756cf364e3461755c65b99961ba50f37fe712914efad98323cca98a97cae
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly INPUT_MANIFEST_FILE_SHA=bd06714ebfe5177b8466778e2bc33ef262544dced48adcfc5739be37ac6488b9

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
PROFILE=${GLM_GREENFIELD_PROMPT_CACHE_ASSOCIATION_PROFILE:-matrix}
TAG=${GLM_GREENFIELD_PROMPT_CACHE_ASSOCIATION_TAG:-greenfield_layer0_prompt_index_cache_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_association/8k/$TAG
ASSOCIATION_DIR=$RUN_DIR/association

[[ $PROFILE == matrix || $PROFILE == chunk_parameter ]] || {
  echo "unsupported prompt-key association profile: $PROFILE" >&2
  exit 2
}

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]] || {
  echo "refusing prompt-key association from the wrong worktree" >&2
  exit 2
}
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing prompt-key association outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing prompt-key association from a dirty worktree" >&2
  exit 2
}
[[ $(git -C "$LEGACY_REPO" rev-parse HEAD) == "$LEGACY_PIN" &&
  -z $(git -C "$LEGACY_REPO" status --porcelain) ]] || {
  echo "accepted legacy oracle checkout drifted" >&2
  exit 2
}
[[ -r $RESULTS_DB && -d $SOURCE_DIR && -d $INPUT_DIR && ! -e $RUN_DIR ]] || {
  echo "association source/DB missing or append-only run path exists" >&2
  exit 2
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

mkdir -p "$RUN_DIR"
say() {
  echo "[prompt-key-association $(date -u +%H:%M:%S)] $*" |
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py|[p]robe_layer0_prompt_index_cache(_association)?[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
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

say "PIN=$PIN RUN_DIR=$RUN_DIR PROFILE=$PROFILE SOURCE_DB=$SOURCE_RUN_ID/$SOURCE_ITEM_ROW_ID"
strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}

for contract in \
  "$SOURCE_SUMMARY_FILE_SHA $SOURCE_DIR/summary.json" \
  "$SOURCE_COMPARISON_FILE_SHA $SOURCE_DIR/comparison/comparison.json" \
  "$SOURCE_CACHE_MANIFEST_FILE_SHA $SOURCE_DIR/prompt_index_cache/manifest.json" \
  "$SOURCE_CACHE_TENSOR_FILE_SHA $SOURCE_DIR/prompt_index_cache/prompt_index_cache.safetensors" \
  "$SOURCE_OBSERVED_TENSOR_FILE_SHA $SOURCE_DIR/comparison/observed_prompt_index_keys.safetensors" \
  "$SOURCE_DB_SNAPSHOT_SHA $SOURCE_DIR/results_ckpt.db" \
  "$SOURCE_SUCCESS_SHA $SOURCE_DIR/SUCCESS" \
  "$SOURCE_REMOTE_OBJECTS_SHA $SOURCE_DIR/remote_objects.json" \
  "$SOURCE_EVIDENCE_SHA $SOURCE_DIR/evidence.sha256" \
  "$SOURCE_PRE_CENSUS_SHA $SOURCE_DIR/census_pre.txt" \
  "$SOURCE_POST_CENSUS_SHA $SOURCE_DIR/census_post.txt" \
  "$INPUT_MANIFEST_FILE_SHA $INPUT_DIR/manifest.json"; do
  expected=${contract%% *}
  path=${contract#* }
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: sealed DB506 source drifted: $path"
    exit 2
  }
done
has_eight_unique_markers "$SOURCE_DIR/census_pre.txt" CENSUS_OK || exit 2
has_eight_unique_markers "$SOURCE_DIR/census_post.txt" CENSUS_OK || exit 2
remote_source_success_sha=$(gcloud storage cat \
  "$SOURCE_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $remote_source_success_sha == "$SOURCE_SUCCESS_SHA" ]] || {
  say "ABORT: approved DB506 SUCCESS drifted"
  exit 2
}

/home/gianl/vllm-env/bin/python - \
  "$RESULTS_DB" "$SOURCE_DIR" "$RUN_DIR/source_validation.json" \
  "$SOURCE_RUN_ID" "$SOURCE_ITEM_ROW_ID" "$SOURCE_CODE_HASH" \
  "$SOURCE_CACHE_MANIFEST_SHA" "$SOURCE_COMPARISON_MANIFEST_SHA" \
  "$SOURCE_LOGICAL_CACHE_SHA" "$SOURCE_OBSERVED_CACHE_SHA" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

(db_path, source_path, output_path, run_id, item_id, code_hash,
 cache_sha, comparison_sha, expected_sha, observed_sha) = sys.argv[1:]
run_id, item_id = int(run_id), int(item_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("DB506 source DB integrity failed")
run = connection.execute(
    "SELECT model,harness_git,env_json,pod,note FROM runs WHERE run_id=?",
    (run_id,),
).fetchone()
item = connection.execute(
    "SELECT benchmark,item_id,correct,score,n_prompt_tokens,latency_ms "
    "FROM items WHERE id=? AND run_id=?", (item_id, run_id),
).fetchone()
if run is None or item is None:
    raise SystemExit("DB506 source row absent")
model, harness, env_json, pod, note = run
env = json.loads(env_json)
if (
    model != "zai-org/GLM-5.2-FP8:greenfield-layer0-prompt-index-cache"
    or harness != code_hash[:7]
    or pod != "db-v4-64-od"
    or "No decoder" not in note
    or item != (
        "greenfield_layer0_prompt_index_cache_comparison",
        "layer0_prompt_positions_0_8154", 0, 0.0, 8155, None,
    )
    or env.get("greenfield_code_hash") != code_hash
    or env.get("source_cache_manifest_sha256") != cache_sha
    or env.get("comparison_manifest_sha256") != comparison_sha
):
    raise SystemExit("DB506 source provenance drifted")
root = Path(source_path)
summary = json.loads((root / "summary.json").read_text())
comparison = json.loads((root / "comparison/comparison.json").read_text())
cache = json.loads((root / "prompt_index_cache/manifest.json").read_text())
if (
    summary["results_db_run_id"] != run_id
    or summary["results_db_item_row_id"] != item_id
    or summary["elementwise_exact"] is not False
    or cache["manifest_sha256"] != cache_sha
    or cache["prompt_index_key_bfloat16_sha256"] != expected_sha
    or comparison["manifest_sha256"] != comparison_sha
    or comparison["comparison"]["observed_bfloat16_sha256"] != observed_sha
    or comparison["comparison"]["mismatch_count"] != 4058
):
    raise SystemExit("DB506 source artifact identity drifted")
Path(output_path).write_text(json.dumps({
    "status": "SUCCESS",
    "source_run_id": run_id,
    "source_item_row_id": item_id,
    "source_code_hash": code_hash,
    "source_db_integrity": "ok",
    "source_cache_manifest_sha256": cache_sha,
    "source_comparison_manifest_sha256": comparison_sha,
}, indent=2, sort_keys=True) + "\n")
connection.close()
PY

if [[ $PROFILE == chunk_parameter ]]; then
  for contract in \
    "$MATRIX_ASSOCIATION_FILE_SHA $MATRIX_DIR/association/association.json" \
    "$MATRIX_SUMMARY_FILE_SHA $MATRIX_DIR/summary.json" \
    "$MATRIX_SUCCESS_SHA $MATRIX_DIR/SUCCESS" \
    "$MATRIX_RESULTS_DB_SHA $MATRIX_DIR/results_ckpt.db" \
    "$MATRIX_EVIDENCE_SHA $MATRIX_DIR/evidence.sha256" \
    "$MATRIX_REMOTE_OBJECTS_SHA $MATRIX_DIR/remote_objects.json" \
    "$MATRIX_PRE_CENSUS_SHA $MATRIX_DIR/census_pre.txt" \
    "$MATRIX_POST_CENSUS_SHA $MATRIX_DIR/census_post.txt"; do
    expected=${contract%% *}
    path=${contract#* }
    [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
      say "ABORT: sealed DB507 matrix drifted: $path"
      exit 2
    }
  done
  matrix_remote_success_sha=$(gcloud storage cat \
    "$MATRIX_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $matrix_remote_success_sha == "$MATRIX_SUCCESS_SHA" ]] || {
    say "ABORT: approved DB507 SUCCESS drifted"
    exit 2
  }
  /home/gianl/vllm-env/bin/python - \
    "$RESULTS_DB" "$MATRIX_DIR" "$RUN_DIR/matrix_validation.json" \
    "$MATRIX_RUN_ID" "$MATRIX_CODE_HASH" \
    "$MATRIX_ASSOCIATION_MANIFEST_SHA" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

db_path, matrix_path, output_path, run_id, code_hash, manifest_sha = sys.argv[1:]
run_id = int(run_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("DB507 matrix source DB integrity failed")
run = connection.execute(
    "SELECT harness_git,env_json,pod FROM runs WHERE run_id=?", (run_id,)
).fetchone()
items = connection.execute(
    "SELECT id,item_id,correct,score FROM items WHERE run_id=? ORDER BY id",
    (run_id,),
).fetchall()
expected_items = [
    (1790, "accepted_xla_m2048_divide_sqrt", 0, 0.0),
    (1791, "accepted_xla_m2048_multiply_rsqrt", 0, 0.0),
    (1792, "production_pallas_m1_divide_sqrt", 0, 0.0),
]
if run is None or items != expected_items:
    raise SystemExit("DB507 matrix rows drifted")
harness, env_json, pod = run
env = json.loads(env_json)
if (
    harness != code_hash[:7]
    or pod != "db-v4-64-od"
    or env.get("association_manifest_sha256") != manifest_sha
    or env.get("exact_candidates") != []
    or env.get("classification") != "declared_matrix_not_sufficient"
):
    raise SystemExit("DB507 matrix provenance drifted")
association = json.loads(
    (Path(matrix_path) / "association/association.json").read_text()
)
best = association["candidates"]["accepted_xla_m2048_divide_sqrt"]
if (
    association["manifest_sha256"] != manifest_sha
    or association["conclusion"] != {
        "classification": "declared_matrix_not_sufficient",
        "exact_candidates": [],
    }
    or best["comparison_to_accepted"]["mismatch_count"] != 45
    or best["observed_bfloat16_sha256"]
    != "52bf55ed5e9ea74a59551351a21ae830fb39e289e82eaff636fb9ab84d7dcd8a"
):
    raise SystemExit("DB507 matrix artifact drifted")
Path(output_path).write_text(json.dumps({
    "status": "SUCCESS",
    "matrix_run_id": run_id,
    "matrix_code_hash": code_hash,
    "matrix_association_manifest_sha256": manifest_sha,
    "matrix_best_mismatch_count": 45,
}, indent=2, sort_keys=True) + "\n")
connection.close()
PY
fi

say "syncing exact greenfield pin to all eight hosts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; idx=${HOSTNAME##*-w-}; if [[ "$idx" == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host code sync failed"
  exit 1
}

say "running three-candidate association matrix on one TPU-v4 host"
started=$(date +%s)
env JAX_PLATFORMS=tpu \
  TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH="$WORKTREE" \
  timeout --signal=TERM --kill-after=60 1800 \
  /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/probe_layer0_prompt_index_cache_association.py" \
  --expected-code-hash "$PIN" \
  --run-tag "$TAG" \
  --input-dir "$INPUT_DIR" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
  --prompt-cache-dir "$SOURCE_DIR/prompt_index_cache" \
  --prompt-cache-manifest-sha256 "$SOURCE_CACHE_MANIFEST_SHA" \
  --baseline-comparison-dir "$SOURCE_DIR/comparison" \
  --baseline-comparison-manifest-sha256 "$SOURCE_COMPARISON_MANIFEST_SHA" \
  --candidate-set "$PROFILE" \
  --output "$ASSOCIATION_DIR" >"$RUN_DIR/association_summary.json"
elapsed=$(( $(date +%s) - started ))
say "association matrix completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$LEGACY_REPO" \
  "$elapsed" "$SOURCE_RUN_ID" "$SOURCE_ITEM_ROW_ID" \
  "$SOURCE_CACHE_MANIFEST_SHA" "$SOURCE_COMPARISON_MANIFEST_SHA" \
  "$PROFILE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

(run_path, pin, db_path, repo, legacy_repo, elapsed, source_run_id,
 source_item_id, cache_sha, baseline_sha, profile) = sys.argv[1:]
root = Path(run_path)
association = json.loads((root / "association/association.json").read_text())
expected_names = (
    {
        "production_pallas_m1_divide_sqrt",
        "accepted_xla_m2048_divide_sqrt",
        "accepted_xla_m2048_multiply_rsqrt",
    }
    if profile == "matrix"
    else {"accepted_xla_m2048_chunk_parameter_divide_sqrt"}
)
if (
    association["status"] != "SUCCESS"
    or association["code_hash"] != pin
    or association["backend"] != "tpu"
    or association["device_count"] != 4
    or association["diagnostic_only"] is not True
    or association["performance_claim"] is not False
    or association["prompt_cache_manifest_sha256"] != cache_sha
    or association["baseline"]["comparison_manifest_sha256"] != baseline_sha
    or association["candidate_set"] != profile
    or set(association["candidates"]) != expected_names
    or association["accepted_adapted_wk"]["byte_sum"] != 193298069
):
    raise SystemExit("prompt-key association result contract failed")
for name, record in association["candidates"].items():
    if not record["hlo"]["contract"]["passed"]:
        raise SystemExit(f"prompt-key association HLO failed: {name}")

source_run_id, source_item_id = int(source_run_id), int(source_item_id)
sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

connection = pv.connect(db_path)
run_id = pv.start_run(
    connection,
    model=(
        "zai-org/GLM-5.2-FP8:greenfield-layer0-prompt-key-association"
    ),
    revision=f"bounded-real-layer0-prompt-key-association-{profile}-v1",
    env={
        "GLM_ENGINE": "greenfield_layer0_prompt_index_cache_association",
        "greenfield_code_hash": pin,
        "source_run_id": source_run_id,
        "source_item_row_id": source_item_id,
        "source_cache_manifest_sha256": cache_sha,
        "baseline_comparison_manifest_sha256": baseline_sha,
        "association_manifest_sha256": association["manifest_sha256"],
        "exact_candidates": association["conclusion"]["exact_candidates"],
        "classification": association["conclusion"]["classification"],
        "candidate_set": profile,
        "matrix_source_run_id": 507 if profile == "chunk_parameter" else None,
    },
    note=(
        "Protected bounded layer-0 prompt-key association matrix. No decoder, "
        "Gate-D, latency, or token-rate claim."
    ),
    harness_repo=repo,
    fork_repo=legacy_repo,
)
item_rows = {}
for name in sorted(expected_names):
    record = association["candidates"][name]
    exact = bool(record["comparison_to_accepted"]["elementwise_exact"])
    pv.record_item(
        connection,
        run_id,
        benchmark="greenfield_layer0_prompt_index_cache_association",
        item_id=name,
        prompt="Immutable DB506 accepted prompt cache association candidate.",
        gold="Elementwise-exact accepted layer-0 BF16 prompt index cache.",
        raw_output=json.dumps(record, sort_keys=True),
        extracted=json.dumps(record["comparison_to_accepted"], sort_keys=True),
        correct=exact,
        score=float(exact),
        n_prompt_tokens=8155,
        latency_ms=None,
    )
    item_rows[name] = connection.execute(
        "SELECT id FROM items WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,)
    ).fetchone()[0]
pv.finalize(
    connection,
    run_id,
    benchmark="greenfield_layer0_prompt_index_cache_association",
    metric="diagnostic_completed",
    value=1.0,
    note="Exactness is per item; execution times are not performance.",
)
connection.close()
summary = {
    "status": "SUCCESS",
    "code_hash": pin,
    "elapsed_seconds": int(elapsed),
    "results_db_run_id": run_id,
    "results_db_item_row_ids": item_rows,
    "source_run_id": source_run_id,
    "source_item_row_id": source_item_id,
    "prompt_cache_manifest_sha256": cache_sha,
    "baseline_comparison_manifest_sha256": baseline_sha,
    "association_manifest_sha256": association["manifest_sha256"],
    "candidate_set": profile,
    "conclusion": association["conclusion"],
    "candidate_comparisons": {
        name: value["comparison_to_accepted"]
        for name, value in association["candidates"].items()
    },
    "claim_scope": association["claim_scope"],
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(root / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
check = sqlite3.connect(root / "results_ckpt.db")
if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("prompt-key association DB snapshot integrity failed")
check.close()
print(
    f"PROMPT_KEY_ASSOCIATION_VALID db_run={run_id} "
    f"classification={association['conclusion']['classification']}"
)
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "freezing and archiving prompt-key association evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find association -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum association_summary.json source_validation.json summary.json \
    results_ckpt.db census_pre.txt census_post.txt sync.txt \
    orchestrator.sealed.log
  if [[ -f matrix_validation.json ]]; then
    sha256sum matrix_validation.json
  fi
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
    "artifact_kind": "glm52_layer0_prompt_index_cache_association",
    "code_hash": sys.argv[3],
    "source_run_id": summary["source_run_id"],
    "source_item_row_id": summary["source_item_row_id"],
    "results_db_run_id": summary["results_db_run_id"],
    "results_db_item_row_ids": json.dumps(
        summary["results_db_item_row_ids"], sort_keys=True,
        separators=(",", ":"),
    ),
    "prompt_cache_manifest_sha256": summary[
        "prompt_cache_manifest_sha256"
    ],
    "baseline_comparison_manifest_sha256": summary[
        "baseline_comparison_manifest_sha256"
    ],
    "association_manifest_sha256": summary[
        "association_manifest_sha256"
    ],
    "candidate_set": summary["candidate_set"],
    "classification": summary["conclusion"]["classification"],
    "exact_candidates": ",".join(summary["conclusion"]["exact_candidates"]),
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
classification=$(sed -n 's/^classification=//p' "$RUN_DIR/SUCCESS")
exact=$(sed -n 's/^exact_candidates=//p' "$RUN_DIR/SUCCESS")
say "SUCCESS DB=$db_run classification=$classification exact=$exact"
