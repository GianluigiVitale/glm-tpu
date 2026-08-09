#!/usr/bin/env bash
# Protected one-host discriminator for physical-M64 projection plus key norm.
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
readonly OBSERVER_PIN=89fc453b6116ac3df71e666db6f4659775b313c3
readonly SOURCE_TAG=greenfield_legacy_layer0_prompt_projection_input_p113_20260809T050055956585082Z
readonly SOURCE_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_COMPARISON_DIR=$SOURCE_DIR/prompt_projection_input_comparison
readonly SOURCE_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_projection_input/8k/$SOURCE_TAG
readonly SOURCE_GREENFIELD_PIN=b8e30ed41e46816461c7e956e4dd7bc85e96d998
readonly SOURCE_HARNESS_PIN=a4a17ac4e90b15f1994bd8b26917ef62daa52660
readonly SOURCE_RUN_ID=515
readonly SOURCE_ITEM_ROW_ID=1800
readonly SOURCE_SUCCESS_SHA=048528e3a985807e672305682fd128ef3972676865cc1108085a8175620cad79
readonly SOURCE_REMOTE_OBJECTS_SHA=06c0b778bbad3eb29045a73d64b5cdf53868ce464d0c5cfd7a192940666a5b6d
readonly SOURCE_DB_SHA=43fdf1215c8c6981d9430b25d34137b2bafa6c61e4b3a668ffc962c9d0ac01c0
readonly SOURCE_COMPARISON_FILE_SHA=bf7b49ab57d69470aa3dd253b41df281479600f1319a6e70b5561f974448a0e3
readonly SOURCE_COMPARISON_MANIFEST_SHA=df048dd7d266c7a393b5efe0835c511665edfadc21521a68f4bb3be7f6b6f258
readonly SOURCE_CAPTURE_FILE_SHA=b54330ff83726d05aff76767da83bf61f3c9f8046637c811e7c8679d56d2aebb
readonly SOURCE_CAPTURE_TENSOR_SHA=753e63d947aa94673389117f94bf4c19cc049fa06f6b95f8d50c977d659f8f2d
readonly SOURCE_CAPTURE_MANIFEST_SHA=64320e979288fa524c495d0436dc6bcf1a908c8b68d976eb014e207866312ef9
readonly SOURCE_CACHE_MANIFEST_FILE_SHA=372d0ad2503860b6fb826045de7d24ca0f22438b8e945af2515e1c8cf3d69c94
readonly SOURCE_CACHE_TENSOR_SHA=36303f0638661b4a56d3c9a1d4023a9b39eb19dbfd6d48e0718c29a45c41c07a
readonly SOURCE_CACHE_MANIFEST_SHA=acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab
readonly ACCEPTED_CACHE_SHA=3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1
readonly SEALED_BF16_CACHE_SHA=52bf55ed5e9ea74a59551351a21ae830fb39e289e82eaff636fb9ab84d7dcd8a
readonly SOURCE_PRE_CENSUS_SHA=d3f0f70f84d09109f0745a042b3da5a3dfff39969d054452a2bbe78b5e6a6eb7
readonly SOURCE_POST_CENSUS_SHA=2787e4ffd1058910c2e85cfd1470ebc6f4d44f9a22f4468551c4c3868aa408ba
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly INPUT_MANIFEST_FILE_SHA=bd06714ebfe5177b8466778e2bc33ef262544dced48adcfc5739be37ac6488b9
readonly LOWERING_TAG=greenfield_accepted_prompt_projection_lowering_recovery_20260809T105858202006975Z
readonly LOWERING_DIR=/home/gianl/glm-run/$LOWERING_TAG
readonly LOWERING_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_projection_lowering/8k/$LOWERING_TAG
readonly LOWERING_SUCCESS_SHA=6502bbffb880549011fc43bf106d3f3b36b3fa8df3059dd4fcfaa04d8e7353d8
readonly LOWERING_SUMMARY_SHA=8b8356b12885cd444a6c39cd8edfa36b3b98ef276c04c42791e8d99250415aa8
readonly LOWERING_MANIFEST_FILE_SHA=222b91d62e422cef8fe9f574bf044567600ba876f577de069f6467d5a56df247
readonly LOWERING_MANIFEST_SHA=d9b492ee2296deaf9c8acea67f62897809cc283d354a0802d1c26023faeefba6
readonly PROJECTION_TAG=greenfield_layer0_prompt_key_projection_m64_20260809T115707855267296Z
readonly PROJECTION_DIR=/home/gianl/glm-run/$PROJECTION_TAG
readonly PROJECTION_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_key_projection_m64/8k/$PROJECTION_TAG
readonly PROJECTION_CODE_HASH=5926b05de3858488d208c4b47df03403fce181c5
readonly PROJECTION_RUN_ID=517
readonly PROJECTION_ITEM_ROW_ID=1802
readonly PROJECTION_SUCCESS_SHA=79f0ea68b66f4a152127b92ea936caeb8505815f37f1ea856c4a9ce1742703c0
readonly PROJECTION_SUMMARY_SHA=7e7c014c37e0ec0485d35a8d4f257443906d2bc30e7692d5a4da3db95858638b
readonly PROJECTION_COMPARISON_SHA=fc2e4e96f67548f39b04ab471326b6406351b17420af6dbaec7a922e3fa7236b
readonly PROJECTION_MANIFEST_SHA=f3587bd40561c45478750e8987fa73cecb25fd4dcf9000bf3feb13f8bfc3fecc
readonly PROJECTION_CACHE_SHA=94cd84d56af875faf062e8ddb46179bd344f4f07adea1776fd9160a561338319
readonly PROJECTION_REMOTE_OBJECTS_SHA=cc42e8e9d19730ec9fafc92a6227d44e0edc814c0c62eecbdf58dfca8257c0de
readonly PROJECTION_DB_SHA=e84093ac2a7f36757364e2f7d4bad06b830b0c0488439101256256e39049dd0e
readonly PROJECTION_PRE_CENSUS_SHA=95dce16ac3a78a3a92462d963a8b182e51cc35d488fc25f08098ed5df5e5d86b
readonly PROJECTION_POST_CENSUS_SHA=7b6d2278094271fe3186e1470a09d5afad65522fcaaf82d475d809729ca8234a

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
readonly WEIGHT_SOURCE=${GLM_GREENFIELD_PROMPT_KEY_WEIGHT_SOURCE:-materialized_parameter}
TAG=${GLM_GREENFIELD_PROMPT_KEY_NORM_TAG:-greenfield_layer0_prompt_key_norm_m64_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_key_norm_m64/8k/$TAG
COMPARISON_DIR=$RUN_DIR/comparison

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" &&
  $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing prompt-key projection probe from wrong branch/worktree" >&2
  exit 2
}
[[ $WEIGHT_SOURCE == materialized_lp4_stage_local ||
  $WEIGHT_SOURCE == materialized_parameter ||
  $WEIGHT_SOURCE == raw_fp8_inside_executable ]] || {
  echo "unknown prompt-key weight source: $WEIGHT_SOURCE" >&2
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
[[ -r $RESULTS_DB && -d $SOURCE_DIR && -d $INPUT_DIR && -d $LOWERING_DIR &&
  -d $PROJECTION_DIR &&
  ! -e $RUN_DIR ]] || {
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
  echo "[prompt-key-norm $(date -u +%H:%M:%S)] $*" |
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

say "PIN=$PIN RUN_DIR=$RUN_DIR SOURCE_DB=$SOURCE_RUN_ID/$SOURCE_ITEM_ROW_ID WEIGHT_SOURCE=$WEIGHT_SOURCE"
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
  "$SOURCE_COMPARISON_FILE_SHA $SOURCE_COMPARISON_DIR/comparison.json" \
  "$SOURCE_CAPTURE_FILE_SHA $SOURCE_COMPARISON_DIR/accepted_capture/capture.json" \
  "$SOURCE_CAPTURE_TENSOR_SHA $SOURCE_COMPARISON_DIR/accepted_capture/accepted_prompt_key_states.npz" \
  "$SOURCE_CACHE_MANIFEST_FILE_SHA $SOURCE_DIR/prompt_index_cache/manifest.json" \
  "$SOURCE_CACHE_TENSOR_SHA $SOURCE_DIR/prompt_index_cache/prompt_index_cache.safetensors" \
  "$SOURCE_PRE_CENSUS_SHA $SOURCE_DIR/census_pre.txt" \
  "$SOURCE_POST_CENSUS_SHA $SOURCE_DIR/census_post.txt" \
  "$INPUT_MANIFEST_FILE_SHA $INPUT_DIR/manifest.json" \
  "$LOWERING_SUCCESS_SHA $LOWERING_DIR/SUCCESS" \
  "$LOWERING_SUMMARY_SHA $LOWERING_DIR/accepted_prompt_projection_lowering/summary.json" \
  "$LOWERING_MANIFEST_FILE_SHA $LOWERING_DIR/accepted_prompt_projection_lowering/manifest.json" \
  "$PROJECTION_SUCCESS_SHA $PROJECTION_DIR/SUCCESS" \
  "$PROJECTION_SUMMARY_SHA $PROJECTION_DIR/summary.json" \
  "$PROJECTION_COMPARISON_SHA $PROJECTION_DIR/comparison/comparison.json" \
  "$PROJECTION_REMOTE_OBJECTS_SHA $PROJECTION_DIR/remote_objects.json" \
  "$PROJECTION_DB_SHA $PROJECTION_DIR/results_ckpt.db" \
  "$PROJECTION_PRE_CENSUS_SHA $PROJECTION_DIR/census_pre.txt" \
  "$PROJECTION_POST_CENSUS_SHA $PROJECTION_DIR/census_post.txt"; do
  expected=${contract%% *}
  path=${contract#* }
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
    say "ABORT: sealed source drifted: $path"
    exit 2
  }
done
has_eight_unique_markers "$SOURCE_DIR/census_pre.txt" CENSUS_OK || exit 2
has_eight_unique_markers "$SOURCE_DIR/census_post.txt" CENSUS_OK || exit 2
has_eight_unique_markers "$PROJECTION_DIR/census_pre.txt" CENSUS_OK || exit 2
has_eight_unique_markers "$PROJECTION_DIR/census_post.txt" CENSUS_OK || exit 2
remote_source_success_sha=$(gcloud storage cat \
  "$SOURCE_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $remote_source_success_sha == "$SOURCE_SUCCESS_SHA" ]] || {
  say "ABORT: approved DB515 SUCCESS drifted"
  exit 2
}
remote_lowering_success_sha=$(gcloud storage cat \
  "$LOWERING_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $remote_lowering_success_sha == "$LOWERING_SUCCESS_SHA" ]] || {
  say "ABORT: approved DB516 lowering SUCCESS drifted"
  exit 2
}
remote_projection_success_sha=$(gcloud storage cat \
  "$PROJECTION_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $remote_projection_success_sha == "$PROJECTION_SUCCESS_SHA" ]] || {
  say "ABORT: approved DB517 projection SUCCESS drifted"
  exit 2
}

/home/gianl/vllm-env/bin/python - "$LOWERING_DIR" \
  "$LOWERING_MANIFEST_SHA" >"$RUN_DIR/lowering_validation.json" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected_manifest = sys.argv[2]
summary = json.loads(
    (root / "accepted_prompt_projection_lowering/summary.json").read_text()
)
manifest = json.loads(
    (root / "accepted_prompt_projection_lowering/manifest.json").read_text()
)
if (
    manifest.get("manifest_sha256") != expected_manifest
    or summary.get("status") != "SUCCESS"
    or summary.get("source_capture_code_hash")
    != "643d0926030c092ee3742e9bfe3b5c8d2511bed4"
    or summary["shape_association"] != {
        "logical_row_count": 2048,
        "physical_hlo_result": "f32[64,128]",
        "physical_row_count": 64,
        "physical_row_shards": 32,
        "profile_physical_result": "f32[64,128]",
    }
    or summary["hlo"]["convolution_count"] != 21
    or summary["hlo"]["emitter"] != "EmitAllBatchInSublanes"
    or summary["profile"]["core_count"] != 64
):
    raise SystemExit("DB516 physical-M64 lowering identity drifted")
print(json.dumps({
    "status": "SUCCESS",
    "manifest_sha256": manifest["manifest_sha256"],
    "shape_association": summary["shape_association"],
    "emitter": summary["hlo"]["emitter"],
}, indent=2, sort_keys=True))
PY

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
    raise SystemExit("DB515 source DB integrity failed")
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
    raise SystemExit("DB515 source row drifted")
model, harness, fork, env_json, pod, note = run
env = json.loads(env_json)
if (
    model != "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
    or harness != harness_pin[:7]
    or len(fork) < 7
    or not oracle_pin.startswith(fork)
    or pod != "db-v4-64-od"
    or "prompt_projection_input" not in note
    or env["os_env"].get("GLM_DSA_DUMP_INTERNALS_CODE_HASH") != observer_pin
    or env["os_env"].get("GLM_DSA_DUMP_INTERNALS_ORACLE_PIN") != oracle_pin
):
    raise SystemExit("DB515 source provenance drifted")
root = Path(source_path)
success = dict(
    line.split("=", 1)
    for line in (root / "SUCCESS").read_text().splitlines()
    if "=" in line
)
comparison = json.loads(
    (root / "prompt_projection_input_comparison/comparison.json").read_text()
)
capture = json.loads(
    (root / "prompt_projection_input_comparison/accepted_capture/capture.json").read_text()
)
cache = json.loads((root / "prompt_index_cache/manifest.json").read_text())
if (
    success.get("code_hash") != greenfield_pin
    or success.get("legacy_repository_pin") != observer_pin
    or success.get("source_run_id") != str(run_id)
    or success.get("source_item_row_id") != str(item_id)
    or comparison.get("manifest_sha256") != comparison_sha
    or comparison.get("conclusion") != {
        "classification": "projection_lowering_association",
        "first_divergent_field": "pre_layer_norm_key",
    }
    or comparison["cache_comparison"]["mismatch_count"] != 45
    or comparison["cache_comparison"]["observed_bfloat16_sha256"]
    != candidate_cache_sha
    or comparison.get("projection_input_comparison", {}).get(
        "elementwise_exact"
    ) is not True
    or capture.get("manifest_sha256") != capture_sha
    or capture.get("capture_mode") != "prompt_key_input"
    or capture.get("oracle_pin") != oracle_pin
    or capture.get("legacy_code_hash") != observer_pin
    or cache.get("manifest_sha256") != cache_manifest_sha
    or cache.get("prompt_index_key_bfloat16_sha256") != accepted_cache_sha
):
    raise SystemExit("DB515 compact source artifact drifted")
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

/home/gianl/vllm-env/bin/python - \
  "$RESULTS_DB" "$PROJECTION_DIR" "$RUN_DIR/projection_validation.json" \
  "$PROJECTION_RUN_ID" "$PROJECTION_ITEM_ROW_ID" "$PROJECTION_CODE_HASH" \
  "$PROJECTION_MANIFEST_SHA" "$PROJECTION_CACHE_SHA" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

(db_path, source_path, output_path, run_id, item_id, code_hash,
 manifest_sha, cache_sha) = sys.argv[1:]
run_id, item_id = int(run_id), int(item_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("DB517 source DB integrity failed")
run = connection.execute(
    "SELECT model,harness_git,pod FROM runs WHERE run_id=?", (run_id,)
).fetchone()
item = connection.execute(
    "SELECT id,benchmark,item_id,correct,score,n_prompt_tokens "
    "FROM items WHERE id=? AND run_id=?", (item_id, run_id)
).fetchone()
if run != (
    "zai-org/GLM-5.2-FP8:greenfield-layer0-prompt-key-projection-m64",
    code_hash[:7],
    "db-v4-64-od",
) or item != (
    item_id,
    "greenfield_layer0_prompt_key_projection_association",
    "adapted_fp32_m64_lax_map_gather_cache_source_rope",
    0,
    0.0,
    8155,
):
    raise SystemExit("DB517 source row drifted")
root = Path(source_path)
success = dict(
    line.split("=", 1)
    for line in (root / "SUCCESS").read_text().splitlines()
    if "=" in line
)
summary = json.loads((root / "summary.json").read_text())
comparison = json.loads((root / "comparison/comparison.json").read_text())
fields = comparison["state_comparison"]["fields"]
if (
    success.get("code_hash") != code_hash
    or success.get("results_db_run_id") != str(run_id)
    or success.get("results_db_item_row_id") != str(item_id)
    or success.get("projection_association_restored") != "false"
    or summary.get("comparison_manifest_sha256") != manifest_sha
    or summary.get("projection_association_restored") is not False
    or comparison.get("manifest_sha256") != manifest_sha
    or comparison.get("projection_mapping_mode") != "physical_m64_lax_map"
    or comparison.get("projection_input_comparison", {}).get(
        "elementwise_exact"
    ) is not True
    or fields["pre_layer_norm_key"]["elementwise_exact"] is not True
    or comparison["state_comparison"].get("first_divergent_field")
    != "pre_rope_key"
    or comparison["state_comparison"].get("classification")
    != "key_layer_norm_association"
    or comparison["cache_comparison"].get("mismatch_count") != 22
    or comparison["cache_comparison"].get("first_mismatch_position") != 114
    or comparison["cache_comparison"].get("observed_bfloat16_sha256")
    != cache_sha
):
    raise SystemExit("DB517 exact-projection evidence drifted")
for name in ("cache", "states"):
    contract = comparison["hlo"][name]["contract"]
    if (
        not contract["passed"]
        or contract["accepted_convolution_count"] != 1
        or contract["bf16_wk_conversion_count"] != 0
        or contract["convolution_weight_f32"] is not True
        or contract["required_shapes"]["physical_m64_projection_input"]
        is not True
        or contract["loop_count"] != 1
    ):
        raise SystemExit(f"DB517 {name} physical projection drifted")
Path(output_path).write_text(json.dumps({
    "status": "SUCCESS",
    "source_run_id": run_id,
    "source_item_row_id": item_id,
    "code_hash": code_hash,
    "comparison_manifest_sha256": manifest_sha,
    "projection_elementwise_exact": True,
    "first_divergent_field": "pre_rope_key",
    "cache_mismatch_count": 22,
}, indent=2, sort_keys=True) + "\n")
connection.close()
PY

say "running physical-M64 projection plus key LayerNorm on one four-chip TPU host"
started=$(date +%s)
env JAX_PLATFORMS=tpu \
  TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH="$WORKTREE" \
  timeout --signal=TERM --kill-after=60 1800 \
  /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/compare_accepted_prompt_key_internals.py" \
  --accepted-capture-dir "$SOURCE_COMPARISON_DIR/accepted_capture" \
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
  --projection-weight-source "$WEIGHT_SOURCE" \
  --projection-mapping-mode physical_m64_projection_keynorm_lax_map \
  --capture-mode prompt_key_input \
  >"$RUN_DIR/comparison_summary.json"
elapsed=$(( $(date +%s) - started ))
say "key LayerNorm discriminator completed in ${elapsed}s"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" \
  "$SOURCE_RUN_ID" "$SOURCE_ITEM_ROW_ID" \
  "$SOURCE_COMPARISON_MANIFEST_SHA" "$SOURCE_CAPTURE_MANIFEST_SHA" \
  "$SOURCE_CACHE_MANIFEST_SHA" "$LOWERING_MANIFEST_SHA" \
  "$PROJECTION_RUN_ID" "$PROJECTION_ITEM_ROW_ID" \
  "$PROJECTION_MANIFEST_SHA" "$WEIGHT_SOURCE" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

(run_path, pin, db_path, repo, elapsed, source_run_id, source_item_id,
 source_comparison_sha, capture_sha, cache_sha, lowering_sha,
 projection_run_id, projection_item_id, projection_manifest_sha,
 weight_source) = sys.argv[1:]
root = Path(run_path)
comparison = json.loads((root / "comparison/comparison.json").read_text())
if (
    comparison["status"] != "SUCCESS"
    or comparison["format_version"] != 2
    or comparison["code_hash"] != pin
    or comparison["backend"] != "tpu"
    or comparison["device_count"] != 4
    or comparison["diagnostic_only"] is not True
    or comparison["performance_claim"] is not False
    or comparison["projection_weight_mode"] != "adapted_fp32"
    or comparison["projection_weight_source"] != weight_source
    or comparison["projection_mapping_mode"]
    != "physical_m64_projection_keynorm_lax_map"
    or comparison["accepted_capture"]["manifest_sha256"] != capture_sha
    or comparison["accepted_cache"]["manifest_sha256"] != cache_sha
):
    raise SystemExit("physical-M64 key LayerNorm comparison identity failed")
for name in ("cache", "states"):
    contract = comparison["hlo"][name]["contract"]
    weight_contract = comparison["hlo"][name]["weight_source_contract"]
    if (
        not contract["passed"]
        or contract["accepted_convolution_count"] != 1
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
        or contract["loop_count"] != 1
        or contract["required_shapes"]["physical_m64_projection"] is not True
        or contract["required_shapes"]["physical_m64_projection_input"] is not True
        or contract["required_shapes"]["physical_m64_key_norm_sqrt"] is not True
        or contract["required_shapes"]["physical_m64_key_norm_affine"] is not True
        or contract["physical_key_norm"]["enabled"] is not True
        or contract["physical_key_norm"]["grouped_sqrt_count"] != 0
        or contract["physical_key_norm"]["sqrt_count"] < 1
        or contract["physical_key_norm"]["affine_count"] < 1
        or weight_contract["passed"] is not True
        or weight_contract["source"] != weight_source
    ):
        raise SystemExit(f"physical-M64 key LayerNorm {name} HLO contract failed")
projection_input = comparison["projection_input_comparison"]
if projection_input is None or projection_input["elementwise_exact"] is not True:
    raise SystemExit("DB515 projection input is not bitwise exact")
lp4 = comparison["lp4_materialized_repair"]
if weight_source == "materialized_lp4_stage_local":
    if (
        lp4 is None
        or lp4["assembled_cache_elementwise_exact"] is not True
        or lp4["assembled_cache_sha256"]
        != comparison["accepted_cache"][
            "prompt_index_key_bfloat16_sha256"
        ]
        or lp4["owner_isolation_exact"] is not True
        or lp4["local_device_ids"] != [0, 1, 2, 3]
        or lp4["per_lane_written_rows"] != [2048, 2048, 2048, 2011]
        or len(lp4["materialized_shards"]) != 4
        or any(
            shard["byte_count"] != 128 * 6144 * 4
            or shard["sha256"]
            != comparison["accepted_adapted_wk"]["sha256"]
            for shard in lp4["materialized_shards"]
        )
    ):
        raise SystemExit("LP4 materialized repair ownership/exactness failed")
    materializer = lp4["hlo"]["lp4_wk_materializer"]["contract"]
    repair = lp4["hlo"]["lp4_cache_repair"]["contract"]
    if (
        not materializer["passed"]
        or materializer["backend"]
        != "external_stage_local_bf16_then_fp32"
        or materializer["expected_slot_count"] != 1
        or materializer["raw_parameter_count"] != 1
        or materializer["scale_parameter_count"] != 1
        or materializer["bf16_round_count"] < 1
        or materializer["fp32_promotion_count"] < 1
        or materializer["collective_count"] != 0
        or materializer["host_markers"]
        or materializer["violations"]
        or not repair["passed"]
        or repair["full_indexer_layer_count"] != 1
        or repair["chunk_count"] != 4
        or repair["expected_call_count"] != 4
        or repair["projection_count"] != 4
        or repair["exact_projection_operand_count"] != 4
        or repair["physical_sqrt_count"] != 8
        or repair["physical_affine_count"] < 4
        or repair["cache_write_count"] < 4
        or repair["materialized_wk_parameter_count"] < 1
        or repair["repair_weight_round_count"] != 0
        or repair["repair_collectives"]
        or repair["forbidden_markers"]
        or repair["violations"]
    ):
        raise SystemExit("LP4 materialized repair HLO contract failed")
elif lp4 is not None:
    raise SystemExit("unrequested LP4 materialized repair evidence is present")
state_exact = comparison["state_comparison"]["all_fields_elementwise_exact"]
cache_exact = comparison["cache_comparison"]["elementwise_exact"]
restored = bool(state_exact and cache_exact)

sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

connection = pv.connect(db_path)
run_id = pv.start_run(
    connection,
    model="zai-org/GLM-5.2-FP8:greenfield-layer0-prompt-key-norm-m64",
    revision="bounded-real-layer0-prompt-key-norm-physical-m64-v1",
    env={
        "GLM_ENGINE": "greenfield_layer0_prompt_key_norm_association",
        "greenfield_code_hash": pin,
        "source_run_id": int(source_run_id),
        "source_item_row_id": int(source_item_id),
        "source_comparison_manifest_sha256": source_comparison_sha,
        "accepted_capture_manifest_sha256": capture_sha,
        "prompt_cache_manifest_sha256": cache_sha,
        "accepted_projection_lowering_manifest_sha256": lowering_sha,
        "source_projection_run_id": int(projection_run_id),
        "source_projection_item_row_id": int(projection_item_id),
        "source_projection_manifest_sha256": projection_manifest_sha,
        "comparison_manifest_sha256": comparison["manifest_sha256"],
        "candidate_cache_sha256": comparison["greenfield_cache"][
            "prompt_index_key_bfloat16_sha256"
        ],
        "projection_weight_mode": "adapted_fp32",
        "projection_weight_source": weight_source,
        "projection_mapping_mode": (
            "physical_m64_projection_keynorm_lax_map"
        ),
        "projection_input_elementwise_exact": True,
        "state_elementwise_exact": state_exact,
        "cache_elementwise_exact": cache_exact,
        "lp4_materialized_repair": (
            {
                "assembled_cache_sha256": lp4[
                    "assembled_cache_sha256"
                ],
                "materializer_hlo_sha256": lp4["hlo"][
                    "lp4_wk_materializer"
                ]["optimized_hlo_sha256"],
                "owner_isolation_exact": lp4["owner_isolation_exact"],
                "per_lane_written_rows": lp4["per_lane_written_rows"],
                "repair_hlo_sha256": lp4["hlo"][
                    "lp4_cache_repair"
                ]["optimized_hlo_sha256"],
            }
            if lp4 is not None
            else None
        ),
    },
    note=(
        "Protected bounded physical-M64 prompt-key projection plus LayerNorm "
        "diagnostic. No decoder, Gate-D, latency, or token-rate claim."
    ),
    harness_repo=repo,
    fork_repo=None,
)
pv.record_item(
    connection,
    run_id,
    benchmark="greenfield_layer0_prompt_key_norm_association",
    item_id=f"adapted_fp32_m64_projection_keynorm_{weight_source}",
    prompt="Sealed DB515 input/cache and DB517 exact projection.",
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
    benchmark="greenfield_layer0_prompt_key_norm_association",
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
    "source_projection_run_id": int(projection_run_id),
    "source_projection_item_row_id": int(projection_item_id),
    "source_projection_manifest_sha256": projection_manifest_sha,
    "comparison_manifest_sha256": comparison["manifest_sha256"],
    "accepted_projection_lowering_manifest_sha256": lowering_sha,
    "key_norm_association_restored": restored,
    "projection_input_elementwise_exact": True,
    "projection_weight_source": weight_source,
    "state_elementwise_exact": state_exact,
    "cache_elementwise_exact": cache_exact,
    "lp4_materialized_repair": lp4,
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
    raise SystemExit("key LayerNorm DB snapshot failed integrity check")
print(f"PROMPT_KEY_NORM_VALID db_run={run_id} restored={restored}")
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "freezing and archiving key LayerNorm evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find comparison -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum comparison_summary.json summary.json source_validation.json \
    lowering_validation.json projection_validation.json results_ckpt.db disk_preflight.txt \
    census_pre.txt census_post.txt \
    orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY' \
  >"$RUN_DIR/remote_objects.json"
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = [
    (path, path.relative_to(root).as_posix())
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.name not in {"SUCCESS", "remote_objects.json"}
]

def local_crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()

def describe(item):
    path, relative = item
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}",
         "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    crc32c = value.get("crc32c_hash") or value.get("crc32c")
    if (
        int(value["size"]) != path.stat().st_size
        or crc32c != local_crc32c(path)
    ):
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
    "artifact_kind": "glm52_layer0_prompt_key_norm_physical_m64",
    "code_hash": sys.argv[3],
    "results_db_run_id": summary["results_db_run_id"],
    "results_db_item_row_id": summary["results_db_item_row_id"],
    "source_run_id": summary["source_run_id"],
    "source_item_row_id": summary["source_item_row_id"],
    "source_projection_run_id": summary["source_projection_run_id"],
    "source_projection_item_row_id": summary[
        "source_projection_item_row_id"
    ],
    "source_projection_manifest_sha256": summary[
        "source_projection_manifest_sha256"
    ],
    "comparison_manifest_sha256": summary["comparison_manifest_sha256"],
    "accepted_projection_lowering_manifest_sha256": summary[
        "accepted_projection_lowering_manifest_sha256"
    ],
    "key_norm_association_restored": str(
        summary["key_norm_association_restored"]
    ).lower(),
    "projection_input_elementwise_exact": "true",
    "projection_mapping_mode": "physical_m64_projection_keynorm_lax_map",
    "projection_weight_source": summary["projection_weight_source"],
    "candidate_cache_sha256": summary["candidate_cache_sha256"],
    "lp4_owner_isolation_exact": str(
        (
            summary["lp4_materialized_repair"] or {}
        ).get("owner_isolation_exact", False)
    ).lower(),
    "lp4_materializer_hlo_sha256": (
        (summary["lp4_materialized_repair"] or {}).get(
            "hlo", {}
        ).get("lp4_wk_materializer", {}).get(
            "optimized_hlo_sha256", "none"
        )
    ),
    "lp4_repair_hlo_sha256": (
        (summary["lp4_materialized_repair"] or {}).get(
            "hlo", {}
        ).get("lp4_cache_repair", {}).get(
            "optimized_hlo_sha256", "none"
        )
    ),
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
restored=$(sed -n 's/^key_norm_association_restored=//p' "$RUN_DIR/SUCCESS")
say "SUCCESS DB=$db_run key_norm_restored=$restored"
