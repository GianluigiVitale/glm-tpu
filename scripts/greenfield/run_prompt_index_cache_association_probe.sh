#!/usr/bin/env bash
# Resume the sealed cache chain and isolate one prompt-key association.
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
readonly MATRIX_BEST_HLO_FILE_SHA=7dbb25c231149e5d8ba811cfaede6bcd8fa692dcc07a4f3fa7d1b9c24440fb89
readonly CHUNK_TAG=greenfield_layer0_prompt_index_cache_association_20260808T221200429613435Z
readonly CHUNK_DIR=/home/gianl/glm-run/$CHUNK_TAG
readonly CHUNK_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_association/8k/$CHUNK_TAG
readonly CHUNK_CODE_HASH=7027cf6b655d76751f7c166b38e0780428918a9e
readonly CHUNK_RUN_ID=508
readonly CHUNK_ITEM_ROW_ID=1793
readonly CHUNK_ASSOCIATION_FILE_SHA=7ec5737b64117b94e055a264b426824dc377962521ab4986a2793b84939fd3f2
readonly CHUNK_SUMMARY_FILE_SHA=d5e663c86270a1711c2965bb34e2499de407083677640577a22e84716cb8a8a3
readonly CHUNK_SUCCESS_SHA=6a38369a4fbe007b0539bba185af49715917738ea995491d23b0a8d287e612e7
readonly CHUNK_RESULTS_DB_SHA=81bff92804a785ca5991e919441b3139e16aba2f29290a3baa8a36794affa7be
readonly CHUNK_EVIDENCE_SHA=48776445ce8971afdc8734f81485880686a00704b0471b1ded3fb73b5627214c
readonly CHUNK_REMOTE_OBJECTS_SHA=11aa387aa711eeba67a83acf12ea1304e17caf0545e3a98a4c5a657fe32034e4
readonly CHUNK_PRE_CENSUS_SHA=0e31ae7e0b5cc66a618965c673926e8cb76fffd79b17f3a73221a208da008ce7
readonly CHUNK_POST_CENSUS_SHA=127b69cd3010860a7e00da92fb521140d76baf36b990cc9a167d87081801a911
readonly CHUNK_MATRIX_VALIDATION_SHA=667c4b1de614f90821c21f18cbf9866888c6ff01d3fb02aa1fb0b62996421070
readonly CHUNK_SOURCE_VALIDATION_SHA=cf9306eab21f030198e9d1195732b0879fb8aba824feec02bab93f68d95c547d
readonly CHUNK_ASSOCIATION_MANIFEST_SHA=8539a81d5300266f2ea35c2df2d16321e3b964e471ce83f7b8fc80459fcd6d07
readonly CHUNK_OBSERVED_CACHE_SHA=db2f77d986fa3a452bc60a429b1a7b0a63e5d77a6408e504dc9a9359eb50a7a1
readonly CHUNK_HLO_FILE_SHA=b2e986161aafaff4823d6d2e527c24fc308de1336c5ebb00d0aa29f33dd2f274
readonly BF16_TAG=greenfield_layer0_prompt_index_cache_association_20260808T225610150435734Z
readonly BF16_DIR=/home/gianl/glm-run/$BF16_TAG
readonly BF16_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_association/8k/$BF16_TAG
readonly BF16_CODE_HASH=8f2545cf25181f002d939ab704bc251897fa377d
readonly BF16_RUN_ID=509
readonly BF16_ITEM_ROW_ID=1794
readonly BF16_ASSOCIATION_FILE_SHA=f2c21baa1a403ae9fddb4f772a6e5b683f0e0703e173573f74b6cae09556dd21
readonly BF16_SUMMARY_FILE_SHA=8275006b016b686078e75c655faa9b71754ae7ef5bbdc2b604719ea49f306ab7
readonly BF16_SUCCESS_SHA=0bea72bac14c8bc59824a30b3c062fd45a489e9db530f3dbbbf75c9b8e459232
readonly BF16_RESULTS_DB_SHA=3820af70ae5c5fb31cd1e16a5092622268154c132ae8c1d2207e32eabf78d88f
readonly BF16_EVIDENCE_SHA=652da6663f2fabc3e10676d07e8c3531a145e306f399920eb62e8bcb31ce1f99
readonly BF16_REMOTE_OBJECTS_SHA=6378c961c177d7c98dc57cb33c1cc5af0a86c5c07f7bc4daf7ee36594c1b7047
readonly BF16_PRE_CENSUS_SHA=12b5fa792e6cba2298d32761f4ea8b16479a6d0597220551723ea92b0c87306e
readonly BF16_POST_CENSUS_SHA=b0f6d2a4141d28ebde870b41addad64f26070f6b177b008abb9b4b222c0cfbe7
readonly BF16_MATRIX_VALIDATION_SHA=667c4b1de614f90821c21f18cbf9866888c6ff01d3fb02aa1fb0b62996421070
readonly BF16_CHUNK_VALIDATION_SHA=5816cd9e3a1cfc2faf52395d9b90dc660a0ce261771d87f3b2f34c8d02ccaa2c
readonly BF16_SOURCE_VALIDATION_SHA=cf9306eab21f030198e9d1195732b0879fb8aba824feec02bab93f68d95c547d
readonly BF16_ASSOCIATION_MANIFEST_SHA=df0b901e9d8f59a9053a52c35fe4b7156ed6814a727ede03c64a5c350c9921c1
readonly BF16_OBSERVED_CACHE_SHA=db2f77d986fa3a452bc60a429b1a7b0a63e5d77a6408e504dc9a9359eb50a7a1
readonly BF16_HLO_FILE_SHA=6e2697569b995b395dadc51a72504cefeb4b95cc96e3de6a842822a883632160
readonly GATHER_TAG=greenfield_layer0_prompt_index_cache_association_20260808T234459065479710Z
readonly GATHER_DIR=/home/gianl/glm-run/$GATHER_TAG
readonly GATHER_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_association/8k/$GATHER_TAG
readonly GATHER_CODE_HASH=6286a06341cb1f798c7059490e565e966ba85dc0
readonly GATHER_RUN_ID=510
readonly GATHER_ITEM_ROW_ID=1795
readonly GATHER_ASSOCIATION_FILE_SHA=7d9c12bebc59a6e7408050bfa6cf73f02d4a7624b8815fce30ca2e487874e224
readonly GATHER_TENSOR_FILE_SHA=d896b17229b0564cc9fe79781847d7c03895a32d5fcd346e646638813ed0227d
readonly GATHER_SUMMARY_FILE_SHA=af65953a35d81adfd796d95c3e6cebd0ac33996f5c64a6a76d8a356c997336e1
readonly GATHER_SUCCESS_SHA=a7660e3ee812abccced8bc293f3552434cc573088cddd3c52c7439022d71e165
readonly GATHER_RESULTS_DB_SHA=50758ad281f1ac280807c9499519d714dd339aecd43eba9deabe450b1943de53
readonly GATHER_EVIDENCE_SHA=9ed5bdfffb5ad9e1ee026aaef3abb28d447e3d5e4c4f0a29bacf064eb8c35b4f
readonly GATHER_REMOTE_OBJECTS_SHA=d4663158c69b875af9bb00a87d38be0753c6a0f57cd4cb4fc4b21ba07bdc5f9e
readonly GATHER_PRE_CENSUS_SHA=d05a3822200060f14abad6138871f15d17b2476b892dbd35eb112312f26f967b
readonly GATHER_POST_CENSUS_SHA=6caec386e35b5d47b9749ea21618799a1e667898b19f4ec788f7f0a765a7f1df
readonly GATHER_MATRIX_VALIDATION_SHA=667c4b1de614f90821c21f18cbf9866888c6ff01d3fb02aa1fb0b62996421070
readonly GATHER_CHUNK_VALIDATION_SHA=5816cd9e3a1cfc2faf52395d9b90dc660a0ce261771d87f3b2f34c8d02ccaa2c
readonly GATHER_BF16_VALIDATION_SHA=4d21da19b35a073a0133f61b692ca5b5ea2c1e91544a7a4f68a350e9dc159e4c
readonly GATHER_SOURCE_VALIDATION_SHA=cf9306eab21f030198e9d1195732b0879fb8aba824feec02bab93f68d95c547d
readonly GATHER_ASSOCIATION_MANIFEST_SHA=3e29aadce4851ad4a2f5ab62ab2d77d1474fc04aed988eaf14657d1142c8b0fb
readonly GATHER_OBSERVED_CACHE_SHA=52bf55ed5e9ea74a59551351a21ae830fb39e289e82eaff636fb9ab84d7dcd8a
readonly GATHER_HLO_FILE_SHA=cd293af30d94b2f96d205dfa9e13e9e08ad86732309fe2950abadf654a947290
readonly CACHE_WRITE_TAG=greenfield_layer0_prompt_index_cache_association_20260809T003039938922204Z
readonly CACHE_WRITE_DIR=/home/gianl/glm-run/$CACHE_WRITE_TAG
readonly CACHE_WRITE_REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_association/8k/$CACHE_WRITE_TAG
readonly CACHE_WRITE_CODE_HASH=31ab626547edd7b622bb23d7d994a889cd311eeb
readonly CACHE_WRITE_RUN_ID=511
readonly CACHE_WRITE_ITEM_ROW_ID=1796
readonly CACHE_WRITE_ASSOCIATION_FILE_SHA=addefc7e530ad2469cd437305e23f0c19262a8ec44c1a4054b32f8351908ebf8
readonly CACHE_WRITE_TENSOR_FILE_SHA=64e6df3b65981b765da8e50dbed39c69a6736a1f39bddc7e59911efe793b130f
readonly CACHE_WRITE_SUMMARY_FILE_SHA=8d420541d4db9358133010a20d463386a6eff1f5f497f38e99c97f6284aae1d2
readonly CACHE_WRITE_SUCCESS_SHA=ec6a43702a1ae217ae601237abd0fba05fb0d68881830ce995cf0963724e55a7
readonly CACHE_WRITE_RESULTS_DB_SHA=d71012bb90a2db44ecb82f48f3bec5915472547a9df4b382c40ac604d2d96eae
readonly CACHE_WRITE_EVIDENCE_SHA=8978b3330fa85e2e78c8d63932a6e0b5f5e8622f13c5e9c44ffe948cdfa6cb25
readonly CACHE_WRITE_REMOTE_OBJECTS_SHA=5d7bb1f6b3709a00cadf2b535e290af321f94fa3b9fa40fb249b38accbec083f
readonly CACHE_WRITE_PRE_CENSUS_SHA=1d49893707020ce32e4f075ab501cb86cbdc135ac780d5ab7f4a87ec064bf785
readonly CACHE_WRITE_POST_CENSUS_SHA=e5d38ffbce7adb88f39115e1b1dad52674c15be35db9448fb771dd1d2c575487
readonly CACHE_WRITE_MATRIX_VALIDATION_SHA=667c4b1de614f90821c21f18cbf9866888c6ff01d3fb02aa1fb0b62996421070
readonly CACHE_WRITE_CHUNK_VALIDATION_SHA=5816cd9e3a1cfc2faf52395d9b90dc660a0ce261771d87f3b2f34c8d02ccaa2c
readonly CACHE_WRITE_BF16_VALIDATION_SHA=4d21da19b35a073a0133f61b692ca5b5ea2c1e91544a7a4f68a350e9dc159e4c
readonly CACHE_WRITE_GATHER_VALIDATION_SHA=d80163cb19f0f85819ca709a7bb2425e0a6d1f11e3f8fb89a4f38b3f3072665e
readonly CACHE_WRITE_SOURCE_VALIDATION_SHA=cf9306eab21f030198e9d1195732b0879fb8aba824feec02bab93f68d95c547d
readonly CACHE_WRITE_ASSOCIATION_MANIFEST_SHA=6cbe954b4bd28161518374897027c91556b5d7030ea4117434d49c19a7f19bb6
readonly CACHE_WRITE_OBSERVED_CACHE_SHA=52bf55ed5e9ea74a59551351a21ae830fb39e289e82eaff636fb9ab84d7dcd8a
readonly CACHE_WRITE_OPTIMIZED_HLO_SHA=d396040f171afcc28a1fe6d1ec36e7f11529d72ff0051aabdfca4fe91a4b3938
readonly CACHE_WRITE_HLO_FILE_SHA=e55cecef0f05142f20748f441df020bb83941bb4770a90903395fb9301babc10
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly INPUT_MANIFEST_FILE_SHA=bd06714ebfe5177b8466778e2bc33ef262544dced48adcfc5739be37ac6488b9

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
PROFILE=${GLM_GREENFIELD_PROMPT_CACHE_ASSOCIATION_PROFILE:-matrix}
TAG=${GLM_GREENFIELD_PROMPT_CACHE_ASSOCIATION_TAG:-greenfield_layer0_prompt_index_cache_association_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_index_cache_association/8k/$TAG
ASSOCIATION_DIR=$RUN_DIR/association

[[ $PROFILE == matrix || $PROFILE == chunk_parameter || \
  $PROFILE == chunk_bf16_weight || \
  $PROFILE == chunk_gather_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_source_rope ]] || {
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

if [[ $PROFILE == chunk_parameter || $PROFILE == chunk_bf16_weight || \
  $PROFILE == chunk_gather_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_source_rope ]]; then
  for contract in \
    "$MATRIX_ASSOCIATION_FILE_SHA $MATRIX_DIR/association/association.json" \
    "$MATRIX_BEST_HLO_FILE_SHA $MATRIX_DIR/association/hlo/accepted_xla_m2048_divide_sqrt.optimized_hlo.txt.gz" \
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

if [[ $PROFILE == chunk_bf16_weight || \
  $PROFILE == chunk_gather_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_source_rope ]]; then
  for contract in \
    "$CHUNK_ASSOCIATION_FILE_SHA $CHUNK_DIR/association/association.json" \
    "$CHUNK_HLO_FILE_SHA $CHUNK_DIR/association/hlo/accepted_xla_m2048_chunk_parameter_divide_sqrt.optimized_hlo.txt.gz" \
    "$CHUNK_SUMMARY_FILE_SHA $CHUNK_DIR/summary.json" \
    "$CHUNK_SUCCESS_SHA $CHUNK_DIR/SUCCESS" \
    "$CHUNK_RESULTS_DB_SHA $CHUNK_DIR/results_ckpt.db" \
    "$CHUNK_EVIDENCE_SHA $CHUNK_DIR/evidence.sha256" \
    "$CHUNK_REMOTE_OBJECTS_SHA $CHUNK_DIR/remote_objects.json" \
    "$CHUNK_PRE_CENSUS_SHA $CHUNK_DIR/census_pre.txt" \
    "$CHUNK_POST_CENSUS_SHA $CHUNK_DIR/census_post.txt" \
    "$CHUNK_MATRIX_VALIDATION_SHA $CHUNK_DIR/matrix_validation.json" \
    "$CHUNK_SOURCE_VALIDATION_SHA $CHUNK_DIR/source_validation.json"; do
    expected=${contract%% *}
    path=${contract#* }
    [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
      say "ABORT: sealed DB508 chunk result drifted: $path"
      exit 2
    }
  done
  chunk_remote_success_sha=$(gcloud storage cat \
    "$CHUNK_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $chunk_remote_success_sha == "$CHUNK_SUCCESS_SHA" ]] || {
    say "ABORT: approved DB508 SUCCESS drifted"
    exit 2
  }
  /home/gianl/vllm-env/bin/python - \
    "$RESULTS_DB" "$CHUNK_DIR" \
    "$RUN_DIR/chunk_parameter_validation.json" \
    "$CHUNK_RUN_ID" "$CHUNK_ITEM_ROW_ID" "$CHUNK_CODE_HASH" \
    "$CHUNK_ASSOCIATION_MANIFEST_SHA" "$CHUNK_OBSERVED_CACHE_SHA" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

(db_path, chunk_path, output_path, run_id, item_id, code_hash,
 manifest_sha, observed_sha) = sys.argv[1:]
run_id, item_id = int(run_id), int(item_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("DB508 chunk source DB integrity failed")
run = connection.execute(
    "SELECT harness_git,env_json,pod FROM runs WHERE run_id=?", (run_id,)
).fetchone()
item = connection.execute(
    "SELECT id,item_id,correct,score FROM items WHERE run_id=?", (run_id,)
).fetchone()
expected_item = (
    item_id,
    "accepted_xla_m2048_chunk_parameter_divide_sqrt",
    0,
    0.0,
)
if run is None or item != expected_item:
    raise SystemExit("DB508 chunk source row drifted")
harness, env_json, pod = run
env = json.loads(env_json)
if (
    harness != code_hash[:7]
    or pod != "db-v4-64-od"
    or env.get("association_manifest_sha256") != manifest_sha
    or env.get("candidate_set") != "chunk_parameter"
    or env.get("exact_candidates") != []
):
    raise SystemExit("DB508 chunk source provenance drifted")
root = Path(chunk_path)
association = json.loads((root / "association/association.json").read_text())
candidate = association["candidates"][
    "accepted_xla_m2048_chunk_parameter_divide_sqrt"
]
if (
    association["manifest_sha256"] != manifest_sha
    or association["candidate_set"] != "chunk_parameter"
    or candidate["comparison_to_accepted"]["mismatch_count"] != 4045
    or candidate["observed_bfloat16_sha256"] != observed_sha
    or candidate["hlo"]["contract"]["loop_count"] != 0
):
    raise SystemExit("DB508 chunk source artifact drifted")
Path(output_path).write_text(json.dumps({
    "status": "SUCCESS",
    "chunk_run_id": run_id,
    "chunk_item_row_id": item_id,
    "chunk_code_hash": code_hash,
    "chunk_association_manifest_sha256": manifest_sha,
    "chunk_mismatch_count": 4045,
}, indent=2, sort_keys=True) + "\n")
connection.close()
PY
fi

if [[ $PROFILE == chunk_gather_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_source_rope ]]; then
  for contract in \
    "$BF16_ASSOCIATION_FILE_SHA $BF16_DIR/association/association.json" \
    "$BF16_HLO_FILE_SHA $BF16_DIR/association/hlo/accepted_xla_m2048_chunk_bf16_weight_divide_sqrt.optimized_hlo.txt.gz" \
    "$BF16_SUMMARY_FILE_SHA $BF16_DIR/summary.json" \
    "$BF16_SUCCESS_SHA $BF16_DIR/SUCCESS" \
    "$BF16_RESULTS_DB_SHA $BF16_DIR/results_ckpt.db" \
    "$BF16_EVIDENCE_SHA $BF16_DIR/evidence.sha256" \
    "$BF16_REMOTE_OBJECTS_SHA $BF16_DIR/remote_objects.json" \
    "$BF16_PRE_CENSUS_SHA $BF16_DIR/census_pre.txt" \
    "$BF16_POST_CENSUS_SHA $BF16_DIR/census_post.txt" \
    "$BF16_MATRIX_VALIDATION_SHA $BF16_DIR/matrix_validation.json" \
    "$BF16_CHUNK_VALIDATION_SHA $BF16_DIR/chunk_parameter_validation.json" \
    "$BF16_SOURCE_VALIDATION_SHA $BF16_DIR/source_validation.json"; do
    expected=${contract%% *}
    path=${contract#* }
    [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
      say "ABORT: sealed DB509 BF16-weight result drifted: $path"
      exit 2
    }
  done
  bf16_remote_success_sha=$(gcloud storage cat \
    "$BF16_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $bf16_remote_success_sha == "$BF16_SUCCESS_SHA" ]] || {
    say "ABORT: approved DB509 SUCCESS drifted"
    exit 2
  }
  /home/gianl/vllm-env/bin/python - \
    "$RESULTS_DB" "$BF16_DIR" \
    "$RUN_DIR/bf16_weight_validation.json" \
    "$BF16_RUN_ID" "$BF16_ITEM_ROW_ID" "$BF16_CODE_HASH" \
    "$BF16_ASSOCIATION_MANIFEST_SHA" "$BF16_OBSERVED_CACHE_SHA" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

(db_path, bf16_path, output_path, run_id, item_id, code_hash,
 manifest_sha, observed_sha) = sys.argv[1:]
run_id, item_id = int(run_id), int(item_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("DB509 BF16-weight source DB integrity failed")
run = connection.execute(
    "SELECT harness_git,env_json,pod FROM runs WHERE run_id=?", (run_id,)
).fetchone()
item = connection.execute(
    "SELECT id,item_id,correct,score FROM items WHERE run_id=?", (run_id,)
).fetchone()
candidate_name = "accepted_xla_m2048_chunk_bf16_weight_divide_sqrt"
expected_item = (item_id, candidate_name, 0, 0.0)
if run is None or item != expected_item:
    raise SystemExit("DB509 BF16-weight source row drifted")
harness, env_json, pod = run
env = json.loads(env_json)
if (
    harness != code_hash[:7]
    or pod != "db-v4-64-od"
    or env.get("association_manifest_sha256") != manifest_sha
    or env.get("candidate_set") != "chunk_bf16_weight"
    or env.get("association_parent_run_id") != 508
    or env.get("exact_candidates") != []
):
    raise SystemExit("DB509 BF16-weight source provenance drifted")
root = Path(bf16_path)
association = json.loads((root / "association/association.json").read_text())
candidate = association["candidates"][candidate_name]
hlo = candidate["hlo"]["contract"]
if (
    association["manifest_sha256"] != manifest_sha
    or association["candidate_set"] != "chunk_bf16_weight"
    or candidate["comparison_to_accepted"]["mismatch_count"] != 4045
    or candidate["observed_bfloat16_sha256"] != observed_sha
    or hlo["loop_count"] != 0
    or hlo["bf16_wk_conversion_count"] != 1
    or hlo["convolution_weight_bf16"] is not True
):
    raise SystemExit("DB509 BF16-weight source artifact drifted")
Path(output_path).write_text(json.dumps({
    "status": "SUCCESS",
    "bf16_weight_run_id": run_id,
    "bf16_weight_item_row_id": item_id,
    "bf16_weight_code_hash": code_hash,
    "bf16_weight_association_manifest_sha256": manifest_sha,
    "bf16_weight_mismatch_count": 4045,
}, indent=2, sort_keys=True) + "\n")
connection.close()
PY
fi

if [[ $PROFILE == chunk_gather_cache_write_bf16_weight || \
  $PROFILE == chunk_gather_cache_write_source_rope ]]; then
  for contract in \
    "$GATHER_ASSOCIATION_FILE_SHA $GATHER_DIR/association/association.json" \
    "$GATHER_TENSOR_FILE_SHA $GATHER_DIR/association/candidate_prompt_index_keys.safetensors" \
    "$GATHER_HLO_FILE_SHA $GATHER_DIR/association/hlo/accepted_xla_m2048_gather_chunk_bf16_weight_divide_sqrt.optimized_hlo.txt.gz" \
    "$GATHER_SUMMARY_FILE_SHA $GATHER_DIR/summary.json" \
    "$GATHER_SUCCESS_SHA $GATHER_DIR/SUCCESS" \
    "$GATHER_RESULTS_DB_SHA $GATHER_DIR/results_ckpt.db" \
    "$GATHER_EVIDENCE_SHA $GATHER_DIR/evidence.sha256" \
    "$GATHER_REMOTE_OBJECTS_SHA $GATHER_DIR/remote_objects.json" \
    "$GATHER_PRE_CENSUS_SHA $GATHER_DIR/census_pre.txt" \
    "$GATHER_POST_CENSUS_SHA $GATHER_DIR/census_post.txt" \
    "$GATHER_MATRIX_VALIDATION_SHA $GATHER_DIR/matrix_validation.json" \
    "$GATHER_CHUNK_VALIDATION_SHA $GATHER_DIR/chunk_parameter_validation.json" \
    "$GATHER_BF16_VALIDATION_SHA $GATHER_DIR/bf16_weight_validation.json" \
    "$GATHER_SOURCE_VALIDATION_SHA $GATHER_DIR/source_validation.json"; do
    expected=${contract%% *}
    path=${contract#* }
    [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
      say "ABORT: sealed DB510 gather result drifted: $path"
      exit 2
    }
  done
  gather_remote_success_sha=$(gcloud storage cat \
    "$GATHER_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $gather_remote_success_sha == "$GATHER_SUCCESS_SHA" ]] || {
    say "ABORT: approved DB510 SUCCESS drifted"
    exit 2
  }
  /home/gianl/vllm-env/bin/python - \
    "$RESULTS_DB" "$GATHER_DIR" \
    "$RUN_DIR/gather_validation.json" \
    "$GATHER_RUN_ID" "$GATHER_ITEM_ROW_ID" "$GATHER_CODE_HASH" \
    "$GATHER_ASSOCIATION_MANIFEST_SHA" "$GATHER_OBSERVED_CACHE_SHA" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

(db_path, gather_path, output_path, run_id, item_id, code_hash,
 manifest_sha, observed_sha) = sys.argv[1:]
run_id, item_id = int(run_id), int(item_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("DB510 gather source DB integrity failed")
run = connection.execute(
    "SELECT harness_git,env_json,pod FROM runs WHERE run_id=?", (run_id,)
).fetchone()
item = connection.execute(
    "SELECT id,item_id,correct,score FROM items WHERE run_id=?", (run_id,)
).fetchone()
candidate_name = (
    "accepted_xla_m2048_gather_chunk_bf16_weight_divide_sqrt"
)
expected_item = (item_id, candidate_name, 0, 0.0)
if run is None or item != expected_item:
    raise SystemExit("DB510 gather source row drifted")
harness, env_json, pod = run
env = json.loads(env_json)
if (
    harness != code_hash[:7]
    or pod != "db-v4-64-od"
    or env.get("association_manifest_sha256") != manifest_sha
    or env.get("candidate_set") != "chunk_gather_bf16_weight"
    or env.get("association_parent_run_id") != 509
    or env.get("exact_candidates") != []
):
    raise SystemExit("DB510 gather source provenance drifted")
root = Path(gather_path)
association = json.loads((root / "association/association.json").read_text())
candidate = association["candidates"][candidate_name]
hlo = candidate["hlo"]["contract"]
if (
    association["manifest_sha256"] != manifest_sha
    or association["candidate_set"] != "chunk_gather_bf16_weight"
    or candidate["comparison_to_accepted"]["mismatch_count"] != 45
    or candidate["observed_bfloat16_sha256"] != observed_sha
    or hlo["loop_count"] != 0
    or hlo["physical_embedding_gather_count"] != 1
    or hlo["gather_coupled_input_rms"] is not True
    or hlo["bf16_wk_conversion_count"] != 1
    or hlo["convolution_weight_bf16"] is not True
):
    raise SystemExit("DB510 gather source artifact drifted")
Path(output_path).write_text(json.dumps({
    "status": "SUCCESS",
    "gather_run_id": run_id,
    "gather_item_row_id": item_id,
    "gather_code_hash": code_hash,
    "gather_association_manifest_sha256": manifest_sha,
    "gather_mismatch_count": 45,
}, indent=2, sort_keys=True) + "\n")
connection.close()
PY
fi

if [[ $PROFILE == chunk_gather_cache_write_source_rope ]]; then
  for contract in \
    "$CACHE_WRITE_ASSOCIATION_FILE_SHA $CACHE_WRITE_DIR/association/association.json" \
    "$CACHE_WRITE_TENSOR_FILE_SHA $CACHE_WRITE_DIR/association/candidate_prompt_index_keys.safetensors" \
    "$CACHE_WRITE_HLO_FILE_SHA $CACHE_WRITE_DIR/association/hlo/accepted_xla_m2048_gather_cache_write_bf16_weight_divide_sqrt.optimized_hlo.txt.gz" \
    "$CACHE_WRITE_SUMMARY_FILE_SHA $CACHE_WRITE_DIR/summary.json" \
    "$CACHE_WRITE_SUCCESS_SHA $CACHE_WRITE_DIR/SUCCESS" \
    "$CACHE_WRITE_RESULTS_DB_SHA $CACHE_WRITE_DIR/results_ckpt.db" \
    "$CACHE_WRITE_EVIDENCE_SHA $CACHE_WRITE_DIR/evidence.sha256" \
    "$CACHE_WRITE_REMOTE_OBJECTS_SHA $CACHE_WRITE_DIR/remote_objects.json" \
    "$CACHE_WRITE_PRE_CENSUS_SHA $CACHE_WRITE_DIR/census_pre.txt" \
    "$CACHE_WRITE_POST_CENSUS_SHA $CACHE_WRITE_DIR/census_post.txt" \
    "$CACHE_WRITE_MATRIX_VALIDATION_SHA $CACHE_WRITE_DIR/matrix_validation.json" \
    "$CACHE_WRITE_CHUNK_VALIDATION_SHA $CACHE_WRITE_DIR/chunk_parameter_validation.json" \
    "$CACHE_WRITE_BF16_VALIDATION_SHA $CACHE_WRITE_DIR/bf16_weight_validation.json" \
    "$CACHE_WRITE_GATHER_VALIDATION_SHA $CACHE_WRITE_DIR/gather_validation.json" \
    "$CACHE_WRITE_SOURCE_VALIDATION_SHA $CACHE_WRITE_DIR/source_validation.json"; do
    expected=${contract%% *}
    path=${contract#* }
    [[ $(sha256sum "$path" | awk '{print $1}') == "$expected" ]] || {
      say "ABORT: sealed DB511 cache-write result drifted: $path"
      exit 2
    }
  done
  cache_write_remote_success_sha=$(gcloud storage cat \
    "$CACHE_WRITE_REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
  [[ $cache_write_remote_success_sha == "$CACHE_WRITE_SUCCESS_SHA" ]] || {
    say "ABORT: approved DB511 SUCCESS drifted"
    exit 2
  }
  /home/gianl/vllm-env/bin/python - \
    "$RESULTS_DB" "$CACHE_WRITE_DIR" \
    "$RUN_DIR/cache_write_validation.json" \
    "$CACHE_WRITE_RUN_ID" "$CACHE_WRITE_ITEM_ROW_ID" \
    "$CACHE_WRITE_CODE_HASH" "$CACHE_WRITE_ASSOCIATION_MANIFEST_SHA" \
    "$CACHE_WRITE_OBSERVED_CACHE_SHA" \
    "$CACHE_WRITE_OPTIMIZED_HLO_SHA" <<'PY'
import gzip
import json
from pathlib import Path
import re
import sqlite3
import sys

(db_path, cache_write_path, output_path, run_id, item_id, code_hash,
 manifest_sha, observed_sha, optimized_hlo_sha) = sys.argv[1:]
run_id, item_id = int(run_id), int(item_id)
connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("DB511 cache-write source DB integrity failed")
run = connection.execute(
    "SELECT harness_git,env_json,pod FROM runs WHERE run_id=?", (run_id,)
).fetchone()
item = connection.execute(
    "SELECT id,item_id,correct,score FROM items WHERE run_id=?", (run_id,)
).fetchone()
candidate_name = (
    "accepted_xla_m2048_gather_cache_write_bf16_weight_divide_sqrt"
)
expected_item = (item_id, candidate_name, 0, 0.0)
if run is None or item != expected_item:
    raise SystemExit("DB511 cache-write source row drifted")
harness, env_json, pod = run
env = json.loads(env_json)
if (
    harness != code_hash[:7]
    or pod != "db-v4-64-od"
    or env.get("association_manifest_sha256") != manifest_sha
    or env.get("candidate_set")
    != "chunk_gather_cache_write_bf16_weight"
    or env.get("association_parent_run_id") != 510
    or env.get("exact_candidates") != []
):
    raise SystemExit("DB511 cache-write source provenance drifted")
root = Path(cache_write_path)
association = json.loads((root / "association/association.json").read_text())
candidate = association["candidates"][candidate_name]
hlo = candidate["hlo"]["contract"]
with gzip.open(
    root / "association/hlo/accepted_xla_m2048_gather_cache_write_"
    "bf16_weight_divide_sqrt.optimized_hlo.txt.gz",
    "rt",
) as stream:
    parent_hlo = stream.read().lower()
parent_rotary = {
    "cosine_count": len(
        re.findall(r"= f32\[2048,32\].*\bcosine\(", parent_hlo)
    ),
    "exponent_constant": "constant(0.015625)" in parent_hlo,
    "power_count": len(re.findall(r"= f32\[32\].*\bpower\(", parent_hlo)),
    "sine_count": len(
        re.findall(r"= f32\[2048,32\].*\bsine\(", parent_hlo)
    ),
    "theta_constant": "constant(8e+06)" in parent_hlo,
}
if (
    association["manifest_sha256"] != manifest_sha
    or association["candidate_set"]
    != "chunk_gather_cache_write_bf16_weight"
    or candidate["comparison_to_accepted"]["mismatch_count"] != 45
    or candidate["observed_bfloat16_sha256"] != observed_sha
    or candidate["hlo"]["optimized_hlo_sha256"] != optimized_hlo_sha
    or hlo["loop_count"] != 0
    or hlo["physical_embedding_gather_count"] != 1
    or hlo["gather_coupled_input_rms"] is not True
    or hlo["physical_cache_scatter_count"] != 1
    or hlo["cache_scatter_update_bf16"] is not True
    or parent_rotary != {
        "cosine_count": 1,
        "exponent_constant": True,
        "power_count": 1,
        "sine_count": 1,
        "theta_constant": True,
    }
):
    raise SystemExit("DB511 cache-write source artifact drifted")
Path(output_path).write_text(json.dumps({
    "status": "SUCCESS",
    "cache_write_run_id": run_id,
    "cache_write_item_row_id": item_id,
    "cache_write_code_hash": code_hash,
    "cache_write_association_manifest_sha256": manifest_sha,
    "cache_write_mismatch_count": 45,
    "cache_write_optimized_hlo_sha256": optimized_hlo_sha,
    "cache_write_rotary_physical_contract": parent_rotary,
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

say "running prompt-key association profile=$PROFILE on one TPU-v4 host"
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
if profile == "matrix":
    expected_names = {
        "production_pallas_m1_divide_sqrt",
        "accepted_xla_m2048_divide_sqrt",
        "accepted_xla_m2048_multiply_rsqrt",
    }
elif profile == "chunk_parameter":
    expected_names = {"accepted_xla_m2048_chunk_parameter_divide_sqrt"}
elif profile == "chunk_bf16_weight":
    expected_names = {
        "accepted_xla_m2048_chunk_bf16_weight_divide_sqrt"
    }
elif profile == "chunk_gather_bf16_weight":
    expected_names = {
        "accepted_xla_m2048_gather_chunk_bf16_weight_divide_sqrt"
    }
elif profile == "chunk_gather_cache_write_source_rope":
    expected_names = {
        "accepted_xla_m2048_gather_cache_write_bf16_weight_"
        "divide_sqrt_source_rope"
    }
else:
    expected_names = {
        "accepted_xla_m2048_gather_cache_write_bf16_weight_divide_sqrt"
    }
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

parent_hlo_evidence = {}
if profile == "chunk_gather_cache_write_source_rope":
    parent = json.loads((root / "cache_write_validation.json").read_text())
    current_hlo = next(iter(association["candidates"].values()))["hlo"]
    current_hlo_sha = current_hlo["optimized_hlo_sha256"]
    parent_hlo_sha = parent["cache_write_optimized_hlo_sha256"]
    current_rotary = dict(current_hlo["contract"]["rotary"])
    current_rotary.pop("source_literal")
    parent_rotary = parent["cache_write_rotary_physical_contract"]
    parent_hlo_evidence = {
        "parent_optimized_hlo_sha256": parent_hlo_sha,
        "optimized_hlo_identical_to_parent": (
            current_hlo_sha == parent_hlo_sha
        ),
        "rotary_physical_contract_identical_to_parent": (
            current_rotary == parent_rotary
        ),
    }

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
        **parent_hlo_evidence,
        "association_parent_run_id": {
            "chunk_parameter": 507,
            "chunk_bf16_weight": 508,
            "chunk_gather_bf16_weight": 509,
            "chunk_gather_cache_write_bf16_weight": 510,
            "chunk_gather_cache_write_source_rope": 511,
        }.get(profile),
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
    **parent_hlo_evidence,
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
  if [[ -f chunk_parameter_validation.json ]]; then
    sha256sum chunk_parameter_validation.json
  fi
  if [[ -f bf16_weight_validation.json ]]; then
    sha256sum bf16_weight_validation.json
  fi
  if [[ -f gather_validation.json ]]; then
    sha256sum gather_validation.json
  fi
  if [[ -f cache_write_validation.json ]]; then
    sha256sum cache_write_validation.json
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
