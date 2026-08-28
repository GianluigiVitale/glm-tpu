#!/bin/bash
# Read-only verifier for the private three-PR GLM-5.2 DSA submission stack.

set -euo pipefail

CODE_REPO="${GLM_DSA_CODE_REPO:-/home/gianl/tpu-inference-glm-baseline}"
DOC_REPO="${GLM_DSA_DOC_REPO:-/home/gianl/glm-tpu-topology-rewrite}"
RUN_ROOT="${GLM_DSA_RUN_ROOT:-/home/gianl/glm-run}"
BUCKET="${GLM_DSA_BUCKET:-gs://driftbench-dsv4-uc}"

BASE=e08b64c14208cb5efc34cc3b41eeaa3402346911
PR1=650b5fccb890b5a872871af489b50fc4c584e8ad
PR2=d837832ab41f947ee9ff759e65ea8417ba1bd5c9
PR3=101ec506d76a3ecb0b688315e432ae7e8d0ab37a
VLLM=d626108b1841888ec90aced33367149a6bbc7e4b
BUNDLE=glm-dsa-private-stack_20260828T090200Z.bundle
BUNDLE_SHA=38800e54e9b03ffe930a8629c0ea6b4039e4cbb19d188a54665889ddb8e31f49
BUNDLE_BYTES=12158343

fail() {
  echo "READINESS_FAIL: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null || fail "missing command: $1"
}

assert_equal() {
  local label="$1"
  local expected="$2"
  local actual="$3"
  [[ "$actual" == "$expected" ]] || fail "$label expected=$expected actual=$actual"
  echo "OK $label $actual"
}

verify_ref() {
  local ref="$1"
  local expected="$2"
  assert_equal "local_ref:$ref" "$expected" "$(git -C "$CODE_REPO" rev-parse "$ref")"
  assert_equal "private_ref:$ref" "$expected" "$(
    git -C "$CODE_REPO" ls-remote origin "refs/heads/$ref" | awk '{print $1}'
  )"
}

verify_range() {
  local name="$1"
  local range="$2"
  local expected_files="$3"
  local expected_adds="$4"
  local expected_dels="$5"
  local expected_commits="$6"
  local stats
  stats="$(git -C "$CODE_REPO" diff --numstat "$range" | awk '
    {files += 1; adds += $1; dels += $2}
    END {printf "%d %d %d", files, adds, dels}')"
  assert_equal "$name:stats" \
    "$expected_files $expected_adds $expected_dels" "$stats"
  assert_equal "$name:commits" "$expected_commits" "$(
    git -C "$CODE_REPO" rev-list --count "$range"
  )"
  local commit
  while read -r commit; do
    git -C "$CODE_REPO" show -s --format=%B "$commit" |
      grep -q '^Signed-off-by:' || fail "$name commit $commit lacks DCO"
  done < <(git -C "$CODE_REPO" rev-list "$range")
  echo "OK $name:DCO $expected_commits/$expected_commits"
}

verify_evidence() {
  local local_tag="$1"
  local remote_rel="$2"
  local expected_sha="$3"
  local expected_tests="${4:-}"
  local evidence_file="$RUN_ROOT/$local_tag/evidence.sha256"
  [[ -f "$evidence_file" ]] || fail "missing evidence $evidence_file"
  assert_equal "evidence:$local_tag" "$expected_sha" "$(
    sha256sum "$evidence_file" | awk '{print $1}'
  )"
  assert_equal "remote_evidence:$local_tag" "$expected_sha" "$(
    gcloud storage cat "$BUCKET/$remote_rel/evidence.sha256" | sha256sum | awk '{print $1}'
  )"
  if [[ -n "$expected_tests" ]]; then
    grep -q "$expected_tests passed" "$RUN_ROOT/$local_tag/pytest.txt" ||
      fail "$local_tag lacks '$expected_tests passed'"
    echo "OK tests:$local_tag $expected_tests"
  fi
  grep -q "TPU_INFERENCE_HEAD" "$RUN_ROOT/$local_tag/provenance.txt" ||
    fail "$local_tag lacks head provenance"
  grep -q "VLLM_PIN $VLLM" "$RUN_ROOT/$local_tag/provenance.txt" ||
    fail "$local_tag has wrong vLLM pin"
}

for command in git awk grep sha256sum stat gcloud python3; do
  require_command "$command"
done
[[ "$(git -C "$CODE_REPO" rev-parse --is-inside-work-tree 2>/dev/null)" == true ]] ||
  fail "not a git worktree: $CODE_REPO"
[[ "$(git -C "$DOC_REPO" rev-parse --is-inside-work-tree 2>/dev/null)" == true ]] ||
  fail "not a git worktree: $DOC_REPO"
[[ -z "$(git -C "$CODE_REPO" status --porcelain --untracked-files=no)" ]] ||
  fail "code repository has tracked changes"

assert_equal origin_url git@github.com:GianluigiVitale/tpu-inference.git "$(
  git -C "$CODE_REPO" remote get-url origin
)"
assert_equal upstream_url https://github.com/vllm-project/tpu-inference.git "$(
  git -C "$CODE_REPO" remote get-url upstream
)"
UPSTREAM_MAIN="$(git -C "$CODE_REPO" ls-remote upstream \
  refs/heads/main | awk '{print $1}')"
git -C "$CODE_REPO" merge-base --is-ancestor "$BASE" "$UPSTREAM_MAIN" ||
  fail "audited base is not an ancestor of current upstream main"
overlap="$({
  git -C "$CODE_REPO" diff --name-only "$BASE..$UPSTREAM_MAIN"
  git -C "$CODE_REPO" diff --name-only "$BASE..$PR1"
} | sort | uniq -d)"
[[ -z "$overlap" ]] || fail "upstream moved across PR1 paths: $overlap"
echo "OK upstream_main_descends_from_base $UPSTREAM_MAIN"
echo "OK upstream_main_PR1_path_overlap none"

verify_ref pr/glm-dsa-kernels-v3 "$PR1"
verify_ref pr/glm-dsa-bridge-v3 "$PR2"
verify_ref pr/glm-dsa-model-ci-v3 "$PR3"
git -C "$CODE_REPO" merge-base --is-ancestor "$BASE" "$PR1" || fail "base !< PR1"
git -C "$CODE_REPO" merge-base --is-ancestor "$PR1" "$PR2" || fail "PR1 !< PR2"
git -C "$CODE_REPO" merge-base --is-ancestor "$PR2" "$PR3" || fail "PR2 !< PR3"
echo "OK ancestry BASE->PR1->PR2->PR3"

verify_range PR1 "$BASE..$PR1" 8 1250 10 5
verify_range PR2 "$PR1..$PR2" 11 1344 34 4
verify_range PR3 "$PR2..$PR3" 2 302 0 2

largest="$(git -C "$CODE_REPO" ls-tree -r --long "$PR3" | awk '
  $4 > max {max=$4; path=$5} END {print max " " path}')"
largest_bytes="${largest%% *}"
((largest_bytes < 1048576)) || fail "tracked file >=1MiB: $largest"
echo "OK largest_tracked_file $largest"
if git -C "$CODE_REPO" diff --name-only "$BASE..$PR3" |
  grep -Eiq '(^|/)(checkpoints?|runs?|environments?|artifacts?)/|[.](safetensors|npy|npz|xplane|trace)$'; then
  fail "bulk/generated path found in stack"
fi
if git -C "$CODE_REPO" diff --unified=0 "$BASE..$PR3" -- \
    tpu_inference tests scripts .buildkite |
  grep -Eq '^\+[^+].*(glm_tpu[./]greenfield|from glm_tpu|import glm_tpu|ray[.])'; then
  fail "greenfield/Ray execution import found"
fi
echo "OK no_bulk_or_greenfield_execution_imports"

verify_evidence upstream_glm_dsa_pr1_rebase_20260828T085145Z \
  results/upstream_glm_dsa_pr1_rebase_20260828T085145Z \
  1e20d3c5342c25537dfc2e98ee51e22aea0ca34ba66bef5e2778be4868bfcf3c 30
verify_evidence upstream_glm_dsa_pr1_benchmark_rebase_20260828T085526Z \
  results/upstream_glm_dsa_pr1_benchmark_rebase_20260828T085526Z \
  54ec928ba42d5426c5d9096ea0959bbf2fede52c45b43eb232e5b01f95119575
verify_evidence upstream_glm_dsa_pr2_rebase_20260828T085714Z \
  results/upstream_glm_dsa_pr2_rebase_20260828T085714Z \
  7731e8397b58b323b1f68e82509005dce24dc121bd2e235c194d95200a0383de 57
verify_evidence upstream_glm_dsa_pr3_rebase_20260828T085938Z \
  results/upstream_glm_dsa_pr3_rebase_20260828T085938Z \
  a3cbb2b461d8329c8dc48f083671ac0cc02732ca1f4f31fcf7cdfaefc6a09df4 3

python3 - "$RUN_ROOT/upstream_glm_dsa_pr1_benchmark_rebase_20260828T085526Z/benchmark.json" <<'PY'
import json
import sys

raw = open(sys.argv[1], encoding="utf-8").read()
data = json.loads(raw[raw.index("{"):])
assert data["config"] == {
    "mla_block_size": 512,
    "pages_per_block": 4,
    "samples": 20,
    "seq_len": 262144,
    "topk": 2048,
    "warmup": 5,
}
expected = {
    "lax_top_k_n262144_k2048": (2.009, 2.041),
    "streamindex_n262144_k2048_bkvp4": (3.216, 3.248),
    "index_cache_insert_n262144": (0.374, 0.434),
    "sparse_mla_k2048_block512": (0.182, 0.201),
    "gather_sparse_mla_n262144_k2048": (0.244, 0.264),
    "dsa_decode_chain_n262144_k2048": (3.292, 3.318),
}
actual = {
    item["name"]: (round(item["p50_ms"], 3), round(item["p99_ms"], 3))
    for item in data["measurements"]
}
for name, values in expected.items():
    assert actual[name] == values, (name, actual[name], values)
print("OK benchmark_json exact config/metrics")
PY

bundle_path="$RUN_ROOT/$BUNDLE"
assert_equal bundle_bytes "$BUNDLE_BYTES" "$(stat -c %s "$bundle_path")"
assert_equal bundle_sha "$BUNDLE_SHA" "$(sha256sum "$bundle_path" | awk '{print $1}')"
assert_equal remote_bundle_sha "$BUNDLE_SHA" "$(
  gcloud storage cat "$BUCKET/backups/tpu-inference-glm-dsa/$BUNDLE" |
    sha256sum | awk '{print $1}'
)"
assert_equal bucket_location US-CENTRAL2 "$(
  gcloud storage buckets describe "$BUCKET" --format='value(location)'
)"

for doc in \
  docs/upstream/glm-dsa-pr1-owner-audit.md \
  docs/upstream/glm-dsa-pr2-owner-audit.md \
  docs/upstream/glm-dsa-pr3-owner-audit.md \
  docs/upstream/glm-dsa-owner-audit-index.md \
  docs/upstream-glm-dsa-pr-drafts.md; do
  [[ -f "$DOC_REPO/$doc" ]] || fail "missing review document: $doc"
done
grep -q "$PR1" "$DOC_REPO/docs/upstream/glm-dsa-owner-audit-index.md" ||
  fail "owner index lacks exact PR1 head"

echo "READINESS_OK_PRE_OWNER_AUDIT"
echo "OWNER_APPROVAL_REQUIRED PR1_HEAD=$PR1"
echo "NO_UPSTREAM_MUTATION_AUTHORIZED"
