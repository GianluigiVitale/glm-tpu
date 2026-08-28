#!/bin/bash
# Read-only verifier for the private three-PR GLM-5.2 DSA submission stack.

set -euo pipefail

CODE_REPO="${GLM_DSA_CODE_REPO:-/home/gianl/tpu-inference-glm-baseline}"
DOC_REPO="${GLM_DSA_DOC_REPO:-/home/gianl/glm-tpu-topology-rewrite}"
RUN_ROOT="${GLM_DSA_RUN_ROOT:-/home/gianl/glm-run}"
BUCKET="${GLM_DSA_BUCKET:-gs://driftbench-dsv4-uc}"

BASE=5e2c7128bc74a75493f07930f3a749bcb272a3cb
PR1=fd29657d336cee859c17d4568f8d38d276ca9707
PR2=dfb28231b9e35c11659d3db3125bc18cc3177ab8
PR3=8aae29ad6da2b2cd778be031b423e31eb4a85a80
VLLM=d626108b1841888ec90aced33367149a6bbc7e4b
BUNDLE=glm-dsa-private-stack_20260828T015742Z.bundle
BUNDLE_SHA=d9b13b3906184286ad67ddeddd6d46d2bff29b04b9465a366c16f88245844bfb
BUNDLE_BYTES=12152990

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
assert_equal upstream_main "$BASE" "$(
  git -C "$CODE_REPO" ls-remote upstream refs/heads/main | awk '{print $1}'
)"

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

verify_evidence upstream_streamindex_test_20260828T011411Z \
  results/upstream_glm_dsa_pr1_correctness_20260828T011411Z \
  1e08dad8c1ee87487df08a380a17b65cebac9134a4e82157240e1e71b8138704 30
verify_evidence upstream_glm_dsa_benchmark_20260828T011531Z \
  results/upstream_glm_dsa_pr1_benchmark_20260828T011531Z \
  20cbf34d8576c8905d337f932584f3f9248865941bcbae63b9a68bf35bb8e285
verify_evidence upstream_glm_dsa_pr2_full_local_bounds_20260828T015207Z \
  results/upstream_glm_dsa_pr2_full_local_bounds_20260828T015207Z \
  f5b39594dda06bd0a3546611568db747a93259d5c1e9022ab570ae7aa02ed473 57
verify_evidence upstream_glm_dsa_pr3_local_bounds_20260828T015441Z \
  results/upstream_glm_dsa_pr3_local_bounds_20260828T015441Z \
  ce0f35f0d12d564132d0eb58fd9cfc5bf3c3b17182057bd79e7d2a8fac4cad31 3

python3 - "$RUN_ROOT/upstream_glm_dsa_benchmark_20260828T011531Z/benchmark.json" <<'PY'
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
    "lax_top_k_n262144_k2048": (2.007, 2.027),
    "streamindex_n262144_k2048_bkvp4": (3.208, 3.223),
    "index_cache_insert_n262144": (0.353, 0.412),
    "sparse_mla_k2048_block512": (0.159, 0.172),
    "gather_sparse_mla_n262144_k2048": (0.222, 0.253),
    "dsa_decode_chain_n262144_k2048": (3.280, 3.294),
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
