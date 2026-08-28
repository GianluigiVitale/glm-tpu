# Upstream GLM-5.2 DSA PR preparation status

Status date: 2026-08-28 UTC. This is private preparation evidence; nothing was
pushed to or opened against `vllm-project/tpu-inference`.

## Pins and stack

- Upstream TPU Inference base: `5e2c7128bc74a75493f07930f3a749bcb272a3cb`.
- vLLM source pin: `d626108b1841888ec90aced33367149a6bbc7e4b`.
- PR 1 branch/head: `pr/glm-dsa-kernels-v3` at
  `bb002856ca32e05e22ffdc49b06eed4c9d47843e` (private fork).
- PR 2 branch/head: `pr/glm-dsa-bridge-v3` at
  `b6d920922984106d8098d4b64226f0d39565c553` (private fork), stacked on PR 1.
- PR 3 branch/head: `pr/glm-dsa-model-ci-v3` at
  `f25c9a914bf6b98c05c657f461052ca355ffbca0` (private fork), stacked on PR 2.
- All eight private commits carry the repository-required DCO signoff. The
  DCO rewrite preserved each PR head's exact tree hash; recoverable local
  pre-rewrite refs remain under `backup/glm-dsa-*-pre-dco`.
- Restored environment: JAX/JAXLIB 0.10.1, fsspec 2025.3.0,
  google-cloud-storage 3.10.1, Tokamax 0.0.13.

## PR 1 — kernel foundation

Three commits provide exact FP8+FP32-scale StreamIndex scoring/top-k, paged
V3.2 index-cache insertion, and sparse MLA selected attention. The protected
suite passed 36/36 in 74.90 seconds, including 262,144-position/top-k-2,048
microbenchmarks and exact HLO checks. Evidence:
`upstream_streamindex_test_20260827T224406Z`; manifest-list SHA-256
`dd461d99feae8b582457954473577e1ee46ca059cf1d96314ee4b4f486daf170`.

The exact final DCO-signed head passed 29/29 focused protected kernel tests in
61.75 seconds. Evidence: `upstream_streamindex_test_20260828T004109Z`;
manifest-list SHA-256
`699505e28e5b7684b046d68972cf91872f5a00065985a0827d329ca976bb1251`.

## PR 2 — TorchAX/vLLM bridge

Three commits add paged slot mapping, the registered out-of-tree
`SparseAttnIndexer`, physical index-cache sizing, shared IndexShare top-k
wiring, and sparse MLA for both prefill and decode. Runtime metadata remains
owned by vLLM/TPU Inference. The implementation never revives legacy model
monkeypatches.

The current public checkpoint config resolves without loading weights as
`GlmMoeDsaForCausalLM`, model type `glm_moe_dsa`, with a 1,048,576-position
architectural limit, `index_topk=2048`, 32 index heads, and head dimension 128.
The protected target runtime was capped and exercised at 262,144 positions.
Production single-stream shape was verified as eight static decode token rows
plus one sequence's metadata. Phase comes from request metadata rather than an
ambiguous bucket shape. DSA and sparse MLA execute only the first live decode
row and return zero padding for the other seven. Sparse MLA explicitly
reconstructs complete query rows across TP before pairing them with global
top-k/cache metadata; a protected two-way-TP prefill regression proves that an
eight-row prefill is not misclassified as decode. TPU bucket-only rows beyond
vLLM's shared-buffer allocation are filled with `-1` sentinels rather than
reading stale indices.

Current explicit limitations are fail-closed: `max_num_seqs=1`,
`data_parallel_size=1`, `additional_config.enable_continue_decode=true`, no DCP/PCP, FP8+UE8M0
block-128 index cache only, and no quantized or transposed main MLA cache.

Validation:

- Focused two-device CPU regressions: 24/24; all repository pre-commit hooks
  pass, including the all-files run.
- Corrected decode/indexer plus sparse MLA: 2/2 protected TPU.
- 16-row sparse-prefill kernel versus reference: 1/1 protected TPU; evidence
  `upstream_streamindex_test_20260827T235002Z`, SHA
  `44f99014c5f38826dda2db2f0a57db908b9b1e35bed52bf9b18603017871b7c0`.
- Exact causal prefill indexer rows/cache writes: 1/1 protected TPU; evidence
  `upstream_streamindex_test_20260827T235116Z`, SHA
  `d38d4e97327393c34371b46df8912912ccf1eeb63a24a81b4b7cb46eae36b551`.
- Final focused bridge suite: 22/22 in 44.04 seconds; evidence
  `upstream_streamindex_test_20260827T235340Z`, SHA
  `6d3a6b445568f8b4a5cd25c97feb50316d97acc0fc48e1259290b14591dd70fe`.
- Every protected run ended with authenticated `CENSUS_OK` on all eight hosts.
- Exact final-head bridge suite: 55/55 in 97.15 seconds, including real v4
  scorer/top-k, index-cache, sparse-MLA, shared-buffer, metadata-phase, and TP2
  tests. Evidence `upstream_streamindex_test_20260828T004235Z`, manifest-list
  SHA-256
  `61a78d59e11eeb17a7a7c8b9bb6c2ee9a3e28b77c407b56b74c2f666be62dffe`.

A deliberately broader 66-test run passed all 63 relevant tests but also
reproduced three existing dense FP8-cache v4 Mosaic legalization failures in
`kv_utils.py`; those tests do not exercise this sparse PR stack. The focused
suite above isolates and passes the claimed contract.

These are correctness/integration results, not full-model latency claims.

## PR 3 — model contract and CI

Two commits add a no-weight GLM-5.2 model-contract test and a model-specific
Buildkite entry. The test verifies that `GlmMoeDsaForCausalLM` resolves through
the current vLLM registry, that all decoder layers receive the same shared
top-k buffer, and that the public GLM-5.2 IndexShare schedule is interpreted as
three initial producers, three consumers, then the next producer. The CI unit
step runs on the single-TPU queue. Accuracy and performance remain explicitly
`unverified`; this PR does not imply that the 753B checkpoint fits that queue.
The unit step also executes the PR 1 kernel and PR 2 bridge/sparse-backend
regressions, so the model CI cannot pass while omitting its stacked runtime.

Validation:

- Focused CPU test: 3/3 in 8.88 seconds; all repository pre-commit hooks pass.
- Protected TPU model-contract test at the committed PR 3 head: 3/3 in 9.09
  seconds; evidence `upstream_streamindex_test_20260828T001254Z`, manifest-list
  SHA-256
  `f8f3bf38cd2dfed2735c82010f86af3b57e2cfe87b350726cdabe8b89690e5d3`.
- The protected run ended with authenticated `CENSUS_OK` on all eight hosts.
- Exact final DCO-signed head: 3/3 in 9.86 seconds; evidence
  `upstream_streamindex_test_20260828T004433Z`, manifest-list SHA-256
  `cbe5fbb9129b425add5136b3c100ed9d19320d8cc49db48e98b5d0aa499c4149`.
- The repository metadata validator was also inspected. Its current empty-env
  query flags the same metadata-free execution steps in existing model YAMLs;
  the new file follows those existing conventions and has unique, complete
  metadata on every result-recording step.

## Stack status and next action

All three private stacked branches are prepared and pushed only to the user's
private fork. The one permitted independent Sol review found TP row/metadata,
bucket-buffer, evidence-head, and CI-coverage gaps; each has been corrected and
revalidated above. Compact evidence is mirrored under matching
`gs://driftbench-dsv4-uc/results/upstream_glm_dsa_pr{1,2,3}_*` prefixes in the
verified `US-CENTRAL2` regional bucket. Present the exact diffs to the user for
audit. No upstream push or PR is authorized yet.
