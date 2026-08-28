# Upstream GLM-5.2 DSA PR preparation status

Status date: 2026-08-28 UTC. This is private preparation evidence; nothing was
pushed to or opened against `vllm-project/tpu-inference`.

## Pins and stack

- Upstream TPU Inference base: `5e2c7128bc74a75493f07930f3a749bcb272a3cb`.
- vLLM source pin: `d626108b1841888ec90aced33367149a6bbc7e4b`.
- PR 1 branch/head: `pr/glm-dsa-kernels-v3` at
  `d3ccafdfe5157dda4c26b779dc50eb505edfa96b` (private fork).
- PR 2 branch/head: `pr/glm-dsa-bridge-v3` at
  `b1c863f44a783b8d44b52210c441fd360d50f691` (private fork), stacked on PR 1.
- PR 3 branch/head: `pr/glm-dsa-model-ci-v3` at
  `f7ed37ea5625577bffd5c6133bca4aa7cbbf2050` (private fork), stacked on PR 2.
- Restored environment: JAX/JAXLIB 0.10.1, fsspec 2025.3.0,
  google-cloud-storage 3.10.1, Tokamax 0.0.13.

## PR 1 — kernel foundation

Three commits provide exact FP8+FP32-scale StreamIndex scoring/top-k, paged
V3.2 index-cache insertion, and sparse MLA selected attention. The protected
suite passed 36/36 in 74.90 seconds, including 262,144-position/top-k-2,048
microbenchmarks and exact HLO checks. Evidence:
`upstream_streamindex_test_20260827T224406Z`; manifest-list SHA-256
`dd461d99feae8b582457954473577e1ee46ca059cf1d96314ee4b4f486daf170`.

## PR 2 — TorchAX/vLLM bridge

Two commits add paged slot mapping, the registered out-of-tree
`SparseAttnIndexer`, physical index-cache sizing, shared IndexShare top-k
wiring, and sparse MLA for both prefill and decode. Runtime metadata remains
owned by vLLM/TPU Inference. The implementation never revives legacy model
monkeypatches.

The real checkpoint config resolves without loading weights as
`GlmMoeDsaForCausalLM`, model type `glm_moe_dsa`, max length 262,144,
`index_topk=2048`, 32 index heads, head dimension 128. Production single-stream
shape was verified as eight static decode token rows plus one sequence's
metadata; DSA and sparse MLA execute only the first live decode row and return
zero padding for the other seven.

Current explicit limitations are fail-closed: `max_num_seqs=1`,
`additional_config.enable_continue_decode=true`, no DCP/PCP, FP8+UE8M0
block-128 index cache only, and no quantized or transposed main MLA cache.

Validation:

- Focused CPU regressions: 26/26; all repository pre-commit hooks pass.
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

These are correctness/integration results, not full-model latency claims.

## PR 3 — model contract and CI

One commit adds a no-weight GLM-5.2 model-contract test and a model-specific
Buildkite entry. The test verifies that `GlmMoeDsaForCausalLM` resolves through
the current vLLM registry, that all decoder layers receive the same shared
top-k buffer, and that the public GLM-5.2 IndexShare schedule is interpreted as
three initial producers, three consumers, then the next producer. The CI unit
step runs on the single-TPU queue. Accuracy and performance remain explicitly
`unverified`; this PR does not imply that the 753B checkpoint fits that queue.

Validation:

- Focused CPU test: 3/3 in 8.88 seconds; all repository pre-commit hooks pass.
- Protected TPU model-contract test: 3/3 in 8.90 seconds; evidence
  `upstream_streamindex_test_20260828T000713Z`, manifest-list SHA-256
  `b8e991f3f494e155226ae77bc97b6b8111bd9206f439e0203ed35342f6de3c1e`.
- The protected run ended with authenticated `CENSUS_OK` on all eight hosts.
- The repository metadata validator was also inspected. Its current empty-env
  query flags the same metadata-free execution steps in existing model YAMLs;
  the new file follows those existing conventions and has unique, complete
  metadata on every result-recording step.

## Stack status and next action

All three private stacked branches are prepared. One final independent Sol
review is in progress over the exact diffs and evidence. Correct any concrete
findings, then provide the user with exact reviewable diffs and ready-to-paste
PR descriptions. The user must audit them before any upstream action.
