# Upstream GLM-5.2 DSA PR preparation status

Status date: 2026-08-28 UTC. Owner-approved PR 1 is open upstream as
https://github.com/vllm-project/tpu-inference/pull/3480. PRs 2-3 remain private
and unsubmitted.

## Pins and stack

- Upstream TPU Inference base: `e08b64c14208cb5efc34cc3b41eeaa3402346911`.
- vLLM source pin: `d626108b1841888ec90aced33367149a6bbc7e4b`.
- PR 1 branch/head: `pr/glm-dsa-kernels-v3` at
  `650b5fccb890b5a872871af489b50fc4c584e8ad` (public submission fork,
  upstream PR #3480).
- PR 2 branch/head: `pr/glm-dsa-bridge-v3` at
  `d837832ab41f947ee9ff759e65ea8417ba1bd5c9` (private fork), stacked on PR 1.
- PR 3 branch/head: `pr/glm-dsa-model-ci-v3` at
  `101ec506d76a3ecb0b688315e432ae7e8d0ab37a` (private fork), stacked on PR 2.
- All eleven private commits carry the repository-required DCO signoff. The
  DCO rewrite preserved each PR head's exact tree hash; recoverable local
  pre-rewrite refs remain under `backup/glm-dsa-*-pre-dco`.
- Restored environment: JAX/JAXLIB 0.10.1, fsspec 2025.3.0,
  google-cloud-storage 3.10.1, Tokamax 0.0.13.

## PR 1 — kernel foundation

Five commits provide exact FP8+FP32-scale StreamIndex scoring/top-k, paged
V3.2 index-cache insertion, sparse MLA selected attention, and a repository-
native bounded TPU benchmark. The protected
suite passed 36/36 in 74.90 seconds, including 262,144-position/top-k-2,048
microbenchmarks and exact HLO checks. Evidence:
`upstream_streamindex_test_20260827T224406Z`; evidence-manifest SHA-256
`dd461d99feae8b582457954473577e1ee46ca059cf1d96314ee4b4f486daf170`.

The exact current head passed 30/30 focused protected kernel/HLO tests in
61.19 seconds. Evidence: `upstream_streamindex_test_20260828T011411Z`;
evidence-manifest SHA-256
`1e08dad8c1ee87487df08a380a17b65cebac9134a4e82157240e1e71b8138704`.
The tracked benchmark then ran independently on that same clean head with five
warmups and 20 individually synchronized samples: scorer p50 3.208 ms, cache
insert 0.353 ms, sparse MLA 0.159 ms, selected gather plus MLA 0.222 ms, and
composed scorer/gather/MLA 3.280 ms. Evidence:
`upstream_glm_dsa_benchmark_20260828T011531Z`; evidence-manifest SHA-256
`20cbf34d8576c8905d337f932584f3f9248865941bcbae63b9a68bf35bb8e285`.
One initial direct-script launch failed before compilation because the repo
root was absent from `PYTHONPATH`; it is preserved as
`upstream_glm_dsa_benchmark_20260828T005800Z` (SHA
`244b114239c4a7a76efa4cef9fc60b3462b99a727b844537d9998b5f40ca1255`).
Both attempts ended 8/8 clean.

After official main advanced by two non-overlapping CI/support-matrix commits,
the stack was rebased without changing any PR patch bytes. The exact new PR 1
head passed 30/30 in 60.16 seconds and the refreshed 256K benchmark measured
the composed chain at 3.292 ms p50 / 3.318 ms p99. Evidence tags are
`upstream_glm_dsa_pr1_rebase_20260828T085145Z` and
`upstream_glm_dsa_pr1_benchmark_rebase_20260828T085526Z`; evidence-manifest
SHA-256 values are respectively
`1e20d3c5342c25537df08a380a17b65cebac9134a4e82157240e1e71b8138704` and
`54ec928ba42d5426c5d9096ea0959bbf2fede52c45b43eb232e5b01f95119575`.

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
Requesting sparse MLA without the model-owned shared top-k buffer also raises
immediately; it can no longer silently select the dense backend.

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
- Exact current-head bridge suite: 56/56 in 97.71 seconds, including real v4
  scorer/top-k, index-cache, sparse-MLA, shared-buffer, metadata-phase, and TP2
  tests. Evidence `upstream_streamindex_test_20260828T011620Z`, manifest-list
  SHA-256
  `8ad57d3ea3ee16008b40c03cb4ed5bc6dc7c4f4a6a18f1978ee15ffad332c608`.
- Corrected fail-closed head: 57/57 in 98.64 seconds, including the missing-
  buffer rejection and all prior real-v4 contracts. Evidence
  `upstream_glm_dsa_pr2_full_local_bounds_20260828T015207Z`, evidence-manifest
  SHA-256
  `f5b39594dda06bd0a3546611568db747a93259d5c1e9022ab570ae7aa02ed473`.

A deliberately broader 66-test run passed all 63 relevant tests but also
reproduced three existing dense FP8-cache v4 Mosaic legalization failures in
`kv_utils.py`; those tests do not exercise this sparse PR stack. The focused
suite above isolates and passes the claimed contract.

These are correctness/integration results, not full-model latency claims.

The rebased exact PR 2 head passed the same 57/57 protected suite in 99.99
seconds. Evidence `upstream_glm_dsa_pr2_rebase_20260828T085714Z`,
evidence-manifest SHA-256
`7731e8397b58b323b1f68e82509005dce24dc121bd2e235c194d95200a0383de`.

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
- Exact current head: 3/3 in 9.59 seconds; evidence
  `upstream_streamindex_test_20260828T011829Z`, evidence-manifest SHA-256
  `0cd31c71911a890ce36f917e624cc60a234a1ede39ee7b7bcd140e5d5743a250`.
- Restacked fail-closed head: 3/3 in 9.24 seconds; evidence
  `upstream_glm_dsa_pr3_local_bounds_20260828T015441Z`, evidence-manifest SHA-256
  `ce0f35f0d12d564132d0eb58fd9cfc5bf3c3b17182057bd79e7d2a8fac4cad31`.
- The repository metadata validator was also inspected. Its current empty-env
  queue parser emits a literal `\t` and consequently reports all queues as
  empty, including plain `cpu`; unchanged GLM-5 and DeepSeek-V3.2 model YAMLs
  fail identically. A separate structural audit passes all six new steps,
  unique keys/target, dependencies, referenced test paths, and recording-step
  metadata. Do not hide the upstream validator defect inside this model PR.
- The rebased exact PR 3 head passed 3/3 in 9.43 seconds. Evidence
  `upstream_glm_dsa_pr3_rebase_20260828T085938Z`, evidence-manifest SHA-256
  `a3cbb2b461d8329c8dc48f083671ac0cc02732ca1f4f31fcf7cdfaefc6a09df4`.

## Stack status and next action

All three stacked branches are prepared. PR 1's exact approved head is public
only on `GianluigiVitale/tpu-inference-glm-dsa` and open upstream as #3480;
PRs 2-3 remain only on the user's private fork. The one permitted independent
Sol review found TP row/metadata, bucket-buffer, evidence-head, and CI-coverage
gaps; each has been corrected and revalidated above. Compact evidence is
mirrored under matching
`gs://driftbench-dsv4-uc/results/upstream_glm_dsa_pr{1,2,3}_*` prefixes in the
verified `US-CENTRAL2` regional bucket. PR 1 is 8 files (+1,250/-10), PR 2 is
11 files (+1,344/-34), and PR 3 is 2 files (+302). The final stack tracks 1,103
files, contains no tracked file above 1 MiB, and has no checkpoint, trace,
cache, environment, or generated run tree. The audited branch base is
`e08b64c14208cb5efc34cc3b41eeaa3402346911`; live `upstream/main` is
`b256da42e879f6070dc517cf171eb2ab8f6b2c9e` with only non-overlapping nightly
support-matrix changes after that base. PR #3480 is open and DCO passes;
Buildkite intentionally stops until a maintainer applies the `ready` label,
so its zero-second failure is not a test failure. The five-minute same-region
cron now mirrors the authoritative `tpu-inference-glm-baseline` worktree to
`gs://driftbench-dsv4-uc/repos/tpu-inference-glm-baseline`. A separately
verified complete-history three-ref bundle is at
`gs://driftbench-dsv4-uc/backups/tpu-inference-glm-dsa/`
`glm-dsa-private-stack_20260828T090200Z.bundle` (12,158,343 bytes; SHA-256
`38800e54e9b03ffe930a8629c0ea6b4039e4cbb19d188a54665889ddb8e31f49`).

The protected harness must set `TPU_PROCESS_BOUNDS=1,1,1`,
`TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1`, and four visible local chips. Omitting the
process bound made libtpu inherit the full `1,2,4` pod and wait in
`CreateTpuSystemState` for absent peer pytest processes. Three bounded failed
attempts and the unnecessary but harmless all-host runtime-service restart are
preserved; the corrected one-test probe passed in 14.74 seconds before the
full suites above. Do not publish or open PR 2 until the owner audits and
explicitly approves exact head `d837832ab41f947ee9ff759e65ea8417ba1bd5c9`.

Exact owner-facing checklists for the dependent patches are
`docs/upstream/glm-dsa-pr2-owner-audit.md` and
`docs/upstream/glm-dsa-pr3-owner-audit.md`. They do not change the submission
order: present PR 1 first, then PR 2, then PR 3.
