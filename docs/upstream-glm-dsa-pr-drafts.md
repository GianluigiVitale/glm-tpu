# Draft upstream GLM-5.2 DSA PR series

Private review material only. Do not open these PRs until the user has audited
the exact diffs. The series is stacked and must be reviewed/merged in order.

## PR 1 — `kernels: add exact V3.2/GLM sparse-attention primitives`

Base: `vllm-project/tpu-inference:main` at `5e2c7128`  
Head: private `pr/glm-dsa-kernels-v3` at `fd29657d`

### Summary

This generalizes the existing experimental StreamIndex kernel for the current
vLLM DeepSeek-V3.2/GLM index-cache format and adds the two missing selected-MLA
primitives. It does not add a model fork or a second execution architecture.

- Decode FP8 index records with packed FP32 power-of-two scales.
- Select exact top-k indices with deterministic low-index tie ordering.
- Quantize, pack, and insert V3.2 indexer keys into the paged cache.
- Gather selected paged MLA rows and compute sparse selected attention.

### Validation

- Protected TPU suite: 36/36, 8/8 hosts clean.
- Exact selected set/order at 2,047/2,048/2,049 boundaries and production
  `n=262144, k=2048`.
- Exact current head: 30/30 kernel/HLO tests in 61.19 seconds, 8/8 hosts clean.
- Warming: 5 iterations; samples: 20 on TPU v4.
- Reproduce from the repository root with
  `PYTHONPATH=. python scripts/benchmarking/kernels/benchmark_glm_dsa.py --mode all`.
- Current tracked-harness median wall times: StreamIndex 3.208 ms, cache insert
  0.353 ms, selected gather plus MLA 0.222 ms, sparse MLA 0.159 ms, composed
  DSA chain 3.280 ms.
- Exact StableHLO captured; no end-to-end throughput claim is made.

Evidence: `upstream_streamindex_test_20260827T224406Z`, evidence-manifest SHA-256
`dd461d99feae8b582457954473577e1ee46ca059cf1d96314ee4b4f486daf170`.
Current-head correctness evidence: `upstream_streamindex_test_20260828T011411Z`,
evidence-manifest SHA-256
`1e08dad8c1ee87487df08a380a17b65cebac9134a4e82157240e1e71b8138704`.
Current-head benchmark evidence:
`upstream_glm_dsa_benchmark_20260828T011531Z`, evidence-manifest SHA-256
`20cbf34d8576c8905d337f932584f3f9248865941bcbae63b9a68bf35bb8e285`.

### Compatibility, risk, and rollback

- Existing DeepSeek-v4 behavior remains the default: E8M0 scale storage and
  approximate top-k are unchanged unless the new explicit options are used.
  New V3.2 cache/sparse-MLA APIs remain under `kernels/experimental`.
- Direct performance evidence is TPU v4 only and covers isolated kernels, not
  full-checkpoint serving, accuracy, HBM capacity, or other TPU generations.
- Rollback is a five-commit revert. It removes only the new experimental
  primitives/tests/benchmark and restores the StreamIndex defaults without a
  checkpoint or state migration.

## PR 2 — `layers: bridge V3.2/GLM sparse attention to TPU`

Base: PR 1 `fd29657d`

Head: private `pr/glm-dsa-bridge-v3` at `dfb28231`

### Summary

This connects vLLM's existing `GlmMoeDsaForCausalLM`/DeepSeek-V3.2 model path
to the PR 1 kernels through TPU out-of-tree custom ops. vLLM remains responsible
for model registration, layer scheduling, weight loading, metadata, and the
IndexShare buffer.

- Register a TPU `SparseAttnIndexer` implementation.
- Allocate the physical index cache and derive paged insertion slots.
- Preserve one shared top-k buffer across IndexShare producer/consumer layers.
- Execute sparse MLA for decode and causal prefill; never silently fall back to
  dense attention for a DSA layer.
- Reconstruct complete token rows across TP before consuming global
  top-k/cache metadata; reject data parallelism until it has its own proof.
- Derive decode from request metadata, not bucket shape, and sentinel-pad TPU
  bucket rows beyond vLLM's unpadded shared-buffer capacity.
- On the static eight-row single-stream decode bucket, execute DSA only for the
  one live row and zero-pad the remaining rows.
- Fail closed for unimplemented configurations: more than one sequence,
  data parallelism, continue-decode disabled, DCP/PCP,
  non-FP8/UE8M0/block-128 index caches, and quantized or transposed MLA caches.
- Fail closed when sparse MLA is requested without the model-owned shared
  top-k buffer; never silently select dense MLA for that configuration.

### Validation

- Focused forced two-device CPU regressions pass; repository pre-commit hooks
  pass.
- Final protected TPU bridge suite: 22/22 in 44.04 seconds, 8/8 hosts clean.
- Additional protected exact causal-prefill and selected-MLA tests pass.
- Corrected current head: 57/57 protected tests in 98.64 seconds, including a
  two-device TP prefill with the same eight-row shape as decode; 8/8 hosts
  clean.
- No full-checkpoint latency, accuracy, or serving claim is made.

Primary evidence: `upstream_streamindex_test_20260827T235340Z`, manifest-list
SHA-256
`6d3a6b445568f8b4a5cd25c97feb50316d97acc0fc48e1259290b14591dd70fe`.
Current-head evidence:
`upstream_glm_dsa_pr2_full_local_bounds_20260828T015207Z`,
evidence-manifest SHA-256
`f5b39594dda06bd0a3546611568db747a93259d5c1e9022ab570ae7aa02ed473`.

### Compatibility, risk, and rollback

- The existing dense MLA path is unchanged when no shared top-k buffer is
  supplied. The bridge uses vLLM's current model, metadata, cache ownership,
  and IndexShare schedule rather than adding a competing model path.
- Unsupported multi-sequence, DP, DCP/PCP, continue-decode-off, and alternate
  cache layouts fail closed. Full 753B serving and quality remain unclaimed.
- Rollback is a three-commit revert of PR 2 (and PR 3 if stacked). PR 1 remains
  independently useful; no persistent checkpoint format is migrated.

## PR 3 — `models: add GLM-5.2 DSA contract CI`

Base: PR 2 `dfb28231`

Head: private `pr/glm-dsa-model-ci-v3` at `8aae29ad`

### Summary

Current vLLM already registers `GlmMoeDsaForCausalLM`; duplicating that
registration in TPU Inference would be incorrect. This PR instead locks down
the cross-repository model contract and makes the model's unit step real.

- Verify the public GLM-5.2 architecture resolves through vLLM's registry.
- Verify one top-k buffer is shared across decoder layers.
- Verify the public three-producer/three-consumer IndexShare prefix and next
  producer are constructed correctly.
- Add a model Buildkite unit step on the single-TPU queue.
- Run the stacked kernel, cache, sparse-MLA, indexer bridge, and backend
  regressions in that unit step, not only the registry contract.
- Leave accuracy and performance explicitly `unverified`: the CI topology is
  not represented as capable of loading the 753B checkpoint.

### Validation

- Focused CPU: 3/3; repository pre-commit hooks pass.
- Protected TPU at exact committed head: 3/3 in 9.09 seconds, 8/8 hosts clean.
- Restacked current head: 3/3 in 9.24 seconds, 8/8 hosts clean.

Evidence: `upstream_streamindex_test_20260828T001254Z`, evidence-manifest SHA-256
`f8f3bf38cd2dfed2735c82010f86af3b57e2cfe87b350726cdabe8b89690e5d3`.
Current-head evidence: `upstream_glm_dsa_pr3_local_bounds_20260828T015441Z`,
evidence-manifest SHA-256
`ce0f35f0d12d564132d0eb58fd9cfc5bf3c3b17182057bd79e7d2a8fac4cad31`.

### Compatibility, risk, and rollback

- This PR adds tests and Buildkite metadata only; current vLLM already owns
  model registration. Accuracy/performance entries intentionally remain
  `unverified` because the single-TPU CI queue cannot load the 753B model.
- Rollback is a two-commit revert with no runtime or state effect. PRs 1-2 can
  remain merged independently.

## Review notes common to the series

- Claim only the first upstream GLM-5.x DSA integration on TPU, not the first
  DSA kernel on TPU; the existing DeepSeek-V4 StreamIndex work predates this.
- TPU v4 results are direct evidence for the target deployment but do not
  establish v6e/v7x performance. Maintainers should choose whether equivalent
  pre-merge measurements on a supported-generation queue are required.
- The patches intentionally reuse current vLLM registration and model logic.
- Official upstream has not been mutated; all heads currently exist only on
  the user's private fork.
- Exact audit commands are:
  `git diff 5e2c7128...fd29657d`,
  `git diff fd29657d...dfb28231`, and
  `git diff dfb28231...8aae29ad`.
