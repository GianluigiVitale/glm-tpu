# Draft upstream GLM-5.2 DSA PR series

Private review material only. Do not open these PRs until the user has audited
the exact diffs. The series is stacked and must be reviewed/merged in order.

## PR 1 — `kernels: add exact V3.2/GLM sparse-attention primitives`

Base: `vllm-project/tpu-inference:main` at `5e2c7128`  
Head: private `pr/glm-dsa-kernels-v3` at `bb002856`

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
- Exact final signed head: 29/29 focused protected tests, 8/8 hosts clean.
- Exact selected set/order at 2,047/2,048/2,049 boundaries and production
  `n=262144, k=2048`.
- Warming: 5 iterations; samples: 20 on TPU v4.
- Median device times: StreamIndex 3.232 ms, cache insert 0.361 ms, selected
  gather 0.216 ms, sparse MLA 0.175 ms, composed DSA chain 3.276 ms.
- Exact StableHLO captured; no end-to-end throughput claim is made.

Evidence: `upstream_streamindex_test_20260827T224406Z`, manifest-list SHA-256
`dd461d99feae8b582457954473577e1ee46ca059cf1d96314ee4b4f486daf170`.
Final-head evidence: `upstream_streamindex_test_20260828T004109Z`,
manifest-list SHA-256
`699505e28e5b7684b046d68972cf91872f5a00065985a0827d329ca976bb1251`.

## PR 2 — `layers: bridge V3.2/GLM sparse attention to TPU`

Base: PR 1 `bb002856`  
Head: private `pr/glm-dsa-bridge-v3` at `b6d92092`

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

### Validation

- Focused CPU regressions: 26/26; repository pre-commit hooks pass.
- Final protected TPU bridge suite: 22/22 in 44.04 seconds, 8/8 hosts clean.
- Additional protected exact causal-prefill and selected-MLA tests pass.
- Exact final signed head: 55/55 protected tests in 97.15 seconds, including a
  two-device TP prefill with the same eight-row shape as decode; 8/8 hosts
  clean.
- No full-checkpoint latency, accuracy, or serving claim is made.

Primary evidence: `upstream_streamindex_test_20260827T235340Z`, manifest-list
SHA-256
`6d3a6b445568f8b4a5cd25c97feb50316d97acc0fc48e1259290b14591dd70fe`.
Final-head evidence: `upstream_streamindex_test_20260828T004235Z`,
manifest-list SHA-256
`61a78d59e11eeb17a7a7c8b9bb6c2ee9a3e28b77c407b56b74c2f666be62dffe`.

## PR 3 — `models: add GLM-5.2 DSA contract CI`

Base: PR 2 `b6d92092`  
Head: private `pr/glm-dsa-model-ci-v3` at `f25c9a91`

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
- Exact final signed head: 3/3 in 9.86 seconds, 8/8 hosts clean.

Evidence: `upstream_streamindex_test_20260828T001254Z`, manifest-list SHA-256
`f8f3bf38cd2dfed2735c82010f86af3b57e2cfe87b350726cdabe8b89690e5d3`.
Final-head evidence: `upstream_streamindex_test_20260828T004433Z`,
manifest-list SHA-256
`cbe5fbb9129b425add5136b3c100ed9d19320d8cc49db48e98b5d0aa499c4149`.

## Review notes common to the series

- Claim only the first upstream GLM-5.x DSA integration on TPU, not the first
  DSA kernel on TPU; the existing DeepSeek-V4 StreamIndex work predates this.
- TPU v4 results are direct evidence for the target deployment but do not
  establish v6e/v7x performance. Maintainers should choose whether equivalent
  pre-merge measurements on a supported-generation queue are required.
- The patches intentionally reuse current vLLM registration and model logic.
- Official upstream has not been mutated; all heads currently exist only on
  the user's private fork.
