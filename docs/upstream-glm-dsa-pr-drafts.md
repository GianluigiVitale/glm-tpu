# Ready-to-paste upstream GLM-5.2 DSA PR series

Private review material only. Do not open these PRs until the owner audits the
exact diffs and explicitly approves submission. The PRs are separately
reviewable but stacked in order: PR 1 -> PR 2 -> PR 3.

## PR 1 title

`kernels: add exact V3.2 sparse-attention primitives`

## PR 1 body

### Description

Current vLLM DeepSeek-V3.2/GLM sparse attention stores each index-cache record
as 128 FP8 E4M3 bytes followed by a power-of-two scale encoded in four raw FP32
bytes. TPU Inference's experimental StreamIndex path currently handles the
older one-byte E8M0 scale record and uses approximate selection.

This patch adds the isolated primitives needed for the current contract:

- exact FP8+FP32-scale record decoding and deterministic top-k selection;
- UE8M0-compatible key quantization, record packing, and physical paged-cache
  insertion;
- selected-position paged MLA gather and one-row sparse attention; and
- a bounded, synchronized real-TPU benchmark for the complete scorer-to-
  consumer chain.

Existing DeepSeek-v4 callers keep the E8M0/approximate defaults. New V3.2 APIs
remain under `kernels/experimental`; this PR contains no TorchAX bridge, model
registration, scheduler change, or CI enablement. It is the dependency-free
foundation of a three-PR series and is independently useful/testable.

This contributes toward #1699 without closing it.

### Tests

Exact review range:

```text
e08b64c14208cb5efc34cc3b41eeaa3402346911..650b5fccb890b5a872871af489b50fc4c584e8ad
```

Real TPU v4 correctness/HLO run:

```bash
pytest -q \
  tests/kernels/deepseek_v4/test_streamindex_topk.py \
  tests/kernels/deepseek_v32/test_indexer_cache.py \
  tests/kernels/deepseek_v32/test_sparse_mla.py
```

Result: 30/30 passed in 60.16 seconds at the exact head above. Coverage includes
exact cache bytes, fragmented physical slots, numerical scorer equivalence,
low-position tie order, `-1` suffixes at 2047/2048/2049, fully masked rows,
legacy compatibility, K=2048, and scorer -> selected gather -> sparse MLA.

Profiler-free benchmark:

```bash
PYTHONPATH=. python scripts/benchmarking/kernels/benchmark_glm_dsa.py \
  --mode all --warmup 5 --samples 20 --seq-len 262144 --topk 2048
```

Four local TPU v4 devices, five warmups and 20 individually synchronized
samples at the same exact head:

| Operation | p50 ms | p99 ms |
|---|---:|---:|
| exact top-k | 2.009 | 2.041 |
| StreamIndex scorer + top-k | 3.216 | 3.248 |
| index-cache insert | 0.374 | 0.434 |
| sparse MLA | 0.182 | 0.201 |
| selected gather + MLA | 0.244 | 0.264 |
| scorer -> gather -> MLA chain | 3.292 | 3.318 |

Both protected runs used a clean worktree, pinned vLLM
`d626108b1841888ec90aced33367149a6bbc7e4b`, and authenticated 8/8 host cleanup.
These are isolated-kernel measurements, not full-model quality, HBM, serving
latency, other-generation performance, or throughput claims.

### Compatibility, limitations, and rollback

- Invalid record width/format, cache geometry, shape, or dtype fails loudly.
- The V3.2 quantizer is intentionally specialized: the common
  `quantize_tensor` helper returns an arbitrary `absmax/448` FP32 scale, while
  this cache contract requires that value rounded up to a power of two and
  serialized as four raw FP32 bytes. Exact-byte tests lock down the difference.
- TPU v4 widens FP8 values to BF16 for scorer arithmetic; cache storage remains
  FP8 and current E4M3 values are exactly representable.
- The production bridge and prefill integration are intentionally deferred to
  PR 2 so this kernel review remains bounded.
- Rollback is the five commits in this range; no checkpoint/state migration is
  introduced.

### Checklist

- [x] I have performed a self-review of my code.
- [x] I have added comments for the non-obvious record and sentinel contracts.
- [x] I have added focused tests and a reproducible TPU benchmark.

---

## PR 2 title

`layers: bridge V3.2 sparse attention to TPU`

## PR 2 body

Before opening, replace `<PR1_URL>` with the submitted PR 1 URL.

### Description

Depends on PR 1: `<PR1_URL>`.

This patch connects current vLLM DeepSeek-V3.2/`GlmMoeDsaForCausalLM`
semantics to the PR 1 TPU kernels through the existing TorchAX/vLLM layer and
out-of-tree custom-op abstractions.

- Register a TPU `SparseAttnIndexer` implementation.
- Map logical tokens to the physical paged index cache and insert exact records.
- Reuse the model-owned shared top-k buffer across IndexShare layers.
- Consume selected positions in sparse MLA for causal prefill and decode.
- Reconstruct complete TP query rows before consuming global top-k/cache
  metadata.
- Derive prefill/decode from request metadata, sentinel-pad TPU-only rows, and
  execute DSA/model work for only the one live single-stream decode row.

Current vLLM remains authoritative for model registration, weight loading,
request metadata, IndexShare scheduling, and buffer lifetime. This patch adds
no model fork, constructor monkeypatch, DSA-disable path, host dispatch, or CPU
fallback.

Unsupported configurations fail closed: more than one sequence, DP, DCP/PCP,
continue-decode disabled, non-FP8/UE8M0/block-128 index caches, quantized or
transposed main MLA caches, or sparse MLA without the model-owned top-k buffer.

### Tests

Exact review range:

```text
650b5fccb890b5a872871af489b50fc4c584e8ad..d837832ab41f947ee9ff759e65ea8417ba1bd5c9
```

The exact-head protected TPU v4 suite passed 57/57 in 99.99 seconds. It covers
the PR 1 kernels plus indexer construction/errors, cache insertion, causal
prefill, one-row decode, shared-buffer reuse, missing-buffer rejection,
metadata phase selection, sparse backend outputs, and TP2 complete-row
reconstruction. Focused forced-two-device CPU tests and all-file pre-commit
also pass.

Protected provenance pins vLLM
`d626108b1841888ec90aced33367149a6bbc7e4b`, a clean worktree, four local TPU v4
devices, and authenticated 8/8 pre/post cleanup. This is bridge correctness
evidence, not full-checkpoint loading, quality, HBM, or serving latency.

### Compatibility, limitations, and rollback

- Dense MLA callers without a DSA/shared-buffer contract retain their existing
  path; a DSA caller cannot silently fall back to dense MLA.
- The explicit single-sequence/DP/DCP/cache restrictions prevent unproved
  layouts from appearing supported.
- Rollback is the four commits in this range. PR 1 remains independently
  useful; no persistent format migration exists.

### Checklist

- [x] I have performed a self-review of my code.
- [x] I have preserved vLLM-owned model/metadata/IndexShare semantics.
- [x] I have added focused unit, TP2, fail-closed, and real-TPU tests.

---

## PR 3 title

`models: add GLM-5.2 DSA contract CI`

## PR 3 body

Before opening, replace `<PR1_URL>` and `<PR2_URL>` with the submitted parent
PR URLs.

### Description

Depends on PR 1 `<PR1_URL>` and PR 2 `<PR2_URL>`.

Current pinned vLLM already registers `GlmMoeDsaForCausalLM`; adding a second
TPU-side registration would fork model semantics. This final stacked patch
instead verifies and enables the cross-repository contract:

- resolve the public GLM-5.2 architecture through vLLM's registry;
- prove all decoder layers receive the same shared top-k buffer;
- prove the public `full, full, full, shared, shared, shared, full` IndexShare
  prefix and producer reuse;
- add a model-specific Buildkite unit step on the single-TPU queue; and
- run the PR 1 kernel and PR 2 bridge/sparse-backend regressions in that step.

Accuracy and performance remain explicitly `unverified`; this patch does not
claim that the 753B checkpoint fits the CI queue or that v4 measurements prove
v6e performance. This contributes toward #1699 without closing it.

### Tests

Exact review range:

```text
d837832ab41f947ee9ff759e65ea8417ba1bd5c9..101ec506d76a3ecb0b688315e432ae7e8d0ab37a
```

```bash
MODEL_IMPL_TYPE=vllm python3 -m pytest -q \
  tests/models/vllm/test_glm_moe_dsa.py
```

Result: 3/3 passed in 9.43 seconds on the exact head, with a clean worktree and
authenticated 8/8 host cleanup. Focused CPU tests and all-file pre-commit pass.

The YAML parses as six unique steps; every dependency and referenced test path
resolves, every result-recording step has complete metadata, and
`CI_TARGET=zai-org/GLM-5.2-FP8` is globally unique. The repository metadata
validator currently has an unrelated queue-parser defect reproduced on
unchanged GLM-5 and DeepSeek-V3.2 YAMLs; that fix is intentionally not mixed
into this model PR.

### Compatibility, limitations, and rollback

- Production code is unchanged; current vLLM remains the registration and
  IndexShare authority.
- The default CI queue proves unit behavior only. Full-checkpoint accuracy,
  performance, HBM, and cross-generation validation remain unclaimed.
- Rollback is the two commits in this range and removes only the test/YAML.
  PRs 1-2 remain valid independently.

### Checklist

- [x] I have performed a self-review of my code.
- [x] I have reused current vLLM registration instead of duplicating it.
- [x] I have added focused model-contract tests and CI coverage.

---

## Submission invariants

- Open and review in order: PR 1, then PR 2, then PR 3.
- Re-check official `main` immediately before each submission. If its base has
  moved, rebase narrowly and repeat exact-head validation before opening.
- Replace dependency placeholders only after parent PR URLs exist.
- Claim GLM-5.2/V3.2 DSA integration, not the first TPU DSA kernel; the merged
  DeepSeek-v4 indexer predates this series.
- Never submit a different diff than the owner-audited range.
