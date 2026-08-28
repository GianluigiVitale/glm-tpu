# Owner audit — PR 1 exact GLM/V3.2 DSA kernel foundation

This document covers PR 1 only. Official upstream is not authorized and has not
been mutated.

## Immutable review range

- Base: `5e2c7128bc74a75493f07930f3a749bcb272a3cb`
- Head: `fd29657d336cee859c17d4568f8d38d276ca9707`
- Branch: private `pr/glm-dsa-kernels-v3`
- Dependency: none. PR 1 contains no TorchAX bridge, model registration, or CI
  enablement from PRs 2-3.
- Size: 8 files, +1,250/-10; no tracked file exceeds 1 MiB.

Inspect the exact patch:

```bash
git -C /home/gianl/tpu-inference-glm-baseline \
  diff --find-renames 5e2c7128bc74a75493f07930f3a749bcb272a3cb..fd29657d336cee859c17d4568f8d38d276ca9707
```

## File-by-file checklist

| File | Change | Owner question | Direct proof |
|---|---|---|---|
| `.../deepseek_v4/indexer/streamindex_topk.py` | Adds explicit FP32-scale record decoding and exact `lax.top_k`; legacy E8M0/approximate defaults remain unchanged | Is widening E4M3 to BF16 on v4 acceptable, and should exact selection remain opt-in? | Legacy tests plus numerical FP32-scale and 2047/2048/2049 tie tests |
| `.../deepseek_v32/indexer_cache.py` | Quantizes UE8M0-style power-of-two scales, stores them as raw FP32 bytes, and flat-scatters physical slots | Does the repository want this under `experimental/deepseek_v32`? | Exact byte, fragmented slot, padding, unpadded-layout, and StableHLO tests |
| `.../deepseek_v32/sparse_mla.py` | Gathers selected paged rows and computes decode-only online-softmax MLA | Is a decode-only selected consumer the right minimal primitive before bridge wiring? | Real-v4 reference, fully masked, fragmented paging, and boundary composition tests |
| `tests/kernels/deepseek_v4/test_streamindex_topk.py` | Extends the existing kernel suite without changing old fixtures | Do exact ties select lower positions and append only `-1` suffixes? | Exact array equality at 2047/2048/2049 |
| `tests/kernels/deepseek_v32/test_indexer_cache.py` | Covers writer bytes/scatter and writer-to-scorer composition | Are both 132-byte vLLM and 256-byte TPU-padded records required? | Both layouts are exercised directly |
| `tests/kernels/deepseek_v32/test_sparse_mla.py` | Covers the selected-position handoff to real sparse MLA | Is top-k proven as consumed rather than merely produced? | Scorer -> paged gather -> sparse MLA equality |
| `scripts/benchmarking/kernels/benchmark_glm_dsa.py` | Adds profiler-free synchronized v4 microbenchmarks | Are isolated-kernel claims scoped narrowly enough? | Five warmups, 20 individually blocked samples, raw distributions and checksums |
| `.../deepseek_v32/__init__.py` | Package marker only | None | Header-only file |

The exact CODEOWNER surfaces are the DeepSeek-v4 experimental owners for the
modified scorer, the general kernel owners for new V3.2 files, and kernel-test
owners. `scripts/` has no required CODEOWNER.

## Evidence to accept or reject

- Exact-head protected kernel/HLO suite: 30/30 in 61.19 seconds; 8/8 hosts
  clean; local tag `upstream_streamindex_test_20260828T011411Z`; remote prefix
  `upstream_glm_dsa_pr1_correctness_20260828T011411Z`; evidence-manifest SHA-256
  `1e08dad8c1ee87487df08a380a17b65cebac9134a4e82157240e1e71b8138704`.
- Same clean head, real-v4 benchmark: scorer 3.208 ms p50, cache insert
  0.353 ms, selected gather + MLA 0.222 ms, sparse MLA 0.159 ms, composed chain
  3.280 ms; local tag `upstream_glm_dsa_benchmark_20260828T011531Z`; remote
  prefix `upstream_glm_dsa_pr1_benchmark_20260828T011531Z`; evidence-manifest
  SHA-256 `20cbf34d8576c8905d337f932584f3f9248865941bcbae63b9a68bf35bb8e285`.
- These are kernel measurements, not full-model correctness, HBM, quality,
  serving latency, or non-v4 performance claims.

## Compatibility and rollback decision

- Existing callers retain `scale_storage="e8m0"` and `exact_topk=False`.
- New APIs are isolated under `kernels/experimental/deepseek_v32`.
- Invalid scale format, record width, cache shape/dtype, or writer geometry
  fails loudly. `-1` selected-position padding is consumed only as a suffix.
- Rollback is the five PR1 commits only; no checkpoint/state migration exists.
- All five commits carry the required DCO trailer.

Owner decision: approve PR 1 as-is, request a named correction, or request a
different kernel/package decomposition. PRs 2-3 should not influence this
foundation decision.
