# Owner audit — PR 1 exact GLM/V3.2 DSA kernel foundation

This document covers PR 1 only. Official upstream is not authorized and has not
been mutated.

## Immutable review range

- Base: `e08b64c14208cb5efc34cc3b41eeaa3402346911`
- Head: `650b5fccb890b5a872871af489b50fc4c584e8ad`
- Branch: private `pr/glm-dsa-kernels-v3`
- Dependency: none. PR 1 contains no TorchAX bridge, model registration, or CI
  enablement from PRs 2-3.
- Size: 8 files, +1,250/-10; no tracked file exceeds 1 MiB.

Inspect the exact patch:

```bash
git -C /home/gianl/tpu-inference-glm-baseline \
  diff --find-renames e08b64c14208cb5efc34cc3b41eeaa3402346911..650b5fccb890b5a872871af489b50fc4c584e8ad
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

- Exact-head protected kernel/HLO suite: 30/30 in 60.16 seconds; 8/8 hosts
  clean; local/remote tag `upstream_glm_dsa_pr1_rebase_20260828T085145Z`;
  evidence-manifest SHA-256
  `1e20d3c5342c25537dfc2e98ee51e22aea0ca34ba66bef5e2778be4868bfcf3c`.
- Same clean head, real-v4 benchmark: scorer 3.216 ms p50, cache insert
  0.374 ms, selected gather + MLA 0.244 ms, sparse MLA 0.182 ms, composed chain
  3.292 ms; local/remote tag
  `upstream_glm_dsa_pr1_benchmark_rebase_20260828T085526Z`; evidence-manifest
  SHA-256 `54ec928ba42d5426c5d9096ea0959bbf2fede52c45b43eb232e5b01f95119575`.
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
