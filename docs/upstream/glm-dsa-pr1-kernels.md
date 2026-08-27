# PR 1 — exact V3.2/GLM DSA kernels

**Prepared:** 2026-08-27 UTC. **Submission state:** private fork only; official upstream has not
been mutated. **Dependency:** none; this is the foundation for the later bridge and CI PRs.

## Pins and diff

- TPU Inference base: `5e2c7128bc74a75493f07930f3a749bcb272a3cb`.
- vLLM LKG: `d626108b1841888ec90aced33367149a6bbc7e4b`.
- Private branch/head: `pr/glm-dsa-kernels-v3` / `53b78b9519352b1bc64a115dee7190e35dad9d00`.
- Commits: `78d7ee30` (current FP8+FP32-scale exact scorer/top-k), `53b78b95` (paged sparse MLA
  consumer).
- The branch contains five changed/new source-test files and no generated data, checkpoints,
  traces, caches, or benchmark run directories.

The patch preserves the existing DeepSeek-v4 E8M0/approximate defaults. Opt-in static arguments
select current vLLM's 128 FP8 bytes + four raw FP32 scale bytes and deterministic `lax.top_k`.
The sparse consumer accepts the normal TPU token-major latent cache, gathers only selected rows,
masks the `-1` suffix, and performs one-row online-softmax MLA.

## Exact clean-head verification

Protected run: `/home/gianl/glm-run/upstream_streamindex_test_20260827T222601Z`.

- Real TPU v4-64 pod, worker 0 single-process bounds, four visible local chips.
- 15/15 tests passed in 53.42 seconds.
- Coverage includes numerical FP8+FP32-scale scoring, exact selected set, stable low-position ties,
  2047/2048/2049 sentinel boundaries, fragmented physical pages, K=2048 full GLM shape, fully
  masked rows, and scorer -> selected-page gather -> sparse-MLA equivalence.
- Pre/post authenticated fleet census: 8/8 unique hosts `CENSUS_OK`.
- Head was clean; worktree snapshot SHA is the empty SHA
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Profiler-free warmed wall distributions (five warmups, 20 synchronized samples):

| One-row v4 operation | p50 ms | p99 ms |
|---|---:|---:|
| exact top-k, N=262144, K=2048 | 2.0165 | 2.0643 |
| current-format scorer + exact top-k | 3.2470 | 3.2605 |
| sparse MLA, K=2048 | 0.1791 | 0.2082 |
| paged gather + sparse MLA from a 256K cache | 0.2347 | 0.2424 |
| complete scorer -> top-k -> gather -> sparse MLA chain | 3.2935 | 3.3160 |

Evidence SHA-256: provenance `22dbf0a7...42392`, pytest/metrics `103faddd...3d8f`, pre-census
`7dc15c23...f1ce8`, post-census `c9e08e8d...ba236`. The benchmark is an isolated kernel result,
not an end-to-end model latency or throughput claim.

The seven compact evidence/harness objects (16,074 bytes total) are mirrored byte-identically to
`gs://driftbench-dsv4-uc/results/upstream_glm_dsa_pr1_20260827T222601Z/`; bucket location was
reverified as the regional `US-CENTRAL2` bucket and all seven remote SHA-256 values match locally.

## Maintainer-facing PR draft

**Title:** `kernels: support exact V3.2/GLM DSA decode on TPU`

**Body:**

Current vLLM uses a V3.2 sparse-attention index record containing 128 FP8 E4M3 values followed by
a power-of-two scale stored as four raw FP32 bytes. TPU Inference's existing experimental
StreamIndex kernel only decodes the older one-byte E8M0 record and uses approximate top-k. This
patch adds opt-in current-record decoding, deterministic exact top-k, and a decode-only paged sparse
MLA consumer. Existing behavior remains the default.

The tests cover numerical dequantized scorer equivalence, stable ties and `-1` suffixes at
2047/2048/2049, fragmented paging, fully masked rows, full K=2048 GLM dimensions, and the complete
selected-position handoff. On TPU v4 the clean-head suite passed 15/15; the warmed one-row 256K
scorer-to-consumer chain measured p50 3.294 ms and p99 3.316 ms over 20 synchronized samples.

This PR intentionally contains no TorchAX/model wiring or registration/CI changes. A follow-up will
connect these kernels through the repository's existing vLLM-owned sparse-MLA metadata path.

## Risks and rollback

- This proves standalone kernel semantics/performance, not serving correctness. PR 2 must wire the
  returned indices without changing vLLM's IndexShare ownership.
- TPU v4 cannot use FP8 as the scorer MXU RHS, so current-format rows widen exactly to BF16 in VMEM;
  the cache remains FP8 and E4M3 values are exactly representable.
- Rollback is deletion of the new V3.2 module and leaving the two new static arguments at their
  original defaults; legacy callers are unchanged.
