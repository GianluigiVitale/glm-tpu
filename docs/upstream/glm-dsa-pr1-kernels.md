# PR 1 — exact V3.2/GLM DSA kernels

**Prepared:** 2026-08-27 UTC. **Submission state:** private fork only; official upstream has not
been mutated. **Dependency:** none; this is the foundation for the later bridge and CI PRs.

## Pins and diff

- TPU Inference base: `5e2c7128bc74a75493f07930f3a749bcb272a3cb`.
- vLLM LKG: `d626108b1841888ec90aced33367149a6bbc7e4b`.
- Private branch/head: `pr/glm-dsa-kernels-v3` / `d3ccafdfe5157dda4c26b779dc50eb505edfa96b`.
- Commits: `78d7ee30` (current FP8+FP32-scale exact scorer/top-k), `53b78b95` (paged sparse MLA
  consumer), and `d3ccafdf` (exact V3.2 index-cache pack/insertion plus legacy-v4 lowering fix).
- The branch contains seven changed/new source-test files and no generated data, checkpoints,
  traces, caches, or benchmark run directories.

The patch preserves the existing DeepSeek-v4 E8M0/approximate defaults. Opt-in static arguments
select current vLLM's 128 FP8 bytes + four raw FP32 scale bytes and deterministic `lax.top_k`.
The V3.2 writer uses vLLM's UE8M0 quantization rule, packs exact raw bytes, and performs a flat
physical-slot scatter for one live decode row. The sparse consumer accepts the normal TPU
token-major latent cache, gathers only selected rows, masks the `-1` suffix, and performs one-row
online-softmax MLA.

## Exact clean-head verification

Primary protected run: `/home/gianl/glm-run/upstream_streamindex_test_20260827T224406Z`.

- Real TPU v4-64 pod, worker 0 single-process bounds, four visible local chips.
- 36/36 checks passed in 74.90 seconds: 29 repository tests, six bounded real-v4 benchmarks and
  one exact StableHLO inspection.
- Coverage includes numerical FP8+FP32-scale scoring, exact selected set, stable low-position ties,
  2047/2048/2049 sentinel boundaries, exact cache bytes, padded/unpadded and fragmented physical
  insertion, all legacy E8M0 tests, K=2048 full GLM shape, fully masked rows, and writer -> scorer
  -> selected-page gather -> sparse-MLA equivalence.
- Pre/post authenticated fleet census: 8/8 unique hosts `CENSUS_OK`.
- Head was clean; worktree snapshot SHA is the empty SHA
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Profiler-free warmed wall distributions (five warmups, 20 synchronized samples):

| One-row v4 operation | p50 ms | p99 ms |
|---|---:|---:|
| exact top-k, N=262144, K=2048 | 2.0103 | 2.0292 |
| current-format scorer + exact top-k | 3.2320 | 3.2413 |
| V3.2 quantize/pack/scatter into a 256K cache | 0.3610 | 0.3964 |
| sparse MLA, K=2048 | 0.1753 | 0.2068 |
| paged gather + sparse MLA from a 256K cache | 0.2159 | 0.2419 |
| scorer -> top-k -> gather -> sparse MLA chain | 3.2760 | 3.2885 |
| full writer -> scorer -> top-k -> gather -> sparse MLA step | 3.5575 | 3.6684 |

The full-step distribution is a separate exact-head run at
`/home/gianl/glm-run/upstream_streamindex_test_20260827T224724Z`; both runs use five warmups and
20 synchronized samples. StableHLO SHA is `65e0c5a1...454e9` and contains one flat cache scatter,
no loop or page-wise read/modify/write. Primary evidence SHA-256: provenance `ff1fba5e...43689`,
pytest/metrics `2c5d2453...c1cf`, pre-census `43641494...c0f`, post-census
`18a56e1c...1d7`. The benchmark is isolated kernel evidence, not end-to-end serving latency or a
throughput claim.

The compact evidence/harness bundle is mirrored byte-identically to
`gs://driftbench-dsv4-uc/results/upstream_glm_dsa_pr1_final_20260827T224724Z/`; bucket location was
reverified as regional `US-CENTRAL2` and every remote SHA-256 matches locally.

## Maintainer-facing PR draft

**Title:** `kernels: support exact V3.2/GLM DSA decode on TPU`

**Body:**

Current vLLM uses a V3.2 sparse-attention index record containing 128 FP8 E4M3 values followed by
a power-of-two scale stored as four raw FP32 bytes. TPU Inference's existing experimental
StreamIndex kernel only decodes the older one-byte E8M0 record and uses approximate top-k. This
patch adds exact current-record pack/insertion and decoding, deterministic exact top-k, and a
decode-only paged sparse MLA consumer. Existing behavior remains the default.

The tests cover exact cache bytes/scatter, numerical dequantized scorer equivalence, stable ties and
`-1` suffixes at 2047/2048/2049, fragmented paging, fully masked rows, full K=2048 GLM dimensions,
legacy compatibility, and the complete selected-position handoff. On TPU v4 the clean-head suite
passed 36/36; the warmed full one-row 256K writer-to-consumer step measured p50 3.557 ms and p99
3.668 ms over 20 synchronized samples.

This PR intentionally contains no TorchAX/model wiring or registration/CI changes. A follow-up will
connect these kernels through the repository's existing vLLM-owned sparse-MLA metadata path.

## Risks and rollback

- This proves standalone kernel semantics/performance, not serving correctness. PR 2 must wire the
  returned indices without changing vLLM's IndexShare ownership.
- TPU v4 cannot use FP8 as the scorer MXU RHS, so both cache formats widen exactly to BF16 in VMEM;
  caches remain FP8 and E4M3 values are exactly representable. Legacy E8M0 scale bytes reconstruct
  BF16 exponent bits directly because current Mosaic cannot lower E8M0 conversion on v4.
- Rollback is deletion of the new V3.2 module and leaving the two new static arguments at their
  original defaults; legacy callers are unchanged.
