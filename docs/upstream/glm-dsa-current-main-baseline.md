# GLM-5.2 DSA upstream baseline

**Captured:** 2026-08-27 UTC. **Submission state:** private preparation only; no upstream push,
PR, issue, or comment.

## Exact inputs

- TPU Inference: official `main` `5e2c7128bc74a75493f07930f3a749bcb272a3cb`.
- Its `.buildkite/vllm_lkg.version`: vLLM `d626108b1841888ec90aced33367149a6bbc7e4b`.
- Research dossier: `docs/deep-research-report.md`, 2,206 lines / 91,647 bytes, SHA-256
  `fbbbfb3a55c2d66aa4ff55befe71c45ec470791735d2a25a1b72040ce71ad8f1`; read in full.
- Model contract: `GlmMoeDsaForCausalLM`, `glm_moe_dsa`, 32 index heads x 128, exact top-k
  2,048, maximum context 1,048,576.

## Untouched-current-main result

The detached baseline worktree is `/home/gianl/tpu-inference-glm-baseline` at the exact official
pin. A serialized, weight-free probe on worker 0 of the existing `db-v4-64-od` pod loaded these
two source pins through `/home/gianl/vllm-env/bin/python`. Runtime detection reported
`TpuPlatform tpu XLA`; invoking the untouched `SparseAttnIndexer.forward_native` produced:

```text
NotImplementedError: SparseAttnIndexer native forward is only implemented for CUDA, ROCm and XPU platforms.
```

The run directory is
`/home/gianl/glm-run/upstream_glm_baseline_env_20260827T214949Z`. Pre/post censuses each
authenticate all eight unique hosts as `CENSUS_OK`. Evidence hashes are:

- pre-census: `6e3d3bfe5880063a27505f04c97ae25de7515064d8c40b2b106a6a640597e937`
- environment/blocker: `c8d467f5df0f55f4b151ba6dfabf09f7ebb642eb4a6e0032dec42829557373d8`
- post-census: `c0da22647a35691e1838f1cd4b3a3238ad2ffce1d26da2b5eeae315455000b5f`

This proves the current TPU dispatch blocker. It is not a performance or end-to-end correctness
claim.

## Scope correction and PR stack

Current vLLM already owns GLM registration/model semantics, FP8 indexer loading, IndexShare
scheduling, and request metadata. TPU Inference's generic MLA wrapper calls the indexer but drops
its return before dense MLA. Therefore a score-only or top-k-only patch is insufficient above
2,048 tokens: selected positions must reach a real sparse MLA consumer.

1. PR1: isolated TPU indexer/top-k and sparse-MLA kernels, exactness tests, boundary cases, and
   real-v4 performance evidence. It must be useful alone.
2. PR2: thin TorchAX/vLLM dispatch plus selected-position consumer wiring, stacked on PR1.
3. PR3: focused IndexShare/model regressions and CI enablement, stacked on PR2; add registration
   only if the submission-day base actually lacks it.

Historical branches are comparison inputs, not submission bases: `e652c163` measures scores only;
the recovered G6 pair ending at `6908b827` has exact BF16 paged kernels and CPU-compatible tests
but does not yet implement the current packed FP8+FP32-scale cache contract or current bridge.
