# Goal — Upstream GLM-5.2 DSA support for TPU Inference

FULL ACCESS: work autonomously. Never push, open, or mutate upstream
`vllm-project/tpu-inference` before the user audits and explicitly approves the exact diff. Keep
below 4,000 characters. After start/compaction, read in full and inspect upstream/local state and
protected evidence.

## Primary objective

Be the first to contribute correct, mergeable GLM-5.2 DSA support to TPU Inference. Do not wait for
the greenfield engine's later Gates E-H. Use its protected v4 results as an oracle and evidence
source, not as code to import into the legacy execution path.

Prepare three small **stacked** PRs. They are separately reviewable, but not independent:

1. **GLM/DSA semantic kernels, exactness tests, and real TPU benchmarks.** This is the foundation
   and must be mergeable/useful alone. It must not depend on PR 2 or PR 3.
2. **Thin TorchAX/vLLM integration.** Base it on PR 1 and keep it limited to wiring the accepted
   kernels into the repository's preferred TorchAX-first path. It depends on PR 1.
3. **Model registration and CI enablement.** Base it on PRs 1-2. Add only the registration,
   configuration, fixtures, and CI coverage needed to enable GLM-5.2. It depends on both.

The review order is PR 1 -> PR 2 -> PR 3. While preparing locally, stacked branches are allowed;
before submission, rebase each onto the exact parent reviewers should see and make dependencies
explicit in the PR descriptions.

## Acceptance-first rules

- Re-audit upstream main, PRs/issues, CONTRIBUTING, CI, ownership, and maintainer feedback before
  finalizing each patch.
- Prefer existing repository abstractions and naming. Avoid a parallel framework, greenfield
  execution imports, speculative refactors, unrelated cleanup, generated dumps, model weights,
  profiler traces, or thousands of tracked artifacts.
- Match exact GLM-5.2 DSA semantics: selected set, deterministic tie/order behavior, padding and
  sentinel rules, IndexShare reuse, dtypes, shapes, and one-live-row decode behavior.
- Every behavior change needs focused tests. Performance claims require protected real-v4
  correctness, profiler-free wall measurements, and exact provenance; CPU/synthetic/HLO-only
  results are not performance proof.
- Keep benchmark evidence compact: scripts/configs, hashes, summarized results, and durable links;
  never commit checkpoints, raw XPlanes, generated environments, caches, or bulk run directories.
- Run prescribed pre-commit, tests, static checks, and relevant CI locally. Preserve compatibility.
- Never use Fable or Opus. At most one fresh independent Sol review of the final current diff.
- Commit and push coherent preparation batches only to the user's private fork/branches and mirror
  compact evidence only to `gs://driftbench-dsv4-uc` in exact `US-CENTRAL2`; never EU. Keep sync
  outside TPU timing and serialize with cron rsync.

## Required deliverables before user audit

For each PR provide: exact base/head pins, a minimal diff, dependency statement, test matrix and
results, real-TPU evidence where applicable, compatibility/risk notes, rollback behavior, and a
ready-to-paste title/body. Prove no generated/bulk files are tracked and no upstream mutation
occurred. Present PR 1 first for personal audit; do not let unfinished PRs 2-3 delay it.

## Definition of done

All three patches are locally complete and reviewable in dependency order; PR 1 independently
passes semantics, exactness, and protected TPU evidence; PRs 2-3 pass their integration and CI
contracts; the user audits each exact diff before upstream submission. After explicit approval,
submit in order, respond to maintainer feedback, rebase narrowly, and continue until all accepted
PRs are merged or the maintainers explicitly request a different decomposition.
