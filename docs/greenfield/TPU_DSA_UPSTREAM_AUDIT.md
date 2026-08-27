# TPU GLM DSA upstream audit

Status on 2026-08-27: **prepared for owner review; not submitted upstream**.

## Exact state

- Real upstream `vllm-project/tpu-inference` has not been written to and no PR exists.
- Audit worktree: `/home/gianl/tpu-inference-dsa-pr`.
- Branch: `feature/tpu-dsa-indexer`.
- Current upstream base: `c5c8a055`.
- Local audit commit: `e652c163` (signed, clean, not pushed).
- The owner's fork retains earlier safety checkpoint `62b1670a`; it must be force-updated only
  after owner approval because the local branch was rebased.
- Scope: 5 new files, 290 lines. No model, loader, serving, environment, or existing code changed.
- DeepSeek V4 is a process lesson only. The diff contains no DeepSeek code, import, cache format,
  or execution path; the scorer is adapted from the protected GLM greenfield kernel.

## Why this slice is reviewable

`CONTRIBUTING.md` prefers Torchax model enablement first and requires unit/CI tests. This PR does
not propose the independent greenfield engine for upstream. It contributes one reusable Pallas
primitive that a later Torchax `SparseAttnIndexer` override can call.

Relevant maintainer precedent:

- GLM PR [#2324](https://github.com/vllm-project/tpu-inference/pull/2324) remains open, disables the
  unported DSA forward, and was explicitly declined by one reviewer at 33 files / 4,400 additions.
- Tuning PR [#3200](https://github.com/vllm-project/tpu-inference/pull/3200) was deferred because the
  change was considered too large for a path being replaced.
- Kernel-only PR [#3295](https://github.com/vllm-project/tpu-inference/pull/3295) merged without its
  caller, establishing that a tested standalone kernel is acceptable.
- Current changed-kernel CI automatically schedules the kernel suite on TPU v6e and v7x. The diff
  includes a real non-interpreter TPU execution test, not only CPU interpretation.

## Evidence

- Repository hooks: all pass (`addlicense`, `isort`, `yapf`, `ruff`, missing-init, filenames).
- Isolated JAX 0.11 CPU suite: `5 passed, 1 skipped`; the skip is the real-TPU execution test.
- Bounded live TPU-v4 audit on the exact final source SHA
  `1b20c1d3f867f41ff7dc37b2c6060505453f509f7c2de75d9b141b10335e60c0`:
  one `[1,32,128] x [65536,128]` score row, exact top-2048 positions, score max/mean error
  `2.861e-6 / 2.417e-7`, named Pallas call, no `[32,65536]` HBM overlay or batch-32 shape.
  Compile was 0.390 s; diagnostic p50/p99 was 0.291/0.318 ms. Pre/post fleet census was 8/8 clean.
- Archived result:
  `gs://driftbench-dsv4-uc/results/glm_dsa_upstream_audit_20260827T180500Z/`.
- Earlier protected production proof remains DB443 (200 warmups / 1,000 samples, full HLO/HBM/DB).

## Likely rejection points and mitigation

1. **“Not integrated.”** State this is scorer-only; #3295 is the kernel-only precedent. Promise
   exact selection and Torchax bridge as separate PRs, not hidden scope.
2. **“Only tested on v4.”** Do not claim v6e/v7x performance. The PR's real hardware test compiles
   and executes in both current CI pipelines; wait for those results before requesting review.
3. **“This is not full DSA.”** Title it “GLM DSA indexer scorer,” never full model/DSA support.
4. **“Why fixed one row/128?”** One row prevents decode batch-32 dead rows; 128 is GLM's checkpoint
   head dimension and the validated TPU tile. Prefill/multi-row support is explicitly out of scope.
5. **“Private evidence.”** Put reproducible commands and numeric results in the PR body; the bucket
   is integrity backup, not required reviewer access.
6. **“Conflicts with #2324.”** There is no file overlap. This kernel replaces its dense-fallback
   limitation later; it does not absorb that PR's loader/MoE/multihost changes.
7. **“Current-generation compatibility.”** Local CPU used repository JAX 0.11; live v4 used 0.10.1.
   The upstream v6e/v7x hardware test is the authority for JAX 0.11 TPU lowering.

## Owner audit checklist

- Inspect `git diff c5c8a055..e652c163` in the audit worktree.
- Confirm the mathematical order: FP32 dot, `128**-0.5`, ReLU, signed FP32 head sum.
- Confirm BF16 cache keys, one live row, padding before bounds-check disabling, and sliced output.
- Confirm all five files and DCO trailer; reject any unrelated change.
- Approve the title/body separately. Only then update the fork and open the upstream PR.
