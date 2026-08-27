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

## Maintainer acceptance audit

`CONTRIBUTING.md` prefers Torchax model enablement first and requires unit/CI tests. This PR does
not propose the independent greenfield engine for upstream. It contributes one reusable Pallas
primitive that a later Torchax `SparseAttnIndexer` override can call.

Official repository rules and routing:

- The PR template asks for a short delta, rationale, implementation details, shortcomings,
  reproducible tests, self-review, comments, and documentation consideration.
- The `ready` label is a merge-blocking check. Do not apply/request it until the owner audit is
  complete and the branch is current.
- DCO sign-off is enforced. The audit commit is signed off.
- Kernel and kernel-test changes trigger the full kernel suite on both v6e and v7x. Those jobs are
  `soft_fail`, so their actual results must be inspected and reported rather than treating the
  overall green check as sufficient.
- Kernel CODEOWNERS are `@kyuyeunk`, `@bythew3i`, `@a1yssan13`, and `@jrplatin`.

Observed maintainer behavior:

- GLM PR [#2324](https://github.com/vllm-project/tpu-inference/pull/2324) remains open, disables the
  unported DSA forward, and was explicitly declined by one reviewer at 33 files / 4,400 additions.
- Tuning PR [#3200](https://github.com/vllm-project/tpu-inference/pull/3200) was deferred because the
  change was considered too large and the path was being replaced. Reviewability and architectural
  currency matter more than raw line count.
- External-contributor PR [#1729](https://github.com/vllm-project/tpu-inference/pull/1729) merged as
  a contained five-file model slice after focused review, MMLU evidence, a rebase, and an explicit
  decision to leave MLA to a follow-up.
- External-contributor kernel PRs [#3349](https://github.com/vllm-project/tpu-inference/pull/3349)
  and [#3411](https://github.com/vllm-project/tpu-inference/pull/3411) were two-file changes and each
  received kernel-owner approval. The first was later reverted after a nightly regression, then
  re-landed, underscoring the need for production-shape coverage rather than only a toy test.
- Experimental kernel PR [#3040](https://github.com/vllm-project/tpu-inference/pull/3040) merged
  before final model integration. Reviewers asked for E2E comparison, but accepted a tested,
  encapsulated kernel as a starting point with follow-up optimization.
- On [#3250](https://github.com/vllm-project/tpu-inference/pull/3250), a kernel owner explicitly
  approved because the code was isolated under `experimental` and could not affect other workloads.

The merge-maximizing shape is therefore exactly one isolated experimental primitive, no existing
runtime behavior change, executable hardware coverage, production-shape correctness evidence, and
an honest scorer-only title. Selection and Torchax integration should be separate PRs.

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

1. **“Not integrated.”** State this is scorer-only; #3040 is the stronger kernel-before-integration
   precedent. Promise exact selection and Torchax bridge as separate PRs, not hidden scope.
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
8. **“Why merge before full model support?”** It is default-inert experimental code with no caller,
   so it cannot regress existing workloads. It also directly removes one bounded blocker from the
   already-open GLM effort without importing that PR's loader or execution architecture.

## Submission sequence after owner approval

1. Rebase onto the then-current `upstream/main`; rerun hooks and the focused CPU suite.
2. Force-update only the owner's fork branch, open as a draft, and link #1699/#2324.
3. Let both changed-kernel jobs execute on v6e and v7x; inspect the individual jobs despite
   `soft_fail`, and add exact results to the PR body.
4. Fix any current-generation lowering issue in this PR only. Do not add selector/model scope.
5. Request kernel CODEOWNER review, then apply/request `ready` only when all evidence is visible.
6. Respond quickly and split any requested follow-up instead of growing the first PR.

## Owner audit checklist

- Inspect `git diff c5c8a055..e652c163` in the audit worktree.
- Confirm the mathematical order: FP32 dot, `128**-0.5`, ReLU, signed FP32 head sum.
- Confirm BF16 cache keys, one live row, padding before bounds-check disabling, and sliced output.
- Confirm all five files and DCO trailer; reject any unrelated change.
- Approve the title/body separately. Only then update the fork and open the upstream PR.
