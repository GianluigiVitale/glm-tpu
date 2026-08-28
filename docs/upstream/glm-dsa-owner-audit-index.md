# Owner audit index — exact private GLM-5.2 DSA stack

Status: 2026-08-28 UTC. Official upstream has not been mutated.

## Live upstream acceptance audit

- The audited PR base is `e08b64c14208cb5efc34cc3b41eeaa3402346911`.
  Live official `main` is `b256da42e879f6070dc517cf171eb2ab8f6b2c9e`;
  its only later changes are nightly support-matrix files and do not overlap
  PR 1. The readiness verifier fails if a later upstream move overlaps PR 1.
- No open PR targets GLM-5.2, TPU DSA, or TPU `SparseAttnIndexer`. Open PR
  #2324 targets GLM-5.1 and remains 33 files, +4,453/-336; a maintainer
  explicitly declined review because of its size.
- Sparse-MLA PR #2457 was closed unmerged after a maintainer requested a real
  performance measurement. PR 1 includes a repository-native, synchronized
  real-v4 benchmark and limits its claims to isolated kernels.
- The focused DeepSeek-v4 indexer PR #2905 and integration PR #2980 were merged
  after direct CODEOWNER review. This series follows the same kernel-then-
  integration boundary and adds a separate model/CI patch.
- Inline #2905 review requested repository-local kernel imports, explicit
  block parameters, no unnecessary clone/zero-token branches, a separate
  numerical reference, and exact power-of-two quantization. The current stack
  satisfies those shapes. Its V3.2 writer cannot reuse the common quantizer
  because that helper does not perform the required UE8M0 power-of-two
  rounding; exact serialized FP32-scale bytes are tested directly.
- Issue #1699 (GLM5 support) remains open. The drafts say “contributes toward”
  rather than closing it before full maintainer acceptance.
- Current CONTRIBUTING prefers TorchAX-first vLLM model enablement, requires
  focused unit and CI tests, and prescribes all-file pre-commit. The stack uses
  those paths and contains eleven DCO-signed commits.

Relevant live pages:

- https://github.com/vllm-project/tpu-inference/issues/1699
- https://github.com/vllm-project/tpu-inference/pull/2324
- https://github.com/vllm-project/tpu-inference/pull/2457
- https://github.com/vllm-project/tpu-inference/pull/2905
- https://github.com/vllm-project/tpu-inference/pull/2980

## Exact ranges to audit

| PR | Dependency | Exact range | Size |
|---|---|---|---:|
| 1 | none | `e08b64c14208cb5efc34cc3b41eeaa3402346911..650b5fccb890b5a872871af489b50fc4c584e8ad` | 8 files, +1,250/-10 |
| 2 | PR 1 | `650b5fccb890b5a872871af489b50fc4c584e8ad..d837832ab41f947ee9ff759e65ea8417ba1bd5c9` | 11 files, +1,344/-34 |
| 3 | PRs 1-2 | `d837832ab41f947ee9ff759e65ea8417ba1bd5c9..101ec506d76a3ecb0b688315e432ae7e8d0ab37a` | 2 files, +302/-0 |

Run the first audit now:

```bash
git -C /home/gianl/tpu-inference-glm-baseline diff --find-renames \
  e08b64c14208cb5efc34cc3b41eeaa3402346911..650b5fccb890b5a872871af489b50fc4c584e8ad
```

Then use the exact ranges above for PRs 2-3. Do not review `main...head` with a
floating merge base.

## Review packet

- PR 1 file/risk/evidence map: `docs/upstream/glm-dsa-pr1-owner-audit.md`
- PR 2 file/risk/evidence map: `docs/upstream/glm-dsa-pr2-owner-audit.md`
- PR 3 file/risk/evidence map: `docs/upstream/glm-dsa-pr3-owner-audit.md`
- Ready-to-paste titles/bodies: `docs/upstream-glm-dsa-pr-drafts.md`
- Full test/evidence status: `docs/upstream-glm-dsa-pr-status.md`

## Integrity proof

- All three private branch heads are pushed only to
  `GianluigiVitale/tpu-inference`.
- Exact PR3 tree: 1,103 tracked files; largest tracked file 319,925 bytes
  (`docs/assets/torchax.png`). No checkpoint, safetensor, NumPy dump, XPlane,
  cache, environment, or generated run tree is tracked.
- Complete three-ref bundle:
  `glm-dsa-private-stack_20260828T090200Z.bundle`, 12,158,343 bytes, SHA-256
  `38800e54e9b03ffe930a8629c0ea6b4039e4cbb19d188a54665889ddb8e31f49`.
- Compact evidence and repository mirrors use only
  `gs://driftbench-dsv4-uc` in exact `US-CENTRAL2`.

## Owner decision boundary

Audit and decide PR 1 first. Approval must name the exact PR 1 head
`650b5fccb890b5a872871af489b50fc4c584e8ad`. No official branch, PR, issue
comment, label, or other upstream mutation is authorized until that explicit
approval is received.
