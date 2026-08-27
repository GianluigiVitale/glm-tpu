# Draft upstream PR — do not submit without owner approval

## Title

`[Kernel][GLM] Add TPU Pallas DSA indexer scorer`

## Description

This adds a self-contained Pallas score kernel for the GLM lightning/DSA indexer decode path.
It computes one FP32 score row from projected FP32 queries, BF16 cached index keys, and signed FP32
head weights without writing the `[heads, context]` intermediate to HBM.

GLM DSA is not currently implemented on the TPU Torchax path. Related GLM support PR #2324 must
disable the indexer and fall back to dense attention. This small kernel PR establishes the score
primitive independently of model loading, serving, and distributed selection.

The arithmetic order matches GLM: highest-precision query/key dot, `128**-0.5` scaling, per-head
ReLU, then the signed FP32 head reduction. Decode is intentionally one row. Contexts not divisible
by 128 are padded before the bounds-check-free kernel and sliced back to their logical width.

Scope is deliberately limited to the scorer. Exact top-k/distributed merge and the Torchax
`SparseAttnIndexer` bridge will be separate follow-ups.

The new code is isolated under `kernels/experimental/glm`, has no caller, and changes no existing
runtime behavior. This is intentionally a kernel-first slice: it can be reviewed and validated
independently while preserving the repository's Torchax-first model path.

Related: #1699, #2324.

## Tests

- `pre-commit run --files <the five changed files>`
- `python -m pytest -q tests/kernels/glm/test_indexer_score.py`
  - CPU/Pallas interpreter: 5 passed; real-TPU test skips off TPU.
  - Changed-kernel CI executes the real kernel on TPU v6e and v7x; exact job results will be added
    here before requesting review because those jobs are soft-fail.
- TPU v4 production-local shape (`[1,32,128] x [65536,128]`):
  - exact top-2048 positions against the readable FP32 reference;
  - max/mean score error `2.861e-6 / 2.417e-7`;
  - HLO contains one named scorer call and no `[32,65536]` score overlay or batch-32 shape;
  - compile 0.390 s; diagnostic 10-warmup/50-sample p50/p99 `0.291/0.318 ms`.

## Checklist

- [x] I have performed a self-review of my code.
- [x] I have added comments for the non-obvious tiling and numerical order.
- [x] I added focused interpreter, shape, contract-drift, and real-TPU tests.
- [x] No documentation change is required for an unintegrated experimental kernel.
