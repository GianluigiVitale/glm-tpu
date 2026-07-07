# PR-G4 — [Quantization] Blockwise-FP8: checkpoint-exact ragged block scales for fused linears

**Branch:** `pr-g4-fp8-ragged-block-scales` (worktree `~/tpu-inference-prs`)
**Commit:** `1cfaf85b`
**Base:** `97938b62` (fork main, 2026-06-13).
Forward-port to `vllm-project/tpu-inference` main @ `0d59fee9` (2026-07-07): **conflict-free**
(`linear.py` and `quantization/fp8.py` are byte-identical between the base and the tip).
**Status:** ready for owner review; needs the owner's TPU parity re-run before submission.

---

## PR body draft

# Description

With `DISABLE_WEIGHT_REQUANTIZATION=1` (serve the checkpoint's own FP8 blocks — the
checkpoint-exact path), fused linears whose parts are **not block-aligned** loaded wrong scales.
Each fused part is block-quantized separately in the checkpoint, so a part whose size is not a
multiple of the block ends in a partial block that still owns a **full scale column**. Concrete
case: GLM's `kv_a_proj_with_mqa` (576 outputs @ block 128 → `ceil(576/128) = 5` scale columns,
the last covering 64 features).

Two fixes:

- `process_blockwise_fp8_linear_weights` derived per-part scale-column counts with **floor**
  division (`s // block`). Flooring silently dropped the partial block's scale column and
  misaligned every following part's scales. Now `ceil(s / block)`.
- `xla_quantized_matmul`'s 2D-block-scale branch expanded scales with a uniform-blocks reshape
  (`block_size_out = out_features // out_blocks`), which cannot represent a ragged tail at all
  (the reshape fails outright at 576 @ 5 columns, and would silently mis-scale if it didn't).
  Replaced with repeat+clip expansion: the block size is derived from the always-block-aligned
  contracting axis, and a hard assert rejects interior-ragged fused concatenations (only the
  LAST fused part may be ragged) instead of silently mis-scaling.

Block-aligned checkpoints produce identical scale layouts (`ceil == div` when aligned) and take
the byte-identical path.

## Why this is not duplicating an existing PR

Searches run (2026-07-07): open PRs matching `block scale`, `fp8 v4`, `gmm`, `kv_cache_dtype`,
`requantization`. Findings: no open PR touches the `DISABLE_WEIGHT_REQUANTIZATION` scale-layout
path. Adjacent-but-different: **#2324** adds a 2D `scale_n_block_size` mode for its HBM-dequant
MoE path (MoE expert weights, not fused attention linears); **#2822** fixes the ragged-tile
scale reshape *inside the gmm_v2 kernel* (per-tile, not per-fused-part checkpoint layout).
Neither addresses the fused-linear ceil/ragged-tail scale-column bug.

# Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests/layers/common/test_fp8_ragged_block_scales.py -q
```
Result: **5 passed** (new file):
- ragged 576@128 expansion **bit-exact** vs an explicit per-block dequantization reference
  (matmul outputs compared with `assert_array_equal`);
- block-aligned expansion bit-exact (no-regression case — also passes on main);
- interior-ragged concatenation fails loudly (assert), never silently mis-scales;
- `process_blockwise_fp8_linear_weights` keeps all `ceil` scale columns per fused part,
  fused (`reorder`) and unfused (`slice`) paths, verified column-by-column.

Adversarial check: 4 of 5 **fail on main before the change** (the reshape raises on the ragged
shape; the floor path silently drops the tail scale column); the aligned case passes on both.

Machine-gated functional run (TPU v4, dev branch): the fp8 GLM engine-parity twin — the HF
reference loads the SAME effective weights via quant-dequant roundtrip, so the diff isolates the
engine's fp8 handling — top-1 1.000 vs both fp32 and bf16 references (RESEARCH_LOG 2026-07-07).
The re-cut branch needs the submitter's parity re-run.

# Risk

Only the `DISABLE_WEIGHT_REQUANTIZATION` / 2D-block-scale path changes. Block-aligned
checkpoints: identical layouts, byte-identical behavior. The new assert converts a silent
mis-scale into a loud failure for the (unsupported) interior-ragged case.

# Disclosure

Portions of this change were developed with AI assistance (Claude); every line has been
reviewed and is defended by the human submitter.
(Commit carries a `Co-authored-by: Claude` trailer.)

---

## Submitter checklist (not part of the PR body)

- [ ] Re-run the fp8 parity twin on the re-cut branch (owner TPU time).
- [ ] `pre-commit run --all-files`; add DCO `Signed-off-by` when pushing.
