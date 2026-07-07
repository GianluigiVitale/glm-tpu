# PR-G4 — [Quantization] Blockwise-FP8: checkpoint-exact ragged block scales for fused linears

**Branch:** `pr-g4-fp8-ragged-block-scales` (worktree `~/tpu-inference-prs`)
**Commit:** `c349d1d6` (re-cut of `1cfaf85b`: comment/commit-message accuracy fixes from the
round-6 adversarial review — same code)
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
  contracting axis. Only the LAST fused part may be ragged. An assert rejects fused layouts
  whose **total** scale-column count is inconsistent with a uniform ceil grid (which catches
  e.g. two or more ragged parts) — but a **lone ragged part in an interior position** produces
  a consistent total count (Σ per-part ceils == ceil of the sum) and **cannot be detected at
  this call site** (per-part sizes are unavailable there); such a layout would be silently
  mis-scaled. No known checkpoint fuses one: GLM's ragged part (`kv_a_proj_with_mqa`) is last,
  DeepSeek/Kimi parts are block-aligned. See Risk for the honest trade-off vs main.

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
- a fused layout with an inconsistent total scale-column count (2×6 scale columns vs 576
  outputs) fails loudly (assert) — this covers the detectable interior-ragged class; a lone
  interior-ragged part is undetectable at this call site (see Description) and has no test;
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
checkpoints: identical layouts, byte-identical behavior. Interior-ragged fused layouts
(unsupported) split into two classes: with an inconsistent total scale-column count (e.g. two
ragged parts) the new assert fails loudly; with a **lone** interior-ragged part (e.g. a
hypothetical `[576, 512]` @ 128 fusion — no known checkpoint) the count check passes and the
expansion silently mis-scales every column from the ragged boundary on, **where main failed
loudly at the uniform reshape** — a loud→silent trade for that class, accepted because
detecting it needs per-part sizes (available in `process_blockwise_fp8_linear_weights`, not
at this call site); flagged in code comments and open to a follow-up that validates part
sizes upstream of the matmul.

# Disclosure

Portions of this change were developed with AI assistance (Claude); every line has been
reviewed and is defended by the human submitter.
(Commit carries a `Co-authored-by: Claude` trailer.)

---

## Submitter checklist (not part of the PR body)

- [ ] Re-run the fp8 parity twin on the re-cut branch (owner TPU time).
- [ ] `pre-commit run --all-files`; add DCO `Signed-off-by` when pushing.
