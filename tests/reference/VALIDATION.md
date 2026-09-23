# Reference model validation (S2e receipt, DESIGN 7.6 / D18)

`tests/reference/` is an unsharded, single-device, op-by-op pure-JAX forward of GLM-5.3 over
the dequantized frozen fixture v1 (`model.py` composing `linear.py`, `norm.py`,
`attention.py`, `dsa.py`, `moe.py`). It calls no production `shard_map` or Pallas code. This
file records its cross-validation against the still-present frozen FP8 oracle and against the
current production composition, what agrees exactly, what does not, and why.

## What was measured

| Item | Value |
|---|---|
| Tree | HEAD `187ecb333b80` plus `tests/reference/` of this commit; production paths identical to `181c013e` (`source.production_paths_equal_baseline = true`) |
| Environment | Python 3.12.13, jax 0.10.1, jaxlib 0.10.1, numpy 2.3.5, ml_dtypes 0.5.4; `JAX_PLATFORMS=cpu`, `XLA_FLAGS=--xla_force_host_platform_device_count=32`; engine Pallas kernels in interpret mode with the TPU-v4 chip description |
| Fixture | `frozen_fixture_v1`, panel geometry, 1,536 slots, DSA `top_k` 128; checkpoint `tree_record` digest `d8664fd5cdd4911a3029b3224633f63ec1e09deca98a4b4e5444f2b72aa66809` |
| Reference weights | dequantized BF16 tree digest `afd42ad1833baa05416310859a40d2d8e343d345a4ea832f1d518d907ca932ee` (identical with the eager and the jitted dequantizer) |
| Engine inputs | the promoted FP32 indexer `wk` tables (`build_wk_programs`) and the host RoPE table equal the reference's bitwise (`engine_inputs_equal_reference = {wk: true, rope: true}`) |
| Prompts | `short` = `[30, 31, 32]` (the prompt of `tests/release/test_optimized_{bf16,prefill_program}.py`), sha256 `91b89f90f71a…`; `a` = G3 prompt A, `(37 i + 11) mod 256` for 157 tokens (a 128-row block and a 29-row tail), sha256 `d3faaaa50b4e…` |
| Schedule | prefill in 128-row blocks (production: B128 + B114 tail program), then 8 greedy decode steps; the baseline decodes its own tokens, the candidate is teacher-forced with the baseline's input token at every step |
| Tolerances | float leaves: `abs(candidate - baseline) <= 0.0625 + 0.02 * abs(baseline)` (the frozen-prefill bounds); integer/boolean leaves: exact |

Systems: `reference` (128-row prefill blocks), `reference-one-block` (the whole prompt in one
block: only matmul shapes, i.e. accumulation order, change), `fp8-oracle`
(`build_ws32_batched_prefill_program` B128 with the production window options, and
`build_ws32_decoder_program(...).observe`, which exposes every full indexer's selection) and
`production` (`OrdinaryRuntime._load`'s composition at 181c013e: BF16-resident tables,
`build_ws32_prefill_challenger_program` B128/B114, `build_packed_decoder_program`; only the
carried selection of layer 6 is observable). `production:fp8-oracle` is the distance between
the two accepted engines (the *engine floor*); `reference-one-block:reference` is the
reference's own rounding floor.

Command (one JSON line per pair; `PAIR` = `CANDIDATE:BASELINE`, `PROMPT` = `short` or `a`):

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 XLA_FLAGS=--xla_force_host_platform_device_count=32 \
  python -B -m tests.reference.oracle_run --pair PAIR --prompt PROMPT --steps 8
JAX_PLATFORMS=cpu pytest -p no:cacheprovider tests/reference            # all tests
JAX_PLATFORMS=cpu pytest -p no:cacheprovider -m "not slow" tests/reference   # fast tier
```

## Results

### The DESIGN 7.6 criteria, per pair

`ref:fp8` = reference vs FP8 oracle, `ref:prod` = reference vs production, `prod:fp8` = the
engine floor, `self` = reference one block vs 128-row blocks.

| Criterion | short ref:fp8 | short ref:prod | short prod:fp8 | short self | A ref:fp8 | A ref:prod | A prod:fp8 | A self |
|---|---|---|---|---|---|---|---|---|
| prefill integer leaves equal (incl. score-ordered `selected_positions`) | yes | yes | yes | yes | no | no | no | no |
| prefill integer state equal (all other integer/boolean leaves) | yes | yes | yes | yes | yes | yes | yes | yes |
| prefill float leaves within rtol 0.02 / atol 0.0625 | yes | yes | yes | yes | no | no | no | yes |
| prefill next token equal | yes | yes | yes | yes | yes | yes | yes | yes |
| greedy tokens equal, 8 decode steps | yes | yes | yes | yes | **yes** | no (2 BF16 ties) | no (1) | no (1) |
| decode integer state equal | yes | yes | yes | yes | yes | yes | yes | yes |
| decode float leaves within the bounds | no | no | no | yes | no | no | no | no |
| DSA selections equal in score order | no | no | no | yes | no | no | no | no |
| DSA selections equal as sets | yes | yes | yes | yes | no (8/32) | no (3/8) | no (5/8) | no (5/32) |
| every set disagreement within the pair's score noise of the top-k boundary | yes | yes | yes | yes | yes | yes | yes | yes |
| decision-free rows within the bounds (see below) | yes | yes | yes | yes | yes | yes | yes | yes |

Tokens (prefill token, then the 8 decode predictions): short, every pair
`[55, 55, 77, 55, 77, 55, 77, 77, 77]`. Prompt A: FP8 oracle and reference
`[218, 121, 218, 218, 218, 218, 218, 218, 218]`; production `[218, 218, …]` (all 218).

### Observed errors (max absolute / max ratio to the bound / fraction of elements outside it)

| Leaf | short ref:fp8 | short ref:prod | short prod:fp8 | A ref:fp8 | A ref:prod | A prod:fp8 | A self |
|---|---|---|---|---|---|---|---|
| prefill `kv_cache_local` | 0.031 / 0.34 / 0 | 0.045 / 0.66 / 0 | 0.039 / 0.54 / 0 | 0.193 / 2.75 / 6.2e-5 | 0.189 / 2.69 / 1.0e-4 | 0.178 / 2.53 / 6.9e-5 | 0.031 / 0.44 / 0 |
| prefill `index_cache_local` (= `repaired_index_local`) | 0.020 / 0.27 / 0 | 0.031 / 0.43 / 0 | 0.031 / 0.46 / 0 | 0.127 / 1.71 / 5.5e-5 | 0.164 / 2.02 / 1.2e-4 | 0.156 / 1.90 / 7.1e-5 | 0.023 / 0.31 / 0 |
| prefill `selected_scores` | 0.010 / 0.15 / 0 | 0.007 / 0.11 / 0 | 0.014 / 0.20 / 0 | 0.026 / 0.30 / 0 | 0.037 / 0.45 / 0 | 0.036 / 0.43 / 0 | 0.017 / 0.18 / 0 |
| decode `kv_cache_local` | 0.174 / 2.74 / 2.1e-5 | 0.146 / 2.24 / 1.5e-5 | 0.169 / 2.66 / 2.7e-5 | 0.193 / 2.75 / 7.9e-5 | 0.189 / 2.69 / 1.0e-4 | 0.178 / 2.53 / 8.6e-5 | 0.047 / 0.56 / 0 |
| decode `index_cache_local` | 0.113 / 1.44 / 1.5e-5 | 0.109 / 1.33 / 3.8e-6 | 0.102 / 1.35 / 1.4e-5 | 0.129 / 1.95 / 7.1e-5 | 0.164 / 2.02 / 1.2e-4 | 0.156 / 1.93 / 8.9e-5 | 0.035 / 0.47 / 0 |
| decode `selected_scores` | 0.060 / 0.94 / 0 | 0.062 / 0.97 / 0 | 0.061 / 0.90 / 0 | 0.114 / 1.11 / 1.6e-2 | 0.055 / 0.52 / 0 | 0.124 / 1.21 / 1.6e-2 | 0.015 / 0.17 / 0 |
| decode `final_residual_local` | 0.239 / 3.64 / 0.19 | 0.234 / 2.81 / 0.15 | 0.261 / 3.96 / 0.18 | 0.188 / 2.94 / 0.16 | 0.104 / 1.44 / 0.019 | 0.221 / 3.46 / 0.22 | 0.125 / 1.68 / 0.022 |

The reference's bound ratios are at or below the engine floor or above it by at most 9 %
(prompt-A KV 2.75 vs 2.53; short decode index 1.44 vs 1.35 and selected scores 0.97 vs 0.90,
the latter still inside the bound); its outside fractions exceed the floor by at most 1.7x
(prompt-A index cache vs production, 1.2e-4 vs 7.1e-5). Decision-free rows: largest bound
ratio 0.28 / 0.33 / 0.35 (short, ref:fp8 / ref:prod / prod:fp8) and 0.40 / 0.49 / 0.49
(prompt A).

## Why the strict criteria fail: discrete decisions at near-ties

The criteria fail in the same places for the two accepted engines (`prod:fp8`) and for the
reference against itself (`self`, where only matmul shapes change), so no implementation with
a different accumulation order can meet them on this fixture. The mechanism, measured:

1. **MoE top-8 routing near-ties.** Accumulation-order rounding moves BF16 activations by one
   ulp; router scores then move by ~1e-4, which flips a top-8 choice whose margin (k-th minus
   (k+1)-th biased sigmoid score, `moe.Routes.margin`) is that small. A flipped row's hidden
   state changes by a whole expert's contribution, so its cache rows leave the float bounds from
   the next layer on and attention passes a smaller perturbation to other rows. Evidence
   (reference vs FP8 oracle, prompt A, first block): no row is outside the bound in layers 0-3;
   11 rows leave it later, 7 of them at the layer right after a router margin of at most 1.8e-4
   (rank at most 7 of 128 in that layer; layer medians 2.5e-3 to 5.8e-3): row 124 (1.8e-4 at
   layer 3), rows 8, 50, 74, 112 (2.0e-5 to 1.5e-4 at layer 4), row 120 (1.2e-7 at layer 5), row
   81 (7.0e-6 at layer 6); rows 20, 32, 76 follow margins of 5.4e-4 to 8.6e-4 and row 6 none
   below 9.8e-4 (spill-over from flipped rows). In decode each jump sits at the layer after a
   margin below 1.6e-4 (the other system may land on either side): short prompt, position 5
   (layer-3 margin 4.6e-5; bound ratio of the new KV row 0.25 at layer 3, 1.64 at layer 4 vs the
   FP8 oracle, no jump vs production), position 8 (layer 3, 1.5e-4; jump at layer 4 vs
   production only), position 9 (layer 6, 4.1e-5; jump at layer 7 vs both), position 10 (layer
   4, 9.4e-5; jump at layer 5 vs both). Rows no routing or DSA choice can reach never leave the
   bounds (the decision-free criterion).
2. **DSA top-k boundary ties.** With 158-165 positions and `top_k` 128 the selection has a
   boundary; every set disagreement swaps positions whose score is 0 to 0.027 above the last
   selected score, within twice the pair's own score difference over commonly selected positions
   (0.005 to 0.094). Score *order* among the selected positions is noise-sensitive in every pair
   (short: 4 to 19 slot mismatches over 8 steps; prompt A: 728 to 1,705); attention reads the
   position-sorted set, so the order is not model output.
3. **Greedy BF16 ties.** Prompt A step 1, the reference's logits: `121: 9.125`, `218: 9.0625`,
   one BF16 ulp apart.
   The FP8 oracle and the reference choose 121, production 218, the reference with the whole
   prompt in one block 218. Step 8 vs production: `121` and `218` both `8.8125` (lowest id wins,
   121). No other step differs in any pair.
4. The fixture is a random-weight model: each of those flips feeds the next layers, so the
   float differences of every pair grow with depth from about one BF16 ulp at layer 0.

The existing oracle tests did not see this because they compare implementations that share
the sharded accumulation structure, start decode from the same prefill state and stop after
three steps (`test_optimized_bf16`) or one three-token block (`test_optimized_prefill_program`).

## Acceptance as implemented (`test_reference_consistency.py`, `cpu32`, `slow`)

Tolerances are unchanged. Every DESIGN 7.6 criterion the two engines meet against each other
is asserted exactly for the reference against each engine:

* short prompt: prefill integer leaves equal, prefill float leaves within the bounds, prefill
  token, all 8 greedy tokens, decode integer state, DSA selection sets of every observable layer
  in every step, decision-free rows within the bounds;
* prompt A: prefill and decode integer state equal, prefill token, decision-free rows within the
  bounds, set disagreements explained by the pair's score noise, greedy tokens equal (FP8 oracle:
  all 8 exactly; production: equal or a BF16 tie in the reference's logits).

Where even the engines fail (decode floats on both prompts, prefill floats, set equality and
score order on prompt A), the reference must be no further from the engine than production is
from the FP8 oracle in the same run: per leaf, the bound ratio at most 1.0 where the engines
are within the bounds and at most 1.25 x the engine floor where they are not, the fraction of
elements outside the bound at most 2 x the floor + 1e-5, and set-unequal steps of the carried
layer at most the floor's + 1. Evaluated on the reports of the command above (the child the
tests run, same code and inputs): all four checks pass.

## Semantic choices (for later stages)

The reference follows the engine's numerical contract, which differs from the vendored Hugging
Face eager modeling (`reference/modeling_glm_moe_dsa.py`) in these documented places:
indexer RoPE pairs are interleaved per `indexer_rope_interleave: true` (the HF code comment says
half-split); q-a/kv-a LoRA RMSNorm uses `rms_norm_eps` 1e-5 (HF constructs them with its 1e-6
default); DSA head weights are FP32 (HF rounds the projection to BF16); `routed_scaling_factor`
multiplies the routed sum (HF: the weights; equal in exact arithmetic); main RoPE output keeps
pairs interleaved (HF concatenates halves; the q.k product is identical). Within the engine
contract, the reference uses one association where the engine has two: the indexer key LayerNorm
is `(x - mean) * rsqrt(var + eps)` (engine decode; engine prefill divides by `sqrt`), and the
weighted routes are summed in FP32 (engine prefill; engine decode sums one owner chip's routes
in BF16 first). Both differ by FP32 rounding only.

## Dependencies on oracle functions (keep for S2f and S4)

`glm_tpu/greenfield/kernels/reference/`: `fp8.dequantize_fp8_bits_block_weight`,
`rmsnorm.rms_norm`, `rmsnorm.fused_add_rms_norm`, `rotary.build_rotary_table_host`,
`rotary.apply_rotary_fp32_final_round` (and, through `dsa`, `rotary.rotary_cos_sin`,
`rotary.apply_rotary`), `dsa.DsaNumericalContract`, `dsa.SelectedPositions`,
`dsa.dsa_index_keys`, `dsa.dsa_query_and_head_weights`, `dsa.dsa_scores`, `dsa.exact_topk`,
`attention.MlaNumericalContract`, `attention.gather_paged_selected_kv`,
`attention.sparse_mla_attention` (and `attention.canonicalize_selected_positions`),
`moe.route_glm_noaux_tc_logits`. DESIGN 3.3 moves the oracle-only ones into `tests/reference/`
at S4; S2f's pruning must keep them. `oracle_run.py` also imports the frozen oracle entry
points (`ws32_batched_prefill.build_ws32_batched_prefill_program`,
`make_ws32_batched_prefill_state`, `finish_ws32_batched_prefill`,
`ws32_decoder.build_ws32_decoder_program`, `build_ws32_main_rope_table`), the production
builders (`optimized.bf16_resident.bf16_resident_weights`,
`optimized.prefill_challenger.build_ws32_prefill_challenger_program`,
`optimized.request_loop.build_packed_decoder_program`,
`scripts.greenfield.ws32_native_benchmark_programs.build_cache_initializer`,
`scripts.greenfield.ws32_compile_originals.build_wk_programs`) and
`tools/equivalence/{fixture,common,lowering,budget}.py`; S2a moves the two `scripts` builders and
S2c introduces `build_program_set`, which `Production` should then use.
`test_reference_matches_frozen_fp8_oracle` and the `fp8-oracle` system are archived with the
frozen oracle in S2f; the engine-floor numbers above are then the recorded floor for
`tests/models/glm_moe_dsa/test_against_reference.py` (S5).

## Runtime (240-core host)

Non-`cpu32` tests: 59 s (fast tier 19 s; the two `slow` ones 27 s and 13 s). Each oracle child
takes 2.0 to 3.2 min wall and 3.6 to 5.3 GB RSS when eight run concurrently; the two `cpu32`
tests run six children in sequence (the engine floor is shared): about 16 min, estimated from
those child times.
