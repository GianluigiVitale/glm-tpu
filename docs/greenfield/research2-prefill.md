# TPU-v4 Deep Research: Grouped-MoE Weight Reuse, Exact DSA Top-2048, and Large Layer Windows

## Executive decision

The highest-value change is **not another small `row_tile` adjustment to the current grouped kernel**. The present eight-row schedule has a structural weight-reuse problem: the same expert can occupy several globally aligned row tiles, and each such visit causes the Pallas program to request that expert's weight block again and execute the FP8→FP32-scale→BF16 decode again. The attached kernel makes that explicit: its grid is `(tiles_n, active_tiles, tiles_k)`, the weight `BlockSpec` depends on the group selected by each `gi`, and the decode is inside the per-`(ni, gi, ki)` kernel invocation. fileciteturn0file1 Upstream Tokamax's TPU grouped-matmul implementation uses the same basic grouping model and even models RHS traffic as `rhs_bytes * group_ids.size`, i.e. expert-weight traffic scales with **group-tile visits**, not just unique experts. fileciteturn4file0

So the source-level answer to the first question is **yes: weights are repeatedly requested and dequantized for repeated expert-row-tile visits**. Whether every one of those requests becomes a physical HBM transaction cannot be proved from source alone, because compiler-managed caching can affect actual memory traffic. Pallas's documented model nevertheless requires HBM inputs to be transferred to VMEM before computation, and the current code has no persistent expert-weight VMEM object spanning different `gi` visits. citeturn10search4turn10search0

The concrete replacement I recommend is an **expert-aligned, weight-panel-stationary grouped kernel**:

| Prompt batch | Mean routed rows / expert | First row panel | N panel | HBM weight buffering | K reduction |
|---|---:|---:|---:|---:|---|
| B512 | 16 | **32 rows** | **512 output channels** | **2 buffers** | 128-wide subtiles, ascending, FP32 accumulator |
| B1024 | 32 | **64 rows** | **512 output channels** | **2 buffers** | 128-wide subtiles, ascending, FP32 accumulator |
| B1024 control | 32 | 32 rows | 512 | 2 buffers | identical |

The critical difference is that **K disappears from the outer Pallas grid**. A program receives one `[TM, K]` activation panel plus one raw `[TN, K]` expert-weight panel in VMEM and walks its 128×128 weight subtiles internally. That converts "load/decode the expert weight once per global 8-row visit" into "load/decode each weight element once per expert-aligned row panel." It keeps the current FP32 accumulation sequence and performs the BF16 conversion at the same existing boundary. TPU matrix multiply natively produces FP32 results in Pallas, and the official guidance is to use BF16 inputs with an FP32 accumulator and downcast only at the chosen output boundary. citeturn10search1turn10search2

For DSA, retain `lax.top_k(..., is_stable=True)` for the **first unsorted 4096-position tile**, but stop repeatedly invoking a generic top-k/sort for every merge. Exact sorted `K=2048` candidate lists should be merged with a **12-stage top-half bitonic merge network** under the explicit total order `(score descending, position ascending)`. A two-list merge then costs 13,312 compare-exchanges per row, versus 159,744 for a full custom bitonic sort of 4096 values. JAX's exact `top_k` is stable, and OpenXLA specifies that ties put the lower input index first. citeturn10search6turn12search11 Across the expert-8 axis, replace the current 8-way `all_gather` plus replicated global merge with a **three-round XOR butterfly using `lax.ppermute` and the same exact merge**. The attached DSA currently materializes all eight lists before its final merge. fileciteturn0file0 JAX documents `ppermute` as the direct point-to-point collective and explicitly shows it as a building block for communication-efficient collective algorithms. citeturn9search0

For 16/32 attention tiles per layer, do **not** extend the Python loop in `ws32_prefill_window.py`. It is currently a trace-time `range()` loop, so making it 16 or 32 iterations creates exactly the unrolling/executable-growth problem you are trying to avoid. fileciteturn0file2 Use one fixed 32-row body inside `lax.scan(..., unroll=1)`. JAX lowers `scan` to a single `WhileOp` specifically to avoid the large XLA programs caused by Python-loop unrolling. citeturn8search0 Carry only the current proposal cache generation, unrepaired index proposal, repaired index proposal, and health state; stack row outputs, **not cache generations**.

The most decisive first measurement is therefore:

> **Capture the exact same natural routed row stream at B512 and B1024, run the current TM8 kernel and expert-aligned TM32/TM64 kernels on the same real layer weights, and compare both the statically predicted expert-weight visit count and XProf kernel time/memory-bandwidth behavior.** If kernel time falls in proportion to the predicted visit reduction while numerical outputs remain exact at the existing FP32/BF16 boundaries, the hypothesis is proved. XProf supports TPU operation timelines and memory-bandwidth utilization; Cloud TPU exposes both memory-bandwidth and TensorCore-utilization metrics on v4 and newer. citeturn16search5turn17search1turn16search4

## Grouped MoE: the eight-row schedule really is throwing away weight reuse

### What the existing code does

The attached `prefill_grouped_fp8.py` has four details that matter here. It accepts `row_tile` only from `{8, 16, 32}`; constructs grouped metadata with that tile size; launches a grid over output-channel tile, grouped-row-tile visit, and K tile; and decodes the selected 128×128 raw FP8 weight tile inside the kernel before the dot. fileciteturn0file1 Conceptually its execution is:

```text
for n128:
    for group_row_tile_visit:
        expert = group_ids[group_row_tile_visit]
        for k128:
            x = activation[row_tile, k128]
            raw_w = weight[expert, n128, k128]
            scale = scale[expert, n128, k128]

            # Executed again for every row-tile visit:
            w_bf16 = (
                bitcast(raw_w, float8_e4m3fn).astype(float32)
                * scale
            ).astype(bfloat16)

            acc_f32 += dot(x_bf16, w_bf16)
```

That is not weight-stationary. It is **row-tile-stationary with an FP32 output accumulator**. An expert covering 24 routed rows under `row_tile=8`, for example, generally requires about three group-tile visits; all of its weight tiles are consequently decoded once for each visit. fileciteturn0file1

The behavior is not accidental to your local implementation. Tokamax's current Pallas TPU grouped matmul constructs group metadata such that a group beginning or ending inside an M tile may revisit a globally aligned M tile, then indexes RHS weights by the group assigned to each such visit. fileciteturn3file0 Its explicit cost estimate computes RHS bytes proportional to the number of entries in `group_ids`, which is exactly the cost model one would write for repeated expert-weight access across grouped-row visits. fileciteturn4file0

There is therefore a useful distinction:

**Repeated dequantization is proven by the source.** The FP8 decode/scale/BF16 conversion is inside the kernel program and re-executes when the same expert is revisited. fileciteturn0file1

**Repeated physical HBM traffic is strongly implied but must be measured.** Pallas inputs originate in HBM and computation occurs from VMEM, but the implementation does not expose every detail of compiler-managed caching between those spaces. citeturn10search4 What is certain is that your code contains no explicit VMEM-resident expert panel whose lifetime spans successive row-tile visits.

### Why the problem becomes much larger at B512/B1024

For a nonempty expert with group size \(s_e\), starting at routed-row offset \(o_e\), a globally aligned M-tile schedule visits that expert approximately

\[
V_e(M)
=
\left\lceil
\frac{(o_e \bmod M)+s_e}{M}
\right\rceil.
\]

The extra visit is caused by expert boundaries that land in the middle of a global row tile. That is the same boundary-revisit mechanism documented in grouped-matmul metadata generation. fileciteturn3file0

Your model routes top-8 to 256 experts, so under the uniform sanity model a particular expert sees a token with probability \(8/256=1/32\), giving mean group sizes of 16 at B512 and 32 at B1024. The local brief gives the same \(B/32\) mean-row/expert relationship. fileciteturn0file3

With TM8 and essentially all experts active, there are \(B\) unavoidable eight-row route tiles because there are \(8B\) routed rows, plus roughly \(7/8\) of an extra visit for every expert boundary. That gives the following **derived source-level traffic model**:

| Case | Approx. current TM8 visits / expert | Expert-aligned target | Expected target panels / expert under uniform routing | Approx. weight-request reduction |
|---|---:|---:|---:|---:|
| B512 | **2.87** | TM32 | **1.0001** | **2.87×** |
| B1024 | **4.87** | TM64 | **1.0000001** | **4.87×** |
| B1024 control | 4.87 | TM32 | **1.453** | **3.35×** |

These are derived estimates, not measured route statistics; the real run should compute the exact ratios directly from its captured `group_sizes`. They are nonetheless strikingly consistent with your existing B128 experiment: your one-call distributed case reported 345 active grouped tiles, versus 868 across eight B16 calls. fileciteturn0file3 With 1,024 routed rows, TM8 implies 128 base row tiles plus boundary revisits, putting a healthy, broadly distributed route pattern in roughly that 340–350 range. That existing result is strong local evidence that **boundary-driven group visits are already a first-order term**.

The weight dimensions strengthen the case. Each local gate or up expert matrix is \(2048\times1536\) raw bytes, and each down matrix is \(1536\times2048\), so each projection has exactly 3 MiB of raw 8-bit weight per local expert. Your expert-8 sharding gives 32 local experts per chip. fileciteturn0file3 Reading each local expert exactly once therefore corresponds to roughly 96 MiB per projection, or 288 MiB for gate+up+down per MoE layer per chip, before scales and other traffic. Under the visit ratios above, the source-level weight requests are roughly 827 MiB at B512 and 1.40 GiB at B1024, versus about 288 MiB if every local expert fits one expert-aligned panel. This is a derived traffic estimate from the attached weight shapes and visit model. fileciteturn0file3turn0file1

TPU v4 is rated at 275 TFLOP/s BF16 and 1,200 GB/s HBM bandwidth per chip, a peak ratio of about 229 FLOP/byte. citeturn11search5 If one raw 8-bit expert weight is read once, the useful matrix-multiply intensity against weight bytes alone is only approximately \(2s_e\): about 32 FLOP/B at B512 and 64 FLOP/B at B1024. Even deliberately padding computation to 32 or 64 rows gives only about 64 or 128 physical FLOP/B against those weight bytes. This roofline calculation ignores activation/output traffic and decode work, so it is deliberately optimistic; it says the proposed kernels can plausibly afford some masked MXU work in exchange for eliminating repeated weight fetches. citeturn11search5turn11search1

That is why I would choose **TM32 at B512 and TM64 at B1024**, rather than TM16/TM32 merely because they match the mean group size exactly. At B512 essentially every uniformly routed expert fits under 32 rows, so going to 64 adds almost no weight reuse but doubles padding. At B1024, TM32 makes roughly 45% extra panels per expert under the uniform model, while TM64 makes virtually every expert a single panel. The natural captured route histogram, not the uniform model, should make the final choice.

### The kernel should become expert-aligned and K-inner

The important change is not just setting `row_tile=32`. A larger **global** row tile still has boundary sharing and therefore still revisits experts. Instead, build metadata whose unit is `(expert, expert_local_panel)`:

```python
# Pseudocode: metadata construction, outside the Pallas body.

for local_expert in local_experts:
    start = group_offsets[local_expert]
    size = group_sizes[local_expert]

    # B512: TM = 32
    # B1024: TM = 64
    for panel in range(ceil_div(size, TM)):
        panel_rows.append(start + panel * TM)
        panel_experts.append(local_expert)
        panel_live_rows.append(min(TM, size - panel * TM))
```

The Pallas grid should then be logically:

```text
grid = (
    ceil_div(N, TN_PANEL),     # parallel: output-channel panel
    num_active_expert_panels,  # arbitrary: expert/panel sequence
)
```

rather than:

```text
(tiles_n, active_global_row_tiles, tiles_k)
```

I recommend `TN_PANEL=512` initially. It is four native 128-wide MXU output tiles; v4 and other TPU generations before v6e use 128×128 MXUs, and matrix multiplication is still performed as those native small contractions internally. citeturn11search1turn10search2

The HBM `BlockSpec`s then become conceptually:

```python
x_spec:
    [TM, K]               # one expert row panel, full contraction dim

w_spec:
    [1, TN_PANEL, K]      # raw uint8, one expert + N panel

scale_spec:
    scales covering
    [TN_PANEL / 128, K / 128]

out_spec:
    [TM, TN_PANEL]
```

and the compute body becomes:

```python
def expert_panel_kernel(
    x_ref,          # BF16 [TM, K], already in VMEM
    raw_w_ref,      # U8   [TN, K], already in VMEM
    scale_ref,
    out_ref,
    acc_f32_ref,    # F32 [TM, TN]
    live_rows,
):
    acc_f32_ref[...] = 0.0

    # N subtiles are independent; K order is deliberately unchanged.
    for nj in range(TN_PANEL // 128):
        for kj in range(K // 128):
            x128 = x_ref[:, kj * 128 : (kj + 1) * 128]

            raw128 = raw_w_ref[
                nj * 128 : (nj + 1) * 128,
                kj * 128 : (kj + 1) * 128,
            ]

            # Same numerical boundary as current kernel.
            w128 = (
                bitcast_convert_type(raw128, float8_e4m3fn)
                .astype(jnp.float32)
                * scale_ref[nj, kj]
            ).astype(jnp.bfloat16)

            acc_f32_ref[
                :, nj * 128 : (nj + 1) * 128
            ] += jax.lax.dot_general(
                x128,
                w128,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

    # Mask only rows beyond this expert.
    # Cast exactly where the current projection casts.
    out_ref[...] = masked_store(
        acc_f32_ref[...].astype(RESULT_DTYPE),
        live_rows,
    )
```

The crucial property is that the raw `[TN_PANEL,K]` expert panel reaches VMEM **once**, after which all `K/128` decodes refer to that resident panel. Every raw weight element is decoded exactly once for that expert-row panel. There is no `ki` HBM grid dimension left.

This also lowers activation traffic. Under the current 128-wide N tiling, the same `[TM,K]` activation region participates in many N tiles. With `TN_PANEL=512`, one activation panel services four native N tiles before another outer-grid invocation.

For the first v4 implementation I would stay with the existing Pallas-call pipeline and **double buffering** rather than make correctness depend on newer nested-pipeline behavior. Current Pallas documentation says inputs and outputs use a default buffer count of two and allows per-argument multiple buffering. citeturn10search0 Upstream current tests for some newer `emit_pipeline` machinery are explicitly restricted to TPU v5+, so there is no good reason to make nested `emit_pipeline` a prerequisite for this v4 patch when ordinary `pallas_call` pipelining already works in your kernel. citeturn14search3

### Conservative VMEM budget

A current JAX `get_tpu_info()` report for TPU v4 shows 16,777,216 bytes of VMEM per TensorCore, along with 128 lanes and MXU column size 128. citeturn11search0 Your actual runtime should query this value rather than hard-code it.

The following budget assumes `TN_PANEL=512`, double-buffered activation/weight/output panels, one FP32 accumulator, one materialized BF16 128×128 decoded weight tile, and conservative padded scale storage. It intentionally overstates some temporaries because a compiler may fuse or shorten them.

| Projection | TM | Double-buffered X | Double-buffered raw W | Output buffers | FP32 accumulator | Decoded 128² BF16 | Approx. total |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate/up, K1536 | 32 | 0.188 MiB | 1.50 MiB | 0.125 MiB | 0.063 MiB | 0.031 MiB | **~1.92 MiB** |
| gate/up, K1536 | 64 | 0.375 MiB | 1.50 MiB | 0.250 MiB | 0.125 MiB | 0.031 MiB | **~2.30 MiB** |
| down, K2048 | 32 | 0.250 MiB | 2.00 MiB | 0.063 MiB | 0.063 MiB | 0.031 MiB | **~2.42 MiB** |
| down, K2048 | 64 | 0.500 MiB | 2.00 MiB | 0.125 MiB | 0.125 MiB | 0.031 MiB | **~2.80 MiB** |

The figures are derived from your attached local matrix dimensions and Pallas's documented two-buffer default. fileciteturn0file3 citeturn10search0 Even with layout/compiler overhead, these proposed panels leave a large margin below the reported 16 MiB v4 VMEM capacity. citeturn11search0

After the single-projection version is proven, gate and up are good candidates for **one paired panel kernel** because they read the same activation panel. That kernel should still produce two separate **FP32** partial outputs. The existing feature-4 FP32 reduction must happen outside it; only after that reduction should the result cross the existing BF16 boundary and enter SwiGLU. Your project brief requires exactly that sequence, as well as BF16 down output/route products followed later by FP32 route and expert reductions. fileciteturn0file3 Do **not** copy Tokamax's optional fused activation pattern here merely because the upstream kernel supports one: it would move your numerical boundary. Tokamax itself accumulates grouped matmuls in FP32 and only optionally applies activation at its chosen output point. fileciteturn4file0

The required numerical order for your replacement is therefore:

```text
gate/up:
  raw FP8 -> FP32 * F32 scale -> BF16 weight
  BF16 activation × BF16 weight
  K tiles 0,1,... in the same 128-wide order
  FP32 accumulator
  feature4 FP32 reduction
  BF16
  SwiGLU

down:
  BF16 SwiGLU activation
  same K-order / FP32 dot accumulator
  existing down-output cast
  existing BF16 route weighting
  existing FP32 route-slot sum
  existing FP32 expert8 combine
  existing final BF16 cast
```

Changing M and N panelization does not require changing any reduction dimension or dtype boundary. fileciteturn0file1turn0file3

## Exact DSA top-2048: keep exact initial selection, replace generic repeated merges

The attached DSA already does the expensive scoring in tiles, limits `key_tile` to at most 4096, and bounds its largest documented score intermediate to 16 MiB at 32 rows × 32 heads × 4096 keys × FP32. It carries `[rows, top_k]` candidates from tile to tile. fileciteturn0file0 For each nonempty visible key tile it computes scores, selects a local top-2048, then invokes `merge_topk_candidates_with_scores` on the previous and new lists. After context-local selection, expert ranks `all_gather` their score and position candidates and perform another global merge. fileciteturn0file0

That overall decomposition is mathematically sound. What needs replacing is the **merge primitive and distributed aggregation pattern**.

### Define one explicit total order

Every selector and merge should use exactly:

```python
def better(sa, pa, sb, pb):
    return (
        (sa > sb)
        | ((sa == sb) & (pa < pb))
    )
```

with invalid candidates separately ordered below every live candidate.

Do not implement the tie break using a floating epsilon. Do not perturb scores. Do not pack `score` and `position` by an arithmetic transformation that changes FP32 equality.

For the raw 4096-position key tile, positions are monotonically increasing among live entries under the metadata contract checked by your DSA. fileciteturn0file0 Therefore:

```python
scores_k, local_idx = jax.lax.top_k(
    masked_fp32_scores,
    2048,
    is_stable=True,
)
positions_k = take(global_positions, local_idx)
```

already gives the required tie rule **inside a single key tile**: JAX's stable top-k preserves relative order among equal elements, and OpenXLA defines top-k ties such that the lower input index appears first. citeturn10search6turn12search11

I would keep that primitive for the **unsorted 4096→2048 step**. There is no documented TPU-v4 exact-top-k primitive in Pallas that gives a stronger reason to replace it, and `approx_max_k` is explicitly disqualified: JAX says it is approximate, does not guarantee stability, and may choose an implementation-defined subset when a tie crosses the K boundary. citeturn12search1

### Once lists are sorted, never sort their 4096-element concatenation from scratch

Suppose `A` and `B` are already sorted top-2048 lists under `(score↓, position↑)`. Then the exact top-2048 of `A∪B` can be produced by a **top-half bitonic merge**.

Construct a bitonic sequence:

```text
A descending
reverse(B) ascending
```

Perform one compare-exchange between positions `i` and `i+2048`, keeping the better candidate in the first half. The first half now contains exactly the winning 2048 candidates but is itself bitonic; recursively bitonic-merge only that first half with strides 1024, 512, ..., 1.

For K=2048 the derived comparator count is:

\[
2048 + \frac{2048}{2}\log_2(2048)
=
2048 + 1024\times11
=
\mathbf{13,312}
\]

compare-exchanges per row.

By comparison, a full custom bitonic sort of 4096 candidates would take

\[
\frac{\log_2 4096(\log_2 4096+1)}2 \times \frac{4096}{2}
=
78\times2048
=
\mathbf{159,744}
\]

compare-exchanges per row.

Those are network-operation counts, not claims about XLA's internal implementation of `lax.sort` or `lax.top_k`. JAX's generic stable sort supports multiple lexicographic keys if you need a correctness oracle, but a full sort unnecessarily solves a much harder problem than merging two already ordered lists. citeturn12search0

A compact implementation shape is:

```python
# K = 2048. score/position arrays have shape [rows, K].

def compare_exchange(score_a, pos_a, score_b, pos_b):
    a_wins = (
        (score_a > score_b)
        | ((score_a == score_b) & (pos_a < pos_b))
    )
    hi_s = jnp.where(a_wins, score_a, score_b)
    hi_p = jnp.where(a_wins, pos_a, pos_b)
    lo_s = jnp.where(a_wins, score_b, score_a)
    lo_p = jnp.where(a_wins, pos_b, pos_a)
    return hi_s, hi_p, lo_s, lo_p


def exact_merge_top2048(a_s, a_p, b_s, b_p):
    # Conceptual representation:
    # x = concat(A_desc, reverse(B_desc)) is bitonic.

    x_s = concat(a_s, reverse(b_s))
    x_p = concat(a_p, reverse(b_p))

    # Distance K: discard lower half.
    x_s, x_p = bitonic_compare_stage(
        x_s, x_p, distance=2048, keep_first_half=True
    )
    x_s, x_p = x_s[:, :2048], x_p[:, :2048]

    # Sort only the winning half.
    for distance in [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        x_s, x_p = bitonic_merge_stage_descending(
            x_s, x_p, distance, comparator=compare_exchange
        )

    return x_s, x_p
```

This is particularly attractive on a TPU vector core because it is twelve fixed, regular compare/permutation stages. TPU v4's vector unit handles the non-matmul operations surrounding its MXUs, while the architecture uses 128-wide matrix arrays for the heavy matrix work. citeturn11search1 A theoretically lower-work merge-path algorithm can approach linear comparison count, but it requires partition searches and more irregular indexed movement; it is the right second experiment only if the regular bitonic merge remains hot after profiling.

There is also a trivial optimization in the attached loop: **do not merge the first real tile with the initialized all-invalid carry**. fileciteturn0file0 Add a `have_candidates` scalar:

```python
new = stable_top_k(tile_scores, 2048)

carry = lax.cond(
    have_candidates,
    lambda: exact_merge_top2048(carry, new),
    lambda: new,
)
have_candidates |= tile_has_visible_keys
```

For a short context with only one or two real score tiles, this removes a surprisingly large fraction of selection work without changing any ordering.

### Use top-k as an associative distributed reduction

Under a strict total order,

\[
TopK(A\cup B)
=
TopK(TopK(A)\cup TopK(B)).
\]

Consequently exact top-K merge is associative. This lets the expert-8 global aggregation become a custom three-stage all-reduce rather than an `all_gather` followed by a replicated 8-way merge.

The candidate payload per expert rank at 32 rows is:

\[
32\times2048\times(4\text{ B score}+4\text{ B position})
=
524,288\text{ B}
=
\mathbf{512\ KiB}.
\]

The current all-gather materializes eight such lists, or **4 MiB/rank**, before the final global merge. fileciteturn0file0 At minimum, each rank must receive seven foreign 512-KiB lists to materialize that result, i.e. 3.5 MiB of candidate payload, independent of the physical all-gather algorithm.

Instead:

```python
scores = local_scores    # [32, 2048]
pos = local_positions

for xor_distance in (1, 2, 4):
    peer_scores = lax.ppermute(
        scores,
        "expert",
        perm=xor_permutation(8, xor_distance),
    )
    peer_pos = lax.ppermute(
        pos,
        "expert",
        perm=xor_permutation(8, xor_distance),
    )

    scores, pos = exact_merge_top2048(
        scores, pos, peer_scores, peer_pos
    )
```

After round 1 each rank represents two original ranks; after round 2, four; after round 4, all eight. `lax.ppermute` directly sends values according to a static source/destination permutation, and JAX's distributed-programming documentation shows it as the primitive from which communication-efficient collective patterns are built. citeturn9search0

Per rank, this changes the logical candidate communication from at least **3.5 MiB received plus a 4-MiB gathered temporary** to **three 512-KiB exchanges = 1.5 MiB sent and 1.5 MiB received**, with a constant 512-KiB current candidate state. It also reduces per-rank global merge work from processing all eight local lists to exactly three two-list merges. These payload numbers are derived from your fixed rows/K/dtypes; physical link traffic will depend on the TPU topology and collective routing. fileciteturn0file0

The tradeoff is synchronization: this is three collective rounds rather than one all-gather call. That is why **all-gather + local bitonic tree** should remain a benchmark control. If ICI latency dominates 512-KiB payloads, one all-gather may still win. If candidate movement and replicated merge dominate, the butterfly should win.

The comparison I would actually run is therefore:

| Exact approach | Raw 4096 selector | Carried merge | Global candidate communication | Key cost |
|---|---|---|---|---|
| Current | current local top-k | current generic merge | 8-way all-gather | baseline |
| Exact baseline | stable `lax.top_k` | stable lexicographic full sort | all-gather | correctness oracle |
| **Recommended** | **stable `lax.top_k`** | **13,312-CE bitonic half-merge** | **3-round `ppermute` butterfly** | regular and bounded |
| Secondary | stable `lax.top_k` | merge-path | butterfly | fewer comparisons, irregular movement |
| Reject | `approx_max_k` | any | any | not exact/stable |

The exactness oracle should explicitly include equal FP32 scores occurring **across different key tiles and across different expert ranks**. That is the case that distinguishes merely stable local top-k from the globally required explicit `(score, position)` comparator. JAX stable top-k alone only preserves the order of its own input array. citeturn10search6

## Large layer windows: roll the 32-row body, not the layer graph

The attached `ws32_prefill_window.py` currently does the right semantic thing for up to 128 rows: it processes at most 32 rows at a time, passes this layer's updated `cache_local` and unrepaired/repaired index-cache proposals to the next prefix tile, concatenates normalized MLP inputs, then runs one broader MoE suffix. fileciteturn0file2 That preserves the intended rule that later prompt tiles can see this layer's earlier proposed KV/unrepaired writes while repaired index state remains a separate proposal. Your project brief explicitly requires prompt attention to continue reading unrepaired index keys through prefill and to keep repaired writes separately staged until the appropriate boundary. fileciteturn0file3

The problem is purely **how the loop is expressed**. It currently uses a Python:

```python
for tile_start in range(0, rows, 32):
    ...
```

which JAX traces by unrolling. fileciteturn0file2 JAX's own documentation says native Python loops inside JIT lead to large XLA computations, whereas `lax.scan` is a primitive lowered to one `WhileOp`, specifically reducing compilation time and executable size. `scan` also has an explicit `unroll` parameter, and `unroll=1` keeps the body rolled. citeturn8search0turn8search1

That should become the production window scheduler.

### Use fixed 32-row tiles and fixed window buckets

I recommend two production buckets:

```text
W512  = 16 attention tiles × 32 rows
W1024 = 32 attention tiles × 32 rows
```

Do **not** compile special 1–31-row terminal bodies. Pad the bucket and carry a scalar `valid_rows`. Every iteration has exactly the same tensor shapes.

The schedule is:

```python
ATTN_TILE = 32
WINDOW = 512       # or 1024
TILES = WINDOW // ATTN_TILE

def scan_body(carry, tile_i):
    (
        cache_proposal,
        unrepaired_proposal,
        repaired_proposal,
        fleet_health,
    ) = carry

    tile_start = tile_i * ATTN_TILE

    hidden_tile = lax.dynamic_slice_in_dim(
        hidden_padded, tile_start, ATTN_TILE, axis=0
    )

    incoming_selected_pos = lax.dynamic_slice_in_dim(
        selected_positions_padded, tile_start, ATTN_TILE, axis=0
    )
    incoming_selected_count = lax.dynamic_slice_in_dim(
        selected_counts_padded, tile_start, ATTN_TILE, axis=0
    )

    tile_live = jnp.clip(
        valid_rows - tile_start,
        0,
        ATTN_TILE,
    )

    tile_offset = absolute_window_start + tile_start

    # Always execute the fixed-shape body.
    # Zero-live trailing tiles are masked; they do not advance state.
    prefix = ws32_prefill_transformer_layer_mapped(
        hidden_local=hidden_tile,
        cache_local=cache_proposal,

        # Later prompt tiles see these previous unrepaired writes.
        unrepaired_index_cache=unrepaired_proposal,

        # Carried as proposal, but not substituted for unrepaired
        # prompt-attention reads.
        repaired_index_cache=repaired_proposal,

        offset=tile_offset,
        count=tile_live,
        rows=ATTN_TILE,
        prefix_only=True,
        ...
    )

    next_health = fleet_health & prefix.contract_valid

    # IMPORTANT:
    # caches remain ONLY in the loop carry; do not emit them as scan ys.
    next_carry = (
        prefix.cache_local,
        prefix.unrepaired_index_cache,
        prefix.repaired_index_cache,
        next_health,
    )

    tile_outputs = (
        prefix.normalized_mlp_local,
        prefix.selected_positions,
        prefix.selected_scores,
        prefix.selected_counts,
        prefix.carried_residual,
        prefix.normalized_input,
        prefix.contract_valid,
    )

    return next_carry, tile_outputs


final_state, tiled_outputs = lax.scan(
    scan_body,
    initial_state,
    jnp.arange(TILES, dtype=jnp.int32),
    unroll=1,
)

# [TILES, 32, ...] -> [WINDOW, ...]
normalized_mlp = flatten_tile_rows(
    tiled_outputs.normalized_mlp_local
)

# One broad grouped MoE suffix for all live rows.
mlp_out = ws32_prefill_mlp_mapped(
    normalized_mlp,
    valid_rows=valid_rows,
    ...
)
```

`scan` requires the carry to keep fixed shapes and dtypes, which these cache proposal arrays do. citeturn8search0 The key memory rule is that **only the final cache values are loop carry**. They must not be members of `tile_outputs`, because `scan` stacks its `y` result over the iteration axis. Keeping caches out of `y` prevents a `[T, cache_shape]` result from ever existing.

This is fundamentally different from building a Python list of sixteen/thirty-two prefix result trees.

### Cache state should have one proposal generation, not one per tile

The first implementation does not need a new cache data structure. Carry the same three fixed-shape proposal arrays through the rolled loop:

```text
KV proposal
unrepaired-index proposal
repaired-index proposal
```

There is no need for 16/32 independently live cache generations. Inside one compiled computation, XLA can reuse storage for functional updates; JAX's buffer-donation documentation specifically notes that such reuse can happen automatically within a compiled computation, while donation is needed primarily to express reuse across a JIT call boundary. citeturn8search3

Because your atomic contract requires the original committed cache to remain valid until the fleet accepts the proposal, **do not donate the committed cache just to force aliasing**. A donated input becomes invalid to its caller, which conflicts with rollback semantics if the proposal later fails. citeturn8search3 It is acceptable to have:

```text
one committed generation
+
one current proposal generation
```

during the call. What is unacceptable is:

```text
committed
+
proposal after tile 0
+
proposal after tile 1
+
...
+
proposal after tile 31.
```

The compiled memory analysis and optimized HLO should tell you which case you obtained. JAX exposes compiler memory analysis including temporary, argument, output, and alias sizes. citeturn8search9

If even committed+one-full-proposal is too expensive, the second-stage design is a **bounded window journal**:

```text
base committed cache      immutable
window KV delta           only W new rows
window unrepaired delta   only W new rows
window repaired delta     only W new rows
frontier                  highest valid proposed row
```

For a selected position during a later tile:

```python
if position < window_start:
    value = read_committed_paged_cache(position)
elif position < current_frontier:
    value = read_window_delta(position - window_start)
else:
    masked_future
```

For index state, only the unrepaired delta participates in prompt attention. Repaired delta remains staged separately. At the fleet-success frontier, apply the bounded journal to the real cache once. This eliminates even the full-size cache proposal during the layer window, but it requires plumbing the overlay through your cache readers and atomic commit path, so I would only do it if the rolled-loop memory profile proves the simpler proposal carry insufficient.

Pallas itself supports explicit input/output aliasing for kernels when a bounded update helper is useful. citeturn8search11turn8search7

### Preserve causality by making frontier state explicit

At iteration `t`:

\[
offset_t = window\_start + 32t
\]

and

\[
live_t = \mathrm{clip}(valid\_rows-32t,0,32).
\]

The prefix body must expose only cache/index writes from iterations `<t` plus earlier positions in the current tile according to the existing causal mask. The attached implementation already advances its proposal arrays sequentially from one tile to the next and uses tile-specific offsets/counts. fileciteturn0file2 Rolling that exact body into `scan` changes graph representation, not the causal dependency.

Zero-live trailing iterations are important. Because every rank executes the same fixed 16 or 32 scan iterations, the collective schedule stays matched even when the prompt ends in an early tile. A zero-live body must therefore leave caches unchanged and produce masked/empty row outputs; it should **not early-break the device loop**.

The same principle applies to health failures. A local failure should set:

```python
fleet_health &= tile_health
```

but the program should continue through the statically matched loop rather than branching out of later collective-bearing layer bodies. Only after the layer/window has completed does the existing all-owner admission protocol decide whether the proposal crosses the state frontier. Your project brief explicitly defines the decoder's all-owner atomic commit as the sole mutable-state frontier. fileciteturn0file3

### The intentional window-sized buffers are modest and predictable

The local hidden shard is 6144/4 = 1536 channels in your feature-4 layout. fileciteturn0file3 Therefore retaining the normalized MLP input for the later broad MoE suffix costs, per chip:

| Window | BF16 `[W,1536]` | If retained as FP32 |
|---|---:|---:|
| W512 | 1.5 MiB | 3 MiB |
| W1024 | 3 MiB | 6 MiB |

The exact DSA final score+position output costs:

| Window | `[W,2048]` score F32 + position I32 |
|---|---:|
| W512 | **8 MiB** |
| W1024 | **16 MiB** |

Those are intentional HBM-resident outputs, not multiplied cache generations. The attached DSA's per-attention-tile logical scoring intermediate remains bounded by its existing 32-row × 4096-key design; rolling the layer loop means that scratch is reused sequentially instead of being represented as sixteen/thirty-two separately unrolled bodies. fileciteturn0file0

This directly addresses the graph-growth evidence already in your brief: the current complete B128 layer-6 graph was reported at roughly 203.7 MB temporary allocation and 49.4 MB code versus 96.1 MB temporary and 17.1 MB code for B32, with more static collectives as well. fileciteturn0file3 Extending the same Python construction from four prefix tiles to 16/32 is therefore the wrong direction. JAX's rolled `scan` is specifically intended to make the loop body representation independent of the static iteration count. citeturn8search0

## Decisive measurements and acceptance gates

The three questions can all be resolved with **small, attributable experiments**. None requires a 78-layer end-to-end run.

### Weight reuse experiment

Use one real MoE layer—layer 6 is a good choice because you already use it for structural tests—and save two natural routing/input captures, one B512 and one B1024. Every candidate kernel must consume the **same sorted routed rows, same `group_sizes`, same raw weights/scales, and same activation tensors**. The existing production numerical boundaries remain the reference. fileciteturn0file3turn0file1

For every capture, calculate before execution:

```python
def current_tm8_weight_visits(group_sizes):
    offsets = exclusive_cumsum(group_sizes)

    return sum(
        0 if s == 0 else
        ceil_div((o % 8) + s, 8)
        for o, s in zip(offsets, group_sizes)
    )

def expert_panel_visits(group_sizes, tm):
    return sum(
        ceil_div(s, tm)
        for s in group_sizes
    )
```

Do that globally and per expert-owner shard. This produces an exact **requested weight-load multiplier for the source schedules**.

Run:

```text
B512:
    current global TM8
    expert-aligned TM32 / TN512

B1024:
    current global TM8
    expert-aligned TM32 / TN512
    expert-aligned TM64 / TN512
```

Do at least one warm compiled invocation, then enough repeated accelerator-side executions that dispatch noise is negligible; Tokamax's own benchmarking guidance emphasizes accelerator-side repeated measurement because Python overhead can be much larger than kernel execution time. citeturn15search1 Capture an XProf trace around the repeated projection/MoE section. XProf's Trace Viewer gives individual operation durations and communication; its TPU overview reports memory-bandwidth utilization. citeturn17search0turn17search1 TPU v4 also supports TensorCore-utilization and memory-bandwidth-utilization metrics. citeturn16search4

The experiment is decisive under this interpretation:

| Observation | Conclusion |
|---|---|
| Visit count falls ~3×/5× and kernel time/memory pressure fall substantially | **Weight reload was real and material** |
| Visit count falls but time barely moves while TensorCore utilization rises sharply | Weight traffic was masked/cached; padding/compute now dominates |
| TM32 wins B512; TM64 wins B1024 | Adopt proposed bucket policy |
| TM32 beats TM64 at B1024 | Natural group tail or masked MXU cost outweighs extra weight loads |
| Exact output diverges before an existing cast/reduction boundary | Reject implementation regardless of speed |

For numerical validation, compare the current and proposed projection outputs **before every existing boundary**: gate FP32 partial, up FP32 partial, post-feature-reduction value, post-BF16 value, down output, restored route slots, FP32 route sum, FP32 expert combine, final BF16. The boundary contract is stated in the local project brief. fileciteturn0file3

The most informative single ratio to print beside every timing is:

\[
\frac{\sum_e V_e^{current}}
     {\sum_e \lceil s_e/TM\rceil}.
\]

If timing scales with that ratio, you have direct causal evidence rather than a generic "new kernel is faster" result.

### DSA selector experiment

Instrument the attached DSA around three scopes separately:

```text
score calculation
raw 4096 -> 2048 selector
candidate merge / global aggregation
```

The current file already cleanly separates score construction, local candidate selection, carried merge, and expert-axis all-gather/global merge. fileciteturn0file0

Use identical FP32 scores for these implementations:

```text
A. current merge
B. stable top_k + stable lexicographic full-sort merge
C. stable top_k + bitonic top-half merge
D. C + expert8 XOR butterfly
```

The correctness corpus needs ordinary real scores plus deliberate adversarial cases:

```text
equal scores inside one 4096 tile
equal scores across two key tiles
equal scores across two expert ranks
equal score exactly at rank 2048/2049
all masked except <2048 entries
more than 2048 equal live entries
```

Every output must equal the lexicographic oracle in **FP32 score bits and integer position**, not merely `allclose`. Stable JAX top-k and lexicographic stable sort provide the primary exactness primitives for that oracle. citeturn10search6turn12search0

For the communication comparison, report:

```text
all_gather:
    candidate bytes materialized per rank
    collective duration
    post-gather merge duration

butterfly:
    3 x ppermute durations
    3 x merge durations
    no 8-list temporary
```

The winner should be chosen on **end-to-end DSA time**, because the butterfly deliberately exchanges one collective synchronization for lower payload and lower local merge work.

### Window scheduler experiment

Before attempting W512/W1024, run the already-planned B128 numerical discriminator on the existing four-tile implementation; your brief notes that a complete B128 numerical/timing experiment had not yet been run. fileciteturn0file3 Then make a B128 `scan` version and require it to match the current Python-loop version.

Compile these three rolled variants:

```text
W128  = 4 × 32
W512  = 16 × 32
W1024 = 32 × 32
```

For each, record:

```python
lowered = f.lower(...)
compiled = lowered.compile()
stats = compiled.memory_analysis()

print(stats.temp_size_in_bytes)
print(stats.argument_size_in_bytes)
print(stats.output_size_in_bytes)
print(stats.alias_size_in_bytes)
```

JAX documents these compiled-memory statistics and their use for identifying real buffer cost. citeturn8search9 Inspect optimized HLO/XProf Graph Viewer and require a rolled loop rather than sixteen/thirty-two duplicated layer bodies; XProf's Graph Viewer exposes optimized HLO structure and execution counts. citeturn17search3

The production gate I would use is:

```text
Correctness:
    W128 scan == current W128 implementation
    W512 scan == sequential 4 × current W128 reference
    W1024 scan == sequential 8 × current W128 reference
    same KV/index cache proposals and health outcomes

Graph:
    one rolled scan/while body
    no per-tile cache outputs
    no 16x/32x duplicated prefix subgraphs

Memory:
    cache generations do not scale with T
    intentional [W,...] row outputs may scale linearly
    temporary scratch does not scale linearly with 16 -> 32 tiles

Failure:
    injected invalid tile changes final health
    no cache proposal is committed
    all ranks still execute the matched number of loop/collective steps

Causality:
    tile t can read earlier same-window unrepaired/KV writes
    tile t cannot read a later tile
    repaired index writes never replace unrepaired prompt reads
```

The especially useful compiler discriminator is **W512 versus W1024 code allocation**. With a properly rolled `scan`, doubling the trip count should not roughly double executable code because `scan` lowers to one WhileOp body; Python-loop unrolling would. citeturn8search0 Intentional HBM output size will double, so total memory cannot be expected to remain constant, but code size and per-tile scratch should remain close.

## Recommended implementation order

These three changes interact, but they should not be landed as one optimization patch.

**First, replace the grouped-MoE row schedule.** The source already identifies a real repeated-decode mechanism, your B128 active-tile result independently points at boundary visits, and B512/B1024 make the effect larger. Keep the first implementation deliberately conservative: expert-aligned TM32/TM64, TN512, K128 in ascending order, two HBM buffers, existing decode expression, existing output dtypes, and no activation fusion. fileciteturn0file1turn0file3 The upstream Tokamax kernel is useful corroboration for FP32 accumulation and grouped metadata, but its pinned heuristics explicitly target newer TPU generations rather than supplying a tuned v4 answer, so its default tiles should not be treated as v4 tuning results. fileciteturn2file0

**Second, replace only DSA merges.** Leave scoring and the exact stable raw 4096→2048 `top_k` in place. Add the first-tile bypass, exact `(score↓, position↑)` bitonic half-merge, then benchmark all-gather versus XOR butterfly. This keeps the change attributable and preserves the exact selector semantics documented by JAX/OpenXLA. citeturn10search6turn12search11

**Third, roll the layer window.** Replace the Python list-building loop with `lax.scan(unroll=1)` before increasing the production bucket to W512 or W1024. Carry only the current cache proposals and health, emit only row-sized outputs, and retain the existing all-owner commit frontier. fileciteturn0file2turn0file3 JAX's rolled-control-flow and buffer-reuse mechanisms directly address the compilation and retained-generation failure modes here. citeturn8search0turn8search3

The core architectural conclusion is therefore unusually decisive: **B512/B1024 should not be built by making the current eight-row grouped schedule run more efficiently. They should be built by changing the reuse unit from "global row tile" to "expert-aligned row panel."** The current numerical design—BF16 operands, FP32 K accumulation and cross-shard reductions, then BF16 only at explicitly chosen boundaries—can remain exactly intact while that schedule changes. fileciteturn0file1turn0file3