# Second prefill research report — checked against the executing engine

2026-09-08. Main agent read `research2-prefill.md` from first line to EOF:
858 logical lines (857 newline characters, no final newline),47,947 bytes,
SHA256 `c4c873346d2afe7959a2f4ef783d42e1cd871fc3ef93aa2f10351b7d8d085acb`.
The owner's report is preserved unchanged. Baseline `ed0a3e05`; JAX0.10.1,
libtpu0.0.41, existing32-chip v4. No model, kernel, environment or TPU change
was made while evaluating it. Embedded citation handles are leads, not locally
verified evidence; primary pages and installed APIs checked below are distinct.

## Decision

This report usefully converts three performance questions into implementable
hypotheses. Keep all three, but **do not apply them together or before diagnosing
the current failure**. The report still describes B128 as unexecuted; it has now
failed on DSA order and five changed router sets. See
[current evidence and next capture](PREFILL_WINDOW_BOUNDARY_DIAGNOSIS.md).
Nothing here relaxes that comparator or the §21 numerical contract.

## 1. Expert-aligned row panels: promising, not a drop-in kernel

Confirmed in `kernels/pallas/prefill_grouped_fp8.py`: the grid is
`(tiles_n, active_tiles, tiles_k)`; rawFP8 decode/scale/BF16 conversion is inside
each invocation, and metadata may revisit one expert across global row tiles.
An expert-aligned panel schedule removes boundary-driven revisits. Actual HBM
transactions and end-to-end gains remain unmeasured; repeated source accesses
are not a physical-bandwidth measurement.

The report's TM32/TM64,N512,K-inner panel is worth a selected-projection test.
Do not conflate three changes: larger row tile, expert-aligned starts, and wider
resident N/K panels. Retain current globalTM8 AND globalTM32 controls where
admitted so timing can distinguish alignment from simple row-tile enlargement.
Moving K inside a program does not by itself remove repeated expert-row panels.

Missing implementation obligations:

- Expert group starts are arbitrary, not necessarily multiples of TM. Ordinary
  blocked indices multiply by block size. A valid unaligned gather/staging design
  is needed; the report's conceptual `BlockSpec` is not executable as written.
- A partial expert panel must not overwrite the next expert's rows. Zero-masking
  values in an overlapping output block is not the same as suppressing writes.
  Account for any scratch/padded route layout and exact restoration to route slots.
- Include raw weight/input/output double buffers, FP32 accumulator, decoded
  weight/scales, alignment and compiler scratch in measured VMEM/HBM admission.
  The proposed2–3MiB arithmetic is a budget sketch, not measured allocation.
- AscendingK128 and unchanged casts do not guarantee bitwise identity when M/N
  geometry or compiler fusion changes. Preserve existing per-boundary contracts;
  use actual operands and outputs, not a numerical guarantee from pseudocode.
- Feature4 FP32 reduction remains BEFORE BF16/SwiGLU; no new packed checkpoint
  or full BF16 weight expansion. Natural-route claims require actual model
  activations, not merely real weights and synthetic inputs.

Primary check: [Pallas grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
documents block-index scaling, shape restrictions and overlapping writes. These
constraints explain the missing address/store design; they do not reject the idea.

## 2. Exact bitonic merge: isolate it before changing communication

The sorted-list half-merge is mathematically plausible. A small CPU NumPy
implementation matched a lexicographic oracle on200 rows acrossK1/2/8/32/2048,
random/tied/all-tied/empty/partially invalid inputs. This is a mechanism sanity
check, NOT a JAX implementation, exhaustive proof, TPU result or latency forecast.
The13,312 compare-exchanges forK2048 do not describe the current compiler's sort
implementation or establish a speedup over it.

Important installed-version correction: JAX0.10.1 reports
`lax.top_k(operand, k, *, axis=-1)`; passing `is_stable=True` raises TypeError.
Its equal-score CPU example returns lower indices first. The
[current online API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html)
does expose `is_stable`; this is a version mismatch, not a reason to upgrade the
running environment. Inspect/test the pinned implementation before using it.

Keep scoring unchanged. Test exact local merge first, including score bits,
absolute-position ties across tiles/owners, invalid sentinels, fewer thanK live
values, and finite-input health. Do not add epsilon score perturbations or
approximate top-k. Lower input-index ties equal lower global-position ties only
when that input ordering is proved; global merges need explicit position keys.

The report's4096→2048 initial tile is not universal: the failed layer6 protocol
uses **key_tile512**, requiring padding/count handling forK2048. Preserve actual
registered geometry. First-visible-tile bypass must also preserve empty owners
and per-row causal lengths, not assume every query has a visible tile.

Then compare unchanged all-gather plus exact merge with the proposed three-round
expert8 XOR exchange. Retain the all-gather control: smaller candidate payload
adds sequential synchronization. Score and position exchanges may lower to two
collectives per round; do not advertise three physical operations from three
logical rounds. Bind physical partner groups, replicated final selections and
disjoint candidate ownership. Choose on complete DSA wall, not comparator counts.

## 3. Rolled prefix loop: useful direction with unchanged state semantics

`ws32_prefill_window.py` currently traces four32-row prefixes in a Python loop.
Use a rolled prefix body as the later B512/B1024 challenger; do NOT roll the
entire batch-one decoder over prompt tokens. A layer's token-parallel prefix
tiles followed by one broad MLP are a different execution model from that old
serial bridge. [JAX scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
documents fixed-shape carry and single-WhileOp lowering, supporting the code-size
argument but not proving buffer reuse or numerical identity.

Carry KV, unrepaired index and repaired index proposals; stack row observations,
not all cache generations. Keep all-owner final commit/rollback and no donation
of still-needed committed state. Inspect actual allocation/aliasing and runtime
peaks: one WhileOp alone cannot prove a single physical proposal allocation.

Correct the sketch before use: preserve capacity-safe offsets even for zero-live
trailing tiles, per-row live/health semantics, original row order, M64 repair and
separate repaired visibility. Never let a local health branch skip a collective
that peers execute. The blanket ban on narrow tails is not adopted: fixed padded
buckets trade compile count against dead work and must be measured. Final-small
tails and bounded buckets are choices, not universal correctness rules.

## Order and independent review

1. Finish the current actual DSA/MLP boundary capture and specific numerical fix
   or valid adjudication. Do not replace the failed comparison with a kernel win.
2. Use preserved original routes for bounded mechanism tests; collect natural
   route statistics separately when the actual model path can provide them.
3. Evaluate expert-aligned panels, local exact merges and rolled windows as
   separate default-off changes, each with its smallest informative test.
4. Choose integration order from measured costs, preregister final128K/256K
   prefill/TTFT targets, then own competitive8K and all new L7/L8 proofs.

Independent existing Astra reviewer agrees with these caveats and next order.
No broad new correctness-proof project, environment upgrade, hardware operation,
checkpoint pack or permission to skip the current failure follows from the report.
