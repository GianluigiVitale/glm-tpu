# Layer-major prefill decoder assembly

2026-09-08. §24 default-off implementation; no new TPU/full-model/performance result.
Starting pin `a6a3bba9e23622969fd6ded488f542904b273a91`. Reuses real-layer
admissions DB583/584/587; original failed reference realizations remain failed.

## Execution and state

`glm_tpu/greenfield/runtime/ws32_batched_prefill.py` is a separate program from the
promoted one-row decoder. One invocation embeds B live prompt rows together, visits
each transformer layer once over those rows, and appends its proposed cache writes.
Python enumerates layers during tracing only; there is no host layer dispatch or
scan of the complete decoder per token. Existing per-layer tiled/row-local kernel
loops retain their meanings; this claim does not say all inner work is parallel.

The initial B1..32 interface matches admitted layer kernels. It is a correctness
integration window, NOT a claim that B32 is sufficient for interactive throughput.
The routing occupancy model in `PREFILL_COST_MODEL.md` still motivates larger
routing/linear windows with smaller attention tiles before performance promotion.
No new long-context performance run is authorized by this CPU assembly.

The immutable configuration explicitly requires raw-layout weights, page512 and
the accepted host main-attention rotary table. Exact-convolution aliases and
StrategyND dense overlay configs are refused, not silently ignored. A future
runner must bind a raw view of the same retained weight arrays and separately
bind the promoted decoder's exact/overlay view; it must not create a new pack.
Repair receives the existing completed, checkpoint-authenticated FP32 wk tensors
for each full-indexer owner. The kernel does not rematerialize wk per block.

`Ws32BatchedPrefillState` carries existing decoder state plus the separate repaired
index buffer, immutable request prompt length, and a finished bit. Fresh allocation
creates zero caches/frontier0/context-length1. A nonzero resume must authenticate
the whole state, including page tables and both index buffers; merely asserting
a nonzero frontier does not prove a populated prefix. There is no resume loader
or persistent-state authenticity claim in this module.

Key invariants:

- Append offset is taken only from carried decoder.position. Context length must
  equal offset+1; count is positive, at most B, and cannot exceed prompt length or
  consume reserved decode capacity. Only live count advances the frontier.
- Explicit full-indexer slot map is0,1,2,6,..., not layer%4. KV indexes all layers;
  index/repair buffers index only full producers. Every shared layer uses its own
  KV and the current producer's compact B×topK selected state.
- Hidden update and carried residual remain separate through both norms. Repair
  is computed inside each producer from that producer's own normalized inputs.
- All prompt chunks score unrepaired keys. Repaired keys become active only in
  the final successful commit. Finished states refuse further prefill calls.
- Only the final block runs the one-row final norm/head on its last LIVE row.
  Intermediate chunks return token−1. No per-prompt-row vocabulary projection.
- Health accumulates every layer/live row and the head, then two explicit scalar
  subgroup reductions (feature4, expert8) produce all-owner agreement once per
  block. Head control flow depends on replicated schedule metadata, not possibly
  different owner health. No repeated32-chip model collective is introduced.
- A failed proposal returns unchanged KV/index/repair/frontier/page tables and
  latches false health everywhere. No cache donation is used yet. Input/output
  aliasing and actual compiled peak HBM remain mandatory hardware admission work.
- `finish_ws32_batched_prefill` requires complete healthy state before releasing
  its one-row decoder state and first greedy token. It synchronizes at the final
  serving boundary, not between layers or prompt tokens.

## Smallest CPU checks

The current test uses actual eight-layer weight pytrees and actual layer kernels
on forced32 CPU, with reduced hidden/expert geometry and distinct layer weights.
It independently wires completed layer calls and compares the composition outputs.
This is a state/ownership composition oracle, not independent model arithmetic.
Repair wk values and prefix activations are synthetic; this cannot certify real
checkpoint resume, real-model numerical exactness, or TPU rounding behavior.

Coverage: layer2→3 dense-to-MoE/IndexShare, layer6→7 full-DSA+MoE producer/share,
two chunks offset505/B17 then11-live tail, nonidentity page table, producer
replacement, separate KV/indices, last-live sampling, padded invalid token IDs and
NaN rotary values, premature/repeated finalization, invalid counts/frontiers/page
aliases, repaired-history independence, final-block single-owner health failure,
and an actual raw one-row decoder step using the final repaired state/token.

Independent reviewer identified invalid initial fixture topK16/segment8 against
the production decoder's segment128 constraint. Corrected fixture topK128 and
segment128; no production guard changed. Tests and exact outcome: latest HANDOFF.

## Next protected evidence

Trace the production78-layer input/weight/state schema without allocating weights.
Wire the existing protected short-decoder campaign to this separate graph pair,
using narrow static tails and the existing completed repair weights/host RoPE.
Inspect actual acquired HLO groups, useful rows, cache aliases and per-chip memory
before numerical execution. Resolve major avoidable allocations before proceeding.
The short decoder must earn its OWN §21 raw-token, cutoff-active own-score DSA,
first-divergent-event adjudication and cache/state protection; DB587 cannot stand
in for that proof. Preregister quantitative prefill/TTFT criteria before candidate
performance trials. Efficient four-depth L7 and full L8 remain open.
