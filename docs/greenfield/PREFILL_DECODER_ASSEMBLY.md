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

## Protected acquisition wiring — 2026-09-08

Starting pin61860d94. `GLM_GREENFIELD_WS32_PREFILL_MODE=layer_major_raw_v1`
selects separate short-context ACQUISITION ONLY, with exact decode/host main
RoPE required, B17 default/B1..32 permitted, `_bp1` tag suffix. Numerical runs,
inherited serial adjudication and all long-context runs are refused for this
mode until its actual compiler/allocation profiles and short proof exist.
New serial long-context launches are also refused (§24); recovery of existing
serial evidence is preserved. Serial mode remains the historical default.

Worker binds raw-prefill views BEFORE discarding loaded arrays. Promoted decode
keeps its original view. Both share original objects; no full checkpoint or
weight copy. Completed producer wk arrays are reused. The two compiled programs
use six inputs (IDs, scalar live count, typed state, raw weights, completed wk,
host RoPE), with no old serial donation indices. Acquisition state placeholders
share existing initial cache buffers. Both actual prefill graphs and the five
existing materializer/decoder/observer/cache graphs are preserved before final
structural authorization. No model-prefill graph executes in this acquisition.

`benchmarking/ws32_batched_prefill.py` currently collects exact payload families,
shapes, helper signatures, live layers and non-ADD reducers from the actual graph.
It ALWAYS reports UNREGISTERED and passedFalse: it is NOT the completed linter.
Matching hashes/labels/local groups cannot approve it. Expected refusal writes
the complete per-host runner envelope with HLO_REFUSED before raising, so the
existing upload trap retains HLO, memory/inventory/provenance and logs remotely.
No HLO_ACQUIRED/SUCCESS/DB performance row is manufactured from that refusal.

Independent review's source-derived inventory for later registration:

- Common attention/norm branch78×; full DSA21×; dense3×; MoE75×; one embedding
  and one conditional final one-row head. Do not multiply whole layer0/3 profiles:
  eighteen full-DSA+MoE layers combine independent branches.
- Raw FP8 semantic callsites588, grouped225, structured156, sparse78:1047 total.
  Acquisition must determine actual textual multiplicity/CSE/shared callees.
- Count tuple reduction operand leaves, not guessed psum counts. Exact physical
  feature4/expert8 groups/global IDs and ADD reducers for all model sums.
- Two S32 scalar MIN health reductions require feature→expert→actual commit
  predicate lineage; names alone do not authorize MIN anywhere else.
- Adapt actual FP32 route-sum SSA/fusion proof to every75 MoE layer and actualB;
  handle tuple leaf forwarding without treating an unrelated attention sum as
  route evidence. No BF16 intermediate/original_type correction is allowed there.
- Acquire exact compiler-helper operands/counts, bias lowering, conditional-head
  realization and cache aliases. Do not reuse single-layer capacity1024/B17
  helpers blindly at8192 or narrow tail. Full cache arrays legitimately exceed
  the old layer-only element ceiling; distinguish them from weight expansion by
  shape AND lineage. Intentional21 repair hidden gathers need their exact consumer.

The host execution adapter is tested but not yet enabled in the numerical worker.
It requires two all-host AND decisions per block: structural health, then budget/
logging continuation. Every rank participates even on final block; a peer-only
refusal prevents the next dispatch. Logging errors vote false before raising.
Actual runtime wiring must supply genuine multihost consensus (existing protected
layer worker uses process_allgather of a compact integer), not identity/host-local
bool. These checks are outside layer execution and inside reported request wall.
This accounting ends at token readiness, explicitly NOT actual-delivery TTFT.

Next: review the concrete acquisition cost/preflight and compile actual main/tail
graphs once; use its preserved per-host evidence to implement the strict linter
and allocation admission. Then wire the numerical adapter/sealer and candidate's
own §21 observation/adjudication. No representative layer arithmetic rerun.

Acquisition launch capsule: `configs/greenfield-ws32-batched-acquisition.json`.
Existing retained checkpoint/overlay metadata hashes rechecked locally. Cold
planning estimate15–25min is not a guarantee; the reviewed worker hard ceiling
is45min (serial default remains4h). Upload/cleanup are outside that worker timer.
The wrapper's batched census now supplements its ordinary process/container
check with `fp8_baseline_guard` root device/inode/libtpu/PID checks, even if the
ordinary census fails. Root output is embedded in the archived ordinary census
file, including expected-refusal exit. Caller holds user rsync lease; wrapper
holds workload lease. No model numerical execution and no automatic retry.
