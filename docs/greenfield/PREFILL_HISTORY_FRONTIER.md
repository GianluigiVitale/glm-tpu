# Bounded full-history frontier — staged, not hardware-admitted

Purpose: localize the corrected batched path's own8K token failure without
another full78-layer retry. Original evidence and limitations are in
[PREFILL_CANONICAL_8K_FAILURE.md](PREFILL_CANONICAL_8K_FAILURE.md).
No model arithmetic, checkpoint, numerical bound or performance target changes.

## Implemented core

`scripts/greenfield/ws32_history_frontier.py` calls the existing embedding and
actual layer-window kernels for layers0..6. It returns completed update and
carried residual SEPARATELY, each layer's normalized input, ordered routes,
weights, owner health and producer0/1/2/6 selected positions/counts/scores.
It carries seven KV owners and four unrepaired/repaired index owners; shared
layers3/4/5 neither acquire their own index slot nor reuse another layer's KV.
Unhealthy or invalid-capacity proposals retain prior caches and return false.
This stateless core does not enforce monotonic host offsets or complete prompt
coverage; the eventual guarded executor must enforce both and chain health.

Four explicit programs retain physicalB128/B114. The candidate sets
canonical_dense=True only for dense0..2. The control sets it FALSE, matching
the retained original live32 run, not a newly corrected control. No head,
generation, full-state load, CLI, deployment or production default is added.

`ws32_history_frontier_prepare.py` reuses the existing current-source guard,
authenticated metadata/schema and selected-layer name inventory. It returns
only ShapeDtypeStruct inputs; it reads no weight payload and places no arrays.
It retains original78-layer geometry while selecting exactly201 raw leaves:
embedding plus seven COMPLETE layer trees. Existing selected loader will verify
those actual bytes; metadata alone does not establish payload integrity.

Persistent raw operand accounting from all32 manifest owners:

| Operand | Bytes/chip |
|---|---:|
| Selected raw weights | 1,424,692,176 |
| Three cache families, ONE branch | 11,272,192 |
| Four completed FP32 WK matrices | 12,582,912 |

Weights/WK/RoPE are shared across branches; caches are independent. Selected
weight reads total45,590,149,632B over32chips, from existing retained owner files,
NOT a new persistent artifact or checkpoint copy. These are NOT peak HBM or a
complete experiment budget: exact-decode operands/overlay, programs, scratch,
simultaneous captures, host originals and archive still need explicit accounting.

## Required continuation (not yet implemented/admitted)

1. Build the bounded first-decode observer from the existing
   `ws32_transformer_layer_mapped`, exact-DSA materializer and verified original
   StrategyND overlay for dense0..2. Raw-prefill weights/WK alone are NOT the
   original observer. Do not fabricate head weights or sample with a reduced head.
2. Preserve original8155 IDs and both schedules: candidate63×128+91live tail;
   control254×32+27live tail. Both tails physical114. Interleave one wide block
   and corresponding narrow blocks, retaining separate branch caches and only
   bounded first-difference operands/locations plus compact comparison records.
3. Continue BOTH complete histories even after finding a stream difference.
   Install repaired keys only after full history; observe original token220 at
   position8155/contextlength8156. Require EACH branch's saved step0 event0..3
   positions/counts/scores to reproduce before any original-cause attribution.
   CPU equality or a first128-only match cannot establish that reproduction.
4. Reuse existing BudgetedCalls, compiler journal, actual HLO/memory guards,
   protected fleet/collector/DB/archive/cleanup with a DISTINCT bounded protocol.
   Neither the old two-layer profile nor these functions authorize execution.
   Bind exact original inputs, source pins and all32 physical owners; register
   selected/overlay bytes, simultaneous HBM/host peak and output/archive caps.
5. Inspect saved actual graphs before numerical dispatch; preserve originals
   before refusals. No new generic symbolic/precision proof or blind MoE fix.

## Local evidence and review

Initial actualCPU32 B128+114tail cache/lifecycle/padding/refusal comparison:
1PASS161.16s. Production metadata-only four-program abstract schema test:
1PASS50.95s; explicit payload-open and device_put traps remain enabled.
Neither executes TPU or proves original numerical reproduction.

Independent gpt-6-astra core/preparation review found no implementation P0–P2,
but identified missing coverage of exported observations and False/live32.
The expanded test compares all exported boundaries/producers to instrumented
ORIGINAL eight-layer runtime returns, for both branches: **2PASS318.12s**.
Candidate128+19 and control32+19 use physical128/114, including poison padding,
cache/repair lifecycle and incoming-unhealthy/out-of-capacity rollback. The
reviewer cleared this coverage/source delta for persistence after that pass;
no P0–P2 remains. Reuse-registry tests4PASS1.95s; git diff --check passes.
Observer and guarded integration above remain pending. No TPU was launched.
