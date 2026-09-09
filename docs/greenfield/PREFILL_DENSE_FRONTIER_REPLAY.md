# Frozen prefill: dense0/1 reproduction before attribution

Status 2026-09-09 13:12Z: CPU mechanism and production abstract preparation passed;
no new TPU execution, root cause, numerical promotion or speed result.
Authority remains goal.md / specification §25. DB603 is frozen.

## Evidence and question

DB604's original five-call first128 diagnostic compares one128-live call with
four32-live calls to the SAME physical B128 executable. All32 owners establish
equal layer0 caches and first differing writer1. Unique expert counts at layer1:
KV271, unrepaired index12, repaired index11 BF16 words. First KV difference is
position44/component21. This does not identify the erroneous operation, nor
prove the cause of the 8K generated-token11 failure. Both layers are dense/full
indexer, before MoE and IndexShare. No MoE/panel or throughput experiment follows.

Original receipt: `docs/artifacts/prefill-frozen-first128-diagnostic-20260909.json`.
Archive ledger SHA: `4c6d4cab7f26281bbdac69a85a296d1298aa5f08709d516aac1f68583242653e`.
Originals remain under `/home/gianl/glm-run/` plus that receipt's tag and
`first_window_collected/`. No new archive copy or new correctness oracle.

## Implemented bounded replay

- `scripts/greenfield/ws32_dense_frontier_program.py`: original embedding and
  original rolled layer kernel twice, physicalB128 on both schedules, key512 and
  frozen options. Carry both layers' three caches across narrow calls, reset
  per-call selection exactly as original runtime, carry layer0 selection into1.
  Outputs are diagnostic proposals, not an atomically committed serving state.
  No final head, generation or repaired-index promotion.
- `checkpoint/ws32_layer_subset.py`: default-off `include_embedding` adds one
  complete original tensor to the existing selected-layer reader. Same byte
  budget, original ledger indices, header/SHA/finiteness and addressable-owner
  checks. No row-only hashing, full-model load or new checkpoint.
- `scripts/greenfield/ws32_dense_frontier_prepare.py`: authenticate actual
  production metadata; bind only55 leaves (27 each layer plus embedding).
  Payload102,589,760B/chip,3,282,872,320B across32 chips. This is selected existing
  weight payload, NOT extra storage or peak HBM. WK/cache/scratch/code/retained
  outputs must additionally enter actual execution admission.
- `scripts/greenfield/ws32_dense_frontier_witness.py`: fixed DB604 source ledger
  and original JSON/NPZ hashes, full original capsule replay before selecting
  layers0/1. Compare EVERY byte in both layers/allthree families/allpages for
  each branch/owner, including untouched zeros and signed-zero bits. A single
  differing byte rejects that realization for attribution. Never compare only
  the difference count or first-writer label.

## Tests and review

- Initial dense program + subset suite:30PASS85.52s. CPU32 actual eight-layer
  reference vs reduced firsttwo caches; narrow carry, padding poison and invalid
  spans. CPU equality is not TPU compiler-realization equivalence.
- Witness + extended subset suite:45PASS25.28s. All64 branch/owner originals,
  source binding, layer0/1 and outside-frontier mutations, wrong branch/shape;
  embedding placement verified on reversed32-device CPU mesh.
- Production metadata + TPU-target raw lowering:1PASS46.39s, no payload reads
  or device_put allowed.55leaves, original78-layer configuration, raw653497B,
  SHA `4f2ec6ed56ac1485bfda976882c74211e168a667018b6caf06be0ae5f4eccb60`.
  This is CPU-side TPU-target lowering, not optimized TPU HLO or measured HBM.
  First attempt failed an erroneous test expectation61leaves; corrected from
  source to27+27+1=55. No production schema was changed to satisfy the test.
- Independent Astra current-delta review: no P0-P2 in program/loader, witness,
  and preparation. Reviewer independently ran witness15PASS7.01s and embedding
  selection1PASS2.38s. Hardware admission remains explicitly separate.

## Exact next action

Wire ONE selected dense0/1 diagnostic through existing protected selected-layer
worker/campaign/leases/journal/publication/collector, reusing BudgetedCalls and
actual live-memory admission. Authenticate original prompt IDs, checkpoint and
physical owners before load. Compile the reduced graph and two existing WK
materializers, preserve actual HLO/compiled memory before any inspection refusal;
admit actual local groups/helper interfaces and per-chip budget. No full checkpoint
or another full-model acquisition is needed for this question.

Execute fixed one128 + four32 calls with the original first128 IDs, original
zero independent cache states/table/RoPE/WK and frozen numerics. Preserve output
bytes before refusal; collect both branches/all32 owners. The existing witness
consumer must reproduce each branch's layers0 AND1, allthree whole-cache hashes.
Only then inspect layer0 update/residual and layer1 normalized input, and use
existing completed-prefix/dense-suffix tools for identical-input attribution.
Exposing more outputs or reducing the graph may perturb fusion: if reproduction
fails, mark it unsuitable; do not fit model arithmetic or declare DB604 wrong.

Do not make another full8K attempt without a demonstrated cause/fix. No row
sweep, precision archaeology, threshold relaxation or slower replacement baseline.
New worker/collector wiring, actual HLO/HBM and protected byte reproduction are
still missing; this document grants no hardware promotion from CPU tests.
