# Frozen prefill: dense0/1 reproduction before attribution

Status 2026-09-09 14:15Z: CPU mechanism, bound runtime/continuation and per-host preflight passed;
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

- `ws32_dense_frontier_runtime.py` binds an EXISTING initialized runtime to
  original DB604 host/process/mesh/topology and actual owners, rechecks selected
  headers/checkpoint/source, validates prompt/RoPE BEFORE selected payload load,
  then reuses the selected loader and original sharded placement. No CLI/init.
- `ws32_dense_frontier_execution.py` preserves allthree compiler originals before
  admission, then SAME BudgetedCalls for four per-layer WK and five modelcalls.
  WK captures bind source operands/shardindices/process/platform, preserve invalid
  bytes before refusal and cap four capsules96MiB/rank. This is ADDITIONAL to the
  five-model128MiB cap, not a whole-archive allowance. Default launch stays off;
  actual-HLO inspector, independent collector and protected entry are pending.
  DB604 omits DSA oracle fields; original8K pins come from SHA83efb10c refusal
  runner, not invented DB604 metadata. Actual oracle/RoPE test validates this.

- `ws32_dense_frontier_protocol.py` / `ws32_dense_frontier_preflight.py` now
  connect the existing campaign's retained-preflight to a distinct dense01 tag.
  Per host: original DB604 runner, four endpoint files and fixed source ledger,
  14.459–35.267MB, not the complete331MB archive on every host. Exact generation,
  size/CRC/SHA, US-CENTRAL2,128MiB reference cap and1GiB disk reserve; existing
  mismatching files refuse instead of overwrite. Actual owner sets, no rank*4.
  Original selected metadata/header checks bind55leaves/layers0/1+embedding,
  checkpoint/source/topology/file ledgers,8192capacity and hostrope. This is
  NOT actual runtime ownership or selected payload verification. Campaign launch
  explicitly refuses before SSH until worker/admission/collector are integrated.

- `scripts/greenfield/ws32_dense_frontier_worker.py`: actual BudgetedCalls
  continuation requires four prior per-layer WK calls, then fixed five model
  calls. Source prompt SHA, independent zero cache allocations, live-memory
  budget with1GiB reserve and voted failure handling. Writes both branch
  endpoints before reproduction refusal. `complete` means captured evidence,
  NOT reproduced bytes or model correctness. Runtime launch routing is unwired.
- `scripts/greenfield/ws32_dense_frontier_capture.py`: exact production shapes,
  dtype, device/process/slot and shard indices before addressable-only reads.
  Retains all live row outputs and full128-row health, full endpoint cache bits;
  skips intermediate cache downloads. Health/nonfinite errors are returned for
  saving before refusal.128MiB is the FIVE MODEL CAPSULES budget, not a complete
  archive budget: outer publication must also budget four WK captures, compiler
  originals, journal/logs and wrapper copies before deployment.

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

- Bound runtime/execution + existing worker/capture:61PASS8.57s. Actual saved
  topology/DB604 owner mapping and complete oracle/RoPE, compiler writer/journal/
  NPZ/fleet votes; device compiler/math/memory/load are explicit fixtures.
  Independent execution21PASS4.62s/runtime17PASS3.71s, noP0-P2. No new TPU result.

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
- Continuation lifecycle8PASS1.99s, production-shape capture15PASS2.07s,
  actual32-device CPU allocation/placement/zero/alias/padding1PASS1.98s.
  Fixture worker compute/memory are explicitly not hardware measurements.
  Reviewer independently ran continuation+capture23PASS2.72s, noP0-P2.
- Allthree abstract compiler jobs:1PASS45.65s, dense raw above unchanged;
  WK decode10725B/SHA8eeefbb0cbc3518ac223b49e1bade70dc1aa78c965c983ebef83c0140284c362,
  WK promote802B/SHA7b277bb821af372bd03687010b1db3630533dc9db46747863dd08cc0742006e5.
  Reuse existing completed-BF16 decode/promote builder. Three compiled programs,
  NINE executed calls once wired: decode/promote for each of two layers +five
  model calls. No shared layer0 WK substitution. Independent helper review noP0-P2.
- Actual controller retained selected payload:220 tensors/410359040B from four
  owners9/13/25/29,42.343914s; each selected tensor header/SHA/finiteness passes.
  Ledger86cce759f824b6b8485f3d04577c1e1fba1357dfc3082552712ac75ee9fd874d.
  Receipt `docs/artifacts/prefill-dense01-controller-selected-bytes-20260909.json`.
  No TPU initialization or complete-checkpoint claim. Other seven hosts still
  require preflight/selected-byte verification before the diagnostic.
- Per-host preflight and original witness:32PASS198.66s/no skips, including
  real all8 original joins and actual metadata; transport/runtime absence tested
  with explicit fixtures, not real cloud downloads or TPU. Independent Astra
  16PASS149.14s/1deselected/noP0-P2. Default witness still requires all32owners;
  optional launcher rank checks exactlyfour matched owners across both branches.

## Exact next action

Retained-preflight and bound continuation are ready; runtime launch remains disabled. Wire ONE selected dense0/1 diagnostic through existing protected selected-layer
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
Fixed HLO inspector, protected parent routing/collector, actual HLO/HBM and byte reproduction are
still missing; this document grants no hardware promotion from CPU tests.
