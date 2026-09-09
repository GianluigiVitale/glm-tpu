# Frozen prefill — actual layer0 norm boundary

2026-09-09. Completion-only diagnostic, not optimization or a new baseline.
Authority: goal.md / §25. DB603 remains the accepted engine. DB605 seals the
original dense01 reproduction; it does not fix the 8K generated-token11 mismatch.

## What the evidence establishes

DB605 reproduces both DB604 full layer0/1 cache branches on all32owners. In
the live rows, layer0 returned residual and input-normalized state match between
schedules; layer0 output differs1200 unique BF16words, first at position44.
Layer1 input-normalized differs202words. Neither observation identifies a defect.

The existing fused RMSNorm adds BF16 attention update and combined residual in
FP32. It uses that unrounded sum for normalization and returns a separately
BF16-rounded residual. Equal returned residual does not imply identical norm
inputs. Saved HLO also shows a BF16[4,32,1536] scan-output stack feeding both
M128 dense projections (one via a completed copy). It does not establish actual
normalizedMLP equality or explain row44 in wide versus row12 in narrow.

## Implemented CPU-only mechanism

`scripts/greenfield/ws32_dense_norm_boundary.py` isolates the new diagnostic.
The original dense builder and all production kernels remain byte-identical.
Only the fixed four-tile scheduling adapter is reproduced; all model arithmetic
calls existing kernels. No legacy import, host callback, new checkpoint, precision
change, row-size sweep, sampling or TPU initialization.

`build_capture_program` uses the original prompt/embedding/two layer trees/WKs,
keeps layer1 unchanged, and records layer0's executing hooks as scan auxiliary
outputs. Carries remain the original three caches. The suffix remains B128.
Owner axes [8,4] preserve physical observations, including nonreplicated partials.

| Field | Local shape/call | dtype |
|---|---|---|
| post_norm update, residual, normalized, carried | [128,1536] | BF16 |
| post_norm summed | [128,1536] | FP32 |
| local_square_sum, square_sum, inverse | [128,1] | FP32 |
| boundary normalized_mlp | [128,1536] | BF16 |
| boundary live | [128] | bool |
| post_norm weight, once/call | [1536] | BF16 |

Fixed device packet:2,757,248B/chip. Save only live rows on host, weight once/call.
Five calls add22,094,848rawB/rank; original model capsules81,933,312B become
104,028,160B before existing allowances, below128MiB. This is payload accounting,
not actual HBM admission. Keep packet capsules separate from original NPZs:
combining fields would exceed the existing reader's128-entry inventory guard.

`build_completed_dense_suffix` reuses the original dense MLP with completed
BF16 normalized inputs and physicalB128. Caller must first reproduce each
schedule's original output from its own captured inputs. Only then compare
identical captured inputs at the two fixed row placements. A separate executable
may perturb results; it cannot be assumed equivalent from HLO or CPU success.

## Exact relevance requirement

`ws32_dense_norm_originals.py` binds the reviewed DB605 original receipt, source
ledger, original runner and allfive NPZs by bytes/SHA/CRC. It reuses the bounded
NPY reader and compares encoded arrays without dtype coercion or tolerance.
Parent transport still authenticates original remote generations.

Require every RETAINED field on both layers and allfive calls. Endpoints retain
all12 fields/full cache pages; intermediate narrow32/64/96 retain9 row fields
and full128-row health, not caches. Never claim comparison of unsaved caches or
run extra hardware just to manufacture them. The existing64 DB604 endpoint
comparisons remain mandatory. If instrumentation changes any retained output,
preserve the packet and refuse attribution to the original realization.

## Decision from actual packet

1. Different update/residual/FP32sum: the difference reaches the norm input.
2. Equal sum but different square sums/inverse: normalization reduction boundary.
3. Equal norm operands/statistics but different normalized result: output boundary.
4. Equal actual masked normalizedMLP but different dense output: suffix realization
   or row placement requires the completed-input discriminator.

None alone establishes the8K token11 cause. Only a demonstrated correction followed
by this frozen path's own protected8K test can close that obligation.

## Verification and exact next integration

- Actual forcedCPU32 two-layer/five-call capture versus original builder:
  all outputs byte-equal, packet ownership/FP32sum/rounding/masks, poisoned padding,
  cache carry, invalid spans, completed suffix equality and M32 refusal PASS46.18s.
- Production abstract shapes/no payload reads/no device allocation and TPU-target
  raw lowering PASS48.78s. New capture raw663350B SHA
  `65379b5f5a87c2aef25687ce846e149749857a831924123a228ff267976f617c`;
  suffix raw28917B SHA
  `f9c1a7fa602a2c255fa85c0c5264b75e1241b0ab75aa3e434a7e27c432e7f73f`.
  These are not actual optimized TPU graphs or measured allocations.
- Original all8host/32owner/5capsule replay plus strict mutations:11PASS4.74s.
  First test caught an incorrect old launch_process_id field; actual DB605 uses
  launch_rank. Corrected locally without a TPU run or altered evidence.
- Independent final reviewer: noP0-P2;10negative tests1.41s. Supports CPU
  persistence only, never actual hardware admission or causal attribution.

Next adapt existing protected dense worker and original-array collector with a
distinct diagnostic identity, bounded packet publication, actual new graph/memory
admission and retained-byte reproduction before attribution. Reuse completed WK,
selected55-leaf loader, both leases, voted calls and 8host cleanup. Do not launch
the original nine-call campaign unchanged or run a new full-model/8K test yet.
No launcher integration, actual packet, cause, numerical promotion or speed claim
exists from this CPU-only implementation.
