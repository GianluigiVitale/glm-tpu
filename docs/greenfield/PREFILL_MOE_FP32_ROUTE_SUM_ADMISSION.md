# FP32 route-sum candidate — numerical preregistration

2026-09-07. Applies only after independent review and persistence, before executing the
new candidate. No performance trial, full decoder promotion or historical relabeling.

## Decision and evidence

DB582 is a boundary diagnostic, not arithmetic acceptance. The original exact B17 test at
31a2f917 remains FAILED. Its candidate final output is reproduced on all32 chips by DB582;
the instrumented reference changes16/32 final witnesses, so DB582 cannot establish the
original graph's exact first divergent boundary.

On the diagnostic's actual operands, all routed/shared FP32 projection partials match.
Shared post-feature and output arrays match. Sixty-four routed post-feature BF16 values
differ; their four-feature FP64 sums round differently from candidate/reference at11/17
of4,456,448 unique entries respectively. Do not chase the old collective strategy.

CPU replay over835,584 owner-local output values demonstrates a specific arithmetic gap:
candidate expert-psum operands equal the sum rounded once to BF16 exactly, whereas reference
operands equal the FP64 sum rounded to FP32 exactly. Candidate differs from that FP32 sum
in112,345 values. Both expert combines reproduce FP64 sum→BF16 on their own actual inputs
with zero differences. This is measured from tensors, not inferred from HLO type labels.
Record: `../artifacts/prefill-real-moe-boundary-replay-v2-20260907.json`. V2 additionally
binds loaded workers to the diagnostic aggregate, requires unique32-owner coverage, and
checks expert/feature replica equality before counting unique collective entries. Original
CPU replay is retained as superseded analysis; the reviewer required these two binding checks.

Choose a new, default-off FP32 route accumulation: retain original BF16 per-route down/weight
products and exact slot identities, promote those values to FP32, sum in FP32, feed the
expert8 combine without an intervening BF16 round, retain the existing post-combine BF16
and routed-scale/shared output boundaries. No change to feature collective strategy,
FP8 projection, routing, checkpoint layout, legacy execution or the sealed decoder.
This is an explicitly new numerical implementation, not undocumented compiler emulation.

## Fixed admission contract for the NEW candidate

Use the retained real layer3 pack/oracle and current physical mesh from the first test.
Two17-row cases: original normal and concentrated distributions from `case_rows`, unchanged
captured row0; no dropped/duplicate/wrong-owner routes. Worst case places all136 selected
routes on one expert owner. The candidate uses grouped kernels, not a token-by-token model.

- CPU tests must prove exact route/order/ownership and the explicit FP32-sum formula from
  BF16 weighted parts, plus unchanged default-off behavior. CPU bit agreement against a
  BF16-reducing CPU reference is not a requirement of this deliberately FP32 implementation.
- Compile/HLO must preserve three grouped raw-U8 calls, three shared calls, only physical
  feature4/expert8 collectives, no full-weight expansion, no BF16 round of the new local sum
  before expert combination. Compiler allocation remains<=1GiB/chip for this one-layer test.
- Compare NEW uninstrumented candidate versus the old uninstrumented one-row TPU path at
  every row/local feature shard. Apply the EXISTING Gate C `REAL_LAYER_OUTPUT_TOLERANCE`,
  unchanged: max absolute error<=0.125, p99<=0.0625, mean<=0.02. Require each row and the
  complete17-row tensor to pass; report errors and bit mismatch counts, never call them exact.
- Compare candidate row0 DIRECTLY with the captured legacy output on every owner using
  those same bounds; checking only the reference versus legacy is insufficient.
- Require finite outputs and all32 health flags, exact supplied inputs across hosts,
  topology/weight ownership and byte/hash checks, measured HBM on all chips, both leases,
  authenticated pre/post census, generation-bound original outputs/HLO and DB linkage.
- Explicit new bounded-arithmetic classification. No timed samples; NULL latency. The prior
  exact-admission failure and diagnostic records remain unchanged and cannot be promoted.

Bounded-internal acceptance is the existing §5.5/§21/§24 contract, not a performance-target
change. The original candidate's normal outputs also fit these existing bounds (all owners:
max0.0625/p990.00390625/mean0.0001926523 versus M1; captured row0 max0.015625/p990.0078125/
mean0.0018774072 versus legacy), but that offline fact does not pass its strict experiment,
cover concentrated routing, justify this new candidate, or prove downstream correctness.

## Escalation and stop rule

One bounded two-case real-layer run after review, tests and persistence. Stop on structural,
health, memory or bounded-error failure; no post-failure widening. A pass closes only this
MoE component's bounded numerical admission. Changed prefill still needs causal layers,
its own short-decoder §21 proof, registered prefill/TTFT targets and efficient L7/L8.
Do not conduct another boundary-capture campaign solely to reproduce a tiny collective
rounding difference. No new checkpoint or long serial run is authorized.

Independent Astra design review: PASS in scope on2026-09-07, with the two replay P2 fixes
above applied and CPU-replayed with identical numerical totals. Implementation review is
separate and pending. Initial14 focused CPU tests pass16.15s, including forced32 formula,
default-path identity, direct-legacy/per-row refusal and original-NPZ comparison replay.

Final independent Astra implementation review PASS for persistence and ONE untimed two-case
TPU admission; no remaining P0-P2. The additional precision linter follows the expert
collective's actual operand through fusion roots/parameters to the eight-route FP32 sum,
checks both scalar-add reducers and rejects intervening BF16/correction or unknown forwarding.
Connected/fusion-forward positives and disconnected/round/correction negatives pass. Broader
CPU check had41 passes and one stale shell-string assertion; updated for the added mode,
all5 affected tests pass0.17s. No numerical tolerance was changed.

## Outcome — DB583 sealed2026-09-07 21:07Z

Both preregistered cases pass all32 owners, every row/aggregate and direct row0 legacy check.
Normal differs from M1 in3/2/6/5 values per unique feature shard, concentrated in zero.
HLO precision/locality and measured HBM pass; worker phase39s, no timing samples, all8 clean.
Exact generation/DB/archive identities and numerical totals:
`../artifacts/prefill-real-moe-fp32-bounded-admission-20260907.json`.
Independent Astra outcome review confirms bounded component admission only; no P0-P2.
Proceed to causal-layer integration. Full prefill/short-decoder/TTFT/L7/L8 remain open.
