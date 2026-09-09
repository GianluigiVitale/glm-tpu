# Expert-relative FP8 panels — next throughput candidate

## 2026-09-09 03:37Z — hardware selection PASS, DB600

Corrected d94126c9 run sealed, unchanged3/10/2 sampling and180s budget;287calls
perhost,9actual graphs,32owner bounded originals/memory,8host64core traces/cleanup.
Suffix10.505594→6.305570ms (ratio0.600211), widepartial44.025856→40.642469ms
(ratio0.923150): both preregistered DB596 criteria pass. Packing/unpacking included.
See prefill-expert-panel-phase-db600-sealed-20260909.json for exact original pins,
generation-readback sources, per-chip peaks and all p50/p99 distributions.
Independent review replayed all8 numerical receipts and found noP0-P2.
Selected default-off component, NOT fused layer/full-model/TTFT admission; prefix
and whole diagnostic traversal did not improve. Historical failedrun preserved.
Next larger-window integration must retain completed prefix boundaries and its
own single-path assembly, not assume old fused mlp_window inherits this proof.

## 2026-09-09 03:16Z — first panel budget refusal, narrow observer correction

879bed7e passed actualgraphs/memory/first bounded comparison on8hosts, failed
180s atwall9 beforetrace. Receipt prefill-expert-panel-budget-refusal-20260909.
Full bounded report in runner caused repeated1.586MB JSON serialization; local
63.3ms becomes3.49ms when only that report is compact. Keep canonicalSHA/bytes/
passed, rederive full unchanged comparison from originalNPZ atcollector. Eight
CPU numerical/tampering/actualpanelcollector-DB tests PASS280.93s. Onecorrected
trial after persistence/freshguards; no deadline/sample/model/bound change.
Preliminary6.373ms suffix and40.629ms widepartial are NOT promoted results.

## Current next action after DB599 — 2026-09-09 03:02Z

The historical missing-budget/target blocker below is resolved: DB598 is sealed
and PREFILL_PERFORMANCE_TARGETS.md/linked JSON are prospectively committed.
DB599 selects exact local merge (1.71/1.74x long-prefix DSA component gain),
independently reviewed; not wired into this panel experiment's retained prefix.
Next bounded test is this already-staged panel candidate against original B32,
with committed targets bound before launch, real selected weights and protected
phase evidence. Verify panel-specific target propagation before deploying it.
No repeat baseline, acquisition-only run or full-model reload between microsteps.
Panel TPU numerical/allocation/performance remains UNMEASURED.

Prospective component selection: compare to sealed DB596 original B128, not only
the slower four-B32 path. Require >=10% lower wide-suffix p50 (packing/unpacking
included) and no increase in wide partial phase-sum p50. Record all p50/p99/trace
results even on rejection; phase sums still omit independent final assembly and
cannot certify TTFT. Fixed final target files and DB596 receipt are SHA-bound by
the existing pre-JAX `registered_programs` call on all eight hosts and every
actual graph report/replay. No historical variant or numerical bound changes.

## Current integration — 2026-09-09

The distinct `ws32_prefill_expert_panel_phase` mode is now wired through the
existing nine-program compiler, 287-call worker, original-array collector,
launcher and diagnostic DB accounting. **Only B128 uses panels. B32 remains the
original grouped kernel**, so its complete DB594 byte witnesses remain an
independent suffix reference. This supersedes the two-changed-suffix proposal below;
the recorded B32-panel hash is unused, not deleted historical evidence.

Before the first timed traversal, the complete first warmup must pass the
unchanged per-row/aggregate output bounds against original B32 and reconstruct
both actual device assemblies. All non-output B128 fields, every prefix/narrow
field and both WK witnesses remain exact DB594. Subsequent candidate outputs
must repeat their first bytes. The collector independently recomputes the bounds
and assembly from retained arrays rather than trusting the worker report.
Packing/unpacking remains inside the B128 suffix timing. Candidate scratch/code
caps are128MiB/32MiB; actual analyses of all nine programs enter live budgets.

The 21 pending original-array/production-shape/variant tests passed227.23s;
actual compiler→WK→287-call→trace composition passed all3variants28.63s.
The all8-owner collector test initially exposed an invalid synthetic fixture:
rank0 cache bytes relabeled as other owners fail exact untouched-cache checks.
Panel mode now uses retained DB596 originals on all8owners, with no download
or weaker production checks. Compiler optimized HLO and timing in this CPU test
remain fixtures, not panel TPU evidence. Final composed test result is in HANDOFF.

Independent Astra source review: no P0–P2 conditional on tests passing.
No TPU panel run is authorized yet: §24 final performance target registration
was missed by the preceding optimization trials. Resolve the two missing budget
inputs in `PREFILL_COST_MODEL.md` before the next optimization timing launch.
Do not change old DB classifications or treat diagnostic scope as a waiver.

2026-09-09. Main-agent source/byte inspection and independent reviewer
`/root/observer_identity_review` agree. This is a design, not a TPU performance result. DB597 has sealed; source freeze is over.

DB597 paired own2K is sealed:30.9739prompt tok/s. See
`../artifacts/prefill-paired-short-sealed-20260909.json`.

## Evidence and priority

Source `pallas/prefill_grouped_fp8.py` uses global row-aligned Megablox metadata,
row_tile8 and grid(Ntile,group_row_tile,Ktile). Experts sharing global row tiles
cause masked revisits. Weight dequantization executes per active route tile.
This confirms scheduled work, not measured HBM reload or utilization.

DB596 local original `fleet/rank0/phase_first.npz` matches archived ledger:
generation1788901524314158, size47677909, CRC32C QGGd7A==,
SHA bcaac31ae51c27bf245ba0de4f8c9dde746b56fda5c1eee65c9a0a76ea13db7b.
wide_12/13/14/15 routes are equal. B128 has1024 routes,117active experts,
maximum33 rows/expert; owner totals150,90,135,120,151,123,109,146.
This is real-weight/synthetic-state layer6 evidence, NOT real-prompt routing.

| M | Global visits | Expert-relative panels | Global max/owner | Relative max/owner |
|---|---:|---:|---:|---:|
|8|234|192|35|29|
|16|172|140|27|20|
|32|144|118|22|18|
|64|131|117|20|18|
|128|123|117|19|18|

Counts derive from global ceil(end/M)-floor(start/M) versus ceil(count/M), with
zero-count experts skipped. No speed forecast follows from these counts.
M32 is a useful first discriminator; M128 quadruples relative padded rows versus
M32 (14976 versus3776) while eliminating only one panel in this fixture.

## Proposed bounded candidate

Default-off expert-relative panels, M32/N256/K128 arithmetic. Preserve stable
expert/route sorting and original route-slot restore/FP32 route sum. Pack only
owned expert rows into aligned [Pmax,32,K] with descriptors(expert, sorted_start,
live_count). Ordinary BlockSpec((32,K)) cannot express arbitrary expert starts.
Exclusive aligned panel outputs prevent overlapping expert boundary writes;
mask padding, unpack to original sorted rows with unowned output zeros.
Pmax=ceil(m/32)+local_groups-1; m=8B, G32 gives63panels/2016rows atB128.
Reuse descriptors gate/up/down; initially preserve existing intermediate shapes,
feature4 reductions, BF16/SwiGLU boundaries and restoration. Include all packing
and unpacking in measured wall. Do not allocate per-expert full-B workspaces.

Grid(N/256,active_panels), full-K rawU8 [256,K] panel resident in VMEM; internally
two independent N128 outputs, each using its correct FP8 scale row and original
increasing K128 accumulation. RawFP8->F32scale->BF16 boundary unchanged. Do NOT
broadcast one scale across N256. Logical double-buffer allowance approximately
1.08MiB K1536 and1.39MiB K2048, not actual compiled VMEM/HBM admission.

Smallest test: reuse interpreted grouped-projection tests and the authenticated
DB594/596 completed-prefix B128 suffix operands; no new scalar taps or baseline.
Test metadata validity, tails/empty/unowned experts, exclusive writes, every live
row retained, source scale geometry, original K order; then actual changed suffix
arithmetic/memory/inclusive wall under existing protected harness.

Larger full-model token windows still needed afterward. Current window code
unrolls four B32 prefixes atB128. Scaling to512/2048 by duplicating prefix graphs
would multiply code; a rolled bounded causal prefix loop is a separate candidate.
Keep per-layer KV, unrepaired/repaired index lifecycle and own-score DSA contract.
Neither panel proposal nor prior shared-prefix agreement proves own8K/L7/L8.
No B32 full-model intermediate, no new checkpoint, no numerical tolerance change,
and no10Ktok/s promise. DB597 now supplies accepted short-model comparison evidence.

## Independent design review

Astra /root/observer_identity_review: no P0 blocker. Preserve independent N128
scale selection; `_scale_value` wraps ki modulo8 and must receive the correct slab.
Test N512/K1152 with distinct K0/K8 and four N128 scales. Guard dynamic count overflow,
invalid/extreme offsets, empty owners and inactive indices before gathers. Prove
exclusive panel writes, inverse mapping, every live row once and masked padding.
Reuse metadata once for all projections. Actual TPU allocation and inclusive
packing/unpacking wall remain required; CPU interpretation cannot prove either.

## Implemented CPU candidate — 2026-09-09

`prefill_expert_panels.py` creates one device plan per routed group. It bounds
int32 sums, handles empty groups, gathers only safe row addresses, and gives
each panel exclusive output storage; inverse gathering restores sorted rows.
`pallas/prefill_panel_fp8.py` implements the proposed M32/N256 full-K U8 panel
with two independent N128 accumulators and absolute-K scale addressing.
`ws32_prefill_moe.py` reuses the same plan for gate/up/down. The opt-in reaches
the existing completed-window builder/abstract preparation; no worker variant
is authorized and all serving defaults remain off. Checkpoint layout unchanged.

15 CPU tests passed19.00s (9 new panel tests plus6 original grouped regressions).
They cover M32 boundaries, safe tails/empty owners, poisoned inactive rows,
concentrated routes, fullK ninth-block scales/N512, F32/BF16 projection equality,
and forced32 full MoE equality/local groups with supplied routes and scaled-down
expert geometry. This is not a real-prompt or protected TPU result.

Production abstract preparation/lowering test passed5.49s: all five original
paired-phase raw graphs remain byte-identical with panels OFF. With panels ON,
WK/prefix remain identical and each suffix contains three new panel calls with
unchanged output interfaces. CPU-only Mosaic13 serializer used as in prior
preregistration; actual compiler/executable allocation still requires TPU.

Preregistered raw suffix hashes (must be reproduced before launch):

- B128 candidate: `b3dc3996d7ab44384ad15f51a89acdc443e134f915757ffec782661b2fef24d5`,123770B.
- B32 control: `6fd6de7f3ca2bccf568521a6fd736abb3d9a055730469ea6ce3fc816eb8e643b`,123267B.

Independent current-source Astra review found no P0-P2; CPU32 was its remaining
persistence condition and has passed. Hardware admission remains separate.
Next adapt the existing DB596/DB594 completed-prefix original-array phase harness
to a distinct panel profile, preserving baseline checks. Compare the new B128
suffix against the old completed-prefix witnesses, with packing/unpacking inside
wall and actual compiled/live memory. Do not run another baseline/acquisition-only
campaign or reopen scalar boundary taps. If numerical differences arise, retain
original arrays and use the existing bounded suffix contract, never relax it.
