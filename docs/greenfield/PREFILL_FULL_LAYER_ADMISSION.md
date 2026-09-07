# Bounded complete-layer prefill admission — v1

2026-09-07. Default-off component correctness experiment under §24, not a protected
model/legacy equivalence or performance claim. Independent Astra design review accepted
the scope with the constraints below. Implementation/execution review remains required.

## Question and scope

Can the new complete batched layer execute on TPU with real retained weights, correct
causal/cache/routing semantics and bounded differences from the existing raw-layout
one-row layer? This is the next discriminator before short-decoder integration.

Run layer0(full DSA+dense) first, then layer3(IndexShare+MoE). Select only their complete
tensor sets through `checkpoint/ws32_layer_subset.py`; no full753B model load/copy.
Source pins and controller-only byte proof are in
`../artifacts/prefill-selected-layer-host-admission-20260907.json`. Validate current
retained files/slots across all8 hosts before launch. Controller byte proof is not fleet proof.

Reference: `ws32_transformer_layer_mapped` with raw final-owner FP8 weights, no
StrategyND dense overlay and no exact-convolution aliases. It is an existing greenfield
arithmetic reference, NOT the promoted exact decoder or an independent legacy oracle.
New full prefill must still earn its own short-context §21 and efficient L7/L8 results.

The DSA arithmetic differs beyond row batching: the raw scalar reference calls
`dsa_index_keys_from_projection` with its default `multiply_rsqrt` normalization and
uses HIGHEST scoring; the candidate uses `divide_sqrt` normalization and DEFAULT scoring.
These source-confirmed differences must appear in the result classification. Neither
reference settings nor bounds are changed to manufacture identity. The bounded cache
and separate own-score checks below are deliberate; no cross-path score identity is claimed.

## Fixed cases and numerical contract

17 static rows, capacity1024 (two512-position logical pages,64 rows/owner/page), real
GLM geometry/top2048, nonidentity page table `[1,0]`. Cases: empty prefix offset0/live17;
initialized prefix offset505/live17; same prefix/live11 with padded NaN inputs.
Initialize identical finite synthetic prefix state explicitly. Activations/cache state
are synthetic with real weights; they are not outputs of correctly computed prior layers.

Candidate executes one layer over the whole block. Reference executes the same layer
one row at a time, carrying its own updated KV and unrepaired index cache. No production
host per-row dispatch is introduced; the loop belongs only to this untimed reference.

- Compare output update and carried residual per live row AND aggregate with existing
  Gate C output bounds: max0.125/p990.0625/mean0.02.
- Cross-path KV, unrepaired index and repaired index written rows use those SAME numeric
  bounds **as an explicitly registered contract of this new experiment**, not as an
  assertion that Gate C previously established a cache-key tolerance. Compare only newly
  written rows per row/aggregate, never dilute errors by averaging the full cache.
- Every untouched cache byte, including future/tail/prefix bytes, must equal its own input
  exactly. IndexShare must leave both supplied index caches unchanged. Padding output and
  carried residual must be exact zero; padded route weights must be exact zero.
- Compute each path's repair from its own actual normalized inputs with the existing M64
  helper. Check candidate repair against its own separate M64 reconstruction exactly;
  distinguish that identity check from bounded cross-path repair agreement. Exposing the
  normalized input is admission observability, not proof about an unobserved executable.
- Layer3 route IDs must be exact against the one-row reference BEFORE routed-output
  acceptance. Route weights use existing max/p99=5e-4,mean=2e-4 per row and aggregate.
  Route mismatch stops this protocol; no tolerance relaxation after seeing results.
- Full-indexer valid counts and causal coverage are exact. Every valid prefix in these
  cases is shorter than top2048: the selected set must contain EVERY causal position once,
  with score order and lowest-position ties canonical against each path's OWN scores.
  This does not establish top2048 cutoff stability on longer prefixes or score arithmetic
  against a separate FP64 forward. Shared layers must preserve supplied selections exactly.
- All32 owners must be healthy. False incoming health must survive propagation. Invalid
  offset/page metadata must fail health and leave proposed caches unchanged.
- Perturbing a future row must leave earlier output/selection/routing unchanged. Replacing
  repaired history must not affect prompt output/selection/unrepaired cache. Padded NaNs
  must not change live results or writes. These are exact candidate-to-candidate checks.

No threshold is fitted to observed TPU results. A failure preserves original bytes and
stops at the smallest failing case; diagnose locally and review the narrow evidence.

## Execution and evidence

Reuse the bounded FP8 wrapper's two leases, authenticated normal/root pre/post8-host
censuses and generation-bound original-file publisher. Add a distinct layer protocol and
DB classification; do not reuse MoE's six-call HLO rule or describe this as that proof.
Existing pod only; never manage TPU/VM/queued resources. Review/publish the frozen source
before deploying to existing worker checkouts. Initial worker budget600s; no serial long run.

Source collective expectations: both branches have three expert8 reductions (selected KV,
attention output, MLP output). Layer0 has three expert8 gathers (query/head and two candidate
gathers), plus one feature4 normalized-input gather. Layer3 has two expert8 router gathers
and no feature4 gather. Feature4 reductions may be tuple-merged: validate their actual
payloads/dependencies, not a guessed count. Separate M64 reconstruction is a declared
side program, never included in these candidate-graph counts.

Pre-execution: acquire actual HLO, verify exact feature4/expert8 groups/counts and bounded
live-row raw-FP8/structured/grouped calls; no full-pod reconstruction, callback/staging or
full BF16 model/expert expansion. Check compiled candidate allocation against a2GiB/chip
admission ceiling; separately record reference/repair program allocations and measured
peak HBM/headroom on all32 chips. This ceiling is not a full-model HBM budget.

Preserve every rank's original output/reference/repair arrays, health and input hashes;
controller re-derives comparisons from authenticated bytes rather than worker booleans.
Bind loader metadata pins, selected tensor hashes, physical slot/device mapping, code hash,
HLO and original generation/size/CRC/SHA receipts. Timing fields remainNULL: execution
duration is operational accounting, not a speedup. New prefill/TTFT targets must be
registered before candidate performance experiments. Complete-layer trace/warmed wall
and later full-decoder protections remain distinct follow-up evidence.

## Worker implementation / CPU admission — 2026-09-07

The distinct guarded mode is `GLM_GREENFIELD_FP8_MATMUL_KERNEL=ws32_prefill_layer_admission`,
with `GLM_GREENFIELD_PREFILL_LAYER=0` (then3 only after layer0 outcome review/cleanup).
Use `scripts/greenfield/run_fp8_matmul_microbench.sh`, never the worker/campaign directly.
The wrapper still owns both leases, root+normal8-host censuses, DB and terminal publication.
Current worker budget600s. Candidate compiled allocation cap2GiB/chip. No warmup/timing.

Before any worker initializes JAX, all8 hosts authenticate retained metadata and four
file sizes/headers. Runtime then binds those slots to actual JAX physical owners and
hashes/loads only the selected layer. The controller checks every observed tensor digest
against its original index in the fixed full manifest. Unselected payload integrity is
explicitly outside this admission; no full checkpoint copy or full-file hash is needed.

Actual candidate and scalar normalization are retained. Reference normalization alone
has a declared observer host roundtrip into the separate M64 reconstruction program,
using local addressable shards (not a global tensor fetch). This is untimed reference
harness work, not candidate transport or a production design. Each program's StableHLO,
optimized HLO and compiler allocation are retained, including separate completed BF16
wk decode/FP32 promotion. Controller comparisons use original NPZ bytes, not verdicts.

111CPU tests12.68s: fixed-input tampering, per-owner tensors/causality/cache interventions,
full-layer HLO/fleet/ledger refusal and real-shape20-input CPU32 tracing. Shape fixture
is16.9KB metadata with original manifest SHA, NOT weights/integrity evidence. Independent
review found a page-table rank error; corrected to[1,2] and tested through actual builders
before TPU. Second review found no P0-P2; approves one layer0 launch after clean commit,
push/mirror and preflights. Layer3 remains conditional. No TPU result is claimed here.

## Layer0 compiler acquisition correction — 2026-09-07

Tag `greenfield_fp8_ws32_prefill_layer_admission_l0_20260907T231756351264474Z`,
pin4a15234c, loaded selected real layer0 and compiled candidate6.44s on controller.
All32 original selected tensor ledgers pass. Candidate HLO `dc1d5a94…a13d9f3` has
the12 intended Pallas calls and exact declared collective payloads/groups. The initial
guard wrongly required EVERY custom-call to be Pallas: compiler-emitted indexing/layout
helpers are not model kernels. Execution stopped before candidate arithmetic,8/8 clean.

Register layer0's exact helper inventory separately:9 AssumeGatherIndicesInBound
(7×s32[1024],2×s32[34816]);7 GatherScatterIndicesBitpacked
(2×s32[17,4096,2],2×s32[17,16384,2],3×s32[17,2048,2]);6 ConcatBitcast
(U8[2048,2048],[3584,512],[1536,2048],3×[1536,1536]). Each concat reconstructs
four contiguous local quarter slices of one U8 parameter, not BF16/global weights.
Index input/output types/shapes match; concat arity/quarter shapes match. Unknown targets,
side effects, missing/extra/wrong-shaped helpers remain refused. Existing one_layer and
runtime/decoder guards already distinguish these compiler helper classes. Layer3 helper
inventory is deliberately unregistered until its own acquired HLO is reviewed.

70CPU tests8.59s and replay of all8 original HLOs PASS; independent narrow Astra review
PASS. No numerical thresholds/model arithmetic changed. One corrected guarded layer0
retry after persistence/preflights; layer3 stays conditional, no performance claim.

Next layer0 tag at ea93a4ee executed empty/boundary successfully on all32 owners;
all8 original NPZ pairs replay PASS. Tail failed before execution because global JAX
device_put numerical cross-host equality rejects identical NaNs. Preserve intentional
padded NaNs: input harness now collectively verifies shape/dtype/byte SHA and initializes
owned slices with explicit sharding callbacks. This is initial fixture transfer outside
all executables, not model transport.78CPU tests and narrow independent review pass.
Original failed tag/bindings in `prefill-layer-tail-transfer-failure-20260907.json`;
complete admission remains open until all three cases pass in a fresh protected tag.

## Layer0 closed — DB584, 2026-09-07 23:34Z

Fresh tag `greenfield_fp8_ws32_prefill_layer_admission_l0_20260907T233201686218983Z`
atf2fb1141 passes all3 cases/all32 owners with generation-bound original-array replay,
exact HLO/local groups and47,517,696B maximum per-chip measured HBM including reference.
Written KV/index/repair and carried residual cross-path errors0; output worstabs6.10e-5.
DB584, same-region terminal SUCCESS, authenticated root+normal8/8 clean. Compact pins
in `../artifacts/prefill-complete-layer0-admission-20260907.json`; HANDOFF has exact tag.
No warm timings/full-model/legacy equivalence claim. Layer3 remains required; its helper
inventory needs its own bounded compile acquisition before numerical execution.
