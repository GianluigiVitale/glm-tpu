# Engine efficiency audit — prefill-first pivot

Status: 2026-09-07, engineering audit/design, NOT performance proof of a new implementation.
Authority: [specification §24](../glm-tpu-revolution.md), owner directive to stop the remaining
serial long-context campaign. Original depth0.05 sealed DB575; its evidence is preserved.
The next experiment is bounded batched prefill, not serial depth0.95 or serial256K E0.

## Outcome and root cause

The long wait is device computation, not prompt upload. Current prefill feeds one token at a
time through all 78 layers using the batch-one step. The outer chunks reduce temporary memory
and host dispatch count; they do not reuse a layer's weights across multiple prompt rows.
DB574 measured **16,425.514998 seconds** fleet-max prefill for **127,363 prompt tokens**.
At 256K, the existing capacity measurement projects approximately 10.3 hours; that is a
projection, not a completed full-prompt 256K measurement.

This was deliberately accepted as a correctness/memory bridge in §23. Treating that bridge as
sufficient to finish the engine was the planning mistake. §24 now requires an efficient
prefill path before completion. We retain every existing correctness result and its limitations.

Independent adversarial audit: `/root/astra_long_run_recovery_review`, gpt-6-astra, 2026-09-07.
The main agent independently inspected the scan, FP8 row tile, DSA page/score ordering, loader
waits and host chunk loop. Agreement: prioritize layer-major multirow prefill. Do not mistake
an independent review for a TPU measurement. Source paths below refer to the current source,
unchanged from run pin `a9bfbbb3` (documentation changes only at this audit).

## Evidence anchors and limits

- DB574: `greenfield_ws32_short_decoder_128k_d0_0_numerical_cap131072_hrope_20260907T064941550123130Z`.
  Summary SHA `33912e8391dde6f0cfed53b608098d6782dc8c9469edc6675c2e7364207a83c2`.
  SUCCESS `f123beba8caab916e1307607ef28b294606c95338f3a7d431ce374a210900667`.
  Approved archive: `gs://driftbench-dsv4-uc/results/<tag>/`.
- DB574 protected decode: fleet p50 143.496482 ms; 6.968812 wall tok/s;
  peak HBM 27,810,852,864 B/chip. These are NOT prefill throughput figures.
- DB574 trace attribution (decode, not prefill): collectives 44.075 ms/step;
  custom-call-other (including FP8 formatting) 44.235; gather/scatter 17.493;
  sort/top-k 7.299; movement 9.638. The first two total approximately 65% of recorded
  135.343 ms busy time. Do not relabel the entire custom-call category as dequantization,
  the entire movement category as cache copies, or any of these as prefill measurements.
- DB574 rank0 load 139.297 s, summed graph compilation approximately 699.122 s. Cold-start
  costs matter but cannot account for 16,425 s measured inside prefill.
- DB572: an 8K prompt at capacity262656, not full256K. Peak 29,655,086,080 B/chip,
  3,359,312,896 B headroom; serial chunk2048 approximately290.8 s. New batched scratch must
  be budgeted and measured independently; this margin is not automatically available to it.
- DB403 legacy128K approximately600 s/item and DB402 legacy256K prefill1337.7 s are historical
  orientation with different timing definitions. They do not establish matched warm TTFT.

## Ranked findings and decisions

### E1 — serial token-major prefill (confirmed; first implementation priority)

Evidence: `glm_tpu/greenfield/runtime/ws32_decoder.py`,
`build_ws32_chunked_prefill_program`, `exact_body`/`scan_step` (around lines1780,1867–1910):
`lax.scan(unroll=1)` calls `ws32_prefill_step_mapped(token[None], ...)`, which calls the full
decoder body. `scripts/greenfield/run_short_decoder_ws32.py:1043` dispatches approximately63
chunks/tail for128K. Roughly263 seconds of device work per main chunk dwarfs host dispatch.

Decision: build a separate layer-major path over real token rows, retaining the existing path
as reference. Larger serial chunks, more host workers or a blind `vmap` of mutable decode state
do not solve this. Use existing checkpoints, physical groups and numerics where possible.

Smallest test: representative dense, full-indexer and MoE layers at 8–32 causal rows against
sequential reference evaluation on identical inputs and prior cache. Compare intermediates,
per-row selection/ties, routing, cache addresses/values and continuation. Initial row counts
are experiment candidates, not promised production tile sizes. Require useful weight reuse in
HLO/trace before assembling all78 layers. CPU success alone is not TPU performance evidence.

### E2 — one-row linear/MoE wrappers (confirmed structure; gain unmeasured)

Evidence: `kernels/ws32_layer.py:1083` and `kernels/ws32.py:266` enforce one live row in
decode; `kernels/pallas/fp8_matmul.py`, `Fp8BlockMatmulConfig.row_tile=8`, pads input rows
to that tile. The underlying raw-FP8 primitive already accepts multiple rows.

Decision: reuse the primitive for prompt rows, then group routed token/expert pairs for
weight-tile reuse. Do not remove the correct one-row invariant from decode. Do not duplicate
the entire checkpoint in BF16 or repack terabytes to test this hypothesis.

Test/risk: captured multirow FP8 inputs against the existing per-row path; keep output dtypes,
BF16/FP32 rounding boundaries and routing ties. Test all routes on one expert owner, uneven
dispatch and tails. No dropping tokens, truncated capacity or silent fallback. Matrix row
geometry can change TPU association; adjudicate numerics rather than assuming bit identity.

### E3 — collectives and FP8 formatting (measured decode cost; prefill attribution pending)

Evidence: DB574 categories above. Prefill's decoder reuse makes repeated fixed overhead a
credible target, but the fraction of prefill saved is not measured.

Test: trace a bounded multirow real layer, classify actual custom-call names, physical group
sizes/counts, useful rows and weight loads. Compare weight-tile reuse and, where numerically
valid, reductions over multiple rows. Preserve feature4/expert8 groups. A fused operation that
changes association needs review and correctness proof. Do not promise linear speedup with
row count: expert coverage, compute, DSA and scratch growth can change the limiting resource.

### E4 — DSA scores allocated history before masking (confirmed structure)

Evidence: `kernels/ws32_layer.py:543–575` gathers logical pages and calls `dsa_scores` on
flattened allocated keys; `kernels/reference/dsa.py` masks valid lengths at local top-k.
Early prompt rows therefore pay for invalid future capacity in the source program. Optimized
HLO/trace must determine retained physical work and savings.

Test: tiled causal scorer/exact top-k across several prefix lengths, page boundaries and tied
cutoffs. Bound score tiles; merge candidates with the canonical lowest-position tie order.
No dense `[rows,heads,context]` temporary. Consider a small number of prefix buckets only if
saved execution outweighs compile cost; avoid a new executable per prompt length.

### E5 — IndexShare address reuse (hypothesis; inspect before implementing)

Evidence: attention boundary in `kernels/ws32_layer.py` around line753. Reused selected
positions can imply reusable sort/page-address metadata, but each layer owns DIFFERENT KV.
Inventory a four-layer group's actual HLO before claiming redundant operations.

Test: reuse only invariant addresses/order/masks; require exact address and attention-result
comparison. Never reuse gathered KV values across layers. Lower priority than E1/E2.

### E6 — cache copies and donation (hypothesis; do not assume all movement is copying)

Evidence: `_ws32_decode_impl` updates whole-state cache containers with
`kv_cache.at[layer_id].set(...)`; no explicit state donation at the audited entry compile.
DB574's movement total is not an attribution to these source expressions.

Test: inspect optimized copy/alias allocation and input/output ownership first. Donation is
safe only when no runner/observer still consumes the old state. Then adversarial cache probes,
measured peak HBM and clean wall A/B. Do not create use-after-donation to save a hypothetical copy.

### E7 — cold load, compile and validation (confirmed waits; optimization benefit pending)

Evidence: `checkpoint/ws32_runtime_checkpoint.py:1175` hashes each tensor and calls
`device_put(...).block_until_ready()` before advancing. Full file integrity checks also exist.
Acquisition and numerical runs compile separately; their provenance purpose is real.

Test: phase timings first, then bounded asynchronous transfers with a strict in-flight byte cap
if the data supports it. Executable-cache reuse requires code/config/topology/compiler identity
and acquired-HLO validation. Do not disable hashes or dirty-source refusal to save startup time.
Batch compatible small tests in one protected workload where its declared protocol permits it;
never overlap TPU workflows. These improvements must not delay the E1 discriminator.

### E8 — TTFT and possibly unnecessary prompt heads (timing gap confirmed; head cost hypothesis)

Evidence: the runner times prefill but compiles observer/decode later; it is an evidence harness,
not a first-token streaming interface. `_ws32_decode_impl` calls final sampling even when used
for teacher forcing. Determine which unused head work survives compiler elimination first.

Test: instrument request input ready, transfer/cache initialization, prefill completion and
actual first-token delivery. Remove nonfinal head work only if it is physically present and
the retained health/final-token contract is equivalent. Report harness-only observation,
tracing and sealing separately from production latency, without hiding required serving work.

### E9 — artifact and experiment overhead (standing constraint)

Reuse current final-layout weights. Live bucket ceiling2.5e12 B, US-CENTRAL2 only, no full-size
safety copies. Explain >100GB artifacts before creation, including temporary/retained bytes
and replacement. Keep compact source/config/manifests/results; do not retain every generation.
Shared gzip HLO layout §23.10 already avoids seven redundant copies; keep that protection.
Preserve authenticated observations and original runs across controller failure: DB574 saved
another4.6h prefill by recovering evidence, not rerunning compute. No infra action is authorized.

## Minimal implementation shape and hardest correctness boundary

For each bounded block of known prompt IDs, embed rows, then execute each layer over those rows.
Construct the layer's current-block keys before causal attention over prior cache plus this
block. Carry per-row position/valid length and `[rows,top_k]` IndexShare state across layers;
group MoE routes without losing token/expert identity. Commit only valid cache rows.

Preserve the TWO index caches: all prompt rows attend using unrepaired keys, including earlier
blocks. Write repaired M64 keys to the separate destination; promote that destination only
after the whole prompt. Repair changes from batching need their own comparison against the
current repair kernel. Main-attention host rotary stays as adopted in B′; indexer rotary stays
on device. Do not reopen the refuted host-indexer-table path because of the pivot.

Before code work, read/update `REUSE_INVENTORY.md` and
`../../configs/greenfield-reuse-inventory.json`. Reuse/adapt APIs deliberately rather than
copying a second complete decoder or importing legacy execution. This is a design direction,
not yet an implemented or accepted interface.

## Performance registration and staged experiments

Final quantitative prefill/TTFT targets are **not yet registered**. This prevents performance
promotion, not source inspection, design, CPU correctness or bounded baseline measurements.
The reproduced arithmetic, routing-reuse model and minimal baseline campaign are in
[`PREFILL_COST_MODEL.md`](PREFILL_COST_MODEL.md). Its planning bands are not acceptance targets.
Before candidate performance experiments, record same-hardware baseline definitions and
compute/weight-traffic/collective/DSA budgets, then fixed128K/256K targets and wall budgets.

Reviewer suggested ≤600s128K/≤1500s256K as INTERIM engineering milestones. Main-agent decision:
do not adopt those as completion thresholds or imply they are interactive. They are historical
orientation, not an expert promise of what this hardware should deliver. Choose justified
final targets before trials and do not relax them to fit a disappointing candidate.

1. Inventory reuse and specify multirow state/causality/repair with reference tests.
2. Bound live/scratch bytes at selected row counts and long capacity; inspect generated HLO.
3. Register targets from baselines/budgets; run the smallest real multirow layer discriminator
   with separate warmed wall and trace, compare outputs and decide before full decoder work.
4. Assemble the short decoder, prove §21 on its own outputs. New first divergence requires
   its own valid preregistration/review; do not mechanically reuse B′'s record.
5. Measure prefix-length scaling and acquire long-capacity memory/HLO. Project full-test cost
   and fail early if the candidate still executes serial full-decoder work per prompt row.
6. Run all four L7 depths and full L8 on the candidate with full protections. Old serial depths
   are useful references, not substitutes for changed-prefill coverage.

Measurements: warm TTFT includes input transfer, cache initialization, prefill and actual
first-token delivery, resident weights/executables but no prefix-cache hit. Cold latency adds
load/compile. Disclose tokenization/transport boundaries. Report prefill tok/s, warm TTFT,
decode p50/p99 and total request wall separately; exclude profiling from measured steady wall.
No candidate speedup or completion ETA is established by this audit.

## Lessons retained

- A memory-safe token scan is not efficient prefill; inspect the loop body, not its name.
- Use cheap structural/correctness discriminators before long end-to-end tests.
- Decode timing cannot stand in for prefill or TTFT; profile the phase being optimized.
- Mathematical equivalence does not guarantee TPU BF16/FP8 identity. Keep existing acceptance
  contracts and judge differences using independent evidence, not implementation familiarity.
- Preserve successful work through process-identity-aware recovery; a tool timeout is not a
  reason to pay for the same computation again.
- Review the current diff and evidence, resolve material findings, then move forward. Neither
  speculative hardening nor renewed full-pod numerical archaeology is the new critical path.

First admission update: six CPU multirow FP8 cases (8/17/32 rows, zero/nonzero output tails)
plus four reuse-registry checks passed,10 total in9.38s. No runtime/enforcement file changed
while the original d0.05 seal was active. TPU association and performance remain unmeasured.

After DB575 sealed, multirow feature/expert linear and reciprocal dense building blocks were
added in `kernels/ws32_prefill_linear.py`, unwired/default-off. Three new tests passed in4.53s,
including forced32 CPU bitwise row comparisons and exact subgroup4/8 HLO membership. They are
not the StrategyND dense overlay and cannot replace it without new numerical evidence.

Real-shape primitive baseline now sealed DB576–579: rows8/32/128/256, K1536 N2048,
F32 output, p50≈0.232–0.248ms per batch, synthetic reference max difference0 in all four.
Each runner took8–9s. Exact pins/archive/memory and limitations are in PREFILL_COST_MODEL.md
and `../artifacts/prefill-fp8-real-shape-baseline-20260907.json`. This resolves the narrow
question whether the existing raw-FP8 primitive can execute these live row counts efficiently
at this local shape. It does NOT resolve E1, grouped expert reuse, causal layers or TTFT.
Next use grouped-row scheduling; do not write another standalone matmul just to obtain rows.

Grouped execution is now implemented as an unwired candidate (`PREFILL_GROUPED_MOE_DESIGN.md`):
active expert-row tile schedule, raw-FP8 tile reuse and original route-slot restoration.
Forced32 CPU output equals the existing MoE path bitwise for distributed and worst-owner
routing, with only physical subgroup4/8 collectives. This resolves the CPU mechanism part of
E2; it does not establish TPU arithmetic, grouped performance, full prefill or TTFT.

DB580 now closes the narrow F32 grouped-projection TPU arithmetic question:136 sorted route
rows, local32 experts/N2048/K1536, distributed/concentrated-owner/empty-owner cases all
bit-exact against old M1 projections. Runner8s, entire protected workflow34s, no timing
samples or performance claim. HLO contains one grouped raw-U8 Pallas call/no full table
decode/no collectives; peak process HBM107012096B including reference. See grouped design
and HANDOFF for exact evidence. BF16 down, real mapped MoE and causal full prefill remain open.

DB581 subsequently closes grouped BF16 down projection arithmetic at G32/N1536/K2048:
all three route cases bit-exact, runner9s, no timing samples. This is not a full MoE result.
E4 now has an unwired bounded causal DSA reference: up to32 rows/key tiles up to4096,
future-only tiles skip scoring/selection, exact local candidate merges and expert8 exchange.
Six CPU tests passed including physical group membership, tails/holes/ties and bad metadata.
The JAX reference may still be slow: repeated candidate sorts, padded key buffers and
per-head tile temporaries need TPU attribution before promotion or Pallas replacement.
Do not call this an efficient production scorer based on CPU correctness alone.

Real layer3 batched arithmetic trial at31a2f917 refused within one minute (20:22Z), before
concentrated case. Input/topology/checkpoint/HLO pass; actual BF16 outputs differ from old M1
by small amounts. Record `../artifacts/prefill-real-moe-arithmetic-refusal-20260907.json` binds
all8 original runner generations; failure census8/8 clean. Changed feature collective strategy
and local route reduction lowering are confirmed graph differences, NOT a proved cause.
Next is one B17 first-boundary diagnostic, not an unchanged retry or relaxed comparator.
Reviewer warns that F32 HLO with original-BF16 correction metadata is not proof of F32
rounding semantics. Synthetic projection identity also cannot stand in for real operands.
This is the intended cheap discriminator: no multi-hour model run was needed to expose it.

DB582 boundary diagnostic sealed20:42Z (46s worker phase). Captured projection partials match;
CPU replay proves an additional BF16 local route-sum round in the batched path compared with
the captured reference's FP32 operand. Candidate finals match original32/32; reference finals
changed16/32 under capture, so do not claim original first-boundary causality. New FP32 route
sum is proposed under `PREFILL_MOE_FP32_ROUTE_SUM_ADMISSION.md`, retaining existing Gate C
bounds and a separate two-case numerical test. The original exact test stays FAILED. Avoid
another compiler-tree investigation over64 routed feature-rounding differences. This is
numerical progress only; E1 end-to-end prefill and TTFT are still unproved.

DB583 now closes that supplied-route real-MoE bounded admission (39s worker phase): normal
bit differences shrink to3/2/6/5 per unique feature, worstabs.0078125/p990/mean4.30e-7;
concentrated routing is M1-bit-exact. Every row/aggregate/direct-legacy test passes unchanged
Gate C bounds. HLO binds FP32 sum to actual expert input; groups4/8, no full-weight expansion.
Peak HBM322160640B including reference, all8 clean, exact DB/archive links in HANDOFF and
`../artifacts/prefill-real-moe-fp32-bounded-admission-20260907.json`. E2's arithmetic obstacle
is resolved for these cases, not its performance or full-model integration. Next E1 causal
attention/cache/layer assembly; do not rerun this cleared MoE arithmetic diagnostic.

Causal attention assembly now passes forced32 CPU admission:17 real prompt rows share
structured kv-b tiles, use row-specific sparse counts/scratch and per-query causal bounds
after a whole-block write. Output and final cache match old sequential attention bitwise
across stripe/page boundaries and partial tails; future-row perturbations leave earlier
outputs unchanged. Only two expert8 attention reductions. qkv-a preparation reuses raw
multirow projections, not production exact-convolution association. The new API is unwired;
DSA producer/dual index lifecycle/full layer, real TPU arithmetic and TTFT remain open.
Adversarial review exposed a null-sink health hole: NaN scores could become finite zero
attention. Explicit live operand finiteness now gates health, with query/rotary/old-cache
NaN refusal tests. Corrected CPU32 test passed12.98s, reviewer PASS for CPU persistence.
This closes a bounded assembly defect without a model run; it does not establish TPU speed.

DSA query/key production and separate M64 repair are now CPU-admitted in
`ws32_prefill_dsa.py`:23tests32.02s,17rows with stripe/page/tail boundaries and own-score
exact sets/values/ties. Replacing repaired history leaves prompt selections unchanged;
the selector reads only unrepaired storage. Future rows remain causally invisible.
Raw projection/head-sum association is a new numerical path. FP32 GEMM/GEMV head sums
are bounded against independent FP64 by a dimension-derived forward-error bound, not
declared bit-identical. Existing M64 repair is CPU-bit-identical for supplied normalization.
The complete layer and actual TPU allocation/arithmetic/performance remain unproved.

The complete layer is now assembled in `ws32_prefill_layer.py`: all four static
full/shared-indexer × dense/MoE branches pass CPU32 at17rows/11live; shared+dense matches
the old full one-row layer's output/carried/cache bitwise on synthetic inputs. Router
has exact own-logit noaux_tc semantics with BF16 weights/F32 bias, explicit finiteness and
zero-weight valid dummy routes for padded rows. Split residuals and local incoming health
remain intact. Padded static MoE rows still execute bounded work; narrow final-block
executables are required to minimize that cost. Real-layer TPU admission is next, not
more algorithm design or a full-model launch. End-to-end prefill/TTFT remains unmeasured.

Selected-layer loader now removes a concrete admission cost: the full loader would place
all2310 leaves and local-layout verification would hash entire24GB owner files. The new
`checkpoint/ws32_layer_subset.py` authenticates the same full metadata but reads/hashes only
complete explicitly selected layers, with a mandatory payload budget. Actual file headers,
original tensor-ledger indices and same-byte finiteness remain checked. Its separate types
and evidence scope cannot certify a complete checkpoint. CPU tests instrument file reads;
unselected corruption is explicitly out of scope and still refused by the full verifier.
Reversed CPU32 ownership, sparse local files, selected corruption, hash-correct NaNs,
wrong metadata/header/slots and incomplete layers are covered.39tests21.80s; independent
Astra review PASS for CPU persistence. This enables the bounded layer test without a new
checkpoint copy; it is not a loader-performance or complete-layer TPU result.

Complete-layer admission now has actual candidate/raw-scalar program builders and a
fixed original-array comparator, not another implementation of the layer. Both expose
their actual normalization for separate own-input M64 reconstruction.39 host comparator
tests and the extended CPU32 layer/repair checks pass40total33.76s. Written-row bounds
and exact untouched bytes avoid hiding local errors in huge unchanged cache averages.
Review caught a scope ambiguity: raw scalar DSA is multiply_rsqrt/HIGHEST; new prefill
is divide_sqrt/DEFAULT. Both are now disclosed in the preregistered component contract
and result, with no reference or threshold change. `PREFILL_FULL_LAYER_ADMISSION.md`
states the real-weight/synthetic-state scope and pending worker/HLO/fleet evidence.
No full-layer TPU run or speedup has been claimed.

The hardware admission worker/collector is now connected to the actual layer. Reuses
existing fleet guards/transport/publication instead of a new full-model campaign.
Before-JAX retained-header checks and selected-only payload loading avoid booting the
753B model for a one-layer question. Original-array controller replay covers fixed
synthetic inputs, per-row bounded comparisons and exact causal/cache interventions.
111CPU tests12.68s pass; real production schemas traced on CPU32. Review caught the
block-table rank mismatch before TPU; fixed with actual20-input regression coverage.
Conditional one layer0 launch approved after clean persistence/preflights; no hardware
result or E1 speedup yet. Next layer0/fullDSA+dense, then layer3/IndexShare+MoE.

First layer0 worker failed before load/compile on a digest-kind wiring error: raw JSON
SHA was compared to canonical inventory self-digest. All8 original failure records
generation-verified, all8 cleanup confirmed23:04Z. Corrected via the existing validated
inventory parser, not a changed pin or disabled check;68 CPU regressions pass. This
avoidable harness error would have been caught by exercising the actual inventory
authentication on CPU, which now passes. No layer arithmetic or speed result yet.

Corrected inventory path now loads layer0 on32owners, all selected tensor ledgers
verified; candidate compiled6.44s. All collectives/Pallas counts match, but the linter
omitted known TPU index/layout helper classes and refused before execution. Reuse the
existing guards' distinction with exact layer0 shapes/counts/operands, not a broad
allowlist.70tests and all8 acquired-HLO replays pass, independent review PASS. Both
harness failures were bounded; no hours-long run. Real layer arithmetic is still open.

Layer0 empty and page-boundary cases now pass32owners, original arrays independently
replayed from all8 generation-bound publications. Tail failed at input transfer due to
JAX's numerical cross-host equality (NaN!=NaN), not model arithmetic. Keep the NaN
padding challenge: authenticate fixture bytes collectively, then transfer owned slices.
78CPU tests+review pass; all3 cases still need a complete protected admission.

DB584 completes real layer0/fullDSA+dense bounded TPU admission: all3 cases32owners,
original-array replay, exact causal/cache/health tests, local HLO and47.5MB peak HBM
including reference. Written cache/carried errors0; output worstabs6.10e-5. All8 clean,
terminal archived. This is E1's first complete-layer hardware correctness proof, not
full-model/legacy numerical admission or a speedup. Next layer3 then short decoder.

Layer3 bounded acquisition compiled4.79s and stopped before execution on exact graph
inventory. Actual bias gather lowered to local expert8 zero-insert+F32 sum; no pod-wide
collective. Register this payload plus exact local indexing/scan scratch helpers, not
a blanket allowlist. All8 originals verified,90CPU tests and captured graph replay PASS,
independent review PASS. Numerical route IDs remain exact; no model/bounds change.
This is graph admission progress, not layer3 numerical success or a prefill speedup.

Layer3 subsequently passes HLO/empty-case arithmetic, then refuses a boundary
row4 route-order swap (same selected experts). All32 original bytes agree on
the signature; normal/root cleanup8/8. This was caught within two minutes, not
in a long decoder run. Equal rounded residual cannot localize postnorm rounding.
Next instrument actual router inputs/partials/logits with an attention+router
prefix, excluding MoE; reproduce original swap before identical-input replays.
CPU prefix/replay builders reuse existing kernels and are independently reviewed.
Do not relax ordered IDs, blind-change precision or rerun full layer unchanged.

Router diagnostic now has bounded fleet wiring, original-route preregistration,
same-input M17/M1 replay, checkpoint-bound observed weights and cross-owner replica
checks. Review caught a precompile API wiring defect (plain shard_map lacks lower);
actual CPU compilation now covers it. Independent current-diff review permits one
guarded diagnostic after persistence/preflights. Diagnostic DB fields cannot promote
numerical admission or performance. No new hardware result or end-to-end speedup yet.

DB585 now seals that92s router diagnostic, with32-owner original-route reproduction,
four guarded graphs, original arrays and8/8cleanup. Row4 captured BF16 input is identical;
standaloneM17/M1 projections agree with each other and FP64, whereas the fused scalar
prefix differs by~0.0019 in logits. HLO contains BF16 conversion, so rounding bypass is
not proved. Independent review favors a new completed-BF16 scalar-reference realization,
with original candidate/comparison bounds unchanged and the failedv1 preserved. See
PREFILL_MATERIALIZED_ROUTER_REFERENCE.md. This avoids fitting the candidate to a dubious
reference boundary, but does not establish full-model correctness or prefill speedup.

The first completed-BF16 v2 admission failed the same boundary route order. Offline
FP64 on its actual captured inputs reproduces all17x8 reference routeIDs on32owners;
its newly simplified prefix changes20963 normalized BF16 values versus DB585. Thus
the completed suffix is not the current discrepancy: prefix realization changed.
Both graphs contain BF16 casts; hardware cast omission remains unproved. Preserve
failedv2 and test exact DB585 observable prefix reproduction before any further layer
admission. No threshold relaxation, blind barrier trials or long full-model run.

The next discriminator now uses only TWO existing programs: exact DB585 scalar
prefix/all12 outputs plus completed-input scalar MLP. All17-row per-owner input
and observed-field hashes must reproduce on every host before suffix execution.
This avoids rerunning the candidate/router baselines already sealed in DB585.
183CPU regressions plus4 reuse tests pass, original32-owner fingerprints replay,
and independent Astra current-diff review finds no P0-P2. Hardware result pending;
full-layer reference assembly, short decoder and E1/TTFT remain open.

DB586 seals that exact12-field boundary (85s worker/collector). DB587 then closes
complete real layer3 admission using actual13th PREnorm observation: first12
fields reproduceDB58532owners, original3cases/interventions pass unchanged,
ordered routes exact, output worstabs1.52588e-5 and writtenKV error0. Reference
prefix17 boundary tuples are reused, avoiding redundant compute. Three guarded
graphs, peak347505664B/chip,106s worker/collector, normal/root8/8clean. Original
v1/v2 remainFAILED, no hardware fusion mechanism or performance claim. Alongside
DB584 layer0 this permits cross-layer/dual-cache assembly and short-decoder proof;
do not spend another trial on cleared layer0/3 arithmetic. First CPU compositions
are2→3 and6→7, with two chunks/tails and repaired-index promotion at prompt end.

CPU assembly now passes: separate `runtime/ws32_batched_prefill.py`, actual eight
layers/two chunks and raw decode handoff; production78-layer B17/B11 abstract tree
also passes without weight allocation. One embedding/block, layer-major kernel
calls, only final-live-row head, dual-cache prompt phase and scalar subgroup
all-owner commit. Invalid proposals preserve both caches/frontier and latch false.
This closes the CPU integration part of E1, not protected short-model admission.
B<=32 remains a correctness window, not sufficient measured MoE reuse. Next
protected runner wiring, acquired HLO/memory, own §21 proof and phase baselines.
E6 caveat: atomic rollback retains old/proposed caches; compiler aliasing/copies
must be inspected and budgeted on the ACTUAL full-model graph before numerical
execution. No assumption that old3.36GB long-capacity headroom covers this path.
No candidate timing, TTFT speedup, long-context result or new checkpoint claimed.

Protected acquisition-only integration now compiles the distinct main/tail API
from shared raw weights, reusing completed wk and exact narrow rows. The host
adapter's local budget decision was a real distributed hang risk: independent
review caught it before deployment; health and budget/logging now require all-host
AND before continuation, including final block. Peer-only refusals are tested.
Another review caught loss of memory/provenance on planned HLO refusal; failed
acquisition now writes the complete runner envelope before raising so the existing
uploader preserves it. Actual compiler helper/route-sum/health/cache-allocation
profiles remain unregistered; new mode cannot execute numerically or seal. This
is protected integration progress, not a TPU or performance result. See assembly
document for exact next acquisition and source-derived full-model inventory.

First full acquisition at58c747f9 exposed a HOST validation scaling defect after
the real main78-layer graph compiled: recursive liveness traversal exceeded the
Python stack. Tail and final memory records were not reached. Original20 remote
objects are generation-bound; all8 logs agree, authenticated8clean03:25Z. Local
explicit-stack fix replays the captured main in46.36s (410255instructions,
409607live,787local4/8collectives, all78layers). Exact fusion parameter pruning
and old materializer closure identities remain intact. No numerical execution.
New lesson: publish partial compile/memory state BEFORE parsing, not only at
the final planned refusal. That journal is required before another acquisition;
no unchanged retry. Main HLO has no input/output alias declaration, so E6 remains
a real allocation-admission question, not an assumed cache-copy optimization.
