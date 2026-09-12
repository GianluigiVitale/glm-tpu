# Long-prefill phase residency — 2026-09-12

## Current — all four batched128K depths sealed (DB616–619, 2026-09-12)

DB619 depth0.95 returns289958 on8/8; full127363-token prefill2801.698858s
(45.459204tok/s), decode144.204211ms p50/147.111221ms p99/6.934610walltok/s.
Actual32-chip peak28,511,790,592B, minimum4,502,608,384B headroom;8XPlanes/64cores,
492regionalobjects/2,855,422,796B, exact generation readback and normal/root8clean.
Original controller33724 exited143; all8 original workers survived and finished.
Monitor14241 authenticated their identities and two idle observations. Recovery
45905 sealed at933d18ca and exited0; execution69b6e142. NO model rerun.
Receipts: docs/artifacts/prefill-delivery-db619-{sealed,recovery}-20260912.json
(repo-relative); recovery capsule preserves20 originals/156172B, including SSH
publication receipts concatenated without newlines, parsed without reupload.

NEXT: restore6GiB local launch floor (2,596,892,672B free after seal) using exact
archived copies, then ONE full256K E0 via PREFILL_LONG_LAUNCH.md. No further128K,
new acquisition, tuning or model-math work. HF-card benchmarks, request/resume/
TTFT and final delivery remain open. Paid judge budget still unapproved.
Older dated entries below are preserved history, not new work queues.


## Current — third batched128K depth sealed DB618 (2026-09-12)

Depth0.05: all8 return824794; full127363-token prefill2797.210928s
(45.532140tok/s), decode143.679207ms p50/145.635190ms p99/6.959949walltok/s.
Actual32-chip peak28.512GB, minimum4.503GB headroom;491regionalobjects/
2,855,388,625B, DB618 and normal/root8clean verified. Original controller exited0.
Execution pinfff29f918d52dd4a7afbb4e3d760e8694972f58e. Receipt:
docs/artifacts/prefill-delivery-db618-sealed-20260912.json (repo-relative).
DB616/617/618 close3/4 depths (1.0/0.0/0.05), NOT256K/card parity/serving.
36 exact archived local copies evicted3,914,678,574B;6,606,495,744B free.
All cloud originals/weights/primaryDB kept; old compiler originals and phase
rank0 partners retained locally. Review/application: docs/artifacts/
delivery-db618-local-copy-{review,eviction}-20260912.json (repo-relative).
Existing leased unlink engine unchanged;63CPU checks pass0.79s. Adversarial
self-review only, no P0-P2. Initial review refused a busy lease without writes;
retry acquired both leases. No protection bypass or source change in execution.
Next ONE128k_d0_95 after fresh6GiB/census checks; full256K/card/serving follow.
No new model/math/acquisition campaign.
Paid judge budget remains unapproved; no paid calls, long tests unblocked.

## Prior — second batched128K depth sealed DB617 (2026-09-12)

Depth0.0: all8 return705269; full127363-token prefill2803.147036s
(45.435719tok/s), decode144.292427ms p50/145.510380ms p99/6.930371walltok/s.
Actual32-chip peak28.512GB, minimum4.503GB headroom;491regionalobjects/
2.852GB, DB617 and normal/root8clean verified. Original controller exited0.
Execution pin0e9e3766. Receipt:
docs/artifacts/prefill-delivery-db617-sealed-20260912.json (repo-relative).
DB616+DB617 close2/4 depths (1.0/0.0), NOT256K/card parity/serving.
24 verified local copies evicted3.571GB;6.914GB free, all cloud originals kept.
Existing leased engine unchanged;52CPU checks pass0.73s. Exact restore records:
docs/artifacts/delivery-db617-pp8-copy-review-20260912.json (repo-relative);
application: delivery-db617-pp8-copy-eviction-20260912.json in the same folder.
Next ONE128k_d0_05 after fresh floor/census; depth0.95/full256K follow.
No new model/math/acquisition. Paid judge budget pending; long tests unblocked.

## Prior — first batched128K depth sealed DB616 (2026-09-12)

Depth1.0: all8 return891482; full127363-token prefill2798.276859s
(45.514796tok/s), decode145.531280ms p50/148.028088ms p99/6.871375walltok/s.
Actual32-chip peak28.512GB, minimum4.503GB headroom;491regionalobjects/
2.850GB, DB616 and normal/root8clean verified. Execution pinff101b8d.
Receipt: docs/artifacts/prefill-delivery-db616-sealed-20260912.json (repo-relative).
One of four depths complete, NOT full256K/card parity/serving completion.
15 archived local copies evicted4.379GB;7.239GB free, cloud originals retained.
41CPU eviction checks pass0.71s; exact restore generations in
docs/artifacts/delivery-archived-trace-copy-review-20260912.json (repo-relative).
Next ONE128k_d0_0 through PREFILL_LONG_LAUNCH.md; no new model/math/acquisition.
Paid benchmark-judge budget asked, not approved; it does not block long tests.

## Prior — long entry and original phase sealing integrated (2026-09-12)

See docs/greenfield/PREFILL_LONG_LAUNCH.md (repo-relative): actual five-context
CLI,42+2 original preparation calls and32-owner memory join now connected.
76CPU checks pass18.03s; final22 pass11.35s (overlap). No long-model result.
31 archived local copies evicted2.139GB; cloud/weights/primaryDB retained.
Next persist/mirror, postcommit source checks, then ONE128k_d1_0 protected run.
No further helper/research/unchanged acquisition campaign. Earlier next lists
below are preserved history; allfour128K/full256K/HF-card/serving remain open.

## 2026-09-12 — bounded long-phase collection connected (CPU only)

Long preparation now publishes48 exact originals/rank (two phase records,
42 call records, four WK HLO texts), compressed and generation/CRC/SHA-bound.
The existing materializer explicitly accepts nine graphs plus these originals;
the existing EXIT uploader has a long-only hook. Historical seven-graph and
256MiB diagnostic rules are unchanged. No weights or full-size copies.
Per-file caps sum136MiB/rank, outer rank ceiling160MiB; collector bounds inflation
and refuses missing/extra/generation-conflicting data. Shared WK graphs retain
the existing single-gzip layout. This is transport, NOT a numerical/HBM verdict.

Next: replay original phase calls/overlay/32-owner memory in the sealer and wire
the actual outer request plus worker entry (both still reject long profiles).
Check producer/pre-write and whole-run storage bounds, then fresh6GiB launch
headroom/fleet census. No new acquisition, tuning or model-math campaign.
Allfour batched128K/full256K, HF-card benchmarks and serving/TTFT remain open.
Prior next-step entries below are history. Source/tests and limitations:
docs/artifacts/prefill-delivery-phase-transport-local-20260912.json.


## 2026-09-12 — long companion/request identity; outer launch still closed

Fixed long workload/owned-plan journal and fresh actual optimized-HLO inspection
now share worker/sealer helpers. Original RAW/source pins remain mandatory;
no optimized debug normalizer, inherited numerical PASS or model math change.
Ten retained capacity companion graphs pass actual writer/JSON/sealer replay;
111 focused CPU checks,55 publication/journal/checkout and7 final request checks
pass (overlap; retained-graph replay predates current-pin/basis annotation).
Current-production CPU lowering passes both capacities/ten graphs in348.16s:
observer/decode differ only in embedded debug info (234 bodies each); all
non-debug Mosaic bodies and surrounding RAW match. Current literal hashes
registered separately; no runtime normalizer or historical request exemption.
Receipt: docs/artifacts/prefill-delivery-companions-local-20260912.json.

NEXT: connect fixed outer request and nested
delivery_wk.rankN/delivery_decode.rankN upload/collection/
sealer replay with32-owner memory binding. Existing42-call full-census planning
is391,263,600B across8hosts, exceeding old256MiB diagnostic collection cap;
use explicitly bounded original publication, not a blanket cap increase.
Outer entry remains disabled. Fresh read-only root fleet census06:18Z:8/8idle;
localfree4.898GB below6GiB launch floor. No TPU workflow/new speed/long pass.
Then actual allfour128K/full256K, HF-card benchmarks, serving/resume/TTFT and
DB/regional archive/8clean. DB610 short-context62.761prefill/7.660decode unchanged.

Status: CPU-tested, opt-in host components; NOT a protected long-model run or
measured TPU memory fit. Base pin `e602465c73cceb410ff1f7134bce8357bfbc41b7`.
Authority remains §25/§26: fix capacity and deliver; no throughput/math search.

## Latest — deferred decode placement and calls use existing protection

The actual long branch now preflights the verified overlay's per-owner tensor
bytes against all-live arrays and allocator peak, allowing two overlay payloads
for placement/assembly while keeping1GiB reserve. It performs the unchanged
hash-verifying loader, checks actual completed per-owner peaks, and votes the
load, weight transition and phase publication before any distributed successor.
This is a capacity allowance, not a new overlay/checkpoint copy.

Both exact decode materializer calls now use existing BudgetedCalls with actual
all-live/compiled output/scratch/code and pre/post owner/peak checks. Original
BF16 completion precedes promotion; the first call budgets one resident program,
the second two. All code roots/cache release is voted. Small output schemas
survive postflight refusal; values are not copied or independently replayed.
The historical default helper body/call stack remains intact behind an explicit
long-only delegate, avoiding a default compiler-identity regression.

91CPU tests19.43s pass, including fixture materializer/HBM plus existing realCPU
phase/ownership tests. This does not prove TPU peak, model values or long success.
Self-review only. Receipt: prefill-delivery-decode-preparation-local-20260912.json.

Still required before enabling launch: fixed request/journal/owned plan, actual
numerical caller and companion graph admission, nested WK and decode preparation
publication/collection/sealer replay, and fresh6GiB/32-chip live guards. Both
new phase records remain staged originals, not yet a protected remote schema.
No renewed compiler-only acquisition or symbolic/numerical archaeology.

## Earlier — phase loading staged in the original worker

The long branch of `run_short_decoder_ws32.py` now owns the raw checkpoint via
PhaseWeights instead of loading the dense overlay early. It invokes the new
`ws32_delivery_wk.prepare`: exactly two original compiler jobs, all42 completed
WK calls through existing HistoryCalls/BudgetedCalls, all-live memory/owner/
reserve checks, finite replicated outputs, original compiler reports and
append-once call records. The helper returns metadata only; its executable/JIT
roots are released before prefill. No full WK tensor archive or new checkpoint.

The live builder selects the same frozen options as abstract preparation.
E0 wraps one consumed-state program, compiles once and publishes both roles
with the identical compiled object. Existing128K keeps B128/B114, no donation.
After actual prefill returns, borrowed raw/WK roots and prefill code are cleared;
the original verified overlay loader and exact decode materializer are invoked.
Their original bodies are preserved by AST comparison, not rewritten arithmetic.
Separate base-load, WK-preparation, overlay-load and decode-preparation wall
fields prevent cold work from disappearing into prefill or decode rates.

This is STAGED integration, not an enabled launch. Existing outer short-profile
guards deliberately still reject it. Before enabling: fixed long request and
journal identity; actual numerical caller HLO identity; WK plus exact/observer/
decode/cache companion graph inventory; uploader/collector/sealer replay of
`delivery_wk.rankN`; explicit owned plan publication; voted memory checks around
deferred overlay and exact decode preparation. The existing exact materializer
still has its historical calls, NOT the new WK budget. Do not infer those calls
are admitted or phase peak fits merely because prefill/WK checks are present.

Tests:98CPU checks18.56s final batch;148 adjacent checks12.16s earlier, overlap.
Actual42-call host orchestration uses fixture model/counters and original WK
HLO, not full real weights. Separate4CPU-shard tests exercise real replicated
output hashes/finite checks at both dtypes. Failures stop successors, preserve
originals and release code; pre-existing phase directories cannot be overwritten.
Self-review only. Receipt: `../artifacts/prefill-delivery-phase-loading-local-20260912.json`.
No TPU execution, runtime-HBM, new speed or completed long-context claim.

## Current execution/collector boundary — 2026-09-12

The existing `_execute_batched_prefill` now accepts the explicit long workload
and its actual `PhaseWeights` owner. It binds source, original graph bytes and
compiler allocations before the existing fleet-voted preflight. Original128K
keeps non-donating B128/B114; E0 requires the same compiled object for both
B128 roles and the consumed-state adapter. Existing post-output token, physical
owner, measured-peak and full-reserve checks remain, including failure evidence.

The sealer's execution and32-owner memory helpers accept only the trusted
caller's long workload label. They independently validate original allocations,
published plans, role sharing, token binding, budgets and actual owner records.
E0's role map cannot claim two executables while budgeting one. New script
authorities are included in the committed enforcement surface.

163CPU checks pass12.96s. Full127363/262144-ID host schedules run996/2048 fake
model calls through actual loop/publication/validators, with physical114 padding
for the three-live128K tail. Model outputs and runtime counters are SYNTHETIC;
this is neither actual TPU memory nor model correctness/performance evidence.
Preflight failures execute zero model calls; late failures retain completed
execution and memory evidence. Existing real CPU donation/phase tests also pass.
Receipt: `../artifacts/prefill-delivery-runtime-local-20260912.json`.

Outer main/CLI/collector still refuses long launch. Connect raw-only loading,
WK graph evidence/42 voted calls, root/code release, long entry and deferred
decode preparation together. The current component binds literal saved
optimized bytes: a new numerical caller may change debug identity. Handle that
explicitly during entry integration, not via a generic normalizer, weaker hash
check or another acquisition-only campaign. This limitation is not runtime fit.

## Long structural check and publication — 2026-09-12

`scripts/greenfield/ws32_delivery_hlo.py` now reuses the fixed-loop, physical
collective, kernel-interface and FP32 route-sum checkers on all three retained
long graphs. The original short-profile floating-size guard stays unchanged
for historical callers. A separate cache-base lineage check permits only the
registered BF16 KV stack (and E0 flat view), traced to ENTRY tuple field2
through actual storage operands, fusion callers and both conditional branches.
Cache-shaped conversion/broadcast, full floating weights, unbound origins and
full-size update operands refuse. This is not a generic raised size threshold.

Scope is explicitly limited: local schedule, kernel interfaces and full-cache
BASE storage lineage. It does NOT prove bounded update indices/values, complete
health/commit implication, opaque kernel arithmetic, runtime HBM or task quality.
Frozen source/RAW and retained semantic tests remain binding; own runtime
state/cache/DSA/health and actual all-live/peak checks remain required. Do not
restart the short profile's twelve symbolic proofs at each long capacity.

The existing worker evidence writer and sealer replay share this inspector under
the distinct `ws32_delivery_long_phase_v1` name. Both bind fixed RAW and actual
optimized bytes; JSON reports have canonical containers. Original files are
written before refusal. This is graph publication/replay ONLY: numerical entry
still rejects this profile, companion graphs remain unregistered, and reports
explicitly carry `dispatch_authorized=false`. No full request is enabled by it.

Remaining critical path: phase load/WK/borrower release, protected long entry
and companion evidence, actual memory admission, then the four128K/full256K
workloads. Reuse DB615 and original128K rather than reacquiring unchanged graphs.
Tests, exact hashes and limits: `../artifacts/prefill-delivery-hlo-local-20260912.json`.

## Concrete defect and retained evidence

The current `run_short_decoder_ws32.py` loads the StrategyND dense overlay and
materializes the entire exact-DSA decode tree before batched prefill. Prefill
instead uses the raw base view plus only the completed WK repair matrices.
The original worker retains `weights`, `loaded_dense_overlay`, and
`exact_dsa_weights` alongside those inputs. Its original loader and materializer
are not corrupt; their overlapping lifetimes are unnecessary for this phase.

DB610's original all-live census records 1,127,473,664 B/chip outside prefill
arguments. Those old records contain allocation sizes/groups, not semantic leaf
names: the apparent size correspondence is NOT an exact owner attribution.
No old census is retroactively relabelled. A new actual all-live census must
decide whether phase separation removes the observed excess.

DB615 is the actual E0 compiler anchor: scratch 3,372,240,896 B/chip, alias
3,630,978,560 B, code 101,054,464 B. Single graph plus 1 GiB reserve leaves
148,082,688 B before additional live allocations. Its 78 late captures are fixed.
Do not repeat its unchanged compilation or infer runtime fit from compiler fit.

## Implementation and exact phase boundary

`scripts/greenfield/ws32_phase_weights.py` introduces an explicit `PhaseWeights`
owner; it does not load, authorize, launch, or certify checkpoint bytes.

| Phase | Retained weights | Not created / released |
|---|---|---|
| Base load | Original verified raw checkpoint, same array objects | Dense overlay and exact decode tree not loaded/materialized |
| Repair preparation | Raw checkpoint + each producer's completed FP32 WK | Each temporary BF16 WK released after its separate promotion |
| Batched prefill | Raw checkpoint + 21 completed WK | No packed exact QKV/query aliases/head materialization or dense overlay |
| Decode handoff | Shared raw leaves + original loader-verified dense overlay | Old raw dense roots and prefill WK released |
| Decode preparation | Original full exact materializer outputs | Prefill executables/arguments must already be released |

WK preparation reuses `probe_ws32_prefill_layer.build_wk_programs`, including its
feature4 gather, completed BF16 decode and separate FP32 promotion. Compile two
programs once; dispatch each producer's own inputs through the existing admitted,
voted, memory-budgeted call path (21 × 2 = 42 calls). Local post-call checks and
root updates also use the existing voted phase, before any successor collective.
Do not add a new dispatcher.
The same primitives and selected full-vs-standalone materializer evidence exist
in DB612; no new numerical archaeology or full exact-tree capture is necessary.

`begin_decode` rebinds the existing decoder schema without tensor copies. For
the production geometry, exactly 18 raw dense leaves are replaced by 12 overlay
leaves; all other selected base leaves retain object identity. Missing/extra
overlay names refuse before root mutation. The overlay MUST come from the
unchanged checksum-verifying loader. This binder is not a replacement verifier.

The parent must drop its loader dictionaries, old argument tuples and borrowed
roots. This object cannot collect another caller's references. Its named roots
are inputs to, not a substitute for, the all-live census. Failed WK preparation
is terminal, not a partial retry. `begin_decode` is only authorized by the caller
after healthy, complete prefill. It alone does not prove request completion.

## Consumed-state memory adapter

`scripts/greenfield/ws32_owned_prefill_memory.py` is a distinct contract. The
existing host block loop accepts it only through the explicit
`state_ownership_contract`; the default remains the old non-donating budget.

- Read actual compiled argument metadata: exactly all leaves of argument 2 are
  donated, never IDs/count/weights/WK/RoPE; state shapes/dtypes match live inputs.
- Reuse the original physical-pointer/all-live census and allocator counters.
  State allocations cannot overlap nonstate inputs; unknown pointer ownership
  refuses this profile rather than assuming donation is safe.
- Subtract only the active executable's actual alias bytes from its output
  allocation. Preserve live baseline, scratch, all resident code, prior peak and
  the full required reserve. Alias bytes cannot exceed accounted state/output.
- E0 roles may share one compiled object; count its code once only for actual
  Python object identity, never from equal HLO, sizes or cached-wrapper guesses.
  Two distinct compiled objects are conservatively counted separately.
- Replay the budget arithmetic from the record; a declared fit is not a verdict.
  The protected sealer still must bind actual graph/profile/owner evidence.

This interface admits no companion executable inventory. Clear WK/prefill code
at the correct boundary; unlisted code cannot be omitted to fit. The unchanged
old budget still rejects all nonzero alias reports. Neither budget substitutes
for measured post-execution peak HBM on all 32 chips.

## Tests / limitations

Final focused run: **106 passed in 7.71 s**, `JAX_PLATFORMS=cpu`, including:

- Actual production 2,310-leaf name/tree ownership; weak references prove raw
  dense and repair arrays disappear after transition while shared weights live.
  Tiny placeholder base values test ownership, NOT checkpoint contents.
- One real reduced-geometry 32-device CPU WK computation equals the original
  full exact materializer bit-for-bit; production 21-producer shape/sharding
  preparation is abstract and allocates no production weights.
- Actual CPU compiled donation metadata, shared vs distinct executable objects,
  nonstate alias refusal, and the real host block loop consuming cache handles.
  CPU allocator counters are explicit fixtures, NOT measured TPU HBM.
- Prior-peak/reserve/tampered-budget failures and historical adapter/memory tests.

Initial fixture failed before computation because `first_dense_layers=3` exceeded
the reduced one-layer geometry; corrected to 1 without changing model code or
test bounds. Self-review caught unvoted local checks after completed WK calls;
existing fleet phases now gate them. Peer-only refusal tests verify no successor
decode/promote dispatch. Current-chat self-review only, not independent review.
Receipt: `../artifacts/prefill-phase-residency-local-20260912.json`.

## Next — finish integration, not another research/compile campaign

Update2026-09-12: `scripts/greenfield/ws32_delivery_programs.py` now selects the
fixed long programs from the current DB615 source tree. The old128K preparation
selected the older canonical source recipe, which correctly rejects today's
default-off cache additions. An explicit source choice fixes this mismatch;
historical recipes and their refusals stay unchanged. L7 uses original B128/B114
without donation/cache variants; E0 uses the original DB615 consumed-state
program for both roles. This is metadata/abstract preparation, not dispatch
authorization. The eventual worker must compile that E0 object ONCE and reuse
the actual compiled object; equal program identity alone does not save code HBM.

Saved DB615 optimized graph `c11cd29d…096099` passes the reused physical
schedule locally:788 static collectives,2 health votes,78 fixed-four prefix
bodies,3 fixed-four dense bodies,75 FP32 route-sum proofs; no missing/extra
collective. This is not the complete long HLO/health/cache admission or a trace.
Reuse this result; do not reacquire the same executable.

Historical HLO gap, addressed by the long structural component above: the old kernel-interface checker deliberately rejects
every BF16/F32 result at least32×2048×1536 elements. At long capacity legitimate
full KV stacks exceed that short-only bound. Original128K has
BF16[78,256,64,640]; DB615 has BF16[78,513,64,640] and its flat
BF16[2560896,640] view. A regex inventory of these shapes is discovery only,
not allocation/lifetime or cache provenance proof. Keep the old short guard;
any distinct long allowance must be source/graph-bound, never a general larger
floating-weight cap. Reuse existing fixed-loop/interface/physical checks and
frozen semantic tests; no new full symbolic arithmetic campaign.

Update2026-09-12: the consumed-state host record formerly declared no donation,
and `validate_execution_record` always selected the old no-donation schema.
Fixed in the actual adapter and shared validator. `plan.identity` and
`validate_execution_record` accept the explicit caller-provided
`state_ownership_contract`; the owned identity records `donate_argnums=[2]`.
Neither selects authorization from an untrusted record. Default callers retain
byte-identical no-donation identities and reject owned records.

The existing `validate_batched_fleet_memory` now accepts the same explicit
contract, replays owned budgets and joins them to all32 authenticated owners
and all three measured-peak boundaries. Shared-code roles expand only for
comparison with the runner's actual pair of compiler analyses. No peak/reserve,
all-live, slot/process, actual-analysis or lifetime monotonicity check is removed.
The owned report has a distinct schema; it is not a historical memory receipt.

202CPU tests pass8.46s, including real donated CPU handles through the host loop
and full record validator (small states, B128/B114 three-live tail and shared
B128 object). Fleet tests reuse DB615 compiler allocations with explicitly
SYNTHETIC runtime counters; they do not establish long-model HBM fit. Original
short sealer helper remains covered. First batch167PASS/1FAIL caught an existing
test regex expecting a mixed-profile message when the earlier unknown-profile
guard correctly refused; only that test expectation changed. No model/source
math, protected entry authorization or TPU execution. Self-review only.

Remaining integration is the protected entry's explicit long profile, graph
authorization and phase-lifetime orchestration, NOT another memory schema.
Call the shared validators with the profile's ownership contract (None for the
original128K non-donating pair; explicit contract for E0), not a value copied
from worker evidence. Receipt: prefill-owned-record-fleet-local-20260912.json.

1. Connect this owner to the protected long worker: raw-only verified load,
   two WK jobs/42 protected calls, clear WK code, prefill, clear prefill borrowers
   and code, load verified overlay, bind decode, original exact materializer.
   Include phase preparation and decode preparation in honest request timings.
2. Use the distinct owned budget for E0 and bind DB615's actual graph evidence.
   Reuse original 128K non-donating graphs where their all-live budget fits;
   do not reacquire unchanged 128K or imply it needs E0's cache variant.
3. Update long worker and sealer/profile together, bind new phase/HLO provenance,
   preserve originals before refusal, then fresh launch-space/fleet/memory guards
   and the real four 128K depths followed by full 256K E0. No full model is
   authorized solely by these host tests. No fabricated complete-model fit.

No TPU workflow, weight copy, cloud eviction, policy change or infrastructure
action in this step. Local free space remains below the 6 GiB launch floor.
Current performance remains DB610 short2K: 62.761 prefill tok/s, 7.660 decode
wall tok/s; all long tests, HF-card benchmarks, request/resume/TTFT remain open.
