# Dense-only canonical row placement — corrective candidate

Authority: goal.md and specification §25. Status: two-layer CPU integration and
bounded compiler-only fleet wiring; numerical parent/collector/admission still
pending, no candidate TPU execution.
This is a correctness intervention, not renewed performance tuning or an accepted
live32 engine. The frozen DB603 baseline is preserved.

## Evidence and decision

DB606 sealed the fixed18-call norm diagnostic on all32 chips. Receipt:
`../artifacts/prefill-dense-norm-db606-sealed-20260909.json`. Its retained-byte
checks reproduced DB605 fields and DB604 cache endpoints before attribution.
Every captured norm field is byte-equal between wide and narrow first128 cases.
Moving identical normalized32-row segments to rows0:32 of the same completed
physicalB128 dense suffix reproduces all128 own-narrow output/health comparisons.
The wide comparison differs by1200 unique BF16 output words, matching DB605.
This localizes the observed difference to dense suffix placement/co-batch
realization; it does not prove a hardware defect or cause of generated token11.

The prior independent reviewer and main analysis selected a direct corrective
candidate instead of another intermediate-projection capture. Current source
receives its own independent adversarial review before deployment.

## Implementation and unchanged boundaries

`glm_tpu/greenfield/kernels/ws32_prefill_dense_canonical.py` reuses
`ws32_prefill_mlp_mapped` with original dense weights and no MoE branch. Four
uniform device-side scan iterations place each32-row segment in a physical128
row input, zero the rest, preserve live-output/finite-health checks and assemble
the original row order. Empty segments still take the uniform path. B114 pads
to128 then crops; its numerical identity is not inherited from B128.

Only the first three dense layers may opt in. Keep B128 host stride, rolled
attention/DSA/cache schedule, MoE panels, FP8/BF16/FP32 boundaries, checkpoint,
decode and numerical contracts unchanged. No serial token-prefill fallback.
Four dense calls replace one for these layers: disclose and measure this
correctness cost; no performance promise follows from CPU equivalence.

## Shortest decisive admission sequence

1. CPU32 compare real scan candidate with four original physicalB128 calls:
   distinct owners, full/partial/empty rows, B114 tails, poisoned padding and
   nonfinite live health. Candidate+reuse tests5PASS23.60s; candidate is one
   pytest case with multiple shapes and an independent original-full-row reference.
   Isolated-owner NaN atrow33 checks later-tile health placement;
   this is synthetic CPU evidence, not actual TPU row-placement reproduction.
2. Reuse the reduced dense0/1 builder, selected checkpoint loader, WK programs,
   protected workflow, retained originals and guards. Distinct candidate profile
   must preserve actual changed HLO/memory before any refusal. Compare one wide
   candidate first128 against retained NARROW row outputs and endpoint caches
   on all32 owners. Do not demand identity to the intentionally corrected wide
   baseline or rerun historical reference/capture campaigns.
3. Only after bounded reproduction: own2K regression including B114 tail, then
   own8K exact raw tokens/cache/DSA under §21. A continued8K failure stops
   escalation and requires evidence-based diagnosis; no authority to broaden
   the intervention to MoE or sweep precision/windows.
4. After own8K: long-capacity HLO/measured HBM, four128K depths, full256K E0,
   serving/resume/actualTTFT, DB/archive/cleanup. Historical results stay historical.

## Current integration — 2026-09-09

Default-off `canonical_dense` now selects the reviewed suffix in the existing
window and reduced dense0/1 builder; unsupported shapes/branches/options refuse.
The full decoder has no opt-in. Actual CPU32 two-layer comparison passed1test
76.16s: original-wide semantic control, four original narrow row outputs/full
health and all six endpoint-cache arrays. This is not TPU placement evidence.

`scripts/greenfield/ws32_dense_canonical.py` supplies the distinct three-graph,
five-call continuation (four original WK preparations + one first128 candidate),
reusing metadata/selected names, compiler jobs, resident-memory budget,
BudgetedCalls and capture. No reference model call is repeated. Completed outputs
are preserved before health/memory/reproduction refusal. The retained mapper
joins four original narrow32 rows and only final saved caches, including full
health. Independent all8 authenticated DB605 bundles each produce96 arrays/
40,965,120 bytes; no fixture-versus-real field mismatch.

41focused CPU tests1.68s cover mapper, actual voted-call/NPZ lifecycle with
fixture compute/counters, refusal preservation, source mutations and invalid
option combinations. Separate earlier29new+8oldworker+4reuse=41tests4.36s.
Production abstract preparation/lowering1PASS49.56s reads only metadata:
55leaves/102,589,760 selected bytes perchip, allinputsabstract, no weight placement.
Candidate raw667699B/SHA d17cbfeaa7a173f5872632c16e4fbcd1de7f73898f0ca96d6d69734b32a41dad;
WK raw hashes remain original. Omitted versus explicitFalse lowering agrees,
candidate scope appears only in debug-enabled IR. Initial test searched plain
IR for debug-only names and failed locally; corrected inspection, no model change.

Final combined new core/source, old worker, actual historical HLO/helper and
reuse regression:104PASS43.69s (overlaps focused selections). Independent final
source/test/evidence review noP0-P2; approved for persistence, not deployment.

Metadata uses its own exact two-file source registration (window + canonical
suffix); every other model path must equal frozenDB603. Historical source guards
and acquired RAW registrations remain unchanged and reject this changed tree.
No original graph is relabelled as the candidate. Later production execution
must receive its own reviewed source/config/graph admission, not bypass these
guards. Actual optimized candidate loops/helpers/collectives/memory, selected
runtime binding, protected parent/collector and generation-qualified archive
remain before one bounded TPU reproduction. No full8K retry yet.

No TPU candidate run,8K fix, new throughput result or complete engine is claimed.

## One changed-graph acquisition, before numerical execution

Main reasoning and independent reviewer select ONE abstract-input acquisition
of `dense01_canonical`, not another baseline or instrumentation capture. The
correction introduces suffix loops/output stacks whose actual optimized shape
cannot be inferred safely from the old ENTRY-only suffix inspector. Preserve
the actual full helper inventory once, then reuse the physical/kernel/copy
checks plus the two fixed-four-iteration suffix loops and their output stacks.
No symbolic arithmetic campaign, WK acquisition or selected weight read.

`ws32_dense_canonical_compile.py` adapts the existing metadata-only preparation,
compiler writer/journal, voted worker, budget campaign, exact generation-qualified
publisher/collector and protected FP8 wrapper. Distinct kernel/protocol/profile;
one graph, five files perhost,64MiB/rank ceiling (512MiB fleet originals),
900s worker ceiling. Controller collection requires512MiB+1GiB free; allow up
to2GiB for complete archive including wrapper copies/DB, not a checkpoint copy.
All8 source/metadata/pin checks precede distributed initialization. Original
graph/memory evidence is preserved before validation. Zero executable calls;
correctness/score/latency NULL. `preserved_pair` is the reused graph-set receipt
field, not a claim that two graphs were compiled. Numerical execution remains
the separate four-WK-plus-one-candidate continuation above.

42CPU lifecycle/CLI/publication/fleet/SQLite tests pass2.93s, including original
compiler mode, partial/memory/peer refusals and generation-size-before-download
checks. Independent acquisition diff review noP0-P2, conditional persistence/
bounded acquisition after production lowering and fresh preflights. Production
adapter preparation/lowering passes1test135.81s:55abstract leaves/102589760Bperchip,
registered raw667699B/d17cbfea, no payload/device_put/WK jobs. Both original WK
hashes also match in regression. Adjacent budget worker/campaign/reuse34tests
pass120.74s. CPU suites ran concurrently; no TPU compilation or timing claim.
