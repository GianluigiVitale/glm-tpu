# GLM-5.2 → TPU v4 — /goal RESUME

SOLO, FULLY AUTONOMOUS. Finish **GLM-5.2-FP8** on TPU v4 — and it DOES NOT SHIP at today's decode
speed. **OWNER RULING 2026-08-03: the throughput null is REVOKED as an endpoint.** ~2.2 tok/s/seq
single-stream decode (397-526 ms/step) on 32 chips is wrong-by-inspection; the measured causes are
IN OUR STACK and must be ENGINEERED AWAY: (1) the TP-32 all-reduce floor — 232 launches ≈127 ms/step,
common-mode, 64% of even the DENSE step; (2) sparse selection machinery ~94 ms/step (gathers 66 +
top-k/sort 36) feeding an attend that costs 12.7 ms. **GOAL: cut the sparse decode step ≥2× (≤~200
ms/step, ≥4.5 tok/s/seq single-stream; stretch ≥3×) with EXACT semantics (selected-set bitwise;
re-gate smoke after each landed lever), and sparse must BEAT dense at 256K.** No workaround gating;
root cause fixed + validated; honest nulls only after the levers are genuinely exhausted.

## RESUME PROOF RULE (mandatory first ~30 min)
Trust NOTHING from this file until DIRECTLY OBSERVED: (a) pgrep pod work BEFORE any action; (b) one
health serve (bench_run health probe or 5K needle) proves the stack: manifest VERIFIED + repair
armed + correct=True; (c) re-derive the step baseline from a fresh 20-step GLM_JAX_TRACE capture —
the parser is scripts/analysis/parse_xplane.py; docs/artifacts/*BREAKDOWN* are the reference
numbers. Claims without a fresh observation are hypotheses.

## THE CAMPAIGN (docs/19 levers → now the critical path; docs/18 ladder = design bank)
1. **All-reduce floor** (~127 ms target, helps every config): fuse/reassociate the ~3/layer
   reductions (78 layers); reduce-scatter+all-gather restructuring; combiner/threshold XLA flags
   ladder (env-only first — cheapest A/B); count MUST drop from 232 — verify by trace.
2. **Selection machinery** (~94 ms): fold the selected-KV gather into the dsa_sparse_decode Pallas
   kernel (index-driven DMA — kills the 66 ms materialized gathers, sparse_mla_kernel.py:635);
   fused partial top-2048 (replaces XLA top_k + full position-sort, 36 ms).
3. Each lever: gated env + CPU-exact tests (bitwise selected-set; jaxpr gate-off identity) →
   single-variable metal A/B with a fresh trace (before/after per-category table) → 128K smoke
   (4 mechanism depths) before the next lever. Adversarial review before each land.
BASELINES (fresh-verify per the proof rule): sparse 389.4 ms device / dense 269.3; all-reduce
232/step both arms; per-category tables in docs/artifacts/.

## STATE (2026-08-03; all in RESEARCH_LOG + results.db)
DONE: engine-lottery solved+validated (repair→refuse→relaunch; docs/17); 128K sparse gate CLOSED
77/77; 256K correctness 4/4; GSM8K 196/200=98.0%; AIME 19/19 completed-correct (32K retry was
mid-run when the owner stopped pod work — 1 chunk committed); GPQA 83.9% completed-item (49-trunc
retry PARKED); MTP M2 PARKED. Benchmarks resume AFTER the throughput campaign (same levers make
them ~2× cheaper). E0 traces: both arms, 8 hosts × 15 steps, banked + parsed.

## OBSERVABILITY (standing kit — USE it, docs/10 + suggestions.md doctrine)
GLM_JAX_TRACE (in-worker decode tracing; flight-recorder decode rule request_distribution[0]==
num_reqs); GLM_FLIGHT_RECORDER; parse_xplane.py; mount+manifest keepers (10 s self-heal, touch
/tmp/golden*.json); protection stack GLM_STATE_HASH_REF + GLM_WK_OOB_DIR + GLM_WK_OOB_GOLDEN +
NaN scans (NEVER disarmed on correctness runs); fingerprint pin GLM_EXPECT_CODE_HASH (tip
94b746433); hardened sync_workers (machine-verified 8×).

## HARD RULES + LANDMINES (unchanged, condensed)
gs://driftbench-dsv4-uc only; NEVER create machines/TPUs. ONE VARIABLE AT A TIME; commit+push every
step; agents in worktrees JAX_PLATFORMS=cpu; serialize TPU; owner submits PRs. WORKER-0 = this VM;
bracket pkill patterns ("x[.]sh"); never edit a running script; setsid nohup </dev/null drivers;
GLM_* raylet AND driver; chunked bench commits (--batch-size); ray stop lies — verify pgrep raylet
+ dashboards; /tmp ages out — keepers touch golden files; purge closed-campaign scratch; detached
chains for pod pipelines (zero-gap); freeze-the-wrapper (SIGSTOP) defuses timeouts without loss.
