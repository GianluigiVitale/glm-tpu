# Goal — Finish the GLM-5.2 TPU speed comparison

Measure whether GLM-5.2-FP8 on our 32 TPU v4 chips can run faster correctly. Keep winners; reject failures and retain ordinary decode. MTP supplies speculative drafts. Exclude GLM-5.3 repos and cancelled campaigns.

## Current facts

Ordinary: ~14.3 decode tok/s; historical 138.85 prefill at 2,034 tokens. Global-max verifier rejected: R2/R3 code token 5 differs; caches differ. R3 target-only 144–149 ms versus 200–202 ms ordinary; no serving gain. Global-max 2K prefill: 135.13 tok/s, 29/29 tokens; recent ordinary 140.59 (separate runs). Paired longer-context results pending.

Replay `140413Z` and prefill `144410Z` completed with strict collection/cleanup. Active: `perf_public_fused_ep_20260920T151240Z`, source `5e0b0aa4`; all eight workers live at 15:13:14 UTC. Final ordinary pairs plus scheduling/7,671-token controls are prepared, not queued. Authenticate identities/receipts; never duplicate a run.

Read AGENTS.md, HANDOFF.md, docs/release/STATUS.md and docs/perf/MTP_PROGRESS_20260919.md. Source details: KAGGLE_MTP_REVIEW_20260920.md and UPSTREAM_MTP_REUSE_20260920.md in docs/perf. Prior plan: `471d71c9:goal.md`.

## Execute in this order

1. Completed: collect trained prefill and verifier rejection, cache/rollback and HLO/memory evidence. No further unchanged verifier replay. A different greedy token fails exactness.
2. Measure candidates individually: global-max; profiling-off/device acceptance; fused EP MoE versus M8 (small-row padding and prefill); occupancy/route indexing. Resolve page boundaries, v4 alignment, IndexShare, single norm, shard argmax and sparse-adapter applicability with evidence. Mark tested/already present/incompatible/rejected; record independent drafter-validation gaps.
3. Measure prefill at ~2K DB610 and an admitted prompt near 8K with output space. Check tokens/cache health. Wider MTP requires separate CPU/trained/memory gates; it must not delay R2/R3.
4. Compare ordinary, optimized ordinary and admitted R2/R3 on matched prose/code/structured cases: two alternating repeats, profiling off. Include sustained generation and a checkable completed answer. Changed prompts/caps require new pairs. Keep rejected modes visible; unfinished/divergent outputs are not correct.
5. Publish results; clean up. Target >=25% higher suite throughput; 30+ tok/s is aspirational. A tested negative result can complete this goal.

## Stop going in circles

Each experiment answers one question and ends in keep/reject or a specific correction. Allow one corrective retest per failing candidate; then reject for this campaign and disclose unresolved causes. No repeated unchanged tests/runs, broad tracing, or framework work unrelated to the next measurement. After short gates, run answers.

Table per prompt/mode/repeat: input/output counts, prefill rate, TTFT, accepted delivered decode tok/s, paired speedup, acceptance, verifier time, peak HBM, token agreement, termination and answer checks, plus receipt/pin. Aggregate all fixed cases as tokens/total wall time; disclose regressions. Decode wall includes drafting, rejected work, commit/refresh, host votes and delivery. Report load/compile/prefill separately. Update README/progress; keep this file <4,000 characters.

## Authority

Work autonomously; private perf branch. One workload under workload/pod leases on db-v4-64-od, us-central2-b; authenticate all eight hosts idle before/after, respect sync/cron locks, no automatic retries or resource changes. pytest: JAX_PLATFORMS=cpu. No environment upgrades, weight safety copies or frozen MODEL_SOURCE edits. Only gs://driftbench-dsv4-uc; no weights, prompts, credentials or raw DBs in Git. Preserve history and DB616–621. Commit/push; no force or Co-Authored-By. Merge eligible work only after release checks, resolved review findings and verified regional backup; self-review is not independent review. State whether main receives implementation or evidence.
