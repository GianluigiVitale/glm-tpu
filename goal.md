# Goal — Finish the GLM-5.2 TPU speed comparison

Measure whether GLM-5.2-FP8 on our 32 TPU v4 chips can run faster correctly. Keep winners; reject failures and retain ordinary decode. MTP supplies speculative drafts. Exclude GLM-5.3 repos and cancelled campaigns.

Editing this file does not launch work. When activated, finish this bounded plan.

## Current facts

Ordinary: ~14.3 decode tok/s; historical 138.85 prefill at 2,034 tokens. Global-max verifier rejected: R2/R3 code token 5 differs; caches differ. R3 target-only 144–149 ms versus 200–202 ms ordinary; no serving gain. Global-max 2K prefill: 135.13 tok/s, 29/29 tokens; recent ordinary 140.59 (separate runs). Paired longer-context results pending.

Replay `140413Z` and prefill `144410Z` completed with strict collection/cleanup. Last recorded run: `perf_public_fused_ep_20260920T151240Z`, source `5e0b0aa4`; eight workers live at 15:13:14 UTC, September 20 (historical status). Final ordinary pairs plus scheduling/7,671-token controls are prepared, not queued. Authenticate identities/receipts; never duplicate a run.

Read AGENTS.md, HANDOFF.md, docs/release/STATUS.md and docs/perf/MTP_PROGRESS_20260919.md. Source reviews and receipts are linked from progress. Prior plan: `471d71c9:goal.md`.

## Execute in this order

1. Collect the existing fused-EP run against M8, including small-row padding and prefill. Finish the short profiling-off/device-acceptance TPU comparison. Reuse recorded decisions for already implemented or incompatible upstream ideas. No new candidate families or literature surveys.
2. Decide keep/reject for each remaining candidate. Require CPU evidence, TPU memory/latency admission and trained correctness before integration. Do not reopen rejected verifiers or pursue wider MTP. A different greedy token fails exactness.
3. Run the prepared real-weight final comparison: ordinary versus admitted optimized ordinary; include R2/R3 only if qualified. Failed MTP must not delay ordinary results. Use fixed prose/code/structured prompts, matched budgets, two alternating repeats and profiling off. Include sustained generation, the separate checkable completed-answer control and ~2K/near-8K prefill with output space. Reuse one model load where practical.
4. Publish results, authenticate cleanup on all eight hosts and finish. Keep ordinary if nothing wins. >=25% higher suite throughput is the success target, not a completion requirement. Do not extend the campaign to chase it.

## Stop going in circles

Each experiment must end in keep/reject or a specific correction. Allow one corrective retest per failing candidate; then reject for this campaign and disclose unresolved causes. No repeated unchanged tests/runs, broad tracing, or framework work unrelated to the next measurement. After short gates, run answers.

Table per prompt/mode/repeat: input/output counts, prefill rate, TTFT, accepted delivered decode tok/s, paired speedup, acceptance, verifier time, peak HBM, token agreement, termination and answer checks, plus receipt/pin. Aggregate all fixed cases as tokens/total wall time; disclose regressions. Decode wall includes drafting, rejected work, commit/refresh, host votes and delivery. Report load/compile/prefill separately. Update README/progress; keep this file <4,000 characters.

## Authority

Work autonomously; private perf branch. One workload under workload/pod leases on db-v4-64-od, us-central2-b; authenticate all eight hosts idle before/after, respect sync/cron locks, no automatic retries or resource changes. pytest: JAX_PLATFORMS=cpu. No environment upgrades, weight safety copies or frozen MODEL_SOURCE edits. Only gs://driftbench-dsv4-uc; no weights, prompts, credentials or raw DBs in Git. Preserve history and DB616–621. Commit/push; no force or Co-Authored-By. Merge eligible work only after release checks, resolved review findings and verified regional backup; self-review is not independent review. State whether main receives implementation or evidence.
