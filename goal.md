# Goal — GLM-5.2-FP8 TPU v4: validate and deliver

FULL ACCESS. Finish §18 under §25/§26. Keep this <4,000 chars.
Read this and docs/glm-tpu-revolution.md IN FULL at start/compaction, then
HANDOFF head, GATE_D_LESSONS tail, ENGINE_EFFICIENCY_AUDIT.md and live state.
Follow docs/greenfield/DELIVERY_PLAN.md, not old next-step lists.

## OWNER PIVOT — finish the current engine

Accept current native-JAX WS32_2D speed. DB603 baseline pin
7456bf6433e1dce966670deb252f4c64bbc5f432; retain tested dense correction DB610.
No legacy execution imports. STOP throughput/precision searches, alternate plans
and exact-continuation archaeology. Archive current diagnostic; no new bit-matching
capture. Fix only structural defects, execution blockers or material card-quality loss.

DB610 own2K:20/20 reference tokens;62.761 prefill tok/s, decode130.554ms p50,
7.660 wall tok/s. SHORT-CONTEXT only; measure long-context rates, never promise.
No speed/speculation gate; slower scaling alone is not failure.

## Acceptance — original Hugging Face benchmarks

Authoritative benchmark reference: https://huggingface.co/zai-org/GLM-5.2-FP8
Compare this port with the GLM-5.2 column, NOT previous local builds.
Pin card bytes/revision and footnotes.
Start with card-listed GPQA-Diamond and AIME2026 using existing bench/ tooling.
Match dataset, prompts/template, sampling, token cap, scorer/judge and aggregation.
Disclose unspecified details/substitutions; no unmatched test labelled card parity.
Old local scores are diagnostics only.
Register sample counts, uncertainty, material-deficit threshold and runtime budget
BEFORE outputs; investigate large deficits. No cherry-picking or tolerance fitting.

§26 removes exact incidental continuation/cross-engine bits as delivery gates.
8K returned correct passkey881446; later prose differs at token index11.
Preserve the old failure; new task-level assessment names §26, not a retroactive PASS.
One passkey is not broad quality. Keep checkpoint/scale/load integrity, causal
masks, cache addresses/validity, own-score DSA ties/selection, routing semantics,
finite state, topology-local collectives, actual per-chip HBM and honest evidence.
Keep bounded kernel tests; no rounding emulation.

## Remaining work — in order

1. DB612 archived/8clean. §26 saved8K task smoke8/8: correct881446, DSA/cache/state;
   Original failure unchanged; receipt in DELIVERY_PLAN.md.
   Worker/sealer use task profile; no prose-only rerun or card-parity claim.
2. Admit batched long-capacity HLO/HBM; run all FOUR128K passkey depths and full
   256K E0 (§23.5). Old serial DB573–575 are references, not new-path coverage.
   Three-graph compiler route staged; tests/budgets in DELIVERY_PLAN.md/HANDOFF.
   Actual HLO/HBM/launch pending; restore fresh6GiB localspace first.
3. Execute registered HF-card benchmarks through the native engine. Keep misses
   and truncations; report score gaps/protocol caveats. Diagnose material deficits.
4. Prove request/resume and first-token delivery; separate input/cache, prefill,
   warm TTFT, cold load/compile, decode and request wall.
5. DB/archive, reproducible commands/results/limitations, commit/push and8/8clean;
   then stop. No bonus tuning.

## Model, safety and persistence

ONLY this chat: GPT-6 Astra High. No Ultra, subagents/external reviewers or
Claude/Fable/Opus/Claude CLI. Adversarial self-review; resolve P0-P2, never call
it independent review. Preserve historical evidence/reviews.
Existing8hosts/32v4 only; NEVER manage TPU/node/VM/queued resources, especially
db-v4-64-od-qr4. Only gs://driftbench-dsv4-uc (US-CENTRAL2), live<2.5e12B,
softdeleteoff. No full-size copies; >100GB needs peak/retained/replacement budget.
Serialize BOTH leases; preserve running jobs. Timeout is not restart authority.
Fresh storage/HBM bounds; exact-generation reviewed eviction; pytest CPU.
Freeze model/enforcement during execution/sealing. Commit/push own branch
rewrite/topology-first-decode and verify regional mirror; cron syncs repos.
