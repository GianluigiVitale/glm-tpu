# Goal — GLM-5.2-FP8 TPU v4: validate and deliver

FULL ACCESS. Finish §18 under §25/§26.
Read this/docs/glm-tpu-revolution.md IN FULL at start/compaction;
HANDOFF head, GATE_D_LESSONS tail, ENGINE_EFFICIENCY_AUDIT.md; live state.
Follow docs/greenfield/DELIVERY_PLAN.md.

## OWNER PIVOT — finish the current engine

Freeze native-JAX WS32_2D speed. DB603 baseline
7456bf6433e1dce966670deb252f4c64bbc5f432; retain tested dense correction DB610.
No legacy execution. STOP throughput/precision searches, alternate plans,
exact-continuation archaeology and bit-matching capture.
Fix only structural defects, execution blockers or material card-quality loss.

DB610 own2K:20/20;62.761 prefill tok/s,130.554ms decode,7.660walltok/s.
Short-context only. No speed/speculation gate.

## Acceptance — original Hugging Face benchmarks

Benchmark: https://huggingface.co/zai-org/GLM-5.2-FP8
Compare GLM-5.2 column, NOT old local builds.
Pin card/footnotes.
Start with card-listed GPQA-Diamond and AIME2026 using existing bench/ tooling.
Match dataset, prompts/template, sampling, token cap, scorer/judge and aggregation.
Disclose protocol gaps/substitutions; no unmatched parity claim.
Preregister samples/uncertainty/deficit/budget before outputs.
Investigate large deficits; no cherry-picking/tolerance fitting.

§26 removes incidental continuation/cross-engine bits as gates. 8K passkey881446
correct; prose differs at token11. Preserve failure, no retroactive PASS or broad
quality claim. Keep checkpoint/scale/load integrity, causal/cache validity,
own-score DSA ties/selection, routing, finite state, topology-local collectives,
actual per-chip HBM, kernel tests and honest evidence. No rounding emulation.

## Remaining work

1. DB612 archived/8clean;8K passkey881446, DSA/cache/state. Not card parity.
2. Batched long-context (§23.5) COMPLETE. DB616–619: FOUR128K depths SEALED8/8.
   DB619 key289958, prefill2801.699s/45.459tok/s;decode144.204ms/6.935walltok/s.
   32-chip peak28.512GB/spare4.503GB;archive/8clean.
   Full256K SEALED DB620:32.157prefill tok/s;162.644/169.402ms p50/p99;
   6.148walltok/s,29.930GB peak/3.084GB spare,8traces/8clean/archive verified.
   NO model rerun. Receipt/limits: HANDOFF head.
   docs/greenfield/PREFILL_LONG_LAUNCH.md: recovery.
3. Execute registered HF-card benchmarks through the native engine. Keep misses
   and truncations; report score gaps/protocol caveats. Diagnose material deficits.
   228 real prompts: configs/greenfield-native-benchmark-protocol.json.
   First native15aefe97 ended before answers; originals collected/8clean.
   Native-only vocabulary HLO lowering fix replays originals. Captured HBM
   refuses262656 all-resident. Benchmark166912 fits all228 prompts+FULL163840cap;
   register new RAW/actual HLO/HBM before ONE new tag. No smaller generation cap.
   HANDOFF head/current receipts govern; no failed-tag reuse or long reruns.
   Healthagent1GiB0..6;7 unchanged. Logs losslessly recovered; fresh6GiB checks.
4. Prove request/resume and first-token delivery; separate input/cache, prefill,
   warm TTFT, cold load/compile, decode and request wall.
5. DB/archive, commands/results/limits, commit/push and8/8clean;
then stop. No tuning.

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
