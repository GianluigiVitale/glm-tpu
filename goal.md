# Goal — GLM-5.2-FP8 TPU v4: finish the engine

FULL ACCESS. Finish §18 under §25. <4K.
At start/compaction read this, docs/glm-tpu-revolution.md IN FULL, HANDOFF head,
GATE_D_LESSONS tail, docs/greenfield/ENGINE_EFFICIENCY_AUDIT.md; inspect live state.

## OWNER PIVOT — freeze performance, finish the project

Accept DB603 prefill/decode speed; freeze its native-JAX WS32_2D implementation at
7456bf6433e1dce966670deb252f4c64bbc5f432 as the completion baseline. No legacy execution
or serial teacher-forcing. STOP throughput tuning, key4096 trials, window searches,
optimization benchmarks; 10K tok/s, 500 milestone, strong/stretch decode targets are NOT
gates. Preserve historical target files/receipts; §25 supersedes their requirement.

DB603:2034tokens/78layers,B128/B114,16calls,31.950s/63.661prefill tok/s;
decode131.433/134.195ms p50/p99,7.608wall tok/s. SHORT-CONTEXT only, not promised
8K/128K/256K rates or deliveredTTFT. Slower scaling does not reopen optimization.
No quality, numerical, integrity, memory, locality, provenance or review waiver.

## Exact remaining work

1. D/G closed DB567/§22; DB603 own2K20/20. 2K doesn't prove truncating top2048;
   never inherit old8K witnesses. Receipts: docs/artifacts/.
2. Complete this frozen batched path's own8K §21 numerical proof.
   DB610 own2K SEALED20/20. Corrected8K at8f919277 fails token11 onall8; DSA/cache pass.
   Live32 agreement reaches producers0..2; first gap producer6. STAGED on CPU, tests PASS:
   layers0..6 core, exact observer (byte parity vs production), two-branch protocol/driver.
   No blindfix; see docs/greenfield/PREFILL_HISTORY_FRONTIER.md "Next".
3. Prove long-capacity HLO and actual32-chip HBM before execution; run all FOUR 128K
   depths and full256K E0 on THIS batched path (§23.5). Serial DB573–575 are references only.
4. Serving/resume and actual first-token delivery; report input/cache, prefill, warm
   deliveredTTFT, cold load/compile, decodep50/p99, requestwall separately; no cache-hit
   or profiler-contaminated claims.
5. Close every §18 item: evidence, DB linkage, regional archive, authenticated8/8cleanup.
   Base vs speculative rates separate; speculation deferred, not a tuning campaign.

## Resume checkpoint

2026-09-11: GPT review cleared history observer/driver/capture; CPU parity2PASS,
owner/failure60PASS. No TPU proof.98recoverable local copies evicted3.412GB;
free5.54GB, cloud originals retained. See docs/greenfield/PREFILL_HISTORY_FRONTIER.md.
Next: persist/mirror; preregister seven graphs/per-graph memory caps; compile-only
acquire, inspect actualHLO/HBM, wire runtime/entry/collector, ONE bounded
fullhistory diagnostic. Never attribute cause before both original events reproduce.

## GPT-only execution and review

Executor and independent adversarial reviewer: **GPT-6 Astra (ultra)**, separate agents.
GPT models ONLY; no Claude/Fable/Opus/Claude CLI. Overrides older model/reviewer
instructions, not historical evidence. Review new changes/evidence; resolve P0-P2.
No cleared-code rereview or symbolic/precision archaeology. Smallest
decisive test first; reuse originals. Observability docs under docs/greenfield.
Changed paths earn their own correctness/HLO/HBM evidence. Defaults stay off.

## Safety and persistence

Existing8hosts/32v4chips ONLY. NEVER manage TPU/node/VM/queued resources, esp.
db-v4-64-od-qr4. Only gs://driftbench-dsv4-uc (US-CENTRAL2), live<2.5e12B
(2.043e12B 09-09), softdeleteoff. Recheck5.07GB controller launch floor.
Exact eviction receipts: docs/artifacts/db602-db609-local-copy-*.
No full-size copies; >100GB needs peak/retained explanation.
Delete only reviewed exact generation/size/CRC targets. Serialize under BOTH leases.
watch_ws32_run.py/WS32_ORPHAN_RECOVERY.md; prove PID/start/boot/libtpu ownership.
Timeout ≠ restart authority. pytest ALWAYS JAX_PLATFORMS=cpu.
Freeze model/enforcement source during execution/sealing. Review, commit/push own
rewrite/topology-first-decode branch, verify regional mirror; cron syncs repos.
