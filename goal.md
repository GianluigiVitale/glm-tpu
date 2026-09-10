# Goal — GLM-5.2-FP8 TPU v4: finish the accepted engine

FULL ACCESS. Finish §18 under §25. <4K chars.
At start/compaction read this and docs/glm-tpu-revolution.md IN FULL; HANDOFF,
GATE_D_LESSONS tails, docs/greenfield/ENGINE_EFFICIENCY_AUDIT.md, then inspect live state.

## OWNER PIVOT — freeze performance, finish the project

Accept DB603 prefill/decode speed. Freeze its
native-JAX WS32_2D implementation at7456bf6433e1dce966670deb252f4c64bbc5f432 as
the completion baseline. No legacy execution or return to serial teacher-forcing.
STOP throughput tuning, key4096 trials, window searches and optimization benchmarks.
10K prompt tok/s,500 milestone and strong/stretch decode targets are NOT completion gates.
No speed tuning before engine completion.
Preserve historical target files/receipts; §25 supersedes their completion requirement.

DB603:2034prompttokens/78layers, B128/B114 in16calls,31.950s/63.661prompt tok/s.
Decode131.433ms p50/134.195ms p99,7.608 wall tok/s. These are SHORT-CONTEXT
measurements, NOT promised8K/128K/256K rates or deliveredTTFT. Measure/report actual
long-context speeds honestly; slower scaling alone does not reopen optimization.
No quality, numerical, integrity, memory, locality, provenance or review waiver.

## Exact remaining work

1. Resume from HANDOFF and docs/artifacts/prefill-rolled-short-db603-sealed-20260909.json.
   D/G closed DB567/§22; DB603 own2K passed20/20oracle tokens; all29IDs equal DB597.
   DSA scores/order/cacheVALUEbits differ; own checks pass.2K does not prove
   truncating top2048. Never inherit old8K numerical witnesses silently.
2. Complete this frozen batched path's own8K §21 numerical proof.
   DB610 own2K SEALED20/20;62.761prefill/7.660decode tok/s.
   Corrected8K at8f919277 still fails token11 onall8;8/8clean. DSA/cache pass.
   Live32 agreement now reaches producers0..2; first observed gap producer6.
   Bounded0..6 core staged; exact observer/guarded execution next. No blindfix.
   See docs/greenfield/PREFILL_HISTORY_FRONTIER.md (failure linked).
   Reuse evidence/protections; required integration/proven fixes only.
3. Prove long-capacity HLO and actual32-chip HBM before execution. Run all FOUR
   128K passkey depths and full256K E0 on THIS batched path (§23.5 classifications).
   Old serial DB573–575 are references, not coverage of changed prefill.
4. Complete serving/resume and actual first-token delivery; report input/cache,
   prefill, warm deliveredTTFT, cold load/compile, decodep50/p99 and requestwall
   separately. No prefix-cache-hit or profiler-contaminated performance claims.
5. Close every remaining §18 item with direct evidence, DB linkage, regional
   archive and authenticated8/8cleanup. Base vs speculative rates stay separate;
   speculation is unmeasured/deferred, not a new pre-completion tuning campaign.

## Efficient, adversarially reviewed execution

Independent gpt-6-astra review of new changes/current evidence; resolve P0-P2.
No cleared-code rereview or symbolic-proof/precision archaeology.
Smallest decisive test first; bulk compatible checks; reuse saved originals.
Observability: EVIDENCE_MAP.md, GATE_D_OBSERVABILITY_PLAYBOOK.md,
ENGINE_EFFICIENCY_AUDIT.md under docs/greenfield, and docs/suggestions.md.
Changed paths earn their own correctness/HLO/HBM evidence. Defaults stay off.

## Safety and persistence

Existing8hosts/32v4chips ONLY. NEVER manage TPU/node/VM/queued resources, especially
db-v4-64-od-qr4. Only gs://driftbench-dsv4-uc (US-CENTRAL2), live<2.5e12B,
softdeleteoff. No full-size copies; before>100GB explain peak/retained/replacement.
Deletion only reviewed exact generation/size/CRC targets. Serialize under BOTH leases.
Use watch_ws32_run.py/WS32_ORPHAN_RECOVERY.md; prove PID/start/boot/libtpu ownership.
Timeout is not restart authority. pytest ALWAYS JAX_PLATFORMS=cpu.
Freeze model/enforcement source during execution/sealing. Review, commit/push own
rewrite/topology-first-decode branch and verify regional mirror; cron syncs repos only.
