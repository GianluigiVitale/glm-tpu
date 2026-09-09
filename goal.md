# Goal — GLM-5.2-FP8 TPU v4: finish the accepted engine

FULL ACCESS. Finish §18 under §25; keep <4000 chars.
At start/compaction read this and docs/glm-tpu-revolution.md IN FULL; HANDOFF and
GATE_D_LESSONS tails, docs/greenfield/ENGINE_EFFICIENCY_AUDIT.md, then inspect live state.

## OWNER PIVOT — freeze performance, finish the project

Owner2026-09-09 accepts DB603 prefill/decode speed. Freeze its
native-JAX WS32_2D implementation at7456bf6433e1dce966670deb252f4c64bbc5f432 as
the completion baseline. No legacy execution or return to serial teacher-forcing.
STOP throughput tuning, key4096 trials, larger-window searches and optimization benchmarks.
10K prompt tok/s,500 milestone and strong/stretch decode targets are NOT completion gates.
Do not spend more time improving speed before completing the working engine.
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
   B128 failed token11; live32 matches20/20 diagnostically, never a new baseline.
   DB608 dense0/1 correction matches narrow on32chips;8K fix still unproven.
   Fullmodel optin/CPU tails pass; acquire both graphs, then own2K/8K.
   See docs/greenfield/PREFILL_DENSE_CANONICAL.md.
   Reuse evidence/protections; only required integration or proven fixes.
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

Independent gpt-6-astra reviewer for new changes/current evidence; resolve P0-P2.
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
