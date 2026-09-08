# Goal — GLM-5.2-FP8 TPU v4: efficient end-to-end inference

FULL ACCESS. Continue autonomously to §18 completion under §24. Owner directive
2026-09-07 supersedes §23's deferred-prefill/decode-only completion: hours-long serial prefill
is NOT a finished engine. Preserve historical evidence. Keep <4000 chars.
At start/compaction read this and docs/glm-tpu-revolution.md IN FULL; read HANDOFF and
GATE_D_LESSONS tails, docs/greenfield/ENGINE_EFFICIENCY_AUDIT.md, then inspect live state.

## PIVOT NOW

Native-JAX; no legacy execution. Existing8-host/32-chip pod. Efficient multi-token
prefill AND protected batch-one decode at256K are required. Serial teacher-forcing is a reference,
not production prefill; memory chunking is not token-parallel prefill.
No serial128K/256K run or DB575 rerun.
Pivot to batched prefill: audit/design -> small exact layer/kernel
experiments -> short decoder -> efficient long-context proofs.
Long reference runs require an evidence gap, cost and review.

## Adversarial efficiency audit

Independent gpt-6-astra design/execution audit:
prefill, DSA/IndexShare, attention/MoE, FP8 layout/dequantization, collectives/gathers,
host/device synchronization, loading/caches, compilation/revalidation, storage and benchmarks.
Findings: file/trace evidence, fact vs hypothesis, benefit,
smallest test, correctness/HBM risk, priority and decision evidence.
Maintain ENGINE_EFFICIENCY_AUDIT.md. Resolve major costs or justify measured tradeoffs.
Seek measured performance; no unproved global maximum claims.
Resolve review P0-P2; no repeated cleared-code review or speculative hardening loops.

## Performance and proof

Before optimization runs preregister quantitative128K/256K prefill and time-to-first-token
targets from same-hardware baselines and compute/memory/communication budgets; no post-failure
loosening. Measure prefill, warm TTFT, decode p50/p99, cold load/compile and total request wall
separately. No cache-hit-only claims or device win hiding end-to-end regression.
Base minimum <=200ms/>=4.5 wall tok/s; strong <=125ms/>=8; stretch <=100ms/>=10.
Speculative/effective throughput separate, unmeasured until proved. No guessed speedups.
Smallest decisive test first: reference/CPU -> real one-layer TPU -> short decoder -> long.
Preserve §21/§23.5 numerical contracts, own-score exact DSA/ties, quality/state/load/cache,
checkpoint bytes/hashes, HLO groups/counts, per-chip peak HBM, fresh8-host XPlanes,
profiler-free wall, DB/archive and authenticated8/8 cleanup. Changed prefill earns its own
protected evidence; old serial passes cannot silently promote it. Optimizations default off.

## Safety and persistence

Only gs://driftbench-dsv4-uc (US-CENTRAL2); live <2,500,000,000,000 B; soft delete off.
No full-size safety copies. Before >100GB explain necessity, peak/retained bytes and replacement.
Deletion only reviewed exact generation/size/CRC targets. NEVER manage TPU/node/VM/queued
resource infrastructure, especially db-v4-64-od-qr4. Serialize TPU workflows under both leases.
A free lease is not idle: exact PID/start/boot, libtpu holders, authenticated fleet census.
Use watch_ws32_run.py and WS32_ORPHAN_RECOVERY.md; timeout is not restart authority.
pytest ALWAYS JAX_PLATFORMS=cpu. Freeze model/enforcement source during execution and sealing.
Review, commit/push own branch, verify same-region mirror; cron syncs repos only.

## Snapshot

Worktree /home/gianl/glm-tpu-topology-rewrite, branch rewrite/topology-first-decode.
D/G closed DB567/§22; B/B'/C closed DB568-572. Serial L7 depths1.0/0.0/0.05 DB573-575 sealed.
DB583 MoE/DB584 layer0/DB587 layer3 PASS; CPU composition/78-layer adapter PASS.
133fe71f acquisition: all7 graphs/8 journals archived,8/8clean04:23Z Sep8.
HLO_REFUSED; route/commit/collective/helper/kernel/stack/repair/writer proofs PASS.
Next model-health coverage, memory/numerical admission, OWN §21.
No acquisition/layer0/3 rerun. Targets/row reuse open; pins in HANDOFF.
Efficient L7/L8 open; speedup unmeasured.
