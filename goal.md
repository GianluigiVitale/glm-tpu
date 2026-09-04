# Goal — GLM-5.2-FP8 TPU v4: <2 TB, Gate D, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read it and
`docs/glm-tpu-revolution.md` in full, then tails of `HANDOFF.md` and
`docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`; inspect state. Numerical fallback:
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`.

## Storage

Keep live storage in **only** `gs://driftbench-dsv4-uc` at the smallest resumable set. Never touch
TPU/queued-resource infrastructure, especially `db-v4-64-od-qr4`. Inspect dependencies;
delete only name+generation+size+CRC-bound objects. Preserve Git, compact evidence and dependencies.
**Hard ceiling: live <2,000,000,000,000 bytes; >2 TB is unacceptable.** No full-size
backup or moving cost elsewhere. Soft delete is user-authorized off
from 2026-09-04; report live and retained soft-deleted bytes separately. Rsync writes only `repos/`.
Before any >100-GB artifact, state need, size and what it replaces.

## Then Gate D

Close **Gate D**: correct 78-layer decoder at 8K with exact tokens/DSA order, state/cache integrity,
local collectives, HBM, trace and wall. Plan adjudication, long contexts and §18 follow.

## Invariants

Native-JAX greenfield; legacy `tpu-inference` is oracle/utilities only. Never create/manage
infra. Serialize TPU work under both leases; every protected run ends with authenticated 8/8
zero-work census. Evidence is append-only,
fail-closed and SHA/code/plan/input bound. CPU/HLO/labels/throughput alone are not proof. Never
weaken history. Optimizations default off.

## Efficiency/review

- Smallest decisive test first; instrument boundaries and stop on first invariant failure. Never
  run 8K while a row/chunk probe can decide.
- Use an independent Sol adversarial reviewer before persistence, install, execution or destructive
  storage apply; resolve every P0–P2. Fable-max may replace it.
- Cron `/home/gianl/bin/sync-glm.sh` runs every 5 min. Commit/push/same-region mirror; verify origin
  plus US-CENTRAL2 mirror before protected work. Never use EU.
- Report result and blockers plainly; never claim unproven.

## Resume — 2026-09-04 15:25Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`. V10
`b770893` is committed/pushed/mirrored; Sol cleared P0--P2, 46/46 tests and V3 install passed.
Protected tag `greenfield_m2048_strategy_nd_20260904T143657392861243Z` passed locks, census,
topology, compile/HLO, then failed closed before numerics. Preserve it. Terminal generation
`1788535490689471`, SHA `5335c322…882f4`; exit census 8/8 clean. See
`docs/artifacts/gate-d-m2048-v10-unit-extent-reduce-failure.json`.

Cause: XLA lowers row 0 as exact `[0:1]` slice, u16 bitcast, then dimension-0 reduce with u16 zero
and scalar add. Extent one combines no payloads; the unary-only linter falsely rejected it. The sole
collective remains the intended full-group bf16 all-reduce. Offline V11 accepted exact HLO/hostile
mutations but was reverted pending review and immutable hashes. Fable hit its usage limit.

Cleanup receipts are committed. Deleted 528 objects/3,936,205,948,614 bytes. Baseline is
55,266 objects/1,945,025,989,379 bytes. Keep canonical GLM (755.66 GB), active direct PP16
(869.67 GB), lineage/results/oracles/repos/evidence and 77.96-GB layer3 until recipes are proven.
Runtime reverified: 32 files/6496 tensors. Soft delete is off; 3.681 TB already retained expires per
object, phase 1 around 2026-09-11. Never breach 2 TB; large artifacts must stream or replace.

Next: one reviewed V11 batch—exact unit-extent validator/hostile tests, fresh V4 paths, certificate,
tag and hashes. Test, Sol review, commit/push/mirror, install, then one M2048 rerun. Never reuse V10
tag or run 8K before M2048 localizes layer-0 arithmetic. Then exact 8K and §18.
