# Goal — GLM-5.2-FP8 TPU v4: <2 TB, Gate D, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read it and
`docs/glm-tpu-revolution.md` in full, then tails of `HANDOFF.md` and
`docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`; inspect state. Numerical fallback:
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`.

## Storage

Keep live storage in **only** `gs://driftbench-dsv4-uc` at the smallest resumable set. Never touch
TPU/queued-resource infrastructure, especially `db-v4-64-od-qr4`. Inspect dependencies;
delete only name+generation+size+CRC-bound objects. Preserve Git, compact evidence and dependencies.
**Hard ceiling: live bytes must stay below 2,000,000,000,000; >2 TB is unacceptable.** No full-size
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

## Resume — 2026-09-04

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; storage pin
`d33b9a7`. TPU was 8/8 zero-work at pause. V7 install-only passed 8/8,
but its protected M2048 run failed before backend/numerics:
probe initialization omitted `local_device_ids`, entered Cloud TPU autodetection and lacked
`requests`. Preserve the V7 failure record/tag. Local uncommitted V8 adds explicit local IDs plus
`cluster_detection_method="deactivate"`, sealed regression and V2 immutable install paths; focused
suite was 42/42. Do not install/run it until final certificate, review and persistence.

Cleanup is complete; receipts/capsules are committed/pushed. Three phases deleted
528 objects/3,936,205,948,614 bytes: superseded runtimes, unrelated DeepSeek models, an incomplete
PP16 derivative and tar dumps. Tar payloads are unrecoverable. Post-state is 55,266 live
objects/1,945,025,989,379 bytes (1.945 TB/1.769 TiB), leaving 54,974,010,621 bytes below the cap.
Keep canonical GLM 150/755,663,676,164; active direct PP16 68/869,671,243,535; PP16 lineage
metadata 4/11,059,060; all NPZ dumps 4,636/28,158,360,488; results/oracles/repos/evidence.
Active runtime reverified 32 files/6496 tensors. Soft delete is 0. Pre-disable retained versions
were 11,750/3,681,290,733,167; each remains billable until its expiry; phase-1 expires around
2026-09-11; none can be purged retroactively. Keep 77.96-GB layer3 until
recipes are proven. Future large
artifacts must stream or replace a runtime; never accumulate another full copy or breach 2 TB.

After cleanup: finish V8 M2048, localize layer-0 consumer arithmetic, then exact protected 8K. After
Gate D finish §18: protected PP8/PP16, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; base and speculative throughput remain separate.
