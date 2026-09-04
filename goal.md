# Goal — GLM-5.2-FP8 TPU v4: urgent minimal storage, then close Gate D and §18

FULL ACCESS. Keep this <4000 chars; compaction-safe authority. At start/compaction read it and
`docs/glm-tpu-revolution.md` in full, then tails of `HANDOFF.md` and
`docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`; inspect live state. Numerical fallback:
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`.

## Immediate priority: storage cost

Before creating or executing more Gate-D work, reduce live storage in **only**
`gs://driftbench-dsv4-uc` to the smallest resumable GLM working set. Never touch TPU/queued-resource
infrastructure, especially `db-v4-64-od-qr4`. Inspect live readers/writers and exact dependencies;
delete only name+generation+size+CRC-bound objects. Preserve Git, compact evidence/recipes and
current dependencies. No full-size backup or moving cost elsewhere. Never change soft-delete policy
without specific user approval; report live and
soft-deleted bytes separately. Rsync writes only `repos/`; prevent recreation. Before any future
hundreds-of-GB artifact, state need, temporary/retained size and what it replaces.

## Engine priority after cleanup

Close **Gate D**: correct 78-layer decoder at 8K with exact tokens/DSA order, state/cache integrity,
local collectives, HBM, trace and wall. Plan adjudication, long contexts and §18 follow.

## Hard invariants

Native-JAX greenfield only; legacy `tpu-inference` is oracle/utilities only. Never create/manage
infra. Serialize TPU work under both leases; every protected run ends with authenticated 8/8
zero-work census. Evidence is append-only,
fail-closed and SHA/code/plan/input bound. CPU/HLO/labels/throughput alone are not proof. Never
weaken history. Optimizations default off.

## Efficiency/review contract

- Smallest decisive test first; instrument boundaries and stop on first invariant failure. Never
  run 8K while a row/chunk probe can decide the hypothesis.
- Use an independent Sol adversarial reviewer before persistence, install, execution or destructive
  storage apply; resolve every P0–P2. Fable-max may replace it when available.
- Cron `/home/gianl/bin/sync-glm.sh` runs every 5 min. Commit/push/same-region mirror; verify origin
  plus US-CENTRAL2 mirror before protected work. Never use EU.
- Report result, percentage and blockers plainly; never claim unproven.

## Resume checkpoint — 2026-09-04

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`; pushed HEAD
`43ba8f7c477d47829b520d4467b8baad5cfb9507`. TPU is 8/8 zero-work. V7 install-only passed 8/8,
but its protected M2048 run failed before backend/numerics:
probe initialization omitted `local_device_ids`, entered Cloud TPU autodetection and lacked
`requests`. Preserve the V7 failure record/tag. Local uncommitted V8 adds explicit local IDs plus
`cluster_detection_method="deactivate"`, sealed regression and V2 immutable install paths; focused
suite was 42/42. Do not install/run it until cleanup, final certificate, review and persistence.

Live bucket inventory is 5,877,908,918,585 bytes. Keep canonical GLM source (755,663,676,164) and
direct PP16 runtime `...qkv_direct_pp16_20260827T164842844148623Z`
(869,671,243,535), plus compact lineage/results/oracles. Generation-pinned cleanup preparation covers
three superseded runtime payloads, three unrelated DeepSeek models and 68 SICK/INFRA tarballs;
expected removal is 3.677 TB while retaining old PP16/PP8 root metadata. Keep 77.96-GB layer3
packs until recipes are separately proven. Current soft-deleted inventory is 4,049,225,835 bytes;
new deletes remain billable seven days unless the user specifically authorizes a policy change.

After cleanup: finish V8 M2048, localize layer-0 consumer arithmetic, then exact protected 8K. After
Gate D finish §18: protected PP8/PP16, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; base and speculative throughput remain separate.
