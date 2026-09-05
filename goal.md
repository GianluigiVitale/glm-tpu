# Goal — GLM-5.2-FP8 TPU v4: <2 TB, Gate D, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read it, `docs/glm-tpu-revolution.md` in full
(incl. §21), tails of `HANDOFF.md`, `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS,
NUMERICAL_CONTRACT}.md`; inspect state. Numerical fallback:
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`.

## Storage

Live storage only in `gs://driftbench-dsv4-uc`, smallest resumable set. Never touch TPU/queued-
resource infra, esp. `db-v4-64-od-qr4`. Delete only name+generation+size+CRC-bound objects after
dependency inspection. **Hard ceiling: live <2,000,000,000,000 bytes.** No full-size backup or
moving cost elsewhere. Soft delete off since 2026-09-04; report live and retained soft-deleted
bytes separately (3.681 TB retained expires per object, phase 1 ≈2026-09-11). Rsync writes only
`repos/`. Before any >100-GB artifact state need/size/replacement. Keep canonical GLM (755.66 GB),
active direct PP16 (869.67 GB), 77.96-GB layer3, lineage/results/oracles/repos/evidence. Census
2026-09-04: 1,945,027,232,816 live bytes.

## Gate D (contract = spec §21, 2026-09-05)

78-layer decoder at 8K: exact raw tokens vs sealed legacy oracle; within-engine exact DSA tie order
vs canonical top-k of own scores; cross-oracle selected sets exact or boundary-explained with
`eps` = oracle's own error vs an independent FP64 CPU reference, relative caps κ=2 on engine
max/std, reference-band swap bound; bias rule |m_e| ≤ 2|m_o| + 3s/√n; first divergent event only; bounded
internal tensors; exact cache/state structure; no repeated 32-chip layer collective; fresh trace,
profiler-free wall, HBM, provenance, archive, 8/8 cleanup. Bit-exact legacy intermediate arithmetic
is NOT required. Then plan adjudication, long contexts, §18.

## Invariants

Native-JAX greenfield; legacy `tpu-inference` is oracle/utilities only. Never create/manage infra.
Serialize TPU work under both leases; protected runs end with authenticated 8/8 zero-work census.
Evidence append-only, fail-closed, SHA/code/plan/input bound. CPU/HLO/labels/throughput alone are
not proof. Never weaken history. Optimizations default off.

## Efficiency/review

- Smallest decisive test first; stop on first invariant failure; prefer offline adjudication
  over TPU runs.
- Adversarial reviewer = separate Fable 5.1 (high) agent (replaces Sol) before persistence,
  install, execution or destructive storage apply; resolve all P0–P2.
- Cron `/home/gianl/bin/sync-glm.sh` every 5 min; verify origin + US-CENTRAL2 mirror before
  protected work. Never use EU.

## Resume — 2026-09-05 04:30Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`. Pins:
V11+§21 `9659af7`, reference `7764cb9`, §21.2 observer `8d8759b` (reviewed, pushed). M2048
DEFERRED. Workers 1–7 clean at `b7708936…`.

Result (§21.5): FP64 reference: event 1 legacy +0.112/0.227, engine −0.006/0.113 (eps 1e-5);
+0.061/0.177, −0.057/0.156 (eps 1e-6); §21.2 items 3–4 PASS under both; swaps = boundary noise.
WS32 8K runs had exact 20/20 tokens, state/cache, event 0, HBM. Pre-registered record
`docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json`.

Checkpoint: WS32 runtime pack (786 GB) deleted 08-27; re-upload breaches 2 TB. Route: STREAM into
tmpfs (hosts 400 GB RAM, 201 GB /dev/shm; 4 slots = 98 GB/host), sealed manifest/SUCCESS reused
verbatim (lineage `results/greenfield_ws32_runtime_pack_20260815T214050854386790Z`), per-slot byte
identity enforced; PP16 direct stays. Code changed since acquisition pin `04d059b`.

Shm batch pushed (`6832b0e`); tmpfs pack DONE 08:07Z (32/32 slots byte-identical, tag
`greenfield_ws32_runtime_shm_pack_20260905T052030829387724Z`). Next:
(3) compile-only 8K acquisition at HEAD. (4) ONE 8K numerical run with
`GLM_GREENFIELD_WS32_DSA_ADJUDICATION=1`, trace, wall, HBM, DB, archive, 8/8 cleanup → Gate D;
tmpfs cleanup. Then Gate G, 128K, 256K, §18.
