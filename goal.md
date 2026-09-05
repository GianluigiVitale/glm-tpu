# Goal — GLM-5.2-FP8 TPU v4: <2 TB, Gate D, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read it, `docs/glm-tpu-revolution.md` in full
(incl. §21), tails of `HANDOFF.md`, `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS,
NUMERICAL_CONTRACT}.md`; inspect state. Numerical fallback:
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`.

## Storage

Live storage only in `gs://driftbench-dsv4-uc`, smallest resumable set. Never touch TPU/queued-
resource infra, esp. `db-v4-64-od-qr4`. Delete only name+generation+size+CRC-bound objects after
dependency inspection. **Hard ceiling: live <2,000,000,000,000 bytes.** No full-size backup
or moving cost elsewhere. Soft delete off since 2026-09-04; report live and retained soft-deleted
bytes separately (3.681 TB retained expires per object, phase 1 ≈2026-09-11). Rsync writes only
`repos/`. Before any >100-GB artifact state need/size/replacement. Keep canonical GLM (755.66 GB),
active direct PP16 (869.67 GB), 77.96-GB layer3, lineage/results/oracles/repos/evidence until
recipes are proven. Census 2026-09-04: 1,945,027,232,816 live bytes.

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

- Smallest decisive test first; stop on first invariant failure; prefer offline adjudication of
  archived arrays over TPU runs.
- Adversarial reviewer = separate Fable 5.1 (high) agent (replaces Sol) before persistence,
  install, execution or destructive storage apply; resolve every P0–P2.
- Cron `/home/gianl/bin/sync-glm.sh` every 5 min; verify origin + US-CENTRAL2 mirror before
  protected work. Never use EU. Re-version install chain only on reviewer finding.

## Resume — 2026-09-05 03:00Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`. V11+§21 at
`9659af7`/`dabfb3c`. M2048 DEFERRED. Workers 1–7 clean at `b7708936…`.

Result (spec §21.5, `docs/artifacts/gate-d-event1-math-reference-adjudication-20260905.json`,
Fable-reviewed): independent FP64 CPU reference (`scripts/greenfield/reference_cpu/`) gives event 1
legacy +0.112/0.227, engine −0.006/0.113 (eps 1e-5) and +0.061/0.177, −0.057/0.156 (eps 1e-6); §21.2
items 3–4 PASS for the engine under both (κ=2, κ=1). Bias signs convention-dependent; no "more
accurate" claim. Seven swaps = boundary noise within legacy's own error; layer-1 defect unsupported.
WS32 8K runs already had exact 20/20 tokens, state/cache, event 0, HBM.

Next: (1) commit/push/mirror the reviewed reference + record. (2) Implement §21.2
observer mode in the WS32 8K runner (refuse only on token mismatch, within-engine inexactness,
cache/state structure, locality, unexplained first-divergent-event swaps; record later events), CPU
tests, review, commit/push/mirror. (3) ONE protected 8K WS32 run with fresh trace, profiler-free
wall, HBM, DB, archive, 8/8 cleanup → closes Gate D. Then Gate G, 128K, 256K, §18. No TPU before (2)
is reviewed.
