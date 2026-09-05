# Goal — GLM-5.2-FP8 TPU v4: <2 TB, Gate D, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read it, `docs/glm-tpu-revolution.md` in full
(incl. §21), tails of `HANDOFF.md`, `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS,
NUMERICAL_CONTRACT}.md`; inspect state. Numerical fallback:
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`.

## Storage

Live storage only in `gs://driftbench-dsv4-uc`, smallest resumable set. Never touch TPU/queued-
resource infra, esp. `db-v4-64-od-qr4`. Delete only name+generation+size+CRC-bound objects
after dependency inspection. **Hard ceiling: live <2,000,000,000,000 bytes.** No full-size backup
or moving cost elsewhere. Soft delete off since 2026-09-04; report live and retained soft-deleted
bytes separately (3.681 TB retained expires per object, phase 1 ≈2026-09-11). Rsync writes only
`repos/`. Before any >100-GB artifact state need/size/replacement. Keep canonical GLM (755.66 GB),
active direct PP16 (869.67 GB), 77.96-GB layer3, lineage/results/oracles/repos/evidence until
recipes are proven. Census 2026-09-04: 1,945,027,232,816 live bytes.

## Gate D (contract = spec §21, 2026-09-05)

78-layer decoder at 8K: exact raw tokens vs sealed legacy oracle; within-engine exact DSA tie order
vs canonical top-k of own scores; cross-oracle selected sets exact or boundary-explained with
`eps` = oracle's own error vs an independent FP32 CPU reference, a pre-registered absolute cap
(DB421 0.003605) and an ambiguity-band swap bound; systematic-bias rule (mean cap 0.000965); bounded
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
  protected work. Never use EU. Re-version install chain only on reviewer finding. Claim nothing
  unproven.

## Resume — 2026-09-05 00:40Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`. V11 (unit-
extent validator accepting real V10 HLO `a96ff87a…9ab2`, V4 paths, fixture, 69/69, Fable review
+ delta clean) is in the commit carrying this goal; its M2048 install/run is DEFERRED under §21.
Preserve V10 evidence. Workers 1–7 clean at `b7708936…`.

Grounds: `greenfield_ws32_short_decoder_8k_numerical_20260826T213125786075567Z` and
`…_20260827T011711674195301Z`: exact 20/20 tokens, state/cache, event 0; HBM 26.4 GB/chip; p50
≈127–129 ms/token; sole refusal seven event-1 selected-position swaps (no aligned stats). Signature is
plan-invariant and survived a legacy-exact layer-1 input → maybe a real defect; bias rule decides.
DB421: PyTorch CPU vs greenfield TPU FP32 scorer swap 2/2,048.

Next: (1) confirm archived engine/oracle event-1 score rows (WS32 NPZ `2be686ff…`), sealed legacy
layer-1 cache `d9058cc6…`, FP32 query at 8155 (oracle `79b813da…`); else one bounded capture run.
(2) Offline §21.2 adjudicator with independent reference on the `20260827…` arrays. (3) Fail →
localize layer-1 defect by chunk/row probes, no 8K. Pass → §21.2 observer, review, ONE protected
8K WS32 run closes Gate D. Then Gate G, 128K, 256K, §18.
