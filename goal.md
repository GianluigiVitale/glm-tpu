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

## Gate D (contract = spec §21) — CLOSED 2026-09-05 (§21.6)

Exact raw tokens vs sealed legacy oracle; within-engine exact DSA order; cross-oracle sets exact or
boundary-explained vs an FP64 reference (first divergent event only); exact cache/state; no repeated
32-chip layer collective; fresh trace, profiler-free wall, HBM, archive, 8/8 cleanup. Bit-exact legacy
arithmetic NOT required. Gate G CLOSED (§22): WS32_2D promoted; PP16 rejected with evidence.

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

## Resume — 2026-09-05 11:30Z

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`, HEAD after
`579b13f` (sealing pin). **GATE D CLOSED (spec §21.6):**
`greenfield_ws32_short_decoder_8k_numerical_20260905T085534575653049Z`, run pin `4286509`, SUCCESS
`e40760f8…f662`, summary `683fe2e1…08fa`, DB run 567, tokens 20/20 exact, event 0 exact, event 1 ==
record `4da05468…`, later events recorded (alarm acked), item 5 by inheritance (Gate C), dense overlay ON, p50 130.369 ms/token (7.67 tok/s), HBM
26.38 GB/chip, XPlane 8/64/2, censuses 8/8. Gate E met, F not. M2048 DEFERRED.

State: live 1,952,694,918,629 bytes. Streamed WS32 checkpoint RETAINED in tmpfs on 8 hosts (98 GB/host,
verified per run; needed by 128K/256K; cleanup script exists, run it before host maintenance).

Long context (spec §23, reviewed): chunked exact prefill (dual buffer, chunk+tail graphs) coded;
128K/256K oracles rebuilt from DB 403/402 provenance (bit-exact). Indexer rotary stays on device
(host DSA table is tombstoned); main-rotary long-position effect measured in Step B (§23.8).
Next: commit Phase A → 8K acquisition (new schema) → Step B: 8K runs C=2048 and C=512 must reproduce
DB567 witnesses → Step C: 8K prompt at capacities 131072 and 262656 (CAPACITY_MEASUREMENT) → L7 four
128K runs (order 1.0, 0.0, 0.05, 0.95) → L8 256K E0 (capacity 262656) → §18 wording. pytest ALWAYS
with JAX_PLATFORMS=cpu (tests/conftest.py enforces; controller = pod worker 0).
