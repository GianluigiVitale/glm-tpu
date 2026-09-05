# Goal — GLM-5.2-FP8 TPU v4: long context, §18

FULL ACCESS. Keep <4000 chars. At start/compaction read it, `docs/glm-tpu-revolution.md` in full
(incl. §21–§23), tails of `HANDOFF.md`, `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS,
NUMERICAL_CONTRACT}.md`; inspect state. Fallback:
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`.

## Storage

Live storage only in `gs://driftbench-dsv4-uc`, smallest resumable set. Never touch TPU/queued-
resource infra, esp. `db-v4-64-od-qr4`. Delete only name+generation+size+CRC-bound objects after
dependency review. **Hard ceiling: live <2,500,000,000,000 bytes** (raised from 2 TB 2026-09-05).
No full-size backup, no moving cost elsewhere. Soft delete off since 2026-09-04; report live and
retained soft-deleted bytes separately. Rsync writes only `repos/`. Before any >100-GB artifact state
need/size/replacement. Keep canonical GLM (755.66 GB), direct PP16 (869.67 GB), 77.96-GB layer3,
lineage/results/oracles/repos/evidence. Live 2026-09-05: 1,972,722,227,769 B.

## Achieved

**Gate D CLOSED (§21.6):** `…8k_numerical_20260905T085534575653049Z`, pin `4286509`, DB 567, SUCCESS
`e40760f8…f662`: tokens 20/20 exact, event 0 exact, event 1 == record `4da05468…`, later events
recorded (alarm acked), item 5 inherited from Gate C, p50 130.369 ms (7.67 tok/s), HBM 26.38 GB/chip,
XPlane 8/64/2, censuses 8/8. Gate E met; F at 2K only. M2048 DEFERRED.
**Gate G CLOSED offline (§22):** PP8 2K DB563 245.6 ms vs WS32 2K DB553 122.6 ms, same oracles, both
exact; PP16_LP2 rejected on measured evidence; WS32_2D promoted; §18 amended.
**§23 Step B:** chunked exact prefill (dual repaired-index buffer, chunk+tail graphs) reproduces
every DB567 witness at C=2048 (DB 568) and C=512 (DB 569) — §23.3.1; 128K/256K oracles rebuilt.

## Invariants

Native-JAX greenfield; legacy `tpu-inference` = oracle/utilities only. Never create/manage infra.
Serialize TPU work under both leases; protected runs end with an authenticated 8/8 zero-work census.
Evidence append-only, fail-closed, SHA/code/plan/input bound. CPU/HLO/labels/throughput alone are not
proof. Never weaken history. Optimizations default off. pytest ALWAYS `JAX_PLATFORMS=cpu` (controller
= pod worker 0). Never edit the run worktree during a protected run or seal.

## Decisions

Never block on the owner. When a question arises, take the decision you would recommend — the one
best serving a correct, provable, finished project — record it with its reasoning and the alternative
in `HANDOFF.md`, and proceed: scope, sequencing, cost/benefit, what to build or drop, pod time. The
reviewer, not the owner, gates persistence, install, execution and destructive apply. Never ask.

## Efficiency/review

- Smallest decisive test first; stop on first invariant failure; prefer offline adjudication to runs.
- Adversarial reviewer = separate **Opus 5** agent (replaced Fable 5.1 2026-09-05); resolve all P0–P2.
- Cron `/home/gianl/bin/sync-glm.sh` every 5 min; verify origin + US-CENTRAL2 mirror before protected
  work. Never use EU.

## Resume — 2026-09-05 20:30Z

Run worktree `/home/gianl/glm-tpu-topology-rewrite` (pushed, mirrored); dev worktree
`/home/gianl/glm-tpu-dev` at `737186f`: evidence layout v2 (one gzip HLO object per graph/form,
≈3 GB/run saved) + storage-headroom refusal, UNREVIEWED, unmerged. WS32 checkpoint RETAINED in tmpfs
on 8 hosts (98 GB/host, verified per run; cleanup script exists).

BLOCKER: the §23.8 rotary diagnostic ran on TPU in the 131,072 acquisition and FAILED (45/128 cells
outside κ=2). Next: (1) read its per-cell record; (2) A′ — default-off main-attention host BF16 rotary
table for WS32 (mirror PP8 `main_rope_table_row` + `apply_rotary_fp32_final_round` at the query and
current-key sites); (3) B′ — 8K adjudication of A′ under §21; (4) Step C capacity runs at 131,072 and
262,656; (5) L7 four 128K runs (1.0, 0.0, 0.05, 0.95); (6) L8 256K E0 at 262,656; (7) §18 direct
proof, incl. base vs effective throughput (Gate H).
