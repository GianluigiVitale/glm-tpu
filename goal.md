# Goal — GLM-5.2-FP8 TPU v4 topology-first engine: close Gate D, then finish §18

FULL ACCESS. Keep this <4000 chars; compaction-safe authority. At
start/compaction read it and `docs/glm-tpu-revolution.md` in full, then only the last ~300 lines of
`HANDOFF.md` and last entries of `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`; never reread
whole histories. Inspect live git/run/lease/pod state. If stuck, reread
`docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md` in full
and adjudicate its hypotheses against local evidence.

## Priority

Close **Gate D**: a correct 78-layer decoder at 8K with exact raw tokens and DSA
set/tie order, state/cache integrity, local repeated collectives, measured HBM, fresh trace and
steady wall. PP8/PP16/WS32 adjudication, 128K/256K, speculation and §18 follow; do not divert now.

## Hard invariants

Native-JAX greenfield only; legacy `tpu-inference` is oracle/utilities only. Use only
`db-v4-64-od` and `gs://driftbench-dsv4-uc`; never create infra. Serialize TPU work under both
leases; every protected run ends with authenticated 8/8 zero-work census. Evidence is append-only
and fail-closed with exact SHA/code/plan/input binding. CPU/HLO/labels/throughput alone are not
proof. Never weaken historical evidence. Optimizations default off. Targets: useful <=200 ms/token,
strong <=125, stretch <=100.

## Efficiency/review contract

- Smallest decisive test first; instrument boundaries and stop on first invariant failure. Do not
  run an 8K decoder while a row/chunk probe can decide the hypothesis.
- One batched review covers source, tests, certificate, install and one fresh-tag command. Use Sol
  `gpt-5.6-sol`, thread `01a05206-57b1-7dc3-a49c-a913529b3937`, read-only, with exact verdict lines.
  Fable-max may replace it when available; otherwise Sol is sufficient.
- Cron `sync-glm.sh` runs every 5 min. Keep work committed/pushed and same-region mirrored; use
  `verify_gate_d_same_region_git_mirror.py` immediately before protected work. Never sync to EU.
- Report result, percentage and blockers plainly; never claim unproven.

## Resume checkpoint — 2026-09-03

Worktree `/home/gianl/glm-tpu-topology-rewrite`, branch `rewrite/topology-first-decode`, last pushed
tip `d549cdd…`; pod READY/HEALTHY, no TPU work. The last two 8K runs have exact token/event 0 and
seven identical event-1 swaps. Offline substitution proves the greenfield layer-1 **prompt cache is
sufficient** for those swaps, not decode-side exactness (`DECODE_SIDE_EXACTNESS_NOT_PROVEN`); the
legacy layer-1 cache differs on 8155/8155 rows.

Active test is one 4-chip chunk-0 legacy-prefill probe; row 0 decides geometry. Sol cleared its
science/provenance chain and rewrite-mirror adapter. V1 installed without launch, but tag
`…122749038361895Z` failed closed before TPU: wrapper confused canonical manifest self-hashes with
raw JSON SHAs (`574f…` vs `bd06…`; legacy `d905…` vs `c11b…`). Published/local tensors are exact.
Tag burned; remote vacant. The narrow fix keeps both hash domains, has a real-artifact regression,
31 CPU tests passing, and preserves v1 while staging launcher v2 (`1eb43b8f…`). Fable-max is
quota-exhausted. Exact next: one Sol batch; if approved commit/push/mirror, install v2, 8/8 preflight,
execute fresh tag `…130504318767505Z`. Then integrate only a proven geometry or use archived HLO/
arrays to localize the next boundary. Do not launch 8K yet. Gate D open.

## After Gate D

Finish §18: protected PP8/PP16 measurements, WS32 result/rejection, 128K four-depth, 256K E0,
DB/archive/clean fleet; report base and speculative throughput separately.
