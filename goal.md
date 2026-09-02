# Goal — GLM-5.2-FP8 TPU v4 topology-first engine: close Gate D, then finish §18

FULL ACCESS: autonomous. Keep this file <4000 chars; it is the compaction-safe authority. At
start/compaction read this file and `docs/glm-tpu-revolution.md` in full, then only the **last
~300 lines** of `HANDOFF.md` and the last entries of `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`;
never reread whole histories. Then inspect live state (git, run dirs, leases, pod). If stuck or
failing, reread `docs/greenfield/compass_artifact_wf-f6f3c189-f49a-5169-bc82-8adefac958df_text_markdown.md`
in full and adjudicate its hypotheses against local evidence.

## Priority

Close **Gate D**: complete 78-layer greenfield decoder at short context (8K) with correct raw tokens,
exact DSA selected sets/tie order, state/cache integrity, no repeated 32-chip layer collective,
measured HBM, fresh trace, steady wall. Everything else (PP8/PP16/WS32 adjudication, 128K/256K,
speculation, §18 finish) comes after and must not consume effort now.

## Hard invariants (never relax)

Native JAX greenfield tree only; legacy `tpu-inference` is oracle/utilities only. Use only
`db-v4-64-od` and `gs://driftbench-dsv4-uc`; never create infra. Serialize TPU work under the pod
lease; every protected run ends with an authenticated 8/8 zero-work census. Append-only evidence,
fail closed, exact SHA binding of code/plan/artifacts. Proof = exact DSA + tokens + integrity on
real hardware; CPU/HLO/labels/throughput alone are not proof. Never modify historical evidence.
Optimizations default-off. Targets: useful `<=200 ms/token`, strong `<=125`, stretch `<=100`.

## Efficiency contract (owner, 2026-09-01)

- **Batch reviews.** One Sol review per milestone covering source + tests + certificate + install
  commands + fresh-tag run command together. Do not request a review per file or per step.
- Sol = Codex `gpt-5.6-sol` sub-agent thread `01a05206-57b1-7dc3-a49c-a913529b3937`; invoke
  `codex exec fork <id> --skip-git-repo-check -c 'sandbox_mode="read-only"' -o <verdict> "<prompt>"`
  (binary in the VS Code Codex extension). Ask for exact verdict lines.
- Mirror = cron `sync-glm.sh` every 5 min; do not add ceremony. Verify with
  `verify_gate_d_same_region_git_mirror.py` only before a protected run.
- Root-cause work first: reuse the installed immutable V1 numerical runtime and existing wrappers;
  add new orchestration only when a run demands it. Prefer diagnosis over new hardening.
- Reports to owner: state result, % and blockers plainly; never claim unproven.

## Resume checkpoint — 2026-09-02 07:00Z

Tooling worktree `/home/gianl/glm-tpu-gate-d-pp16-numerical`. **Bounded TPU result** (tag
`gate_d_layer1_rms_schedule_20260902T061305905714981Z`, diagnostic archive; HANDOFF): with the
accepted `f32[32,6144]{T(8,128)} -> f32[32]` variance reduce and an FP32 carry, the greenfield
layer-1 row equals the accepted legacy row bit-for-bit; the layer-1 one-ULP miss was the
single-row reduce schedule. DSA rotary table refuted and tombstoned.

Staged: decoder flag `rms_accepted_schedule` (default off): every decode-step RMS variance over a
32-row barrier-carried operand; HLO contract (all `rsqrt` 32-row, none scalar);
`--rms-accepted-schedule`; runner env `GLM_GREENFIELD_RMS_ACCEPTED_SCHEDULE`, suffix `_ras`. Tests:
rmsnorm unit, forced-CPU decoder, compile/runner static pass; kernel suites pending.
Next: commit/push; one batched Sol review (decoder change + tests + 8K launch); merge → rewrite,
mirror, preflight; one protected 8K run `GLM_GREENFIELD_RMS_ACCEPTED_SCHEDULE=1 bash
scripts/greenfield/run_short_decoder_gate_d_pp8_8k.sh` from the rewrite worktree; accept only
exact tokens + all DSA contracts + protections + 8/8 cleanup. Then Gate D evidence.

## Finish (after Gate D)

§18 of the spec: PP8/PP16 protected measurements, WS32 result or evidence-backed rejection, 128K
four-depth, 256K E0, DB/archive, clean fleet, base vs speculative throughput reported separately.
