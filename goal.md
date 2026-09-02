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

## Resume checkpoint — 2026-09-02 08:40Z

Tooling worktree `/home/gianl/glm-tpu-gate-d-pp16-numerical`. DSA host-rotary table refuted, tombstoned
(`…8k-refusal-adjudication.json` `2a7c8fb5…`). **Layer-1 frontier certified on CPU**
(`gate-d-layer1-scale-frontier-certificate.json` `e8abfb9b…`): DB548 row `9b52a04e…` and accepted
row `9936ee1e…` are exact functions of the SAME FP32 RMS input (`dense + attention_update +
combined_residual`, unrounded), weight `10e34f4f…`, eps 1e-5, one rounding; they differ
only in `s=rsqrt(mean+eps)`: greenfield s0−4…s0−1 ulps, legacy s0…s0+14. Legacy
reduces RMS variance over `f32[32,6144]{T(8,128)}` dims={1}; greenfield over `f32[1,1,6144]`.

Committed: captured-RMS probe `accepted_split` arm = FP32-carry accepted-schedule arm (FP32 sum,
one FP32 barrier on `[32,6144]`, reduce dims={1}); matcher `fp32_carry_schedule`; tests 7/7. Next: Sol review; merge → rewrite, push, mirror; one sub-minute protected
probe `GLM_GREENFIELD_CAPTURED_RMS_REPLAY=1 bash scripts/greenfield/run_layer0_projection_reduction_probe.sh`
from the rewrite worktree. Control must equal DB548; arm exact ⇔ accepted row; then decoder change, one 8K run.

## Finish (after Gate D)

§18 of the spec: PP8/PP16 protected measurements, WS32 result or evidence-backed rejection, 128K
four-depth, 256K E0, DB/archive, clean fleet, base vs speculative throughput reported separately.
