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

## Resume checkpoint — 2026-09-03 02:30Z (then the HANDOFF tail)

**The blocker is now prefill, not the decode-side norms.** Three executed 8K runs on the accepted RMS schedule:
token at 8,155 exact each; the first (q-a on the 32-row schedule) broke event 0 by one swap, event 1 = 9, reverted;
the two later kept event 0 exact, event 1 = 7, identical set. CPU capsules: the greenfield
layer-1 prompt cache alone reproduces the device selection and the oracle's 7/7 swaps — a **sufficient cause** only
(`DECODE_SIDE_EXACTNESS_NOT_PROVEN`) — and the legacy layer-1 prompt cache captured on TPU differs from it in
**8,155/8,155 rows** (BF16-ulp level, from position 0). Hypothesis: legacy computes prompt rows at
M=2048/owner; greenfield uses one-row decode forms.

Pins (code): tooling `c92b24cf94bdcd4b0d0570aa7530c8b2c7194c34`, rewrite `f204223e4b8c16f657041da870b91d133e1ef663`;
later commits are records only. Mirror/verifier pass; pod READY; no lease.

Next: finish the bounded **chunk-0 legacy-geometry probe** (row 0 decisive; control = layer-0 keys equal DB518).
Source + 28 CPU tests pass; Sol round 32 leaves three open provenance P1s — fix per HANDOFF "02:30Z", then one Sol
batch and one run, **fresh tag**. Do NOT launch the 8K decoder.

## Finish (after Gate D)

§18 of the spec: PP8/PP16 protected measurements, WS32 result or evidence-backed rejection, 128K
four-depth, 256K E0, DB/archive, clean fleet, base vs speculative throughput reported separately.
