# Goal — GLM-5.2-FP8 TPU v4 topology-first engine: close Gate D, then finish §18

FULL ACCESS: autonomous. Keep this file <4000 chars; it is the compaction-safe authority. At
start/compaction read this file and `docs/glm-tpu-revolution.md` in full, then only the **last
~300 lines** of `HANDOFF.md` and the last entries of `docs/greenfield/{EVIDENCE_MAP,GATE_D_LESSONS}.md`;
never reread whole histories. Then inspect live state (git, run dirs, leases, pod).

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
  (binary under `~/.vscode-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex`). Ask for
  exact verdict lines. Fable CLI review may substitute when Sol is unavailable.
- Mirror = cron `sync-glm.sh` every 5 min; do not add ceremony. Verify with
  `verify_gate_d_same_region_git_mirror.py` only before a protected run.
- Root-cause work first: reuse the installed immutable V1 numerical runtime and existing wrappers;
  add new orchestration only when a run demands it. Prefer diagnosis over new hardening.
- Reports to owner: state result, % and blockers plainly; never claim unproven.

## Resume checkpoint — 2026-09-02 01:10Z

Worktree `/home/gianl/glm-tpu-gate-d-pp16-numerical`, branch
`tooling/gate-d-compensated-pp16-numerical`, HEAD `ecf723d7…` (+ evidence commit) on origin+mirror.
**Root cause proven at bounded scope:** V3 tag `…004306002075694Z` executed once with a host FP32
DSA rotary row: `NUMERICAL_ACCEPTED`; key within 4.99e-7 of the F64 reference (rotary 1.25e-7 vs
2.29e-3 in V2; implied cos/sin error 9.9e-3 → 6.5e-7); HLO has 0 transcendentals/0 collectives.
On-device cos/sin at large rotary angles was the DSA key divergence. Artifact
`gate-d-projection-host-rope-pp16-numerical-success-adjudication.json`.

Exact next: carry host FP32 DSA rotary rows into the decoder (`dsa_host_rope.py` keys/queries at
stage_local key/query sites, `prefill_index.py`, decoder plumbing mirroring `main_rope_table`),
prove DSA selection exactness on a bounded captured witness (one batched Sol review), then seek
separate authority before any 8K run. Gate D remains open; no performance claim.

## Finish (after Gate D)

§18 of the spec: PP8/PP16 protected measurements, WS32 result or evidence-backed rejection, 128K
four-depth, 256K E0, DB/archive, clean fleet, base vs speculative throughput reported separately.
