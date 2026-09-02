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

## Resume checkpoint — 2026-09-02 04:20Z

Tooling worktree `/home/gianl/glm-tpu-gate-d-pp16-numerical`. **Host-rotary-table hypothesis
REFUTED at engine level**: protected 8K run `…mainrope_dr_…_20260902T021346091708582Z` (pin
`ba7d1e72`, `GLM_GREENFIELD_DSA_ROPE_TABLE=1`) failed closed at exact DSA. Without the table
(2026-08-28 run) event 0/layer 0 was bit-exact vs the legacy oracle; with it, layer-0 scores move
≤2.6e-3 (query, position 0 too) while event 1/layer 1 is unchanged (6 swaps, mean |Δscore| 0.0148
in both runs). Legacy indexer computes `jnp.cos/sin` on device (`glm_dsa_indexer.py:1078`).
Artifact `gate-d-dsa-rope-table-8k-refusal-adjudication.json` `0f7c36d2…`. Censuses 8/8; no
SUCCESS/DB. `dsa_rope_table` stays default-off; never relaunch with it.

Consequence: the layer-1 divergence arises between the exact layer-0 DSA event and the layer-1
indexer input (layer-0 attention output / MoE / residual), at bf16-rounding scale (~1e-3 rel).

Exact next (no TPU launch): reread the Compass artifact; adjudicate its hidden-state hypotheses
against existing layer-0 discriminator/ingredient artifacts, DSA internal observer and layer-1
internal reference; pick one discriminator isolating the layer-0→1 hidden-state delta; one batched
Sol review of this refusal record plus that plan before any run.

## Finish (after Gate D)

§18 of the spec: PP8/PP16 protected measurements, WS32 result or evidence-backed rejection, 128K
four-depth, 256K E0, DB/archive, clean fleet, base vs speculative throughput reported separately.
