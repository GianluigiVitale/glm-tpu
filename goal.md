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

## Resume checkpoint — 2026-09-03 01:10Z

Tooling `/home/gianl/glm-tpu-gate-d-pp16-numerical`; run pin = rewrite merge (see HANDOFF tail). Three
executed 8K runs with `GLM_GREENFIELD_RMS_ACCEPTED_SCHEDULE=1`: token exact, event 0 exact, **event 1 = 7
mismatches (same set)** — decode-side norm hypothesis exhausted. Offline CPU capsule
`docs/artifacts/gate-d-event1-layer1-prompt-cache-offline-diagnosis.json`: live/bounded reduce fusions
byte-identical; legacy layer-1 query/head-weights/current-key over the greenfield DB518 layer-1 prompt
cache reproduces the device selection exactly (0/0) and the oracle's 7/7 swaps → the **layer-1 prompt
index cache built by the teacher-forced prefill is a sufficient cause of the event-1 mismatch** (decode-side
exactness not proven). Do NOT launch the 8K decoder. Next: Sol re-review of the four round-24 fixes, then
one legacy oracle capture `GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID=1 bash
scripts/greenfield/run_capture_legacy_prompt_index_cache.sh` (legacy slot 2 = 2·layer; sealing fails
closed on any other geometry), then `compare_layer1_prompt_index_cache_offline.py` against DB518
`result.npz` `534bacc5…` → row-level mismatch map → decide the prefill plan. Sealed layer-0 DSA input
restored to `/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z`.

## Finish (after Gate D)

§18 of the spec: PP8/PP16 protected measurements, WS32 result or evidence-backed rejection, 128K
four-depth, 256K E0, DB/archive, clean fleet, base vs speculative throughput reported separately.
