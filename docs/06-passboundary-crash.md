# docs/06 — The "pass-2" device-fatal crash: RuntimeUnexpectedCoreHalt at request-FINISH steps

> Root-cause analysis of the two GLM pod benchmark crashes (2026-07-07), from logs + code only
> (READ-ONLY session; no TPU touched). Deliverable: ranked candidates, a discriminating probe
> ladder for the pod owner (cheapest first), and a fix sketch per candidate.

## 0. TL;DR — the framing correction first

**The crash is NOT at the start of the second `llm.generate()`.** Both halts (and a third,
previously uncounted wedge) happened **minutes into pass 2, at a scheduler step whose
`SchedulerOutput` has non-empty `finished_req_ids`** — i.e. the first step after a request
finishes, when the persistent batch condenses, the finished request's KV blocks are freed
(and possibly immediately reused), and — under async scheduling — a "zombie" extra decode
step for the finished request is still in flight in the 2-deep batch queue.

- 0 failures in ~32 pass-1 finish-steps across the three runs; 3/3 failures at pass-2
  finish-steps (2 core halts + 1 silent wedge). Under a uniform per-finish-step hazard this
  split has p ≈ 0.003 → "pass-2-ness" is real, but the *trigger site* is the finish-step.
- The pass boundary matters only as **state carried across it**: stale `_pre_async_results`
  from pass 1's last step, pass-1's freed KV pages available for reuse, and (observed in try1)
  a fully-empty `SchedulerOutput` delivered to all 8 workers at the boundary.
- **Zero serving-time backend compiles were logged for 26–40 min before either halt** → the
  DSV4 "staggered live compile → launch-group desync" racer (moe-tpu docs/15) is *not visibly*
  the mechanism here (it remains testable — probe P1).

## 1. Evidence

### 1.1 The three failing runs (all: GSM8K n=32, TP=32 on the v4-64, greedy, max_new=1024)

Common config (from the `gsm8k_n32.log` engine banner + `scripts/launch_glm_32chip.sh`):
Ray compiled-DAG executor, **async scheduling ON** (`step_with_batch_queue`, queue depth 2 —
`~/vllm-build/vllm/v1/engine/core.py:505-568`), chunked prefill `max_num_batched_tokens=512`,
**one compiled token bucket** (`TPU_MIN_TOKEN_BUCKET=512`, `compile_ranges_endpoints=[512]`)
so every step — 7-token decode or 74-token mixed — runs the *same* 512-padded program,
`enable_prefix_caching=False`, `TPU_DISABLE_DSA_INDEXER=1` (attention = dense MLA via mla.v2
+ the GLM FP8 GMM path; the new DSA sparse kernel is NOT in these runs), `max_num_seqs=8`.

| run | chunking | pass 1 | pass 2 failure | fatal step (from the vLLM `dump_input` block) |
|---|---|---|---|---|
| `~/glm-run/gsm8k_n32.log` | 2×16 | clean (1773 s, ~16 finish/admit steps) | **HALT 08:08:09**, 833 s / 7 of 16 done | 7 decodes + **new req `30` prefill (67 tok, block [51])**, `finished_req_ids=['25']`, total=74 (log :5692) |
| `~/glm-run/gsm8k_n32_try1.log` | 2×8 | clean (1108.8 s, 8 finish-steps) | **HALT 08:46:22**, 454 s in — **exactly at the FIRST finish of pass 2** | 7 decodes, `finished_req_ids=['11']`, **no** new reqs; survivors' `num_computed_tokens` 466–520 (straddling the 512-token page boundary); `num_output_tokens` all ≈386–388 (log :5374) |
| `~/glm-run/gsm8k_n32_try2.log` | 2×8 | clean (1109.5 s) | **WEDGE ~09:23:47** — progress stopped right after pass 2's first finish (`1/8 [04:07]`), idle until teardown 09:29:30 | no dump (no exception surfaced — the wedge form) |

Control: `~/glm-run/smoke16.log` — 4 items as sequential single-prompt generates — clean.
(Single-seq finish-steps have no surviving requests to condense, no zombie step worth racing,
and the batch queue drains at each generate end.)

### 1.2 The halt signature

- Root fatal on **one chip**, message class `Unknown` (NOT the DSV4 "different launch id /
  unexpected peer" text): run A `tpu455:pe1:0` on the **driver host .21**; try1 `tpu449:pe1:1`
  on **host .17** — a *different* host each time (not a worker-3-style fixed straggler).
- Peers all report `Got error from device: UNKNOWN: Fatal error; (root error from
  launch_id=894207886)` — the error propagated through a **launch-group (collective) program**,
  so the halting program was one of the cross-host collectives (i.e. essentially any step
  program at TP=32).
- The Python-side surfacing point is irrelevant to causality: `get_output →
  runner_utils.host_extract_sampled_tokens → jax.device_get(next_tokens)`
  (`~/tpu-inference/tpu_inference/runner/tpu_runner.py:170`, `runner/utils.py:1021`) — that is
  the per-step blocking sync documented in docs/04; it merely observes the already-fatal device.
- Because of the 2-deep queue, the halted program belongs to the window
  **[zombie step N+1, dumped step N+2]** around the finish (see §2.1).

### 1.3 Compile timeline (the DSV4-class check, done from the logs for free)

All logged XLA backend compiles (`isa_program_util_common` "Executable fingerprint" lines)
cluster in warmup + the **first serving step** (07:24 run A; 08:20 try1): `jit_split`,
`jit__select_from_array_fn`, `jit_compute_logits_func`, `jit_compute_and_gather_logprobs`,
`jit__lambda`, `jit_stage`, `jit__threefry_seed`, `jit_sample`,
`jit__substitute_placeholder_token`. **None in the 40 min (run A) / 26 min (try1) before the
halts.** docs/03 independently concluded "zero mid-run recompiles". With a single 512 bucket
and `num_reqs` padded to 8, there is no shape-driven recompile surface at the fatal steps.

### 1.4 The boundary empty cycle (try1)

At **08:39:21**, one second before pass 2's prompts arrived, **all 8 workers** logged
`Should not schedule a request that does nothing!` (`tpu_runner.py:1245`) — vLLM's scheduler
emitted a completely empty `SchedulerOutput` (no tokens, no finished ids) at the drain/refill
boundary. The runner short-circuits it (`tpu_runner.py:1234-1249`, returns
`EMPTY_MODEL_RUNNER_OUTPUT`, no device program) — benign *if* every worker sees it and
handles it identically, but it is a boundary-only code path that skips all async bookkeeping
(`_modify_prev_results` is never called for it) and it is the one step class the warmup never
exercises.

## 2. Ranked root-cause candidates

### C1 (most likely): async-scheduling finish-step interaction (zombie step × condense × block free/reuse)

Under async scheduling, when request X samples EOS at step N, step N+1 was **already
scheduled with X still in it** (the standard async overshoot; its token is discarded via
`discard_sampled_tokens_req_indices`). X's removal lands at step N+2 — exactly the dumped
step in both halts: `finished_req_ids=[X]` + `input_batch` condense
(`persistent_batch_manager.update_states`, called at `tpu_runner.py:1232` *before* input
prep) + scheduler-side free of X's KV blocks, which can be **reallocated to a new request in
the same step** (run A: new req 30 gets block [51] in the very step req 25 is freed). All of
this mutates worker state while step N+1 is still executing and while the on-device
placeholder-substitution path (`_substitute_placeholder_token` `tpu_runner.py:253-286`,
indices built in `_prepare_async_token_substitution_indices` :2197-2263 from a mix of the
**pre-step** placeholder map and the **post-condense** `req_id_to_index`) rewires the freshly
condensed batch to the previous step's `next_tokens` rows.

- For: both dumps are exactly step N+2; try2's wedge sits at the same site; smoke16
  (no survivors at finish) clean; the machinery is young TPU-specific code with known
  hairy edge cases (the padding comment at :262-265, the preemption guard at :2227-2232).
- Against/open: pass-1 finish-steps use the same code and survived (needs the pass-2
  state ingredient — stale `_pre_async_results` across the boundary, or first-time page
  *reuse*, which in the 2×8 runs genuinely occurs only in pass 2; in run A pass 1 had reuse
  and survived, so reuse alone is not sufficient either). A host-side indexing bug normally
  yields wrong tokens, not a chip fatal — the fatal needs C1 to *feed* C2 (bad
  seq_len/block-table state into the kernel) or to desync a worker (C3 form).
- Fix sketch: (i) immediate: run **sync scheduling** — at today's 1.37 s/step the 8–20 ms
  host floor (docs/04) costs ~1%; (ii) proper: defer worker-side condense + scheduler-side
  block free for a request until its zombie step's future has resolved (a one-step delayed
  `finished_req_ids` application, queue-depth aware), and audit
  `_prepare_async_token_substitution_indices`/`_modify_prev_results` (:1121-1179) under
  condense-at-N+2 with a CPU unit test that replays the dumped sequence.

### C2: data-dependent kernel fault (mla.v2 dense-MLA w/ GLM Stage-1 mods, or the per-tile FP8 GMM dequant) on inconsistent finish-step metadata

A single-chip `Unknown` fatal that moves between hosts is the signature of an out-of-bounds
DMA / bad scalar-prefetch grid on whichever shard holds the affected rows. The mla.v2 kernel
walks pages from scalar-prefetched `block_tables`/`seq_lens`; a condensed or padded slot whose
`seq_lens` exceeds its block-table row's valid prefix (an off-by-one that only the
finish/condense transition produces) drives the page walk into garbage entries. Note try1's
survivors sat at 466–520 — straddling the 512-token page boundary — when it died.

- For: matches the single-chip, host-roaming, data-dependent fatal; the FP8 GMM + block-scale
  paths are new (fork commits `8ad980e7`, `3f745ddc`) and v4-specific.
- Against: the same program ran thousands of steps incl. 32 clean finish-steps; stale
  block-table entries are usually *valid* page ids (correctness bug, not a fault), so this
  most plausibly fires only with C1 supplying truly inconsistent metadata.
- Fix sketch: cheap host-side pre-dispatch asserts under a debug env (pad-slot `seq_lens==0`;
  every walked block-table entry `< num_blocks`; `seq_len ≤ 512·row_valid_len`); replay the
  dumped fatal-step state on the single-host sub-cube to fuzz finish/admit transitions at
  page-boundary lengths.

### C3: multi-host divergence at the finish/boundary step (the DSV4 docs/15 class, non-compile variant)

One worker running (or skipping) a different program than the other 31 → the collective's
launch group breaks → wedge (try2) or fatal (try1/run A). The classic *cause* in DSV4 —
staggered serving-time live compiles — is **contra-indicated here** (§1.3: zero compiles for
26–40 min before the halts). A residual variant would need a worker-local decision diverging
(e.g. asymmetric handling of the boundary empty cycle, or a rare compile invisible in these
logs). Note `enable_continue_decode` is OFF (default; requires sync scheduling), so the
decode-only program fork at `tpu_runner.py:1253-1256` is inert.

- For: wedge+halt duality is exactly the desync family; error rides a launch_id.
- Against: no compile evidence; empty cycles reached all 8 workers symmetrically; no fixed
  straggler host.
- Fix sketch: if P1's observer names a racing program, lockstep-precompile it (the PR-10
  pattern, already in this fork).

### C4: hardware/environment transient (chip-level fatal)

Kept last: 3/3 failures at pass-2 finish-steps is far too patterned for random chip flakes,
and the halted chip moved (`tpu455` host .21, then `tpu449` host .17). Escalate only if the
probe ladder returns all-null. On the next halt, grab the halted host's fresh libtpu log
(the try1 trace shows libtpu opened a new log at 08:46:22 on w-2 — it names the halting
program): `gcloud compute tpus tpu-vm ssh <pod> --worker=<w> --command "ls -lt /tmp/tpu_logs | head; tail -200 /tmp/tpu_logs/<newest>"`.

## 3. Discriminating probe ladder for the pod owner (cheapest first)

- **P1 — observer rerun (one benchmark run, zero code changes).** Re-run GSM8K n=32 with
  `EXTRA_ENVS="DSV4_OBSERVE_COMPILES=1"` baked into the launcher and `--batch-size 8` (the
  fastest reproducer: try1/try2 died at pass 2's *first* finish, ~35–40 min total). The
  observer is already wired into `execute_model`/`sample_tokens`/`get_output`
  (`tpu_runner.py:492-510,1015`; `runner/utils.py:300`). Crash + any per-worker serving
  compile named at/near the fatal step → **C3**; crash + zero compiles → **C1/C2**. Also
  collect the halted host's `/tmp/tpu_logs` (names the halted HLO program → discriminates
  C2's kernel vs a collective).
- **P2 — async OFF A/B (the sharpest single discriminator, ~1% perf cost today).** Same run
  with async scheduling disabled (engine arg `async_scheduling=False` in `bench/engine.py`'s
  `LLM(...)`; vLLM enables it by default because the Ray executor supports it — docs/04).
  3/3 clean → **C1 confirmed as the enabling condition and the immediate mitigation**;
  still crashes → C1 is out, weigh C2 vs C4 (async-independent).
- **P3 — cheap synthetic reproducer (~10 min/rep).** Two back-to-back generates of 8 short
  prompts with staggered `max_tokens` (e.g. 32..96) so pass 2 has early finish-steps with
  survivors; 3–5 reps, observer on. Reproduces → a fast A/B vehicle for P2/P4 and for
  separating "boundary state" from "finish-step alone" (variant: run pass 2 in a FRESH engine
  — if a fresh engine's generate-2-equivalent never crashes, cross-boundary state is load-bearing).
- **P4 — mitigation to get the benchmark number now (config-only).** `--batch-size 0`
  (all 32 items in ONE generate): pass-1-equivalent conditions went 32-for-32 finish-steps
  clean across the three runs; combined with the retry loop this is the highest-yield cheap
  path to a recorded n=32 run. (Risk-reduction, not a fix — finish-steps still occur.)
  Alternative known-good: sequential single-prompt generates (smoke16 mode), ~8× slower.
- **P5 — if C2 leads after P1/P2:** sub-cube replay of the dumped `SchedulerOutput` sequence
  (7 decodes + finish + admit at 466–520/851–861 lengths, page size 512) with the host-side
  asserts from §C2 enabled.

## 4. Artifacts / provenance

- Fatal dumps: `gsm8k_n32.log:5692` (SchedulerOutput), `:5694+` (EngineCore traceback);
  `gsm8k_n32_try1.log:5374`. Halt roots: run A `tpu455:pe1:0` (.21), try1 `tpu449:pe1:1`
  (.17), peers `root error from launch_id=894207886`.
- Boundary empty cycle: `gsm8k_n32_try1.log:3725,3860` (08:39:21, ×8 workers).
- try2 wedge: last progress `1/8 [04:07]` ≈09:23:47; raylet-only logs until 09:29:30.
- Compile timeline greps: `grep -o "I0707 [0-9:]*\.[0-9]*[^]]*] (HLO module jit[a-zA-Z_0-9]*" <log>`.
- Code: `tpu_runner.py` :1232-1256 (empty cycle + update_states), :1799-1833
  (`_pre_async_results`), :2197-2300 (substitution), :1121-1179 (`_modify_prev_results`),
  :164-207 (`get_output`); `vllm/v1/engine/core.py:505-568` (`step_with_batch_queue`).
