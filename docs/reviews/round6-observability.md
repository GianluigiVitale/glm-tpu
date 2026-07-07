# Round-6 adversarial review — observability stack (flight recorder + triage + GLM_LOG_STATS)

**Target:** fork `glm-5.2-v4` @ `3f6f5854` + merge `87abdf53` (flight recorder, live in
`~/tpu-inference` = the editable install the pod workers run), `~/glm-tpu` @ `ef68d6e`
(`scripts/triage_crash.sh`, launcher passthrough, docs/10), and the `GLM_LOG_STATS` knob
(actually landed in `9f2f2c3`, see F9).
**Reviewer constraints honored:** read-only except this file; no TPU touched (all probes
`JAX_PLATFORMS=cpu` before python; no pod ssh — waveB3 was live during review).
**Stakes:** crash conclusions (the step-417 / kv_len-513 page-crossing RCA and the
probeA5 "153 steps behind" narrative) are being drawn from this instrument. Both were
attacked; one survives, one does not.

---

## Verdict in one paragraph

The recorder itself is sound: gated, host-only, no device syncs, per-line durable, cost
claim reproduced (14.3 µs mean @16 reqs), 11/11 unit tests pass. **The step-417/kv_len-513
conclusion SURVIVES for the waveB/waveB2 crashes** — but only because those runs verifiably
ran `async_scheduling=False`; under the default async mode the same reading would be
off-by-one *at exactly the 512-page boundary being blamed* (F4). The triage script's
cross-host alignment is **unreliable as shipped**: `gcloud --worker=all` interleaves the 8
ssh streams and the awk misattributes hosts — reproduced with the script's own awk,
producing both the observed "NO flight data" hosts and a **false** "diverged" flag (F1).
The fail-open design has a single-error silent-death path (F2) and a lifetime error
counter (F3) that can kill the black box exactly when needed. The probeA5 "halted host 153
steps behind" number is a Ray-dedup artifact and its RESEARCH_LOG attribution to the
flight recorder is false — the recorder did not exist in probeA5 (F5).

---

## CONFIRMED (positive assurance — verified, not assumed)

| # | Claim | How verified |
|---|---|---|
| C1 | **Gate-off = zero behavioral change.** `maybe_create_flight_recorder` returns `None` unless `GLM_FLIGHT_RECORDER=1`; all 5 hook sites are `if self._flight_recorder is not None` (tpu_runner.py 507, 962, 1019, 1077, 1624). Only an added import runs when off. | Read every hook in the 3f6f5854 diff; gate test passes. |
| C2 | **No device syncs on the hot path.** `padded_num_reqs` is a host `Optional[int]` (ExecuteModelState:245, from `get_padded_num_reqs_with_upper_limit`, utils.py:143); `input_ids.shape` is metadata; `request_distribution` is `list[int]` (input_batch.py:146); `num_tokens` is host numpy. `record_step` never calls `.item()`/`device_get`. | Type-traced each field to its producer. |
| C3 | **Cost claim reproduces.** 16-req decode batch, this VM (worker-0 hardware class): **mean 14.3 µs, p50 13.7, p99 20.2, max 152 µs** over 20k calls (author claimed 15 µs). 256 scheduled ids: 62.5 µs — still under the 100 µs target. | `scratchpad/fr_probe.py`, 20k-sample distribution. |
| C4 | **Tests: 11/11 pass** (`tests/runner/test_flight_recorder.py`, `JAX_PLATFORMS=cpu`, vllm-env, on the live branch @ 87abdf53). | Ran them. |
| C5 | **decode_only mirrors the runner's dispatch test** — same source (`input_batch.request_distribution[0] == num_reqs`) as tpu_runner.py:1270. | Code identity. |
| C6 | **Durability mechanism is right**: single `os.write` to an `O_APPEND` fd per line; rotation keeps exactly 2 files (test passes); a SIGKILL/halt loses at most the in-flight line. | Code + rotation test. |
| C7 | **GLM_LOG_STATS plumbing works end-to-end**: driver-env → `LLM(disable_log_stats=False)` overrides offline-LLM's force-default; propagates to EngineCore via vllm_config (no raylet baking needed, as documented). **Observed live**: waveB3 log emits `loggers.py:273` stats every ~10 s. | bench/engine.py:82-83 + live log. |
| C8 | **Launcher passthrough correct**: `GLM_FLIGHT_RECORDER=${GLM_FLIGHT_RECORDER:-0}` inside the single ENVS string used verbatim on head + all workers (launch_glm_32chip.sh:94); recorder confirmed ON on all 8 hosts in waveB3 (`flight_recorder.py:81 ... ON` ×1 explicit + `[repeated 7x across cluster]`). | Log evidence. |
| C9 | **`triage_crash.sh` is shellcheck-clean** (v0.10.0) and its log-parse sections work on probeA5.log (halted `.20`, first fatal `E0707 11:01:39.201727` — matches the commit message). | Ran shellcheck + the script `--no-fetch`. |
| C10 | **req_ids_hash** is order-independent and set-sensitive (test) → cross-host comparable as designed. | Test passes. |

---

## The two headline questions

### Q1 — Is "last logged step" the last EXECUTED step or the last DISPATCHED step?

**It is always the last DISPATCHED step** (the hook runs right after `_execute_model`
returns, i.e., after async dispatch — the author's docstring and docs/10 field note say
this correctly). Whether that equals the step executing at death **depends on the
scheduling mode**, which neither the code, the doc, nor the recorded lines capture:

- **Sync (`GLM_ASYNC_SCHED=0` → `async_scheduling=False`):** `sample_tokens(N)` blocks on
  step N's own tokens (`host_extract_sampled_tokens` → `jax.device_get`, utils.py:1021).
  The host cannot write line N+1 until step N fully completed on device. **Last line N ⇒
  death during step N's model or sampler. Attribution exact (±0).**
- **Async (default when `GLM_ASYNC_SCHED` unset):** `_modify_prev_results` is called
  inside `_sample_from_logits` (tpu_runner.py:1826-1829) and blocks only on step **N−1**'s
  tokens. A worker whose device halts in model(N) still sails through `sample_tokens(N)`,
  dispatches model(N+1), **writes line N+1**, and only dies at `sample_tokens(N+1)`.
  **Last line = halted step + 1** — unless the poisoned runtime fails the h2d transfers in
  `execute_model(N+1)` first, in which case it's +0. **Ambiguous ±1.**

**Verdict on step-417/kv_len-513: SURVIVES.** waveB2 verifiably ran sync:
`gsm8k_waveB2.log:16` shows `'async_scheduling': False` in the engine args, and the crash
traceback frames are the sync block (`tpu_runner.py:1131 → utils.py:1021
host_extract_sampled_tokens → jax.device_get`), not `_modify_prev_results`. So the halted
host's last flight line IS the dying step, no off-by-one. Had the run been async (the
default!), the identical flight data would have meant the device died at the **kv=512
(single-page) step** and the 513 two-page program was never executed — the opposite RCA
target. See F4 for the required guardrails.

Residual caveats on the 513 conclusion (do not overturn it):
- The line pins the *step*, not the *program*: death could be in model(417) **or**
  sampler(417). The repro (`repro_mla_513.py`) assumes the MLA page walk; the sampler is
  not excluded by the flight data alone (the jit-names-near-fatal triage section is only
  a hint). An on-TPU A/B (e.g., same step shapes with attention stubbed) discriminates.
- `kv_len_max` is a max over 11 reqs — "one req crosses" is an inference (min=…,max=513
  can hide multiple boundary reqs). Same kernel-shape class either way.
- It rests on pod-side flight lines I could not re-fetch (read-only + waveB3 live). The
  SPMD argument makes it robust to F1's missing hosts: all 8 workers log identical step
  compositions (same broadcast scheduler output, same global batch), so ANY host's line
  417 pins the composition — the halted-host ID comes independently from the runtime's
  `ip=` fatal line.

### Q2 — kv_len_min/max: which field, PRE- or POST-step?

From `input_batch.num_tokens[:num_reqs]` (flight_recorder.py:158). For a **decode** row
this is prompt + outputs **including the token sampled at step N−1 — which is step N's
query token**. So `kv_len` on line N = the attention span of step N = the **post-write**
KV length (the step *writes* slot `kv_len` and attends over `kv_len` entries). Verified
identical under both scheduling modes (sync: set at the previous `sample_tokens`,
tpu_runner.py:1912; async: placeholder at `_update_placeholder`:1224 corrected at
`_modify_prev_results`:1191 — both counted before the hook fires). **Therefore
`kv_len_max=513` means the logged step is itself the first step whose attention crosses
the 512-page boundary — the author's reading is the correct one; no off-by-one from the
field semantics.** The off-by-one risk lives entirely in Q1's mode dependence.

**BUT (F6):** for a req still (chunked-)prefilling, `num_tokens` is the **full prompt
length**, not the computed KV (probe: a 900-token prompt with 100 computed logs
`kv_len_max=900`). The fields mean "KV length" only on `decode_only=1` lines. The 513
line was decode-only, so the conclusion is unaffected — but the doc must say this before
someone reads a mixed-batch line the same way.

---

## Findings (most severe first)

### F1 — HIGH — `triage_crash.sh` §6: `gcloud --worker=all` interleaves the 8 ssh streams; the awk misattributes hosts → the observed "NO flight data" (w-4/w-5/w-7) and possible FALSE "diverged" flags
`scripts/triage_crash.sh:170-236`.
gcloud's `tpu-vm ssh --worker=all` runs one **thread per worker**, each ssh child writing
directly to the **shared stdout** (verified in the installed SDK:
`/snap/google-cloud-cli/current/lib/surface/compute/tpus/tpu_vm/ssh.py` — threads +
`WaitForBatchCompletion`; per-worker files only with `--output-directory`). Worker
outputs interleave at pipe-chunk granularity, so a host's `FLIGHT_HOST $(hostname)`
marker and its `tail` payload are NOT contiguous in `$FLIGHT_RAW`. The awk attributes
every line to the *most recent marker seen*.
**Reproduced with the script's own awk block (extracted verbatim):** w-3's marker printed,
w-4's marker interleaved before w-3's data ⇒ output was
`w-3 NO flight data`, w-4 credited with w-3's step 417, and **w-5 falsely flagged
"stopped 1 step(s) EARLY (diverged?)"**. This mechanism produces exactly the reported
w-4/w-5/w-7 "NO flight data" symptom, and — worse — can flag a healthy host as the
diverged one, or hide the real one. Mid-line chunk splits can additionally garble JSON
lines (silently dropped by the regex).
**Fix (pick one):** (a) `--output-directory=DIR` → per-worker files, no shared stdout;
(b) tag remotely per line: `tail -n 100 $GLOB | sed "s/^/FR|$(hostname)|/"` and key the
awk on the tag instead of stateful markers; (c) loop `--worker=0..7` sequentially
(8× slower, trivially correct). Until fixed, treat every §6 per-host row as suspect;
only the *set* of step lines (host-agnostic) is trustworthy.

### F2 — HIGH — flight recorder: a single rotation failure silently kills the black box (bypasses the 3-error allowance)
`tpu_inference/runner/flight_recorder.py:183-190` (`_rotate`) with `:192-204` (`_on_error`).
`_rotate` sets `self._fd = -1` *before* `os.replace`/`os.open`; if either raises (dir
perms, ENOSPC creating the new file, .1 unlink race), the exception lands in
`_on_error` with `_errors=1` — but every later `record_step` returns at the `_fd < 0`
guard, so the counter never reaches 3 and the **"disabled after N errors" message never
fires**; the only trace is one `record failed (1/3)` line in a Ray-deduped stream.
**Probe confirmed:** after one forced `os.replace` failure, `fd=-1, errors=1`, zero bytes
ever written again. A recorder that dies mid-run makes that host "stop logging early" —
**the triage heuristic then reports a healthy host as the halted one** (inversion of the
instrument's purpose). Fix: on rotation failure, attempt to reopen the ORIGINAL path
(rotation is an optimization; losing it should not lose the recorder), and make any
`_fd=-1` transition log loudly ("BLACK BOX DEAD at ts=…").

### F3 — MED — fail-open counts LIFETIME errors, never reset: three transient blips (hours apart) permanently disable it; /tmp pressure at crash time is the worst case
`flight_recorder.py:49,192-204`. `_errors` is monotone; **probe confirmed** 200 successful
writes between errors do not reset it. A transient `/tmp` full (core dumps and TPU
runtime debug dumps land there *at crash time*; stale flight files accumulate across
relaunches — 5+ launches today, files never pruned, docs/10:63-65) yields 3 strikes and
a silently dead recorder for the rest of a multi-hour run. Fix: reset the counter on a
successful `_write` (making it "3 consecutive"), and/or retry-once-per-N-steps after
disable. Triage countermeasure regardless of fix: print each host's **last line ts** next
to its last step and flag any host whose recorder went quiet long before the fatal ts
(recorder-death vs worker-death discrimination) — the data is already in the fetch.

### F4 — MED — the ±1-step interpretation rule is mode-dependent and nowhere recorded; under the DEFAULT async scheduling the 513-style conclusion would flip
See Q1. `docs/10-observability.md:69-74` states the dispatch-async caveat but not that
the lag is **0 under sync and 0-or-1 under async** — and async is the default
(`bench/engine.py:74` only forces sync when `GLM_ASYNC_SCHED=0`). At a page boundary
this ±1 is precisely the difference between "died in the two-page MLA program (513)" and
"died in the single-page program (512)". Also `docs/10:73-74` "the halted worker stops
appending while peers keep going" is wrong for the dominant fatal class: a mid-collective
halt hangs the peers at the same step (sync) or +1 (async) — expected spread 0-1, not a
growing gap; the heuristic discriminates only for halts that let peers progress.
**Fix:** (a) record `async_scheduling` (and `GLM_ASYNC_SCHED`) in the `recorder_init`
event so every triage knows which rule applies; (b) add the mode-dependent
interpretation table to docs/10; (c) triage should print "halted step = last line (sync)
/ last line − 0..1 (async)".

### F5 — MED — the probeA5 "halted host 153 steps behind" evidence is a Ray-dedup artifact, and RESEARCH_LOG falsely attributes it to the flight recorder
`docs/RESEARCH_LOG.md` (12:40 entry): "the flight-recorder triage of probeA5 showed the
halted host **153 steps behind**". Refuted on two grounds: (1) **probeA5 predates the
recorder** — `grep -c flight_recorder probeA5.log` = 0 (run 11:01; recorder commit 11:45,
first wired launch 11:46). The number came from §3's `[OBSERVE_COMPILES]` log lines.
(2) Those per-host numbers are unusable for skew: Ray's dedup canonicalizer **strips every
whitespace-token containing a digit** (`ray/_private/ray_logging/__init__.py:201-210`),
so `step=233` and `step=386` lines from different hosts are "identical" and all but one
per ~5 s window are suppressed — per-host "last step" measures *who last won a dedup
window*, essentially rotating. Internal evidence: in the same triage output, healthy
hosts show −12, −24, −33, −36 spreads — impossible under sync scheduling's ≤1-step host
skew — so −153 for `.20` carries no more meaning. The script itself labels §3 a "lower
bound" (triage_crash.sh:12-14, correct and confirmed); the RESEARCH_LOG upgraded a lower
bound into quantitative skew evidence under the wrong instrument's name. The
sharedbin=1 A/B (waveA2 clean) stands on its own, but note waveB (12:53, `.20`) and
waveB2 (13:15, `.18`) **crashed with sharedbin=1** — "CORE-HALT ROOT CAUSE CLOSED" was
overbroad, as the ongoing 513 investigation implicitly concedes. Recommend a correcting
RESEARCH_LOG entry (the repo's honest-nulls discipline).

### F6 — LOW — `kv_len_*` is total-known-tokens, not computed KV: meaningless for prefill rows; `num_prefill_chunks` will miscount when MTP lands
`flight_recorder.py:154-160` (kv) and `:116-121` (chunks). Probe: a chunked-prefill req
with 100/900 tokens computed logs `kv_len_max=900`. Only `decode_only=1` lines admit the
"KV length" reading (the 513 line qualifies). Separately, "reqs scheduled with >1 token
are prefill chunks" becomes false under spec decode (a decode req schedules 1+k tokens)
— Stage-3 MTP will silently misclassify decode steps as prefill. Document the former in
docs/10; guard the latter with `scheduled_spec_decode_tokens` when Stage 3 lands.

### F7 — LOW — triage §6: a fresh post-relaunch file (lifecycle events only) is invisible; lifecycle info collected but never reported
`triage_crash.sh:192-199,207-215`. `lastev[k]` is gathered but never printed, and
`best[h]` iterates `lastts` (step lines only) — a host whose newest file has
`recorder_init`/`model_loaded` but no steps (crashed in load/warmup — a *very*
interesting divergence signal) prints as bare "NO flight data". Also `ts` in the step
block reuses the previous line's value if a line lacks `"ts":` (stale-variable reuse;
`st` is guarded by `else next`, `ts` is not). Print `lastev` + its ts for step-less
hosts; reset `ts` per line.

### F8 — LOW — /tmp accumulation across relaunches is unbounded and feeds F3
Files are per-pid and never pruned (docs/10:63-65 acknowledges); ≥5 recorder-on launches
today ⇒ up to ~100 MB × launches × host. The launcher stop phase (which already kills
the processes) should prune `glm_flight_*` older than the previous launch, AFTER a
triage fetch if a crash occurred (the doc's "fetch BEFORE relaunching" run-book covers
the ordering).

### F9 — INFO — commit-message provenance: ef68d6e claims the bench/engine.py change, but `git show ef68d6e -- bench/engine.py` is empty
`GLM_LOG_STATS` landed in `9f2f2c3` (verified with `git log -S`). Cosmetic, but this
repo's provenance discipline is load-bearing; note it wherever ef68d6e is cited as the
knob's origin.

---

## What only on-TPU / pod-side runs can verify (this review could not)

1. **The actual waveB/waveB2 flight lines** — the 511→512→513 `kv_len_max` progression,
   `decode_only=1`, 11 reqs / T=32 on the halted host's own file. Fetch with a
   non-interleaving method (F1 fix) **before** any relaunch wipes or buries `/tmp`.
   WaveB3 was still running during this review; nothing was fetched.
2. **Model-vs-sampler within step 417** (Q1 residual): needs the fatal-window jit names
   from an un-deduped source (per-host stderr, `--output-directory` fetch of raylet
   logs) or an on-TPU A/B with the attention/page-walk stubbed.
3. **Healthy-run host skew** (claimed ≤1 step here from code analysis): one clean
   recorder-on run, align per-host `ts` for the same step — also validates the F1 fix.
4. **On-pod recorder cost** under real serving cadence (my 14.3 µs is this VM = worker-0
   hardware, idle; contention with the engine process not measured — expected
   negligible vs multi-second steps).
5. **Fetch-path behavior against real gcloud** (the author's stub test could not exhibit
   thread interleaving — that is exactly what it missed).

## Probe artifacts (scratchpad, reproducible)

- `scratchpad/fr_probe.py` — cost distribution, rotation-death, error-accumulation,
  prefill-kv_len probes (all CPU).
- `scratchpad/triage_test/{sec6.awk,interleaved.txt}` — the F1 misattribution repro using
  the script's verbatim awk.
