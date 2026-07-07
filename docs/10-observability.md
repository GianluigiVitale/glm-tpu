# 10 — Observability for pod runs (see the failing state, don't grep blind)

> Philosophy (docs/suggestions.md): instrument BEFORE debugging. The agent must
> be able to SEE what the system was doing at the moment of failure — a
> post-mortem that can only grep a Ray-deduplicated log is flying blind.
> Concrete blindness this stack had: pod runs die with roaming single-host
> fatal `Error Interrupt` / `RuntimeUnexpectedCoreHalt` (probeA2/3/5/6 —
> different host each time: .15+.17, .15, .20, .15) and nothing recorded WHAT
> each worker was executing at death, nor per-step throughput.

## What exists now

| Instrument | Where | Gate (default OFF) | What it shows |
|---|---|---|---|
| **Flight recorder** (the black box) | fork `tpu_inference/runner/flight_recorder.py` + hooks in `tpu_runner.py` (branch `glm-5.2-v4-obs`) | `GLM_FLIGHT_RECORDER=1` **baked into the raylet env** (launcher passthrough) | One compact JSON line per serving step, per worker process, appended to `/tmp/glm_flight_<host>_<pid>.jsonl`: `ts, step, num_reqs, real_tokens, padded_tokens` (bucket), `decode_only, num_prefill_chunks, finished, req_ids_hash, kv_len_min/max, padded_num_reqs` + lifecycle events (`recorder_init`, `model_loaded`, `warmup_done`, `continue_decode` loop summaries). Host-side only, outside jit, no device syncs; per-line `os.write` to an `O_APPEND` fd → survives SIGKILL/core-halt; ~15 µs/step measured (target <100 µs); rotates at 50 MB keeping the last 2 files. Round-6 hardening (branch `glm-5.2-v4-r6fix`): `recorder_init` records `async_scheduling` (which step-attribution rule applies); a failed rotation falls back to appending the original path instead of silently dying; the fail-open disable now needs 3 CONSECUTIVE errors (counter resets on success — transient blips hours apart no longer kill it); ANY death logs a loud `BLACK BOX DEAD at ts=…` line so triage can tell recorder-death from worker-death. |
| **Crash triage** | `scripts/triage_crash.sh <run_log> [--no-fetch]` | — (read-only) | Automates the every-crash hand-grep: halted host IP(s) + first fatal timestamp, deduped error lines, per-host last step + composition (from log `[OBSERVE_COMPILES]` lines AND from the flight-recorder files fetched off all 8 hosts), the step delta identifying the diverged worker, jit program names near the fatal window, RESOURCE/ICI lines deduped. Round-6 fixes: the fetch uses `--output-directory` per-worker files + remote per-line host tags (`gcloud --worker=all` on shared stdout INTERLEAVES the 8 streams — the old awk misattributed hosts, producing false "NO flight data" / "diverged" rows); the output header prints the sync/async step-attribution rule; per-host `ts` + a recorder-death mark discriminate a dead recorder from a dead worker; step-less hosts show their newest lifecycle event instead of "NO flight data". |
| **Per-step engine stats** | `bench/engine.py` | `GLM_LOG_STATS=1` (driver env; no raylet baking) | vLLM's periodic stats logger (~10 s): prompt/generation tok/s, running/waiting request counts. Offline `LLM()` force-defaults `disable_log_stats=True`, so without this the only throughput signal is the tqdm bar. |
| **Serving-compile observer** | fork `tpu_runner.py` (`_observe_serving_compiles`, from the DSV4 port) | `DSV4_OBSERVE_COMPILES=1` (or `trace`) in the raylet env | Logs `[OBSERVE_COMPILES] step=N <label>: k backend compile(s)` per serving half-step — names the exact programs that live-compile at serving time (the launch-group race class). Caveat: these go through Ray's log stream, which **deduplicates across the cluster** → per-host last-step from the log is only a lower bound (the flight recorder is the authoritative per-host record). |
| **vLLM crash dump** | vLLM `dump_input.py` (upstream, always on) | — | On an engine-core exception, dumps the full engine config + the SchedulerOutput of the dying step (req ids, `num_scheduled_tokens`, finished ids). Engine-side view only — it shows what was *scheduled*, not which *worker* diverged. |

## Run-books

### Pod run crashed (Error Interrupt / core halt)

```
bash scripts/triage_crash.sh <run_log>          # full report (fetches flight files from all 8 hosts)
bash scripts/triage_crash.sh <run_log> --no-fetch   # log-parsing only (works on any old probe log)
```

Read the report bottom-up: the flight-recorder section aligns every host's
last recorded step. Apply the step-attribution rule the header prints (sync:
halted step = last line; async — the default — last line − 0..1; see Field
notes), and mind the caveat: a mid-collective halt hangs the peers at the same
step ±1, so a LOW spread does not exonerate anyone — the per-host `ts` column
and the recorder-death mark separate "worker died" from "recorder died". The
last line of the halted host tells you the step composition it died on
(decode-only vs mixed, padding bucket, kv-len range — decode-only lines only,
see Field notes — req-set hash). Compare with the halted IP from the fatal
section (they should agree) and with the `dump_input` SchedulerOutput in the
log (what the engine *wanted* that step). Validated on probeA5.log: halted
host 192.168.0.20, first fatal `E0707 11:01:39.201727`. (probeA5's per-host
"last steps" in §3 are Ray-dedup artifacts — lower bounds only, see
RESEARCH_LOG 2026-07-07 correction; the flight recorder did not exist for
probeA5.)

### Next pod run (always, until the halts are closed)

Enable the black box + throughput stats — both are default-off, zero-cost
choices we should make on every diagnostic run:

```
GLM_FLIGHT_RECORDER=1 bash scripts/launch_glm_32chip.sh   # bakes it into every raylet
GLM_LOG_STATS=1 python bench/run_bench.py ...             # driver-side, 10s tok/s + queue depth
```

(`DSV4_OBSERVE_COMPILES=1` optionally via `EXTRA_ENVS=` when hunting live
compiles; it is noisier and goes through Ray's deduplicated log stream.)

### Throughput looks wrong / run seems stalled

`GLM_LOG_STATS=1` gives running/waiting + tok/s every ~10 s. For per-step
cadence, `tail -f /tmp/glm_flight_*.jsonl` on any worker (or worker-0
locally): the `ts` deltas between step lines ARE the step times; `padded_*`
fields expose bucket thrash (recompile suspects) without a profiler.

### Flight-file hygiene

Files live in `/tmp` (per-host, per-pid), rotate at 50 MB keeping 2 → bounded
at ~100 MB/proc worst case; a pod restart wipes `/tmp`. Fetch them BEFORE
relaunching (`triage_crash.sh` does this). Round-6 F8: the launcher's stop
phase now prunes all but the 8 newest `/tmp/glm_flight_*` per host (unbounded
accumulation across relaunches fed the recorder's fail-open via /tmp pressure);
the just-crashed run's files are the newest and survive exactly one relaunch —
the fetch-before-relaunch run-book still applies (triage picks the newest file
per host by timestamp).

## Field notes / limits

- **The last logged step is the last DISPATCHED step, and the lag to the step
  executing at death is SCHEDULING-MODE-DEPENDENT** (round-6 F4,
  docs/reviews/round6-observability.md — the rule the triage output now
  prints, and `recorder_init` now records `async_scheduling`):

  | mode | engine arg | halted step vs last logged line |
  |---|---|---|
  | sync (`GLM_ASYNC_SCHED=0`) | `async_scheduling=False` | **= last line (lag 0)** — `sample_tokens(N)` blocks on step N's own tokens (`host_extract_sampled_tokens` → `device_get`), so line N+1 cannot exist unless step N completed |
  | async (**the vLLM default** when `GLM_ASYNC_SCHED` unset) | `async_scheduling=True` | **= last line − 0 or 1** — `_modify_prev_results` blocks only on step N−1, so a worker halted in model(N) can still dispatch and log N+1 before dying at `sample_tokens(N+1)` |

  At a page boundary the async ±1 is exactly the difference between "died in
  the two-page program (kv=513)" and "died in the single-page program
  (kv=512)" — check the log's `async_scheduling` before reading any flight
  line (waveB/waveB2 ran sync, so their step-417/kv-513 attribution is exact).
- **The divergence heuristic only works for halts that let peers progress.**
  For the dominant fatal class (mid-collective core halt) the peers hang at
  the SAME step (sync) or +1 (async) — expected cross-host spread 0–1, NOT a
  growing gap. A large spread means a halt that let peers keep serving, or a
  host whose RECORDER died early (see the triage ts column and the
  recorder-death mark; the recorder logs `BLACK BOX DEAD` on any death since
  the round-6 fix).
- **`kv_len_min/max` is `input_batch.num_tokens` = total KNOWN tokens, not
  computed KV** (round-6 F6). On a `decode_only=1` line it IS the step's
  attention span (post-write: the step writes slot kv_len and attends over
  kv_len entries — the 513 reading is correct). On a line with prefilling
  requests it is the FULL prompt length regardless of how much is computed —
  do not read kv_len off mixed-batch lines. Also `num_prefill_chunks` counts
  "reqs scheduled with >1 token", which will misclassify spec-decode steps
  when Stage-3 MTP lands (a decode req schedules 1+k tokens) — guard with
  `scheduled_spec_decode_tokens` then.
- `continue_decode` runs its inner loop on-device (inside jit): per-inner-step
  lines are impossible; the recorder logs the wrapper step plus a
  `continue_decode` summary event (actual_steps, termination reason).
- `padded_num_reqs=-1` on the continue_decode path (honestly unknown there);
  `padded_tokens` falls back to the dp=1 bucket recompute when
  `execute_model_state` is absent.
- CPU tests: fork `tests/runner/test_flight_recorder.py` (gate-off writes
  nothing, field values, hash stability, rotation, fail-open, cost smoke).
