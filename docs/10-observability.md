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
| **Flight recorder** (the black box) | fork `tpu_inference/runner/flight_recorder.py` + hooks in `tpu_runner.py` (branch `glm-5.2-v4-obs`) | `GLM_FLIGHT_RECORDER=1` **baked into the raylet env** (launcher passthrough) | One compact JSON line per serving step, per worker process, appended to `/tmp/glm_flight_<host>_<pid>.jsonl`: `ts, step, num_reqs, real_tokens, padded_tokens` (bucket), `decode_only, num_prefill_chunks, finished, req_ids_hash, kv_len_min/max, padded_num_reqs` + lifecycle events (`recorder_init`, `model_loaded`, `warmup_done`, `continue_decode` loop summaries). Host-side only, outside jit, no device syncs; per-line `os.write` to an `O_APPEND` fd → survives SIGKILL/core-halt; ~15 µs/step measured (target <100 µs); rotates at 50 MB keeping the last 2 files. |
| **Crash triage** | `scripts/triage_crash.sh <run_log> [--no-fetch]` | — (read-only) | Automates the every-crash hand-grep: halted host IP(s) + first fatal timestamp, deduped error lines, per-host last step + composition (from log `[OBSERVE_COMPILES]` lines AND from the flight-recorder files fetched off all 8 hosts), the step delta identifying the diverged worker, jit program names near the fatal window, RESOURCE/ICI lines deduped. |
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
last recorded step — **the host that stopped logging EARLIER than its peers is
the diverged/halted one**; its last line tells you the step composition it
died on (decode-only vs mixed, padding bucket, kv-len range, req-set hash).
Compare with the halted IP from the fatal section (they should agree) and with
the `dump_input` SchedulerOutput in the log (what the engine *wanted* that
step). Validated on probeA5.log: halted host 192.168.0.20, first fatal
`E0707 11:01:39.201727`, last logged step 233 vs cluster max 386.

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
relaunching (`triage_crash.sh` does this) — the launcher's stop phase kills the
processes but leaves the files, a relaunch leaves stale pids' files behind
(triage picks the newest file per host by timestamp).

## Field notes / limits

- The flight recorder hook sits in `TPUModelRunner.execute_model` AFTER
  `_execute_model` returns — dispatch is async, so the line records what was
  *submitted*; a worker whose device halted mid-program still logs that step
  (the crash surfaces at the next host sync, typically `sample_tokens`'s
  `device_get`). The signal is the *silence afterwards*: the halted worker
  stops appending while peers keep going.
- `continue_decode` runs its inner loop on-device (inside jit): per-inner-step
  lines are impossible; the recorder logs the wrapper step plus a
  `continue_decode` summary event (actual_steps, termination reason).
- `padded_num_reqs=-1` on the continue_decode path (honestly unknown there);
  `padded_tokens` falls back to the dp=1 bucket recompute when
  `execute_model_state` is absent.
- CPU tests: fork `tests/runner/test_flight_recorder.py` (gate-off writes
  nothing, field values, hash stability, rotation, fail-open, cost smoke).
