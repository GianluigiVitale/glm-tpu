# Native trace failure and bounded recovery — 2026-09-13

Execution `9c80ebac85b7dca0ee5423638eddbaf41089e027`, tag
`greenfield_ws32_native_benchmark_20260913T141500000000000Z`.
All eight original workers passed six HLO inspections and fresh memory admission,
completed first GPQA prefill, then failed after the first advancing observer call.
Rank0 delivered one token to the local JSONL sink. No completed answer/score.
Original worker exits and publication exits are all1; root postcensus all8idle.

## Cause and correction

Original XPlanes:244,036,416–244,713,162B per host; JSON gzip~17.5MB.
Rank0 XPlane SHA `1dec2c1452ac988036d2a32e751da699870d6c9ef4e75a7b9eef4f85bd65dc93`.
Host metadata80,653,530B, CPU plane60,817B; device metadata dominates the remainder.
Python tracer was already0. The128MiB raw cap was a harness planning error, not
a model arithmetic failure. Publication used that same cap and also refused.

Correction:320MiB original/inflated trace ceiling;128MiB stored trace ceiling.
Use existing lossless gzip publication and generation/CRC/SHA readback, with
pre-publication compressed-size screening. Whole-run regional limit stays10GiB.
Original rank0 passes corrected finalizer and requires51,697,023 stored bytes
including unchanged JSON. No trace events/metadata removed. Local/inflation
reserve checks count ORIGINAL bytes. Preserve DSA observation before finalizer.

Tests: initial41 focused CPU checks passed14.88s, including compressed roundtrip,
raw/stored over-budget refusal, changed generation/bytes/SHA, original preservation,
first observer failure, result/archive/outer composition. Recovery ownership tests
added; final combined44 tests passed14.67s. Self-review only; no P0-P2 remaining.

## Recover originals without another model call

After clean publication and regional mirror verification, run:

```bash
JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python -m scripts.greenfield.recover_native_trace_originals --recovery-pin <published-HEAD>
```

Fixed failed tag/pin only, both leases, fresh root idle and original ended/boot
identity before existing-repo deployment. Exact conditional publication preserves
all failed markers. New receipt `trace_original_recovery.json` in original run
directory records execution and recovery pins separately. No deletion, checkpoint
copy, resource management, supervisor/model restart or historical SUCCESS rewrite.
Recovery publication is not a quality pass. Preserve partial first-token output
privately; never paste benchmark questions/golds/answers into Git.

After recovery, ONE corrected new-tag benchmark may start after fresh6GiB/fleet
guards. Keep228 requests/full163840 cap/capacity166912, no baseline long reruns.
All four128K and256K evidence DB616–620 stays complete. AIME judging still needs
explicit paid-judge approval; no paid calls or quiet scorer substitution.
