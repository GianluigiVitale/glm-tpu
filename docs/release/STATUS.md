# Release status

**Current question-release finding:** the corrected-cache run at `4acb873a`,
`optimized_request_20260920T180158114672Z`, compiled and admitted all six graphs
for the 166,912-slot profile. Its first selected question exhausted an explicit
32,768-token output allowance without a final answer. The second was interrupted
while reasoning; the remaining eight were not run. The attempt was deliberately
stopped to correct the avoidable output cutoff, with originals preserved and
authenticated cleanup on all eight hosts. This is not a ten-answer pass or a
completed 131,072-token input check.

The `ask` default now allocates each question its available output space, up to
the retained 163,840-token maximum. Explicit caps remain exact; a full 131,072-token
input leaves 35,840 output tokens. Thirty affected CPU checks passed, and a
real-tokenizer prepare-only check preserved all ten input token arrays exactly.
No numerical program changed. Corrected trained results and main promotion are
still pending. The [private operational handoff](../../HANDOFF.md) points to the
originals; no failed or interrupted output has been relabeled successful.

**Earlier question interface candidate:** `bae824a0` on
`release/optimized-ordinary-20260920` adds `python -m glm_tpu ask`, ten queued
questions and an explicit 128K input profile (166,912 combined slots). The owner
accepted queued generation, one answer at a time. Real-weight run
`optimized_request_20260920T173446591702Z` attempted five randomly selected
GPQA and five AIME questions; one GPQA input is exactly 131,072 tokens using
synthetic irrelevant filler. It is a capacity check, not a long-document
reasoning benchmark. It stopped before any answer: the B128 prefill compiler
needed 31.45 GiB against 30.75 GiB, exceeding HBM by 720.25 MiB. All eight
workers exited and authenticated cleanup passed. The correction reuses the
legacy long-context path's exclusive state donation, avoiding a redundant cache
copy; it changes no numerical kernel. Corrected completed-answer results and
main promotion are pending. The question selection and originals are preserved.

The preceding 8K integration at `4f551e6b` completed and was collected:
[receipt](ordinary-integration-20260920.json). It matched the 29-token reference
prefix on all eight hosts, admitted fresh graphs/memory and cleaned up. It
delivered 14.35 decode tokens/s and 142.04 prompt tokens/s at 2,034 prompt tokens;
cold load/compile took 1,095.33 seconds. The 256-token limit stopped during
reasoning, so this does not provide a completed answer or admit 128K operation.

The [existing CPU release check](optimized-cpu-check-20260920.json) passed
571 tests, one skipped, plus source/content/package checks on the 8K candidate.
The queued extension passed 114 affected checks with one optional skip, and
40 portable checks passed from an extracted source archive without Git history.
These sets overlap and must not be added into a unique-test total. Its content,
package, frozen-source and curation checks passed. [Promotion gates](OPTIMIZED_PROMOTION.md)
separate source checks, actual answers and historical measurements.

The records below describe earlier releases and remain historical evidence.
They do not admit the optimized candidate.

**2026-09-20 research freeze:** the owner stopped optimization. [Frozen results and history](../perf/README.md) preserve the ordinary ~14.3 wall tok/s research baseline, rejected MTP/upstream variants and unmeasured cancelled work. That preceding publication updated documentation only; its supported engine and protected DB616–621 were unchanged. Freeze refs and regional backup evidence: `gs://driftbench-dsv4-uc/results/glm52_research_freeze_20260920/final-promotion.json`.

**Repository curation completed 2026-09-15** ([curation objective](CURATION_PLAN.md)).
Every tracked file has a justified role in [the curation ledger](../curation/README.md):
1,971 starting files became 616, with 1,380 originals removed and
exactly recoverable. Curation changed no numerical source, admitted identity or
historical evidence; the deployment evidence below is unchanged history.

Initial release, 2026-09-14: the supported single-request deployment passed its
protected smoke test, **DB621**, and was promoted to private main. That historical
promotion is bound by the final record below. It is not approval of the current
curation branch or a claim that main is already lean. No further benchmark campaign.

## Verified scope

| Capability | Direct evidence | Limit |
|---|---|---|
| Full short decoder | DB567; corrected batched 2K DB610 | Not broad task quality |
| 128K retrieval | DB616–619, all four depths | Four protected passkey prompts |
| Full 256K E0 | DB620 | Capacity/performance, NO_CORRECTNESS_ORACLE |
| Ordinary user response | DB621, release execution ca545aa4 | One site-specific request, sampled capacity166912 |
| Request continuation | DB621, same live session/cache/RNG | No crash recovery or persistent KV |
| First-token delivery | DB621, rank0 local JSONL write/flush | Not a network-delivered latency claim |
| Cold/HLO/memory/trace/cleanup | DB621, 32 owners, 8-host traces, authenticated idle | Same retained hardware/assets only |
| Installation | Fresh63-dependency install and CPU imports | Not a portable hosted service |
| Automated checks (initial release) | Final431passed,1optional skip,2upstream warnings | CPU tests do not replace hardware proof |
| Automated checks (curated main) | Release check passed on 2026-09-16 (pytest step 524 passed, 1 skipped; doctor, content audit, frozen-source pin `edecdd94`, compileall and isolated wheel install all clean); whole CPU tree 2,832 passed, 122 skipped, 173 failed, 0 errors in 1 h 30 min on code pin `cc2b36de` ([TESTING](TESTING.md)) | Same scope; skips are local-evidence replays |

## User-response admission

[Original sealed receipt](user-response-db621-sealed-20260914.json).
Execution/replay commit: ca545aa4a47b5c00ecde179623f9e094bddfdb22.
The ordinary prompt returned the requested word READY, ended at EOS after71
tokens, and all8 workers exited/published successfully. All-rank original replay,
checkpoint/scale/load, HLO, per-chip memory, trace and ownership checks passed;
DB621 and its regional generation-bound archive are linked by the receipt.

Cold loading/compilation:2283.431s. Prompt:19tokens. Local TTFT:9.368s.
Request wall:50.241s, **instrumented**; not steady profiler-free service latency.
69 ordinary decode samples, excluding the instrumented sample:147.975/240.609ms
p50/p99,6.572walltok/s. Maximum measured HBM:28,398,567,936bytes/chip.
These are one short smoke request's measurements, not a speed target or benchmark.
The separate cold replay checks44calls/rank and10unique graphs; request execution
and cache validation come from the joined request replay, not the cold check alone.

## Benchmark cancellation and quality limits

Owner cancelled remaining questions on2026-09-14 to finish the repository.
All8 original workers were stopped by authenticated process identity; supervisors
published originals successfully. Completed rank0 records:15GPQA-Diamond answers,
12marked correct by the original scorer,3marked incorrect; all15ended at EOS.
Item015 was interrupted after57,267token records. No AIME questions completed.

This incomplete prefix is **not a dataset accuracy estimate or model-card parity**.
A known full-option-versus-letter GPQA extraction defect remains recorded in the
original scoring audit; no revised scores replace originals. No extra model run,
paid AIME judge or full benchmark-success seal is required for this release.
[Cancellation evidence](benchmark-owner-cancellation-20260914.json).
Full GPQA/AIME quality, public-card parity and broader task quality remain unknown.

## Completion checklist

- [x] Supported imports/assets and entry points mapped; obsolete removal recoverable in Git.
- [x] Fresh pinned dependency installation and CPU imports.
- [x] Real ordinary user request and protected release deployment (DB621).
- [x] Exact published release pin accepted without using the research branch as execution.
- [x] Checkpoint loading used retained verified assets; direct recovery recipe and metadata checked.
- [x] Automated CPU release checks; source and package checks.
- [x] Known third-party notices and bounded secret/private-content/history audit.
- [x] Partial benchmark and cancellation scope recorded without a false success claim.
- [x] Same-live-session continuation, local delivery and operational limits documented.
- [x] Adversarial self-review; no independent review claimed.
- [x] Pre-promotion release/Git-store checksum and generation verification:10,262files.
- Final main promotion/ref/private/idle verification: see the authoritative record below.

## Final publication record

`gs://driftbench-dsv4-uc/results/private_release_20260914/final-promotion.json`
records the exact published main/release pins, retained research pin, canonical
detached checkout, final checksum/generation mirror and authenticated idle fleet.
The full mirror manifest is alongside it. These small final-pin-dependent records
live outside Git to avoid a self-referential commit; missing/failed verification
must not be interpreted as a successful promotion. No branch is force-pushed.

The installed five-minute same-region mirror now explicitly includes this release
tree and shared Git history with content comparison; its original locks, other
destinations, region guard and model-weight exclusions are preserved.

The source remains private. Future public release still needs an owner-selected
license for original code and renewed historical provenance/privacy review.
No HTTP endpoint, concurrent batching, durable KV recovery, speculative decode,
portable infrastructure provisioning or bundled model distribution is supported.
See [operations](OPERATIONS.md), [inference](INFERENCE.md), [checkpoint recovery](CHECKPOINTS.md)
and [audit scope](READINESS_AUDIT.md). Do not rerun DB616–621 for cosmetic cleanup.
