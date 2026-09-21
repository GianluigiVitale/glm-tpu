# Release status

The September21 [concurrent-conversation candidate](CONCURRENT.md) failed its
eight-chat,32K-per-chat TPU compilation gate:34.09GiB required versus30.75GiB
available per chip. Zero answers were generated; authenticated cleanup passed
on all eight hosts. The candidate remains on its private release branch and
has not been promoted to main. No retry was launched. See the
[failure receipt](concurrent-failure-20260921.json).

## Ordinary implementation: validated release

The actual ordinary engine, question interface and protected controller are
included in this release. Executable source `9469cd733df7fabe7f6f421ab0fad60801cf3138`
passed the owner's final answer criterion: **one correctly completed GSM8K
example**, replacing the cancelled ten-question requirement.

[Answer and hardware receipt](single-answer-20260920.json): pinned GSM8K test
row0 returned18, matching the gold and the independently checked arithmetic
`(16 − 3 − 4) × 2`. It ended at EOS after265 output tokens. All eight hosts
agreed, admitted fresh graphs/memory and exited successfully; authenticated idle
checks passed on all eight. This is a functional example, not a dataset score.

| Slowest-host measurement | Value |
|---|---|
| Cold load/compile | 1,102.028924 s |
| Prefill,92tokens | 0.937823 s;98.099531 tokens/s |
| Decode,264timed tokens | 18.147396 s;14.547542 tokens/s |
| Maximum observed HBM per chip | 28,228,733,440 bytes |
| Scope | Ordinary greedy8K;32owners;8,192 combined slots |

The preceding [8K integration receipt](ordinary-integration-20260920.json) at
`4f551e6b` supplies29/29 reference-prefix agreement. It stopped during reasoning;
it is not the completed-answer example. The final executable retains that
numerical path, with the queued interface, long-cache ownership and explicit
output-budget changes covered by affected CPU checks.

## Checks, packaging and promotion

The [original CPU release check](optimized-cpu-check-20260920.json) passed
571tests, one skipped, plus source/content/package checks. The extension passed
114affected checks with one optional skip; the donation changes passed20prefill/
runtime and2decode checks, sync-lock changes13checks, and final output-budget
changes30checks. Sets overlap and must not be added into a unique-test total.
Forty portable checks also passed from an extracted source archive without Git.
[Final check index](ordinary-release-checks-20260920.json) binds reused evidence
and the documented self-review scope. Final documentation/content/package and
archive checks are recorded alongside the private publication receipt.

No numerical source changed after the successful answer. Final commit identities,
source archive hash, regional generation-bound readbacks and verified private
main promotion are recorded outside Git in
`/home/gianl/glm-run/gsm8k_acceptance_20260920/publication/final-promotion.json`.
This avoids a self-referential final commit. See [package instructions](SHAREABLE_PACKAGE.md).
A missing or failed publication record does not establish main promotion.

## Evidence limits and preserved history

One request generates at a time. The queue accepts up to ten; simultaneous
model batching is not implemented. Long-capacity compilation succeeded after
the cache-ownership fix, but no full131072-token input or ten-answer check
completed. The difficult run was explicitly cancelled by the owner, with
authenticated cleanup, before the single GSM8K example. No failures or partial
outputs were relabeled successful. [Detailed integration history](../perf/ordinary-release-20260920.md)
preserves those attempts and recovery references.

Review is assistant self-review, not independent review. Original code,
attribution, licenses, research branches, frozen MODEL_SOURCE and DB616–621
remain preserved. No resources were created or upgraded and no research campaign
was resumed. The repository and reviewer archive remain private.

## Earlier releases — historical evidence only

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
