# Goal — Finish the project for the PhD application

## Latest owner test — four at the same time

The owner explicitly requested one test of four concurrent conversations after
the eight-chat failures. The unchanged implementation ata252eb01 ran four32K
caches concurrently, passed fresh graph/memory gates and all-rank agreement,
and exited successfully with authenticated cleanup on all eight hosts.
Three GSM8K answers completed correctly; one exhausted1024output tokens without
a final answer. This is hardware concurrency success, not four-for-four answer
completion. Preserve the result, do not automatically retry, and leave main
unchanged while the answer gate remains incomplete. The slash goal stays paused.
See /home/gianl/glm-run/batched_four_20260921/HANDOFF.md and the four-chat receipt.

## Owner-requested memory repair — September21

The changed-code test at18dda892 is now terminal and failed batch-decoder
compilation:31.17GiB required versus30.75GiB available per chip,437.87MiB over.
Zero answers; all eight workers exited1 and authenticated cleanup passed.
The temporary3.05GiB cache copy remains. No automatic retry or main promotion;
this requested feature is still unfinished. The following authorization and
original failure are retained as history, not authority to replay the workload.

After the failed eight-chat test, the owner explicitly asked to fix the memory
problem and make the requested shared-model batching work. Repair the cache
ownership defect, check affected CPU behavior, then test the changed implementation
with the same eight GSM8K questions at32K slots each on the existing fleet.
This authorizes a corrected-code test; it is not an automatic retry of21174374.
Preserve that failed run, all existing constraints and the working main release.
No new resources, upgrades, unrelated tuning, duplicate workload or automatic
retry loop. Use the existing controller and all workload/sync locks; keep the
slash goal paused. Operational handoff:
/home/gianl/glm-run/batched_memory_fix_20260921/HANDOFF.md.

## New authorized feature — eight batched conversations, September21

The release below is complete. The owner now explicitly requests one loaded
model generating for up to eight concurrent conversations through batched
decoding, with32Ktotal context per conversation, tested on easy cached GSM8K
questions. This authorizes the necessary isolated implementation and one bounded
hardware test; it does not reopen unrelated optimization or cancelled queues.
Preserve the released single-request path and all prior evidence. Develop on
release/batched-conversations-20260921, prove request isolation and CPU numerical
behavior, then use the existing protected fleet controller and memory gates.
Carry forward private visibility, frozen source, no resources/upgrades, both
workload/sync leases and no automatic retries. Use completion-triggered waiting.
The operational handoff is /home/gianl/glm-run/batched_conversations_20260921/HANDOFF.md.

The authorized test at21174374 failed during TPU batch-decoder compilation:
34.09GiB required against30.75GiB available per chip. No answer was generated.
All eight workers exited and authenticated idle cleanup passed. The candidate
is preserved on its private branch; main21f495de remains the working release.
No automatic retry or smaller substitute run is authorized by this result.

## Latest owner steering — usable answers first

The owner subsequently replaced the ten-question acceptance gate with **one
correctly completed answer**, explicitly allowing an easier GSM8K question.
One cached GSM8K test example through the existing 8K ordinary path is sufficient
for that answer gate. Do not resume the difficult ten-question campaign or
require a full 128K question to finish this release. Keep the implemented queue
and larger profile, with their actual evidence and limits described honestly.
The original implementation, README, history, archive, backup and private-main
deliverables still apply. The slash goal is paused; use the saved handoff and
completion-triggered continuation, without repeated model polling.

That one-answer check passed at executable9469cd73: GSM8K test row0 returned18
and ended at EOS; all eight workers exited successfully and cleanup passed.
No further question runs are required. Finish only the documented final checks,
verified regional backup, commit-bound package and private-main promotion.
The publication receipt named in docs/release/STATUS.md establishes their final
identities and completion. Earlier research and queue instructions below are
preserved context, not additional acceptance gates.

The preceding instruction prioritized submitting ten concurrent questions, correct completed
answers, and roughly 128K context. Exercise actual randomly selected difficult
benchmark questions and inspect their final answers. Do not treat an 8K
single-request release or token-prefix agreement as completion of that request.
The owner explicitly accepted ten requests queued for generation, one at a time.
Keep this scheduling behavior clear. Reuse the existing long-context runtime where possible;
preserve the active request and never duplicate its workload. The private-repo,
existing-hardware, lease, frozen-source, history and no-environment-upgrade
constraints remain. The prior release/package requirements below are retained,
but documentation work must not substitute for working question answering.

## Prior release objective

The retained release instruction is to finish on private main with
usable code, clear instructions, honest measurements, preserved history and a
compact source/docs archive tied to the final commit. This supersedes earlier
optimization goals. Finish the release; do not start another research campaign.

Reuse candidate `4f551e6b` in this worktree; its release checks passed 571 tests,
one skipped, plus source/content/package checks. Do not rebuild the integration.
Reconcile and collect existing validation
`optimized_request_20260920T170223095076Z`; never duplicate it. Writing this goal
does not launch a new workload. Historical speeds do not admit this release.

Required additions to the gates below: a bounded, checkable completed-answer
example using suitable same-code evidence; separate startup/prefill/decode
measurements; a short project summary; an offline CPU reviewer path; and a
private source/docs archive excluding weights, secrets, private prompts, raw
outputs and databases. Token agreement is not answer correctness. Explain
original contributions versus reuse and retain attribution/licenses.

Reuse passing checks and rerun only those affected by changes or failures.
Resolve material findings and verify regional backup before commit, push and
merge of the actual implementation. A documentation-only merge is incomplete.
The final handoff must name main, usable speed, validation, README/archive paths
and concrete limitations. Identify any failed gate without claiming readiness.
No new engines/controllers, acquisitions or benchmark suites. Keep the repository
private; do not send messages or publish externally. Work autonomously.

## Retained implementation scope and operational limits

The owner stopped optimization research, then requested that main actually run
the retained approximately 14.3 tok/s ordinary engine. A documentation-only merge
does not meet that goal. Follow the earlier supported-code extraction and curation
at c7f6d42f / 493b67de; preserve research history and protected originals.

Scope: promote D1 grouped experts with the empty-slot fix, D8 BF16 non-routed
weights, D10 DSA, D4 packed delivery, and D8/P1/P2 prefill. The measured profile is
greedy decoding with 8,192 total prompt/output slots. Preserve the legacy sampled
long-context entry explicitly; never silently substitute its sampling or capacity.
MTP, decode D5, fused reductions, global-max and fused EP remain excluded.

1. Extract only supported dependencies from pinned, trained research code.
   Keep frozen MODEL_SOURCE unchanged. Prove numerical/host interfaces on CPU.
2. Connect private request preparation, protected fleet launch, real checkpoint
   loading, fresh graph/memory admission, token delivery and authenticated cleanup.
3. Run one bounded real-weight release validation through the actual new entry.
   Check the DB610 reference prefix, all-rank agreement, timing and memory. This
   validates integration; it does not restart cancelled answer or MTP campaigns.
4. Record exact source, checks and receipts. Main README must describe the usable
   release and its measured limits, with future work at most two sentences at the
   end. Put detailed failed/superseded research in the retained history index.
5. Update file dispositions, pass release checks, resolve review findings, verify
   regional backup, and merge the implementation into private main. Report actual
   publication state. Self-review is not independent review.

Current worktree: /home/gianl/glm-tpu-optimized-release, branch
release/optimized-ordinary-20260920, from main e9ef0dda. Extraction and the 8K
integration check are complete. Queued 128K questions and final promotion are
tracked in docs/release/STATUS.md; do not rerun the completed 8K request.
Research remains recoverable at e3290fd8 and the pre-cleanup preservation ref.
See docs/release/OPTIMIZED_PROMOTION.md for provenance and gates.

Existing pod only: db-v4-64-od, us-central2-b, eight hosts / 32 v4 chips. One
workload under both workload leases; respect both sync leases during staging.
Authenticate all eight hosts idle before and after. No automatic workload retries,
resource lifecycle operations, environment upgrades or full-model safety copies.
pytest always uses JAX_PLATFORMS=cpu. Keep private inputs, weights and raw output
outside Git. Only gs://driftbench-dsv4-uc in US-CENTRAL2; retain storage bounds.
Commit/push private release milestones without asking; no force-push, history
rewrites, Co-Authored-By lines, subagents or external reviewers.
