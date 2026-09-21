# Goal — Run GLM-5.3 on TPU and measure real answer quality and speed

Deliver a working, understandable GLM-5.3-FP8 project on private main using the
existing shared-model implementation and TPU fleet. Run the same four GSM8K
questions concurrently, inspect their actual completed answers, and report
measured speed honestly. Preserve the completed GLM-5.2 release and history.
Finish this migration; do not start another optimization or benchmark campaign.

## Reuse the existing implementation and acquisition

- GLM-5.2 is frozen at private tag `glm-5.2`, commit
  `edbced29315b6afce92cf994ae70785bf8f6d995`. Its code, reviewer archive,
  measurements, originals and history remain preserved. Its weight payloads
  were intentionally retired; do not restore or reacquire them.
- Work in `/home/gianl/glm-tpu-glm53` on `release/glm-5.3`. Reuse the existing
  ordinary engine, four-conversation batching, packing path and protected
  controller. Adapt only what GLM-5.3 requires.
- Use official FP8 `zai-org/GLM-5.3`, pinned at
  `aca966e4e02791568aa6a4ced368624b3d897f42`: 141 shards, 755,632,050,320 bytes.
  Do not substitute the BF16 checkpoint.
- The existing source acquisition is already dispatched. Read
  `/home/gianl/glm-run/glm53_migration_20260921/HANDOFF.md`, `dispatch.json`
  and its completion receipts before acting. Never duplicate or automatically
  retry it. A process exit is not success.
- Architecture/configuration and tensor mappings appear compatible; the weights,
  chat template and license changed. Pin the actual GLM-5.3 assets, preserve
  their attribution, and establish compatibility instead of assuming it.
  GLM-5.2 receipts do not qualify GLM-5.3.

## Required working behavior

Complete source verification and prepare the required owner shards and dense
overlay on the existing hosts. Connect the real GLM-5.3 weights and chat template
to the existing command. One loaded model must decode four independent
conversations concurrently, with separate histories, output streams and stopping.
Retain the 32,768-total-slot capacity per conversation and the sequential prompt
prefill behavior unless a concrete necessary change is identified and reported.

Use the same four questions already selected in
`/home/gianl/glm-run/batched_four_20260921/questions.json`: GSM8K test rows 0–3
at revision `740312add88f781978c0658806c59bc2815b9866`.
Keep `private-cases.json` and its reference answers separate from model inputs.
Do not replace difficult examples, resample, insert gold answers or use a new
benchmark suite. Prepare new GLM-5.3 requests from those question strings;
do not reuse GLM-5.2 prepared requests or their old output budgets.

## No artificial output or thinking cap

Remove the old 1,024-token and 2,048-token output defaults from this acceptance
path. Do not impose a separate token cap on reasoning, a smaller hidden output
budget, or a low-effort/forced-short-answer workaround. Use thinking enabled at
maximum effort and let each conversation generate until its own normal EOS.

Finite context still applies: each request receives all remaining slots in its
32,768-slot cache after tokenizing the complete input, including the chat
template and any history. Reasoning and the final answer share that remaining
space. Wire that full per-request allowance through preparation, validation,
runtime and the documented command; omitting an output-cap flag must not silently
restore the old 2,048-token default. Do not advertise infinite context.

If the model exhausts its context, reaches an operational deadline, fails, or
never produces a final answer, report it as incomplete. Do not count an
intermediate calculation or a correct number inside unfinished reasoning as a
completed correct answer. Choose an operational deadline appropriate for the
full allowance; keep cleanup and failure handling intact. Do not silently
increase context, retry a failed workload or change the questions.

## Answer quality and speed

The acceptance check is four completed correct answers to the fixed questions.
Inspect each final answer and verify its arithmetic against the private reference.
Record normal EOS versus context exhaustion, interruption or failure separately.
The old GLM-5.2 result was three correct completed answers and one capped answer;
preserve that fact without using it as the GLM-5.3 result.

Measure from this same real-weight run rather than launching an extra speed
campaign. Report cold loading/compilation, prompt prefill, and decoding separately.
Include prompt/output token counts, per-conversation decode speed while active,
aggregate batch throughput, end-to-end batch duration and maximum observed
memory per chip. State the timing boundaries and use the slowest host where
appropriate. Do not confuse aggregate throughput with each conversation's speed.

Require fresh graph/memory admission, independent conversation state, all-host
token agreement and authenticated cleanup on all eight hosts. Reuse existing
passing checks for unchanged code; rerun checks affected by the migration or
output-budget changes. Token agreement is not answer correctness. Four short
GSM8K examples are a functional check, not broad accuracy or full-32K-input proof.

## Deliver the release

Keep main clear and reviewer-friendly. Update the README, short project summary,
installation/inference instructions, measured GLM-5.3 results and concrete limits.
Provide one recommended command that actually uses the intended full remaining
output allowance, plus an offline CPU inspection/test path. Link GLM-5.2 and
failed/superseded attempts as history; do not mix their headline measurements
with current results or turn the README into a progress diary.

Preserve justified code, tests, configuration, licenses, original evidence,
research branches and DB616–621. Update the curation ledger and retain exact
recovery before removing clutter. Prepare a compact final-commit source/docs
archive excluding weights, secrets, private prompts, raw outputs and databases.

Pass the affected release checks, resolve material findings, verify the regional
backup, then commit, push and merge the actual GLM-5.3 implementation into private
main. Self-review is not independent review. A documentation-only merge, acquired
weights, or successful compilation alone is not completion.

Final handoff: main commit, usable command, measured speed, the four answer
outcomes, README/archive locations and remaining limitations. If a required check
fails, identify the exact failure; do not claim GLM-5.3 is ready or substitute a
different task.

## Operating boundaries and waiting

Read AGENTS.md, HANDOFF.md and current migration/release status. Use only the
existing `db-v4-64-od` fleet in `us-central2-b`: eight hosts, 32 TPU v4 chips.
One workload at a time under both workload leases; respect both sync locks.
Authenticate idle/cleanup on all eight hosts. Only
`gs://driftbench-dsv4-uc`, US-CENTRAL2, within existing storage bounds.

The necessary GLM-5.3 acquisition, packing, integration and this four-question
check are authorized. No new resources, resource lifecycle changes, environment
upgrades, MTP/speculative decoding, kernel-tuning campaign, new engines/controllers
or benchmark suites. Preserve the frozen GLM-5.2 source identities; add explicit
GLM-5.3 provenance rather than rewriting old receipts. No force-push or history
rewrite. Tests and source audits use `JAX_PLATFORMS=cpu`.

Keep the repository private; do not send messages or publish externally.
Use completion-triggered wakeups with a durable handoff instead of repeated
polling. Inspect a completion once; if the job is still running, preserve it and
stop polling. Never duplicate or automatically retry a workload. Leave the
paused slash goal paused. Editing this prompt does not launch another job.
