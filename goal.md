# Goal — Deliver working GLM-5.3 inference on TPU

Finish the GLM-5.3-FP8 project on private main: real weights, four concurrent
conversations, completed correct answers to the fixed four GSM8K questions,
measured speed, clear instructions and a reviewer-ready source archive.
Carry this through to a working release. Preserve the completed GLM-5.2 release
and its history.

## Authority to finish

The owner authorizes downloading and resuming the weights, diagnosing failures,
fixing code or operational defects, and retrying acquisition, packing, tests and
inference as needed to achieve this goal. Do this autonomously without asking
again for ordinary fixes or retries. Earlier no-retry, no-fix and restrictive
workflow wording is superseded by this instruction.

Use evidence to choose the next action. Preserve successful work and failure
receipts, distinguish a live job from a terminated one, and continue from verified
progress. A failed attempt is a problem to repair, not a reason to abandon the
goal. Keep one owner of the fleet and respect workload/sync locks so recovery
does not duplicate or interrupt a live job.

Reuse the existing implementation and environment where they work. Make the
engineering changes needed for compatibility, correctness, memory, usability and
reproducibility. Record material changes and test their effects. Prioritize the
working release over additional research.

## Starting point

- Work in /home/gianl/glm-tpu-glm53 on release/glm-5.3.
- GLM-5.2 is preserved at tag glm-5.2, commit
  edbced29315b6afce92cf994ae70785bf8f6d995. Its weight payloads were retired.
- Use official FP8 zai-org/GLM-5.3 at
  aca966e4e02791568aa6a4ced368624b3d897f42:
  141 shards, 755,632,050,320 bytes.
- Read /home/gianl/glm-run/glm53_migration_20260921/HANDOFF.md and actual
  dispatch, terminal and reconciliation receipts. The first acquisition failed;
  17 verified shards are preserved. Resume the missing source, retaining the
  verified generations instead of downloading everything again.
- Use the existing db-v4-64-od fleet in us-central2-b: eight hosts, 32 TPU v4
  chips, with gs://driftbench-dsv4-uc in US-CENTRAL2 for storage and backup.
- Pin the actual new weights, configuration, tokenizer, chat template and license.
  Establish compatibility through inspection and execution. Historical GLM-5.2
  measurements remain historical evidence.

## Working behavior and answer quality

Verify the complete source and prepare the owner shards and other assets actually
consumed by the engine. Inspect the legacy dense-overlay dependency: prepare it
if required; if the ordinary path already carries those weights in its owner
shards, document that fact and use the working path.

One loaded model must decode four independent conversations concurrently, with
separate histories, output streams and stopping. Use 32,768 combined slots per
conversation and sequential prompt prefill as the starting configuration.

Use the same four question strings in
/home/gianl/glm-run/batched_four_20260921/questions.json: GSM8K test rows 0–3 at
revision 740312add88f781978c0658806c59bc2815b9866.
Keep private-cases.json and reference answers separate from model inputs.
Acceptance requires four completed correct final answers, with arithmetic checked
against the references. Report failed or unfinished attempts honestly; mentioning
a correct number during unfinished reasoning is not a completed answer.
These four examples are a functional check, not broad model accuracy.

Enable thinking at maximum effort. Give each request all remaining context slots
after tokenizing the full input, template and history. Reasoning and the final
answer share this space. Remove the old 1,024/2,048-token defaults and hidden
thinking/output limits from this path. Let each conversation stop at normal EOS.
Finite cache capacity still applies: report context exhaustion, timeout or failure
as incomplete. Set operational deadlines to accommodate the full allowance and
repair/retry incomplete executions when evidence supports doing so.

## Measurements and tests

Run the CPU test suite and required real-weight checks. Use JAX_PLATFORMS=cpu for
CPU tests and source audits. Verify independent conversation state, current graph
and memory fit, all-host token agreement and authenticated eight-host cleanup.
Token agreement and answer correctness must both be checked.

Measure the actual four-question run: cold loading/compilation, prompt prefill,
per-conversation decode speed while active, aggregate batch throughput, token
counts, end-to-end duration and peak memory per chip. State timing boundaries
and use the slowest host where appropriate. Keep aggregate throughput distinct
from each conversation's speed. Report the actual hardware and capacity limits;
short inputs do not prove full-32K-input quality.

## Release deliverables

Update the README, project summary and installation/inference instructions around
GLM-5.3, with one recommended command using the full remaining output allowance,
current measurements and receipts, concrete limitations, and an offline CPU
inspection/test path. Keep the presentation clear and suitable for PhD reviewers.
Explain the original engineering contribution versus reused model/compiler work.
Link GLM-5.2 and failed or superseded attempts as history.

Keep justified code, tests, configuration, licenses and detailed historical
evidence. Preserve original source identities, research branches, DB616–621 and
Git history. Update the curation ledger and record recovery before removing
clutter. Prepare a compact source/docs archive tied to the final commit, excluding
weights, secrets, private prompts, raw outputs and databases.

Resolve material findings, finish release checks, verify the regional backup,
then commit, push and merge the actual working implementation into private main.
Self-review must be described as self-review. Keep the repository private and
retain attribution and licenses.

## Continuity and finish line

Read AGENTS.md and current handoffs, applying this latest owner authorization
over older retry restrictions. Use completion-triggered wakeups and durable
handoffs for long jobs rather than repeated polling. On completion, verify the
real result and cleanup, repair problems, and continue until the release works.

Final handoff: main commit, usable command, measured speed, four answer outcomes,
README/archive locations and concrete limitations. Completion requires the
working release and evidence; downloaded weights, compilation, documentation or
passing CPU tests alone are intermediate progress.
