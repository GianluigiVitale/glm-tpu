# Release status

What was measured on the hardware, and what the current tree is. Two releases
are covered: the **GLM-5.3 inference release** of 2026-09-22, whose
measurements below are unchanged, and the **public-structure release** (this
tree), which restructured that engine without changing what runs on the TPU and
proved it on the CPU and on the fleet. The state of the work is in the
[handoff](../development/HANDOFF.md).

## The current tree

The public-structure release, tag `v1.0.0` on `main` (developed on the branch
`release/public-structure-20260922`), restructures the released engine
(the research tree at `181c013e`, preserved at the tag
`archive/research-20260922`) into the layout of the
[README](../../README.md#repository-layout): one package with public module,
class, function and Pallas kernel names; one untracked, validated site file for
every deployment value ([template](../../examples/site.example.toml)); a launch
policy that admits only a clean checkout of a pushed commit on an allowed
branch; a command line split into subcommands, with `checkpoint inventory`,
`checkpoint verify` and `collect-env`; rewritten documentation; the Apache
License 2.0. The pre-refactor release is kept at the tag
`pre-refactor-main-20260922`.

The restructuring was checked on the CPU, commit by commit, against records of
`181c013e` ([equivalence harness](../../tools/equivalence/README.md)): the TPU
device programs of the 98 fixture-tier and 66 production-tier programs, lowered
for the TPU with the compiler options production binds; the CPU execution
goldens; the checkpoint identities; the import closures; the executed functions;
the wire formats. A few early commits deferred the heavy gates to their stage's
close, as their messages state. The device-program, numerics and identity
records have not changed since the harness was finished (`d54572a3`, before the
first restructuring commit) and were only ever recorded from the `181c013e`
production paths. After that the other records changed only in separate,
reviewed re-baseline commits that name the stage, work unit or host change
behind the difference.

Behaviour changes, each declared and tested in its commit: the resident stop
keeps every collected record (the release crashed there with `FileExistsError`;
fixed in `79e9b39e`); the site file replaces the committed site configuration
and the launch policy replaces the frozen-source guard; the remote helpers are
real modules whose exact text is pinned for each run; `read_bounded` refuses a
directory like any other input that is not a regular file; the HLO admission
profile, the admission functions, help texts and error messages no longer carry
development labels; `collect-env` replaces the environment file (`doctor` stays
as an alias).

## TPU comparison of the current tree

On 2026-09-27 the tree at `ab4c6582` ran on the eight hosts and 32 TPU v4 chips
of the release, against golden runs of `181c013e` made the same day on the same
fleet (the original golden runs were lost when the TPU slice was recreated on
2026-09-24). Both used a checkpoint re-packed from the pinned source whose 32
slot records equal the sealed manifest, 32,768 combined slots, greedy decoding
and the same prepared GSM8K requests. No file under `glm_tpu/` changed after
`ab4c6582`. [Method](../../tools/equivalence/README.md#tpu-comparison-compare_runpy).

| Check | Result |
|---|---|
| B1: 10 sequential requests | 10/10 token streams and answers identical |
| B2: 4 concurrent conversations | 4/4 identical |
| R3: resident session, first request and two inbox rounds, then stop | 3/3 identical; the stop exits 0 with every record collected, where `181c013e` ends in `FileExistsError` |
| Slowest-host decode speed, this tree / `181c013e` | 0.994–1.024 |
| Compiled programs (`authenticity`) | the TPU originals of both trees equal the programs the harness lowers on the CPU: 9/9 programs, 1,062/1,062 Pallas kernels |
| Site identity (`site-check`) | pass before and after the runs; a baseline with one changed fact fails |
| Launch policy | a checkout with an untracked file is refused; the remote helpers each run staged were the commit's own |
| Chat UI and `/v1` API on a resident session | the model list, a chat answer and a streamed chat answer returned 200, the stream ended with `[DONE]`, the UI page carried its content-security policy, and the stop exited 0 |

This proves token identity for these 17 requests at 32,768 slots. It is not a
new speed or quality measurement. The other capacities are covered on the CPU:
the program fingerprints include 8,192, 32,768 and 166,912 slots, and 262,144
slots is covered by a unit test. The raw run directories are kept privately;
this page and the harness README are the committed record.

## GLM-5.3 release measurements

Resident ordinary inference was executed at
`5c3c1d6bee18817fdd4d665923747e233e76ee4a`. It produced a correct solo answer,
then reused the same loaded model for 770 independent GSM8K requests.
[Resident result receipt](glm53-resident-results-20260922.json).

That 32K session was stopped on 2026-09-22 and replaced by a 166,912-slot
session for agent work. Every measurement below
describes the earlier 32K session. The 166,912-slot session has one completed
answer of its own at 11.20 decode tokens/s and 28.94 GB peak HBM per chip; a
262,144-slot profile was refused by HBM admission.
[Capacity profiles](glm53-context-profiles-20260922.json). That session ended
when the TPU slice was recreated on 2026-09-24; at the close of the TPU
comparison on 2026-09-27 no resident session ran.

### Single-chat measurement

| Measurement | Slowest-host result |
|---|---:|
| Decode | 13.570447 tokens/s |
| Prompt prefill | 85 tokens / 0.965029 s |
| Decode time | 239 timed tokens / 17.611801 s |
| Total output including thinking and prefill-produced first token | 240 tokens |
| Cold verification/loading/compilation | 1,219.894875 s |
| Worker including initialization, cold startup, warmup and answer | 1,249.783933 s |
| Peak HBM per chip | 28,228,733,440 bytes |

Decode includes fleet votes and local token writes. It excludes prefill, startup,
queue overhead and final text decoding. Worker wall excludes controller staging
and SSH. Compilation is included in cold time; overlapping intervals must not
be added. The completed final answer was 70,000, checked against reference and
arithmetic. All eight hosts agreed on tokens and graphs, and memory checks
passed. Authenticated workers remained live with libtpu after answering.

### Partial GSM8K evaluation

The evaluation covered the first 770 of the 1,319 test questions of
`openai/gsm8k` (configuration `main`, revision
`740312add88f781978c0658806c59bc2815b9866`): test rows 0–769, in order. The
remaining 549 questions were not run. All 770 were freshly executed in
independent conversations; prior demonstration answers were not reused.

| Outcome | Count |
|---|---:|
| Correct completed final answers | 740 |
| Incorrect scored final answers with normal EOS | 27 |
| Context exhausted, counted incorrect | 3 |
| Total processed | 770 |
| Not run (test rows 770–1,318) | 549 |
| Accuracy on the processed subset | 96.103896% |

Prompts were original questions plus a boxed-answer format instruction, without
brevity instructions or examples. References were separate from model inputs.
Decoding was greedy, thinking enabled/max, with every remaining slot after full
tokenization in a 32,768-slot cache. The four-hour operational deadline applied
per question. No answer-driven retries, resampling or hidden output caps.

Scoring considered only the final channel after `</think>` at normal EOS. It
used the last numeric boxed answer, then a `####` numeric marker, then the last
number in the final channel, comparing normalized Decimal values exactly.
There were 763 boxed extractions and four last-number fallbacks; all four
fallbacks were incorrect. Three unfinished context-exhausted outputs were never
scored from numbers in their reasoning. Scoring verifies final numeric results,
not the validity of every intermediate step.

The run emitted 298,578 tokens including thinking. Of these, 297,808 were timed
decode tokens; summed slowest-host decode time was 22,067.408499 s, giving
13.495377 tokens/s. Prefill totaled 60,764 input tokens in 756.988795 s.
These sums exclude queue/SSH collection overhead and the prior cold startup.
Peak HBM stayed at 28,228,733,440 bytes/chip. All-host token/graph identities and
memory evidence were checked per result. The remaining queue was withdrawn;
the scorer/notifier exited, while the model and its workload leases were retained.

This is **not a full-test-set GSM8K result**. Public benchmark familiarity,
ordered-prefix selection and a prefix length that was not fixed in advance
limit comparison with independently chosen complete evaluations. No full-32K-input quality claim.

### Four concurrent chats

The separate four-chat run at `c2f60efe` completed four correct normal-EOS answers,
with 4.89–5.12 tokens/s per active chat and 9.269431 aggregate tokens/s.
Its 317-token sequential prefill took 3.884926 s; cold startup took 1,178.091731 s;
peak HBM was 28,789,189,632 bytes/chip. That invocation verified eight-host cleanup.
The [receipt](glm53-four-answers-20260921.json) preserves all boundaries.
Its concise-explanation suffix differs from the partial benchmark prompt.

The receipts are the release's records. When this page was revised on
2026-09-27, the TPU node name and the zone in the four-chat and resident
receipts, and a host path in the four-chat receipt, were replaced by
`<redacted>`; every measured value, hash and run name is unchanged, and the
originals are in the history (`git show 60b68b4a:docs/release/<file>`).

## Verification and review

At the commit that last revised this page (2026-09-27), the CPU suite outside
the equivalence wrappers had 785 passing tests and one skip (the site-bound test,
without a site file); the light equivalence wrappers (`-m "not cpu32"`) had 154
passing tests and the `cpu32` wrappers 8; `check`, `check --tier production`,
`check --gates G6-static` and `selftest` (23 cases) passed with every record
unchanged; the helper contract tests had 49 passing tests
([TESTING](TESTING.md)).

Before the restructure, the release's main passed 629 CPU tests with one skip,
and its resident change passed 65 affected tests; unchanged numerical code
reused those and the TPU evidence. Its final release checks were recorded in
publication receipts kept outside the repository.

Review: both releases were built with AI coding agents under the author's
direction. The GLM-5.3 release was reviewed by the agent that built it. In the
public-structure release, from the commit that made the Pallas kernel names
public (`27b1b411`) on, each unit's code commit was checked with its raw
equivalence-check evidence by a separate adversarial verifier (another agent
session) before it was pushed (in all units but one, on the local commit);
fixes for its findings were amended into the unpushed commit, the other
findings were recorded as follow-ups, and a re-baseline commit, where the unit
needed one, was recorded after that pass. Before that commit there was no
standing verifier step: some commits cite review rounds or a verifier, many cite
none. The maintainer reviews the units after they are pushed. None of this is
independent human review.

## Limitations

Source: 141 verified shards / 755,632,050,320 bytes; packed checkpoint: 32 slot files /
786,181,673,984 bytes. The checkpoint and source identities are pins in the
untracked site file's `[checkpoint]` table ([CHECKPOINTS](CHECKPOINTS.md),
[template](../../examples/site.example.toml)). The [local chat UI](../UI.md) and
its stateless [OpenAI-compatible API](../API.md) attach to a resident session
through its inbox. Ordinary capacity profiles are 8,192 / 32,768 / 166,912 /
262,144 combined slots; the 256K agent profile carries no measured speed or
quality claim of its own. No dynamic cache capacity, online batch admission or
durable restart recovery.

The equivalence proof lowers the TPU programs on the CPU but does not run XLA's
TPU compilation, so a compiler or libtpu change is caught only by a TPU
comparison; the site check reads rank 0's assets only. The Git history keeps
infrastructure names of earlier commits; a full-history scan found no
credential in it ([SECURITY](../../SECURITY.md#history-scan)).
[History](GLM53_MIGRATION.md) preserves earlier results and failures.
