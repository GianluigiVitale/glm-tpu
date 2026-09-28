# GLM-5.3 ordinary inference

Run on rank 0 of the TPU v4 slice, from a clean, published checkout on a branch
the site's launch policy allows, with the [pinned environment](INSTALLATION.md),
a site file and the verified [checkpoint](CHECKPOINTS.md). The wheel alone is not
a deployment. [OPERATIONS](OPERATIONS.md) describes what the controller does.

## Start a resident session

When no session owns the fleet:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask "Your question" --keep-loaded --wall-seconds 14400
```

The default is 32K combined prompt, history, thinking and output slots
(`--context 32k`); the other capacities are `8k` (8,192 slots), `128k` (128K
prompt / 163K total: a prompt of up to 131,072 tokens in 166,912 slots) and
`256k` (262,144 slots, offered but refused
by HBM admission on 32 TPU v4 chips when it was tried). The cache is allocated and
the TPU programs are compiled for the chosen capacity at load, and every request
of the session must use it. Thinking is on at maximum effort; omitting
`--max-new-tokens` gives generation every slot left after the full input is
tokenized (at most 163,840). Input is never silently truncated. `--site` names
the site file when it is not in the default place.

The command prints `RUN <run directory>` and keeps the loaded model and its
compiled programs after answering. Token events stream to a private JSONL file;
the first answer is `item000/answer.txt` in the run directory. Each completed round writes
`resident-measurement.json` with the all-host agreement. The cold start is paid
once per session. A second launch while a session owns the fleet is refused; use
the session's inbox instead.

## Submit another question to that model

Prepare the input without launching anything:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask "Next question" --prepare-only
```

It prints `PRIVATE_INPUT <prepared request.json>`. On the controller host,
atomically hard-link that file into the session's inbox; for the first follow-up
of a session:

```bash
ln /path/to/prepared/request.json /path/to/run-directory/inbox/0001.json
```

Use the next unused sequence number (`0002.json`, and so on), one producer at a
time. After a round completes, `resident-ready.json` names its sequence. Never
overwrite, delete or replace an admitted input. The prepared file must be
owner-only and on the same filesystem for the hard link. The controller checks
the input's integrity and capacity and sends the identical request to all eight
workers.

Follow-up results are in `resident-0001/` and so on: tokens, decoded text,
per-host reports and `resident-measurement.json`. Every request starts from a
fresh state: include the full message history to continue a conversation. The
Python API `glm_tpu.engine.request.from_messages` prepares user and assistant
history with the pinned tokenizer and template for a given `context_capacity`;
`ask` prepares a fresh user question. `glm-tpu prepare-request` prepares a
message file into a request for the `ordinary-greedy-8k` or `ordinary-greedy-128k`
profile only; use `ask --prepare-only` for the 32K and 256K capacities. A
prepared batch can queue up to ten questions that run one after another, sharing
the weights but not the conversation state.

Raw answer text contains the reasoning and the model's control markers. The
final answer is the text after `</think>`, and it is complete only at a normal
EOS. Context exhaustion or a failure is incomplete, whatever numbers appear in
unfinished thinking. These private files stay outside Git.

## Stop and failure behaviour

A resident session has no idle timeout. `--wall-seconds` bounds the start-up
together with the first request group, and then each later request group. There
is no process-restart or durable KV recovery. An invalid input or a worker or
controller failure ends the session through authenticated cleanup; validate
prepared inputs before publishing them.

To unload the model after outstanding work finishes, atomically place an
owner-only file containing `{"stop":true}` at `inbox/stop.json`. The controller
then verifies cleanup on all eight hosts and releases its workload locks. This
stops the model, not just the queue. There is no in-flight cancellation that
keeps the model loaded.

A successful resident record says `all_hosts_idle_after=false`: the workers and
both workload locks were meant to stay owned. The live-worker checks and the
final shutdown cleanup are separate receipts.

## Four conversations at once

The separate batch path uses one shared model and four independent 32K caches,
prefilling the prompts one after another and decoding them together:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask --questions /path/to/questions.json --concurrent --wall-seconds 14400
```

The input is a private JSON array of one to four question strings. The
invocation exits after its group and verifies cleanup on all eight hosts.
`--keep-loaded` applies to sequential requests only, not to a concurrent batch;
do not run a batch while a resident session owns the fleet.
[CONCURRENT](CONCURRENT.md).

[Results](STATUS.md) separate the solo, resident-evaluation and four-chat
evidence. The [chat UI](../UI.md) and the [`/v1` API](../API.md) attach to an
existing resident session (one started by `ask --keep-loaded` included, once it
has answered its first request) and use its queue one request at a time. There is no
online batch admission and no claim of full-32K-input quality. The 8K and 128K
profiles have their own recorded scope; the 256K profile has no measured speed
or quality result of its own. The legacy sampled GLM-5.2 interface is history,
preserved at the tag `archive/research-20260922`
(`git show archive/research-20260922:docs/release/INFERENCE.md`).
