# GLM-5.3 ordinary inference

Use a clean, published full checkout on authenticated rank0 of the existing
8-host/32-chip TPU v4 site, with the [pinned environment](INSTALLATION.md) and
verified [checkpoint assets](CHECKPOINTS.md). The wheel alone is not a deployment.

## Start a resident session

When no session owns the fleet:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask "Your question" --keep-loaded --wall-seconds 14400
```

The default is 32K combined prompt/history/thinking/output slots. Thinking is
on/max; omitting `--max-new-tokens` gives generation all remaining slots after
full input tokenization. There is no separate 1024/2048 thinking/output default.
The cache is allocated and TPU graphs compiled for the chosen capacity at load.
Each request must match that capacity. Input is not silently truncated.

The command prints `RUN /private/run/path` and retains the loaded model and
compiled graphs after answering. Token events stream to local private JSONL;
the first answer is under `item000/answer.txt`. Each completed round produces
`resident-measurement.json` with all-host agreement. Cold startup is paid once
per live session. Existing resident ownership is a refusal to launch a second
model; use its inbox instead.

## Submit another question to that model

Prepare inputs without launching hardware:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask "Next question" --prepare-only
```

This prints `PRIVATE_INPUT /private/prepared/request.json`. On the controller,
atomically link that prepared file into the existing session inbox. For the first
follow-up in a new session:

```bash
ln /private/prepared/request.json /private/run/path/inbox/0001.json
```

Use the next unused sequence number (`0002.json`, etc.), one producer at a time.
After a round is complete, `resident-ready.json` identifies that round's sequence.
Do not overwrite, delete or replace an admitted input. Owner-only files on the
same filesystem are required for this atomic hard-link operation. The controller
checks input integrity/capacity and sends identical requests to all eight workers.

Follow-up results are in `resident-0001/`, etc., with tokens, decoded text,
per-host reports and `resident-measurement.json`. All requests use fresh state;
include the full message history when continuing a conversation. The Python
`glm_tpu.optimized.request.from_messages` preparation API accepts user/assistant
history, pinned tokenizer/template and `context_capacity=32768`; the simple `ask`
interface prepares a fresh user question. A prepared batch can queue up to ten
questions sequentially, sharing weights but not conversation state.

Raw answer text contains reasoning and model control markers. Inspect final text
after `</think>` and require normal EOS for a completed answer. Context exhaustion
or failure is incomplete, regardless of correct numbers inside unfinished thinking.
These private files stay outside Git and reviewer archives.

## Stop and failure behavior

Resident idle time has no automatic timeout. `--wall-seconds` bounds initial
startup plus the first request group, and then each subsequent admitted request
group. No process-restart or durable KV recovery is provided. An invalid submitted
input or worker/controller failure can terminate the session through authenticated
cleanup; validate prepared inputs before publication.

To intentionally unload after outstanding work finishes, atomically place an
owner-only file containing `{"stop":true}` at `inbox/stop.json`. The controller
then verifies eight-host cleanup and releases its workload leases. This stops
the model, not just the question queue. Removing future work requires an operator
to preserve admitted requests and avoid racing a live controller read; there is
no supported in-flight cancellation API that keeps the model alive.

Successful residency deliberately records `all_hosts_idle_after=false`. It is
not a cleanup failure: all workers and both workload leases remain owned. Live
worker identity checks and final shutdown cleanup are distinct receipts.

## Four conversations at once

The separate fixed-batch path uses one shared model and four independent 32K
caches, with sequential prefill followed by concurrent decoding:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask --questions /private/questions.json --concurrent --wall-seconds 14400
```

The input is a private JSON array of one to four question strings. This invocation
exits after its group and authenticates eight-host cleanup. `--keep-loaded` is
currently limited to sequential ordinary requests, not concurrent batches. Do
not run the batch command while a resident session owns the fleet.

[Results](STATUS.md) separate solo, resident evaluation and four-chat evidence.
The optional [local chat UI](../UI.md) attaches to an existing resident session.
No online batch admission or full-32K-input quality claim. Retained 8K and
128K-input/166912-total-slot profiles have separate historical scope; the
latter retains its 163840 output ceiling. [Legacy sampling](INFERENCE.md) remains
GLM-5.2 history, with retired weight payloads. Source-bound launch, both workload
leases, sync locks, fresh graph/memory checks and private receipts remain enforced.
