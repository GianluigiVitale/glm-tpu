# GLM-5.3 chat workspace

A private, loopback web interface with saved conversations, streamed thinking
and answers, copy, rename/delete, light/dark themes and a mobile layout. It
attaches to an **already running 32K resident session**. It never loads weights,
starts TPU workers or unloads the model when the browser or UI server closes.

## Open the workspace

Run on the resident controller's host, from the published checkout and documented
Python environment. Use the original controller dispatch receipt (PID, start
ticks, command and code hash), not a completion-notifier receipt:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu.entrypoints.serve.server \
  --run /absolute/path/to/resident-run \
  --dispatch /absolute/path/to/controller-dispatch.json \
  --state /absolute/path/outside-the-repository/private-chats \
  --port 8011
```

Forward port **8011** through VS Code's Ports panel or an SSH tunnel, using the
same local port, then open **http://127.0.0.1:8011**. The server listens only on
loopback. This is a single-owner workspace, without accounts or public hosting;
other local users who can reach that port are inside its trust boundary. Keep
the tunnel private. Host/origin checks intentionally reject public proxy hosts.
The UI uses only bundled assets, with no CDN, telemetry or external fonts.

The server needs the pinned local tokenizer and the checkout's model/template
assets, but does not load model weights or initialize JAX devices. Runtime
dependencies and site requirements remain in [installation](release/INSTALLATION.md).
`--repo /path/to/checkout` locates those assets when using an installed wheel.

## Conversation behavior

Each turn includes that conversation's previous user messages and completed
final answers. Thinking is displayed separately and is not replayed as an
assistant answer. Other conversations' histories never enter the request.
The complete template/history/input is tokenized without truncation; thinking
and answer receive **all remaining slots of the 32,768-slot cache**. There is
no separate short-output default or thinking budget. Oversized input is rejected
before submission. Context exhaustion is visibly incomplete.

The current resident executable serves **one answer at a time**. Up to ten
pending turns across separate conversations can wait in the UI queue. This is
distinct from the separate four-chat batched invocation. A new turn in the same
conversation waits for a completed answer. There is no in-flight cancel control:
the loaded controller cannot cancel generation while retaining its workers.

The UI holds the resident producer lock, which is why the stateless
[`/v1` API](API.md) is served by this same process. Do not run another inbox
producer at the same time. Request identity and sequence are saved before atomic publication;
restarting this UI with the same state directory reconciles the same admitted
request rather than duplicating it. This does not recover a failed model process
or its KV state. On a backend error the queue pauses and displays the reason.
Closing the UI does not cancel already accepted work in the resident controller.

Histories, prepared requests and UI errors are private files (0700 directory,
0600 files) outside Git. Deleting a conversation removes it from the workspace;
original resident inference receipts remain as private execution evidence.
Browser storage holds only the chosen theme and active conversation ID.

The browser receives token progress as the server reads rank0's existing stream.
An answer becomes complete only after normal EOS and all eight completion
receipts agree on request, tokens, graphs and memory admission. Displayed decode
speed is the slowest host's decode rate, including thinking, excluding prefill,
startup and queue wait. This UI does not change the model's measured performance.

## Design provenance

The palette, sidebar, message layout, composer and responsive styling are adapted
from the owner's private [as-pt](https://github.com/GianluigiVitale/as-pt) project,
`aspt_rag/web/index.html`, commit
`4ed7937910538eef2754b31d0af316c6888ad9ba`. Original file SHA256:
`3b93ef59be703bf7caba4bdee9b74d3c0f10e00d3dbb944867110560feb925d3`.
The GLM resident bridge and conversation frontend are implemented here; the
reference archive's retrieval/citation logic and private data are not included.

Read-only inspection confirmed that the exact same HTML is backed up at
`repos/aspt/aspt_rag/web/index.html` in both reference buckets:

| Bucket | Exact generation | Region |
|---|---:|---|
| `driftbench-storage` | `1783351597625998` | EUROPE-WEST4 |
| `driftbench-v5e-euw4` | `1788025839615601` | EUROPE-WEST4 |

Neither reference bucket was changed. New GLM source backups remain in the
approved US-CENTRAL2 bucket. Reuse is owner-authorized inside this private
project; this notice does not assign a new license to the reference UI.

## Offline checks

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/entrypoints/serve/test_http_handler.py tests/entrypoints/ui/test_conversations.py
node --check glm_tpu/entrypoints/ui/static/app.js
```

The CPU checks use a synthetic resident adapter to verify queue ordering, history
isolation, duplicate suppression, interrupted-publication recovery, capacity
rejection and HTTP origin boundaries. They require no weights, TPU or cloud.
Real-model and browser validation receipts remain outside Git; only their
sanitized summary belongs in the release documentation.

On September22, two real browser turns completed at normal EOS: an arithmetic
answer and a follow-up recalling both its result and a supplied project name.
They used the existing loaded model (resident sequences771–772), with37/127
input tokens,279/90 output tokens and13.51/13.46 decode tokens/s. Both eight-host
receipts passed. Browser checks covered desktop, mobile without horizontal
overflow, dark mode and restored conversation history after reload. These are
interface smoke checks, not additional GSM8K benchmark results. Raw prompts,
answers and screenshots remain private outside the source archive.
