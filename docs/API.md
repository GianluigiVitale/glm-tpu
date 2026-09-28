# GLM-5.3 local OpenAI-compatible API

A stateless `/v1` surface for local development tools, served by the **same
process** as the [chat workspace](UI.md). It renders each request with the pinned
GLM-5.3 template, including tool definitions, and submits the resulting tokens to
the **existing resident session**. It never loads weights, starts TPU workers or
unloads the model.

One process is required: that server holds the resident producer lock, and a
second producer would race the inbox sequence. The browser workspace and `/v1`
therefore share one sequential queue.

## Start and connect

The API is on by default when the workspace starts; `--no-api` serves only the
browser. A key is created at `<state>/api-key` (0600) on first start and printed
by path, never by value (`--api-key-file` names another location). Run the server
on the resident controller's host, from the published checkout, after the session
answered its first request ([how it finds the controller](UI.md#open-the-workspace)):

```bash
JAX_PLATFORMS=cpu python -m glm_tpu.entrypoints.serve.server \
  --run /absolute/path/to/resident-run \
  --state /absolute/path/outside-the-repository/private-chats \
  --port 8011
```

Forward port 8011 to the same local port, then point a client at
`http://127.0.0.1:8011/v1` with `Authorization: Bearer <key>`. The server listens
on loopback only and still applies the Host check, so the forwarded local port
must be 8011. Requests to `/v1` do not use the browser's `X-GLM-UI` gate, and the
API key does not open `/api/chat`.

```bash
export GLM_API_KEY="$(ssh <controller-host> cat /absolute/path/outside-the-repository/private-chats/api-key)"
curl -s http://127.0.0.1:8011/v1/models -H "Authorization: Bearer $GLM_API_KEY"
```

## Surface

`GET /v1/models` and `POST /v1/chat/completions`. The request is **stateless**:
send the whole `messages` array every time, including `system`. Nothing is
stored, no conversation is created, and API traffic never appears in the browser
workspace.

Supported fields: `messages` (`system`, `user`, `assistant`, `tool`; string or
text-part content), `tools`, `tool_choice`, `max_tokens`, `stream`,
`reasoning_effort` (`low`, `high`, `max`; default `max`). Responses carry
`usage`, and `finish_reason` is `stop`, `tool_calls` or `length`.

Three model ids select reasoning effort for clients that route by model rather
than by field: `glm-5.3`, `glm-5.3-low` and `glm-5.3-high`. The alias overrides
`reasoning_effort`, so a client can point its cheap side calls — titles,
summaries, compaction — at `glm-5.3-low` and keep them off the full-effort path.
All three address the same loaded model and the same sequential queue.
`request_deadline_seconds` in the listing bounds one request; size an expected
output against it and observed throughput rather than `max_output_tokens`, which
is only the profile's hard ceiling.

Thinking is returned in `reasoning_content` — on the message when buffered, on
`delta.reasoning_content` when streaming. It is **never** placed in `content`,
because a stray `<think>` block breaks tool-call parsing in agent loops.

### Tool calling

Tool definitions are rendered by the pinned template's own `<tools>` block, and
the model answers in GLM's `<tool_call>` format. The server translates in both
directions: incoming `function.arguments` arrive as a JSON **string** and are
decoded into the mapping the template iterates; outgoing calls are re-serialized
as a JSON string with `finish_reason: "tool_calls"`. Send results back as
`role: "tool"` messages carrying `tool_call_id`.

`tool_choice` has no template equivalent, so `none`, `required` and a named
function are expressed as an in-band system instruction rather than a decoding
constraint. Treat it as a strong request, not a guarantee.

### Streaming

`stream: true` returns `text/event-stream` chunks terminated by `data: [DONE]`.
Text is withheld once a `<tool_call>` opens, so partial markup never reaches
`content`; tool calls are emitted as complete fragments with a stable `index`.
The final chunk carries `finish_reason` and `usage`. Chunks are coalesced from
the resident token stream at roughly four per second, not one per token.

## Limits that will shape an agent loop

The resident session decodes **one request at a time** at about 13.5 tokens/s.
Concurrent client requests queue rather than run together; up to ten may wait.
There is no way to make this parallel without a different session.

- **The window is the loaded session's capacity**, shared by system, tools,
  history, thinking and answer. Ordinary profiles are 8,192 / 32,768 / 166,912
  (128K prompt / 163K total) / 262,144 total slots, chosen with
  `--context 8k|32k|128k|256k` when the session
  starts and compiled in at load. `GET /v1/models` is served by whichever session
  is running; the server rejects a request that no longer leaves room to generate.
- **Generation is capped at 163,840 tokens** regardless of capacity, so a 256K
  window reserves the remainder for input and history.
- **Greedy decoding only.** `temperature`, `top_p` and `seed` are ignored; the
  resident profile is pinned and cannot be changed without redeploying the model.
- **No cancellation.** An aborted HTTP request does not stop an admitted
  generation. Closing the connection leaves it running to completion.
- `n > 1`, logprobs, images, audio and embeddings are not supported.
- Use `reasoning_effort: "low"` for cheap side calls. At `max`, a routine request
  can think for minutes before its first visible token.
- A backend failure pauses the shared queue and is reported to waiting clients;
  it is never retried under a fresh sequence.

## Client configuration

For an OpenAI-compatible client, set the base URL to `http://127.0.0.1:8011/v1`,
the key from `GLM_API_KEY` and the model id `glm-5.3`. Declare tool support where
the client asks for it, and prefer streaming: a buffered request must wait for
the whole answer, and the server's own deadline is 30 minutes.

Tell the client the real window so its own compaction triggers at the right
point — for a 256K session that is a context limit of 262,144 and an output
limit of 163,840. Because prefill cost follows the actual prompt rather than the
allocated window, a large window with client-side compaction is cheaper than
repeatedly refilling a small one.

## Offline checks

```bash
JAX_PLATFORMS=cpu python -m pytest -q -p no:cacheprovider tests/entrypoints/openai/test_serving_chat.py tests/entrypoints/openai/test_serving_models.py tests/entrypoints/serve/test_http_handler.py tests/entrypoints/serve/test_server.py
```

These cover message conversion, tool-call render and parse, `tool_choice`
handling, statelessness, capacity rejection, streaming order, key boundaries,
backend-failure reporting and the server's command line with a synthetic
resident. They need no weights, TPU or cloud access. The server's code is
`glm_tpu/entrypoints/serve/` and `glm_tpu/entrypoints/openai/`.

On 2026-09-22 four bounded requests ran against the loaded model: a plain answer,
a real `get_weather` tool call, the tool result, and a streamed answer —
36.7 s of model time in total, all at normal EOS with the model retained. Raw
receipts stay private outside the source archive.
