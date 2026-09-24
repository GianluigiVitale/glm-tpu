"""Stateless OpenAI-compatible surface over the existing resident GLM session.

This module owns no model and no device. It renders a complete chat with the
pinned GLM-5.3 template, hands the resulting token ids to the same sequential
resident queue the chat UI uses, and shapes the reply as `/v1/chat/completions`.
Conversation state is never stored: every request carries its own messages.
"""

import json
import time
import uuid

from glm_tpu.engine.request import PROMPT_LIMITS, MAX_NEW
from glm_tpu.entrypoints.openai.chat_utils import check_tools, convert, instruct
from glm_tpu.entrypoints.openai.protocol import ALIASES, ApiError, EFFORTS, MODEL_ID
from glm_tpu.entrypoints.openai.tool_parser import parse


def finish(job, calls):
    if calls:
        return "tool_calls"
    if job["status"] == "complete":
        return "stop"
    return "length" if job.get("stop_cause") in ("context_exhausted", "output_cap") else "stop"


def usage(job):
    return dict(
        prompt_tokens=job["prompt_tokens"],
        completion_tokens=job["output_tokens"],
        total_tokens=job["prompt_tokens"] + job["output_tokens"],
    )


class Api:
    """Stateless request shaping in front of the shared sequential resident queue."""

    def __init__(self, chats, backend, *, capacity, wait_seconds=1800):
        self.chats = chats
        self.backend = backend
        self.capacity = capacity
        self.wait_seconds = wait_seconds

    def models(self):
        # Report the loaded session's real window so a client sizes its own
        # compaction correctly instead of assuming a default.
        # max_input_tokens is what a client must compact against: this profile's
        # prompt ceiling can be lower than its total capacity.
        return dict(
            object="list",
            data=[
                dict(
                    id=name,
                    object="model",
                    owned_by="local",
                    created=0,
                    context_window=self.capacity,
                    max_input_tokens=min(PROMPT_LIMITS[self.capacity], self.capacity - 1),
                    max_output_tokens=min(MAX_NEW, self.capacity - 1),
                    # A request is also bounded by this server deadline; a client should
                    # size its own output expectation against observed throughput.
                    request_deadline_seconds=self.wait_seconds,
                    reasoning_effort=forced or "max",
                    supports=dict(
                        tools=True,
                        streaming=True,
                        reasoning_effort=list(EFFORTS),
                        parallel_requests=False,
                        sampling=False,
                    ),
                )
                for name, forced in ALIASES.items()
            ],
        )

    def prepare(self, data):
        if type(data) is not dict:
            raise ApiError("expected a JSON object")
        name = data.get("model") or MODEL_ID
        if name not in ALIASES:
            raise ApiError("unknown model: " + str(name), status=404, kind="model_not_found")
        if data.get("n") not in (None, 1):
            raise ApiError("only one choice per request is supported")
        effort = data.get("reasoning_effort", "max")
        if effort not in EFFORTS:
            raise ApiError("reasoning_effort must be low, high or max")
        # The alias wins, so a client that cannot send the field still gets it.
        effort = ALIASES[name] or effort
        budget = data.get("max_tokens")
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise ApiError("max_tokens must be a positive integer")
        tools = check_tools(data.get("tools"))
        messages = instruct(convert(data.get("messages")), data.get("tool_choice"), tools)
        key = uuid.uuid4().hex
        payload = self.backend.prepare_api(
            messages, key, tools=tools, effort=effort, max_new_tokens=budget, context_capacity=self.capacity
        )
        return payload, key, name

    def wait(self, key, *, deadline):
        while True:
            job = self.chats.job(key)
            if job is None:
                raise ApiError("request was not retained", status=500, kind="server_error")
            if job["status"] in ("complete", "incomplete"):
                return job
            if job.get("error"):
                raise ApiError(job["error"], status=503, kind="server_error")
            if time.time() > deadline:
                raise ApiError(
                    "the resident model did not finish within the server deadline; use stream=true for long answers",
                    status=504,
                    kind="timeout",
                )
            time.sleep(0.2)

    def completion(self, data):
        payload, key, name = self.prepare(data)
        self.chats.submit(payload, key, label="api")
        job = self.wait(key, deadline=time.time() + self.wait_seconds)
        content, calls = parse(job["answer"])
        message = dict(role="assistant", content=content or None)
        if job["thinking"]:
            message["reasoning_content"] = job["thinking"]
        if calls:
            message["tool_calls"] = [dict(id=c["id"], type="function", function=c["function"]) for c in calls]
        return dict(
            id="chatcmpl-" + key,
            object="chat.completion",
            created=int(job["created"]),
            model=name,
            choices=[dict(index=0, message=message, logprobs=None, finish_reason=finish(job, calls))],
            usage=usage(job),
        )

    def stream(self, data):
        """Yield SSE chunks. Tool calls are emitted once the final channel is known."""
        payload, key, name = self.prepare(data)
        self.chats.submit(payload, key, label="api")
        identity = dict(id="chatcmpl-" + key, object="chat.completion.chunk", model=name)
        created = int(time.time())
        yield self.chunk(dict(identity, created=created), dict(role="assistant", content=""), None)
        deadline = time.time() + self.wait_seconds
        sent_thinking = sent_answer = 0
        while True:
            job = self.chats.job(key)
            if job is None:
                raise ApiError("request was not retained", status=500, kind="server_error")
            if job.get("error"):
                raise ApiError(job["error"], status=503, kind="server_error")
            terminal = job["status"] in ("complete", "incomplete")
            thinking, answer = job["thinking"], job["answer"]
            if len(thinking) > sent_thinking:
                yield self.chunk(
                    dict(identity, created=created), dict(reasoning_content=thinking[sent_thinking:]), None
                )
                sent_thinking = len(thinking)
            # Hold back text until the end when a tool call may still be forming.
            visible = answer if terminal else answer.split("<tool_call>", 1)[0]
            if not terminal and len(visible) > sent_answer:
                yield self.chunk(dict(identity, created=created), dict(content=visible[sent_answer:]), None)
                sent_answer = len(visible)
            if terminal:
                break
            if time.time() > deadline:
                raise ApiError(
                    "the resident model did not finish within the server deadline", status=504, kind="timeout"
                )
            time.sleep(0.2)
        content, calls = parse(job["answer"])
        if len(content) > sent_answer:
            yield self.chunk(dict(identity, created=created), dict(content=content[sent_answer:]), None)
        for call in calls:
            yield self.chunk(
                dict(identity, created=created),
                dict(
                    tool_calls=[
                        dict(
                            index=call["index"],
                            id=call["id"],
                            type="function",
                            function=dict(name=call["function"]["name"], arguments=call["function"]["arguments"]),
                        )
                    ]
                ),
                None,
            )
        yield self.chunk(dict(identity, created=created), {}, finish(job, calls), extra=dict(usage=usage(job)))
        yield b"data: [DONE]\n\n"

    @staticmethod
    def chunk(identity, delta, finish_reason, *, extra=None):
        body = dict(identity, choices=[dict(index=0, delta=delta, logprobs=None, finish_reason=finish_reason)])
        if extra:
            body.update(extra)
        return b"data: " + json.dumps(body, ensure_ascii=False).encode() + b"\n\n"
