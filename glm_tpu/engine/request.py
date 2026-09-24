"""Private greedy requests for the fixed ordinary release profile."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import struct

from glm_tpu.utils import json_utils
from glm_tpu.utils import io_utils
from glm_tpu.config import model


SCHEMA = "glm_ws32_optimized_request_v2"
CAPACITY = 8192
LONG_CAPACITY = 166912
CONCURRENT_CAPACITY = 32768
AGENT_CAPACITY = 262144
CONCURRENT_LIMIT = 4
MAX_PROMPT = 131072
CAPACITIES = (CAPACITY, LONG_CAPACITY, CONCURRENT_CAPACITY, AGENT_CAPACITY)
# Per-profile prompt ceiling. The 128K-input profile keeps its validated limit;
# the others are bound by their own capacity once an output budget is reserved.
PROMPT_LIMITS = {
    CAPACITY: CAPACITY,
    CONCURRENT_CAPACITY: CONCURRENT_CAPACITY,
    LONG_CAPACITY: MAX_PROMPT,
    AGENT_CAPACITY: AGENT_CAPACITY,
}
BATCH_SCHEMA = "glm_ws32_optimized_batch_v1"
CONCURRENT_SCHEMA = "glm_ws32_concurrent_batch_v1"

MAX_CAPACITY = 1048576
MAX_NEW = 163840
VOCAB = 154880
EOS = (154820, 154827, 154829)
PAYLOAD_CAP = 16 << 20
MESSAGES_CAP = 1 << 20


def from_token_ids(ids, *, request_id, max_new_tokens, context_capacity=CAPACITY):
    # Reuse strict token/type/identity validation, then bind the narrower profile.
    if type(context_capacity) is not int or context_capacity not in CAPACITIES:
        raise ValueError("unsupported ordinary context capacity")
    original = _validate_prompt_ids(
        ids, request_id=request_id, max_new_tokens=max_new_tokens, capacity=context_capacity
    )
    if len(ids) > PROMPT_LIMITS[context_capacity] or len(ids) + max_new_tokens > context_capacity:
        raise ValueError("prompt and complete output budget exceed the selected capacity")
    body = dict(
        schema=SCHEMA,
        request_id=request_id,
        prompt_ids=original["prompt_ids"],
        prompt_ids_sha256=original["prompt_ids_sha256"],
        max_new_tokens=max_new_tokens,
        context_capacity=context_capacity,
        vocab_size=VOCAB,
        eos_ids=list(EOS),
        decode_policy="greedy",
        thinking="on/max",
        tokenizer_files=dict(model.TOKENIZER_FILES),
        chat_template_sha256=model.TEMPLATE_SHA,
        model_id=model.MODEL_ID,
        model_revision=model.REVISION,
    )
    return dict(body, request_sha256=sha256(json_utils.canonical(body)).hexdigest())


def validate(value):
    if type(value) is not dict:
        raise ValueError("optimized request must be an object")
    try:
        expected = from_token_ids(
            value["prompt_ids"],
            request_id=value["request_id"],
            max_new_tokens=value["max_new_tokens"],
            context_capacity=value["context_capacity"],
        )
    except KeyError as exc:
        raise ValueError("missing optimized request fields") from exc
    if json_utils.canonical(value) != json_utils.canonical(expected):
        raise ValueError("optimized request identity or fixed profile differs")


def batch(values, *, concurrent=False):
    """Bind either a sequential queue or an explicitly concurrent 32K batch."""
    if type(concurrent) is not bool:
        raise ValueError("concurrent must be boolean")
    limit = CONCURRENT_LIMIT if concurrent else 10
    if type(values) is not list or not 1 <= len(values) <= limit:
        raise ValueError(f"submit between one and {limit} requests")
    for value in values:
        validate(value)
    if len({v["request_id"] for v in values}) != len(values):
        raise ValueError("request IDs must be unique")
    capacities = {v["context_capacity"] for v in values}
    if len(capacities) != 1:
        raise ValueError("one loaded model requires one context capacity")
    capacity = capacities.pop()
    if concurrent and capacity != CONCURRENT_CAPACITY:
        raise ValueError("concurrent conversations require 32K total slots each")
    body = dict(schema=CONCURRENT_SCHEMA if concurrent else BATCH_SCHEMA, requests=values, context_capacity=capacity)
    return dict(body, request_sha256=sha256(json_utils.canonical(body)).hexdigest())


def validate_payload(value):
    if type(value) is dict and value.get("schema") in (BATCH_SCHEMA, CONCURRENT_SCHEMA):
        if json_utils.canonical(value) != json_utils.canonical(
            batch(value.get("requests"), concurrent=value["schema"] == CONCURRENT_SCHEMA)
        ):
            raise ValueError("batch identity differs")
    else:
        validate(value)


def requests(value):
    validate_payload(value)
    return value["requests"] if value.get("schema") in (BATCH_SCHEMA, CONCURRENT_SCHEMA) else [value]


def stop_cause(value, finish_reason):
    """Distinguish normal EOS, context exhaustion and an explicit shorter cap."""
    if finish_reason == "eos":
        return "eos"
    if finish_reason == "length":
        return (
            "context_exhausted"
            if len(value["prompt_ids"]) + value["max_new_tokens"] == value["context_capacity"]
            else "output_cap"
        )
    raise ValueError("request did not reach a terminal token condition")


def read(path, *, expected_sha256=None):
    raw = io_utils.read_bounded(Path(path), PAYLOAD_CAP)
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("optimized request file digest differs")
    value = json.loads(raw)
    validate_payload(value)
    return value


def prepare_file(*, messages_path, output, repo, tokenizer_root, request_id, max_new_tokens, context_capacity=CAPACITY):
    messages = json.loads(io_utils.read_bounded(messages_path, MESSAGES_CAP))
    template = model.verified_template(repo, tokenizer_root)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, trust_remote_code=False)
    value = from_messages(
        messages,
        tokenizer=tokenizer,
        chat_template=template,
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        context_capacity=context_capacity,
    )
    output = io_utils._plain(output)
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError("private requests must be outside the source checkout")
    raw = json_utils.canonical(value) + b"\n"
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(
        request_sha256=value["request_sha256"],
        file_sha256=sha256(raw).hexdigest(),
        prompt_tokens=len(value["prompt_ids"]),
        max_new_tokens=value["max_new_tokens"],
        context_capacity=context_capacity,
        decode_policy="greedy",
        model_executions=0,
    )


def from_messages(messages, *, tokenizer, chat_template, request_id, max_new_tokens=None, context_capacity=CAPACITY):
    """Tokenize the complete chat at maximum thinking effort, without truncation."""
    if (
        type(messages) is not list
        or not messages
        or len(json_utils.canonical(messages)) > MESSAGES_CAP
        or any(
            type(m) is not dict
            or set(m) != {"role", "content"}
            or m["role"] not in ("system", "user", "assistant")
            or type(m["content"]) is not str
            for m in messages
        )
        or messages[-1]["role"] != "user"
        or any(m["role"] == "system" for m in messages[1:])
    ):
        raise ValueError("expected bounded text-only chat ending with a user message")
    if sha256(chat_template.encode()).hexdigest() != model.TEMPLATE_SHA:
        raise ValueError("chat template differs from pinned GLM-5.3")
    if type(context_capacity) is not int or context_capacity not in CAPACITIES:
        raise ValueError("unsupported ordinary context capacity")
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        chat_template=chat_template,
        enable_thinking=True,
        reasoning_effort="max",
    )
    if max_new_tokens is None:
        max_new_tokens = min(MAX_NEW, context_capacity - len(ids))
    return from_token_ids(ids, request_id=request_id, max_new_tokens=max_new_tokens, context_capacity=context_capacity)


def _parameters(request_id: str, max_new_tokens: int) -> None:
    if not isinstance(request_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", request_id) is None:
        raise ValueError("request id must be 1..128 simple ASCII characters")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW:
        raise ValueError("generation cap must be 1..163840; no implicit truncation")


def _validate_prompt_ids(
    ids: list[int],
    *,
    request_id: str,
    max_new_tokens: int,
    capacity: int,
) -> dict:
    """The validated prompt: ``prompt_ids`` and ``prompt_ids_sha256`` (little-endian int32)."""
    _parameters(request_id, max_new_tokens)
    if type(capacity) is not int or not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError("context capacity must be 1..1048576")
    if (
        type(ids) is not list
        or not ids
        or len(ids) + max_new_tokens > capacity
        or any(type(x) is not int or not 0 <= x < VOCAB for x in ids)
    ):
        raise ValueError("prompt IDs must fit the full requested generation budget")
    return dict(
        prompt_ids=list(ids),
        prompt_ids_sha256=sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest(),
    )
