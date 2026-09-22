"""Bounded user prompts for the frozen native sampled runtime; no model imports.

This is not a benchmark protocol or launch authorization. Prompt IDs are private
data: write them outside Git. Sampling/capacity stay at the admitted graph values.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any

SCHEMA = "glm_ws32_user_request_v1"
CAPACITY = 166912
MAX_CAPACITY = 1048576
MAX_NEW = 163840
VOCAB = 154880
EOS = (154820, 154827, 154829)
PAYLOAD_CAP = 16 << 20
MESSAGES_CAP = 1 << 20
TEMPLATE_SHA = "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679"
TOKENIZER_FILES = {
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config.json": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _plain(path: Path) -> Path:
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("request paths must not traverse symlinks")
    return path


def read_bounded(path: Path, cap: int) -> bytes:
    path = _plain(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        facts = os.fstat(stream.fileno())
        if not stat.S_ISREG(facts.st_mode) or not 0 < facts.st_size <= cap:
            raise ValueError("request input is not a bounded regular file")
        raw = stream.read(cap + 1)
    if not 0 < len(raw) <= cap:
        raise ValueError("request input changed beyond its byte cap")
    return raw


def _parameters(request_id: str, seed: int, max_new_tokens: int) -> None:
    if (
        not isinstance(request_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", request_id) is None
    ):
        raise ValueError("request id must be 1..128 simple ASCII characters")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("request seed must be a uint64 integer")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW:
        raise ValueError("generation cap must be 1..163840; no implicit truncation")


def from_token_ids(
    ids: list[int],
    *,
    request_id: str,
    seed: int,
    max_new_tokens: int,
    capacity: int = CAPACITY,
) -> dict:
    # The legacy profile keeps its own 166912 ceiling; a caller binding a wider
    # ordinary profile states that capacity explicitly.
    _parameters(request_id, seed, max_new_tokens)
    if type(capacity) is not int or not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError("context capacity must be 1..1048576")
    if (
        type(ids) is not list
        or not ids
        or len(ids) + max_new_tokens > capacity
        or any(type(x) is not int or not 0 <= x < VOCAB for x in ids)
    ):
        raise ValueError("prompt IDs must fit the full requested generation budget")
    body = dict(
        schema=SCHEMA,
        request_id=request_id,
        seed=seed,
        prompt_ids=list(ids),
        prompt_ids_sha256=sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest(),
        max_new_tokens=max_new_tokens,
        context_capacity=capacity,
        vocab_size=VOCAB,
        eos_ids=list(EOS),
        temperature=1.0,
        top_p=0.95,
        tokenizer_files=dict(TOKENIZER_FILES),
        chat_template_sha256=TEMPLATE_SHA,
        thinking="on/max",
        benchmark=False,
    )
    return dict(body, request_sha256=sha256(canonical(body)).hexdigest())


def validate(value: dict) -> None:
    if type(value) is not dict:
        raise ValueError("user request must be an object")
    try:
        expected = from_token_ids(
            value["prompt_ids"],
            request_id=value["request_id"],
            seed=value["seed"],
            max_new_tokens=value["max_new_tokens"],
        )
    except KeyError as exc:
        raise ValueError("user request is missing required fields") from exc
    # Compare canonical bytes so bool/int and float/int substitutions cannot pass.
    if canonical(value) != canonical(expected) or len(canonical(value)) > PAYLOAD_CAP:
        raise ValueError("user request fields, frozen profile or digest differ")


def from_messages(
    messages: list[dict],
    *,
    tokenizer: Any,
    chat_template: str,
    request_id: str,
    seed: int,
    max_new_tokens: int,
) -> dict:
    _parameters(request_id, seed, max_new_tokens)
    if (
        type(messages) is not list
        or not messages
        or len(canonical(messages)) > MESSAGES_CAP
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
    if sha256(chat_template.encode()).hexdigest() != TEMPLATE_SHA:
        raise ValueError("chat template differs from the frozen model template")
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        chat_template=chat_template,
        enable_thinking=True,
        reasoning_effort="max",
    )
    return from_token_ids(
        ids, request_id=request_id, seed=seed, max_new_tokens=max_new_tokens
    )


def write_private(path: Path, value: dict, *, repo: Path) -> dict:
    """Exclusive owner-only output outside the source checkout; never overwrite."""
    validate(value)
    path = _plain(path)
    if path.resolve().is_relative_to(repo.resolve()):
        raise ValueError("private prompts must be written outside the repository")
    raw = canonical(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return dict(
        request_sha256=value["request_sha256"],
        file_sha256=sha256(raw).hexdigest(),
        bytes=len(raw),
        prompt_tokens=len(value["prompt_ids"]),
        max_new_tokens=value["max_new_tokens"],
        model_executions=0,
    )


def prepare_file(
    *,
    messages_path: Path,
    output: Path,
    repo: Path,
    tokenizer_root: Path,
    request_id: str,
    seed: int,
    max_new_tokens: int,
) -> dict:
    """Local tokenizer only; no cloud access, downloads, device or model loading."""
    _parameters(request_id, seed, max_new_tokens)
    messages = json.loads(read_bounded(messages_path, MESSAGES_CAP))
    template = read_bounded(
        repo / "reference/hf-repo/chat_template.jinja", 64 << 10
    ).decode()
    for name, digest in TOKENIZER_FILES.items():
        if sha256(read_bounded(tokenizer_root / name, 32 << 20)).hexdigest() != digest:
            raise ValueError("local tokenizer bytes differ from the frozen profile")
    # Import only after input and identity checks; never fetch custom model code.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True, trust_remote_code=False
    )
    value = from_messages(
        messages,
        tokenizer=tokenizer,
        chat_template=template,
        request_id=request_id,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    return write_private(output, value, repo=repo)
