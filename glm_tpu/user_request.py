"""Bounded request inputs and the canonical request bytes; no model imports.

``from_token_ids`` validates a prompt and its full generation budget for the ordinary request
profiles (``glm_tpu.optimized.request``); ``canonical`` is the canonical JSON wire encoding of
request bodies and resident commands; ``read_bounded`` reads a bounded regular file without
following symlinks. The GLM-5.2 sampled request profile this module served until S2f
(``prepare-request --profile legacy-sampled``) is archived at ``archive/research-20260922``.
Prompt IDs are private data: write them outside Git.
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

MAX_CAPACITY = 1048576
MAX_NEW = 163840
VOCAB = 154880
EOS = (154820, 154827, 154829)
PAYLOAD_CAP = 16 << 20
MESSAGES_CAP = 1 << 20


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


def _parameters(request_id: str, max_new_tokens: int) -> None:
    if (
        not isinstance(request_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", request_id) is None
    ):
        raise ValueError("request id must be 1..128 simple ASCII characters")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW:
        raise ValueError("generation cap must be 1..163840; no implicit truncation")


def from_token_ids(
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
