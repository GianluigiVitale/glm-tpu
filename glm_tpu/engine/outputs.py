"""The token stream: one ``TokenEvent`` per generated token (its fields are a wire format)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    index: int
    token_id: int
    finish_reason: str | None
