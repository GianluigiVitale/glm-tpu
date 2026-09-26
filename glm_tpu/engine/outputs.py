"""The generated output: the token stream, one ``TokenEvent`` per generated token (its fields are a wire
format), and ``final_channel``, the split of the generated text into reasoning and final answer."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    index: int
    token_id: int
    finish_reason: str | None


def final_channel(text):
    if "</think>" not in text:
        return text.removeprefix("<think>").strip(), ""
    thinking, answer = text.split("</think>", 1)
    answer = re.split(r"<\|(?:user|endoftext|observation)\|>", answer, maxsplit=1)[0]
    return thinking.removeprefix("<think>").strip(), answer.strip()
