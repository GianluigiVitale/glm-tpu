#!/usr/bin/env python3
"""Answer extractors + scorers for GLM-5.2 benchmark reproduction.

Pure, offline-testable (no model, no TPU). GLM-5.2 is a reasoning model that emits
a long chain-of-thought then a final answer, so scoring is generation-based:
extract the final answer from the model's verbatim reply, then score vs gold.

The model reply may contain a <think>...</think> block; strip it before extraction
(the answer lives after it). Extractors are deliberately forgiving (multiple answer
conventions) but never guess — return None when no answer is found (scored wrong,
and the raw output is stored for audit).
"""
from __future__ import annotations

import re

_LETTERS = "ABCDEFGHIJ"


def strip_think(text: str) -> str:
    """Drop a leading/enclosed <think>...</think> reasoning block."""
    if not text:
        return ""
    # remove complete <think>..</think> blocks; if an unclosed <think> remains,
    # keep only what follows the last </think> (or the whole thing if none).
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


def extract_mc_letter(text: str, n_choices: int = 4) -> str | None:
    """Extract a multiple-choice answer letter (A..). Tries, in order:
    \\boxed{X}, 'Answer: X', 'answer is X', 'option X', then a final standalone letter."""
    if not text:
        return None
    body = strip_think(text)
    valid = _LETTERS[:n_choices]
    pat = f"[{valid}]"
    for rx in (rf"\\boxed\{{\s*({pat})\s*\}}",
               rf"(?:final\s+answer|answer)\s*(?:is|:)?\s*\(?\s*({pat})\b",
               rf"\boption\s*\(?\s*({pat})\b",
               rf"\b({pat})\b(?!.*\b[{valid}]\b)"):   # last standalone valid letter
        m = re.search(rx, body, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).upper()
    return None


def extract_boxed(text: str) -> str | None:
    r"""Extract the content of the last \boxed{...}, brace-balanced."""
    if not text:
        return None
    body = strip_think(text)
    idx = body.rfind(r"\boxed{")
    if idx < 0:
        return None
    i = idx + len(r"\boxed{")
    depth, out = 1, []
    while i < len(body) and depth:
        c = body[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() or None


def extract_final_number(text: str) -> str | None:
    """Last number in the reply (GSM8K-style). Handles $, commas, #### markers."""
    if not text:
        return None
    body = strip_think(text)
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", body)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", body)
    return nums[-1].replace(",", "") if nums else None


def norm_math(s: str | None) -> str | None:
    """Light normalization for math-answer string comparison."""
    if s is None:
        return None
    s = s.strip().strip("$").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = s.rstrip(".")
    # 003 -> 3 ; 3.0 -> 3
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return s


# ---- scorers: (extracted, gold) -> bool -------------------------------------

def score_mc(extracted: str | None, gold: str) -> bool:
    return extracted is not None and extracted.upper() == str(gold).upper()


def score_math(extracted: str | None, gold: str) -> bool:
    a, b = norm_math(extracted), norm_math(gold)
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return False
