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
    """Extract a multiple-choice answer letter (A..). The LATEST explicit
    answer wins: the last \\boxed{X} and the last 'Answer: X' / 'answer is X' /
    'option X' phrase compete BY POSITION (a boxed candidate later revised in
    prose is superseded); falls back to a final standalone-letter line."""
    if not text:
        return None
    body = strip_think(text)
    # The captured LETTER class is UPPERCASE + case-sensitive (answer letters are
    # A..); only the KEYWORDS are case-insensitive (inline (?i:...)). A global
    # re.IGNORECASE would make [A-J] match lowercase 'a'/'i' in prose, so
    # "the answer is a function" would fabricate 'A' — we must never guess.
    pat = f"[{_LETTERS[:n_choices]}]"
    # THE LATEST EXPLICIT ANSWER WINS (round3-ac37c8d finding 1): a model may
    # box a candidate then revise in prose ("try \\boxed{(A)}... the final
    # answer is B") — so the last \boxed{} and the last keyword phrase COMPETE
    # BY POSITION instead of boxed always pre-empting.
    # \boxed{}: brace-balanced via extract_boxed so \boxed{\text{C}} /
    # \boxed{(B)} / \boxed{D.} all resolve; participates only if its content
    # is a single valid letter (never guess).
    boxed_letter, boxed_pos = None, -1
    boxed = extract_boxed(body)
    if boxed:
        b = re.sub(r"\\text(?:bf|rm|it)?\s*\{([^{}]*)\}", r"\1", boxed)
        m = re.fullmatch(rf"[\s(\[*_]*({pat})[\s)\]*_.:]*", b)
        if m:
            boxed_letter, boxed_pos = m.group(1), body.rfind(r"\boxed{")
    # keyword patterns: [\s(*_]* admits '(' and markdown emphasis ('**B**');
    # the letter itself stays UPPERCASE-only (see above). LAST match wins.
    kw_letter, kw_pos = None, -1
    for rx in (rf"(?i:final\s+answer|answer)\s*(?i:is|:)?\s*[\s(*_]*({pat})\b",
               rf"(?i:option)\s*(?i:is|:)?\s*[\s(*_]*({pat})\b"):
        for m in re.finditer(rx, body, flags=re.DOTALL):   # NOT IGNORECASE
            if m.start() > kw_pos:
                kw_letter, kw_pos = m.group(1), m.start()
    if boxed_letter is not None and boxed_pos > kw_pos:
        return boxed_letter          # boxed is the latest explicit answer
    if kw_letter is not None:
        return kw_letter             # explicit prose AFTER the boxed wins
    # Conservative fallback: accept a bare letter ONLY if the last non-empty line
    # IS just that (uppercase) letter, optionally wrapped/punctuated. Never guess an
    # answer from a letter buried in prose. No clear answer -> None -> scored wrong,
    # with the raw output stored for audit.
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if lines:
        m = re.fullmatch(rf"\(?\s*({pat})\s*\)?[.):]?", lines[-1])   # NOT IGNORECASE
        if m:
            return m.group(1)
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


# ---- card-protocol extraction (docs/07-card-protocol-fidelity.md) -----------
# The GLM-5.2 card evaluates AIME/HMMT/IMOAnswerBench with a system prompt that
# mandates the reply format
#     Explanation: {...}\nExact Answer: {...}\nConfidence: {...}
# and grades with a GPT-5.5 judge. Our judge SUBSTITUTE is: extract the
# 'Exact Answer:' field, then exact-match (score_math) — labeled as such in
# provenance; extract_exact_answer below is that extractor.

# (?<![A-Za-z0-9]) not \b: '__Exact Answer__:' must match (underscore is a
# word char, so \b would reject it) while 'inexact answer:' must not.
_EXACT_ANSWER_MARK = re.compile(
    r"(?i)(?<![A-Za-z0-9])exact\s+answer\s*(?:\*\*|__)?\s*[:：]")
_CONFIDENCE_MARK = re.compile(r"(?i)[\s,;.]*(?:\*\*|__)?\s*confidence\s*(?:\*\*|__)?\s*[:：]")
_CONFIDENCE_LINE = re.compile(r"(?im)^\s*(?:\*\*|__)?\s*confidence\s*(?:\*\*|__)?\s*[:：][^\n]*$")


def extract_exact_answer(text: str) -> str | None:
    """Extract the 'Exact Answer:' field mandated by the GLM-5.2 card's
    Explanation/Exact Answer/Confidence system prompt (the AIME/HMMT/
    IMOAnswerBench card protocol).

    The LAST 'Exact Answer:' marker wins (a reasoning model may draft the
    format, then restate it); the field is cut at the next 'Confidence:'
    marker (same line or later) and reduced to its first non-empty line;
    a \\boxed{...} inside the field is unwrapped and markdown emphasis /
    stray whitespace stripped. The literal colon is REQUIRED — prose like
    'the exact answer might be 7' never matches (never guess; the card-mode
    fallback chain handles nonconforming replies). Returns None when the
    field is absent or empty (scored wrong, raw output stored for audit)."""
    if not text:
        return None
    body = strip_think(text)
    marks = list(_EXACT_ANSWER_MARK.finditer(body))
    if not marks:
        return None
    tail = body[marks[-1].end():]
    cut = _CONFIDENCE_MARK.search(tail)
    if cut:
        tail = tail[:cut.start()]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return None
    ans = lines[0]
    if r"\boxed{" in ans:
        ans = extract_boxed(ans) or ans
    # strip markdown emphasis / backticks / stray whitespace at both ends
    # ('**42**', '** 42', '`42`'); interior characters are never touched.
    ans = re.sub(r"^[\s*_`]+", "", re.sub(r"[\s*_`]+$", "", ans))
    return ans or None


def drop_confidence_lines(text: str) -> str:
    """Remove whole 'Confidence: ...' lines (the card format's third field)
    so card-mode FALLBACK extraction (last \\boxed{} / final number on a reply
    that ignored the format) can never grab the confidence percentage as the
    answer ('Confidence: 95%' -> extract_final_number would return '95')."""
    if not text:
        return ""
    return _CONFIDENCE_LINE.sub("", text)


def norm_math(s: str | None) -> str | None:
    """Light normalization for math-answer string comparison."""
    if s is None:
        return None
    s = s.strip().replace(" ", "")
    s = re.sub(r"\\?\$", "", s)                       # $ / \$ (money, math-mode)
    s = re.sub(r"\\?%$", "", s)                        # trailing % / \% ("60\%" == "60")
    # "160 minutes" / "12dollars" -> "160" / "12": drop a PURELY alphabetic
    # unit tail after a number (never digits/operators in the tail, so
    # composites like "2 of 3" or "3x+1" are untouched).
    m = re.fullmatch(
        r"(-?\d[\d,]*(?:\.\d+)?)(?:\s+[a-zA-Z\\ ]+|[a-zA-Z]{2,}[a-zA-Z\\ ]*)", s)
    if m:  # single letters without a space ("3x") stay algebraic
        s = m.group(1)
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = s.rstrip(".")
    # Thousands-comma stripping ONLY when the ENTIRE value is a plain grouped
    # number (optional sign/decimals): "1,234,567" -> "1234567". NEVER inside
    # composite tokens — "(1,200)" is a tuple/interval and "0,001" a European
    # decimal; both are kept verbatim (round3-ac37c8d finding 2: the old
    # in-place regex rewrote "(1,200)" -> "(1200)" and "0,001" -> "1").
    if re.fullmatch(r"-?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
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
