#!/usr/bin/env python3
"""Benchmark registry for GLM-5.2 reproduction.

Each benchmark is a BenchSpec: how to load its items from HuggingFace, how to
build the prompt, how to extract the answer from the model reply, how to score.
Loaders are deterministic (fixed seed for any shuffling) so the stored prompt is
reproducible and auditable.

Scope now = the TRACTABLE benchmarks that are readily available on HF and give a
Stage-1 correctness signal (GPQA-Diamond, MMLU-Pro, GSM8K, and AIME best-effort).
The agentic/coding card benchmarks (SWE-bench Pro, Terminal Bench, ...) need full
agentic harnesses and are deferred (see PLAN.md); their card targets live in
provenance.CARD_TARGETS.

Datasets are cached under bench/data/ (HF datasets cache; gitignored). Needs
`datasets` + an HF token (GPQA is gated) — export HF_TOKEN from ~/glm-tpu/.env.
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Callable

import extract as ex

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(DATA_DIR, "hf_cache"))

_THINK_HINT = ("Reason step by step, then give your final answer. "
               "Put ONLY the final answer in \\boxed{}.")


# ---- The HF card's ACTUAL evaluation protocol (--protocol card) --------------
# VERBATIM from reference/hf-repo/README.md §Footnote, first bullet (the only
# bullet covering the reasoning benchmarks; fetched 2026-07-06, quote kept
# byte-exact incl. the double space in "evaluate  each"). Full fidelity map:
# docs/07-card-protocol-fidelity.md.
CARD_FOOTNOTE_REASONING = (
    "**Humanity’s Last Exam (HLE) & other reasoning tasks**: We use sampling "
    "parameters of `temperature=1.0`, `top_p=0.95` for evaluation. We evaluate "
    "with a maximum generation length of `163,840` tokens. By default, we "
    "report the text-only subset; results marked with * are from the full "
    "set. For AIME, HMMT and IMOAnswerBench, we evaluate  each question using "
    "the following system prompt: `Your response should be in the following "
    "format:\\nExplanation: {your explanation for your final answer}\\nExact "
    "Answer: {your succinct, final answer}\\nConfidence: {your confidence "
    "score between 0% and 100% for your answer}.` We use GPT-5.5 (medium) as "
    "the judge model. For HLE-with-tools, we use a maximum context length of "
    "300,000 tokens, with no context management strategy.")

# The card's answer-format system prompt for AIME/HMMT/IMOAnswerBench, decoded
# from the backticked string above (the README's literal \n are real newlines;
# the trailing period after the closing brace IS part of the prompt). A CPU
# test re-derives this from reference/hf-repo/README.md so it can never drift.
CARD_REASONING_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}.")


@dataclass
class CardProtocol:
    """One benchmark's protocol as the HF model card ACTUALLY specifies it
    (--protocol card). Every field is either quoted from the card (`source`)
    or an explicitly-labeled honest substitute (`substitutes`) — nothing here
    is invented. build/extract of None mean 'the greedy machinery, unchanged'
    (used when the card is silent on prompt/extraction)."""
    temperature: float
    top_p: float
    max_new: int                       # the card's max generation length
    system_prompt: str | None = None   # exact card system prompt (None = card gives none)
    build: Callable | None = None      # card-mode item builder (None = greedy items)
    extract: Callable | None = None    # card-mode extractor (None = greedy extractor)
    source: str = ""                   # verbatim card quote this protocol came from
    substitutes: str = ""              # what we CANNOT reproduce + the substitute used


def card_math_extract(reply, it):
    """Card-mode math extraction — the honest substitute for the card's
    GPT-5.5 (medium) judge, labeled as such: (1) the 'Exact Answer:' field
    the card's system prompt mandates; (2) if the reply ignored the format,
    the greedy extractors (last \\boxed{} / final number) AFTER dropping
    'Confidence:' lines (so a confidence percentage is never mistaken for
    the answer) — a stand-in for the judge's ability to read a nonconforming
    reply. Scoring stays exact-match (score_math); it is NOT a semantic judge."""
    ans = ex.extract_exact_answer(reply)
    if ans is not None:
        return ans
    fallback_body = ex.drop_confidence_lines(reply)
    return ex.extract_boxed(fallback_body) or ex.extract_final_number(fallback_body)


@dataclass
class Item:
    item_id: str
    question: str          # verbatim question (stored)
    prompt: str            # the full prompt sent to the model (stored)
    gold: str              # gold answer (letter for MC, value for math)
    n_choices: int = 0     # >0 for MC
    system_prompt: str | None = None   # card-mode system prompt (None = none)
    meta: dict = field(default_factory=dict)


@dataclass
class BenchSpec:
    name: str
    card_key: str          # key into provenance.CARD_TARGETS
    hf_path: str
    hf_config: str | None
    hf_split: str
    build: Callable        # (raw_row, idx) -> Item
    extract: Callable      # (reply, item) -> extracted  (item carries per-item n_choices)
    score: Callable        # (extracted, gold) -> bool
    # PINNED dataset commit sha (HfApi.dataset_info, fetched 2026-07-07):
    # an unpinned load_dataset() silently tracks upstream pushes — gold
    # answers (and for GPQA even row ORDER) could drift with no error
    # (review round3-ac37c8d finding 3). Recorded per item in Item.meta and
    # in the run's provenance env (run_bench._run_env).
    hf_revision: str | None = None
    note: str = ""
    # The HF card's protocol for this benchmark (--protocol card); None =
    # the card publishes no protocol for it (not on the card, or the agentic
    # harness is out of scope) — card mode refuses to run it.
    card: CardProtocol | None = None


def _mc_prompt(question: str, choices: list[str]) -> tuple[str, int]:
    lines = [question, ""]
    for i, c in enumerate(choices):
        lines.append(f"{ex._LETTERS[i]}. {c}")
    lines += ["", _THINK_HINT]
    return "\n".join(lines), len(choices)


# ---- GPQA-Diamond (gated: Idavidrein/gpqa) ----------------------------------
def _gpqa_build(row, idx):
    q = row["Question"]
    correct = row["Correct Answer"]
    choices = [correct, row["Incorrect Answer 1"],
               row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    # Per-item shuffle seed = a STABLE CONTENT HASH of the question text, NOT
    # the row index: an upstream row reorder must not silently relabel gold
    # letters (review round3-ac37c8d finding 3). NOTE: this CHANGES the gold
    # letters vs the idx-seeded scheme used before 2026-07-07 — acceptable, no
    # real (non-stub) runs were recorded under the old scheme.
    seed = int.from_bytes(hashlib.sha256(q.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)                 # deterministic per-item ordering
    order = list(range(4)); rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    gold_letter = ex._LETTERS[order.index(0)]  # where the correct answer landed
    prompt, n = _mc_prompt(q, shuffled)
    return Item(f"gpqa_{idx}", q, prompt, gold_letter, n_choices=n,
                meta={"category": row.get("Subdomain", ""),
                      "shuffle_seed": seed})


GPQA_DIAMOND = BenchSpec(
    "gpqa_diamond", "gpqa_diamond", "Idavidrein/gpqa", "gpqa_diamond", "train",
    _gpqa_build, lambda r, it: ex.extract_mc_letter(r, it.n_choices), ex.score_mc,
    hf_revision="633f5ee89ab8ad4522a9f850766b73f62147ffdd",  # pinned 2026-07-07
    note="gated on HF — needs HF_TOKEN; only a train split exists (198 items).",
    card=CardProtocol(
        temperature=1.0, top_p=0.95, max_new=163840,
        source=CARD_FOOTNOTE_REASONING,
        substitutes=(
            "The card specifies NO GPQA-specific protocol: footnote 1 only "
            "sets sampling (temperature=1.0, top_p=0.95) and the 163,840-"
            "token generation cap for 'HLE & other reasoning tasks', and "
            "scopes the Exact-Answer system prompt to AIME, HMMT and "
            "IMOAnswerBench ONLY. It says nothing about GPQA's MC prompt "
            "format (letter vs full option), choice ordering, answer "
            "extraction, or a judge. Card mode therefore keeps the greedy MC "
            "machinery unchanged (content-hash-shuffled A-D choices, "
            "\\boxed{} hint, letter extraction, exact letter match) and "
            "changes ONLY sampling params + the generation cap — the prompt/"
            "extraction side is a HARNESS choice, not a card reproduction.")))


# ---- MMLU-Pro (TIGER-Lab/MMLU-Pro) ------------------------------------------
def _mmlu_pro_build(row, idx):
    q = row["question"]
    choices = list(row["options"])
    gold = row["answer"]  # already a letter
    prompt, n = _mc_prompt(q, choices)
    return Item(f"mmlupro_{idx}", q, prompt, gold, n_choices=n,
                meta={"category": row.get("category", "")})


MMLU_PRO = BenchSpec(
    "mmlu_pro", "mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test",
    _mmlu_pro_build, lambda r, it: ex.extract_mc_letter(r, it.n_choices), ex.score_mc,
    hf_revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",  # pinned 2026-07-07
    note="generation-based MC; a fast sanity gate (NOT on the GLM-5.2 card).")


# ---- GSM8K (openai/gsm8k) ---------------------------------------------------
def _gsm8k_build(row, idx):
    q = row["question"]
    gold = row["answer"].split("####")[-1].strip().replace(",", "")
    prompt = f"{q}\n\n{_THINK_HINT}"
    return Item(f"gsm8k_{idx}", q, prompt, gold, meta={})


GSM8K = BenchSpec(
    "gsm8k", "gsm8k", "openai/gsm8k", "main", "test",
    _gsm8k_build,
    lambda r, it: (ex.extract_boxed(r) or ex.extract_final_number(r)),
    ex.score_math,
    hf_revision="740312add88f781978c0658806c59bc2815b9866",  # pinned 2026-07-07
    note="grade-school math; fast generation sanity gate.")


# ---- AIME 2026 (best-effort; verify dataset path at download time) ----------
def _aime_fields(row) -> tuple[str, str]:
    q = row.get("problem") or row.get("question") or ""
    ans = row.get("answer")
    gold = str(ans).strip() if ans is not None else str(row.get("solution") or "").strip()
    return q, gold


def _aime_build(row, idx):
    q, gold = _aime_fields(row)
    prompt = f"{q}\n\n{_THINK_HINT}"
    return Item(f"aime_{idx}", q, prompt, gold,
                meta={"problem_idx": row.get("problem_idx")})


def _aime_card_build(row, idx):
    """CARD-protocol item: the bare problem text as the user turn (VERBATIM —
    no \\boxed hint, nothing appended; the card specifies only the system
    prompt) + the card's Explanation/Exact Answer/Confidence system prompt on
    Item.system_prompt."""
    q, gold = _aime_fields(row)
    return Item(f"aime_{idx}", q, q, gold,
                system_prompt=CARD_REASONING_SYSTEM_PROMPT,
                meta={"problem_idx": row.get("problem_idx"), "protocol": "card"})


AIME_2026 = BenchSpec(
    "aime_2026", "aime_2026", "MathArena/aime_2026", None, "train",
    _aime_build, lambda r, it: (ex.extract_boxed(r) or ex.extract_final_number(r)),
    ex.score_math,
    hf_revision="d2de22f3c656b4f56cf8981212186377d1e23bc3",  # pinned 2026-07-07
    note="VERIFIED 2026-07-07: MathArena/aime_2026 [train] = the official 30-problem "
         "set (fields problem_idx/answer/problem; problem_idx 1-15 = AIME I, 16-30 = "
         "AIME II; all integer answers 0-999). All 30 answers cross-checked against "
         "two independent HF copies (MathArena/aime_2026_I, 96kevinli29/aime2026-en). "
         "opencompass/AIME2026 does not exist on the Hub.",
    card=CardProtocol(
        temperature=1.0, top_p=0.95, max_new=163840,
        system_prompt=CARD_REASONING_SYSTEM_PROMPT,
        build=_aime_card_build, extract=card_math_extract,
        source=CARD_FOOTNOTE_REASONING,
        substitutes=(
            "The card judges with GPT-5.5 (medium); we SUBSTITUTE exact-match "
            "(score_math normalization) on the extracted 'Exact Answer:' "
            "field, falling back to \\boxed{}/final-number (Confidence lines "
            "dropped first) when the reply ignores the format — labeled as a "
            "judge substitute, never a semantic judge. The card does NOT "
            "specify k samples or averaging; --samples N (avg@N over "
            "independent seeds) is a harness-side variance knob, not a card "
            "spec. Seeds are ours (recorded), the card publishes none.")))


REGISTRY = {b.name: b for b in (GPQA_DIAMOND, MMLU_PRO, GSM8K, AIME_2026)}


def load_items(spec: BenchSpec, limit: int | None = None,
               protocol: str = "greedy", offset: int = 0,
               ids: list[str] | set[str] | None = None) -> list[Item]:
    """Load + build items for a benchmark (needs the `datasets` lib + HF_TOKEN).
    Loads the PINNED dataset revision (spec.hf_revision) and stamps it into
    every item's meta so each stored row is traceable to the exact dataset
    commit it was built from.

    protocol='greedy' (default) builds items exactly as before (byte-identical
    prompts). protocol='card' builds the HF card's protocol items via
    spec.card.build (falling back to spec.build when the card is silent on the
    prompt — e.g. GPQA); refuses benchmarks with no card protocol mapped.

    `ids` (the --ids retry selector) loads ONLY the items whose item_id is in
    the given collection (e.g. {'gsm8k_3', 'gsm8k_17'}): the FULL dataset is
    scanned and items are returned in dataset order. Item ids are INDEX-
    derived ('gsm8k_{idx}') and stable ONLY under the pinned hf_revision — a
    revision bump that reorders rows would silently rename the questions, so
    never retry ids across a revision change (merge_runs' gold-mismatch
    warning catches scored drift, not a same-gold reorder). Every requested
    id must exist —
    a missing id raises instead of silently retrying a partial set. Refused
    together with limit/offset (both are slices of the same axis; --ids IS
    the selection). Composes with protocol: the ids name the same items under
    either builder."""
    if protocol not in ("greedy", "card"):
        raise ValueError(f"unknown protocol {protocol!r} (greedy|card)")
    wanted: set[str] | None = None
    if ids is not None:
        wanted = {str(s).strip() for s in ids if str(s).strip()}
        if not wanted:
            raise ValueError("ids given but empty")
        if limit is not None or offset:
            raise ValueError(
                "ids does not compose with limit/offset — the id list IS the "
                "selection (run a separate slice run instead)")
    build = spec.build
    if protocol == "card":
        if spec.card is None:
            raise ValueError(
                f"{spec.name} has no card protocol mapped (not on the GLM-5.2 "
                "card, or its harness is out of scope) — run it with "
                "--protocol greedy")
        build = spec.card.build or spec.build
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.hf_split,
                      revision=spec.hf_revision, token=token,
                      cache_dir=os.environ["HF_DATASETS_CACHE"])
    items = []
    for idx, row in enumerate(ds):
        if wanted is None:
            if idx < offset:
                continue
            if limit and idx >= offset + limit:
                break
        it = build(row, idx)
        if wanted is not None and it.item_id not in wanted:
            continue
        it.meta["hf_revision"] = spec.hf_revision
        items.append(it)
    if wanted is not None:
        missing = sorted(wanted - {it.item_id for it in items})
        if missing:
            raise ValueError(
                f"{spec.name}: requested item id(s) not in the dataset: "
                f"{missing} (have e.g. {items[0].item_id if items else '?'} — "
                "ids are the stored items.item_id values)")
    return items
