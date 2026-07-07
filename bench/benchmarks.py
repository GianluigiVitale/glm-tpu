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


@dataclass
class Item:
    item_id: str
    question: str          # verbatim question (stored)
    prompt: str            # the full prompt sent to the model (stored)
    gold: str              # gold answer (letter for MC, value for math)
    n_choices: int = 0     # >0 for MC
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
    note="gated on HF — needs HF_TOKEN; only a train split exists (198 items).")


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
def _aime_build(row, idx):
    q = row.get("problem") or row.get("question") or ""
    ans = row.get("answer")
    gold = str(ans).strip() if ans is not None else str(row.get("solution") or "").strip()
    prompt = f"{q}\n\n{_THINK_HINT}"
    return Item(f"aime_{idx}", q, prompt, gold,
                meta={"problem_idx": row.get("problem_idx")})


AIME_2026 = BenchSpec(
    "aime_2026", "aime_2026", "MathArena/aime_2026", None, "train",
    _aime_build, lambda r, it: (ex.extract_boxed(r) or ex.extract_final_number(r)),
    ex.score_math,
    hf_revision="d2de22f3c656b4f56cf8981212186377d1e23bc3",  # pinned 2026-07-07
    note="VERIFIED 2026-07-07: MathArena/aime_2026 [train] = the official 30-problem "
         "set (fields problem_idx/answer/problem; problem_idx 1-15 = AIME I, 16-30 = "
         "AIME II; all integer answers 0-999). All 30 answers cross-checked against "
         "two independent HF copies (MathArena/aime_2026_I, 96kevinli29/aime2026-en). "
         "opencompass/AIME2026 does not exist on the Hub.")


REGISTRY = {b.name: b for b in (GPQA_DIAMOND, MMLU_PRO, GSM8K, AIME_2026)}


def load_items(spec: BenchSpec, limit: int | None = None) -> list[Item]:
    """Load + build items for a benchmark (needs the `datasets` lib + HF_TOKEN).
    Loads the PINNED dataset revision (spec.hf_revision) and stamps it into
    every item's meta so each stored row is traceable to the exact dataset
    commit it was built from."""
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.hf_split,
                      revision=spec.hf_revision, token=token,
                      cache_dir=os.environ["HF_DATASETS_CACHE"])
    items = []
    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break
        it = spec.build(row, idx)
        it.meta["hf_revision"] = spec.hf_revision
        items.append(it)
    return items
