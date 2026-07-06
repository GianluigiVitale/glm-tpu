#!/usr/bin/env python3
"""CPU tests for the GLM-5.2 bench machinery — no model, no network, no TPU.
Validates the provenance DB round-trip, the extractors/scorers, and the
benchmark item-builders (on synthetic rows). Run:  ~/vllm-env/bin/python bench/test_bench.py
"""
from __future__ import annotations

import os
import tempfile

import extract as ex
import provenance as pv
import benchmarks as B


def test_extractors():
    assert ex.extract_mc_letter("...therefore the answer is \\boxed{C}", 4) == "C"
    assert ex.extract_mc_letter("<think>lots</think>\nAnswer: B", 4) == "B"
    assert ex.extract_mc_letter("I pick option D.", 4) == "D"
    assert ex.extract_mc_letter("no letter here 123", 4) is None
    assert ex.extract_mc_letter("A is wrong, B is wrong, final answer: J", 10) == "J"
    # a trailing bare letter on its own line is accepted...
    assert ex.extract_mc_letter("Long reasoning here.\n\nD", 4) == "D"
    # ...but a valid letter buried in prose is NOT guessed (the fallback misfire fix,
    # incl. the IGNORECASE-on-letter bug: lowercase 'a'/'i' in prose must NOT match):
    assert ex.extract_mc_letter("I am unsure. A, B, C all plausible.", 10) is None
    assert ex.extract_mc_letter("The letters A and I appear often in text.", 4) is None
    assert ex.extract_mc_letter("The result is a well-known constant", 10) is None
    assert ex.extract_mc_letter("It follows directly, as I noted earlier", 10) is None
    assert ex.extract_mc_letter("the answer is a function of x", 4) is None
    assert ex.extract_boxed(r"work \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert ex.extract_boxed(r"nested \boxed{x^{2}+1}") == "x^{2}+1"
    assert ex.extract_final_number("so 12 + 30 = 42. #### 42") == "42"
    assert ex.extract_final_number("the total is $1,234.5") == "1234.5"
    assert ex.strip_think("<think>hidden</think>visible").strip() == "visible"
    print("  extractors OK")


def test_scorers():
    assert ex.score_mc("C", "C") and not ex.score_mc("C", "D")
    assert not ex.score_mc(None, "A")
    assert ex.score_math("42", "42") and ex.score_math("42.0", "42")
    assert ex.score_math(r"\frac{1}{2}", "1/2") is False  # exact-string, not sympy (Stage 2)
    assert ex.score_math("003", "3")
    assert not ex.score_math(None, "5")
    print("  scorers OK")


def test_db_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        conn = pv.connect(db)
        rid = pv.start_run(conn, model="STUB", env={"dsa": False}, note="test")
        for i, (ex_ans, gold, ok) in enumerate([("A", "A", 1), ("B", "C", 0),
                                                 ("D", "D", 1), ("A", "A", 1)]):
            pv.record_item(conn, rid, benchmark="gpqa_diamond", item_id=f"q{i}",
                           prompt="Q?", gold=gold, raw_output="...",
                           extracted=ex_ans, correct=bool(ok), score=float(ok))
        summ = pv.finalize(conn, rid, benchmark="gpqa_diamond", metric="acc")
        assert summ["n"] == 4, summ
        assert abs(summ["value"] - 75.0) < 1e-9, summ           # 3/4
        assert summ["card_value"] == 91.2                        # from CARD_TARGETS
        assert abs(summ["delta"] - (75.0 - 91.2)) < 1e-9, summ
        # every item is retrievable (traceability)
        rows = conn.execute("SELECT prompt,raw_output,extracted,correct FROM items"
                            " WHERE run_id=?", (rid,)).fetchall()
        assert len(rows) == 4 and all(r[0] == "Q?" for r in rows)
        assert conn.execute("SELECT harness_git FROM runs WHERE run_id=?",
                            (rid,)).fetchone()[0]  # a git sha (or 'unknown')
    print("  DB round-trip + provenance OK")


def test_item_builders():
    # GPQA: synthetic row → deterministic 4-way MC, gold letter tracks the correct answer
    row = {"Question": "2+2?", "Correct Answer": "4",
           "Incorrect Answer 1": "3", "Incorrect Answer 2": "5",
           "Incorrect Answer 3": "6", "Subdomain": "math"}
    it = B._gpqa_build(row, 7)
    assert it.n_choices == 4 and it.gold in "ABCD"
    assert "4" in it.prompt and "\\boxed{}" in it.prompt
    # gold letter must point at the "4" choice
    lines = [l for l in it.prompt.splitlines() if l[:2] in
             ("A.", "B.", "C.", "D.")]
    gold_line = [l for l in lines if l.startswith(it.gold + ".")][0]
    assert gold_line.endswith("4"), gold_line
    # deterministic: same idx → same ordering
    assert B._gpqa_build(row, 7).gold == it.gold
    # MMLU-Pro + GSM8K builders
    mp = B._mmlu_pro_build({"question": "q", "options": ["a", "b", "c"],
                            "answer": "B", "category": "x"}, 0)
    assert mp.gold == "B" and mp.n_choices == 3
    # spec.extract now takes (reply, item) and clamps to the item's real n_choices:
    assert B.MMLU_PRO.extract("Answer: B", mp) == "B"
    assert B.MMLU_PRO.extract("Answer: E", mp) is None   # E out of range (3 choices)
    g = B._gsm8k_build({"question": "q", "answer": "steps #### 1,234"}, 0)
    assert g.gold == "1234"
    print("  item builders OK")


if __name__ == "__main__":
    test_extractors()
    test_scorers()
    test_db_roundtrip()
    test_item_builders()
    print("ALL bench CPU tests passed.")
