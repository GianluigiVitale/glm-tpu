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


def test_adversarial_extractors():
    """Red cases found 2026-07-07 (hand-written adversarial replies; all were
    misses before the extract.py fixes). The prompt asks for \\boxed{}, and a
    reasoning model wraps the letter in \\text{}/parens/punct/markdown."""
    # MC: boxed-letter variants (brace-balanced, \text-unwrapped, punct-stripped)
    assert ex.extract_mc_letter(
        "<think>maybe A or B</think>\nAfter analysis, the final answer is "
        r"\boxed{\text{C}}", 4) == "C"
    assert ex.extract_mc_letter(r"\boxed{(B)}", 4) == "B"
    assert ex.extract_mc_letter(r"So the answer is \boxed{D.}", 4) == "D"
    # MC: markdown-bold letter + 'option is'
    assert ex.extract_mc_letter("The final answer is **B**.", 4) == "B"
    assert ex.extract_mc_letter("The correct option is (C).", 4) == "C"
    # MC: the LAST boxed is the final answer
    assert ex.extract_mc_letter(r"try \boxed{A}... reconsider: \boxed{B}", 4) == "B"
    # MC round3-ac37c8d flip cases — the LATEST explicit answer wins, in BOTH
    # directions (no silent-wrong, no lost extraction):
    # (i) explicit final-answer prose AFTER the last boxed supersedes it...
    assert ex.extract_mc_letter(
        r"Let me try \boxed{(A)} first. Hmm, that fails the check. "
        "The final answer is B.", 4) == "B"
    # (ii) ...while a decorated boxed AFTER earlier boxed/prose still wins
    assert ex.extract_mc_letter(r"maybe \boxed{A}? No. Final: \boxed{(C)}", 4) == "C"
    assert ex.extract_mc_letter("The final answer is B, i.e. "
                                r"\boxed{B}", 4) == "B"
    # (iii) prose revised by later prose: last keyword phrase wins
    assert ex.extract_mc_letter("the answer is A... wait, "
                                "the final answer is D", 4) == "D"
    # MC never-guess guards must survive the forgiving boxed path:
    assert ex.extract_mc_letter(r"\boxed{42}", 4) is None      # non-letter boxed
    assert ex.extract_mc_letter(r"\boxed{AB}", 4) is None      # two letters
    assert ex.extract_mc_letter("the answer is a function of x", 4) is None
    # math: thousands-separator comma + (escaped) dollar inside \boxed{}
    assert ex.score_math(ex.extract_boxed(r"The total is \boxed{1,234}."), "1234")
    assert ex.score_math(ex.extract_boxed(r"she makes \boxed{\$18} per day"), "18")
    assert ex.norm_math("1,234,567") == "1234567"
    assert ex.norm_math("(1,2)") == "(1,2)"    # NOT a thousands separator — kept
    assert ex.norm_math("$42.") == "42"
    # round3-ac37c8d comma flip cases: strip ONLY a whole plain grouped number
    assert ex.norm_math("(1,200)") == "(1,200)"   # tuple/interval — kept
    assert not ex.score_math("(1,200)", "1200")   # no silent-wrong inflation
    assert ex.norm_math("0,001") == "0,001"       # European decimal — kept
    assert not ex.score_math("0,001", "1")        # (old code scored this True)
    assert ex.score_math("1,234.56", "1234.56")   # grouped decimal still works
    assert ex.score_math("$1,234.56", "1234.56")
    assert ex.score_math("-1,234", "-1234")
    assert ex.score_math(ex.extract_boxed(r"\boxed{1,234}."), "1234")  # trailing .
    print("  adversarial extractors OK")


def test_exact_answer_extractor():
    """extract_exact_answer — the card-protocol 'Exact Answer:' field (the
    labeled GPT-5.5-judge substitute), incl. adversarial replies."""
    E = ex.extract_exact_answer
    # the card format, verbatim shape
    assert E("Explanation: because reasons\nExact Answer: 42\n"
             "Confidence: 95%") == "42"
    # think block stripped; the field lives after it
    assert E("<think>maybe 7? no.</think>\nExplanation: x\n"
             "Exact Answer: 128\nConfidence: 100%") == "128"
    # markdown emphasis on marker and/or answer
    assert E("**Exact Answer:** 042") == "042"
    assert E("Exact Answer: **17**\nConfidence: 80%") == "17"
    assert E("__Exact Answer__: 5") == "5"
    # \boxed{} inside the field is unwrapped (brace-balanced)
    assert E("Exact Answer: \\boxed{128}\nConfidence: 99%") == "128"
    assert E("Exact Answer: \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    # Confidence on the SAME line is cut off
    assert E("Exact Answer: 42 Confidence: 90%") == "42"
    assert E("Exact Answer: 42. Confidence: 90%") == "42"
    # the LAST marker wins (drafted format restated later)
    assert E("Exact Answer: 7\n...wait, recompute...\n"
             "Exact Answer: 128\nConfidence: 99%") == "128"
    # answer on the NEXT line still found
    assert E("Exact Answer:\n128\nConfidence: 99%") == "128"
    # ADVERSARIAL — never guess:
    assert E("the exact answer might be 7") is None          # no colon
    assert E("the exact answer is 7, I think") is None       # no colon
    assert E("an inexact answer: 7") is None                 # \b guard
    assert E("Exact Answer:\nConfidence: 90%") is None       # empty field
    assert E("Exact Answer:   \n\nConfidence: 90%") is None  # whitespace field
    assert E("") is None and E("no field here 42") is None
    # marker ONLY inside a closed think block does not count
    assert E("<think>Exact Answer: 5</think>\nThe answer is 6") is None
    # case-insensitive marker, uppercase answer preserved
    assert E("EXACT ANSWER: 10") == "10"
    # normalization compatibility with score_math (trailing '.', leading 0s, $)
    assert ex.score_math(E("Exact Answer: 042.\nConfidence: 1%"), "42")
    assert ex.score_math(E("Exact Answer: \\boxed{1,234}\nConfidence: 5%"), "1234")
    assert ex.score_math(E("Exact Answer: $18\nConfidence: 5%"), "18")
    # drop_confidence_lines: the fallback path must never grab a confidence
    # percentage as the answer (card reply that skipped 'Exact Answer:')
    nonconforming = "I believe the result is 128.\nConfidence: 95%"
    assert ex.extract_final_number(nonconforming) == "95"     # the trap...
    body = ex.drop_confidence_lines(nonconforming)
    assert ex.extract_final_number(body) == "128"             # ...defused
    import benchmarks as _B
    assert _B.card_math_extract(nonconforming, None) == "128"
    assert _B.card_math_extract(
        "Explanation: y\nExact Answer: 42\nConfidence: 90%", None) == "42"
    # fallback chain still finds a boxed answer in a nonconforming reply
    assert _B.card_math_extract(
        "<think>t</think> the answer is \\boxed{60}", None) == "60"
    # a reply that is ONLY a confidence line extracts nothing (scored wrong)
    assert _B.card_math_extract("Confidence: 100%", None) is None
    print("  exact-answer extractor (card protocol) OK")


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


def test_batched_run():
    """run_benchmark's BATCHED path on a fake generate_batch (no vllm, no TPU):
    ordering + provenance. The fake returns canned replies out of a dict keyed
    by the verbatim prompt, so any order slip inside run_benchmark would pair a
    reply with the wrong item and flip correct/extracted below."""
    import run_bench as rb

    # the stub must NOT grow a generate_batch — --stub stays sequential
    assert not hasattr(rb.stub_generate, "generate_batch")

    items = [B.Item("b0", "1+1?", "PROMPT-ZERO", "2"),
             B.Item("b1", "2+3?", "PROMPT-ONE", "5"),
             B.Item("b2", "3+4?", "PROMPT-TWO", "7")]
    canned = {  # prompt -> (verbatim reply, n_gen_tokens); b1 deliberately wrong
        "PROMPT-ZERO": (r"<think>t</think> so \boxed{2}", 11),
        "PROMPT-ONE":  (r"<think>t</think> so \boxed{99}", 22),
        "PROMPT-TWO":  (r"<think>t</think> so \boxed{7}", 33)}
    calls = []

    def fake_gen(prompt):
        raise AssertionError("sequential generate() must NOT be called when "
                             "generate_batch is exposed")

    def fake_batch(prompts):
        calls.append(list(prompts))
        return [canned[p] for p in prompts]

    fake_gen.generate_batch = fake_batch

    orig_load = B.load_items
    B.load_items = (lambda spec, limit=None, protocol="greedy", offset=0:
                    list(items[:limit] if limit else items))
    try:
        with tempfile.TemporaryDirectory() as d:
            conn = pv.connect(os.path.join(d, "t.db"))

            # ---- batch_size=0: ALL items in ONE generate_batch call
            rid = pv.start_run(conn, model="FAKE-BATCH", env={}, note="unit")
            summ = rb.run_benchmark(conn, rid, B.GSM8K, fake_gen, batch_size=0)
            assert calls == [["PROMPT-ZERO", "PROMPT-ONE", "PROMPT-TWO"]]
            rows = conn.execute(
                "SELECT item_id,prompt,gold,raw_output,extracted,correct,"
                "n_gen_tokens,latency_ms FROM items WHERE run_id=? ORDER BY id",
                (rid,)).fetchall()
            assert [r[0] for r in rows] == ["b0", "b1", "b2"]
            for r, it in zip(rows, items):  # verbatim provenance, per item
                assert r[1] == it.prompt and r[2] == it.gold
                assert r[3] == canned[it.prompt][0]
                assert r[6] == canned[it.prompt][1]
                assert r[7] is None          # latency never faked in a batch
            assert rows[0][4:6] == ("2", 1)   # extracted + correct track...
            assert rows[1][4:6] == ("99", 0)  # ...EACH item's own reply
            assert rows[2][4:6] == ("7", 1)
            assert summ["n"] == 3 and abs(summ["value"] - 200.0 / 3) < 1e-9
            note = conn.execute("SELECT note FROM summary WHERE run_id=?",
                                (rid,)).fetchone()[0]
            assert note.startswith("batched:3 chunks=1 batch_wall_ms="), note

            # ---- batch_size=2: chunked [2, 1], order preserved across chunks
            calls.clear()
            rid2 = pv.start_run(conn, model="FAKE-BATCH", env={}, note="unit2")
            rb.run_benchmark(conn, rid2, B.GSM8K, fake_gen, batch_size=2)
            assert calls == [["PROMPT-ZERO", "PROMPT-ONE"], ["PROMPT-TWO"]]
            rows2 = conn.execute(
                "SELECT item_id,extracted,correct FROM items WHERE run_id=?"
                " ORDER BY id", (rid2,)).fetchall()
            assert rows2 == [("b0", "2", 1), ("b1", "99", 0), ("b2", "7", 1)]
            note2 = conn.execute("SELECT note FROM summary WHERE run_id=?",
                                 (rid2,)).fetchone()[0]
            assert note2.startswith("batched:3 chunks=2 batch_wall_ms="), note2

            # ---- a generator WITHOUT generate_batch still runs sequentially
            rid3 = pv.start_run(conn, model="FAKE-SEQ", env={}, note="unit3")
            rb.run_benchmark(conn, rid3, B.GSM8K,
                             lambda p: canned[p], batch_size=0)
            rows3 = conn.execute(
                "SELECT extracted,correct,latency_ms FROM items WHERE run_id=?"
                " ORDER BY id", (rid3,)).fetchall()
            assert [r[:2] for r in rows3] == [("2", 1), ("99", 0), ("7", 1)]
            assert all(r[2] is not None for r in rows3)  # sequential keeps latency
    finally:
        B.load_items = orig_load
    print("  batched run (ordering + provenance + chunking) OK")


def test_card_protocol_specs():
    """The card-protocol registry mapping — what the HF card ACTUALLY
    specifies (docs/07-card-protocol-fidelity.md), pinned so it cannot drift."""
    # the card system prompt is BYTE-DERIVED from the committed README copy:
    # decode the backticked \n-escaped string and compare
    readme = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "reference", "hf-repo", "README.md")
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    escaped = B.CARD_REASONING_SYSTEM_PROMPT.replace("\n", "\\n")
    assert f"`{escaped}`" in text, \
        "CARD_REASONING_SYSTEM_PROMPT drifted from reference/hf-repo/README.md"
    # sampling params + generation cap, as the footnote states them
    for spec in (B.AIME_2026, B.GPQA_DIAMOND):
        cp = spec.card
        assert cp is not None
        assert cp.temperature == 1.0 and cp.top_p == 0.95, spec.name
        assert cp.max_new == 163840, spec.name
        assert "temperature=1.0" in cp.source and "top_p=0.95" in cp.source
        assert "163,840" in cp.source
    # AIME: the card's system prompt + Exact-Answer extractor; bare question
    assert B.AIME_2026.card.system_prompt == B.CARD_REASONING_SYSTEM_PROMPT
    assert B.AIME_2026.card.extract is B.card_math_extract
    assert "GPT-5.5" in B.AIME_2026.card.substitutes   # judge substitute labeled
    row = {"problem": "Find x such that x+1=2.", "answer": 1, "problem_idx": 3}
    it = B._aime_card_build(row, 2)
    assert it.prompt == "Find x such that x+1=2."     # VERBATIM, no \boxed hint
    assert "\\boxed" not in it.prompt
    assert it.system_prompt == B.CARD_REASONING_SYSTEM_PROMPT
    assert it.gold == "1" and it.item_id == "aime_2"
    # greedy AIME items are unchanged (byte-identical default protocol)
    itg = B._aime_build(row, 2)
    assert itg.prompt.endswith(B._THINK_HINT) and itg.system_prompt is None
    # GPQA: the card is SILENT on prompt/extraction — card mode keeps the
    # greedy MC machinery (build/extract None => spec.build/spec.extract)
    assert B.GPQA_DIAMOND.card.build is None
    assert B.GPQA_DIAMOND.card.extract is None
    assert B.GPQA_DIAMOND.card.system_prompt is None
    assert "NO GPQA-specific protocol" in B.GPQA_DIAMOND.card.substitutes
    # not-on-card benchmarks have NO card protocol and card mode refuses them
    assert B.MMLU_PRO.card is None and B.GSM8K.card is None
    try:
        B.load_items(B.GSM8K, limit=1, protocol="card")
        raise AssertionError("card protocol on gsm8k must be refused")
    except ValueError as e:
        assert "no card protocol" in str(e)
    try:
        B.load_items(B.GSM8K, limit=1, protocol="typo")
        raise AssertionError("unknown protocol must be refused")
    except ValueError as e:
        assert "unknown protocol" in str(e)
    print("  card-protocol specs (README-pinned) OK")


def test_card_protocol_run():
    """run_benchmark --protocol card end-to-end on a fake generator (no vllm):
    card sampling params + system prompts + per-sample seeds reach the
    generator; every sample is its own provenance row ('#sN' ids, seed
    recorded); the summary is avg@N; the stub accepts card kwargs."""
    import run_bench as rb

    # the stub swallows card kwargs (--stub --protocol card works offline)
    assert rb.stub_generate("x", system_prompt="s", sampling={}, seed=3) == ""

    sysp = B.CARD_REASONING_SYSTEM_PROMPT
    items = [B.Item("a0", "Q0", "Q0", "42", system_prompt=sysp),
             B.Item("a1", "Q1", "Q1", "7", system_prompt=sysp)]
    canned = {  # (prompt, seed) -> reply; sample 1 gets a1 wrong
        ("Q0", 5): "Explanation: e\nExact Answer: 42\nConfidence: 90%",
        ("Q1", 5): "Explanation: e\nExact Answer: 7\nConfidence: 90%",
        ("Q0", 6): "Explanation: f\nExact Answer: 42\nConfidence: 80%",
        ("Q1", 6): "Explanation: f\nExact Answer: 9\nConfidence: 80%"}
    calls = []

    def fake_gen(prompt, **kw):
        raise AssertionError("batched path must be used")

    def fake_batch(prompts, *, system_prompts=None, sampling=None, seed=None):
        calls.append((list(prompts), list(system_prompts), dict(sampling), seed))
        return [(canned[(p, seed)], 5, "stop") for p in prompts]

    fake_gen.generate_batch = fake_batch

    orig_load = B.load_items
    seen_protocols = []

    def fake_load(spec, limit=None, protocol="greedy", offset=0):
        seen_protocols.append(protocol)
        return list(items)

    B.load_items = fake_load
    try:
        with tempfile.TemporaryDirectory() as d:
            conn = pv.connect(os.path.join(d, "t.db"))
            rid = pv.start_run(conn, model="FAKE-CARD", env={}, note="unit")
            summ = rb.run_benchmark(conn, rid, B.AIME_2026, fake_gen,
                                    batch_size=0, protocol="card", samples=2,
                                    seed=5)
            assert seen_protocols == ["card"]
            # card sampling params + system prompts + per-sample seeds
            assert len(calls) == 2
            card_sampling = {"temperature": 1.0, "top_p": 0.95,
                             "max_new": 163840}
            assert calls[0] == (["Q0", "Q1"], [sysp, sysp], card_sampling, 5)
            assert calls[1] == (["Q0", "Q1"], [sysp, sysp], card_sampling, 6)
            # per-sample provenance rows: '#sN' ids, per-row seed, own replies
            rows = conn.execute(
                "SELECT item_id,seed,extracted,correct,raw_output FROM items"
                " WHERE run_id=? ORDER BY id", (rid,)).fetchall()
            assert [r[0] for r in rows] == ["a0#s0", "a1#s0", "a0#s1", "a1#s1"]
            assert [r[1] for r in rows] == [5, 5, 6, 6]
            assert [(r[2], r[3]) for r in rows] == \
                [("42", 1), ("7", 1), ("42", 1), ("9", 0)]
            assert rows[3][4] == canned[("Q1", 6)]      # verbatim reply stored
            # avg@2 = mean over ALL sample rows = 3/4
            assert summ["n"] == 4 and abs(summ["value"] - 75.0) < 1e-9
            note = conn.execute("SELECT note FROM summary WHERE run_id=?",
                                (rid,)).fetchone()[0]
            assert note.startswith("protocol=card temp=1.0 top_p=0.95 "
                                   "card_max_new=163840 samples=2 "
                                   "base_seed=5"), note

            # samples=1 keeps plain item ids (no suffix); card seed recorded
            calls.clear()
            rid2 = pv.start_run(conn, model="FAKE-CARD", env={}, note="unit2")
            rb.run_benchmark(conn, rid2, B.AIME_2026, fake_gen, batch_size=0,
                             protocol="card", samples=1, seed=5)
            rows2 = conn.execute(
                "SELECT item_id,seed FROM items WHERE run_id=? ORDER BY id",
                (rid2,)).fetchall()
            assert rows2 == [("a0", 5), ("a1", 5)]

            # SEQUENTIAL card path (no generate_batch): kwargs + latency kept
            seq_calls = []

            def fake_seq(prompt, *, system_prompt=None, sampling=None,
                         seed=None):
                seq_calls.append((prompt, system_prompt, dict(sampling), seed))
                return (canned[(prompt, seed)], 5, "stop")

            rid3 = pv.start_run(conn, model="FAKE-CARD", env={}, note="unit3")
            rb.run_benchmark(conn, rid3, B.AIME_2026, fake_seq, batch_size=0,
                             protocol="card", samples=1, seed=5)
            assert seq_calls == [("Q0", sysp, card_sampling, 5),
                                 ("Q1", sysp, card_sampling, 5)]
            rows3 = conn.execute(
                "SELECT extracted,correct,latency_ms FROM items WHERE run_id=?"
                " ORDER BY id", (rid3,)).fetchall()
            assert [r[:2] for r in rows3] == [("42", 1), ("7", 1)]
            assert all(r[2] is not None for r in rows3)

            # greedy refuses samples>1
            try:
                rb.run_benchmark(conn, rid3, B.AIME_2026, fake_gen,
                                 protocol="greedy", samples=2)
                raise AssertionError("greedy samples>1 must be refused")
            except ValueError as e:
                assert "card" in str(e)
    finally:
        B.load_items = orig_load
    print("  card-protocol run (sampling/seeds/avg@N provenance) OK")


def test_seed_provenance_gating():
    """Round-6 F1/F10 (docs/reviews/round6-mtp-card.md): items.seed records
    the seed the ENGINE actually received. A generator that declares
    per_request_seeds=False (make_generate's platform probe result on the TPU
    stack) still RECEIVES the per-sample seed (label/keying) but the rows
    record NULL and the note says seed_passthrough=off. Greedy rows record
    NULL (no seed is ever sent; previously they recorded the unused --seed
    value, so the column meant different things across protocols)."""
    import run_bench as rb

    sysp = B.CARD_REASONING_SYSTEM_PROMPT
    items = [B.Item("a0", "Q0", "Q0", "42", system_prompt=sysp)]
    reply = "Explanation: e\nExact Answer: 42\nConfidence: 90%"
    calls = []

    def fake_gen(prompt, **kw):
        raise AssertionError("batched path must be used")

    def fake_batch(prompts, *, system_prompts=None, sampling=None, seed=None):
        calls.append(seed)
        return [(reply, 5, "stop") for _ in prompts]

    fake_gen.generate_batch = fake_batch
    fake_gen.per_request_seeds = False          # the TPU engine path

    orig_load = B.load_items
    B.load_items = (lambda spec, limit=None, protocol="greedy", offset=0:
                    list(items))
    try:
        with tempfile.TemporaryDirectory() as d:
            conn = pv.connect(os.path.join(d, "t.db"))

            # card, 2 samples: the generator still gets 5 then 6 (it is the
            # layer that omits them from SamplingParams); rows record NULL
            rid = pv.start_run(conn, model="FAKE", env={}, note="unit")
            rb.run_benchmark(conn, rid, B.AIME_2026, fake_gen, batch_size=0,
                             protocol="card", samples=2, seed=5)
            assert calls == [5, 6]
            rows = conn.execute("SELECT item_id, seed FROM items WHERE "
                                "run_id=? ORDER BY id", (rid,)).fetchall()
            assert rows == [("a0#s0", None), ("a0#s1", None)]
            note = conn.execute("SELECT note FROM summary WHERE run_id=?",
                                (rid,)).fetchone()[0]
            assert "seed_passthrough=off" in note and "labels only" in note

            # a seed-capable generator (no attribute = passthrough on, the
            # pre-existing fake/stub behavior) records the real seeds
            del fake_gen.per_request_seeds
            calls.clear()
            rid2 = pv.start_run(conn, model="FAKE", env={}, note="unit2")
            rb.run_benchmark(conn, rid2, B.AIME_2026, fake_gen, batch_size=0,
                             protocol="card", samples=2, seed=5)
            assert calls == [5, 6]
            rows2 = conn.execute("SELECT seed FROM items WHERE run_id=? "
                                 "ORDER BY id", (rid2,)).fetchall()
            assert [r[0] for r in rows2] == [5, 6]
            note2 = conn.execute("SELECT note FROM summary WHERE run_id=?",
                                 (rid2,)).fetchone()[0]
            assert "seed_passthrough=on" in note2

            # greedy: no seed is ever sent -> NULL even with --seed 9
            def gfake(prompt):
                raise AssertionError("batched path must be used")

            def gfake_batch(prompts):
                return [(r"\boxed{2}", 3, "stop") for _ in prompts]

            gfake.generate_batch = gfake_batch
            rid3 = pv.start_run(conn, model="FAKE", env={}, note="unit3")
            rb.run_benchmark(conn, rid3, B.GSM8K, gfake, batch_size=0, seed=9)
            rows3 = conn.execute("SELECT seed FROM items WHERE run_id=?",
                                 (rid3,)).fetchall()
            assert [r[0] for r in rows3] == [None]
    finally:
        B.load_items = orig_load
    print("  seed provenance gating (F1/F10: items.seed = engine seed) OK")


def test_platform_seed_probe():
    """Round-6 F1's engine-side gap: 'seeds reach the generator' was tested,
    'the engine accepts them' was NOT. Probe the REAL installed vllm platform
    exactly as make_generate does. On this stack vllm resolves the fork's
    TpuPlatform, whose validate_request rejects SamplingType.RANDOM_SEED —
    the probe MUST return False here (that is what keeps --protocol card from
    crashing on its first request after a ~45-min engine build), and the
    seedless params card mode now sends MUST pass validation. The real-vllm
    probe runs in a SUBPROCESS (JAX_PLATFORMS=cpu forced before python
    starts) so this test process never imports vllm — test_longctx's offline
    contract ('vllm' not in sys.modules after a --stub run) must keep holding
    in a combined pytest session. Prints a SKIP when vllm is not importable
    so the suite stays runnable on any box."""
    import subprocess
    import sys

    import run_bench as rb

    # fail-SAFE contract: a broken/unknown platform API counts as unsupported
    class _Boom:
        def __init__(self, *a, **kw):
            raise TypeError("no SamplingParams here")

    ok, msg = rb._per_request_seed_support(_Boom)
    assert ok is False and "probe failed" in msg, (ok, msg)

    here = os.path.dirname(os.path.abspath(__file__))
    script = f"""
import os, sys
sys.path.insert(0, {here!r})
import run_bench as rb
try:
    # LLM first: platform resolution must settle before any vllm submodule
    # import (the exact order make_generate uses — see its comment).
    from vllm import LLM  # noqa: F401
    from vllm import SamplingParams
    from vllm.platforms import current_platform
except Exception as exc:
    print("SKIP:" + type(exc).__name__); sys.exit(0)
ok, msg = rb._per_request_seed_support(SamplingParams)
plat = type(current_platform).__name__
if "tpu" in plat.lower():
    assert ok is False, (plat, msg)
    assert "per-request seed" in msg, msg
    # what card mode now sends (seed omitted) must be accepted
    current_platform.validate_request(
        None, SamplingParams(temperature=1.0, top_p=0.95, seed=None))
print("RESULT:%s:per_request_seeds=%s" % (plat, ok))
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu")     # never touch a TPU
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert tail.startswith(("RESULT:", "SKIP:")), (proc.stdout, proc.stderr)
    print(f"  engine-side seed probe OK ({tail})")


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
    # deterministic AND row-order independent: the shuffle seed is a sha256
    # content hash of the question text, NOT the row index — an upstream row
    # reorder must not relabel gold letters (round3-ac37c8d finding 3; this
    # CHANGED gold letters vs the pre-2026-07-07 idx-seeded scheme — fine, no
    # real runs were recorded under it)
    assert B._gpqa_build(row, 7).gold == it.gold
    it3 = B._gpqa_build(row, 3)
    assert it3.gold == it.gold and it3.n_choices == it.n_choices
    assert [l.split(". ", 1)[1] for l in it3.prompt.splitlines()
            if l[:2] in ("A.", "B.", "C.", "D.")] == \
           [l.split(". ", 1)[1] for l in lines]
    # pinned canary (guards a hypothetical CPython shuffle change AND the
    # content-hash seed derivation itself — round3-ac37c8d finding 5):
    assert it.meta["shuffle_seed"] == 3507495222521963811
    assert it.gold == "C" and [l.split(". ", 1)[1] for l in lines] == \
        ["5", "6", "4", "3"]
    # every registered benchmark carries a pinned 40-hex dataset revision
    import re as _re
    for spec in B.REGISTRY.values():
        assert spec.hf_revision and _re.fullmatch(r"[0-9a-f]{40}",
                                                  spec.hf_revision), spec.name
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


def test_dcp_engine_kwarg():
    """GLM_DCP plumbing (runbook §5 — decode context parallelism): unset (or
    0/empty) -> decode_context_parallel_size ABSENT from the LLM(...) kwargs,
    engine args byte-identical to the pre-DCP harness; GLM_DCP=N -> the kwarg
    present with value N AND recorded in the engine-built log line. The kwarg
    name is vLLM's EngineArgs.decode_context_parallel_size (verified against
    ~/vllm-build/vllm/engine/arg_utils.py). Monkeypatched vllm.LLM capture —
    no real vllm import, no TPU."""
    import contextlib
    import io
    import sys
    import types

    import engine

    calls = []

    class _CaptureLLM:
        def __init__(self, **kw):
            calls.append(kw)

    fake = types.ModuleType("vllm")
    fake.LLM = _CaptureLLM
    had_vllm = "vllm" in sys.modules
    old_mod = sys.modules.get("vllm")
    old_env = os.environ.pop("GLM_DCP", None)
    try:
        sys.modules["vllm"] = fake
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            engine.build_llm("stub-model")
        assert "decode_context_parallel_size" not in calls[0]  # unset -> absent
        assert "dcp=" not in out.getvalue()          # log line unchanged too

        os.environ["GLM_DCP"] = "4"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            engine.build_llm("stub-model")
        assert calls[1]["decode_context_parallel_size"] == 4
        assert "dcp=4" in out.getvalue()             # engine-built log line
        # every OTHER engine arg byte-identical to the unset build:
        assert {k: v for k, v in calls[1].items()
                if k != "decode_context_parallel_size"} == calls[0]

        os.environ["GLM_DCP"] = "0"                  # 0 = off = absent
        with contextlib.redirect_stdout(io.StringIO()):
            engine.build_llm("stub-model")
        assert "decode_context_parallel_size" not in calls[2]

        os.environ["GLM_DCP"] = "-2"                 # negative: readable
        try:                                         # harness-boundary error,
            engine.build_llm("stub-model")           # not a pydantic trace
            raise AssertionError("GLM_DCP=-2 should have raised")
        except ValueError as e:
            assert "GLM_DCP" in str(e)
        assert len(calls) == 3                       # LLM never constructed
    finally:
        if old_env is None:
            os.environ.pop("GLM_DCP", None)
        else:
            os.environ["GLM_DCP"] = old_env
        if had_vllm:
            sys.modules["vllm"] = old_mod
        else:
            sys.modules.pop("vllm", None)
    print("  DCP engine kwarg plumbing OK")


def test_spec_engine_kwarg():
    """GLM_SPEC_K plumbing (runbook §7 — Stage-3 MTP spec-decode, docs/08 §7
    M2): unset (or 0/empty) -> speculative_config ABSENT from the LLM(...)
    kwargs, engine args byte-identical to the pre-MTP harness (the M2 gate's
    'non-spec serving numbers unregressed' precondition); GLM_SPEC_K=k -> the
    kwarg present as {"method": "mtp", "num_speculative_tokens": k} (shape
    verified against ~/vllm-build/vllm/engine/arg_utils.py EngineArgs.
    speculative_config -> SpeculativeConfig(**dict)) AND recorded in the
    engine-built log line. Monkeypatched vllm.LLM capture — no real vllm
    import, no TPU."""
    import contextlib
    import io
    import sys
    import types

    import engine

    calls = []

    class _CaptureLLM:
        def __init__(self, **kw):
            calls.append(kw)

    fake = types.ModuleType("vllm")
    fake.LLM = _CaptureLLM
    had_vllm = "vllm" in sys.modules
    old_mod = sys.modules.get("vllm")
    old_env = os.environ.pop("GLM_SPEC_K", None)
    try:
        sys.modules["vllm"] = fake
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            engine.build_llm("stub-model")
        assert "speculative_config" not in calls[0]   # unset -> absent
        assert "spec=" not in out.getvalue()          # log line unchanged too

        os.environ["GLM_SPEC_K"] = "1"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            engine.build_llm("stub-model")
        assert calls[1]["speculative_config"] == {
            "method": "mtp", "num_speculative_tokens": 1}
        assert "spec=mtp:k=1" in out.getvalue()       # engine-built log line
        # every OTHER engine arg byte-identical to the unset build:
        assert {k: v for k, v in calls[1].items()
                if k != "speculative_config"} == calls[0]

        os.environ["GLM_SPEC_K"] = "5"                # runbook 7d: k=5
        with contextlib.redirect_stdout(io.StringIO()):
            engine.build_llm("stub-model")
        assert calls[2]["speculative_config"] == {
            "method": "mtp", "num_speculative_tokens": 5}

        os.environ["GLM_SPEC_K"] = "0"                # 0 = off = absent
        with contextlib.redirect_stdout(io.StringIO()):
            engine.build_llm("stub-model")
        assert "speculative_config" not in calls[3]
        assert calls[3] == calls[0]                   # byte-identical again

        os.environ["GLM_SPEC_K"] = "-3"               # negative: readable
        try:                                          # harness-boundary error,
            engine.build_llm("stub-model")            # not a pydantic trace
            raise AssertionError("GLM_SPEC_K=-3 should have raised")
        except ValueError as e:
            assert "GLM_SPEC_K" in str(e)

        os.environ["GLM_SPEC_K"] = "abc"              # garbage: readable too
        try:                                          # (round-9 review LOW-4)
            engine.build_llm("stub-model")
            raise AssertionError("GLM_SPEC_K=abc should have raised")
        except ValueError as e:
            assert "GLM_SPEC_K" in str(e) and "'abc'" in str(e)
        assert len(calls) == 4                        # LLM never constructed
    finally:
        if old_env is None:
            os.environ.pop("GLM_SPEC_K", None)
        else:
            os.environ["GLM_SPEC_K"] = old_env
        if had_vllm:
            sys.modules["vllm"] = old_mod
        else:
            sys.modules.pop("vllm", None)
    print("  MTP spec-decode engine kwarg plumbing OK")


def test_warn_worker_only_envs_classification():
    """The warn list's boolean-vs-value-carrying classification (review of the
    2026-07-17 safety commit: the list had needed two review-fix cycles with
    zero coverage). Boolean-gated names at '0'/'false' must NOT warn; armed
    booleans and any non-empty value-carrying value MUST; the two new names
    are worker-side booleans."""
    import engine as eng
    saved = {k: os.environ.pop(k, None) for k in eng._WORKER_SIDE_OBS_ENVS}
    try:
        assert eng.warn_worker_only_envs() == []
        assert "GLM_DSA_DCP_HEADSPLIT_UNSAFE" in eng._WORKER_SIDE_OBS_ENVS
        assert "GLM_WRITE_PROBE" in eng._WORKER_SIDE_OBS_ENVS
        assert "GLM_DSA_DCP_HEADSPLIT_UNSAFE" not in eng._VALUE_CARRYING_OBS_ENVS
        assert "GLM_WRITE_PROBE" not in eng._VALUE_CARRYING_OBS_ENVS
        # boolean-gated: explicit-off never warns, armed warns
        os.environ["GLM_WRITE_PROBE"] = "0"
        os.environ["GLM_DSA_DCP_HEADSPLIT_UNSAFE"] = "false"
        assert eng.warn_worker_only_envs() == []
        os.environ["GLM_WRITE_PROBE"] = "1"
        os.environ["GLM_DSA_DCP_HEADSPLIT_UNSAFE"] = "1"
        flagged = eng.warn_worker_only_envs()
        assert "GLM_WRITE_PROBE" in flagged
        assert "GLM_DSA_DCP_HEADSPLIT_UNSAFE" in flagged
        # value-carrying: ANY non-empty value warns, including "0"
        for k in ("GLM_WRITE_PROBE", "GLM_DSA_DCP_HEADSPLIT_UNSAFE"):
            del os.environ[k]
        os.environ["GLM_DCP_CACHE_DUMP"] = "0"
        assert "GLM_DCP_CACHE_DUMP" in eng.warn_worker_only_envs()
        del os.environ["GLM_DCP_CACHE_DUMP"]
    finally:
        for k in eng._WORKER_SIDE_OBS_ENVS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    print("warn_worker_only_envs classification OK")


def test_run_env_provenance_fields():
    """run_bench._run_env must actually RETURN (2026-07-08 fix: the audit
    commit assigned env['attention_path'] before `env` existed — a NameError
    that crashed EVERY run, stub included, at pv.start_run) and must carry
    attention_path (dense-mla default; dsa-sparse:<mode> under GLM_DSA_MODE)
    plus the GLM_* os_env sweep — GLM_DCP lands there with no extra code."""
    import argparse

    import run_bench as rb

    args = argparse.Namespace(
        model="STUB", stub=True, max_len=8192, max_new=64, max_seqs=8,
        max_batched_tokens=4096, batch_size=8, gmu=0.94, num_gpu_blocks=0,
        protocol="greedy", samples=1, seed=1234)
    saved = {k: os.environ.pop(k, None)
             for k in ("GLM_DSA_MODE", "GLM_DCP", "GLM_SPEC_K")}
    try:
        env = rb._run_env(args, ["gsm8k"])
        assert env["attention_path"] == "dense-mla"     # default: DSA bypassed
        assert "GLM_DCP" not in env["os_env"]
        assert "GLM_SPEC_K" not in env["os_env"]
        os.environ["GLM_DCP"] = "4"
        os.environ["GLM_DSA_MODE"] = "topk2048"
        os.environ["GLM_SPEC_K"] = "5"
        env = rb._run_env(args, ["gsm8k"])
        assert env["attention_path"] == "dsa-sparse:topk2048"
        assert env["os_env"]["GLM_DCP"] == "4"          # the GLM_* sweep
        assert env["os_env"]["GLM_DSA_MODE"] == "topk2048"
        assert env["os_env"]["GLM_SPEC_K"] == "5"       # runbook §7 (M2) —
        # the M2 comparison script (mtp_m2_check.py) reads THIS field to
        # orient which run was spec-on; no extra provenance code needed.
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("  run_env provenance fields (attention_path + GLM_DCP sweep) OK")


if __name__ == "__main__":
    test_extractors()
    test_adversarial_extractors()
    test_exact_answer_extractor()
    test_scorers()
    test_db_roundtrip()
    test_batched_run()
    test_card_protocol_specs()
    test_card_protocol_run()
    test_seed_provenance_gating()
    test_platform_seed_probe()
    test_item_builders()
    test_dcp_engine_kwarg()
    test_spec_engine_kwarg()
    test_run_env_provenance_fields()
    test_warn_worker_only_envs_classification()
    print("ALL bench CPU tests passed.")
