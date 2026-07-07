#!/usr/bin/env python3
"""CPU tests for bench/rescore.py — a SYNTHETIC provenance DB, never the real
results.db. Covers: flips up (the %-strip + unit-tail norm_math rules), flips
down, unscored(NULL)-transition labeling, no-flip stability, MC n_choices
reconstruction (contiguous-block anchoring), card-protocol extractor
selection, read-only safety of the analysis connection, row-identical
originals (and byte-identical DB in report-only mode), the write authorizer,
and the rescored_summary rows themselves."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile

import benchmarks as B
import provenance as pv
import rescore as rs


def _synthetic_db(path: str) -> tuple[int, int]:
    """Two runs of stored items whose `correct` verdicts were recorded by an
    OLD extractor (pre %-strip / unit-tail normalization)."""
    conn = pv.connect(path)
    run_gsm = pv.start_run(conn, model="SYNTH", env={}, note="synthetic gsm8k")
    # 1) old extractor scored '60\%' vs gold '60' as a MISS -> flips UP now
    pv.record_item(conn, run_gsm, benchmark="gsm8k", item_id="g_pct",
                   prompt="p", gold="60", raw_output=r"... \boxed{60\%}",
                   extracted=r"60\%", correct=False, score=0.0)
    # 2) old extractor scored '160 minutes' vs gold '160' as a MISS -> UP
    pv.record_item(conn, run_gsm, benchmark="gsm8k", item_id="g_unit",
                   prompt="p", gold="160", raw_output=r"\boxed{160 minutes}",
                   extracted="160 minutes", correct=False, score=0.0)
    # 3) stable hit — must NOT flip
    pv.record_item(conn, run_gsm, benchmark="gsm8k", item_id="g_hit",
                   prompt="p", gold="42", raw_output=r"\boxed{42}",
                   extracted="42", correct=True, score=1.0)
    # 4) stable miss (a genuinely wrong answer) — must NOT flip
    pv.record_item(conn, run_gsm, benchmark="gsm8k", item_id="g_miss",
                   prompt="p", gold="9", raw_output=r"\boxed{7}",
                   extracted="7", correct=False, score=0.0)
    # 5) recorded as a HIT but the stored reply does not support it (simulates
    #    a rule that tightened) -> flips DOWN
    pv.record_item(conn, run_gsm, benchmark="gsm8k", item_id="g_down",
                   prompt="p", gold="9", raw_output=r"\boxed{7}",
                   extracted="9", correct=True, score=1.0)
    # 6) recorded UNSCORED (correct NULL — extract/score raised at run time);
    #    now scorable -> must be labeled unscored->hit, NOT a miss->hit flip
    pv.record_item(conn, run_gsm, benchmark="gsm8k", item_id="g_null",
                   prompt="p", gold="5", raw_output=r"\boxed{5}",
                   extracted=None, correct=None, score=None)
    pv.finalize(conn, run_gsm, benchmark="gsm8k")

    run_mc = pv.start_run(conn, model="SYNTH", env={}, note="synthetic mc")
    mc_prompt = "Which?\n\nA. x\nB. y\nC. z\nD. w\n\nhint"
    # old extractor missed the letter (extracted None) -> flips UP now
    pv.record_item(conn, run_mc, benchmark="gpqa_diamond", item_id="m_up",
                   prompt=mc_prompt, gold="C", raw_output="Answer: C",
                   extracted=None, correct=False, score=0.0)
    # out-of-range letter must stay clamped by the RECONSTRUCTED n_choices=4
    pv.record_item(conn, run_mc, benchmark="gpqa_diamond", item_id="m_oor",
                   prompt=mc_prompt, gold="C", raw_output="Answer: E",
                   extracted=None, correct=False, score=0.0)
    pv.finalize(conn, run_mc, benchmark="gpqa_diamond")
    conn.close()
    return run_gsm, run_mc


def _dump(path: str, table: str):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
    finally:
        conn.close()


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_rescore_flips_and_immutability():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "synth.db")
        run_gsm, run_mc = _synthetic_db(db)
        before = {t: _dump(db, t) for t in ("runs", "items", "summary")}

        ro = rs.connect_ro(db)
        # -- GSM8K: 2 up (percent + unit tail), 1 down, 1 unscored->hit ----
        r = rs.rescore_run_benchmark(ro, run_gsm, "gsm8k")
        assert r["n"] == 6 and r["protocol"] == "greedy"
        assert r["old_value"] == 100.0 * 2 / 5     # NULL row outside the denom
        assert r["new_value"] == 100.0 * 4 / 6     # g_pct,g_unit,g_hit,g_null
        assert r["n_flip_up"] == 2 and r["n_flip_down"] == 1
        by_id = {f["item_id"]: f for f in r["flips"]}
        assert by_id["g_pct"]["flip"] == "miss->hit"
        assert by_id["g_pct"]["new_extracted"] == r"60\%"   # scoring flipped it
        assert by_id["g_unit"]["flip"] == "miss->hit"
        assert by_id["g_down"]["flip"] == "hit->miss"
        # the NULL row is reported, distinctly labeled, NOT in up/down counts
        assert by_id["g_null"]["flip"] == "unscored->hit"
        assert "g_hit" not in by_id and "g_miss" not in by_id

        # -- MC: n_choices reconstructed from the CONTIGUOUS choice block ---
        assert rs._n_choices_from_prompt("Q\n\nA. x\nB. y\nC. z\nD. w") == 4
        assert rs._n_choices_from_prompt("no choices here 5.") == 0
        # prose 'E. ...' NOT adjacent to the block must not inflate the count
        assert rs._n_choices_from_prompt(
            "Q says\nE. coli grows\n\nA. x\nB. y\nC. z\nD. w\n\nhint") == 4
        m = rs.rescore_run_benchmark(ro, run_mc, "gpqa_diamond")
        mby = {f["item_id"]: f for f in m["flips"]}
        assert mby["m_up"]["flip"] == "miss->hit"
        assert mby["m_up"]["new_extracted"] == "C"
        assert "m_oor" not in mby                        # 'E' clamped -> None

        # -- the analysis connection is physically read-only --------------
        try:
            ro.execute("UPDATE items SET correct=1")
            raise AssertionError("read-only connection accepted a write")
        except sqlite3.OperationalError:
            # the failed write leaves an open (read) transaction on ro; roll
            # it back so it cannot hold a shared lock against the writer below
            ro.rollback()

        # -- selection + unknown benchmarks -------------------------------
        assert rs.select_targets(ro, None, None) == [
            (run_gsm, "gsm8k"), (run_mc, "gpqa_diamond")]
        assert rs.select_targets(ro, [run_gsm], "gsm8k") == [(run_gsm, "gsm8k")]
        assert rs.rescore_run_benchmark(ro, run_gsm, "not_a_bench") is None

        # -- write-summary appends ONLY rescored_summary -------------------
        n = rs.write_rescored_summary(db, [r, m], harness_git="testgit",
                                      note="unit test")
        assert n == 2
        rows = _dump(db, "rescored_summary")
        assert len(rows) == 2
        pragma_conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cols = [c[1] for c in
                pragma_conn.execute("PRAGMA table_info(rescored_summary)")]
        pragma_conn.close()
        row = dict(zip(cols, rows[0]))
        assert (row["run_id"], row["benchmark"]) == (run_gsm, "gsm8k")
        assert row["protocol"] == "greedy"
        assert row["n_flip_up"] == 2 and row["n_flip_down"] == 1
        assert "g_pct" in row["flips_json"]

        # -- the write authorizer DENIES writes outside rescored_summary ---
        rw = sqlite3.connect(db)
        rw.set_authorizer(rs._rescore_authorizer)
        try:
            rw.execute("UPDATE items SET correct=0")
            raise AssertionError("authorizer let a write at items through")
        except sqlite3.DatabaseError:
            pass
        finally:
            rw.close()

        # -- originals row-identical after the whole rescore ----------------
        after = {t: _dump(db, t) for t in ("runs", "items", "summary")}
        assert after == before, "rescore mutated an original table"
        ro.close()
    print("  rescore flips + immutability OK")


def test_rescore_card_protocol():
    """A run recorded with protocol=card must be rescored with the CARD
    extractor: on a card-format reply the greedy extractor would grab the
    Confidence percentage ('95') and manufacture a false hit->miss flip."""
    card = getattr(B.AIME_2026, "card", None)
    if card is None or getattr(card, "extract", None) is None:
        print("  card-protocol rescore SKIPPED (registry predates card mode)")
        return
    reply = "Explanation: because.\nExact Answer: 42\nConfidence: 95%"
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "synth.db")
        conn = pv.connect(db)
        run_card = pv.start_run(conn, model="SYNTH",
                                env={"protocol": "card"}, note="synth card")
        pv.record_item(conn, run_card, benchmark="aime_2026", item_id="c_fmt",
                       prompt="p", gold="42", raw_output=reply,
                       extracted="42", correct=True, score=1.0)
        # same reply stored under a GREEDY run really was scored greedily
        # ('95' -> miss); the rescore must reproduce that too — per-run choice
        run_greedy = pv.start_run(conn, model="SYNTH",
                                  env={"protocol": "greedy"}, note="synth greedy")
        pv.record_item(conn, run_greedy, benchmark="aime_2026",
                       item_id="c_fmt", prompt="p", gold="42",
                       raw_output=reply, extracted="95", correct=False,
                       score=0.0)
        conn.close()

        ro = rs.connect_ro(db)
        assert rs.run_protocol(ro, run_card) == "card"
        assert rs.run_protocol(ro, run_greedy) == "greedy"
        rc = rs.rescore_run_benchmark(ro, run_card, "aime_2026")
        assert rc["protocol"] == "card"
        assert rc["flips"] == [], f"false card flip: {rc['flips']}"
        assert rc["details"][0]["new_extracted"] == "42"
        rg = rs.rescore_run_benchmark(ro, run_greedy, "aime_2026")
        assert rg["flips"] == [] and rg["details"][0]["new_extracted"] == "95"
        ro.close()
    print("  card-protocol rescore OK")


def test_rescore_cli_end_to_end():
    """main() over the synthetic DB: report-only leaves the DB BYTE-identical
    (no rescored_summary table); --write-summary creates/appends it."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "synth.db")
        _synthetic_db(db)
        sha_before = _sha(db)
        assert rs.main(["--db", db, "--all"]) == 0
        assert _sha(db) == sha_before, "report-only rescore changed the DB file"
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        assert conn.execute("SELECT name FROM sqlite_master WHERE "
                            "name='rescored_summary'").fetchone() is None
        conn.close()
        assert rs.main(["--db", db, "--all", "--benchmark", "gsm8k",
                        "--write-summary", "--note", "cli test"]) == 0
        rows = _dump(db, "rescored_summary")
        assert len(rows) == 1 and rows[0][2] == 1  # one row, run_id 1
    print("  rescore CLI end-to-end OK")


if __name__ == "__main__":
    test_rescore_flips_and_immutability()
    test_rescore_card_protocol()
    test_rescore_cli_end_to_end()
    print("ALL rescore CPU tests passed.")
