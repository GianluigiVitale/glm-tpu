#!/usr/bin/env python3
"""CPU tests for bench/report_passkey.py — SYNTHETIC passkey rows (stub), never
the real results.db. Covers: the passkey_L{L}_d{d} cell parser, per-cell
recomputation from item rows (finalize's non-NULL-correct convention), the
length x depth matrix + marginals (the '99% shallow / 60% deep is a failure
wearing an average' breakdown), the >=95%-per-cell gate verdict, unscored
cells, attention_path in the header, run auto-selection, and the exit codes
(0 pass / 2 gate-fail / 1 no data).

Run: JAX_PLATFORMS=cpu python3 -m pytest bench/test_report_passkey.py -q
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile

import provenance as pv
import report_passkey as rp


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _cell(conn, rid, L, d, hits, n=3):
    bench = f"passkey_L{L}_d{d}"
    for t in range(n):
        hit = t < hits
        pv.record_item(conn, rid, benchmark=bench, item_id=f"t{t}", prompt="p",
                       gold="123456", raw_output=("123456" if hit else ""),
                       extracted=("123456" if hit else None), correct=hit,
                       score=float(hit), seed=1000 + t)
    pv.finalize(conn, rid, benchmark=bench, metric="acc")


def _mk_db(path: str) -> tuple[int, int]:
    conn = pv.connect(path)
    # run A: shallow depths perfect, the DEEP 128K cell failing — exactly the
    # failure mode the audit says an average would hide
    rid_a = pv.start_run(conn, model="M", note="ladder A",
                         env={"attention_path": "dense-mla",
                              "prompt_mode": "raw_completion", "trials": 3})
    _cell(conn, rid_a, 1024, 0.25, hits=3)
    _cell(conn, rid_a, 1024, 0.75, hits=3)
    _cell(conn, rid_a, 131072, 0.25, hits=3)
    _cell(conn, rid_a, 131072, 0.75, hits=1)          # 33.3% deep
    # one UNSCORED trial (correct NULL) in its own cell: never counts as a pass
    pv.record_item(conn, rid_a, benchmark="passkey_L262144_d0.5", item_id="t0",
                   prompt="p", gold="123456", raw_output="x", extracted=None,
                   correct=None, score=None)
    # a non-passkey row in the same run must be ignored by the report
    pv.record_item(conn, rid_a, benchmark="gsm8k", item_id="g0", prompt="p",
                   gold="1", raw_output="1", extracted="1", correct=True,
                   score=1.0)
    # run C: every cell perfect but the ladder stops at 1024 — the per-cell
    # gate holds, the >=128K COVERAGE does not (must exit 2, review finding 5)
    rid_c = pv.start_run(conn, model="M", note="ladder C (short)",
                         env={"attention_path": "dense-mla",
                              "prompt_mode": "raw_completion", "trials": 3})
    for d in (0.25, 0.75):
        _cell(conn, rid_c, 1024, d, hits=3)
    # run B (latest): every cell perfect out to 128K, on the DSA path
    rid_b = pv.start_run(conn, model="M", note="ladder B",
                         env={"attention_path": "dsa-sparse:topk2048",
                              "prompt_mode": "raw_completion", "trials": 3})
    for L in (1024, 131072):
        for d in (0.25, 0.75):
            _cell(conn, rid_b, L, d, hits=3)
    conn.close()
    return rid_a, rid_c, rid_b


def test_cell_parser():
    assert rp.parse_cell("passkey_L131072_d0.75") == (131072, 0.75)
    assert rp.parse_cell("passkey_L1024_d0.5") == (1024, 0.5)
    assert rp.parse_cell("gsm8k") is None
    assert rp.parse_cell("passkey_L10_dxx") is None
    assert rp.parse_cell("longctx_passkey") is None    # the aggregate row
    assert rp.parse_cell("") is None and rp.parse_cell(None) is None
    print("  cell parser OK")


def test_cells_matrix_and_verdict():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "synth.db")
        rid_a, _rid_c, rid_b = _mk_db(db)
        conn = rp.connect_ro(db)

        cells = rp.cells_from_items(conn, rid_a)
        assert set(cells) == {(1024, 0.25), (1024, 0.75), (131072, 0.25),
                              (131072, 0.75), (262144, 0.5)}
        assert cells[(1024, 0.25)] == {"n": 3, "n_scored": 3, "correct": 3,
                                       "acc": 100.0}
        assert abs(cells[(131072, 0.75)]["acc"] - 100.0 / 3) < 1e-9
        assert cells[(262144, 0.5)]["acc"] is None     # unscored cell
        # gsm8k rows ignored: total passkey n
        assert sum(c["n"] for c in cells.values()) == 13

        # matrix: per-cell acc (n) with '!' on failing cells + marginals that
        # EXPOSE the deep-depth failure the overall average hides
        text = rp.format_matrix(cells, gate=95.0)
        assert "d=0.25" in text and "d=0.75" in text and "d=0.5" in text
        assert "L=1024" in text and "L=131072" in text
        assert "33.3% (3)!" in text                    # the failing deep cell
        assert "100.0% (3) " in text                   # a passing cell
        assert "all-d" in text and "all-L" in text
        # per-depth marginal: d=0.75 over both lengths = 4/6 = 66.7 (failing)
        assert "66.7% (6)!" in text
        # per-depth marginal: d=0.25 = 6/6 = 100.0 (passing)
        assert "100.0% (6) " in text
        # overall = 10/12 = 83.3 — 'a failure wearing an average'
        assert "83.3% (12)!" in text

        ok, failing = rp.verdict(cells, gate=95.0)
        assert not ok and len(failing) == 2
        assert any("L=131072 d=0.75" in f and "33.3%" in f for f in failing)
        assert any("L=262144 d=0.5" in f and "no scored trials" in f
                   for f in failing)

        # run B: every cell 100% -> PASS
        ok_b, failing_b = rp.verdict(rp.cells_from_items(conn, rid_b), 95.0)
        assert ok_b and failing_b == []
        conn.close()
    print("  cells/matrix/verdict (per-depth breakdown) OK")


def _main(argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = rp.main(argv)
    return rc, out.getvalue()


def test_main_cli_and_exit_codes():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "synth.db")
        rid_a, rid_c, rid_b = _mk_db(db)
        sha_before = _sha(db)

        # default = the LATEST passkey run (B) -> PASS, exit 0, DSA path shown
        rc, text = _main(["--db", db])
        assert rc == 0 and f"run {rid_b}:" in text
        assert "attention_path=dsa-sparse:topk2048" in text
        assert "GATE (>= 95% per cell): PASS" in text
        assert "max length 131072 >= 128K" in text

        # pinned run A -> FAIL, exit 2, failing cells listed, dense path shown
        rc, text = _main(["--db", db, "--run-id", str(rid_a)])
        assert rc == 2 and f"run {rid_a}:" in text
        assert "attention_path=dense-mla" in text
        assert "prompt_mode=raw_completion" in text
        assert "GATE (>= 95% per cell): FAIL" in text
        assert "L=131072 d=0.75" in text and "33.3%" in text
        assert "no scored trials" in text

        # run C: per-cell gate PASSES but the ladder stops at 1024 — the
        # COVERAGE shortfall must exit 2 (a 4K smoke ladder never gates
        # Stage-2->Stage-3; review finding 5)
        rc, text = _main(["--db", db, "--run-id", str(rid_c)])
        assert rc == 2
        assert "GATE (>= 95% per cell): PASS" in text
        assert "does not reach" in text and "max length 1024 < 128K" in text

        # a lower --gate flips run A's 33% cell but not the unscored one
        rc, text = _main(["--db", db, "--run-id", str(rid_a), "--gate", "30"])
        assert rc == 2 and "no scored trials" in text

        # unknown run id -> exit 1 with the available runs listed
        rc, text = _main(["--db", db, "--run-id", "999"])
        assert rc == 1 and str(rid_a) in text

        # nonexistent DB path -> exit 1 and NO file is created (the report
        # opens read-only; review finding 4)
        missing = os.path.join(td, "typo.db")
        rc, text = _main(["--db", missing])
        assert rc == 1 and "no such provenance DB" in text
        assert not os.path.exists(missing)

        # the report NEVER writes the DB it reads (mode=ro, no migrations)
        assert _sha(db) == sha_before

        # a DB whose rows LIKE-match but parse to no cells -> exit 1, not a
        # traceback; and the escaped LIKE must not select 'passkeyXL...' rows
        db2 = os.path.join(td, "weird.db")
        conn = pv.connect(db2)
        r_broken = pv.start_run(conn, model="M", env={}, note="broken name")
        pv.record_item(conn, r_broken, benchmark="passkey_Lbroken_dxx",
                       item_id="t0", prompt="p", gold="1", raw_output="",
                       extracted=None, correct=False, score=0.0)
        r_bait = pv.start_run(conn, model="M", env={}, note="LIKE bait")
        pv.record_item(conn, r_bait, benchmark="passkeyXL1024_d0.5",
                       item_id="t0", prompt="p", gold="1", raw_output="",
                       extracted=None, correct=False, score=0.0)
        conn.close()
        rc, text = _main(["--db", db2, "--run-id", str(r_broken)])
        assert rc == 1 and "no parseable" in text
        rc, text = _main(["--db", db2, "--run-id", str(r_bait)])
        assert rc == 1 and "no passkey items" in text   # the `_` was escaped
    print("  report CLI + exit codes (0 pass / 2 fail-or-short / 1 no data) OK")


if __name__ == "__main__":
    test_cell_parser()
    test_cells_matrix_and_verdict()
    test_main_cli_and_exit_codes()
    print("ALL report_passkey CPU tests passed.")
