#!/usr/bin/env python3
"""CPU tests for bench/merge_runs.py — a SYNTHETIC provenance DB, never the
real results.db. Covers: retry-overrides-base per item_id, later-retry-wins
ordering, in-run-duplicate latest-row-wins, retry-only inclusion + warning,
gold/config mismatch warnings, merged acc + n_truncated recomputed AFTER the
merge, the merged_summary row (labeled with the source run ids), physical
immutability of the originals (read-only analysis connection + the write
authorizer), benchmark inference, and the CLI round-trip.

Run: JAX_PLATFORMS=cpu python3 -m pytest bench/test_merge_runs.py -q
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sqlite3
import tempfile

import provenance as pv
import merge_runs as mr


def _rec(conn, rid, iid, correct, truncated, gold="G", bench="gpqa_diamond"):
    pv.record_item(conn, rid, benchmark=bench, item_id=iid, prompt="p",
                   gold=gold, raw_output="r", extracted=None, correct=correct,
                   score=None if correct is None else float(bool(correct)),
                   finish_reason=(None if truncated is None else
                                  ("length" if truncated else "stop")),
                   truncated=truncated)


def _mk_db(path: str) -> tuple[int, int, int]:
    """base: 6 gpqa items (2 truncated misses, 1 unscored, 1 in-run duplicate)
    + 1 gsm8k item (multi-benchmark, so inference from base must refuse);
    retry1 (@64K): fixes g1, g3 still truncated;
    retry2 (@128K): fixes g3 (later retry wins), overrides g5 with a DIFFERENT
    gold (mismatch warning), a mismatched attention_path AND samples (config
    warnings), plus a retry-only id g9."""
    conn = pv.connect(path)
    env = {"model": "M", "protocol": "greedy", "attention_path": "dense-mla",
           "temperature": 0.0, "top_p": 1.0, "max_new": 16384, "ids": None,
           "samples": 1}
    base = pv.start_run(conn, model="M", env=env, note="base gpqa @16K")
    _rec(conn, base, "g0", True, 0)
    _rec(conn, base, "g1", False, 1)          # truncated miss -> retried
    _rec(conn, base, "g2", False, 0)          # stale in-run duplicate...
    _rec(conn, base, "g2", True, 0)           # ...the LATEST row wins
    _rec(conn, base, "g3", False, 1)          # truncated miss -> retried twice
    _rec(conn, base, "g4", None, None)        # unscored (extract raised)
    _rec(conn, base, "g5", False, 0)          # honest miss, never retried
    _rec(conn, base, "x0", True, 0, bench="gsm8k")   # second benchmark
    pv.finalize(conn, base, benchmark="gpqa_diamond")

    r1 = pv.start_run(conn, model="M",
                      env=dict(env, max_new=65536, ids="g1,g3"),
                      note="retry @64K")
    _rec(conn, r1, "g1", True, 0)
    _rec(conn, r1, "g3", False, 1)            # still truncated at 64K
    pv.finalize(conn, r1, benchmark="gpqa_diamond")

    r2 = pv.start_run(conn, model="M",
                      env=dict(env, max_new=131072, ids="g3",
                               attention_path="dsa-sparse:topk2048",
                               samples=2),
                      note="retry @128K")
    _rec(conn, r2, "g3", True, 0)             # resolved -> later retry wins
    _rec(conn, r2, "g5", False, 0, gold="H")  # gold drift -> loud warning
    _rec(conn, r2, "g9", True, 0)             # retry-only id -> included+warned
    pv.finalize(conn, r2, benchmark="gpqa_diamond")
    conn.close()
    return base, r1, r2


def _dump(path: str, table: str):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
    finally:
        conn.close()


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_merge_semantics():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "synth.db")
        base, r1, r2 = _mk_db(db)
        ro = mr.connect_ro(db)

        m = mr.merge_runs(ro, base, [r1, r2], "gpqa_diamond")
        # sources: retry rows override base rows per item_id; later retry wins
        assert m["sources"] == {"g0": base, "g1": r1, "g2": base, "g3": r2,
                                "g4": base, "g5": r2, "g9": r2}
        by_id = {r["item_id"]: r for r in m["rows"]}
        assert by_id["g2"]["correct"] == 1          # in-run latest row won
        assert by_id["g3"]["correct"] == 1 and by_id["g3"]["truncated"] == 0
        # merged acc: scored = g0,g1,g2,g3,g5,g9 (g4 stays unscored/NULL);
        # hits = g0,g1,g2,g3,g9 -> 5/6
        assert m["n"] == 7 and m["n_scored"] == 6
        assert abs(m["value"] - 500.0 / 6) < 1e-9
        assert m["card_value"] == 91.2
        assert abs(m["delta"] - round(500.0 / 6 - 91.2, 2)) < 1e-9
        # n_truncated recomputed AFTER the merge (base had 2, both resolved)
        assert m["n_truncated"] == 0
        assert m["n_overridden"] == 3               # g1, g3, g5 (base rows)
        assert m["n_retry_only"] == 1               # g9
        # base order preserved, retry-only appended
        assert [r["item_id"] for r in m["rows"]] == \
            ["g0", "g1", "g2", "g3", "g4", "g5", "g9"]
        # warnings: retry-only id, gold drift, config mismatches (incl. the
        # card-samples key — '#sN' vs plain item_ids can never match) — loud
        w = "\n".join(m["warnings"])
        assert "'g9'" in w and "NOT in base" in w
        assert "gold mismatch on 'g5'" in w
        assert f"run {base} 'G'" in w            # names the overridden source
        assert "config mismatch attention_path" in w
        assert "config mismatch samples" in w
        assert len(m["warnings"]) == 4
        assert f"{mr.merge_label(m)}" == \
            f"merge gpqa_diamond: base=run{base} retries=" \
            f"{['run%d' % r1, 'run%d' % r2]}"

        # RETRY ORDER MATTERS: [r2, r1] means r1 re-overrides g3 (still
        # truncated) — later-listed retries win
        m_rev = mr.merge_runs(ro, base, [r2, r1], "gpqa_diamond")
        assert m_rev["sources"]["g3"] == r1
        assert m_rev["n_truncated"] == 1
        assert abs(m_rev["value"] - 400.0 / 6) < 1e-9

        # benchmark inference: refused on the multi-benchmark base, inferred
        # on a single-benchmark base (r1 as base)
        try:
            mr.merge_runs(ro, base, [r1])
            raise AssertionError("multi-benchmark base must demand --benchmark")
        except ValueError as e:
            assert "--benchmark" in str(e)
        assert mr.merge_runs(ro, r1, [r2])["benchmark"] == "gpqa_diamond"

        # a retry with no items for the benchmark is refused
        try:
            mr.merge_runs(ro, base, [r1], "gsm8k")
            raise AssertionError("retry without the benchmark must be refused")
        except ValueError as e:
            assert "no 'gsm8k' items" in str(e)
        # base==retry / duplicate retries are refused
        for retries in ([base], [r1, r1]):
            try:
                mr.merge_runs(ro, base, retries, "gpqa_diamond")
                raise AssertionError(f"retries={retries} must be refused")
            except ValueError as e:
                assert "distinct" in str(e)
        ro.close()
    print("  merge semantics (override/order/acc/truncation/warnings) OK")


def test_merged_summary_write_and_immutability():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "synth.db")
        base, r1, r2 = _mk_db(db)
        before = {t: _dump(db, t) for t in ("runs", "items", "summary")}
        sha_before = _sha(db)

        # --dry-run: report only, byte-identical DB
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = mr.main(["--db", db, "--base", str(base),
                          "--retry", str(r1), "--retry", str(r2),
                          "--benchmark", "gpqa_diamond", "--dry-run"])
        assert rc == 0 and _sha(db) == sha_before
        text = out.getvalue()
        assert "merged acc=83.33" in text
        assert "n_truncated=0 (after merge)" in text
        assert f"retry:{r2}" in text and f"base:{base}" in text  # per-item source
        assert "WARNING" in text

        # real write: ONE merged_summary row, originals row-identical
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = mr.main(["--db", db, "--base", str(base),
                          "--retry", str(r1), "--retry", str(r2),
                          "--benchmark", "gpqa_diamond", "--note", "unit"])
        assert rc == 0 and "merged_summary row 1 appended" in out.getvalue()
        for t in ("runs", "items", "summary"):
            assert _dump(db, t) == before[t], f"{t} was mutated"
        rows = _dump(db, "merged_summary")
        assert len(rows) == 1
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM merged_summary").fetchone()
        conn.close()
        assert row["benchmark"] == "gpqa_diamond"
        assert row["base_run_id"] == base
        assert json.loads(row["retry_run_ids"]) == [r1, r2]
        assert f"base=run{base}" in row["label"]      # labeled with source ids
        assert f"run{r1}" in row["label"] and f"run{r2}" in row["label"]
        assert row["n"] == 7 and row["n_scored"] == 6
        assert abs(row["value"] - 500.0 / 6) < 1e-9
        assert row["metric"] == "acc" and row["card_value"] == 91.2
        assert row["n_overridden"] == 3 and row["n_retry_only"] == 1
        assert row["n_truncated"] == 0
        # the WARNINGS are persisted in the durable row, not stdout-only
        assert row["note"].startswith("unit | 4 WARNING(s): "), row["note"]
        assert "gold mismatch on 'g5'" in row["note"]
        assert "config mismatch samples" in row["note"]
        srcs = json.loads(row["sources_json"])
        assert srcs["g3"] == r2 and srcs["g0"] == base

        # the analysis connection is physically read-only
        ro = mr.connect_ro(db)
        try:
            ro.execute("UPDATE items SET correct=0")
            raise AssertionError("read-only connection accepted a write")
        except sqlite3.OperationalError:
            pass
        ro.close()
        # the writer's authorizer DENIES any write outside merged_summary
        rw = sqlite3.connect(db)
        rw.set_authorizer(mr._merged_authorizer)
        for stmt in ("UPDATE items SET correct=0",
                     "DELETE FROM summary",
                     "INSERT INTO runs(created_utc,model) VALUES ('t','m')",
                     "UPDATE merged_summary SET value=100.0",
                     "DROP TABLE merged_summary",
                     # review finding 8: no sqlite_sequence allowance at all
                     # (merged_summary has no AUTOINCREMENT)
                     "UPDATE sqlite_sequence SET seq=0 WHERE name='items'",
                     "INSERT INTO sqlite_sequence(name,seq) VALUES ('x',1)"):
            try:
                rw.execute(stmt)
                raise AssertionError(f"authorizer allowed: {stmt}")
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                pass
        rw.close()
    print("  merged_summary write + originals immutability OK")


if __name__ == "__main__":
    test_merge_semantics()
    test_merged_summary_write_and_immutability()
    print("ALL merge_runs CPU tests passed.")
