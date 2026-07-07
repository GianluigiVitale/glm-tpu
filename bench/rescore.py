#!/usr/bin/env python3
"""Offline rescore utility — re-run extraction+scoring over STORED outputs.

Re-runs the CURRENT extractors/scorers (bench/extract.py via the
bench/benchmarks.py registry) over the VERBATIM `items.raw_output` already
recorded in bench/results.db, and reports what would flip — WITHOUT mutating
any existing row. Use case: the extractor gains a rule (e.g. the 2026-07-07
%-stripping + unit-tail normalization in norm_math) and we want to quantify
how many past misses were extraction false-negatives, with full provenance.

Guarantees (enforced, not just promised):
  * The analysis connection is opened READ-ONLY (sqlite URI mode=ro) — the
    original runs/items/summary rows CANNOT be modified by this tool.
  * `--write-summary` uses a SEPARATE read-write connection carrying a sqlite
    AUTHORIZER that DENIES every write outside the rescored_summary table
    (append-only, same discipline as the provenance DB). Originals are never
    touched; a rescore is a new, timestamped record.
  * Scoring semantics mirror run_bench._score_and_record: extract/score
    inside try/except -> (None, None) on failure; accuracy is
    provenance.finalize's convention (percent over items whose correct is
    NOT NULL).
  * PROTOCOL-AWARE extraction: a run recorded with env_json protocol=card is
    rescored with the card extractor (spec.card.extract — e.g. the
    Exact-Answer field extractor) exactly as run_bench chose it. Rescoring a
    card reply with the greedy extractor would, e.g., grab the 'Confidence:
    95%' percentage as the answer — a false flip.

MC caveat: items.n_choices was never stored, so for multiple-choice
benchmarks it is reconstructed from the stored prompt (the longest run of
CONTIGUOUS "A. ...", "B. ..." lines that _mc_prompt wrote — anchored so a
question whose own text contains "E. something" cannot inflate the count).
If reconstruction fails (<2 letters found) the extractor falls back to the
full A-J range with a warning — it still never guesses (an explicit answer
letter is required), but an out-of-range letter would no longer be clamped
to None.

Usage:
    # report only (stdout), one run:
    JAX_PLATFORMS=cpu python3 rescore.py --run-id 28
    # all runs, GSM8K only, and record the rescore in rescored_summary:
    JAX_PLATFORMS=cpu python3 rescore.py --all --benchmark gsm8k --write-summary
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

import benchmarks as B
import extract as ex
import provenance as pv

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = pv.DEFAULT_DB

# Append-only rescore record. NEVER touches runs/items/summary — a rescore is
# a new fact ("with extractor X, run N would score Y"), not an edit of the old
# one. flips_json carries the per-item flip details verbatim for audit.
_RESCORED_SCHEMA = """
CREATE TABLE IF NOT EXISTS rescored_summary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    rescored_utc  TEXT NOT NULL,
    run_id        INTEGER NOT NULL,
    benchmark     TEXT NOT NULL,
    protocol      TEXT,
    harness_git   TEXT,
    n             INTEGER,
    old_value     REAL,
    new_value     REAL,
    n_flip_up     INTEGER,
    n_flip_down   INTEGER,
    flips_json    TEXT,
    note          TEXT
)"""
# Append-only migration for a rescored_summary created before a column
# existed (provenance._ITEM_MIGRATIONS pattern; never rewrites rows).
_RESCORED_MIGRATIONS = {
    "protocol": "ALTER TABLE rescored_summary ADD COLUMN protocol TEXT",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_ro(path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Open the provenance DB READ-ONLY. Any write through this connection
    raises sqlite3.OperationalError — the originals are physically safe."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _n_choices_from_prompt(prompt: str) -> int:
    """Reconstruct n_choices from a stored MC prompt: _mc_prompt writes the
    options as CONTIGUOUS 'A. ...', 'B. ...' lines. Returns the length of the
    longest run of ADJACENT lines whose letters follow A, B, C, ... in order
    (0 = no choice block, i.e. a math prompt). Anchoring to adjacent lines
    means a question whose own prose contains an 'E. something' line cannot
    inflate the count past the real choice block."""
    best = run = 0
    for line in (prompt or "").splitlines():
        m = re.match(r"([A-J])\.\s", line)
        if m and run < len(ex._LETTERS) and m.group(1) == ex._LETTERS[run]:
            run += 1
        elif m and m.group(1) == "A":
            run = 1                      # a new block starts over at 'A'
        else:
            run = 0
        best = max(best, run)
    return best


def _extract_fn(spec: B.BenchSpec, protocol: str):
    """The extractor run_bench would use for this (spec, protocol): the card
    extractor when the run was a card-protocol run and the spec defines one
    (run_bench passes extract_fn = spec.card.extract), spec.extract otherwise.
    getattr-guarded so this also works against a benchmarks.py predating the
    card-protocol machinery."""
    card = getattr(spec, "card", None)
    if protocol == "card":
        if card is None:
            print(f"[rescore] WARNING: {spec.name}: run was recorded with "
                  f"protocol=card but the current registry has no card "
                  f"protocol for it — rescoring with the greedy extractor",
                  file=sys.stderr)
        elif getattr(card, "extract", None) is not None:
            return card.extract
    return spec.extract


def rescore_reply(spec: B.BenchSpec, prompt: str, gold, raw_output: str,
                  protocol: str = "greedy"):
    """Re-run the CURRENT extractor + spec.score on one stored reply, using
    the extractor the run's protocol selects (see _extract_fn). Mirrors
    run_bench._score_and_record: any exception during extract/score yields
    (None, None) — never crash a rescore over one row."""
    n_choices = _n_choices_from_prompt(prompt)
    if spec.score is ex.score_mc and n_choices < 2:
        print(f"[rescore] WARNING: {spec.name}: could not reconstruct "
              f"n_choices from the stored prompt — falling back to "
              f"{len(ex._LETTERS)} (full A-J range)", file=sys.stderr)
        n_choices = len(ex._LETTERS)
    it = B.Item("rescore", prompt, prompt, gold, n_choices=n_choices)
    extract = _extract_fn(spec, protocol)
    try:
        extracted = extract(raw_output or "", it)
        correct = spec.score(extracted, it.gold)
    except Exception:
        extracted, correct = None, None
    return extracted, correct


def _acc(corrects) -> float | None:
    """provenance.finalize's convention: percent over non-NULL corrects."""
    scored = [c for c in corrects if c is not None]
    return 100.0 * sum(bool(c) for c in scored) / len(scored) if scored else None


def run_protocol(ro: sqlite3.Connection, run_id: int) -> str:
    """The protocol the run was recorded under (runs.env_json 'protocol');
    'greedy' for runs predating protocol support or unparseable env_json."""
    rows = ro.execute("SELECT env_json FROM runs WHERE run_id=?",
                      (run_id,)).fetchall()
    try:
        return json.loads(rows[0][0] or "{}").get("protocol", "greedy") \
            if rows else "greedy"
    except (ValueError, TypeError):
        return "greedy"


def _verdict(v) -> str:
    """A stored/computed correct value as a word: NULL correct means the item
    was never scored (extract/score raised) — NOT a miss, so a NULL->scored
    transition must not masquerade as a hit->miss/miss->hit flip."""
    return "hit" if v is True else ("miss" if v is False else "unscored")


def rescore_run_benchmark(ro: sqlite3.Connection, run_id: int,
                          benchmark: str) -> dict | None:
    """Rescore every stored item of (run_id, benchmark) with the CURRENT
    extractor (protocol-aware). Returns the report dict (None if the
    benchmark is unknown): old/new accuracy, flip lists, per-item details."""
    spec = B.REGISTRY.get(benchmark)
    if spec is None:
        print(f"[rescore] WARNING: run {run_id}: benchmark {benchmark!r} not "
              f"in the registry ({list(B.REGISTRY)}) — skipped",
              file=sys.stderr)
        return None
    protocol = run_protocol(ro, run_id)
    rows = ro.execute(
        "SELECT id, item_id, prompt, gold, raw_output, extracted, correct"
        " FROM items WHERE run_id=? AND benchmark=? ORDER BY id",
        (run_id, benchmark)).fetchall()
    details, flips, n_extract_changed = [], [], 0
    for _rowid, item_id, prompt, gold, raw, old_extracted, old_correct in rows:
        old_correct = None if old_correct is None else bool(old_correct)
        new_extracted, new_correct = rescore_reply(spec, prompt, gold, raw,
                                                   protocol=protocol)
        d = {"item_id": item_id, "gold": gold,
             "old_extracted": old_extracted, "new_extracted": new_extracted,
             "old_correct": old_correct, "new_correct": new_correct}
        details.append(d)
        if old_correct is not new_correct:
            d["flip"] = f"{_verdict(old_correct)}->{_verdict(new_correct)}"
            flips.append(d)
        elif new_extracted != old_extracted:
            n_extract_changed += 1
    return {
        "run_id": run_id, "benchmark": benchmark, "protocol": protocol,
        "n": len(rows),
        "old_value": _acc(d["old_correct"] for d in details),
        "new_value": _acc(d["new_correct"] for d in details),
        "flips": flips,
        # up/down count REAL verdict flips only (miss<->hit between scored
        # items); unscored(NULL) transitions are in the flips list with their
        # own labels but never inflate these counters.
        "n_flip_up": sum(1 for f in flips if f["flip"] == "miss->hit"),
        "n_flip_down": sum(1 for f in flips if f["flip"] == "hit->miss"),
        "n_extract_changed_same_score": n_extract_changed,
        "details": details,
    }


def select_targets(ro: sqlite3.Connection, run_ids: list[int] | None,
                   benchmark: str | None) -> list[tuple[int, str]]:
    """(run_id, benchmark) pairs that actually have items, in run order.
    run_ids=None means ALL runs with items."""
    q = "SELECT DISTINCT run_id, benchmark FROM items"
    where, args = [], []
    if run_ids:
        where.append(f"run_id IN ({','.join('?' * len(run_ids))})")
        args += run_ids
    if benchmark:
        where.append("benchmark=?")
        args.append(benchmark)
    if where:
        q += " WHERE " + " AND ".join(where)
    return ro.execute(q + " ORDER BY run_id, benchmark", args).fetchall()


def run_meta(ro: sqlite3.Connection, run_id: int) -> tuple[str, str]:
    # fetchall (not fetchone): an unexhausted SELECT statement keeps a shared
    # read lock on the DB, which would block the --write-summary writer.
    rows = ro.execute("SELECT harness_git, note FROM runs WHERE run_id=?",
                      (run_id,)).fetchall()
    return (rows[0][0] or "?", rows[0][1] or "") if rows else ("?", "")


def _rescore_authorizer(action, arg1, arg2, dbname, source):
    """sqlite authorizer for the --write-summary connection: reads are free;
    writes are allowed ONLY for rescored_summary (+ the sqlite-internal
    bookkeeping its CREATE/AUTOINCREMENT require). Anything that would touch
    runs/items/summary is DENIED at the engine level — the append-only
    guarantee is mechanical, not a promise."""
    ok, deny = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                  sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_FUNCTION):
        return ok
    allowed = ("rescored_summary", "sqlite_sequence", "sqlite_master")
    if action == sqlite3.SQLITE_INSERT and arg1 in allowed:
        return ok
    if action == sqlite3.SQLITE_UPDATE and arg1 in ("sqlite_sequence",
                                                    "sqlite_master"):
        return ok
    if action == sqlite3.SQLITE_CREATE_TABLE and arg1 in (
            "rescored_summary", "sqlite_sequence"):  # AUTOINCREMENT bookkeeping
        return ok
    if action == sqlite3.SQLITE_ALTER_TABLE and arg2 == "rescored_summary":
        return ok
    if action == sqlite3.SQLITE_PRAGMA and arg1 == "table_info":
        return ok
    return deny


def write_rescored_summary(db_path: str, results: list[dict],
                           harness_git: str, note: str = "") -> int:
    """Append the rescore records to rescored_summary via a separate
    read-write connection whose authorizer (above) denies every write outside
    that table. timeout=30 rides out a concurrent engine run writing its own
    items rows."""
    rw = sqlite3.connect(db_path, timeout=30.0)
    try:
        rw.set_authorizer(_rescore_authorizer)
        rw.execute(_RESCORED_SCHEMA)
        have = {r[1] for r in rw.execute("PRAGMA table_info(rescored_summary)")}
        for col, ddl in _RESCORED_MIGRATIONS.items():
            if col not in have:
                rw.execute(ddl)
        for r in results:
            rw.execute(
                "INSERT INTO rescored_summary(rescored_utc, run_id, benchmark,"
                " protocol, harness_git, n, old_value, new_value, n_flip_up,"
                " n_flip_down, flips_json, note)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (_utc(), r["run_id"], r["benchmark"],
                 r.get("protocol", "greedy"), harness_git, r["n"],
                 r["old_value"], r["new_value"], r["n_flip_up"],
                 r["n_flip_down"], json.dumps(r["flips"]), note))
        rw.commit()
        return len(results)
    finally:
        rw.close()


def extractor_git() -> str:
    """glm-tpu short HEAD hash, '-dirty'-suffixed when the bench/ tree has
    uncommitted changes — the whole point of a rescore is usually a JUST-
    EDITED extractor, and attributing that to a clean commit would lie."""
    h = pv._git(HERE)
    try:
        dirty = subprocess.check_output(
            ["git", "-C", HERE, "status", "--porcelain", "--", "."],
            stderr=subprocess.DEVNULL, text=True).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return h


def _fmt(v) -> str:
    return "  -  " if v is None else f"{v:5.1f}"


def print_report(ro: sqlite3.Connection, results: list[dict]) -> None:
    print(f"run  benchmark     proto   n  old_acc  new_acc     Δ  up  down  "
          f"(extractor: glm-tpu {extractor_git()})")
    for r in results:
        delta = (None if r["old_value"] is None or r["new_value"] is None
                 else r["new_value"] - r["old_value"])
        print(f"{r['run_id']:>3}  {r['benchmark']:<12} "
              f"{r.get('protocol', 'greedy'):<6} {r['n']:>3}  "
              f"{_fmt(r['old_value'])}    {_fmt(r['new_value'])} "
              f"{_fmt(delta)}  {r['n_flip_up']:>2}  {r['n_flip_down']:>4}")
    print()
    any_flip = False
    for r in results:
        for f in r["flips"]:
            any_flip = True
            git, note = run_meta(ro, r["run_id"])
            print(f"FLIP {f['flip']:<14} run {r['run_id']} "
                  f"[{r['benchmark']}] {f['item_id']}: extracted "
                  f"{f['old_extracted']!r} -> {f['new_extracted']!r}  "
                  f"gold {f['gold']!r}  (run: {note or git})")
    if not any_flip:
        print("no flips — every stored verdict matches the current extractor.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--run-id", type=int, action="append", default=None,
                    help="run to rescore (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="rescore every run that has items")
    ap.add_argument("--benchmark", default=None,
                    help="restrict to one benchmark (default: all present)")
    ap.add_argument("--write-summary", action="store_true",
                    help="append the rescore to the rescored_summary table "
                         "(new table; originals are NEVER touched)")
    ap.add_argument("--note", default="", help="note stored with --write-summary")
    args = ap.parse_args(argv)
    if not args.all and not args.run_id:
        ap.error("provide --run-id N (repeatable) or --all")

    ro = connect_ro(args.db)
    targets = select_targets(ro, None if args.all else args.run_id,
                             args.benchmark)
    if not targets:
        print("[rescore] nothing to rescore (no matching items).")
        return 1
    results = [r for run_id, bench in targets
               if (r := rescore_run_benchmark(ro, run_id, bench)) is not None]
    print_report(ro, results)
    # Close the read-only connection BEFORE writing: a lingering read
    # transaction on it would block the writer's commit (database is locked).
    ro.close()
    if args.write_summary:
        n = write_rescored_summary(args.db, results, extractor_git(),
                                   args.note)
        print(f"\n[rescore] {n} rescored_summary row(s) appended "
              f"(originals untouched).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
