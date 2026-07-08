#!/usr/bin/env python3
"""Merge a BASE benchmark run with one or more RETRY runs — the companion of
run_bench.py --ids (retry a truncated tail at a longer --max-new, then fold
the retry verdicts back into the base run's numbers, with provenance).

MERGE SEMANTICS (append-only, originals NEVER mutated):
  * All runs must carry items for the SAME benchmark (inferred from the base
    run when it has exactly one; --benchmark pins it otherwise).
  * Start from the base run's item rows; for each retry run IN THE ORDER
    GIVEN, a retry row REPLACES the base (or earlier-retry) row with the same
    item_id — later retries win. Within one run, the LATEST row per item_id
    wins (max items.id), so an accidental in-run duplicate cannot double-count.
  * Retry-only item_ids (present in a retry but not the base) are INCLUDED
    and loudly flagged — a retry of a truncated tail should be a subset of
    the base; anything else is reported, not hidden.
  * Merged accuracy follows provenance.finalize's convention: percent over
    merged rows whose `correct` is NOT NULL. n_truncated is recounted over
    the MERGED rows (the whole point: a retried item's truncated=1 base row
    is superseded by its retry row).
  * The result is written as ONE row in a separate `merged_summary` table,
    labeled with the source run ids (base_run_id, retry_run_ids JSON, and a
    human label) plus the per-item source map (sources_json). The write goes
    through a connection whose sqlite AUTHORIZER denies every write outside
    merged_summary — runs/items/summary are physically untouchable (the
    rescore.py discipline). --dry-run prints without writing.

Comparability: model / protocol / attention_path / temperature / top_p /
samples are compared across the runs and any mismatch printed as a loud
WARNING (a merge across different serving numerics is not one result; a
samples mismatch means '#sN'-suffixed vs plain item_ids that can never
match). ALL warnings are also persisted into the merged_summary note — the
durable row never looks cleaner than the merge was. max_new is printed per
run — it is EXPECTED to differ (that is why the retry exists).

Usage:
    # merge base run 12 with retry runs 15 then 16 (16 wins on overlap):
    JAX_PLATFORMS=cpu python3 merge_runs.py --base 12 --retry 15 --retry 16
    # report only, no merged_summary row:
    JAX_PLATFORMS=cpu python3 merge_runs.py --base 12 --retry 15 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance as pv

DEFAULT_DB = pv.DEFAULT_DB

# env_json keys that must MATCH for a merge to be one result; max_new is
# deliberately absent (the retry's longer cap is the point of the merge).
# `samples` IS present: a base card run with --samples 2 stores 'a0#s0/#s1'
# rows while a samples=1 retry stores plain 'a0' — the item_ids then never
# match, the retry rows enter as retry-only, and the merged number is wrong;
# the samples-mismatch warning names that cause (review 2026-07-08 finding 3).
_COMPARABILITY_KEYS = ("model", "protocol", "attention_path",
                       "temperature", "top_p", "samples")

# Plain INTEGER PRIMARY KEY (no AUTOINCREMENT): append-only semantics are
# unchanged (nothing may DELETE here), and it keeps sqlite_sequence out of
# the writer authorizer's allow-list entirely (review finding 8).
_MERGED_SCHEMA = """
CREATE TABLE IF NOT EXISTS merged_summary (
    id            INTEGER PRIMARY KEY,
    created_utc   TEXT NOT NULL,
    benchmark     TEXT NOT NULL,
    base_run_id   INTEGER NOT NULL,
    retry_run_ids TEXT NOT NULL,
    label         TEXT NOT NULL,
    n             INTEGER,
    n_scored      INTEGER,
    metric        TEXT,
    value         REAL,
    card_value    REAL,
    delta         REAL,
    n_overridden  INTEGER,
    n_retry_only  INTEGER,
    n_truncated   INTEGER,
    sources_json  TEXT,
    note          TEXT
)"""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_ro(path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Open the provenance DB READ-ONLY — the analysis side of the merge can
    physically never modify the originals."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def run_env(ro: sqlite3.Connection, run_id: int) -> dict:
    rows = ro.execute("SELECT env_json FROM runs WHERE run_id=?",
                      (run_id,)).fetchall()
    if not rows:
        raise ValueError(f"run_id {run_id} not found")
    try:
        return json.loads(rows[0][0] or "{}")
    except ValueError:
        return {}


def run_benchmarks(ro: sqlite3.Connection, run_id: int) -> list[str]:
    return [r[0] for r in ro.execute(
        "SELECT DISTINCT benchmark FROM items WHERE run_id=? ORDER BY benchmark",
        (run_id,))]


def latest_rows(ro: sqlite3.Connection, run_id: int,
                benchmark: str) -> dict[str, dict]:
    """{item_id: row-dict} for one (run, benchmark) — id order, so the LATEST
    stored row per item_id wins (in-run duplicates cannot double-count)."""
    out: dict[str, dict] = {}
    for (rowid, item_id, gold, extracted, correct, truncated,
         finish_reason) in ro.execute(
            "SELECT id, item_id, gold, extracted, correct, truncated,"
            " finish_reason FROM items WHERE run_id=? AND benchmark=?"
            " ORDER BY id", (run_id, benchmark)):
        out[item_id] = dict(rowid=rowid, item_id=item_id, gold=gold,
                            extracted=extracted, correct=correct,
                            truncated=truncated, finish_reason=finish_reason,
                            source=run_id)
    return out


def merge_runs(ro: sqlite3.Connection, base: int, retries: list[int],
               benchmark: str | None = None) -> dict:
    """Compute the merged view (no writes). Returns
    {benchmark, base, retries, rows, warnings, n, n_scored, value, card_value,
     delta, n_overridden, n_retry_only, n_truncated, sources} where rows is
    the ordered merged item list (base order, retry-only ids appended)."""
    if not retries:
        raise ValueError("need at least one retry run")
    if base in retries or len(set(retries)) != len(retries):
        raise ValueError(f"base/retry run ids must be distinct "
                         f"(base={base}, retries={retries})")
    base_benches = run_benchmarks(ro, base)
    if benchmark is None:
        if len(base_benches) != 1:
            raise ValueError(
                f"base run {base} has {len(base_benches)} benchmarks "
                f"{base_benches} — pass --benchmark to pick one")
        benchmark = base_benches[0]
    if benchmark not in base_benches:
        raise ValueError(f"base run {base} has no {benchmark!r} items "
                         f"(has {base_benches})")

    warnings: list[str] = []
    env_base = run_env(ro, base)
    for rid in retries:
        if benchmark not in run_benchmarks(ro, rid):
            raise ValueError(f"retry run {rid} has no {benchmark!r} items — "
                             "a merge needs the SAME benchmark in every run")
        env_r = run_env(ro, rid)
        for k in _COMPARABILITY_KEYS:
            if env_base.get(k) != env_r.get(k):
                warnings.append(
                    f"config mismatch {k}: base run {base}="
                    f"{env_base.get(k)!r} vs retry run {rid}={env_r.get(k)!r}"
                    " — the merged number is NOT one protocol/config")

    merged = latest_rows(ro, base, benchmark)   # insertion order = base order
    n_overridden = 0
    retry_only: list[str] = []
    for rid in retries:                          # later retries win
        for item_id, row in latest_rows(ro, rid, benchmark).items():
            old = merged.get(item_id)
            if old is None:
                retry_only.append(item_id)
                warnings.append(
                    f"retry run {rid} item {item_id!r} is NOT in base run "
                    f"{base} — included, but a tail retry should be a subset")
            else:
                if old["gold"] != row["gold"]:
                    warnings.append(
                        f"gold mismatch on {item_id!r}: run {old['source']} "
                        f"{old['gold']!r} vs retry run {rid} {row['gold']!r} "
                        "— dataset/builder drift, the merge may be comparing "
                        "apples to oranges")
                if old["source"] == base:
                    n_overridden += 1            # count base items overridden
            merged[item_id] = row

    rows = list(merged.values())
    scored = [r for r in rows if r["correct"] is not None]
    value = (100.0 * sum(bool(r["correct"]) for r in scored) / len(scored)
             if scored else None)
    card = pv.CARD_TARGETS.get(benchmark, {}).get("value")
    delta = None if (value is None or card is None) else round(value - card, 2)
    return {
        "benchmark": benchmark, "base": base, "retries": list(retries),
        "rows": rows, "warnings": warnings,
        "n": len(rows), "n_scored": len(scored),
        "value": value, "card_value": card, "delta": delta,
        "n_overridden": n_overridden, "n_retry_only": len(retry_only),
        "n_truncated": sum(1 for r in rows if r["truncated"] == 1),
        "sources": {r["item_id"]: r["source"] for r in rows},
    }


def merge_label(m: dict) -> str:
    return (f"merge {m['benchmark']}: base=run{m['base']} "
            f"retries={['run%d' % r for r in m['retries']]}")


def _merged_authorizer(action, arg1, arg2, dbname, source):
    """sqlite authorizer for the writer connection: reads free; writes ONLY
    to merged_summary (+ the sqlite-internal bookkeeping its CREATE/
    AUTOINCREMENT need). runs/items/summary are DENIED at the engine level —
    never-mutates-originals is mechanical, not a promise."""
    ok, deny = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                  sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_FUNCTION):
        return ok
    # merged_summary has NO AUTOINCREMENT, so sqlite_sequence needs no
    # allowance at all (review finding 8: the old allow-list let the writer
    # touch other tables' sequence bookkeeping). sqlite_master writes are the
    # CREATE TABLE's own schema row.
    if action == sqlite3.SQLITE_INSERT and arg1 in ("merged_summary",
                                                    "sqlite_master"):
        return ok
    if action == sqlite3.SQLITE_UPDATE and arg1 == "sqlite_master":
        return ok
    if action == sqlite3.SQLITE_CREATE_TABLE and arg1 == "merged_summary":
        return ok
    if action == sqlite3.SQLITE_PRAGMA and arg1 == "table_info":
        return ok
    return deny


def write_merged_summary(db_path: str, m: dict, note: str = "") -> int:
    """Append ONE merged_summary row via an authorizer-guarded read-write
    connection. timeout=30 rides out a concurrent engine run. Returns the
    new row id. The merge WARNINGS are persisted into the stored note —
    stdout-only warnings would leave the durable row looking clean (review
    finding 3)."""
    if m["warnings"]:
        note = ((note + " | " if note else "")
                + f"{len(m['warnings'])} WARNING(s): "
                + "; ".join(m["warnings"]))
    rw = sqlite3.connect(db_path, timeout=30.0)
    try:
        rw.set_authorizer(_merged_authorizer)
        rw.execute(_MERGED_SCHEMA)
        cur = rw.execute(
            "INSERT INTO merged_summary(created_utc, benchmark, base_run_id,"
            " retry_run_ids, label, n, n_scored, metric, value, card_value,"
            " delta, n_overridden, n_retry_only, n_truncated, sources_json,"
            " note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_utc(), m["benchmark"], m["base"], json.dumps(m["retries"]),
             merge_label(m), m["n"], m["n_scored"], "acc", m["value"],
             m["card_value"], m["delta"], m["n_overridden"],
             m["n_retry_only"], m["n_truncated"], json.dumps(m["sources"]),
             note))
        rw.commit()
        return cur.lastrowid
    finally:
        rw.close()


def _run_header(ro: sqlite3.Connection, run_id: int, role: str) -> str:
    rows = ro.execute(
        "SELECT created_utc, model, harness_git, note FROM runs WHERE run_id=?",
        (run_id,)).fetchall()
    created, model, git, note = rows[0] if rows else ("?", "?", "?", "")
    env = run_env(ro, run_id)
    return (f"[merge] {role}: run {run_id}  [{created}]  model={model}  "
            f"harness={git}  protocol={env.get('protocol')}  "
            f"attention_path={env.get('attention_path')}  "
            f"max_new={env.get('max_new')}  ids={env.get('ids')}  "
            f"note={note!r}")


def print_report(ro: sqlite3.Connection, m: dict) -> None:
    print(_run_header(ro, m["base"], "base"))
    for rid in m["retries"]:
        print(_run_header(ro, rid, "retry"))
    for w in m["warnings"]:
        print(f"[merge] WARNING: {w}")
    print(f"\n{'item_id':<24} {'source':>9}  correct  truncated")
    for r in m["rows"]:
        src = ("base:%d" % m["base"] if r["source"] == m["base"]
               else "retry:%d" % r["source"])
        print(f"{r['item_id']:<24} {src:>9}  {_fmt(r['correct']):>7}  "
              f"{_fmt(r['truncated']):>9}")
    print(f"\n[merge] {merge_label(m)}")
    print(f"[merge] merged acc={_fmt(m['value'], '{:.2f}')} "
          f"(n={m['n']}, scored={m['n_scored']})  "
          f"card={_fmt(m['card_value'], '{:.1f}')} "
          f"Δ={_fmt(m['delta'], '{:+.2f}')}  "
          f"n_truncated={m['n_truncated']} (after merge)  "
          f"overridden={m['n_overridden']}  retry_only={m['n_retry_only']}")


def _fmt(v, spec="{}", none="-"):
    return none if v is None else spec.format(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--base", type=int, required=True,
                    help="the BASE run_id (the full run with the truncated tail)")
    ap.add_argument("--retry", type=int, action="append", required=True,
                    help="a RETRY run_id (repeatable; later ones override "
                         "earlier ones on overlapping item_ids)")
    ap.add_argument("--benchmark", default=None,
                    help="benchmark to merge (default: the base run's only one)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the merged view without writing merged_summary")
    ap.add_argument("--note", default="", help="note stored on the merged row")
    args = ap.parse_args(argv)

    ro = connect_ro(args.db)
    try:
        m = merge_runs(ro, args.base, args.retry, args.benchmark)
        print_report(ro, m)
    finally:
        # close BEFORE writing: a lingering read txn would block the writer
        ro.close()
    if args.dry_run:
        print("[merge] --dry-run: no merged_summary row written.")
    else:
        rowid = write_merged_summary(args.db, m, args.note)
        print(f"[merge] merged_summary row {rowid} appended "
              f"(originals untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
