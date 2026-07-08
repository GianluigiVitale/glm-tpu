#!/usr/bin/env python3
"""Print the passkey LENGTH x DEPTH accuracy matrix from bench/results.db —
the per-cell view of the Stage-2 long-context gate.

The audit requires per-depth reporting: '99% shallow / 60% deep is a failure
wearing an average'. bench/glm_longctx.py stores every trial under
benchmark = passkey_L{L}_d{depth} (the depth rides in the benchmark name of
BOTH the item rows and the per-cell summary rows), so this tool recomputes
each cell straight from the ITEM rows — a number with no items behind it is
not a result — and prints:

    * the L x depth matrix: accuracy% (n) per cell, failing cells marked '!'
    * per-length and per-depth marginals (the deep-vs-shallow breakdown)
    * the >=GATE%-PER-CELL verdict (default 95%, PLAN.md's Stage-2 bar),
      plus whether the ladder reaches the >=128K coverage the gate demands
    * the run's recorded attention_path (dense-mla vs dsa-sparse:<mode>) —
      no passkey number without its attention path.

Run selection: --run-id pins a run; default is the LATEST run with passkey
items. The DB is opened READ-ONLY (sqlite mode=ro): a report can never touch
the provenance originals, and a typo'd --db path errors instead of silently
creating an empty DB. Exit codes: 0 = the Stage-2 gate holds (every cell at
or above the bar AND the ladder reaches >= 128K), 2 = gate FAIL (a failing
cell OR a coverage shortfall — a 4K smoke ladder must never exit 0),
1 = no passkey data.

    JAX_PLATFORMS=cpu python3 report_passkey.py                # latest run
    JAX_PLATFORMS=cpu python3 report_passkey.py --run-id 17
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance as pv

_CELL_RE = re.compile(r"^passkey_L(\d+)_d([0-9.]+)$")
_GATE_DEFAULT = 95.0
_GATE_COVER_L = 128 * 1024      # the Stage-2 gate: >=95% out to >=128K
# LIKE with `_` ESCAPED: `_` is a single-char wildcard, so an unescaped
# 'passkey_L%' would also match e.g. 'passkeyXL...' (review finding 2).
_LIKE = "benchmark LIKE 'passkey\\_L%' ESCAPE '\\'"


def connect_ro(path: str) -> sqlite3.Connection:
    """READ-ONLY provenance DB handle (the merge_runs/rescore discipline):
    pv.connect would run schema DDL + migrations read-write and CREATE a file
    on a typo'd path — the opposite of what an audit report may do."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def parse_cell(benchmark: str) -> tuple[int, float] | None:
    """passkey_L{L}_d{d} -> (L, depth); None for any other benchmark."""
    m = _CELL_RE.match(benchmark or "")
    if not m:
        return None
    try:
        return int(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def passkey_run_ids(conn) -> list[int]:
    return [r[0] for r in conn.execute(
        f"SELECT DISTINCT run_id FROM items WHERE {_LIKE} ORDER BY run_id")]


def cells_from_items(conn, run_id: int) -> dict[tuple[int, float], dict]:
    """{(L, depth): {n, n_scored, correct, acc}} recomputed from the ITEM rows
    (the provenance ground truth; finalize's convention — acc over items whose
    correct is NOT NULL)."""
    cells: dict[tuple[int, float], dict] = {}
    for bench, correct in conn.execute(
            f"SELECT benchmark, correct FROM items WHERE run_id=? AND {_LIKE}",
            (run_id,)):
        key = parse_cell(bench)
        if key is None:
            continue
        c = cells.setdefault(key, {"n": 0, "n_scored": 0, "correct": 0})
        c["n"] += 1
        if correct is not None:
            c["n_scored"] += 1
            c["correct"] += int(bool(correct))
    for c in cells.values():
        c["acc"] = (100.0 * c["correct"] / c["n_scored"]
                    if c["n_scored"] else None)
    return cells


def _acc(cells_subset: list[dict]) -> tuple[float | None, int]:
    ok = sum(c["correct"] for c in cells_subset)
    n = sum(c["n_scored"] for c in cells_subset)
    return (100.0 * ok / n if n else None), n


def _cell_txt(acc, n, gate) -> str:
    if acc is None:
        return f"- ({n})"
    flag = "!" if acc < gate else " "
    return f"{acc:5.1f}% ({n}){flag}"


def format_matrix(cells: dict[tuple[int, float], dict],
                  gate: float = _GATE_DEFAULT) -> str:
    """The L x depth accuracy matrix + marginals, failing cells marked '!'."""
    lengths = sorted({L for L, _ in cells})
    depths = sorted({d for _, d in cells})
    width = max(12, *(len(_cell_txt(c["acc"], c["n_scored"], gate))
                      for c in cells.values())) + 2
    hdr = f"{'':>10}" + "".join(f"{'d=%g' % d:>{width}}" for d in depths) \
        + f" | {'all-d':>{width}}"
    lines = [hdr, "-" * len(hdr)]
    for L in lengths:
        row = [f"{'L=%d' % L:>10}"]
        for d in depths:
            c = cells.get((L, d))
            row.append(f"{_cell_txt(c['acc'], c['n_scored'], gate) if c else '-':>{width}}")
        macc, mn = _acc([cells[(L, d)] for d in depths if (L, d) in cells])
        row.append(f" | {_cell_txt(macc, mn, gate):>{width}}")
        lines.append("".join(row))
    lines.append("-" * len(hdr))
    row = [f"{'all-L':>10}"]
    for d in depths:
        macc, mn = _acc([cells[(L, d)] for L in lengths if (L, d) in cells])
        row.append(f"{_cell_txt(macc, mn, gate):>{width}}")
    oacc, on = _acc(list(cells.values()))
    row.append(f" | {_cell_txt(oacc, on, gate):>{width}}")
    lines.append("".join(row))
    return "\n".join(lines)


def verdict(cells: dict[tuple[int, float], dict],
            gate: float = _GATE_DEFAULT) -> tuple[bool, list[str]]:
    """(gate_pass, failing-cell descriptions). A cell passes only if it has
    scored trials AND acc >= gate — an empty/unscored cell can never pass."""
    failing = []
    for (L, d) in sorted(cells):
        c = cells[(L, d)]
        if c["acc"] is None or c["acc"] < gate:
            failing.append(
                f"L={L} d={d} -> "
                + ("no scored trials" if c["acc"] is None else
                   f"{c['acc']:.1f}% ({c['correct']}/{c['n_scored']})"))
    return not failing, failing


def _run_header(conn, run_id: int) -> str:
    row = conn.execute(
        "SELECT created_utc, model, harness_git, fork_git, note FROM runs "
        "WHERE run_id=?", (run_id,)).fetchone()
    try:
        env = json.loads(conn.execute(
            "SELECT env_json FROM runs WHERE run_id=?",
            (run_id,)).fetchone()[0] or "{}")
    except (TypeError, ValueError):
        env = {}
    return (f"run {run_id}: attention_path={env.get('attention_path')}  "
            f"prompt_mode={env.get('prompt_mode')}  [{row[0]}]\n"
            f"  model={row[1]}  harness={row[2]}  fork={row[3]}  "
            f"trials/cell={env.get('trials')}  note={row[4]!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=None,
                    help="provenance DB path (default bench/results.db)")
    ap.add_argument("--run-id", type=int, default=None,
                    help="passkey run to report (default: latest run with "
                         "passkey items)")
    ap.add_argument("--gate", type=float, default=_GATE_DEFAULT,
                    help="per-cell accuracy gate in %% (default 95, the "
                         "Stage-2 bar)")
    args = ap.parse_args(argv)

    try:
        conn = connect_ro(args.db or pv.DEFAULT_DB)   # READ-ONLY, always
    except FileNotFoundError as exc:
        print(f"[passkey] no such provenance DB: {exc} — a report never "
              "creates one.")
        return 1
    have = passkey_run_ids(conn)
    run_id = args.run_id if args.run_id is not None else \
        (have[-1] if have else None)
    if run_id is None or run_id not in have:
        print(f"[passkey] no passkey items for run_id={run_id} "
              f"(runs with passkey items: {have}) — run bench/glm_longctx.py "
              "first.")
        return 1

    cells = cells_from_items(conn, run_id)
    if not cells:
        # LIKE-matched rows whose names the strict cell regex rejects (e.g. a
        # malformed manual insert): the no-data path, never a traceback
        # (review finding 2 — format_matrix on {} would TypeError).
        print(f"[passkey] run {run_id} has passkey-LIKE items but no "
              "parseable passkey_L{L}_d{depth} cells — nothing to report.")
        return 1
    print(_run_header(conn, run_id))
    print()
    print(format_matrix(cells, args.gate))
    print("\n(cell = accuracy% (n scored trials); '!' = below the "
          f"{args.gate:g}% per-cell gate; depth = fraction of prompt tokens "
          "before the needle)")

    ok, failing = verdict(cells, args.gate)
    # NOTE: max_l counts ANY parsed cell, scored or not — an all-unscored
    # 256K cell satisfies coverage while failing the per-cell gate, so the
    # combined verdict below still cannot exit 0 on it.
    max_l = max(L for L, _ in cells)
    covered = max_l >= _GATE_COVER_L
    if ok:
        print(f"\nGATE (>= {args.gate:g}% per cell): PASS — all {len(cells)} "
              "cells at or above the bar.")
    else:
        print(f"\nGATE (>= {args.gate:g}% per cell): FAIL — "
              f"{len(failing)}/{len(cells)} cell(s) below the bar:")
        for f in failing:
            print(f"  {f}")
    if not covered:
        print(f"coverage: max length {max_l} < 128K — the Stage-2 gate needs "
              f">= {args.gate:g}% out to >= 128K; this ladder does not reach "
              "it yet (exit 2: a short ladder must never read as a gate "
              "pass).")
    else:
        print(f"coverage: max length {max_l} >= 128K (the Stage-2 gate's "
              "reach).")
    return 0 if (ok and covered) else 2


if __name__ == "__main__":
    raise SystemExit(main())
