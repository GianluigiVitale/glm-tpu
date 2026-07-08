#!/usr/bin/env python3
"""Print the DSA-sparse vs dense-MLA decode-throughput A/B table from
bench/results.db — the Δ view of the >=256K FLOP/throughput gate.

The instrument is bench/dsa_throughput.py run TWICE (two cluster launches,
GLM_DSA_MODE unset vs baked — the attention_path recorded in each run's
env_json); this tool joins the two runs' per-ctx summary rows
(benchmark = dsa_throughput_ctx<L>, metric = decode_tok_s) and prints

    ctx | dense tok/s | dsa tok/s | Δ | ratio | prefill walls | output-identity

Output-identity compares the VERBATIM stored output token ids item-by-item
(same seeds -> same prompts -> greedy outputs should match where the sparse
path is exact; a divergence count is evidence, not noise). Runs are
auto-selected (the LATEST run per attention_path class) or pinned via
--run-dense/--run-dsa. Comparability (num_seqs / measure_tokens / model /
dcp) is checked and any mismatch printed as a loud WARNING — a Δ across
mismatched configs is not the gate.

    python3 report_throughput.py                 # auto: latest dense vs dsa
    python3 report_throughput.py --run-dense 12 --run-dsa 13
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance as pv

_PREFIX = "dsa_throughput_ctx"
_COMPARABILITY_KEYS = ("num_seqs", "measure_tokens", "model", "dcp",
                       "max_batched_tokens", "base_seed")


def _note_json(note: str | None) -> dict:
    """Parse a summary note written as JSON; tolerate the ' n_truncated=N'
    suffix provenance.finalize may append."""
    if not note:
        return {}
    for cand in (note, note.rsplit(" n_truncated=", 1)[0]):
        try:
            return json.loads(cand)
        except (ValueError, TypeError):
            continue
    return {}


def run_env(conn, run_id: int) -> dict:
    row = conn.execute("SELECT env_json FROM runs WHERE run_id=?",
                       (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run_id {run_id} not found")
    try:
        return json.loads(row[0] or "{}")
    except ValueError:
        return {}


def classify(env: dict) -> str | None:
    """dense | dsa | None, from the recorded attention_path (the audit's A/B
    axis — never inferred from anything else)."""
    ap = env.get("attention_path") or ""
    if ap == "dense-mla":
        return "dense"
    if ap.startswith("dsa-sparse"):
        return "dsa"
    return None


def throughput_run_ids(conn) -> list[int]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT run_id FROM summary WHERE benchmark LIKE ? "
        "ORDER BY run_id", (_PREFIX + "%",))]


def latest_ab(conn) -> dict:
    """{'dense': run_id|None, 'dsa': run_id|None} — the latest throughput run
    per attention_path class."""
    out: dict = {"dense": None, "dsa": None}
    for rid in throughput_run_ids(conn):
        cls = classify(run_env(conn, rid))
        if cls:
            out[cls] = rid
    return out


def rungs(conn, run_id: int) -> dict:
    """{ctx: {'decode_tok_s': float, ...note fields...}} for one run."""
    out = {}
    for bench, value, note in conn.execute(
            "SELECT benchmark, value, note FROM summary WHERE run_id=? AND "
            "benchmark LIKE ? ORDER BY id", (run_id, _PREFIX + "%")):
        try:
            ctx = int(bench[len(_PREFIX):])
        except ValueError:
            continue
        d = _note_json(note)
        d["decode_tok_s"] = value
        out[ctx] = d      # later rows (re-runs within a run) win
    return out


def _out_ids(raw: str | None):
    try:
        return json.loads(raw or "")["token_ids"]
    except (ValueError, TypeError, KeyError):
        return None


def output_identity(conn, run_a: int, run_b: int) -> dict:
    """{ctx: (n_match, n_total)} comparing verbatim output token ids joined on
    (benchmark, item_id) across the two runs."""
    def items(rid):
        return {(b, i): _out_ids(r) for b, i, r in conn.execute(
            "SELECT benchmark, item_id, raw_output FROM items WHERE run_id=? "
            "AND benchmark LIKE ?", (rid, _PREFIX + "%"))}
    a, b = items(run_a), items(run_b)
    out: dict = {}
    for key in sorted(set(a) & set(b)):
        ctx = int(key[0][len(_PREFIX):])
        match, total = out.get(ctx, (0, 0))
        same = a[key] is not None and a[key] == b[key]
        out[ctx] = (match + int(same), total + 1)
    return out


def ab_table(conn, run_dense: int, run_dsa: int) -> tuple[list[dict], list[str]]:
    """Join the two runs by ctx. Returns (rows, warnings)."""
    env_d, env_s = run_env(conn, run_dense), run_env(conn, run_dsa)
    warnings = []
    if classify(env_d) != "dense":
        warnings.append(f"run {run_dense} attention_path="
                        f"{env_d.get('attention_path')!r} is NOT dense-mla")
    if classify(env_s) != "dsa":
        warnings.append(f"run {run_dsa} attention_path="
                        f"{env_s.get('attention_path')!r} is NOT dsa-sparse")
    for k in _COMPARABILITY_KEYS:
        if env_d.get(k) != env_s.get(k):
            warnings.append(f"config mismatch {k}: dense={env_d.get(k)!r} "
                            f"vs dsa={env_s.get(k)!r} — the Δ is NOT a "
                            f"controlled A/B on this key")
    rd, rs = rungs(conn, run_dense), rungs(conn, run_dsa)
    ident = output_identity(conn, run_dense, run_dsa)
    rows = []
    for ctx in sorted(set(rd) | set(rs)):
        d, s = rd.get(ctx), rs.get(ctx)
        dv = d.get("decode_tok_s") if d else None
        sv = s.get("decode_tok_s") if s else None
        rows.append({
            "ctx": ctx,
            "dense_tok_s": dv, "dsa_tok_s": sv,
            "delta_tok_s": (round(sv - dv, 3)
                            if dv is not None and sv is not None else None),
            "ratio": (round(sv / dv, 3)
                      if dv not in (None, 0) and sv is not None else None),
            "dense_prefill_s": d.get("t_prefill_s") if d else None,
            "dsa_prefill_s": s.get("t_prefill_s") if s else None,
            "outputs_match": ident.get(ctx),
        })
    return rows, warnings


def _fmt(v, spec="{:.2f}", none="-"):
    return none if v is None else spec.format(v)


def format_table(rows: list[dict]) -> str:
    hdr = (f"{'ctx':>8} | {'dense tok/s':>11} | {'dsa tok/s':>10} | "
           f"{'Δ tok/s':>9} | {'ratio':>6} | {'prefill d/s (s)':>17} | outputs")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        om = r["outputs_match"]
        lines.append(
            f"{r['ctx']:>8} | {_fmt(r['dense_tok_s']):>11} | "
            f"{_fmt(r['dsa_tok_s']):>10} | {_fmt(r['delta_tok_s']):>9} | "
            f"{_fmt(r['ratio'], '{:.3f}'):>6} | "
            f"{_fmt(r['dense_prefill_s']):>8}/{_fmt(r['dsa_prefill_s']):<8} | "
            + (f"{om[0]}/{om[1]} identical" if om else "-"))
    return "\n".join(lines)


def _run_header(conn, run_id: int) -> str:
    row = conn.execute(
        "SELECT created_utc, model, harness_git, fork_git, note FROM runs "
        "WHERE run_id=?", (run_id,)).fetchone()
    env = run_env(conn, run_id)
    return (f"run {run_id}: {env.get('attention_path')}  [{row[0]}]  "
            f"model={row[1]}  harness={row[2]}  fork={row[3]}  "
            f"dcp={env.get('dcp')}  num_seqs={env.get('num_seqs')}  "
            f"measure_tokens={env.get('measure_tokens')}  note={row[4]!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=None,
                    help="provenance DB path (default bench/results.db)")
    ap.add_argument("--run-dense", type=int, default=None,
                    help="dense-mla run_id (default: latest dense throughput run)")
    ap.add_argument("--run-dsa", type=int, default=None,
                    help="dsa-sparse run_id (default: latest dsa throughput run)")
    ap.add_argument("--json", dest="out_json", default=None,
                    help="also write the joined table to this JSON path")
    args = ap.parse_args(argv)

    conn = pv.connect(args.db) if args.db else pv.connect()
    auto = latest_ab(conn)
    run_dense = args.run_dense if args.run_dense is not None else auto["dense"]
    run_dsa = args.run_dsa if args.run_dsa is not None else auto["dsa"]
    if run_dense is None or run_dsa is None:
        have = throughput_run_ids(conn)
        print("[report] need TWO dsa_throughput runs (one dense-mla, one "
              f"dsa-sparse); found dense={run_dense} dsa={run_dsa} "
              f"(throughput runs in this DB: {have}).\n[report] run "
              "bench/dsa_throughput.py under the other launch env first "
              "(the A/B knob is GLM_DSA_MODE at launch).", flush=True)
        return 1

    rows, warnings = ab_table(conn, run_dense, run_dsa)
    print(_run_header(conn, run_dense))
    print(_run_header(conn, run_dsa))
    for w in warnings:
        print(f"[report] WARNING: {w}", flush=True)
    print()
    print(format_table(rows))
    print("\n(ratio = dsa/dense decode tok/s; outputs = per-ctx count of "
          "sequences whose verbatim output token ids are identical across "
          "the two runs)")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"run_dense": run_dense, "run_dsa": run_dsa,
                       "warnings": warnings, "rows": rows}, f, indent=2)
        print(f"[report] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
