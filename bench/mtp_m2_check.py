#!/usr/bin/env python3
"""MTP M2 gate checker — spec-decode greedy MUST be output-identical to
non-spec greedy (docs/08 §7 M2, runbook §7).

Given two run ids from the provenance DB (one GLM_SPEC_K spec-decode run, one
non-spec baseline — EITHER order; the roles are oriented from the recorded
runs.env_json, never trusted from argument position), assert per-item exact
identity of the generated token sequence. Greedy rejection sampling accepts a
draft token iff it equals the target argmax, so at temperature 0 spec-decode
is distribution-identical by construction — token-level equality is the M2
gate, and the only permitted exception is a documented tie flip (top-2 logit
gap below bf16 eps at the mismatch step; docs/08's protocol — this script
FINDS the mismatch, the logit-dump classification is a separate rerun).

What "token sequence" means here: the items table stores the VERBATIM
detokenized completion (items.raw_output = vLLM's output.text) plus the
generated-token count (items.n_gen_tokens = len(output.token_ids)) and
finish_reason. Detokenization is a deterministic function of the token ids,
so identical token sequences imply identical (text, count, finish_reason)
triples, and any triple mismatch proves the token sequences differ. The
converse gap (two DIFFERENT id sequences of the same length detokenizing to
the same text) is the one thing the DB cannot see; it is vanishingly unlikely
under greedy and, if it occurred, the OUTPUT would still be identical — the
M2 property being served. A triple match is therefore accepted as the M2
identity signal; raw token-id capture is an M3 instrumentation item.

Acceptance stats: per-item acceptance is NOT attributable from what M2 runs
record (SpecDecodingStats aggregation is an M3 deliverable — docs/08 §7 M3).
What IS available now: pass the spec run's driver log via --log and the
script scrapes vLLM's periodic "SpecDecoding metrics:" lines (emitted by the
stats logger — the run needs GLM_LOG_STATS=1, which the runbook §7 BASE env
sets; each line is an interval aggregate that resets after logging) and
reports per-interval + run-aggregate acceptance. Run-level tok/s A/B comes
from the two runs' summary notes (batch_wall_ms / gen_tok).

Exit codes: 0 = identity PASS; 1 = one or more item mismatches (apply the
docs/08 tie-flip protocol); 2 = runs not comparable (role/protocol/item-set
mismatch, stub or all-empty-output runs — a PASS must never be vacuous) or a
usage error (bad --db/--log path, unknown run id).

Usage (pod, after the runbook §7c pair):
    python mtp_m2_check.py <run_id_a> <run_id_b> [--log ~/glm-run/m2_k1.log]
    python mtp_m2_check.py --latest [--log ...]     # the two newest runs

CPU-only, stdlib-only: never imports vllm/engine — stub-testable anywhere
(test_mtp_m2_check.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "results.db")


class CompareError(Exception):
    """The two runs cannot be M2-compared (exit 2) — wrong roles, non-greedy
    protocol, or different item sets/prompts. Distinct from a MISMATCH (exit
    1), which is a comparable pair whose outputs differ."""


# ---------------------------------------------------------------- run loading

def load_run(conn: sqlite3.Connection, run_id: int) -> dict:
    row = conn.execute(
        "SELECT run_id, created_utc, model, harness_git, fork_git, env_json,"
        " note FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise CompareError(f"run_id {run_id} not found in the DB")
    env = {}
    try:
        env = json.loads(row[5] or "{}")
    except json.JSONDecodeError:
        pass
    return {"run_id": row[0], "created_utc": row[1], "model": row[2],
            "harness_git": row[3], "fork_git": row[4], "env": env,
            "note": row[6] or ""}


def spec_k_of(run: dict) -> int:
    """The run's GLM_SPEC_K, from the recorded env_json GLM_* sweep
    (run_bench._run_env os_env). 0 = spec-decode off."""
    raw = (run["env"].get("os_env") or {}).get("GLM_SPEC_K")
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def orient_runs(run_a: dict, run_b: dict) -> tuple[dict, dict]:
    """Return (spec_run, base_run) from the recorded provenance. Exactly one
    of the two runs must have GLM_SPEC_K >= 1 — argument order is NEVER
    trusted for the roles."""
    ka, kb = spec_k_of(run_a), spec_k_of(run_b)
    if bool(ka) == bool(kb):
        raise CompareError(
            f"cannot orient roles: run {run_a['run_id']} has GLM_SPEC_K="
            f"{ka or 'unset'}, run {run_b['run_id']} has GLM_SPEC_K="
            f"{kb or 'unset'} — exactly one run must be the spec-decode run "
            "(env_json os_env GLM_SPEC_K)")
    return (run_a, run_b) if ka else (run_b, run_a)


# ------------------------------------------------------------- comparability

def check_comparable(spec_run: dict, base_run: dict) -> list[str]:
    """Hard-fail (CompareError) on anything that voids the M2 theory bar;
    return WARNING strings for generation-relevant env drift that the item
    comparison itself will surface."""
    warnings = []
    for run in (spec_run, base_run):
        # vacuous-PASS guard (round-9 review MED-2): a --stub run stores an
        # empty reply for every item — two stubs would "pass" identity while
        # comparing nothing. compare_items adds the all-empty-output twin.
        if run["env"].get("stub") or run["model"] == "STUB":
            raise CompareError(
                f"run {run['run_id']} is a --stub run (model="
                f"{run['model']!r}) — no model output to compare; M2 needs "
                "real engine runs")
        proto = run["env"].get("protocol")
        temp = run["env"].get("temperature")
        if proto is None:
            warnings.append(
                f"run {run['run_id']} has no recorded protocol (pre-protocol "
                "or unparseable env_json) — greedy assumed, verify manually")
        if proto not in (None, "greedy"):
            raise CompareError(
                f"run {run['run_id']} used protocol={proto!r} — the M2 "
                "identity gate only holds for greedy decode (temperature 0: "
                "argmax-match acceptance = exact)")
        if temp not in (None, 0, 0.0):
            raise CompareError(
                f"run {run['run_id']} recorded temperature={temp!r} — M2 "
                "requires greedy (temperature 0.0)")
    for key in ("model", "max_len", "max_new", "attention_path"):
        va, vb = spec_run["env"].get(key), base_run["env"].get(key)
        if va != vb:
            warnings.append(f"env[{key!r}] differs: spec={va!r} base={vb!r}")
    osa = spec_run["env"].get("os_env") or {}
    osb = base_run["env"].get("os_env") or {}
    for key in ("GLM_DSA_MODE", "GLM_DCP", "TPU_DISABLE_DSA_INDEXER",
                "DISABLE_WEIGHT_REQUANTIZATION", "TPU_MIN_TOKEN_BUCKET"):
        if osa.get(key) != osb.get(key):
            warnings.append(f"os_env[{key!r}] differs: "
                            f"spec={osa.get(key)!r} base={osb.get(key)!r}")
    if spec_run["fork_git"] != base_run["fork_git"]:
        warnings.append(f"fork_git differs: spec={spec_run['fork_git']!r} "
                        f"base={base_run['fork_git']!r}")
    return warnings


def fetch_items(conn: sqlite3.Connection, run_id: int) -> dict:
    """(benchmark, item_id) -> item row dict for one run."""
    rows = conn.execute(
        "SELECT benchmark, item_id, prompt, raw_output, n_gen_tokens,"
        " finish_reason, latency_ms FROM items WHERE run_id=?",
        (run_id,)).fetchall()
    return {(r[0], r[1]): {"prompt": r[2], "raw_output": r[3],
                           "n_gen_tokens": r[4], "finish_reason": r[5],
                           "latency_ms": r[6]}
            for r in rows}


# ------------------------------------------------------------ item comparison

def _first_divergence(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n  # one is a prefix of the other (or equal)


def compare_items(spec_items: dict, base_items: dict) -> dict:
    """Per-item exact-identity comparison. Returns
    {"n": ..., "mismatches": [...]} or raises CompareError when the runs are
    not over the same items (different keys or different prompts)."""
    if set(spec_items) != set(base_items):
        only_s = sorted(set(spec_items) - set(base_items))[:5]
        only_b = sorted(set(base_items) - set(spec_items))[:5]
        raise CompareError(
            f"item sets differ ({len(spec_items)} spec vs {len(base_items)} "
            f"base items); only-spec={only_s} only-base={only_b}")
    if not spec_items:
        raise CompareError("no items recorded for these runs")
    # vacuous-PASS guard twin (round-9 review MED-2): every output empty on
    # both sides = nothing was compared (all-SKIP prompts, or stub-shaped
    # rows that dodged the env guard) — never a PASS.
    if all(not (it["raw_output"] or "")
           for d in (spec_items, base_items) for it in d.values()):
        raise CompareError(
            "every stored raw_output is EMPTY in both runs — vacuous "
            "identity (stub run or all prompts SKIPped?); M2 needs real "
            "generations")
    mismatches = []
    for key in sorted(spec_items):
        s, b = spec_items[key], base_items[key]
        if s["prompt"] != b["prompt"]:
            raise CompareError(
                f"item {key} was asked DIFFERENT prompts in the two runs — "
                "not the same benchmark slice (check --limit/--offset)")
        # NULL and "" are DIFFERENT (round-9 review LOW-5): compare the
        # stored values directly; the excerpt math below may coalesce.
        sa, ba = s["raw_output"] or "", b["raw_output"] or ""
        identical = (s["raw_output"] == b["raw_output"]
                     and s["n_gen_tokens"] == b["n_gen_tokens"]
                     and s["finish_reason"] == b["finish_reason"])
        if not identical:
            div = _first_divergence(sa, ba)
            mismatches.append({
                "benchmark": key[0], "item_id": key[1],
                "char_divergence": div,
                "spec_excerpt": sa[max(0, div - 40):div + 40],
                "base_excerpt": ba[max(0, div - 40):div + 40],
                "spec_gen_tokens": s["n_gen_tokens"],
                "base_gen_tokens": b["n_gen_tokens"],
                "spec_finish": s["finish_reason"],
                "base_finish": b["finish_reason"],
            })
    return {"n": len(spec_items), "mismatches": mismatches}


# -------------------------------------------------- acceptance-stats scraping

# vLLM's interval spec-decode log line (v1/spec_decode/metrics.py, verified
# against ~/vllm-build 2026-07-08). Emitted only when the stats logger runs
# (GLM_LOG_STATS=1 -> disable_log_stats=False); each line aggregates one
# logging interval and then RESETS.
_SPEC_LINE = re.compile(
    r"SpecDecoding metrics: "
    r"Mean acceptance length: (?P<mal>[\d.]+|nan), "
    r"Accepted throughput: (?P<acc_tps>[\d.]+|nan) tokens/s, "
    r"Drafted throughput: (?P<draft_tps>[\d.]+|nan) tokens/s, "
    r"Accepted: (?P<accepted>\d+) tokens, "
    r"Drafted: (?P<drafted>\d+) tokens, "
    # any per-position entry can be nan (a num_drafts=0 interval divides 0/0
    # for EVERY position — round-9 review LOW-3):
    r"Per-position acceptance rate: (?P<per_pos>(?:[\d.]+|nan)(?:, (?:[\d.]+|nan))*), "
    r"Avg Draft acceptance rate: (?P<rate>[\d.]+|nan)%")


def parse_spec_log(text: str) -> dict:
    """Scrape every interval 'SpecDecoding metrics:' line from a driver log.
    Returns {'intervals': [...], 'aggregate': {...} | None}. The aggregate
    mean acceptance length is RECOVERED from the logged 2-decimal values
    (num_drafts_i = accepted_i / (mal_i - 1)) — approximate by construction
    (an interval whose true mal rounds down to 1.00 with accepted>0 is
    excluded from the drafts denominator but not the accepted numerator —
    a small upward bias; and vLLM logs the final idle-time flush at DEBUG,
    invisible at INFO, so the tail interval can be missing). Exact counters
    are the M3 get_metrics deliverable."""
    intervals = []
    for m in _SPEC_LINE.finditer(text):
        d = m.groupdict()
        intervals.append({
            "mean_acceptance_length": float(d["mal"]),
            "accepted_tokens": int(d["accepted"]),
            "drafted_tokens": int(d["drafted"]),
            "per_position_rates": [float(x) for x in d["per_pos"].split(", ")],
            "draft_acceptance_rate_pct": float(d["rate"]),
        })
    if not intervals:
        return {"intervals": [], "aggregate": None}
    tot_acc = sum(i["accepted_tokens"] for i in intervals)
    tot_draft = sum(i["drafted_tokens"] for i in intervals)
    drafts_est = sum(i["accepted_tokens"] / (i["mean_acceptance_length"] - 1)
                     for i in intervals
                     if i["mean_acceptance_length"] > 1
                     and i["accepted_tokens"] > 0)
    aggregate = {
        "accepted_tokens": tot_acc,
        "drafted_tokens": tot_draft,
        "draft_acceptance_rate_pct": (100.0 * tot_acc / tot_draft
                                      if tot_draft else float("nan")),
        "mean_acceptance_length_est": (1.0 + tot_acc / drafts_est
                                       if drafts_est else float("nan")),
        "n_intervals": len(intervals),
    }
    return {"intervals": intervals, "aggregate": aggregate}


_NOTE_WALL = re.compile(r"batch_wall_ms=(\d+)")
_NOTE_TOK = re.compile(r"gen_tok=(\d+)")


def run_tok_s(conn: sqlite3.Connection, run_id: int) -> dict | None:
    """Run-level END-TO-END tok/s from the summary note (batch_wall_ms wraps
    the whole generate_batch call INCLUDING prefill — not decode-only;
    run_bench's batched-path wall-clock provenance). Reads the run's newest
    summary row — right for §7's single-benchmark runs, do not trust it for
    multi-benchmark runs. None when absent."""
    row = conn.execute(
        "SELECT note FROM summary WHERE run_id=? ORDER BY id DESC LIMIT 1",
        (run_id,)).fetchone()
    if not row or not row[0]:
        return None
    mw, mt = _NOTE_WALL.search(row[0]), _NOTE_TOK.search(row[0])
    if not (mw and mt):
        return None
    wall_ms, tok = int(mw.group(1)), int(mt.group(1))
    return {"gen_tok": tok, "wall_ms": wall_ms,
            "tok_s": (1000.0 * tok / wall_ms) if wall_ms else float("nan")}


# ------------------------------------------------------------------- the gate

def m2_check(conn: sqlite3.Connection, run_id_a: int, run_id_b: int,
             log_text: str | None = None) -> dict:
    """The full M2 comparison. Returns a report dict (see keys below);
    raises CompareError when the pair is not comparable."""
    run_a, run_b = load_run(conn, run_id_a), load_run(conn, run_id_b)
    spec_run, base_run = orient_runs(run_a, run_b)
    warnings = check_comparable(spec_run, base_run)
    items = compare_items(fetch_items(conn, spec_run["run_id"]),
                          fetch_items(conn, base_run["run_id"]))
    return {
        "spec_run_id": spec_run["run_id"],
        "base_run_id": base_run["run_id"],
        "spec_k": spec_k_of(spec_run),
        "warnings": warnings,
        "n_items": items["n"],
        "mismatches": items["mismatches"],
        "identical": not items["mismatches"],
        "spec_tok_s": run_tok_s(conn, spec_run["run_id"]),
        "base_tok_s": run_tok_s(conn, base_run["run_id"]),
        "acceptance": (parse_spec_log(log_text) if log_text is not None
                       else None),
    }


def print_report(rep: dict) -> None:
    k = rep["spec_k"]
    print(f"M2 identity gate: spec run {rep['spec_run_id']} (mtp k={k}) vs "
          f"non-spec baseline run {rep['base_run_id']} — "
          f"{rep['n_items']} items")
    for w in rep["warnings"]:
        print(f"  WARNING: {w}")
    if rep["identical"]:
        print(f"  PASS: all {rep['n_items']} items token-identical "
              "(raw_output + n_gen_tokens + finish_reason)")
    else:
        print(f"  FAIL: {len(rep['mismatches'])}/{rep['n_items']} items "
              "differ — apply the docs/08 tie-flip protocol (dump step "
              "logits at the divergence; a top-2 gap < bf16 eps is a "
              "documented pass-with-note, anything else is a bug):")
        for m in rep["mismatches"]:
            print(f"    {m['benchmark']}/{m['item_id']}: first divergence at "
                  f"char {m['char_divergence']}; gen_tok "
                  f"{m['spec_gen_tokens']} vs {m['base_gen_tokens']}; finish "
                  f"{m['spec_finish']} vs {m['base_finish']}")
            print(f"      spec: ...{m['spec_excerpt']!r}...")
            print(f"      base: ...{m['base_excerpt']!r}...")
    st, bt = rep["spec_tok_s"], rep["base_tok_s"]
    if st and bt:
        ratio = (st["tok_s"] / bt["tok_s"]) if bt["tok_s"] else float("nan")
        print(f"  tok/s A/B (run-level, batch_wall_ms): spec "
              f"{st['tok_s']:.2f} vs base {bt['tok_s']:.2f} "
              f"({ratio:.2f}x) — [gen_tok {st['gen_tok']} vs "
              f"{bt['gen_tok']}]")
    else:
        print("  tok/s A/B: unavailable (summary note lacks batch_wall_ms/"
              "gen_tok on one side)")
    acc = rep["acceptance"]
    if acc is None:
        print("  acceptance stats: no --log given (per-item acceptance is "
              "not recorded by M2 runs; interval stats need the spec run's "
              "driver log with GLM_LOG_STATS=1)")
    elif not acc["intervals"]:
        print("  acceptance stats: NO 'SpecDecoding metrics:' lines in the "
              "log — was GLM_LOG_STATS=1 set on the spec run?")
    else:
        a = acc["aggregate"]
        print(f"  acceptance (from log, {a['n_intervals']} interval lines; "
              "per-RUN aggregate — per-item attribution is an M3 "
              "deliverable):")
        print(f"    accepted {a['accepted_tokens']} / drafted "
              f"{a['drafted_tokens']} tokens = "
              f"{a['draft_acceptance_rate_pct']:.1f}% draft acceptance; "
              f"mean acceptance length ~{a['mean_acceptance_length_est']:.2f}"
              " (recovered from 2-dp interval values — approximate)")
        for i, iv in enumerate(acc["intervals"]):
            rates = ", ".join(f"{r:.3f}" for r in iv["per_position_rates"])
            print(f"    interval {i}: mal={iv['mean_acceptance_length']:.2f} "
                  f"acc={iv['accepted_tokens']} draft={iv['drafted_tokens']} "
                  f"per-pos=[{rates}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="MTP M2 gate: spec-decode greedy == non-spec greedy "
                    "(exact token-sequence identity per item)")
    ap.add_argument("run_ids", nargs="*", type=int,
                    help="the two run ids (either order — roles are oriented "
                         "from env_json GLM_SPEC_K)")
    ap.add_argument("--latest", action="store_true",
                    help="compare the two NEWEST runs in the DB (the runbook "
                         "§7c back-to-back pair)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--log", default=None,
                    help="the SPEC run's driver log — scraped for vLLM's "
                         "interval 'SpecDecoding metrics:' acceptance lines "
                         "(requires GLM_LOG_STATS=1 on that run)")
    args = ap.parse_args(argv)

    if args.latest == bool(args.run_ids):
        ap.error("give exactly two run ids, or --latest (not both)")
    if args.run_ids and len(args.run_ids) != 2:
        ap.error(f"need exactly two run ids, got {len(args.run_ids)}")

    # usage errors must exit 2, never masquerade as exit-1 "mismatch"
    # (round-9 review MED-1): sqlite3.connect CREATES a missing file, so
    # check the path first; wrap file/DB errors explicitly.
    if not os.path.exists(args.db):
        print(f"ERROR (usage): DB not found: {args.db}", file=sys.stderr)
        return 2
    log_text = None
    if args.log:
        try:
            with open(args.log, encoding="utf-8", errors="replace") as f:
                log_text = f.read()
        except OSError as exc:
            print(f"ERROR (usage): cannot read --log {args.log}: {exc}",
                  file=sys.stderr)
            return 2
    conn = sqlite3.connect(args.db)
    try:
        if args.latest:
            ids = [r[0] for r in conn.execute(
                "SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 2")]
            if len(ids) < 2:
                print("ERROR: --latest needs at least two runs in the DB",
                      file=sys.stderr)
                return 2
            run_id_a, run_id_b = ids
        else:
            run_id_a, run_id_b = args.run_ids
        rep = m2_check(conn, run_id_a, run_id_b, log_text=log_text)
    except CompareError as exc:
        print(f"ERROR (not comparable): {exc}", file=sys.stderr)
        return 2
    except sqlite3.Error as exc:
        print(f"ERROR (db): {args.db}: {exc} — is this a provenance "
              "results.db?", file=sys.stderr)
        return 2
    finally:
        conn.close()
    print_report(rep)
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
