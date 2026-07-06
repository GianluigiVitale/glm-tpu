#!/usr/bin/env python3
"""Provenance DB for GLM-5.2-on-TPU-v4 benchmark reproduction.

The rule (owner directive): **do NOT run anything that cannot be traced back.**
Every benchmark item — the exact question asked, WHEN it was asked, the model's
verbatim reply, the extracted answer, and pass/fail — is stored in a SQLite DB
(`bench/results.db`) alongside full run-level provenance (model+revision, the
harness + fork git commits, the env/mesh/flags, the pod). A benchmark number with
no `items` rows behind it is not a result.

This module is model-agnostic and CPU-only: it is the storage layer. The harness
(`run_bench.py`) calls it; the model is plugged in later (Stage 1).

Usage:
    import provenance as pv
    conn = pv.connect()                         # creates schema if absent
    run = pv.start_run(conn, model="zai-org/GLM-5.2-FP8", revision="<sha>",
                       env={"DSA": False, "attn_dp": 8}, pod="db-v4-64-od", note="dense-MLA")
    pv.record_item(conn, run, benchmark="gpqa_diamond", item_id="q0",
                   prompt=..., gold="A", raw_output=..., extracted="A",
                   correct=True, score=1.0, n_gen_tokens=512, latency_ms=1234.5)
    pv.finalize(conn, run, benchmark="gpqa_diamond", metric="acc", value=90.5)
"""
from __future__ import annotations

import json
import os
import subprocess
import sqlite3
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "results.db")

# HF model-card published values (GLM-5.2, verified 2026-07-06). Signed Δ is
# computed against these. Some are agentic/heavy (need full harnesses) — flagged.
CARD_TARGETS = {
    # tractable-first (reasoning / short generation / loglikelihood-style)
    "gpqa_diamond":        {"value": 91.2, "kind": "reasoning"},
    "aime_2026":           {"value": 99.2, "kind": "math"},
    "hmmt_nov_2025":       {"value": 94.4, "kind": "math"},
    "hmmt_feb_2026":       {"value": 92.5, "kind": "math"},
    "imo_answer_bench":    {"value": 91.0, "kind": "math"},
    "hle":                 {"value": 40.5, "kind": "reasoning"},
    "hle_tools":           {"value": 54.7, "kind": "agentic"},
    "critpt":              {"value": 20.9, "kind": "reasoning"},
    # agentic / coding (heavier harnesses — later)
    "swe_bench_pro":       {"value": 62.1, "kind": "agentic"},
    "nl2repo":             {"value": 48.9, "kind": "agentic"},
    "deepswe":             {"value": 46.2, "kind": "agentic"},
    "program_bench":       {"value": 63.7, "kind": "coding"},
    "terminal_bench_2_1":  {"value": 81.0, "kind": "agentic"},
    "frontier_swe":        {"value": 74.4, "kind": "agentic"},
    "post_train_bench":    {"value": 34.3, "kind": "agentic"},
    "swe_marathon":        {"value": 13.0, "kind": "agentic"},
    "mcp_atlas":           {"value": 76.8, "kind": "agentic"},
    "tool_decathlon":      {"value": 48.2, "kind": "agentic"},
    # a standard loglikelihood sanity gate (not on the card; a fast Stage-1 check)
    "mmlu_pro":            {"value": None, "kind": "loglikelihood"},
    "gsm8k":               {"value": None, "kind": "math"},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc   TEXT NOT NULL,
    model         TEXT NOT NULL,
    model_revision TEXT,
    harness_git   TEXT,
    fork_git      TEXT,
    env_json      TEXT,
    pod           TEXT,
    note          TEXT
);
CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(run_id),
    benchmark     TEXT NOT NULL,
    item_id       TEXT NOT NULL,
    asked_utc     TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    gold          TEXT,
    raw_output    TEXT,
    extracted     TEXT,
    correct       INTEGER,
    score         REAL,
    n_prompt_tokens INTEGER,
    n_gen_tokens  INTEGER,
    latency_ms    REAL,
    seed          INTEGER
);
CREATE TABLE IF NOT EXISTS summary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(run_id),
    benchmark     TEXT NOT NULL,
    created_utc   TEXT NOT NULL,
    n             INTEGER,
    metric        TEXT,
    value         REAL,
    card_value    REAL,
    delta         REAL,
    note          TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_run ON items(run_id, benchmark);
CREATE INDEX IF NOT EXISTS idx_summary_run ON summary(run_id, benchmark);
"""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def connect(path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def start_run(conn, *, model, revision=None, env=None, pod="db-v4-64-od",
              note="", harness_repo=None, fork_repo=None) -> int:
    harness_git = _git(harness_repo or os.path.abspath(os.path.join(HERE, "..")))
    fork_git = _git(fork_repo or os.path.expanduser("~/tpu-inference"))
    cur = conn.execute(
        "INSERT INTO runs(created_utc,model,model_revision,harness_git,fork_git,env_json,pod,note)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (_utc(), model, revision, harness_git, fork_git,
         json.dumps(env or {}, sort_keys=True), pod, note))
    conn.commit()
    return cur.lastrowid


def record_item(conn, run_id, *, benchmark, item_id, prompt, gold=None,
                raw_output=None, extracted=None, correct=None, score=None,
                n_prompt_tokens=None, n_gen_tokens=None, latency_ms=None,
                seed=None, asked_utc=None):
    conn.execute(
        "INSERT INTO items(run_id,benchmark,item_id,asked_utc,prompt,gold,raw_output,"
        "extracted,correct,score,n_prompt_tokens,n_gen_tokens,latency_ms,seed)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, benchmark, str(item_id), asked_utc or _utc(), prompt, gold,
         raw_output, extracted, None if correct is None else int(bool(correct)),
         score, n_prompt_tokens, n_gen_tokens, latency_ms, seed))
    conn.commit()


def finalize(conn, run_id, *, benchmark, metric="acc", value=None, note=""):
    """Compute the summary from the stored items and the card target; store it.
    Recomputes n/value from items when value is None (single source of truth)."""
    rows = conn.execute(
        "SELECT correct FROM items WHERE run_id=? AND benchmark=? AND correct IS NOT NULL",
        (run_id, benchmark)).fetchall()
    n = len(rows)
    if value is None and n:
        value = 100.0 * sum(r[0] for r in rows) / n
    card = CARD_TARGETS.get(benchmark, {}).get("value")
    delta = None if (value is None or card is None) else round(value - card, 2)
    conn.execute(
        "INSERT INTO summary(run_id,benchmark,created_utc,n,metric,value,card_value,delta,note)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (run_id, benchmark, _utc(), n, metric, value, card, delta, note))
    conn.commit()
    return {"benchmark": benchmark, "n": n, "value": value,
            "card_value": card, "delta": delta}


def report(conn, run_id=None):
    """Human-readable summary of stored runs (for quick inspection)."""
    q = "SELECT run_id,benchmark,n,metric,value,card_value,delta,created_utc FROM summary"
    args = ()
    if run_id is not None:
        q += " WHERE run_id=?"; args = (run_id,)
    return conn.execute(q + " ORDER BY run_id,benchmark", args).fetchall()


if __name__ == "__main__":
    # smoke: create the DB, print schema + card targets.
    c = connect()
    print(f"DB ready at {DEFAULT_DB}")
    print("tables:", [r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()])
    print(f"card targets: {len(CARD_TARGETS)} benchmarks")
