#!/usr/bin/env python3
"""CPU stub tests for mtp_m2_check.py (the MTP M2 identity gate, runbook §7).

Builds throwaway provenance DBs with synthetic spec-on/spec-off run pairs —
no vllm, no TPU, no network. Run: pytest test_mtp_m2_check.py  (or python)."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

import provenance as pv
import mtp_m2_check as m2


def _env(spec_k: int | None, **over) -> dict:
    """A minimal env_json of the run_bench._run_env shape."""
    os_env = {"GLM_TP": "32", "TPU_DISABLE_DSA_INDEXER": "1"}
    if spec_k:
        os_env["GLM_SPEC_K"] = str(spec_k)
    env = {"protocol": "greedy", "temperature": 0.0, "model": "gs://m",
           "max_len": 4096, "max_new": 1024, "attention_path": "dense-mla",
           "os_env": os_env}
    env.update(over)
    return env


def _mk_pair(conn, *, spec_k=1, items_a=None, items_b=None,
             env_a=None, env_b=None, note_a="", note_b=""):
    """Two runs: A = spec-on (unless env_a overrides), B = baseline.
    items_* : list of (item_id, prompt, raw_output, n_gen, finish)."""
    default = [("q0", "p0", "out zero", 10, "stop"),
               ("q1", "p1", "out one", 20, "stop")]
    ra = pv.start_run(conn, model="gs://m", env=env_a or _env(spec_k),
                      note=note_a)
    rb = pv.start_run(conn, model="gs://m", env=env_b or _env(None),
                      note=note_b)
    for rid, rows in ((ra, items_a or default), (rb, items_b or default)):
        for iid, prompt, out, n_gen, finish in rows:
            pv.record_item(conn, rid, benchmark="gsm8k", item_id=iid,
                           prompt=prompt, gold="1", raw_output=out,
                           extracted=None, correct=False, score=0.0,
                           n_gen_tokens=n_gen, finish_reason=finish)
    return ra, rb


@contextlib.contextmanager
def _db():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.db")
        conn = pv.connect(path)
        try:
            yield conn, path
        finally:
            conn.close()


def test_identity_pass():
    """Identical items in both runs -> identical=True, exit 0 through main(),
    roles oriented from env_json regardless of argument order."""
    with _db() as (conn, path):
        ra, rb = _mk_pair(conn, spec_k=1,
                          note_b="batched:2 chunks=2 batch_wall_ms=2000 gen_tok=30",
                          note_a="batched:2 chunks=2 batch_wall_ms=1000 gen_tok=30")
        pv.finalize(conn, ra, benchmark="gsm8k",
                    note="batched:2 chunks=2 batch_wall_ms=1000 gen_tok=30")
        pv.finalize(conn, rb, benchmark="gsm8k",
                    note="batched:2 chunks=2 batch_wall_ms=2000 gen_tok=30")
        for order in ((ra, rb), (rb, ra)):     # either order: roles from env
            rep = m2.m2_check(conn, *order)
            assert rep["identical"] and rep["n_items"] == 2
            assert rep["spec_run_id"] == ra and rep["base_run_id"] == rb
            assert rep["spec_k"] == 1
        # tok/s A/B from the summary notes (2x uplift here)
        rep = m2.m2_check(conn, ra, rb)
        assert abs(rep["spec_tok_s"]["tok_s"] - 30.0) < 1e-9
        assert abs(rep["base_tok_s"]["tok_s"] - 15.0) < 1e-9
        # CLI end-to-end: exit 0, PASS printed
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = m2.main([str(ra), str(rb), "--db", path])
        assert rc == 0 and "PASS: all 2 items" in out.getvalue()
        # --latest picks the same (only) pair
        with contextlib.redirect_stdout(io.StringIO()):
            assert m2.main(["--latest", "--db", path]) == 0
    print("  identity PASS path OK")


def test_mismatch_detected():
    """A text divergence AND a count-only divergence both flag; the first-
    divergence offset is right; exit code 1."""
    with _db() as (conn, path):
        base = [("q0", "p0", "out zero", 10, "stop"),
                ("q1", "p1", "out one", 20, "stop"),
                ("q2", "p2", "same text", 7, "stop")]
        spec = [("q0", "p0", "out zERO", 10, "stop"),      # text differs @5
                ("q1", "p1", "out one", 20, "stop"),       # identical
                ("q2", "p2", "same text", 8, "stop")]      # count-only differs
        ra, rb = _mk_pair(conn, spec_k=5, items_a=spec, items_b=base)
        rep = m2.m2_check(conn, ra, rb)
        assert not rep["identical"]
        bad = {m["item_id"]: m for m in rep["mismatches"]}
        assert set(bad) == {"q0", "q2"}
        assert bad["q0"]["char_divergence"] == 5
        assert bad["q2"]["spec_gen_tokens"] == 8
        assert bad["q2"]["base_gen_tokens"] == 7
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = m2.main([str(ra), str(rb), "--db", path])
        assert rc == 1 and "tie-flip" in out.getvalue()
    print("  mismatch detection OK")


def test_role_and_protocol_guards():
    """Ambiguous roles (both or neither spec) and non-greedy runs are NOT
    comparable: CompareError / exit 2."""
    with _db() as (conn, path):
        # neither run has GLM_SPEC_K
        ra, rb = _mk_pair(conn, spec_k=1, env_a=_env(None))
        try:
            m2.m2_check(conn, ra, rb)
            raise AssertionError("ambiguous roles should have raised")
        except m2.CompareError as e:
            assert "GLM_SPEC_K" in str(e)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            assert m2.main([str(ra), str(rb), "--db", path]) == 2
        assert "not comparable" in err.getvalue()
        # both runs spec-on
        rc, rd = _mk_pair(conn, spec_k=1, env_b=_env(3))
        try:
            m2.m2_check(conn, rc, rd)
            raise AssertionError("double-spec should have raised")
        except m2.CompareError:
            pass
        # card protocol / nonzero temperature void the theory bar
        re_, rf = _mk_pair(conn, spec_k=1,
                           env_b=_env(None, protocol="card", temperature=1.0))
        try:
            m2.m2_check(conn, re_, rf)
            raise AssertionError("card protocol should have raised")
        except m2.CompareError as e:
            assert "greedy" in str(e)
    print("  role/protocol guards OK")


def test_item_set_and_prompt_guards():
    """Different item sets or different prompts per id -> CompareError (the
    runs are not over the same benchmark slice)."""
    with _db() as (conn, _):
        ra, rb = _mk_pair(
            conn, spec_k=1,
            items_a=[("q0", "p0", "x", 1, "stop")],
            items_b=[("q0", "p0", "x", 1, "stop"),
                     ("q1", "p1", "y", 1, "stop")])
        try:
            m2.m2_check(conn, ra, rb)
            raise AssertionError("item-set mismatch should have raised")
        except m2.CompareError as e:
            assert "item sets differ" in str(e)
        rc, rd = _mk_pair(conn, spec_k=1,
                          items_a=[("q0", "PROMPT A", "x", 1, "stop"),
                                   ("q1", "p1", "y", 1, "stop")],
                          items_b=[("q0", "PROMPT B", "x", 1, "stop"),
                                   ("q1", "p1", "y", 1, "stop")])
        try:
            m2.m2_check(conn, rc, rd)
            raise AssertionError("prompt mismatch should have raised")
        except m2.CompareError as e:
            assert "DIFFERENT prompts" in str(e)
    print("  item-set/prompt guards OK")


def test_vacuous_pass_guards():
    """Round-9 review MED-2: two --stub runs, or a pair whose every stored
    raw_output is empty, must be NOT COMPARABLE (exit 2) — never a PASS."""
    with _db() as (conn, path):
        # env-declared stub run
        ra, rb = _mk_pair(conn, spec_k=1, env_a=_env(1, stub=True))
        try:
            m2.m2_check(conn, ra, rb)
            raise AssertionError("stub run should have raised")
        except m2.CompareError as e:
            assert "stub" in str(e)
        # model == "STUB" (what run_bench actually stores for --stub)
        rc = pv.start_run(conn, model="STUB", env=_env(1))
        rd = pv.start_run(conn, model="gs://m", env=_env(None))
        for rid in (rc, rd):
            pv.record_item(conn, rid, benchmark="gsm8k", item_id="q0",
                           prompt="p0", gold="1", raw_output="",
                           n_gen_tokens=0)
        try:
            m2.m2_check(conn, rc, rd)
            raise AssertionError("model=STUB should have raised")
        except m2.CompareError as e:
            assert "stub" in str(e)
        # all-empty outputs on both sides (e.g. every prompt SKIPped)
        empty = [("q0", "p0", "", 0, None), ("q1", "p1", "", 0, None)]
        re_, rf = _mk_pair(conn, spec_k=1, items_a=empty, items_b=empty)
        try:
            m2.m2_check(conn, re_, rf)
            raise AssertionError("all-empty pair should have raised")
        except m2.CompareError as e:
            assert "EMPTY" in str(e)
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            assert m2.main([str(re_), str(rf), "--db", path]) == 2
    print("  vacuous-PASS guards OK")


def test_null_vs_empty_output_mismatch():
    """Round-9 review LOW-5: a NULL raw_output vs '' is a MISMATCH, not
    identity (one non-empty item keeps the vacuous guard out of the way)."""
    with _db() as (conn, _):
        ra, rb = _mk_pair(conn, spec_k=1,
                          items_a=[("q0", "p0", None, 0, None),
                                   ("q1", "p1", "real", 3, "stop")],
                          items_b=[("q0", "p0", "", 0, None),
                                   ("q1", "p1", "real", 3, "stop")])
        rep = m2.m2_check(conn, ra, rb)
        assert [m["item_id"] for m in rep["mismatches"]] == ["q0"]
    print("  NULL-vs-empty mismatch OK")


def test_usage_errors_exit_2():
    """Round-9 review MED-1: a typo'd --db or --log path is a USAGE error
    (exit 2), never exit 1 'mismatch' — and the checker must not create a
    missing DB file as a connect() side effect."""
    with _db() as (conn, path):
        ra, rb = _mk_pair(conn, spec_k=1)
        missing_db = path + ".nope"
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            assert m2.main([str(ra), str(rb), "--db", missing_db]) == 2
        assert "DB not found" in err.getvalue()
        assert not os.path.exists(missing_db)      # no empty-DB side effect
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            assert m2.main([str(ra), str(rb), "--db", path,
                            "--log", path + ".nolog"]) == 2
        assert "cannot read --log" in err.getvalue()
        # a real file that is not a provenance DB -> sqlite error -> 2
        notdb = path + ".txt"
        with open(notdb, "w") as f:
            f.write("not a database, just text long enough to have a header")
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            assert m2.main([str(ra), str(rb), "--db", notdb]) == 2
        # unknown run id -> CompareError -> 2
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            assert m2.main([str(ra), "9999", "--db", path]) == 2
    print("  usage-error exit codes OK")


def test_env_drift_warnings():
    """Generation-relevant env drift (max_new, GLM_DSA_MODE, fork_git) warns
    but does not block — the item comparison itself is the gate."""
    with _db() as (conn, _):
        env_b = _env(None, max_new=2048)
        env_b["os_env"]["GLM_DSA_MODE"] = "xla_ref"
        ra, rb = _mk_pair(conn, spec_k=1, env_b=env_b)
        rep = m2.m2_check(conn, ra, rb)
        assert rep["identical"]                    # still compared
        joined = " | ".join(rep["warnings"])
        assert "max_new" in joined and "GLM_DSA_MODE" in joined
    print("  env-drift warnings OK")


def test_parse_spec_log():
    """The SpecDecoding interval-line scraper: per-interval fields + the
    run-aggregate recovery (format pinned to ~/vllm-build
    vllm/v1/spec_decode/metrics.py)."""
    log = (
        "INFO 07-08 [loggers.py] Engine 000: SpecDecoding metrics: "
        "Mean acceptance length: 4.50, Accepted throughput: 100.00 tokens/s, "
        "Drafted throughput: 140.00 tokens/s, Accepted: 700 tokens, "
        "Drafted: 1000 tokens, Per-position acceptance rate: 0.900, 0.800, "
        "0.700, 0.600, 0.500, Avg Draft acceptance rate: 70.0%\n"
        "unrelated line\n"
        "INFO 07-08 [loggers.py] Engine 000: SpecDecoding metrics: "
        "Mean acceptance length: 3.00, Accepted throughput: 50.00 tokens/s, "
        "Drafted throughput: 100.00 tokens/s, Accepted: 300 tokens, "
        "Drafted: 600 tokens, Per-position acceptance rate: 0.700, 0.500, "
        "0.400, 0.250, 0.150, Avg Draft acceptance rate: 50.0%\n"
        # a num_drafts=0 interval: EVERY field nan-degenerate at k=5
        # (round-9 review LOW-3 — must still parse, zero contribution)
        "INFO 07-08 [loggers.py] Engine 000: SpecDecoding metrics: "
        "Mean acceptance length: nan, Accepted throughput: 0.00 tokens/s, "
        "Drafted throughput: 0.00 tokens/s, Accepted: 0 tokens, "
        "Drafted: 0 tokens, Per-position acceptance rate: nan, nan, nan, "
        "nan, nan, Avg Draft acceptance rate: nan%\n")
    got = m2.parse_spec_log(log)
    assert len(got["intervals"]) == 3
    i0, i1 = got["intervals"][:2]
    inan = got["intervals"][2]
    assert inan["accepted_tokens"] == 0 and inan["drafted_tokens"] == 0
    assert len(inan["per_position_rates"]) == 5
    assert all(r != r for r in inan["per_position_rates"])   # all nan
    assert i0["mean_acceptance_length"] == 4.50
    assert i0["accepted_tokens"] == 700 and i0["drafted_tokens"] == 1000
    assert i0["per_position_rates"] == [0.9, 0.8, 0.7, 0.6, 0.5]
    assert i1["draft_acceptance_rate_pct"] == 50.0
    agg = got["aggregate"]
    assert agg["accepted_tokens"] == 1000 and agg["drafted_tokens"] == 1600
    assert abs(agg["draft_acceptance_rate_pct"] - 62.5) < 1e-9
    # drafts: 700/3.5 = 200; 300/2.0 = 150 -> mal = 1 + 1000/350 = 3.857...
    assert abs(agg["mean_acceptance_length_est"] - (1 + 1000 / 350)) < 1e-9
    assert m2.parse_spec_log("no spec lines here") == {
        "intervals": [], "aggregate": None}
    print("  SpecDecoding log scraper OK")


def test_cli_log_flag():
    """--log wires the scraper into the report; a log with no spec lines
    prints the GLM_LOG_STATS hint."""
    with _db() as (conn, path):
        ra, rb = _mk_pair(conn, spec_k=1)
        with tempfile.NamedTemporaryFile("w", suffix=".log",
                                         delete=False) as f:
            f.write("engine built\nno metrics\n")
            logpath = f.name
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = m2.main([str(ra), str(rb), "--db", path,
                              "--log", logpath])
            assert rc == 0
            assert "GLM_LOG_STATS=1" in out.getvalue()
        finally:
            os.unlink(logpath)
    print("  CLI --log wiring OK")


if __name__ == "__main__":
    test_identity_pass()
    test_mismatch_detected()
    test_role_and_protocol_guards()
    test_item_set_and_prompt_guards()
    test_vacuous_pass_guards()
    test_null_vs_empty_output_mismatch()
    test_usage_errors_exit_2()
    test_env_drift_warnings()
    test_parse_spec_log()
    test_cli_log_flag()
    print("ALL mtp_m2_check CPU tests passed.")
