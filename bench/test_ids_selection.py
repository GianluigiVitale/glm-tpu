#!/usr/bin/env python3
"""CPU tests for run_bench --ids (the retry selector for a truncated tail) —
its OWN file, deliberately: the first version of this test lived inside
test_bench.py and was clobbered by a concurrent workstream's rewrite of that
shared file (adversarial review 2026-07-08, finding 1). A FAKE datasets
module (no network) + temp DBs only.

Covers: load_items id filtering (dataset order, duplicate/whitespace
normalization, missing-id refusal, limit/offset refusal, empty refusal),
run_benchmark end-to-end into a temp provenance DB, run_benchmark's own
ids+limit refusal (with `ids is not None` semantics: an EMPTY computed id
list must refuse, never fall through to a full run), _run_env provenance,
and the CLI refusals via subprocess — with pv.connect redirected to a TEMP
DB so a refusal regression can never append junk to the real results.db
(review finding 6).

Run: JAX_PLATFORMS=cpu python3 -m pytest bench/test_ids_selection.py -q
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
import provenance as pv
import run_bench as rb

HERE = os.path.dirname(os.path.abspath(__file__))


def _fake_datasets(rows):
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda *a, **kw: rows
    return mod


def test_load_items_ids():
    rows = [{"question": f"q{i}", "answer": f"steps #### {i}"}
            for i in range(6)]
    had = "datasets" in sys.modules
    old_mod = sys.modules.get("datasets")
    try:
        sys.modules["datasets"] = _fake_datasets(rows)
        # selection returns DATASET order (not request order), stamps meta
        items = B.load_items(B.GSM8K, ids=["gsm8k_4", "gsm8k_1"])
        assert [it.item_id for it in items] == ["gsm8k_1", "gsm8k_4"]
        assert [it.gold for it in items] == ["1", "4"]
        assert all(it.meta["hf_revision"] == B.GSM8K.hf_revision
                   for it in items)
        # duplicates + whitespace normalize away
        items = B.load_items(B.GSM8K, ids=[" gsm8k_2 ", "gsm8k_2"])
        assert [it.item_id for it in items] == ["gsm8k_2"]
        # an unknown id is REFUSED (a silent partial retry would corrupt a
        # later merge) and the error names ONLY the missing ids
        try:
            B.load_items(B.GSM8K, ids=["gsm8k_1", "gsm8k_99"])
            raise AssertionError("missing id must be refused")
        except ValueError as e:
            assert "gsm8k_99" in str(e) and "'gsm8k_1'" not in str(e)
        # ids + limit/offset refused — the id list IS the selection
        for kw in ({"limit": 2}, {"offset": 1}):
            try:
                B.load_items(B.GSM8K, ids=["gsm8k_1"], **kw)
                raise AssertionError(f"ids+{kw} must be refused")
            except ValueError as e:
                assert "compose" in str(e)
        # empty / whitespace-only ids refused
        for bad in ([], [" "]):
            try:
                B.load_items(B.GSM8K, ids=bad)
                raise AssertionError(f"ids={bad!r} must be refused")
            except ValueError as e:
                assert "empty" in str(e)
    finally:
        if had:
            sys.modules["datasets"] = old_mod
        else:
            sys.modules.pop("datasets", None)
    print("  load_items --ids filtering + refusals OK")


def test_run_benchmark_ids_end_to_end():
    rows = [{"question": f"q{i}", "answer": f"steps #### {i}"}
            for i in range(6)]
    had = "datasets" in sys.modules
    old_mod = sys.modules.get("datasets")
    try:
        sys.modules["datasets"] = _fake_datasets(rows)
        with tempfile.TemporaryDirectory() as d:
            conn = pv.connect(os.path.join(d, "t.db"))
            rid = pv.start_run(conn, model="STUB", env={}, note="ids unit")
            summ = rb.run_benchmark(conn, rid, B.GSM8K, rb.stub_generate,
                                    ids=["gsm8k_0", "gsm8k_5"])
            got = [r[0] for r in conn.execute(
                "SELECT item_id FROM items WHERE run_id=? ORDER BY id",
                (rid,))]
            assert got == ["gsm8k_0", "gsm8k_5"] and summ["n"] == 2
            # ids+limit refused at the run_benchmark layer too
            try:
                rb.run_benchmark(conn, rid, B.GSM8K, rb.stub_generate,
                                 ids=["gsm8k_0"], limit=2)
                raise AssertionError("run_benchmark ids+limit must be refused")
            except ValueError as e:
                assert "--ids" in str(e)
            # review finding 7: an EMPTY computed id list must REFUSE (via
            # load_items), never silently run the full dataset
            try:
                rb.run_benchmark(conn, rid, B.GSM8K, rb.stub_generate, ids=[])
                raise AssertionError("ids=[] must be refused")
            except ValueError as e:
                assert "empty" in str(e)
            n_after = conn.execute("SELECT COUNT(*) FROM items WHERE run_id=?",
                                   (rid,)).fetchone()[0]
            assert n_after == 2                  # nothing ran on the refusals
    finally:
        if had:
            sys.modules["datasets"] = old_mod
        else:
            sys.modules.pop("datasets", None)
    print("  run_benchmark --ids end-to-end + empty-list refusal OK")


def test_run_env_records_ids():
    args = argparse.Namespace(
        model="STUB", stub=True, max_len=8192, max_new=64, max_seqs=8,
        max_batched_tokens=4096, batch_size=0, gmu=0.94, num_gpu_blocks=0,
        protocol="greedy", samples=1, seed=0, ids="gsm8k_3,gsm8k_17")
    assert rb._run_env(args, ["gsm8k"])["ids"] == "gsm8k_3,gsm8k_17"
    args.ids = None
    assert rb._run_env(args, ["gsm8k"])["ids"] is None
    print("  _run_env records the --ids selection OK")


def test_cli_refusals_subprocess():
    """argparse exit 2 for every refused combination. pv.connect is
    REDIRECTED to a temp DB inside the subprocess (review finding 6): if a
    refusal ever regresses, the stub run pollutes the throwaway DB — never
    the real bench/results.db."""
    with tempfile.TemporaryDirectory() as d:
        tmpdb = os.path.join(d, "guard.db")
        cases = [
            (["--benchmark", "gsm8k", "--stub", "--ids", "gsm8k_1",
              "--limit", "2"], "--ids"),
            (["--benchmark", "gsm8k", "--stub", "--ids", "gsm8k_1",
              "--offset", "1"], "--ids"),
            (["--benchmarks", "gsm8k,mmlu_pro", "--stub", "--ids", "gsm8k_1"],
             "ONE benchmark"),
            (["--benchmark", "gsm8k", "--stub", "--ids", " , "], "empty"),
        ]
        env = dict(os.environ, JAX_PLATFORMS="cpu")   # CPU before python starts
        for cli, needle in cases:
            script = (
                f"import sys; sys.path.insert(0, {HERE!r})\n"
                "import provenance as pv\n"
                "_orig = pv.connect\n"
                f"pv.connect = lambda path=None: _orig({tmpdb!r})\n"
                "import run_bench as rb\n"
                f"sys.argv = ['run_bench.py'] + {cli!r}\n"
                "rb.main()\n")
            proc = subprocess.run([sys.executable, "-c", script], env=env,
                                  capture_output=True, text=True, timeout=120)
            assert proc.returncode == 2, (cli, proc.stdout, proc.stderr)
            assert needle in proc.stderr, (cli, proc.stderr)
        assert not os.path.exists(tmpdb)     # refusals precede ANY db touch
    print("  CLI --ids refusals (subprocess, temp-DB-guarded) OK")


if __name__ == "__main__":
    test_load_items_ids()
    test_run_benchmark_ids_end_to_end()
    test_run_env_records_ids()
    test_cli_refusals_subprocess()
    print("ALL --ids selection CPU tests passed.")
