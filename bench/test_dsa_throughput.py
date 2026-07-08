#!/usr/bin/env python3
"""CPU unit tests for the DSA decode-throughput A/B harness
(dsa_throughput.py + report_throughput.py). No model, no network, no vllm,
no TPU.

Run: JAX_PLATFORMS=cpu ~/vllm-env/bin/python bench/test_dsa_throughput.py
 or: JAX_PLATFORMS=cpu ~/vllm-env/bin/python -m pytest bench/test_dsa_throughput.py -q
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsa_throughput as DT
import engine
import report_throughput as RT


@contextlib.contextmanager
def _env(**kv):
    """Temporarily set/unset env vars (None = unset); always restore."""
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# deterministic prompt construction
# ---------------------------------------------------------------------------
def test_prompt_determinism():
    a = DT.build_prompt_ids(1024, seed=7, vocab_size=DT.STUB_VOCAB_SIZE)
    b = DT.build_prompt_ids(1024, seed=7, vocab_size=DT.STUB_VOCAB_SIZE)
    assert a == b                                    # same params -> identical
    assert DT.prompt_sha256(a) == DT.prompt_sha256(b)
    c = DT.build_prompt_ids(1024, seed=8, vocab_size=DT.STUB_VOCAB_SIZE)
    assert c != a and DT.prompt_sha256(c) != DT.prompt_sha256(a)
    d = DT.build_prompt_ids(1024, seed=7, vocab_size=1000)  # range change
    assert d != a


def test_prompt_shape_and_vocab_range():
    import glm_longctx as LC
    for ctx in (16, 256, 8192):
        ids = DT.build_prompt_ids(ctx, seed=3, vocab_size=DT.STUB_VOCAB_SIZE)
        assert len(ids) == ctx                       # EXACTLY ctx tokens
        assert ids[:2] == LC.PROMPT_PREFIX_IDS       # [gMASK]<sop> prepended
        body = ids[2:]
        # sampled from the REAL vocab id range, never a special token: the
        # stub vocab == the real GLM vocab (154880), so the sampling band is
        # [256, 154820) — below <|endoftext|> and the whole special block
        low, high = DT.sample_range(DT.STUB_VOCAB_SIZE)
        assert (low, high) == (256, 154820)
        assert all(low <= t < high for t in body)
        assert not any(t in engine.EOS_IDS for t in body)
    # tiny vocab (mock tokenizers): the band clamps to the vocab
    assert DT.sample_range(1000) == (256, 1000)


def test_prompt_descriptor_reproducible_not_verbatim():
    ids = DT.build_prompt_ids(4096, seed=11, vocab_size=DT.STUB_VOCAB_SIZE)
    d = json.loads(DT.prompt_descriptor(ids, 4096, 11, DT.STUB_VOCAB_SIZE))
    # everything needed to rebuild + verify, WITHOUT storing 4096 ids of text
    assert d["seed"] == 11 and d["ctx"] == 4096 and d["n_tokens"] == 4096
    assert d["sample_low"] == 256 and d["sample_high"] == 154820
    assert d["sha256"] == DT.prompt_sha256(ids)
    rebuilt = DT.build_prompt_ids(d["ctx"], d["seed"], d["vocab_size"])
    assert DT.prompt_sha256(rebuilt) == d["sha256"]  # the rebuild contract
    assert "token_ids" not in d                      # ids NOT stored verbatim


def test_seq_seeds_distinct():
    seeds = {DT.seq_seed(12345, ctx, i)
             for ctx in (8192, 32768, 131072, 262144) for i in range(8)}
    assert len(seeds) == 4 * 8                       # no collisions on the ladder


# ---------------------------------------------------------------------------
# tok/s math
# ---------------------------------------------------------------------------
def test_throughput_math():
    # 4 seqs, N=101: pass A = 4x1 tok, pass B = 4x101 tok; walls 10 s / 20 s
    # -> 400 decode steps in 10 s = 40 tok/s aggregate, 10/seq, 100 ms/step,
    # e2e = 404/20 = 20.2 tok/s
    s = DT.throughput_from_walls(10.0, 20.0, tokens_a=4, tokens_b=404,
                                 num_seqs=4)
    assert s["decode_tok_s"] == 40.0
    assert s["per_seq_tok_s"] == 10.0
    assert s["ms_per_step"] == 100.0
    assert s["e2e_tok_s"] == 20.2
    assert s["t_prefill_s"] == 10.0 and s["t_full_s"] == 20.0
    assert not s["nonpositive_dt"]
    # single seq (the 256K gate config)
    s1 = DT.throughput_from_walls(100.0, 227.0, 1, 128, num_seqs=1)
    assert s1["decode_tok_s"] == round(127 / 127.0, 3) == 1.0
    assert s1["ms_per_step"] == 1000.0


def test_throughput_math_degenerate():
    # nonpositive delta (timer noise) must flag, never divide by zero
    s = DT.throughput_from_walls(20.0, 20.0, 4, 404, num_seqs=4)
    assert s["nonpositive_dt"] and s["decode_tok_s"] == 0.0
    assert s["ms_per_step"] is None
    s = DT.throughput_from_walls(21.0, 20.0, 4, 404, num_seqs=4)
    assert s["nonpositive_dt"] and s["decode_tok_s"] == 0.0


# ---------------------------------------------------------------------------
# KV math guard (docs/05 ladder + docs/11 §5 block math)
# ---------------------------------------------------------------------------
def test_kv_plan_256k_needs_dcp8():
    # the docs/05 ladder at the 128-block pool: 262144+128 tokens needs
    # ceil(262272/512/dcp) blocks/seq -> dcp=4: 129 > 128 REFUSE; dcp=8:
    # 65 <= 128 OK for ONE seq
    p4 = DT.kv_plan([262144], 1, 128, dcp=4, pool_blocks=128)
    assert not p4["fits"]
    assert p4["rungs"][0]["blocks_per_seq"] == 129
    assert p4["rungs"][0]["min_dcp"] == 8            # the docs/05 headline
    p8 = DT.kv_plan([262144], 1, 128, dcp=8, pool_blocks=128)
    assert p8["fits"]
    assert p8["rungs"][0]["blocks_per_seq"] == 65
    assert p8["logical_block_tokens"] == 4096        # 512 x dcp (#2398)
    # 2 concurrent 256K seqs do NOT fit dcp=8 (130 > 128) -> min_dcp=16
    p8x2 = DT.kv_plan([262144], 2, 128, dcp=8, pool_blocks=128)
    assert not p8x2["fits"] and p8x2["rungs"][0]["min_dcp"] == 16


def test_kv_plan_default_ladder():
    # the full default ladder at dcp=8 / 1 seq fits the 128-block pool
    ladder = [8192, 32768, 131072, 262144]
    p = DT.kv_plan(ladder, 1, 128, dcp=8, pool_blocks=128)
    assert p["fits"] and [r["ctx"] for r in p["rungs"]] == ladder
    # baseline (dcp=1) pool = 65,536 tokens (docs/05 §1.1): 8K+128 x 4 seqs
    # fits (68 blocks), x 8 seqs does NOT (136 > 128)
    assert DT.kv_plan([8192], 4, 128, dcp=1, pool_blocks=128)["fits"]
    p8s = DT.kv_plan([8192], 8, 128, dcp=1, pool_blocks=128)
    assert not p8s["fits"] and p8s["rungs"][0]["required_blocks"] == 136
    assert p8s["pool_tokens"] == 65536


def test_kv_plan_pool_override_and_validation():
    # an explicit smaller pool (e.g. --num-gpu-blocks 64, the runbook §3
    # halved pool) governs
    assert not DT.kv_plan([32768], 1, 128, dcp=1, pool_blocks=64)["fits"]
    assert DT.kv_plan([32768], 1, 128, dcp=2, pool_blocks=64)["fits"]
    for bad_dcp in (3, 5, 7, 64):
        try:
            DT.kv_plan([8192], 1, 128, dcp=bad_dcp, pool_blocks=128)
            raise AssertionError(f"dcp={bad_dcp} should have raised")
        except ValueError:
            pass
    try:
        DT.kv_plan([8192], 1, 128, dcp=1, pool_blocks=0)
        raise AssertionError("pool_blocks=0 should have raised")
    except ValueError:
        pass


def test_main_refuses_unfittable_config():
    """The guard REFUSES pre-build (exit 2) instead of letting the pod
    preempt/thrash silently (docs/16): 256K at dcp=4 (driver env GLM_DCP=4)
    cannot fit the 128-block pool — no engine build, no DB row."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        with _env(GLM_DCP="4", GLM_DSA_MODE=None):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = DT.main(["--stub", "--ctxs", "262144", "--num-seqs", "1",
                              "--measure-tokens", "128", "--db", db])
        assert rc == 2
        assert "REFUSE" in out.getvalue() and "GLM_DCP=8" in out.getvalue()
        # the refusal happens BEFORE any engine build or DB connect — no run
        # row, in fact no DB file at all
        assert not os.path.exists(db)


# ---------------------------------------------------------------------------
# stub pipeline -> provenance rows
# ---------------------------------------------------------------------------
def _run_stub(db, note, dsa_mode=None, ctxs="256,512", num_seqs=2,
              measure=8, seed=12345):
    with _env(GLM_DSA_MODE=dsa_mode, GLM_DCP=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = DT.main(["--stub", "--ctxs", ctxs, "--num-seqs",
                          str(num_seqs), "--measure-tokens", str(measure),
                          "--seed", str(seed), "--db", db, "--note", note,
                          "--out-json", os.path.join(
                              os.path.dirname(db), f"{note}.json")])
    assert rc == 0, out.getvalue()
    return out.getvalue()


def test_stub_pipeline_records_provenance():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        _run_stub(db, "stubA")
        assert "vllm" not in sys.modules              # the offline contract
        conn = sqlite3.connect(db)
        # runs row: attention_path + the A/B-critical config in env_json
        model, env_json = conn.execute(
            "SELECT model, env_json FROM runs").fetchone()
        env = json.loads(env_json)
        assert model == "STUB"
        assert env["attention_path"] == "dense-mla"   # GLM_DSA_MODE unset
        assert env["benchmark"] == "dsa_throughput"
        assert env["base_seed"] == 12345 and env["dcp"] == 1
        assert env["kv_plan"]["fits"] is True
        assert env["sample_low"] == 256 and env["sample_high"] == 154820
        # items: per (ctx, seq) with reproducible prompt + verbatim output
        items = conn.execute(
            "SELECT benchmark, item_id, prompt, raw_output, n_prompt_tokens,"
            " n_gen_tokens, seed, latency_ms FROM items ORDER BY id").fetchall()
        assert len(items) == 2 * 2                    # 2 ctxs x 2 seqs
        assert {r[0] for r in items} == {"dsa_throughput_ctx256",
                                         "dsa_throughput_ctx512"}
        for bench, item_id, prompt, raw, n_p, n_g, seed, lat in items:
            ctx = int(bench[len("dsa_throughput_ctx"):])
            p = json.loads(prompt)
            assert p["sha256"] and p["ctx"] == ctx == n_p
            i = int(item_id[3:])
            assert seed == DT.seq_seed(12345, ctx, i)  # recorded per item
            rebuilt = DT.build_prompt_ids(p["ctx"], p["seed"], p["vocab_size"])
            assert DT.prompt_sha256(rebuilt) == p["sha256"]
            out = json.loads(raw)                     # verbatim OUTPUT tokens
            assert len(out["token_ids"]) == 8 == n_g
            assert lat is None       # batched: per-item latency never faked
        # summary: one decode_tok_s row per rung + the aggregate row
        summ = conn.execute(
            "SELECT benchmark, metric, value, note FROM summary "
            "ORDER BY id").fetchall()
        names = [r[0] for r in summ]
        assert names == ["dsa_throughput_ctx256", "dsa_throughput_ctx512",
                         "dsa_throughput"]
        for bench, metric, value, note in summ[:2]:
            assert metric == "decode_tok_s"
            n = json.loads(note)
            # stub fake clock: wall = n_tokens * 1e-4 s -> dt = 7e-4 s for
            # 2x7 decode steps = 20000 tok/s aggregate, deterministic
            assert value == n["decode_tok_s"] == 20000.0
            assert n["per_seq_tok_s"] == 10000.0
            assert n["attention_path"] == "dense-mla"
            assert not n["count_mismatch"]
        assert summ[2][2] is None                     # aggregate: no fake value
        agg = json.loads(summ[2][3])
        assert agg["rungs"] == {"256": 20000.0, "512": 20000.0}


def test_stub_prompts_identical_across_attention_paths():
    """The A/B contract: the SAME seeds produce byte-identical prompts under
    both launch envs (prompt sha256 joins the two runs' items)."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        _run_stub(db, "stubA", dsa_mode=None)
        _run_stub(db, "stubB", dsa_mode="pallas_decode")
        conn = sqlite3.connect(db)
        shas = {}
        for run_id, bench, item_id, prompt in conn.execute(
                "SELECT run_id, benchmark, item_id, prompt FROM items"):
            shas.setdefault((bench, item_id), []).append(
                json.loads(prompt)["sha256"])
        assert all(len(v) == 2 and v[0] == v[1] for v in shas.values())


# ---------------------------------------------------------------------------
# the A/B report
# ---------------------------------------------------------------------------
def test_report_ab_table():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        _run_stub(db, "A-dense", dsa_mode=None)
        _run_stub(db, "B-dsa", dsa_mode="pallas_decode")
        import provenance as pv
        conn = pv.connect(db)
        ab = RT.latest_ab(conn)
        assert ab["dense"] == 1 and ab["dsa"] == 2
        assert RT.classify(RT.run_env(conn, 1)) == "dense"
        assert RT.classify(RT.run_env(conn, 2)) == "dsa"
        rows, warnings = RT.ab_table(conn, 1, 2)
        assert warnings == []                        # identical configs
        assert [r["ctx"] for r in rows] == [256, 512]
        for r in rows:
            assert r["dense_tok_s"] == r["dsa_tok_s"] == 20000.0
            assert r["delta_tok_s"] == 0.0 and r["ratio"] == 1.0
            assert r["outputs_match"] == (2, 2)      # stub outputs are a
            # function of the prompt only -> identical across paths
            assert r["dense_prefill_s"] is not None
        table = RT.format_table(rows)
        assert "256" in table and "2/2 identical" in table
        # main() end-to-end (auto-selection)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = RT.main(["--db", db])
        assert rc == 0
        assert "dense-mla" in out.getvalue()
        assert "dsa-sparse:pallas_decode" in out.getvalue()


def test_report_warns_on_config_mismatch():
    """A Δ across mismatched configs is NOT the gate — the report must say so."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        _run_stub(db, "A", dsa_mode=None, num_seqs=1)
        _run_stub(db, "B", dsa_mode="pallas_decode", num_seqs=2)
        import provenance as pv
        conn = pv.connect(db)
        rows, warnings = RT.ab_table(conn, 1, 2)
        assert any("num_seqs" in w for w in warnings)


def test_report_needs_two_runs():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        _run_stub(db, "only-dense", dsa_mode=None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = RT.main(["--db", db])
        assert rc == 1 and "need TWO" in out.getvalue()


def test_note_json_tolerates_finalize_suffix():
    assert RT._note_json('{"a": 1}') == {"a": 1}
    assert RT._note_json('{"a": 1} n_truncated=3') == {"a": 1}
    assert RT._note_json("not json") == {}
    assert RT._note_json(None) == {}


# ---------------------------------------------------------------------------
# the A/B axis derivation (shared engine.attention_path)
# ---------------------------------------------------------------------------
def test_attention_path_derivation():
    with _env(GLM_DSA_MODE=None):
        assert engine.attention_path() == "dense-mla"
    with _env(GLM_DSA_MODE="off"):
        assert engine.attention_path() == "dense-mla"
    with _env(GLM_DSA_MODE="pallas_decode"):
        assert engine.attention_path() == "dsa-sparse:pallas_decode"
    with _env(GLM_DSA_MODE="xla_ref"):
        assert engine.attention_path() == "dsa-sparse:xla_ref"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
