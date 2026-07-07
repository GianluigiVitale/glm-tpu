#!/usr/bin/env python3
"""CPU unit tests for the GLM-5.2 long-context passkey harness (glm_longctx).
Mock whitespace tokenizer — no model, no network, no vllm, no TPU.

Run: JAX_PLATFORMS=cpu ~/vllm-env/bin/python bench/test_longctx.py
 or: JAX_PLATFORMS=cpu ~/vllm-env/bin/python -m pytest bench/test_longctx.py -q
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_longctx as LC


def _tok():
    return LC._MockTok()   # whitespace: ~1 token per space-separated word


class _MergingTok:
    """Boundary-MERGING mock, mimicking the real BPE failure mode (round3
    review, finding 1): a sentence's trailing space tokenizes standalone as its
    own token but merges into the next sentence's first word when concatenated
    (ByteLevel 'Gword'), so per-sentence counts are NOT additive — summing
    standalone counts overshoots the true concatenated length by ~20%. The
    pre-fix per-sentence-sum targeting undershoots real length by the same
    factor and MUST fail the concatenated-length test below."""
    def __call__(self, s, add_special_tokens=False):
        toks = re.findall(r"\s*\S+", s)          # space attaches to NEXT word
        if s and s[-1].isspace():
            toks.append("<trail>")               # standalone-only trailing tok
        return {"input_ids": toks}


def _needle_frac(ctx: str) -> float:
    """Token-position fraction of the needle's first token (mock tokenizer)."""
    words = ctx.split()
    before = len(ctx[:ctx.index("Remember this.")].split())
    return before / len(words)


def test_token_length_targeting():
    tok = _tok()
    for L in [256, 1024, 4096, 32768]:
        ctx, key = LC.build_trial(tok, L, 0.5, seed=L)
        n = len(tok(ctx)["input_ids"])
        # measured on the CONCATENATED prompt: within max(one unit, 1% of L),
        # never above L
        assert L - max(8, L // 100) <= n <= L, f"L={L} got {n}"


def test_token_length_targeting_merging_tokenizer():
    """The round-3 regression guard: with a BPE-like boundary-MERGING tokenizer
    (per-sentence counts non-additive), the CONCATENATED prompt length must
    still land within 1% of the target. The old per-sentence-sum targeting
    produced ~80% of L here (-17.2% with the real GLM tokenizer) and fails."""
    tok = _MergingTok()
    # sanity: the mock really merges — standalone sums must overshoot concat
    standalone = sum(len(tok(u)["input_ids"]) for u in LC.FILLER_SENTENCES)
    concat = len(tok("".join(LC.FILLER_SENTENCES))["input_ids"])
    assert standalone > concat, (standalone, concat)
    for L in [1024, 4096, 32768]:
        ctx, _ = LC.build_trial(tok, L, 0.5, seed=L)
        n = len(tok(ctx)["input_ids"])
        assert L * 0.99 <= n <= L, f"L={L} got {n} ({100.0 * n / L:.1f}% of L)"
    # depth placement survives the merging tokenizer too (fractions are
    # preserved under uniform boundary-merge deflation)
    for d in [0.25, 0.5, 0.75]:
        ctx, _ = LC.build_trial(tok, 8192, d, seed=3)
        before = len(tok(ctx[:ctx.index("Remember this.")])["input_ids"])
        frac = before / len(tok(ctx)["input_ids"])
        assert abs(frac - d) <= 0.02, f"d={d} got {frac:.4f}"


def test_depth_placement_within_2pct():
    """The Stage-2 instrument bar: the passkey's token position lands within
    +-2% of the requested depth at every ladder length."""
    tok = _tok()
    for L in [1024, 4096, 16384]:
        for d in [0.1, 0.25, 0.5, 0.75, 0.9]:
            ctx, _ = LC.build_trial(tok, L, d, seed=7)
            frac = _needle_frac(ctx)
            assert abs(frac - d) <= 0.02, f"L={L} d={d} got {frac:.4f}"


def test_depth_monotonic_and_endpoints():
    tok = _tok()
    fracs = [_needle_frac(LC.build_trial(tok, 2048, d, seed=7)[0])
             for d in [0.0, 0.25, 0.5, 0.75, 1.0]]
    assert all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1)), fracs
    assert fracs[0] < 0.02 and fracs[-1] > 0.95, fracs


def test_needle_once_and_prompt_shape():
    tok = _tok()
    ctx, key = LC.build_trial(tok, 1024, 0.5, seed=1)
    assert len(key) == 6 and key.isdigit()               # always a 6-digit key
    assert f"passcode is {key}" in ctx                   # needle embedded
    assert ctx.count(key) == 1                           # appears exactly once
    assert ctx.endswith("The secret passcode is")       # ends at the raw-completion cue
    assert "<|user|>" not in ctx and "<think>" not in ctx  # raw completion, no chat markup


def test_determinism():
    tok = _tok()
    a = LC.build_trial(tok, 1024, 0.5, seed=99)
    b = LC.build_trial(tok, 1024, 0.5, seed=99)
    assert a == b                                        # same seed -> identical trial
    c = LC.build_trial(tok, 1024, 0.5, seed=100)
    assert c[1] != a[1]                                  # trial seeds vary the key


def test_extractor():
    x = LC.extract_passkey
    assert x(" 483920.") == "483920"                     # the expected greedy reply
    assert x("The secret passcode is 483920. Do not") == "483920"
    assert x("483,920 — remember it") == "483920"        # thousands-comma normalized
    assert x("<think>code 111111?</think> 222222") == "222222"  # think block stripped
    assert x("maybe 42, or 7") is None                   # short numbers never guessed
    assert x("1234567 is what I recall") == "1234567"    # long run NOT truncated —
    assert x("48392") == "48392"                          #   wrong-length runs are
    assert x("") is None and x(None) is None              #   returned as-is and fail
    # the exact-match scoring (extracted == 6-digit gold) honestly.


def test_raw_prompt_protocol():
    """RAW-protocol facts (round3 findings 2+4): [gMASK]<sop> ids are prepended
    EXPLICITLY (add_special_tokens is a no-op for the GLM tokenizer — plain
    ByteLevel post-processor) and <|assistant|> is a stop id (an emitted chat
    token must not burn the decode budget)."""
    import engine
    ids = LC._prompt_ids(_tok(), "hello world")
    assert ids[:2] == [154822, 154824] == LC.PROMPT_PREFIX_IDS  # [gMASK]<sop>
    assert len(ids) == 2 + 2                                    # prefix + words
    assert LC.ASSISTANT_ID == 154828
    assert LC.RAW_STOP_IDS == engine.EOS_IDS + [LC.ASSISTANT_ID]
    assert LC.ASSISTANT_ID not in engine.EOS_IDS  # generation_config unchanged


def test_parse_lengths():
    p = LC.parse_lengths
    assert p("1024,2048") == [1024, 2048]
    assert p("32k,128K") == [32 * 1024, 128 * 1024]
    # the 1M-endpoint cell: MAX_TARGET_LEN leaves the +256 auto max_len
    # headroom and a >=32-token answer budget inside max_position_embeddings
    assert LC.MAX_TARGET_LEN == LC.MAX_CONTEXT - 256 - 32 == 1048288
    assert p(str(LC.MAX_TARGET_LEN)) == [LC.MAX_TARGET_LEN]
    # a bare 1M target cannot fit its own answer -> rejected (round3 finding 3:
    # '--lengths 1M' used to crash at engine build with max_len 1048832 > 1M)
    for bad in ("1M", "2M", "0", "abc"):
        try:
            p(bad)
            raise AssertionError(f"{bad!r} should have raised")
        except ValueError:
            pass


def test_stub_pipeline_records_provenance():
    """End-to-end --stub run (no vllm import): every trial lands in the DB with
    prompt/gold/raw_output/correct; per-cell + aggregate summary rows exist."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        out = os.path.join(d, "res.json")
        rc = LC.main(["--stub", "--lengths", "256,512", "--depths", "0.5",
                      "--trials", "2", "--db", db, "--out-json", out])
        assert rc == 0
        assert "vllm" not in sys.modules                 # the offline contract
        conn = sqlite3.connect(db)
        items = conn.execute(
            "SELECT benchmark, prompt, gold, raw_output, correct, seed "
            "FROM items ORDER BY id").fetchall()
        assert len(items) == 4                            # 2 lengths x 1 depth x 2 trials
        assert {r[0] for r in items} == {"passkey_L256_d0.5", "passkey_L512_d0.5"}
        for bench, prompt, gold, raw, correct, seed in items:
            assert prompt.endswith("The secret passcode is")  # verbatim prompt stored
            assert len(gold) == 6 and gold.isdigit()
            assert raw == "" and correct == 0             # stub: empty reply, scored wrong
            assert seed is not None                       # reconstructible
        summ = conn.execute("SELECT benchmark, n, value FROM summary").fetchall()
        names = {r[0] for r in summ}
        assert names == {"passkey_L256_d0.5", "passkey_L512_d0.5", "longctx_passkey"}
        assert all(r[2] == 0.0 for r in summ)             # stub accuracy 0 everywhere
        res = json.load(open(out))
        assert res["overall_accuracy"] == 0.0 and len(res["grid"]) == 2
        assert res["prompt_mode"] == "stub"
        assert res["max_len"] == 512 + 256      # auto max_len = max(L) + 256


def test_prompt_storage_cap():
    ctx, _ = LC.build_trial(_tok(), 4096, 0.5, seed=3)
    assert LC._prompt_for_db(ctx, cap=10 ** 9) == ctx     # under cap: verbatim
    capped = LC._prompt_for_db(ctx, cap=1000)
    assert "sha256=" in capped and ctx[:2048] in capped and ctx[-2048:] in capped


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
