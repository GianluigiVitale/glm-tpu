"""Compile accounting schema checks, not TPU performance evidence."""
from copy import deepcopy

import pytest

from scripts.greenfield.seal_short_decoder_ws32 import _require_compile_timing


def record():
    return {
        "compile_seconds": {"prefill_chunk": 2.0, "prefill_tail": 0.0, "decode": 3.0},
        "graphs": {name: {"stablehlo_sha256": "a" * 64, "optimized_hlo_sha256": "b" * 64}
                   for name in ("prefill_chunk", "prefill_tail")},
        "compiled_memory_analysis": {name: {"temp_size_in_bytes": 1024}
                                     for name in ("prefill_chunk", "prefill_tail")},
    }


def check(value, reused=True):
    _require_compile_timing(value, expected_graphs={"prefill_chunk", "prefill_tail", "decode"},
                            rank=0, reused_e0_tail=reused)


def test_explicit_reused_tail_zero_is_additional_compile_time():
    value = record()
    original = deepcopy(value)
    check(value)
    assert value == original


def test_historical_positive_times_unchanged():
    value = record()
    value["compile_seconds"]["prefill_tail"] = 1.0
    check(value, reused=False)


def test_zero_rejected_outside_explicit_e0_reuse():
    with pytest.raises(SystemExit, match="compile timing"):
        check(record(), reused=False)


@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail", "decode"])
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), 0, False, "0", None])
def test_invalid_times_rejected(graph, bad):
    value = record()
    value["compile_seconds"][graph] = bad
    with pytest.raises(SystemExit, match="compile timing"):
        check(value)


@pytest.mark.parametrize("graph", ["prefill_chunk", "decode"])
def test_other_zero_times_rejected(graph):
    value = record()
    value["compile_seconds"][graph] = 0.0
    with pytest.raises(SystemExit, match="compile timing"):
        check(value)


@pytest.mark.parametrize("field", ["graphs", "compiled_memory_analysis"])
@pytest.mark.parametrize("mutation", ["different", "missing_tail", "missing_both", "empty", "wrong_type"])
def test_reuse_requires_matching_original_records(field, mutation):
    value = record()
    if mutation == "different": value[field]["prefill_tail"]["changed"] = True
    elif mutation == "missing_tail": del value[field]["prefill_tail"]
    elif mutation == "missing_both": del value[field]
    elif mutation == "empty": value[field] = {"prefill_chunk": {}, "prefill_tail": {}}
    else: value[field] = None
    with pytest.raises(SystemExit, match="compile timing"):
        check(value)


@pytest.mark.parametrize("mutation", ["missing", "extra", "none"])
def test_exact_compile_graph_set_required(mutation):
    value = record()
    if mutation == "missing": del value["compile_seconds"]["decode"]
    elif mutation == "extra": value["compile_seconds"]["extra"] = 1.0
    else: value["compile_seconds"] = None
    with pytest.raises(SystemExit, match="compile timing"):
        check(value)
