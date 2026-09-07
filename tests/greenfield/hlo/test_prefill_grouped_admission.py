"""CPU-only protocol tests for the bounded grouped TPU arithmetic probe."""

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.greenfield import probe_prefill_grouped_fp8 as probe


def test_cases_cover_sparse_shared_tiles_concentration_and_empty_owner():
    local_rows = []
    for name in probe.CASES:
        counts = probe.counts_for_case(name)
        assert counts.dtype == np.int32 and counts.shape == (256,)
        assert counts.sum() == 136 and (counts >= 0).all()
        local_rows.append(int(counts[64:96].sum()))
    assert local_rows == [32, 136, 0]
    with pytest.raises(ValueError):
        probe.counts_for_case("unknown")


def test_hlo_requires_raw_weight_call_and_no_collective_or_full_overlay():
    hlo = (
        "%call = f32[136,2048] custom-call(%x, u8[32,2048,1536] %w), "
        'custom_call_target="tpu_custom_call", name="greenfield_prefill_grouped_raw_fp8"'
    )
    assert probe.check_hlo(hlo)["passed"]
    for changed in (
        hlo + "\nx = f32[100663296] copy(y)",
        hlo + "\nx = bf16[1536,32,2048] transpose(y)",
        hlo + "\nx = f32[136,2048] all-reduce-start(y)",
        hlo + "\n" + hlo,
        hlo.replace("u8[32,2048,1536]", "bf16[32,2048,1536]"),
        "",
    ):
        assert not probe.check_hlo(changed)["passed"]


def valid_record():
    return dict(
        kernel=probe.KERNEL,
        protocol=probe.PROTOCOL,
        admission_only=True,
        baseline_only=False,
        performance_claim=False,
        latency=None,
        warmup=0,
        iterations=0,
        shape={"lhs": [136, 1536], "weights": [32, 2048, 1536], "output": [136, 2048]},
        dtype_contract={"output": "float32"},
        compiled_memory_estimate={"temp_size_in_bytes": 0},
        comparison={
            "passed": True,
            "cases": [
                dict(case=name, passed=True, bit_mismatches=0) for name in probe.CASES
            ],
        },
    )


def test_protocol_cannot_be_promoted_to_timing_or_partial_coverage():
    record = valid_record()
    probe.validate_record(record)
    for key, value in (
        ("latency", {"p50_ms": 0}),
        ("performance_claim", True),
        ("iterations", 1000),
        ("protocol", "other"),
        ("baseline_only", True),
        ("admission_only", False),
    ):
        changed = copy.deepcopy(record)
        changed[key] = value
        with pytest.raises(ValueError):
            probe.validate_record(changed)
    record["comparison"]["cases"].pop()
    with pytest.raises(ValueError):
        probe.validate_record(record)


def test_wrapper_reuses_bounded_guards_and_records_no_latency():
    source = (
        Path(__file__).resolve().parents[3]
        / "scripts/greenfield/run_fp8_matmul_microbench.sh"
    ).read_text()
    assert "$GROUPED_ADMISSION != 1 ]] || BOUNDED_PREFILL=1" in source
    assert "$KERNEL != ws32_grouped_down_admission &&" in source
    assert "ws32_prefill_moe_admission ]] || GROUPED_ADMISSION=1" in source
    assert source.count("if [[ $BOUNDED_PREFILL == 1 ]]; then") == 4
    assert (
        "[[ $WARMUP == 0 && $ITERATIONS == 0 && $DIAGNOSTIC_REFERENCE == 0 ]]" in source
    )
    assert "latency_ms=None if admission else" in source
    assert 'runner["kernel"] != expected_kernel' in source
    assert "validate_record(runner)" in source


def test_down_is_distinct_bf16_transposed_geometry_not_up_relabeling():
    kernel, protocol, k, n, dtype = probe.projection_contract("down")
    assert (k, n, dtype) == (2048, 1536, "bfloat16")
    record = valid_record()
    record.update(
        direction="down",
        kernel=kernel,
        protocol=protocol,
        shape={"lhs": [136, k], "weights": [32, n, k], "output": [136, n]},
        dtype_contract={"output": dtype},
    )
    probe.validate_record(record)
    for key, value in (
        ("kernel", probe.KERNEL),
        ("direction", "up"),
        ("dtype_contract", {"output": "float32"}),
    ):
        changed = copy.deepcopy(record)
        changed[key] = value
        with pytest.raises(ValueError):
            probe.validate_record(changed)
    hlo = (
        "%call = bf16[136,1536] custom-call(%x, u8[32,1536,2048] %w), "
        'custom_call_target="tpu_custom_call", name="greenfield_prefill_grouped_raw_fp8"'
    )
    assert probe.check_hlo(hlo, direction="down")["passed"]
    assert not probe.check_hlo(hlo, direction="up")["passed"]
    with pytest.raises(ValueError):
        probe.projection_contract("other")
