from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.dense_rms_replay import (
    DENSE_RMS_SOURCE_NPZ_SHA256,
    load_dense_rms_inputs,
    validate_strategy_nd_dense_rms_hlo,
)
from glm_tpu.greenfield.benchmarking.association_fingerprint import (
    ACCEPTED_DECODE_RESULT_LAYOUT,
    STRATEGY_ND_ALGORITHM,
)
from glm_tpu.greenfield.validation.strategy_nd_dense_rms_replay import (
    EXPECTED_SOURCE,
    _is_int_list,
    _recompute_comparison,
    _validate_capture_record,
)


REPO = Path(__file__).resolve().parents[3]
REAL_RMS_SOURCE = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)


def _synthetic_exact_dense_rms_hlo() -> str:
    collective_backend = json.dumps(
        {"collective_algorithm_config": STRATEGY_ND_ALGORITHM},
        separators=(",", ":"),
    )
    schedule_backend = json.dumps(
        {
            "megacore_config": {
                "megacore_allreduce_bytes": "4096",
                "megacore_split_dim": "0",
            },
            "window_config": {
                "cost_model_type": "COST_MODEL_TYPE_INVALID",
                "input_window_bounds": [],
                "is_mask": False,
                "iteration_bounds": ["2", "1"],
                "kernel_window_bounds": [],
                "output_window_bounds": ["2", "48"],
                "pad_input_on_minor_dim": "0",
                "pad_output_on_minor_dim": "0",
            },
        },
        separators=(",", ":"),
    )
    members = ",".join(str(value) for value in range(32))
    layout = ACCEPTED_DECODE_RESULT_LAYOUT
    return f'''HloModule exact_dense_rms, num_partitions=32

%bf16_add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %add = bf16[] add(%x, %y)
}}

%f32_add (x: f32[], y: f32[]) -> f32[] {{
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}}

%input_body (p: u16[1,32,6144]) -> bf16[32,6144] {{
  %p = u16[1,32,6144]{{2,1,0:T(8,128)(2,1)}} parameter(0)
  %bits = bf16[1,32,6144]{{2,1,0:T(8,128)(2,1)}} bitcast-convert(%p)
  ROOT %flat = bf16[32,6144]{layout} bitcast(%bits)
}}

%reduce_body (dense: bf16[32,6144], attention: bf16[1,6144], residual: bf16[1,6144]) -> f32[32] {{
  %dense = bf16[32,6144]{layout} parameter(0)
  %attention = bf16[1,6144]{{1,0}} parameter(1)
  %residual = bf16[1,6144]{{1,0}} parameter(2)
  %zero_bf16 = bf16[] constant(0)
  %attention_pad = bf16[32,6144]{{1,0}} pad(%attention, %zero_bf16), padding=0_31x0_0
  %residual_pad = bf16[32,6144]{{1,0}} pad(%residual, %zero_bf16), padding=0_31x0_0
  %attention_f32 = f32[32,6144]{{1,0}} convert(%attention_pad)
  %residual_f32 = f32[32,6144]{{1,0}} convert(%residual_pad)
  %residual_add = f32[32,6144]{{1,0}} add(%attention_f32, %residual_f32)
  %carried = bf16[32,6144]{{1,0}} convert(%residual_add)
  %carried_f32 = f32[32,6144]{{1,0}} convert(%carried)
  %dense_f32 = f32[32,6144]{{1,0}} convert(%dense)
  %summed = f32[32,6144]{{1,0}} add(%dense_f32, %carried_f32)
  %square = f32[32,6144]{{1,0}} multiply(%summed, %summed)
  %zero = f32[] constant(0)
  ROOT %reduce = f32[32]{{0}} reduce(%square, %zero), dimensions={{1}}, to_apply=%f32_add
}}

%rsqrt_body (reduced: f32[32]) -> f32[32] {{
  %reduced = f32[32]{{0}} parameter(0)
  %scale = f32[] constant(0.000162760422)
  %scale_wide = f32[32]{{0}} broadcast(%scale), dimensions={{}}
  %mean = f32[32]{{0}} multiply(%reduced, %scale_wide)
  %epsilon = f32[] constant(1e-05)
  %epsilon_wide = f32[32]{{0}} broadcast(%epsilon), dimensions={{}}
  %variance = f32[32]{{0}} add(%mean, %epsilon_wide)
  ROOT %inverse = f32[32]{{0}} rsqrt(%variance)
}}

%output_body (dense: bf16[32,6144], attention: bf16[1,6144], residual: bf16[1,6144], inverse: f32[32], weight: bf16[6144]) -> u16[1,6144] {{
  %dense = bf16[32,6144]{layout} parameter(0)
  %attention = bf16[1,6144]{{1,0}} parameter(1)
  %residual = bf16[1,6144]{{1,0}} parameter(2)
  %inverse = f32[32]{{0}} parameter(3)
  %weight = bf16[6144]{{0}} parameter(4)
  %zero_bf16 = bf16[] constant(0)
  %attention_pad = bf16[32,6144]{{1,0}} pad(%attention, %zero_bf16), padding=0_31x0_0
  %residual_pad = bf16[32,6144]{{1,0}} pad(%residual, %zero_bf16), padding=0_31x0_0
  %attention_f32 = f32[32,6144]{{1,0}} convert(%attention_pad)
  %residual_f32 = f32[32,6144]{{1,0}} convert(%residual_pad)
  %residual_add = f32[32,6144]{{1,0}} add(%attention_f32, %residual_f32)
  %carried = bf16[32,6144]{{1,0}} convert(%residual_add)
  %carried_f32 = f32[32,6144]{{1,0}} convert(%carried)
  %dense_f32 = f32[32,6144]{{1,0}} convert(%dense)
  %summed = f32[32,6144]{{1,0}} add(%dense_f32, %carried_f32)
  %inverse_wide = f32[32,6144]{{1,0}} broadcast(%inverse), dimensions={{0}}
  %normalized = f32[32,6144]{{1,0}} multiply(%summed, %inverse_wide)
  %normalized_bf16 = bf16[32,6144]{{1,0}} convert(%normalized)
  %normalized_f32 = f32[32,6144]{{1,0}} convert(%normalized_bf16)
  %weight_wide = bf16[32,6144]{{1,0}} broadcast(%weight), dimensions={{1}}
  %weight_f32 = f32[32,6144]{{1,0}} convert(%weight_wide)
  %weighted = f32[32,6144]{{1,0}} multiply(%normalized_f32, %weight_f32)
  %weighted_bf16 = bf16[32,6144]{{1,0}} convert(%weighted)
  %row = bf16[1,6144]{{1,0}} slice(%weighted_bf16), slice={{[0:1], [0:6144]}}
  ROOT %bits = u16[1,6144]{{1,0}} bitcast-convert(%row)
}}

ENTRY %main (p0: u16[1,32,6144], p1: bf16[1,6144], p2: bf16[1,6144], p3: bf16[6144]) -> u16[1,6144] {{
  %p0 = u16[1,32,6144]{{2,1,0:T(8,128)(2,1)}} parameter(0)
  %p1 = bf16[1,6144]{{1,0}} parameter(1)
  %p2 = bf16[1,6144]{{1,0}} parameter(2)
  %p3 = bf16[6144]{{0}} parameter(3)
  %input = bf16[32,6144]{layout} fusion(%p0), kind=kLoop, calls=%input_body
  %collective = bf16[32,6144]{layout} all-reduce(%input), channel_id=1, replica_groups={{{{{members}}}}}, use_global_device_ids=true, to_apply=%bf16_add, metadata={{op_name="strategy_nd_dense_rms_collective/psum"}}, backend_config={collective_backend}
  %scheduled = f32[32]{{0}} fusion(%collective, %p1, %p2), kind=kLoop, calls=%reduce_body, metadata={{op_name="strategy_nd_dense_rms_layer1/reduce_sum"}}, backend_config={schedule_backend}
  %inverse = f32[32]{{0}} fusion(%scheduled), kind=kLoop, calls=%rsqrt_body
  ROOT %output = u16[1,6144]{{1,0}} fusion(%collective, %p1, %p2, %inverse, %p3), kind=kLoop, calls=%output_body
}}
'''


def _synthetic_exact_direct_dense_rms_hlo() -> str:
    text = _synthetic_exact_dense_rms_hlo()
    reduce_start = text.index("%reduce_body ")
    reduce_end = text.index("\n%rsqrt_body", reduce_start)
    layout = ACCEPTED_DECODE_RESULT_LAYOUT
    reduce_body = f'''%reduce_body (dense: bf16[32,6144], residual: bf16[1,6144]) -> f32[32] {{
  %dense = bf16[32,6144]{layout} parameter(0)
  %residual = bf16[1,6144]{{1,0}} parameter(1)
  %zero_bf16 = bf16[] constant(0)
  %residual_pad = bf16[32,6144]{{1,0}} pad(%residual, %zero_bf16), padding=0_31x0_0
  %residual_f32 = f32[32,6144]{{1,0}} convert(%residual_pad)
  %dense_f32 = f32[32,6144]{{1,0}} convert(%dense)
  %summed = f32[32,6144]{{1,0}} add(%dense_f32, %residual_f32)
  %square = f32[32,6144]{{1,0}} multiply(%summed, %summed)
  %zero = f32[] constant(0)
  ROOT %reduce = f32[32]{{0}} reduce(%square, %zero), dimensions={{1}}, to_apply=%f32_add
}}
'''
    text = text[:reduce_start] + reduce_body + text[reduce_end:]
    output_start = text.index("%output_body ")
    output_end = text.index("\nENTRY %main", output_start)
    output_body = f'''%output_body (dense: bf16[32,6144], residual: bf16[1,6144], inverse: f32[32], weight: bf16[6144]) -> u16[1,6144] {{
  %dense = bf16[32,6144]{layout} parameter(0)
  %residual = bf16[1,6144]{{1,0}} parameter(1)
  %inverse = f32[32]{{0}} parameter(2)
  %weight = bf16[6144]{{0}} parameter(3)
  %zero_bf16 = bf16[] constant(0)
  %residual_pad = bf16[32,6144]{{1,0}} pad(%residual, %zero_bf16), padding=0_31x0_0
  %residual_f32 = f32[32,6144]{{1,0}} convert(%residual_pad)
  %dense_f32 = f32[32,6144]{{1,0}} convert(%dense)
  %summed = f32[32,6144]{{1,0}} add(%dense_f32, %residual_f32)
  %inverse_wide = f32[32,6144]{{1,0}} broadcast(%inverse), dimensions={{0}}
  %normalized = f32[32,6144]{{1,0}} multiply(%summed, %inverse_wide)
  %normalized_bf16 = bf16[32,6144]{{1,0}} convert(%normalized)
  %normalized_f32 = f32[32,6144]{{1,0}} convert(%normalized_bf16)
  %weight_wide = bf16[32,6144]{{1,0}} broadcast(%weight), dimensions={{1}}
  %weight_f32 = f32[32,6144]{{1,0}} convert(%weight_wide)
  %weighted = f32[32,6144]{{1,0}} multiply(%normalized_f32, %weight_f32)
  %weighted_bf16 = bf16[32,6144]{{1,0}} convert(%weighted)
  %row = bf16[1,6144]{{1,0}} slice(%weighted_bf16), slice={{[0:1], [0:6144]}}
  ROOT %bits = u16[1,6144]{{1,0}} bitcast-convert(%row)
}}
'''
    text = text[:output_start] + output_body + text[output_end:]
    entry_start = text.index("ENTRY %main")
    old_entry = text[entry_start:]
    collective_line = next(
        line for line in old_entry.splitlines() if " all-reduce(" in line
    )
    scheduled_backend = next(
        line for line in old_entry.splitlines() if " %scheduled = " in line
    ).split("backend_config=", 1)[1]
    entry = f'''ENTRY %main (p0: u16[1,32,6144], p1: bf16[1,6144], p2: bf16[6144]) -> u16[1,6144] {{
  %p0 = u16[1,32,6144]{{2,1,0:T(8,128)(2,1)}} parameter(0)
  %p1 = bf16[1,6144]{{1,0}} parameter(1)
  %p2 = bf16[6144]{{0}} parameter(2)
  %input = bf16[32,6144]{layout} fusion(%p0), kind=kLoop, calls=%input_body
{collective_line}
  %scheduled = f32[32]{{0}} fusion(%collective, %p1), kind=kLoop, calls=%reduce_body, metadata={{op_name="strategy_nd_dense_rms_layer1/reduce_sum"}}, backend_config={scheduled_backend}
  %inverse = f32[32]{{0}} fusion(%scheduled), kind=kLoop, calls=%rsqrt_body
  ROOT %output = u16[1,6144]{{1,0}} fusion(%collective, %p1, %inverse, %p2), kind=kLoop, calls=%output_body
}}
'''
    return text[:entry_start] + entry


def _synthetic_exact_dense_rms_tuple_hlo() -> str:
    text = _synthetic_exact_dense_rms_hlo()
    text = text.replace(
        "%reduce_body (dense: bf16[32,6144], attention: bf16[1,6144], "
        "residual: bf16[1,6144]) -> f32[32] {",
        "%reduce_body (dense: bf16[32,6144], attention: bf16[1,6144], "
        "residual: bf16[1,6144]) -> (f32[32], f32[32,6144]) {",
        1,
    ).replace(
        "ROOT %reduce = f32[32]{0} reduce(%square, %zero), dimensions={1}, "
        "to_apply=%f32_add",
        "%reduce = f32[32]{0} reduce(%square, %zero), dimensions={1}, "
        "to_apply=%f32_add\n"
        "  ROOT %reduce_tuple = (f32[32]{0}, f32[32,6144]{1,0}) "
        "tuple(%reduce, %summed)",
        1,
    )
    output_start = text.index("%output_body ")
    output_end = text.index("\nENTRY %main", output_start)
    output_body = '''%output_body (summed: f32[32,6144], inverse: f32[32], weight: bf16[6144]) -> u16[1,6144] {
  %summed = f32[32,6144]{1,0} parameter(0)
  %inverse = f32[32]{0} parameter(1)
  %weight = bf16[6144]{0} parameter(2)
  %inverse_wide = f32[32,6144]{1,0} broadcast(%inverse), dimensions={0}
  %normalized = f32[32,6144]{1,0} multiply(%summed, %inverse_wide)
  %normalized_bf16 = bf16[32,6144]{1,0} convert(%normalized)
  %normalized_f32 = f32[32,6144]{1,0} convert(%normalized_bf16)
  %weight_wide = bf16[32,6144]{1,0} broadcast(%weight), dimensions={1}
  %weight_f32 = f32[32,6144]{1,0} convert(%weight_wide)
  %weighted = f32[32,6144]{1,0} multiply(%normalized_f32, %weight_f32)
  %weighted_bf16 = bf16[32,6144]{1,0} convert(%weighted)
  %row = bf16[1,6144]{1,0} slice(%weighted_bf16), slice={[0:1], [0:6144]}
  ROOT %bits = u16[1,6144]{1,0} bitcast-convert(%row)
}
'''
    text = text[:output_start] + output_body + text[output_end:]
    text = text.replace(
        '%scheduled = f32[32]{0} fusion(%collective, %p1, %p2),',
        '%scheduled = (f32[32]{0}, f32[32,6144]{1,0}) '
        'fusion(%collective, %p1, %p2),',
        1,
    ).replace(
        '"megacore_split_dim":"0"', '"megacore_split_dim":"1"', 1
    ).replace(
        '"iteration_bounds":["2","1"]',
        '"iteration_bounds":["1","2"]',
        1,
    ).replace(
        '"output_window_bounds":["2","48"]',
        '"output_window_bounds":["4","24"]',
        1,
    ).replace(
        "  %inverse = f32[32]{0} fusion(%scheduled), kind=kLoop, calls=%rsqrt_body",
        "  %scheduled_reduction = f32[32]{0} get-tuple-element(%scheduled), index=0\n"
        "  %scheduled_sum = f32[32,6144]{1,0} get-tuple-element(%scheduled), index=1\n"
        "  %inverse = f32[32]{0} fusion(%scheduled_reduction), kind=kLoop, calls=%rsqrt_body",
        1,
    ).replace(
        "ROOT %output = u16[1,6144]{1,0} fusion(%collective, %p1, %p2, %inverse, %p3), kind=kLoop, calls=%output_body",
        "ROOT %output = u16[1,6144]{1,0} fusion(%scheduled_sum, %inverse, %p3), kind=kLoop, calls=%output_body",
        1,
    )
    return text


def _synthetic_exact_dense_rms_m1_hlo() -> str:
    text = _synthetic_exact_dense_rms_tuple_hlo()
    output_start = text.index("%output_body ")
    output_end = text.index("\nENTRY %main", output_start)
    output_body = '''%output_body (summed: f32[32,6144], inverse: f32[1], weight: bf16[6144]) -> u16[1,6144] {
  %summed = f32[32,6144]{1,0} parameter(0)
  %inverse = f32[1]{0:T(128)S(3)} parameter(1)
  %weight = bf16[6144]{0} parameter(2)
  %summed_row = f32[1,6144]{1,0} slice(%summed), slice={[0:1], [0:6144]}
  %inverse_wide = f32[1,6144]{1,0} broadcast(%inverse), dimensions={0}
  %normalized = f32[1,6144]{1,0} multiply(%summed_row, %inverse_wide)
  %normalized_bf16 = bf16[1,6144]{1,0} convert(%normalized)
  %normalized_f32 = f32[1,6144]{1,0} convert(%normalized_bf16)
  %weight_wide = bf16[1,6144]{1,0} broadcast(%weight), dimensions={1}
  %weight_f32 = f32[1,6144]{1,0} convert(%weight_wide)
  %weighted = f32[1,6144]{1,0} multiply(%normalized_f32, %weight_f32)
  %weighted_bf16 = bf16[1,6144]{1,0} convert(%weighted)
  ROOT %bits = u16[1,6144]{1,0} bitcast-convert(%weighted_bf16)
}
'''
    text = text[:output_start] + output_body + text[output_end:]
    text = text.replace(
        "%inverse = f32[32]{0} fusion(%scheduled_reduction), kind=kLoop, "
        "calls=%rsqrt_body",
        "%inverse = f32[32]{0:T(128)S(3)} fusion(%scheduled_reduction), "
        "kind=kLoop, calls=%rsqrt_body\n"
        "  %inverse_row = f32[1]{0:T(128)S(3)} bitcast(%inverse)",
        1,
    ).replace(
        "ROOT %output = u16[1,6144]{1,0} fusion(%scheduled_sum, %inverse, %p3), kind=kLoop, calls=%output_body",
        "ROOT %output = u16[1,6144]{1,0} fusion(%scheduled_sum, %inverse_row, %p3), kind=kLoop, calls=%output_body",
        1,
    )
    return text


@pytest.mark.skipif(not REAL_RMS_SOURCE.is_file(), reason="sealed RMS source absent")
def test_sealed_rms_source_loads_exact_target() -> None:
    inputs = load_dense_rms_inputs(REAL_RMS_SOURCE)
    assert inputs.post_attention_residual_bits.shape == (1, 6144)
    assert inputs.layer1_norm_bits.shape == (6144,)
    assert inputs.accepted_layer1_bits.shape == (6144,)
    assert int(inputs.accepted_layer1_bits[2795]) == 48423
    assert DENSE_RMS_SOURCE_NPZ_SHA256 == (
        "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
    )


def test_dense_rms_comparison_classifies_exact_control_and_new_result() -> None:
    expected = np.zeros((6144,), dtype=np.uint16)
    exact = _recompute_comparison(expected.copy(), expected)
    assert exact["classification"] == "global_strategy_nd_rms_exact_accepted"
    assert exact["elementwise_exact"] is True
    assert exact["mismatch_count"] == 0
    assert exact["first_mismatch_index"] is None

    observed = expected.copy()
    observed[2795] = np.uint16(1)
    nonexact = _recompute_comparison(observed, expected)
    assert nonexact["classification"] == "global_strategy_nd_rms_nonexact_new_result"
    assert nonexact["elementwise_exact"] is False
    assert nonexact["mismatch_count"] == 1
    assert nonexact["first_mismatch_index"] == 2795


def test_dense_rms_capture_recomputes_distributed_input_and_exact_types() -> None:
    physical = np.arange(32 * 6144, dtype=np.uint16).reshape(32, 6144)
    output = np.arange(6144, dtype=np.uint16)
    from glm_tpu.greenfield.benchmarking.association_fingerprint import array_sha256

    distributed = np.ascontiguousarray(
        np.broadcast_to(physical[:, None, :], (32, 32, 6144))
    )
    output_sha = array_sha256(output)
    capture = {
        "determinism_repeat_invocations": 1,
        "distributed_input_sha256": array_sha256(distributed),
        "invocation_count": 2,
        "local_replica_output_sha256": [output_sha] * 4,
        "output_bits_sha256": output_sha,
        "repeated_local_replica_output_sha256": [output_sha] * 4,
        "repeated_output_bits_sha256": output_sha,
    }
    _validate_capture_record(capture, physical, output)
    for key, value in (
        ("distributed_input_sha256", "0" * 64),
        ("determinism_repeat_invocations", True),
        ("invocation_count", True),
    ):
        mutation = dict(capture)
        mutation[key] = value
        with pytest.raises(ValueError, match="deterministic capture"):
            _validate_capture_record(mutation, physical, output)
    assert not _is_int_list([False, *range(1, 32)], list(range(32)))


def test_dense_rms_optimized_hlo_binds_exact_live_arithmetic() -> None:
    from jaxlib import xla_client

    accepted = _synthetic_exact_dense_rms_hlo()
    xla_client._xla.hlo_module_from_text(accepted)
    _, _, contract = validate_strategy_nd_dense_rms_hlo(
        accepted, tuple(range(32))
    )
    assert contract["exact_collective_input"] is True
    assert contract["exact_reduction_operand_graph"] is True
    assert contract["exact_weighted_operand_graph"] is True
    direct_accepted = _synthetic_exact_direct_dense_rms_hlo()
    xla_client._xla.hlo_module_from_text(direct_accepted)
    _, _, direct_contract = validate_strategy_nd_dense_rms_hlo(
        direct_accepted, tuple(range(32))
    )
    assert direct_contract["exact_direct_residual"] is True
    assert direct_contract["exact_residual_round"] is False
    assert direct_contract["residual_source_mode"] == (
        "direct_post_attention_residual"
    )
    for mutation in (
        direct_accepted.replace(
            "%summed = f32[32,6144]{1,0} add(%dense_f32, %residual_f32)",
            "%summed = f32[32,6144]{1,0} add(%dense_f32, %dense_f32)",
        ),
        direct_accepted.replace(
            "%residual_pad = bf16[32,6144]{1,0} pad(%residual, %zero_bf16), padding=0_31x0_0",
            "%residual_pad = bf16[32,6144]{1,0} pad(%residual, %zero_bf16), padding=31_0x0_0",
        ),
    ):
        xla_client._xla.hlo_module_from_text(mutation)
        with pytest.raises(Exception):
            validate_strategy_nd_dense_rms_hlo(mutation, tuple(range(32)))
    tuple_accepted = _synthetic_exact_dense_rms_tuple_hlo()
    xla_client._xla.hlo_module_from_text(tuple_accepted)
    _, _, tuple_contract = validate_strategy_nd_dense_rms_hlo(
        tuple_accepted, tuple(range(32))
    )
    assert tuple_contract["exact_result_binding"] is True
    m1_accepted = _synthetic_exact_dense_rms_m1_hlo()
    xla_client._xla.hlo_module_from_text(m1_accepted)
    _, _, m1_contract = validate_strategy_nd_dense_rms_hlo(
        m1_accepted, tuple(range(32))
    )
    assert m1_contract["exact_result_binding"] is True
    for mutation in (
        m1_accepted.replace(
            "%inverse_row = f32[1]{0:T(128)S(3)} bitcast(%inverse)",
            "%inverse_row = f32[1]{0} bitcast(%inverse)",
            1,
        ),
        m1_accepted.replace(
            "%inverse = f32[32]{0:T(128)S(3)} fusion(%scheduled_reduction)",
            "%inverse = f32[32]{0} fusion(%scheduled_reduction)",
            1,
        ).replace(
            "%inverse_row = f32[1]{0:T(128)S(3)} bitcast(%inverse)",
            "%inverse_row = f32[1]{0} bitcast(%inverse)",
            1,
        ),
    ):
        xla_client._xla.hlo_module_from_text(mutation)
        with pytest.raises(Exception):
            validate_strategy_nd_dense_rms_hlo(mutation, tuple(range(32)))
    mutations = {
        "arbitrary_schedule": accepted.replace(
            "ROOT %reduce = f32[32]{0} reduce(%square, %zero), "
            "dimensions={1}, to_apply=%f32_add",
            "ROOT %reduce = f32[32]{0} broadcast(%zero), dimensions={}",
            1,
        ),
        "rogue_input_layout": accepted.replace(
            "%bits = bf16[1,32,6144]{2,1,0:T(8,128)(2,1)} "
            "bitcast-convert(%p)",
            "%bits = bf16[1,32,6144]{1,2,0:T(8,128)(2,1)} "
            "bitcast-convert(%p)",
            1,
        ),
        "missing_residual_round": accepted.replace(
            "%carried = bf16[32,6144]{1,0} convert(%residual_add)",
            "%carried = bf16[32,6144]{1,0} convert(%residual_f32)",
            1,
        ),
        "rogue_normalized_arithmetic": accepted.replace(
            "%normalized = f32[32,6144]{1,0} multiply(%summed, %inverse_wide)",
            "%normalized = f32[32,6144]{1,0} add(%summed, %inverse_wide)",
            1,
        ),
        "rogue_live_result": accepted.replace(
            "%row = bf16[1,6144]{1,0} slice(%weighted_bf16), "
            "slice={[0:1], [0:6144]}",
            "%row0 = bf16[1,6144]{1,0} slice(%weighted_bf16), "
            "slice={[0:1], [0:6144]}\n"
            "  %row = bf16[1,6144]{1,0} add(%row0, %row0)",
            1,
        ),
    }
    for name, mutation in mutations.items():
        xla_client._xla.hlo_module_from_text(mutation)
        with pytest.raises(Exception):
            validate_strategy_nd_dense_rms_hlo(mutation, tuple(range(32)))


def test_dense_rms_stablehlo_binds_collective_residual_norm_and_result() -> None:
    code = f"""
import sys
import re
sys.path.insert(0, {str(REPO)!r})
from glm_tpu.greenfield.benchmarking.dense_rms_replay import build_strategy_nd_dense_rms_replay
from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import validate_strategy_nd_dense_rms_stablehlo

text = build_strategy_nd_dense_rms_replay(
    tuple(range(32)), enforce_optimized_hlo_contract=False
).stablehlo
assert validate_strategy_nd_dense_rms_stablehlo(text)["passed"]
all_reduce = next(line for line in text.splitlines() if '"stablehlo.all_reduce"' in line)
residual_add = next(
    line for line in text.splitlines()
    if "stablehlo.add" in line and "tensor<32x6144xf32>" in line
)
reducer_add = next(
    line for line in text.splitlines()
    if "stablehlo.add" in line and "tensor<bf16>" in line
)
live_slice = next(
    line for line in text.splitlines()
    if "stablehlo.slice" in line and "[0:1, 0:6144]" in line
)
residual_operands = re.search(
    r"stablehlo.add\\s+(%[A-Za-z0-9_.-]+),\\s*(%[A-Za-z0-9_.-]+)",
    residual_add,
).groups()
mutations = {{
    "wrong_group": text.replace(
        "30, 31]]> : tensor<1x32xi64>",
        "31, 30]]> : tensor<1x32xi64>",
        1,
    ),
    "wrong_reducer": text.replace(
        reducer_add,
        reducer_add.replace("stablehlo.add", "stablehlo.maximum", 1),
        1,
    ),
    "wrong_reducer_quoted_decoy": text.replace(
        reducer_add,
        reducer_add.replace("stablehlo.add", "stablehlo.maximum", 1)
        + ' loc("' + reducer_add.strip() + '")',
        1,
    ),
    "rogue_collective_input": text.replace(
        all_reduce,
        "      %rogue = stablehlo.add %2, %2 : tensor<32x6144xbf16>\\n"
        + all_reduce.replace("(%2)", "(%rogue)"),
        1,
    ),
    "duplicate_residual_source": text.replace(
        residual_add,
        residual_add.replace(
            f"{{residual_operands[0]}}, {{residual_operands[1]}}",
            f"{{residual_operands[0]}}, {{residual_operands[0]}}",
        ),
        1,
    ),
    "wrong_live_row": text.replace(
        live_slice,
        live_slice.replace("[0:1, 0:6144]", "[1:2, 0:6144]"),
        1,
    ),
    "wrong_outer_result": text.replace(
        "return %0 : tensor<1x6144xui16>",
        "return %arg0 : tensor<32x32x6144xui16>",
        1,
    ),
}}
for name, mutation in mutations.items():
    result = validate_strategy_nd_dense_rms_stablehlo(mutation)
    assert not result["passed"], (name, result)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(REPO),
            "XLA_FLAGS": "--xla_force_host_platform_device_count=32",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_protected_wrapper_keeps_rms_replay_default_off_and_success_last() -> None:
    wrapper = (
        REPO / "scripts/greenfield/run_strategy_nd_dense_replay.sh"
    ).read_text()
    assert "GLM_GREENFIELD_STRATEGY_ND_RMS_REPLAY:-0" in wrapper
    assert "--mode strategy_nd_dense_rms_replay" in wrapper
    assert '--association-rms-input "$run/source_rms/dense_partial_capture.npz"' in wrapper
    assert "validate_strategy_nd_dense_rms_replay" in wrapper
    assert 'evidence_dirs+=(source_rms)' in wrapper
    assert '"$REMOTE_PREFIX/$REPLAY_OUTPUT_DIR/*"' in wrapper
    assert wrapper.index("remote nonterminal object set drifted") < wrapper.index(
        '"$REMOTE_PREFIX/SUCCESS" >/dev/null'
    )


def test_protected_wrapper_success_heredoc_executes_both_modes(
    tmp_path: Path,
) -> None:
    wrapper = (
        REPO / "scripts/greenfield/run_strategy_nd_dense_replay.sh"
    ).read_text()
    marker = (
        '/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" '
        '"$RMS_REPLAY" <<\'PY\'\nfrom hashlib import sha256'
    )
    start = wrapper.index(marker) + marker.index("from hashlib")
    body = wrapper[start : wrapper.index("\nPY\n", start)]
    common = {
        "artifact_kind": "unit",
        "classification": "unit",
        "code_hash": "a" * 40,
    }
    summaries = (
        (
            "1",
            {
                **common,
                "elementwise_exact": True,
                "expected_hidden_2795_bfloat16_bits": 48423,
                "expected_raw_sha256": "a" * 64,
                "mismatch_count": 0,
                "observed_hidden_2795_bfloat16_bits": 48423,
                "observed_raw_sha256": "a" * 64,
                "optimized_hlo_sha256": "b" * 64,
                "source": EXPECTED_SOURCE,
                "stablehlo_sha256": "c" * 64,
                "topology_hash": "d" * 64,
            },
        ),
        (
            "0",
            {
                **common,
                "hardware_hidden_2795_bfloat16_bits": 47808,
                "hardware_row0_raw_sha256": "a" * 64,
                "row0_exact": True,
                "row0_mismatch_count": 0,
                "software_hidden_2795_bfloat16_bits": 47808,
                "software_row0_raw_sha256": "a" * 64,
                "source": {
                    "db_run_id": 550,
                    "npz_sha256": "b" * 64,
                    "success_sha256": "c" * 64,
                },
            },
        ),
    )
    for mode, summary in summaries:
        run_dir = tmp_path / mode
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps(summary))
        for name in ("evidence.sha256", "census_post.txt", "remote_objects.json"):
            (run_dir / name).write_text(name)
        completed = subprocess.run(
            [sys.executable, "-", str(run_dir), "gs://unit/result", mode],
            input=body,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        success = (run_dir / "SUCCESS").read_text()
        assert "classification=unit\n" in success
