from __future__ import annotations

import importlib.util
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import google_crc32c


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/greenfield/probe_layer0_isolated_dense.py"
WRAPPER = REPO / "scripts/greenfield/run_layer0_projection_reduction_probe.sh"
REAL_CAPTURE = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
REAL_DB548 = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_envelope_cross_layer_"
    "20260813T120703034434907Z/dense_envelope_cross_layer.npz"
)
REAL_ACCEPTED_M32 = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/"
    "decode_projection_lowering/8k/"
    "greenfield_accepted_decode_projection_lowering_20260811T184908676873350Z"
)
SPEC = importlib.util.spec_from_file_location("isolated_dense_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _stablehlo(*, bit_dtype: str = "jnp.uint8") -> str:
    program = f'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scripts.greenfield.probe_layer0_isolated_dense import _build_isolated_partial
mesh = Mesh(np.asarray(jax.devices()), ("lp4",))
rep = NamedSharding(mesh, P())
slot = NamedSharding(mesh, P("lp4", None, None, None))
contracts = (
    ((1,6144), jnp.bfloat16, rep),
    ((1,6144), jnp.bfloat16, rep),
    ((6144,), jnp.bfloat16, rep),
    ((4,1,6144,768), {bit_dtype}, slot),
    ((4,1,48,768), jnp.float32, slot),
    ((4,1,384,6144), {bit_dtype}, slot),
    ((4,1,3,6144), jnp.float32, slot),
)
arguments = tuple(
    jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    for shape, dtype, sharding in contracts
)
print(jax.jit(_build_isolated_partial(mesh)).lower(*arguments).as_text())
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _config(label: str) -> str:
    expected = (
        {
            "kernel_window_bounds": ["384", "6"],
            "input_window_bounds": ["4", "24"],
            "output_window_bounds": ["4", "6"],
            "iteration_bounds": ["1", "1", "2"],
            "megacore_split_dim": "2",
            "megacore_allreduce_bytes": "98304",
        }
        if label == "gate"
        else {
            "kernel_window_bounds": ["48", "6"],
            "input_window_bounds": ["4", "3"],
            "output_window_bounds": ["4", "6"],
            "iteration_bounds": ["8", "1", "1"],
            "megacore_split_dim": "0",
            "megacore_allreduce_bytes": None,
        }
    )
    return json.dumps(
        {
            "convolution_algorithm_config": {
                "emitter": "EmitAllBatchInSublanes"
            },
            "window_config": {
                "kernel_window_bounds": expected["kernel_window_bounds"],
                "input_window_bounds": expected["input_window_bounds"],
                "output_window_bounds": expected["output_window_bounds"],
                "iteration_bounds": expected["iteration_bounds"],
                "cost_model_type": "COST_MODEL_TYPE_CLASSIC",
                "is_mask": False,
                "pad_input_on_minor_dim": "0",
                "pad_output_on_minor_dim": "0",
            },
            "megacore_config": {
                "megacore_split_dim": expected["megacore_split_dim"],
                "megacore_allreduce_bytes": expected[
                    "megacore_allreduce_bytes"
                ],
            },
        },
        separators=(",", ":"),
    )


def _optimized_hlo(
    *,
    external_gate_singleton: bool = True,
    fused_gate_dequant: bool = True,
) -> str:
    gate_computation = ""
    gate_decode_entry = '''  %gate_decoded = f32[6144,768]{1,0} convert(%gate_bits_flat), metadata={op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}
  %gate_scaled = f32[6144,768]{1,0} multiply(%gate_decoded, %gate_scale_wide), metadata={op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}
  %gate_rhs = bf16[6144,768]{1,0} convert(%gate_scaled), metadata={op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}
'''
    if external_gate_singleton and fused_gate_dequant:
        gate_computation = '''gate_bits_copy_fusion {
  %gate_bits_copy_parameter = f8e4m3fn[6144,768]{1,0} parameter(0)
  ROOT %gate_bits_copy_root = f8e4m3fn[6144,768]{1,0} copy(%gate_bits_copy_parameter)
}

gate_dequant_fusion {
  %gate_scale_inner = f32[6144,768]{1,0} parameter(0)
  %gate_bits_inner = f8e4m3fn[6144,768]{1,0} parameter(1)
  %gate_bits_copied_inner = f8e4m3fn[6144,768]{1,0} fusion(%gate_bits_inner), kind=kLoop, calls=%gate_bits_copy_fusion
  %gate_decoded_inner = f32[6144,768]{1,0} convert(%gate_bits_copied_inner), metadata={op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}
  %gate_scaled_inner = f32[6144,768]{1,0} multiply(%gate_scale_inner, %gate_decoded_inner), metadata={op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}
  ROOT %gate_rhs_inner = bf16[6144,768]{1,0} convert(%gate_scaled_inner), metadata={op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}
}

gate_fusion {
  %gate_lhs = bf16[32,6144]{1,0} parameter(0)
  %gate_scale_parameter = f32[6144,768]{1,0} parameter(1)
  %gate_bits_parameter = f8e4m3fn[6144,768]{1,0} parameter(2)
  %gate_rhs_internal = bf16[6144,768]{1,0} fusion(%gate_scale_parameter, %gate_bits_parameter), kind=kLoop, calls=%gate_dequant_fusion
  %gate_dot = f32[32,768]{1,0} convolution(%gate_lhs, %gate_rhs_internal), dim_labels=bf_io->bf, metadata={op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/conv_general_dilated"}
  %gate_round = bf16[32,768]{1,0} convert(%gate_dot)
  ROOT %gate_singleton_root = bf16[32,1,768]{2,0,1} reshape(%gate_round)
}
'''
        gate_execution = (
            '  %gate_singleton = bf16[32,1,768]{2,0,1} '
            'fusion(%carried, %gate_scale_wide, %gate_bits_flat), '
            'kind=kOutput, calls=%gate_fusion, '
            'metadata={op_name="jit/local/'
            'greenfield_dense_convolution_virtual_rank_00/'
            'conv_general_dilated"}, backend_config='
            + _config("gate")
        )
        gate_decode_entry = ""
    elif external_gate_singleton:
        gate_computation = '''gate_fusion {
  %gate_lhs = bf16[32,6144]{1,0} parameter(0)
  %gate_weight = bf16[6144,768]{1,0} parameter(1)
  %gate_dot = f32[32,768]{1,0} convolution(%gate_lhs, %gate_weight), dim_labels=bf_io->bf, metadata={op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/conv_general_dilated"}
  %gate_round = bf16[32,768]{1,0} convert(%gate_dot)
  ROOT %gate_singleton_root = bf16[32,1,768]{2,0,1} reshape(%gate_round)
}
'''
        gate_execution = (
            '  %gate_singleton = bf16[32,1,768]{2,0,1} '
            'fusion(%carried, %gate_rhs), kind=kOutput, calls=%gate_fusion, '
            'metadata={op_name="jit/local/'
            'greenfield_dense_convolution_virtual_rank_00/'
            'conv_general_dilated"}, backend_config='
            + _config("gate")
        )
    else:
        gate_execution = (
            '  %gate = f32[32,768]{1,0} convolution(%carried, %gate_rhs), '
            'dim_labels=bf_io->bf, metadata={op_name="jit/local/'
            'greenfield_dense_convolution_virtual_rank_00/'
            'conv_general_dilated"}, backend_config='
            + _config("gate")
            + '\n  %gate_round = bf16[32,768]{1,0} convert(%gate)'
            + '\n  %gate_singleton = bf16[32,1,768]{2,0,1} '
            'reshape(%gate_round)'
        )
    return f'''HloModule isolated_dense, is_scheduled=true, replica_count=1, num_partitions=4

{gate_computation}

ENTRY main {{
  %attention = bf16[1,6144]{{1,0}} parameter(0)
  %residual = bf16[1,6144]{{1,0}} parameter(1)
  %norm = bf16[6144]{{0}} parameter(2)
  %zero = bf16[] constant(0)
  %attention_pad = bf16[32,6144]{{1,0}} pad(%attention, %zero), padding=0_31x0_0
  %residual_pad = bf16[32,6144]{{1,0}} pad(%residual, %zero), padding=0_31x0_0
  %attention_wide = f32[32,6144]{{1,0}} convert(%attention_pad)
  %residual_wide = f32[32,6144]{{1,0}} convert(%residual_pad)
  %combined = f32[32,6144]{{1,0}} add(%attention_wide, %residual_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_predense_rmsnorm/add"}}
  %carried = bf16[32,6144]{{1,0}} convert(%combined), metadata={{op_name="jit/local/greenfield_dense_convolution_predense_rmsnorm/convert_element_type"}}
  %gate_bits = f8e4m3fn[1,1,6144,768]{{3,2,1,0}} parameter(3)
  %gate_scale = f32[1,1,48,768]{{3,2,1,0}} parameter(4)
  %down_bits = f8e4m3fn[1,1,384,6144]{{3,2,1,0}} parameter(5)
  %down_scale = f32[1,1,3,6144]{{3,2,1,0}} parameter(6)
  %gate_bits_slice = f8e4m3fn[1,1,6144,768]{{3,2,1,0}} slice(%gate_bits), slice={{[0:1],[0:1],[0:6144],[0:768]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %gate_bits_flat = f8e4m3fn[6144,768]{{1,0}} reshape(%gate_bits_slice), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %gate_scale_slice = f32[1,1,48,768]{{3,2,1,0}} slice(%gate_scale), slice={{[0:1],[0:1],[0:48],[0:768]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %gate_scale_seed = f32[48,768]{{1,0}} reshape(%gate_scale_slice), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %gate_scale_expanded = f32[48,128,768]{{2,1,0}} broadcast(%gate_scale_seed), dimensions={{0,2}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/broadcast_in_dim"}}
  %gate_scale_wide = f32[6144,768]{{1,0}} reshape(%gate_scale_expanded), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
{gate_decode_entry}
{gate_execution}
  %gate_slice_rank3 = bf16[32,1,384]{{2,0,1}} slice(%gate_singleton), slice={{[0:32],[0:1],[0:384]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %up_slice_rank3 = bf16[32,1,384]{{2,0,1}} slice(%gate_singleton), slice={{[0:32],[0:1],[384:768]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %gate_slice = bf16[32,384]{{1,0}} reshape(%gate_slice_rank3)
  %up_slice = bf16[32,384]{{1,0}} reshape(%up_slice_rank3)
  %gate_wide = f32[32,384]{{1,0}} convert(%gate_slice)
  %up_wide = f32[32,384]{{1,0}} convert(%up_slice)
  %one = bf16[] constant(1)
  %one_broadcast = bf16[32,384]{{1,0}} broadcast(%one), dimensions={{}}
  %one_wide = f32[32,384]{{1,0}} convert(%one_broadcast)
  %negate = f32[32,384]{{1,0}} negate(%gate_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/neg"}}
  %exponential = f32[32,384]{{1,0}} exponential(%negate), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/exp"}}
  %denominator = f32[32,384]{{1,0}} add(%exponential, %one_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/add"}}
  %sigmoid = f32[32,384]{{1,0}} divide(%one_wide, %denominator), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/div"}}
  %silu = f32[32,384]{{1,0}} multiply(%gate_wide, %sigmoid), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}}, backend_config={{"float_type_correction_info":{{"original_type":"BF16"}}}}
  %activated = f32[32,384]{{1,0}} multiply(%silu, %up_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}}, backend_config={{"float_type_correction_info":{{"original_type":"BF16"}}}}
  %activated_round = bf16[32,384]{{1,0}} convert(%activated)
  %down_bits_slice = f8e4m3fn[1,1,384,6144]{{3,2,1,0}} slice(%down_bits), slice={{[0:1],[0:1],[0:384],[0:6144]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %down_bits_flat = f8e4m3fn[384,6144]{{1,0}} reshape(%down_bits_slice), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %down_decoded = f32[384,6144]{{1,0}} convert(%down_bits_flat), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}}
  %down_scale_slice = f32[1,1,3,6144]{{3,2,1,0}} slice(%down_scale), slice={{[0:1],[0:1],[0:3],[0:6144]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %down_scale_seed = f32[3,6144]{{1,0}} reshape(%down_scale_slice), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %down_scale_expanded = f32[3,128,6144]{{2,1,0}} broadcast(%down_scale_seed), dimensions={{0,2}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/broadcast_in_dim"}}
  %down_scale_wide = f32[384,6144]{{1,0}} reshape(%down_scale_expanded), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %down_scaled = f32[384,6144]{{1,0}} multiply(%down_decoded, %down_scale_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}}
  %down_rhs = bf16[384,6144]{{1,0}} convert(%down_scaled), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}}
  %down = f32[32,6144]{{1,0}} convolution(%activated_round, %down_rhs), dim_labels=bf_io->bf, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/conv_general_dilated"}}, backend_config={_config("down")}
  %partial = bf16[32,6144]{{1,0}} convert(%down)
  %stack = bf16[1,32,6144]{{2,1,0}} reshape(%partial)
  %barrier = bf16[1,32,6144]{{2,1,0}} optimization-barrier(%stack)
  %live = bf16[1,1,6144]{{2,1,0}} slice(%barrier), slice={{[0:1],[0:1],[0:6144]}}
  ROOT %result = (bf16[1,1,6144]{{2,1,0}}, bf16[32,6144]{{1,0}}) tuple(%live, %carried)
}}
'''


def test_isolated_dense_stablehlo_contract_accepts_u8_and_fp8() -> None:
    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_isolated_dense_partial_stablehlo,
    )

    for bit_dtype, expected_bitcasts in (
        ("jnp.uint8", 2),
        ("jnp.float8_e4m3fn", 0),
    ):
        stablehlo = _stablehlo(bit_dtype=bit_dtype)
        result = validate_isolated_dense_partial_stablehlo(
            stablehlo,
            accepted_gate_singleton=True,
            accepted_gate_dequant_fusion=True,
        )
        assert result["passed"], result
        assert result["accepted_gate_dequant_fusion"] is True
        assert result["accepted_gate_singleton"] is True
        assert result["convolution_count"] == 2
        assert result["matched_virtual_shards"] == [0]
        assert result["runtime_u8_bitcast_count"] == expected_bitcasts
        assert result["collective_counts"] == {
            "all_gather": 0,
            "all_reduce": 0,
            "all_to_all": 0,
            "collective_broadcast": 0,
            "collective_permute": 0,
            "reduce_scatter": 0,
        }


def test_isolated_dense_stablehlo_contract_refuses_wrong_arithmetic() -> None:
    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_isolated_dense_partial_stablehlo,
    )

    stablehlo = _stablehlo()
    mutations = (
        stablehlo.replace("applies stablehlo.add", "applies stablehlo.maximum", 1),
        stablehlo.replace("[0:1, 0:1, 0:6144]", "[0:1, 1:2, 0:6144]", 1),
        stablehlo.replace("stablehlo.multiply %47, %40", "stablehlo.add %47, %40", 1),
        stablehlo.replace("sdy.return %60, %6", "sdy.return %60, %20", 1),
    )
    assert all(value != stablehlo for value in mutations)
    assert all(
        not validate_isolated_dense_partial_stablehlo(
            value,
            accepted_gate_singleton=True,
            accepted_gate_dequant_fusion=True,
        )["passed"]
        for value in mutations
    )


def test_isolated_dense_optimized_contract_pins_schedule_and_liveness() -> None:
    accepted = _optimized_hlo()
    result = MODULE._validate_isolated_optimized_hlo(accepted)
    assert result["passed"], result
    assert result["accepted_gate_dequant_fusion"] is True
    assert result["accepted_gate_singleton"] is True
    assert result["exact_activation_graph"]
    assert result["exact_carried_residual_binding"]
    assert result["exact_gate_singleton_external_boundary"]
    assert result["exact_gate_dequant_fusion_boundary"]
    assert result["exact_packed_weight_lineage"]
    assert result["exact_result_binding"]
    direct_fp8 = accepted.replace(
        '''gate_bits_copy_fusion {
  %gate_bits_copy_parameter = f8e4m3fn[6144,768]{1,0} parameter(0)
  ROOT %gate_bits_copy_root = f8e4m3fn[6144,768]{1,0} copy(%gate_bits_copy_parameter)
}

''',
        "",
        1,
    ).replace(
        "  %gate_bits_copied_inner = f8e4m3fn[6144,768]{1,0} fusion(%gate_bits_inner), kind=kLoop, calls=%gate_bits_copy_fusion\n",
        "",
        1,
    ).replace(
        "convert(%gate_bits_copied_inner)",
        "convert(%gate_bits_inner)",
        1,
    )
    direct_fp8_result = MODULE._validate_isolated_optimized_hlo(direct_fp8)
    assert direct_fp8_result["passed"], direct_fp8_result
    assert direct_fp8_result["exact_gate_dequant_fusion_boundary"]
    direct_singleton = MODULE._validate_isolated_optimized_hlo(
        _optimized_hlo(external_gate_singleton=False)
    )
    assert not direct_singleton["passed"]
    assert not direct_singleton[
        "exact_gate_singleton_external_boundary"
    ]
    materialized_gate_rhs = MODULE._validate_isolated_optimized_hlo(
        _optimized_hlo(fused_gate_dequant=False)
    )
    assert materialized_gate_rhs[
        "exact_gate_singleton_external_boundary"
    ]
    assert not materialized_gate_rhs["passed"]
    assert not materialized_gate_rhs[
        "exact_gate_dequant_fusion_boundary"
    ]
    mutations = (
        accepted.replace(
            "%gate_scale_inner = f32[6144,768]{1,0}",
            "%gate_scale_inner = f32[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "%gate_bits_inner = f8e4m3fn[6144,768]{1,0}",
            "%gate_bits_inner = f8e4m3fn[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "%gate_scale_parameter = f32[6144,768]{1,0}",
            "%gate_scale_parameter = f32[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "%gate_bits_parameter = f8e4m3fn[6144,768]{1,0}",
            "%gate_bits_parameter = f8e4m3fn[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "%gate_decoded_inner = f32[6144,768]{1,0}",
            "%gate_decoded_inner = f32[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "%gate_scaled_inner = f32[6144,768]{1,0}",
            "%gate_scaled_inner = f32[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "ROOT %gate_rhs_inner = bf16[6144,768]{1,0}",
            "ROOT %gate_rhs_inner = bf16[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "kind=kLoop, calls=%gate_dequant_fusion",
            "kind=kInput, calls=%gate_dequant_fusion",
            1,
        ),
        accepted.replace(
            "kind=kLoop, calls=%gate_dequant_fusion",
            "kind=kOutput, calls=%gate_dequant_fusion",
            1,
        ),
        accepted.replace(
            "%gate_bits_copy_parameter = f8e4m3fn[6144,768]{1,0}",
            "%gate_bits_copy_parameter = f8e4m3fn[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "%gate_bits_copy_root = f8e4m3fn[6144,768]{1,0}",
            "%gate_bits_copy_root = f8e4m3fn[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "%gate_bits_copied_inner = f8e4m3fn[6144,768]{1,0}",
            "%gate_bits_copied_inner = f8e4m3fn[6144,768]{0,1}",
            1,
        ),
        accepted.replace(
            "kind=kLoop, calls=%gate_bits_copy_fusion",
            "kind=kInput, calls=%gate_bits_copy_fusion",
            1,
        ),
        accepted.replace(
            "copy(%gate_bits_copy_parameter)",
            "negate(%gate_bits_copy_parameter)",
            1,
        ),
        accepted.replace('"iteration_bounds":["1","1","2"]', '"iteration_bounds":["1","1","3"]', 1),
        accepted.replace(
            "  %down = f32[32,6144]",
            "  %rogue_activation = bf16[32,384]{1,0} add(%activated_round, %activated_round)\n"
            "  %down = f32[32,6144]",
            1,
        ).replace("convolution(%activated_round, %down_rhs)", "convolution(%rogue_activation, %down_rhs)", 1),
        accepted.replace(
            "  %partial = bf16[32,6144]",
            "  %rogue_down = f32[32,6144]{1,0} add(%down, %down)\n"
            "  %partial = bf16[32,6144]",
            1,
        ).replace("convert(%down)", "convert(%rogue_down)", 1),
        accepted.replace(
            "  ROOT %result",
            "  %rogue_live = bf16[1,1,6144]{2,1,0} add(%live, %live)\n"
            "  ROOT %result",
            1,
        ).replace("tuple(%live, %carried)", "tuple(%rogue_live, %carried)", 1),
        accepted.replace(
            "(bf16[1,1,6144]{2,1,0}, bf16[32,6144]{1,0}) tuple(%live, %carried)",
            "(bf16[1,1,6144]{2,1,0}, bf16[32,6144]{1,0}, bf16[32,6144]{1,0}) tuple(%live, %carried, %carried)",
            1,
        ),
        accepted.replace(
            "%gate_rhs_internal = bf16[6144,768]{1,0} fusion",
            "%gate_rhs_internal = bf16[6144,768]{0,1} fusion",
            1,
        ),
        accepted.replace(
            "  %gate_dot = f32[32,768]{1,0} convolution",
            "  %rogue_gate_weight = bf16[6144,768]{1,0} "
            "add(%gate_rhs_internal, %gate_rhs_internal)\n"
            "  %gate_dot = f32[32,768]{1,0} convolution",
            1,
        ).replace(
            "convolution(%gate_lhs, %gate_rhs_internal)",
            "convolution(%gate_lhs, %rogue_gate_weight)",
            1,
        ),
        accepted.replace(
            "%down_rhs = bf16[384,6144]{1,0} convert",
            "%down_rhs = bf16[384,6144]{0,1} convert",
            1,
        ),
        accepted.replace(
            "  %down = f32[32,6144]",
            "  %rogue_down_rhs = bf16[384,6144]{1,0} add(%down_rhs, %down_rhs)\n"
            "  %down = f32[32,6144]",
            1,
        ).replace("convolution(%activated_round, %down_rhs)", "convolution(%activated_round, %rogue_down_rhs)", 1),
        accepted.replace(
            "%live = bf16[1,1,6144]{2,1,0} slice(%barrier), slice={[0:1],[0:1],[0:6144]}",
            "%live = bf16[1,1,6144]{2,1,0} slice(%barrier), slice={[0:1],[1:2],[0:6144]}",
            1,
        ),
        accepted.replace(
            "tuple(%live, %carried)",
            "tuple(%live, %attention_pad)",
            1,
        ),
        accepted.replace(
            "%zero = bf16[] constant(0)",
            '%zero = bf16[] constant(1), metadata={op_name="constant(0)"}',
            1,
        ),
        accepted.replace(
            "%one = bf16[] constant(1)",
            '%one = bf16[] constant(0), metadata={op_name="constant(1)"}',
            1,
        ),
        accepted.replace(
            "  ROOT %result",
            "  %rogue_carried = bf16[32,6144]{1,0} add(%carried, %carried)\n"
            "  ROOT %result",
            1,
        ).replace(
            "tuple(%live, %carried)",
            "tuple(%live, %rogue_carried)",
            1,
        ),
        accepted.replace(
            "  ROOT %result",
            "  %extra = bf16[32,6144]{1,0} all-reduce(%partial), replica_groups={{0,1,2,3}}, to_apply=%add\n"
            "  ROOT %result",
            1,
        ),
    )
    assert all(value != accepted for value in mutations)
    accepted_mutations = [
        index
        for index, value in enumerate(mutations)
        if MODULE._validate_isolated_optimized_hlo(value)["passed"]
    ]
    assert not accepted_mutations, accepted_mutations


def test_isolated_partial_comparison_reports_each_virtual_rank() -> None:
    reference = np.zeros((4, 8, 1, 6144), dtype=np.uint16)
    observed = reference.copy()
    observed[2, 5, 0, 2795] = 1
    result = MODULE._partial_comparison(reference, observed)
    assert not result["elementwise_exact"]
    assert result["mismatch_count"] == 1
    assert result["first_mismatch_index"] == [2, 5, 0, 2795]
    assert result["per_virtual_rank"][5] == {
        "mismatch_count": 1,
        "virtual_rank": 5,
    }


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(),
    reason="sealed isolated-dense source is unavailable",
)
def test_isolated_dense_sensitivity_deduplicates_reduced_values() -> None:
    with np.load(REAL_CAPTURE, allow_pickle=False) as payload:
        captured = np.ascontiguousarray(
            payload["dense_virtual_partials_bfloat16_bits"]
        )
    candidates = MODULE._build_sensitivity_candidates(captured)
    assert [item["candidate_id"] for item in candidates] == list(
        range(len(candidates))
    )
    assert [item["dense_update_bits"] for item in candidates] == list(
        range(47802, 47817)
    )
    baseline = next(
        item for item in candidates if item["dense_update_bits"] == 47808
    )
    assert baseline["representative_model_rank"] == -1
    assert baseline["representative_partial_bit_delta"] == 0
    plus_one = next(
        item for item in candidates if item["dense_update_bits"] == 47809
    )
    assert plus_one["minimum_delta_producers"] == [
        {"model_rank": 25, "partial_bit_delta": 5}
    ]


def test_isolated_dense_wrapper_is_default_off_and_no_db() -> None:
    wrapper = WRAPPER.read_text()
    assert "GLM_GREENFIELD_ISOLATED_DENSE_REPLAY:-0" in wrapper
    assert "probe_layer0_isolated_dense.py" in wrapper
    assert '--accepted-m32-hlo "$ACCEPTED_M32_HLO"' in wrapper
    assert '--accepted-m32-summary "$ACCEPTED_M32_SUMMARY"' in wrapper
    assert '--accepted-m32-success "$ACCEPTED_M32_SUCCESS"' in wrapper
    assert "bounded replays are exclusive" in wrapper
    assert wrapper.index("validate_isolated_dense_replay()") < wrapper.index(
        'if [[ $ISOLATED_DENSE_REPLAY == 1 ]]; then\n'
        "  validate_isolated_dense_replay"
    )
    rollback = wrapper[wrapper.index("rollback_provisional_db()") :]
    assert (
        'if [[ $CAPTURED_RMS_REPLAY == 1 || '
        '$ISOLATED_DENSE_REPLAY == 1 ]]; then' in rollback
    )
    assert "NO_PROVISIONAL_DB_RUN" in rollback


@pytest.mark.skipif(
    not REAL_ACCEPTED_M32.exists()
    or not (
        REAL_DB548.parent / "hlo/dense_convolution.optimized_hlo.txt"
    ).exists(),
    reason="pinned accepted/DB548 HLO sources are unavailable",
)
def test_gate_singleton_discriminator_is_bound_to_pinned_hlo() -> None:
    result, source = MODULE._validate_gate_singleton_discriminator(
        SimpleNamespace(
            accepted_m32_hlo=(
                REAL_ACCEPTED_M32
                / "accepted_decode_projection_lowering/"
                "jit_step_fun_impl.m32.after_codegen_hlo.txt.gz"
            ),
            accepted_m32_hlo_sha256=(
                "25041bfbcf319b6c6fc4c5888cb22548b246cccba784791796fe9e8f57199e4c"
            ),
            accepted_m32_summary=(
                REAL_ACCEPTED_M32
                / "accepted_decode_projection_lowering/summary.json"
            ),
            accepted_m32_summary_sha256=(
                "409c845c2c9d67a1d6de36f0cccd25d2982850ee86c35645b0839fc78a1507a3"
            ),
            accepted_m32_success=REAL_ACCEPTED_M32 / "SUCCESS",
            accepted_m32_success_sha256=(
                "6ef516dc42e046a996aa1fe542a4b11af5c2a14450e1aba7bfac98ddbf257278"
            ),
            db548_hlo=(
                REAL_DB548.parent / "hlo/dense_convolution.optimized_hlo.txt"
            ),
            db548_hlo_sha256=(
                "5f4dd83793da67be6a8c580949920e93f8c64fe8205816738e7d04890640a877"
            ),
        )
    )
    assert result == {
        "accepted_internal_gate_dequant_count": 3,
        "accepted_rank2_gate_count": 0,
        "accepted_rank3_gate_count": 3,
        "challenger_selected": True,
        "db548_materialized_bf16_gate_count": 8,
        "db548_rank2_gate_count": 8,
        "db548_rank3_gate_count": 0,
    }
    assert source["accepted_m32_hlo_raw_sha256"] == (
        "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775"
    )


@pytest.mark.skipif(
    not REAL_CAPTURE.exists() or not REAL_DB548.exists(),
    reason="sealed isolated-dense sources are unavailable",
)
def test_isolated_dense_wrapper_validates_exact_synthetic_result(
    tmp_path: Path,
) -> None:
    import re

    programs = re.findall(
        r"<<'PY'\n(.*?)\nPY\n", WRAPPER.read_text(), re.DOTALL
    )
    record_program = next(
        value for value in programs if "ISOLATED_DENSE_REPLAY_VALID" in value
    )
    run_dir = tmp_path / "isolated"
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir(parents=True)
    with np.load(REAL_CAPTURE, allow_pickle=False) as payload:
        accepted = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
        captured = np.ascontiguousarray(
            payload["dense_virtual_partials_bfloat16_bits"]
        )
    with np.load(REAL_DB548, allow_pickle=False) as payload:
        control = np.ascontiguousarray(
            payload["layer1_normalized_bfloat16_bits"]
        )
    for name in ("isolated_dense", "layer1_replay"):
        (hlo_dir / f"{name}.stablehlo.mlir").write_text(f"{name} stable\n")
        (hlo_dir / f"{name}.optimized_hlo.txt").write_text(
            f"{name} optimized\n"
        )
    records = MODULE._FINAL_DENSE_LAYOUT_RECORDS
    zero_collectives = {
        "all_gather": 0,
        "all_reduce": 0,
        "all_to_all": 0,
        "collective_broadcast": 0,
        "collective_permute": 0,
        "reduce_scatter": 0,
    }
    isolated_stable = {
        "accepted_gate_dequant_fusion": True,
        "accepted_gate_singleton": True,
        "collective_counts": zero_collectives,
        "convolution_count": 2,
        "exact_result_binding": True,
        "gate_up_layout_constraint_count": 0,
        "matched_virtual_shards": [0],
        "runtime_u8_bitcast_count": 0,
        "virtual_contractions_per_chip": 1,
        "passed": True,
        "violations": [],
    }
    isolated_optimized = {
        "accepted_gate_dequant_fusion": True,
        "accepted_gate_singleton": True,
        "accepted_down_schedule": True,
        "accepted_gate_up_schedule": True,
        "async_collectives": [],
        "collective_count": 0,
        "compile_rows": 32,
        "convolution_count": 2,
        "dense_envelope": True,
        "down_convolution_count": 1,
        "exact_accepted_kernel_geometry": True,
        "exact_accepted_weight_layout": True,
        "exact_activation_graph": True,
        "exact_carried_residual_binding": True,
        "exact_gate_singleton_external_boundary": True,
        "exact_gate_dequant_fusion_boundary": True,
        "exact_packed_weight_lineage": True,
        "exact_result_binding": True,
        "final_dense_layout": True,
        "gate_up_convolution_count": 1,
        "isolated_dense": True,
        "live_rows": 1,
        "num_partitions": 4,
        "num_replicas": 1,
        "performance_claim": False,
        "passed": True,
        "violations": [],
    }
    rms_stable = {
        "collective_counts": {**zero_collectives, "all_gather": 1},
        "exact_result_binding": True,
        "live_rows": 1,
        "split_layer1_rms": False,
        "passed": True,
        "violations": [],
    }
    rms_optimized = {
        "accepted_scheduled_reduction_values": [],
        "async_collectives": [],
        "collective_count": 1,
        "exact_output_fusion": True,
        "exact_result_binding": True,
        "exact_scheduled_reduction_binding": True,
        "live_rows": 1,
        "num_partitions": 4,
        "num_replicas": 1,
        "performance_claim": False,
        "split_layer1_rms": False,
        "passed": True,
        "violations": [],
    }
    source_names = (
        "capture_tensor_sha256",
        "capture_runner_sha256",
        "capture_summary_sha256",
        "capture_success_sha256",
        "db548_tensor_sha256",
        "db548_runner_sha256",
        "db548_summary_sha256",
        "db548_success_sha256",
        "db548_hlo_sha256",
        "db549_tensor_sha256",
        "db549_runner_sha256",
        "db549_summary_sha256",
        "db549_success_sha256",
        "db549_hlo_sha256",
        "accepted_m32_hlo_sha256",
        "accepted_m32_summary_sha256",
        "accepted_m32_success_sha256",
        "checkpoint_manifest_sha256",
    )
    source_hashes = [f"{index:064x}" for index in range(1, 19)]
    ordered_values = dict(zip(source_names, source_hashes, strict=True))
    source = {
        "accepted_m32_hlo_raw_sha256": "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775",
        "accepted_m32_hlo_sha256": ordered_values[
            "accepted_m32_hlo_sha256"
        ],
        "accepted_m32_success_sha256": ordered_values[
            "accepted_m32_success_sha256"
        ],
        "accepted_m32_summary_sha256": ordered_values[
            "accepted_m32_summary_sha256"
        ],
        "capture_runner_sha256": ordered_values["capture_runner_sha256"],
        "capture_summary_sha256": ordered_values["capture_summary_sha256"],
        "capture_success_sha256": ordered_values["capture_success_sha256"],
        "capture_tensor_sha256": ordered_values["capture_tensor_sha256"],
        "checkpoint_manifest_sha256": ordered_values[
            "checkpoint_manifest_sha256"
        ],
        "db548_hlo_sha256": ordered_values["db548_hlo_sha256"],
        "db548_runner_sha256": ordered_values["db548_runner_sha256"],
        "db548_summary_sha256": ordered_values["db548_summary_sha256"],
        "db548_success_sha256": ordered_values["db548_success_sha256"],
        "db548_tensor_sha256": ordered_values["db548_tensor_sha256"],
        "db549_hlo_sha256": ordered_values["db549_hlo_sha256"],
        "db549_runner_sha256": ordered_values["db549_runner_sha256"],
        "db549_summary_sha256": ordered_values["db549_summary_sha256"],
        "db549_success_sha256": ordered_values["db549_success_sha256"],
        "db549_tensor_sha256": ordered_values["db549_tensor_sha256"],
    }
    partial_comparison = MODULE._partial_comparison(captured, captured)
    comparison = MODULE._compare_bits(accepted, accepted)
    sensitivity_candidates = MODULE._build_sensitivity_candidates(captured)
    for candidate in sensitivity_candidates:
        candidate["comparison"] = comparison
        candidate["output_sha256"] = MODULE._array_sha256(accepted)
    sensitivity = {
        "baseline_dense_update_bits": 47808,
        "candidate_count": len(sensitivity_candidates),
        "candidates": sensitivity_candidates,
        "exact_candidate_ids": list(range(len(sensitivity_candidates))),
        "hidden_index": 2795,
        "partial_bit_delta_limit": 32,
    }
    pin = "a" * 40
    runner = {
        "accepted_gate_dequant_fusion": True,
        "accepted_gate_singleton": True,
        "artifact_kind": "glm52_layer0_isolated_dense_replay",
        "classification": "isolated_virtual_contractions_exact",
        "code_hash": pin,
        "control_admissible": True,
        "exact": True,
        "exact_arms": ["isolated_virtual_contractions"],
        "final_layout_records": records,
        "gate_singleton_discriminator": {
            "accepted_internal_gate_dequant_count": 3,
            "accepted_rank2_gate_count": 0,
            "accepted_rank3_gate_count": 3,
            "challenger_selected": True,
            "db548_materialized_bf16_gate_count": 8,
            "db548_rank2_gate_count": 8,
            "db548_rank3_gate_count": 0,
        },
        "hlo": {},
        "isolated_layer1_comparison": comparison,
        "isolated_layer1_sha256": MODULE._array_sha256(accepted),
        "partial_comparison": partial_comparison,
        "performance_claim": False,
        "position": 8155,
        "sensitivity": sensitivity,
        "source": source,
        "status": "SUCCESS",
        "virtual_contractions_per_chip": 1,
        "virtual_rank_batches": 8,
    }
    for name, stable, optimized in (
        ("isolated_dense", isolated_stable, isolated_optimized),
        ("layer1_replay", rms_stable, rms_optimized),
    ):
        runner["hlo"][name] = {
            "optimized_contract": optimized,
            "optimized_sha256": MODULE._array_sha256(
                np.frombuffer(
                    (hlo_dir / f"{name}.optimized_hlo.txt").read_bytes(),
                    dtype=np.uint8,
                )
            ),
            "stablehlo_contract": stable,
            "stablehlo_sha256": MODULE._array_sha256(
                np.frombuffer(
                    (hlo_dir / f"{name}.stablehlo.mlir").read_bytes(),
                    dtype=np.uint8,
                )
            ),
        }
    (run_dir / "runner.json").write_text(json.dumps(runner))
    np.savez(
        run_dir / "isolated_dense_replay.npz",
        accepted_layer1_normalized_bfloat16_bits=accepted,
        captured_dense_virtual_partials_bfloat16_bits=captured,
        control_layer1_normalized_bfloat16_bits=control,
        isolated_dense_virtual_partials_bfloat16_bits=captured,
        isolated_layer1_normalized_bfloat16_bits=accepted,
        sensitivity_candidate_dense_update_bfloat16_bits=np.asarray(
            [item["dense_update_bits"] for item in sensitivity_candidates],
            dtype=np.uint16,
        ),
        sensitivity_candidate_model_rank=np.asarray(
            [
                item["representative_model_rank"]
                for item in sensitivity_candidates
            ],
            dtype=np.int16,
        ),
        sensitivity_candidate_partial_bit_delta=np.asarray(
            [
                item["representative_partial_bit_delta"]
                for item in sensitivity_candidates
            ],
            dtype=np.int16,
        ),
        sensitivity_candidate_partial_bits=np.asarray(
            [
                item["representative_mutated_partial_bits"]
                for item in sensitivity_candidates
            ],
            dtype=np.int32,
        ),
        sensitivity_layer1_normalized_bfloat16_bits=np.stack(
            [accepted] * len(sensitivity_candidates)
        ),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(run_dir),
            pin,
            "5",
            "isolated_dense_test",
            *source_hashes,
        ],
        input=record_program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["results_db_run_id"] is None
    assert summary["classification"] == "isolated_virtual_contractions_exact"

    success_program = next(
        value
        for value in programs
        if "isolated dense replay summary drifted before SUCCESS" in value
    )
    evidence_paths = [
        run_dir / "runner.json",
        run_dir / "isolated_dense_replay.npz",
        run_dir / "summary.json",
        *(hlo_dir / f"{name}.{suffix}"
          for name in ("isolated_dense", "layer1_replay")
          for suffix in ("optimized_hlo.txt", "stablehlo.mlir")),
    ]
    (run_dir / "evidence.sha256").write_text(
        "".join(
            f"{sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(run_dir).as_posix()}\n"
            for path in evidence_paths
        )
    )

    def crc32c(path: Path) -> str:
        checksum = google_crc32c.Checksum()
        checksum.update(path.read_bytes())
        return base64.b64encode(checksum.digest()).decode()

    ledger_paths = [
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name not in {"SUCCESS", "remote_objects.json"}
    ]
    (run_dir / "remote_objects.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "crc32c": crc32c(path),
                        "generation": str(index + 1),
                        "path": path.relative_to(run_dir).as_posix(),
                        "size": path.stat().st_size,
                    }
                    for index, path in enumerate(sorted(ledger_paths))
                ]
            }
        )
    )
    sealed = subprocess.run(
        [
            sys.executable,
            "-",
            str(run_dir),
            "gs://driftbench-dsv4-uc/results/isolated_dense_test",
            pin,
        ],
        input=success_program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert sealed.returncode == 0, sealed.stderr
    success = (run_dir / "SUCCESS").read_text()
    assert "artifact_kind=glm52_layer0_isolated_dense_replay\n" in success
    assert "results_db_run_id=none\n" in success
    assert "accepted_gate_dequant_fusion=true\n" in success
    assert "accepted_gate_singleton=true\n" in success
    assert "gate_dequant_accepted_internal_count=3\n" in success
    assert "gate_dequant_db548_materialized_count=8\n" in success
    assert "gate_singleton_accepted_rank3_count=3\n" in success
    assert "gate_singleton_db548_rank2_count=8\n" in success
    assert "sensitivity_baseline_dense_update_bits=47808\n" in success
    assert "sensitivity_candidate_count=15\n" in success
    assert (
        "sensitivity_exact_candidate_ids="
        + ",".join(str(value) for value in range(15))
        + "\n"
    ) in success
    assert "sensitivity_hidden_index=2795\n" in success
