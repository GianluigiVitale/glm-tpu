from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from glm_tpu.greenfield.errors import PlanValidationError


def _exact_head_key_hlo(*, slots: int = 5, layers: int = 21) -> str:
    parameters = "\n".join(
        f"  %wk.{slot} = f32[128,6144] parameter({slot})"
        for slot in range(slots)
    )
    operations = []
    for layer in range(layers):
        operations.extend(
            (
                f"  %normalized.{layer} = f32[1,6144] convert(%hidden), "
                f'metadata={{op_name="jit(mapped)/shard_map/convert.{layer}"}}',
                f"  %key.{layer} = (f32[], f32[128]) fusion(%wk.0, "
                f"%normalized.{layer}), kind=kLoop, calls=%key_projection, "
                f'metadata={{op_name="jit(mapped)/shard_map/dot_general.{layer}"}}, '
                'backend_config={"megacore_config":'
                '{"megacore_allreduce_bytes":"8192"}}',
                f"  %sqrt.{layer} = f32[] sqrt(%epsilon), "
                f'metadata={{op_name="jit(mapped)/shard_map/sqrt.{layer}"}}',
            )
        )
    operations_text = "\n".join(operations)
    return (
        "HloModule head_key, num_partitions=32\n\n"
        "%key_projection (wk: f32[128,6144], hidden: f32[1,6144]) "
        "-> (f32[], f32[128]) {\n"
        "  %wk = f32[128,6144] parameter(0)\n"
        "  %hidden = f32[1,6144] parameter(1)\n"
        "  %zero = f32[] constant(0)\n"
        "  %key = f32[128] constant({...})\n"
        "  ROOT %tuple = (f32[], f32[128]) tuple(%zero, %key)\n"
        "}\n\n"
        f"ENTRY %main ({', '.join(f'wk.{slot}: f32[128,6144]' for slot in range(slots))}) "
        "-> f32[] {\n"
        f"{parameters}\n"
        "  %hidden = bf16[1,6144] constant({...})\n"
        "  %epsilon = f32[] constant(1)\n"
        f"{operations_text}\n"
        f"  ROOT %out = f32[] copy(%sqrt.{layers - 1})\n"
        "}\n"
    )


def test_exact_head_key_hlo_contract_pins_db527_mechanism() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_dsa_head_key_decoder_association,
    )

    hlo = _exact_head_key_hlo()
    accepted = _validate_dsa_head_key_decoder_association(
        hlo,
        full_indexer_layers=21,
        maximum_full_indexer_slots=5,
        exact_association=True,
        prefill_index_repair=True,
    )
    assert accepted["passed"], accepted
    assert accepted["external_wk_parameter_count"] == 5
    assert accepted["key_projection_count"] == 21
    assert accepted["key_sqrt_count"] == 21
    assert accepted["normalized_f32_convert_count"] == 21

    missing_sqrt = _validate_dsa_head_key_decoder_association(
        hlo.replace(" sqrt(%epsilon)", " copy(%epsilon)", 1),
        full_indexer_layers=21,
        maximum_full_indexer_slots=5,
        exact_association=True,
    )
    assert not missing_sqrt["passed"]
    assert missing_sqrt["key_sqrt_count"] == 20

    escaped = _validate_dsa_head_key_decoder_association(
        hlo + "\n%escaped = f32[4,128,6144] parameter(99)\n",
        full_indexer_layers=21,
        maximum_full_indexer_slots=5,
        exact_association=True,
    )
    assert not escaped["passed"]
    assert escaped["forbidden_global_shapes"] == ["f32[4,128,6144]"]


def test_default_head_key_hlo_allows_only_prefill_repair_state() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_dsa_head_key_decoder_association,
    )

    prefill_state_only = _exact_head_key_hlo(layers=1).split(
        "  %normalized.0", 1
    )[0] + "  ROOT %out = f32[] copy(%epsilon)\n}\n"
    allowed = _validate_dsa_head_key_decoder_association(
        prefill_state_only,
        full_indexer_layers=21,
        maximum_full_indexer_slots=5,
        exact_association=False,
        prefill_index_repair=True,
    )
    assert allowed["passed"], allowed
    rejected = _validate_dsa_head_key_decoder_association(
        prefill_state_only,
        full_indexer_layers=21,
        maximum_full_indexer_slots=5,
        exact_association=False,
        prefill_index_repair=False,
    )
    assert not rejected["passed"]


def test_fused_qkv_a_hlo_contract_requires_db502_primitive() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_fused_qkv_a_decoder_association,
    )

    hlo = '''HloModule fused_qkv, replica_count=1, num_partitions=32

ENTRY main {
  %hidden = bf16[1,6144] parameter(0)
  %weight = u8[32,6144,82] parameter(1)
  %scale = f32[32,48,82] parameter(2)
  ROOT %qkv = f32[1,82] convolution(%weight, %scale), dim_labels=bf_io->bf
}
'''
    contract = _validate_fused_qkv_a_decoder_association(hlo, layers=1)
    assert contract["passed"], contract
    assert contract["convolution_count"] == 1

    dead = _validate_fused_qkv_a_decoder_association(
        hlo.replace(
            "%weight = u8[32,6144,82] parameter(1)",
            "%weight = u8[32,6144,82] parameter(1)\n"
            "  %dead = bf16[32,6144] parameter(3)",
        ),
        layers=1,
    )
    assert dead["violations"]
    assert dead["forbidden_shapes"] == ["bf16[32,6144]"]


def _synthetic_dense_final_layout_hlo() -> tuple[str, str]:
    optimized = [
        "HloModule dense_final_layout, replica_count=1, num_partitions=32",
        "",
        "ENTRY main {",
        "  %gate_lhs = bf16[1,6144]{1,0} parameter(0)",
        "  %gate_rhs = bf16[6144,768]{1,0} parameter(1)",
        "  %down_lhs = bf16[1,384]{1,0} parameter(2)",
        "  %down_rhs = bf16[384,6144]{1,0} parameter(3)",
    ]
    stable = ["module {", "  func.func @main() {"]
    for rank in range(8):
        optimized.extend(
            (
                f"  %gate_{rank} = f32[1,768]{{1,0}} convolution("
                "%gate_lhs, %gate_rhs), dim_labels=bf_io->bf, "
                "metadata={op_name=\"jit(mapped)/shard_map/"
                f"greenfield_dense_convolution_virtual_rank_{rank:02d}/"
                "conv_general_dilated\"}",
                f"  %down_{rank} = f32[1,6144]{{1,0}} convolution("
                f"%activated_{rank}, %down_rhs), dim_labels=bf_io->bf, "
                "metadata={op_name=\"jit(mapped)/shard_map/"
                f"greenfield_dense_convolution_virtual_rank_{rank:02d}/"
                "conv_general_dilated\"}",
            )
        )
        optimized.insert(
            -1,
            f"  %gate_round_{rank} = bf16[1,768]{{1,0}} convert(%gate_{rank})",
        )
        optimized.insert(
            -1,
            f"  %activated_{rank} = bf16[1,384]{{1,0}} "
            f"slice(%gate_round_{rank}), slice={{[0:1], [0:384]}}",
        )
        optimized.append(
            f"  %down_round_{rank} = bf16[1,6144]{{1,0}} convert(%down_{rank})"
        )
        stable.extend(
            (
                f"    %layout_{rank} = stablehlo.custom_call "
                f"@LayoutConstraint(%weight_{rank}) : "
                "(tensor<6144x768xbf16>) -> tensor<6144x768xbf16>",
                f"    %gate_{rank} = stablehlo.convolution(%lhs, "
                f"%layout_{rank}) dim_numbers = [b, f]x[i, o]->[b, f], "
                "window = {stride = [], pad = [], lhs_dilate = [], "
                "rhs_dilate = [], reverse = []} {batch_group_count = 1 : "
                "i64, feature_group_count = 1 : i64, precision_config = "
                "[#stablehlo<precision DEFAULT>, #stablehlo<precision "
                "DEFAULT>]} : (tensor<1x6144xbf16>, "
                "tensor<6144x768xbf16>) -> tensor<1x768xf32>",
                f"    %down_{rank} = stablehlo.convolution(%activated, "
                f"%down_weight_{rank}) dim_numbers = [b, f]x[i, o]->[b, f], "
                "window = {stride = [], pad = [], lhs_dilate = [], "
                "rhs_dilate = [], reverse = []} {batch_group_count = 1 : "
                "i64, feature_group_count = 1 : i64, precision_config = "
                "[#stablehlo<precision DEFAULT>, #stablehlo<precision "
                "DEFAULT>]} : (tensor<1x384xbf16>, "
                "tensor<384x6144xbf16>) -> tensor<1x6144xf32>",
            )
        )
    down_types = ", ".join("bf16[1,6144]" for _ in range(8))
    down_values = ", ".join(f"%down_round_{rank}" for rank in range(8))
    optimized.extend((f"  ROOT %out = ({down_types}) tuple({down_values})", "}"))
    stable.extend(("    return", "  }", "}"))
    return "\n".join(optimized), "\n".join(stable)


@lru_cache(maxsize=1)
def _runtime_dense_final_layout_stablehlo() -> str:
    program = r'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
    _reduce_virtual_tp32_bf16_partials,
    _virtual_dense_final_layout_convolution_down_partials,
)

mesh = Mesh(np.asarray(jax.devices()), ("lp4",))
replicated = NamedSharding(mesh, P())
slot = NamedSharding(mesh, P("lp4", None, None, None))
contracts = (
    ((1, 6144), jnp.bfloat16, replicated),
    ((4, 8, 6144, 768), jnp.uint8, slot),
    ((4, 8, 48, 768), jnp.float32, slot),
    ((4, 8, 384, 6144), jnp.uint8, slot),
    ((4, 8, 3, 6144), jnp.float32, slot),
)
arguments = tuple(
    jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    for shape, dtype, sharding in contracts
)

def local(normalized, merged_bits, merged_scale, down_bits, down_scale):
    partials = _virtual_dense_final_layout_convolution_down_partials(
        normalized,
        merged_bits[0],
        merged_scale[0],
        down_bits[0],
        down_scale[0],
        block_shape=(128, 128),
        compile_rows=1,
    )
    return _reduce_virtual_tp32_bf16_partials(
        partials,
        axis_name="lp4",
        groups=((0, 1, 2, 3),),
        association=STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
    )

mapped = jax.shard_map(
    local,
    mesh=mesh,
    in_specs=(P(), P("lp4", None, None, None),
              P("lp4", None, None, None),
              P("lp4", None, None, None),
              P("lp4", None, None, None)),
    out_specs=P(),
    check_vma=False,
)
print(jax.jit(mapped).lower(*arguments).as_text())
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_dense_final_layout_hlo_pins_one_row_geometry_and_ranks() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_dense_final_layout_convolution_hlo,
    )

    optimized, _stable_fixture = _synthetic_dense_final_layout_hlo()
    stable = _runtime_dense_final_layout_stablehlo()
    accepted = _validate_dense_final_layout_convolution_hlo(
        optimized,
        stable,
        dense_layers=1,
        enabled=True,
    )
    assert accepted["passed"], accepted
    assert accepted["gate_up_convolution_count"] == 8
    assert accepted["down_convolution_count"] == 8
    assert accepted["optimized_exact_gate_down_bijection"]
    assert accepted["optimized_exact_down_result_liveness"]
    assert accepted["stablehlo_exact_arithmetic_contract"] == {
        "dense_layer_group_count": 1,
        "exact_live_result_count": 1,
        "exact_runtime_u8_bitcast_count": 2,
        "gate_up_layout_constraint_count": 8,
        "matched_virtual_shard_count": 8,
        "passed": True,
        "violations": [],
    }
    wrong_row = _validate_dense_final_layout_convolution_hlo(
        optimized.replace(
            "%gate_0 = f32[1,768]{1,0} convolution(",
            "%gate_0 = f32[32,768]{1,0} convolution(",
            1,
        ),
        stable.replace("tensor<1x6144xbf16>", "tensor<32x6144xbf16>", 1),
        dense_layers=1,
        enabled=True,
    )
    assert not wrong_row["passed"]
    missing_rank = _validate_dense_final_layout_convolution_hlo(
        optimized.replace(
            "greenfield_dense_convolution_virtual_rank_07/",
            "greenfield_dense_convolution_virtual_rank_06/",
        ),
        stable,
        dense_layers=1,
        enabled=True,
    )
    assert not missing_rank["passed"]

    rogue_rhs = _validate_dense_final_layout_convolution_hlo(
        optimized.replace(
            "  %down_lhs = bf16[1,384]{1,0} parameter(2)",
            "  %down_lhs = bf16[1,384]{1,0} parameter(2)\n"
            "  %rogue_gate_rhs = bf16[6144,768]{1,0} "
            "add(%gate_rhs, %gate_rhs)",
            1,
        ).replace(
            "convolution(%gate_lhs, %gate_rhs)",
            "convolution(%gate_lhs, %rogue_gate_rhs)",
            1,
        ),
        stable,
        dense_layers=1,
        enabled=True,
    )
    assert not rogue_rhs["passed"]

    unrelated_constraint = re.sub(
        r"(@LayoutConstraint\()%[A-Za-z0-9_.$#-]+(\))",
        r"\1%arg5\2",
        stable,
        count=1,
    )
    unrelated = _validate_dense_final_layout_convolution_hlo(
        optimized,
        unrelated_constraint,
        dense_layers=1,
        enabled=True,
    )
    assert not unrelated["passed"]
    assert not unrelated["stablehlo_exact_arithmetic_contract"]["passed"]

    wrong_tuple_element = optimized.replace(
        "  %down_rhs = bf16[384,6144]{1,0} parameter(3)",
        "  %down_rhs = bf16[384,6144]{1,0} parameter(3)\n"
        "  %rogue_gate_value = bf16[1,768]{1,0} parameter(4)",
        1,
    ).replace(
        "  %activated_0 = bf16[1,384]{1,0} slice(%gate_round_0)",
        "  %tuple_decoy = (bf16[1,768]{1,0}, bf16[1,768]{1,0}) "
        "tuple(%gate_round_0, %rogue_gate_value)\n"
        "  %gte_decoy = bf16[1,768]{1,0} "
        "get-tuple-element(%tuple_decoy), index=1\n"
        "  %activated_0 = bf16[1,384]{1,0} slice(%gte_decoy)",
        1,
    )
    wrong_element = _validate_dense_final_layout_convolution_hlo(
        wrong_tuple_element,
        stable,
        dense_layers=1,
        enabled=True,
    )
    assert not wrong_element["passed"]
    assert not wrong_element["optimized_exact_gate_down_bijection"]

    copied_wrong_tuple_element = wrong_tuple_element.replace(
        "  %gte_decoy = bf16[1,768]{1,0} "
        "get-tuple-element(%tuple_decoy), index=1",
        "  %tuple_copy = (bf16[1,768]{1,0}, bf16[1,768]{1,0}) "
        "copy(%tuple_decoy)\n"
        "  %gte_decoy = bf16[1,768]{1,0} "
        "get-tuple-element(%tuple_copy), index=1",
        1,
    )
    copied_wrong_element = _validate_dense_final_layout_convolution_hlo(
        copied_wrong_tuple_element,
        stable,
        dense_layers=1,
        enabled=True,
    )
    assert not copied_wrong_element["passed"]
    assert not copied_wrong_element["optimized_exact_gate_down_bijection"]

    dead = _validate_dense_final_layout_convolution_hlo(
        re.sub(
            r"  ROOT %out = .*",
            "  ROOT %out = bf16[1,6144]{1,0} copy(%down_round_0)",
            optimized,
            count=1,
        ),
        stable,
        dense_layers=1,
        enabled=True,
    )
    assert not dead["passed"]
    assert not dead["optimized_exact_down_result_liveness"]


def test_dense_final_layout_kernel_traces_loader_u8_storage() -> None:
    import jax
    import jax.numpy as jnp

    from glm_tpu.greenfield.kernels.stage_local import (
        _virtual_dense_final_layout_convolution_down_partials,
    )

    arguments = (
        jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((8, 6144, 768), jnp.uint8),
        jax.ShapeDtypeStruct((8, 48, 768), jnp.float32),
        jax.ShapeDtypeStruct((8, 384, 6144), jnp.uint8),
        jax.ShapeDtypeStruct((8, 3, 6144), jnp.float32),
    )
    result = jax.eval_shape(
        lambda *values: _virtual_dense_final_layout_convolution_down_partials(
            *values,
            block_shape=(128, 128),
            compile_rows=1,
        ),
        *arguments,
    )
    assert result.shape == (8, 1, 6144)
    assert result.dtype == jnp.bfloat16


def _repair_scoped_shape_hlo(*, include_unscoped: bool = False) -> str:
    unrelated = ""
    if include_unscoped:
        unrelated = '''
%unrelated (dead_rows: f32[32,6144], overlay: bf16[2048,6144]) -> bf16[2048,6144] {
  %dead_rows = f32[32,6144] parameter(0)
  %overlay = bf16[2048,6144] parameter(1)
  ROOT %unrelated_root = bf16[2048,6144] copy(%overlay), metadata={op_name="decode/recurrent_overlay"}
}
'''
    return f'''HloModule repair_scope, replica_count=1, num_partitions=32

%repair_weight (wk: f32[128,6144]) -> f32[128,6144] {{
  %wk = f32[128,6144] parameter(0)
  ROOT %weight_root = f32[128,6144] copy(%wk)
}}

%repair (chunk: bf16[2048,6144], rows: f32[32,6144], wk: f32[128,6144]) -> bf16[2048,6144] {{
  %chunk = bf16[2048,6144] parameter(0)
  %rows = f32[32,6144] parameter(1)
  %wk = f32[128,6144] parameter(2)
  %weight = f32[128,6144] fusion(%wk), kind=kLoop, calls=%repair_weight
  ROOT %repair_root = bf16[2048,6144] copy(%chunk)
}}
{unrelated}
ENTRY %main (chunk: bf16[2048,6144], rows: f32[32,6144], wk: f32[128,6144]) -> bf16[2048,6144] {{
  %chunk = bf16[2048,6144] parameter(0)
  %rows = f32[32,6144] parameter(1)
  %wk = f32[128,6144] parameter(2)
  ROOT %repair_call = bf16[2048,6144] fusion(%chunk, %rows, %wk), kind=kLoop, calls=%repair, metadata={{op_name="jit(execute)/shard_map/cond/branch_0_fun/repair_stage_local_prompt_index_cache"}}
}}
'''


def test_prefill_repair_shape_scope_is_not_a_module_wide_exception() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
        _validate_pallas_stage_linear_decoder_calls,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    hlo = _repair_scoped_shape_hlo()
    module = parse_hlo_module(hlo)
    stage = _validate_pallas_stage_linear_decoder_calls(
        hlo,
        layers=0,
        dense_layers=0,
        full_indexer_layers=0,
        dsa_query_backend="reference",
        prefill_index_repair=True,
        module=module,
    )
    assert stage["passed"], stage
    assert set(stage["allowed_prefill_index_repair_shape_counts"]) == {
        "bf16[2048,6144]",
        "f32[128,6144]",
    }

    live = _classify_decoder_live_tensor_shapes(
        module,
        config=_real_8k_decoder_config(),
        full_indexer_layers=0,
        backend_contract="cpu_reference",
        prefill_index_repair=True,
    )
    assert live["passed"], live
    assert live["allowed_prefill_index_repair_shapes"]

    default_stage = _validate_pallas_stage_linear_decoder_calls(
        hlo,
        layers=0,
        dense_layers=0,
        full_indexer_layers=0,
        dsa_query_backend="reference",
    )
    assert not default_stage["passed"]
    assert default_stage["forbidden_decoded_weight_overlays"] == [
        "bf16[2048,6144]",
        "f32[128,6144]",
    ]

    unscoped_hlo = _repair_scoped_shape_hlo(include_unscoped=True)
    unscoped_module = parse_hlo_module(unscoped_hlo)
    unscoped_stage = _validate_pallas_stage_linear_decoder_calls(
        unscoped_hlo,
        layers=0,
        dense_layers=0,
        full_indexer_layers=0,
        dsa_query_backend="reference",
        prefill_index_repair=True,
        module=unscoped_module,
    )
    assert not unscoped_stage["passed"]
    assert unscoped_stage["forbidden_decoded_weight_overlays"] == [
        "bf16[2048,6144]"
    ]

    unscoped_live = _classify_decoder_live_tensor_shapes(
        unscoped_module,
        config=_real_8k_decoder_config(),
        full_indexer_layers=0,
        backend_contract="cpu_reference",
        prefill_index_repair=True,
    )
    assert not unscoped_live["passed"]
    assert any(
        item["op_name"] == "decode/recurrent_overlay"
        or item["computation"].startswith("%unrelated ")
        for item in unscoped_live["forbidden_shapes"]
    )


def test_fused_qkv_dead_row_gate_scopes_prefill_repair_rows() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_fused_qkv_a_decoder_association,
    )

    hlo = '''HloModule fused_qkv_repair, num_partitions=32

%repair_rows (rows: f32[32,6144]) -> f32[32,6144] {
  %rows = f32[32,6144] parameter(0)
  ROOT %rows_root = f32[32,6144] copy(%rows)
}

ENTRY %main (hidden: bf16[1,6144], weight: u8[32,6144,82], scale: f32[32,48,82], rows: f32[32,6144]) -> f32[1,82] {
  %hidden = bf16[1,6144] parameter(0)
  %weight = u8[32,6144,82] parameter(1)
  %scale = f32[32,48,82] parameter(2)
  %rows = f32[32,6144] parameter(3)
  %repair_call = f32[32,6144] fusion(%rows), kind=kLoop, calls=%repair_rows, metadata={op_name="jit(execute)/shard_map/cond/branch_0_fun/repair_stage_local_prompt_index_cache"}
  ROOT %qkv = f32[1,82] convolution(%weight, %scale), dim_labels=bf_io->bf
}
'''
    scoped = _validate_fused_qkv_a_decoder_association(
        hlo,
        layers=1,
        prefill_index_repair=True,
    )
    assert scoped["passed"], scoped
    assert scoped["allowed_prefill_index_repair_shape_counts"][
        "f32[32,6144]"
    ] > 0

    default = _validate_fused_qkv_a_decoder_association(hlo, layers=1)
    assert not default["passed"]
    assert default["forbidden_shapes"] == ["f32[32,6144]"]

    unscoped = hlo.replace(
        "ENTRY %main",
        '''%unrelated_rows (rows: f32[32,6144]) -> f32[32,6144] {
  %rows = f32[32,6144] parameter(0)
  ROOT %root = f32[32,6144] copy(%rows), metadata={op_name="decode/dead_rows"}
}

ENTRY %main''',
    )
    rejected = _validate_fused_qkv_a_decoder_association(
        unscoped,
        layers=1,
        prefill_index_repair=True,
    )
    assert not rejected["passed"]
    assert rejected["forbidden_shapes"] == ["f32[32,6144]"]


def _exact_wk_feature_slice_hlo(*, include_dead: bool = False) -> str:
    dead = (
        "  %dead = f32[32,6144]{1,0} parameter(4)\n"
        if include_dead
        else ""
    )
    return f'''HloModule exact_wk_feature_slice, num_partitions=32

ENTRY %main (hidden: bf16[1,6144], weight: u8[32,6144,82], scale: f32[32,48,82], wk: f32[128,6144]) -> f32[1,82] {{
  %hidden = bf16[1,6144] parameter(0)
  %weight = u8[32,6144,82] parameter(1)
  %scale = f32[32,48,82] parameter(2)
  %wk = f32[128,6144] parameter(3)
{dead}  %slice-start = ((f32[128,6144]), f32[32,6144], s32[]) slice-start(%wk), slice={{[0:32], [0:6144]}}
  %slice-start.1 = ((f32[128,6144]), f32[32,6144], s32[]) slice-start(%wk), slice={{[32:64], [0:6144]}}
  %slice-start.2 = ((f32[128,6144]), f32[32,6144], s32[]) slice-start(%wk), slice={{[64:96], [0:6144]}}
  %slice-start.3 = ((f32[128,6144]), f32[32,6144], s32[]) slice-start(%wk), slice={{[96:128], [0:6144]}}
  %slice-done = f32[32,6144] slice-done(%slice-start)
  %slice-done.1 = f32[32,6144] slice-done(%slice-start.1)
  %slice-done.2 = f32[32,6144] slice-done(%slice-start.2)
  %slice-done.3 = f32[32,6144] slice-done(%slice-start.3)
  %restored = f32[128,6144] custom-call(%slice-done, %slice-done.1, %slice-done.2, %slice-done.3), custom_call_target="ConcatBitcast"
  ROOT %qkv = f32[1,82] convolution(%weight, %scale), dim_labels=bf_io->bf
}}
'''


def test_exact_wk_feature_slices_are_not_dead_rows() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
        _validate_fused_qkv_a_decoder_association,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    hlo = _exact_wk_feature_slice_hlo()
    fused = _validate_fused_qkv_a_decoder_association(
        hlo,
        layers=1,
        dsa_head_key_exact_association=True,
    )
    assert fused["passed"], fused
    assert fused["allowed_exact_wk_feature_slice_shape_counts"] == {
        "f32[32,6144]": 16
    }
    assert fused["exact_wk_feature_slice_contract"]["group_count"] == 1

    module = parse_hlo_module(hlo)
    live = _classify_decoder_live_tensor_shapes(
        module,
        config=_real_8k_decoder_config(),
        full_indexer_layers=0,
        backend_contract="cpu_reference",
        dsa_head_key_exact_association=True,
    )
    assert live["passed"], live
    assert len(live["allowed_exact_wk_feature_slices"]) == 16

    default = _validate_fused_qkv_a_decoder_association(hlo, layers=1)
    assert not default["passed"]
    assert default["forbidden_shapes"] == ["f32[32,6144]"]


def test_exact_wk_feature_slice_scope_rejects_unrelated_rows() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
        _validate_fused_qkv_a_decoder_association,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    hlo = _exact_wk_feature_slice_hlo(include_dead=True)
    fused = _validate_fused_qkv_a_decoder_association(
        hlo,
        layers=1,
        dsa_head_key_exact_association=True,
    )
    assert not fused["passed"]
    assert fused["forbidden_shapes"] == ["f32[32,6144]"]

    live = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(hlo),
        config=_real_8k_decoder_config(),
        full_indexer_layers=0,
        backend_contract="cpu_reference",
        dsa_head_key_exact_association=True,
    )
    assert not live["passed"]
    assert len(live["forbidden_shapes"]) == 1
    assert live["forbidden_shapes"][0]["instruction"] == "%dead"


def test_fused_qkv_a_runtime_binding_omits_separate_projection_state() -> None:
    from glm_tpu.greenfield.runtime.decoder import _attention_weights

    loaded = []

    def weight(name: str) -> str:
        loaded.append(name)
        return name

    attention = _attention_weights(weight, 0, "fused_n82_convolution")
    assert attention.q_a_bits is None
    assert attention.q_a_scale is None
    assert attention.kv_a_bits is None
    assert attention.kv_a_scale is None
    assert attention.qkv_a_bits == "attention.slot_00.qkv_a.weight_bits"
    assert attention.qkv_a_scale == "attention.slot_00.qkv_a.scale_inv"
    assert "attention.slot_00.q_a.weight_bits" not in loaded
    assert "attention.slot_00.kv_a.weight_bits" not in loaded


def _synthetic_8k_dsa_score_hlo() -> str:
    return '''HloModule dsa_score, replica_count=1, num_partitions=32

%score (q: f32[32,128], key: bf16[2048,128], weight: f32[32]) -> f32[2048] {
  %q = f32[32,128] parameter(0)
  %key = bf16[2048,128] parameter(1)
  %weight = f32[32] parameter(2)
  %weight_broadcast = f32[1,32,2048] broadcast(%weight), dimensions={1}, metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/rh,rhs->rs/dot_general"}
  %contraction = f32[32,2048] convolution(%q, %key), dim_labels=bf_oi->bf, operand_precision={highest,highest}, metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/rhd,sd->rhs/dot_general"}
  %scale = f32[] constant(0.0883883461)
  %scale_broadcast = f32[32,2048] broadcast(%scale), dimensions={}, metadata={op_name="jit(mapped_token)/shard_map/broadcast.19787"}
  %scaled = f32[32,2048] multiply(%contraction, %scale_broadcast), metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/mul"}
  %zero = f32[] constant(0)
  %zero_broadcast = f32[32,2048] broadcast(%zero), dimensions={}, metadata={op_name="jit(mapped_token)/shard_map/broadcast.20646"}
  %clamped = f32[32,2048] maximum(%scaled, %zero_broadcast), metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/max"}
  %expanded = f32[1,32,2048] bitcast(%clamped), metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/max"}
  %weighted = f32[1,32,2048] multiply(%weight_broadcast, %expanded), metadata={op_name="jit(mapped_token)/shard_map/multiply.611"}
  ROOT %reduced = f32[2048] reduce(%weighted, %zero), dimensions={0,1}, metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/rh,rhs->rs/dot_general"}
}
'''


def _real_8k_decoder_config(*, dsa_score_default_precision: bool = False):
    from glm_tpu.greenfield.runtime.decoder import DecoderStepConfig

    return DecoderStepConfig(
        stage_count=8,
        local_parallel_size=4,
        hidden_size=6144,
        selected_width=2048,
        context_capacity=8192,
        maximum_layer_slots=10,
        maximum_full_indexer_slots=3,
        logical_page_size=256,
        local_rows_per_page=64,
        packed_cache_width=192,
        index_key_width=128,
        dsa_indexer_heads=32,
        vocab_size=154880,
        dsa_score_default_precision=dsa_score_default_precision,
    )


def _pregathered_b512_hlo(
    *, folded_cache: bool = True, old_scope: bool = False
) -> str:
    cache_shape = "bf16[1,4,512,640]" if folded_cache else "bf16[1,2048,640]"
    exchange = (
        f"  %sum = {cache_shape} all-reduce(%cache), "
        "replica_groups={{0,1,2,3}}, to_apply=%add, "
        'metadata={op_name="jit(mapped)/shard_map/'
        'greenfield_selected_cache_lp4_exchange/psum"}'
    )
    kernel = (
        "  %call = bf16[1,16,512] custom-call(%count, %query, %sum), "
        'custom_call_target="tpu_custom_call", metadata={op_name="jit(mapped)/'
        "shard_map/greenfield_pregathered_b512_attention/"
        'greenfield_pregathered_sparse_mla_h16_k2048_b512_w640/pallas_call"}, '
        'backend_config={"body":"'
        'greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"}'
    )
    old = (
        '  %old = bf16[1,16,512] copy(%call), '
        'metadata={op_name="jit(mapped)/shard_map/'
        'greenfield_owner_split_attention_output_gather/copy"}\n'
        if old_scope
        else ""
    )
    root = "%old" if old_scope else "%call"
    return f'''HloModule pregathered, replica_count=1, num_partitions=4

%add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %out = bf16[] add(%x, %y)
}}

ENTRY %main (cache: {cache_shape}) -> bf16[1,16,512] {{
  %cache = {cache_shape} parameter(0)
{exchange}
  %count = s32[1] constant({{2048}})
  %query = bf16[1,16,640] constant({{...}})
{kernel}
{old}  ROOT %root = bf16[1,16,512] copy({root})
}}
'''


def test_pregathered_b512_attention_hlo_contract_is_fail_closed() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_pregathered_b512_attention_hlo,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    config = replace(_real_8k_decoder_config(), packed_cache_width=640)
    for folded_cache in (False, True):
        hlo = _pregathered_b512_hlo(folded_cache=folded_cache)
        contract = _validate_pregathered_b512_attention_hlo(
            hlo,
            module=parse_hlo_module(hlo),
            config=config,
            layers=1,
            enabled=True,
        )
        assert contract["passed"], contract
        assert contract["exchange_count"] == 1
        assert contract["exchange_bijection"]
        assert contract["kernel_count"] == 1
        assert len(contract["kernel_exchange_links"]) == 1

        bypassed = hlo.replace(
            "custom-call(%count, %query, %sum)",
            "custom-call(%count, %query, %cache)",
        )
        rejected_bypass = _validate_pregathered_b512_attention_hlo(
            bypassed,
            module=parse_hlo_module(bypassed),
            config=config,
            layers=1,
            enabled=True,
        )
        assert not rejected_bypass["passed"]
        assert not rejected_bypass["exchange_bijection"]
        assert any(
            "cache operands do not depend" in violation
            for violation in rejected_bypass["violations"]
        )

    wrong_block = _pregathered_b512_hlo().replace("b512", "b128")
    rejected_block = _validate_pregathered_b512_attention_hlo(
        wrong_block,
        module=parse_hlo_module(wrong_block),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not rejected_block["passed"]
    assert rejected_block["kernel_count"] == 0

    suffixed_kernel = _pregathered_b512_hlo().replace(
        "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640",
        "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640_wrong",
    )
    rejected_suffix = _validate_pregathered_b512_attention_hlo(
        suffixed_kernel,
        module=parse_hlo_module(suffixed_kernel),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not rejected_suffix["passed"]
    assert rejected_suffix["kernel_count"] == 0

    out_of_scope = _pregathered_b512_hlo().replace(
        "greenfield_pregathered_b512_attention/",
        "greenfield_unscoped_b512_attention/",
    )
    out_of_scope = out_of_scope.replace(
        "  %count =",
        "  %scoped_marker = bf16[1,4,512,640] copy(%sum), "
        'metadata={op_name="jit(mapped)/shard_map/'
        'greenfield_pregathered_b512_attention/unrelated_marker"}\n'
        "  %count =",
        1,
    )
    rejected_scope = _validate_pregathered_b512_attention_hlo(
        out_of_scope,
        module=parse_hlo_module(out_of_scope),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not rejected_scope["passed"]
    assert rejected_scope["attention_instruction_count"] == 1
    assert rejected_scope["kernel_count"] == 0

    old_scope = _pregathered_b512_hlo(old_scope=True)
    rejected_old_scope = _validate_pregathered_b512_attention_hlo(
        old_scope,
        module=parse_hlo_module(old_scope),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not rejected_old_scope["passed"]
    assert rejected_old_scope["old_scope_instruction_count"] == 1

    default = _validate_pregathered_b512_attention_hlo(
        _pregathered_b512_hlo(),
        module=parse_hlo_module(_pregathered_b512_hlo()),
        config=config,
        layers=1,
        enabled=False,
    )
    assert not default["passed"]


def test_pregathered_b512_reduction_contract_removes_old_merge_results() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _expected_tpu_decoder_reductions,
    )

    arities, shapes = _expected_tpu_decoder_reductions(
        layers=78,
        dense_layers=3,
        sparse_layers=75,
        pregathered_b512_attention=True,
        feature_reconstruct_down_fp32=True,
        complete_token_path=True,
        split_residual_state=True,
        token_observation_candidates=1,
    )
    assert arities == {"1": 312}
    assert shapes == {
        "bf16[1,6144]": 81,
        "bf16[2,1,6144]": 75,
        "bf16[1,2048,640]": 78,
        "f32[8,6144]": 75,
        "bf16[1,1,6144]": 1,
        "bf16[4]": 1,
        "s32[4]": 1,
    }
    assert "f32[256]" not in shapes
    assert "u32[1,1,128]" not in shapes

    default_arities, default_shapes = _expected_tpu_decoder_reductions(
        layers=78,
        dense_layers=3,
        sparse_layers=75,
        pregathered_b512_attention=False,
        feature_reconstruct_down_fp32=True,
        complete_token_path=True,
        split_residual_state=True,
        token_observation_candidates=1,
    )
    assert default_arities == {"1": 355, "2": 16, "3": 1}
    assert default_shapes["f32[256]"] == 78
    assert default_shapes["u32[1,1,128]"] == 78


def _strategy_nd_attention_projection_hlo(*, folded: bool = False) -> str:
    call_input = "bf16[8,512]" if folded else "bf16[1,512]"
    scale = "f32[8,128]" if folded else "f32[48,4]"
    call_output = "bf16[8,6144]" if folded else "bf16[1,6144]"
    calls: list[str] = []
    partials: list[str] = []
    for index in range(8):
        calls.append(
            f"  %call.{index} = {call_output} custom-call(%lhs, %rhs, %scale), "
            'custom_call_target="tpu_custom_call", metadata={op_name="jit(mapped)/'
            "shard_map/greenfield_strategy_nd_row0_attention_output/"
            f"greenfield_fp8_strategy_nd_o_m8_k512_n6144/call.{index}\"}}, "
            'backend_config={"body":"greenfield_fp8_strategy_nd_o_m8_k512_n6144"}'
        )
        if folded:
            calls.append(
                f"  %partial.{index} = bf16[1,6144] slice(%call.{index}), "
                "slice={[0:1],[0:6144]}"
            )
            partials.append(f"%partial.{index}")
        else:
            partials.append(f"%call.{index}")
    return f'''HloModule strategy_nd_attention, replica_count=1, num_partitions=4

ENTRY %main (lhs: {call_input}, rhs: u8[6144,512], scale: {scale}) -> bf16[4,8,1,6144] {{
  %lhs = {call_input} parameter(0)
  %rhs = u8[6144,512] parameter(1)
  %scale = {scale} parameter(2)
{chr(10).join(calls)}
  %partials = bf16[8,6144] concatenate({', '.join(partials)}), dimensions={{0}}
  %logical = bf16[8,1,6144] reshape(%partials)
  %gather = bf16[4,8,1,6144] all-gather(%logical), dimensions={{0}}, replica_groups={{{{0,1,2,3}}}}, use_global_device_ids=true, metadata={{op_name="jit(mapped)/shard_map/greenfield_strategy_nd_row0_attention_output/greenfield_strategy_nd_row0_association/greenfield_strategy_nd_row0_association_gather/all_gather"}}
  ROOT %root = bf16[4,8,1,6144] copy(%gather)
}}
'''


def test_strategy_nd_attention_projection_hlo_is_fail_closed() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_strategy_nd_attention_projection_hlo,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    for folded in (False, True):
        hlo = _strategy_nd_attention_projection_hlo(folded=folded)
        accepted = _validate_strategy_nd_attention_projection_hlo(
            parse_hlo_module(hlo), layers=1, enabled=True
        )
        assert accepted["passed"], accepted
        assert accepted["gather_count"] == 1
        assert accepted["kernel_count"] == 8
        assert accepted["gather_source_counts"] == {"%gather": 8}
        assert accepted["partial_gather_bijection"]
        assert accepted["exclusive_partial_dataflow"]
        assert not accepted["exclusive_source_violations"]
        assert accepted["all_gathers_live"]

    hlo = _strategy_nd_attention_projection_hlo()
    bypass = hlo.replace("all-gather(%logical)", "all-gather(%lhs)")
    rejected_bypass = _validate_strategy_nd_attention_projection_hlo(
        parse_hlo_module(bypass), layers=1, enabled=True
    )
    assert not rejected_bypass["passed"]
    assert rejected_bypass["gather_source_counts"] == {"%gather": 0}

    duplicate = hlo.replace("%call.7)", "%call.0)")
    rejected_duplicate = _validate_strategy_nd_attention_projection_hlo(
        parse_hlo_module(duplicate), layers=1, enabled=True
    )
    assert not rejected_duplicate["passed"]
    assert not rejected_duplicate["partial_gather_bijection"]

    injected = hlo.replace(
        "scale: f32[48,4])",
        "scale: f32[48,4], extra: bf16[1,6144])",
    ).replace(
        "  %scale = f32[48,4] parameter(2)",
        "  %scale = f32[48,4] parameter(2)\n"
        "  %extra = bf16[1,6144] parameter(3)",
    ).replace(
        "  %partials = bf16[8,6144] concatenate(",
        "  %mixed = bf16[9,6144] concatenate(",
    ).replace(
        "), dimensions={0}\n  %logical =",
        ", %extra), dimensions={0}\n"
        "  %partials = bf16[8,6144] slice(%mixed), "
        "slice={[0:8],[0:6144]}\n  %logical =",
        1,
    )
    rejected_injected = _validate_strategy_nd_attention_projection_hlo(
        parse_hlo_module(injected), layers=1, enabled=True
    )
    assert not rejected_injected["passed"]
    assert rejected_injected["partial_gather_bijection"]
    assert not rejected_injected["exclusive_partial_dataflow"]
    assert rejected_injected["exclusive_source_violations"]

    dead = hlo.replace("copy(%gather)", "copy(%logical)")
    rejected_dead = _validate_strategy_nd_attention_projection_hlo(
        parse_hlo_module(dead), layers=1, enabled=True
    )
    assert not rejected_dead["passed"]
    assert not rejected_dead["all_gathers_live"]

    suffixed = hlo.replace(
        "greenfield_fp8_strategy_nd_o_m8_k512_n6144",
        "greenfield_fp8_strategy_nd_o_m8_k512_n6144_wrong",
    )
    rejected_suffix = _validate_strategy_nd_attention_projection_hlo(
        parse_hlo_module(suffixed), layers=1, enabled=True
    )
    assert not rejected_suffix["passed"]
    assert rejected_suffix["kernel_count"] == 0

    disabled = _validate_strategy_nd_attention_projection_hlo(
        parse_hlo_module(hlo), layers=1, enabled=False
    )
    assert not disabled["passed"]
    assert not disabled["applicable"]


def test_strategy_nd_attention_kernel_identity_is_disjoint_from_moe_down() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_pallas_feature_decoder_calls,
        _validate_pallas_stage_linear_decoder_calls,
    )

    stage_names = (
        "greenfield_fp8_block_matmul_m8_k6144_n2048",
        "greenfield_fp8_block_matmul_m8_k2048_n4096",
        "greenfield_fp8_block_matmul_m8_k6144_n640",
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512",
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256",
        *(["greenfield_fp8_strategy_nd_o_m8_k512_n6144"] * 8),
    )
    stage_hlo = "\n".join(
        f'%call.{index} = custom-call(), custom_call_target="tpu_custom_call", '
        f'backend_config="{name}"'
        for index, name in enumerate(stage_names)
    )
    # Same-geometry MoE down is present but cannot inflate StrategyND counts.
    stage_hlo += (
        '\n%moe = custom-call(), custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_block_matmul_m8_k512_n6144"'
    )
    stage = _validate_pallas_stage_linear_decoder_calls(
        stage_hlo,
        layers=1,
        dense_layers=0,
        full_indexer_layers=0,
        attention_projection_backend="separate",
        strategy_nd_attention_projection=True,
    )
    assert stage["passed"], stage
    assert stage["kernel_counts"][
        "greenfield_fp8_strategy_nd_o_m8_k512_n6144"
    ] == 8

    feature_names = (
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512",
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512",
        "greenfield_fp8_block_up_gate_m8_k6144_n512",
        "greenfield_fp8_block_up_gate_m8_k6144_n512",
        "greenfield_fp8_block_matmul_m8_k512_n6144",
    )
    feature_hlo = "\n".join(
        f'%feature.{index} = custom-call(u8[256,6144,512], '
        'u8[256,6144,512], u8[256,512,6144]), '
        'custom_call_target="tpu_custom_call", '
        f'backend_config="{name}"'
        for index, name in enumerate(feature_names)
    )
    feature_hlo += "\n" + stage_hlo
    feature = _validate_pallas_feature_decoder_calls(
        feature_hlo, sparse_layers=2, feature_output_tile=128
    )
    # Only the two explicit MoE identities count; eight StrategyND calls do not.
    assert feature["kernel_counts"] == feature["expected_kernel_counts"]
    assert feature["kernel_counts"][
        "greenfield_fp8_block_matmul_m8_k512_n6144"
    ] == 2


def test_strategy_nd_attention_reduction_contract_replaces_projection_sum() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _expected_tpu_decoder_reductions,
    )

    arities, shapes = _expected_tpu_decoder_reductions(
        layers=78,
        dense_layers=3,
        sparse_layers=75,
        pregathered_b512_attention=True,
        strategy_nd_attention_projection=True,
        feature_reconstruct_down_fp32=True,
        complete_token_path=True,
        split_residual_state=True,
        token_observation_candidates=1,
    )
    assert arities == {"1": 234}
    assert shapes["bf16[1,6144]"] == 3
    assert shapes["bf16[1,2048,640]"] == 78


def test_live_tensor_contract_admits_only_local_strategy_nd_virtual_partials() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    hlo = _strategy_nd_attention_projection_hlo().replace(
        "  ROOT %root = bf16[4,8,1,6144] copy(%gather)",
        "  %folded = bf16[32,1,6144] bitcast(%gather)\n"
        "  ROOT %root = bf16[32,1,6144] copy(%folded)",
    )
    module = parse_hlo_module(hlo)
    rejected = _classify_decoder_live_tensor_shapes(
        module,
        config=_real_8k_decoder_config(),
        full_indexer_layers=0,
        backend_contract="cpu_reference",
        strategy_nd_attention_projection=False,
    )
    assert not rejected["passed"]
    assert rejected["forbidden_shapes"]
    accepted = _classify_decoder_live_tensor_shapes(
        module,
        config=_real_8k_decoder_config(),
        full_indexer_layers=0,
        backend_contract="cpu_reference",
        strategy_nd_attention_projection=True,
    )
    assert accepted["passed"], accepted
    assert accepted["allowed_strategy_nd_virtual_partials"]
    assert not accepted["forbidden_shapes"]


def _main_rope_table_hlo(
    *,
    include_forbidden: bool = False,
    round_scope: bool = True,
    duplicate_round: bool = False,
    round_products_to_bf16: bool = False,
    table_dataflow: bool = True,
) -> str:
    scoped_name = "jit(mapped_token)/shard_map/greenfield_main_rope_table"
    multiplies = "\n".join(
        f'  %mul.{index} = f32[1,32] multiply(%f32, %f32), '
        f'metadata={{op_name="{scoped_name}/mul.{index}"}}'
        for index in range(8)
    )
    product_rounds = (
        "\n".join(
            f"  %mul.round.{index} = bf16[1,32] convert(%mul.{index})\n"
            f"  %mul.widen.{index} = f32[1,32] convert(%mul.round.{index})"
            for index in range(8)
        )
        if round_products_to_bf16
        else ""
    )
    product_name = "mul.widen" if round_products_to_bf16 else "mul"
    combines = "\n".join(
        f'  %add.{index} = f32[1,32] add(%{product_name}.{2 * index}, '
        f'%{product_name}.{2 * index + 1}), '
        f'metadata={{op_name="{scoped_name}/add.{index}"}}'
        for index in range(4)
    )
    forbidden = (
        f'  %bad = bf16[1,32] multiply(%round.0, %round.1), '
        f'metadata={{op_name="{scoped_name}/bad"}}\n'
        if include_forbidden
        else ""
    )
    rounds = "\n".join(
        f"  %round.{index} = bf16[1,32] convert(%add.{index})"
        + (
            f', metadata={{op_name="{scoped_name}/round.{index}"}}'
            if round_scope
            else ""
        )
        for index in range(2)
    )
    duplicate = (
        "  %round.duplicate = bf16[1,32] convert(%add.0)\n"
        if duplicate_round
        else ""
    )
    return f'''HloModule main_rope, num_partitions=32

ENTRY %main (main_rope_table: bf16[8192,64]) -> bf16[1,32] {{
  %main_rope_table = bf16[8192,64] parameter(0), metadata={{op_name="main_rope_table"}}
  %zero = s32[] constant(0)
  %table.row = bf16[1,64] dynamic-slice(%main_rope_table, %zero, %zero), dynamic_slice_sizes={{1,64}}, metadata={{op_name="greenfield_main_rope_table_lookup/dynamic_slice"}}
  %table.half = bf16[1,32] slice(%table.row), slice={{[0:1], [0:32]}}
  %table.f32 = f32[1,32] convert(%table.half)
  %f32 = f32[1,32] {"copy(%table.f32)" if table_dataflow else "constant({...})"}
{multiplies}
{product_rounds}
{combines}
{rounds}
{duplicate}{forbidden}  ROOT %out = bf16[1,32] copy(%round.0)
}}
'''


def test_main_rope_table_hlo_contract_is_fail_closed() -> None:
    from dataclasses import replace

    from glm_tpu.greenfield.runtime.decoder import _validate_main_rope_table_hlo
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    config = _real_8k_decoder_config()
    with pytest.raises(PlanValidationError, match="table width must be even"):
        replace(config, main_rope_table_width=63)
    accepted = _validate_main_rope_table_hlo(
        parse_hlo_module(_main_rope_table_hlo()),
        config=config,
        layers=1,
        enabled=True,
    )
    assert accepted["passed"], accepted
    assert accepted["expected_table_shape"] == [8192, 64]
    assert accepted["table_parameter_count"] == 1
    assert accepted["named_table_parameter_count"] == 1
    assert accepted["table_dependent_lookup_count"] == 1
    assert accepted["table_dependent_fp32_multiply_count"] == 8
    assert accepted["table_dependent_fp32_combine_count"] == 4
    assert accepted["table_dependent_final_round_count"] == 2
    assert accepted["fp32_multiply_count"] == 8
    assert accepted["fp32_combine_count"] == 4
    assert accepted["combines_with_direct_fp32_product_operands"] == 4
    assert accepted["premature_product_rounds"] == []
    assert accepted["final_round_count"] == 2
    assert accepted["scoped_final_round_count"] == 2
    assert accepted["direct_dataflow_final_round_count"] == 0
    assert accepted["combines_with_sole_convert_user"] == 2

    unused_table = _validate_main_rope_table_hlo(
        parse_hlo_module(_main_rope_table_hlo(table_dataflow=False)),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not unused_table["passed"]
    assert unused_table["table_dependent_lookup_count"] == 1
    assert unused_table["table_dependent_fp32_multiply_count"] == 0
    assert any(
        "products do not depend on the table lookup" in violation
        for violation in unused_table["violations"]
    )

    metadata_stripped = _validate_main_rope_table_hlo(
        parse_hlo_module(_main_rope_table_hlo(round_scope=False)),
        config=config,
        layers=1,
        enabled=True,
    )
    assert metadata_stripped["passed"], metadata_stripped
    assert metadata_stripped["final_round_count"] == 2
    assert metadata_stripped["scoped_final_round_count"] == 0
    assert metadata_stripped["direct_dataflow_final_round_count"] == 2
    assert metadata_stripped["combines_with_sole_convert_user"] == 2

    premature_product_rounds = _validate_main_rope_table_hlo(
        parse_hlo_module(
            _main_rope_table_hlo(round_products_to_bf16=True)
        ),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not premature_product_rounds["passed"]
    assert (
        premature_product_rounds[
            "combines_with_direct_fp32_product_operands"
        ]
        == 0
    )
    assert len(premature_product_rounds["premature_product_rounds"]) == 8
    assert (
        "main-RoPE FP32 products are rounded before combine"
        in premature_product_rounds["violations"]
    )
    assert (
        "main-RoPE FP32 combines do not directly consume scoped products"
        in premature_product_rounds["violations"]
    )

    duplicated = _validate_main_rope_table_hlo(
        parse_hlo_module(
            _main_rope_table_hlo(
                round_scope=False,
                duplicate_round=True,
            )
        ),
        config=config,
        layers=1,
        enabled=True,
    )
    assert duplicated["passed"], duplicated
    assert duplicated["final_round_count"] == 2
    assert duplicated["direct_dataflow_final_round_count"] == 2
    assert duplicated["combines_with_sole_convert_user"] == 1

    unrelated_rounds = _main_rope_table_hlo(round_scope=False).replace(
        "convert(%add.0)", "convert(%f32)"
    ).replace("convert(%add.1)", "convert(%f32)")
    unrelated = _validate_main_rope_table_hlo(
        parse_hlo_module(unrelated_rounds),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not unrelated["passed"]
    assert unrelated["final_round_count"] == 0
    assert unrelated["direct_dataflow_final_round_count"] == 0

    cross_computation_round = (
        "\n%other_round (value: f32[1,32]) -> bf16[1,32] {\n"
        "  %value = f32[1,32] parameter(0)\n"
        "  ROOT %round.0 = bf16[1,32] convert(%add.0)\n"
        "}\n\n"
    )
    cross_computation_hlo = _main_rope_table_hlo(round_scope=False).replace(
        "  %round.0 = bf16[1,32] convert(%add.0)\n",
        "",
        1,
    ).replace(
        "HloModule main_rope, num_partitions=32\n\n",
        "HloModule main_rope, num_partitions=32\n"
        + cross_computation_round,
    )
    cross_computation = _validate_main_rope_table_hlo(
        parse_hlo_module(cross_computation_hlo),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not cross_computation["passed"]
    assert cross_computation["final_round_count"] == 1
    assert cross_computation["direct_dataflow_final_round_count"] == 1

    forbidden = _validate_main_rope_table_hlo(
        parse_hlo_module(_main_rope_table_hlo(include_forbidden=True)),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not forbidden["passed"]
    assert forbidden["bf16_arithmetic"]

    unnamed = _validate_main_rope_table_hlo(
        parse_hlo_module(
            _main_rope_table_hlo().replace(
                'metadata={op_name="main_rope_table"}', ""
            )
        ),
        config=config,
        layers=1,
        enabled=True,
    )
    assert not unnamed["passed"]
    assert unnamed["named_table_parameter_count"] == 0

    default = _validate_main_rope_table_hlo(
        parse_hlo_module(_main_rope_table_hlo()),
        config=config,
        layers=1,
        enabled=False,
    )
    assert not default["passed"]


def _layer0_residual_discriminator_hlo(variant_name: str) -> str:
    flags = {
        "baseline_bf16": (False, False),
        "attention_output_fp32": (True, False),
        "dense_down_fp32": (False, True),
        "attention_output_and_dense_down_fp32": (True, True),
    }
    attention_fp32, dense_fp32 = flags[variant_name]
    replica_groups = (
        "{{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15},"
        "{16,17,18,19},{20,21,22,23},{24,25,26,27},{28,29,30,31}}"
    )
    gathers = "\n".join(
        f"  %ag.{index} = s32[4,1] all-gather(%s), dimensions={{0}}, "
        f"replica_groups={replica_groups}, channel_id={index + 1}, "
        "use_global_device_ids=true"
        for index in range(5)
    )
    reduction_dtypes = (
        *("f32",) * (int(attention_fp32) + int(dense_fp32)),
        *("bf16",) * (3 - int(attention_fp32) - int(dense_fp32)),
    )
    reductions = "\n".join(
        f"  %ar.{index} = {dtype}[1,6144] all-reduce("
        f"%{'f' if dtype == 'f32' else 'b'}), "
        f"replica_groups={replica_groups}, channel_id={index + 11}, "
        "use_global_device_ids=true, to_apply=%add"
        for index, dtype in enumerate(reduction_dtypes)
    )
    kernels = []
    if attention_fp32:
        kernels.append(
            '  %attention = f32[1,6144] custom-call(%f), '
            'custom_call_target="tpu_custom_call", '
            'backend_config="greenfield_fp8_block_matmul_f32_m8_k4096_n6144"'
        )
    else:
        kernels.append(
            '  %attention.bf16 = bf16[1,6144] custom-call(%b), '
            'custom_call_target="tpu_custom_call", '
            'backend_config="greenfield_fp8_block_matmul_m8_k4096_n6144"'
        )
    if dense_fp32:
        kernels.append(
            '  %dense = f32[1,6144] custom-call(%f), '
            'custom_call_target="tpu_custom_call", '
            'backend_config="greenfield_fp8_fused_block_swiglu_'
            'm8_h6144_i3072_o6144_downf32"'
        )
    else:
        kernels.append(
            '  %dense.bf16 = bf16[1,6144] custom-call(%b), '
            'custom_call_target="tpu_custom_call", '
            'backend_config="greenfield_fp8_fused_block_swiglu_'
            'm8_h6144_i3072_o6144"'
        )
    for index in range(int(attention_fp32) + int(dense_fp32)):
        kernels.append(
            f'  %boundary.{index} = bf16[1,6144] custom-call(%f), '
            'custom_call_target="tpu_custom_call", '
            'backend_config="greenfield_fp32_to_bf16_r8_h6144"'
        )
    kernels_text = "\n".join(kernels)
    return f'''HloModule layer0_{variant_name}, replica_count=1, num_partitions=32

%add (x: f32[], y: f32[]) -> f32[] {{
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %sum = f32[] add(%x, %y)
}}

ENTRY %main () -> (s32[1,2048], f32[1,2048], s32[1,1], bf16[1,6144], pred[1,1]) {{
  %s = s32[1] constant({{0}})
  %positions = s32[1,2048] constant({{0}})
  %scores = f32[1,2048] constant({{0}})
  %count = s32[1,1] constant({{0}})
  %valid = pred[1,1] constant({{true}})
  %f = f32[1,6144] constant({{0}})
  %b = bf16[1,6144] constant({{0}})
{gathers}
{reductions}
{kernels_text}
  %normalized = bf16[1,6144] constant({{0}}), metadata={{op_name="jit(probe)/greenfield_layer1_input_norm_{variant_name}/mul"}}
  ROOT %result = (s32[1,2048], f32[1,2048], s32[1,1], bf16[1,6144], pred[1,1]) tuple(%positions, %scores, %count, %normalized, %valid)
}}
'''


def _layer0_attention_schedule_discriminator_hlo(
    variant_name: str,
    *,
    tpu_rewritten_control: bool = False,
    tpu_flattened_cache_gather: bool = False,
) -> str:
    if variant_name not in {
        "attention_schedule_control",
        "replicated_monolithic_attention",
    }:
        raise ValueError("unknown attention-schedule test variant")
    monolithic = variant_name == "replicated_monolithic_attention"
    if monolithic and tpu_rewritten_control:
        raise ValueError("TPU owner-split rewrite applies only to the control")
    if tpu_flattened_cache_gather and not monolithic:
        raise ValueError(
            "TPU cache-gather flattening applies only to the challenger"
        )
    replica_groups = (
        "{{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15},"
        "{16,17,18,19},{20,21,22,23},{24,25,26,27},{28,29,30,31}}"
    )
    generic_gathers = "\n".join(
        f"  %ag.{index} = s32[4,1] all-gather(%s), dimensions={{0}}, "
        f"replica_groups={replica_groups}, channel_id={index + 1}, "
        "use_global_device_ids=true"
        for index in range(2)
    )
    if monolithic:
        gathered_cache_shape = (
            "bf16[96,64,192]"
            if tpu_flattened_cache_gather
            else "bf16[4,24,64,192]"
        )
        attention_gathers = (
            "  %cache = bf16[24,64,192] constant({0})\n"
            f"  %cache.ag = {gathered_cache_shape} all-gather(%cache), "
            f"dimensions={{0}}, replica_groups={replica_groups}, "
            "channel_id=3, use_global_device_ids=true, "
            "metadata={op_name=\"jit(probe)/"
            "greenfield_replicated_monolithic_attention_cache_gather/"
            "all_gather\"}\n"
            "  %attention.schedule = bf16[1,128,512] constant({0}), "
            "metadata={op_name=\"jit(probe)/"
            "greenfield_replicated_monolithic_attention/output\"}"
        )
    elif tpu_rewritten_control:
        attention_gathers = (
            "  %partial.output = bf16[1,128,512] constant({0})\n"
            "  %partial.output.ag = bf16[4,1,128,512] "
            "all-gather(%partial.output), dimensions={0}, "
            f"replica_groups={replica_groups}, channel_id=3, "
            "use_global_device_ids=true, metadata={op_name=\"jit(probe)/"
            "greenfield_owner_split_attention_output_gather/all_gather\"}\n"
            "  %partial.lse = f32[256] constant({0})\n"
            "  %partial.lse.ar = f32[256] all-reduce(%partial.lse), "
            f"replica_groups={replica_groups}, channel_id=4, "
            "use_global_device_ids=true, to_apply=%add_f32\n"
            "  %partial.lse.reshape = f32[4,1,64] "
            "reshape(%partial.lse.ar), metadata={op_name=\"jit(probe)/"
            "shard_map/cond/branch_1_fun/all_gather\"}"
        )
    else:
        scoped_shapes = (
            ("bf16[4,1,128,512]", "greenfield_owner_split_attention_output_gather"),
            ("f32[4,1,128]", "greenfield_owner_split_attention_lse_gather"),
            ("pred[4,1]", "greenfield_owner_split_attention_validity_gather"),
        )
        local_shapes = ("bf16[1,128,512]", "f32[1,128]", "pred[1]")
        attention_gathers = "\n".join(
            f"  %partial.{index} = {local_shapes[index]} constant({{0}})\n"
            f"  %partial.ag.{index} = {shape} all-gather(%partial.{index}), "
            f"dimensions={{0}}, replica_groups={replica_groups}, "
            f"channel_id={index + 3}, use_global_device_ids=true, "
            f'metadata={{op_name="jit(probe)/{scope}/all_gather"}}'
            for index, (shape, scope) in enumerate(scoped_shapes)
        )
    reductions = "\n".join(
        f"  %ar.{index} = bf16[1,6144] all-reduce(%b), "
        f"replica_groups={replica_groups}, channel_id={index + 11}, "
        "use_global_device_ids=true, to_apply=%add"
        for index in range(3)
    )
    main_rope_scope = "jit(probe)/shard_map/greenfield_main_rope_table"
    multiplies = "\n".join(
        f"  %rope.mul.{index} = f32[1,32] multiply(%f32, %f32), "
        f'metadata={{op_name="{main_rope_scope}/mul.{index}"}}'
        for index in range(8)
    )
    combines = "\n".join(
        f"  %rope.add.{index} = f32[1,32] add(%rope.mul.{2 * index}, "
        f"%rope.mul.{2 * index + 1}), "
        f'metadata={{op_name="{main_rope_scope}/add.{index}"}}'
        for index in range(4)
    )
    rounds = "\n".join(
        f"  %rope.round.{index} = bf16[1,32] convert(%rope.add.{index}), "
        f'metadata={{op_name="{main_rope_scope}/round.{index}"}}'
        for index in range(2)
    )
    return f'''HloModule layer0_{variant_name}, replica_count=1, num_partitions=32

%add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}

%add_f32 (x: f32[], y: f32[]) -> f32[] {{
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %sum = f32[] add(%x, %y)
}}

ENTRY %main (main_rope_table: bf16[8192,64]) -> (s32[1,2048], f32[1,2048], s32[1,1], bf16[1,6144], pred[1,1]) {{
  %main_rope_table = bf16[8192,64] parameter(0), metadata={{op_name="main_rope_table"}}
  %s = s32[1] constant({{0}})
  %positions = s32[1,2048] constant({{0}})
  %scores = f32[1,2048] constant({{0}})
  %count = s32[1,1] constant({{0}})
  %valid = pred[1,1] constant({{true}})
  %rope.zero = s32[] constant(0)
  %rope.table.row = bf16[1,64] dynamic-slice(%main_rope_table, %rope.zero, %rope.zero), dynamic_slice_sizes={{1,64}}, metadata={{op_name="greenfield_main_rope_table_lookup/dynamic_slice"}}
  %rope.table.half = bf16[1,32] slice(%rope.table.row), slice={{[0:1], [0:32]}}
  %f32 = f32[1,32] convert(%rope.table.half)
  %b = bf16[1,6144] constant({{0}})
{multiplies}
{combines}
{rounds}
{generic_gathers}
{attention_gathers}
{reductions}
  %attention = bf16[1,6144] custom-call(%b), custom_call_target="tpu_custom_call", backend_config="greenfield_fp8_block_matmul_m8_k4096_n6144"
  %dense = bf16[1,6144] custom-call(%b), custom_call_target="tpu_custom_call", backend_config="greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144"
  %normalized = bf16[1,6144] constant({{0}}), metadata={{op_name="jit(probe)/greenfield_layer1_input_norm_{variant_name}/mul"}}
  ROOT %result = (s32[1,2048], f32[1,2048], s32[1,1], bf16[1,6144], pred[1,1]) tuple(%positions, %scores, %count, %normalized, %valid)
}}
'''


def _layer0_attention_output_association_discriminator_hlo(
    variant_name: str,
    *,
    tpu_rewritten_owner_split: bool = False,
) -> str:
    control_name = "attention_output_association_control"
    associations = {
        "dcp_then_model_sequential_bf16",
        "dcp_then_model_pairwise_bf16",
        "model_then_dcp_sequential_bf16",
        "model_then_dcp_pairwise_bf16",
    }
    association = variant_name.removeprefix("attention_output_")
    if variant_name != control_name and association not in associations:
        raise ValueError("unknown attention-output association test variant")

    hlo = _layer0_attention_schedule_discriminator_hlo(
        "attention_schedule_control",
        tpu_rewritten_control=tpu_rewritten_owner_split,
    ).replace("attention_schedule_control", variant_name)
    if variant_name == control_name:
        return hlo

    replica_groups = (
        "{{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15},"
        "{16,17,18,19},{20,21,22,23},{24,25,26,27},{28,29,30,31}}"
    )
    old_reductions = "\n".join(
        f"  %ar.{index} = bf16[1,6144] all-reduce(%b), "
        f"replica_groups={replica_groups}, channel_id={index + 11}, "
        "use_global_device_ids=true, to_apply=%add"
        for index in range(3)
    )
    virtual_shape = (
        "bf16[1,6144]"
        if association.startswith("dcp_then_model_")
        else "bf16[8,1,6144]"
    )
    new_reductions = "\n".join(
        (
            f"  %ar.{index} = "
            f"{virtual_shape if index == 0 else 'bf16[1,6144]'} "
            "all-reduce(%b), "
            f"replica_groups={replica_groups}, channel_id={index + 11}, "
            "use_global_device_ids=true, to_apply=%add"
            + (
                ", metadata={op_name=\"jit(probe)/"
                f"greenfield_virtual_tp32_{association}/psum\""
                "}"
                if index == 0
                else ""
            )
        )
        for index in range(3)
    )
    production_attention = (
        "  %attention = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_block_matmul_m8_k4096_n6144"'
    )
    virtual_attention = "\n".join(
        f"  %attention.{index} = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_strategy_nd_o_m8_k512_n6144"'
        for index in range(8)
    )
    return hlo.replace(old_reductions, new_reductions).replace(
        production_attention,
        virtual_attention,
    )


def _layer0_strategy_nd_row0_discriminator_hlo(
    variant_name: str,
    *,
    flattened_gathers: bool = False,
) -> str:
    if variant_name not in {
        "strategy_nd_row0_control",
        "strategy_nd_row0_both",
    }:
        raise ValueError("unknown StrategyND row-zero test variant")
    hlo = _layer0_attention_schedule_discriminator_hlo(
        "attention_schedule_control",
        tpu_rewritten_control=True,
    ).replace("attention_schedule_control", variant_name)
    if variant_name == "strategy_nd_row0_control":
        return hlo

    replica_groups = (
        "{{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15},"
        "{16,17,18,19},{20,21,22,23},{24,25,26,27},{28,29,30,31}}"
    )
    result_shape = (
        "bf16[32,1,6144]"
        if flattened_gathers
        else "bf16[4,8,1,6144]"
    )
    gathers = (
        "  %strategy.partials = bf16[8,1,6144] constant({0})\n"
        + "\n".join(
            f"  %strategy.ag.{index} = {result_shape} all-gather("
            "%strategy.partials), dimensions={0}, "
            f"replica_groups={replica_groups}, channel_id={20 + index}, "
            "use_global_device_ids=true, metadata={op_name=\"jit(probe)/"
            + (
                "greenfield_strategy_nd_row0_attention_output/"
                if index == 0
                else "greenfield_strategy_nd_row0_dense_down/"
            )
            + "greenfield_strategy_nd_row0_association/"
            "greenfield_strategy_nd_row0_association_gather/all_gather\"}"
            for index in range(2)
        )
        + "\n"
    )
    production_attention = (
        "  %attention = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_block_matmul_m8_k4096_n6144"'
    )
    virtual_attention = "\n".join(
        f"  %attention.{index} = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_strategy_nd_o_m8_k512_n6144"'
        for index in range(8)
    )
    production_dense = (
        "  %dense = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_fused_block_swiglu_'
        'm8_h6144_i3072_o6144"'
    )
    virtual_dense = "\n".join(
        f"  %dense.{index} = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_fused_block_swiglu_'
        'm8_h6144_i384_o6144"'
        for index in range(8)
    )
    return (
        hlo.replace(
            "  %normalized =",
            gathers + "  %normalized =",
            1,
        )
        .replace(production_attention, virtual_attention)
        .replace(production_dense, virtual_dense)
    )


def _layer0_virtual_tp32_discriminator_hlo(variant_name: str) -> str:
    variants = {
        "dcp_then_model_sequential_bf16",
        "dcp_then_model_pairwise_bf16",
        "model_then_dcp_sequential_bf16",
        "model_then_dcp_pairwise_bf16",
    }
    if variant_name not in variants:
        raise ValueError("unknown virtual TP32 test variant")
    dcp_first = variant_name.startswith("dcp_then_model_")
    replica_groups = (
        "{{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15},"
        "{16,17,18,19},{20,21,22,23},{24,25,26,27},{28,29,30,31}}"
    )
    gathers = "\n".join(
        f"  %ag.{index} = s32[4,1] all-gather(%s), dimensions={{0}}, "
        f"replica_groups={replica_groups}, channel_id={index + 1}, "
        "use_global_device_ids=true"
        for index in range(5)
    )
    hidden_shapes = (
        ("bf16[1,6144]",) * 3
        if dcp_first
        else (
            "bf16[1,6144]",
            "bf16[8,1,6144]",
            "bf16[8,1,6144]",
        )
    )
    reductions = "\n".join(
        f"  %ar.{index} = {shape} all-reduce(%b), "
        f"replica_groups={replica_groups}, channel_id={index + 11}, "
        "use_global_device_ids=true, to_apply=%add, "
        f'metadata={{op_name="jit(probe)/greenfield_virtual_tp32_{variant_name}/psum"}}'
        for index, shape in enumerate(hidden_shapes)
    )
    attention = "\n".join(
        f"  %attention.{index} = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_strategy_nd_o_m8_k512_n6144"'
        for index in range(8)
    )
    dense = "\n".join(
        f"  %dense.{index} = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        'backend_config="greenfield_fp8_fused_block_swiglu_'
        'm8_h6144_i384_o6144"'
        for index in range(8)
    )
    return f'''HloModule layer0_{variant_name}, replica_count=1, num_partitions=32

%add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}

ENTRY %main () -> (s32[1,2048], f32[1,2048], s32[1,1], bf16[1,6144], pred[1,1]) {{
  %s = s32[1] constant({{0}})
  %positions = s32[1,2048] constant({{0}})
  %scores = f32[1,2048] constant({{0}})
  %count = s32[1,1] constant({{0}})
  %valid = pred[1,1] constant({{true}})
  %b = bf16[1,6144] constant({{0}})
{gathers}
{reductions}
{attention}
{dense}
  %normalized = bf16[1,6144] constant({{0}}), metadata={{op_name="jit(probe)/greenfield_layer1_input_norm_{variant_name}/mul"}}
  ROOT %result = (s32[1,2048], f32[1,2048], s32[1,1], bf16[1,6144], pred[1,1]) tuple(%positions, %scores, %count, %normalized, %valid)
}}
'''


def _layer0_ingredients_hlo(
    *,
    main_rope_table: bool = False,
    table_dataflow: bool = True,
    round_scope: bool = True,
    root_from_final_rounds: bool = True,
) -> str:
    replica_groups = (
        "{{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15},"
        "{16,17,18,19},{20,21,22,23},{24,25,26,27},{28,29,30,31}}"
    )
    gathers = "\n".join(
        f"  %ag.{index} = s32[4,1] all-gather(%s), dimensions={{0}}, "
        f"replica_groups={replica_groups}, channel_id={index + 1}, "
        "use_global_device_ids=true"
        for index in range(5)
    )
    reductions = "\n".join(
        f"  %ar.{index} = bf16[1,6144] all-reduce(%b), "
        f"replica_groups={replica_groups}, channel_id={index + 11}, "
        "use_global_device_ids=true, to_apply=%add"
        for index in range(3)
    )
    kernels = [
        "greenfield_fp8_block_matmul_m8_k4096_n6144",
        *(["greenfield_fp8_strategy_nd_o_m8_k512_n6144"] * 8),
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144",
        *(
            ["greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"]
            * 8
        ),
    ]
    kernel_text = "\n".join(
        f"  %kernel.{index} = bf16[1,6144] custom-call(%b), "
        'custom_call_target="tpu_custom_call", '
        f'backend_config="{name}"'
        for index, name in enumerate(kernels)
    )
    roots = (
        ("s32", (1, 2048)),
        ("f32", (1, 2048)),
        ("s32", (1, 1)),
        ("bf16", (1, 6144)),
        ("bf16", (1, 6144)),
        ("bf16", (1, 640)),
        ("s32", (1, 2048)),
        ("s32", (1, 1)),
        ("bf16", (1, 2048, 640)),
        ("pred", (1, 1)),
        ("bf16", (1, 64, 512)),
        ("f32", (1, 64)),
        ("pred", (1, 1)),
        ("bf16", (1, 64, 512)),
        ("f32", (1, 64)),
        ("pred", (1, 1)),
        ("bf16", (1, 16, 256)),
        ("bf16", (1, 4096)),
        ("bf16", (1, 8, 6144)),
        *(("bf16", (1, 6144)),) * 4,
        ("bf16", (1, 8, 6144)),
        *(("bf16", (1, 6144)),) * 4,
        ("pred", (1, 1)),
    )

    def shape(dtype: str, dimensions: tuple[int, ...]) -> str:
        return f"{dtype}[{','.join(str(value) for value in dimensions)}]"

    root_shapes = ", ".join(shape(*item) for item in roots)
    constants = []
    for index, item in enumerate(roots):
        if main_rope_table and table_dataflow and index == 17:
            continue
        metadata = (
            ', metadata={op_name="jit(probe)/'
            'greenfield_layer0_ingredients_layer1_norm/mul"}'
            if index == 27
            else ""
        )
        constants.append(
            f"  %root.{index} = {shape(*item)} constant({{0}}){metadata}"
        )
    operands = ", ".join(
        "%rope.output.input"
        if main_rope_table and table_dataflow and index == 17
        else f"%root.{index}"
        for index in range(len(roots))
    )
    table_signature = (
        "main_rope_table: bf16[8192,64]" if main_rope_table else ""
    )
    table_scope = "jit(probe)/shard_map/greenfield_main_rope_table"
    table_lines = ""
    if main_rope_table:
        products = "\n".join(
            f"  %rope.mul.{index} = f32[1,32] multiply(%rope.f32, "
            f"%rope.f32), metadata={{op_name=\"{table_scope}/mul.{index}\"}}"
            for index in range(8)
        )
        combines = "\n".join(
            f"  %rope.add.{index} = f32[1,32] add(%rope.mul.{2 * index}, "
            f"%rope.mul.{2 * index + 1}), "
            f"metadata={{op_name=\"{table_scope}/add.{index}\"}}"
            for index in range(4)
        )
        rounds = "\n".join(
            f"  %rope.round.{index} = bf16[1,32] convert(%rope.add.{index})"
            + (
                f", metadata={{op_name=\"{table_scope}/round.{index}\"}}"
                if round_scope
                else ""
            )
            for index in range(2)
        )
        rope_source = (
            "%rope.table.f32" if table_dataflow else "%rope.constant.f32"
        )
        output_lines = ""
        if table_dataflow:
            root_rounds = "%rope.rounds"
            bypass_lines = ""
            if not root_from_final_rounds:
                bypass_lines = (
                    "  %rope.bypass.add.0 = f32[1,32] add("
                    "%rope.add.0, %rope.add.0)\n"
                    "  %rope.bypass.add.1 = f32[1,32] add("
                    "%rope.add.1, %rope.add.1)\n"
                    "  %rope.bypass.round.0 = bf16[1,32] convert("
                    "%rope.bypass.add.0)\n"
                    "  %rope.bypass.round.1 = bf16[1,32] convert("
                    "%rope.bypass.add.1)\n"
                    "  %rope.bypass.rounds = bf16[1,64] concatenate("
                    "%rope.bypass.round.0, %rope.bypass.round.1), "
                    "dimensions={1}\n"
                )
                root_rounds = "%rope.bypass.rounds"
            output_lines = (
                "  %rope.rounds = bf16[1,64] concatenate(%rope.round.0, "
                "%rope.round.1), dimensions={1}\n"
                + bypass_lines
                + "  %rope.pad.value = bf16[] constant(0)\n"
                "  %rope.output.input = bf16[1,4096] pad("
                f"{root_rounds}, %rope.pad.value), padding=0_0x0_4032\n"
            )
        table_lines = f'''  %main_rope_table = bf16[8192,64] parameter(0), metadata={{op_name="main_rope_table"}}
  %rope.zero = s32[] constant(0)
  %rope.table.row = bf16[1,64] dynamic-slice(%main_rope_table, %rope.zero, %rope.zero), dynamic_slice_sizes={{1,64}}, metadata={{op_name="{table_scope}_lookup/dynamic_slice"}}
  %rope.table.half = bf16[1,32] slice(%rope.table.row), slice={{[0:1], [0:32]}}
  %rope.table.f32 = f32[1,32] convert(%rope.table.half)
  %rope.constant.f32 = f32[1,32] constant({{0}})
  %rope.f32 = f32[1,32] copy({rope_source})
{products}
{combines}
{rounds}
{output_lines}
'''
    return f'''HloModule layer0_ingredients, replica_count=1, num_partitions=32

%add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}

ENTRY %main ({table_signature}) -> ({root_shapes}) {{
{table_lines}  %s = s32[1] constant({{0}})
  %b = bf16[1,6144] constant({{0}})
{gathers}
{reductions}
{kernel_text}
{chr(10).join(constants)}
  ROOT %result = ({root_shapes}) tuple({operands})
}}
'''


def test_layer0_residual_discriminator_hlo_pins_isolated_arms() -> None:
    from glm_tpu.greenfield.runtime import (
        validate_layer0_residual_discriminator_hlo,
    )

    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4))
        for stage in range(8)
    )
    variant_names = (
        "baseline_bf16",
        "attention_output_fp32",
        "dense_down_fp32",
        "attention_output_and_dense_down_fp32",
    )
    for variant_name in variant_names:
        hlo = _layer0_residual_discriminator_hlo(variant_name)
        accepted = validate_layer0_residual_discriminator_hlo(
            hlo,
            config=_real_8k_decoder_config(dsa_score_default_precision=True),
            groups=groups,
            variant_name=variant_name,
        )
        assert accepted["passed"], accepted
        assert (
            accepted["kernel_counts"]
            == accepted["expected_kernel_counts"]
        )
        assert accepted["normalization_scope_present"]
        assert not accepted["multi_hidden_results"]
        assert not accepted["multi_scalar_reductions"]

    hlo = _layer0_residual_discriminator_hlo("attention_output_fp32")
    demoted = validate_layer0_residual_discriminator_hlo(
        hlo.replace("f32[1,6144] all-reduce", "bf16[1,6144] all-reduce"),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        variant_name="attention_output_fp32",
    )
    assert not demoted["passed"]
    assert any(
        "FP32 local-combine count drifted" in item
        for item in demoted["violations"]
    )

    escaped = validate_layer0_residual_discriminator_hlo(
        hlo.replace("{0,1,2,3}", "{0,1,2,4}"),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        variant_name="attention_output_fp32",
    )
    assert not escaped["passed"]
    assert escaped["escaped_collectives"]

    contaminated = _layer0_residual_discriminator_hlo(
        "baseline_bf16"
    ).replace(
        "  ROOT %result =",
        "  %bad.hidden = (bf16[1,6144], bf16[1,6144]) "
        "tuple(%normalized, %normalized), "
        "metadata={op_name=\"jit(probe)/"
        "greenfield_layer1_input_norm_baseline_bf16/mul\"}\n"
        "  ROOT %result =",
    )
    contaminated_contract = validate_layer0_residual_discriminator_hlo(
        contaminated,
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        variant_name="baseline_bf16",
    )
    assert not contaminated_contract["passed"]
    assert (
        "layer-0 discriminator tuple-fused multiple hidden rows"
        in contaminated_contract["violations"]
    )

    with pytest.raises(PlanValidationError, match="unknown.*variant"):
        validate_layer0_residual_discriminator_hlo(
            hlo,
            config=_real_8k_decoder_config(dsa_score_default_precision=True),
            groups=groups,
            variant_name="unknown",
        )


def test_layer0_attention_schedule_hlo_pins_control_and_challenger() -> None:
    from glm_tpu.greenfield.runtime import (
        validate_layer0_residual_discriminator_hlo,
    )

    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4))
        for stage in range(8)
    )
    replica_groups = "{" + ",".join(
        "{" + ",".join(str(rank) for rank in group) + "}"
        for group in groups
    ) + "}"
    config = _real_8k_decoder_config(dsa_score_default_precision=True)
    contracts = {}
    for variant_name in (
        "attention_schedule_control",
        "replicated_monolithic_attention",
    ):
        hlo = _layer0_attention_schedule_discriminator_hlo(variant_name)
        contracts[variant_name] = validate_layer0_residual_discriminator_hlo(
            hlo,
            config=config,
            groups=groups,
            variant_name=variant_name,
            main_rope_table_enabled=True,
        )
        assert contracts[variant_name]["passed"], contracts[variant_name]
        assert contracts[variant_name]["main_rope_table_contract"]["passed"]

    control = contracts["attention_schedule_control"]
    challenger = contracts["replicated_monolithic_attention"]
    assert len(control["owner_split_gathers"]) == 3
    assert len(control["owner_split_output_gathers"]) == 1
    assert not control["monolithic_cache_gathers"]
    assert not control["cache_shaped_gathers"]
    assert not control["monolithic_attention_scope_present"]
    assert not challenger["owner_split_gathers"]
    assert len(challenger["monolithic_cache_gathers"]) == 1
    assert len(challenger["cache_shaped_gathers"]) == 1
    assert challenger["monolithic_attention_scope_present"]

    tpu_challenger_hlo = _layer0_attention_schedule_discriminator_hlo(
        "replicated_monolithic_attention",
        tpu_flattened_cache_gather=True,
    )
    tpu_challenger = validate_layer0_residual_discriminator_hlo(
        tpu_challenger_hlo,
        config=config,
        groups=groups,
        variant_name="replicated_monolithic_attention",
        main_rope_table_enabled=True,
    )
    assert tpu_challenger["passed"], tpu_challenger
    assert len(tpu_challenger["monolithic_cache_gathers"]) == 1
    assert len(tpu_challenger["cache_shaped_gathers"]) == 1

    tpu_control_hlo = _layer0_attention_schedule_discriminator_hlo(
        "attention_schedule_control",
        tpu_rewritten_control=True,
    )
    tpu_control = validate_layer0_residual_discriminator_hlo(
        tpu_control_hlo,
        config=config,
        groups=groups,
        variant_name="attention_schedule_control",
        main_rope_table_enabled=True,
    )
    assert tpu_control["passed"], tpu_control
    assert len(tpu_control["owner_split_gathers"]) == 1
    assert len(tpu_control["owner_split_output_gathers"]) == 1
    assert not tpu_control["cache_shaped_gathers"]

    rogue_cache_hlo = tpu_control_hlo.replace(
        "  %normalized =",
        "  %rogue.cache = bf16[24,64,192] constant({0})\n"
        "  %rogue.cache.ag = bf16[4,24,64,192] "
        "all-gather(%rogue.cache), dimensions={0}, "
        f"replica_groups={replica_groups}, channel_id=99, "
        "use_global_device_ids=true\n"
        "  %normalized =",
        1,
    )
    rogue_cache = validate_layer0_residual_discriminator_hlo(
        rogue_cache_hlo,
        config=config,
        groups=groups,
        variant_name="attention_schedule_control",
        main_rope_table_enabled=True,
    )
    assert not rogue_cache["passed"]
    assert len(rogue_cache["cache_shaped_gathers"]) == 1
    assert "layer-0 attention control contains a full-cache gather" in (
        rogue_cache["violations"]
    )

    challenger_hlo = _layer0_attention_schedule_discriminator_hlo(
        "replicated_monolithic_attention"
    )
    missing_cache = validate_layer0_residual_discriminator_hlo(
        challenger_hlo.replace(
            "greenfield_replicated_monolithic_attention_cache_gather",
            "missing_monolithic_cache_scope",
        ),
        config=config,
        groups=groups,
        variant_name="replicated_monolithic_attention",
        main_rope_table_enabled=True,
    )
    assert not missing_cache["passed"]
    assert not missing_cache["monolithic_cache_gathers"]

    escaped = validate_layer0_residual_discriminator_hlo(
        challenger_hlo.replace("{0,1,2,3}", "{0,1,2,4}"),
        config=config,
        groups=groups,
        variant_name="replicated_monolithic_attention",
        main_rope_table_enabled=True,
    )
    assert not escaped["passed"]
    assert escaped["escaped_collectives"]

    contaminated_control = validate_layer0_residual_discriminator_hlo(
        _layer0_attention_schedule_discriminator_hlo(
            "attention_schedule_control"
        ).replace(
            "greenfield_owner_split_attention_output_gather",
            "greenfield_replicated_monolithic_attention",
        ),
        config=config,
        groups=groups,
        variant_name="attention_schedule_control",
        main_rope_table_enabled=True,
    )
    assert not contaminated_control["passed"]
    assert contaminated_control["monolithic_attention_scope_present"]


def test_layer0_attention_output_association_hlo_isolates_projection() -> None:
    from glm_tpu.greenfield.runtime import (
        validate_layer0_residual_discriminator_hlo,
    )

    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4))
        for stage in range(8)
    )
    config = _real_8k_decoder_config(dsa_score_default_precision=True)
    variants = (
        "attention_output_association_control",
        "attention_output_dcp_then_model_sequential_bf16",
        "attention_output_dcp_then_model_pairwise_bf16",
        "attention_output_model_then_dcp_sequential_bf16",
        "attention_output_model_then_dcp_pairwise_bf16",
    )
    contracts = {}
    for variant_name in variants:
        hlo = _layer0_attention_output_association_discriminator_hlo(
            variant_name
        )
        contract = validate_layer0_residual_discriminator_hlo(
            hlo,
            config=config,
            groups=groups,
            variant_name=variant_name,
            main_rope_table_enabled=True,
        )
        assert contract["passed"], contract
        assert contract["discriminator_kind"] == (
            "attention_output_association"
        )
        assert len(contract["owner_split_output_gathers"]) == 1
        assert not contract["cache_shaped_gathers"]
        contracts[variant_name] = contract

    control = contracts["attention_output_association_control"]
    assert control["kernel_counts"][
        "greenfield_fp8_block_matmul_m8_k4096_n6144"
    ] == 1
    assert control["kernel_counts"][
        "greenfield_fp8_strategy_nd_o_m8_k512_n6144"
    ] == 0
    for variant_name in variants[1:]:
        candidate = contracts[variant_name]
        assert candidate["kernel_counts"][
            "greenfield_fp8_block_matmul_m8_k4096_n6144"
        ] == 0
        assert candidate["kernel_counts"][
            "greenfield_fp8_strategy_nd_o_m8_k512_n6144"
        ] == 8
        assert candidate["kernel_counts"][
            "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144"
        ] == 1
        assert candidate["kernel_counts"][
            "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"
        ] == 0
        assert candidate["virtual_tp32_association_scope_present"]

    tpu_rewritten = validate_layer0_residual_discriminator_hlo(
        _layer0_attention_output_association_discriminator_hlo(
            "attention_output_model_then_dcp_pairwise_bf16",
            tpu_rewritten_owner_split=True,
        ),
        config=config,
        groups=groups,
        variant_name="attention_output_model_then_dcp_pairwise_bf16",
        main_rope_table_enabled=True,
    )
    assert tpu_rewritten["passed"], tpu_rewritten
    assert tpu_rewritten["collective_counts"]["all-gather"] == 3

    model_first_hlo = (
        _layer0_attention_output_association_discriminator_hlo(
            "attention_output_model_then_dcp_pairwise_bf16"
        )
    )
    wrong_shape = validate_layer0_residual_discriminator_hlo(
        model_first_hlo.replace("bf16[8,1,6144]", "bf16[1,6144]"),
        config=config,
        groups=groups,
        variant_name="attention_output_model_then_dcp_pairwise_bf16",
        main_rope_table_enabled=True,
    )
    assert not wrong_shape["passed"]
    assert any(
        "virtual TP32 collective association drifted" in item
        for item in wrong_shape["violations"]
    )

    virtual_dense = validate_layer0_residual_discriminator_hlo(
        model_first_hlo.replace(
            "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144",
            "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144",
        ),
        config=config,
        groups=groups,
        variant_name="attention_output_model_then_dcp_pairwise_bf16",
        main_rope_table_enabled=True,
    )
    assert not virtual_dense["passed"]
    assert any(
        "kernel association drifted" in item
        for item in virtual_dense["violations"]
    )


def test_layer0_attention_output_specs_leave_dense_on_production_path() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _layer0_residual_discriminator_specs,
    )

    specs = _layer0_residual_discriminator_specs(
        "attention_output_association"
    )
    assert tuple(spec[0] for spec in specs) == (
        "attention_output_association_control",
        "attention_output_dcp_then_model_sequential_bf16",
        "attention_output_dcp_then_model_pairwise_bf16",
        "attention_output_model_then_dcp_sequential_bf16",
        "attention_output_model_then_dcp_pairwise_bf16",
    )
    assert specs[0][3:] == (None, False, False)
    for spec in specs[1:]:
        assert spec[1:3] == (False, False)
        assert spec[3] is not None
        assert spec[4:6] == (True, False)


def test_layer0_strategy_nd_row0_hlo_pins_two_local_partial_gathers() -> None:
    from glm_tpu.greenfield.runtime import (
        validate_layer0_residual_discriminator_hlo,
    )

    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4))
        for stage in range(8)
    )
    config = _real_8k_decoder_config(dsa_score_default_precision=True)
    control = validate_layer0_residual_discriminator_hlo(
        _layer0_strategy_nd_row0_discriminator_hlo(
            "strategy_nd_row0_control"
        ),
        config=config,
        groups=groups,
        variant_name="strategy_nd_row0_control",
        main_rope_table_enabled=True,
    )
    assert control["passed"], control
    assert control["discriminator_kind"] == "strategy_nd_row0_association"
    assert not control["strategy_nd_gathers"]

    for flattened in (False, True):
        candidate_hlo = _layer0_strategy_nd_row0_discriminator_hlo(
            "strategy_nd_row0_both",
            flattened_gathers=flattened,
        )
        candidate = validate_layer0_residual_discriminator_hlo(
            candidate_hlo,
            config=config,
            groups=groups,
            variant_name="strategy_nd_row0_both",
            main_rope_table_enabled=True,
        )
        assert candidate["passed"], candidate
        assert len(candidate["strategy_nd_gathers"]) == 2
        assert candidate["strategy_nd_shaped_gather_count"] == 2
        assert candidate["strategy_nd_attention_gather_count"] == 1
        assert candidate["strategy_nd_dense_gather_count"] == 1
        assert candidate["kernel_counts"][
            "greenfield_fp8_strategy_nd_o_m8_k512_n6144"
        ] == 8
        assert candidate["kernel_counts"][
            "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"
        ] == 8
        assert candidate["virtual_tp32_association_scope_present"]

    missing = validate_layer0_residual_discriminator_hlo(
        _layer0_strategy_nd_row0_discriminator_hlo(
            "strategy_nd_row0_both"
        ).replace(
            "greenfield_strategy_nd_row0_association_gather",
            "missing_strategy_nd_gather_scope",
            1,
        ),
        config=config,
        groups=groups,
        variant_name="strategy_nd_row0_both",
        main_rope_table_enabled=True,
    )
    assert not missing["passed"]
    assert len(missing["strategy_nd_gathers"]) == 1

    wrong_shape = validate_layer0_residual_discriminator_hlo(
        _layer0_strategy_nd_row0_discriminator_hlo(
            "strategy_nd_row0_both"
        ).replace("bf16[4,8,1,6144]", "bf16[4,7,1,6144]", 1),
        config=config,
        groups=groups,
        variant_name="strategy_nd_row0_both",
        main_rope_table_enabled=True,
    )
    assert not wrong_shape["passed"]
    assert wrong_shape["strategy_nd_shaped_gather_count"] == 1

    extra_unscoped = validate_layer0_residual_discriminator_hlo(
        _layer0_strategy_nd_row0_discriminator_hlo(
            "strategy_nd_row0_both"
        ).replace(
            "  %normalized =",
            "  %extra.strategy.ag = bf16[4,8,1,6144] all-gather("
            "%strategy.partials), dimensions={0}, "
            f"replica_groups={{{{0,1,2,3}},{{4,5,6,7}},"
            "{8,9,10,11},{12,13,14,15},{16,17,18,19},"
            "{20,21,22,23},{24,25,26,27},{28,29,30,31}}}, "
            "channel_id=99, use_global_device_ids=true\n"
            "  %normalized =",
            1,
        ),
        config=config,
        groups=groups,
        variant_name="strategy_nd_row0_both",
        main_rope_table_enabled=True,
    )
    assert not extra_unscoped["passed"]
    assert extra_unscoped["strategy_nd_shaped_gather_count"] == 3
    assert any(
        "unscoped partial gather" in violation
        for violation in extra_unscoped["violations"]
    )


def test_layer0_strategy_nd_row0_specs_apply_to_attention_and_dense() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _layer0_residual_discriminator_specs,
    )

    specs = _layer0_residual_discriminator_specs(
        "strategy_nd_row0_association"
    )
    assert specs == (
        (
            "strategy_nd_row0_control",
            False,
            False,
            None,
            False,
            False,
        ),
        (
            "strategy_nd_row0_both",
            False,
            False,
            "strategy_nd_row0_bf16",
            False,
            False,
        ),
    )


def test_layer0_ingredients_hlo_pins_primitive_capture() -> None:
    from glm_tpu.greenfield.runtime import (
        validate_layer0_ingredients_observer_hlo,
    )

    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4)) for stage in range(8)
    )
    hlo = _layer0_ingredients_hlo()
    accepted = validate_layer0_ingredients_observer_hlo(
        hlo,
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
    )
    assert accepted["passed"], accepted
    assert accepted["kernel_counts"] == accepted["expected_kernel_counts"]
    assert len(accepted["ingredient_names"]) == 29

    table_on = validate_layer0_ingredients_observer_hlo(
        _layer0_ingredients_hlo(main_rope_table=True),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        main_rope_table_enabled=True,
    )
    assert table_on["passed"], table_on
    assert table_on["main_rope_table_contract"]["passed"]
    assert table_on["main_rope_table_contract"][
        "required_entry_root_indices"
    ] == [17]
    assert table_on["main_rope_table_contract"][
        "table_dependent_root_indices"
    ] == [17]
    assert table_on["main_rope_table_contract"][
        "final_round_dependent_root_indices"
    ] == [17]

    unused_table = validate_layer0_ingredients_observer_hlo(
        _layer0_ingredients_hlo(
            main_rope_table=True,
            table_dataflow=False,
        ),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        main_rope_table_enabled=True,
    )
    assert not unused_table["passed"]
    assert any(
        "products do not depend on the table lookup" in violation
        for violation in unused_table["violations"]
    )
    assert any(
        "captured roots do not depend on the table" in violation
        for violation in unused_table["violations"]
    )

    bypassed_final_rounds = validate_layer0_ingredients_observer_hlo(
        _layer0_ingredients_hlo(
            main_rope_table=True,
            round_scope=False,
            root_from_final_rounds=False,
        ),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        main_rope_table_enabled=True,
    )
    assert not bypassed_final_rounds["passed"]
    assert bypassed_final_rounds["main_rope_table_contract"][
        "table_dependent_root_indices"
    ] == [17]
    assert bypassed_final_rounds["main_rope_table_contract"][
        "final_round_dependent_root_indices"
    ] == []
    assert any(
        "captured roots do not depend on final BF16 rounds" in violation
        for violation in bypassed_final_rounds["violations"]
    )

    missing_table = validate_layer0_ingredients_observer_hlo(
        hlo,
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        main_rope_table_enabled=True,
    )
    assert not missing_table["passed"]
    assert not missing_table["main_rope_table_contract"]["passed"]

    escaped = validate_layer0_ingredients_observer_hlo(
        hlo.replace("{0,1,2,3}", "{0,1,2,4}"),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
    )
    assert not escaped["passed"]
    assert escaped["escaped_collectives"]

    missing_partial = validate_layer0_ingredients_observer_hlo(
        hlo.replace(
            "greenfield_fp8_strategy_nd_o_m8_k512_n6144",
            "greenfield_fp8_block_matmul_m8_k4096_n6144",
            1,
        ),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
    )
    assert not missing_partial["passed"]
    assert any(
        "kernel set drifted" in item
        for item in missing_partial["violations"]
    )


def test_layer0_virtual_tp32_discriminator_hlo_pins_subshards() -> None:
    from glm_tpu.greenfield.runtime import (
        validate_layer0_residual_discriminator_hlo,
    )

    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4))
        for stage in range(8)
    )
    variants = (
        "dcp_then_model_sequential_bf16",
        "dcp_then_model_pairwise_bf16",
        "model_then_dcp_sequential_bf16",
        "model_then_dcp_pairwise_bf16",
    )
    for variant_name in variants:
        accepted = validate_layer0_residual_discriminator_hlo(
            _layer0_virtual_tp32_discriminator_hlo(variant_name),
            config=_real_8k_decoder_config(dsa_score_default_precision=True),
            groups=groups,
            variant_name=variant_name,
        )
        assert accepted["passed"], accepted
        assert accepted["discriminator_kind"] == "virtual_tp32"
        assert accepted["kernel_counts"] == accepted["expected_kernel_counts"]
        assert accepted["virtual_tp32_association_scope_present"]

    model_first = _layer0_virtual_tp32_discriminator_hlo(
        "model_then_dcp_pairwise_bf16"
    )
    wrong_shape = validate_layer0_residual_discriminator_hlo(
        model_first.replace("bf16[8,1,6144]", "bf16[1,6144]"),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        variant_name="model_then_dcp_pairwise_bf16",
    )
    assert not wrong_shape["passed"]
    assert any(
        "virtual TP32 collective association drifted" in item
        for item in wrong_shape["violations"]
    )

    missing_subshard = validate_layer0_residual_discriminator_hlo(
        model_first.replace(
            "greenfield_fp8_strategy_nd_o_m8_k512_n6144",
            "greenfield_fp8_block_matmul_m8_k4096_n6144",
            1,
        ),
        config=_real_8k_decoder_config(dsa_score_default_precision=True),
        groups=groups,
        variant_name="model_then_dcp_pairwise_bf16",
    )
    assert not missing_subshard["passed"]
    assert any(
        "kernel association drifted" in item
        for item in missing_subshard["violations"]
    )


def test_layer0_residual_discriminator_traces_real_stage0_shape() -> None:
    program = r'''
import json
from dataclasses import replace
import jax
import jax.numpy as jnp
import ml_dtypes
from jax.sharding import NamedSharding
from glm_tpu.greenfield.model import build_decoder_runtime_weight_layout, build_decoder_state_layout, build_pipeline_schedule
from glm_tpu.greenfield.runtime import build_decoder_step_program
from tests.greenfield.checkpoint.test_runtime_pack import _small_plan

base = _small_plan()
geometry = replace(
    base.geometry,
    num_layers=9,
    mlp_layer_types=("dense", "dense", "dense", *("sparse",) * 6),
    indexer_types=("full",) * 9,
)
assignments = tuple(
    replace(
        assignment,
        layer_start=0 if stage == 0 else stage + 1,
        layer_end_exclusive=2 if stage == 0 else stage + 2,
    )
    for stage, assignment in enumerate(base.stage_assignments)
)
plan = replace(base, geometry=geometry, stage_assignments=assignments)
schedule = build_pipeline_schedule(plan)
state = build_decoder_state_layout(
    plan,
    schedule,
    context_capacity=8,
    logical_page_size=8,
    packed_kv_width=8,
)
weight_layout = build_decoder_runtime_weight_layout(plan, schedule)
groups = tuple(
    tuple(stage * 4 + slot for slot in range(4))
    for stage in range(8)
)
pairs = tuple(
    (groups[stage][slot], groups[(stage + 1) % 8][slot])
    for stage in range(8)
    for slot in range(4)
)
decoder = build_decoder_step_program(
    plan,
    schedule,
    state,
    weight_layout,
    groups,
    pairs,
    complete_token_path=True,
    split_residual_state=True,
    build_layer0_residual_discriminator=True,
)
assert len(decoder.layer0_residual_discriminators) == 4
exact_decoder = build_decoder_step_program(
    plan,
    schedule,
    state,
    weight_layout,
    groups,
    pairs,
    complete_token_path=True,
    split_residual_state=True,
    dsa_query_backend="reference",
    dsa_query_exact_association=True,
    dsa_head_key_exact_association=True,
    build_layer0_residual_discriminator=True,
)
assert len(exact_decoder.layer0_residual_discriminators) == 4
virtual_decoder = build_decoder_step_program(
    plan,
    schedule,
    state,
    weight_layout,
    groups,
    pairs,
    complete_token_path=True,
    split_residual_state=True,
    build_layer0_residual_discriminator=True,
    layer0_residual_discriminator_kind="virtual_tp32",
)
assert len(virtual_decoder.layer0_residual_discriminators) == 4
attention_decoder = build_decoder_step_program(
    plan,
    schedule,
    state,
    weight_layout,
    groups,
    pairs,
    complete_token_path=True,
    split_residual_state=True,
    main_rope_table_enabled=True,
    build_layer0_residual_discriminator=True,
    layer0_residual_discriminator_kind="attention_schedule",
)
assert len(attention_decoder.layer0_residual_discriminators) == 2
attention_output_decoder = build_decoder_step_program(
    plan,
    schedule,
    state,
    weight_layout,
    groups,
    pairs,
    complete_token_path=True,
    split_residual_state=True,
    main_rope_table_enabled=True,
    build_layer0_residual_discriminator=True,
    layer0_residual_discriminator_kind="attention_output_association",
)
assert len(attention_output_decoder.layer0_residual_discriminators) == 5
strategy_nd_decoder = build_decoder_step_program(
    plan,
    schedule,
    state,
    weight_layout,
    groups,
    pairs,
    complete_token_path=True,
    split_residual_state=True,
    main_rope_table_enabled=True,
    build_layer0_residual_discriminator=True,
    layer0_residual_discriminator_kind="strategy_nd_row0_association",
)
assert len(strategy_nd_decoder.layer0_residual_discriminators) == 2

def abstract(shape, dtype, spec):
    return jax.ShapeDtypeStruct(
        shape,
        dtype,
        sharding=NamedSharding(decoder.mesh, spec),
    )

weights = {}
for spec in weight_layout.specs:
    dtype = {
        "F8_E4M3": jnp.uint8,
        "F32": jnp.float32,
    }.get(spec.dtype, ml_dtypes.bfloat16)
    weights[spec.name] = abstract(
        (32, *spec.shape),
        dtype,
        decoder.input_specs[0][spec.name],
    )
state_shape = state.padded_state_shape_per_device
inputs = (
    weights,
    abstract((32, 2, 1, 8), ml_dtypes.bfloat16, decoder.input_specs[1]),
    abstract((32, *state_shape["kv"]), ml_dtypes.bfloat16, decoder.input_specs[2]),
    abstract((32, *state_shape["index_keys"]), ml_dtypes.bfloat16, decoder.input_specs[3]),
    abstract((32, 1, decoder.config.metadata_width), jnp.int32, decoder.input_specs[4]),
    abstract((32, 1), jnp.int32, decoder.input_specs[5]),
    abstract((1,), jnp.int32, decoder.input_specs[6]),
    abstract((1, 1), jnp.int32, decoder.input_specs[7]),
    abstract((1,), jnp.int32, decoder.input_specs[8]),
)
specs_by_name = {spec.name: spec for spec in weight_layout.specs}
query_owners = tuple(
    abstract(
        (
            32,
            *specs_by_name[
                f"indexer.slot_{slot:02d}.wq_b.weight_bits"
            ].shape,
        ),
        jnp.float32,
        exact_decoder.input_specs[1][0][0][slot],
    )
    for slot in range(exact_decoder.config.maximum_full_indexer_slots)
)
query_aliases = (query_owners,) * 4
wk_owners = tuple(
    abstract(
        (
            32,
            *specs_by_name[
                f"indexer.slot_{slot:02d}.wk.weight_bits"
            ].shape,
        ),
        jnp.float32,
        exact_decoder.input_specs[1][1][slot],
    )
    for slot in range(exact_decoder.config.maximum_full_indexer_slots)
)
exact_inputs = (weights, (query_aliases, wk_owners), *inputs[1:])
attention_inputs = (
    *inputs,
    abstract(
        (state.context_capacity, geometry.qk_rope_head_dim),
        ml_dtypes.bfloat16,
        attention_decoder.input_specs[-1],
    ),
)
stablehlos = {
    name: jax.jit(discriminator).lower(*inputs).as_text()
    for name, discriminator in decoder.layer0_residual_discriminators
}
exact_stablehlos = {
    name: jax.jit(discriminator).lower(*exact_inputs).as_text()
    for name, discriminator in exact_decoder.layer0_residual_discriminators
}
attention_stablehlos = {
    name: jax.jit(discriminator).lower(*attention_inputs).as_text()
    for name, discriminator in attention_decoder.layer0_residual_discriminators
}
print(json.dumps({
    "all_reduce_counts": {
        name: stablehlo.count("stablehlo.all_reduce")
        for name, stablehlo in stablehlos.items()
    },
    "distinct_stablehlo": len(set(stablehlos.values())),
    "exact_distinct_stablehlo": len(set(exact_stablehlos.values())),
    "exact_single_variant_outputs": {
        name: "tensor<32x8xbf16>" in stablehlo
        for name, stablehlo in exact_stablehlos.items()
    },
    "exact_variant_names": list(exact_stablehlos),
    "attention_discriminator_kind": (
        attention_decoder.layer0_residual_discriminator_kind
    ),
    "attention_distinct_stablehlo": len(set(attention_stablehlos.values())),
    "attention_main_rope_input": len(attention_decoder.input_specs) == 10,
    "attention_variant_names": list(attention_stablehlos),
    "attention_all_gather_counts": {
        name: stablehlo.count("stablehlo.all_gather")
        for name, stablehlo in attention_stablehlos.items()
    },
    "attention_output_discriminator_kind": (
        attention_output_decoder.layer0_residual_discriminator_kind
    ),
    "attention_output_main_rope_input": (
        len(attention_output_decoder.input_specs) == 10
    ),
    "attention_output_program_count": len(
        attention_output_decoder.layer0_residual_discriminators
    ),
    "attention_output_program_objects_distinct": len({
        id(discriminator)
        for _, discriminator in (
            attention_output_decoder.layer0_residual_discriminators
        )
    }),
    "attention_output_variant_names": [
        name
        for name, _ in (
            attention_output_decoder.layer0_residual_discriminators
        )
    ],
    "strategy_nd_discriminator_kind": (
        strategy_nd_decoder.layer0_residual_discriminator_kind
    ),
    "strategy_nd_program_count": len(
        strategy_nd_decoder.layer0_residual_discriminators
    ),
    "strategy_nd_program_objects_distinct": len({
        id(discriminator)
        for _, discriminator in (
            strategy_nd_decoder.layer0_residual_discriminators
        )
    }),
    "strategy_nd_variant_names": [
        name
        for name, _ in strategy_nd_decoder.layer0_residual_discriminators
    ],
    "virtual_discriminator_kind": (
        virtual_decoder.layer0_residual_discriminator_kind
    ),
    "virtual_variant_names": [
        name for name, _ in virtual_decoder.layer0_residual_discriminators
    ],
    "multi_variant_outputs": {
        name: "tensor<32x4x8xbf16>" in stablehlo
        for name, stablehlo in stablehlos.items()
    },
    "single_variant_outputs": {
        name: "tensor<32x8xbf16>" in stablehlo
        for name, stablehlo in stablehlos.items()
    },
    "stage0_layers": [layer.layer_id for layer in schedule.stages[0].layers],
    "variant_names": list(stablehlos),
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["stage0_layers"] == [0, 1]
    assert result["variant_names"] == [
        "baseline_bf16",
        "attention_output_fp32",
        "dense_down_fp32",
        "attention_output_and_dense_down_fp32",
    ]
    assert result["distinct_stablehlo"] == 4
    assert result["exact_distinct_stablehlo"] == 4
    assert result["exact_variant_names"] == result["variant_names"]
    assert result["attention_discriminator_kind"] == "attention_schedule"
    assert result["attention_distinct_stablehlo"] == 2
    assert result["attention_main_rope_input"]
    assert result["attention_variant_names"] == [
        "attention_schedule_control",
        "replicated_monolithic_attention",
    ]
    assert result["attention_output_discriminator_kind"] == (
        "attention_output_association"
    )
    assert result["attention_output_main_rope_input"]
    assert result["attention_output_program_count"] == 5
    assert result["attention_output_program_objects_distinct"] == 5
    assert result["attention_output_variant_names"] == [
        "attention_output_association_control",
        "attention_output_dcp_then_model_sequential_bf16",
        "attention_output_dcp_then_model_pairwise_bf16",
        "attention_output_model_then_dcp_sequential_bf16",
        "attention_output_model_then_dcp_pairwise_bf16",
    ]
    assert result["strategy_nd_discriminator_kind"] == (
        "strategy_nd_row0_association"
    )
    assert result["strategy_nd_program_count"] == 2
    assert result["strategy_nd_program_objects_distinct"] == 2
    assert result["strategy_nd_variant_names"] == [
        "strategy_nd_row0_control",
        "strategy_nd_row0_both",
    ]
    assert (
        result["attention_all_gather_counts"][
            "replicated_monolithic_attention"
        ]
        == result["attention_all_gather_counts"][
            "attention_schedule_control"
        ]
        - 2
    )
    assert result["virtual_discriminator_kind"] == "virtual_tp32"
    assert result["virtual_variant_names"] == [
        "dcp_then_model_sequential_bf16",
        "dcp_then_model_pairwise_bf16",
        "model_then_dcp_sequential_bf16",
        "model_then_dcp_pairwise_bf16",
    ]
    assert all(result["exact_single_variant_outputs"].values())
    assert all(result["single_variant_outputs"].values())
    assert not any(result["multi_variant_outputs"].values())
    assert all(
        count >= 3 for count in result["all_reduce_counts"].values()
    )


def test_8k_dsa_head_score_shape_requires_exact_dataflow() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    config = _real_8k_decoder_config()
    hlo = _synthetic_8k_dsa_score_hlo()
    record = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(hlo),
        config=config,
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert record["passed"], record

    assert record["score_body_count"] == 1
    assert record["score_dimensions"] == [32, 2048]
    assert len(record["allowed_dsa_score_shapes"]) == 10
    assert record["forbidden_shapes"] == []
    assert record["body_records"][0] == {
        "computation": (
            "%score (q: f32[32,128], key: bf16[2048,128], "
            "weight: f32[32]) -> f32[2048]"
        ),
        "contraction_count": 1,
        "head_weight_broadcast_count": 1,
        "head_weight_multiply_count": 1,
        "highest_precision_contraction_count": 1,
        "opcode_counts": {
            "bitcast": 1,
            "broadcast": 2,
            "convolution": 1,
            "maximum": 1,
            "multiply": 1,
        },
        "reduction_count": 1,
        "score_shape_occurrences": 10,
        "valid": True,
    }
    assert record["score_precision"] == "highest"

    default_hlo = hlo.replace(
        ", operand_precision={highest,highest}", ""
    )
    default_record = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(default_hlo),
        config=_real_8k_decoder_config(
            dsa_score_default_precision=True
        ),
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert default_record["passed"], default_record
    assert default_record["score_precision"] == "default"
    assert default_record["body_records"][0][
        "highest_precision_contraction_count"
    ] == 0

    wrong_default = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(default_hlo),
        config=config,
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not wrong_default["passed"]
    wrong_highest = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(hlo),
        config=_real_8k_decoder_config(
            dsa_score_default_precision=True
        ),
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not wrong_highest["passed"]

    drifted = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(
            hlo.replace(
                'op_name="jit(mapped_token)/shard_map/cond/'
                'branch_1_fun/mul"',
                'op_name="jit(mapped_token)/shard_map/batch_rows/mul"',
            )
        ),
        config=config,
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not drifted["passed"]
    assert drifted["score_body_count"] == 0
    assert drifted["allowed_dsa_score_shapes"] == []
    assert len(drifted["forbidden_shapes"]) == 10
    assert drifted["violations"] == [
        "decoder local DSA score-body contract drifted: expected=1 observed=0"
    ]


def test_8k_dsa_shape_exception_does_not_admit_dead_rows() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    hlo = _synthetic_8k_dsa_score_hlo() + '''
ENTRY %main (unrelated: f32[32,2048], wrong_dtype: bf16[32,2048], hidden: bf16[32,6144], dead_query: f32[32,32,128]) -> f32[32,2048] {
  %unrelated = f32[32,2048] parameter(0)
  %wrong_dtype = bf16[32,2048] parameter(1)
  %hidden = bf16[32,6144] parameter(2)
  %dead_query = f32[32,32,128] parameter(3)
  ROOT %root = f32[32,2048] copy(%unrelated), metadata={op_name="batch_rows/copy"}
}
'''
    record = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(hlo),
        config=_real_8k_decoder_config(),
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not record["passed"]
    assert record["score_body_count"] == 1
    assert len(record["allowed_dsa_score_shapes"]) == 10
    assert record["violations"] == []
    forbidden = {
        (
            item["instruction"],
            item["shape"]["dtype"],
            tuple(item["shape"]["dimensions"]),
        )
        for item in record["forbidden_shapes"]
    }
    assert forbidden == {
        ("%unrelated", "f32", (32, 2048)),
        ("%wrong_dtype", "bf16", (32, 2048)),
        ("%hidden", "bf16", (32, 6144)),
        ("%dead_query", "f32", (32, 32, 128)),
        ("%root", "f32", (32, 2048)),
        ("%root", "f32", (32, 2048)),
    }


def test_teacher_forced_prefill_builder_rejects_invalid_contracts() -> None:
    from types import SimpleNamespace

    from glm_tpu.greenfield.errors import PlanValidationError
    from glm_tpu.greenfield.runtime import build_teacher_forced_prefill_program

    config = SimpleNamespace(context_capacity=8, total_devices=32)
    body_only = SimpleNamespace(complete_token_path=False, config=config)
    complete = SimpleNamespace(
        complete_token_path=True,
        config=config,
        dense_final_layout_convolution=False,
        main_rope_table_enabled=False,
        observe_prefill_index_inputs=False,
    )

    with pytest.raises(PlanValidationError, match="complete-token decoder"):
        build_teacher_forced_prefill_program(body_only, prompt_length=2)
    for value in (True, 0, -1):
        with pytest.raises(PlanValidationError, match="must be positive"):
            build_teacher_forced_prefill_program(complete, prompt_length=value)
    with pytest.raises(PlanValidationError, match="leave capacity"):
        build_teacher_forced_prefill_program(complete, prompt_length=8)

    import jax.numpy as jnp

    disabled = build_teacher_forced_prefill_program(complete, prompt_length=2)
    with pytest.raises(PlanValidationError, match="table state drifted"):
        disabled.execute(
            None,
            None,
            None,
            None,
            None,
            jnp.asarray([1, 2], jnp.int32),
            None,
            None,
            None,
            None,
            None,
            jnp.zeros((8, 64), jnp.bfloat16),
        )
    enabled = build_teacher_forced_prefill_program(
        SimpleNamespace(
            complete_token_path=True,
            config=config,
            dense_final_layout_convolution=False,
            main_rope_table_enabled=True,
            observe_prefill_index_inputs=False,
        ),
        prompt_length=2,
    )
    with pytest.raises(PlanValidationError, match="table state drifted"):
        enabled.execute(
            None,
            None,
            None,
            None,
            None,
            jnp.asarray([1, 2], jnp.int32),
            None,
            None,
            None,
        )


def test_complete_token_tpu_collective_lowering_is_exact_and_local() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_complete_token_collective_lowering,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4)) for stage in range(8)
    )
    pairs = tuple(
        (groups[stage][slot], groups[(stage + 1) % 8][slot])
        for stage in range(8)
        for slot in range(4)
    )
    group_text = "{" + ",".join(
        "{" + ",".join(map(str, group)) + "}" for group in groups
    ) + "}"
    pair_text = "{" + ",".join(
        "{" + ",".join(map(str, pair)) + "}" for pair in pairs
    ) + "}"
    hlo = f'''HloModule complete_token, replica_count=1, num_partitions=32

%add.1 (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}

%add.2 (x: s32[], y: s32[]) -> s32[] {{
  %x = s32[] parameter(0)
  %y = s32[] parameter(1)
  ROOT %sum = s32[] add(%x, %y)
}}

ENTRY %main (scores: bf16[4], ids: s32[4], token: s32[1]) -> s32[1] {{
  %scores = bf16[4] parameter(0)
  %ids = s32[4] parameter(1)
  %token = s32[1] parameter(2)
  %score_exchange = bf16[4] all-reduce(%scores), replica_groups={group_text}, use_global_device_ids=true, to_apply=%add.1
  %id_exchange = s32[4] all-reduce(%ids), replica_groups={group_text}, use_global_device_ids=true, to_apply=%add.2
  ROOT %token_return = (s32[1], s32[1], u32[], u32[]) collective-permute-start(%token), source_target_pairs={pair_text}, metadata={{op_name="jit(mapped_token)/shard_map/ppermute"}}
}}
'''
    module = parse_hlo_module(hlo)
    record = _validate_complete_token_collective_lowering(
        module,
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert record["passed"], record
    assert record["lowering"] == "local_one_hot_all_reduce"
    assert record["score_exchange"][0]["operand_shapes"] == ["bf16[4]"]
    assert record["token_id_exchange"][0]["operand_shapes"] == ["s32[4]"]
    assert record["token_return"][0]["operand_shapes"] == ["s32[1]"]
    assert record["accepted_token_return_op_names"] == [
        "jit(mapped_token)/shard_map/ppermute",
        "jit(execute)/while/body/closed_call/shard_map/ppermute",
    ]

    exact_query_hlo = hlo.replace(
        "jit(mapped_token)/shard_map/ppermute",
        "jit(mapped_token_exact_query)/shard_map/ppermute",
    )
    exact_query_record = _validate_complete_token_collective_lowering(
        parse_hlo_module(exact_query_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        dsa_query_exact_association=True,
    )
    assert exact_query_record["passed"], exact_query_record
    assert exact_query_record["accepted_token_return_op_names"] == [
        "jit(mapped_token_exact_query)/shard_map/ppermute",
        "jit(execute)/while/body/closed_call/shard_map/ppermute",
    ]
    assert not _validate_complete_token_collective_lowering(
        parse_hlo_module(exact_query_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )["passed"]
    assert not _validate_complete_token_collective_lowering(
        module,
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        dsa_query_exact_association=True,
    )["passed"]

    main_rope_hlo = hlo.replace(
        "jit(mapped_token)/shard_map/ppermute",
        "jit(mapped_token_main_rope)/shard_map/ppermute",
    )
    main_rope_record = _validate_complete_token_collective_lowering(
        parse_hlo_module(main_rope_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        main_rope_table_enabled=True,
    )
    assert main_rope_record["passed"], main_rope_record
    assert main_rope_record["accepted_token_return_op_names"] == [
        "jit(mapped_token_main_rope)/shard_map/ppermute",
        "jit(execute)/while/body/closed_call/shard_map/ppermute",
    ]
    assert not _validate_complete_token_collective_lowering(
        parse_hlo_module(main_rope_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )["passed"]

    exact_main_rope_hlo = hlo.replace(
        "jit(mapped_token)/shard_map/ppermute",
        "jit(mapped_token_exact_query_main_rope)/shard_map/ppermute",
    )
    exact_main_rope_record = _validate_complete_token_collective_lowering(
        parse_hlo_module(exact_main_rope_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        dsa_query_exact_association=True,
        main_rope_table_enabled=True,
    )
    assert exact_main_rope_record["passed"], exact_main_rope_record
    assert exact_main_rope_record["accepted_token_return_op_names"] == [
        "jit(mapped_token_exact_query_main_rope)/shard_map/ppermute",
        "jit(execute)/while/body/closed_call/shard_map/ppermute",
    ]
    assert not _validate_complete_token_collective_lowering(
        parse_hlo_module(exact_main_rope_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        dsa_query_exact_association=True,
    )["passed"]
    assert not _validate_complete_token_collective_lowering(
        parse_hlo_module(exact_query_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        dsa_query_exact_association=True,
        main_rope_table_enabled=True,
    )["passed"]

    wide_hlo = hlo.replace("bf16[4]", "bf16[4,16]").replace(
        "s32[4]", "s32[4,16]"
    )
    wide_record = _validate_complete_token_collective_lowering(
        parse_hlo_module(wide_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        token_observation_candidates=16,
    )
    assert wide_record["passed"], wide_record
    assert wide_record["token_observation_candidates"] == 16
    assert wide_record["score_exchange"][0]["operand_shapes"] == [
        "bf16[4,16]"
    ]
    assert wide_record["token_id_exchange"][0]["operand_shapes"] == [
        "s32[4,16]"
    ]

    wrong_candidate_width = _validate_complete_token_collective_lowering(
        parse_hlo_module(hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        token_observation_candidates=16,
    )
    assert not wrong_candidate_width["passed"]

    prefill_record = _validate_complete_token_collective_lowering(
        parse_hlo_module(
            hlo.replace(
                "jit(mapped_token)/shard_map/ppermute",
                "jit(execute)/while/body/closed_call/shard_map/ppermute",
            )
        ),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert prefill_record["passed"], prefill_record

    wrong_source = _validate_complete_token_collective_lowering(
        parse_hlo_module(
            hlo.replace(
                "jit(mapped_token)/shard_map/ppermute",
                "jit(unrelated)/shard_map/ppermute",
            )
        ),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not wrong_source["passed"]
    assert any(
        "source operation drifted" in item
        for item in wrong_source["violations"]
    )

    nonlocal_hlo = hlo.replace(
        f"replica_groups={group_text}",
        "replica_groups={{" + ",".join(map(str, range(32))) + "}}",
    )
    rejected = _validate_complete_token_collective_lowering(
        parse_hlo_module(nonlocal_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not rejected["passed"]
    assert any("escaped PP8 local groups" in item for item in rejected["violations"])

    wrong_return = _validate_complete_token_collective_lowering(
        parse_hlo_module(hlo.replace("token: s32[1]", "token: s32[2]").replace(
            "%token = s32[1]", "%token = s32[2]"
        )),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not wrong_return["passed"]
    assert any("exactly one s32[1]" in item for item in wrong_return["violations"])


def test_feature_decoder_hlo_contract_pins_all_raw_kernels_and_overlays() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_pallas_feature_decoder_calls,
    )

    selected = (
        "out = bf16[8,8,6144] custom-call("
        "u8[256,6144,512], u8[256,6144,512], u8[256,512,6144]), "
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_fp8_fused_selected_moe_'
        'r8_g256_h6144_i512"}'
    )
    shared_up = (
        "up = bf16[1,512] custom-call(u8[512,6144]), "
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_fp8_block_up_gate_m8_k6144_n512"}'
    )
    shared_down = (
        "down = bf16[1,6144] custom-call(u8[6144,512]), "
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_fp8_block_matmul_m8_k512_n6144"}'
    )
    hlo = "\n".join((selected, shared_up, shared_down) * 75)
    record = _validate_pallas_feature_decoder_calls(
        hlo,
        sparse_layers=75,
        feature_output_tile=128,
    )
    assert record["passed"], record
    assert record["feature_output_tile"] == 128

    wide_hlo = hlo.replace(
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512",
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
    )
    wide = _validate_pallas_feature_decoder_calls(wide_hlo, sparse_layers=75)
    assert wide["passed"], wide
    assert wide["feature_output_tile"] == 256
    fp32_hlo = wide_hlo.replace(
        "out = bf16[8,8,6144]",
        "out = f32[8,8,6144]",
    ).replace(
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256_downf32",
    )
    fp32_hlo += "\n" + "\n".join(
        "cast = bf16[8,6144] custom-call(f32[8,6144]), "
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_fp32_to_bf16_r8_h6144"}'
        for _ in range(75)
    )
    fp32 = _validate_pallas_feature_decoder_calls(
        fp32_hlo,
        sparse_layers=75,
        reconstruct_down_fp32=True,
    )
    assert fp32["passed"], fp32
    assert fp32["reconstruct_down_fp32"] is True
    stale_fp32_shape = _validate_pallas_feature_decoder_calls(
        wide_hlo.replace(
            "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
            "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256_downf32",
        ),
        sparse_layers=75,
        reconstruct_down_fp32=True,
    )
    assert not stale_fp32_shape["passed"]
    with pytest.raises(PlanValidationError, match="incompatible"):
        _validate_pallas_feature_decoder_calls(
            fp32_hlo,
            sparse_layers=75,
            fuse_route_weighting=True,
            reconstruct_down_fp32=True,
        )
    fused_hlo = wide_hlo.replace(
        "out = bf16[8,8,6144]",
        "out = bf16[8,6144]",
    ).replace(
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256_wsum",
    )
    fused = _validate_pallas_feature_decoder_calls(
        fused_hlo,
        sparse_layers=75,
        fuse_route_weighting=True,
    )
    assert fused["passed"], fused
    assert fused["fuse_route_weighting"] is True
    stale_shape = _validate_pallas_feature_decoder_calls(
        wide_hlo.replace(
            "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
            "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256_wsum",
        ),
        sparse_layers=75,
        fuse_route_weighting=True,
    )
    assert not stale_shape["passed"]
    wrong_fingerprint = _validate_pallas_feature_decoder_calls(
        wide_hlo,
        sparse_layers=75,
        feature_output_tile=128,
    )
    assert not wrong_fingerprint["passed"]

    rejected = _validate_pallas_feature_decoder_calls(
        hlo + "\noverlay = bf16[256,6144,512] parameter(0)",
        sparse_layers=75,
        feature_output_tile=128,
    )
    assert not rejected["passed"]
    assert rejected["forbidden_decoded_expert_overlays"]

    formatted = _validate_pallas_feature_decoder_calls(
        hlo + "\nformatted = f8e4m3fn[512,6144] parameter(0)",
        sparse_layers=75,
        feature_output_tile=128,
    )
    assert not formatted["passed"]
    assert formatted["forbidden_formatted_shared_overlays"]


def test_stage_linear_decoder_hlo_contract_pins_kernels_and_overlays() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_pallas_stage_linear_decoder_calls,
    )

    names = (
        "greenfield_fp8_block_matmul_m8_k6144_n2048",
        "greenfield_fp8_block_matmul_m8_k2048_n4096",
        "greenfield_fp8_block_matmul_m8_k6144_n640",
        "greenfield_fp8_block_matmul_m8_k4096_n6144",
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512",
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256",
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024",
        "greenfield_fp8_block_matmul_f32_m8_k6144_n128",
    )
    calls = [
        f'%{name} = bf16[1,128] custom-call(u8[1,128]), '
        'custom_call_target="tpu_custom_call", '
        f'metadata={{op_name="{name}"}}'
        for name in names
        for _ in range(
            21 if "block_matmul_f32" in name else 78
        )
    ]
    dense = "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144"
    calls.extend(
        f'%{dense} = bf16[1,6144] custom-call(u8[3072,6144]), '
        'custom_call_target="tpu_custom_call", '
        f'metadata={{op_name="{dense}"}}'
        for _ in range(3)
    )
    hlo = "\n".join(calls)
    record = _validate_pallas_stage_linear_decoder_calls(
        hlo, layers=78, dense_layers=3, full_indexer_layers=21
    )
    assert record["passed"], record

    reference_dsa_hlo = "\n".join(
        line
        for line in calls
        if "greenfield_fp8_block_matmul_f32_m8_k2048_n1024" not in line
    )
    reference_dsa_hlo += "\n" + "\n".join(
        f"%dsa_owner_{index} = f32[1024,2048] parameter(0)"
        for index in range(21)
    )
    reference_dsa = _validate_pallas_stage_linear_decoder_calls(
        reference_dsa_hlo,
        layers=78,
        dense_layers=3,
        full_indexer_layers=21,
        dsa_query_backend="reference",
    )
    assert reference_dsa["passed"], reference_dsa
    assert reference_dsa["expected_kernel_counts"][
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024"
    ] == 0

    fused_qkv_hlo = "\n".join(
        line
        for line in calls
        if not any(
            kernel in line
            for kernel in (
                "greenfield_fp8_block_matmul_m8_k6144_n2048",
                "greenfield_fp8_block_matmul_m8_k6144_n640",
            )
        )
    )
    fused_qkv = _validate_pallas_stage_linear_decoder_calls(
        fused_qkv_hlo,
        layers=78,
        dense_layers=3,
        full_indexer_layers=21,
        attention_projection_backend="fused_n82_convolution",
    )
    assert fused_qkv["passed"], fused_qkv
    assert fused_qkv["attention_projection_backend"] == (
        "fused_n82_convolution"
    )
    assert fused_qkv["expected_kernel_counts"][
        "greenfield_fp8_block_matmul_m8_k6144_n2048"
    ] == 0
    assert fused_qkv["expected_kernel_counts"][
        "greenfield_fp8_block_matmul_m8_k6144_n640"
    ] == 0

    rejected = _validate_pallas_stage_linear_decoder_calls(
        hlo + "\noverlay = bf16[2048,6144] parameter(0)",
        layers=78,
        dense_layers=3,
        full_indexer_layers=21,
    )
    assert not rejected["passed"]
    assert rejected["forbidden_decoded_weight_overlays"]

    formatted = _validate_pallas_stage_linear_decoder_calls(
        hlo + "\nformatted = f8e4m3fn[6144,4096] parameter(0)",
        layers=78,
        dense_layers=3,
        full_indexer_layers=21,
    )
    assert not formatted["passed"]
    assert formatted["forbidden_formatted_weight_overlays"]

    from glm_tpu.greenfield.runtime.decoder import (
        _validate_dsa_query_decoder_association,
    )

    unrelated_16k = (
        reference_dsa_hlo
        + '\n%unrelated = bf16[4096] fusion(%value), '
        + 'backend_config={"megacore_config":'
        + '{"megacore_allreduce_bytes":"16384"}}'
    )
    association = _validate_dsa_query_decoder_association(
        unrelated_16k,
        full_indexer_layers=21,
        local_parallel_size=4,
        dsa_indexer_heads=32,
        index_key_width=128,
        backend="reference",
    )
    assert association["passed"], association
    tuple_result = "(" + ",".join(["f32[1024]"] * 4) + ")"
    exact_hlo = unrelated_16k + "\n" + "\n".join(
        f"%query_tuple_{index} = {tuple_result} fusion(%value), "
        'backend_config={"megacore_config":'
        '{"megacore_allreduce_bytes":"16384"}}'
        for index in range(21)
    )
    exact = _validate_dsa_query_decoder_association(
        exact_hlo,
        full_indexer_layers=21,
        local_parallel_size=4,
        dsa_indexer_heads=32,
        index_key_width=128,
        backend="reference",
        exact_association=True,
    )
    assert exact["passed"], exact
    assert exact["tuple4_reduction_fusion_count"] == 21
    assert exact["all_16k_reduction_fusion_count"] == 22
    collapsed = _validate_dsa_query_decoder_association(
        exact_hlo.replace('"16384"', '"4096"', 2),
        full_indexer_layers=21,
        local_parallel_size=4,
        dsa_indexer_heads=32,
        index_key_width=128,
        backend="reference",
        exact_association=True,
    )
    assert not collapsed["passed"]
    assert "expected=21 observed=20" in collapsed["violations"][0]
    global_owner = _validate_dsa_query_decoder_association(
        reference_dsa_hlo + "\n%global = f32[4096,2048] parameter(0)",
        full_indexer_layers=21,
        local_parallel_size=4,
        dsa_indexer_heads=32,
        index_key_width=128,
        backend="reference",
    )
    assert not global_owner["passed"]
    assert global_owner["forbidden_global_shapes"] == ["f32[4096,2048]"]


def test_dsa_query_weight_materializer_stays_local() -> None:
    from glm_tpu.greenfield.runtime import (
        validate_dsa_query_weight_materializer_hlo,
    )

    hlo = '''HloModule query_materializer, num_partitions=32

ENTRY main (raw0: u8[1024,2048], raw1: u8[1024,2048], raw2: u8[1024,2048], scale0: f32[8,16], scale1: f32[8,16], scale2: f32[8,16]) -> (f32[1024,2048], f32[1024,2048], f32[1024,2048]) {
  %raw0 = u8[1024,2048] parameter(0)
  %raw1 = u8[1024,2048] parameter(1)
  %raw2 = u8[1024,2048] parameter(2)
  %scale0 = f32[8,16] parameter(3)
  %scale1 = f32[8,16] parameter(4)
  %scale2 = f32[8,16] parameter(5)
  %out0 = f32[1024,2048] convert(%raw0)
  %out1 = f32[1024,2048] convert(%raw1)
  %out2 = f32[1024,2048] convert(%raw2)
  ROOT %root = (f32[1024,2048], f32[1024,2048], f32[1024,2048]) tuple(%out0, %out1, %out2)
}
'''
    contract = validate_dsa_query_weight_materializer_hlo(
        hlo,
        full_indexer_slots=3,
        local_output_width=1024,
        q_lora_rank=2048,
        total_devices=32,
    )
    assert contract["passed"], contract
    metadata_calls = validate_dsa_query_weight_materializer_hlo(
        hlo.replace(
            "  ROOT %root",
            '  %bounded = s32[1] custom-call(), '
            'custom_call_target="AssumeGatherIndicesInBound"\n'
            '  %bitpacked = s32[1] custom-call(), '
            'custom_call_target="GatherScatterIndicesBitpacked"\n'
            "  ROOT %root",
        ),
        full_indexer_slots=3,
        local_output_width=1024,
        q_lora_rank=2048,
        total_devices=32,
    )
    assert metadata_calls["passed"], metadata_calls
    assert metadata_calls["custom_call_targets"] == [
        "AssumeGatherIndicesInBound",
        "GatherScatterIndicesBitpacked",
    ]
    unknown_call = validate_dsa_query_weight_materializer_hlo(
        hlo.replace(
            "  ROOT %root",
            '  %unknown = f32[1] custom-call(), '
            'custom_call_target="tpu_custom_call"\n'
            "  ROOT %root",
        ),
        full_indexer_slots=3,
        local_output_width=1024,
        q_lora_rank=2048,
        total_devices=32,
    )
    assert not unknown_call["passed"]
    assert unknown_call["forbidden_custom_call_targets"] == [
        "tpu_custom_call"
    ]
    escaped = validate_dsa_query_weight_materializer_hlo(
        hlo.replace(
            "  ROOT %root",
            "  %global = f32[4096,2048] broadcast(%out0), dimensions={0,1}\n"
            "  ROOT %root",
        ),
        full_indexer_slots=3,
        local_output_width=1024,
        q_lora_rank=2048,
        total_devices=32,
    )
    assert not escaped["passed"]
    assert escaped["forbidden_global_shapes"] == ["f32[4096,2048]"]


def test_decoder_sparse_backend_fails_closed_on_layout_mismatch() -> None:
    from dataclasses import replace

    from glm_tpu.greenfield.errors import PlanValidationError
    from glm_tpu.greenfield.model import (
        FEATURE_EXPERT_RUNTIME_LAYOUT,
        build_decoder_feature_fused_qkv_runtime_weight_layout,
        build_decoder_feature_runtime_weight_layout,
        build_decoder_runtime_weight_layout,
        build_decoder_state_layout,
        build_pipeline_schedule,
    )
    from glm_tpu.greenfield.runtime import build_decoder_step_program
    from tests.greenfield.checkpoint.test_runtime_pack import (
        _small_feature_source_plan,
    )

    source_plan = _small_feature_source_plan()
    source_plan = replace(
        source_plan,
        geometry=replace(
            source_plan.geometry,
            hidden_size=128,
            q_lora_rank=128,
            kv_lora_rank=30,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            moe_intermediate_size=512,
            fp8_block_shape=(128, 128),
        ),
    )
    source_schedule = build_pipeline_schedule(source_plan)
    source_state = build_decoder_state_layout(
        source_plan,
        source_schedule,
        context_capacity=8,
        logical_page_size=8,
        packed_kv_width=8,
    )
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        source_schedule,
    )
    feature_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    feature_schedule = build_pipeline_schedule(feature_plan)
    feature_state = build_decoder_state_layout(
        feature_plan,
        feature_schedule,
        context_capacity=8,
        logical_page_size=8,
        packed_kv_width=8,
    )
    feature_layout = build_decoder_feature_runtime_weight_layout(
        feature_plan,
        feature_schedule,
        source_layout,
    )
    fused_feature_layout = (
        build_decoder_feature_fused_qkv_runtime_weight_layout(
            feature_plan,
            feature_schedule,
            source_layout,
        )
    )
    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4)) for stage in range(8)
    )
    pairs = tuple(
        (groups[stage][slot], groups[(stage + 1) % 8][slot])
        for stage in range(8)
        for slot in range(4)
    )

    with pytest.raises(PlanValidationError, match="expected 32 devices"):
        build_decoder_step_program(
            feature_plan,
            feature_schedule,
            feature_state,
            fused_feature_layout,
            groups,
            pairs,
            sparse_moe_backend="pallas_feature",
            attention_projection_backend="fused_n82_convolution",
            devices=(object(),),
        )

    with pytest.raises(PlanValidationError, match="backend and runtime"):
        build_decoder_step_program(
            feature_plan,
            feature_schedule,
            feature_state,
            feature_layout,
            groups,
            pairs,
        )
    with pytest.raises(PlanValidationError, match="backend and runtime"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            sparse_moe_backend="pallas_feature",
        )
    with pytest.raises(PlanValidationError, match="backend is unknown"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            sparse_moe_backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="non-default feature"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            feature_output_tile=256,
        )
    with pytest.raises(PlanValidationError, match="requires pallas_feature"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            feature_reconstruct_down_fp32=True,
        )
    with pytest.raises(PlanValidationError, match="incompatible"):
        build_decoder_step_program(
            feature_plan,
            feature_schedule,
            feature_state,
            feature_layout,
            groups,
            pairs,
            sparse_moe_backend="pallas_feature",
            feature_fuse_route_weighting=True,
            feature_reconstruct_down_fp32=True,
        )
    with pytest.raises(PlanValidationError, match="linear backend is unknown"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            linear_backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="DSA query backend is unknown"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            dsa_query_backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="head/key association flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            dsa_head_key_exact_association=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="requires exact DSA query"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            dsa_head_key_exact_association=True,
        )
    with pytest.raises(PlanValidationError, match="score-precision flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            dsa_score_default_precision=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="requires exact DSA head/key"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            dsa_score_default_precision=True,
        )
    with pytest.raises(PlanValidationError, match="attention backend and runtime"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            attention_projection_backend="fused_n82_convolution",
        )
    with pytest.raises(PlanValidationError, match="main-RoPE table flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            main_rope_table_enabled=1,  # type: ignore[arg-type]
        )
    with pytest.raises(
        PlanValidationError,
        match="attention projection backend is unknown",
    ):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            attention_projection_backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="token-path flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="event-observation flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_dsa_events=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="internal-observation flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_dsa_internals=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="complete-token path"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_dsa_events=True,
        )
    with pytest.raises(PlanValidationError, match="residual-observation flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_layer_residuals=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="split residual-state flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            split_residual_state=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="discriminator flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            build_layer0_residual_discriminator=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="ingredient-observer flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            build_layer0_ingredients_observer=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="complete split-token path"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            build_layer0_residual_discriminator=True,
        )
    with pytest.raises(PlanValidationError, match="admits only"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            split_residual_state=True,
            build_layer0_residual_discriminator=True,
            main_rope_table_enabled=True,
        )
    with pytest.raises(PlanValidationError, match="requires the proven main-RoPE"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            split_residual_state=True,
            build_layer0_residual_discriminator=True,
            layer0_residual_discriminator_kind="attention_schedule",
        )
    with pytest.raises(PlanValidationError, match="requires the proven main-RoPE"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            split_residual_state=True,
            build_layer0_residual_discriminator=True,
            layer0_residual_discriminator_kind=(
                "attention_output_association"
            ),
        )
    with pytest.raises(PlanValidationError, match="proven complete split-token"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            build_layer0_ingredients_observer=True,
        )
    with pytest.raises(
        PlanValidationError, match="prefill index-input observation flag"
    ):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_prefill_index_inputs=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="complete-token path"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_prefill_index_inputs=True,
        )
    with pytest.raises(PlanValidationError, match="must be isolated"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            observe_dsa_events=True,
            observe_prefill_index_inputs=True,
        )
    with pytest.raises(PlanValidationError, match="isolated DSA observer"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            observe_layer_residuals=True,
        )
    with pytest.raises(PlanValidationError, match="isolated DSA observer"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            observe_dsa_internals=True,
        )
    with pytest.raises(PlanValidationError, match="must be isolated"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            observe_dsa_events=True,
            observe_dsa_internals=True,
            observe_layer_residuals=True,
        )


def test_complete_small_decoder_token_step_runs_all_stages_on_forced_cpu() -> None:
    program = r'''
import json
from dataclasses import replace
from hashlib import sha256
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding
from glm_tpu.greenfield.model import build_decoder_runtime_weight_layout, build_decoder_state_layout, build_pipeline_schedule
from glm_tpu.greenfield.runtime import build_decoder_step_program, build_teacher_forced_prefill_program, validate_decoder_step_hlo, validate_teacher_forced_prefill_hlo
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.checkpoint.test_runtime_pack import _small_plan

source_plan = _small_plan()
plan = replace(
    source_plan,
    geometry=replace(source_plan.geometry, vocab_size=96),
)
schedule = build_pipeline_schedule(plan)
state = build_decoder_state_layout(plan, schedule, context_capacity=8, logical_page_size=8, packed_kv_width=8)
weight_layout = build_decoder_runtime_weight_layout(plan, schedule)
groups = tuple(tuple(stage * 4 + slot for slot in range(4)) for stage in range(8))
pairs = tuple((groups[stage][slot], groups[(stage + 1) % 8][slot]) for stage in range(8) for slot in range(4))
decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True)
exact_query_decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, dsa_query_backend='reference', dsa_query_exact_association=True)
main_rope_decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, main_rope_table_enabled=True)
explicit_default = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=False, split_residual_state=False)
observer = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True, observe_layer_residuals=True)
internal_observer = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True, observe_dsa_internals=True)
split = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, split_residual_state=True)
layer0_discriminator = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, split_residual_state=True, build_layer0_residual_discriminator=True)
repair_decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_prefill_index_inputs=True, split_residual_state=True)
split_boundary_regression = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True, observe_layer_residuals=True, split_residual_state=True)
split_internal_regression = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True, observe_dsa_internals=True, split_residual_state=True)
prefill = build_teacher_forced_prefill_program(decoder, prompt_length=2)
exact_query_prefill = build_teacher_forced_prefill_program(exact_query_decoder, prompt_length=2)
main_rope_prefill = build_teacher_forced_prefill_program(main_rope_decoder, prompt_length=2)
split_prefill = build_teacher_forced_prefill_program(split, prompt_length=2)
repair_prefill = build_teacher_forced_prefill_program(repair_decoder, prompt_length=2)

weights = {}
weight_specs = decoder.input_specs[0]
for spec in weight_layout.specs:
    shape = (32, *spec.shape)
    if spec.dtype == 'F8_E4M3':
        value = np.zeros(shape, np.uint8)
    elif spec.dtype == 'F32':
        value = np.ones(shape, np.float32) if spec.value_class == 'fp8_scale' else np.zeros(shape, np.float32)
    else:
        value = np.ones(shape, dtype=ml_dtypes.bfloat16)
        if spec.name.endswith('key_norm_bias'):
            value.fill(0)
    weights[spec.name] = jax.device_put(value, NamedSharding(decoder.mesh, weight_specs[spec.name]))

regression_weights = dict(weights)
for spec in weight_layout.specs:
    if spec.dtype == 'F8_E4M3' and spec.name.startswith('dense.'):
        value = np.full((32, *spec.shape), 0x38, np.uint8)
        regression_weights[spec.name] = jax.device_put(
            value, NamedSharding(decoder.mesh, weight_specs[spec.name])
        )

residual_host = np.zeros((32, 1, 8), dtype=ml_dtypes.bfloat16)
initial_row = np.asarray([[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]], dtype=ml_dtypes.bfloat16)
for rank in groups[0]: residual_host[rank] = initial_row
split_residual_host = np.zeros((32, 2, 1, 8), dtype=ml_dtypes.bfloat16)
split_addend = np.asarray([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]], dtype=ml_dtypes.bfloat16)
for rank in groups[0]:
    split_residual_host[rank, 0] = initial_row
    split_residual_host[rank, 1] = split_addend
kv_host = np.ones((32, 1, 1, 2, 8), dtype=ml_dtypes.bfloat16)
index_host = np.ones((32, 1, 1, 2, 2), dtype=ml_dtypes.bfloat16)
metadata_host = np.full((32, 1, decoder.config.metadata_width), -1, np.int32)
metadata_host[..., decoder.config.count_index] = 0
metadata_host[..., decoder.config.producer_index] = -1
metadata_host[..., decoder.config.visited_index] = 0
metadata_host[..., decoder.config.health_index] = 1
metadata_host[..., decoder.config.active_index] = 0
for rank in groups[0]: metadata_host[rank, 0, decoder.config.active_index] = 1
token_host = np.full((32, 1), -1, np.int32)
for rank in groups[0]: token_host[rank, 0] = 5

put = lambda value, spec: jax.device_put(value, NamedSharding(decoder.mesh, spec))
inputs = (
    weights,
    put(residual_host, decoder.input_specs[1]),
    put(kv_host, decoder.input_specs[2]),
    put(index_host, decoder.input_specs[3]),
    put(metadata_host, decoder.input_specs[4]),
    put(token_host, decoder.input_specs[5]),
    put(np.asarray([0], np.int32), decoder.input_specs[6]),
    put(np.asarray([[0]], np.int32), decoder.input_specs[7]),
    put(np.asarray([1], np.int32), decoder.input_specs[8]),
)
assert main_rope_decoder.main_rope_table_host is not None
main_rope_table = put(
    main_rope_decoder.main_rope_table_host,
    main_rope_decoder.input_specs[-1],
)
main_rope_inputs = (*inputs, main_rope_table)
query_names = exact_query_decoder.dsa_query_weight_names
query_bits = tuple(weights[bits_name] for bits_name, _ in query_names)
query_scales = tuple(weights[scale_name] for _, scale_name in query_names)
query_materializer = exact_query_decoder.materialize_dsa_query_weights_fp32
assert query_materializer is not None
materialized_query_weights = tuple(
    jax.jit(query_materializer)(query_bits, query_scales)
)
query_aliases = (materialized_query_weights,) * 4
exact_query_inputs = (weights, query_aliases, *inputs[1:])
split_inputs = (
    weights,
    put(split_residual_host, split.input_specs[1]),
    *inputs[2:],
)
lowered = jax.jit(decoder.execute).lower(*inputs)
default_stablehlo = lowered.as_text()
explicit_default_stablehlo = jax.jit(explicit_default.execute).lower(*inputs).as_text()
compiled = lowered.compile()
main_rope_lowered = jax.jit(main_rope_decoder.execute).lower(
    *main_rope_inputs
)
main_rope_compiled = main_rope_lowered.compile()
exact_query_lowered = jax.jit(exact_query_decoder.execute).lower(*exact_query_inputs)
exact_query_stablehlo = exact_query_lowered.as_text()
exact_query_compiled = exact_query_lowered.compile()
split_compiled = jax.jit(split.execute).lower(*split_inputs).compile()
repair_step_compiled = jax.jit(repair_decoder.execute).lower(*split_inputs).compile()
split_boundary_regression_compiled = jax.jit(split_boundary_regression.execute).lower(*split_inputs).compile()
split_internal_regression_compiled = jax.jit(split_internal_regression.execute).lower(*split_inputs).compile()
observer_compiled = jax.jit(observer.execute).lower(*inputs).compile()
internal_observer_compiled = jax.jit(internal_observer.execute).lower(*inputs).compile()
observed = observer_compiled(*inputs)
internal_observed = internal_observer_compiled(*inputs)
first = compiled(*inputs)
main_rope_first = main_rope_compiled(*main_rope_inputs)
exact_query_first = exact_query_compiled(*exact_query_inputs)
second = compiled(weights, *first)
split_first = split_compiled(*split_inputs)
split_second = split_compiled(weights, *split_first)
regression_inputs = (regression_weights, *split_inputs[1:])
repair_step = repair_step_compiled(*regression_inputs)
split_boundary_observed = split_boundary_regression_compiled(*regression_inputs)
split_internal_observed = split_internal_regression_compiled(*regression_inputs)
prefill_inputs = (
    weights,
    *inputs[1:5],
    jnp.asarray([5, 6], dtype=jnp.int32),
    *inputs[6:9],
)
prefill_compiled = jax.jit(prefill.execute).lower(*prefill_inputs).compile()
prefilled = prefill_compiled(*prefill_inputs)
main_rope_prefill_inputs = (
    *prefill_inputs,
    None,
    None,
    main_rope_table,
)
main_rope_prefill_compiled = jax.jit(main_rope_prefill.execute).lower(
    *main_rope_prefill_inputs
).compile()
main_rope_prefilled = main_rope_prefill_compiled(
    *main_rope_prefill_inputs
)
exact_query_prefill_inputs = (*prefill_inputs, None, query_aliases)
exact_query_prefill_compiled = jax.jit(exact_query_prefill.execute).lower(*exact_query_prefill_inputs).compile()
exact_query_prefilled = exact_query_prefill_compiled(*exact_query_prefill_inputs)
split_prefill_inputs = (
    weights,
    *split_inputs[1:5],
    jnp.asarray([5, 6], dtype=jnp.int32),
    *split_inputs[6:9],
)
split_prefill_compiled = jax.jit(split_prefill.execute).lower(*split_prefill_inputs).compile()
split_prefilled = split_prefill_compiled(*split_prefill_inputs)
repair_prefill_inputs = (
    weights,
    *split_inputs[1:5],
    jnp.asarray([5, 6], dtype=jnp.int32),
    *split_inputs[6:9],
)
repair_wk_names = repair_decoder.prefill_index_weight_names
repair_wk_bits = tuple(weights[bits_name] for bits_name, _ in repair_wk_names)
repair_wk_scales = tuple(weights[scale_name] for _, scale_name in repair_wk_names)
repair_wk_decoder = repair_decoder.decode_prefill_index_weights_bf16
repair_wk_promoter = repair_decoder.promote_prefill_index_weights_fp32
assert repair_wk_decoder is not None and repair_wk_promoter is not None
repair_wk_decoded = jax.jit(repair_wk_decoder)(
    repair_wk_bits, repair_wk_scales
)
repair_wk_materialized = jax.jit(repair_wk_promoter)(repair_wk_decoded)
repair_prefill_inputs = (*repair_prefill_inputs, repair_wk_materialized)
repair_prefill_compiled = jax.jit(repair_prefill.execute).lower(*repair_prefill_inputs).compile()
repair_prefilled = repair_prefill_compiled(*repair_prefill_inputs)
residual, kv, index, metadata, next_token, next_position, next_blocks, next_lengths = map(np.asarray, jax.device_get(second))
split_residual, split_kv, split_index, split_metadata, split_next_token, split_next_position, split_next_blocks, split_next_lengths = map(np.asarray, jax.device_get(split_second))
prefill_values = list(map(np.asarray, jax.device_get(prefilled)))
main_rope_values = list(map(np.asarray, jax.device_get(main_rope_first)))
main_rope_prefill_values = list(
    map(np.asarray, jax.device_get(main_rope_prefilled))
)
exact_query_values = list(map(np.asarray, jax.device_get(exact_query_first)))
exact_query_prefill_values = list(map(np.asarray, jax.device_get(exact_query_prefilled)))
split_prefill_values = list(map(np.asarray, jax.device_get(split_prefilled)))
repair_prefill_values = list(map(np.asarray, jax.device_get(repair_prefilled)))
repair_step_history = np.asarray(jax.device_get(repair_step[8]))
split_boundary_history = np.asarray(jax.device_get(split_boundary_observed[10]))
split_internal_history = np.asarray(
    jax.device_get(split_internal_observed[10].normalized_hidden)
)
repair_stage_rows = np.stack([
    repair_step_history[group[0], 0] for group in groups
])
normalized_stage_rows = np.stack([
    split_internal_history[group[0], 0] for group in groups
])
rounded_stage_rows = np.stack([
    split_boundary_history[group[0], stage]
    for stage, group in enumerate(groups)
])
observation = np.asarray(jax.device_get(observed[8]))
token_observation = np.asarray(jax.device_get(observed[9]))
layer_residual_observation = np.asarray(jax.device_get(observed[10]))
internal_observation = {
    name: np.asarray(jax.device_get(value))
    for name, value in zip(
        internal_observed[10]._fields,
        internal_observed[10],
        strict=True,
    )
}
observation_rows = []
for stage, group in enumerate(groups):
    rows = observation[list(group), 0]
    observation_rows.append({
        'all_stage_lanes_equal': bool(np.all(rows == rows[0])),
        'row': rows[0].tolist(),
        'stage': stage,
    })
token_observation_lanes = token_observation[list(groups[-1])]
token_candidate_width = observer.config.token_observation_candidates
token_candidate_scores = token_observation_lanes[
    0, token_candidate_width:
].view(np.float32)
active = np.flatnonzero(metadata[:, 0, decoder.config.active_index] == 1)
module = parse_hlo_module(compiled.as_text())
observer_module = parse_hlo_module(observer_compiled.as_text())
internal_observer_module = parse_hlo_module(internal_observer_compiled.as_text())
hlo_contract = validate_decoder_step_hlo(compiled.as_text(), config=decoder.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True)
main_rope_hlo_contract = validate_decoder_step_hlo(main_rope_compiled.as_text(), config=main_rope_decoder.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True, main_rope_table_enabled=True)
split_hlo_contract = validate_decoder_step_hlo(split_compiled.as_text(), config=split.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True, split_residual_state=True)
observer_hlo_contract = validate_decoder_step_hlo(observer_compiled.as_text(), config=observer.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True, token_observation_candidates=observer.config.token_observation_candidates)
internal_observer_hlo_contract = validate_decoder_step_hlo(internal_observer_compiled.as_text(), config=internal_observer.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True, token_observation_candidates=internal_observer.config.token_observation_candidates)
prefill_hlo_contract = validate_teacher_forced_prefill_hlo(prefill_compiled.as_text(), program=prefill, schedule=schedule, backend_contract='cpu_reference')
main_rope_prefill_hlo_contract = validate_teacher_forced_prefill_hlo(main_rope_prefill_compiled.as_text(), program=main_rope_prefill, schedule=schedule, backend_contract='cpu_reference')
split_prefill_hlo_contract = validate_teacher_forced_prefill_hlo(split_prefill_compiled.as_text(), program=split_prefill, schedule=schedule, backend_contract='cpu_reference')
repair_prefill_hlo_contract = validate_teacher_forced_prefill_hlo(repair_prefill_compiled.as_text(), program=repair_prefill, schedule=schedule, backend_contract='cpu_reference')
counts = {}
for item in module.collectives: counts[item.opcode] = counts.get(item.opcode, 0) + 1
local_groups = tuple(tuple(group) for group in groups)
collectives_local = all(
    item.replica_groups == local_groups
    for item in module.collectives
    if item.opcode in ('all-gather', 'all-reduce')
)
kv_writes = []
index_writes = []
for stage in range(8):
    owner = groups[stage][0]
    kv_writes.append(bool(np.all(kv[owner, 0, 0, 0] == 0)))
    index_writes.append(bool(np.all(index[owner, 0, 0, 0] == 0)))

print(json.dumps({
    'active': active.tolist(),
    'collectives_local': collectives_local,
    'complete_token_path': decoder.complete_token_path,
    'layer0_residual_discriminator_built': len(layer0_discriminator.layer0_residual_discriminators) == 4,
    'default_observation_off_stablehlo_identical': default_stablehlo == explicit_default_stablehlo,
    'counts': counts,
    'exact_query': {
        'default_outputs_exact': all(np.array_equal(exact_query_values[index], np.asarray(jax.device_get(first[index]))) for index in range(8)),
        'input_spec_count': len(exact_query_decoder.input_specs),
        'materialized_slot_count': len(materialized_query_weights),
        'prefill_outputs_exact': all(np.array_equal(exact_query_prefill_values[index], prefill_values[index]) for index in range(8)),
        'program_flag': exact_query_decoder.dsa_query_exact_association,
        'stablehlo_dot_count': exact_query_stablehlo.count('stablehlo.dot_general'),
        'stablehlo_parameter_alias_count': exact_query_stablehlo.count('tensor<32x2x4xf32>'),
        'stablehlo_barrier_count': exact_query_stablehlo.count('stablehlo.optimization_barrier'),
    },
    'health': sorted(set(metadata[active, 0, decoder.config.health_index].tolist())),
    'hlo_contract': {key: hlo_contract[key] for key in ('collective_count', 'collective_counts', 'passed', 'violations')},
    'index_writes': index_writes,
    'main_rope_table': {
        'bytes_per_device': main_rope_decoder.main_rope_table_bytes_per_device,
        'default_asset_absent': decoder.main_rope_table_host is None and decoder.main_rope_table_sha256 is None and decoder.main_rope_table_bytes_per_device == 0,
        'device_table_exact': bool(np.array_equal(
            np.asarray(jax.device_get(main_rope_table)),
            np.asarray(main_rope_decoder.main_rope_table_host),
        )),
        'hlo_contract': {
            key: main_rope_hlo_contract['main_rope_table_contract'][key]
            for key in (
                'bf16_arithmetic',
                'expected_table_shape',
                'final_round_count',
                'forbidden_instructions',
                'fp32_combine_count',
                'fp32_multiply_count',
                'named_table_parameter_count',
                'passed',
                'table_parameter_count',
                'violations',
            )
        },
        'input_spec_delta': len(main_rope_decoder.input_specs) - len(decoder.input_specs),
        'position_zero_outputs_exact': all(
            np.array_equal(main_rope_values[index], np.asarray(jax.device_get(first[index])))
            for index in range(8)
        ),
        'prefill_finite': all(np.all(np.isfinite(value)) for value in main_rope_prefill_values[:2]),
        'prefill_hlo_passed': main_rope_prefill_hlo_contract['passed'],
        'prefill_table_contract_enabled': main_rope_prefill_hlo_contract['decoder_contract']['main_rope_table_enabled'],
        'program_flag': main_rope_decoder.main_rope_table_enabled,
        'shape': list(main_rope_decoder.main_rope_table_host.shape),
        'sha256_matches': sha256(
            np.ascontiguousarray(main_rope_decoder.main_rope_table_host)
            .view(np.uint16)
            .tobytes()
        ).hexdigest() == main_rope_decoder.main_rope_table_sha256,
    },
    'untargeted_owner_cache_unchanged': bool(np.all(kv[[rank for group in groups for rank in group[1:]], 0, 0, 1] == 1)),
    'kv_writes': kv_writes,
    'next_tokens': sorted(set(next_token[active, 0].tolist())),
    'next_position': next_position.tolist(),
    'next_lengths': next_lengths.tolist(),
    'observer': {
        'collective_counts': {
            opcode: sum(item.opcode == opcode for item in observer_module.collectives)
            for opcode in ('all-gather', 'all-reduce', 'collective-permute')
        },
        'host_callback_absent': all(marker not in observer_compiled.as_text().lower() for marker in ('host_callback', 'outside_compilation', 'xla_ffi_python_cpu_callback', 'xla_python_cpu_callback')),
        'hlo_contract': {key: observer_hlo_contract[key] for key in ('passed', 'token_observation_candidates', 'violations')},
        'production_outputs_exact': all(np.array_equal(np.asarray(jax.device_get(observed[index])), np.asarray(jax.device_get(first[index]))) for index in range(8)),
        'rows': observation_rows,
        'layer_residuals': {
            'dtype': layer_residual_observation.dtype.name,
            'shape': list(layer_residual_observation.shape),
            'stage_lane_replication': all(
                np.array_equal(
                    layer_residual_observation[list(group), stage],
                    np.broadcast_to(
                        layer_residual_observation[group[0], stage],
                        layer_residual_observation[list(group), stage].shape,
                    ),
                )
                for stage, group in enumerate(groups)
            ),
        },
        'token_observation': {
            'candidate_ids': token_observation_lanes[0, :token_candidate_width].tolist(),
            'candidate_scores': token_candidate_scores.tolist(),
            'candidate_width': token_candidate_width,
            'final_stage_lanes_equal': bool(np.all(token_observation_lanes == token_observation_lanes[0])),
            'inactive_lanes_sentinel': bool(np.all(token_observation[:groups[-1][0]] == -1)),
        },
    },
    'internal_observer': {
        'collective_counts': {
            opcode: sum(item.opcode == opcode for item in internal_observer_module.collectives)
            for opcode in ('all-gather', 'all-reduce', 'collective-permute')
        },
        'dtypes': {name: value.dtype.name for name, value in internal_observation.items()},
        'finite': all(np.all(np.isfinite(value)) for name, value in internal_observation.items() if name != 'producer_layer_ids'),
        'hlo_contract': {key: internal_observer_hlo_contract[key] for key in ('passed', 'token_observation_candidates', 'violations')},
        'host_callback_absent': all(marker not in internal_observer_compiled.as_text().lower() for marker in ('host_callback', 'outside_compilation', 'xla_ffi_python_cpu_callback', 'xla_python_cpu_callback')),
        'producer_layer_ids': [
            internal_observation['producer_layer_ids'][group[0], 0].item()
            for group in groups
        ],
        'production_outputs_exact': all(np.array_equal(np.asarray(jax.device_get(internal_observed[index])), np.asarray(jax.device_get(first[index]))) for index in range(8)),
        'program_flag': internal_observer.observe_dsa_internals,
        'shapes': {name: list(value.shape) for name, value in internal_observation.items()},
        'stage_lane_replication': all(
            all(
                np.array_equal(
                    value[list(group)],
                    np.broadcast_to(value[group[0]], value[list(group)].shape),
                )
                for value in internal_observation.values()
            )
            for group in groups
        ),
    },
    'block_tables_unchanged': bool(np.array_equal(next_blocks, np.asarray([[0]], np.int32))),
    'positions': sorted(set(tuple(row) for row in metadata[active, 0, :4].tolist())),
    'producer': sorted(set(metadata[active, 0, decoder.config.producer_index].tolist())),
    'prefill': {
        'next_lengths': prefill_values[7].tolist(),
        'next_position': prefill_values[5].tolist(),
        'next_tokens': sorted(set(prefill_values[4][active, 0].tolist())),
        'positions': sorted(set(tuple(row) for row in prefill_values[3][active, 0, :4].tolist())),
        'valid_counts': sorted(set(prefill_values[3][active, 0, decoder.config.count_index].tolist())),
    },
    'prefill_hlo_contract': {
        'collective_count': prefill_hlo_contract['decoder_contract']['collective_count'],
        'collective_counts': prefill_hlo_contract['decoder_contract']['collective_counts'],
        'dead_prompt_rows': prefill_hlo_contract['dead_prompt_rows'],
        'host_transfer_markers': prefill_hlo_contract['host_transfer_markers'],
        'host_transfer_opcodes': prefill_hlo_contract['host_transfer_opcodes'],
        'loop_count': prefill_hlo_contract['loop_count'],
        'loop_contract': {
            key: prefill_hlo_contract['loop_contract'][key]
            for key in (
                'expected_fused_qkv_internal_loop_count',
                'expected_total_loop_count',
                'fused_qkv_internal_loop_count',
                'loop_count',
                'outer_loop_count',
                'passed',
                'unclassified_loops',
                'violations',
            )
        },
        'outer_loop_count': prefill_hlo_contract['outer_loop_count'],
        'passed': prefill_hlo_contract['passed'],
        'prompt_shape_present': prefill_hlo_contract['prompt_shape_parameter_count'] > 0,
        'violations': prefill_hlo_contract['violations'],
    },
    'prefill_index_repair': {
        'backend': repair_prefill_hlo_contract['index_repair_backend'],
        'decoder_contract_passed': repair_prefill_hlo_contract['decoder_contract']['passed'],
        'exact_projection_operand_count': repair_prefill_hlo_contract['index_repair_contract']['exact_projection_operand_count'],
        'expected_call_count': repair_prefill_hlo_contract['index_repair_contract']['expected_call_count'],
        'forbidden_markers': repair_prefill_hlo_contract['index_repair_contract']['forbidden_markers'],
        'full_pod_history_shapes': repair_prefill_hlo_contract['index_repair_contract']['full_pod_history_shapes'],
        'history_estimated_bytes_per_device': repair_prefill_hlo_contract['index_repair_contract']['history_estimated_bytes_per_device'],
        'history_shapes_compact': bool(
            repair_prefill_hlo_contract['index_repair_contract']['history_shapes']
            and len(repair_prefill_hlo_contract['index_repair_contract']['history_shapes']) <= 8
            and all(
                shape['dtype'] == 'bf16' and 32 not in shape['dimensions']
                for shape in repair_prefill_hlo_contract['index_repair_contract']['history_shapes']
            )
        ),
        'loop_count': repair_prefill_hlo_contract['loop_contract']['prefill_index_repair_loop_count'],
        'passed': repair_prefill_hlo_contract['passed'],
        'production_outputs_exact': all(np.array_equal(repair_prefill_values[index], split_prefill_values[index]) for index in range(8)),
        'projection_count': repair_prefill_hlo_contract['index_repair_contract']['projection_count'],
        'recorded_normalized_input_exact': bool(np.array_equal(repair_stage_rows, normalized_stage_rows)),
        'repair_collectives': repair_prefill_hlo_contract['index_repair_contract']['repair_collectives'],
        'rounded_boundary_is_distinct': bool(np.any(normalized_stage_rows != rounded_stage_rows)),
        'split_residual_state': repair_decoder.split_residual_state,
        'violations': repair_prefill_hlo_contract['violations'],
    },
    'residual_exact': bool(np.array_equal(residual[active], np.ones((4, 1, 8), dtype=ml_dtypes.bfloat16))),
    'split': {
        'active': np.flatnonzero(split_metadata[:, 0, split.config.active_index] == 1).tolist(),
        'collective_counts': split_hlo_contract['collective_counts'],
        'finite': bool(np.all(np.isfinite(split_residual))),
        'hlo_contract': {key: split_hlo_contract[key] for key in ('passed', 'residual_transport_count', 'residual_transport_dimensions', 'residual_transport_dtype', 'split_residual_state', 'violations')},
        'next_lengths': split_next_lengths.tolist(),
        'next_position': split_next_position.tolist(),
        'next_tokens': sorted(set(split_next_token[active, 0].tolist())),
        'prefill': {
            'hlo_passed': split_prefill_hlo_contract['passed'],
            'next_lengths': split_prefill_values[7].tolist(),
            'next_position': split_prefill_values[5].tolist(),
            'next_tokens': sorted(set(split_prefill_values[4][active, 0].tolist())),
            'residual_shape': list(split_prefill_values[0].shape),
            'violations': split_prefill_hlo_contract['violations'],
        },
        'program_flag': split.split_residual_state,
        'residual_shape': list(split_residual.shape),
        'visited': sorted(set(split_metadata[active, 0, split.config.visited_index].tolist())),
    },
    'sparse_moe_backend': decoder.sparse_moe_backend,
    'valid_counts': sorted(set(metadata[active, 0, decoder.config.count_index].tolist())),
    'visited': sorted(set(metadata[active, 0, decoder.config.visited_index].tolist())),
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=420,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["active"] == [0, 1, 2, 3]
    assert result["residual_exact"]
    assert result["complete_token_path"]
    assert result["layer0_residual_discriminator_built"]
    assert result["default_observation_off_stablehlo_identical"]
    assert result["exact_query"]["program_flag"]
    assert result["exact_query"]["input_spec_count"] == 10
    assert result["exact_query"]["materialized_slot_count"] == 1
    assert result["exact_query"]["default_outputs_exact"]
    assert result["exact_query"]["prefill_outputs_exact"]
    assert result["exact_query"]["stablehlo_dot_count"] >= 4
    assert result["exact_query"]["stablehlo_parameter_alias_count"] >= 4
    assert result["exact_query"]["stablehlo_barrier_count"] >= 2
    main_rope = result["main_rope_table"]
    assert main_rope["program_flag"]
    assert main_rope["default_asset_absent"]
    assert main_rope["device_table_exact"]
    assert main_rope["input_spec_delta"] == 1
    assert main_rope["shape"] == [8, 2]
    assert main_rope["bytes_per_device"] == 32
    assert main_rope["sha256_matches"]
    assert main_rope["position_zero_outputs_exact"]
    assert main_rope["prefill_finite"]
    assert main_rope["prefill_hlo_passed"]
    assert main_rope["prefill_table_contract_enabled"]
    main_rope_hlo = main_rope["hlo_contract"]
    assert main_rope_hlo["passed"], main_rope_hlo
    assert main_rope_hlo["violations"] == []
    assert main_rope_hlo["expected_table_shape"] == [8, 2]
    assert main_rope_hlo["table_parameter_count"] == 1
    assert main_rope_hlo["named_table_parameter_count"] == 1
    assert main_rope_hlo["fp32_multiply_count"] >= 8 * 8
    assert main_rope_hlo["fp32_combine_count"] >= 8 * 4
    assert main_rope_hlo["final_round_count"] >= 8 * 2
    assert main_rope_hlo["forbidden_instructions"] == []
    assert main_rope_hlo["bf16_arithmetic"] == []
    assert result["split"] == {
        "active": [0, 1, 2, 3],
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "finite": True,
        "hlo_contract": {
            "passed": True,
            "residual_transport_count": 8,
            "residual_transport_dimensions": [2, 1, 8],
            "residual_transport_dtype": "f32",
            "split_residual_state": True,
            "violations": [],
        },
        "next_lengths": [3],
        "next_position": [2],
        "next_tokens": [0],
        "prefill": {
            "hlo_passed": True,
            "next_lengths": [3],
            "next_position": [2],
            "next_tokens": [0],
            "residual_shape": [32, 2, 1, 8],
            "violations": [],
        },
        "program_flag": True,
        "residual_shape": [32, 2, 1, 8],
        "visited": [255],
    }
    assert result["next_tokens"] == [0]
    assert result["next_position"] == [2]
    assert result["next_lengths"] == [3]
    assert result["prefill_index_repair"] == {
        "backend": "physical_m64_chunk",
        "decoder_contract_passed": True,
        "exact_projection_operand_count": 8,
        "expected_call_count": 8,
        "forbidden_markers": [],
        "full_pod_history_shapes": [],
        "history_estimated_bytes_per_device": 32,
        "history_shapes_compact": True,
        "loop_count": 8,
        "passed": True,
        "production_outputs_exact": True,
        "projection_count": 8,
        "recorded_normalized_input_exact": True,
        "repair_collectives": [],
        "rounded_boundary_is_distinct": True,
        "split_residual_state": True,
        "violations": [],
    }
    assert result["observer"] == {
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "host_callback_absent": True,
        "hlo_contract": {
            "passed": True,
            "token_observation_candidates": 16,
            "violations": [],
        },
        "production_outputs_exact": True,
        "layer_residuals": {
            "dtype": "bfloat16",
            "shape": [32, 9, 8],
            "stage_lane_replication": True,
        },
        "rows": [
            {
                "all_stage_lanes_equal": True,
                "row": [
                    0,
                    -1,
                    -1,
                    -1,
                    0,
                    -8388608,
                    -8388608,
                    -8388608,
                    1,
                    stage,
                ],
                "stage": stage,
            }
            for stage in range(8)
        ],
        "token_observation": {
            "candidate_ids": list(range(16)),
            "candidate_scores": [8.0] * 16,
            "candidate_width": 16,
            "final_stage_lanes_equal": True,
            "inactive_lanes_sentinel": True,
        },
    }
    assert result["internal_observer"] == {
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "dtypes": {
            "current_key": "float32",
            "head_weights": "float32",
            "normalized_hidden": "bfloat16",
            "producer_layer_ids": "int32",
            "q_a_state": "bfloat16",
            "query": "float32",
        },
        "finite": True,
        "hlo_contract": {
            "passed": True,
            "token_observation_candidates": 16,
            "violations": [],
        },
        "host_callback_absent": True,
        "producer_layer_ids": list(range(8)),
        "production_outputs_exact": True,
        "program_flag": True,
        "shapes": {
            "current_key": [32, 1, 2],
            "head_weights": [32, 1, 4],
            "normalized_hidden": [32, 1, 8],
            "producer_layer_ids": [32, 1],
            "q_a_state": [32, 1, 4],
            "query": [32, 1, 4, 2],
        },
        "stage_lane_replication": True,
    }
    assert result["prefill"] == {
        "next_lengths": [3],
        "next_position": [2],
        "next_tokens": [0],
        "positions": [[0, 1, -1, -1]],
        "valid_counts": [2],
    }
    assert result["prefill_hlo_contract"] == {
        "collective_count": 92,
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "dead_prompt_rows": [],
        "host_transfer_markers": [],
        "host_transfer_opcodes": [],
        "loop_count": 1,
        "loop_contract": {
            "expected_fused_qkv_internal_loop_count": 0,
            "expected_total_loop_count": 1,
            "fused_qkv_internal_loop_count": 0,
            "loop_count": 1,
            "outer_loop_count": 1,
            "passed": True,
            "unclassified_loops": [],
            "violations": [],
        },
        "outer_loop_count": 1,
        "passed": True,
        "prompt_shape_present": True,
        "violations": [],
    }
    assert result["block_tables_unchanged"]
    assert result["sparse_moe_backend"] == "reference"
    assert result["positions"] == [[0, 1, -1, -1]]
    assert result["valid_counts"] == [2]
    assert result["producer"] == [7]
    assert result["visited"] == [255]
    assert result["health"] == [1]
    assert all(result["kv_writes"])
    assert all(result["index_writes"])
    assert result["untargeted_owner_cache_unchanged"]
    assert result["collectives_local"]
    assert result["hlo_contract"] == {
        "collective_count": 92,
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "passed": True,
        "violations": [],
    }
    assert result["counts"] == {
        "all-gather": 58,
        "all-reduce": 17,
        "collective-permute": 17,
    }
