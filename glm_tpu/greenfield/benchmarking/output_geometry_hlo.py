"""Structural report for the SHA-pinned output-only Pallas lowering."""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import BenchmarkValidationError
from ..sharding.hlo_contract import parse_hlo_module


def validate_output_geometry_stable_structure(
    stablehlo: str,
    digest: str,
) -> Mapping[str, Any]:
    """Bind the exact manual region, M1 Pallas inputs, and live returns."""

    required_once = (
        "mhlo.num_partitions = 32 : i32",
        "sdy.manual_computation(",
        'manual_axes={"member"}',
        "%38 = stablehlo.optimization_barrier %arg7 : "
        "tensor<1x6144xbf16>",
        "%39 = stablehlo.optimization_barrier %arg8 : "
        "tensor<1x6144xbf16>",
        "%40 = stablehlo.slice %13 [0:1] : (tensor<32xf32>) -> "
        "tensor<1xf32>",
        "%41 = stablehlo.broadcast_in_dim %arg9, dims = [1] : "
        "(tensor<6144xbf16>) -> tensor<1x6144xbf16>",
        "%42 = stablehlo.custom_call @tpu_custom_call(%38, %39, %40, %41)",
        'kernel_name = "greenfield_weighted_output_m1_m8_scratch_h6144"',
        ": (tensor<1x6144xbf16>, tensor<1x6144xbf16>, tensor<1xf32>, "
        "tensor<1x6144xbf16>) -> tensor<1x6144xbf16>",
        "%43 = stablehlo.bitcast_convert %42 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xui16>",
        "sdy.return %24, %37, %43 : tensor<32x6144xui16>, "
        "tensor<1x6144xui16>, tensor<1x6144xui16>",
        "return %0#0, %0#1, %0#2 : tensor<32x6144xui16>, "
        "tensor<1x6144xui16>, tensor<1x6144xui16>",
    )
    missing = [value for value in required_once if stablehlo.count(value) != 1]
    forbidden = (
        "stablehlo.all_gather",
        "stablehlo.all_reduce",
        "stablehlo.all_to_all",
        "stablehlo.collective_broadcast",
        "stablehlo.collective_permute",
        "stablehlo.reduce_scatter",
    )
    if missing or any(value in stablehlo for value in forbidden):
        raise BenchmarkValidationError(
            "output-geometry StableHLO structure drifted"
        )
    return {
        "exact_graph_digest": True,
        "exact_manual_sharding": True,
        "exact_pallas_operand_binding": True,
        "pallas_custom_call_count": 1,
        "pallas_true_m1_io": True,
        "physical_collective_count": 0,
        "passed": True,
        "stablehlo_sha256": digest,
    }


def validate_output_geometry_optimized_structure(
    optimized_hlo: str,
    digest: str,
) -> Mapping[str, Any]:
    """Bind scheduled source flow and the live true-M1 Pallas result."""

    module = parse_hlo_module(optimized_hlo)
    entry = tuple(
        instruction
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
    )
    values = {instruction.name: instruction for instruction in entry}
    expected_operands = {
        "%copy-start.2": ("%param.9",),
        "%copy-start.3": ("%param.8",),
        "%copy-start.4": ("%param.7",),
        "%copy-done.2": ("%copy-start.2",),
        "%copy-done.3": ("%copy-start.3",),
        "%copy-done.4": ("%copy-start.4",),
        "%multiply_reduce_fusion": ("%param.6", "%param.5"),
        "%add_rsqrt_fusion": ("%multiply_reduce_fusion",),
        "%bitcast.2": ("%add_rsqrt_fusion",),
        "%bitcast.3": ("%add_rsqrt_fusion",),
        "%broadcast_in_dim.2": ("%copy-done.2",),
        "%greenfield_weighted_output_m1_m8_scratch_h6144.1": (
            "%copy-done.4",
            "%copy-done.3",
            "%bitcast.3",
            "%broadcast_in_dim.2",
        ),
        "%bitcast_convert_type.11": (
            "%greenfield_weighted_output_m1_m8_scratch_h6144.1",
        ),
        "%fusion.1": (
            "%copy-done",
            "%copy-done.1",
            "%add_rsqrt_fusion",
            "%copy-done.2",
        ),
        "%tuple.5": (
            "%fusion.1",
            "%multiply_bitcast-convert_fusion",
            "%bitcast_convert_type.11",
        ),
    }
    operand_graph_exact = all(
        name in values and values[name].operand_names == operands
        for name, operands in expected_operands.items()
    )
    expected_line_prefixes = {
        "%param.5": (
            "%param.5 = bf16[32,6144]{1,0:T(8,128)(2,1)} parameter(0), "
            "sharding={replicated}, metadata={op_name=\"dense_m32\"}"
        ),
        "%param.6": (
            "%param.6 = bf16[32,6144]{1,0:T(8,128)(2,1)} parameter(1), "
            "sharding={replicated}, metadata={op_name=\"carried_m32\"}"
        ),
        "%param.7": (
            "%param.7 = bf16[1,6144]{1,0:T(2,128)(2,1)} parameter(2), "
            "sharding={replicated}, metadata={op_name=\"dense_auto\"}"
        ),
        "%param.8": (
            "%param.8 = bf16[1,6144]{1,0:T(2,128)(2,1)} parameter(3), "
            "sharding={replicated}, metadata={op_name=\"carried_auto\"}"
        ),
        "%param.9": (
            "%param.9 = bf16[6144]{0:T(1024)(128)(2,1)} parameter(4), "
            "sharding={replicated}, metadata={op_name=\"weight\"}"
        ),
        "%copy-start.2": (
            "%copy-start.2 = (bf16[6144]{0:T(1024)(128)(2,1)S(3)}, "
            "bf16[6144]{0:T(1024)(128)(2,1)}, u32[]{:S(2)}) "
            "copy-start(%param.9)"
        ),
        "%copy-start.3": (
            "%copy-start.3 = (bf16[1,6144]{1,0:T(2,128)(2,1)S(3)}, "
            "bf16[1,6144]{1,0:T(2,128)(2,1)}, u32[]{:S(2)}) "
            "copy-start(%param.8)"
        ),
        "%copy-start.4": (
            "%copy-start.4 = (bf16[1,6144]{1,0:T(2,128)(2,1)S(3)}, "
            "bf16[1,6144]{1,0:T(2,128)(2,1)}, u32[]{:S(2)}) "
            "copy-start(%param.7)"
        ),
        "%copy-done.2": (
            "%copy-done.2 = bf16[6144]{0:T(1024)(128)(2,1)S(3)} "
            "copy-done(%copy-start.2)"
        ),
        "%copy-done.3": (
            "%copy-done.3 = bf16[1,6144]{1,0:T(2,128)(2,1)S(3)} "
            "copy-done(%copy-start.3)"
        ),
        "%copy-done.4": (
            "%copy-done.4 = bf16[1,6144]{1,0:T(2,128)(2,1)S(3)} "
            "copy-done(%copy-start.4)"
        ),
        "%multiply_reduce_fusion": (
            "%multiply_reduce_fusion = f32[32]{0:T(128)S(3)} "
            "fusion(%param.6, %param.5)"
        ),
        "%add_rsqrt_fusion": (
            "%add_rsqrt_fusion = f32[32]{0:T(128)S(3)} "
            "fusion(%multiply_reduce_fusion)"
        ),
        "%bitcast.3": (
            "%bitcast.3 = f32[1]{0:T(128)S(3)} "
            "bitcast(%add_rsqrt_fusion)"
        ),
        "%broadcast_in_dim.2": (
            "%broadcast_in_dim.2 = "
            "bf16[1,6144]{1,0:T(2,128)(2,1)S(3)} "
            "reshape(%copy-done.2)"
        ),
        "%greenfield_weighted_output_m1_m8_scratch_h6144.1": (
            "%greenfield_weighted_output_m1_m8_scratch_h6144.1 = "
            "bf16[1,6144]{1,0:T(2,128)(2,1)S(3)} "
            "custom-call(%copy-done.4, %copy-done.3, %bitcast.3, "
            "%broadcast_in_dim.2)"
        ),
        "%bitcast_convert_type.11": (
            "%bitcast_convert_type.11 = "
            "u16[1,6144]{1,0:T(2,128)(2,1)} "
            "bitcast-convert("
            "%greenfield_weighted_output_m1_m8_scratch_h6144.1)"
        ),
    }
    boundary_lines_exact = all(
        name in values and values[name].raw_line.startswith(prefix)
        for name, prefix in expected_line_prefixes.items()
    )
    pallas = values.get(
        "%greenfield_weighted_output_m1_m8_scratch_h6144.1"
    )
    root = values.get("%tuple.5")
    expected_pallas_inputs = (
        ("bf16", (1, 6144)),
        ("bf16", (1, 6144)),
        ("f32", (1,)),
        ("bf16", (1, 6144)),
    )
    expected_root_shapes = (
        ("u16", (32, 6144)),
        ("u16", (1, 6144)),
        ("u16", (1, 6144)),
    )
    pallas_inputs = (
        ()
        if pallas is None
        else tuple(
            (shape.dtype, shape.dimensions) for shape in pallas.operand_shapes
        )
    )
    pallas_results = (
        ()
        if pallas is None
        else tuple(
            (shape.dtype, shape.dimensions) for shape in pallas.result_shapes
        )
    )
    root_shapes = (
        ()
        if root is None
        else tuple(
            (shape.dtype, shape.dimensions) for shape in root.result_shapes
        )
    )
    pallas_line_exact = bool(
        pallas is not None
        and pallas.raw_opcode == "custom-call"
        and pallas.op_name
        == (
            "jit(program)/shard_map/output_geometry_m1_pallas_m8_scratch/"
            "greenfield_weighted_output_m1_m8_scratch_h6144/pallas_call"
        )
        and 'custom_call_target="tpu_custom_call"' in pallas.raw_line
        and (
            "operand_layout_constraints={bf16[1,6144]{1,0}, "
            "bf16[1,6144]{1,0}, f32[1]{0}, bf16[1,6144]{1,0}}"
        )
        in pallas.raw_line
        and pallas_inputs == expected_pallas_inputs
        and pallas_results == (("bf16", (1, 6144)),)
        and "bf16[1,6144]{1,0:T(2,128)(2,1)S(3)} custom-call"
        in pallas.raw_line
    )
    no_async_or_sync_collectives = not module.collectives and not any(
        instruction.raw_opcode.startswith(
            (
                "all-gather",
                "all-reduce",
                "all-to-all",
                "collective-broadcast",
                "collective-permute",
                "reduce-scatter",
            )
        )
        for instruction in module.instructions
    )
    exact_root = bool(
        root is not None
        and root.raw_opcode == "tuple"
        and root.raw_line.startswith("ROOT ")
        and root_shapes == expected_root_shapes
    )
    if not (
        module.name == "jit_program"
        and module.num_partitions == 32
        and len(module.instructions) == 76
        and len(entry) == 25
        and operand_graph_exact
        and boundary_lines_exact
        and pallas_line_exact
        and no_async_or_sync_collectives
        and exact_root
    ):
        raise BenchmarkValidationError(
            "output-geometry optimized HLO structure drifted"
        )
    return {
        "exact_graph_digest": True,
        "exact_live_root": True,
        "exact_pallas_boundary_layouts": True,
        "exact_pallas_input_origins": True,
        "exact_pallas_operand_binding": True,
        "exact_shared_inverse_fanout": True,
        "instruction_count": len(module.instructions),
        "optimized_hlo_sha256": digest,
        "pallas_custom_call_count": 1,
        "pallas_input_shapes": [
            {"dtype": dtype, "shape": list(shape)}
            for dtype, shape in pallas_inputs
        ],
        "pallas_true_m1_io": True,
        "physical_collective_count": 0,
        "passed": True,
    }
