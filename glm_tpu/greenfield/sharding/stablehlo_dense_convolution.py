"""Exact pre-fusion contract for the layer-0 dense-convolution probe."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

from .stablehlo_strategy_nd import (
    _MatchError,
    _StableGraph,
    _StableNode,
    _expect_concat,
    _expect_node,
    _expect_reshape,
    _expect_slice,
    _match_reduction_tree,
    _only,
    _parse_graphs,
)


_CONVOLUTION_ATTRIBUTES = (
    "dim_numbers=[b,f]x[i,o]->[b,f]",
    "window={stride=[],pad=[],lhs_dilate=[],rhs_dilate=[],reverse=[]}",
    "batch_group_count=1:i64,feature_group_count=1:i64",
    "precision_config=[#stablehlo<precisionDEFAULT>,"
    "#stablehlo<precisionDEFAULT>]",
)


def _expect_unary(
    graph: _StableGraph,
    source: str,
    *,
    opcode: str,
    result_type: str,
) -> _StableNode:
    return _only(
        (
            node
            for node in graph.matching_users(
                source, opcode=opcode, result_type=result_type
            )
            if node.operands == (source,)
        ),
        f"{opcode} {source} -> {result_type}",
    )


def _expect_binary(
    graph: _StableGraph,
    left: str,
    right: str,
    *,
    opcode: str,
    result_type: str,
    commutative: bool = False,
) -> _StableNode:
    expected = Counter((left, right)) if commutative else (left, right)
    return _only(
        (
            node
            for node in graph.nodes.values()
            if node.opcode == opcode
            and node.result_type == result_type
            and (
                Counter(node.operands) == expected
                if commutative
                else node.operands == expected
            )
        ),
        f"{opcode} {left}, {right} -> {result_type}",
    )


def _expect_transpose(
    graph: _StableGraph,
    source: str,
    *,
    dimensions: tuple[int, ...],
    result_type: str,
) -> _StableNode:
    return _only(
        (
            node
            for node in graph.matching_users(
                source, opcode="transpose", result_type=result_type
            )
            if node.operands == (source,) and node.dimensions == dimensions
        ),
        f"transpose {source} {dimensions} -> {result_type}",
    )


def _expect_broadcast(
    graph: _StableGraph,
    source: str,
    *,
    dimensions: tuple[int, ...],
    result_type: str,
) -> _StableNode:
    return _only(
        (
            node
            for node in graph.matching_users(
                source, opcode="broadcast_in_dim", result_type=result_type
            )
            if node.operands == (source,) and node.dimensions == dimensions
        ),
        f"broadcast {source} {dimensions} -> {result_type}",
    )


def _expect_constant_broadcast(
    graph: _StableGraph,
    *,
    literal: str,
    scalar_type: str,
    result_type: str,
    dimensions: tuple[int, ...],
) -> _StableNode:
    return _only(
        (
            broadcast
            for constant in graph.nodes.values()
            if constant.opcode == "constant"
            and constant.result_type == scalar_type
            and constant.constant_literal == literal
            for broadcast in graph.matching_users(
                constant.name,
                opcode="broadcast_in_dim",
                result_type=result_type,
            )
            if broadcast.operands == (constant.name,)
            and broadcast.dimensions == dimensions
        ),
        f"constant {literal} broadcast -> {result_type}",
    )


def _require_constant_broadcast(
    graph: _StableGraph,
    node: _StableNode,
    *,
    literal: str,
    scalar_type: str,
    result_type: str,
    dimensions: tuple[int, ...],
) -> None:
    if (
        node.opcode != "broadcast_in_dim"
        or node.result_type != result_type
        or node.dimensions != dimensions
        or len(node.operands) != 1
    ):
        raise _MatchError(f"{node.name}: constant broadcast geometry drifted")
    constant = _expect_node(
        graph,
        node.operands[0],
        opcode="constant",
        result_type=scalar_type,
    )
    if constant.constant_literal != literal:
        raise _MatchError(f"{constant.name}: constant literal drifted")


def _expect_convolution(
    node: _StableNode,
    *,
    operands: tuple[str, str],
    tensor_types: tuple[str, str, str],
) -> None:
    normalized = re.sub(r"\s+", "", node.raw_line)
    if (
        node.opcode != "convolution"
        or node.operands != operands
        or node.tensor_types[-3:] != tensor_types
        or node.result_type != tensor_types[-1]
        or any(marker not in normalized for marker in _CONVOLUTION_ATTRIBUTES)
    ):
        raise _MatchError(f"{node.name}: dense convolution contract drifted")


def _match_fp8_decode(
    graph: _StableGraph,
    output: str,
    *,
    bit_type: str,
    fp8_type: str,
    scale_seed_type: str,
    scale_middle_type: str,
    scale_wide_type: str,
    weight_type: str,
    first_broadcast_dimensions: tuple[int, ...],
) -> tuple[str, str]:
    rounded = _expect_node(
        graph, output, opcode="convert", result_type=weight_type
    )
    if len(rounded.operands) != 1:
        raise _MatchError(f"{rounded.name}: BF16 weight conversion drifted")
    scaled = _expect_node(
        graph, rounded.operands[0], opcode="multiply", result_type=scale_wide_type
    )
    if len(scaled.operands) != 2:
        raise _MatchError(f"{scaled.name}: FP8 scale multiplication drifted")
    fp32 = _expect_node(
        graph, scaled.operands[0], opcode="convert", result_type=scale_wide_type
    )
    fp8 = _expect_node(
        graph, fp32.operands[0], opcode="bitcast_convert", result_type=fp8_type
    )
    if fp8.tensor_types[-2:] != (bit_type, fp8_type):
        raise _MatchError(f"{fp8.name}: FP8 bitcast geometry drifted")
    wide_scale = _expect_node(
        graph, scaled.operands[1], opcode="reshape", result_type=scale_wide_type
    )
    second_broadcast = _expect_node(
        graph,
        wide_scale.operands[0],
        opcode="broadcast_in_dim",
        result_type=scale_middle_type,
    )
    if second_broadcast.dimensions != (0, 1):
        raise _MatchError(f"{second_broadcast.name}: inner scale expansion drifted")
    middle_scale = _expect_node(
        graph,
        second_broadcast.operands[0],
        opcode="reshape",
        result_type=scale_seed_type,
    )
    first_broadcast = _expect_node(
        graph,
        middle_scale.operands[0],
        opcode="broadcast_in_dim",
        result_type=middle_scale.tensor_types[-2],
    )
    if first_broadcast.dimensions != first_broadcast_dimensions:
        raise _MatchError(f"{first_broadcast.name}: outer scale expansion drifted")
    return fp8.operands[0], first_broadcast.operands[0]


def _match_one_shard(
    graph: _StableGraph,
    gate_up: _StableNode,
    *,
    compile_rows: int,
) -> tuple[int, tuple[str, ...], str]:
    row_type = f"tensor<{compile_rows}x"
    if len(gate_up.operands) != 2:
        raise _MatchError(f"{gate_up.name}: gate/up operand arity drifted")
    normalized = gate_up.operands[0]
    gate_bits, gate_scales = _match_fp8_decode(
        graph,
        gate_up.operands[1],
        bit_type="tensor<6144x768xui8>",
        fp8_type="tensor<6144x768xf8E4M3FN>",
        scale_seed_type="tensor<6144x6xf32>",
        scale_middle_type="tensor<6144x6x128xf32>",
        scale_wide_type="tensor<6144x768xf32>",
        weight_type="tensor<6144x768xbf16>",
        first_broadcast_dimensions=(0, 2),
    )
    _expect_convolution(
        gate_up,
        operands=(normalized, gate_up.operands[1]),
        tensor_types=(
            f"{row_type}6144xbf16>",
            "tensor<6144x768xbf16>",
            f"{row_type}768xf32>",
        ),
    )
    merged_bits = _expect_node(
        graph,
        gate_bits,
        opcode="concatenate",
        result_type="tensor<6144x768xui8>",
    )
    if merged_bits.concatenate_dimension != 1 or len(merged_bits.operands) != 2:
        raise _MatchError(f"{merged_bits.name}: gate/up bit ordering drifted")
    bit_slices = []
    for operand in merged_bits.operands:
        transpose = _expect_node(
            graph,
            operand,
            opcode="transpose",
            result_type="tensor<6144x384xui8>",
        )
        if transpose.dimensions != (1, 0) or len(transpose.operands) != 1:
            raise _MatchError(f"{transpose.name}: gate/up bit transpose drifted")
        bit_slices.append(
            _expect_node(
                graph,
                transpose.operands[0],
                opcode="slice",
                result_type="tensor<384x6144xui8>",
            )
        )
    first_ranges = bit_slices[0].slice_ranges
    if (
        first_ranges is None
        or first_ranges[1] != (0, 6144)
        or first_ranges[0][1] != first_ranges[0][0] + 384
        or first_ranges[0][0] % 384
    ):
        raise _MatchError(f"{bit_slices[0].name}: gate shard slice drifted")
    shard = first_ranges[0][0] // 384
    expected_bit_ranges = ((shard * 384, (shard + 1) * 384), (0, 6144))
    if any(
        node.slice_ranges != expected_bit_ranges or len(node.operands) != 1
        for node in bit_slices
    ):
        raise _MatchError(f"{gate_up.name}: gate/up shard slices cross-wired")

    merged_scales = _expect_node(
        graph,
        gate_scales,
        opcode="concatenate",
        result_type="tensor<48x6xf32>",
    )
    if merged_scales.concatenate_dimension != 1 or len(merged_scales.operands) != 2:
        raise _MatchError(f"{merged_scales.name}: gate/up scale ordering drifted")
    scale_slices = []
    for operand in merged_scales.operands:
        transpose = _expect_node(
            graph, operand, opcode="transpose", result_type="tensor<48x3xf32>"
        )
        if transpose.dimensions != (1, 0) or len(transpose.operands) != 1:
            raise _MatchError(f"{transpose.name}: gate/up scale transpose drifted")
        scale_slices.append(
            _expect_node(
                graph,
                transpose.operands[0],
                opcode="slice",
                result_type="tensor<3x48xf32>",
            )
        )
    expected_scale_ranges = ((shard * 3, (shard + 1) * 3), (0, 48))
    if any(
        node.slice_ranges != expected_scale_ranges or len(node.operands) != 1
        for node in scale_slices
    ):
        raise _MatchError(f"{gate_up.name}: gate/up scale slices cross-wired")

    gate_up_bf16 = _expect_unary(
        graph,
        gate_up.name,
        opcode="convert",
        result_type=f"{row_type}768xbf16>",
    )
    gate = _expect_slice(
        graph,
        gate_up_bf16.name,
        ((0, compile_rows), (0, 384)),
        f"{row_type}384xbf16>",
    )
    up = _expect_slice(
        graph,
        gate_up_bf16.name,
        ((0, compile_rows), (384, 768)),
        f"{row_type}384xbf16>",
    )
    negated = _expect_unary(
        graph,
        gate.name,
        opcode="negate",
        result_type=f"{row_type}384xbf16>",
    )
    exponential = _expect_unary(
        graph,
        negated.name,
        opcode="exponential",
        result_type=f"{row_type}384xbf16>",
    )
    denominator = _only(
        (
            node
            for node in graph.matching_users(
                exponential.name,
                opcode="add",
                result_type=f"{row_type}384xbf16>",
            )
            if len(node.operands) == 2
            and node.operands[1] == exponential.name
        ),
        f"sigmoid denominator for {gate_up.name}",
    )
    one_denominator = graph.node(denominator.operands[0])
    _require_constant_broadcast(
        graph,
        one_denominator,
        literal="1.000000e+00",
        scalar_type="tensor<bf16>",
        result_type=f"{row_type}384xbf16>",
        dimensions=(),
    )
    sigmoid = _only(
        (
            node
            for node in graph.matching_users(
                denominator.name,
                opcode="divide",
                result_type=f"{row_type}384xbf16>",
            )
            if len(node.operands) == 2
            and node.operands[1] == denominator.name
        ),
        f"sigmoid reciprocal for {gate_up.name}",
    )
    one_numerator = graph.node(sigmoid.operands[0])
    _require_constant_broadcast(
        graph,
        one_numerator,
        literal="1.000000e+00",
        scalar_type="tensor<bf16>",
        result_type=f"{row_type}384xbf16>",
        dimensions=(),
    )
    if one_numerator.name == one_denominator.name:
        raise _MatchError(f"{gate_up.name}: sigmoid constants were conflated")
    silu = _expect_binary(
        graph,
        gate.name,
        sigmoid.name,
        opcode="multiply",
        result_type=f"{row_type}384xbf16>",
    )
    activated = _expect_binary(
        graph,
        silu.name,
        up.name,
        opcode="multiply",
        result_type=f"{row_type}384xbf16>",
    )
    down = _only(
        (
            node
            for node in graph.matching_users(
                activated.name,
                opcode="convolution",
                result_type=f"{row_type}6144xf32>",
            )
            if len(node.operands) == 2 and node.operands[0] == activated.name
        ),
        f"down convolution for virtual shard {shard}",
    )
    down_bits, down_scales = _match_fp8_decode(
        graph,
        down.operands[1],
        bit_type="tensor<384x6144xui8>",
        fp8_type="tensor<384x6144xf8E4M3FN>",
        scale_seed_type="tensor<384x48xf32>",
        scale_middle_type="tensor<384x48x128xf32>",
        scale_wide_type="tensor<384x6144xf32>",
        weight_type="tensor<384x6144xbf16>",
        first_broadcast_dimensions=(0, 2),
    )
    _expect_convolution(
        down,
        operands=(activated.name, down.operands[1]),
        tensor_types=(
            f"{row_type}384xbf16>",
            "tensor<384x6144xbf16>",
            f"{row_type}6144xf32>",
        ),
    )
    down_bits_transpose = _expect_node(
        graph,
        down_bits,
        opcode="transpose",
        result_type="tensor<384x6144xui8>",
    )
    if down_bits_transpose.dimensions != (1, 0):
        raise _MatchError(f"{down_bits}: down bit transpose drifted")
    down_bit_slice = _expect_node(
        graph,
        down_bits_transpose.operands[0],
        opcode="slice",
        result_type="tensor<6144x384xui8>",
    )
    if down_bit_slice.slice_ranges != (
        (0, 6144),
        (shard * 384, (shard + 1) * 384),
    ):
        raise _MatchError(f"{down_bit_slice.name}: down bit slice drifted")
    down_scale_transpose = _expect_node(
        graph,
        down_scales,
        opcode="transpose",
        result_type="tensor<3x48xf32>",
    )
    if down_scale_transpose.dimensions != (1, 0):
        raise _MatchError(f"{down_scales}: down scale transpose drifted")
    down_scale_slice = _expect_node(
        graph,
        down_scale_transpose.operands[0],
        opcode="slice",
        result_type="tensor<48x3xf32>",
    )
    if down_scale_slice.slice_ranges != (
        (0, 48),
        (shard * 3, (shard + 1) * 3),
    ):
        raise _MatchError(f"{down_scale_slice.name}: down scale slice drifted")
    down_bf16 = _expect_unary(
        graph,
        down.name,
        opcode="convert",
        result_type=f"{row_type}6144xbf16>",
    )
    roots = (
        normalized,
        bit_slices[0].operands[0],
        scale_slices[0].operands[0],
        bit_slices[1].operands[0],
        scale_slices[1].operands[0],
        down_bit_slice.operands[0],
        down_scale_slice.operands[0],
    )
    return shard, roots, down_bf16.name


def _validate_m32_input_pad(
    graph: _StableGraph,
    helpers: dict[str, _StableGraph],
    call_name: str,
) -> str:
    call = _expect_node(
        graph,
        call_name,
        opcode="call",
        result_type="tensor<32x6144xbf16>",
    )
    if (
        len(call.operands) != 2
        or call.tensor_types[-3:]
        != (
            "tensor<1x6144xbf16>",
            "tensor<bf16>",
            "tensor<32x6144xbf16>",
        )
        or call.callee is None
        or call.callee not in helpers
    ):
        raise _MatchError("M32 discriminator row-pad call drifted")
    zero = _expect_node(
        graph,
        call.operands[1],
        opcode="constant",
        result_type="tensor<bf16>",
    )
    if zero.constant_literal not in {"0.000000e+00", "0.000000E+00"}:
        raise _MatchError("M32 discriminator padding is not BF16 zero")
    helper = helpers[call.callee]
    expected_header = (
        f"func.funcprivate@{call.callee}(%arg0:tensor<1x6144xbf16>,"
        "%arg1:tensor<bf16>)->tensor<32x6144xbf16>{"
    )
    if re.sub(r"\s+", "", helper.header) != expected_header:
        raise _MatchError("M32 discriminator row-pad helper signature drifted")
    pads = [node for node in helper.nodes.values() if node.opcode == "pad"]
    returns = [
        node for node in helper.nodes.values() if node.opcode == "return"
    ]
    if len(pads) != 1 or len(returns) != 1 or len(helper.nodes) != 2:
        raise _MatchError("M32 discriminator row-pad helper body drifted")
    pad = pads[0]
    if (
        pad.operands != ("%arg0", "%arg1")
        or pad.tensor_types[-3:]
        != (
            "tensor<1x6144xbf16>",
            "tensor<bf16>",
            "tensor<32x6144xbf16>",
        )
        or pad.result_type != "tensor<32x6144xbf16>"
        or pad.pad_low != (0, 0)
        or pad.pad_high != (31, 0)
        or pad.pad_interior != (0, 0)
        or returns[0].operands != (pad.name,)
    ):
        raise _MatchError("M32 discriminator row-pad helper semantics drifted")
    return call.operands[0]


def _match_rmsnorm(
    graph: _StableGraph, dense_output: str, *, layer1_only: bool = False
) -> tuple[str, str, str]:
    dense_f32 = _expect_unary(
        graph, dense_output, opcode="convert", result_type="tensor<1x6144xf32>"
    )
    combined = _only(
        (
            node
            for node in graph.matching_users(
                dense_f32.name, opcode="add", result_type="tensor<1x6144xf32>"
            )
            if len(node.operands) == 2 and node.operands[0] == dense_f32.name
        ),
        "dense residual addition",
    )
    residual = _expect_node(
        graph,
        combined.operands[1],
        opcode="convert",
        result_type="tensor<1x6144xf32>",
    )
    square = _expect_unary(
        graph, combined.name, opcode="square", result_type="tensor<1x6144xf32>"
    )
    reduce = _only(
        (
            node
            for node in graph.matching_users(
                square.name, opcode="reduce", result_type="tensor<1xf32>"
            )
            if len(node.operands) == 2 and node.operands[0] == square.name
        ),
        "RMSNorm sum reduction",
    )
    if reduce.dimensions != (1,) or len(reduce.operands) != 2:
        raise _MatchError(f"{reduce.name}: RMSNorm reduction drifted")
    zero = _expect_node(
        graph, reduce.operands[1], opcode="constant", result_type="tensor<f32>"
    )
    if zero.constant_literal != "0.000000e+00":
        raise _MatchError(f"{zero.name}: RMSNorm zero drifted")
    summed = _expect_broadcast(
        graph, reduce.name, dimensions=(0,), result_type="tensor<1x1xf32>"
    )
    width = _expect_constant_broadcast(
        graph,
        literal="6.144000e+03",
        scalar_type="tensor<f32>",
        result_type="tensor<1x1xf32>",
        dimensions=(),
    )
    mean = _expect_binary(
        graph,
        summed.name,
        width.name,
        opcode="divide",
        result_type="tensor<1x1xf32>",
    )
    epsilon = _expect_constant_broadcast(
        graph,
        literal="9.99999974E-6",
        scalar_type="tensor<f32>",
        result_type="tensor<1x1xf32>",
        dimensions=(),
    )
    variance = _expect_binary(
        graph,
        mean.name,
        epsilon.name,
        opcode="add",
        result_type="tensor<1x1xf32>",
    )
    reciprocal = _expect_unary(
        graph, variance.name, opcode="rsqrt", result_type="tensor<1x1xf32>"
    )
    scale = _expect_broadcast(
        graph,
        reciprocal.name,
        dimensions=(0, 1),
        result_type="tensor<1x6144xf32>",
    )
    normalized = _expect_binary(
        graph,
        combined.name,
        scale.name,
        opcode="multiply",
        result_type="tensor<1x6144xf32>",
    )
    rounded = _expect_unary(
        graph, normalized.name, opcode="convert", result_type="tensor<1x6144xbf16>"
    )
    norm_weight = _only(
        (
            node
            for node in graph.nodes.values()
            if node.opcode == "broadcast_in_dim"
            and node.dimensions == (1,)
            and node.result_type == "tensor<1x6144xbf16>"
            and len(node.operands) == 1
        ),
        "layer-1 RMSNorm weight broadcast",
    )
    output = _expect_binary(
        graph,
        rounded.name,
        norm_weight.name,
        opcode="multiply",
        result_type="tensor<1x6144xbf16>",
    )
    expected_return = (output.name,) if layer1_only else (dense_output, output.name)
    _only(
        (
            node
            for node in graph.nodes.values()
            if node.opcode == "return"
            and node.operands == expected_return
        ),
        "probe layer-1-only return"
        if layer1_only
        else "probe dense/layer-1 return",
    )
    if layer1_only and graph.users.get(dense_output, ()) != [dense_f32]:
        raise _MatchError("layer-1-only probe externalizes the dense update")
    return residual.operands[0], norm_weight.operands[0], output.name


def validate_dense_convolution_stablehlo(
    stablehlo: str, *, compile_rows: int = 1, layer1_only: bool = False
) -> dict[str, object]:
    """Validate all eight exact dense chains and the exact StrategyND tree."""

    violations: list[str] = []
    if compile_rows not in (1, 32):
        raise ValueError("dense StableHLO compile rows must be 1 or 32")
    graphs, parse_errors = _parse_graphs(stablehlo)
    violations.extend(parse_errors)
    matched_shards: list[int] = []
    try:
        graph = _only(
            (item for item in graphs if item.name == "main"),
            "dense probe main graph",
        )
        gate_up = [
            node
            for node in graph.nodes.values()
            if node.opcode == "convolution"
            and node.result_type == f"tensor<{compile_rows}x768xf32>"
        ]
        down = [
            node
            for node in graph.nodes.values()
            if node.opcode == "convolution"
            and node.result_type == f"tensor<{compile_rows}x6144xf32>"
        ]
        if len(gate_up) != 8 or len(down) != 8:
            raise _MatchError(
                f"dense convolution count drifted: gate_up={len(gate_up)} "
                f"down={len(down)}"
            )
        rows: dict[int, tuple[tuple[str, ...], str]] = {}
        for convolution in gate_up:
            shard, roots, result = _match_one_shard(
                graph, convolution, compile_rows=compile_rows
            )
            if shard in rows:
                raise _MatchError(f"duplicate dense virtual shard {shard}")
            rows[shard] = (roots, result)
        if set(rows) != set(range(8)):
            raise _MatchError(f"dense virtual shard set drifted: {sorted(rows)}")
        matched_shards = sorted(rows)
        if len({roots for roots, _result in rows.values()}) != 1:
            raise _MatchError("dense virtual shards use different layer sources")
        broadcast_rows = compile_rows
        broadcasts = tuple(
            _expect_broadcast(
                graph,
                rows[shard][1],
                dimensions=(1, 2),
                result_type=f"tensor<1x{broadcast_rows}x6144xbf16>",
            ).name
            for shard in range(8)
        )
        stack = _expect_concat(
            graph,
            broadcasts,
            dimension=0,
            result_type=f"tensor<8x{broadcast_rows}x6144xbf16>",
        )
        if compile_rows == 1:
            live_stack = stack
        else:
            anchored_stack = _expect_unary(
                graph,
                stack.name,
                opcode="optimization_barrier",
                result_type="tensor<8x32x6144xbf16>",
            )
            live_stack = _expect_slice(
                graph,
                anchored_stack.name,
                ((0, 8), (0, 1), (0, 6144)),
                "tensor<8x1x6144xbf16>",
            )
        gathered_input = _expect_broadcast(
            graph,
            live_stack.name,
            dimensions=(1, 2, 3),
            result_type="tensor<1x8x1x6144xbf16>",
        )
        gather = _only(
            (
                node
                for node in graph.matching_users(
                    gathered_input.name,
                    opcode="all_gather",
                    result_type="tensor<4x8x1x6144xbf16>",
                )
                if node.operands == (gathered_input.name,)
            ),
            "dense StrategyND all-gather",
        )
        normalized_gather = re.sub(r"\s+", "", gather.raw_line)
        if (
            gather.all_gather_dimension != 0
            or not gather.use_global_device_ids
            or "replica_groups=dense<[[0,1,2,3]]>:tensor<1x4xi64>"
            not in normalized_gather
        ):
            raise _MatchError(f"{gather.name}: dense gather group drifted")
        dense_output = _match_reduction_tree(graph, gather)
        residual, norm_weight, _layer1 = _match_rmsnorm(
            graph, dense_output, layer1_only=layer1_only
        )
        layer_roots = next(iter(rows.values()))[0]
        root_contracts = (
            ("tensor<1x3072x6144xui8>", "tensor<3072x6144xui8>"),
            ("tensor<1x24x48xf32>", "tensor<24x48xf32>"),
            ("tensor<1x3072x6144xui8>", "tensor<3072x6144xui8>"),
            ("tensor<1x24x48xf32>", "tensor<24x48xf32>"),
            ("tensor<1x6144x3072xui8>", "tensor<6144x3072xui8>"),
            ("tensor<1x48x24xf32>", "tensor<48x24xf32>"),
        )
        normalized_root = layer_roots[0]
        if compile_rows == 32:
            helpers = {item.name: item for item in graphs if item is not graph}
            normalized_root = _validate_m32_input_pad(
                graph, helpers, normalized_root
            )
        manual_arguments = [normalized_root]
        for root, tensor_types in zip(layer_roots[1:], root_contracts, strict=True):
            node = _expect_node(
                graph,
                root,
                opcode="reshape",
                result_type=tensor_types[1],
            )
            if node.tensor_types[-2:] != tensor_types or len(node.operands) != 1:
                raise _MatchError(f"{root}: dense checkpoint source reshape drifted")
            manual_arguments.append(node.operands[0])
        manual_arguments.extend((residual, norm_weight))
        manual_call = re.search(
            r"sdy\.manual_computation\(([^)]*)\)", stablehlo
        )
        outer_arguments = (
            []
            if manual_call is None
            else [value.strip() for value in manual_call.group(1).split(",")]
        )
        if (
            len(set(manual_arguments)) != 9
            or any(name in graph.nodes for name in manual_arguments)
            or manual_arguments
            != [
                "%arg9",
                "%arg11",
                "%arg12",
                "%arg13",
                "%arg14",
                "%arg15",
                "%arg16",
                "%arg10",
                "%arg17",
            ]
            or outer_arguments != [f"%arg{index}" for index in range(9)]
        ):
            raise _MatchError("dense probe manual argument ownership drifted")
        expected_graph_count = 2 if compile_rows == 32 else 1
        if len(graphs) != expected_graph_count:
            raise _MatchError("dense probe contains unexpected helper functions")
    except _MatchError as error:
        violations.append(str(error))
    collective_counts = {
        opcode: stablehlo.count(f"stablehlo.{opcode}")
        for opcode in (
            "all_gather",
            "all_reduce",
            "all_to_all",
            "collective_broadcast",
            "collective_permute",
            "reduce_scatter",
        )
    }
    if collective_counts != {
        "all_gather": 1,
        "all_reduce": 0,
        "all_to_all": 0,
        "collective_broadcast": 0,
        "collective_permute": 0,
        "reduce_scatter": 0,
    }:
        violations.append(f"pre-fusion collective contract drifted: {collective_counts}")
    if compile_rows == 1 and "tensor<32x6144xbf16>" in stablehlo:
        violations.append("dense convolution reconstructed 32 token rows")
    if "xla_python_cpu_callback" in stablehlo or "host_callback" in stablehlo:
        violations.append("pre-fusion module contains a host callback")
    return {
        "collective_counts": collective_counts,
        "compile_rows": compile_rows,
        "convolution_count": stablehlo.count("stablehlo.convolution"),
        "down_convolution_count": 8 if matched_shards == list(range(8)) else 0,
        "gate_up_convolution_count": 8 if matched_shards == list(range(8)) else 0,
        "matched_virtual_shards": matched_shards,
        "live_rows": 1,
        "result_mode": "layer1_only" if layer1_only else "dense_and_layer1",
        "passed": not violations,
        "violations": violations,
    }
