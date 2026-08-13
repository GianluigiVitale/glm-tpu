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


def _expand_dependency_barriers(stablehlo: str) -> tuple[str, list[str]]:
    """Expose the two exact results of the virtual-shard ordering barrier.

    The shared StableHLO parser models one SSA result per node, while JAX emits
    ``%n:2`` plus ``%n#0``/``%n#1`` for a tuple-valued optimization barrier.
    Expand only the exact two-value form used here into parser-only nodes.  Each
    result retains both operands, so the dense contract can prove the live
    weight result is ordered after the preceding shard result.
    """

    pattern = re.compile(
        r"^(?P<indent>\s*)(?P<name>%[A-Za-z0-9_.$-]+):2\s*=\s*"
        r"stablehlo\.optimization_barrier\s+"
        r"(?P<left>%[A-Za-z0-9_.$#-]+),\s*"
        r"(?P<right>%[A-Za-z0-9_.$#-]+)\s*:\s*"
        r"(?P<left_type>tensor<[^>]+>),\s*"
        r"(?P<right_type>tensor<[^>]+>)\s*$"
    )
    lines: list[str] = []
    errors: list[str] = []
    for line_number, line in enumerate(stablehlo.splitlines(), start=1):
        if ":2" not in line or "stablehlo.optimization_barrier" not in line:
            lines.append(line)
            continue
        match = pattern.fullmatch(line)
        if match is None:
            errors.append(
                "dense dependency barrier has an unknown StableHLO form at "
                f"line {line_number}"
            )
            lines.append(line)
            continue
        values = match.groupdict()
        lines.extend(
            (
                f"{values['indent']}{values['name']}#0 = "
                "stablehlo.optimization_barrier "
                f"{values['left']}, {values['right']} : "
                f"{values['left_type']}",
                f"{values['indent']}{values['name']}#1 = "
                "stablehlo.optimization_barrier "
                f"{values['right']}, {values['left']} : "
                f"{values['right_type']}",
            )
        )
    return "\n".join(lines), errors


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


def _expect_accepted_gate_up_layout(
    graph: _StableGraph,
    output: str,
) -> tuple[str, str | None, str | None]:
    """Bind one gate/up RHS to the exact accepted layout constraint."""

    dependency: str | None = None
    dependency_output: str | None = None
    node = _expect_node(
        graph,
        output,
        opcode=(
            "optimization_barrier"
            if "#" in output
            else "custom_call"
        ),
        result_type="tensor<6144x768xbf16>",
    )
    if node.opcode == "optimization_barrier":
        if (
            not node.name.endswith("#0")
            or len(node.operands) != 2
            or graph.node(node.operands[1]).result_type
            != "tensor<32x6144xbf16>"
        ):
            raise _MatchError(
                f"{node.name}: virtual-shard dependency barrier drifted"
            )
        dependency = node.operands[1]
        dependency_output = node.name[:-1] + "1"
        sibling = _expect_node(
            graph,
            dependency_output,
            opcode="optimization_barrier",
            result_type="tensor<32x6144xbf16>",
        )
        if sibling.operands != (dependency, node.operands[0]):
            raise _MatchError(
                f"{sibling.name}: predecessor barrier result drifted"
            )
        node = _expect_node(
            graph,
            node.operands[0],
            opcode="custom_call",
            result_type="tensor<6144x768xbf16>",
        )
    normalized = re.sub(r"\s+", "", node.raw_line)
    if (
        len(node.operands) != 1
        or node.tensor_types[-2:]
        != (
            "tensor<6144x768xbf16>",
            "tensor<6144x768xbf16>",
        )
        or "stablehlo.custom_call@LayoutConstraint(" not in normalized
        or "operand_layouts=[dense<[0,1]>:tensor<2xindex>]" not in normalized
        or "result_layouts=[dense<[1,0]>:tensor<2xindex>]" not in normalized
    ):
        raise _MatchError(f"{node.name}: accepted gate/up layout drifted")
    return node.operands[0], dependency, dependency_output


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
    direct_fp8_expanded_output_scale: bool = False,
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
    if direct_fp8_expanded_output_scale:
        fp8 = _expect_node(
            graph, fp32.operands[0], opcode="reshape", result_type=fp8_type
        )
        if fp8.result_type != bit_type:
            raise _MatchError(f"{fp8.name}: direct FP8 geometry drifted")
    else:
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
    if direct_fp8_expanded_output_scale:
        if second_broadcast.dimensions != first_broadcast_dimensions:
            raise _MatchError(
                f"{second_broadcast.name}: expanded-output scale drifted"
            )
        return fp8.name, second_broadcast.operands[0]
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
    final_dense_layout: bool = False,
) -> tuple[int, tuple[str, ...], str, str | None, str | None]:
    row_type = f"tensor<{compile_rows}x"
    if len(gate_up.operands) != 2:
        raise _MatchError(f"{gate_up.name}: gate/up operand arity drifted")
    normalized = gate_up.operands[0]
    decoded_gate_up = gate_up.operands[1]
    dependency: str | None = None
    dependency_output: str | None = None
    if final_dense_layout:
        (
            decoded_gate_up,
            dependency,
            dependency_output,
        ) = _expect_accepted_gate_up_layout(graph, decoded_gate_up)
    gate_bits, gate_scales = _match_fp8_decode(
        graph,
        decoded_gate_up,
        bit_type=(
            "tensor<6144x768xf8E4M3FN>"
            if final_dense_layout
            else "tensor<6144x768xui8>"
        ),
        fp8_type="tensor<6144x768xf8E4M3FN>",
        scale_seed_type=(
            "tensor<48x768xf32>"
            if final_dense_layout
            else "tensor<6144x6xf32>"
        ),
        scale_middle_type=(
            "tensor<48x128x768xf32>"
            if final_dense_layout
            else "tensor<6144x6x128xf32>"
        ),
        scale_wide_type="tensor<6144x768xf32>",
        weight_type="tensor<6144x768xbf16>",
        first_broadcast_dimensions=(0, 2),
        direct_fp8_expanded_output_scale=final_dense_layout,
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
    if final_dense_layout:
        gate_bits_reshape = _expect_node(
            graph,
            gate_bits,
            opcode="reshape",
            result_type="tensor<6144x768xf8E4M3FN>",
        )
        gate_bits_slice = _expect_node(
            graph,
            gate_bits_reshape.operands[0],
            opcode="slice",
            result_type="tensor<1x6144x768xf8E4M3FN>",
        )
        if gate_bits_slice.slice_ranges is None:
            raise _MatchError(f"{gate_bits_slice.name}: packed gate slice absent")
        shard = gate_bits_slice.slice_ranges[0][0]
        if gate_bits_slice.slice_ranges != (
            (shard, shard + 1),
            (0, 6144),
            (0, 768),
        ):
            raise _MatchError(f"{gate_bits_slice.name}: packed gate slice drifted")
        gate_scale_reshape = _expect_node(
            graph,
            gate_scales,
            opcode="reshape",
            result_type="tensor<48x768xf32>",
        )
        gate_scale_slice = _expect_node(
            graph,
            gate_scale_reshape.operands[0],
            opcode="slice",
            result_type="tensor<1x48x768xf32>",
        )
        if gate_scale_slice.slice_ranges != (
            (shard, shard + 1),
            (0, 48),
            (0, 768),
        ):
            raise _MatchError(
                f"{gate_scale_slice.name}: packed gate scale slice drifted"
            )
        gate_bit_root = gate_bits_slice.operands[0]
        gate_scale_root = gate_scale_slice.operands[0]
    else:
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
        gate_bit_root = bit_slices[0].operands[0]
        gate_scale_root = scale_slices[0].operands[0]

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
        bit_type=(
            "tensor<384x6144xf8E4M3FN>"
            if final_dense_layout
            else "tensor<384x6144xui8>"
        ),
        fp8_type="tensor<384x6144xf8E4M3FN>",
        scale_seed_type=(
            "tensor<3x6144xf32>"
            if final_dense_layout
            else "tensor<384x48xf32>"
        ),
        scale_middle_type=(
            "tensor<3x128x6144xf32>"
            if final_dense_layout
            else "tensor<384x48x128xf32>"
        ),
        scale_wide_type="tensor<384x6144xf32>",
        weight_type="tensor<384x6144xbf16>",
        first_broadcast_dimensions=(0, 2),
        direct_fp8_expanded_output_scale=final_dense_layout,
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
    if final_dense_layout:
        down_bits_reshape = _expect_node(
            graph,
            down_bits,
            opcode="reshape",
            result_type="tensor<384x6144xf8E4M3FN>",
        )
        down_bit_slice = _expect_node(
            graph,
            down_bits_reshape.operands[0],
            opcode="slice",
            result_type="tensor<1x384x6144xf8E4M3FN>",
        )
        if down_bit_slice.slice_ranges != (
            (shard, shard + 1),
            (0, 384),
            (0, 6144),
        ):
            raise _MatchError(f"{down_bit_slice.name}: packed down slice drifted")
        down_scale_reshape = _expect_node(
            graph,
            down_scales,
            opcode="reshape",
            result_type="tensor<3x6144xf32>",
        )
        down_scale_slice = _expect_node(
            graph,
            down_scale_reshape.operands[0],
            opcode="slice",
            result_type="tensor<1x3x6144xf32>",
        )
        if down_scale_slice.slice_ranges != (
            (shard, shard + 1),
            (0, 3),
            (0, 6144),
        ):
            raise _MatchError(
                f"{down_scale_slice.name}: packed down scale slice drifted"
            )
        down_bit_root = down_bit_slice.operands[0]
        down_scale_root = down_scale_slice.operands[0]
    else:
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
        down_bit_root = down_bit_slice.operands[0]
        down_scale_root = down_scale_slice.operands[0]
    down_bf16 = _expect_unary(
        graph,
        down.name,
        opcode="convert",
        result_type=f"{row_type}6144xbf16>",
    )
    roots = (
        normalized,
        gate_bit_root,
        gate_scale_root,
        *((down_bit_root, down_scale_root) if final_dense_layout else (
            bit_slices[1].operands[0],
            scale_slices[1].operands[0],
            down_bit_root,
            down_scale_root,
        )),
    )
    return shard, roots, down_bf16.name, dependency, dependency_output


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
    graph: _StableGraph,
    helpers: dict[str, _StableGraph],
    dense_output: str,
    *,
    layer1_only: bool = False,
    residual_is_m32: bool = False,
) -> tuple[str, str, str]:
    rows = 32 if layer1_only else 1
    row_type = f"tensor<{rows}x6144x"
    scalar_row_type = f"tensor<{rows}x1xf32>"

    def exact_m32_pad(source: str) -> str:
        call = _only(
            (
                node
                for node in graph.matching_users(
                    source,
                    opcode="call",
                    result_type="tensor<32x6144xbf16>",
                )
                if node.operands and node.operands[0] == source
            ),
            f"M32 RMSNorm pad for {source}",
        )
        if _validate_m32_input_pad(graph, helpers, call.name) != source:
            raise _MatchError(f"{call.name}: M32 RMSNorm pad source drifted")
        return call.name

    dense_value = exact_m32_pad(dense_output) if layer1_only else dense_output
    dense_f32 = _expect_unary(
        graph,
        dense_value,
        opcode="convert",
        result_type=f"{row_type}f32>",
    )
    combined = _only(
        (
            node
            for node in graph.matching_users(
                dense_f32.name,
                opcode="add",
                result_type=f"{row_type}f32>",
            )
            if len(node.operands) == 2 and node.operands[0] == dense_f32.name
        ),
        "dense residual addition",
    )
    residual = _expect_node(
        graph,
        combined.operands[1],
        opcode="convert",
        result_type=f"{row_type}f32>",
    )
    residual = residual.operands[0]
    if layer1_only and not residual_is_m32:
        residual_pad = _expect_node(
            graph,
            residual,
            opcode="call",
            result_type="tensor<32x6144xbf16>",
        )
        if len(residual_pad.operands) != 2:
            raise _MatchError(f"{residual_pad.name}: residual pad arity drifted")
        residual = _validate_m32_input_pad(
            graph, helpers, residual_pad.name
        )
    square = _expect_unary(
        graph,
        combined.name,
        opcode="square",
        result_type=f"{row_type}f32>",
    )
    reduce = _only(
        (
            node
            for node in graph.matching_users(
                square.name,
                opcode="reduce",
                result_type=f"tensor<{rows}xf32>",
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
        graph,
        reduce.name,
        dimensions=(0,),
        result_type=scalar_row_type,
    )
    mean = _only(
        (
            node
            for node in graph.matching_users(
                summed.name,
                opcode="divide",
                result_type=scalar_row_type,
            )
            if len(node.operands) == 2 and node.operands[0] == summed.name
        ),
        "RMSNorm mean division",
    )
    width = _expect_node(
        graph,
        mean.operands[1],
        opcode="broadcast_in_dim",
        result_type=scalar_row_type,
    )
    _require_constant_broadcast(
        graph,
        width,
        literal="6.144000e+03",
        scalar_type="tensor<f32>",
        result_type=scalar_row_type,
        dimensions=(),
    )
    variance = _only(
        (
            node
            for node in graph.matching_users(
                mean.name,
                opcode="add",
                result_type=scalar_row_type,
            )
            if len(node.operands) == 2 and mean.name in node.operands
        ),
        "RMSNorm epsilon addition",
    )
    epsilon_name = next(name for name in variance.operands if name != mean.name)
    epsilon = _expect_node(
        graph,
        epsilon_name,
        opcode="broadcast_in_dim",
        result_type=scalar_row_type,
    )
    _require_constant_broadcast(
        graph,
        epsilon,
        literal="9.99999974E-6",
        scalar_type="tensor<f32>",
        result_type=scalar_row_type,
        dimensions=(),
    )
    reciprocal = _expect_unary(
        graph,
        variance.name,
        opcode="rsqrt",
        result_type=scalar_row_type,
    )
    scale = _expect_broadcast(
        graph,
        reciprocal.name,
        dimensions=(0, 1),
        result_type=f"{row_type}f32>",
    )
    normalized = _expect_binary(
        graph,
        combined.name,
        scale.name,
        opcode="multiply",
        result_type=f"{row_type}f32>",
    )
    rounded = _expect_unary(
        graph,
        normalized.name,
        opcode="convert",
        result_type=f"{row_type}bf16>",
    )
    output = _only(
        (
            node
            for node in graph.matching_users(
                rounded.name,
                opcode="multiply",
                result_type=f"{row_type}bf16>",
            )
            if len(node.operands) == 2 and rounded.name in node.operands
        ),
        "layer-1 RMSNorm weighted output",
    )
    norm_weight_name = next(
        name for name in output.operands if name != rounded.name
    )
    norm_weight = _expect_node(
        graph,
        norm_weight_name,
        opcode="broadcast_in_dim",
        result_type=f"{row_type}bf16>",
    )
    if layer1_only:
        if norm_weight.dimensions != (0, 1) or len(norm_weight.operands) != 1:
            raise _MatchError("layer-1 RMSNorm M32 weight broadcast drifted")
        norm_weight_seed = _expect_node(
            graph,
            norm_weight.operands[0],
            opcode="broadcast_in_dim",
            result_type="tensor<1x6144xbf16>",
        )
    else:
        norm_weight_seed = norm_weight
    if norm_weight_seed.dimensions != (1,) or len(norm_weight_seed.operands) != 1:
        raise _MatchError("layer-1 RMSNorm weight seed drifted")
    returned_output = output.name
    if layer1_only:
        returned_output = _expect_slice(
            graph,
            output.name,
            ((0, 1), (0, 6144)),
            "tensor<1x6144xbf16>",
        ).name
    expected_return = (
        (returned_output,)
        if layer1_only
        else (dense_output, returned_output)
    )
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
    if layer1_only and graph.users.get(dense_output, ()) != [
        graph.node(dense_value)
    ]:
        raise _MatchError("layer-1-only probe externalizes the dense update")
    return residual, norm_weight_seed.operands[0], returned_output


def _match_predense_rmsnorm(
    graph: _StableGraph,
    helpers: dict[str, _StableGraph],
    normalized_output: str,
    carried_residual: str,
) -> tuple[tuple[str, str], str]:
    """Bind the exact M32 fused add/RMSNorm feeding every gate convolution."""

    row_bf16 = "tensor<32x6144xbf16>"
    row_f32 = "tensor<32x6144xf32>"
    scalar_row = "tensor<32x1xf32>"
    output = _expect_node(
        graph, normalized_output, opcode="multiply", result_type=row_bf16
    )
    if len(output.operands) != 2:
        raise _MatchError("pre-dense RMSNorm weighted output arity drifted")
    rounded_candidates = [
        graph.node(name)
        for name in output.operands
        if graph.node(name).opcode == "convert"
        and graph.node(name).result_type == row_bf16
    ]
    weight_candidates = [
        graph.node(name)
        for name in output.operands
        if graph.node(name).opcode == "broadcast_in_dim"
        and graph.node(name).result_type == row_bf16
        and graph.node(name).dimensions == (0, 1)
    ]
    rounded = _only(rounded_candidates, "pre-dense RMSNorm BF16 round")
    weight = _only(weight_candidates, "pre-dense RMSNorm weight broadcast")
    normalized = _expect_node(
        graph, rounded.operands[0], opcode="multiply", result_type=row_f32
    )
    if len(normalized.operands) != 2:
        raise _MatchError("pre-dense normalized multiply arity drifted")
    combined_candidates = [
        graph.node(name)
        for name in normalized.operands
        if graph.node(name).opcode == "add"
        and graph.node(name).result_type == row_f32
    ]
    scale_candidates = [
        graph.node(name)
        for name in normalized.operands
        if graph.node(name).opcode == "broadcast_in_dim"
        and graph.node(name).result_type == row_f32
        and graph.node(name).dimensions == (0, 1)
    ]
    combined = _only(combined_candidates, "pre-dense FP32 residual add")
    scale = _only(scale_candidates, "pre-dense reciprocal broadcast")
    carried = _expect_node(
        graph, carried_residual, opcode="convert", result_type=row_bf16
    )
    if carried.operands != (combined.name,):
        raise _MatchError("pre-dense carried residual differs from the FP32 sum")
    square = _expect_unary(
        graph, combined.name, opcode="square", result_type=row_f32
    )
    reduce = _only(
        (
            node
            for node in graph.matching_users(
                square.name, opcode="reduce", result_type="tensor<32xf32>"
            )
            if len(node.operands) == 2 and node.operands[0] == square.name
        ),
        "pre-dense RMSNorm sum reduction",
    )
    if reduce.dimensions != (1,):
        raise _MatchError("pre-dense RMSNorm reduction axis drifted")
    if "appliesstablehlo.add" not in re.sub(r"\s+", "", reduce.raw_line):
        raise _MatchError("pre-dense RMSNorm reduction combiner drifted")
    zero = _expect_node(
        graph, reduce.operands[1], opcode="constant", result_type="tensor<f32>"
    )
    if zero.constant_literal != "0.000000e+00":
        raise _MatchError("pre-dense RMSNorm reduction zero drifted")
    if len(scale.operands) != 1:
        raise _MatchError("pre-dense reciprocal broadcast source drifted")
    reciprocal = _expect_node(
        graph, scale.operands[0], opcode="rsqrt", result_type=scalar_row
    )
    variance = _expect_node(
        graph, reciprocal.operands[0], opcode="add", result_type=scalar_row
    )
    if len(variance.operands) != 2:
        raise _MatchError("pre-dense variance add arity drifted")
    mean = _only(
        (
            graph.node(name)
            for name in variance.operands
            if graph.node(name).opcode == "divide"
            and graph.node(name).result_type == scalar_row
        ),
        "pre-dense RMSNorm mean",
    )
    epsilon = _only(
        (
            graph.node(name)
            for name in variance.operands
            if graph.node(name).opcode == "broadcast_in_dim"
            and graph.node(name).result_type == scalar_row
        ),
        "pre-dense RMSNorm epsilon",
    )
    _require_constant_broadcast(
        graph,
        epsilon,
        literal="9.99999974E-6",
        scalar_type="tensor<f32>",
        result_type=scalar_row,
        dimensions=(),
    )
    if len(mean.operands) != 2:
        raise _MatchError("pre-dense mean division arity drifted")
    summed = _expect_node(
        graph,
        mean.operands[0],
        opcode="broadcast_in_dim",
        result_type=scalar_row,
    )
    if summed.dimensions != (0,) or summed.operands != (reduce.name,):
        raise _MatchError("pre-dense reduction broadcast drifted")
    width = _expect_node(
        graph,
        mean.operands[1],
        opcode="broadcast_in_dim",
        result_type=scalar_row,
    )
    _require_constant_broadcast(
        graph,
        width,
        literal="6.144000e+03",
        scalar_type="tensor<f32>",
        result_type=scalar_row,
        dimensions=(),
    )
    sources = []
    for name in combined.operands:
        converted = _expect_node(
            graph, name, opcode="convert", result_type=row_f32
        )
        call = _expect_node(
            graph,
            converted.operands[0],
            opcode="call",
            result_type=row_bf16,
        )
        sources.append(_validate_m32_input_pad(graph, helpers, call.name))
    if len(set(sources)) != 2:
        raise _MatchError("pre-dense add inputs are duplicated")
    weight_seed = _expect_node(
        graph,
        weight.operands[0],
        opcode="broadcast_in_dim",
        result_type="tensor<1x6144xbf16>",
    )
    if weight_seed.dimensions != (1,) or len(weight_seed.operands) != 1:
        raise _MatchError("pre-dense RMSNorm weight seed drifted")
    return (sources[0], sources[1]), weight_seed.operands[0]


def validate_dense_convolution_stablehlo(
    stablehlo: str,
    *,
    compile_rows: int = 1,
    layer1_only: bool = False,
    final_dense_layout: bool = False,
    dense_envelope: bool = False,
) -> dict[str, object]:
    """Validate all eight exact dense chains and the exact StrategyND tree."""

    violations: list[str] = []
    if compile_rows not in (1, 32):
        raise ValueError("dense StableHLO compile rows must be 1 or 32")
    if final_dense_layout and (compile_rows != 32 or not layer1_only):
        raise ValueError(
            "final-layout dense StableHLO proof requires M32 layer1-only"
        )
    if dense_envelope and not final_dense_layout:
        raise ValueError(
            "dense-envelope StableHLO proof requires final-layout mode"
        )
    parsed_stablehlo, dependency_errors = _expand_dependency_barriers(
        stablehlo
    )
    graphs, parse_errors = _parse_graphs(parsed_stablehlo)
    violations.extend(dependency_errors)
    violations.extend(parse_errors)
    matched_shards: list[int] = []
    layout_constraint_count = sum(
        1
        for item in graphs
        for node in item.nodes.values()
        if node.opcode == "custom_call"
        and "@LayoutConstraint(" in node.raw_line
    )
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
        layout_constraints = [
            node
            for node in graph.nodes.values()
            if node.opcode == "custom_call"
            and "@LayoutConstraint(" in node.raw_line
        ]
        expected_layout_constraints = 8 if final_dense_layout else 0
        if len(layout_constraints) != expected_layout_constraints:
            raise _MatchError(
                "dense gate/up layout-constraint count drifted: "
                f"expected={expected_layout_constraints} "
                f"found={len(layout_constraints)}"
            )
        rows: dict[
            int,
            tuple[tuple[str, ...], str, str | None, str | None],
        ] = {}
        for convolution in gate_up:
            (
                shard,
                roots,
                result,
                dependency,
                dependency_output,
            ) = _match_one_shard(
                graph,
                convolution,
                compile_rows=compile_rows,
                final_dense_layout=final_dense_layout,
            )
            if shard in rows:
                raise _MatchError(f"duplicate dense virtual shard {shard}")
            rows[shard] = (
                roots,
                result,
                dependency,
                dependency_output,
            )
        if set(rows) != set(range(8)):
            raise _MatchError(f"dense virtual shard set drifted: {sorted(rows)}")
        matched_shards = sorted(rows)
        if len(
            {
                roots
                for roots, _result, _dependency, _dependency_output
                in rows.values()
            }
        ) != 1:
            raise _MatchError("dense virtual shards use different layer sources")
        if final_dense_layout and (
            rows[0][2] is not None
            or any(
                rows[shard][2] != rows[shard - 1][1]
                for shard in range(1, 8)
            )
        ):
            raise _MatchError(
                "dense virtual shards lost the exact rank-ordered dependency"
            )
        broadcast_rows = compile_rows
        broadcasts = tuple(
            _expect_broadcast(
                graph,
                (
                    rows[shard + 1][3]
                    if final_dense_layout and shard < 7
                    else rows[shard][1]
                ),
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
        helpers = {item.name: item for item in graphs if item is not graph}
        residual, norm_weight, _layer1 = _match_rmsnorm(
            graph,
            helpers,
            dense_output,
            layer1_only=layer1_only,
            residual_is_m32=dense_envelope,
        )
        layer_roots = next(iter(rows.values()))[0]
        root_contracts = (
            (
                (
                    "tensor<1x8x6144x768xf8E4M3FN>",
                    "tensor<8x6144x768xf8E4M3FN>",
                ),
                ("tensor<1x8x48x768xf32>", "tensor<8x48x768xf32>"),
                (
                    "tensor<1x8x384x6144xf8E4M3FN>",
                    "tensor<8x384x6144xf8E4M3FN>",
                ),
                ("tensor<1x8x3x6144xf32>", "tensor<8x3x6144xf32>"),
            )
            if final_dense_layout
            else (
                ("tensor<1x3072x6144xui8>", "tensor<3072x6144xui8>"),
                ("tensor<1x24x48xf32>", "tensor<24x48xf32>"),
                ("tensor<1x3072x6144xui8>", "tensor<3072x6144xui8>"),
                ("tensor<1x24x48xf32>", "tensor<24x48xf32>"),
                ("tensor<1x6144x3072xui8>", "tensor<6144x3072xui8>"),
                ("tensor<1x48x24xf32>", "tensor<48x24xf32>"),
            )
        )
        normalized_root = layer_roots[0]
        if compile_rows == 32 and not dense_envelope:
            normalized_root = _validate_m32_input_pad(
                graph, helpers, normalized_root
            )
        if dense_envelope:
            predense_sources, predense_norm = _match_predense_rmsnorm(
                graph,
                helpers,
                layer_roots[0],
                residual,
            )
            gate_users = {
                node.name
                for node in graph.users.get(layer_roots[0], ())
                if node.opcode == "convolution"
            }
            if gate_users != {node.name for node in gate_up}:
                raise _MatchError(
                    "pre-dense RMSNorm output does not feed exactly eight gate convolutions"
                )
            manual_arguments = [*predense_sources, predense_norm]
        else:
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
        if dense_envelope:
            manual_arguments.append(norm_weight)
        else:
            manual_arguments.extend((residual, norm_weight))
        manual_call = re.search(
            r"sdy\.manual_computation\(([^)]*)\)", stablehlo
        )
        outer_arguments = (
            []
            if manual_call is None
            else [value.strip() for value in manual_call.group(1).split(",")]
        )
        expected_manual_arguments = (
            [
                "%arg8",
                "%arg9",
                "%arg10",
                "%arg11",
                "%arg12",
                "%arg13",
                "%arg14",
                "%arg15",
            ]
            if dense_envelope
            else ["%arg7", "%arg9", "%arg10", "%arg11", "%arg12", "%arg8", "%arg13"]
            if final_dense_layout
            else [
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
        )
        expected_outer_arguments = [
            f"%arg{index}"
            for index in range(
                8 if dense_envelope else 7 if final_dense_layout else 9
            )
        ]
        if (
            len(set(manual_arguments)) != len(expected_manual_arguments)
            or any(name in graph.nodes for name in manual_arguments)
            or manual_arguments != expected_manual_arguments
            or outer_arguments != expected_outer_arguments
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
        "dense_envelope": dense_envelope,
        "final_dense_layout": final_dense_layout,
        "gate_up_convolution_count": 8 if matched_shards == list(range(8)) else 0,
        "gate_up_layout_constraint_count": layout_constraint_count,
        "matched_virtual_shards": matched_shards,
        "live_rows": 1,
        "result_mode": "layer1_only" if layer1_only else "dense_and_layer1",
        "passed": not violations,
        "violations": violations,
    }
