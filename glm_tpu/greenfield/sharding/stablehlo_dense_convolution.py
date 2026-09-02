"""Exact pre-fusion contract for the layer-0 dense-convolution probe."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, Sequence

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
    *,
    compile_rows: int,
) -> tuple[str, str | None, str | None]:
    """Bind one gate/up RHS to the exact accepted layout constraint."""

    dependency: str | None = None
    dependency_output: str | None = None
    node = graph.node(output)
    if (
        node.opcode != "optimization_barrier"
        or node.result_type != "tensor<6144x768xbf16>"
    ):
        raise _MatchError(f"{node.name}: accepted gate/up barrier drifted")
    if node.name.endswith("#0"):
        if (
            len(node.operands) != 2
            or graph.node(node.operands[1]).result_type
            != f"tensor<{compile_rows}x6144xbf16>"
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
            result_type=f"tensor<{compile_rows}x6144xbf16>",
        )
        if sibling.operands != (dependency, node.operands[0]):
            raise _MatchError(
                f"{sibling.name}: predecessor barrier result drifted"
            )
    elif "#" in node.name or len(node.operands) != 1:
        raise _MatchError(
            f"{node.name}: rank-zero materialization barrier drifted"
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
    single_virtual_shard: bool = False,
    accepted_gate_singleton: bool = False,
    accepted_gate_dequant_fusion: bool = False,
) -> tuple[int, tuple[str, ...], str, str | None, str | None]:
    row_type = f"tensor<{compile_rows}x"
    if len(gate_up.operands) != 2:
        raise _MatchError(f"{gate_up.name}: gate/up operand arity drifted")
    normalized = gate_up.operands[0]
    decoded_gate_up = gate_up.operands[1]
    dependency: str | None = None
    dependency_output: str | None = None
    if final_dense_layout and not accepted_gate_dequant_fusion:
        (
            decoded_gate_up,
            dependency,
            dependency_output,
        ) = _expect_accepted_gate_up_layout(
            graph, decoded_gate_up, compile_rows=compile_rows
        )
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
        def exact_single_rank_root(
            source: str,
            *,
            singleton_type: str,
            owner_type: str,
            ui8_type: str | None = None,
        ) -> str:
            node = graph.node(source)
            if ui8_type is not None and node.opcode == "bitcast_convert":
                if (
                    len(node.operands) != 1
                    or node.tensor_types[-2:] != (ui8_type, singleton_type)
                ):
                    raise _MatchError(
                        f"{node.name}: isolated runtime FP8 bitcast drifted"
                    )
                node = graph.node(node.operands[0])
                singleton_type = ui8_type
                owner_type = owner_type.replace("xf8E4M3FN>", "xui8>")
            if (
                node.opcode != "reshape"
                or len(node.operands) != 1
                or node.tensor_types[-2:] != (owner_type, singleton_type)
                or node.operands[0] in graph.nodes
            ):
                raise _MatchError(
                    f"{node.name}: isolated virtual-rank source drifted"
                )
            return node.operands[0]

        gate_bits_reshape = _expect_node(
            graph,
            gate_bits,
            opcode="reshape",
            result_type="tensor<6144x768xf8E4M3FN>",
        )
        if single_virtual_shard:
            shard = 0
            gate_bit_root = exact_single_rank_root(
                gate_bits_reshape.operands[0],
                singleton_type="tensor<1x6144x768xf8E4M3FN>",
                owner_type="tensor<1x1x6144x768xf8E4M3FN>",
                ui8_type="tensor<1x6144x768xui8>",
            )
        else:
            gate_bits_slice = _expect_node(
                graph,
                gate_bits_reshape.operands[0],
                opcode="slice",
                result_type="tensor<1x6144x768xf8E4M3FN>",
            )
            if gate_bits_slice.slice_ranges is None:
                raise _MatchError(
                    f"{gate_bits_slice.name}: packed gate slice absent"
                )
            shard = gate_bits_slice.slice_ranges[0][0]
            if gate_bits_slice.slice_ranges != (
                (shard, shard + 1),
                (0, 6144),
                (0, 768),
            ):
                raise _MatchError(
                    f"{gate_bits_slice.name}: packed gate slice drifted"
                )
            gate_bit_root = gate_bits_slice.operands[0]
        gate_scale_reshape = _expect_node(
            graph,
            gate_scales,
            opcode="reshape",
            result_type="tensor<48x768xf32>",
        )
        if single_virtual_shard:
            gate_scale_root = exact_single_rank_root(
                gate_scale_reshape.operands[0],
                singleton_type="tensor<1x48x768xf32>",
                owner_type="tensor<1x1x48x768xf32>",
            )
        else:
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
    if accepted_gate_singleton:
        gate_up_value = _expect_unary(
            graph,
            gate_up_bf16.name,
            opcode="reshape",
            result_type=f"tensor<{compile_rows}x1x768xbf16>",
        )
        gate_rank3 = _expect_slice(
            graph,
            gate_up_value.name,
            ((0, compile_rows), (0, 1), (0, 384)),
            f"tensor<{compile_rows}x1x384xbf16>",
        )
        up_rank3 = _expect_slice(
            graph,
            gate_up_value.name,
            ((0, compile_rows), (0, 1), (384, 768)),
            f"tensor<{compile_rows}x1x384xbf16>",
        )
        gate = _expect_unary(
            graph,
            gate_rank3.name,
            opcode="reshape",
            result_type=f"{row_type}384xbf16>",
        )
        up = _expect_unary(
            graph,
            up_rank3.name,
            opcode="reshape",
            result_type=f"{row_type}384xbf16>",
        )
    else:
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
        if single_virtual_shard:
            down_bit_root = exact_single_rank_root(
                down_bits_reshape.operands[0],
                singleton_type="tensor<1x384x6144xf8E4M3FN>",
                owner_type="tensor<1x1x384x6144xf8E4M3FN>",
                ui8_type="tensor<1x384x6144xui8>",
            )
        else:
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
                raise _MatchError(
                    f"{down_bit_slice.name}: packed down slice drifted"
                )
            down_bit_root = down_bit_slice.operands[0]
        down_scale_reshape = _expect_node(
            graph,
            down_scales,
            opcode="reshape",
            result_type="tensor<3x6144xf32>",
        )
        if single_virtual_shard:
            down_scale_root = exact_single_rank_root(
                down_scale_reshape.operands[0],
                singleton_type="tensor<1x3x6144xf32>",
                owner_type="tensor<1x1x3x6144xf32>",
            )
        else:
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
    dense_is_m32: bool = False,
    residual_is_m32: bool = False,
    return_bitcast_u16: bool = False,
    split_layer1_rms: bool = False,
    fp32_carry_schedule: bool = False,
) -> tuple[str, str, str]:
    """Match the layer-1 RMSNorm.

    ``fp32_carry_schedule`` matches the accepted-schedule arm with an FP32
    carry: ``dense + attention + combined`` summed in FP32 with no BF16
    rounding, one FP32 ``optimization_barrier`` on the [rows,6144] sum, and
    both the variance reduce and the normalized output consuming that
    barrier.  It returns the FP32 carry ``add`` as the residual so the caller
    can bind the two residual sources.
    """
    if fp32_carry_schedule and split_layer1_rms:
        raise ValueError("fp32 carry schedule excludes the BF16 split arm")
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

    dense_value = (
        dense_output
        if dense_is_m32 or not layer1_only
        else exact_m32_pad(dense_output)
    )
    reduction_dense_value = dense_value
    if split_layer1_rms:
        reduction_dense_value = _expect_unary(
            graph,
            dense_value,
            opcode="optimization_barrier",
            result_type=f"{row_type}bf16>",
        ).name
    dense_f32 = _expect_unary(
        graph,
        reduction_dense_value,
        opcode="convert",
        result_type=f"{row_type}f32>",
    )
    carry_barrier = None
    if fp32_carry_schedule:
        first_add = _only(
            (
                node
                for node in graph.matching_users(
                    dense_f32.name,
                    opcode="add",
                    result_type=f"{row_type}f32>",
                )
                if len(node.operands) == 2 and node.operands[0] == dense_f32.name
            ),
            "FP32 carry dense+attention addition",
        )
        _expect_node(
            graph,
            first_add.operands[1],
            opcode="convert",
            result_type=f"{row_type}f32>",
        )
        carry_add = _only(
            (
                node
                for node in graph.matching_users(
                    first_add.name,
                    opcode="add",
                    result_type=f"{row_type}f32>",
                )
                if len(node.operands) == 2 and node.operands[0] == first_add.name
            ),
            "FP32 carry +combined addition",
        )
        _expect_node(
            graph,
            carry_add.operands[1],
            opcode="convert",
            result_type=f"{row_type}f32>",
        )
        carry_barrier = _expect_unary(
            graph,
            carry_add.name,
            opcode="optimization_barrier",
            result_type=f"{row_type}f32>",
        )
        if graph.users.get(carry_add.name, ()) != [carry_barrier]:
            raise _MatchError("FP32 carry sum escapes its barrier")
        if any(
            node.opcode == "convert" and "bf16" in node.result_type
            for node in graph.users.get(carry_barrier.name, ())
        ):
            raise _MatchError("FP32 carry is rounded to BF16 after the barrier")
    combined = carry_barrier if carry_barrier is not None else _only(
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
    if fp32_carry_schedule:
        residual_value = carry_add.name
        reduction_residual_value = residual_value
    else:
        residual_conversion = _expect_node(
            graph,
            combined.operands[1],
            opcode="convert",
            result_type=f"{row_type}f32>",
        )
        reduction_residual_value = residual_conversion.operands[0]
        residual_value = reduction_residual_value
    if split_layer1_rms:
        reduction_residual_barrier = _expect_node(
            graph,
            reduction_residual_value,
            opcode="optimization_barrier",
            result_type=f"{row_type}bf16>",
        )
        if len(reduction_residual_barrier.operands) != 1:
            raise _MatchError(
                f"{reduction_residual_barrier.name}: split residual barrier "
                "arity drifted"
            )
        residual_value = reduction_residual_barrier.operands[0]
    residual = residual_value
    if layer1_only and not residual_is_m32 and not fp32_carry_schedule:
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
    sanitized_reduce = re.sub(
        r'"(?:\\.|[^"\\])*"', '""', reduce.raw_line
    )
    sanitized_reduce = re.sub(r"/\*.*?\*/", "", sanitized_reduce)
    if re.findall(
        r"\bapplies\s+stablehlo\.([a-z0-9_]+)", sanitized_reduce
    ) != ["add"]:
        raise _MatchError(f"{reduce.name}: RMSNorm reduction combiner drifted")
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
    normalized_sum = combined.name
    if split_layer1_rms:
        dense_output_f32 = _expect_unary(
            graph,
            dense_value,
            opcode="convert",
            result_type=f"{row_type}f32>",
        )
        residual_output_f32 = _expect_unary(
            graph,
            residual_value,
            opcode="convert",
            result_type=f"{row_type}f32>",
        )
        normalized_sum = _expect_binary(
            graph,
            dense_output_f32.name,
            residual_output_f32.name,
            opcode="add",
            result_type=f"{row_type}f32>",
        ).name
        if normalized_sum == combined.name:
            raise _MatchError("split RMSNorm reused the reduction residual sum")
    normalized = _expect_binary(
        graph,
        normalized_sum,
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
    if return_bitcast_u16:
        if not layer1_only:
            raise _MatchError("U16 RMS replay return requires one live M32 row")
        returned_output = _expect_unary(
            graph,
            returned_output,
            opcode="bitcast_convert",
            result_type="tensor<1x6144xui16>",
        ).name
    expected_return = (
        (returned_output,) if layer1_only else (dense_output, returned_output)
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
    if layer1_only and not dense_is_m32 and graph.users.get(dense_output, ()) != [
        graph.node(dense_value)
    ]:
        raise _MatchError("layer-1-only probe externalizes the dense update")
    return residual, norm_weight_seed.operands[0], returned_output


def validate_captured_dense_rms_stablehlo(
    stablehlo: str,
    *,
    split_layer1_rms: bool,
) -> dict[str, object]:
    """Prove a captured-partial StrategyND reduction and layer-1 RMSNorm.

    This is the pre-fusion half of the bounded Gate-D replay.  It deliberately
    contains no dense contractions: the input is the sealed set of 32 real
    BF16 down partials.  The contract reuses the exact DB533 tree and exact
    layer-1 RMS matcher used by the full dense probe, then additionally binds
    the four replay inputs and the outer ``sdy.manual_computation`` result.
    """

    if not isinstance(split_layer1_rms, bool):
        raise TypeError("captured RMS split flag must be boolean")
    graphs, violations = _parse_graphs(stablehlo)
    try:
        graph = _only(
            (item for item in graphs if item.name == "main"),
            "captured RMS main graph",
        )
        helpers = {item.name: item for item in graphs if item is not graph}
        gather = _only(
            (
                node
                for node in graph.nodes.values()
                if node.opcode == "all_gather"
                and node.result_type == "tensor<4x8x1x6144xbf16>"
                and node.tensor_types[-2:]
                == (
                    "tensor<1x8x1x6144xbf16>",
                    "tensor<4x8x1x6144xbf16>",
                )
            ),
            "captured RMS LP4 all-gather",
        )
        sanitized_gather = re.sub(
            r'"(?:\\.|[^"\\])*"', '""', gather.raw_line
        )
        sanitized_gather = re.sub(r"/\*.*?\*/", "", sanitized_gather)
        normalized_gather = re.sub(r"\s+", "", sanitized_gather)
        groups = re.findall(
            r"\breplica_groups=dense<\[\[[0-9,]+\]\]>:tensor<1x4xi64>",
            normalized_gather,
        )
        if (
            gather.all_gather_dimension != 0
            or not gather.use_global_device_ids
            or groups
            != [
                "replica_groups=dense<[[0,1,2,3]]>:tensor<1x4xi64>"
            ]
            or len(gather.operands) != 1
        ):
            raise _MatchError(f"{gather.name}: captured RMS gather drifted")

        allowed_input_ops = {
            "broadcast_in_dim",
            "copy",
            "optimization_barrier",
            "reshape",
            "slice",
        }
        pending = [gather.operands[0]]
        seen: set[str] = set()
        terminals: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name not in graph.nodes:
                terminals.add(name)
                continue
            node = graph.node(name)
            if node.opcode not in allowed_input_ops or len(node.operands) != 1:
                raise _MatchError(
                    f"{node.name}: captured partial input arithmetic drifted"
                )
            input_type, output_type = node.tensor_types[-2:]
            input_dims = tuple(
                int(value)
                for value in re.search(r"tensor<([0-9x]+)xbf16>", input_type)
                .group(1)
                .split("x")
            )
            output_dims = tuple(
                int(value)
                for value in re.search(r"tensor<([0-9x]+)xbf16>", output_type)
                .group(1)
                .split("x")
            )
            input_elements = 1
            output_elements = 1
            for dimension in input_dims:
                input_elements *= dimension
            for dimension in output_dims:
                output_elements *= dimension
            if input_elements != output_elements:
                raise _MatchError(
                    f"{node.name}: captured partial layout changed element count"
                )
            if node.opcode == "slice" and (
                node.slice_ranges is None
                or tuple(stop - start for start, stop in node.slice_ranges)
                != output_dims
            ):
                raise _MatchError(
                    f"{node.name}: captured partial slice drifted"
                )
            pending.extend(node.operands)
        if terminals != {"%arg4"}:
            raise _MatchError(
                f"captured partial gather source drifted: {sorted(terminals)}"
            )

        dense_output = _match_reduction_tree(graph, gather)
        residual, norm_weight, returned_output = _match_rmsnorm(
            graph,
            helpers,
            dense_output,
            layer1_only=True,
            residual_is_m32=True,
            fp32_carry_schedule=split_layer1_rms,
        )
        if split_layer1_rms:
            # FP32 carry: residual is the `(dense + attention) + combined`
            # add; bind attention (%arg5) and combined (%arg6) through their
            # converts and exact M32 pads.  No BF16 round exists on this path.
            carry_add = _expect_node(
                graph,
                residual,
                opcode="add",
                result_type="tensor<32x6144xf32>",
            )
            first_add = _expect_node(
                graph,
                carry_add.operands[0],
                opcode="add",
                result_type="tensor<32x6144xf32>",
            )
            residual_operands = (first_add.operands[1], carry_add.operands[1])
        else:
            residual_round = _expect_node(
                graph,
                residual,
                opcode="convert",
                result_type="tensor<32x6144xbf16>",
            )
            if len(residual_round.operands) != 1:
                raise _MatchError("captured carried-residual round arity drifted")
            residual_add = _expect_node(
                graph,
                residual_round.operands[0],
                opcode="add",
                result_type="tensor<32x6144xf32>",
            )
            if len(residual_add.operands) != 2:
                raise _MatchError("captured carried-residual add arity drifted")
            residual_operands = tuple(residual_add.operands)
        residual_sources: list[str] = []
        for operand in residual_operands:
            conversion = _expect_node(
                graph,
                operand,
                opcode="convert",
                result_type="tensor<32x6144xf32>",
            )
            if len(conversion.operands) != 1:
                raise _MatchError(
                    "captured carried-residual conversion arity drifted"
                )
            pad = _expect_node(
                graph,
                conversion.operands[0],
                opcode="call",
                result_type="tensor<32x6144xbf16>",
            )
            residual_sources.append(
                _validate_m32_input_pad(graph, helpers, pad.name)
            )
        if tuple(residual_sources) != ("%arg5", "%arg6") or norm_weight != "%arg7":
            raise _MatchError(
                "captured RMS residual-source/norm identity drifted: "
                f"sources={residual_sources} norm={norm_weight}"
            )

        manual_matches = list(
            re.finditer(
                r"(?m)^\s*(%[A-Za-z0-9_.-]+)\s*=\s*"
                r"sdy\.manual_computation\(([^)]*)\)",
                stablehlo,
            )
        )
        if len(manual_matches) != 1:
            raise _MatchError("captured RMS manual computation is not unique")
        manual_result = manual_matches[0].group(1)
        manual_operands = tuple(
            re.findall(r"%[A-Za-z0-9_.$#-]+", manual_matches[0].group(2))
        )
        if manual_operands != ("%arg0", "%arg1", "%arg2", "%arg3"):
            raise _MatchError(
                f"captured RMS manual operands drifted: {manual_operands}"
            )
        manual_line = next(
            line
            for line in stablehlo.splitlines()
            if f"{manual_result} = sdy.manual_computation" in line
        )
        if re.findall(
            r"(%arg[0-9]+):\s*(tensor<[^>]+>)", manual_line
        ) != [
            ("%arg4", "tensor<1x8x1x6144xbf16>"),
            ("%arg5", "tensor<1x6144xbf16>"),
            ("%arg6", "tensor<1x6144xbf16>"),
            ("%arg7", "tensor<6144xbf16>"),
        ]:
            raise _MatchError("captured RMS manual block arguments drifted")
        _only(
            (
                node
                for node in graph.nodes.values()
                if node.opcode == "return"
                and node.raw_line.startswith("return ")
                and node.operands == (manual_result,)
            ),
            "captured RMS outer manual return",
        )
        if returned_output not in graph.nodes:
            raise _MatchError("captured RMS exact result disappeared")
    except (AttributeError, _MatchError, ValueError) as error:
        violations.append(str(error))

    collective_counts = {
        opcode: sum(
            node.opcode == opcode
            for item in graphs
            for node in item.nodes.values()
        )
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
        violations.append(
            f"captured RMS collective contract drifted: {collective_counts}"
        )
    if "stablehlo.convolution" in stablehlo:
        violations.append("captured RMS replay contains a convolution")
    if "xla_python_cpu_callback" in stablehlo or "host_callback" in stablehlo:
        violations.append("captured RMS replay contains a host callback")
    return {
        "collective_counts": collective_counts,
        "exact_result_binding": not violations,
        "live_rows": 1,
        "split_layer1_rms": split_layer1_rms,
        "passed": not violations,
        "violations": violations,
    }


def validate_strategy_nd_dense_rms_stablehlo(
    stablehlo: str,
) -> dict[str, object]:
    """Prove the global StrategyND result is consumed by exact layer-1 RMSNorm."""

    graphs, violations = _parse_graphs(stablehlo)
    residual_source_mode: str | None = None
    try:
        graph = _only(
            (item for item in graphs if item.name == "main"),
            "StrategyND RMS main graph",
        )
        helpers = {item.name: item for item in graphs if item is not graph}
        manual_matches = list(
            re.finditer(
                r"(?m)^\s*(%[A-Za-z0-9_.-]+)\s*=\s*"
                r"sdy\.manual_computation\(([^)]*)\)",
                stablehlo,
            )
        )
        if len(manual_matches) != 1:
            raise _MatchError("StrategyND RMS manual computation is not unique")
        manual_result = manual_matches[0].group(1)
        outer_operands = tuple(
            re.findall(r"%[A-Za-z0-9_.$#-]+", manual_matches[0].group(2))
        )
        manual_line = next(
            line
            for line in stablehlo.splitlines()
            if f"{manual_result} = sdy.manual_computation" in line
        )
        block_arguments = tuple(
            re.findall(r"(%arg[0-9]+):\s*(tensor<[^>]+>)", manual_line)
        )
        direct_arguments = (
            ("%arg3", "tensor<1x32x6144xui16>"),
            ("%arg4", "tensor<1x6144xbf16>"),
            ("%arg5", "tensor<6144xbf16>"),
        )
        hybrid_arguments = (
            ("%arg4", "tensor<1x32x6144xui16>"),
            ("%arg5", "tensor<1x6144xbf16>"),
            ("%arg6", "tensor<1x6144xbf16>"),
            ("%arg7", "tensor<6144xbf16>"),
        )
        if outer_operands == ("%arg0", "%arg1", "%arg2") and (
            block_arguments == direct_arguments
        ):
            residual_source_mode = "direct_post_attention_residual"
            physical_input = "%arg3"
            residual_inputs = ("%arg4",)
            expected_norm = "%arg5"
        elif outer_operands == ("%arg0", "%arg1", "%arg2", "%arg3") and (
            block_arguments == hybrid_arguments
        ):
            residual_source_mode = "hybrid_attention_plus_combined_control"
            physical_input = "%arg4"
            residual_inputs = ("%arg5", "%arg6")
            expected_norm = "%arg7"
        else:
            raise _MatchError("StrategyND RMS outer/block arguments drifted")
        collective = _only(
            (node for node in graph.nodes.values() if node.opcode == "all_reduce"),
            "StrategyND RMS all-reduce",
        )
        reshape = _only(
            (
                node
                for node in graph.nodes.values()
                if node.opcode == "reshape"
                and node.operands == (physical_input,)
                and node.tensor_types[-2:]
                == (
                    "tensor<1x32x6144xui16>",
                    "tensor<32x6144xui16>",
                )
            ),
            "StrategyND RMS physical input reshape",
        )
        payload = _expect_unary(
            graph,
            reshape.name,
            opcode="bitcast_convert",
            result_type="tensor<32x6144xbf16>",
        )
        if (
            payload.tensor_types[-2:]
            != ("tensor<32x6144xui16>", "tensor<32x6144xbf16>")
            or collective.operands != (payload.name,)
        ):
            raise _MatchError("StrategyND RMS collective input lineage drifted")

        without_comments = re.sub(r"/\*.*?\*/", "", stablehlo, flags=re.S)
        expected_members = ", ".join(str(value) for value in range(32))
        collective_head = re.compile(
            rf'^\s*{re.escape(collective.name)}\s*=\s*'
            rf'"stablehlo\.all_reduce"\({re.escape(payload.name)}\)\s*'
            rf'<\{{channel_handle\s*=\s*#stablehlo\.channel_handle'
            rf'<handle\s*=\s*1,\s*type\s*=\s*1>,\s*'
            rf'replica_groups\s*=\s*dense<\[\[{expected_members}\]\]>'
            rf'\s*:\s*tensor<1x32xi64>,\s*use_global_device_ids\}}>\s*\(\{{\s*$',
            re.M,
        )
        collective_heads = tuple(collective_head.finditer(without_comments))
        if len(collective_heads) != 1:
            raise _MatchError("StrategyND RMS replica group/channel drifted")
        reducer_region = re.compile(
            r'\A\s*'
            r'\s*\^bb0\((%[A-Za-z0-9_.$#-]+):\s*tensor<bf16>,\s*'
            r'(%[A-Za-z0-9_.$#-]+):\s*tensor<bf16>\):\s*\n'
            r'\s*(%[A-Za-z0-9_.$#-]+)\s*=\s*stablehlo\.add\s+\1,\s*\2\s*'
            r':\s*tensor<bf16>(?:\s+loc\([^\n]+\))?\s*\n'
            r'\s*stablehlo\.return\s+\3\s*:\s*tensor<bf16>'
            r'(?:\s+loc\([^\n]+\))?\s*\n'
            r'\s*\}\)\s*:\s*\(tensor<32x6144xbf16>\)\s*->\s*'
            r'tensor<32x6144xbf16>(?:\s+loc\([^\n]+\))?\s*(?:\n|\Z)',
        )
        # Match only the selected collective's immediately following region.
        # Quoted location text is removed from that tail so neither metadata
        # nor a later reducer can satisfy the live arithmetic contract.
        reducer_tail = re.sub(
            r'"(?:\\.|[^"\\])*"', '""',
            without_comments[collective_heads[0].end():],
        )
        if reducer_region.match(reducer_tail) is None:
            raise _MatchError("StrategyND RMS BF16 reducer/result drifted")

        residual, norm_weight, returned_output = _match_rmsnorm(
            graph,
            helpers,
            collective.name,
            layer1_only=True,
            dense_is_m32=True,
            residual_is_m32=True,
            return_bitcast_u16=True,
            split_layer1_rms=False,
        )
        residual_sources: list[str]
        if residual_source_mode == "direct_post_attention_residual":
            pad = _expect_node(
                graph,
                residual,
                opcode="call",
                result_type="tensor<32x6144xbf16>",
            )
            residual_sources = [
                _validate_m32_input_pad(graph, helpers, pad.name)
            ]
        else:
            residual_round = _expect_node(
                graph,
                residual,
                opcode="convert",
                result_type="tensor<32x6144xbf16>",
            )
            residual_add = _expect_node(
                graph,
                residual_round.operands[0],
                opcode="add",
                result_type="tensor<32x6144xf32>",
            )
            residual_sources = []
            for operand in residual_add.operands:
                conversion = _expect_node(
                    graph,
                    operand,
                    opcode="convert",
                    result_type="tensor<32x6144xf32>",
                )
                pad = _expect_node(
                    graph,
                    conversion.operands[0],
                    opcode="call",
                    result_type="tensor<32x6144xbf16>",
                )
                residual_sources.append(
                    _validate_m32_input_pad(graph, helpers, pad.name)
                )
        if tuple(residual_sources) != residual_inputs or norm_weight != expected_norm:
            raise _MatchError(
                "StrategyND RMS residual/norm source identity drifted: "
                f"sources={residual_sources} norm={norm_weight}"
            )
        _only(
            (
                node
                for node in graph.nodes.values()
                if node.opcode == "return"
                and node.raw_line.startswith("return ")
                and node.operands == (manual_result,)
            ),
            "StrategyND RMS outer return",
        )
        if returned_output not in graph.nodes:
            raise _MatchError("StrategyND RMS exact result disappeared")
    except (AttributeError, _MatchError, StopIteration, ValueError) as error:
        violations.append(str(error))

    collective_counts = {
        opcode: sum(
            node.opcode == opcode
            for item in graphs
            for node in item.nodes.values()
        )
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
        "all_gather": 0,
        "all_reduce": 1,
        "all_to_all": 0,
        "collective_broadcast": 0,
        "collective_permute": 0,
        "reduce_scatter": 0,
    }:
        violations.append(
            f"StrategyND RMS collective contract drifted: {collective_counts}"
        )
    if "stablehlo.convolution" in stablehlo:
        violations.append("StrategyND RMS replay contains a convolution")
    if "xla_python_cpu_callback" in stablehlo or "host_callback" in stablehlo:
        violations.append("StrategyND RMS replay contains a host callback")
    return {
        "collective_counts": collective_counts,
        "exact_result_binding": not violations,
        "live_rows": 1,
        "passed": not violations,
        "residual_source_mode": (
            residual_source_mode if not violations else None
        ),
        "violations": violations,
    }


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


def validate_isolated_dense_partial_stablehlo(
    stablehlo: str,
    *,
    accepted_gate_singleton: bool = False,
    accepted_gate_dequant_fusion: bool = False,
) -> dict[str, object]:
    """Prove one exact M32 contraction per LP4 chip with no collective.

    The same compiled program is executed eight times with separately sealed
    virtual-rank weight slices.  This contract therefore proves exactly one
    pre-dense RMSNorm -> gate/SwiGLU/down chain and its exact live row; the
    artifact manifest binds each of the eight runtime input batches.
    """

    if accepted_gate_dequant_fusion and not accepted_gate_singleton:
        raise ValueError(
            "accepted gate dequant fusion requires the singleton contract"
        )
    parsed, dependency_errors = _expand_dependency_barriers(stablehlo)
    graphs, parse_errors = _parse_graphs(parsed)
    violations = [*dependency_errors, *parse_errors]
    matched = False
    runtime_u8_bitcasts = 0
    try:
        graph = _only(
            (item for item in graphs if item.name == "main"),
            "isolated dense main graph",
        )
        helpers = {item.name: item for item in graphs if item is not graph}
        gate_up = [
            node
            for node in graph.nodes.values()
            if node.opcode == "convolution"
            and node.result_type == "tensor<32x768xf32>"
        ]
        down = [
            node
            for node in graph.nodes.values()
            if node.opcode == "convolution"
            and node.result_type == "tensor<32x6144xf32>"
        ]
        if len(gate_up) != 1 or len(down) != 1:
            raise _MatchError(
                "isolated dense convolution count drifted: "
                f"gate_up={len(gate_up)} down={len(down)}"
            )
        shard, roots, down_result, dependency, dependency_output = (
            _match_one_shard(
                graph,
                gate_up[0],
                compile_rows=32,
                final_dense_layout=True,
                single_virtual_shard=True,
                accepted_gate_singleton=accepted_gate_singleton,
                accepted_gate_dequant_fusion=(
                    accepted_gate_dequant_fusion
                ),
            )
        )
        if shard != 0 or dependency is not None or dependency_output is not None:
            raise _MatchError("isolated dense virtual-rank identity drifted")
        stacked = _expect_broadcast(
            graph,
            down_result,
            dimensions=(1, 2),
            result_type="tensor<1x32x6144xbf16>",
        )
        anchored = _expect_unary(
            graph,
            stacked.name,
            opcode="optimization_barrier",
            result_type="tensor<1x32x6144xbf16>",
        )
        live = _expect_slice(
            graph,
            anchored.name,
            ((0, 1), (0, 1), (0, 6144)),
            "tensor<1x1x6144xbf16>",
        )
        inner_return = _only(
            (
                node
                for node in graph.nodes.values()
                if node.opcode == "return"
                and node.raw_line.startswith("sdy.return ")
            ),
            "isolated dense manual return",
        )
        if inner_return.operands[0] != live.name or len(inner_return.operands) != 2:
            raise _MatchError("isolated dense live partial return drifted")
        predense_sources, predense_norm = _match_predense_rmsnorm(
            graph,
            helpers,
            roots[0],
            inner_return.operands[1],
        )
        if (
            predense_sources != ("%arg7", "%arg8")
            or predense_norm != "%arg9"
            or roots[1:] != ("%arg10", "%arg11", "%arg12", "%arg13")
        ):
            raise _MatchError("isolated dense manual source ownership drifted")
        gate_users = {
            node.name
            for node in graph.users.get(roots[0], ())
            if node.opcode == "convolution"
        }
        if gate_users != {gate_up[0].name}:
            raise _MatchError(
                "isolated pre-dense RMSNorm does not feed exactly one gate"
            )
        layout_constraints = [
            node
            for node in graph.nodes.values()
            if node.opcode == "custom_call"
            and "@LayoutConstraint(" in node.raw_line
        ]
        expected_layout_constraints = (
            0 if accepted_gate_dequant_fusion else 1
        )
        if len(layout_constraints) != expected_layout_constraints:
            raise _MatchError("isolated dense layout-constraint count drifted")
        manual_matches = list(
            re.finditer(
                r"(?m)^\s*(%[A-Za-z0-9_.-]+):2\s*=\s*"
                r"sdy\.manual_computation\(([^)]*)\)",
                stablehlo,
            )
        )
        if len(manual_matches) != 1:
            raise _MatchError("isolated dense manual computation is not unique")
        manual_result = manual_matches[0].group(1)
        if [value.strip() for value in manual_matches[0].group(2).split(",")] != [
            f"%arg{index}" for index in range(7)
        ]:
            raise _MatchError("isolated dense outer manual operands drifted")
        manual_line = next(
            line
            for line in stablehlo.splitlines()
            if f"{manual_result}:2 = sdy.manual_computation" in line
        )
        block_arguments = re.findall(
            r"(%arg(?:7|8|9|10|11|12|13)):\s*(tensor<[^>]+>)",
            manual_line,
        )
        allowed_block_arguments = (
            [
                ("%arg7", "tensor<1x6144xbf16>"),
                ("%arg8", "tensor<1x6144xbf16>"),
                ("%arg9", "tensor<6144xbf16>"),
                ("%arg10", "tensor<1x1x6144x768xf8E4M3FN>"),
                ("%arg11", "tensor<1x1x48x768xf32>"),
                ("%arg12", "tensor<1x1x384x6144xf8E4M3FN>"),
                ("%arg13", "tensor<1x1x3x6144xf32>"),
            ],
            [
                ("%arg7", "tensor<1x6144xbf16>"),
                ("%arg8", "tensor<1x6144xbf16>"),
                ("%arg9", "tensor<6144xbf16>"),
                ("%arg10", "tensor<1x1x6144x768xui8>"),
                ("%arg11", "tensor<1x1x48x768xf32>"),
                ("%arg12", "tensor<1x1x384x6144xui8>"),
                ("%arg13", "tensor<1x1x3x6144xf32>"),
            ],
        )
        if block_arguments not in allowed_block_arguments:
            raise _MatchError("isolated dense manual block arguments drifted")
        runtime_u8_bitcasts = stablehlo.count("stablehlo.bitcast_convert")
        expected_bitcasts = 2 if block_arguments == allowed_block_arguments[1] else 0
        if runtime_u8_bitcasts != expected_bitcasts:
            raise _MatchError("isolated dense runtime U8 bitcast count drifted")
        outer_return = _only(
            (
                node
                for node in graph.nodes.values()
                if node.opcode == "return"
                and node.raw_line.startswith("return ")
            ),
            "isolated dense outer return",
        )
        if outer_return.operands != (
            f"{manual_result}#0",
            f"{manual_result}#1",
        ):
            raise _MatchError("isolated dense outer results drifted")
        if len(graphs) != 2 or set(helpers) != {"_pad"}:
            raise _MatchError("isolated dense helper set drifted")
        matched = True
    except (AttributeError, _MatchError, ValueError) as error:
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
    if any(collective_counts.values()):
        violations.append(
            f"isolated dense collective contract drifted: {collective_counts}"
        )
    if "xla_python_cpu_callback" in stablehlo or "host_callback" in stablehlo:
        violations.append("isolated dense module contains a host callback")
    return {
        "accepted_gate_dequant_fusion": accepted_gate_dequant_fusion,
        "accepted_gate_singleton": accepted_gate_singleton,
        "collective_counts": collective_counts,
        "convolution_count": stablehlo.count("stablehlo.convolution"),
        "exact_result_binding": matched,
        "gate_up_layout_constraint_count": stablehlo.count("@LayoutConstraint("),
        "matched_virtual_shards": [0] if matched else [],
        "runtime_u8_bitcast_count": runtime_u8_bitcasts,
        "virtual_contractions_per_chip": 1,
        "passed": not violations,
        "violations": violations,
    }


def validate_dense_convolution_stablehlo(
    stablehlo: str,
    *,
    compile_rows: int = 1,
    layer1_only: bool = False,
    final_dense_layout: bool = False,
    dense_envelope: bool = False,
    split_layer1_rms: bool = False,
    partials_only: bool = False,
) -> dict[str, object]:
    """Validate all eight exact dense chains and the exact StrategyND tree."""

    violations: list[str] = []
    if compile_rows not in (1, 32):
        raise ValueError("dense StableHLO compile rows must be 1 or 32")
    if final_dense_layout and (
        compile_rows != 32 or not (layer1_only or partials_only)
    ):
        raise ValueError(
            "final-layout dense StableHLO proof requires M32 layer1 or partial capture"
        )
    if dense_envelope and not final_dense_layout:
        raise ValueError(
            "dense-envelope StableHLO proof requires final-layout mode"
        )
    if split_layer1_rms and not (
        dense_envelope and layer1_only and compile_rows == 32
    ):
        raise ValueError(
            "split layer-1 RMS proof requires the M32 dense envelope"
        )
    if partials_only and not (
        dense_envelope
        and final_dense_layout
        and compile_rows == 32
        and not split_layer1_rms
    ):
        raise ValueError(
            "dense partial capture requires the unsplit M32 final-layout envelope"
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
        sanitized_gather = re.sub(
            r'"(?:\\.|[^"\\])*"', '""', gather.raw_line
        )
        sanitized_gather = re.sub(r"/\*.*?\*/", "", sanitized_gather)
        normalized_gather = re.sub(r"\s+", "", sanitized_gather)
        replica_group_attributes = re.findall(
            r"\breplica_groups=dense<\[\[[0-9,]+\]\]>:tensor<1x4xi64>",
            normalized_gather,
        )
        if (
            gather.all_gather_dimension != 0
            or not gather.use_global_device_ids
            or replica_group_attributes
            != [
                "replica_groups=dense<[[0,1,2,3]]>:tensor<1x4xi64>"
            ]
        ):
            raise _MatchError(f"{gather.name}: dense gather group drifted")
        helpers = {item.name: item for item in graphs if item is not graph}
        manual_matches = list(
            re.finditer(
                r"(?m)^\s*(%[A-Za-z0-9_.-]+)(?::([0-9]+))?\s*=\s*"
                r"sdy\.manual_computation\(([^)]*)\)",
                stablehlo,
            )
        )
        if len(manual_matches) != 1:
            raise _MatchError("dense probe manual computation is not unique")
        manual_match = manual_matches[0]
        manual_result = manual_match.group(1)
        if partials_only:
            returned = _only(
                (
                    node
                    for node in graph.nodes.values()
                    if node.opcode == "return"
                    and node.raw_line.startswith("sdy.return ")
                ),
                "dense partial-capture return",
            )
            if len(returned.operands) != 2 or returned.operands[0] != gather.name:
                raise _MatchError(
                    "dense partial capture does not return the exact gathered partials"
                )
            outer_return = _only(
                (
                    node
                    for node in graph.nodes.values()
                    if node.opcode == "return"
                    and node.raw_line.startswith("return ")
                ),
                "dense partial-capture outer return",
            )
            if (
                manual_match.group(2) != "2"
                or outer_return.operands
                != (f"{manual_result}#0", f"{manual_result}#1")
            ):
                raise _MatchError(
                    "dense partial capture outer manual results drifted"
                )
            residual = returned.operands[1]
            norm_weight = None
        else:
            dense_output = _match_reduction_tree(graph, gather)
            residual, norm_weight, _layer1 = _match_rmsnorm(
                graph,
                helpers,
                dense_output,
                layer1_only=layer1_only,
                residual_is_m32=dense_envelope,
                split_layer1_rms=split_layer1_rms,
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
        if dense_envelope and not partials_only:
            assert norm_weight is not None
            manual_arguments.append(norm_weight)
        elif not dense_envelope:
            assert norm_weight is not None
            manual_arguments.extend((residual, norm_weight))
        outer_arguments = (
            [value.strip() for value in manual_match.group(3).split(",")]
        )
        expected_manual_arguments = (
            [f"%arg{index}" for index in range(7, 14)]
            if partials_only
            else [
                "%arg8", "%arg9", "%arg10", "%arg11", "%arg12",
                "%arg13", "%arg14", "%arg15",
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
                7
                if partials_only
                else 8
                if dense_envelope
                else 7
                if final_dense_layout
                else 9
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
        "partials_only": partials_only,
        "result_mode": (
            "partials_only"
            if partials_only
            else "layer1_only"
            if layer1_only
            else "dense_and_layer1"
        ),
        "split_layer1_rms": split_layer1_rms,
        "passed": not violations,
        "violations": violations,
    }


def validate_dense_final_layout_decoder_stablehlo(
    stablehlo: str,
    *,
    dense_layers: int,
    replica_groups: Sequence[Sequence[int]],
) -> dict[str, object]:
    """Prove every live one-row dense chain in the complete decoder.

    The layer-0 probe validator above deliberately accepts one isolated
    manual-computation body.  A complete decoder contains three independent
    dense groups in the same body, so cardinality alone is unsafe: dead or
    cross-wired convolutions could satisfy it.  This contract reuses the exact
    per-shard matcher, groups its eight shards by their five physical inputs,
    binds each group to its StrategyND tree, and requires every tree result to
    reach the function return.
    """

    if not isinstance(dense_layers, int) or isinstance(dense_layers, bool):
        raise ValueError("dense decoder layer count must be an integer")
    if dense_layers < 0:
        raise ValueError("dense decoder layer count must be non-negative")
    parsed, dependency_errors = _expand_dependency_barriers(stablehlo)
    graphs, parse_errors = _parse_graphs(parsed)
    violations = [*dependency_errors, *parse_errors]
    canonical_groups = tuple(
        tuple(int(rank) for rank in group) for group in replica_groups
    )
    if not canonical_groups or any(
        len(group) != 4 for group in canonical_groups
    ):
        raise ValueError("dense decoder replica groups must be non-empty LP4 groups")
    flattened_groups = tuple(
        rank for group in canonical_groups for rank in group
    )
    if len(set(flattened_groups)) != len(flattened_groups):
        raise ValueError("dense decoder replica groups must be disjoint")
    normalized_replica_groups = (
        "replica_groups=dense<["
        + ",".join(
            "[" + ",".join(str(rank) for rank in group) + "]"
            for group in canonical_groups
        )
        + f"]>:tensor<{len(canonical_groups)}x4xi64>"
    )
    matched_groups = 0
    matched_shards = 0
    live_results = 0
    runtime_u8_bitcasts = 0
    layout_constraints = 0
    dead_row_nodes = 0
    try:
        candidates = [
            (graph, node)
            for graph in graphs
            for node in graph.nodes.values()
            if node.opcode == "convolution"
            and node.result_type == "tensor<1x768xf32>"
        ]
        dense_graph_names = {graph.name for graph, _node in candidates}
        dead_row_nodes = sum(
            node.result_type
            in {
                "tensor<32x6144xbf16>",
                "tensor<32x768xf32>",
                "tensor<32x384xbf16>",
            }
            for graph in graphs
            if graph.name in dense_graph_names
            for node in graph.nodes.values()
        )
        if dead_row_nodes:
            raise _MatchError(
                "dense decoder arithmetic graph contains M32 token rows"
            )
        if len(candidates) != 8 * dense_layers:
            raise _MatchError(
                "dense decoder gate/up convolution count drifted: "
                f"expected={8 * dense_layers} found={len(candidates)}"
            )
        rows_by_group: dict[
            tuple[str, tuple[str, ...]],
            dict[int, tuple[str, str | None, str | None]],
        ] = {}
        graph_by_name = {graph.name: graph for graph in graphs}
        for graph, convolution in candidates:
            shard, roots, result, dependency, dependency_output = (
                _match_one_shard(
                    graph,
                    convolution,
                    compile_rows=1,
                    final_dense_layout=True,
                )
            )
            group = rows_by_group.setdefault((graph.name, roots), {})
            if shard in group:
                raise _MatchError(
                    f"duplicate dense decoder virtual shard {shard}"
                )
            group[shard] = (result, dependency, dependency_output)
            matched_shards += 1

        if len(rows_by_group) != dense_layers:
            raise _MatchError(
                "dense decoder physical-input group count drifted: "
                f"expected={dense_layers} found={len(rows_by_group)}"
            )

        dense_outputs: set[tuple[str, str]] = set()
        external_runtime_sources: set[str] = set()

        def exact_owner_unit_source(
            graph: _StableGraph,
            value: str,
            *,
            local_type: str,
        ) -> str:
            node = graph.nodes.get(value)
            if node is None:
                return value
            owner_type = local_type.replace("tensor<", "tensor<1x", 1)
            if (
                node.opcode != "reshape"
                or len(node.operands) != 1
                or node.tensor_types[-2:] != (owner_type, local_type)
                or node.operands[0] in graph.nodes
            ):
                raise _MatchError(
                    f"{value}: dense owner-unit removal drifted"
                )
            return node.operands[0]

        for (graph_name, roots), rows in rows_by_group.items():
            graph = graph_by_name[graph_name]
            if set(rows) != set(range(8)):
                raise _MatchError(
                    "dense decoder virtual-shard set drifted: "
                    f"{sorted(rows)}"
                )
            if rows[0][1] is not None or any(
                rows[shard][1] != rows[shard - 1][0]
                for shard in range(1, 8)
            ):
                raise _MatchError(
                    "dense decoder virtual shards lost rank ordering"
                )

            # The runtime loader exposes exact FP8 storage bytes as U8.  Both
            # packed bit roots must cross one shape-preserving bitcast before
            # any shard slice; numeric conversion is forbidden.
            for root, ui8_type, fp8_type in (
                (
                    roots[1],
                    "tensor<8x6144x768xui8>",
                    "tensor<8x6144x768xf8E4M3FN>",
                ),
                (
                    roots[3],
                    "tensor<8x384x6144xui8>",
                    "tensor<8x384x6144xf8E4M3FN>",
                ),
            ):
                bitcast = _expect_node(
                    graph, root, opcode="bitcast_convert", result_type=fp8_type
                )
                if (
                    len(bitcast.operands) != 1
                    or bitcast.tensor_types[-2:] != (ui8_type, fp8_type)
                ):
                    raise _MatchError(
                        f"{root}: dense runtime U8/FP8 bitcast drifted"
                    )
                external_runtime_sources.add(
                    exact_owner_unit_source(
                        graph,
                        bitcast.operands[0],
                        local_type=ui8_type,
                    )
                )
                runtime_u8_bitcasts += 1
            for scale_root, scale_type in (
                (roots[2], "tensor<8x48x768xf32>"),
                (roots[4], "tensor<8x3x6144xf32>"),
            ):
                external_runtime_sources.add(
                    exact_owner_unit_source(
                        graph, scale_root, local_type=scale_type
                    )
                )

            broadcasts = tuple(
                _expect_broadcast(
                    graph,
                    (
                        rows[shard + 1][2]
                        if shard < 7
                        else rows[shard][0]
                    ),
                    dimensions=(1, 2),
                    result_type="tensor<1x1x6144xbf16>",
                ).name
                for shard in range(8)
            )
            stack = _expect_concat(
                graph,
                broadcasts,
                dimension=0,
                result_type="tensor<8x1x6144xbf16>",
            )
            gathered_input = _expect_broadcast(
                graph,
                stack.name,
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
                "dense decoder StrategyND all-gather",
            )
            normalized_gather = re.sub(r"\s+", "", gather.raw_line)
            if (
                gather.all_gather_dimension != 0
                or not gather.use_global_device_ids
                or normalized_replica_groups not in normalized_gather
            ):
                raise _MatchError(
                    f"{gather.name}: dense decoder gather group drifted"
                )
            dense_output = _match_reduction_tree(graph, gather)
            output_key = (graph.name, dense_output)
            if output_key in dense_outputs:
                raise _MatchError("dense decoder groups share one tree result")
            dense_outputs.add(output_key)

            pending = [dense_output]
            reached = {dense_output}
            is_live = False
            while pending:
                current = pending.pop()
                for user in graph.users.get(current, ()):
                    if user.opcode == "return":
                        is_live = True
                        pending.clear()
                        break
                    if user.name not in reached:
                        reached.add(user.name)
                        pending.append(user.name)
            if not is_live:
                raise _MatchError(
                    f"{dense_output}: dense decoder result is not live"
                )
            live_results += 1
            matched_groups += 1

        if len(external_runtime_sources) != 4 * dense_layers:
            raise _MatchError(
                "dense decoder packed bits/scales source bijection drifted"
            )
        layout_constraints = sum(
            1
            for graph in graphs
            for node in graph.nodes.values()
            if node.opcode == "custom_call"
            and "@LayoutConstraint(" in node.raw_line
            and node.result_type == "tensor<6144x768xbf16>"
        )
        if layout_constraints != 8 * dense_layers:
            raise _MatchError(
                "dense decoder layout-constraint count drifted: "
                f"expected={8 * dense_layers} found={layout_constraints}"
            )
    except _MatchError as error:
        violations.append(str(error))

    return {
        "dead_row_node_count": dead_row_nodes,
        "dense_layer_group_count": matched_groups,
        "exact_live_result_count": live_results,
        "exact_runtime_u8_bitcast_count": runtime_u8_bitcasts,
        "gate_up_layout_constraint_count": layout_constraints,
        "matched_virtual_shard_count": matched_shards,
        "passed": not violations,
        "violations": violations,
    }
