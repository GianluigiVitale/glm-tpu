"""Exact StableHLO contract for DB533's StrategyND attention projection.

The optimized-HLO contract pins physical collectives and Pallas calls.  This
module pins the numerical program before backend fusion: ordered contiguous
K512 contractions, DB533's physical row permutation, and every BF16-rounded
``y -> x -> z`` add.  StableHLO is used because it retains slice bounds and
``optimization_barrier`` operations explicitly while optimized TPU HLO may
legitimately fuse the same tree into several loop fusions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
import re
from typing import Iterable

from ..kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)


_SSA = r"%[A-Za-z0-9_.$#-]+"
_ASSIGNMENT_RE = re.compile(rf"^\s*({_SSA})(?::[0-9]+)?\s*=\s*(.*)$")
_FUNCTION_RE = re.compile(
    r"^\s*func\.func\s+(?:(?:public|private)\s+)?@([^\s(]+)"
)
_TENSOR_RE = re.compile(r"tensor<[^>\n]+>")
_SLICE_RE = re.compile(
    rf"stablehlo\.slice\s+({_SSA})\s+\[([0-9:,\s-]+)\]"
)
_DIMS_RE = re.compile(r"\bdims\s*=\s*\[([^]]*)\]")
_ALL_GATHER_DIM_RE = re.compile(r"\ball_gather_dim\s*=\s*([0-9]+)")
_CONCAT_DIM_RE = re.compile(r"\bdim\s*=\s*([0-9]+)")
_KERNEL_RE = re.compile(r'\bkernel_name\s*=\s*"([^"]+)"')
_CONSTANT_RE = re.compile(r"\bconstant\s+dense<([^>]+)>")
_CALLEE_RE = re.compile(r"\bfunc\.call\s+@([^\s(]+)")
_PAD_LOW_RE = re.compile(r"\blow\s*=\s*\[([^]]*)\]")
_PAD_HIGH_RE = re.compile(r"\bhigh\s*=\s*\[([^]]*)\]")
_PAD_INTERIOR_RE = re.compile(r"\binterior\s*=\s*\[([^]]*)\]")
_KERNEL_NAME = "greenfield_fp8_block_matmul_m8_k512_n6144"


def _normalize_type(value: str) -> str:
    return re.sub(r"\s+", "", value)


@dataclass(frozen=True, slots=True)
class _StableNode:
    name: str
    opcode: str
    operands: tuple[str, ...]
    result_type: str | None
    tensor_types: tuple[str, ...]
    slice_ranges: tuple[tuple[int, int], ...] | None
    dimensions: tuple[int, ...] | None
    concatenate_dimension: int | None
    kernel_name: str | None
    constant_literal: str | None
    callee: str | None
    pad_low: tuple[int, ...] | None
    pad_high: tuple[int, ...] | None
    pad_interior: tuple[int, ...] | None
    all_gather_dimension: int | None
    use_global_device_ids: bool
    line_number: int


@dataclass(slots=True)
class _StableGraph:
    name: str
    header: str
    nodes: dict[str, _StableNode]
    users: dict[str, list[_StableNode]]

    def node(self, name: str) -> _StableNode:
        try:
            return self.nodes[name]
        except KeyError as error:
            raise _MatchError(f"undefined StableHLO value {name}") from error

    def matching_users(
        self,
        source: str,
        *,
        opcode: str | None = None,
        result_type: str | None = None,
    ) -> list[_StableNode]:
        return [
            node
            for node in self.users.get(source, ())
            if (opcode is None or node.opcode == opcode)
            and (result_type is None or node.result_type == result_type)
        ]


class _MatchError(ValueError):
    pass


def _opcode(rhs: str) -> str | None:
    quoted = re.match(r'"stablehlo\.([a-z0-9_]+)"', rhs)
    if quoted is not None:
        return quoted.group(1)
    stable = re.match(r"stablehlo\.([a-z0-9_]+)", rhs)
    if stable is not None:
        return stable.group(1)
    if rhs.startswith("func.call"):
        return "call"
    return None


def _slice_ranges(rhs: str) -> tuple[tuple[int, int], ...] | None:
    match = _SLICE_RE.search(rhs)
    if match is None:
        return None
    result: list[tuple[int, int]] = []
    for item in match.group(2).split(","):
        bounds = item.strip().split(":")
        if len(bounds) != 2:
            return None
        result.append((int(bounds[0]), int(bounds[1])))
    return tuple(result)


def _dimensions(rhs: str) -> tuple[int, ...] | None:
    match = _DIMS_RE.search(rhs)
    if match is None:
        return None
    value = match.group(1).strip()
    return () if not value else tuple(int(item) for item in value.split(","))


def _integer_array(
    pattern: re.Pattern[str], rhs: str
) -> tuple[int, ...] | None:
    match = pattern.search(rhs)
    if match is None:
        return None
    value = match.group(1).strip()
    return () if not value else tuple(int(item) for item in value.split(","))


def _parse_graphs(stablehlo: str) -> tuple[list[_StableGraph], list[str]]:
    graphs: list[_StableGraph] = []
    errors: list[str] = []
    current: _StableGraph | None = None
    return_index = 0
    for line_number, line in enumerate(StringIO(stablehlo), start=1):
        function = _FUNCTION_RE.match(line)
        if function is not None:
            current = _StableGraph(
                function.group(1), line.strip(), {}, defaultdict(list)
            )
            graphs.append(current)
            return_index = 0
        if current is None:
            continue
        assignment = _ASSIGNMENT_RE.match(line)
        if assignment is None:
            returned = re.match(
                rf"^\s*(?:(?:stablehlo|sdy)\.)?return\b(.*)$", line
            )
            if returned is not None:
                operands = tuple(re.findall(_SSA, returned.group(1)))
                name = f"%__return_{return_index}"
                return_index += 1
                node = _StableNode(
                    name=name,
                    opcode="return",
                    operands=operands,
                    result_type=None,
                    tensor_types=(),
                    slice_ranges=None,
                    dimensions=None,
                    concatenate_dimension=None,
                    kernel_name=None,
                    constant_literal=None,
                    callee=None,
                    pad_low=None,
                    pad_high=None,
                    pad_interior=None,
                    all_gather_dimension=None,
                    use_global_device_ids=False,
                    line_number=line_number,
                )
                current.nodes[name] = node
                for operand in operands:
                    current.users[operand].append(node)
            continue
        name, rhs = assignment.groups()
        opcode = _opcode(rhs)
        if opcode is None:
            continue
        if name in current.nodes:
            # StableHLO region arguments and temporaries have nested SSA
            # scope and may reuse an outer function's spelling.  StrategyND
            # lives in the enclosing manual-computation body, so retain the
            # first definition and ignore later region-local spellings.
            continue
        tensor_types = tuple(
            _normalize_type(item) for item in _TENSOR_RE.findall(rhs)
        )
        dims = _dimensions(rhs)
        concat_match = _CONCAT_DIM_RE.search(rhs)
        kernel_match = _KERNEL_RE.search(rhs)
        constant_match = _CONSTANT_RE.search(rhs)
        callee_match = _CALLEE_RE.search(rhs)
        all_gather_dim_match = _ALL_GATHER_DIM_RE.search(rhs)
        node = _StableNode(
            name=name,
            opcode=opcode,
            operands=tuple(re.findall(_SSA, rhs)),
            result_type=tensor_types[-1] if tensor_types else None,
            tensor_types=tensor_types,
            slice_ranges=_slice_ranges(rhs),
            dimensions=dims,
            concatenate_dimension=(
                None if concat_match is None else int(concat_match.group(1))
            ),
            kernel_name=(
                None if kernel_match is None else kernel_match.group(1)
            ),
            constant_literal=(
                None if constant_match is None else constant_match.group(1)
            ),
            callee=None if callee_match is None else callee_match.group(1),
            pad_low=_integer_array(_PAD_LOW_RE, rhs),
            pad_high=_integer_array(_PAD_HIGH_RE, rhs),
            pad_interior=_integer_array(_PAD_INTERIOR_RE, rhs),
            all_gather_dimension=(
                None
                if all_gather_dim_match is None
                else int(all_gather_dim_match.group(1))
            ),
            use_global_device_ids="use_global_device_ids" in rhs,
            line_number=line_number,
        )
        current.nodes[name] = node
        for operand in node.operands:
            current.users[operand].append(node)
    if not graphs:
        errors.append("StableHLO contains no func.func computation")
    return graphs, errors


def _only(candidates: Iterable[_StableNode], description: str) -> _StableNode:
    values = list(candidates)
    if len(values) != 1:
        raise _MatchError(
            f"{description}: expected one node, found {len(values)}"
        )
    return values[0]


def _expect_node(
    graph: _StableGraph,
    name: str,
    *,
    opcode: str,
    result_type: str,
) -> _StableNode:
    node = graph.node(name)
    if node.opcode != opcode or node.result_type != result_type:
        raise _MatchError(
            f"{name}: expected {opcode} {result_type}, found "
            f"{node.opcode} {node.result_type}"
        )
    return node


def _expect_slice(
    graph: _StableGraph,
    source: str,
    ranges: tuple[tuple[int, int], ...],
    result_type: str,
) -> _StableNode:
    return _only(
        (
            node
            for node in graph.matching_users(
                source, opcode="slice", result_type=result_type
            )
            if node.slice_ranges == ranges and node.operands == (source,)
        ),
        f"slice {source} {ranges} -> {result_type}",
    )


def _expect_reshape(
    graph: _StableGraph, source: str, result_type: str
) -> _StableNode:
    return _only(
        (
            node
            for node in graph.matching_users(
                source, opcode="reshape", result_type=result_type
            )
            if node.operands == (source,)
        ),
        f"reshape {source} -> {result_type}",
    )


def _expect_concat(
    graph: _StableGraph,
    operands: tuple[str, ...],
    *,
    dimension: int,
    result_type: str,
) -> _StableNode:
    return _only(
        (
            node
            for node in graph.nodes.values()
            if node.opcode == "concatenate"
            and node.operands == operands
            and node.concatenate_dimension == dimension
            and node.result_type == result_type
        ),
        f"concatenate dim={dimension} -> {result_type}",
    )


def _expect_add_barrier(
    graph: _StableGraph,
    left: str,
    right: str,
    result_type: str,
) -> _StableNode:
    expected = Counter((left, right))
    add = _only(
        (
            node
            for node in graph.nodes.values()
            if node.opcode == "add"
            and node.result_type == result_type
            and Counter(node.operands) == expected
        ),
        f"BF16 add {left} + {right}",
    )
    return _only(
        (
            node
            for node in graph.matching_users(
                add.name,
                opcode="optimization_barrier",
                result_type=result_type,
            )
            if node.operands == (add.name,)
        ),
        f"BF16 barrier after {add.name}",
    )


def _expect_reduce_four(
    graph: _StableGraph,
    values: tuple[str, str, str, str],
    *,
    cross: bool,
    result_type: str,
) -> _StableNode:
    first_pair, second_pair = (
        ((0, 3), (1, 2)) if cross else ((0, 1), (2, 3))
    )
    first = _expect_add_barrier(
        graph,
        values[first_pair[0]],
        values[first_pair[1]],
        result_type,
    )
    second = _expect_add_barrier(
        graph,
        values[second_pair[0]],
        values[second_pair[1]],
        result_type,
    )
    return _expect_add_barrier(
        graph, first.name, second.name, result_type
    )


def _expect_zero_i32(graph: _StableGraph, name: str) -> None:
    node = _expect_node(
        graph, name, opcode="constant", result_type="tensor<i32>"
    )
    if node.constant_literal not in ("0", "0 : i32"):
        raise _MatchError(f"{name}: K512 pad constant is not exact zero")


def _validate_exact_pad_helper(
    helpers: dict[str, _StableGraph],
    verified: set[tuple[str, str, str, tuple[int, ...]]],
    *,
    callee: str | None,
    input_type: str,
    scalar_type: str,
    result_type: str,
    high: tuple[int, ...],
) -> None:
    if callee is None or callee not in helpers:
        raise _MatchError(f"unknown K512 pad helper {callee!r}")
    key = (callee, input_type, result_type, high)
    if key in verified:
        return
    helper = helpers[callee]
    normalized_header = re.sub(r"\s+", "", helper.header)
    expected_header = (
        f"func.funcprivate@{callee}(%arg0:{input_type},"
        f"%arg1:tensor<i32>)->{result_type}{{"
    )
    if normalized_header != expected_header:
        raise _MatchError(f"@{callee}: K512 pad signature drifted")
    converts = [
        node for node in helper.nodes.values() if node.opcode == "convert"
    ]
    pads = [node for node in helper.nodes.values() if node.opcode == "pad"]
    returns = [
        node for node in helper.nodes.values() if node.opcode == "return"
    ]
    if len(converts) != 1 or len(pads) != 1 or len(returns) != 1:
        raise _MatchError(f"@{callee}: K512 pad body arity drifted")
    convert = converts[0]
    pad = pads[0]
    returned = returns[0]
    if (
        convert.operands != ("%arg1",)
        or convert.tensor_types[-2:]
        != ("tensor<i32>", scalar_type)
        or convert.result_type != scalar_type
    ):
        raise _MatchError(f"@{callee}: K512 pad scalar conversion drifted")
    if (
        pad.operands != ("%arg0", convert.name)
        or pad.tensor_types[-3:]
        != (input_type, scalar_type, result_type)
        or pad.result_type != result_type
        or pad.pad_low != (0, 0)
        or pad.pad_high != high
        or pad.pad_interior != (0, 0)
    ):
        raise _MatchError(f"@{callee}: K512 pad placement drifted")
    if returned.operands != (pad.name,):
        raise _MatchError(f"@{callee}: K512 pad return lineage drifted")
    if len(helper.nodes) != 3:
        raise _MatchError(f"@{callee}: unexpected K512 pad arithmetic")
    verified.add(key)


def _match_kernel_stack(
    graph: _StableGraph,
    gather: _StableNode,
    *,
    helpers: dict[str, _StableGraph],
    verified_pad_helpers: set[tuple[str, str, str, tuple[int, ...]]],
) -> tuple[tuple[str, str, str], tuple[str, ...]]:
    if (
        gather.result_type != "tensor<4x8x1x6144xbf16>"
        or gather.tensor_types[-2:]
        != (
            "tensor<1x8x1x6144xbf16>",
            "tensor<4x8x1x6144xbf16>",
        )
        or gather.all_gather_dimension != 0
        or not gather.use_global_device_ids
    ):
        raise _MatchError(
            f"{gather.name}: StrategyND StableHLO gather contract drifted"
        )
    if len(gather.operands) != 1:
        raise _MatchError(f"{gather.name}: gather operand count drifted")
    broadcast = _expect_node(
        graph,
        gather.operands[0],
        opcode="broadcast_in_dim",
        result_type="tensor<1x8x1x6144xbf16>",
    )
    if broadcast.dimensions != (1, 2, 3) or len(broadcast.operands) != 1:
        raise _MatchError(f"{broadcast.name}: local stack broadcast drifted")
    stack = _expect_node(
        graph,
        broadcast.operands[0],
        opcode="concatenate",
        result_type="tensor<8x1x6144xbf16>",
    )
    if stack.concatenate_dimension != 0 or len(stack.operands) != 8:
        raise _MatchError(f"{stack.name}: K512 stack arity/order drifted")

    roots: list[tuple[str, str, str]] = []
    kernels: list[str] = []
    for shard, row_name in enumerate(stack.operands):
        row_broadcast = _expect_node(
            graph,
            row_name,
            opcode="broadcast_in_dim",
            result_type="tensor<1x1x6144xbf16>",
        )
        if row_broadcast.dimensions != (1, 2) or len(row_broadcast.operands) != 1:
            raise _MatchError(
                f"{row_broadcast.name}: K512 row placement drifted"
            )
        row = _expect_node(
            graph,
            row_broadcast.operands[0],
            opcode="slice",
            result_type="tensor<1x6144xbf16>",
        )
        if row.slice_ranges != ((0, 1), (0, 6144)) or len(row.operands) != 1:
            raise _MatchError(f"{row.name}: K512 live-row slice drifted")
        kernel = _expect_node(
            graph,
            row.operands[0],
            opcode="custom_call",
            result_type="tensor<8x6144xbf16>",
        )
        if (
            kernel.kernel_name != _KERNEL_NAME
            or len(kernel.operands) != 3
            or kernel.tensor_types[-4:]
            != (
                "tensor<8x512xbf16>",
                "tensor<6144x512xui8>",
                "tensor<8x128xf32>",
                "tensor<8x6144xbf16>",
            )
        ):
            raise _MatchError(f"{kernel.name}: exact K512 call drifted")
        start = shard * 512
        scale_start = shard * 4

        lhs_pad = _expect_node(
            graph,
            kernel.operands[0],
            opcode="call",
            result_type="tensor<8x512xbf16>",
        )
        if len(lhs_pad.operands) != 2:
            raise _MatchError(f"{lhs_pad.name}: K512 lhs pad arity drifted")
        _expect_zero_i32(graph, lhs_pad.operands[1])
        _validate_exact_pad_helper(
            helpers,
            verified_pad_helpers,
            callee=lhs_pad.callee,
            input_type="tensor<1x512xbf16>",
            scalar_type="tensor<bf16>",
            result_type="tensor<8x512xbf16>",
            high=(7, 0),
        )
        lhs_slice = _expect_node(
            graph,
            lhs_pad.operands[0],
            opcode="slice",
            result_type="tensor<1x512xbf16>",
        )
        if (
            lhs_slice.slice_ranges != ((0, 1), (start, start + 512))
            or lhs_slice.tensor_types[-2:]
            != (
                "tensor<1x4096xbf16>",
                "tensor<1x512xbf16>",
            )
            or len(lhs_slice.operands) != 1
        ):
            raise _MatchError(
                f"{lhs_slice.name}: ordered K512 lhs slice {shard} drifted"
            )

        weight_slice = _expect_node(
            graph,
            kernel.operands[1],
            opcode="slice",
            result_type="tensor<6144x512xui8>",
        )
        if (
            weight_slice.slice_ranges
            != ((0, 6144), (start, start + 512))
            or weight_slice.tensor_types[-2:]
            != (
                "tensor<6144x4096xui8>",
                "tensor<6144x512xui8>",
            )
            or len(weight_slice.operands) != 1
        ):
            raise _MatchError(
                f"{weight_slice.name}: ordered K512 weight slice {shard} drifted"
            )

        scale_pad = _expect_node(
            graph,
            kernel.operands[2],
            opcode="call",
            result_type="tensor<8x128xf32>",
        )
        if len(scale_pad.operands) != 2:
            raise _MatchError(f"{scale_pad.name}: K512 scale pad arity drifted")
        _expect_zero_i32(graph, scale_pad.operands[1])
        _validate_exact_pad_helper(
            helpers,
            verified_pad_helpers,
            callee=scale_pad.callee,
            input_type="tensor<4x48xf32>",
            scalar_type="tensor<f32>",
            result_type="tensor<8x128xf32>",
            high=(4, 80),
        )
        transpose = _expect_node(
            graph,
            scale_pad.operands[0],
            opcode="transpose",
            result_type="tensor<4x48xf32>",
        )
        if transpose.dimensions != (1, 0) or len(transpose.operands) != 1:
            raise _MatchError(f"{transpose.name}: K512 scale transpose drifted")
        scale_slice = _expect_node(
            graph,
            transpose.operands[0],
            opcode="slice",
            result_type="tensor<48x4xf32>",
        )
        if (
            scale_slice.slice_ranges
            != ((0, 48), (scale_start, scale_start + 4))
            or scale_slice.tensor_types[-2:]
            != ("tensor<48x32xf32>", "tensor<48x4xf32>")
            or len(scale_slice.operands) != 1
        ):
            raise _MatchError(
                f"{scale_slice.name}: ordered K512 scale slice {shard} drifted"
            )
        roots.append(
            (
                lhs_slice.operands[0],
                weight_slice.operands[0],
                scale_slice.operands[0],
            )
        )
        kernels.append(kernel.name)
    if len(set(roots)) != 1:
        raise _MatchError(
            f"{gather.name}: K512 partials are cross-wired across layer sources"
        )
    return roots[0], tuple(kernels)


def _flatten_concatenate(
    graph: _StableGraph, name: str, *, dimension: int
) -> tuple[str, ...]:
    node = graph.node(name)
    if (
        node.opcode != "concatenate"
        or node.concatenate_dimension != dimension
    ):
        return (name,)
    result: list[str] = []
    for operand in node.operands:
        child = graph.node(operand)
        if (
            child.opcode == "concatenate"
            and child.concatenate_dimension == dimension
        ):
            result.extend(
                _flatten_concatenate(
                    graph, operand, dimension=dimension
                )
            )
        else:
            result.append(operand)
    return tuple(result)


def _physical_row_index(
    graph: _StableGraph, name: str, flat_gather: str
) -> int:
    broadcast = _expect_node(
        graph,
        name,
        opcode="broadcast_in_dim",
        result_type="tensor<1x1x6144xbf16>",
    )
    if broadcast.dimensions != (1, 2) or len(broadcast.operands) != 1:
        raise _MatchError(f"{broadcast.name}: physical-row broadcast drifted")
    reshaped = _expect_node(
        graph,
        broadcast.operands[0],
        opcode="reshape",
        result_type="tensor<1x6144xbf16>",
    )
    if len(reshaped.operands) != 1:
        raise _MatchError(f"{reshaped.name}: physical-row reshape drifted")
    row = _expect_node(
        graph,
        reshaped.operands[0],
        opcode="slice",
        result_type="tensor<1x1x6144xbf16>",
    )
    if len(row.operands) != 1 or row.operands[0] != flat_gather:
        raise _MatchError(f"{row.name}: physical row escaped its gather")
    ranges = row.slice_ranges
    if (
        ranges is None
        or len(ranges) != 3
        or ranges[1:] != ((0, 1), (0, 6144))
        or ranges[0][1] != ranges[0][0] + 1
    ):
        raise _MatchError(f"{row.name}: physical-row slice drifted")
    return ranges[0][0]


def _match_reduction_tree(
    graph: _StableGraph, gather: _StableNode
) -> str:
    flat = _expect_reshape(
        graph, gather.name, "tensor<32x1x6144xbf16>"
    )
    if graph.users.get(gather.name) != [flat]:
        raise _MatchError(
            f"{gather.name}: gather must feed only the exact reduction tree"
        )
    candidates: list[_StableNode] = []
    for transpose in graph.nodes.values():
        if (
            transpose.opcode != "transpose"
            or transpose.result_type != "tensor<4x2x4x1x6144xbf16>"
            or transpose.dimensions != (1, 2, 0, 3, 4)
            or len(transpose.operands) != 1
        ):
            continue
        reshape = graph.node(transpose.operands[0])
        if (
            reshape.opcode != "reshape"
            or reshape.result_type != "tensor<4x4x2x1x6144xbf16>"
            or len(reshape.operands) != 1
        ):
            continue
        stack = graph.node(reshape.operands[0])
        if stack.result_type != "tensor<32x1x6144xbf16>":
            continue
        try:
            leaves = _flatten_concatenate(graph, stack.name, dimension=0)
            rows = tuple(
                _physical_row_index(graph, leaf, flat.name) for leaf in leaves
            )
        except _MatchError:
            continue
        if rows == STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE:
            candidates.append(transpose)
    transposed = _only(candidates, f"{gather.name}: physical DB533 reorder")

    y_results: list[str] = []
    for hidden_start, cross in ((0, False), (2048, True), (4096, False)):
        chunk = _expect_slice(
            graph,
            transposed.name,
            (
                (0, 4),
                (0, 2),
                (0, 4),
                (0, 1),
                (hidden_start, hidden_start + 2048),
            ),
            "tensor<4x2x4x1x2048xbf16>",
        )
        values: list[str] = []
        for row in range(4):
            sliced = _expect_slice(
                graph,
                chunk.name,
                (
                    (row, row + 1),
                    (0, 2),
                    (0, 4),
                    (0, 1),
                    (0, 2048),
                ),
                "tensor<1x2x4x1x2048xbf16>",
            )
            values.append(
                _expect_reshape(
                    graph, sliced.name, "tensor<2x4x1x2048xbf16>"
                ).name
            )
        y_results.append(
            _expect_reduce_four(
                graph,
                tuple(values),  # type: ignore[arg-type]
                cross=cross,
                result_type="tensor<2x4x1x2048xbf16>",
            ).name
        )
    y_reduced = _expect_concat(
        graph,
        tuple(y_results),
        dimension=3,
        result_type="tensor<2x4x1x6144xbf16>",
    )

    x_values: list[str] = []
    for row in range(2):
        sliced = _expect_slice(
            graph,
            y_reduced.name,
            ((row, row + 1), (0, 4), (0, 1), (0, 6144)),
            "tensor<1x4x1x6144xbf16>",
        )
        x_values.append(
            _expect_reshape(
                graph, sliced.name, "tensor<4x1x6144xbf16>"
            ).name
        )
    x_reduced = _expect_add_barrier(
        graph,
        x_values[0],
        x_values[1],
        "tensor<4x1x6144xbf16>",
    )

    z_results: list[str] = []
    for segment, hidden_start in enumerate(range(0, 6144, 256)):
        chunk = _expect_slice(
            graph,
            x_reduced.name,
            ((0, 4), (0, 1), (hidden_start, hidden_start + 256)),
            "tensor<4x1x256xbf16>",
        )
        values = []
        for row in range(4):
            sliced = _expect_slice(
                graph,
                chunk.name,
                ((row, row + 1), (0, 1), (0, 256)),
                "tensor<1x1x256xbf16>",
            )
            values.append(
                _expect_reshape(
                    graph, sliced.name, "tensor<1x256xbf16>"
                ).name
            )
        z_results.append(
            _expect_reduce_four(
                graph,
                tuple(values),  # type: ignore[arg-type]
                cross=bool(segment % 2),
                result_type="tensor<1x256xbf16>",
            ).name
        )
    output = _only(
        (
            node
            for node in graph.nodes.values()
            if node.opcode == "concatenate"
            and node.concatenate_dimension == 1
            and node.result_type == "tensor<1x6144xbf16>"
            and _flatten_concatenate(
                graph, node.name, dimension=1
            )
            == tuple(z_results)
        ),
        "exact 24-segment StrategyND z concatenate",
    )
    ancestors = {output.name}
    pending = [output.name]
    while pending:
        value = pending.pop()
        for operand in graph.node(value).operands:
            if operand in graph.nodes and operand not in ancestors:
                ancestors.add(operand)
                pending.append(operand)

    descendants = {gather.name}
    pending = [gather.name]
    while pending:
        value = pending.pop()
        for user in graph.users.get(value, ()):
            if user.name in descendants:
                continue
            descendants.add(user.name)
            if user.name != output.name:
                pending.append(user.name)
    rogue_branches = sorted(descendants - ancestors)
    if rogue_branches:
        raise _MatchError(
            f"{gather.name}: gather bypasses the exact StrategyND output via "
            f"{rogue_branches[:8]}"
        )

    pending = [output.name]
    seen = {output.name}
    reaches_return = False
    while pending and not reaches_return:
        value = pending.pop()
        for user in graph.users.get(value, ()):
            if user.opcode == "return":
                reaches_return = True
                break
            if user.name not in seen:
                seen.add(user.name)
                pending.append(user.name)
    if not reaches_return:
        raise _MatchError(
            f"{output.name}: exact StrategyND output does not reach a return"
        )
    return output.name


def validate_strategy_nd_attention_stablehlo(
    stablehlo: str | None,
    *,
    layers: int,
    enabled: bool,
) -> dict[str, object]:
    """Validate the exact pre-fusion StrategyND attention program."""

    if not isinstance(enabled, bool):
        raise TypeError("StrategyND StableHLO flag must be boolean")
    if not isinstance(layers, int) or isinstance(layers, bool) or layers < 0:
        raise ValueError("StrategyND StableHLO layer count must be nonnegative")
    if stablehlo is None:
        violations = (
            ["StrategyND StableHLO is required when the path is enabled"]
            if enabled
            else []
        )
        return {
            "applicable": enabled,
            "expected_gather_count": layers if enabled else 0,
            "gather_count": 0,
            "kernel_count": 0,
            "matched_trees": [],
            "passed": not violations,
            "violations": violations,
        }

    graphs, violations = _parse_graphs(stablehlo)
    helpers: dict[str, _StableGraph] = {}
    for graph in graphs:
        if graph.name in helpers:
            violations.append(f"duplicate StableHLO function @{graph.name}")
        else:
            helpers[graph.name] = graph
    gathered: list[tuple[_StableGraph, _StableNode]] = []
    all_kernels: list[tuple[str, str]] = []
    for graph in graphs:
        for node in graph.nodes.values():
            if node.opcode == "custom_call" and node.kernel_name == _KERNEL_NAME:
                all_kernels.append((graph.name, node.name))
            if (
                node.opcode == "all_gather"
                and node.result_type == "tensor<4x8x1x6144xbf16>"
                and node.tensor_types[-2:]
                == (
                    "tensor<1x8x1x6144xbf16>",
                    "tensor<4x8x1x6144xbf16>",
                )
            ):
                gathered.append((graph, node))

    expected_gathers = layers if enabled else 0
    expected_kernels = 8 * expected_gathers
    if len(gathered) != expected_gathers:
        violations.append(
            "StrategyND StableHLO gather count drifted: "
            f"expected={expected_gathers} observed={len(gathered)}"
        )
    if len(all_kernels) != expected_kernels:
        violations.append(
            "StrategyND StableHLO K512 count drifted: "
            f"expected={expected_kernels} observed={len(all_kernels)}"
        )
    if not enabled and (gathered or all_kernels):
        violations.append(
            "default StableHLO unexpectedly contains StrategyND attention"
        )

    matched_trees: list[dict[str, object]] = []
    matched_kernels: set[tuple[str, str]] = set()
    roots: set[tuple[str, str, str, str]] = set()
    verified_pad_helpers: set[
        tuple[str, str, str, tuple[int, ...]]
    ] = set()
    if enabled:
        for graph, gather in gathered:
            try:
                source_roots, kernels = _match_kernel_stack(
                    graph,
                    gather,
                    helpers=helpers,
                    verified_pad_helpers=verified_pad_helpers,
                )
                final = _match_reduction_tree(graph, gather)
                kernel_keys = {(graph.name, name) for name in kernels}
                if matched_kernels & kernel_keys:
                    raise _MatchError(
                        f"{gather.name}: K512 call reused by another gather"
                    )
                matched_kernels.update(kernel_keys)
                root_key = (graph.name, *source_roots)
                if root_key in roots:
                    raise _MatchError(
                        f"{gather.name}: layer source triple reused"
                    )
                roots.add(root_key)
                matched_trees.append(
                    {
                        "computation": graph.name,
                        "final_output": final,
                        "gather": gather.name,
                        "kernel_count": len(kernels),
                        "source_roots": list(source_roots),
                    }
                )
            except _MatchError as error:
                violations.append(
                    f"{graph.name}/{gather.name}: {error}"
                )
    if matched_kernels != set(all_kernels):
        missing = sorted(set(all_kernels) - matched_kernels)
        extra = sorted(matched_kernels - set(all_kernels))
        if missing or extra:
            violations.append(
                "StrategyND StableHLO K512/gather bijection failed: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
    return {
        "applicable": enabled,
        "expected_gather_count": expected_gathers,
        "expected_kernel_count": expected_kernels,
        "gather_count": len(gathered),
        "kernel_count": len(all_kernels),
        "matched_tree_count": len(matched_trees),
        "matched_trees": matched_trees,
        "passed": not violations,
        "violations": violations,
    }
