"""Whole-module HLO admission for the Gate-D chunk-0 embedding correction.

This module is deliberately stdlib-only.  The protected probe imports it from
the sealed Git archive before invoking the executable; the publisher repeats
the same check on the archived HLO.  Merely naming a fusion ``gather`` or
``reduce_sum`` is not evidence: called computation bodies and caller/callee
parameter mappings are part of the contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
import struct
from typing import Any


_INSTRUCTION = re.compile(
    r"^\s*(?P<root>ROOT )?(?P<name>%[A-Za-z0-9_.-]+) = "
    r"(?P<result>.*?)\b(?P<opcode>[a-z][a-z0-9-]*)\("
)
_HEADER = re.compile(
    r"^(?P<entry>ENTRY )?(?P<name>%[A-Za-z0-9_.-]+)\s*\(.*\)\s*->\s*.*\{$"
)
_SHAPE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\[([0-9,]*)\]")
_NAME = re.compile(r"%[A-Za-z0-9_.-]+")


def _balanced_end(value: str, start: int) -> int:
    depth = 0
    for index in range(start, len(value)):
        if value[index] == "(":
            depth += 1
        elif value[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise RuntimeError("chunk-0 embedding HLO call syntax is unbalanced")


def _split_operands(value: str) -> tuple[str, ...]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(value[start:index].strip())
            start = index + 1
    if value.strip():
        result.append(value[start:].strip())
    names = []
    for operand in result:
        match = _NAME.search(re.sub(r"/\*.*?\*/", "", operand))
        if match is not None:
            names.append(match.group(0))
    return tuple(names)


def _shapes(value: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            match.group(1),
            tuple(int(item) for item in match.group(2).split(",") if item),
        )
        for match in _SHAPE.finditer(value)
    )


def _module_graph(
    optimized: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    computations: dict[str, dict[str, Any]] = {}
    current = ""
    entry = ""
    for line in optimized.splitlines():
        header = _HEADER.match(line)
        if header is not None:
            current = header.group("name")
            if current in computations:
                raise RuntimeError("chunk-0 embedding HLO computation is duplicated")
            computations[current] = {"nodes": {}, "root": ""}
            if header.group("entry"):
                if entry:
                    raise RuntimeError("chunk-0 embedding HLO has multiple ENTRY computations")
                entry = current
            continue
        if line == "}":
            current = ""
            continue
        if not current:
            continue
        match = _INSTRUCTION.match(line)
        if match is None:
            continue
        call_start = match.end() - 1
        call_end = _balanced_end(line, call_start)
        name = match.group("name")
        parameter = None
        if match.group("opcode") == "parameter":
            number = re.fullmatch(r"\s*([0-9]+)\s*", line[call_start + 1 : call_end])
            if number is None:
                raise RuntimeError("chunk-0 embedding HLO parameter number is malformed")
            parameter = int(number.group(1))
        called = re.search(r"\b(?:calls|to_apply)=(%[A-Za-z0-9_.-]+)", line[call_end + 1 :])
        body = re.search(r"\bbody=(%[A-Za-z0-9_.-]+)", line[call_end + 1 :])
        condition = re.search(
            r"\bcondition=(%[A-Za-z0-9_.-]+)", line[call_end + 1 :]
        )
        tuple_index = None
        if match.group("opcode") == "get-tuple-element":
            selected = re.search(r"\bindex=([0-9]+)", line[call_end + 1 :])
            if selected is None:
                raise RuntimeError("chunk-0 embedding HLO tuple index is missing")
            tuple_index = int(selected.group(1))
        nodes = computations[current]["nodes"]
        if name in nodes:
            raise RuntimeError("chunk-0 embedding HLO instruction is duplicated")
        nodes[name] = {
            "called": None if called is None else called.group(1),
            "body": None if body is None else body.group(1),
            "condition": None if condition is None else condition.group(1),
            "opcode": match.group("opcode"),
            "operands": _split_operands(line[call_start + 1 : call_end]),
            "parameter": parameter,
            "raw": line,
            "shapes": _shapes(match.group("result")),
            "tuple_index": tuple_index,
        }
        if match.group("root"):
            if computations[current]["root"]:
                raise RuntimeError("chunk-0 embedding HLO computation has multiple roots")
            computations[current]["root"] = name
    if not entry or not computations.get(entry, {}).get("root"):
        raise RuntimeError("chunk-0 embedding HLO ENTRY graph is incomplete")
    for computation in computations.values():
        if computation["nodes"] and not computation["root"]:
            raise RuntimeError("chunk-0 embedding HLO called computation has no root")
    return computations, entry


def _ancestors(nodes: Mapping[str, Mapping[str, Any]], name: str) -> set[str]:
    pending = [name]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        if current not in nodes:
            raise RuntimeError("chunk-0 embedding HLO operand is unresolved")
        visited.add(current)
        pending.extend(nodes[current]["operands"])
        if len(visited) > 20000:
            raise RuntimeError("chunk-0 embedding HLO graph is unbounded")
    return visited


def _transparent_origin(
    nodes: Mapping[str, Mapping[str, Any]],
    name: str,
    *,
    allowed: frozenset[str] = frozenset(
        {"bitcast", "copy", "copy-done", "copy-start", "reshape", "transpose"}
    ),
) -> str:
    """Follow a single-input representation-only chain to its producer."""

    visited: set[str] = set()
    while nodes[name]["opcode"] in allowed:
        if name in visited or len(nodes[name]["operands"]) != 1:
            raise RuntimeError("chunk-0 embedding HLO transparent chain is malformed")
        visited.add(name)
        name = nodes[name]["operands"][0]
        if name not in nodes:
            raise RuntimeError("chunk-0 embedding HLO transparent operand is unresolved")
    return name


def _exact_bf16_to_f32_lineage(
    nodes: Mapping[str, Mapping[str, Any]], name: str, source_value: str
) -> bool:
    """Require exactly one BF16-to-F32 promotion and no other conversion."""

    representation = frozenset({"bitcast", "copy", "reshape", "transpose"})
    converted = _transparent_origin(nodes, name, allowed=representation)
    node = nodes[converted]
    if (
        node["opcode"] != "convert"
        or node["shapes"] != (("f32", (2048, 6144)),)
        or len(node["operands"]) != 1
    ):
        return False
    source = _transparent_origin(
        nodes, node["operands"][0], allowed=representation
    )
    return (
        source == source_value
        and nodes[source_value]["shapes"] == (("bf16", (2048, 6144)),)
    )


def _f32_bits(value: float) -> bytes:
    return struct.pack(">f", value)


def _exact_f32_broadcast_lineage(
    nodes: Mapping[str, Mapping[str, Any]],
    name: str,
    parameter: str,
    *,
    parameter_shape: tuple[int, ...],
    dimensions: str,
) -> bool:
    representation = frozenset({"bitcast", "copy", "reshape", "transpose"})
    broadcast = _transparent_origin(nodes, name, allowed=representation)
    node = nodes[broadcast]
    if (
        node["opcode"] != "broadcast"
        or node["shapes"] != (("f32", (2048, 6144)),)
        or f"dimensions={{{dimensions}}}" not in node["raw"]
        or len(node["operands"]) != 1
    ):
        return False
    source = _transparent_origin(
        nodes, node["operands"][0], allowed=representation
    )
    return (
        source == parameter
        and nodes[parameter]["shapes"] == (("f32", parameter_shape),)
    )


def _exact_weight_broadcast_lineage(
    nodes: Mapping[str, Mapping[str, Any]], name: str, parameter: str
) -> bool:
    if nodes[parameter]["shapes"] == (("f32", (6144,)),):
        return _exact_f32_broadcast_lineage(
            nodes,
            name,
            parameter,
            parameter_shape=(6144,),
            dimensions="1",
        )
    if nodes[parameter]["shapes"] != (("bf16", (6144,)),):
        return False
    representation = frozenset({"bitcast", "copy", "reshape", "transpose"})
    converted = _transparent_origin(nodes, name, allowed=representation)
    convert = nodes[converted]
    if (
        convert["opcode"] != "convert"
        or convert["shapes"] != (("f32", (2048, 6144)),)
        or len(convert["operands"]) != 1
    ):
        return False
    broadcast = _transparent_origin(
        nodes, convert["operands"][0], allowed=representation
    )
    broadcast_node = nodes[broadcast]
    if (
        broadcast_node["opcode"] != "broadcast"
        or broadcast_node["shapes"] != (("bf16", (2048, 6144)),)
        or "dimensions={1}" not in broadcast_node["raw"]
        or len(broadcast_node["operands"]) != 1
    ):
        return False
    return (
        _transparent_origin(
            nodes, broadcast_node["operands"][0], allowed=representation
        )
        == parameter
    )


def _parameters(computation: Mapping[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, node in computation["nodes"].items():
        number = node["parameter"]
        if number is not None:
            if number in result:
                raise RuntimeError("chunk-0 embedding HLO parameter number is duplicated")
            result[number] = name
    if result and set(result) != set(range(len(result))):
        raise RuntimeError("chunk-0 embedding HLO parameter numbering is sparse")
    return result


def _constant_values(
    nodes: Mapping[str, Mapping[str, Any]], names: set[str]
) -> tuple[float, ...]:
    values = []
    for name in names:
        node = nodes[name]
        if node["opcode"] != "constant":
            continue
        match = re.search(r"\bconstant\(([-+0-9.eE]+)\)", node["raw"])
        if match is not None:
            values.append(float(match.group(1)))
    return tuple(values)


def _callee(
    computations: Mapping[str, Mapping[str, Any]], node: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    called = node["called"]
    return computations.get(called) if called is not None else None


def _gather_body(computation: Mapping[str, Any]) -> dict[str, int] | None:
    nodes = computation["nodes"]
    params = _parameters(computation)
    if len(params) != 2:
        return None
    if nodes[params[0]]["shapes"] != (("bf16", (37, 6144)),) or nodes[params[1]]["shapes"] != (("s32", (2048,)),):
        return None
    gathers = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "gather"
        and node["shapes"] == (("bf16", (2048, 6144)),)
        and all(
            token in node["raw"]
            for token in (
                "offset_dims={1}",
                "collapsed_slice_dims={0}",
                "start_index_map={0}",
                "index_vector_dim=1",
                "slice_sizes={1,6144}",
            )
        )
    ]
    if len(gathers) != 1:
        return None
    gather = gathers[0]
    operands = nodes[gather]["operands"]
    root_live = _ancestors(nodes, computation["root"])
    table_origin = _transparent_origin(nodes, operands[0]) if len(operands) == 2 else ""
    index_live = _ancestors(nodes, operands[1]) if len(operands) == 2 else set()
    index_operations = {
        nodes[name]["opcode"] for name in index_live if name != params[1]
    }
    if (
        len(operands) != 2
        or table_origin != params[0]
        or params[1] not in index_live
        or not index_operations
        <= {"bitcast", "copy", "custom-call", "reshape", "slice", "transpose"}
        or any(
            nodes[name]["opcode"] == "custom-call"
            and 'custom_call_target="AssumeGatherIndicesInBound"' not in nodes[name]["raw"]
            for name in index_live
        )
        or gather not in root_live
        or _transparent_origin(nodes, computation["root"]) != gather
        or not set(params.values()) <= root_live
    ):
        return None
    return {"table": 0, "indices": 1}


def _rms_body(computation: Mapping[str, Any]) -> int | None:
    nodes = computation["nodes"]
    params = _parameters(computation)
    chunk_params = [
        number
        for number, name in params.items()
        if nodes[name]["shapes"] == (("bf16", (2048, 6144)),)
    ]
    reductions = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "reduce"
        and node["shapes"] == (("f32", (2048,)),)
        and "dimensions={1}" in node["raw"]
    ]
    if len(chunk_params) != 1 or len(reductions) != 1:
        return None
    chunk_number = chunk_params[0]
    reduction = reductions[0]
    reduction_operands = nodes[reduction]["operands"]
    if len(reduction_operands) != 2:
        return None
    square = nodes.get(reduction_operands[0])
    if square is None or square["opcode"] != "multiply" or len(square["operands"]) != 2 or square["operands"][0] != square["operands"][1]:
        return None
    root_live = _ancestors(nodes, computation["root"])
    square_value = square["operands"][0]
    initializer = _transparent_origin(nodes, reduction_operands[1])
    initializer_values = _constant_values(nodes, {initializer})
    if (
        reduction not in root_live
        or _transparent_origin(nodes, computation["root"]) != reduction
        or not _exact_bf16_to_f32_lineage(
            nodes, square_value, params[chunk_number]
        )
        or not set(params.values()) <= root_live
        or nodes[initializer]["opcode"] != "constant"
        or nodes[initializer]["shapes"] != (("f32", ()),)
        or len(initializer_values) != 1
        or _f32_bits(initializer_values[0]) != _f32_bits(0.0)
        or nodes[computation["root"]]["shapes"] != (("f32", (2048,)),)
    ):
        return None
    return chunk_number


def _add_region_is_sum(computation: Mapping[str, Any]) -> bool:
    nodes = computation["nodes"]
    params = _parameters(computation)
    root = nodes.get(computation["root"])
    return (
        len(params) == 2
        and root is not None
        and root["opcode"] == "add"
        and set(root["operands"]) == set(params.values())
    )


def _inverse_body(computation: Mapping[str, Any]) -> int | None:
    nodes = computation["nodes"]
    params = _parameters(computation)
    if len(params) != 1 or nodes[params[0]]["shapes"] != (("f32", (2048,)),):
        return None
    root = nodes[computation["root"]]
    if root["opcode"] != "rsqrt" or len(root["operands"]) != 1:
        return None
    add = nodes.get(root["operands"][0])
    if add is None or add["opcode"] != "add" or len(add["operands"]) != 2:
        return None
    multiply_names = [
        name for name in add["operands"] if nodes[name]["opcode"] == "multiply"
    ]
    if len(multiply_names) != 1:
        return None
    multiply_name = multiply_names[0]
    multiply = nodes[multiply_name]
    if len(multiply["operands"]) != 2:
        return None
    scalar_origin = lambda name: _transparent_origin(
        nodes,
        name,
        allowed=frozenset(
            {"bitcast", "broadcast", "copy", "reshape", "transpose"}
        ),
    )
    multiply_origins = {scalar_origin(name) for name in multiply["operands"]}
    add_origins = {scalar_origin(name) for name in add["operands"]}
    reciprocal = [
        name
        for name in multiply_origins
        if nodes[name]["opcode"] == "constant"
        and any(
            _f32_bits(value) == _f32_bits(1.0 / 6144.0)
            for value in _constant_values(nodes, {name})
        )
    ]
    epsilon = [
        name
        for name in add_origins
        if nodes[name]["opcode"] == "constant"
        and any(
            _f32_bits(value) == _f32_bits(1e-5)
            for value in _constant_values(nodes, {name})
        )
    ]
    if (
        multiply_origins != {params[0], *reciprocal}
        or len(reciprocal) != 1
        or add_origins != {multiply_name, *epsilon}
        or len(epsilon) != 1
    ):
        return None
    return 0


def _semantic_ancestors(
    computations: Mapping[str, Mapping[str, Any]],
    computation_name: str,
    name: str,
    *,
    blocked: frozenset[str] = frozenset(),
) -> set[str]:
    """Output-sensitive ancestry through fusions, tuples and while state slots."""

    nodes = computations[computation_name]["nodes"]
    pending = [name]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited or current in blocked:
            continue
        visited.add(current)
        node = nodes[current]
        pending.extend(
            _semantic_predecessors(computations, computation_name, current)
        )
    return visited


def _selected_callee_operands(
    computations: Mapping[str, Mapping[str, Any]],
    caller: Mapping[str, Any],
    index: int | None,
) -> tuple[str, ...]:
    callee = _callee(computations, caller)
    if callee is None:
        raise RuntimeError("chunk-0 embedding HLO selected fusion has no callee")
    root = callee["nodes"][callee["root"]]
    if index is None:
        selected = callee["root"]
    else:
        if root["opcode"] != "tuple" or index >= len(root["operands"]):
            raise RuntimeError("chunk-0 embedding HLO fusion tuple selection is invalid")
        selected = root["operands"][index]
    live = _ancestors(callee["nodes"], selected)
    parameters = _parameters(callee)
    return tuple(
        operand
        for number, operand in enumerate(caller["operands"])
        if number in parameters and parameters[number] in live
    )


def _body_input_slots(
    computations: Mapping[str, Mapping[str, Any]],
    computation_name: str,
    selected: str,
    tuple_parameter: str,
) -> set[int]:
    nodes = computations[computation_name]["nodes"]
    pending = [selected]
    visited: set[str] = set()
    slots: set[int] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        node = nodes[current]
        if (
            node["opcode"] == "get-tuple-element"
            and len(node["operands"]) == 1
            and node["operands"][0] == tuple_parameter
        ):
            if node["tuple_index"] is None:
                raise RuntimeError("chunk-0 embedding HLO body tuple selection is invalid")
            slots.add(node["tuple_index"])
            continue
        pending.extend(
            _semantic_predecessors(computations, computation_name, current)
        )
    return slots


def _while_initial_operands(
    computations: Mapping[str, Mapping[str, Any]],
    computation_name: str,
    node: Mapping[str, Any],
    index: int,
) -> tuple[str, ...]:
    nodes = computations[computation_name]["nodes"]
    if len(node["operands"]) != 1 or node["body"] not in computations:
        raise RuntimeError("chunk-0 embedding HLO while body is unresolved")
    initial_name = _transparent_origin(nodes, node["operands"][0])
    initial = nodes[initial_name]
    body = computations[node["body"]]
    body_root = body["nodes"][body["root"]]
    body_parameters = _parameters(body)
    if (
        initial["opcode"] != "tuple"
        or body_root["opcode"] != "tuple"
        or len(body_parameters) != 1
        or len(initial["operands"]) != len(body_root["operands"])
        or index >= len(initial["operands"])
    ):
        raise RuntimeError("chunk-0 embedding HLO while state is malformed")
    tuple_parameter = body_parameters[0]
    transitions = tuple(
        _body_input_slots(computations, node["body"], value, tuple_parameter)
        for value in body_root["operands"]
    )
    if node["condition"] not in computations:
        raise RuntimeError("chunk-0 embedding HLO while condition is unresolved")
    condition = computations[node["condition"]]
    condition_parameters = _parameters(condition)
    if len(condition_parameters) != 1:
        raise RuntimeError("chunk-0 embedding HLO while condition is malformed")
    condition_slots = _body_input_slots(
        computations,
        node["condition"],
        condition["root"],
        condition_parameters[0],
    )
    if any(slot >= len(initial["operands"]) for slot in condition_slots):
        raise RuntimeError("chunk-0 embedding HLO while condition slot is invalid")
    required = {index, *condition_slots}
    while True:
        expanded = required | {
            dependency for slot in required for dependency in transitions[slot]
        }
        if any(slot >= len(initial["operands"]) for slot in expanded):
            raise RuntimeError("chunk-0 embedding HLO while transition slot is invalid")
        if expanded == required:
            break
        required = expanded
    return tuple(initial["operands"][slot] for slot in sorted(required))


def _semantic_predecessors(
    computations: Mapping[str, Mapping[str, Any]],
    computation_name: str,
    name: str,
) -> tuple[str, ...]:
    nodes = computations[computation_name]["nodes"]
    node = nodes[name]
    callee = _callee(computations, node)
    if node["opcode"] == "fusion" and callee is not None:
        return _selected_callee_operands(computations, node, None)
    if node["opcode"] != "get-tuple-element":
        return tuple(node["operands"])
    if len(node["operands"]) != 1 or node["tuple_index"] is None:
        raise RuntimeError("chunk-0 embedding HLO tuple selection is malformed")
    producer = nodes[node["operands"][0]]
    index = node["tuple_index"]
    if producer["opcode"] == "tuple":
        if index >= len(producer["operands"]):
            raise RuntimeError("chunk-0 embedding HLO tuple selection is out of range")
        return (producer["operands"][index],)
    if producer["opcode"] == "fusion":
        return _selected_callee_operands(computations, producer, index)
    if producer["opcode"] == "while":
        return _while_initial_operands(
            computations, computation_name, producer, index
        )
    # Async and custom calls may also return tuples.  Keeping all their data
    # operands is conservative; unlike fusion/while they have no HLO body whose
    # ignored result operands can be proven dead here.
    return tuple(producer["operands"])


def _normalization_body(
    computation: Mapping[str, Any]
) -> dict[str, int] | None:
    nodes = computation["nodes"]
    params = _parameters(computation)
    chunk = [number for number, name in params.items() if nodes[name]["shapes"] == (("bf16", (2048, 6144)),)]
    inverse = [number for number, name in params.items() if nodes[name]["shapes"] == (("f32", (2048,)),)]
    weight = [
        number
        for number, name in params.items()
        if nodes[name]["shapes"] in {(("bf16", (6144,)),), (("f32", (6144,)),)}
    ]
    if len(chunk) != 1 or len(inverse) != 1 or len(weight) != 1:
        return None
    chunk_name, inverse_name, weight_name = params[chunk[0]], params[inverse[0]], params[weight[0]]
    first_multiply = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "multiply"
        and node["shapes"] == (("f32", (2048, 6144)),)
        and {chunk_name, inverse_name} <= _ancestors(nodes, name)
        and len(node["operands"]) == 2
        and any(
            _exact_bf16_to_f32_lineage(nodes, operand, chunk_name)
            and not _exact_f32_broadcast_lineage(
                nodes,
                operand,
                inverse_name,
                parameter_shape=(2048,),
                dimensions="0",
            )
            for operand in node["operands"]
        )
        and any(
            _exact_f32_broadcast_lineage(
                nodes,
                operand,
                inverse_name,
                parameter_shape=(2048,),
                dimensions="0",
            )
            and not _exact_bf16_to_f32_lineage(nodes, operand, chunk_name)
            for operand in node["operands"]
        )
    ]
    rounded = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "convert"
        and node["shapes"] == (("bf16", (2048, 6144)),)
        and len(node["operands"]) == 1
        and _transparent_origin(
            nodes,
            node["operands"][0],
            allowed=frozenset({"bitcast", "copy", "reshape", "transpose"}),
        )
        in first_multiply
    ]
    weighted = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "multiply"
        and node["shapes"] == (("f32", (2048, 6144)),)
        and weight_name in _ancestors(nodes, name)
        and bool(set(rounded) & _ancestors(nodes, name))
        and len(node["operands"]) == 2
        and any(
            any(
                _exact_bf16_to_f32_lineage(nodes, operand, rounded_name)
                for rounded_name in rounded
            )
            and not _exact_weight_broadcast_lineage(
                nodes, operand, weight_name
            )
            for operand in node["operands"]
        )
        and any(
            _exact_weight_broadcast_lineage(nodes, operand, weight_name)
            and not any(
                _exact_bf16_to_f32_lineage(nodes, operand, rounded_name)
                for rounded_name in rounded
            )
            for operand in node["operands"]
        )
    ]
    root_live = _ancestors(nodes, computation["root"])
    representation = frozenset({"bitcast", "copy", "reshape", "transpose"})
    final_convert_name = _transparent_origin(
        nodes, computation["root"], allowed=representation
    )
    final_convert = nodes[final_convert_name]
    final_source = (
        _transparent_origin(
            nodes, final_convert["operands"][0], allowed=representation
        )
        if final_convert["opcode"] == "convert"
        and len(final_convert["operands"]) == 1
        else ""
    )
    if (
        not first_multiply
        or not rounded
        or not weighted
        or final_convert["opcode"] != "convert"
        or final_convert["shapes"] != (("bf16", (2048, 6144)),)
        or final_source not in weighted
        or not {chunk_name, inverse_name, weight_name} <= root_live
        or not set(params.values()) <= root_live
    ):
        return None
    return {"chunk": chunk[0], "inverse": inverse[0], "weight": weight[0]}


def _tiled_normalization_branch(
    computation: Mapping[str, Any], index: int
) -> dict[str, int] | None:
    """Recognize one exact M2048 RMSNorm result inside a tuple fusion."""

    nodes = computation["nodes"]
    parameters = _parameters(computation)
    root = nodes[computation["root"]]
    if root["opcode"] != "tuple" or index >= len(root["operands"]):
        return None
    logical = (32, 64)
    full = (*logical, 6144)
    representation = frozenset({"bitcast", "copy", "reshape", "transpose"})

    def origin(name: str) -> str:
        return _transparent_origin(nodes, name, allowed=representation)

    final_name = origin(root["operands"][index])
    final = nodes[final_name]
    if (
        final["opcode"] != "convert"
        or final["shapes"] != (("bf16", full),)
        or len(final["operands"]) != 1
    ):
        return None
    weighted_name = origin(final["operands"][0])
    weighted = nodes[weighted_name]
    if (
        weighted["opcode"] != "multiply"
        or weighted["shapes"] != (("f32", full),)
        or len(weighted["operands"]) != 2
    ):
        return None

    def broadcast_parameter(
        name: str, shape: tuple[int, ...], dimensions: str
    ) -> str | None:
        broadcast_name = origin(name)
        broadcast = nodes[broadcast_name]
        if (
            broadcast["opcode"] != "broadcast"
            or broadcast["shapes"] != (("f32", full),)
            or f"dimensions={{{dimensions}}}" not in broadcast["raw"]
            or len(broadcast["operands"]) != 1
        ):
            return None
        parameter = origin(broadcast["operands"][0])
        return (
            parameter
            if parameter in parameters.values()
            and nodes[parameter]["shapes"] == (("f32", shape),)
            else None
        )

    weight_candidates = [
        (operand, broadcast_parameter(operand, (6144,), "2"))
        for operand in weighted["operands"]
    ]
    weight_candidates = [item for item in weight_candidates if item[1] is not None]
    if len(weight_candidates) != 1:
        return None
    weight_operand, weight_parameter = weight_candidates[0]
    rounded_operands = [
        operand for operand in weighted["operands"] if operand != weight_operand
    ]
    if len(rounded_operands) != 1:
        return None
    rounded_f32_name = origin(rounded_operands[0])
    rounded_f32 = nodes[rounded_f32_name]
    if (
        rounded_f32["opcode"] != "convert"
        or rounded_f32["shapes"] != (("f32", full),)
        or len(rounded_f32["operands"]) != 1
    ):
        return None
    rounded_name = origin(rounded_f32["operands"][0])
    rounded = nodes[rounded_name]
    if (
        rounded["opcode"] != "convert"
        or rounded["shapes"] != (("bf16", full),)
        or len(rounded["operands"]) != 1
    ):
        return None
    scaled_name = origin(rounded["operands"][0])
    scaled = nodes[scaled_name]
    if (
        scaled["opcode"] != "multiply"
        or scaled["shapes"] != (("f32", full),)
        or len(scaled["operands"]) != 2
    ):
        return None
    inverse_candidates = [
        (operand, broadcast_parameter(operand, logical, "0,1"))
        for operand in scaled["operands"]
    ]
    inverse_candidates = [item for item in inverse_candidates if item[1] is not None]
    if len(inverse_candidates) != 1:
        return None
    inverse_operand, inverse_parameter = inverse_candidates[0]
    chunk_operands = [
        operand for operand in scaled["operands"] if operand != inverse_operand
    ]
    if len(chunk_operands) != 1:
        return None
    chunk_parameter = origin(chunk_operands[0])
    if (
        chunk_parameter not in parameters.values()
        or nodes[chunk_parameter]["shapes"] != (("f32", full),)
    ):
        return None
    selected_live = _ancestors(nodes, root["operands"][index])
    live_parameters = set(parameters.values()) & selected_live
    if live_parameters != {chunk_parameter, inverse_parameter, weight_parameter}:
        return None
    by_name = {name: number for number, name in parameters.items()}
    return {
        "chunk": by_name[chunk_parameter],
        "inverse": by_name[inverse_parameter],
        "weight": by_name[weight_parameter],
    }


def _exact_partitioned_chunk_f32_lineage(
    nodes: Mapping[str, Mapping[str, Any]], name: str, gather: str
) -> bool:
    """Accept only the TPU compiler's lossless four-slice M2048 reassembly."""

    representation = frozenset({"bitcast", "copy", "reshape", "transpose"})
    converted_name = _transparent_origin(nodes, name, allowed=representation)
    converted = nodes[converted_name]
    if (
        converted["opcode"] != "convert"
        or converted["shapes"] != (("f32", (2048, 6144)),)
        or len(converted["operands"]) != 1
    ):
        return False
    source = _transparent_origin(
        nodes, converted["operands"][0], allowed=representation
    )
    if source == gather:
        return True
    concat = nodes[source]
    if (
        concat["opcode"] != "custom-call"
        or concat["shapes"] != (("bf16", (2048, 6144)),)
        or 'custom_call_target="ConcatBitcast"' not in concat["raw"]
        or len(concat["operands"]) != 4
    ):
        return False
    observed: list[tuple[int, int]] = []
    for operand in concat["operands"]:
        done_name = _transparent_origin(
            nodes, operand, allowed=frozenset({"bitcast", "copy", "reshape", "transpose"})
        )
        done = nodes[done_name]
        # slice-done is deliberately not transparent: the corresponding
        # slice-start carries the exact range being reconstructed.
        if done["opcode"] != "slice-done" or len(done["operands"]) != 1:
            return False
        start = nodes[done["operands"][0]]
        match = re.search(
            r"\bslice=\{\[([0-9]+):([0-9]+)\], \[0:6144\]\}",
            start["raw"],
        )
        if (
            start["opcode"] != "slice-start"
            or start["shapes"][:2]
            != (("bf16", (2048, 6144)), ("bf16", (512, 6144)))
            or len(start["operands"]) != 1
            or match is None
            or _transparent_origin(nodes, start["operands"][0]) != gather
        ):
            return False
        observed.append((int(match.group(1)), int(match.group(2))))
    return tuple(observed) == (
        (0, 512),
        (512, 1024),
        (1024, 1536),
        (1536, 2048),
    )


def _exact_bf16_vector_to_f32_lineage(
    nodes: Mapping[str, Mapping[str, Any]], name: str, parameter: str
) -> bool:
    representation = frozenset(
        {"bitcast", "copy", "copy-done", "copy-start", "reshape", "transpose"}
    )
    converted_name = _transparent_origin(nodes, name, allowed=representation)
    converted = nodes[converted_name]
    return bool(
        converted["opcode"] == "convert"
        and converted["shapes"] == (("f32", (6144,)),)
        and len(converted["operands"]) == 1
        and _transparent_origin(
            nodes, converted["operands"][0], allowed=representation
        )
        == parameter
        and nodes[parameter]["shapes"] == (("bf16", (6144,)),)
    )


def _tiled_normalization_witnesses(
    computations: Mapping[str, Mapping[str, Any]],
    entry_name: str,
    gather: str,
    inverse: str,
    input_weight: str,
) -> set[str]:
    nodes = computations[entry_name]["nodes"]
    witnesses: set[str] = set()
    for name, node in nodes.items():
        if node["opcode"] != "get-tuple-element" or len(node["operands"]) != 1:
            continue
        producer = nodes[node["operands"][0]]
        if producer["opcode"] != "fusion" or node["tuple_index"] is None:
            continue
        callee = _callee(computations, producer)
        mapping = (
            None
            if callee is None
            else _tiled_normalization_branch(callee, node["tuple_index"])
        )
        if mapping is None:
            continue
        operands = {
            role: _mapped_operand(producer, mapping, role) for role in mapping
        }
        if any(value is None for value in operands.values()):
            continue
        if (
            _exact_partitioned_chunk_f32_lineage(
                nodes, operands["chunk"], gather
            )
            and _transparent_origin(nodes, operands["inverse"]) == inverse
            and _exact_bf16_vector_to_f32_lineage(
                nodes, operands["weight"], input_weight
            )
        ):
            witnesses.add(name)
    return witnesses


def _mapped_operand(
    caller: Mapping[str, Any], mapping: Mapping[str, int], role: str
) -> str | None:
    number = mapping[role]
    operands = caller["operands"]
    return operands[number] if number < len(operands) else None


def require_embedding_gather_hlo(
    main_optimized: str, main_stablehlo: str
) -> dict[str, Any]:
    """Require a live M2048 embedding gather feeding exact input RMSNorm."""

    computations, entry_name = _module_graph(main_optimized)
    entry = computations[entry_name]
    nodes = entry["nodes"]
    root = nodes[entry["root"]]
    if root["opcode"] != "tuple" or len(root["operands"]) != 12:
        raise RuntimeError("chunk-0 device embedding-gather boundary drifted")
    params = _parameters(entry)
    by_shape = lambda wanted: {
        name for name in params.values() if nodes[name]["shapes"] == (wanted,)
    }
    unique = by_shape(("bf16", (37, 6144)))
    direct = by_shape(("bf16", (2048, 6144)))
    indices = by_shape(("s32", (2048,)))
    rows = {name for name in indices if "embedding_rows" in name}
    positions = {name for name in indices if "positions" in name}
    input_weights = {
        name
        for name in params.values()
        if nodes[name]["shapes"] == (("bf16", (6144,)),)
        and "input_norm0" in name
    }

    gather_calls: list[tuple[str, dict[str, int]]] = []
    for name, node in nodes.items():
        callee = _callee(computations, node)
        mapping = None if callee is None else _gather_body(callee)
        if node["opcode"] == "fusion" and node["shapes"] == (("bf16", (2048, 6144)),) and mapping is not None:
            gather_calls.append((name, mapping))

    valid: list[tuple[str, str, str, str]] = []
    for gather_name, gather_map in gather_calls:
        gather_node = nodes[gather_name]
        table_operand = _mapped_operand(gather_node, gather_map, "table")
        index_operand = _mapped_operand(gather_node, gather_map, "indices")
        if table_operand is None or index_operand is None or not unique <= _ancestors(nodes, table_operand) or not rows <= _ancestors(nodes, index_operand):
            continue
        for rms_name, rms_node in nodes.items():
            callee = _callee(computations, rms_node)
            chunk_number = None if callee is None else _rms_body(callee)
            if rms_node["opcode"] != "fusion" or rms_node["shapes"] != (("f32", (2048,)),) or chunk_number is None:
                continue
            reduction = next(node for node in callee["nodes"].values() if node["opcode"] == "reduce" and node["shapes"] == (("f32", (2048,)),))
            region = computations.get(reduction["called"])
            if region is None or not _add_region_is_sum(region):
                continue
            rms_operand = rms_node["operands"][chunk_number] if chunk_number < len(rms_node["operands"]) else None
            if rms_operand is None or gather_name not in _ancestors(nodes, rms_operand):
                continue
            for inverse_name, inverse_node in nodes.items():
                inverse_callee = _callee(computations, inverse_node)
                inverse_number = None if inverse_callee is None else _inverse_body(inverse_callee)
                if inverse_node["opcode"] != "fusion" or inverse_number is None or inverse_number >= len(inverse_node["operands"]) or rms_name not in _ancestors(nodes, inverse_node["operands"][inverse_number]):
                    continue
                for norm_name, norm_node in nodes.items():
                    norm_callee = _callee(computations, norm_node)
                    norm_map = None if norm_callee is None else _normalization_body(norm_callee)
                    if (
                        norm_node["opcode"] != "fusion"
                        or norm_node["shapes"]
                        != (("bf16", (2048, 6144)),)
                        or norm_map is None
                    ):
                        continue
                    mapped = {role: _mapped_operand(norm_node, norm_map, role) for role in norm_map}
                    if any(value is None for value in mapped.values()):
                        continue
                    if (
                        gather_name in _ancestors(nodes, mapped["chunk"])
                        and inverse_name in _ancestors(nodes, mapped["inverse"])
                        and input_weights <= _ancestors(nodes, mapped["weight"])
                    ):
                        valid.append((gather_name, rms_name, inverse_name, norm_name))

    signature_match = re.search(r"^  func\.func public @main\((.*)\) -> .* \{$", main_stablehlo, re.M)
    if signature_match is None:
        raise RuntimeError("chunk-0 device embedding-gather boundary drifted")
    stable_parameters = signature_match.group(1)
    contract = {
        "direct_chunk_parameter_count": stable_parameters.count("tensor<2048x6144xbf16>"),
        "embedding_row_parameter_count": stable_parameters.count("tensor<2048xi32>"),
        "gather_coupled_input_rms": len(valid) == 1,
        "physical_embedding_gather_count": len(gather_calls),
        "unique_embedding_parameter_count": stable_parameters.count("tensor<37x6144xbf16>"),
    }
    if len(valid) == 1:
        norm = valid[0][3]
        # Root tuple operand 5 is the layer-0 key control compared bitwise with
        # DB518.  Operand 6 is the layer-1 key and legitimately also depends on
        # the raw residual path, so normalization cannot dominate that graph;
        # layer-1 evidence receives standing only after this control is exact.
        gather = valid[0][0]
        inverse = valid[0][2]
        input_weight = next(iter(input_weights))
        normalization_witnesses = {
            norm,
            *_tiled_normalization_witnesses(
                computations,
                entry_name,
                gather,
                inverse,
                input_weight,
            ),
        }
        key0_live = _semantic_ancestors(
            computations, entry_name, root["operands"][5]
        )
        key0_without_norm = _semantic_ancestors(
            computations,
            entry_name,
            root["operands"][5],
            blocked=frozenset(normalization_witnesses),
        )
        contract["passed"] = bool(
            normalization_witnesses & key0_live
            and gather not in key0_without_norm
        )
    else:
        contract["passed"] = False
    contract["passed"] = bool(
        contract["passed"]
        and len(unique) == 1
        and not direct
        and len(indices) == 2
        and len(rows) == 1
        and len(positions) == 1
        and len(input_weights) == 1
        and contract
        == {
            "direct_chunk_parameter_count": 0,
            "embedding_row_parameter_count": 2,
            "gather_coupled_input_rms": True,
            "passed": True,
            "physical_embedding_gather_count": 1,
            "unique_embedding_parameter_count": 1,
        }
    )
    if not contract["passed"]:
        raise RuntimeError("chunk-0 device embedding-gather boundary drifted")
    return contract
