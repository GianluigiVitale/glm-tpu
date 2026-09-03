"""HLO contracts for the isolated DB518 normalization-to-key discriminator."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any, Mapping

from .chunk0_embedding_hlo import (
    _ancestors,
    _body_input_slots,
    _callee,
    _constant_values,
    _f32_bits,
    _inverse_body,
    _mapped_operand,
    _module_graph,
    _parameters,
    _semantic_ancestors,
    _transparent_origin,
)

_NORMALIZATION_STABLEHLO_SHA256 = (
    "3cd10543ac6ea5142ebdd1424b78a45f0010665e9ceaeb3c394587d0eabda976"
)
_KEY_CONTROL_STABLEHLO_SHA256 = (
    "a2dbe03f58456cf7c9195c55251e4e25ddfcc3a08d0badb72eb59b929ab42b33"
)

_BOUNDARY_FORBIDDEN = (
    "all-gather(",
    "all-reduce(",
    "all-to-all(",
    "collective-permute(",
    "reduce-scatter(",
    "recv(",
    "send(",
    "host_callback",
    "python_callback",
    "stablehlo.all_gather",
    "stablehlo.all_reduce",
    "stablehlo.collective_permute",
    "stablehlo.recv",
    "stablehlo.send",
    "stablehlo.custom_call @xla_python",
)


def _boundary_entry(
    optimized: str,
) -> tuple[
    dict[str, dict[str, Any]],
    str,
    Mapping[str, Any],
    dict[int, str],
    set[str],
]:
    computations, entry_name = _module_graph(optimized)
    entry = computations[entry_name]
    nodes = entry["nodes"]
    root_name = entry["root"]
    params = _parameters(entry)
    live = _semantic_ancestors(computations, entry_name, root_name)
    if set(params.values()) - live:
        raise RuntimeError("completed boundary has a semantically dead input")
    return computations, entry_name, nodes[root_name], params, live


def _stable_main_signature(stablehlo: str) -> tuple[str, str]:
    match = re.search(
        r"^  func\.func public @main\((.*)\) -> \((.*)\) \{$",
        stablehlo,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("completed boundary StableHLO signature is absent")
    return match.group(1), match.group(2)


def _entry_sources(
    computations: Mapping[str, Mapping[str, Any]],
    entry_name: str,
    name: str,
    parameters: Mapping[int, str],
) -> set[str]:
    live = _semantic_ancestors(computations, entry_name, name)
    return set(parameters.values()) & live


def _require_stable_identity(
    stablehlo: str, *, module_name: str, expected_sha256: str
) -> str:
    """Bind every portable-IR operation, type, constant, SSA edge and root."""

    observed = sha256(stablehlo.encode("utf-8")).hexdigest()
    if (
        not stablehlo.startswith(f"module @{module_name} attributes ")
        or stablehlo.count("func.func public @main(") != 1
        or observed != expected_sha256
    ):
        raise RuntimeError("completed boundary StableHLO graph identity drifted")
    return observed


def _exact_gather_callee(computation: Mapping[str, Any]) -> bool:
    nodes = computation["nodes"]
    parameters = _parameters(computation)
    table = {
        name
        for name in parameters.values()
        if nodes[name]["shapes"] == (("bf16", (37, 6144)),)
    }
    rows = {
        name
        for name in parameters.values()
        if nodes[name]["shapes"] == (("s32", (2048,)),)
    }
    gathers = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "gather"
        and len(node["operands"]) == 2
        and (
            (
                "offset_dims={1}" in node["raw"]
                and "collapsed_slice_dims={0}" in node["raw"]
            )
            or (
                "offset_dims={1,2}" in node["raw"]
                and "collapsed_slice_dims={}" in node["raw"]
            )
        )
        and all(
            token in node["raw"]
            for token in (
                "start_index_map={0}",
                "index_vector_dim=1",
                "slice_sizes={1,6144}",
            )
        )
    ]
    if len(table) != 1 or len(rows) != 1 or len(gathers) != 1:
        return False
    gather = gathers[0]
    table_name = next(iter(table))
    rows_name = next(iter(rows))
    table_live = set(parameters.values()) & _ancestors(
        nodes, nodes[gather]["operands"][0]
    )
    rows_live = set(parameters.values()) & _ancestors(
        nodes, nodes[gather]["operands"][1]
    )
    root_live = _ancestors(nodes, computation["root"])
    root_shapes = nodes[computation["root"]]["shapes"]
    return bool(
        table_live == {table_name}
        and rows_live == {rows_name}
        and gather in root_live
        and set(parameters.values()) <= root_live
        and root_shapes
        in {
            (("bf16", (2048, 6144)),),
            (("bf16", (2048, 1, 6144)),),
        }
    )


def _exact_square_callee(computation: Mapping[str, Any]) -> bool:
    nodes = computation["nodes"]
    root_name = _transparent_origin(nodes, computation["root"])
    root = nodes[root_name]
    parameters = _parameters(computation)
    return bool(
        root["opcode"] == "multiply"
        and root["shapes"] == (("f32", (2048, 6144)),)
        and len(root["operands"]) == 2
        and root["operands"][0] == root["operands"][1]
        and set(parameters.values()) <= _ancestors(nodes, root_name)
    )


def _exact_inverse_callee(
    computation: Mapping[str, Any],
    *,
    length: int,
    reciprocal: float,
    epsilon: float,
) -> bool:
    nodes = computation["nodes"]
    parameters = _parameters(computation)
    if (
        len(parameters) != 1
        or nodes[parameters[0]]["shapes"] != (("f32", (length,)),)
    ):
        return False
    root = nodes[computation["root"]]
    if root["opcode"] != "rsqrt" or len(root["operands"]) != 1:
        return False
    add = nodes.get(root["operands"][0])
    if add is None or add["opcode"] != "add" or len(add["operands"]) != 2:
        return False
    multiply_names = [
        name for name in add["operands"] if nodes[name]["opcode"] == "multiply"
    ]
    if len(multiply_names) != 1:
        return False
    multiply_name = multiply_names[0]
    multiply = nodes[multiply_name]
    if len(multiply["operands"]) != 2:
        return False
    allowed = frozenset({"bitcast", "broadcast", "copy", "reshape", "transpose"})
    origin = lambda name: _transparent_origin(nodes, name, allowed=allowed)
    multiply_origins = {origin(name) for name in multiply["operands"]}
    add_origins = {origin(name) for name in add["operands"]}
    reciprocal_constants = {
        name
        for name in multiply_origins
        if nodes[name]["opcode"] == "constant"
        and any(
            _f32_bits(value) == _f32_bits(reciprocal)
            for value in _constant_values(nodes, {name})
        )
    }
    epsilon_constants = {
        name
        for name in add_origins
        if nodes[name]["opcode"] == "constant"
        and any(
            _f32_bits(value) == _f32_bits(epsilon)
            for value in _constant_values(nodes, {name})
        )
    }
    return bool(
        multiply_origins == {parameters[0], *reciprocal_constants}
        and len(reciprocal_constants) == 1
        and add_origins == {multiply_name, *epsilon_constants}
        and len(epsilon_constants) == 1
    )


def _exact_single_convert_callee(
    computation: Mapping[str, Any],
    *,
    source_shape: tuple[int, ...],
    result_shape: tuple[int, ...],
) -> bool:
    nodes = computation["nodes"]
    parameters = _parameters(computation)
    root = nodes[computation["root"]]
    return bool(
        len(parameters) == 1
        and nodes[parameters[0]]["shapes"] == (("f32", source_shape),)
        and root["opcode"] == "convert"
        and root["shapes"] == (("bf16", result_shape),)
        and len(root["operands"]) == 1
        and _transparent_origin(nodes, root["operands"][0]) == parameters[0]
    )


def _exact_dot_callee(computation: Mapping[str, Any]) -> bool:
    nodes = computation["nodes"]
    parameters = _parameters(computation)
    root = nodes[computation["root"]]
    return bool(
        len(parameters) == 2
        and nodes[parameters[0]]["shapes"] == (("f32", (64, 6144)),)
        and nodes[parameters[1]]["shapes"] == (("f32", (128, 6144)),)
        and root["opcode"] == "dot"
        and root["shapes"] == (("f32", (64, 128)),)
        and root["operands"] == (parameters[0], parameters[1])
        and "lhs_contracting_dims={1}" in root["raw"]
        and "rhs_contracting_dims={1}" in root["raw"]
        and "operand_precision={default,highest}" in root["raw"]
    )


def require_completed_normalization_boundary_hlo(
    optimized: str, stablehlo: str
) -> dict[str, Any]:
    """Require one completed BF16 M2048 gather+RMS executable boundary."""

    if any(token in optimized or token in stablehlo for token in _BOUNDARY_FORBIDDEN):
        raise RuntimeError("normalization boundary contains communication/callback")
    computations, entry_name, root, params, _ = _boundary_entry(optimized)
    nodes = computations[entry_name]["nodes"]
    parameter_shapes = tuple(
        nodes[params[index]]["shapes"] for index in range(len(params))
    )
    expected_parameters = (
        (("bf16", (37, 6144)),),
        (("s32", (2048,)),),
        (("bf16", (6144,)),),
    )
    stable_parameters, stable_result = _stable_main_signature(stablehlo)
    stable_sha256 = _require_stable_identity(
        stablehlo,
        module_name="jit_layer0_prompt_normalized_hidden_boundary_chunk",
        expected_sha256=_NORMALIZATION_STABLEHLO_SHA256,
    )
    gather_calls = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "fusion"
        and (callee := _callee(computations, node)) is not None
        and _exact_gather_callee(callee)
        and _entry_sources(computations, entry_name, name, params)
        == {params[0], params[1]}
    ]
    square_calls = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "fusion"
        and (callee := _callee(computations, node)) is not None
        and _exact_square_callee(callee)
        and _entry_sources(computations, entry_name, name, params)
        == {params[0], params[1]}
        and len(gather_calls) == 1
        and gather_calls[0]
        in _semantic_ancestors(computations, entry_name, name)
    ]
    inverse_calls = [
        name
        for name, node in nodes.items()
        if node["opcode"] == "fusion"
        and (callee := _callee(computations, node)) is not None
        and _exact_inverse_callee(
            callee, length=2048, reciprocal=1.0 / 6144.0, epsilon=1e-5
        )
        and _entry_sources(computations, entry_name, name, params)
        == {params[0], params[1]}
        and len(square_calls) == 1
        and square_calls[0]
        in _semantic_ancestors(computations, entry_name, name)
    ]
    root_callee = _callee(computations, root)
    root_callee_live = (
        set()
        if root_callee is None
        else _ancestors(root_callee["nodes"], root_callee["root"])
    )
    root_callee_parameters = (
        {} if root_callee is None else _parameters(root_callee)
    )
    root_operand_sources = tuple(
        frozenset(_entry_sources(computations, entry_name, name, params))
        for name in root["operands"]
    )
    contract = {
        "optimized_exact_gather_call_count": len(gather_calls),
        "optimized_exact_inverse_call_count": len(inverse_calls),
        "optimized_exact_square_call_count": len(square_calls),
        "optimized_parameter_shapes": parameter_shapes,
        "optimized_root_operand_sources": [
            sorted(value) for value in root_operand_sources
        ],
        "optimized_root_opcode": root["opcode"],
        "optimized_root_shapes": root["shapes"],
        "stable_f32_rms_reduce_count": stablehlo.count(
            "(tensor<2048x6144xf32>, tensor<f32>) -> tensor<2048xf32>"
        ),
        "stable_gather_count": len(
            re.findall(
                r'(?:(?:"stablehlo\.gather")|(?:stablehlo\.gather))\s*\(',
                stablehlo,
            )
        ),
        "stable_rsqrt_count": stablehlo.count("stablehlo.rsqrt"),
        "stable_sha256": stable_sha256,
    }
    contract["passed"] = bool(
        parameter_shapes == expected_parameters
        and root["shapes"] == (("bf16", (2048, 6144)),)
        and root["opcode"] == "fusion"
        and len(gather_calls) == 1
        and len(square_calls) == 1
        and len(inverse_calls) == 1
        and root_operand_sources
        == (
            frozenset({params[2]}),
            frozenset({params[0], params[1]}),
            frozenset({params[0], params[1]}),
            frozenset({params[1]}),
        )
        and root_callee is not None
        and root_callee["nodes"][root_callee["root"]]["opcode"] == "convert"
        and root_callee["nodes"][root_callee["root"]]["shapes"]
        == (("bf16", (2048, 6144)),)
        and set(root_callee_parameters.values()) <= root_callee_live
        and stable_parameters.count("tensor<37x6144xbf16>") == 1
        and stable_parameters.count("tensor<2048xi32>") == 1
        and stable_parameters.count("tensor<6144xbf16>") == 1
        and "tensor<2048x6144xbf16>" not in stable_parameters
        and "tensor<2048x6144xbf16>" in stable_result
        and contract["stable_gather_count"] == 1
        and contract["stable_f32_rms_reduce_count"] == 1
        and contract["stable_rsqrt_count"] == 1
        and "stablehlo.dot_general" not in stablehlo
        and "stablehlo.while" not in stablehlo
    )
    if not contract["passed"]:
        raise RuntimeError("completed normalization boundary HLO drifted")
    return contract


def require_normalized_key_control_boundary_hlo(
    optimized: str, stablehlo: str
) -> dict[str, Any]:
    """Require an M64 key control fed only by a completed BF16 row buffer."""

    if any(token in optimized or token in stablehlo for token in _BOUNDARY_FORBIDDEN):
        raise RuntimeError("normalized key control contains communication/callback")
    computations, entry_name, root, params, _ = _boundary_entry(optimized)
    nodes = computations[entry_name]["nodes"]
    parameter_shapes = tuple(
        nodes[params[index]]["shapes"] for index in range(len(params))
    )
    expected_parameters = (
        (("bf16", (2048, 6144)),),
        (("s32", (2048,)),),
        (("f32", (128, 6144)),),
        (("bf16", (128,)),),
        (("bf16", (128,)),),
    )
    stable_parameters, stable_result = _stable_main_signature(stablehlo)
    stable_sha256 = _require_stable_identity(
        stablehlo,
        module_name=(
            "jit_layer0_prompt_index_key_from_normalized_boundary_chunk"
        ),
        expected_sha256=_KEY_CONTROL_STABLEHLO_SHA256,
    )
    while_count = sum(
        node["opcode"] == "while"
        for computation in computations.values()
        for node in computation["nodes"].values()
    )
    entry_while = [
        (name, node)
        for name, node in nodes.items()
        if node["opcode"] == "while"
    ]
    initial_sources: tuple[frozenset[str], ...] = ()
    initial_shapes: tuple[tuple[tuple[str, tuple[int, ...]], ...], ...] = ()
    body_slot_sources: tuple[frozenset[int], ...] = ()
    exact_dot_calls: list[str] = []
    if len(entry_while) == 1:
        _, while_node = entry_while[0]
        if len(while_node["operands"]) == 1:
            initial_name = _transparent_origin(nodes, while_node["operands"][0])
            initial = nodes[initial_name]
            if initial["opcode"] == "tuple":
                initial_sources = tuple(
                    frozenset(
                        _entry_sources(
                            computations, entry_name, operand, params
                        )
                    )
                    for operand in initial["operands"]
                )
                initial_shapes = tuple(
                    nodes[operand]["shapes"] for operand in initial["operands"]
                )
            body = computations.get(while_node["body"])
            if body is not None:
                body_parameters = _parameters(body)
                body_root = body["nodes"][body["root"]]
                if len(body_parameters) == 1 and body_root["opcode"] == "tuple":
                    tuple_parameter = body_parameters[0]
                    body_slot_sources = tuple(
                        frozenset(
                            _body_input_slots(
                                computations,
                                while_node["body"],
                                operand,
                                tuple_parameter,
                            )
                        )
                        for operand in body_root["operands"]
                    )
                    for name, node in body["nodes"].items():
                        callee = _callee(computations, node)
                        if (
                            node["opcode"] == "fusion"
                            and callee is not None
                            and _exact_dot_callee(callee)
                            and _body_input_slots(
                                computations,
                                while_node["body"],
                                name,
                                tuple_parameter,
                            )
                            == {0, 2, 3}
                            and name
                            in _ancestors(body["nodes"], body_root["operands"][1])
                        ):
                            exact_dot_calls.append(name)
    root_callee = _callee(computations, root)
    root_input = (
        ""
        if len(root["operands"]) != 1
        else _transparent_origin(nodes, root["operands"][0])
    )
    concatenate = nodes.get(root_input)
    concatenate_operand_sources = (
        ()
        if concatenate is None
        else tuple(
            frozenset(
                _entry_sources(computations, entry_name, operand, params)
            )
            for operand in concatenate["operands"]
        )
    )
    contract = {
        "optimized_body_slot_sources": [
            sorted(value) for value in body_slot_sources
        ],
        "optimized_exact_dot_call_count": len(exact_dot_calls),
        "optimized_initial_shapes": initial_shapes,
        "optimized_initial_sources": [
            sorted(value) for value in initial_sources
        ],
        "optimized_parameter_shapes": parameter_shapes,
        "optimized_root_opcode": root["opcode"],
        "optimized_root_shapes": root["shapes"],
        "optimized_while_count": while_count,
        "stable_cosine_count": stablehlo.count("stablehlo.cosine"),
        "stable_dot_count": stablehlo.count("stablehlo.dot_general"),
        "stable_key_reduce_count": stablehlo.count(
            "(tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>"
        ),
        "stable_sine_count": stablehlo.count("stablehlo.sine"),
        "stable_sha256": stable_sha256,
        "stable_while_count": stablehlo.count("stablehlo.while"),
    }
    contract["passed"] = bool(
        parameter_shapes == expected_parameters
        and root["shapes"] == (("bf16", (2048, 128)),)
        and root["opcode"] == "fusion"
        and while_count == 1
        and initial_shapes
        == (
            (("s32", ()),),
            (("f32", (32, 64, 128)),),
            (("f32", (128, 6144)),),
            (("f32", (32, 64, 6144)),),
            (("f32", (128,)),),
            (("f32", (128,)),),
        )
        and initial_sources
        == (
            frozenset(),
            frozenset(),
            frozenset({params[2]}),
            frozenset({params[0]}),
            frozenset({params[3]}),
            frozenset({params[4]}),
        )
        and body_slot_sources
        == (
            frozenset({0}),
            frozenset({0, 1, 2, 3, 4, 5}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
            frozenset({5}),
        )
        and len(exact_dot_calls) == 1
        and root_callee is not None
        and _exact_single_convert_callee(
            root_callee,
            source_shape=(2048, 128),
            result_shape=(2048, 128),
        )
        and concatenate is not None
        and concatenate["opcode"] == "concatenate"
        and concatenate["shapes"] == (("f32", (2048, 128)),)
        and "dimensions={1}" in concatenate["raw"]
        and tuple(nodes[name]["shapes"] for name in concatenate["operands"])
        == ((('f32', (2048, 64)),), (('f32', (2048, 64)),))
        and concatenate_operand_sources
        == (
            frozenset(params.values()),
            frozenset({params[0], params[2], params[3], params[4]}),
        )
        and stable_parameters.count("tensor<2048x6144xbf16>") == 1
        and stable_parameters.count("tensor<2048xi32>") == 1
        and stable_parameters.count("tensor<128x6144xf32>") == 1
        and stable_parameters.count("tensor<128xbf16>") == 2
        and "tensor<37x6144xbf16>" not in stable_parameters
        and "tensor<6144xbf16>" not in stable_parameters
        and "tensor<2048x128xbf16>" in stable_result
        and contract["stable_while_count"] == 1
        and contract["stable_dot_count"] == 1
        and contract["stable_key_reduce_count"] == 2
        and contract["stable_cosine_count"] == 1
        and contract["stable_sine_count"] == 1
        and "stablehlo.gather" not in stablehlo
        and "stablehlo.scatter" not in stablehlo
        and "stablehlo.rsqrt" not in stablehlo
    )
    if not contract["passed"]:
        raise RuntimeError("normalized key-control boundary HLO drifted")
    return contract
