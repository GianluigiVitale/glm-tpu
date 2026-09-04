"""Fail-closed HLO contract for the completed-normalization layer consumer."""

from __future__ import annotations

from typing import Any, Mapping

from .chunk0_embedding_hlo import (
    _ancestors,
    _body_input_slots,
    _callee,
    _parameters,
    _semantic_ancestors,
    _transparent_origin,
)
from .original_db518_normalized_boundary_hlo import (
    _BOUNDARY_FORBIDDEN,
    _boundary_entry,
    _entry_sources,
    _optimized_single_device_placement,
    _require_stable_identity,
    _stable_main_signature,
)

_REAL_CONSUMER_STABLEHLO_SHA256 = (
    "fdc207c69be05b936c4a16b65a5970708ce9193e850d085ee6b2606b5c15b44e"
)


def _replicated_axes(rank: int) -> str:
    return "[" + ", ".join("{}" for _ in range(rank)) + "]"


_REAL_CONSUMER_PARAMETER_SHARDINGS = (
    _replicated_axes(2),
    _replicated_axes(1),
    None,
    _replicated_axes(1),
    _replicated_axes(3),
    _replicated_axes(3),
    _replicated_axes(3),
    _replicated_axes(3),
    _replicated_axes(1),
    _replicated_axes(1),
    _replicated_axes(1),
    _replicated_axes(1),
    _replicated_axes(3),
    _replicated_axes(3),
    _replicated_axes(1),
    _replicated_axes(1),
    _replicated_axes(3),
    _replicated_axes(3),
    _replicated_axes(3),
    _replicated_axes(3),
    _replicated_axes(2),
    _replicated_axes(4),
    _replicated_axes(4),
    _replicated_axes(2),
)

_EXPECTED_PARAMETERS = (
    (("bf16", (37, 6144)),),
    (("s32", (2048,)),),
    (("bf16", (2048, 6144)),),
    (("s32", (2048,)),),
    (("u8", (32, 384, 6144)),),
    (("f32", (32, 3, 48)),),
    (("u8", (32, 6144, 768)),),
    (("f32", (32, 48, 6)),),
    (("bf16", (6144,)),),
    (("bf16", (128,)),),
    (("bf16", (128,)),),
    (("bf16", (512,)),),
    (("u8", (32, 512, 6144)),),
    (("f32", (32, 4, 48)),),
    (("bf16", (6144,)),),
    (("bf16", (2048,)),),
    (("u8", (32, 2048, 512)),),
    (("f32", (32, 16, 4)),),
    (("u8", (32, 6144, 82)),),
    (("f32", (32, 48, 82)),),
    (("bf16", (256, 64)),),
    (("bf16", (32, 2, 192, 512)),),
    (("bf16", (32, 2, 512, 256)),),
    (("f32", (128, 6144)),),
)

_EXPECTED_ROOT_SHAPES = (
    ("bf16", (6144,)),
    ("bf16", (6144,)),
    ("bf16", (6144,)),
    ("bf16", (6144,)),
    ("bf16", (6144,)),
    ("bf16", (2048, 128)),
    ("bf16", (512,)),
    ("bf16", (6144,)),
    ("bf16", (2048, 6144)),
    ("bf16", (6144,)),
    ("bf16", (2048,)),
)

_PARAMETER_MARKERS = (
    "unique_embeddings",
    "embedding_rows",
    "normalized0",
    "positions",
    "weights__down_bits__",
    "weights__down_scale__",
    "weights__gu_bits__",
    "weights__gu_scale__",
    "weights__input_norm1__",
    "weights__k_norm1_bias__",
    "weights__k_norm1_weight__",
    "weights__kv_a_norm__",
    "weights__o_bits__",
    "weights__o_scale__",
    "weights__post_norm__",
    "weights__q_a_norm__",
    "weights__qb_bits__",
    "weights__qb_scale__",
    "weights__qkv_bits__",
    "weights__qkv_scale__",
    "weights__rope_table__",
    "weights__w_uk_t__",
    "weights__w_uv__",
    "weights__wk1__",
)


def _qkv_projection_contract(
    computations: Mapping[str, Mapping[str, Any]],
    entry_name: str,
    key_root: str,
    params: Mapping[int, str],
) -> dict[str, Any]:
    """Bind the completed buffer to the output-live layer-0 QKV projection.

    Merely finding every ENTRY parameter somewhere under ``key_root`` is not
    sufficient: a dead exact QKV computation plus an alternate all-input root,
    or a live raw-embedding QKV path plus a normalized-input decoy, would pass
    that test.  This contract identifies the selected QKV while result, proves
    the contraction is live in its body, binds each persistent input slot to
    its sole ENTRY source, and then cuts the selected result from the returned
    key.  After the cut, none of the normalized/QKV inputs may remain live.
    """

    entry_nodes = computations[entry_name]["nodes"]
    key_live = _semantic_ancestors(computations, entry_name, key_root)
    candidates: list[dict[str, Any]] = []
    for selected_name, selected in entry_nodes.items():
        if (
            selected_name not in key_live
            or selected["opcode"] != "get-tuple-element"
            or selected["tuple_index"] != 1
            or selected["shapes"] != (("bf16", (32, 2048, 82)),)
            or len(selected["operands"]) != 1
        ):
            continue
        while_name = selected["operands"][0]
        while_node = entry_nodes[while_name]
        if (
            while_node["opcode"] != "while"
            or len(while_node["operands"]) != 1
            or while_node["body"] not in computations
            or while_node["condition"] not in computations
        ):
            continue
        initial_name = _transparent_origin(entry_nodes, while_node["operands"][0])
        initial = entry_nodes[initial_name]
        body_name = while_node["body"]
        body = computations[body_name]
        body_nodes = body["nodes"]
        body_params = _parameters(body)
        body_root = body_nodes[body["root"]]
        if (
            initial["opcode"] != "tuple"
            or len(initial["operands"]) != 6
            or len(body_params) != 1
            or body_root["opcode"] != "tuple"
            or len(body_root["operands"]) != 6
        ):
            continue
        slot_shapes = tuple(entry_nodes[name]["shapes"] for name in initial["operands"])
        if not (
            slot_shapes[0] == (("s32", ()),)
            and slot_shapes[1] == (("bf16", (32, 2048, 82)),)
            and slot_shapes[2] == (("u8", (32, 6144, 82)),)
            and slot_shapes[3] == (("f32", (32, 48, 82)),)
            and slot_shapes[4]
            in {
                (("bf16", (2048, 6144)),),
                (("f32", (2048, 6144)),),
            }
            and slot_shapes[5] in {(("s32", ()),), (("s32", (6144,)),)}
        ):
            continue
        slot_sources = tuple(
            frozenset(_entry_sources(computations, entry_name, operand, params))
            for operand in initial["operands"]
        )
        if slot_sources != (
            frozenset(),
            frozenset(),
            frozenset({params[18]}),
            frozenset({params[19]}),
            frozenset({params[2]}),
            frozenset(),
        ):
            continue

        body_tuple_parameter = body_params[0]
        transition_name = body_root["operands"][1]
        transition = body_nodes[transition_name]
        transition_live = _semantic_ancestors(computations, body_name, transition_name)
        transition_slots = _body_input_slots(
            computations, body_name, transition_name, body_tuple_parameter
        )
        projection_kind = ""
        projection_slots: set[int] = set()
        direct = [
            name
            for name in transition_live
            if body_nodes[name]["opcode"] == "dot"
            and body_nodes[name]["shapes"] == (("f32", (2048, 82)),)
            and "lhs_contracting_dims={1}" in body_nodes[name]["raw"]
            and "rhs_contracting_dims={0}" in body_nodes[name]["raw"]
        ]
        if len(direct) == 1:
            projection_kind = "body_dot"
            projection_slots = _body_input_slots(
                computations, body_name, direct[0], body_tuple_parameter
            )
        else:
            callee = _callee(computations, transition)
            if transition["opcode"] == "fusion" and callee is not None:
                callee_nodes = callee["nodes"]
                callee_live = _ancestors(callee_nodes, callee["root"])
                convolutions = [
                    name
                    for name in callee_live
                    if callee_nodes[name]["opcode"] == "convolution"
                    and callee_nodes[name]["shapes"] == (("f32", (2048, 82)),)
                    and "dim_labels=bf_io->bf" in callee_nodes[name]["raw"]
                ]
                if len(convolutions) == 1:
                    callee_params = _parameters(callee)
                    convolution_live = _ancestors(callee_nodes, convolutions[0])
                    used_numbers = {
                        number
                        for number, parameter in callee_params.items()
                        if parameter in convolution_live
                    }
                    if all(
                        number < len(transition["operands"]) for number in used_numbers
                    ):
                        projection_kind = "fused_convolution"
                        projection_slots = set().union(
                            *(
                                _body_input_slots(
                                    computations,
                                    body_name,
                                    transition["operands"][number],
                                    body_tuple_parameter,
                                )
                                for number in used_numbers
                            )
                        )
        if (
            projection_kind
            and projection_slots in ({0, 2, 3, 4}, {0, 2, 3, 4, 5})
            and transition_slots in ({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5})
        ):
            candidates.append(
                {
                    "projection_kind": projection_kind,
                    "projection_slots": sorted(projection_slots),
                    "selected_name": selected_name,
                    "slot_shapes": slot_shapes,
                    "transition_slots": sorted(transition_slots),
                }
            )
    if len(candidates) != 1:
        raise RuntimeError(
            "real layer consumer has no unique output-live QKV projection"
        )
    witness = candidates[0]
    selected_name = witness["selected_name"]
    selected_sources = frozenset(
        _entry_sources(computations, entry_name, selected_name, params)
    )
    expected_sources = frozenset({params[2], params[18], params[19]})
    cut_live = _semantic_ancestors(
        computations,
        entry_name,
        key_root,
        blocked=frozenset({selected_name}),
    )
    cut_sources = frozenset(params.values()) & cut_live
    if selected_sources != expected_sources or expected_sources & cut_sources:
        raise RuntimeError("real layer consumer normalized/QKV path is bypassable")
    return {
        **witness,
        "cut_sources": sorted(cut_sources),
        "selected_sources": sorted(selected_sources),
        "selected_source_roles_exact": True,
        "selected_result_dominates_normalized_qkv_sources": True,
    }


def require_real_layer_consumer_hlo(optimized: str, stablehlo: str) -> dict[str, Any]:
    """Prove the real layer consumes, rather than recomputes, input RMS.

    The canonical StableHLO pins the complete source graph.  The optimized
    contract separately proves that every input, especially the completed
    BF16 normalization parameter, remains live into the layer-1 key root.
    """

    if any(token in optimized or token in stablehlo for token in _BOUNDARY_FORBIDDEN):
        raise RuntimeError("real layer consumer contains communication/callback")
    canonical, stable_identity = _require_stable_identity(
        stablehlo,
        module_name=("jit_legacy_geometry_chunk_consumer_gather_from_normalized"),
        expected_sha256=_REAL_CONSUMER_STABLEHLO_SHA256,
        parameter_shardings=_REAL_CONSUMER_PARAMETER_SHARDINGS,
    )
    stable_parameters, stable_result = _stable_main_signature(canonical)
    computations, entry_name, root, params, root_live = _boundary_entry(optimized)
    nodes = computations[entry_name]["nodes"]
    parameter_shapes = tuple(
        nodes[params[index]]["shapes"] for index in range(len(params))
    )
    parameter_names_bound = tuple(
        marker in params[index] for index, marker in enumerate(_PARAMETER_MARKERS)
    )
    root_sources = tuple(
        frozenset(_entry_sources(computations, entry_name, operand, params))
        for operand in root["operands"]
    )
    all_parameters = frozenset(params.values())
    keys1_sources = root_sources[5] if len(root_sources) > 5 else frozenset()
    normalized_echo_sources = root_sources[7] if len(root_sources) > 7 else frozenset()
    qkv_projection = _qkv_projection_contract(
        computations,
        entry_name,
        root["operands"][5],
        params,
    )
    if stable_identity["stable_single_device_sharding_bound"]:
        optimized_backend_form = "tpu_v4_single_device"
        optimized_placement_bound = _optimized_single_device_placement(
            optimized,
            nodes,
            params,
            _REAL_CONSUMER_PARAMETER_SHARDINGS,
        )
    else:
        optimized_backend_form = "cpu_unsharded"
        optimized_placement_bound = True
    contract = {
        "optimized_backend_form": optimized_backend_form,
        "optimized_keys1_source_count": len(keys1_sources),
        "optimized_keys1_sources_all_parameters": (keys1_sources == all_parameters),
        "optimized_normalized_echo_sources": sorted(normalized_echo_sources),
        "optimized_parameter_names_bound": all(parameter_names_bound),
        "optimized_parameter_shapes": parameter_shapes,
        "optimized_placement_bound": optimized_placement_bound,
        "optimized_qkv_projection": qkv_projection,
        "optimized_root_live_count": len(root_live),
        "optimized_root_opcode": root["opcode"],
        "optimized_root_shapes": root["shapes"],
        "stable_convolution_count": canonical.count("stablehlo.convolution"),
        "stable_dot_general_count": canonical.count("stablehlo.dot_general"),
        "stable_gather_count": canonical.count("stablehlo.gather"),
        "stable_hidden_rms_reduce_count": canonical.count(
            "(tensor<2048x6144xf32>, tensor<f32>) -> tensor<2048xf32>"
        ),
        "stable_rsqrt_count": canonical.count("stablehlo.rsqrt"),
        "stable_while_count": canonical.count("stablehlo.while"),
        **stable_identity,
    }
    contract["passed"] = bool(
        parameter_shapes == _EXPECTED_PARAMETERS
        and len(params) == len(_REAL_CONSUMER_PARAMETER_SHARDINGS)
        and all(parameter_names_bound)
        and "input_norm0" not in optimized
        and "input_norm0" not in canonical
        and root["opcode"] == "tuple"
        and root["shapes"] == _EXPECTED_ROOT_SHAPES
        and len(root["operands"]) == len(_EXPECTED_ROOT_SHAPES)
        and keys1_sources == all_parameters
        and normalized_echo_sources == {params[2]}
        and qkv_projection["selected_source_roles_exact"]
        and qkv_projection["selected_result_dominates_normalized_qkv_sources"]
        and optimized_placement_bound
        and stable_parameters.count("tensor<37x6144xbf16>") == 1
        and stable_parameters.count("tensor<2048x6144xbf16>") == 1
        and stable_parameters.count("tensor<2048xi32>") == 2
        and stable_result.count("tensor<2048x128xbf16>") == 1
        and contract["stable_hidden_rms_reduce_count"] == 2
        and contract["stable_rsqrt_count"] == 4
        and contract["stable_gather_count"] == 14
        and contract["stable_dot_general_count"] == 6
        and contract["stable_convolution_count"] == 5
        and contract["stable_while_count"] == 4
    )
    if not contract["passed"]:
        raise RuntimeError("real layer consumer HLO drifted")
    return contract
