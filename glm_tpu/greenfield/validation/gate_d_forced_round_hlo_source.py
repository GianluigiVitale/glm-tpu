"""Static fail-closed audit for the forced-round PP16 compile-only builder."""

from __future__ import annotations

import ast
from hashlib import sha256
from typing import Any


class ForcedRoundHloSourceError(RuntimeError):
    """Raised when compile-only source structure drifts from its contract."""


EXPECTED_BUILDER_MODULE_AST_SHA256 = (
    "6c52174e0f69f8c33203f8ae42e4f9da86f33d3ca53eea31b266186b761e814f"
)


def _call_name(node: ast.Call) -> str:
    value: ast.expr = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _assigned_call(function: ast.FunctionDef, name: str) -> ast.Call:
    matches = [
        node.value
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
        and isinstance(node.value, ast.Call)
    ]
    if len(matches) != 1:
        raise ForcedRoundHloSourceError(f"forced-round assignment drifted: {name}")
    return matches[0]


def _assigned_value(function: ast.FunctionDef, name: str) -> ast.expr:
    matches = [
        node.value
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise ForcedRoundHloSourceError(f"forced-round assignment drifted: {name}")
    return matches[0]


def _matches_expression(node: ast.AST, expected: str) -> bool:
    expected_node = ast.parse(expected, mode="eval").body
    return ast.dump(node, include_attributes=False) == ast.dump(
        expected_node, include_attributes=False
    )


def _is_forbidden_runtime_call(node: ast.Call) -> bool:
    forbidden = {"block_until_ready", "compile", "execute", "lower"}
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in forbidden
    if isinstance(node.func, ast.Name):
        return node.func.id in forbidden
    return False


def audit_forced_round_pp16_hlo_source(source_raw: bytes) -> dict[str, Any]:
    """Prove the isolated builder lineages without importing or lowering JAX."""

    try:
        source = source_raw.decode("utf-8", errors="strict")
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ForcedRoundHloSourceError(
            "forced-round compile-only source is not canonical Python"
        ) from error
    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_gate_d_forced_round_pp16_hlo_replay"
    ]
    results = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GateDForcedRoundPp16ReplayResult"
    ]
    if len(builders) != 1 or len(results) != 1:
        raise ForcedRoundHloSourceError(
            "forced-round builder/result definition drifted"
        )
    builder = builders[0]
    mapped = [
        node
        for node in builder.body
        if isinstance(node, ast.FunctionDef) and node.name == "mapped"
    ]
    if len(mapped) != 1:
        raise ForcedRoundHloSourceError("forced-round mapped body drifted")
    mapped_function = mapped[0]

    runtime_devices = _assigned_value(builder, "runtime_devices")
    expected_stage_zero = _assigned_value(builder, "expected_stage_zero")
    observed_stage_zero = _assigned_value(builder, "observed_stage_zero")
    mesh = _assigned_value(builder, "mesh")
    cache_layout = _assigned_value(builder, "cache_layout")
    groups = _assigned_value(builder, "groups")
    owner_matrix = _assigned_value(builder, "owner_matrix")
    owner_vector = _assigned_value(builder, "owner_vector")
    expected_assignments = (
        (runtime_devices, "tuple(devices)"),
        (
            expected_stage_zero,
            (
                "((0, 'tpu', 'TPU v4', 0, (0, 0, 0), 0), "
                "(1, 'tpu', 'TPU v4', 0, (1, 0, 0), 0))"
            ),
        ),
        (
            observed_stage_zero,
            (
                "tuple((int(device.id), str(device.platform), "
                "str(device.device_kind), int(device.process_index), "
                "tuple(int(coordinate) for coordinate in device.coords), "
                "int(device.core_on_chip)) for device in runtime_devices)"
            ),
        ),
        (mesh, "Mesh(np.asarray(runtime_devices, dtype=object), (axis_name,))"),
        (
            cache_layout,
            (
                "StageLocalKvLayout(logical_page_size="
                "GATE_D_CAPSULE_LOGICAL_PAGE_SIZE, local_parallel_size=2, "
                "packed_cache_width=640)"
            ),
        ),
        (groups, "((0, 1),)"),
        (owner_matrix, "P(axis_name, None, None)"),
        (owner_vector, "P(axis_name, None)"),
    )
    if any(
        not _matches_expression(value, expected)
        for value, expected in expected_assignments
    ):
        raise ForcedRoundHloSourceError("forced-round platform/group contract drifted")
    required_guards = (
        "len(runtime_devices) != 2 or len({device.id for device in runtime_devices}) != 2",
        "observed_stage_zero != expected_stage_zero",
        "axis_name != 'feature'",
    )
    builder_guards = [node.test for node in builder.body if isinstance(node, ast.If)]
    if any(
        sum(_matches_expression(guard, expected) for guard in builder_guards) != 1
        for expected in required_guards
    ):
        raise ForcedRoundHloSourceError("forced-round platform/group contract drifted")

    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 2
        and node.module == "kernels.reference.rmsnorm"
    ]
    if len(imports) != 1 or [alias.name for alias in imports[0].names] != [
        "fused_add_rms_norm_with_forced_bf16_boundary"
    ]:
        raise ForcedRoundHloSourceError("forced-round kernel import drifted")

    forced_assignments = [
        node
        for node in mapped_function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Tuple)
        and [item.id for item in node.targets[0].elts if isinstance(item, ast.Name)]
        == ["normalized", "_"]
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "fused_add_rms_norm_with_forced_bf16_boundary"
    ]
    if len(forced_assignments) != 1:
        raise ForcedRoundHloSourceError("forced-round normalized assignment drifted")
    forced_call = forced_assignments[0].value
    if (
        len(forced_call.args) != 3
        or [item.id for item in forced_call.args if isinstance(item, ast.Name)]
        != ["hidden_update", "residual", "rms_weight"]
        or len(forced_call.keywords) != 1
        or forced_call.keywords[0].arg != "epsilon"
        or not isinstance(forced_call.keywords[0].value, ast.Constant)
        or forced_call.keywords[0].value.value != 1e-5
    ):
        raise ForcedRoundHloSourceError("forced-round kernel arguments drifted")

    qkv_call = _assigned_call(mapped_function, "qkv_a")
    if (
        _call_name(qkv_call) != "one_row_fused_qkv_a_convolution"
        or not qkv_call.args
        or not isinstance(qkv_call.args[0], ast.Name)
        or qkv_call.args[0].id != "normalized"
    ):
        raise ForcedRoundHloSourceError("forced-round QKV lineage drifted")
    dsa_call = _assigned_call(mapped_function, "dsa")
    if _call_name(dsa_call) != "stage_local_dsa_fp8_mapped":
        raise ForcedRoundHloSourceError("forced-round DSA call drifted")
    dsa_keywords = {keyword.arg: keyword.value for keyword in dsa_call.keywords}
    expected_dsa_keywords = {
        "axis_name": "axis_name",
        "contract": "dsa_contract",
        "cache_layout": "cache_layout",
        "axis_index_groups": "groups",
        "precomputed_normalized": "normalized",
        "precomputed_q_residual": "qkv_a.q_residual",
        "linear_backend": "'pallas'",
        "dsa_query_backend": "'reference'",
        "dsa_query_weight_aliases": "(wq_b_weight[0],) * 4",
        "precomputed_wk_weight": "wk_weight[0]",
        "dsa_head_key_exact_association": "True",
        "dsa_score_precision": "'default'",
    }
    if set(dsa_keywords) != set(expected_dsa_keywords) or any(
        not _matches_expression(dsa_keywords[name], expected)
        for name, expected in expected_dsa_keywords.items()
    ):
        raise ForcedRoundHloSourceError("forced-round DSA input lineage drifted")

    mapped_returns = [
        node for node in mapped_function.body if isinstance(node, ast.Return)
    ]
    if len(mapped_returns) != 1 or not isinstance(mapped_returns[0].value, ast.Call):
        raise ForcedRoundHloSourceError("forced-round rooted output drifted")
    result_call = mapped_returns[0].value
    expected_result_arguments = (
        "normalized[None, ...]",
        "dsa.internals.query[None, ...]",
        "dsa.internals.head_weights[None, ...]",
        "dsa.internals.current_key[None, ...]",
        "dsa.index_cache[None, ...]",
        "dsa.selected_positions[None, ...]",
        "dsa.valid_counts[None, ...]",
        "dsa.selected_scores[None, ...]",
        "jnp.asarray(dsa.contract_valid, dtype=jnp.uint8).reshape(1)",
    )
    if _call_name(result_call) != "GateDForcedRoundPp16ReplayResult" or len(
        result_call.args
    ) != len(expected_result_arguments):
        raise ForcedRoundHloSourceError("forced-round normalized output is not rooted")
    if any(
        not _matches_expression(argument, expected)
        for argument, expected in zip(
            result_call.args, expected_result_arguments, strict=True
        )
    ):
        raise ForcedRoundHloSourceError("forced-round primary output lineage drifted")

    result_fields = [
        node.target.id
        for node in results[0].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    expected_fields = [
        "normalized_hidden_owners",
        "query_owners",
        "head_weights_owners",
        "current_key_owners",
        "index_cache_owners",
        "selected_positions_owners",
        "valid_counts_owners",
        "selected_scores_owners",
        "contract_valid_owners",
    ]
    if result_fields != expected_fields:
        raise ForcedRoundHloSourceError("forced-round result field order drifted")

    all_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_names = [_call_name(node) for node in all_calls]
    if any(_is_forbidden_runtime_call(node) for node in all_calls):
        raise ForcedRoundHloSourceError(
            "forced-round source performs lowering, compilation, or execution"
        )

    builder_returns = [node for node in builder.body if isinstance(node, ast.Return)]
    if len(builder_returns) != 1 or not isinstance(builder_returns[0].value, ast.Call):
        raise ForcedRoundHloSourceError("forced-round JIT/shard_map wrapper drifted")
    jit_call = builder_returns[0].value
    if (
        _call_name(jit_call) != "jax.jit"
        or len(jit_call.args) != 1
        or jit_call.keywords
        or not isinstance(jit_call.args[0], ast.Call)
    ):
        raise ForcedRoundHloSourceError("forced-round JIT/shard_map wrapper drifted")
    shard_map_call = jit_call.args[0]
    shard_map_keywords = {
        keyword.arg: keyword.value for keyword in shard_map_call.keywords
    }
    expected_in_specs = (
        "(P(), P(), P(), P(axis_name, None, None, None), P(), P(), P(), "
        "owner_matrix, owner_matrix, owner_matrix, owner_matrix, owner_matrix, "
        "owner_matrix, owner_vector, owner_vector, owner_matrix)"
    )
    expected_out_specs = (
        "GateDForcedRoundPp16ReplayResult(P(axis_name, None, None), "
        "P(axis_name, None, None, None), P(axis_name, None, None), "
        "P(axis_name, None, None), P(axis_name, None, None, None), "
        "P(axis_name, None, None), P(axis_name, None), "
        "P(axis_name, None, None), P(axis_name))"
    )
    if (
        _call_name(shard_map_call) != "jax.shard_map"
        or len(shard_map_call.args) != 1
        or not isinstance(shard_map_call.args[0], ast.Name)
        or shard_map_call.args[0].id != "mapped"
        or set(shard_map_keywords) != {"mesh", "in_specs", "out_specs", "check_vma"}
        or not _matches_expression(shard_map_keywords["mesh"], "mesh")
        or not _matches_expression(shard_map_keywords["in_specs"], expected_in_specs)
        or not _matches_expression(shard_map_keywords["out_specs"], expected_out_specs)
        or not _matches_expression(shard_map_keywords["check_vma"], "False")
    ):
        raise ForcedRoundHloSourceError("forced-round JIT/shard_map wrapper drifted")

    if call_names.count("fused_add_rms_norm_with_forced_bf16_boundary") != 1:
        raise ForcedRoundHloSourceError("forced-round kernel call count drifted")
    if call_names.count("jax.jit") != 1 or call_names.count("jax.shard_map") != 1:
        raise ForcedRoundHloSourceError("forced-round JIT/shard_map wrapper drifted")
    module_ast_sha256 = sha256(
        ast.dump(tree, annotate_fields=True, include_attributes=False).encode("ascii")
    ).hexdigest()
    if module_ast_sha256 != EXPECTED_BUILDER_MODULE_AST_SHA256:
        raise ForcedRoundHloSourceError("forced-round builder module AST drifted")
    return {
        "builder": "build_gate_d_forced_round_pp16_hlo_replay",
        "builder_source_sha256": sha256(
            ast.get_source_segment(source, builder).encode("utf-8")
        ).hexdigest(),
        "forced_round_call_count": 1,
        "hlo_lowering_or_compile_call_count": 0,
        "in_spec_count": 16,
        "module_ast_sha256": module_ast_sha256,
        "normalized_feeds_dsa": True,
        "normalized_feeds_qkv": True,
        "normalized_is_rooted": True,
        "output_field_count": 9,
        "pp16_group": [0, 1],
        "pp16_stage_coordinates": [[0, 0, 0], [1, 0, 0]],
        "pp16_stage_id": 0,
        "pp16_stage_process_index": 0,
        "source_sha256": sha256(source_raw).hexdigest(),
        "tpu_device_count_required": 2,
    }
