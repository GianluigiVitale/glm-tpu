"""Static admission for the bounded Gate-D projection discriminator source."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

EXPECTED_BUILDER_AST_SHA256 = (
    "58d6f0082b81fceee253bbd70365e7e7e687fc227a1f6925044a284237c00704"
)
EXPECTED_DIRECT_DEPENDENCY_SHA256S = {
    "glm_tpu/__init__.py": (
        "902e500f466fdaf18a637710b7d1c54d15514e3dd35d838dade74c03f6a07f06"
    ),
    "glm_tpu/greenfield/__init__.py": (
        "d09e3bc884c4044669eb5791b90b97b642f5e5a3b14c266867d9996830c30ea7"
    ),
    "glm_tpu/greenfield/benchmarking/__init__.py": (
        "7af76bb93d634aced6adc3940fd3f6c7a80cc702a8e113ab9de01c4a397e5173"
    ),
    "glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py": (
        "0f1930c079bd7244452e84dca6d0bbf9da077ea133c685f0f908760379e5313d"
    ),
    "glm_tpu/greenfield/errors.py": (
        "f7b8a079ba16f6114714124598d03ae871a2c21d365de4417b934f00cddaae64"
    ),
    "glm_tpu/greenfield/kernels/__init__.py": (
        "1345c01c8adeb22549f2b8bb42c46cc26edb760900bf80815991dfb58170f378"
    ),
    "glm_tpu/greenfield/kernels/reference/__init__.py": (
        "1f9284813df1dd38f3e8d1acfb9417da372e06c7d5336c8262348dad2f631404"
    ),
    "glm_tpu/greenfield/kernels/reference/dsa.py": (
        "c4b451ab7ca2bf79b7cc7b148a7b1b146996051891c1c82f1952d8ab9ef53fac"
    ),
    "glm_tpu/greenfield/kernels/reference/linear.py": (
        "c3d679dab63d1975f0fbfd1ceda7bd8e198621face135151f5fe05cedd07a0f6"
    ),
    "glm_tpu/greenfield/kernels/reference/rotary.py": (
        "cb17440803331de962328b409212e9647fc736590c3b525a4a2d43daa46b05b3"
    ),
}


class ProjectionContractionSourceError(RuntimeError):
    """Raised when the source or its evidence predecessor drifts."""


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ProjectionContractionSourceError("builder function catalogue drifted")
    return matches[0]


def _literal_keyword(call: ast.Call, name: str) -> Any:
    matches = [item.value for item in call.keywords if item.arg == name]
    if len(matches) != 1:
        raise ProjectionContractionSourceError(f"{name} keyword drifted")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as error:
        raise ProjectionContractionSourceError(f"{name} keyword drifted") from error


def audit_projection_contraction_pp16_source(
    source: bytes,
    *,
    dependency_sha256s: Mapping[str, str],
    predecessor: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a source-only, one-row, two-owner projection replay boundary."""

    if dict(dependency_sha256s) != EXPECTED_DIRECT_DEPENDENCY_SHA256S:
        raise ProjectionContractionSourceError("direct dependency bytes drifted")
    if (
        predecessor.get("artifact_kind")
        != "gate_d_projection_arithmetic_frontier_analysis"
        or predecessor.get("classification")
        != (
            "CPU_F32_DOT_CONTROL_EXACT_ACCEPTED_KEY_CAPTURED_INPUT;"
            "ENUMERATED_REDUCTION_PROBES_REJECTED;"
            "TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN"
        )
        or predecessor.get("variant_count") != 27
        or predecessor.get("cpu_jax_dot_control", {})
        .get("accepted", {})
        .get("bit_mismatch_count")
        != 0
        or predecessor.get("cpu_jax_dot_control", {}).get("projection_sha256")
        != "f16ad9903c6d219c9796f2a1aac32b0fd7b09135ecdfadb9a19dbef5cc1a9aa6"
        or predecessor.get("root_cause_proven") is not False
        or predecessor.get("projection_mechanism_authorized") is not False
        or predecessor.get("full_dsa_or_8k_authorized") is not False
        or predecessor.get("gate_d_closed") is not False
    ):
        raise ProjectionContractionSourceError("projection predecessor drifted")
    groups = topology.get("pp16_lp2", {}).get("groups")
    expected_group = {
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "device_ids": [0, 1],
        "process_index": 0,
        "stage_id": 0,
    }
    if (
        topology.get("artifact_kind") != "gate_d_runtime_physical_locality_authority"
        or not isinstance(groups, list)
        or len(groups) != 16
        or groups[0] != expected_group
        or topology.get("tpu_successor_authorized") is not False
    ):
        raise ProjectionContractionSourceError("PP16 topology predecessor drifted")
    try:
        tree = ast.parse(source.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ProjectionContractionSourceError("builder source is invalid") from error
    _function(tree, "build_gate_d_projection_contraction_pp16")
    local = _function(tree, "_projection_contraction_local")
    wrapper = _function(tree, "_build_projection_contraction_shard_map")
    forbidden = {
        "block_until_ready",
        "compile",
        "device_put",
        "devices",
        "distributed",
        "host_callback",
        "io_callback",
        "lower",
        "make_array_from_single_device_arrays",
        "process_allgather",
        "pure_callback",
    }
    observed_forbidden = sorted(name for name in forbidden if _calls(tree, name))
    if observed_forbidden:
        raise ProjectionContractionSourceError(
            "source contains lowering, compilation, execution, or distributed work"
        )
    barriers = _calls(local, "optimization_barrier")
    if (
        len(barriers) != 1
        or len(barriers[0].args) != 1
        or ast.unparse(barriers[0].args[0]) != "normalized_hidden_owner[0]"
    ):
        raise ProjectionContractionSourceError("normalized input barrier drifted")
    linear_calls = _calls(local, "linear")
    if len(linear_calls) != 1:
        raise ProjectionContractionSourceError("projection call catalogue drifted")
    linear_call = linear_calls[0]
    if (
        len(linear_call.args) != 2
        or not isinstance(linear_call.args[0], ast.Name)
        or linear_call.args[0].id != "normalized"
        or not isinstance(linear_call.args[1], ast.Name)
        or linear_call.args[1].id != "wk_weight"
        or len(linear_call.keywords) != 1
        or linear_call.keywords[0].arg != "output_dtype"
        or ast.unparse(linear_call.keywords[0].value) != "jnp.float32"
    ):
        raise ProjectionContractionSourceError("projection input/dtype lineage drifted")
    precision_contexts = [
        node
        for node in ast.walk(local)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "default_matmul_precision"
            and len(item.context_expr.args) == 1
            and isinstance(item.context_expr.args[0], ast.Constant)
            and item.context_expr.args[0].value == "highest"
            for item in node.items
        )
    ]
    if len(precision_contexts) != 1 or linear_call not in list(
        ast.walk(precision_contexts[0])
    ):
        raise ProjectionContractionSourceError("highest precision scope drifted")
    suffix_calls = _calls(local, "dsa_index_keys_from_projection")
    if len(suffix_calls) != 1:
        raise ProjectionContractionSourceError("key suffix call catalogue drifted")
    suffix = suffix_calls[0]
    if (
        len(suffix.args) != 4
        or not isinstance(suffix.args[0], ast.Name)
        or suffix.args[0].id != "projected_key"
        or _literal_keyword(suffix, "key_norm_mode") != "divide_sqrt"
        or ast.unparse(suffix.args[3])
        != "jnp.asarray([GATE_D_CAPSULE_POSITION], dtype=jnp.int32)"
    ):
        raise ProjectionContractionSourceError("key suffix lineage drifted")
    shard_calls = _calls(wrapper, "shard_map")
    if (
        len(shard_calls) != 1
        or _literal_keyword(shard_calls[0], "check_vma") is not True
    ):
        raise ProjectionContractionSourceError("PP16 shard_map contract drifted")
    source_text = source.decode("utf-8")
    required_fragments = (
        '(0, "tpu", "TPU v4", 0, (0, 0, 0), 0)',
        '(1, "tpu", "TPU v4", 0, (1, 0, 0), 0)',
        'axis_name: str = "feature"',
        "normalized_hidden_owner.shape != (1, 1, contract.hidden_size)",
        "wk_weight_owner.shape != (1, contract.head_dim, contract.hidden_size)",
        "in_specs=(owner_matrix, owner_matrix, owner_vector, owner_vector)",
    )
    if any(source_text.count(fragment) != 1 for fragment in required_fragments):
        raise ProjectionContractionSourceError("static PP16/input contract drifted")
    result_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GateDProjectionContractionPp16Result"
    ]
    fields = (
        tuple(
            node.target.id
            for node in result_classes[0].body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        )
        if len(result_classes) == 1
        else ()
    )
    if fields != (
        "normalized_hidden_owners",
        "projected_key_owners",
        "current_key_owners",
    ):
        raise ProjectionContractionSourceError("rooted output contract drifted")
    result_calls = _calls(tree, "GateDProjectionContractionPp16Result")
    result_signatures = {
        tuple(ast.unparse(argument) for argument in call.args) for call in result_calls
    }
    if result_signatures != {
        (
            "normalized[None, ...]",
            "projected_key[None, ...]",
            "current_key[None, ...]",
        ),
        ("owner_matrix", "owner_matrix", "owner_matrix"),
    }:
        raise ProjectionContractionSourceError("rooted output lineage drifted")
    collective_names = (
        "all_gather",
        "all_to_all",
        "ppermute",
        "pshuffle",
        "psum",
        "psum_scatter",
    )
    collective_call_count = sum(len(_calls(tree, name)) for name in collective_names)
    if collective_call_count:
        raise ProjectionContractionSourceError(
            "projection source contains a collective"
        )
    builder_ast_sha256 = sha256(
        ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()
    if builder_ast_sha256 != EXPECTED_BUILDER_AST_SHA256:
        raise ProjectionContractionSourceError("complete builder module AST drifted")
    return {
        "builder_ast_sha256": builder_ast_sha256,
        "collective_call_count": collective_call_count,
        "compiled_executable_invocation_count": 0,
        "direct_dependency_sha256s": dict(sorted(dependency_sha256s.items())),
        "input_owner_shapes": {
            "key_norm_bias": [2, 128],
            "key_norm_weight": [2, 128],
            "normalized_hidden": [2, 1, 6144],
            "wk_weight": [2, 128, 6144],
        },
        "one_live_row": True,
        "output_fields": list(fields),
        "pp16_stage_zero": expected_group,
        "projection_call_count": 1,
        "projection_dtype": "float32",
        "projection_precision": "highest",
        "suffix_key_norm_mode": "divide_sqrt",
        "tpu_compile_or_execution_performed": False,
    }
