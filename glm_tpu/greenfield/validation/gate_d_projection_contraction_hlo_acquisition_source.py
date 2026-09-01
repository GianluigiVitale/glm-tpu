"""Static admission for the projection-only PP16 HLO acquirer."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

EXPECTED_ACQUIRER_AST_SHA256 = (
    "42596a1781b0c0d6650a528d7e7141dc5205ca73848020a7aa6afea991973e6f"
)
EXPECTED_SOURCE_SHA256 = (
    "5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f"
)
EXPECTED_TOPOLOGY_SHA256 = (
    "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
)
EXPECTED_INPUT_SPEC = (
    ("normalized_hidden_bf16", (2, 1, 6144), "bfloat16"),
    ("wk_weight_fp32", (2, 128, 6144), "float32"),
    ("key_norm_weight_bf16", (2, 128), "bfloat16"),
    ("key_norm_bias_bf16", (2, 128), "bfloat16"),
)
EXPECTED_OUTPUT_SPEC = (
    ("normalized_hidden_owners", (2, 1, 6144), "bfloat16"),
    ("projected_key_owners", (2, 1, 128), "float32"),
    ("current_key_owners", (2, 1, 128), "float32"),
)


class ProjectionContractionHloSourceError(RuntimeError):
    """Raised when compile-only source or its predecessors drift."""


def _assignment(tree: ast.Module, name: str) -> ast.expr:
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise ProjectionContractionHloSourceError(f"{name} assignment drifted")
    return matches[0]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ProjectionContractionHloSourceError(f"{name} function drifted")
    return matches[0]


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


def _literal(tree: ast.Module, name: str) -> Any:
    try:
        return ast.literal_eval(_assignment(tree, name))
    except (ValueError, TypeError) as error:
        raise ProjectionContractionHloSourceError(f"{name} literal drifted") from error


def audit_projection_contraction_hlo_acquisition_source(
    source: bytes,
    *,
    source_authority: Mapping[str, Any],
    source_authority_sha256: str,
    topology: Mapping[str, Any],
    topology_sha256: str,
) -> dict[str, Any]:
    """Prove one compile-only, never-invoked projection HLO boundary."""

    if source_authority_sha256 != EXPECTED_SOURCE_SHA256:
        raise ProjectionContractionHloSourceError("source authority digest drifted")
    if (
        source_authority.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_source"
        or source_authority.get("classification")
        != (
            "PROJECTION_ONLY_PP16_SOURCE_ACCEPTED;COMPILE_UNPROVEN;"
            "TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN"
        )
        or source_authority.get("authorization")
        != {
            "full_dsa_or_8k": False,
            "hlo_acquisition": False,
            "persistence_only": True,
            "tpu_compile": False,
            "tpu_execution": False,
        }
        or source_authority.get("gate_d_closed") is not False
        or source_authority.get("performance_claim") is not False
        or source_authority.get("tpu_compile_or_execution_performed") is not False
    ):
        raise ProjectionContractionHloSourceError("source authority drifted")
    groups = topology.get("pp16_lp2", {}).get("groups")
    stage_zero = {
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "device_ids": [0, 1],
        "process_index": 0,
        "stage_id": 0,
    }
    if (
        topology_sha256 != EXPECTED_TOPOLOGY_SHA256
        or topology.get("artifact_kind") != "gate_d_runtime_physical_locality_authority"
        or not isinstance(groups, list)
        or len(groups) != 16
        or groups[0] != stage_zero
        or topology.get("tpu_successor_authorized") is not False
    ):
        raise ProjectionContractionHloSourceError("topology authority drifted")
    try:
        tree = ast.parse(source.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ProjectionContractionHloSourceError(
            "acquirer source is invalid"
        ) from error
    main = _function(tree, "main")
    if _literal(tree, "INPUT_SPEC") != EXPECTED_INPUT_SPEC:
        raise ProjectionContractionHloSourceError("projection input spec drifted")
    if _literal(tree, "OUTPUT_SPEC") != EXPECTED_OUTPUT_SPEC:
        raise ProjectionContractionHloSourceError("projection output spec drifted")
    imports = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.ImportFrom)
        and node.module
        == "glm_tpu.greenfield.benchmarking.gate_d_projection_contraction_pp16"
    ]
    if len(imports) != 1 or [(item.name, item.asname) for item in imports[0].names] != [
        ("build_gate_d_projection_contraction_pp16", None)
    ]:
        raise ProjectionContractionHloSourceError("projection builder import drifted")
    lower_calls = _calls(main, "lower")
    compile_calls = _calls(main, "compile")
    eval_shape_calls = _calls(main, "eval_shape")
    if (
        len(lower_calls) != 1
        or ast.unparse(lower_calls[0].func) != "replay.lower"
        or len(compile_calls) != 1
        or ast.unparse(compile_calls[0].func) != "lowered.compile"
        or len(eval_shape_calls) != 1
        or ast.unparse(eval_shape_calls[0].func) != "jax.eval_shape"
    ):
        raise ProjectionContractionHloSourceError("compile-only call chain drifted")
    forbidden = {
        "block_until_ready",
        "device_put",
        "execute",
        "host_callback",
        "io_callback",
        "process_allgather",
        "pure_callback",
    }
    if any(_calls(tree, name) for name in forbidden) or any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compiled"
        for node in ast.walk(main)
    ):
        raise ProjectionContractionHloSourceError("executable invocation path detected")
    text = source.decode("utf-8")
    required = (
        '"compiled_executable_invocation_count": 0',
        '"tpu_numerical_execution_performed": False',
        '"numerical_claim": False',
        '"performance_claim": False',
        '"gate_d_closed": False',
        '"status": "HLO_ACQUIRED_UNADJUDICATED"',
        'parser.add_argument("--compile-only", type=int, choices=(1,), required=True)',
    )
    if any(text.count(fragment) != 1 for fragment in required):
        raise ProjectionContractionHloSourceError("result authority contract drifted")
    forbidden_text = ("admission-report", "capsule_input_authority", "forced_round")
    if any(token in text for token in forbidden_text):
        raise ProjectionContractionHloSourceError("unrelated predecessor path detected")
    digest = sha256(
        ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()
    if digest != EXPECTED_ACQUIRER_AST_SHA256:
        raise ProjectionContractionHloSourceError("complete acquirer AST drifted")
    return {
        "acquirer_ast_sha256": digest,
        "compile_call_count": len(compile_calls),
        "compiled_executable_invocation_count": 0,
        "eval_shape_call_count": len(eval_shape_calls),
        "input_spec": [
            {"dtype": dtype, "name": name, "shape": list(shape)}
            for name, shape, dtype in EXPECTED_INPUT_SPEC
        ],
        "lower_call_count": len(lower_calls),
        "output_spec": [
            {"dtype": dtype, "name": name, "shape": list(shape)}
            for name, shape, dtype in EXPECTED_OUTPUT_SPEC
        ],
        "pp16_stage_zero": stage_zero,
        "source_authority_sha256": source_authority_sha256,
        "topology_sha256": topology_sha256,
        "tpu_numerical_execution_authorized": False,
    }
