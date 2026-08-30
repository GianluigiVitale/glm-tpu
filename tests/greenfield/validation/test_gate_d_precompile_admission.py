from __future__ import annotations

import ast
import base64
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile

import numpy as np
import pytest

from glm_tpu.greenfield.errors import BenchmarkValidationError
import glm_tpu.greenfield.gate_d_precompile_admission as admission_module
from glm_tpu.greenfield.gate_d_precompile_admission import (
    admit_gate_d_precompile_candidates,
    write_gate_d_precompile_admission_report,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CURRENT_CONTRACT = REPO_ROOT / "configs/greenfield-gate-d-precompile-admission-v2.json"
CURRENT_CONTRACT_SHA256 = "74e8c06fefc71ebee9f7b583c6bc34c5fe5724830b46ffd64fdc5572642a7752"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return _sha(path)


def _current_concrete_source_nodes() -> tuple[
    dict[str, ast.AST], dict[str, ast.Module]
]:
    rms_tree = ast.parse(
        (REPO_ROOT / "glm_tpu/greenfield/kernels/reference/rmsnorm.py").read_text()
    )
    layer_tree = ast.parse(
        (REPO_ROOT / "glm_tpu/greenfield/kernels/layer.py").read_text()
    )
    names = {
        "candidate.source:FusedAddRmsNormAuxiliaryResult": (
            rms_tree,
            "FusedAddRmsNormAuxiliaryResult",
        ),
        "candidate.source:fused_add_rms_norm": (
            rms_tree,
            "fused_add_rms_norm",
        ),
        "candidate.source:fused_add_rms_norm_with_auxiliary": (
            rms_tree,
            "fused_add_rms_norm_with_auxiliary",
        ),
        "candidate.callsite:StageLocalSplitLayerFp8AuxiliaryResult": (
            layer_tree,
            "StageLocalSplitLayerFp8AuxiliaryResult",
        ),
        "candidate.callsite:stage_local_transformer_layer_fp8_split_mapped": (
            layer_tree,
            "stage_local_transformer_layer_fp8_split_mapped",
        ),
    }
    return (
        {
            source_id: admission_module._qualified_ast_node(tree, name)
            for source_id, (tree, name) in names.items()
        },
        {
            "candidate.source": rms_tree,
            "candidate.callsite": layer_tree,
        },
    )


def test_current_concrete_tuple_source_semantics_are_bound() -> None:
    nodes, imports = _current_concrete_source_nodes()
    report = admission_module._concrete_tuple_source_semantics(nodes, imports)
    assert report["authority_scope"] == "concrete.committed.jax.source"
    assert report["frontier"] == "layer1.rms_input_fp32"


def test_concrete_tuple_source_refuses_precision_drift() -> None:
    nodes, imports = _current_concrete_source_nodes()
    candidate_id = "candidate.source:fused_add_rms_norm_with_auxiliary"
    candidate = ast.parse(ast.unparse(nodes[candidate_id])).body[0]
    assert isinstance(candidate, ast.FunctionDef)
    for node in ast.walk(candidate):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "jnp"
            and node.attr == "float32"
        ):
            node.attr = "float16"
            break
    nodes[candidate_id] = candidate
    with pytest.raises(BenchmarkValidationError, match="primary arithmetic drifted"):
        admission_module._concrete_tuple_source_semantics(nodes, imports)


def test_concrete_tuple_source_refuses_shared_baseline_precision_drift() -> None:
    nodes, imports = _current_concrete_source_nodes()
    for candidate_id in (
        "candidate.source:fused_add_rms_norm",
        "candidate.source:fused_add_rms_norm_with_auxiliary",
    ):
        candidate = ast.parse(ast.unparse(nodes[candidate_id])).body[0]
        assert isinstance(candidate, ast.FunctionDef)
        for node in ast.walk(candidate):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "jnp"
                and node.attr == "float32"
            ):
                node.attr = "float16"
        nodes[candidate_id] = candidate
    with pytest.raises(BenchmarkValidationError, match="accepted source primary arithmetic"):
        admission_module._concrete_tuple_source_semantics(nodes, imports)


@pytest.mark.parametrize(
    ("module_id", "statement", "message"),
    [
        ("candidate.source", "jnp = object()", "global import was rebound"),
        (
            "candidate.source",
            "if True:\n    jnp = object()",
            "global import was rebound",
        ),
        (
            "candidate.callsite",
            "fused_add_rms_norm_with_auxiliary = fused_add_rms_norm",
            "caller import was rebound",
        ),
    ],
)
def test_concrete_tuple_source_refuses_global_rebinding(
    module_id: str, statement: str, message: str
) -> None:
    nodes, modules = _current_concrete_source_nodes()
    modules[module_id].body.extend(ast.parse(statement).body)
    with pytest.raises(BenchmarkValidationError, match=message):
        admission_module._concrete_tuple_source_semantics(nodes, modules)


def test_concrete_tuple_source_refuses_hidden_executable_statement() -> None:
    nodes, modules = _current_concrete_source_nodes()
    candidate_id = "candidate.source:fused_add_rms_norm_with_auxiliary"
    candidate = ast.parse(ast.unparse(nodes[candidate_id])).body[0]
    assert isinstance(candidate, ast.FunctionDef)
    candidate.body.insert(-1, ast.parse("jax.debug.print('hidden')").body[0])
    nodes[candidate_id] = candidate
    with pytest.raises(BenchmarkValidationError, match="executable body drifted"):
        admission_module._concrete_tuple_source_semantics(nodes, modules)


def test_concrete_tuple_source_refuses_named_tuple_behavior() -> None:
    nodes, modules = _current_concrete_source_nodes()
    result_id = "candidate.source:FusedAddRmsNormAuxiliaryResult"
    result = ast.parse(ast.unparse(nodes[result_id])).body[0]
    assert isinstance(result, ast.ClassDef)
    result.body.extend(ast.parse("def hidden(self):\n    return self.output\n").body)
    nodes[result_id] = result
    with pytest.raises(BenchmarkValidationError, match="not one exact NamedTuple"):
        admission_module._concrete_tuple_source_semantics(nodes, modules)


def _run_git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _absolute_current_contract() -> dict[str, Any]:
    contract = json.loads(CURRENT_CONTRACT.read_text())
    base = CURRENT_CONTRACT.parent
    for name in ("contract", "core", "frontier"):
        binding = contract["inherited_v1"][name]
        binding["path"] = str((base / binding["path"]).resolve())
    for name in ("cli", "core", "git", "stablehlo_validator", "validator_python"):
        binding = contract["implementation"][name]
        binding["path"] = str((base / binding["path"]).resolve())
    contract["implementation"]["validator_pythonpath"] = str(
        (base / contract["implementation"]["validator_pythonpath"]).resolve()
    )
    for evidence in contract["evidence"]:
        evidence["path"] = str((base / evidence["path"]).resolve())
    locality = contract["physical_locality_authority"]
    locality["path"] = str((base / locality["path"]).resolve())
    return contract


def _source_authority(
    root: Path,
    candidate: dict[str, Any],
    locality: dict[str, Any],
    *,
    auxiliary_affects_primary: bool = False,
    committed_symlink: bool = False,
    source_hlo_mismatch: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = root / "source-repo"
    repository.mkdir()
    _run_git(repository, "init", "-q")
    _run_git(repository, "config", "user.email", "gate-d@example.invalid")
    _run_git(repository, "config", "user.name", "Gate D fixture")
    source = repository / "candidate.py"
    source_text = admission_module._candidate_source_text(candidate["id"])
    if source_hlo_mismatch:
        source_text = source_text.replace(
            "return candidate_dependency(hidden_update, residual, weight)",
            "return candidate_dependency(residual, hidden_update, weight)",
        )
    if committed_symlink:
        target = root / "candidate-target.py"
        target.write_text(source_text)
        source.symlink_to(target)
    else:
        source.write_text(source_text)
    _run_git(repository, "add", "candidate.py")
    _run_git(repository, "commit", "-q", "-m", "fixture")
    code_pin = _run_git(repository, "rev-parse", "HEAD")
    raw = source_text.encode("utf-8")
    tree = ast.parse(raw.decode("utf-8"))
    node, callsite = [
        item for item in tree.body if isinstance(item, ast.FunctionDef)
    ]

    def ast_sha(item: ast.AST) -> str:
        return sha256(
            ast.dump(item, annotate_fields=True, include_attributes=False).encode(
                "utf-8"
            )
        ).hexdigest()

    candidate_ast_sha = ast_sha(node)
    callsite_ast_sha = ast_sha(callsite)
    object_id = _run_git(repository, "rev-parse", f"{code_pin}:candidate.py")
    files = [
        {
            "id": "candidate.source",
            "repo_path": "candidate.py",
            "sha256": sha256(raw).hexdigest(),
            "symbols": [
                {
                    "ast_sha256": candidate_ast_sha,
                    "qualified_name": "candidate_dependency",
                },
                {
                    "ast_sha256": callsite_ast_sha,
                    "qualified_name": "candidate_callsite",
                },
            ],
        }
    ]
    source_set_payload = {
        "code_pin": code_pin,
        "files": [
            {
                "git_object_id": object_id,
                **files[0],
                "symbols": sorted(
                    files[0]["symbols"], key=lambda item: item["qualified_name"]
                ),
            }
        ],
    }
    source_set_sha = sha256(_canonical(source_set_payload).encode("ascii")).hexdigest()
    fingerprint_sha = sha256(
        _canonical(candidate["mechanism_fingerprint"]).encode("ascii")
    ).hexdigest()
    direct_rejection = json.loads(
        (
            REPO_ROOT
            / "docs/artifacts/plan-local-persistent-fp32-shadow-source-rejection.json"
        ).read_text()
    )
    accepted_semantics = admission_module._accepted_semantics_from_evidence(
        direct_rejection
    )
    semantic_tree = (
        ast.parse(admission_module._candidate_source_text(candidate["id"]))
        if source_hlo_mismatch
        else tree
    )
    semantic_node, semantic_callsite = [
        item for item in semantic_tree.body if isinstance(item, ast.FunctionDef)
    ]
    source_semantics = admission_module._candidate_source_semantics(
        semantic_node, semantic_callsite, candidate["id"]
    )
    source_semantic_sha = sha256(
        _canonical(source_semantics).encode("ascii")
    ).hexdigest()
    semantics = {
        "accepted_authority_sha256": accepted_semantics["authority_sha256"],
        "arithmetic_contract": {
            "auxiliary_affects_primary_arithmetic": auxiliary_affects_primary,
            "normalization_input": "transient.fp32.sum",
            "primary_outputs_bitwise_identical_by_construction": True,
            "primary_recurrence": "bf16.rounded",
        },
        "authority_kind": "declarative.semantic.dsl",
        "candidate_id": candidate["id"],
        "candidate_callsite_symbol": "candidate.source:candidate_callsite",
        "candidate_semantic_sha256": source_semantic_sha,
        "candidate_symbol": "candidate.source:candidate_dependency",
        "causal_frontier_action": candidate["causal_frontier_action"],
        "claim_scope": "Synthetic source-semantics fixture only.",
        "code_pin": code_pin,
        "compensation_claim": (
            "graph.identity.only;numeric.cancellation.unproven"
            if candidate["id"] == "compensated_auxiliary_dependency"
            else "none"
        ),
        "locality": locality,
        "mechanism_fingerprint_sha256": fingerprint_sha,
        "normal_form": candidate["normal_form"],
        "schema_version": 2,
        "source_set_sha256": source_set_sha,
    }
    semantics_path = root / "source-semantics.json"
    semantics_sha = _write_json(semantics_path, semantics)
    authority = {
        "files": files,
        "repository": {"commit": code_pin, "root": str(repository)},
        "semantics_certificate": {
            "path": str(semantics_path),
            "sha256": semantics_sha,
        },
    }
    authority_sha = sha256(
        _canonical(
            {
                "certificate_sha256": semantics_sha,
                "code_pin": code_pin,
                "accepted_authority_sha256": accepted_semantics[
                    "authority_sha256"
                ],
                "authority_kind": semantics["authority_kind"],
                "candidate_ast_sha256": candidate_ast_sha,
                "callsite_ast_sha256": callsite_ast_sha,
                "compensation_claim": semantics["compensation_claim"],
                "mechanism_fingerprint_sha256": fingerprint_sha,
                "source_semantic_sha256": source_semantic_sha,
                "source_set_sha256": source_set_sha,
            }
        ).encode("ascii")
    ).hexdigest()
    return authority, {
        "authority_sha256": authority_sha,
        "callsite_symbol": {
            "ast_sha256": callsite_ast_sha,
            "id": "candidate.source:candidate_callsite",
        },
        "candidate_symbol": {
            "ast_sha256": candidate_ast_sha,
            "id": "candidate.source:candidate_dependency",
        },
        "code_pin": code_pin,
        "source_semantic_sha256": source_semantic_sha,
        "source_set_sha256": source_set_sha,
    }


def _plan_authority(
    root: Path, plan_name: str = "PP16_LP2"
) -> tuple[dict[str, Any], dict[str, Any]]:
    locality = json.loads(
        (REPO_ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json").read_text()
    )
    locality_key = "pp16_lp2" if plan_name == "PP16_LP2" else "pp8_lp4"
    groups = [record["device_ids"] for record in locality[locality_key]["groups"]]
    owner_ids = groups[0]
    layout = "lp2.local" if plan_name == "PP16_LP2" else "lp4.local"
    watchpoint_schema = admission_module._required_watchpoint_schema(len(owner_ids))
    watchpoints = {
        item: {"layout": layout, "owner_ids": owner_ids}
        for item in watchpoint_schema
    }
    payload = {
        "local_device_groups": groups,
        "plan": plan_name,
        "topology_hash": locality["topology_hash"],
        "watchpoints": watchpoints,
    }
    plan_sha = sha256(_canonical(payload).encode("ascii")).hexdigest()
    plan = {**payload, "plan_sha256": plan_sha, "schema_version": 2}
    path = root / "plan.json"
    file_sha = _write_json(path, plan)
    return (
        {
            "path": str(path),
            "sha256": file_sha,
        },
        {
            "file_sha256": file_sha,
            "local_device_groups": groups,
            "path": path,
            "local_group_size": len(owner_ids),
            "owner_group": owner_ids,
            "plan": plan_name,
            "plan_sha256": plan_sha,
            "topology_hash": locality["topology_hash"],
            "watchpoints": watchpoints,
            "watchpoint_schema": watchpoint_schema,
        },
    )


def _stablehlo_authority(
    root: Path,
    candidate: dict[str, Any],
    locality: dict[str, Any],
    source: dict[str, Any],
    plan: dict[str, Any],
    *,
    primary_depends_on_auxiliary: bool = False,
    alter_accepted_primary: bool = False,
    constant_primary: bool = False,
    host_effect: bool = False,
    group_size: int = 1,
    implicit_group: bool = False,
    nonlocal_pair: bool = False,
    opaque_call: bool = False,
    string_edge: bool = False,
    non_cancelling_compensation: bool = False,
    force_tuple_implementation: bool = False,
    extra_argument: bool = False,
    extra_result: bool = False,
    dead_operation: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    comment = "\n    // xla_python_cpu_callback" if host_effect else ""
    collective = ""
    auxiliary_base = "%sum"
    if group_size > 1 or nonlocal_pair:
        pairs = (
            [[0, 31]]
            if nonlocal_pair
            else [[rank, (rank + 1) % group_size] for rank in range(group_size)]
        )
        pair_text = ", ".join(f"[{source}, {target}]" for source, target in pairs)
        if implicit_group:
            collective = (
                "    %transport = \"stablehlo.collective_permute\"(%sum) "
                ": (tensor<1x6144xf32>) -> tensor<1x6144xf32>\n"
            )
        else:
            collective = (
                "    %transport = \"stablehlo.collective_permute\"(%sum) "
                f"<{{source_target_pairs = dense<[{pair_text}]> : "
                f"tensor<{len(pairs)}x2xi64>}}> "
                ": (tensor<1x6144xf32>) -> tensor<1x6144xf32>\n"
            )
        auxiliary_base = "%transport"
    metadata = {
        "gate_d.callsite_ast_sha256": source["callsite_symbol"]["ast_sha256"],
        "gate_d.candidate_ast_sha256": source["candidate_symbol"]["ast_sha256"],
        "gate_d.source_set_sha256": source["source_set_sha256"],
    }
    if string_edge:
        metadata["gate_d.unexpected"] = "%sum"
    module_header = "module attributes {" + ", ".join(
        f'{key} = "{value}"' for key, value in sorted(metadata.items())
    ) + "} {\n"
    primary_extra = ""
    normalization_source = "%sum"
    if primary_depends_on_auxiliary:
        primary_extra = (
            "    %primary_shadow = stablehlo.add %sum, %sum : "
            "tensor<1x6144xf32>\n"
        )
        normalization_source = "%primary_shadow"
    carried_prefix = ""
    carried_source = "%sum"
    if constant_primary:
        carried_prefix = (
            "    %zero_row = stablehlo.constant dense<0.0> : "
            "tensor<1x6144xf32>\n"
        )
        carried_source = "%zero_row"
    if opaque_call:
        auxiliary_operation = (
            f'    %aux = "stablehlo.custom_call"({auxiliary_base}) '
            '<{call_target_name = "opaque", has_side_effect = false}> '
            ': (tensor<1x6144xf32>) -> tensor<1x6144xf32>\n'
        )
        auxiliary_value = "%aux"
    elif (
        candidate["id"] == "compensated_auxiliary_dependency"
        and not force_tuple_implementation
    ):
        final_operator = (
            "stablehlo.add"
            if non_cancelling_compensation
            else "stablehlo.subtract"
        )
        auxiliary_operation = (
            "    %rounded = stablehlo.convert %carried : "
            "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
            f"    %correction = stablehlo.subtract {auxiliary_base}, %rounded : "
            "tensor<1x6144xf32>\n"
            "    %restored = stablehlo.add %rounded, %correction : tensor<1x6144xf32>\n"
            f"    %aux = {final_operator} %restored, %correction : tensor<1x6144xf32>\n"
        )
        auxiliary_value = "%aux"
    else:
        auxiliary_operation = ""
        auxiliary_value = auxiliary_base
    candidate_text = (
        module_header
        + "  func.func @candidate(%arg0: tensor<1x6144xbf16>, "
        "%arg1: tensor<1x6144xbf16>, %arg2: tensor<1x6144xbf16>) "
        "-> (tensor<1x6144xbf16>, tensor<1x6144xbf16>, "
        "tensor<1x6144xf32>) {\n"
        "    %hidden = stablehlo.convert %arg0 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %residual = stablehlo.convert %arg1 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %sum = stablehlo.add %hidden, %residual : tensor<1x6144xf32>\n"
        f"{carried_prefix}"
        f"    %carried = stablehlo.convert {carried_source} : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        f"{primary_extra}"
        f"    %square = stablehlo.multiply {normalization_source}, "
        f"{normalization_source} : tensor<1x6144xf32>\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n"
        "    %variance_sum = \"stablehlo.reduce\"(%square, %zero) "
        "<{dimensions = array<i64: 1>}> ({\n"
        "    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):\n"
        "      %added = stablehlo.add %lhs, %rhs : tensor<f32>\n"
        "      stablehlo.return %added : tensor<f32>\n"
        "    }) : (tensor<1x6144xf32>, tensor<f32>) -> tensor<1xf32>\n"
        "    %divisor = stablehlo.constant dense<6.144000e+03> : tensor<1xf32>\n"
        "    %variance = stablehlo.divide %variance_sum, %divisor : tensor<1xf32>\n"
        "    %epsilon = stablehlo.constant dense<9.99999997e-07> : tensor<1xf32>\n"
        "    %variance_epsilon = stablehlo.add %variance, %epsilon : tensor<1xf32>\n"
        "    %inverse = stablehlo.rsqrt %variance_epsilon : tensor<1xf32>\n"
        "    %scale = stablehlo.broadcast_in_dim %inverse, dims = [0] : "
        "(tensor<1xf32>) -> tensor<1x6144xf32>\n"
        f"    %normalized = stablehlo.multiply {normalization_source}, %scale : "
        "tensor<1x6144xf32>\n"
        "    %normalized_bf16 = stablehlo.convert %normalized : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        "    %normalized_f32 = stablehlo.convert %normalized_bf16 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %weight = stablehlo.convert %arg2 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %weighted_f32 = stablehlo.multiply %normalized_f32, %weight : "
        "tensor<1x6144xf32>\n"
        "    %weighted = stablehlo.convert %weighted_f32 : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        f"{collective}"
        f"{auxiliary_operation}"
        f"{comment}\n"
        f"    return %weighted, %carried, {auxiliary_value} : "
        "tensor<1x6144xbf16>, tensor<1x6144xbf16>, tensor<1x6144xf32>\n"
        "  }\n"
        "}\n"
    )
    if extra_argument:
        candidate_text = candidate_text.replace(
            "%arg2: tensor<1x6144xbf16>) ",
            "%arg2: tensor<1x6144xbf16>, %arg3: tensor<1x6144xbf16>) ",
            1,
        )
    if extra_result:
        candidate_text = candidate_text.replace(
            "tensor<1x6144xf32>) {\n",
            "tensor<1x6144xf32>, tensor<1x6144xf32>) {\n",
            1,
        ).replace(
            f"return %weighted, %carried, {auxiliary_value} : ",
            f"return %weighted, %carried, {auxiliary_value}, %sum : ",
            1,
        ).replace(
            "tensor<1x6144xbf16>, tensor<1x6144xbf16>, tensor<1x6144xf32>\n",
            "tensor<1x6144xbf16>, tensor<1x6144xbf16>, "
            "tensor<1x6144xf32>, tensor<1x6144xf32>\n",
            1,
        )
    if dead_operation:
        candidate_text = candidate_text.replace(
            "    return %weighted,",
            "    %dead = stablehlo.add %sum, %sum : tensor<1x6144xf32>\n"
            "    return %weighted,",
            1,
        )
    accepted_normalization_source = "%hidden" if alter_accepted_primary else "%sum"
    accepted_text = (
        "module {\n"
        "  func.func @accepted(%arg0: tensor<1x6144xbf16>, "
        "%arg1: tensor<1x6144xbf16>, %arg2: tensor<1x6144xbf16>) "
        "-> (tensor<1x6144xbf16>, tensor<1x6144xbf16>) {\n"
        "    %hidden = stablehlo.convert %arg0 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %residual = stablehlo.convert %arg1 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %sum = stablehlo.add %hidden, %residual : tensor<1x6144xf32>\n"
        "    %carried = stablehlo.convert %sum : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        f"    %square = stablehlo.multiply {accepted_normalization_source}, "
        f"{accepted_normalization_source} : tensor<1x6144xf32>\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n"
        "    %variance_sum = \"stablehlo.reduce\"(%square, %zero) "
        "<{dimensions = array<i64: 1>}> ({\n"
        "    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):\n"
        "      %added = stablehlo.add %lhs, %rhs : tensor<f32>\n"
        "      stablehlo.return %added : tensor<f32>\n"
        "    }) : (tensor<1x6144xf32>, tensor<f32>) -> tensor<1xf32>\n"
        "    %divisor = stablehlo.constant dense<6.144000e+03> : tensor<1xf32>\n"
        "    %variance = stablehlo.divide %variance_sum, %divisor : tensor<1xf32>\n"
        "    %epsilon = stablehlo.constant dense<9.99999997e-07> : tensor<1xf32>\n"
        "    %variance_epsilon = stablehlo.add %variance, %epsilon : tensor<1xf32>\n"
        "    %inverse = stablehlo.rsqrt %variance_epsilon : tensor<1xf32>\n"
        "    %scale = stablehlo.broadcast_in_dim %inverse, dims = [0] : "
        "(tensor<1xf32>) -> tensor<1x6144xf32>\n"
        f"    %normalized = stablehlo.multiply {accepted_normalization_source}, "
        "%scale : tensor<1x6144xf32>\n"
        "    %normalized_bf16 = stablehlo.convert %normalized : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        "    %normalized_f32 = stablehlo.convert %normalized_bf16 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %weight = stablehlo.convert %arg2 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %weighted_f32 = stablehlo.multiply %normalized_f32, %weight : "
        "tensor<1x6144xf32>\n"
        "    %weighted = stablehlo.convert %weighted_f32 : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        "    return %weighted, %carried : tensor<1x6144xbf16>, "
        "tensor<1x6144xbf16>\n"
        "  }\n"
        "}\n"
    )
    candidate_path = root / "candidate.stablehlo.mlir"
    accepted_path = root / "accepted-primary.stablehlo.mlir"
    candidate_path.write_text(candidate_text)
    accepted_path.write_text(accepted_text)
    candidate_sha = _sha(candidate_path)
    accepted_sha = _sha(accepted_path)
    fingerprint_sha = sha256(
        _canonical(candidate["mechanism_fingerprint"]).encode("ascii")
    ).hexdigest()
    validator_contract = {
        "auxiliary_result_index": 2,
        "carried_residual_result_index": 1,
        "weighted_output_result_index": 0,
    }
    certificate = {
        "accepted_primary_stablehlo_sha256": accepted_sha,
        "candidate_id": candidate["id"],
        "candidate_source_semantic_sha256": source["source_semantic_sha256"],
        "candidate_stablehlo_sha256": candidate_sha,
        "causal_frontier": {
            "action": candidate["causal_frontier_action"],
            "id": "layer1.rms_input_fp32",
        },
        "claim_scope": "Synthetic parser-validated StableHLO fixture only.",
        "code_pin": source["code_pin"],
        "locality": locality,
        "mechanism_fingerprint_sha256": fingerprint_sha,
        "normal_form": candidate["normal_form"],
        "plan_sha256": plan["plan_sha256"],
        "schema_version": 2,
        "source_set_sha256": source["source_set_sha256"],
        "validator_contract": validator_contract,
    }
    certificate_path = root / "stablehlo-certificate.json"
    certificate_sha = _write_json(certificate_path, certificate)
    authority = {
        "accepted_primary": {"path": str(accepted_path), "sha256": accepted_sha},
        "candidate": {"path": str(candidate_path), "sha256": candidate_sha},
        "certificate": {
            "path": str(certificate_path),
            "sha256": certificate_sha,
        },
    }
    request = {
        "accepted_stablehlo_base64": base64.b64encode(
            accepted_text.encode("utf-8")
        ).decode("ascii"),
        "accepted_stablehlo_sha256": accepted_sha,
        "auxiliary_result_index": 2,
        "callsite_ast_sha256": source["callsite_symbol"]["ast_sha256"],
        "candidate_ast_sha256": source["candidate_symbol"]["ast_sha256"],
        "candidate_stablehlo_base64": base64.b64encode(
            candidate_text.encode("utf-8")
        ).decode("ascii"),
        "candidate_stablehlo_sha256": candidate_sha,
        "carried_residual_result_index": 1,
        "expected_parser_files": {
            item["path"]: item["sha256"]
            for item in _absolute_current_contract()["implementation"]["validator_imports"]
        },
        "local_device_groups": plan["local_device_groups"],
        "source_set_sha256": source["source_set_sha256"],
        "weighted_output_result_index": 0,
    }
    implementation = admission_module._verify_implementation(
        _absolute_current_contract()["implementation"], CURRENT_CONTRACT.parent
    )
    try:
        validator = admission_module._run_stablehlo_validator(request, implementation)
    except BenchmarkValidationError:
        return authority, {"authority_sha256": "0" * 64}
    validator_sha = sha256(_canonical(validator).encode("ascii")).hexdigest()
    authority_sha = sha256(
        _canonical(
            {
                "accepted_primary_sha256": accepted_sha,
                "accepted_primary_slice_sha256": validator[
                    "accepted_primary_slice_sha256"
                ],
                "candidate_sha256": candidate_sha,
                "certificate_sha256": certificate_sha,
                "mechanism_fingerprint_sha256": fingerprint_sha,
                "plan_sha256": plan["plan_sha256"],
                "source_semantic_sha256": source["source_semantic_sha256"],
                "source_set_sha256": source["source_set_sha256"],
                "validator_report_sha256": validator_sha,
            }
        ).encode("ascii")
    ).hexdigest()
    return authority, {"authority_sha256": authority_sha}


def _capsule(
    root: Path,
    candidate: dict[str, Any],
    source: dict[str, Any],
    stablehlo: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    owner_ids = plan["owner_group"]
    owner_count = len(owner_ids)
    values: dict[str, np.ndarray] = {
        "rms_hidden_update": np.full((6144,), 0x3F80, dtype=np.uint16),
        "rms_residual": np.full((6144,), 0x4000, dtype=np.uint16),
        "rms_input": np.full((6144,), 3.0, dtype=np.float32),
        "normalized": np.full((owner_count, 1, 6144), 0x3F80, dtype=np.uint16),
        "cache_history": np.full(
            (owner_count, 16, 256, 128), 0x4000, dtype=np.uint16
        ),
        "query": np.full((owner_count, 1, 32, 128), 2.0, dtype=np.float32),
        "head_weights": np.full((owner_count, 1, 32), 3.0, dtype=np.float32),
        "current_key": np.full((128,), 4.0, dtype=np.float32),
        "event1_positions": np.full((1, 2048), 7, dtype=np.int32),
        "event1_scores": np.full((1, 2048), 5.0, dtype=np.float32),
        "event1_valid_count": np.array([1], dtype=np.int32),
    }
    artifact = root / "candidate-state.npz"
    np.savez(artifact, **values)
    artifact_sha = _sha(artifact)
    descriptions = {
        "layer1.rms_operands_bf16": [
            ("hidden_update", "rms_hidden_update", "bf16_bits", [], None),
            ("residual", "rms_residual", "bf16_bits", [], None),
        ],
        "layer1.rms_input_fp32": [("value", "rms_input", "float32", [], None)],
        "layer1.normalized": [
            (f"owner{slot}", "normalized", "bf16_bits", [slot, 0], owner_id)
            for slot, owner_id in enumerate(owner_ids)
        ],
        "layer1.cache_history": [
            ("value", "cache_history", "bf16_bits", [], None)
        ],
        "layer1.query": [
            (f"owner{slot}", "query", "float32", [slot, 0], owner_id)
            for slot, owner_id in enumerate(owner_ids)
        ],
        "layer1.head_weights": [
            (f"owner{slot}", "head_weights", "float32", [slot, 0], owner_id)
            for slot, owner_id in enumerate(owner_ids)
        ],
        "layer1.current_key": [("value", "current_key", "float32", [], None)],
        "layer1.scorer_event1": [
            ("positions", "event1_positions", "int32", [], None),
            ("scores", "event1_scores", "float32", [], None),
            ("valid_count", "event1_valid_count", "int32", [], None),
        ],
    }
    storage = {
        np.dtype(np.float32): "<f4",
        np.dtype(np.int32): "<i4",
        np.dtype(np.uint16): "<u2",
    }
    watchpoints = []
    for watchpoint_id, specs in descriptions.items():
        arrays = []
        for role, key, semantic_dtype, index_prefix, owner_id in specs:
            expected_spec = plan["watchpoint_schema"][
                watchpoint_id
            ]["arrays"][role]
            array = values[key]
            selected = array[tuple(index_prefix)] if index_prefix else array
            arrays.append(
                {
                    "array_key": key,
                    "array_sha256": sha256(
                        np.ascontiguousarray(selected).tobytes(order="C")
                    ).hexdigest(),
                    "index_prefix": index_prefix,
                    "owner_axis": expected_spec["owner_axis"],
                    "owner_axis_ids": [
                        plan["watchpoints"][watchpoint_id]["owner_ids"][slot]
                        for slot in expected_spec["owner_axis_slots"]
                    ],
                    "owner_id": owner_id,
                    "role": role,
                    "semantic_dtype": semantic_dtype,
                    "shape": list(selected.shape),
                    "storage_dtype": storage[array.dtype],
                }
            )
        watchpoints.append(
            {
                "arrays": arrays,
                "id": watchpoint_id,
                "layer": 1,
                "layout": plan["watchpoints"][watchpoint_id]["layout"],
                "owner_ids": owner_ids,
                "position": 8155,
            }
        )
    document = {
        "artifact": {"path": str(artifact), "sha256": artifact_sha},
        "candidate_id": candidate["id"],
        "claim_scope": "Synthetic candidate-coherent capsule fixture only.",
        "code_pin": source["code_pin"],
        "coherence_id": "synthetic.gate.d.precompile",
        "plan_sha256": plan["plan_sha256"],
        "schema_version": 2,
        "source_authority_sha256": source["authority_sha256"],
        "stablehlo_authority_sha256": stablehlo["authority_sha256"],
        "watchpoints": watchpoints,
    }
    path = root / "capsule.json"
    capsule_sha = _write_json(path, document)
    return {"path": str(path), "sha256": capsule_sha}, path, artifact


def _prepared_contract(
    root: Path,
    *,
    candidate_index: int = 0,
    auxiliary_affects_primary: bool = False,
    primary_depends_on_auxiliary: bool = False,
    alter_accepted_primary: bool = False,
    constant_primary: bool = False,
    host_effect: bool = False,
    group_size: int = 1,
    implicit_group: bool = False,
    committed_symlink: bool = False,
    nonlocal_pair: bool = False,
    opaque_call: bool = False,
    string_edge: bool = False,
    source_hlo_mismatch: bool = False,
    non_cancelling_compensation: bool = False,
    force_tuple_implementation: bool = False,
    extra_argument: bool = False,
    extra_result: bool = False,
    dead_operation: bool = False,
    plan_name: str = "PP16_LP2",
) -> tuple[Path, dict[str, Any], Path, Path]:
    contract = _absolute_current_contract()
    candidate = contract["candidates"][candidate_index]
    locality = contract["locality_contract"]
    source_authority, source = _source_authority(
        root,
        candidate,
        locality,
        auxiliary_affects_primary=auxiliary_affects_primary,
        committed_symlink=committed_symlink,
        source_hlo_mismatch=source_hlo_mismatch,
    )
    plan_authority, plan = _plan_authority(root, plan_name)
    stablehlo_authority, stablehlo = _stablehlo_authority(
        root,
        candidate,
        locality,
        source,
        plan,
        primary_depends_on_auxiliary=primary_depends_on_auxiliary,
        alter_accepted_primary=alter_accepted_primary,
        constant_primary=constant_primary,
        host_effect=host_effect,
        group_size=group_size,
        implicit_group=implicit_group,
        nonlocal_pair=nonlocal_pair,
        opaque_call=opaque_call,
        string_edge=string_edge,
        non_cancelling_compensation=non_cancelling_compensation,
        force_tuple_implementation=force_tuple_implementation,
        extra_argument=extra_argument,
        extra_result=extra_result,
        dead_operation=dead_operation,
    )
    capsule_binding, capsule_path, artifact = _capsule(
        root, candidate, source, stablehlo, plan
    )
    candidate["source_authority"] = source_authority
    candidate["plan_authority"] = plan_authority
    candidate["stablehlo_authority"] = stablehlo_authority
    candidate["coherent_state_capsule"] = capsule_binding
    path = root / "contract.json"
    _write_json(path, contract)
    return path, contract, capsule_path, artifact


def _rebind_artifact(
    contract_path: Path,
    contract: dict[str, Any],
    capsule_path: Path,
    artifact: Path,
) -> None:
    capsule = json.loads(capsule_path.read_text())
    capsule["artifact"]["sha256"] = _sha(artifact)
    capsule_sha = _write_json(capsule_path, capsule)
    contract["candidates"][0]["coherent_state_capsule"]["sha256"] = capsule_sha
    _write_json(contract_path, contract)


def _rebind_capsule(
    contract_path: Path,
    contract: dict[str, Any],
    capsule_path: Path,
    capsule: dict[str, Any],
) -> None:
    capsule_sha = _write_json(capsule_path, capsule)
    contract["candidates"][0]["coherent_state_capsule"]["sha256"] = capsule_sha
    _write_json(contract_path, contract)


def test_current_contract_fails_closed_without_jax() -> None:
    before = set(sys.modules)
    report = admit_gate_d_precompile_candidates(
        CURRENT_CONTRACT, CURRENT_CONTRACT_SHA256
    )
    assert report["admitted_candidate_ids"] == []
    assert report["classification"] == (
        "NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;"
        "NO_JAX_OR_TPU_SUCCESSOR"
    )
    assert report["tpu_successor_authorized"] is False
    assert report["jax_compile_or_tpu_work_performed"] is False
    assert "jax" not in set(sys.modules) - before
    assert {
        item["id"]: item["reasons"] for item in report["candidate_results"]
    } == {
        "auxiliary_device_tuple_dependency": [
            "MISSING_PLAN_AUTHORITY",
            "MISSING_CAUSAL_STABLEHLO_AUTHORITY",
            "MISSING_CANDIDATE_COHERENT_CAPSULE",
        ],
        "compensated_auxiliary_dependency": [
            "MISSING_SOURCE_AST_AUTHORITY",
            "MISSING_PLAN_AUTHORITY",
            "MISSING_CAUSAL_STABLEHLO_AUTHORITY",
            "MISSING_CANDIDATE_COHERENT_CAPSULE",
        ],
    }
    source = report["candidate_results"][0]["source_authority"]
    assert source["authority_kind"] == "concrete.committed.jax.source"
    assert source["executable_source_authority"] is True
    assert source["code_pin"] == "c8b220067577004ddfb824667eadc746642baaca"


def test_complete_synthetic_fixture_never_admits_precompile(tmp_path: Path) -> None:
    path, _, _, _ = _prepared_contract(tmp_path)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    assert report["admitted_candidate_ids"] == []
    assert report["classification"] == (
        "NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;"
        "NO_JAX_OR_TPU_SUCCESSOR"
    )
    candidate = report["candidate_results"][0]
    assert candidate["reasons"] == [
        "MISSING_EXECUTABLE_SOURCE_AUTHORITY",
        "MISSING_IMMUTABLE_STABLEHLO_PARSER_AUTHORITY",
        "MISSING_PINNED_COHERENT_CAPSULE_PRODUCER",
    ]
    assert candidate["source_authority"]["executable_source_authority"] is False
    assert (
        candidate["stablehlo_authority"]["immutable_parser_authority"]
        is False
    )
    assert candidate["capsule"]["producer_provenance_verified"] is False
    assert report["compile_only_review_required"] is True
    assert report["gate_d_closed"] is False
    assert report["tpu_successor_authorized"] is False


def test_complete_pp8_authority_uses_four_real_owners(tmp_path: Path) -> None:
    path, _, _, _ = _prepared_contract(tmp_path, plan_name="PP8_LP4")
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    assert report["admitted_candidate_ids"] == []
    candidate = report["candidate_results"][0]
    assert candidate["plan_authority"]["local_group_size"] == 4
    assert candidate["plan_authority"]["owner_group"] == [0, 2, 1, 3]
    assert candidate["capsule"]["derived_rms_input_sha256"] == sha256(
        np.full((6144,), 3.0, dtype=np.float32).tobytes()
    ).hexdigest()


def test_complete_compensated_authority_is_distinct_and_offline_only(
    tmp_path: Path,
) -> None:
    path, _, _, _ = _prepared_contract(tmp_path, candidate_index=1)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    assert report["admitted_candidate_ids"] == []
    candidate = report["candidate_results"][1]
    assert candidate["stablehlo_authority"]["auxiliary_slice_sha256"] == (
        admission_module._EXPECTED_AUXILIARY_SLICE_SHA256[
            "compensated_auxiliary_dependency"
        ]
    )
    assert report["tpu_successor_authorized"] is False


def test_compensated_survivor_cannot_reuse_tuple_implementation(tmp_path: Path) -> None:
    path, _, _, _ = _prepared_contract(
        tmp_path,
        candidate_index=1,
        force_tuple_implementation=True,
    )
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][1]
    assert "INVALID_CAUSAL_STABLEHLO_AUTHORITY" in candidate["reasons"]
    assert "validator result drifted" in candidate["stablehlo_authority"]["refusal"]


def test_non_cancelling_add_sub_is_not_compensated_authority(tmp_path: Path) -> None:
    path, _, _, _ = _prepared_contract(
        tmp_path,
        candidate_index=1,
        non_cancelling_compensation=True,
    )
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][1]
    assert "INVALID_CAUSAL_STABLEHLO_AUTHORITY" in candidate["reasons"]
    assert "validator result drifted" in candidate["stablehlo_authority"]["refusal"]


def test_rehashed_source_cannot_reuse_unrelated_hlo(tmp_path: Path) -> None:
    path, _, _, _ = _prepared_contract(tmp_path, source_hlo_mismatch=True)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_SOURCE_AST_AUTHORITY" in candidate["reasons"]
    assert "source callsite semantics drifted" in candidate["source_authority"]["refusal"]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (
            {"auxiliary_affects_primary": True},
            "INVALID_SOURCE_AST_AUTHORITY",
        ),
        ({"host_effect": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        (
            {"primary_depends_on_auxiliary": True},
            "INVALID_CAUSAL_STABLEHLO_AUTHORITY",
        ),
        ({"constant_primary": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        ({"alter_accepted_primary": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        ({"group_size": 5}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        (
            {"group_size": 2, "implicit_group": True},
            "INVALID_CAUSAL_STABLEHLO_AUTHORITY",
        ),
        ({"nonlocal_pair": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        ({"opaque_call": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        ({"string_edge": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        ({"extra_argument": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        ({"extra_result": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
        ({"dead_operation": True}, "INVALID_CAUSAL_STABLEHLO_AUTHORITY"),
    ],
)
def test_authority_attacks_fail_closed(
    tmp_path: Path, kwargs: dict[str, Any], reason: str
) -> None:
    path, _, _, _ = _prepared_contract(tmp_path, **kwargs)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert candidate["admitted_precompile"] is False
    assert reason in candidate["reasons"]


def test_source_blob_sha_cannot_be_relabelled(tmp_path: Path) -> None:
    path, contract, _, _ = _prepared_contract(tmp_path)
    contract["candidates"][0]["source_authority"]["files"][0]["sha256"] = "0" * 64
    _write_json(path, contract)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    assert "INVALID_SOURCE_AST_AUTHORITY" in report["candidate_results"][0]["reasons"]


def test_committed_source_symlink_is_not_accepted_as_regular_blob(tmp_path: Path) -> None:
    path, _, _, _ = _prepared_contract(tmp_path, committed_symlink=True)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_SOURCE_AST_AUTHORITY" in candidate["reasons"]
    assert "not a committed regular blob" in candidate["source_authority"]["refusal"]


def test_git_queries_ignore_hostile_path_and_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, _, _ = _prepared_contract(tmp_path)
    fake = tmp_path / "fake-bin"
    fake.mkdir()
    fake_git = fake / "git"
    fake_git.write_text("#!/bin/sh\nexit 99\n")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong.git"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "wrong-objects"))
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    assert report["admitted_candidate_ids"] == []


def test_git_replace_object_cannot_change_bound_source(tmp_path: Path) -> None:
    path, contract, _, _ = _prepared_contract(tmp_path)
    repository = Path(
        contract["candidates"][0]["source_authority"]["repository"]["root"]
    )
    original = _run_git(repository, "rev-parse", "HEAD:candidate.py")
    replacement = tmp_path / "replacement.py"
    replacement.write_text("def accepted_primary(*args): return 0\n")
    replacement_id = _run_git(repository, "hash-object", "-w", str(replacement))
    _run_git(repository, "replace", original, replacement_id)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    assert report["admitted_candidate_ids"] == []


def test_mutated_same_version_parser_tree_is_rejected(tmp_path: Path) -> None:
    path, contract, _, _ = _prepared_contract(tmp_path)
    original_root = Path(contract["implementation"]["validator_pythonpath"])
    parser_root = tmp_path / "mutated-parser"
    for item in contract["implementation"]["validator_imports"]:
        relative = Path(item["path"])
        source = original_root / relative
        destination = parser_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".so":
            os.link(source, destination)
        else:
            destination.write_bytes(source.read_bytes())
    with (parser_root / "jaxlib/mlir/ir.py").open("ab") as stream:
        stream.write(b"\n# hostile same-version replacement\n")
    contract["implementation"]["validator_pythonpath"] = str(parser_root)
    _write_json(path, contract)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CAUSAL_STABLEHLO_AUTHORITY" in candidate["reasons"]
    assert "parser import" in candidate["stablehlo_authority"]["refusal"]
    assert "drifted" in candidate["stablehlo_authority"]["refusal"]


def test_parser_copy_hashes_the_same_bytes_it_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    replacement = tmp_path / "replacement.bin"
    destination = tmp_path / "sealed" / "source.bin"
    original = b"a" * (1024 * 1024) + b"b" * 257
    source.write_bytes(original)
    replacement.write_bytes(b"z" * len(original))
    real_write = admission_module.os.write
    replaced = False

    def replace_source_after_first_write(descriptor: int, value: Any) -> int:
        nonlocal replaced
        written = real_write(descriptor, value)
        if not replaced:
            os.replace(replacement, source)
            replaced = True
        return written

    monkeypatch.setattr(admission_module.os, "write", replace_source_after_first_write)
    admission_module._copy_bound_parser_file(
        source,
        destination,
        expected_bytes=len(original),
        expected_sha256=sha256(original).hexdigest(),
    )
    assert replaced is True
    assert destination.read_bytes() == original
    assert source.read_bytes() != original


def test_self_declared_nonlocal_plan_is_rejected(tmp_path: Path) -> None:
    path, contract, _, _ = _prepared_contract(tmp_path)
    binding = contract["candidates"][0]["plan_authority"]
    plan_path = Path(binding["path"])
    plan = json.loads(plan_path.read_text())
    plan["local_device_groups"][0] = [0, 8]
    plan["local_device_groups"][1] = [1, 9]
    payload = {
        "local_device_groups": plan["local_device_groups"],
        "plan": plan["plan"],
        "topology_hash": plan["topology_hash"],
        "watchpoints": plan["watchpoints"],
    }
    plan["plan_sha256"] = sha256(_canonical(payload).encode("ascii")).hexdigest()
    binding["sha256"] = _write_json(plan_path, plan)
    _write_json(path, contract)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_PLAN_AUTHORITY" in candidate["reasons"]
    assert "physical allowlist" in candidate["plan_authority"]["refusal"]


def test_mutated_candidate_array_is_rejected_even_with_new_artifact_sha(
    tmp_path: Path,
) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    with np.load(artifact, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["query"][0] += np.float32(1.0)
    np.savez(artifact, **values)
    _rebind_artifact(path, contract, capsule_path, artifact)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in report["candidate_results"][0]["reasons"]


def test_rms_input_must_equal_sum_derived_from_bf16_operands(tmp_path: Path) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    with np.load(artifact, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["rms_input"][2795] = np.float32(4.0)
    np.savez(artifact, **values)
    capsule = json.loads(capsule_path.read_text())
    capsule["artifact"]["sha256"] = _sha(artifact)
    rms_input = next(
        item for item in capsule["watchpoints"] if item["id"] == "layer1.rms_input_fp32"
    )
    rms_input["arrays"][0]["array_sha256"] = sha256(
        values["rms_input"].tobytes(order="C")
    ).hexdigest()
    _rebind_capsule(path, contract, capsule_path, capsule)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert "not derived from its sealed BF16 operands" in candidate["capsule"]["refusal"]


def test_capsule_rejects_unreferenced_owner_storage_rows(tmp_path: Path) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    with np.load(artifact, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["normalized"] = np.concatenate(
        [values["normalized"], values["normalized"][:1]], axis=0
    )
    np.savez(artifact, **values)
    _rebind_artifact(path, contract, capsule_path, artifact)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert "storage shape has unreferenced rows" in candidate["capsule"]["refusal"]


def test_capsule_rejects_boolean_owner_ids(tmp_path: Path) -> None:
    path, contract, capsule_path, _ = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    capsule["watchpoints"][0]["owner_ids"][0] = True
    _rebind_capsule(path, contract, capsule_path, capsule)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert "watchpoint owners are invalid" in candidate["capsule"]["refusal"]


def test_missing_compound_scorer_role_is_rejected(tmp_path: Path) -> None:
    path, contract, capsule_path, _ = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    scorer = next(
        item for item in capsule["watchpoints"] if item["id"] == "layer1.scorer_event1"
    )
    scorer["arrays"] = [item for item in scorer["arrays"] if item["role"] != "scores"]
    capsule_sha = _write_json(capsule_path, capsule)
    contract["candidates"][0]["coherent_state_capsule"]["sha256"] = capsule_sha
    _write_json(path, contract)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in report["candidate_results"][0]["reasons"]


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ("wrong_position", "layer/position drifted"),
        ("tiny_shape", "storage shape has unreferenced rows"),
        ("wrong_dtype", "array metadata drifted"),
        ("reused_slice", "owner-axis prefix drifted"),
        ("swapped_owner_prefixes", "owner-axis prefix drifted"),
        ("reversed_cache_owner_axis", "owner-axis mapping drifted"),
        ("forged_layout", "not sealed plan authority"),
    ],
)
def test_capsule_gate_d_schema_attacks_are_rejected(
    tmp_path: Path, mutation: str, detail: str
) -> None:
    path, contract, capsule_path, _ = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    if mutation == "wrong_position":
        capsule["watchpoints"][0]["position"] = 0
    elif mutation == "tiny_shape":
        capsule["watchpoints"][0]["arrays"][0]["shape"] = [1]
    elif mutation == "wrong_dtype":
        capsule["watchpoints"][0]["arrays"][0]["semantic_dtype"] = "int32"
    elif mutation == "reused_slice":
        query = next(
            item for item in capsule["watchpoints"] if item["id"] == "layer1.query"
        )
        query["arrays"][1]["index_prefix"] = query["arrays"][0]["index_prefix"]
    elif mutation == "swapped_owner_prefixes":
        query = next(
            item for item in capsule["watchpoints"] if item["id"] == "layer1.query"
        )
        query["arrays"][0]["index_prefix"], query["arrays"][1]["index_prefix"] = (
            query["arrays"][1]["index_prefix"],
            query["arrays"][0]["index_prefix"],
        )
    elif mutation == "reversed_cache_owner_axis":
        cache = next(
            item
            for item in capsule["watchpoints"]
            if item["id"] == "layer1.cache_history"
        )
        cache["arrays"][0]["owner_axis_ids"] = [1, 0]
    elif mutation == "forged_layout":
        capsule["watchpoints"][0]["layout"] = "lp4.local"
    _rebind_capsule(path, contract, capsule_path, capsule)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert detail in candidate["capsule"]["refusal"]


def test_structured_npy_dtype_is_rejected_as_validation_error(tmp_path: Path) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    with np.load(artifact, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["structured_attack"] = np.zeros((1,), dtype=[("field", "<i4")])
    np.savez(artifact, **values)
    _rebind_artifact(path, contract, capsule_path, artifact)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert "dtype/layout is unsupported" in candidate["capsule"]["refusal"]


def test_candidate_catalogue_cannot_drop_a_survivor(tmp_path: Path) -> None:
    contract = _absolute_current_contract()
    contract["candidates"] = contract["candidates"][:1]
    path = tmp_path / "contract.json"
    _write_json(path, contract)
    with pytest.raises(BenchmarkValidationError, match="catalogue drifted"):
        admit_gate_d_precompile_candidates(path, _sha(path))


def test_closed_v1_fingerprint_cannot_be_renamed(tmp_path: Path) -> None:
    contract = _absolute_current_contract()
    contract["candidates"][0]["mechanism_fingerprint"] = {
        "association": "accepted.m32",
        "consumer_boundary": "rms.weighted",
        "reduction": "m32.local",
        "representation": "sentinel.rows",
        "transport": "full.pod.hidden",
    }
    path = tmp_path / "contract.json"
    _write_json(path, contract)
    with pytest.raises(BenchmarkValidationError, match="sealed survivor"):
        admit_gate_d_precompile_candidates(path, _sha(path))


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    raw = CURRENT_CONTRACT.read_text().replace(
        '  "schema_version": 2\n',
        '  "schema_version": 2,\n  "schema_version": 2\n',
        1,
    )
    path = tmp_path / "contract.json"
    path.write_text(raw)
    with pytest.raises(BenchmarkValidationError, match="duplicate JSON key"):
        admit_gate_d_precompile_candidates(path, _sha(path))


@pytest.mark.parametrize("compression", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_unsupported_npz_compression_is_rejected(
    tmp_path: Path, compression: int
) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    payload = io.BytesIO()
    np.save(payload, np.array([1.0], dtype=np.float32), allow_pickle=False)
    with zipfile.ZipFile(artifact, "w", compression=compression) as archive:
        archive.writestr("rms_input.npy", payload.getvalue())
    _rebind_artifact(path, contract, capsule_path, artifact)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    refusal = report["candidate_results"][0]["capsule"]["refusal"]
    assert "compression is unsupported" in refusal


def test_aggregate_npz_budget_is_checked_before_member_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    payload = io.BytesIO()
    np.save(payload, np.zeros(64, dtype=np.float32), allow_pickle=False)
    with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("rms_input.npy", payload.getvalue())
    _rebind_artifact(path, contract, capsule_path, artifact)
    monkeypatch.setattr(admission_module, "_MAX_TOTAL_UNCOMPRESSED_BYTES", 64)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    refusal = report["candidate_results"][0]["capsule"]["refusal"]
    assert "uncompressed bytes exceed limit" in refusal


def test_python_s_cli_is_byte_stable_and_does_not_import_jax(tmp_path: Path) -> None:
    outputs = [tmp_path / "one.json", tmp_path / "two.json"]
    for output in outputs:
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                str(REPO_ROOT / "scripts/greenfield/admit_gate_d_precompile.py"),
                "--contract",
                str(CURRENT_CONTRACT),
                "--contract-sha256",
                CURRENT_CONTRACT_SHA256,
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert "NO_PRECOMPILE_CANDIDATE_ADMITTED" in completed.stdout
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_intermediate_symlink_is_rejected(tmp_path: Path) -> None:
    contract = _absolute_current_contract()
    evidence = Path(contract["evidence"][0]["path"])
    real = tmp_path / "real"
    real.mkdir()
    copied = real / evidence.name
    copied.write_bytes(evidence.read_bytes())
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    contract["evidence"][0]["path"] = str(linked / evidence.name)
    path = tmp_path / "contract.json"
    _write_json(path, contract)
    with pytest.raises(BenchmarkValidationError, match="cannot safely open"):
        admit_gate_d_precompile_candidates(path, _sha(path))


def test_writer_refuses_occupied_output(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("keep\n")
    with pytest.raises(BenchmarkValidationError, match="occupied"):
        write_gate_d_precompile_admission_report(output, {"value": 1})
    assert output.read_text() == "keep\n"


def test_writer_does_not_unlink_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    replacement = tmp_path / "replacement.json"
    replacement.write_text("replacement\n")
    real_fsync = admission_module.os.fsync
    calls = 0

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            os.replace(replacement, output)
            raise OSError("injected fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(admission_module.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="injected"):
        write_gate_d_precompile_admission_report(output, {"value": 1})
    assert output.read_text() == "replacement\n"
