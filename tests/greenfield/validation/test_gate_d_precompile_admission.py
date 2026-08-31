from __future__ import annotations

import ast
import base64
import errno
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import runpy
import stat
import struct
import subprocess
import sys
from types import SimpleNamespace
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
CURRENT_CONTRACT_SHA256 = "228bbfe697b269678d3fc1c4762ce3024dfe78e8d6d3a8210e98493352eee4ea"


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


def _write_canonical_json(path: Path, value: dict[str, Any]) -> str:
    path.write_text(_canonical(value) + "\n")
    return _sha(path)


def _current_lowering_dependencies() -> dict[str, list[dict[str, Any]]]:
    contract = json.loads(CURRENT_CONTRACT.read_text())
    binding = contract["candidates"][0]["stablehlo_authority"]["producer_receipt"]
    path = Path(binding["path"])
    if not path.is_absolute():
        path = CURRENT_CONTRACT.parent / path
    receipt = json.loads(path.read_text())
    return json.loads(json.dumps(receipt["loaded_dependencies"]))


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


def _current_capsule_execution_records() -> list[dict[str, str]]:
    raw = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(REPO_ROOT),
            "ls-tree",
            "-r",
            "-z",
            "HEAD",
            "--",
            "glm_tpu",
        ],
        check=True,
        capture_output=True,
    ).stdout
    records = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.split(b" ", 2)
        assert kind == b"blob"
        path = raw_path.decode("utf-8")
        if path in admission_module._CAPSULE_EXECUTION_MANIFEST_EXCLUDED_PATHS:
            continue
        records.append(
            {
                "git_object_id": object_id.decode("ascii"),
                "mode": mode.decode("ascii"),
                "path": path,
            }
        )
    replay = admission_module._EXPECTED_CAPSULE_REPLAY_SOURCE
    if not any(item["path"] == replay["repo_path"] for item in records):
        records.append(
            {
                "git_object_id": replay["git_object_id"],
                "mode": "100644",
                "path": replay["repo_path"],
            }
        )
    return sorted(records, key=lambda item: item["path"])


def test_current_concrete_tuple_source_semantics_are_bound() -> None:
    nodes, imports = _current_concrete_source_nodes()
    report = admission_module._concrete_tuple_source_semantics(nodes, imports)
    assert report["authority_scope"] == "concrete.committed.jax.source"
    assert report["frontier"] == "layer1.rms_input_fp32"


@pytest.mark.parametrize(
    "mutation",
    ("producer_path", "producer_blob", "replay_blob", "imported_kernel"),
)
def test_real_capsule_execution_implementation_attacks_fail_closed(
    mutation: str,
) -> None:
    producer = dict(admission_module._EXPECTED_CAPSULE_PRODUCER_SOURCE)
    records = _current_capsule_execution_records()
    replay_blob = (
        REPO_ROOT / admission_module._EXPECTED_CAPSULE_REPLAY_SOURCE["repo_path"]
    ).read_bytes()
    report = admission_module._verify_real_capsule_execution_source(
        producer, records, replay_blob
    )
    assert report["execution_manifest"] == (
        admission_module._EXPECTED_CAPSULE_EXECUTION_SOURCE_MANIFEST
    )
    if mutation == "producer_path":
        producer["repo_path"] = "scripts/greenfield/hostile_capsule.py"
    elif mutation == "producer_blob":
        producer["git_object_id"] = "0" * 40
        producer["sha256"] = "0" * 64
    elif mutation == "replay_blob":
        replay_blob += b"\n# hostile replay\n"
    elif mutation == "imported_kernel":
        kernel = next(
            item
            for item in records
            if item["path"] == "glm_tpu/greenfield/kernels/reference/rmsnorm.py"
        )
        kernel["git_object_id"] = "0" * 40
    with pytest.raises(BenchmarkValidationError, match="independently reviewed|closure drifted"):
        admission_module._verify_real_capsule_execution_source(
            producer, records, replay_blob
        )


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


def _synthetic_source_snapshot(repository: Path, code_pin: str) -> dict[str, Any]:
    environment = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    tree = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "ls-tree",
            "-r",
            "-z",
            code_pin,
            "--",
            "glm_tpu",
        ],
        check=True,
        capture_output=True,
        env=environment,
    ).stdout
    records = []
    for raw_entry in tree.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, kind, object_id = metadata.split(b" ", 2)
        assert kind == b"blob"
        records.append(
            {
                "git_object_id": object_id.decode("ascii"),
                "mode": mode.decode("ascii"),
                "path": raw_path.decode("utf-8"),
            }
        )
    archive = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "archive",
            "--format=zip",
            code_pin,
            "glm_tpu",
        ],
        check=True,
        capture_output=True,
        env=environment,
    ).stdout
    return {
        "archive_sha256": sha256(archive).hexdigest(),
        "file_manifest": {
            "count": len(records),
            "sha256": sha256(_canonical(records).encode("ascii")).hexdigest(),
        },
        "repository": {"commit": code_pin, "root": str(repository)},
    }


def _absolute_current_contract() -> dict[str, Any]:
    contract = json.loads(CURRENT_CONTRACT.read_text())
    base = CURRENT_CONTRACT.parent
    contract["implementation"]["core"]["sha256"] = _sha(
        REPO_ROOT / "glm_tpu/greenfield/gate_d_precompile_admission.py"
    )
    for name in ("contract", "core", "frontier"):
        binding = contract["inherited_v1"][name]
        binding["path"] = str((base / binding["path"]).resolve())
    for name in (
        "cli",
        "core",
        "git",
        "stablehlo_validator",
        "validator_python",
        "validator_python_provisioner",
        "validator_python_provisioner_source",
    ):
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
    producer_repo_path = "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py"
    producer_path = repository / producer_repo_path
    producer_path.parent.mkdir(parents=True)
    producer_raw = b"#!/usr/bin/env python3\n# Synthetic capsule producer fixture.\n"
    producer_path.write_bytes(producer_raw)
    producer_path.chmod(0o644)
    package = repository / "glm_tpu/__init__.py"
    package.parent.mkdir()
    package.write_text('"""Synthetic sealed source package."""\n')
    _run_git(
        repository,
        "add",
        "candidate.py",
        producer_repo_path,
        "glm_tpu/__init__.py",
    )
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
    producer_object_id = _run_git(
        repository, "rev-parse", f"{code_pin}:{producer_repo_path}"
    )
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
        "certificate_sha256": semantics_sha,
        "files": files,
        "source_semantic_sha256": source_semantic_sha,
        "source_set_sha256": source_set_sha,
        "producer_code_pin": code_pin,
        "producer_git_object_id": producer_object_id,
        "producer_repo_path": producer_repo_path,
        "producer_repository_root": str(repository),
        "producer_sha256": sha256(producer_raw).hexdigest(),
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
    producer_attack: str | None = None,
    metadata_attack: str | None = None,
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
    module_attributes = (
        'gate_d.unexpected = "%sum", ' if string_edge else ""
    ) + "mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32"
    module_header = (
        "module @jit_gate_d_tuple_auxiliary_rms attributes {"
        + module_attributes
        + "} {\n"
    )
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
        + "  func.func public @main(%arg0: tensor<1x6144xbf16>, "
        "%arg1: tensor<1x6144xbf16>, %arg2: tensor<6144xbf16>) "
        "-> (tensor<1x6144xbf16> {jax.result_info = \"result.output\"}, "
        "tensor<1x6144xbf16> {jax.result_info = \"result.carried_residual\"}, "
        "tensor<1x6144xf32> {jax.result_info = \"result.rms_input_fp32\"}) {\n"
        "    %hidden = stablehlo.convert %arg0 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %residual = stablehlo.convert %arg1 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %sum = stablehlo.add %hidden, %residual : tensor<1x6144xf32>\n"
        f"{carried_prefix}"
        f"    %carried = stablehlo.convert {carried_source} : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        f"{primary_extra}"
        f"    %square = chlo.square {normalization_source} : "
        "tensor<1x6144xf32> -> tensor<1x6144xf32>\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n"
        "    %variance_sum = \"stablehlo.reduce\"(%square, %zero) "
        "<{dimensions = array<i64: 1>}> ({\n"
        "    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):\n"
        "      %added = stablehlo.add %lhs, %rhs : tensor<f32>\n"
        "      stablehlo.return %added : tensor<f32>\n"
        "    }) : (tensor<1x6144xf32>, tensor<f32>) -> tensor<1xf32>\n"
        "    %variance_row = stablehlo.broadcast_in_dim %variance_sum, dims = [0] : "
        "(tensor<1xf32>) -> tensor<1x1xf32>\n"
        "    %divisor = stablehlo.constant dense<6.144000e+03> : tensor<f32>\n"
        "    %divisor_row = stablehlo.broadcast_in_dim %divisor, dims = [] : "
        "(tensor<f32>) -> tensor<1x1xf32>\n"
        "    %variance = stablehlo.divide %variance_row, %divisor_row : "
        "tensor<1x1xf32>\n"
        "    %epsilon = stablehlo.constant dense<9.99999974E-6> : tensor<f32>\n"
        "    %epsilon_row = stablehlo.broadcast_in_dim %epsilon, dims = [] : "
        "(tensor<f32>) -> tensor<1x1xf32>\n"
        "    %variance_epsilon = stablehlo.add %variance, %epsilon_row : "
        "tensor<1x1xf32>\n"
        "    %inverse = stablehlo.rsqrt %variance_epsilon : tensor<1x1xf32>\n"
        "    %scale = stablehlo.broadcast_in_dim %inverse, dims = [0, 1] : "
        "(tensor<1x1xf32>) -> tensor<1x6144xf32>\n"
        f"    %normalized = stablehlo.multiply {normalization_source}, %scale : "
        "tensor<1x6144xf32>\n"
        "    %normalized_bf16 = stablehlo.convert %normalized : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        "    %weight = stablehlo.broadcast_in_dim %arg2, dims = [1] : "
        "(tensor<6144xbf16>) -> tensor<1x6144xbf16>\n"
        "    %weighted = stablehlo.multiply %normalized_bf16, %weight : "
        "tensor<1x6144xbf16>\n"
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
            "%arg2: tensor<6144xbf16>) ",
            "%arg2: tensor<6144xbf16>, %arg3: tensor<1x6144xbf16>) ",
            1,
        )
    if extra_result:
        candidate_text = candidate_text.replace(
            'tensor<1x6144xf32> {jax.result_info = "result.rms_input_fp32"}) {\n',
            'tensor<1x6144xf32> {jax.result_info = "result.rms_input_fp32"}, '
            'tensor<1x6144xf32> {jax.result_info = "hostile"}) {\n',
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
    if metadata_attack == "candidate_replicas":
        candidate_text = candidate_text.replace(
            "mhlo.num_replicas = 1 : i32", "mhlo.num_replicas = 32 : i32", 1
        )
    elif metadata_attack == "function_name":
        candidate_text = candidate_text.replace("public @main", "public @hostile", 1)
    elif metadata_attack == "function_visibility":
        candidate_text = candidate_text.replace("func.func public", "func.func private", 1)
    elif metadata_attack == "arg_sharding":
        candidate_text = candidate_text.replace(
            "%arg0: tensor<1x6144xbf16>",
            '%arg0: tensor<1x6144xbf16> {mhlo.sharding = "{replicated}"}',
            1,
        )
    elif metadata_attack == "arg_alias":
        candidate_text = candidate_text.replace(
            "%arg0: tensor<1x6144xbf16>",
            "%arg0: tensor<1x6144xbf16> {tf.aliasing_output = 0 : i32}",
            1,
        )
    elif metadata_attack == "result_sharding":
        candidate_text = candidate_text.replace(
            '{jax.result_info = "result.output"}',
            '{jax.result_info = "result.output", mhlo.sharding = "{replicated}"}',
            1,
        )
    candidate_raw_text = candidate_text
    candidate_annotated, annotation_sha = admission_module._annotate_lowered_candidate(
        candidate_raw_text.encode("utf-8"), source
    )
    candidate_text = candidate_annotated.decode("utf-8")
    if producer_attack == "candidate_raw_drift":
        candidate_raw_text += "\n"
    accepted_normalization_source = "%hidden" if alter_accepted_primary else "%sum"
    accepted_text = (
        "module @jit_gate_d_accepted_rms attributes {mhlo.num_partitions = 1 : i32, "
        "mhlo.num_replicas = 1 : i32} {\n"
        "  func.func public @main(%arg0: tensor<1x6144xbf16>, "
        "%arg1: tensor<1x6144xbf16>, %arg2: tensor<6144xbf16>) "
        "-> (tensor<1x6144xbf16> {jax.result_info = \"result[0]\"}, "
        "tensor<1x6144xbf16> {jax.result_info = \"result[1]\"}) {\n"
        "    %hidden = stablehlo.convert %arg0 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %residual = stablehlo.convert %arg1 : "
        "(tensor<1x6144xbf16>) -> tensor<1x6144xf32>\n"
        "    %sum = stablehlo.add %hidden, %residual : tensor<1x6144xf32>\n"
        "    %carried = stablehlo.convert %sum : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        f"    %square = chlo.square {accepted_normalization_source} : "
        "tensor<1x6144xf32> -> tensor<1x6144xf32>\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n"
        "    %variance_sum = \"stablehlo.reduce\"(%square, %zero) "
        "<{dimensions = array<i64: 1>}> ({\n"
        "    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):\n"
        "      %added = stablehlo.add %lhs, %rhs : tensor<f32>\n"
        "      stablehlo.return %added : tensor<f32>\n"
        "    }) : (tensor<1x6144xf32>, tensor<f32>) -> tensor<1xf32>\n"
        "    %variance_row = stablehlo.broadcast_in_dim %variance_sum, dims = [0] : "
        "(tensor<1xf32>) -> tensor<1x1xf32>\n"
        "    %divisor = stablehlo.constant dense<6.144000e+03> : tensor<f32>\n"
        "    %divisor_row = stablehlo.broadcast_in_dim %divisor, dims = [] : "
        "(tensor<f32>) -> tensor<1x1xf32>\n"
        "    %variance = stablehlo.divide %variance_row, %divisor_row : "
        "tensor<1x1xf32>\n"
        "    %epsilon = stablehlo.constant dense<9.99999974E-6> : tensor<f32>\n"
        "    %epsilon_row = stablehlo.broadcast_in_dim %epsilon, dims = [] : "
        "(tensor<f32>) -> tensor<1x1xf32>\n"
        "    %variance_epsilon = stablehlo.add %variance, %epsilon_row : "
        "tensor<1x1xf32>\n"
        "    %inverse = stablehlo.rsqrt %variance_epsilon : tensor<1x1xf32>\n"
        "    %scale = stablehlo.broadcast_in_dim %inverse, dims = [0, 1] : "
        "(tensor<1x1xf32>) -> tensor<1x6144xf32>\n"
        f"    %normalized = stablehlo.multiply {accepted_normalization_source}, "
        "%scale : tensor<1x6144xf32>\n"
        "    %normalized_bf16 = stablehlo.convert %normalized : "
        "(tensor<1x6144xf32>) -> tensor<1x6144xbf16>\n"
        "    %weight = stablehlo.broadcast_in_dim %arg2, dims = [1] : "
        "(tensor<6144xbf16>) -> tensor<1x6144xbf16>\n"
        "    %weighted = stablehlo.multiply %normalized_bf16, %weight : "
        "tensor<1x6144xbf16>\n"
        "    return %weighted, %carried : tensor<1x6144xbf16>, "
        "tensor<1x6144xbf16>\n"
        "  }\n"
        "}\n"
    )
    if metadata_attack == "accepted_partitions":
        accepted_text = accepted_text.replace(
            "mhlo.num_partitions = 1 : i32", "mhlo.num_partitions = 2 : i32", 1
        )
    candidate_path = root / "candidate.stablehlo.mlir"
    candidate_raw_path = root / "candidate.raw.stablehlo.mlir"
    accepted_path = root / "accepted-primary.stablehlo.mlir"
    candidate_path.write_text(candidate_text)
    candidate_raw_path.write_text(candidate_raw_text)
    accepted_path.write_text(accepted_text)
    candidate_sha = _sha(candidate_path)
    candidate_raw_sha = _sha(candidate_raw_path)
    accepted_sha = _sha(accepted_path)
    fingerprint_sha = sha256(
        _canonical(candidate["mechanism_fingerprint"]).encode("ascii")
    ).hexdigest()
    validator_contract = {
        "auxiliary_result_index": 2,
        "carried_residual_result_index": 1,
        "weighted_output_result_index": 0,
    }
    producer_source_path = (
        REPO_ROOT
        / "scripts/greenfield/produce_gate_d_tuple_auxiliary_stablehlo.py"
    )
    producer_source_sha = _sha(producer_source_path)
    producer_code_pin = "6a95811d6f0e6c8f51a38a59e36c2a2b2149bead"
    loaded_dependencies = _current_lowering_dependencies()
    receipt = {
        "artifacts": {
            "accepted.raw.stablehlo": {
                "bytes": len(accepted_text.encode("utf-8")),
                "sha256": accepted_sha,
            },
            "candidate.raw.stablehlo": {
                "bytes": len(candidate_raw_text.encode("utf-8")),
                "sha256": candidate_raw_sha,
            },
            "candidate.stablehlo": {
                "bytes": len(candidate_text.encode("utf-8")),
                "sha256": candidate_sha,
            },
        },
        "backend": {"device_count": 1, "platform": "cpu"},
        "claim_scope": admission_module._EXPECTED_LOWERING_CLAIM_SCOPE,
        "environment": {
            **admission_module._EXPECTED_LOWERING_ENVIRONMENT,
            "python_version": "3.12.13 synthetic fixture",
        },
        "inputs": admission_module._EXPECTED_LOWERING_INPUTS,
        "loaded_dependencies": loaded_dependencies,
        "metadata_annotation_sha256": annotation_sha,
        "plan_authority_sha256": plan["file_sha256"],
        "producer": {
            "code_pin": producer_code_pin,
            "installed_path": admission_module._EXPECTED_LOWERING_PRODUCER_INSTALLED_PATH,
            "sha256": producer_source_sha,
            "source_path": admission_module._EXPECTED_LOWERING_PRODUCER_SOURCE_PATH,
        },
        "schema_version": 1,
        "source": {
            "callsite_ast_sha256": source["callsite_symbol"]["ast_sha256"],
            "candidate_ast_sha256": source["candidate_symbol"]["ast_sha256"],
            "certificate_sha256": source["certificate_sha256"],
            "code_pin": source["code_pin"],
            "files": {
                item["repo_path"]: item["sha256"] for item in source["files"]
            },
            "source_set_sha256": source["source_set_sha256"],
        },
    }
    if producer_attack == "backend":
        receipt["backend"]["device_count"] = 2
    elif producer_attack == "dependency_escape":
        receipt["loaded_dependencies"]["python_modules"].append(
            {
                "bytes": 1,
                "path": "/tmp/hostile.py",
                "sha256": "0" * 64,
            }
        )
    elif producer_attack == "dependency_truncate":
        receipt["loaded_dependencies"]["python_modules"].pop()
    elif producer_attack == "dependency_rebind":
        receipt["loaded_dependencies"]["native_mappings"][0]["sha256"] = "0" * 64
    elif producer_attack == "dependency_dotdot":
        receipt["loaded_dependencies"]["native_mappings"][0]["path"] = (
            "/usr/lib/../tmp/hostile.so"
        )
    elif producer_attack == "source_tuple":
        receipt["source"]["source_set_sha256"] = "0" * 64
    elif producer_attack == "producer_identity":
        receipt["producer"]["installed_path"] = "/tmp/hostile-producer.py"
    producer_receipt_path = root / "producer_receipt.json"
    producer_receipt_sha = _write_canonical_json(producer_receipt_path, receipt)
    success = {
        "producer_receipt_sha256": producer_receipt_sha,
        "schema_version": 1,
    }
    if producer_attack == "success":
        success["schema_version"] = 2
    success_path = root / "SUCCESS"
    success_sha = _write_canonical_json(success_path, success)
    certificate = {
        "accepted_primary_stablehlo_sha256": accepted_sha,
        "candidate_id": candidate["id"],
        "candidate_raw_stablehlo_sha256": candidate_raw_sha,
        "candidate_source_semantic_sha256": source["source_semantic_sha256"],
        "candidate_stablehlo_sha256": candidate_sha,
        "causal_frontier": {
            "action": candidate["causal_frontier_action"],
            "id": "layer1.rms_input_fp32",
        },
        "claim_scope": admission_module._EXPECTED_STABLEHLO_CERTIFICATE_CLAIM_SCOPE,
        "code_pin": source["code_pin"],
        "locality": locality,
        "mechanism_fingerprint_sha256": fingerprint_sha,
        "normal_form": candidate["normal_form"],
        "plan_sha256": plan["plan_sha256"],
        "producer_code_pin": producer_code_pin,
        "producer_receipt_sha256": producer_receipt_sha,
        "producer_source_sha256": producer_source_sha,
        "schema_version": 2,
        "source_set_sha256": source["source_set_sha256"],
        "success_sha256": success_sha,
        "validator_contract": validator_contract,
    }
    if producer_attack == "certificate_claim":
        certificate["claim_scope"] = "Hostile overclaim."
    certificate_path = root / "stablehlo-certificate.json"
    certificate_sha = _write_json(certificate_path, certificate)
    authority = {
        "accepted_primary": {"path": str(accepted_path), "sha256": accepted_sha},
        "candidate": {"path": str(candidate_path), "sha256": candidate_sha},
        "candidate_raw": {
            "path": str(candidate_raw_path),
            "sha256": candidate_raw_sha,
        },
        "certificate": {
            "path": str(certificate_path),
            "sha256": certificate_sha,
        },
        "producer_receipt": {
            "path": str(producer_receipt_path),
            "sha256": producer_receipt_sha,
        },
        "producer_repository": {
            "commit": producer_code_pin,
            "root": str(REPO_ROOT),
        },
        "producer_source": {
            "path": str(producer_source_path),
            "sha256": producer_source_sha,
        },
        "success": {"path": str(success_path), "sha256": success_sha},
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
                "candidate_raw_sha256": candidate_raw_sha,
                "certificate_sha256": certificate_sha,
                "mechanism_fingerprint_sha256": fingerprint_sha,
                "plan_sha256": plan["plan_sha256"],
                "producer_receipt_sha256": producer_receipt_sha,
                "producer_source_sha256": producer_source_sha,
                "source_semantic_sha256": source["source_semantic_sha256"],
                "source_set_sha256": source["source_set_sha256"],
                "success_sha256": success_sha,
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
    current_key = np.full((128,), 4.0, dtype=np.float32)
    cache_history = np.full(
        (owner_count, 16, 256, 128), 0x4000, dtype=np.uint16
    )
    logical_page_size = owner_count * 256
    current_page = 8155 // logical_page_size
    current_page_row = 8155 % logical_page_size
    current_owner_slot = current_page_row // 256
    current_local_row = current_page_row % 256
    cache_history[current_owner_slot, current_page, current_local_row] = current_key.astype(
        np.dtype("<f4"), copy=False
    ).view(np.uint32).__rshift__(16).astype(np.uint16)
    values: dict[str, np.ndarray] = {
        "rms_hidden_update": np.full((6144,), 0x3F80, dtype=np.uint16),
        "rms_residual": np.full((6144,), 0x4000, dtype=np.uint16),
        "rms_input": np.full((6144,), 3.0, dtype=np.float32),
        "normalized": np.full((owner_count, 1, 6144), 0x3F80, dtype=np.uint16),
        "cache_history": cache_history,
        "query": np.full((owner_count, 1, 32, 128), 2.0, dtype=np.float32),
        "head_weights": np.full((owner_count, 1, 32), 3.0, dtype=np.float32),
        "current_key": current_key,
        "event1_positions": np.arange(2048, dtype=np.int32)[None, :],
        "event1_scores": np.arange(2048, 0, -1, dtype=np.float32)[None, :],
        "event1_valid_count": np.array([2048], dtype=np.int32),
    }
    artifact = root / "candidate-state.npz"
    np.savez(artifact, **values)
    artifact_sha = _sha(artifact)
    device_values: dict[str, np.ndarray] = {
        "contract_valid_owners": np.ones((2,), dtype=np.uint8),
        "current_key_owners": np.stack(
            [values["current_key"][None, :], values["current_key"][None, :]]
        ),
        "rms_input_fp32_owners": np.stack(
            [values["rms_input"][None, :], values["rms_input"][None, :]]
        ),
        "selected_positions_owners": np.stack(
            [values["event1_positions"], values["event1_positions"]]
        ),
        "selected_scores_owners": np.stack(
            [values["event1_scores"], values["event1_scores"]]
        ),
        "valid_counts_owners": np.stack(
            [values["event1_valid_count"], values["event1_valid_count"]]
        ),
    }
    device_artifact = root / "candidate-device-evidence.npz"
    np.savez(device_artifact, **device_values)
    device_artifact_sha = _sha(device_artifact)
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
    input_values: dict[str, np.ndarray] = {
        "head_weight_bf16_bits": np.zeros((2, 16, 6144), dtype=np.uint16),
        "key_norm_bias_bf16_bits": np.zeros((2, 128), dtype=np.uint16),
        "key_norm_weight_bf16_bits": np.zeros((2, 128), dtype=np.uint16),
        "prompt_cache_bf16_bits": np.zeros((2, 16, 256, 128), dtype=np.uint16),
        "q_a_norm_bf16_bits": np.zeros((2048,), dtype=np.uint16),
        "qkv_a_scale_inv": np.zeros((32, 48, 82), dtype=np.float32),
        "qkv_a_weight_bits": np.zeros((32, 6144, 82), dtype=np.uint8),
        "rms_hidden_update_bf16_bits": values["rms_hidden_update"],
        "rms_residual_bf16_bits": values["rms_residual"],
        "rms_weight_bf16_bits": np.zeros((6144,), dtype=np.uint16),
        "wk_scale_inv": np.zeros((2, 1, 48), dtype=np.float32),
        "wk_weight_bits": np.zeros((2, 128, 6144), dtype=np.uint8),
        "wq_b_scale_inv": np.zeros((2, 16, 16), dtype=np.float32),
        "wq_b_weight_bits": np.zeros((2, 2048, 2048), dtype=np.uint8),
    }
    input_artifact = root / "candidate-inputs.npz"
    np.savez_compressed(input_artifact, **input_values)
    input_artifact_sha = _sha(input_artifact)
    input_records = {
        name: {
            "array_sha256": sha256(
                np.ascontiguousarray(value).tobytes(order="C")
            ).hexdigest(),
            "shape": list(value.shape),
            "storage_dtype": storage.get(value.dtype, "|u1"),
        }
        for name, value in sorted(input_values.items())
    }
    source_snapshot = _synthetic_source_snapshot(
        Path(source["producer_repository_root"]), source["producer_code_pin"]
    )
    loaded_dependencies: dict[str, list[dict[str, Any]]] = {
        "native_mappings": [],
        "python_modules": [],
    }
    installed_path = (
        Path(source["producer_repository_root"]) / source["producer_repo_path"]
    )
    installed_metadata = installed_path.stat()
    installed_producer = {
        "bytes": installed_metadata.st_size,
        "gid": installed_metadata.st_gid,
        "mode": stat.S_IMODE(installed_metadata.st_mode),
        "path": str(installed_path),
        "sha256": source["producer_sha256"],
        "uid": installed_metadata.st_uid,
    }
    tensor_receipts: list[dict[str, Any]] = []
    upstream_inputs = {
        "synthetic_input_artifact": {
            "bytes": input_artifact.stat().st_size,
            "path": str(input_artifact),
            "sha256": input_artifact_sha,
        }
    }
    manifest_records = sorted(
        (
            {
                **watchpoint,
                "arrays": sorted(
                    watchpoint["arrays"], key=lambda item: item["role"]
                ),
            }
            for watchpoint in watchpoints
        ),
        key=lambda item: item["id"],
    )
    receipt = {
        "artifact": {"bytes": artifact.stat().st_size, "sha256": artifact_sha},
        "backend": {"device_count": 2, "device_ids": [0, 1], "platform": "cpu"},
        "candidate": {
            "code_pin": source["code_pin"],
            "id": candidate["id"],
            "plan_sha256": plan["plan_sha256"],
            "source_authority_sha256": source["authority_sha256"],
            "stablehlo_authority_sha256": stablehlo["authority_sha256"],
        },
        "claim_scope": admission_module._EXPECTED_CAPSULE_PRODUCER_CLAIM_SCOPE,
        "coherence_id": "synthetic.gate.d.precompile",
        "device_evidence": {
            "bytes": device_artifact.stat().st_size,
            "sha256": device_artifact_sha,
        },
        "environment": {
            **admission_module._EXPECTED_CAPSULE_ENVIRONMENT,
            "python_version": sys.version,
        },
        "execution": admission_module._EXPECTED_CAPSULE_EXECUTION,
        "input_arrays": input_records,
        "installed_producer": installed_producer,
        "loaded_dependencies": loaded_dependencies,
        "producer": {
            "git_object_id": source["producer_git_object_id"],
            "repo_path": source["producer_repo_path"],
            "repository": {
                "commit": source["producer_code_pin"],
                "root": source["producer_repository_root"],
            },
            "sha256": source["producer_sha256"],
        },
        "schema_version": 1,
        "source_snapshot": source_snapshot,
        "tensor_receipts": tensor_receipts,
        "upstream_inputs": upstream_inputs,
        "watchpoint_manifest_sha256": sha256(
            _canonical({"watchpoints": manifest_records}).encode("ascii")
        ).hexdigest(),
    }
    receipt_path = root / "capsule-producer-receipt.json"
    receipt_sha = _write_canonical_json(receipt_path, receipt)
    success = {
        "artifact_sha256": artifact_sha,
        "device_evidence_sha256": device_artifact_sha,
        "input_artifact_sha256": input_artifact_sha,
        "producer_receipt_sha256": receipt_sha,
        "schema_version": 1,
    }
    success_path = root / "capsule-SUCCESS"
    success_sha = _write_canonical_json(success_path, success)
    document = {
        "artifact": {"path": str(artifact), "sha256": artifact_sha},
        "candidate_id": candidate["id"],
        "claim_scope": "Synthetic candidate-coherent capsule fixture only.",
        "code_pin": source["code_pin"],
        "coherence_id": "synthetic.gate.d.precompile",
        "plan_sha256": plan["plan_sha256"],
        "producer_input_artifact": {
            "path": str(input_artifact),
            "sha256": input_artifact_sha,
        },
        "producer_device_evidence": {
            "path": str(device_artifact),
            "sha256": device_artifact_sha,
        },
        "producer_receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "producer_success": {"path": str(success_path), "sha256": success_sha},
        "schema_version": 2,
        "source_authority_sha256": source["authority_sha256"],
        "stablehlo_authority_sha256": stablehlo["authority_sha256"],
        "watchpoints": watchpoints,
    }
    path = root / "capsule.json"
    capsule_sha = _write_json(path, document)
    return {"path": str(path), "sha256": capsule_sha}, path, artifact


def _capsule_execution_authority(
    root: Path,
    candidate: dict[str, Any],
    capsule_path: Path,
    artifact: Path,
) -> dict[str, str]:
    capsule = json.loads(capsule_path.read_text())
    receipt_path = _binding_path(
        capsule_path.parent, capsule["producer_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text())
    with np.load(artifact, allow_pickle=False) as archive:
        positions = np.ascontiguousarray(archive["event1_positions"])
        scores = np.ascontiguousarray(archive["event1_scores"])
        valid_count = int(np.asarray(archive["event1_valid_count"]).reshape(-1)[0])
        hidden_update = np.ascontiguousarray(archive["rms_hidden_update"])
        residual = np.ascontiguousarray(archive["rms_residual"])
    authority = {
        "authority_kind": "gate.d.capsule.execution.v1",
        "candidate_id": candidate["id"],
        "environment": receipt["environment"],
        "expected_outputs": {
            "event1_positions_sha256": sha256(positions.tobytes(order="C")).hexdigest(),
            "event1_scores_sha256": sha256(scores.tobytes(order="C")).hexdigest(),
            "event1_valid_count": valid_count,
            "rms_hidden_update_sha256": sha256(
                hidden_update.tobytes(order="C")
            ).hexdigest(),
            "rms_residual_sha256": sha256(residual.tobytes(order="C")).hexdigest(),
        },
        "input_arrays": receipt["input_arrays"],
        "installed_producer": receipt["installed_producer"],
        "loaded_dependencies": receipt["loaded_dependencies"],
        "producer": receipt["producer"],
        "schema_version": 1,
        "source_snapshot": receipt["source_snapshot"],
        "tensor_receipts": receipt["tensor_receipts"],
        "upstream_inputs": receipt["upstream_inputs"],
    }
    path = root / "capsule-execution-authority.json"
    return {"path": str(path), "sha256": _write_canonical_json(path, authority)}


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
    producer_attack: str | None = None,
    metadata_attack: str | None = None,
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
        producer_attack=producer_attack,
        metadata_attack=metadata_attack,
    )
    capsule_binding, capsule_path, artifact = _capsule(
        root, candidate, source, stablehlo, plan
    )
    candidate["source_authority"] = source_authority
    candidate["plan_authority"] = plan_authority
    candidate["stablehlo_authority"] = stablehlo_authority
    candidate["capsule_execution_authority"] = _capsule_execution_authority(
        root, candidate, capsule_path, artifact
    )
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


def _rebind_capsule_execution_authority(
    contract_path: Path, contract: dict[str, Any], authority_path: Path
) -> None:
    contract["candidates"][0]["capsule_execution_authority"]["sha256"] = _sha(
        authority_path
    )
    _write_json(contract_path, contract)


def _binding_path(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _refresh_watchpoint_array_hashes(
    capsule: dict[str, Any], values: dict[str, np.ndarray], array_key: str
) -> None:
    for watchpoint in capsule["watchpoints"]:
        for array in watchpoint["arrays"]:
            if array["array_key"] != array_key:
                continue
            value = values[array_key]
            prefix = tuple(array["index_prefix"])
            selected = value[prefix] if prefix else value
            array["array_sha256"] = sha256(
                np.ascontiguousarray(selected).tobytes(order="C")
            ).hexdigest()


def _rebind_capsule_producer_chain(
    contract_path: Path,
    contract: dict[str, Any],
    capsule_path: Path,
    capsule: dict[str, Any],
    *,
    artifact: Path | None = None,
) -> None:
    receipt_path = _binding_path(
        capsule_path.parent, capsule["producer_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text())
    if artifact is not None:
        artifact_sha = _sha(artifact)
        capsule["artifact"]["sha256"] = artifact_sha
        receipt["artifact"] = {
            "bytes": artifact.stat().st_size,
            "sha256": artifact_sha,
        }
    manifest_records = sorted(
        (
            {
                **watchpoint,
                "arrays": sorted(
                    watchpoint["arrays"], key=lambda item: item["role"]
                ),
            }
            for watchpoint in capsule["watchpoints"]
        ),
        key=lambda item: item["id"],
    )
    receipt["watchpoint_manifest_sha256"] = sha256(
        _canonical({"watchpoints": manifest_records}).encode("ascii")
    ).hexdigest()
    receipt_sha = _write_canonical_json(receipt_path, receipt)
    capsule["producer_receipt"]["sha256"] = receipt_sha
    success_path = _binding_path(
        capsule_path.parent, capsule["producer_success"]["path"]
    )
    success = json.loads(success_path.read_text())
    success["producer_receipt_sha256"] = receipt_sha
    if artifact is not None:
        success["artifact_sha256"] = capsule["artifact"]["sha256"]
    capsule["producer_success"]["sha256"] = _write_canonical_json(
        success_path, success
    )
    _rebind_capsule(contract_path, contract, capsule_path, capsule)


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
            "MISSING_PINNED_COHERENT_CAPSULE_PRODUCER",
            "MISSING_CANDIDATE_COHERENT_CAPSULE",
        ],
        "compensated_auxiliary_dependency": [
            "MISSING_SOURCE_AST_AUTHORITY",
            "MISSING_PLAN_AUTHORITY",
            "MISSING_CAUSAL_STABLEHLO_AUTHORITY",
            "MISSING_PINNED_COHERENT_CAPSULE_PRODUCER",
            "MISSING_CANDIDATE_COHERENT_CAPSULE",
        ],
    }
    source = report["candidate_results"][0]["source_authority"]
    assert source["authority_kind"] == "concrete.committed.jax.source"
    assert source["executable_source_authority"] is True
    assert source["code_pin"] == "c8b220067577004ddfb824667eadc746642baaca"
    plan = report["candidate_results"][0]["plan_authority"]
    assert plan["plan"] == "PP16_LP2"
    assert plan["local_group_size"] == 2
    assert plan["owner_group"] == [0, 1]
    assert plan["plan_sha256"] == (
        "d824c19c4393e3b54767ea3a4a228bc865bac1b218fbce63eb04df7a05bc5833"
    )
    stablehlo = report["candidate_results"][0]["stablehlo_authority"]
    assert stablehlo["accepted_primary_slice_sha256"] == (
        "5037b5a7ef21226f8405d0175c83bc7528fadf588811755e719d20a75f295610"
    )
    assert stablehlo["auxiliary_slice_sha256"] == (
        "02eee1d7a84fd447396577c21c1d2988f23d9b6312b8fa0df4838a9b41642f7c"
    )
    assert stablehlo["collectives"] == []
    assert stablehlo["lowering_receipt"]["backend"] == {
        "device_count": 1,
        "platform": "cpu",
    }


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
    ]
    assert candidate["source_authority"]["executable_source_authority"] is False
    assert (
        candidate["stablehlo_authority"]["immutable_parser_authority"]
        is True
    )
    assert candidate["capsule"]["producer_provenance_verified"] is True
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


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ("source_archive", "committed source snapshot drifted"),
        ("upstream_file", "upstream input drifted"),
        ("installed_producer", "installed capsule producer drifted"),
        ("python_dependency_escape", "escaped sealed roots"),
    ],
)
def test_capsule_execution_authority_attacks_fail_closed(
    tmp_path: Path, mutation: str, detail: str
) -> None:
    path, contract, _, _ = _prepared_contract(tmp_path)
    binding = contract["candidates"][0]["capsule_execution_authority"]
    authority_path = Path(binding["path"])
    authority = json.loads(authority_path.read_text())
    if mutation == "source_archive":
        authority["source_snapshot"]["archive_sha256"] = "0" * 64
        _write_canonical_json(authority_path, authority)
    elif mutation == "upstream_file":
        upstream = Path(
            authority["upstream_inputs"]["synthetic_input_artifact"]["path"]
        )
        with upstream.open("ab") as stream:
            stream.write(b"hostile")
    elif mutation == "installed_producer":
        installed = Path(authority["installed_producer"]["path"])
        installed.write_bytes(installed.read_bytes() + b"# hostile\n")
        installed.chmod(0o644)
    elif mutation == "python_dependency_escape":
        escaped = tmp_path / "hostile.py"
        escaped.write_bytes(b"x = 1\n")
        escaped.chmod(0o644)
        authority["loaded_dependencies"]["python_modules"] = [
            {
                "bytes": escaped.stat().st_size,
                "path": str(escaped),
                "sha256": _sha(escaped),
            }
        ]
        _write_canonical_json(authority_path, authority)
    _rebind_capsule_execution_authority(path, contract, authority_path)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CAPSULE_EXECUTION_AUTHORITY" in candidate["reasons"]
    assert detail in candidate["capsule_execution_authority"]["refusal"]


def test_real_capsule_tensor_receipts_bind_manifest_header_and_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "runtime_manifest.json"
    files = []
    receipts = []
    for slot in (0, 1):
        payload = struct.pack("<ff", float(slot + 1), float(slot + 2))
        header_document = {
            "tensor": {
                "data_offsets": [0, len(payload)],
                "dtype": "F32",
                "shape": [2],
            }
        }
        header_json = _canonical(header_document).encode("ascii")
        raw_header = struct.pack("<Q", len(header_json)) + header_json
        owner_path = tmp_path / f"owner-{slot}.safetensors"
        owner_path.write_bytes(raw_header + payload)
        payload_sha = sha256(payload).hexdigest()
        files.append(
            {
                "destination_filename": owner_path.name,
                "device_slot": slot,
                "file_bytes": owner_path.stat().st_size,
                "header_bytes": len(raw_header),
                "header_sha256": sha256(raw_header).hexdigest(),
                "stage_id": 0,
                "tensors": [
                    {
                        "byte_count": len(payload),
                        "name": "tensor",
                        "sha256": payload_sha,
                    }
                ],
            }
        )
        receipts.append(
            {
                "byte_count": len(payload),
                "device_slot": slot,
                "file_bytes": owner_path.stat().st_size,
                "file_path": str(owner_path),
                "name": "tensor",
                "offset": len(raw_header),
                "sha256": payload_sha,
                "shape": [2],
            }
        )
    manifest_raw = (_canonical({"files": files}) + "\n").encode("ascii")
    manifest_path.write_bytes(manifest_raw)
    monkeypatch.setattr(
        admission_module, "_EXPECTED_CAPSULE_TENSOR_NAMES", ("tensor",)
    )
    monkeypatch.setattr(
        admission_module,
        "_CAPSULE_RUNTIME_INPUT_SOURCES",
        {"runtime_input": ("tensor", (0, 1))},
    )
    monkeypatch.setattr(
        admission_module,
        "_CAPSULE_INPUT_SCHEMA",
        {"runtime_input": ("<f4", (2, 2))},
    )
    monkeypatch.setattr(
        admission_module,
        "_EXPECTED_CAPSULE_UPSTREAM_INPUTS",
        {"runtime_manifest": (str(manifest_path), sha256(manifest_raw).hexdigest())},
    )
    monkeypatch.setattr(
        admission_module, "_EXPECTED_CAPSULE_RUNTIME_DATA_ROOT", tmp_path
    )
    runtime_paths = tuple(tmp_path / f"owner-{slot}.safetensors" for slot in (0, 1))
    with runtime_paths[0].open("rb") as stream:
        metadata = os.fstat(stream.fileno())
        descriptor_fields = dict(
            line.split(":", 1)
            for line in Path(f"/proc/self/fdinfo/{stream.fileno()}")
            .read_text()
            .splitlines()
            if ":" in line
        )
    runtime_mount_authority = {
        "major": os.major(metadata.st_dev),
        "minor": os.minor(metadata.st_dev),
        "mount_id": int(descriptor_fields["mnt_id"].strip()),
    }
    observed_receipts, observed_inputs = (
        admission_module._verify_capsule_tensor_receipts(
            receipts,
            real_source=True,
            runtime_manifest_raw=manifest_raw,
            runtime_paths=runtime_paths,
            runtime_mount_authority=runtime_mount_authority,
        )
    )
    assert observed_receipts == receipts
    assert observed_inputs == {
        "runtime_input": {
            "array_sha256": sha256(
                struct.pack("<ffff", 1.0, 2.0, 2.0, 3.0)
            ).hexdigest(),
            "shape": [2, 2],
            "storage_dtype": "<f4",
        }
    }
    hostile = json.loads(json.dumps(receipts))
    hostile[0]["offset"] += 1
    with pytest.raises(BenchmarkValidationError, match="tensor receipt drifted"):
        admission_module._verify_capsule_tensor_receipts(
            hostile,
            real_source=True,
            runtime_manifest_raw=manifest_raw,
            runtime_paths=runtime_paths,
            runtime_mount_authority=runtime_mount_authority,
        )


def test_capsule_producer_uses_pre_execution_snapshots_and_retained_fds() -> None:
    source_path = (
        REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py"
    )
    tree = ast.parse(source_path.read_text())
    functions = {
        item.name: item for item in tree.body if isinstance(item, ast.FunctionDef)
    }
    main = functions["main"]
    snapshot_lines = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_snapshot_regular"
    ]
    execute_lines = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_execute"
    ]
    mount_lines = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_verify_runtime_data_mount"
    ]
    assert snapshot_lines and execute_lines
    assert mount_lines
    assert min(mount_lines) < min(execute_lines)
    assert max(snapshot_lines) < min(execute_lines)
    input_source = ast.unparse(functions["_input_values"])
    runtime_source = ast.unparse(functions["_read_runtime"])
    assert "_file_sha256" not in input_source
    assert "read_text" not in input_source
    assert "BytesIO(snapshots" in input_source
    assert "os.pread" in runtime_source
    assert ".open(" not in runtime_source
    assert "_revalidate_snapshot" in functions
    assert "_revalidate_tensor_receipts" in functions

    admission_tree = ast.parse(
        (REPO_ROOT / "glm_tpu/greenfield/gate_d_precompile_admission.py").read_text()
    )
    execution_authority = next(
        item
        for item in admission_tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_verify_capsule_execution_authority"
    )
    admission_mount_lines = [
        node.lineno
        for node in ast.walk(execution_authority)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_verify_capsule_runtime_data_mount"
    ]
    receipt_lines = [
        node.lineno
        for node in ast.walk(execution_authority)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_verify_capsule_tensor_receipts"
    ]
    assert len(admission_mount_lines) == 2
    assert len(receipt_lines) == 1
    assert min(admission_mount_lines) < receipt_lines[0] < max(admission_mount_lines)


_EXACT_RUNTIME_MOUNTINFO = (
    b"535 516 0:4 mnt:[4026532862] /run/snapd/ns/lxd.mnt rw - nsfs nsfs rw\n"
    b"44 73 0:43 / /home/gianl/gcs-models ro,nosuid,nodev,relatime shared:49 - "
    b"fuse.gcsfuse driftbench-dsv4-uc "
    b"ro,user_id=2001,group_id=2001,default_permissions\n"
)
_RUNTIME_DATA_ROOT = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/"
    "greenfield_runtime_feature_qkv_direct_pp16_20260827T164842844148623Z"
)
_RUNTIME_PAYLOAD_PATHS = tuple(
    _RUNTIME_DATA_ROOT
    / "base_decoder_runtime_feature"
    / "stage_00"
    / f"device_slot_{slot:02d}.safetensors"
    for slot in (0, 1)
)


@pytest.mark.parametrize(
    "hostile_mountinfo",
    (
        _EXACT_RUNTIME_MOUNTINFO.replace(b"ro,nosuid", b"rw,nosuid", 1),
        _EXACT_RUNTIME_MOUNTINFO.replace(
            b"driftbench-dsv4-uc", b"driftbench-storage"
        ),
        _EXACT_RUNTIME_MOUNTINFO.replace(
            b"/home/gianl/gcs-models", b"/home/gianl/gcs-models-other"
        ),
        _EXACT_RUNTIME_MOUNTINFO + _EXACT_RUNTIME_MOUNTINFO,
    ),
)
def test_capsule_runtime_data_mount_authority_fails_closed(
    hostile_mountinfo: bytes,
) -> None:
    producer = runpy.run_path(
        REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py",
        run_name="gate_d_capsule_mount_test",
    )
    assert producer["RUNTIME_AUTHORITY_ROOT"] != producer["RUNTIME_DATA_ROOT"]
    expected = {
        "major": 0,
        "minor": 43,
        "mount_id": 44,
        "raw_record": _EXACT_RUNTIME_MOUNTINFO.splitlines()[1],
    }
    assert producer["_verify_runtime_data_mount"](
        _EXACT_RUNTIME_MOUNTINFO, _RUNTIME_PAYLOAD_PATHS
    ) == expected
    assert admission_module._verify_capsule_runtime_data_mount(
        _EXACT_RUNTIME_MOUNTINFO, _RUNTIME_PAYLOAD_PATHS
    ) == expected
    with pytest.raises(RuntimeError, match="runtime data mount authority"):
        producer["_verify_runtime_data_mount"](
            hostile_mountinfo, _RUNTIME_PAYLOAD_PATHS
        )
    with pytest.raises(
        BenchmarkValidationError, match="runtime data mount authority"
    ):
        admission_module._verify_capsule_runtime_data_mount(
            hostile_mountinfo, _RUNTIME_PAYLOAD_PATHS
        )


def test_capsule_runtime_payload_nested_mount_fails_closed() -> None:
    nested = (
        b"99 44 0:99 / "
        + str(_RUNTIME_DATA_ROOT / "base_decoder_runtime_feature").encode("ascii")
        + b" rw,nosuid,nodev,relatime - tmpfs hostile rw\n"
    )
    hostile_mountinfo = _EXACT_RUNTIME_MOUNTINFO + nested
    producer = runpy.run_path(
        REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py",
        run_name="gate_d_capsule_nested_mount_test",
    )
    with pytest.raises(RuntimeError, match="covered by a nested mount"):
        producer["_verify_runtime_data_mount"](
            hostile_mountinfo, _RUNTIME_PAYLOAD_PATHS
        )
    with pytest.raises(BenchmarkValidationError, match="covered by a nested mount"):
        admission_module._verify_capsule_runtime_data_mount(
            hostile_mountinfo, _RUNTIME_PAYLOAD_PATHS
        )


def test_capsule_payload_descriptor_binds_mount_identity(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"payload")
    producer = runpy.run_path(
        REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py",
        run_name="gate_d_capsule_descriptor_mount_test",
    )
    with payload.open("rb") as stream:
        metadata = os.fstat(stream.fileno())
        fields = dict(
            line.split(":", 1)
            for line in Path(f"/proc/self/fdinfo/{stream.fileno()}")
            .read_text()
            .splitlines()
            if ":" in line
        )
        authority = {
            "major": os.major(metadata.st_dev),
            "minor": os.minor(metadata.st_dev),
            "mount_id": int(fields["mnt_id"].strip()),
        }
        producer["_verify_payload_descriptor_mount"](
            stream.fileno(), metadata, authority
        )
        admission_module._verify_capsule_payload_descriptor_mount(
            stream.fileno(), metadata, authority
        )
        for key in ("mount_id", "major", "minor"):
            hostile = dict(authority)
            hostile[key] += 1
            with pytest.raises(RuntimeError, match="escaped mount authority"):
                producer["_verify_payload_descriptor_mount"](
                    stream.fileno(), metadata, hostile
                )
            with pytest.raises(
                BenchmarkValidationError, match="escaped mount authority"
            ):
                admission_module._verify_capsule_payload_descriptor_mount(
                    stream.fileno(), metadata, hostile
                )


def test_capsule_runtime_data_root_cannot_escape_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = runpy.run_path(
        REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py",
        run_name="gate_d_capsule_mount_escape_test",
    )
    producer["_verify_runtime_data_mount"].__globals__["RUNTIME_DATA_ROOT"] = Path(
        "/tmp/hostile-runtime-root"
    )
    with pytest.raises(RuntimeError, match="data root escaped"):
        producer["_verify_runtime_data_mount"](
            _EXACT_RUNTIME_MOUNTINFO, _RUNTIME_PAYLOAD_PATHS
        )
    monkeypatch.setattr(
        admission_module,
        "_EXPECTED_CAPSULE_RUNTIME_DATA_ROOT",
        Path("/tmp/hostile-runtime-root"),
    )
    with pytest.raises(BenchmarkValidationError, match="data root escaped"):
        admission_module._verify_capsule_runtime_data_mount(
            _EXACT_RUNTIME_MOUNTINFO, _RUNTIME_PAYLOAD_PATHS
        )


@pytest.mark.parametrize(
    "hostile_success",
    (
        b'{"runtime_manifest": true}\n',
        (
            b"b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab "
            b"runtime_manifest.json\n"
        ),
        (
            b"b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab\t"
            b"runtime_manifest.json\n"
        ),
        b"0" * 64 + b"  runtime_manifest.json\n",
        (
            b"b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab  "
            b"runtime_manifest.json\nextra\n"
        ),
    ),
)
def test_capsule_producer_binds_exact_runtime_success_format(
    hostile_success: bytes,
) -> None:
    scope = runpy.run_path(
        REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py",
        run_name="gate_d_capsule_producer_test",
    )
    manifest = {"manifest_sha256": scope["RUNTIME_MANIFEST_SELF_SHA256"]}
    scope["_verify_runtime_success"](manifest, scope["RUNTIME_SUCCESS_PAYLOAD"])
    with pytest.raises(RuntimeError, match="runtime SUCCESS contract drifted"):
        scope["_verify_runtime_success"](manifest, hostile_success)


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ("backend", "not forced CPU"),
        ("environment", "environment drifted"),
        ("execution", "execution scope drifted"),
        ("producer_object", "committed capsule producer drifted"),
        ("input_manifest", "input array drifted"),
        ("source_snapshot", "not the pinned execution authority"),
        ("dependencies", "not the pinned execution authority"),
        ("upstream_inputs", "not the pinned execution authority"),
        ("tensor_receipts", "not the pinned execution authority"),
        ("success", "SUCCESS drifted"),
    ],
)
def test_capsule_producer_receipt_attacks_fail_closed(
    tmp_path: Path, mutation: str, detail: str
) -> None:
    path, contract, capsule_path, _ = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    receipt_path = _binding_path(
        capsule_path.parent, capsule["producer_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text())
    if mutation == "backend":
        receipt["backend"]["device_count"] = 1
    elif mutation == "environment":
        receipt["environment"]["jax_version"] = "0.0.0"
    elif mutation == "execution":
        receipt["execution"]["tpus_used"] = 1
    elif mutation == "producer_object":
        receipt["producer"]["git_object_id"] = "0" * 40
    elif mutation == "input_manifest":
        receipt["input_arrays"]["qkv_a_weight_bits"]["array_sha256"] = "0" * 64
    elif mutation == "source_snapshot":
        receipt["source_snapshot"]["archive_sha256"] = "0" * 64
    elif mutation == "dependencies":
        receipt["loaded_dependencies"]["python_modules"].append(
            {"bytes": 1, "path": "/tmp/hostile.py", "sha256": "0" * 64}
        )
    elif mutation == "upstream_inputs":
        receipt["upstream_inputs"]["synthetic_input_artifact"]["sha256"] = "0" * 64
    elif mutation == "tensor_receipts":
        receipt["tensor_receipts"] = [{"hostile": True}]
    elif mutation == "success":
        success_path = _binding_path(
            capsule_path.parent, capsule["producer_success"]["path"]
        )
        success = json.loads(success_path.read_text())
        success["schema_version"] = 2
        _write_canonical_json(success_path, success)
    _write_canonical_json(receipt_path, receipt)
    _rebind_capsule_producer_chain(path, contract, capsule_path, capsule)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert detail in candidate["capsule"]["refusal"]


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ("normalized_owner", "normalized owner values differ"),
        ("cache_current_key", "not the BF16 round of its current key"),
        ("duplicate_position", "not one exact ordered cutoff-active set"),
        ("ascending_score", "not one exact ordered cutoff-active set"),
    ],
)
def test_capsule_coherence_attacks_fail_closed_after_full_rebinding(
    tmp_path: Path, mutation: str, detail: str
) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    with np.load(artifact, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    if mutation == "normalized_owner":
        values["normalized"][1, 0, 2795] ^= np.uint16(1)
        changed_keys = ("normalized",)
    elif mutation == "cache_current_key":
        values["cache_history"][1, 15, 219] = np.uint16(0)
        changed_keys = ("cache_history",)
    elif mutation == "duplicate_position":
        values["event1_positions"][0, 1] = values["event1_positions"][0, 0]
        changed_keys = ("event1_positions",)
    else:
        values["event1_scores"][0, 100] = np.float32(-1.0)
        changed_keys = ("event1_scores",)
    np.savez(artifact, **values)
    for key in changed_keys:
        _refresh_watchpoint_array_hashes(capsule, values, key)
    _rebind_capsule_producer_chain(
        path, contract, capsule_path, capsule, artifact=artifact
    )
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert detail in candidate["capsule"]["refusal"]


@pytest.mark.parametrize(
    ("array_key", "hostile_value"),
    [
        ("normalized", np.uint16(0x7F80)),
        ("query", np.float32(np.inf)),
        ("head_weights", np.float32(np.nan)),
    ],
)
def test_capsule_nonfinite_watchpoints_fail_closed_after_rebinding(
    tmp_path: Path, array_key: str, hostile_value: Any
) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    with np.load(artifact, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values[array_key].reshape(-1)[0] = hostile_value
    if array_key in {"normalized", "query", "head_weights"}:
        owner_width = values[array_key][0].size
        values[array_key].reshape(-1)[owner_width] = hostile_value
    np.savez(artifact, **values)
    _refresh_watchpoint_array_hashes(capsule, values, array_key)
    _rebind_capsule_producer_chain(
        path, contract, capsule_path, capsule, artifact=artifact
    )
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert "contains non-finite" in candidate["capsule"]["refusal"]


def test_capsule_nonfinite_input_scale_fails_after_full_authority_rebinding(
    tmp_path: Path,
) -> None:
    path, contract, capsule_path, _ = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    input_path = _binding_path(
        capsule_path.parent, capsule["producer_input_artifact"]["path"]
    )
    with np.load(input_path, allow_pickle=False) as archive:
        inputs = {name: archive[name].copy() for name in archive.files}
    inputs["qkv_a_scale_inv"][0, 0, 0] = np.float32(np.inf)
    np.savez_compressed(input_path, **inputs)
    input_sha = _sha(input_path)
    input_array_sha = sha256(
        np.ascontiguousarray(inputs["qkv_a_scale_inv"]).tobytes(order="C")
    ).hexdigest()
    capsule["producer_input_artifact"]["sha256"] = input_sha
    receipt_path = _binding_path(
        capsule_path.parent, capsule["producer_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["input_arrays"]["qkv_a_scale_inv"]["array_sha256"] = input_array_sha
    receipt["upstream_inputs"]["synthetic_input_artifact"] = {
        "bytes": input_path.stat().st_size,
        "path": str(input_path),
        "sha256": input_sha,
    }
    receipt_sha = _write_canonical_json(receipt_path, receipt)
    capsule["producer_receipt"]["sha256"] = receipt_sha
    success_path = _binding_path(
        capsule_path.parent, capsule["producer_success"]["path"]
    )
    success = json.loads(success_path.read_text())
    success["input_artifact_sha256"] = input_sha
    success["producer_receipt_sha256"] = receipt_sha
    capsule["producer_success"]["sha256"] = _write_canonical_json(
        success_path, success
    )
    authority_binding = contract["candidates"][0]["capsule_execution_authority"]
    authority_path = Path(authority_binding["path"])
    authority = json.loads(authority_path.read_text())
    authority["input_arrays"] = receipt["input_arrays"]
    authority["upstream_inputs"] = receipt["upstream_inputs"]
    authority_binding["sha256"] = _write_canonical_json(authority_path, authority)
    _rebind_capsule(path, contract, capsule_path, capsule)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert "contains non-finite FP32 values" in candidate["capsule"]["refusal"]


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


@pytest.mark.parametrize(
    ("producer_attack", "refusal"),
    [
        ("backend", "not forced CPU"),
        ("candidate_raw_drift", "annotation drifted"),
        ("dependency_escape", "escaped immutable roots"),
        ("dependency_truncate", "python modules manifest drifted"),
        ("dependency_rebind", "native mappings manifest drifted"),
        ("dependency_dotdot", "unsafe or duplicated"),
        ("producer_identity", "producer identity drifted"),
        ("source_tuple", "producer source tuple drifted"),
        ("success", "SUCCESS drifted"),
        ("certificate_claim", "certificate claim scope drifted"),
    ],
)
def test_producer_authority_attacks_fail_closed(
    tmp_path: Path, producer_attack: str, refusal: str
) -> None:
    path, _, _, _ = _prepared_contract(
        tmp_path, producer_attack=producer_attack
    )
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert candidate["admitted_precompile"] is False
    assert "INVALID_CAUSAL_STABLEHLO_AUTHORITY" in candidate["reasons"]
    assert refusal in candidate["stablehlo_authority"]["refusal"]


@pytest.mark.parametrize(
    ("metadata_attack", "refusal"),
    [
        ("candidate_replicas", "candidate module metadata drifted"),
        ("accepted_partitions", "accepted-primary module metadata drifted"),
        ("function_name", "candidate function metadata drifted"),
        ("function_visibility", "candidate function metadata drifted"),
        ("arg_sharding", "candidate function metadata drifted"),
        ("arg_alias", "candidate function metadata drifted"),
        ("result_sharding", "candidate function metadata drifted"),
    ],
)
def test_compilation_metadata_attacks_fail_closed(
    tmp_path: Path, metadata_attack: str, refusal: str
) -> None:
    path, _, _, _ = _prepared_contract(
        tmp_path, metadata_attack=metadata_attack
    )
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert candidate["admitted_precompile"] is False
    assert "INVALID_CAUSAL_STABLEHLO_AUTHORITY" in candidate["reasons"]
    assert refusal in candidate["stablehlo_authority"]["refusal"]


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


def test_parser_source_replace_after_memfd_seal_cannot_change_child_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _absolute_current_contract()
    candidate = contract["candidates"][0]
    locality = contract["locality_contract"]
    _, source = _source_authority(tmp_path, candidate, locality)
    _, plan = _plan_authority(tmp_path, "PP16_LP2")
    original_root = Path(contract["implementation"]["validator_pythonpath"])
    parser_root = tmp_path / "parser"
    for item in contract["implementation"]["validator_imports"]:
        relative = Path(item["path"])
        source_path = original_root / relative
        destination = parser_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == ".so":
            os.link(source_path, destination)
        else:
            destination.write_bytes(source_path.read_bytes())
    contract["implementation"]["validator_pythonpath"] = str(parser_root)
    monkeypatch.setattr(
        sys.modules[__name__], "_absolute_current_contract", lambda: contract
    )
    target = parser_root / "jaxlib/mlir/ir.py"
    replacement = parser_root / "replacement.py"
    replacement.write_bytes(b"raise RuntimeError('hostile parser replacement')\n")
    real_run = admission_module.subprocess.run
    replaced = False

    def replace_after_sealing(*args: Any, **kwargs: Any) -> Any:
        nonlocal replaced
        os.replace(replacement, target)
        replaced = True
        return real_run(*args, **kwargs)

    monkeypatch.setattr(admission_module.subprocess, "run", replace_after_sealing)
    _, stablehlo = _stablehlo_authority(
        tmp_path, candidate, locality, source, plan
    )
    assert replaced is True
    assert stablehlo["authority_sha256"] != "0" * 64
    assert target.read_bytes().startswith(b"raise RuntimeError")


def test_parser_memfd_hashes_the_same_bytes_it_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    replacement = tmp_path / "replacement.bin"
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
    descriptor = admission_module._sealed_parser_memfd(
        source,
        logical_path="jaxlib/source.bin",
        expected_bytes=len(original),
        expected_sha256=sha256(original).hexdigest(),
    )
    try:
        assert replaced is True
        assert os.pread(descriptor, len(original), 0) == original
        assert source.read_bytes() != original
        assert admission_module.fcntl.fcntl(
            descriptor, admission_module._F_GET_SEALS
        ) == admission_module._PARSER_MEMFD_SEALS
        with pytest.raises(OSError):
            os.pwrite(descriptor, b"z", 0)
    finally:
        os.close(descriptor)


def test_parser_memfd_is_independently_rehashed_after_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    payload = b"causal-parser-authority"
    source.write_bytes(payload)

    def forged_pread(descriptor: int, size: int, offset: int) -> bytes:
        del descriptor, offset
        return b"z" * size

    monkeypatch.setattr(admission_module.os, "pread", forged_pread)
    with pytest.raises(BenchmarkValidationError, match="revalidation drifted"):
        admission_module._sealed_parser_memfd(
            source,
            logical_path="jaxlib/source.bin",
            expected_bytes=len(payload),
            expected_sha256=sha256(payload).hexdigest(),
        )


def test_user_owned_python_runtime_is_rejected(tmp_path: Path) -> None:
    contract = _absolute_current_contract()
    original = Path(contract["implementation"]["validator_python"]["path"])
    runtime = tmp_path / "runtime"
    executable = runtime / "bin/python3.12"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(original.read_bytes())
    executable.chmod(0o755)
    contract["implementation"]["validator_python"] = {
        "path": str(executable),
        "sha256": _sha(executable),
    }
    contract["implementation"]["validator_python_runtime_root"] = str(runtime)
    with pytest.raises(BenchmarkValidationError, match="not root-owned read-only"):
        admission_module._verify_implementation(
            contract["implementation"], CURRENT_CONTRACT.parent
        )


def test_root_admission_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _absolute_current_contract()
    monkeypatch.setattr(admission_module.os, "geteuid", lambda: 0)
    with pytest.raises(BenchmarkValidationError, match="unprivileged user"):
        admission_module._verify_implementation(
            contract["implementation"], CURRENT_CONTRACT.parent
        )


def test_bound_nonroot_uid_normalizes_across_hosts() -> None:
    assert admission_module._normalize_bound_nonroot_uid(2001, 2001) == (
        admission_module._normalize_bound_nonroot_uid(12345, 12345)
    )
    with pytest.raises(BenchmarkValidationError, match="UID binding drifted"):
        admission_module._normalize_bound_nonroot_uid(2001, 12345)


def test_runtime_provisioner_strips_privilege_modes_and_xattrs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provisioner = runpy.run_path(
        str(REPO_ROOT / "scripts/greenfield/provision_gate_d_python_runtime.py")
    )
    hostile = tmp_path / "hostile"
    safe = tmp_path / "safe"
    hostile.mkdir()
    safe.mkdir()
    hostile_python = hostile / "python3.12"
    safe_python = safe / "python3.12"
    hostile_python.write_bytes(b"exact-runtime-bytes")
    safe_python.write_bytes(b"exact-runtime-bytes")
    hostile_python.chmod(0o6755)
    safe_python.chmod(0o755)
    os.setxattr(hostile_python, "user.gate_d_hostile", b"capability-like-payload")
    assert provisioner["_tree_sha256"](hostile) == provisioner["_tree_sha256"](
        safe
    )
    safe_python.chmod(0o700)
    assert provisioner["_tree_sha256"](hostile) != provisioner["_tree_sha256"](
        safe
    )
    safe_python.chmod(0o755)
    safe.chmod(0o700)
    assert provisioner["_tree_sha256"](hostile) != provisioner["_tree_sha256"](
        safe
    )
    monkeypatch.setattr(provisioner["os"], "chown", lambda *args, **kwargs: None)
    provisioner["_seal_ownership"](hostile)
    assert stat.S_IMODE(hostile_python.stat().st_mode) == 0o755
    assert os.listxattr(hostile_python) == []


def test_runtime_provisioner_rejects_permissive_staging_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner = runpy.run_path(
        str(REPO_ROOT / "scripts/greenfield/provision_gate_d_python_runtime.py")
    )
    monkeypatch.setattr(
        provisioner["os"],
        "fstat",
        lambda descriptor: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o777,
            st_uid=0,
            st_gid=0,
        ),
    )
    monkeypatch.setattr(
        provisioner["os"], "listxattr", lambda *args, **kwargs: []
    )
    with pytest.raises(SystemExit, match="exact root-owned 0755"):
        provisioner["_verify_directory_descriptor"](3, Path("/opt/glm-tpu"))


def test_runtime_provisioner_rejects_nonroot_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner = runpy.run_path(
        str(REPO_ROOT / "scripts/greenfield/provision_gate_d_python_runtime.py")
    )
    monkeypatch.setattr(
        provisioner["os"],
        "fstat",
        lambda descriptor: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o755,
            st_uid=0,
            st_gid=1000,
        ),
    )
    monkeypatch.setattr(
        provisioner["os"], "listxattr", lambda *args, **kwargs: []
    )
    with pytest.raises(SystemExit, match="exact root-owned 0755"):
        provisioner["_verify_directory_descriptor"](3, Path("/opt/glm-tpu"))


def test_runtime_provisioner_requires_exact_isolated_interpreter() -> None:
    provisioner = runpy.run_path(
        str(REPO_ROOT / "scripts/greenfield/provision_gate_d_python_runtime.py")
    )
    with pytest.raises(SystemExit, match="exact /usr/bin/python3 -I -S"):
        provisioner["_verify_root_interpreter"]()
    completed = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            (
                "import runpy; "
                f"m=runpy.run_path({str(REPO_ROOT / 'scripts/greenfield/provision_gate_d_python_runtime.py')!r}); "
                "m['_verify_root_interpreter']()"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_runtime_provisioner_publication_never_replaces_dangling_target(
    tmp_path: Path,
) -> None:
    provisioner = runpy.run_path(
        str(REPO_ROOT / "scripts/greenfield/provision_gate_d_python_runtime.py")
    )
    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    target.symlink_to("missing")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(tmp_path, flags)
    try:
        with pytest.raises(OSError) as error:
            provisioner["_rename_noreplace"](
                staging.name,
                target.name,
                directory_fd=descriptor,
            )
        assert error.value.errno == errno.EEXIST
    finally:
        os.close(descriptor)
    assert target.is_symlink()
    assert staging.is_dir()


def test_hostile_cwd_stdlib_shadows_cannot_enter_isolated_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _absolute_current_contract()
    candidate = contract["candidates"][0]
    locality = contract["locality_contract"]
    _, source = _source_authority(tmp_path, candidate, locality)
    _, plan = _plan_authority(tmp_path, "PP16_LP2")
    hostile = tmp_path / "hostile-cwd"
    hostile.mkdir()
    for name in ("json.py", "ctypes.py", "hashlib.py", "pathlib.py"):
        (hostile / name).write_text("raise RuntimeError('hostile cwd shadow loaded')\n")
    monkeypatch.chdir(hostile)
    real_run = admission_module.subprocess.run
    observed_isolation = False

    def assert_isolated(*args: Any, **kwargs: Any) -> Any:
        nonlocal observed_isolation
        command = args[0]
        assert command[1:4] == ["-I", "-S", "-c"]
        assert kwargs["cwd"] == "/"
        assert "PYTHONPATH" not in kwargs["env"]
        observed_isolation = True
        return real_run(*args, **kwargs)

    monkeypatch.setattr(admission_module.subprocess, "run", assert_isolated)
    _, stablehlo = _stablehlo_authority(
        tmp_path, candidate, locality, source, plan
    )
    assert observed_isolation is True
    assert stablehlo["authority_sha256"] != "0" * 64


def test_renamed_space_path_unsealed_native_mapping_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_library = Path("/usr/lib/x86_64-linux-gnu/libuuid.so.1.3.0")
    if not system_library.exists():
        pytest.skip("host libuuid fixture is unavailable")
    unsealed = tmp_path / "renamed dependency with spaces.so"
    unsealed.write_bytes(system_library.read_bytes())
    contract = _absolute_current_contract()
    original_validator = Path(
        contract["implementation"]["stablehlo_validator"]["path"]
    ).read_text()
    marker = "    return context, module\n"
    assert original_validator.count(marker) == 1
    injected = original_validator.replace(
        marker,
        f"    ctypes.CDLL({str(unsealed)!r})\n" + marker,
    )
    validator = tmp_path / "hostile-validator.py"
    validator.write_text(injected)
    contract["implementation"]["stablehlo_validator"] = {
        "path": str(validator),
        "sha256": _sha(validator),
    }
    monkeypatch.setattr(
        sys.modules[__name__], "_absolute_current_contract", lambda: contract
    )
    candidate = contract["candidates"][0]
    locality = contract["locality_contract"]
    _, source = _source_authority(tmp_path, candidate, locality)
    _, plan = _plan_authority(tmp_path, "PP16_LP2")
    _, stablehlo = _stablehlo_authority(
        tmp_path, candidate, locality, source, plan
    )
    assert stablehlo["authority_sha256"] == "0" * 64


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


def test_self_declared_plan_layout_is_rejected(tmp_path: Path) -> None:
    path, contract, _, _ = _prepared_contract(tmp_path)
    binding = contract["candidates"][0]["plan_authority"]
    plan_path = Path(binding["path"])
    plan = json.loads(plan_path.read_text())
    for watchpoint in plan["watchpoints"].values():
        watchpoint["layout"] = "forged.local"
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
    assert "layout is not exact PP16_LP2 authority" in candidate["plan_authority"][
        "refusal"
    ]


def test_self_declared_plan_owner_stage_is_rejected(tmp_path: Path) -> None:
    path, contract, _, _ = _prepared_contract(tmp_path)
    binding = contract["candidates"][0]["plan_authority"]
    plan_path = Path(binding["path"])
    plan = json.loads(plan_path.read_text())
    wrong_stage = plan["local_device_groups"][1]
    for watchpoint in plan["watchpoints"].values():
        watchpoint["owner_ids"] = wrong_stage
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
    assert "stage-zero owner group" in candidate["plan_authority"]["refusal"]


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


def test_capsule_rejects_replicated_device_owner_disagreement(tmp_path: Path) -> None:
    path, contract, capsule_path, _ = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    device_path = _binding_path(
        capsule_path.parent, capsule["producer_device_evidence"]["path"]
    )
    with np.load(device_path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["rms_input_fp32_owners"][1, 0, 2795] += np.float32(1.0)
    np.savez(device_path, **values)
    device_sha = _sha(device_path)
    capsule["producer_device_evidence"]["sha256"] = device_sha
    receipt_path = _binding_path(
        capsule_path.parent, capsule["producer_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["device_evidence"] = {
        "bytes": device_path.stat().st_size,
        "sha256": device_sha,
    }
    _write_canonical_json(receipt_path, receipt)
    success_path = _binding_path(
        capsule_path.parent, capsule["producer_success"]["path"]
    )
    success = json.loads(success_path.read_text())
    success["device_evidence_sha256"] = device_sha
    _write_canonical_json(success_path, success)
    _rebind_capsule_producer_chain(path, contract, capsule_path, capsule)
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert "replicated device owners disagree" in candidate["capsule"]["refusal"]


def test_capsule_cannot_reseal_wrong_event_against_outer_authority(
    tmp_path: Path,
) -> None:
    path, contract, capsule_path, artifact = _prepared_contract(tmp_path)
    capsule = json.loads(capsule_path.read_text())
    with np.load(artifact, allow_pickle=False) as archive:
        state = {name: archive[name].copy() for name in archive.files}
    state["event1_scores"] += np.float32(1.0)
    np.savez(artifact, **state)
    _refresh_watchpoint_array_hashes(capsule, state, "event1_scores")
    device_path = _binding_path(
        capsule_path.parent, capsule["producer_device_evidence"]["path"]
    )
    with np.load(device_path, allow_pickle=False) as archive:
        device = {name: archive[name].copy() for name in archive.files}
    device["selected_scores_owners"] += np.float32(1.0)
    np.savez(device_path, **device)
    device_sha = _sha(device_path)
    capsule["producer_device_evidence"]["sha256"] = device_sha
    receipt_path = _binding_path(
        capsule_path.parent, capsule["producer_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["device_evidence"] = {
        "bytes": device_path.stat().st_size,
        "sha256": device_sha,
    }
    _write_canonical_json(receipt_path, receipt)
    success_path = _binding_path(
        capsule_path.parent, capsule["producer_success"]["path"]
    )
    success = json.loads(success_path.read_text())
    success["device_evidence_sha256"] = device_sha
    _write_canonical_json(success_path, success)
    _rebind_capsule_producer_chain(
        path, contract, capsule_path, capsule, artifact=artifact
    )
    report = admit_gate_d_precompile_candidates(path, _sha(path))
    candidate = report["candidate_results"][0]
    assert "INVALID_CANDIDATE_COHERENT_CAPSULE" in candidate["reasons"]
    assert "does not match the pinned accepted authority" in candidate["capsule"][
        "refusal"
    ]


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
