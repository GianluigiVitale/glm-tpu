#!/usr/bin/env python3
"""Publish one bounded PP16 Gate-D numerical discriminator append-only."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import stat
import subprocess
import sys
import types
import zipfile
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
SOURCE_PATH = "scripts/greenfield/publish_gate_d_forced_round_pp16_numerical.py"
DRIVER_PATH = "scripts/greenfield/run_gate_d_forced_round_pp16_numerical.py"
PRIMITIVE_PATH = "scripts/greenfield/publish_gate_d_compensated_pp16_hlo.py"
PRIMITIVE_SHA256 = "75c296a2b46aef1b878a95ee7b1dfabdaf062687bfc496416ffca102f1180ee3"
HLO_SOURCE_CODE_HASH = "94518b7d4ce788157afa98b7bc1f8144613852d5"
TAG_PATTERN = re.compile(r"gate_d_forced_round_pp16_numerical_[0-9]{8}T[0-9]{15}Z")
REMOTE_ROOT = "results/greenfield/glm52/gate_d_forced_round_pp16_numerical/"
EXPECTED_STABLEHLO_SHA256 = (
    "45eae705b60783bf8b65a1d3d209c79e8d43685cec0aa912b3141caf36fb19e1"
)
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "ccd6ffb4909b1bc4dca5a36f106cde4a84304b230161afb484afb5667bb7206c"
)
EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256 = (
    "a0b87e2b43ba81bfe1549fbcc13434d2fbdca91b11c8b2f9a317ff9a9dbd9d45"
)
EXPECTED_FORCED_ROUND_SOURCE_SHA256 = (
    "518be87b650729d365dd09aba20b5f5a03d4bcddccc81978357cdbb02cbe02c6"
)
EXPECTED_POSITIONS_SHA256 = (
    "e55e66c6dcb35de94b9dce54d8fff602704cf26ae92d4afb333bd2501ab88ad7"
)
EXPECTED_SCORES_SHA256 = (
    "a61587a9d18bd169c1697ecb0060b39e8f1aad1caf532bca835b15a426c0b0e7"
)
EXPECTED_HOST_AUTHORITY_SHA256 = (
    "a7e5b393f7c61181f6b2aab9531d788fb27594d9cf980055c165b6f278ba4f99"
)
EXPECTED_QUERY_SHA256 = (
    "e460e727c123856b92fe45f63c2ca2f332a988965324cc824cda2c0fcd51338f"
)
EXPECTED_WK_SHA256 = "b2c67e0fdf4d7292494778233e9813d8256f2fb88b6f7376870379832fbab24b"
EXPECTED_CAPSULE_BINDING = {
    "capsule_sha256": "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4",
    "execution_authority_sha256": (
        "8b8c9cc79a18679628582a66c418a7f63e06553091928df58defbdde3971a660"
    ),
    "input_sha256": "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b",
    "state_sha256": "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236",
}
EXPECTED_CLAIM_SCOPE = (
    "One exact-input, one-invocation, two-chip PP16 stage-zero numerical "
    "discriminator. No decoder, token, performance or Gate-D closure claim."
)
EXPECTED_CAUSAL_WATCHPOINT_LABELS = {
    "layer1.cache_history:value",
    "layer1.current_key:value",
    "layer1.head_weights:owner0",
    "layer1.head_weights:owner1",
    "layer1.normalized:owner0",
    "layer1.normalized:owner1",
    "layer1.query:owner0",
    "layer1.query:owner1",
    "layer1.rms_input_fp32:value",
    "layer1.rms_operands_bf16:hidden_update",
    "layer1.rms_operands_bf16:residual",
}
EXPECTED_HLO_SOURCE_AUTHORITY = {
    "builder_path": "glm_tpu/greenfield/benchmarking/gate_d_forced_round_pp16_hlo.py",
    "builder_sha256": "4f5934dee8532294f67f77cca8b3e41f8ce00d29b23c3fc6996ab1cd886abefe",
    "candidate_id": "forced_normalized_bf16_boundary",
    "direct_runtime_dependency_sha256s": {
        "glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py": "0f1930c079bd7244452e84dca6d0bbf9da077ea133c685f0f908760379e5313d",
        "glm_tpu/greenfield/errors.py": "f7b8a079ba16f6114714124598d03ae871a2c21d365de4417b934f00cddaae64",
        "glm_tpu/greenfield/kernels/reference/attention.py": "0314931e50dedb9b40b2570c7716b38b32421a8c8749eb36bb5636717787c54a",
        "glm_tpu/greenfield/kernels/reference/dsa.py": "c4b451ab7ca2bf79b7cc7b148a7b1b146996051891c1c82f1952d8ab9ef53fac",
        "glm_tpu/greenfield/kernels/reference/qkv_a.py": "1e6d2cb6464d1733708c227283898f0687c965e1883cbea4c441ece2a40d3bb3",
        "glm_tpu/greenfield/kernels/reference/rmsnorm.py": "d6fca18425fb851c2ba3e1a12dacd45c8dadef43b33d6a1e6da72dbfb2eb6276",
        "glm_tpu/greenfield/kernels/stage_local.py": "0fa80ae74390a79bb63e6ac82357a08770b644bc07093139a825eb68b82ac490",
    },
    "source_classification": (
        "ISOLATED_PP16_SOURCE;FORCED_ROUND_PRIMARY_LINEAGE;"
        "NO_LOWER_COMPILE_EXECUTE_CALLS;PERSISTENCE_ONLY;GATE_D_OPEN"
    ),
}
EXPECTED_HLO_SOURCE_LOCATION_METADATA = {
    "derived_byte_count": 263876,
    "derived_sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
    "replacement_count": 3,
    "replacements": [
        {
            "new": (
                "/usr/local/libexec/glm-tpu/"
                "gate-d-forced-round-pp16-numerical-v2/"
                "run_gate_d_forced_round_pp16_numerical.py"
            ),
            "old": (
                "/usr/local/libexec/glm-tpu/gate-d-forced-round-pp16-hlo-v2/"
                "acquire_gate_d_forced_round_pp16_hlo.py"
            ),
            "occurrence_count": 1,
            "surface": "FileNames",
        },
        {
            "new": "line=1273 end_line=1273",
            "old": "line=1568 end_line=1568",
            "occurrence_count": 1,
            "surface": "FileLocations module call",
        },
        {
            "new": "line=1098 end_line=1098",
            "old": "line=1425 end_line=1425",
            "occurrence_count": 1,
            "surface": "FileLocations lower call",
        },
    ],
    "source_byte_count": 263868,
    "source_sha256": EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256,
}
EXPECTED_COMPILER_ENVIRONMENT = {
    "HOME": "/home/gianl",
    "JAX_ENABLE_COMPILATION_CACHE": "0",
    "JAX_PLATFORMS": "tpu",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_VISIBLE_DEVICES": "0,1,2,3",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
}
EXPECTED_OUTPUT_SPEC = {
    "contract_valid_owners": ([2], "uint8", "|u1"),
    "current_key_owners": ([2, 1, 128], "float32", "<f4"),
    "head_weights_owners": ([2, 1, 32], "float32", "<f4"),
    "index_cache_owners": ([2, 16, 256, 128], "bfloat16", "<u2"),
    "normalized_hidden_owners": ([2, 1, 6144], "bfloat16", "<u2"),
    "query_owners": ([2, 1, 32, 128], "float32", "<f4"),
    "selected_positions_owners": ([2, 1, 2048], "int32", "<i4"),
    "selected_scores_owners": ([2, 1, 2048], "float32", "<f4"),
    "valid_counts_owners": ([2, 1], "int32", "<i4"),
}
WRAPPER_WRITE_MEMBERS = {
    "census_failure_exit.txt",
    "census_post.txt",
    "census_pre.txt",
    "mirror.sha256",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.log",
    "sync.txt",
}
SUCCESS_PAYLOAD = {
    "census_post.txt",
    "census_pre.txt",
    "dependencies.json",
    "evidence.json",
    "hlo/forced_round_pp16_stage0.optimized_hlo.txt",
    "hlo/forced_round_pp16_stage0.stablehlo.mlir",
    "hlo/acquired_preimage.optimized_hlo.txt",
    "hlo/acquired_preimage.stablehlo.mlir",
    "mirror.sha256",
    "orchestrator.sealed.log",
    "outputs.npz",
    "publisher_runtime.json",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.json",
    "runner.log",
    "summary.json",
    "sync.txt",
}


def _observed_names(
    bucket: Any,
    prefix: str,
    *,
    versions: bool = False,
    soft_deleted: bool = False,
) -> set[str]:
    if versions and soft_deleted:
        raise RuntimeError("incompatible Gate-D numerical history listing scopes")
    blobs = bucket.list_blobs(
        prefix=prefix,
        versions=versions or None,
        soft_deleted=soft_deleted or None,
    )
    names = set()
    for blob in blobs:
        if not blob.name.startswith(prefix):
            raise RuntimeError(
                "Gate-D numerical history listing escaped its exact prefix"
            )
        names.add(blob.name.removeprefix(prefix))
    return names


def _require_never_used_prefix(bucket: Any, prefix: str) -> None:
    observations = {
        "live": _observed_names(bucket, prefix),
        "all_versions": _observed_names(bucket, prefix, versions=True),
        "soft_deleted": _observed_names(bucket, prefix, soft_deleted=True),
    }
    occupied = {scope: names for scope, names in observations.items() if names}
    if occupied:
        raise RuntimeError(
            "Gate-D numerical remote prefix has prior "
            "live/versioned/soft-deleted history: "
            + ",".join(sorted(occupied))
        )


def _validate_remote_vacancy_evidence(
    raw: bytes, summary: bytes, remote: str
) -> None:
    no_objects = "ERROR: (gcloud.storage.ls) One or more URLs matched no objects."
    expected_raw = (
        "scope=live flags=none returncode=1\n"
        f"{no_objects}\n"
        "scope=all_versions flags=--all-versions returncode=1\n"
        f"{no_objects}\n"
        "scope=soft_deleted flags=--soft-deleted,--exhaustive returncode=1\n"
        f"{no_objects}\n"
    ).encode("ascii")
    expected_summary = (
        f"VACANT live {remote}\n"
        f"VACANT all_versions {remote}\n"
        f"VACANT soft_deleted {remote}\n"
    ).encode("ascii")
    if raw != expected_raw or summary != expected_summary:
        raise RuntimeError(
            "Gate-D numerical canonical remote vacancy evidence drifted"
        )


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _snapshot_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe publisher source identity: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"publisher source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(WORKTREE), *arguments],
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_SSH_COMMAND": "/bin/false",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )


def _load_primitives(code_pin: str) -> types.ModuleType:
    path = WORKTREE / PRIMITIVE_PATH
    raw = _snapshot_regular(path)
    if (
        raw != _git_bytes("show", f"{code_pin}:{PRIMITIVE_PATH}")
        or raw != _git_bytes("show", f"{HLO_SOURCE_CODE_HASH}:{PRIMITIVE_PATH}")
        or sha256(raw).hexdigest() != PRIMITIVE_SHA256
    ):
        raise RuntimeError("audited publication primitive bytes drifted")
    module = types.ModuleType("_gate_d_numerical_publication_primitives")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    module.REPO = WORKTREE
    module.RUN_ROOT = RUN_ROOT
    module.TAG_PATTERN = TAG_PATTERN
    module.REMOTE_ROOT = REMOTE_ROOT
    module.COMPILER_DRIVER_PATH = WORKTREE / DRIVER_PATH
    module._git_bytes = _git_bytes
    return module


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    raw = _snapshot_regular(Path(__file__))
    head = _git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    resolved = _git_bytes("rev-parse", f"{code_pin}^{{commit}}").decode("ascii").strip()
    if (
        head != code_pin
        or resolved != code_pin
        or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("Gate-D numerical publisher is not the committed blob")


def _publication_runtime(primitives: Any) -> dict[str, Any]:
    runtime = primitives.validate_publication_runtime()
    runtime["artifact_kind"] = "gate_d_forced_round_pp16_numerical_publisher_runtime"
    runtime["publication_primitive_path"] = PRIMITIVE_PATH
    runtime["publication_primitive_sha256"] = PRIMITIVE_SHA256
    return runtime


def _validate_dependencies(
    primitives: Any,
    runner: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    dependencies_raw: bytes,
    code_pin: str,
) -> None:
    identity = runner.get("compiler_dependency_manifest")
    accelerator = dependencies.get("accelerator_device_nodes_observed_mapped")
    native = dependencies.get("native_mappings")
    python = dependencies.get("python_modules")
    dependency_source = dependencies.get("sealed_project_source")
    runner_source = runner.get("sealed_project_source")
    if (
        set(dependencies)
        != {
            "accelerator_device_observation_scope",
            "accelerator_device_nodes_observed_mapped",
            "artifact_kind",
            "code_hash",
            "dependency_sites",
            "environment",
            "native_mappings",
            "python_modules",
            "python_runtime",
            "sealed_project_source",
        }
        or dependencies.get("artifact_kind")
        != "gate_d_forced_round_pp16_numerical_dependencies"
        or dependencies.get("code_hash") != code_pin
        or dependencies.get("environment") != EXPECTED_COMPILER_ENVIRONMENT
        or dependencies.get("python_runtime")
        != {
            "python_executable": str(primitives.PYTHON),
            "python_runtime_root": str(primitives.PYTHON_RUNTIME_ROOT),
            "python_runtime_tree_sha256": primitives.PYTHON_RUNTIME_TREE_SHA256,
            "python_sha256": primitives.PYTHON_SHA256,
        }
        or dependencies.get("dependency_sites")
        != {
            "jax": {
                "manifest_sha256": primitives.JAX_SITE_MANIFEST_SHA256,
                "root": str(primitives.JAX_SITE_ROOT),
                "tree_sha256": primitives.JAX_SITE_TREE_SHA256,
            },
            "libtpu": {
                "manifest_sha256": primitives.LIBTPU_SITE_MANIFEST_SHA256,
                "root": str(primitives.LIBTPU_SITE_ROOT),
                "tree_sha256": primitives.LIBTPU_SITE_TREE_SHA256,
            },
        }
        or dependencies.get("accelerator_device_observation_scope")
        != primitives._ACCELERATOR_DEVICE_OBSERVATION_SCOPE
        or not isinstance(identity, Mapping)
        or set(identity)
        != {
            "accelerator_device_node_count",
            "accelerator_device_nodes_sha256",
            "byte_count",
            "filename",
            "native_mapping_count",
            "python_module_count",
            "sha256",
        }
        or identity.get("filename") != "dependencies.json"
        or identity.get("byte_count") != len(dependencies_raw)
        or identity.get("sha256") != sha256(dependencies_raw).hexdigest()
        or not isinstance(accelerator, list)
        or not isinstance(native, list)
        or not isinstance(python, list)
        or identity.get("accelerator_device_node_count") != len(accelerator)
        or identity.get("accelerator_device_nodes_sha256")
        != primitives._accelerator_device_nodes_sha256(accelerator)
        or identity.get("native_mapping_count") != len(native)
        or identity.get("python_module_count") != len(python)
        or not isinstance(dependency_source, Mapping)
        or set(dependency_source)
        != {"archive_sha256", "file_manifest_count", "file_manifest_sha256"}
        or not isinstance(runner_source, Mapping)
        or set(runner_source)
        != {
            "archive_sha256",
            "file_manifest_count",
            "file_manifest_sha256",
            "loaded_module_count",
        }
        or dependency_source
        != {
            key: runner_source.get(key)
            for key in ("archive_sha256", "file_manifest_count", "file_manifest_sha256")
        }
    ):
        raise RuntimeError("Gate-D numerical dependency manifest drifted")
    primitives._validate_accelerator_device_mapping_records(
        accelerator, verify_live=True
    )
    primitives._validate_dependency_records(native, "native_mappings", verify_live=True)
    primitives._validate_dependency_records(python, "python_modules", verify_live=True)


def _validate_output_arrays(runner: Mapping[str, Any]) -> None:
    arrays = runner.get("output_arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(EXPECTED_OUTPUT_SPEC):
        raise RuntimeError("Gate-D numerical output array catalogue drifted")
    for name, (shape, semantic_dtype, storage_dtype) in EXPECTED_OUTPUT_SPEC.items():
        record = arrays[name]
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {"array_sha256", "semantic_dtype", "shape", "storage_dtype"}
            or record.get("shape") != shape
            or record.get("semantic_dtype") != semantic_dtype
            or record.get("storage_dtype") != storage_dtype
            or type(record.get("array_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", record["array_sha256"]) is None
        ):
            raise RuntimeError(f"Gate-D numerical output array drifted: {name}")


def _validate_runner(runner: Mapping[str, Any], code_pin: str) -> str:
    status = runner.get("status")
    accepted = status == "NUMERICAL_ACCEPTED"
    expected_claim = (
        "BOUNDED_TPU_NUMERICAL_ACCEPTED;FULL_DECODER_UNPROVEN;GATE_D_OPEN;"
        "NO_PERFORMANCE_CLAIM"
        if accepted
        else "BOUNDED_TPU_NUMERICAL_REJECTED;FORCED_ROUND_MECHANISM_REJECTED;"
        "GATE_D_OPEN;NO_PERFORMANCE_CLAIM"
    )
    numerical = runner.get("numerical")
    observed = (
        numerical.get("observed_event1") if isinstance(numerical, Mapping) else None
    )
    owners = (
        numerical.get("owner_agreement") if isinstance(numerical, Mapping) else None
    )
    host = runner.get("host_materialization")
    watchpoints = (
        numerical.get("candidate_watchpoint_matches")
        if isinstance(numerical, Mapping)
        else None
    )
    expected_host = {
        "authority_sha256": EXPECTED_HOST_AUTHORITY_SHA256,
        "backend": "cpu",
        "capsule_input_sha256": (
            "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
        ),
        "query_weight": {
            "array_sha256": EXPECTED_QUERY_SHA256,
            "semantic_dtype": "float32",
            "shape": [2, 2048, 2048],
        },
        "wk_weight": {
            "array_sha256": EXPECTED_WK_SHA256,
            "semantic_dtype": "float32",
            "shape": [2, 128, 6144],
        },
    }
    if (
        set(runner)
        != {
            "artifact_kind",
            "capsule",
            "claim_scope",
            "classification",
            "code_hash",
            "compiler_dependency_manifest",
            "compiled_executable_invocation_count",
            "gate_d_closed",
            "hlo",
            "hlo_source_authority",
            "hlo_source_location_metadata",
            "host_materialization",
            "host_transfer_count",
            "memory_after_compile",
            "memory_after_execute",
            "memory_before_compile",
            "numerical",
            "numerical_claim_scope",
            "output_arrays",
            "output_artifact",
            "performance_claim",
            "persistent_compilation_cache_enabled",
            "physical_group",
            "policy",
            "runtime",
            "schema_version",
            "sealed_project_source",
            "status",
            "tpu_execution_elapsed_ns_diagnostic_only",
            "tpu_numerical_execution_performed",
        }
        or status not in {"NUMERICAL_ACCEPTED", "NUMERICAL_REJECTED"}
        or runner.get("artifact_kind") != "gate_d_forced_round_pp16_numerical_replay"
        or runner.get("capsule") != EXPECTED_CAPSULE_BINDING
        or runner.get("claim_scope") != EXPECTED_CLAIM_SCOPE
        or runner.get("classification") != expected_claim
        or runner.get("code_hash") != code_pin
        or type(runner.get("schema_version")) is not int
        or runner.get("schema_version") != 1
        or type(runner.get("compiled_executable_invocation_count")) is not int
        or runner.get("compiled_executable_invocation_count") != 1
        or type(runner.get("host_transfer_count")) is not int
        or runner.get("host_transfer_count") != 1
        or runner.get("persistent_compilation_cache_enabled") is not False
        or runner.get("performance_claim") is not False
        or runner.get("gate_d_closed") is not False
        or runner.get("tpu_numerical_execution_performed") is not True
        or type(runner.get("tpu_execution_elapsed_ns_diagnostic_only")) is not int
        or runner.get("tpu_execution_elapsed_ns_diagnostic_only") <= 0
        or runner.get("numerical_claim_scope") != "bounded_event1_only"
        or runner.get("hlo")
        != {
            "optimized_hlo_sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
            "stablehlo_sha256": EXPECTED_STABLEHLO_SHA256,
        }
        or runner.get("hlo_source_location_metadata")
        != EXPECTED_HLO_SOURCE_LOCATION_METADATA
        or runner.get("hlo_source_authority") != EXPECTED_HLO_SOURCE_AUTHORITY
        or runner.get("physical_group")
        != {
            "coordinates": [[0, 0, 0], [1, 0, 0]],
            "device_ids": [0, 1],
            "process_index": 0,
            "stage_id": 0,
        }
        or not isinstance(host, Mapping)
        or set(host) != set(expected_host) | {"observed_derived_weights"}
        or {key: host.get(key) for key in expected_host} != expected_host
        or host.get("observed_derived_weights")
        != {
            "query_weight": expected_host["query_weight"],
            "wk_weight": expected_host["wk_weight"],
        }
        or not isinstance(numerical, Mapping)
        or numerical.get("accepted_tpu_event_match") is not accepted
        or not isinstance(observed, Mapping)
        or type(observed.get("valid_count")) is not int
        or not isinstance(owners, Mapping)
        or set(owners)
        != {
            "current_key_owners",
            "head_weights_owners",
            "normalized_hidden_owners",
            "query_owners",
            "selected_positions_owners",
            "selected_scores_owners",
            "valid_counts_owners",
        }
        or not all(type(value) is bool for value in owners.values())
        or type(numerical.get("tie_order_exact")) is not bool
        or not isinstance(numerical.get("contract_valid"), list)
        or len(numerical["contract_valid"]) != 2
        or not all(
            type(value) is int and value in {0, 1}
            for value in numerical["contract_valid"]
        )
        or not isinstance(watchpoints, Mapping)
        or len(watchpoints) != 14
        or not all(
            type(name) is str and type(value) is bool
            for name, value in watchpoints.items()
        )
        or type(numerical.get("causal_candidate_watchpoints_exact")) is not bool
        or runner.get("policy")
        != {
            "positions_sha256": EXPECTED_POSITIONS_SHA256,
            "scores_sha256": EXPECTED_SCORES_SHA256,
            "valid_count": 2048,
        }
        or runner.get("runtime")
        != {
            "jax": "0.10.1",
            "jaxlib": "0.10.1",
            "libtpu": "0.0.41",
            "ml_dtypes": "0.5.4",
            "numpy": "2.3.5",
        }
    ):
        raise RuntimeError("Gate-D numerical runner claim boundary drifted")
    if accepted and (
        observed
        != {
            "positions_sha256": EXPECTED_POSITIONS_SHA256,
            "scores_sha256": EXPECTED_SCORES_SHA256,
            "valid_count": 2048,
        }
        or not all(value is True for value in owners.values())
        or numerical.get("causal_candidate_watchpoints_exact") is not True
        or not all(
            watchpoints.get(name) is True
            for name in EXPECTED_CAUSAL_WATCHPOINT_LABELS
        )
        or numerical.get("tie_order_exact") is not True
        or numerical.get("contract_valid") != [1, 1]
    ):
        raise RuntimeError("Gate-D accepted numerical boundary drifted")
    _validate_output_arrays(runner)
    return status


def _npy_payload(raw: bytes, expected_shape: list[int], expected_dtype: str) -> bytes:
    if not raw.startswith(b"\x93NUMPY") or len(raw) < 10:
        raise RuntimeError("Gate-D numerical output NPY magic drifted")
    major, minor = raw[6], raw[7]
    if (major, minor) == (1, 0):
        header_size_width = 2
    elif (major, minor) in {(2, 0), (3, 0)}:
        header_size_width = 4
    else:
        raise RuntimeError("Gate-D numerical output NPY version drifted")
    prefix = 8 + header_size_width
    if len(raw) < prefix:
        raise RuntimeError("Gate-D numerical output NPY header is truncated")
    header_size = int.from_bytes(raw[8:prefix], "little")
    payload_offset = prefix + header_size
    if header_size <= 0 or payload_offset > len(raw):
        raise RuntimeError("Gate-D numerical output NPY header length drifted")
    try:
        header = ast.literal_eval(raw[prefix:payload_offset].decode("ascii"))
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("Gate-D numerical output NPY header drifted") from error
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header.get("descr") != expected_dtype
        or header.get("fortran_order") is not False
        or not isinstance(header.get("shape"), tuple)
        or list(header["shape"]) != expected_shape
        or not all(type(item) is int and item >= 0 for item in header["shape"])
    ):
        raise RuntimeError("Gate-D numerical output NPY schema drifted")
    item_sizes = {"|u1": 1, "<f4": 4, "<i4": 4, "<u2": 2}
    if expected_dtype not in item_sizes:
        raise RuntimeError("Gate-D numerical output storage dtype is unsupported")
    expected_bytes = item_sizes[expected_dtype]
    for dimension in expected_shape:
        expected_bytes *= dimension
    payload = raw[payload_offset:]
    if len(payload) != expected_bytes:
        raise RuntimeError("Gate-D numerical output NPY payload length drifted")
    return payload


def _validate_output_archive(raw: bytes, records: Mapping[str, Any]) -> None:
    expected = {f"{name}.npy" for name in EXPECTED_OUTPUT_SPEC}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        infos = archive.infolist()
        if (
            len(names) != len(set(names))
            or set(names) != expected
            or sum(info.file_size for info in infos) > 16 * 1024 * 1024
            or any(info.is_dir() or info.flag_bits & 0x1 for info in infos)
        ):
            raise RuntimeError("Gate-D numerical output archive catalogue drifted")
        for name, (
            shape,
            _semantic_dtype,
            storage_dtype,
        ) in EXPECTED_OUTPUT_SPEC.items():
            member = archive.read(f"{name}.npy")
            payload = _npy_payload(member, shape, storage_dtype)
            if sha256(payload).hexdigest() != records[name]["array_sha256"]:
                raise RuntimeError(
                    f"Gate-D numerical output array bytes drifted: {name}"
                )


def _prepare_success(
    primitives: Any,
    run_fd: int,
    *,
    code_pin: str,
    run_tag: str,
    remote: str,
    elapsed: int,
) -> tuple[dict[str, bytes], str]:
    runner_raw = primitives.snapshot_member(run_fd, "runner.json")
    dependencies_raw = primitives.snapshot_member(run_fd, "dependencies.json")
    runtime_raw = primitives.snapshot_member(
        run_fd, "publisher_runtime.json", limit=1 << 20
    )
    outputs_raw = primitives.snapshot_member(run_fd, "outputs.npz", limit=64 << 20)
    runner = json.loads(runner_raw)
    dependencies = json.loads(dependencies_raw)
    if not isinstance(runner, Mapping) or not isinstance(dependencies, Mapping):
        raise TypeError("Gate-D numerical core artifacts must be JSON objects")
    status = _validate_runner(runner, code_pin)
    _validate_dependencies(primitives, runner, dependencies, dependencies_raw, code_pin)
    output_identity = runner.get("output_artifact")
    if output_identity != {
        "byte_count": len(outputs_raw),
        "filename": "outputs.npz",
        "sha256": sha256(outputs_raw).hexdigest(),
    }:
        raise RuntimeError("Gate-D numerical output artifact identity drifted")
    _validate_output_archive(outputs_raw, runner["output_arrays"])
    acquired_hlo_raw = primitives.snapshot_member(
        run_fd, "hlo/acquired_preimage.optimized_hlo.txt", limit=256 << 20
    )
    if (
        len(acquired_hlo_raw) != 263868
        or sha256(acquired_hlo_raw).hexdigest()
        != EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256
    ):
        raise RuntimeError("Gate-D accepted optimized-HLO preimage bytes drifted")
    acquired_stablehlo_raw = primitives.snapshot_member(
        run_fd, "hlo/acquired_preimage.stablehlo.mlir", limit=256 << 20
    )
    if (
        len(acquired_stablehlo_raw) != 76529
        or sha256(acquired_stablehlo_raw).hexdigest() != EXPECTED_STABLEHLO_SHA256
    ):
        raise RuntimeError("Gate-D accepted StableHLO preimage bytes drifted")
    hlo_payload: dict[str, bytes] = {}
    for kind, relative in (
        ("optimized_hlo_sha256", "hlo/forced_round_pp16_stage0.optimized_hlo.txt"),
        ("stablehlo_sha256", "hlo/forced_round_pp16_stage0.stablehlo.mlir"),
    ):
        raw = primitives.snapshot_member(run_fd, relative, limit=256 << 20)
        if sha256(raw).hexdigest() != runner["hlo"][kind]:
            raise RuntimeError(f"Gate-D numerical {kind} bytes drifted")
        hlo_payload[relative] = raw
    census_pre = primitives.snapshot_member(run_fd, "census_pre.txt")
    census_post = primitives.snapshot_member(run_fd, "census_post.txt")
    primitives._require_census(census_pre, "CENSUS_OK")
    primitives._require_census(census_post, "CENSUS_OK")
    orchestrator = primitives.snapshot_member(
        run_fd, "orchestrator.log", limit=16 << 20
    )
    primitives.write_member_exclusive(run_fd, "orchestrator.sealed.log", orchestrator)
    summary_raw = _canonical(
        {
            "artifact_kind": "gate_d_forced_round_pp16_numerical_summary",
            "classification": runner["classification"],
            "code_hash": code_pin,
            "elapsed_seconds_diagnostic_only": elapsed,
            "gate_d_closed": False,
            "acquired_optimized_hlo_sha256": (
                EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256
            ),
            "derived_optimized_hlo_sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
            "forced_round_source_sha256": EXPECTED_FORCED_ROUND_SOURCE_SHA256,
            "performance_claim": False,
            "remote_prefix": remote,
            "run_tag": run_tag,
            "status": status,
            "tpu_numerical_execution_performed": True,
        }
    )
    primitives.write_member_exclusive(run_fd, "summary.json", summary_raw)
    payload = {
        "census_post.txt": census_post,
        "census_pre.txt": census_pre,
        "dependencies.json": dependencies_raw,
        "hlo/acquired_preimage.optimized_hlo.txt": acquired_hlo_raw,
        "hlo/acquired_preimage.stablehlo.mlir": acquired_stablehlo_raw,
        **hlo_payload,
        "orchestrator.sealed.log": orchestrator,
        "outputs.npz": outputs_raw,
        "publisher_runtime.json": runtime_raw,
        "runner.json": runner_raw,
        "summary.json": summary_raw,
    }
    for relative in (
        "mirror.sha256",
        "remote_vacancy.raw.txt",
        "remote_vacancy.txt",
        "runner.log",
        "sync.txt",
    ):
        payload[relative] = primitives.snapshot_member(run_fd, relative, limit=32 << 20)
    evidence_raw = _canonical(
        {
            "artifact_kind": "gate_d_forced_round_pp16_numerical_local_evidence",
            "code_hash": code_pin,
            "files": [
                primitives._record(name, payload[name]) for name in sorted(payload)
            ],
            "gate_d_closed": False,
            "acquired_optimized_hlo_sha256": (
                EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256
            ),
            "derived_optimized_hlo_sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
            "forced_round_source_sha256": EXPECTED_FORCED_ROUND_SOURCE_SHA256,
            "performance_claim": False,
            "run_tag": run_tag,
            "status": status,
        }
    )
    primitives.write_member_exclusive(run_fd, "evidence.json", evidence_raw)
    payload["evidence.json"] = evidence_raw
    if set(payload) != SUCCESS_PAYLOAD:
        raise RuntimeError("Gate-D numerical success payload inventory drifted")
    if primitives.local_members(run_fd) != SUCCESS_PAYLOAD | {"orchestrator.log"}:
        raise RuntimeError("Gate-D numerical local preterminal inventory drifted")
    return payload, status


def publish_success(
    primitives: Any,
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    elapsed: int,
    publication_runtime_raw: bytes,
    run_dir_fd: int | None,
) -> None:
    run_tag = primitives.validate_run_dir(run_dir)
    run_fd = primitives._run_fd(run_dir, run_dir_fd)
    try:
        primitives.require_preterminal(run_fd)
        primitives.require_publication_runtime(run_fd, publication_runtime_raw)
        payload, status = _prepare_success(
            primitives,
            run_fd,
            code_pin=code_pin,
            run_tag=run_tag,
            remote=remote,
            elapsed=elapsed,
        )
        _validate_remote_vacancy_evidence(
            payload["remote_vacancy.raw.txt"],
            payload["remote_vacancy.txt"],
            remote,
        )
        bucket, prefix = primitives._bucket_and_prefix(remote, None)
        _require_never_used_prefix(bucket, prefix)
        records = []
        for relative in sorted(payload):
            record = primitives._upload_bound(
                bucket, prefix + relative, payload[relative]
            )
            record["path"] = relative
            records.append(record)
        ledger_raw = _canonical(
            {
                "artifact_kind": "gate_d_forced_round_pp16_numerical_remote_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        primitives.write_member_exclusive(run_fd, "remote_objects.json", ledger_raw)
        ledger = primitives._upload_bound(
            bucket, prefix + "remote_objects.json", ledger_raw
        )
        ledger["path"] = "remote_objects.json"
        for record in records:
            primitives._replay_bound(bucket, prefix + record["path"], record)
        primitives._replay_bound(bucket, prefix + "remote_objects.json", ledger)
        expected_remote = set(payload) | {"remote_objects.json"}
        if _observed_names(bucket, prefix) != expected_remote:
            raise RuntimeError("Gate-D numerical preterminal remote inventory drifted")
        terminal_raw = _canonical(
            {
                "artifact_kind": "gate_d_forced_round_pp16_numerical_terminal",
                "evidence_sha256": sha256(payload["evidence.json"]).hexdigest(),
                "gate_d_closed": False,
                "acquired_optimized_hlo_sha256": (
                    EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256
                ),
                "derived_optimized_hlo_sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
                "forced_round_source_sha256": EXPECTED_FORCED_ROUND_SOURCE_SHA256,
                "performance_claim": False,
                "remote_ledger": ledger,
                "run_tag": run_tag,
                "status": status,
                "summary_sha256": sha256(payload["summary.json"]).hexdigest(),
                "tpu_numerical_execution_performed": True,
            }
        )
        primitives.write_member_exclusive(run_fd, status, terminal_raw)
        expected_terminal_local = SUCCESS_PAYLOAD | {
            "orchestrator.log",
            "remote_objects.json",
            status,
        }
        if primitives.local_members(run_fd) != expected_terminal_local:
            raise RuntimeError("Gate-D numerical local terminal inventory drifted")
        terminal = primitives._upload_bound(bucket, prefix + status, terminal_raw)
        terminal["path"] = status
        primitives._replay_bound(bucket, prefix + status, terminal)
        if primitives._observed_names(bucket, prefix) != expected_remote | {status}:
            raise RuntimeError("Gate-D numerical terminal remote inventory drifted")
        primitives.write_member_exclusive(
            run_fd,
            "terminal_upload_receipt.json",
            _canonical(
                {
                    "artifact_kind": (
                        "gate_d_forced_round_pp16_numerical_terminal_receipt"
                    ),
                    "remote": remote + "/" + status,
                    "terminal": terminal,
                }
            ),
        )
        if primitives.local_members(run_fd) != expected_terminal_local | {
            "terminal_upload_receipt.json"
        }:
            raise RuntimeError("Gate-D numerical terminal receipt inventory drifted")
    finally:
        os.close(run_fd)


def publish_diagnostic(
    primitives: Any,
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    status: int,
    publication_runtime_raw: bytes,
    run_dir_fd: int | None,
) -> None:
    run_tag = primitives.validate_run_dir(run_dir)
    run_fd = primitives._run_fd(run_dir, run_dir_fd)
    try:
        primitives.require_preterminal(run_fd)
        primitives.require_publication_runtime(run_fd, publication_runtime_raw)
        failure_raw = _canonical(
            {
                "artifact_kind": "gate_d_forced_round_pp16_numerical_failure",
                "code_hash": code_pin,
                "exit_status": status,
                "gate_d_closed": False,
                "performance_claim": False,
                "run_tag": run_tag,
                "terminal_payload_present": False,
            }
        )
        primitives.write_member_exclusive(run_fd, "failure_status.json", failure_raw)
        orchestrator = primitives.snapshot_member(
            run_fd, "orchestrator.log", limit=16 << 20
        )
        primitives.write_member_exclusive(
            run_fd, "orchestrator.failure.log", orchestrator
        )
        excluded = {
            "orchestrator.log",
            "NUMERICAL_ACCEPTED",
            "NUMERICAL_REJECTED",
            "diagnostic_objects.json",
        }
        members = [
            name
            for name in sorted(primitives.local_members(run_fd))
            if name not in excluded
        ]
        if not members or len(members) > 40:
            raise RuntimeError("Gate-D numerical diagnostic inventory is unsafe")
        payload = {
            name: primitives.snapshot_member(run_fd, name, limit=256 << 20)
            for name in members
        }
        _validate_remote_vacancy_evidence(
            payload["remote_vacancy.raw.txt"],
            payload["remote_vacancy.txt"],
            remote,
        )
        bucket, prefix = primitives._bucket_and_prefix(remote, None)
        diagnostic_prefix = prefix + "diagnostic/"
        _require_never_used_prefix(bucket, prefix)
        records = []
        for relative in sorted(payload):
            record = primitives._upload_bound(
                bucket, diagnostic_prefix + relative, payload[relative]
            )
            record["path"] = relative
            records.append(record)
        for record in records:
            primitives._replay_bound(bucket, diagnostic_prefix + record["path"], record)
        if _observed_names(bucket, diagnostic_prefix) != set(payload):
            raise RuntimeError("Gate-D numerical diagnostic remote inventory drifted")
        ledger_raw = _canonical(
            {
                "artifact_kind": (
                    "gate_d_forced_round_pp16_numerical_diagnostic_ledger"
                ),
                "objects": records,
                "run_tag": run_tag,
            }
        )
        primitives.write_member_exclusive(run_fd, "diagnostic_objects.json", ledger_raw)
        ledger = primitives._upload_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_raw
        )
        ledger["path"] = "diagnostic_objects.json"
        primitives._replay_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger
        )
        if _observed_names(bucket, diagnostic_prefix) != set(payload) | {
            "diagnostic_objects.json"
        }:
            raise RuntimeError("Gate-D numerical diagnostic terminal inventory drifted")
        primitives.write_member_exclusive(
            run_fd,
            "diagnostic_upload_receipt.json",
            _canonical(
                {
                    "artifact_kind": (
                        "gate_d_forced_round_pp16_numerical_diagnostic_receipt"
                    ),
                    "remote": remote + "/diagnostic/diagnostic_objects.json",
                    "terminal": ledger,
                }
            ),
        )
    finally:
        os.close(run_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--run-dir-fd", type=int)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--run-dir", type=Path, required=True)
    success = subparsers.add_parser("success")
    success.add_argument("--run-dir", type=Path, required=True)
    success.add_argument("--remote-prefix", required=True)
    success.add_argument("--elapsed", type=int, required=True)
    diagnostic = subparsers.add_parser("diagnostic")
    diagnostic.add_argument("--run-dir", type=Path, required=True)
    diagnostic.add_argument("--remote-prefix", required=True)
    diagnostic.add_argument("--status", type=int, required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--run-dir", type=Path, required=True)
    write.add_argument("--member", choices=sorted(WRAPPER_WRITE_MEMBERS), required=True)
    append = subparsers.add_parser("append-log")
    append.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    arguments = parse_args()
    primitives = _load_primitives(arguments.expected_code_hash)
    primitives.validate_environment()
    _verify_running_source(
        arguments.expected_code_hash, arguments.expected_source_sha256
    )
    publication_runtime_raw = _canonical(_publication_runtime(primitives))
    if arguments.mode == "init":
        print(
            "RUN_IDENTITY "
            + primitives.initialize_run_dir(
                arguments.run_dir, publication_runtime_raw=publication_runtime_raw
            )
        )
    elif arguments.mode == "write":
        run_fd = primitives._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            primitives.require_publication_runtime(run_fd, publication_runtime_raw)
            primitives.write_preterminal_member(
                run_fd, arguments.member, sys.stdin.buffer
            )
        finally:
            os.close(run_fd)
    elif arguments.mode == "append-log":
        run_fd = primitives._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            primitives.require_publication_runtime(run_fd, publication_runtime_raw)
            primitives.append_preterminal_log(run_fd, sys.stdin.buffer.read())
        finally:
            os.close(run_fd)
    elif arguments.mode == "success":
        publish_success(
            primitives,
            arguments.run_dir,
            arguments.remote_prefix,
            code_pin=arguments.expected_code_hash,
            elapsed=arguments.elapsed,
            publication_runtime_raw=publication_runtime_raw,
            run_dir_fd=arguments.run_dir_fd,
        )
    else:
        publish_diagnostic(
            primitives,
            arguments.run_dir,
            arguments.remote_prefix,
            code_pin=arguments.expected_code_hash,
            status=arguments.status,
            publication_runtime_raw=publication_runtime_raw,
            run_dir_fd=arguments.run_dir_fd,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
