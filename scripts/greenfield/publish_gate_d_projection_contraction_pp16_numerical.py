#!/usr/bin/env python3
"""Append-only publisher for the bounded PP16 projection numerical replay."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import types
import zipfile
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

REPO = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
SOURCE_PATH = (
    "scripts/greenfield/publish_gate_d_projection_contraction_pp16_numerical.py"
)
BASE_PATH = "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
BASE_PIN = "986378238ac6458307aea69ef1f5e12bf82bc020"
BASE_SHA256 = "f3f20a01fd37bb82988cd77f69fa7b0a780d120568bab0db4162f42e7855bc97"
BUCKET_NAME = "driftbench-dsv4-uc"
REMOTE_ROOT = "results/greenfield/glm52/gate_d_projection_contraction_pp16_numerical/"
TAG_PATTERN_TEXT = r"gate_d_projection_contraction_pp16_numerical_[0-9]{8}T[0-9]{15}Z"
TAG_PATTERN = re.compile(TAG_PATTERN_TEXT)
EXPECTED_STABLEHLO_SHA256 = (
    "4b3fa252e837d381208453b06d7e369947fa4c6c58c795edc8ff5a8ea8c9e2e3"
)
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "817ba2ed87c33ec928f834fcf3a003ce63d3dedf4061a7c64fe354a26ac498ea"
)
EXPECTED_WK_WEIGHT_FP32_SHA256 = (
    "b2c67e0fdf4d7292494778233e9813d8256f2fb88b6f7376870379832fbab24b"
)
EXPECTED_OWNER_SHA256 = {
    "current_key_owners": "5006ad4f7224047652bbc46e6275b4c82bec0dead13f5f2b5f91494a2415329b",
    "normalized_hidden_owners": "28b7db466b74f462b5cd3109cb052bcce0c8be185d3f24199239f75136b22c20",
    "projected_key_owners": "f16ad9903c6d219c9796f2a1aac32b0fd7b09135ecdfadb9a19dbef5cc1a9aa6",
}
EXPECTED_OUTPUT_LAYOUT = {
    "current_key_owners": ((2, 1, 128), "<f4", 4),
    "normalized_hidden_owners": ((2, 1, 6144), "<u2", 2),
    "projected_key_owners": ((2, 1, 128), "<f4", 4),
}
EXPECTED_PREDECESSORS = {
    "capsule_input_sha256": "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b",
    "capsule_sha256": "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4",
    "capsule_state_sha256": "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236",
    "execution_authority_sha256": "8b8c9cc79a18679628582a66c418a7f63e06553091928df58defbdde3971a660",
    "frontier_authority_sha256": "4a6be0f33f221df46f384cd2047da1aa664ae4c45763e68f0f1df05f8644e20c",
    "hlo_success_authority_sha256": "54eb6105b81d9ffdbdd3b90735059fb324701509fe8efd003033bb3cff7e0ad7",
    "host_materialization_authority_sha256": "a7e5b393f7c61181f6b2aab9531d788fb27594d9cf980055c165b6f278ba4f99",
}
EXPECTED_PHYSICAL_GROUP = {
    "coordinates": [[0, 0, 0], [1, 0, 0]],
    "device_ids": [0, 1],
    "local_device_count_visible": 4,
    "mesh_device_count": 2,
    "process_index": 0,
    "stage_id": 0,
}
EXPECTED_COMPILER_ENVIRONMENT = {
    "GLM_GATE_D_PROJECTION_CONTRACTION_NUMERICAL": "1",
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
SUCCESS_PAYLOAD = (
    "census_post.txt",
    "census_pre.txt",
    "dependencies.json",
    "evidence.json",
    "hlo/acquired_preimage.optimized_hlo.txt",
    "hlo/acquired_preimage.stablehlo.mlir",
    "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt",
    "hlo/projection_contraction_pp16_stage0.stablehlo.mlir",
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
)
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
_GIT_ENVIRONMENT = {
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
}


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(REPO), *arguments], env=_GIT_ENVIRONMENT
    )


def _snapshot(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe publisher source: {path}")
        chunks = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            len(raw) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"publisher source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    if _git_bytes("for-each-ref", "--format=%(refname)", "refs/replace"):
        raise RuntimeError("projection numerical publisher found replacement refs")
    raw = _snapshot(Path(__file__))
    if (
        _git_bytes("rev-parse", "HEAD").decode().strip() != code_pin
        or _git_bytes("rev-parse", f"{code_pin}^{{commit}}").decode().strip()
        != code_pin
        or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("projection numerical publisher is not the committed blob")


def _load_base(code_pin: str) -> types.ModuleType:
    path = REPO / BASE_PATH
    raw = _snapshot(path)
    if (
        raw != _git_bytes("show", f"{code_pin}:{BASE_PATH}")
        or raw != _git_bytes("show", f"{BASE_PIN}:{BASE_PATH}")
        or sha256(raw).hexdigest() != BASE_SHA256
    ):
        raise RuntimeError("projection numerical publisher base bytes drifted")
    module = types.ModuleType("_gate_d_projection_publisher_base")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    module.REPO = REPO
    module.RUN_ROOT = RUN_ROOT
    module.BUCKET_NAME = BUCKET_NAME
    module.REMOTE_ROOT = REMOTE_ROOT
    module.TAG_PATTERN = TAG_PATTERN
    module._EXPECTED_COMPILER_ENVIRONMENT = dict(EXPECTED_COMPILER_ENVIRONMENT)
    module._git_bytes = lambda *args: _git_bytes(*args)
    return module


def _publication_runtime(base: Any) -> dict[str, Any]:
    result = base.validate_publication_runtime()
    result["artifact_kind"] = (
        "gate_d_projection_contraction_pp16_numerical_publisher_runtime"
    )
    return result


def _require_preterminal(base: Any, run_fd: int) -> None:
    base.require_preterminal(run_fd)
    for name in ("NUMERICAL_RESULT", "terminal_upload_receipt.json"):
        try:
            os.stat(name, dir_fd=run_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise RuntimeError("projection numerical terminal publication already began")


def _expected_status(runner: Mapping[str, Any]) -> tuple[bool, str, str]:
    numerical = runner.get("numerical")
    accepted = (
        numerical.get("accepted_tpu_projection_match")
        if isinstance(numerical, Mapping)
        else None
    )
    if type(accepted) is not bool:
        raise RuntimeError("projection numerical classification is absent")
    status = "NUMERICAL_ACCEPTED" if accepted else "NUMERICAL_REJECTED"
    classification = (
        "BOUNDED_TPU_PROJECTION_NUMERICAL_ACCEPTED;ROOT_CAUSE_FIX_UNPROVEN;GATE_D_OPEN"
        if accepted
        else "BOUNDED_TPU_PROJECTION_NUMERICAL_REJECTED;PROJECTION_CONTRACTION_MISMATCH;GATE_D_OPEN"
    )
    return accepted, status, classification


def _parse_npy(raw: bytes) -> tuple[tuple[int, ...], str, bytes]:
    if not raw.startswith(b"\x93NUMPY") or len(raw) < 10:
        raise RuntimeError("projection numerical NPY header is invalid")
    major, minor = raw[6], raw[7]
    if (major, minor) == (1, 0):
        header_size = int.from_bytes(raw[8:10], "little")
        header_start = 10
    elif major in (2, 3) and minor == 0:
        if len(raw) < 12:
            raise RuntimeError("projection numerical NPY v2 header is truncated")
        header_size = int.from_bytes(raw[8:12], "little")
        header_start = 12
    else:
        raise RuntimeError("projection numerical NPY version is unsupported")
    header_end = header_start + header_size
    if header_end > len(raw) or header_size > 1 << 20:
        raise RuntimeError("projection numerical NPY header size is unsafe")
    try:
        header = ast.literal_eval(raw[header_start:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("projection numerical NPY header is malformed") from error
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header.get("fortran_order") is not False
        or not isinstance(header.get("shape"), tuple)
        or not all(type(item) is int and item >= 0 for item in header["shape"])
        or not isinstance(header.get("descr"), str)
    ):
        raise RuntimeError("projection numerical NPY schema drifted")
    return header["shape"], header["descr"], raw[header_end:]


def _validate_output_npz(
    raw: bytes, runner: Mapping[str, Any], numerical: Mapping[str, Any]
) -> bool:
    expected_names = {f"{name}.npy" for name in EXPECTED_OUTPUT_LAYOUT}
    runner_arrays = runner.get("output_arrays")
    numerical_arrays = numerical.get("outputs")
    if (
        not isinstance(runner_arrays, Mapping)
        or set(runner_arrays) != set(EXPECTED_OUTPUT_LAYOUT)
        or not isinstance(numerical_arrays, Mapping)
        or set(numerical_arrays) != set(EXPECTED_OUTPUT_LAYOUT)
    ):
        raise RuntimeError("projection numerical output array catalogue drifted")
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or set(names) != expected_names
            or sum(item.file_size for item in infos) > 1 << 20
            or any(item.is_dir() or item.flag_bits & 0x1 for item in infos)
        ):
            raise RuntimeError("projection numerical output NPZ catalogue drifted")
        for name, (shape, dtype, item_size) in EXPECTED_OUTPUT_LAYOUT.items():
            member = archive.read(f"{name}.npy")
            observed_shape, observed_dtype, payload = _parse_npy(member)
            owner_size = (len(payload) // 2) if observed_shape == shape else -1
            owner_hashes = [
                sha256(
                    payload[index * owner_size : (index + 1) * owner_size]
                ).hexdigest()
                if owner_size >= 0
                else None
                for index in range(2)
            ]
            runner_record = runner_arrays[name]
            numerical_record = numerical_arrays[name]
            shape_exact = observed_shape == shape
            dtype_exact = observed_dtype == dtype
            payload_size_exact = (
                len(payload) == item_size * shape[0] * shape[1] * shape[2]
            )
            if dtype == "<f4" and payload_size_exact:
                finite = all(
                    math.isfinite(value)
                    for (value,) in struct.iter_unpack("<f", payload)
                )
            elif dtype == "<u2" and payload_size_exact:
                finite = all(
                    int.from_bytes(payload[offset : offset + 2], "little") & 0x7F80
                    != 0x7F80
                    for offset in range(0, len(payload), 2)
                )
            else:
                finite = False
            owners_equal = (
                shape_exact
                and dtype_exact
                and payload_size_exact
                and payload[:owner_size] == payload[owner_size:]
            )
            witness_exact = owner_hashes == [
                EXPECTED_OWNER_SHA256[name],
                EXPECTED_OWNER_SHA256[name],
            ]
            derived_record = {
                "dtype_exact": dtype_exact,
                "finite": finite,
                "owner_sha256": owner_hashes,
                "owners_equal": owners_equal,
                "shape_exact": shape_exact,
                "witness_exact": witness_exact,
            }
            if (
                not shape_exact
                or not dtype_exact
                or not payload_size_exact
                or not isinstance(runner_record, Mapping)
                or runner_record
                != {
                    "array_sha256": sha256(payload).hexdigest(),
                    "shape": list(shape),
                    "storage_dtype": dtype,
                }
                or not isinstance(numerical_record, Mapping)
                or numerical_record != derived_record
            ):
                raise RuntimeError(f"projection numerical output bytes drifted: {name}")
    derived_accepted = all(
        all(
            record.get(flag) is True
            for flag in (
                "dtype_exact",
                "finite",
                "owners_equal",
                "shape_exact",
                "witness_exact",
            )
        )
        for record in numerical_arrays.values()
    )
    if numerical.get("accepted_tpu_projection_match") is not derived_accepted:
        raise RuntimeError(
            "projection numerical acceptance flag differs from output bytes"
        )
    return derived_accepted


def _validate_dependencies(
    base: Any,
    runner: Mapping[str, Any],
    dependencies_raw: bytes,
    code_pin: str,
) -> None:
    dependencies = json.loads(dependencies_raw)
    identity = runner.get("compiler_dependency_manifest")
    if not isinstance(dependencies, Mapping) or not isinstance(identity, Mapping):
        raise TypeError("projection numerical dependency authority is absent")
    accelerator = dependencies.get("accelerator_device_nodes_observed_mapped")
    native = dependencies.get("native_mappings")
    modules = dependencies.get("python_modules")
    base._validate_accelerator_device_mapping_records(accelerator, verify_live=True)
    base._validate_dependency_records(native, "native_mappings", verify_live=True)
    base._validate_dependency_records(modules, "python_modules", verify_live=True)
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
        or dependencies.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_numerical_dependencies"
        or dependencies.get("code_hash") != code_pin
        or dependencies.get("environment") != EXPECTED_COMPILER_ENVIRONMENT
        or identity.get("filename") != "dependencies.json"
        or identity.get("byte_count") != len(dependencies_raw)
        or identity.get("sha256") != sha256(dependencies_raw).hexdigest()
        or identity.get("accelerator_device_node_count") != len(accelerator)
        or identity.get("accelerator_device_nodes_sha256")
        != base._accelerator_device_nodes_sha256(accelerator)
        or identity.get("native_mapping_count") != len(native)
        or identity.get("python_module_count") != len(modules)
        or runner.get("sealed_project_source")
        != dependencies.get("sealed_project_source")
    ):
        raise RuntimeError("projection numerical dependency authority drifted")


def _prepare_success(
    base: Any,
    run_fd: int,
    *,
    code_pin: str,
    run_tag: str,
    remote: str,
    elapsed: int,
) -> dict[str, bytes]:
    runner_raw = base.snapshot_member(run_fd, "runner.json")
    dependencies_raw = base.snapshot_member(run_fd, "dependencies.json")
    publisher_runtime_raw = base.snapshot_member(
        run_fd, "publisher_runtime.json", limit=1 << 20
    )
    runner = json.loads(runner_raw)
    if not isinstance(runner, Mapping):
        raise TypeError("projection numerical runner report is not an object")
    accepted, status, classification = _expected_status(runner)
    expected_keys = {
        "artifact_kind",
        "claim_scope",
        "classification",
        "code_hash",
        "compiled_executable_invocation_count",
        "compiler_dependency_manifest",
        "gate_d_closed",
        "host_transfer_count",
        "hlo",
        "memory_after_compile",
        "memory_after_execute",
        "memory_before_compile",
        "numerical",
        "output_artifact",
        "output_arrays",
        "performance_claim",
        "physical_group",
        "predecessors",
        "projection_source_authority",
        "root_cause_fix_proven",
        "runtime",
        "schema_version",
        "sealed_project_source",
        "status",
        "tpu_execution_elapsed_ns_diagnostic_only",
        "tpu_numerical_execution_performed",
        "wk_weight",
    }
    if (
        set(runner) != expected_keys
        or runner.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_numerical_replay"
        or runner.get("schema_version") != 1
        or runner.get("code_hash") != code_pin
        or runner.get("status") != status
        or runner.get("classification") != classification
        or runner.get("claim_scope")
        != "One exact-input, one-invocation, two-chip PP16 projection/current-key witness only. No decoder, 8K, token, performance, or Gate-D closure claim."
        or runner.get("compiled_executable_invocation_count") != 1
        or runner.get("host_transfer_count") != 1
        or runner.get("tpu_numerical_execution_performed") is not True
        or runner.get("gate_d_closed") is not False
        or runner.get("root_cause_fix_proven") is not False
        or runner.get("performance_claim") is not False
        or runner.get("physical_group") != EXPECTED_PHYSICAL_GROUP
        or runner.get("predecessors") != EXPECTED_PREDECESSORS
        or runner.get("projection_source_authority")
        != base._projection_contraction_source_authority(code_pin)
        or runner.get("wk_weight")
        != {
            "array_sha256": EXPECTED_WK_WEIGHT_FP32_SHA256,
            "semantic_dtype": "float32",
            "shape": [2, 128, 6144],
        }
        or type(runner.get("tpu_execution_elapsed_ns_diagnostic_only")) is not int
        or runner["tpu_execution_elapsed_ns_diagnostic_only"] <= 0
    ):
        raise RuntimeError("projection numerical runner claim boundary drifted")
    _validate_dependencies(base, runner, dependencies_raw, code_pin)
    numerical = runner["numerical"]
    numerical_outputs = numerical.get("outputs")
    if (
        set(numerical) != {"accepted_tpu_projection_match", "outputs"}
        or not isinstance(numerical_outputs, Mapping)
        or set(numerical_outputs)
        != {
            "current_key_owners",
            "normalized_hidden_owners",
            "projected_key_owners",
        }
        or not all(isinstance(record, Mapping) for record in numerical_outputs.values())
        or any(
            set(record)
            != {
                "dtype_exact",
                "finite",
                "owner_sha256",
                "owners_equal",
                "shape_exact",
                "witness_exact",
            }
            for record in numerical_outputs.values()
        )
        or accepted
        != all(
            all(
                record.get(flag) is True
                for flag in (
                    "dtype_exact",
                    "finite",
                    "owners_equal",
                    "shape_exact",
                    "witness_exact",
                )
            )
            for record in numerical_outputs.values()
        )
    ):
        raise RuntimeError("projection numerical output classification drifted")
    hlo_files = {
        "optimized_hlo": "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt",
        "stablehlo": "hlo/projection_contraction_pp16_stage0.stablehlo.mlir",
    }
    hlo_payload: dict[str, bytes] = {}
    for kind, relative in hlo_files.items():
        raw = base.snapshot_member(run_fd, relative)
        preimage = base.snapshot_member(
            run_fd,
            "hlo/acquired_preimage."
            + ("optimized_hlo.txt" if kind == "optimized_hlo" else "stablehlo.mlir"),
        )
        expected_sha = (
            EXPECTED_OPTIMIZED_HLO_SHA256
            if kind == "optimized_hlo"
            else EXPECTED_STABLEHLO_SHA256
        )
        identity = runner.get("hlo", {}).get(kind, {})
        if (
            raw != preimage
            or sha256(raw).hexdigest() != expected_sha
            or identity != {"byte_count": len(raw), "sha256": expected_sha}
        ):
            raise RuntimeError(f"projection numerical {kind} identity drifted")
        hlo_payload[relative] = raw
        hlo_payload[
            "hlo/acquired_preimage."
            + ("optimized_hlo.txt" if kind == "optimized_hlo" else "stablehlo.mlir")
        ] = preimage
    output_raw = base.snapshot_member(run_fd, "outputs.npz")
    if runner.get("output_artifact") != {
        "byte_count": len(output_raw),
        "filename": "outputs.npz",
        "sha256": sha256(output_raw).hexdigest(),
    }:
        raise RuntimeError("projection numerical output artifact drifted")
    if _validate_output_npz(output_raw, runner, numerical) is not accepted:
        raise RuntimeError("projection numerical status differs from output bytes")
    mirror_raw = base.snapshot_member(run_fd, "mirror.sha256", limit=2 << 20)
    base._validate_mirror_replay(mirror_raw, code_pin)
    sync_raw = base.snapshot_member(run_fd, "sync.txt", limit=1 << 20)
    expected_sync = (
        f"SYNC_OK {os.uname().nodename} {code_pin} numerical_host_only=1 "
        "sealed_source_archive=1\n"
    ).encode("ascii")
    if sync_raw != expected_sync:
        raise RuntimeError("projection numerical host code authority drifted")
    vacancy_raw = base.snapshot_member(run_fd, "remote_vacancy.raw.txt", limit=1 << 20)
    vacancy_summary = base.snapshot_member(run_fd, "remote_vacancy.txt", limit=1 << 20)
    base._validate_remote_vacancy_evidence(vacancy_raw, vacancy_summary, remote)
    census_pre = base.snapshot_member(run_fd, "census_pre.txt")
    census_post = base.snapshot_member(run_fd, "census_post.txt")
    base._require_census(census_pre, "CENSUS_OK")
    base._require_census(census_post, "CENSUS_OK")
    orchestrator = base.snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
    base.write_member_exclusive(run_fd, "orchestrator.sealed.log", orchestrator)
    summary = {
        "artifact_kind": "gate_d_projection_contraction_pp16_numerical_summary",
        "classification": classification,
        "code_hash": code_pin,
        "elapsed_seconds": elapsed,
        "gate_d_closed": False,
        "performance_claim": False,
        "remote_prefix": remote,
        "root_cause_fix_proven": False,
        "run_tag": run_tag,
        "status": status,
        "tpu_numerical_execution_performed": True,
    }
    summary_raw = base._canonical(summary)
    base.write_member_exclusive(run_fd, "summary.json", summary_raw)
    payload = {
        "census_post.txt": census_post,
        "census_pre.txt": census_pre,
        "dependencies.json": dependencies_raw,
        **hlo_payload,
        "mirror.sha256": mirror_raw,
        "orchestrator.sealed.log": orchestrator,
        "outputs.npz": output_raw,
        "publisher_runtime.json": publisher_runtime_raw,
        "remote_vacancy.raw.txt": vacancy_raw,
        "remote_vacancy.txt": vacancy_summary,
        "runner.json": runner_raw,
        "runner.log": base.snapshot_member(run_fd, "runner.log", limit=256 << 20),
        "summary.json": summary_raw,
        "sync.txt": sync_raw,
    }
    evidence = {
        "artifact_kind": "gate_d_projection_contraction_pp16_numerical_local_evidence",
        "classification": classification,
        "code_hash": code_pin,
        "files": [base._record(name, payload[name]) for name in sorted(payload)],
        "gate_d_closed": False,
        "performance_claim": False,
        "root_cause_fix_proven": False,
        "run_tag": run_tag,
        "status": status,
    }
    evidence_raw = base._canonical(evidence)
    base.write_member_exclusive(run_fd, "evidence.json", evidence_raw)
    payload["evidence.json"] = evidence_raw
    if tuple(sorted(payload)) != tuple(sorted(SUCCESS_PAYLOAD)):
        raise RuntimeError("projection numerical success inventory drifted")
    if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {"orchestrator.log"}:
        raise RuntimeError("projection numerical local preterminal inventory drifted")
    return payload


def _result_authority_line(
    status: str,
    marker_sha256: str,
    terminal_record: Mapping[str, Any],
) -> str:
    generation = terminal_record.get("generation")
    terminal_sha256 = terminal_record.get("sha256")
    if (
        status not in {"NUMERICAL_ACCEPTED", "NUMERICAL_REJECTED"}
        or not re.fullmatch(r"[0-9a-f]{64}", marker_sha256)
        or type(generation) is not str
        or not re.fullmatch(r"[0-9]+", generation)
        or type(terminal_sha256) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", terminal_sha256)
    ):
        raise RuntimeError("projection numerical result authority drifted")
    return (
        f"NUMERICAL_RESULT status={status} marker_sha256={marker_sha256} "
        f"terminal_generation={generation} terminal_sha256={terminal_sha256}"
    )


def _publish_success(
    base: Any,
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    elapsed: int,
    publication_runtime_raw: bytes,
    run_dir_fd: int,
) -> str:
    run_tag = base.validate_run_dir(run_dir)
    run_fd = base._run_fd(run_dir, run_dir_fd)
    try:
        _require_preterminal(base, run_fd)
        base.require_publication_runtime(run_fd, publication_runtime_raw)
        payload = _prepare_success(
            base,
            run_fd,
            code_pin=code_pin,
            run_tag=run_tag,
            remote=remote,
            elapsed=elapsed,
        )
        runner = json.loads(payload["runner.json"])
        bucket, prefix = base._bucket_and_prefix(remote, None)
        base._require_never_used_prefix(bucket, prefix)
        records = []
        for relative in sorted(payload):
            record = base._upload_bound(bucket, prefix + relative, payload[relative])
            record["path"] = relative
            records.append(record)
        ledger_raw = base._canonical(
            {
                "artifact_kind": "gate_d_projection_contraction_pp16_numerical_remote_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "remote_objects.json", ledger_raw)
        ledger_record = base._upload_bound(
            bucket, prefix + "remote_objects.json", ledger_raw
        )
        ledger_record["path"] = "remote_objects.json"
        for record in records:
            base._replay_bound(bucket, prefix + record["path"], record)
        base._replay_bound(bucket, prefix + "remote_objects.json", ledger_record)
        expected_remote = set(payload) | {"remote_objects.json"}
        if base._observed_names(bucket, prefix) != expected_remote:
            raise RuntimeError("projection numerical preterminal remote set drifted")
        marker = {
            "artifact_kind": "gate_d_projection_contraction_pp16_numerical_result",
            "evidence_sha256": sha256(payload["evidence.json"]).hexdigest(),
            "gate_d_closed": False,
            "performance_claim": False,
            "remote_ledger": ledger_record,
            "root_cause_fix_proven": False,
            "run_tag": run_tag,
            "status": runner["status"],
            "summary_sha256": sha256(payload["summary.json"]).hexdigest(),
            "tpu_numerical_execution_performed": True,
        }
        marker["marker_payload_sha256"] = sha256(base._canonical(marker)).hexdigest()
        terminal_raw = base._canonical(marker)
        base.write_member_exclusive(run_fd, "NUMERICAL_RESULT", terminal_raw)
        if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {
            "orchestrator.log",
            "remote_objects.json",
            "NUMERICAL_RESULT",
        }:
            raise RuntimeError("projection numerical local terminal inventory drifted")
        # This must remain the final remote mutation in the successful result path.
        terminal_record = base._upload_bound(
            bucket, prefix + "NUMERICAL_RESULT", terminal_raw
        )
        terminal_record["path"] = "NUMERICAL_RESULT"
        base._replay_bound(bucket, prefix + "NUMERICAL_RESULT", terminal_record)
        if base._observed_names(bucket, prefix) != expected_remote | {
            "NUMERICAL_RESULT"
        }:
            raise RuntimeError("projection numerical terminal remote set drifted")
        base.write_member_exclusive(
            run_fd,
            "terminal_upload_receipt.json",
            base._canonical(
                {
                    "artifact_kind": "gate_d_projection_contraction_pp16_numerical_terminal_receipt",
                    "remote": remote + "/NUMERICAL_RESULT",
                    "terminal": terminal_record,
                }
            ),
        )
        if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {
            "orchestrator.log",
            "remote_objects.json",
            "NUMERICAL_RESULT",
            "terminal_upload_receipt.json",
        }:
            raise RuntimeError(
                "projection numerical terminal receipt inventory drifted"
            )
        return _result_authority_line(
            runner["status"],
            marker["marker_payload_sha256"],
            terminal_record,
        )
    finally:
        os.close(run_fd)


def _publish_diagnostic(
    base: Any,
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    status: int,
    publication_runtime_raw: bytes,
    run_dir_fd: int,
) -> None:
    run_tag = base.validate_run_dir(run_dir)
    run_fd = base._run_fd(run_dir, run_dir_fd)
    try:
        _require_preterminal(base, run_fd)
        base.require_publication_runtime(run_fd, publication_runtime_raw)
        failure_raw = base._canonical(
            {
                "artifact_kind": "gate_d_projection_contraction_pp16_numerical_failure",
                "code_hash": code_pin,
                "exit_status": status,
                "gate_d_closed": False,
                "performance_claim": False,
                "root_cause_fix_proven": False,
                "run_tag": run_tag,
                "terminal_payload_present": False,
            }
        )
        base.write_member_exclusive(run_fd, "failure_status.json", failure_raw)
        orchestrator = base.snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
        base.write_member_exclusive(run_fd, "orchestrator.failure.log", orchestrator)
        excluded = {
            "orchestrator.log",
            "NUMERICAL_RESULT",
            "diagnostic_objects.json",
            "diagnostic_upload_receipt.json",
        }
        members = [
            relative
            for relative in sorted(base.local_members(run_fd))
            if relative not in excluded
        ]
        if not members or len(members) > 64:
            raise RuntimeError("projection numerical diagnostic inventory is unsafe")
        payload = {name: base.snapshot_member(run_fd, name) for name in members}
        vacancy_raw = base.snapshot_member(
            run_fd, "remote_vacancy.raw.txt", limit=1 << 20
        )
        vacancy_summary = base.snapshot_member(
            run_fd, "remote_vacancy.txt", limit=1 << 20
        )
        base._validate_remote_vacancy_evidence(vacancy_raw, vacancy_summary, remote)
        bucket, prefix = base._bucket_and_prefix(remote, None)
        base._require_never_used_prefix(bucket, prefix)
        diagnostic_prefix = prefix + "diagnostic/"
        records = []
        for relative in sorted(payload):
            record = base._upload_bound(
                bucket, diagnostic_prefix + relative, payload[relative]
            )
            record["path"] = relative
            records.append(record)
        for record in records:
            base._replay_bound(bucket, diagnostic_prefix + record["path"], record)
        if base._observed_names(bucket, diagnostic_prefix) != set(payload):
            raise RuntimeError("projection numerical diagnostic object set drifted")
        ledger_raw = base._canonical(
            {
                "artifact_kind": "gate_d_projection_contraction_pp16_numerical_diagnostic_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "diagnostic_objects.json", ledger_raw)
        # With no numerical result terminal, the diagnostic ledger is final remotely.
        ledger_record = base._upload_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_raw
        )
        ledger_record["path"] = "diagnostic_objects.json"
        base._replay_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_record
        )
        if base._observed_names(bucket, diagnostic_prefix) != set(payload) | {
            "diagnostic_objects.json"
        }:
            raise RuntimeError("projection numerical diagnostic terminal set drifted")
        base.write_member_exclusive(
            run_fd,
            "diagnostic_upload_receipt.json",
            base._canonical(
                {
                    "artifact_kind": "gate_d_projection_contraction_pp16_numerical_diagnostic_terminal_receipt",
                    "remote": remote + "/diagnostic/diagnostic_objects.json",
                    "terminal": ledger_record,
                }
            ),
        )
    finally:
        os.close(run_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--run-dir-fd", type=int)
    modes = parser.add_subparsers(dest="mode", required=True)
    init = modes.add_parser("init")
    init.add_argument("--run-dir", type=Path, required=True)
    success = modes.add_parser("success")
    success.add_argument("--run-dir", type=Path, required=True)
    success.add_argument("--remote-prefix", required=True)
    success.add_argument("--elapsed", type=int, required=True)
    diagnostic = modes.add_parser("diagnostic")
    diagnostic.add_argument("--run-dir", type=Path, required=True)
    diagnostic.add_argument("--remote-prefix", required=True)
    diagnostic.add_argument("--status", type=int, required=True)
    write = modes.add_parser("write")
    write.add_argument("--run-dir", type=Path, required=True)
    write.add_argument("--member", choices=sorted(WRAPPER_WRITE_MEMBERS), required=True)
    append = modes.add_parser("append-log")
    append.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    arguments = parse_args()
    base = _load_base(arguments.expected_code_hash)
    base.validate_environment()
    publication_runtime_raw = base._canonical(_publication_runtime(base))
    _verify_running_source(
        arguments.expected_code_hash, arguments.expected_source_sha256
    )
    if arguments.mode != "init" and arguments.run_dir_fd != 7:
        raise RuntimeError("projection numerical publisher requires run-directory fd 7")
    if arguments.mode == "init":
        print(
            "RUN_IDENTITY "
            + base.initialize_run_dir(
                arguments.run_dir, publication_runtime_raw=publication_runtime_raw
            ),
            flush=True,
        )
    elif arguments.mode == "success":
        print(
            _publish_success(
                base,
                arguments.run_dir,
                arguments.remote_prefix,
                code_pin=arguments.expected_code_hash,
                elapsed=arguments.elapsed,
                publication_runtime_raw=publication_runtime_raw,
                run_dir_fd=arguments.run_dir_fd,
            ),
            flush=True,
        )
    elif arguments.mode == "diagnostic":
        _publish_diagnostic(
            base,
            arguments.run_dir,
            arguments.remote_prefix,
            code_pin=arguments.expected_code_hash,
            status=arguments.status,
            publication_runtime_raw=publication_runtime_raw,
            run_dir_fd=arguments.run_dir_fd,
        )
    elif arguments.mode == "write":
        run_fd = base._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            _require_preterminal(base, run_fd)
            base.require_publication_runtime(run_fd, publication_runtime_raw)
            base.write_preterminal_member(run_fd, arguments.member, sys.stdin.buffer)
        finally:
            os.close(run_fd)
    else:
        run_fd = base._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            _require_preterminal(base, run_fd)
            base.require_publication_runtime(run_fd, publication_runtime_raw)
            payload = sys.stdin.buffer.read((1 << 20) + 1)
            if len(payload) > 1 << 20:
                raise RuntimeError("projection numerical log append exceeds limit")
            base.append_preterminal_log(run_fd, payload)
        finally:
            os.close(run_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
