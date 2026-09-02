"""Authenticate the completed compensated PP16 numerical rejection offline.

This module does not open JAX or a TPU backend.  It replays the exact historical
validators against the generation-bound diagnostic bytes and applies one narrow
recovery-policy overlay for the compiler helper that the historical publisher
did not admit as a dependency root.
"""

from __future__ import annotations

import base64
import json
import os
import re
import stat
import subprocess
import types
from collections.abc import Mapping, Sequence
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import google_crc32c
import numpy as np

from ..errors import BenchmarkValidationError

SOURCE_TAG = "gate_d_compensated_pp16_numerical_20260901T094622505067868Z"
SOURCE_CODE_HASH = "68c513af6e9f2e47880507422a120fdd9e478e24"
HLO_SOURCE_CODE_HASH = "94518b7d4ce788157afa98b7bc1f8144613852d5"
SOURCE_REMOTE = (
    "gs://driftbench-dsv4-uc/results/greenfield/glm52/gate_d_pp16_numerical/"
    f"{SOURCE_TAG}/diagnostic"
)
SOURCE_LEDGER_SHA256 = (
    "aa138ccafabfa2860b21c0bb1d92c0d6d1d8d01e8df1db74cdead0ecb975d6ae"
)
SOURCE_OUTPUT_SHA256 = (
    "bd017d4c7e3f42b3f60fb4b918811ea028447656227c361c84fffae2544dc1a3"
)
SOURCE_RUNNER_SHA256 = (
    "41c855fdf918976cc3bcd00f1fecd627d56e10e8ac758b8f5f79e7f8750d3b88"
)
SOURCE_DEPENDENCIES_SHA256 = (
    "3ac3abda4f3d6756c69132f3695850f7c5ab96057e8b9b4cae7f18c9dfec2189"
)
SOURCE_DRIVER_PATH = "scripts/greenfield/run_gate_d_compensated_pp16_numerical.py"
SOURCE_DRIVER_SHA256 = (
    "03d25f209a2aa28a94a577905c7c3ac0dac31b96386768e6e3a4f21eceef34cf"
)
SOURCE_PUBLISHER_PATH = (
    "scripts/greenfield/publish_gate_d_compensated_pp16_numerical.py"
)
SOURCE_PUBLISHER_SHA256 = (
    "2bb44eba989cb47cd05dca3f09a2091565b0a845b45654175c41cb97a833ed56"
)
SOURCE_LAUNCHER_SHA256 = (
    "997c57f0113ef6975ffc853bddb9aa7c04326d4561ff396291c651ece2ecc635"
)
SOURCE_WRAPPER_SHA256 = (
    "1f14411435f870ffb5b5aca9f0fb1e31bd0a65fabd1428ba839d82c224428e27"
)
SOURCE_HELPER_PATH = (
    "/home/gianl/glm-tpu-gate-d-pp16-numerical/scripts/greenfield/"
    "acquire_gate_d_compensated_pp16_hlo.py"
)
SOURCE_HELPER_REPO_PATH = "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py"
SOURCE_HELPER_SHA256 = (
    "a4599e0d88a1ef8a2df897d1677196e8cd0db5009f12b866e7316ecaadb5dd15"
)
SOURCE_HELPER_BYTES = 55_384
CAPSULE_SHA256 = "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4"
CAPSULE_INPUT_SHA256 = (
    "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
)
CAPSULE_STATE_SHA256 = (
    "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236"
)
CAPSULE_AUTHORITY_SHA256 = (
    "8b8c9cc79a18679628582a66c418a7f63e06553091928df58defbdde3971a660"
)
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "fc6384d8b266518b95bf38fbbbc188325b88bfcb781da2532d40ab05348d4305"
)
EXPECTED_STABLEHLO_SHA256 = (
    "55d7940c2aa9f0f7cda463cb816f3a5475985aab89050a5aafc22c2c88792a83"
)
EXPECTED_PREIMAGE_HLO_SHA256 = (
    "b63623498d82f67824b3be8998c09448440870b753cc1aa95d7a7765422c692b"
)
EXPECTED_BRIDGE_SHA256 = (
    "1f793179e6559db9898ae3274dfef34b2517dcc09ea3c8193c5a87fab69d8885"
)

_DIAGNOSTIC_MEMBERS = {
    "census_post.txt",
    "census_pre.txt",
    "dependencies.json",
    "failure_status.json",
    "hlo/acquired_preimage.optimized_hlo.txt",
    "hlo/compensated_pp16_stage0.optimized_hlo.txt",
    "hlo/compensated_pp16_stage0.stablehlo.mlir",
    "hlo/source_location_bridge.json",
    "mirror.sha256",
    "orchestrator.failure.log",
    "outputs.npz",
    "publisher_runtime.json",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.json",
    "runner.log",
    "sync.txt",
}
_LOCAL_EXTRA_MEMBERS = {
    "diagnostic_objects.json",
    "diagnostic_upload_receipt.json",
    "orchestrator.log",
}
_HASH = re.compile(r"[0-9a-f]{64}")


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _crc32c(raw: bytes) -> str:
    checksum = google_crc32c.Checksum(raw)
    return base64.b64encode(checksum.digest()).decode("ascii")


def _snapshot_regular(path: Path, *, limit: int = 64 << 20) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > limit
        ):
            raise BenchmarkValidationError(f"unsafe recovery source member: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 8 * 1024 * 1024):
            raw.extend(block)
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
            raise BenchmarkValidationError(
                f"recovery source member changed while reading: {path}"
            )
        return bytes(raw)
    finally:
        os.close(descriptor)


def _git_blob(worktree: Path, pin: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(worktree), "show", f"{pin}:{relative}"],
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )


def _historical_module(
    worktree: Path, pin: str, relative: str, expected_sha256: str, name: str
) -> types.ModuleType:
    raw = _git_blob(worktree, pin, relative)
    current = _snapshot_regular(worktree / relative, limit=2 << 20)
    if raw != current or _sha256(raw) != expected_sha256:
        raise BenchmarkValidationError(
            f"historical recovery module drifted: {relative}"
        )
    module = types.ModuleType(name)
    module.__file__ = str(worktree / relative)
    module.__package__ = None
    exec(compile(raw, str(worktree / relative), "exec"), module.__dict__)  # noqa: S102
    return module


def _require_census(raw: bytes) -> list[str]:
    lines = raw.decode("utf-8", errors="strict").splitlines()
    hosts = [line.split()[1] for line in lines if line.startswith("CENSUS_OK ")]
    if (
        len(hosts) != 8
        or len(set(hosts)) != 8
        or any("CENSUS_BAD" in line or "CENSUS_BUSY" in line for line in lines)
    ):
        raise BenchmarkValidationError(
            "recovery source census is not authenticated 8/8 zero work"
        )
    return sorted(hosts)


def _validate_ledger(source: Path) -> tuple[dict[str, dict[str, Any]], bytes]:
    raw = _snapshot_regular(source / "diagnostic_objects.json", limit=1 << 20)
    if _sha256(raw) != SOURCE_LEDGER_SHA256:
        raise BenchmarkValidationError("source diagnostic ledger bytes drifted")
    report = json.loads(raw)
    objects = report.get("objects") if isinstance(report, Mapping) else None
    if (
        set(report) != {"artifact_kind", "objects", "run_tag"}
        or report.get("artifact_kind")
        != "gate_d_compensated_pp16_numerical_diagnostic_ledger"
        or report.get("run_tag") != SOURCE_TAG
        or not isinstance(objects, list)
        or len(objects) != len(_DIAGNOSTIC_MEMBERS)
    ):
        raise BenchmarkValidationError("source diagnostic ledger schema drifted")
    records: dict[str, dict[str, Any]] = {}
    for record in objects:
        if not isinstance(record, Mapping) or set(record) != {
            "crc32c",
            "generation",
            "path",
            "sha256",
            "size",
        }:
            raise BenchmarkValidationError("source diagnostic object record drifted")
        relative = record["path"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != relative
            or relative in records
            or type(record["generation"]) is not str
            or not record["generation"].isdecimal()
            or type(record["size"]) is not int
            or record["size"] < 0
            or type(record["crc32c"]) is not str
            or type(record["sha256"]) is not str
            or _HASH.fullmatch(record["sha256"]) is None
        ):
            raise BenchmarkValidationError("source diagnostic object identity drifted")
        records[relative] = dict(record)
    if set(records) != _DIAGNOSTIC_MEMBERS:
        raise BenchmarkValidationError("source diagnostic object catalogue drifted")
    return records, raw


def _validate_local_catalogue(source: Path) -> None:
    observed: set[str] = set()
    for path in source.rglob("*"):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise BenchmarkValidationError("source diagnostic contains a symlink")
        if path.is_file():
            observed.add(relative)
        elif not path.is_dir():
            raise BenchmarkValidationError("source diagnostic contains a special file")
    if observed != _DIAGNOSTIC_MEMBERS | _LOCAL_EXTRA_MEMBERS:
        raise BenchmarkValidationError("source local diagnostic catalogue drifted")


def _validate_local_members(
    source: Path, records: Mapping[str, Mapping[str, Any]]
) -> dict[str, bytes]:
    snapshots: dict[str, bytes] = {}
    for relative, record in records.items():
        raw = _snapshot_regular(source / relative)
        if (
            len(raw) != record["size"]
            or _sha256(raw) != record["sha256"]
            or _crc32c(raw) != record["crc32c"]
        ):
            raise BenchmarkValidationError(
                f"source diagnostic local bytes drifted: {relative}"
            )
        snapshots[relative] = raw
    if (
        _snapshot_regular(source / "orchestrator.log", limit=1 << 20)
        != snapshots["orchestrator.failure.log"]
    ):
        raise BenchmarkValidationError("source diagnostic orchestrator copies disagree")
    return snapshots


def _validate_diagnostic_receipt(source: Path, ledger_raw: bytes) -> dict[str, Any]:
    raw = _snapshot_regular(source / "diagnostic_upload_receipt.json", limit=1 << 20)
    receipt = json.loads(raw)
    terminal = receipt.get("terminal") if isinstance(receipt, Mapping) else None
    expected_remote = f"{SOURCE_REMOTE}/diagnostic_objects.json"
    if (
        set(receipt) != {"artifact_kind", "remote", "terminal"}
        or receipt.get("artifact_kind")
        != "gate_d_compensated_pp16_numerical_diagnostic_receipt"
        or receipt.get("remote") != expected_remote
        or not isinstance(terminal, Mapping)
        or set(terminal) != {"crc32c", "generation", "path", "sha256", "size"}
        or terminal.get("path") != "diagnostic_objects.json"
        or terminal.get("sha256") != SOURCE_LEDGER_SHA256
        or terminal.get("size") != len(ledger_raw)
        or terminal.get("crc32c") != _crc32c(ledger_raw)
        or type(terminal.get("generation")) is not str
        or not terminal["generation"].isdecimal()
    ):
        raise BenchmarkValidationError("source diagnostic receipt drifted")
    return dict(terminal)


def _validate_dependency_record_schema(
    records: Any, category: str, *, verify_live: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(records, list) or not records:
        raise BenchmarkValidationError(f"source dependency records absent: {category}")
    roots = (
        Path("/opt/glm-tpu/gate-d-python-3.12.13-021044895e95"),
        Path("/opt/glm-tpu/gate-d-jax-site-55233c63939e"),
        Path("/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3"),
        Path("/lib"),
        Path("/lib64"),
        Path("/usr"),
    )
    allowed: list[dict[str, Any]] = []
    escaped: list[dict[str, Any]] = []
    paths: list[str] = []
    for value in records:
        if not isinstance(value, Mapping) or set(value) != {
            "bytes",
            "device",
            "inode",
            "path",
            "sha256",
        }:
            raise BenchmarkValidationError(
                f"source dependency schema drifted: {category}"
            )
        record = dict(value)
        path = record["path"]
        integers = (record["bytes"], record["device"], record["inode"])
        if (
            type(path) is not str
            or not path.startswith("/")
            or os.path.normpath(path) != path
            or os.path.realpath(path) != path
            or any(type(item) is not int for item in integers)
            or record["bytes"] < 0
            or record["device"] < 0
            or record["inode"] <= 0
            or type(record["sha256"]) is not str
            or _HASH.fullmatch(record["sha256"]) is None
        ):
            raise BenchmarkValidationError(
                f"source dependency identity drifted: {category}"
            )
        resolved = Path(path)
        target = (
            allowed
            if any(resolved == root or root in resolved.parents for root in roots)
            else escaped
        )
        target.append(record)
        paths.append(path)
        if verify_live:
            raw = _snapshot_regular(resolved, limit=max(record["bytes"], 1) + 1)
            metadata = os.stat(resolved, follow_symlinks=False)
            if (
                len(raw) != record["bytes"]
                or _sha256(raw) != record["sha256"]
                or metadata.st_dev != record["device"]
                or metadata.st_ino != record["inode"]
            ):
                raise BenchmarkValidationError(
                    f"source dependency live identity drifted: {path}"
                )
    if paths != sorted(set(paths), key=lambda item: os.fsencode(item)):
        raise BenchmarkValidationError(f"source dependency order drifted: {category}")
    return allowed, escaped


def _validate_dependencies(
    raw: bytes,
    runner: Mapping[str, Any],
    worktree: Path,
    *,
    verify_live: bool,
) -> dict[str, Any]:
    if _sha256(raw) != SOURCE_DEPENDENCIES_SHA256:
        raise BenchmarkValidationError("source dependency manifest bytes drifted")
    report = json.loads(raw)
    identity = runner.get("compiler_dependency_manifest")
    expected_runtime = {
        "python_executable": (
            "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12"
        ),
        "python_runtime_root": "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95",
        "python_runtime_tree_sha256": (
            "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616"
        ),
        "python_sha256": (
            "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
        ),
    }
    expected_sites = {
        "jax": {
            "manifest_sha256": (
                "ef454caafd2e4ba5da4bc7f7f73bef4e8f157afd6c795a91b09e319961941eff"
            ),
            "root": "/opt/glm-tpu/gate-d-jax-site-55233c63939e",
            "tree_sha256": (
                "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df"
            ),
        },
        "libtpu": {
            "manifest_sha256": (
                "d34064f4a0dfcdcd9ec13288ce967ae060a4650b3e86e53bc47e49756c0e97fa"
            ),
            "root": "/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3",
            "tree_sha256": (
                "db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca"
            ),
        },
    }
    expected_environment = {
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
    if (
        not isinstance(report, Mapping)
        or set(report)
        != {
            "accelerator_device_nodes_observed_mapped",
            "accelerator_device_observation_scope",
            "artifact_kind",
            "code_hash",
            "dependency_sites",
            "environment",
            "native_mappings",
            "python_modules",
            "python_runtime",
            "sealed_project_source",
        }
        or report.get("artifact_kind")
        != "gate_d_compensated_pp16_numerical_dependencies"
        or report.get("code_hash") != SOURCE_CODE_HASH
        or report.get("python_runtime") != expected_runtime
        or report.get("dependency_sites") != expected_sites
        or report.get("environment") != expected_environment
        or report.get("accelerator_device_observation_scope")
        != (
            "Unique canonical /dev/accel node paths observed mapped in the compiler "
            "process, deduplicated by path; not a complete VMA catalogue."
        )
        or not isinstance(identity, Mapping)
        or identity.get("filename") != "dependencies.json"
        or identity.get("byte_count") != len(raw)
        or identity.get("sha256") != SOURCE_DEPENDENCIES_SHA256
        or identity.get("native_mapping_count") != len(report["native_mappings"])
        or identity.get("python_module_count") != len(report["python_modules"])
        or report.get("sealed_project_source")
        != {
            key: runner["sealed_project_source"][key]
            for key in ("archive_sha256", "file_manifest_count", "file_manifest_sha256")
        }
    ):
        raise BenchmarkValidationError("source dependency manifest linkage drifted")
    _, native_escapes = _validate_dependency_record_schema(
        report["native_mappings"], "native_mappings", verify_live=verify_live
    )
    _, python_escapes = _validate_dependency_record_schema(
        report["python_modules"], "python_modules", verify_live=verify_live
    )
    _require_exact_helper_escape(native_escapes, python_escapes)

    helper_at_execution = _git_blob(worktree, SOURCE_CODE_HASH, SOURCE_HELPER_REPO_PATH)
    helper_at_hlo = _git_blob(worktree, HLO_SOURCE_CODE_HASH, SOURCE_HELPER_REPO_PATH)
    helper_live = _snapshot_regular(worktree / SOURCE_HELPER_REPO_PATH, limit=1 << 20)
    if (
        helper_at_execution != helper_at_hlo
        or helper_at_execution != helper_live
        or len(helper_at_execution) != SOURCE_HELPER_BYTES
        or _sha256(helper_at_execution) != SOURCE_HELPER_SHA256
    ):
        raise BenchmarkValidationError("escaped compiler helper Git authority drifted")
    return {
        "exception_count": 1,
        "exception_path": SOURCE_HELPER_PATH,
        "exception_sha256": SOURCE_HELPER_SHA256,
        "hlo_source_code_hash": HLO_SOURCE_CODE_HASH,
        "native_mapping_count": len(report["native_mappings"]),
        "policy": "one_exact_git_authenticated_helper_path_only",
        "python_module_count": len(report["python_modules"]),
        "source_code_hash": SOURCE_CODE_HASH,
    }


def _require_exact_helper_escape(
    native_escapes: Sequence[Mapping[str, Any]],
    python_escapes: Sequence[Mapping[str, Any]],
) -> None:
    """Apply the sole recovery-policy exception to historical dependency roots."""

    expected_escape = {
        "bytes": SOURCE_HELPER_BYTES,
        "device": 2049,
        "inode": 1_565_814,
        "path": SOURCE_HELPER_PATH,
        "sha256": SOURCE_HELPER_SHA256,
    }
    if native_escapes or python_escapes != [expected_escape]:
        raise BenchmarkValidationError(
            "recovery policy requires exactly one authenticated helper escape"
        )


def _validate_remote_manifest(
    report: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    receipt: Mapping[str, Any],
) -> None:
    objects = report.get("objects") if isinstance(report, Mapping) else None
    if (
        set(report)
        != {
            "artifact_kind",
            "exact_object_count",
            "objects",
            "source_remote",
            "status",
        }
        or report.get("artifact_kind")
        != "gate_d_compensated_pp16_source_remote_manifest"
        or report.get("source_remote") != SOURCE_REMOTE
        or report.get("status") != "SOURCE_DIAGNOSTIC_AUTHENTICATED"
        or report.get("exact_object_count") != len(_DIAGNOSTIC_MEMBERS) + 1
        or not isinstance(objects, list)
        or len(objects) != len(_DIAGNOSTIC_MEMBERS) + 1
    ):
        raise BenchmarkValidationError("source remote manifest schema drifted")
    expected = {name: dict(record) for name, record in records.items()}
    expected["diagnostic_objects.json"] = dict(receipt)
    seen: set[str] = set()
    for item in objects:
        if not isinstance(item, Mapping) or set(item) != {
            "crc32c",
            "generation",
            "path",
            "sha256",
            "size",
            "uri",
        }:
            raise BenchmarkValidationError("source remote object record drifted")
        path = item["path"]
        if path in seen or path not in expected:
            raise BenchmarkValidationError("source remote object catalogue drifted")
        if {
            key: item[key] for key in ("crc32c", "generation", "path", "sha256", "size")
        } != expected[path]:
            raise BenchmarkValidationError(f"source remote identity drifted: {path}")
        if item["uri"] != f"{SOURCE_REMOTE}/{path}":
            raise BenchmarkValidationError(f"source remote URI drifted: {path}")
        seen.add(path)
    if seen != set(expected):
        raise BenchmarkValidationError("source remote object set is incomplete")
    ledger_generation = int(receipt["generation"])
    if any(
        int(record["generation"]) >= ledger_generation for record in records.values()
    ):
        raise BenchmarkValidationError("source diagnostic ledger was not terminal-last")


def authenticate_compensated_pp16_rejection(
    source_run_dir: Path,
    *,
    worktree: Path,
    capsule: Path,
    capsule_inputs: Path,
    capsule_state: Path,
    capsule_execution_authority: Path,
    source_remote_manifest: Path | bytes,
    verify_live_dependencies: bool = True,
) -> dict[str, Any]:
    """Recompute and authenticate the rejected one-invocation TPU result."""

    _validate_local_catalogue(source_run_dir)
    records, ledger_raw = _validate_ledger(source_run_dir)
    snapshots = _validate_local_members(source_run_dir, records)
    receipt = _validate_diagnostic_receipt(source_run_dir, ledger_raw)
    remote_manifest_raw = (
        source_remote_manifest
        if isinstance(source_remote_manifest, bytes)
        else _snapshot_regular(source_remote_manifest, limit=1 << 20)
    )
    if len(remote_manifest_raw) > 1 << 20:
        raise BenchmarkValidationError("source remote manifest is oversized")
    remote_report = json.loads(remote_manifest_raw)
    _validate_remote_manifest(remote_report, records, receipt)

    runner_raw = snapshots["runner.json"]
    outputs_raw = snapshots["outputs.npz"]
    if (
        _sha256(runner_raw) != SOURCE_RUNNER_SHA256
        or _sha256(outputs_raw) != SOURCE_OUTPUT_SHA256
    ):
        raise BenchmarkValidationError("source runner/output bytes drifted")
    runner = json.loads(runner_raw)
    historical_driver = _historical_module(
        worktree,
        SOURCE_CODE_HASH,
        SOURCE_DRIVER_PATH,
        SOURCE_DRIVER_SHA256,
        "_historical_gate_d_compensated_pp16_driver",
    )
    historical_publisher = _historical_module(
        worktree,
        SOURCE_CODE_HASH,
        SOURCE_PUBLISHER_PATH,
        SOURCE_PUBLISHER_SHA256,
        "_historical_gate_d_compensated_pp16_publisher",
    )
    if (
        historical_publisher._validate_runner(runner, SOURCE_CODE_HASH)
        != "NUMERICAL_REJECTED"
    ):
        raise BenchmarkValidationError("historical runner did not validate as rejected")
    historical_publisher._validate_output_archive(outputs_raw, runner["output_arrays"])

    dependency = _validate_dependencies(
        snapshots["dependencies.json"],
        runner,
        worktree,
        verify_live=verify_live_dependencies,
    )
    if verify_live_dependencies:
        primitives = historical_publisher._load_primitives(SOURCE_CODE_HASH)
        # Recovery-policy overlay: the execution loaded the authenticated HLO helper,
        # not the numerical driver path admitted by the historical generic primitive.
        primitives.COMPILER_DRIVER_PATH = Path(SOURCE_HELPER_PATH)
        historical_publisher._validate_dependencies(
            primitives,
            runner,
            json.loads(snapshots["dependencies.json"]),
            snapshots["dependencies.json"],
            SOURCE_CODE_HASH,
        )
    acquired = snapshots["hlo/acquired_preimage.optimized_hlo.txt"]
    optimized = snapshots["hlo/compensated_pp16_stage0.optimized_hlo.txt"]
    stablehlo = snapshots["hlo/compensated_pp16_stage0.stablehlo.mlir"]
    bridge_raw = snapshots["hlo/source_location_bridge.json"]
    if (
        _sha256(acquired) != EXPECTED_PREIMAGE_HLO_SHA256
        or _sha256(optimized) != EXPECTED_OPTIMIZED_HLO_SHA256
        or _sha256(stablehlo) != EXPECTED_STABLEHLO_SHA256
        or _sha256(bridge_raw) != EXPECTED_BRIDGE_SHA256
    ):
        raise BenchmarkValidationError("source numerical HLO identity drifted")
    derived, bridge_binding = historical_driver.validate_hlo_source_location_bridge(
        json.loads(bridge_raw), acquired
    )
    if derived != optimized or bridge_binding != runner["hlo_source_location_bridge"]:
        raise BenchmarkValidationError("source numerical HLO bridge replay drifted")

    capsule_raw = _snapshot_regular(capsule, limit=1 << 20)
    capsule_input_raw = _snapshot_regular(capsule_inputs)
    capsule_state_raw = _snapshot_regular(capsule_state)
    authority_raw = _snapshot_regular(capsule_execution_authority, limit=1 << 20)
    if (
        _sha256(capsule_raw) != CAPSULE_SHA256
        or _sha256(capsule_input_raw) != CAPSULE_INPUT_SHA256
        or _sha256(capsule_state_raw) != CAPSULE_STATE_SHA256
        or _sha256(authority_raw) != CAPSULE_AUTHORITY_SHA256
        or runner["capsule"]
        != {
            "capsule_sha256": CAPSULE_SHA256,
            "execution_authority_sha256": CAPSULE_AUTHORITY_SHA256,
            "input_sha256": CAPSULE_INPUT_SHA256,
            "state_sha256": CAPSULE_STATE_SHA256,
        }
    ):
        raise BenchmarkValidationError("source capsule authority drifted")
    capsule_report = json.loads(capsule_raw)
    authority = json.loads(authority_raw)
    inputs = historical_driver._load_npz_arrays(
        capsule_input_raw, authority["input_arrays"], np
    )
    with np.load(BytesIO(outputs_raw), allow_pickle=False) as archive:
        host = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    recomputed = historical_driver.classify_outputs(host, capsule_report, inputs, np)
    if recomputed != runner["numerical"]:
        raise BenchmarkValidationError("source numerical classification replay drifted")
    hidden_update = np.ascontiguousarray(
        inputs["rms_hidden_update_bf16_bits"], dtype=np.uint16
    )
    residual = np.ascontiguousarray(inputs["rms_residual_bf16_bits"], dtype=np.uint16)
    hidden_update_fp32 = np.left_shift(
        hidden_update.astype(np.uint32), np.uint32(16)
    ).view(np.float32)
    residual_fp32 = np.left_shift(residual.astype(np.uint32), np.uint32(16)).view(
        np.float32
    )
    derived_rms_input = np.empty_like(hidden_update_fp32)
    np.add(hidden_update_fp32, residual_fp32, out=derived_rms_input)
    observed_rms_input = np.ascontiguousarray(
        host["rms_input_fp32_owners"][0, 0], dtype=np.float32
    )
    if (
        derived_rms_input.shape != (6144,)
        or derived_rms_input.tobytes() != observed_rms_input.tobytes()
    ):
        raise BenchmarkValidationError(
            "source RMS input is not the exact BF16-operand-derived FP32 sum"
        )
    watchpoints = recomputed["cpu_candidate_watchpoint_matches_diagnostic_only"]
    exact_frontier = {
        "layer1.rms_input_fp32:value": True,
        "layer1.rms_operands_bf16:hidden_update": True,
        "layer1.rms_operands_bf16:residual": True,
        "layer1.normalized:owner0": False,
        "layer1.normalized:owner1": False,
    }
    if any(
        watchpoints.get(name) is not expected
        for name, expected in exact_frontier.items()
    ):
        raise BenchmarkValidationError("source first-divergence frontier drifted")
    if (
        recomputed["accepted_tpu_event_match"] is not False
        or recomputed["contract_valid"] != [1, 1]
        or recomputed["tie_order_exact"] is not True
        or not all(recomputed["owner_agreement"].values())
    ):
        raise BenchmarkValidationError("source rejection policy drifted")

    failure = json.loads(snapshots["failure_status.json"])
    if failure != {
        "artifact_kind": "gate_d_compensated_pp16_numerical_failure",
        "code_hash": SOURCE_CODE_HASH,
        "exit_status": 1,
        "gate_d_closed": False,
        "performance_claim": False,
        "run_tag": SOURCE_TAG,
        "terminal_payload_present": False,
    }:
        raise BenchmarkValidationError("source publisher failure boundary drifted")
    if snapshots["runner.log"].decode("ascii") != (
        "GATE_D_COMPENSATED_PP16_NUMERICAL_REJECTED execution_count=1 "
        "accepted_event=false gate_d_open=true\n"
    ):
        raise BenchmarkValidationError("source runner log drifted")
    census_pre = _require_census(snapshots["census_pre.txt"])
    census_post = _require_census(snapshots["census_post.txt"])
    mirror = json.loads(snapshots["mirror.sha256"])
    mirror_blobs = {
        item.get("path"): item.get("sha256")
        for item in mirror.get("bound_blobs", [])
        if isinstance(item, Mapping)
    }
    required_runtime_blobs = {
        SOURCE_HELPER_REPO_PATH: SOURCE_HELPER_SHA256,
        "scripts/greenfield/launch_gate_d_compensated_pp16_numerical.py": (
            SOURCE_LAUNCHER_SHA256
        ),
        SOURCE_PUBLISHER_PATH: SOURCE_PUBLISHER_SHA256,
        SOURCE_DRIVER_PATH: SOURCE_DRIVER_SHA256,
        "scripts/greenfield/run_gate_d_compensated_pp16_numerical.sh": (
            SOURCE_WRAPPER_SHA256
        ),
    }
    if (
        mirror.get("commit") != SOURCE_CODE_HASH
        or mirror.get("mirror_uri") != "gs://driftbench-dsv4-uc/repos/glm-tpu/.git"
        or mirror.get("origin_remote_exact_record") is not True
        or mirror.get("commit_connectivity_fsck") is not True
        or any(
            mirror_blobs.get(path) != digest
            for path, digest in required_runtime_blobs.items()
        )
        or snapshots["sync.txt"].decode("ascii").split()[2] != SOURCE_CODE_HASH
    ):
        raise BenchmarkValidationError("source Git mirror authority drifted")
    return {
        "artifact_kind": "gate_d_compensated_pp16_rejection_authentication",
        "claim_scope": (
            "Offline authentication of one already-completed protected TPU numerical "
            "rejection; the historical publisher failed after execution and did not accept."
        ),
        "compiled_executable_invocation_count": 1,
        "capsule_authority": dict(runner["capsule"]),
        "dependency_recovery_policy": dependency,
        "first_divergence": {
            "accepted_operands_and_rms_input_exact": True,
            "derived_rms_input_from_bf16_operands_bit_exact": True,
            "derived_rms_input_sha256": historical_driver._array_sha256(
                derived_rms_input
            ),
            "first_mismatching_watchpoint": "layer1.normalized",
            "watchpoint_matches": watchpoints,
        },
        "gate_d_closed": False,
        "hlo": {
            "bridge_byte_replay_exact": True,
            "optimized_hlo_sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
            "preimage_hlo_sha256": EXPECTED_PREIMAGE_HLO_SHA256,
            "stablehlo_sha256": EXPECTED_STABLEHLO_SHA256,
        },
        "historical_publisher_accepted": False,
        "historical_runtime_blobs": required_runtime_blobs,
        "numerical": recomputed,
        "numerical_policy": {
            "accepted_event_policy": "exact_array_payload_sha256_and_valid_count",
            "internal_watchpoint_policy": "exact_shape_dtype_and_payload_sha256",
            "owner_policy": "bitwise_equal",
            "tie_order_policy": "descending_score_then_lower_position",
            "tolerance": None,
        },
        "performance_claim": False,
        "source_census_post_hosts": census_post,
        "source_census_pre_hosts": census_pre,
        "source_code_hash": SOURCE_CODE_HASH,
        "source_diagnostic_ledger_sha256": SOURCE_LEDGER_SHA256,
        "source_diagnostic_terminal_generation": receipt["generation"],
        "source_output_sha256": SOURCE_OUTPUT_SHA256,
        "source_remote": SOURCE_REMOTE,
        "source_remote_manifest_sha256": _sha256(remote_manifest_raw),
        "source_tag": SOURCE_TAG,
        "status": "NUMERICAL_REJECTED_RECOVERED",
        "tpu_rerun_performed": False,
    }


def require_eight_host_census(path: Path) -> dict[str, Any]:
    """Return the authenticated identity of one fresh zero-work census."""

    raw = _snapshot_regular(path, limit=1 << 20)
    return {"hosts": _require_census(raw), "sha256": _sha256(raw)}


def require_eight_host_census_bytes(raw: bytes) -> dict[str, Any]:
    """Authenticate an in-memory zero-work census without reopening a pathname."""

    if len(raw) > 1 << 20:
        raise BenchmarkValidationError("recovery census is oversized")
    return {"hosts": _require_census(raw), "sha256": _sha256(raw)}


def canonical_json(value: Any) -> bytes:
    """Encode a deterministic append-only evidence object."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def validate_recovery_remote_vacancy(
    live: Sequence[str], all_versions: Sequence[str], soft_deleted: Sequence[str]
) -> None:
    """Reject any live, noncurrent, or soft-deleted recovery-prefix history."""

    if live or all_versions or soft_deleted:
        raise BenchmarkValidationError(
            "recovery remote prefix is not historically vacant"
        )
