#!/usr/bin/env python3
"""Authenticate the burned forced-round PP16 diagnostic without rerunning TPU.

This program is deliberately read-only.  It validates the local generation ledger,
replays every generation-qualified GCS object, reclassifies the raw output arrays
against the sealed accepted capsule, and emits one JSON report on stdout.  It never
calls the mutating success publisher and never compiles or executes a JAX graph.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import stat
import subprocess
import types
import zipfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import google_crc32c

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
RUN_PIN = "8db00de0da3f166f3b165668687a1fed88b99fa6"
HLO_ACQUISITION_PIN = "b8bdeb14c12a3742f3cc27ef91a4a2e406d412a5"
RUN_TAG = "gate_d_forced_round_pp16_numerical_20260901T164835192240185Z"
RUN_DIR = Path("/home/gianl/gate-d-runs") / RUN_TAG
REMOTE = (
    "gs://driftbench-dsv4-uc/results/greenfield/glm52/"
    f"gate_d_forced_round_pp16_numerical/{RUN_TAG}/diagnostic"
)
BASE_REMOTE = REMOTE.removesuffix("/diagnostic")
SOURCE_PATH = (
    "scripts/greenfield/"
    "adjudicate_gate_d_forced_round_pp16_numerical_diagnostic.py"
)
PUBLISHER_PATH = "scripts/greenfield/publish_gate_d_forced_round_pp16_numerical.py"
DRIVER_PATH = "scripts/greenfield/run_gate_d_forced_round_pp16_numerical.py"
FORCED_HLO_HELPER_SOURCE_PATH = (
    "scripts/greenfield/acquire_gate_d_forced_round_pp16_hlo.py"
)
FORCED_HLO_HELPER_PATH = WORKTREE / FORCED_HLO_HELPER_SOURCE_PATH
FORCED_HLO_HELPER_SHA256 = (
    "046040d382567ccef94792b97e160b0f79b673ab1f0f1d46bfd20c57e884faf7"
)
CAPSULE_ROOT = Path(
    "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z"
)
CAPSULE_PATH = CAPSULE_ROOT / "capsule.json"
CAPSULE_INPUTS_PATH = CAPSULE_ROOT / "candidate-inputs.npz"
CAPSULE_AUTHORITY_PATH = Path(
    "/home/gianl/gate-d-runs/"
    "greenfield_gate_d_compensated_capsule_20260831T124838Z-execution-authority.json"
)
LEDGER_SHA256 = "1b6da67c43371ab9dfe043164a708bcefd11e5e21873af23f631305d7e4fe3f0"
RECEIPT_SHA256 = "50b855f911c407fa6dfc2294c6ade740063841acc0e77cbeada4481972b0566a"
EXPECTED_CLASSIFICATION = (
    "BOUNDED_TPU_NUMERICAL_REJECTED;FORCED_ROUND_MECHANISM_REJECTED;"
    "GATE_D_OPEN;NO_PERFORMANCE_CLAIM"
)
FINAL_CLASSIFICATION = (
    "BOUNDED_TPU_NUMERICAL_REJECTED;SUCCESS_PUBLICATION_POLICY_FAILURE;"
    "DIAGNOSTIC_ARCHIVE_AUTHENTICATED;PROJECTION_FRONTIER_LOCALIZED;"
    "GATE_D_OPEN"
)
PAYLOAD_MEMBERS = {
    "census_post.txt",
    "census_pre.txt",
    "dependencies.json",
    "failure_status.json",
    "hlo/acquired_preimage.optimized_hlo.txt",
    "hlo/acquired_preimage.stablehlo.mlir",
    "hlo/forced_round_pp16_stage0.optimized_hlo.txt",
    "hlo/forced_round_pp16_stage0.stablehlo.mlir",
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
LOCAL_EXTRA_MEMBERS = {
    "diagnostic_objects.json",
    "diagnostic_upload_receipt.json",
    "orchestrator.log",
}
_HASH = re.compile(r"[0-9a-f]{64}")
_GCLOUD_ENV = {
    "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/snap/bin:/usr/bin:/bin",
    "PYTHONWARNINGS": "ignore",
}
_GIT_ENV = {
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


class DiagnosticValidationError(RuntimeError):
    """Raised when the burned-run evidence is not independently authoritative."""


def _canonical_json(value: Any) -> bytes:
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


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _crc32c(raw: bytes) -> str:
    return base64.b64encode(google_crc32c.Checksum(raw).digest()).decode("ascii")


def _snapshot_regular(path: Path, *, limit: int = 256 << 20) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise DiagnosticValidationError(f"unsafe diagnostic member: {path}")
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
            raise DiagnosticValidationError(f"diagnostic member changed: {path}")
        return raw
    finally:
        os.close(descriptor)


def _run(
    command: Sequence[str], *, environment: Mapping[str, str], timeout: int = 180
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            env=dict(environment),
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DiagnosticValidationError(
            f"read-only command failed to execute: {command[0]}"
        ) from error


def _git(*arguments: str) -> bytes:
    result = _run(
        ["/usr/bin/git", "-C", str(WORKTREE), *arguments],
        environment=_GIT_ENV,
    )
    if result.returncode:
        raise DiagnosticValidationError("local-only Git identity check failed")
    return result.stdout


def _load_committed_module(
    relative: str, pin: str, name: str, *, require_live: bool = True
) -> types.ModuleType:
    committed = _git("show", f"{pin}:{relative}")
    if require_live and committed != _snapshot_regular(WORKTREE / relative, limit=4 << 20):
        raise DiagnosticValidationError(f"committed module drifted: {relative}")
    module = types.ModuleType(name)
    module.__file__ = str(WORKTREE / relative)
    module.__package__ = None
    exec(compile(committed, str(WORKTREE / relative), "exec"), module.__dict__)  # noqa: S102
    return module


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    raw = _snapshot_regular(Path(__file__), limit=4 << 20)
    head = _git("rev-parse", "HEAD").decode("ascii").strip()
    resolved = _git("rev-parse", f"{code_pin}^{{commit}}").decode("ascii").strip()
    if (
        head != code_pin
        or resolved != code_pin
        or raw != _git("show", f"{code_pin}:{SOURCE_PATH}")
        or _sha256(raw) != expected_sha256
    ):
        raise DiagnosticValidationError("diagnostic adjudicator is not committed")


def _validate_local_catalogue(run_dir: Path) -> None:
    observed: set[str] = set()
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_symlink():
            raise DiagnosticValidationError("diagnostic contains a symlink")
        if path.is_file():
            observed.add(relative)
        elif not path.is_dir():
            raise DiagnosticValidationError("diagnostic contains a special file")
    if observed != PAYLOAD_MEMBERS | LOCAL_EXTRA_MEMBERS:
        raise DiagnosticValidationError("local diagnostic catalogue drifted")


def _validate_ledger(run_dir: Path) -> tuple[dict[str, dict[str, Any]], bytes]:
    raw = _snapshot_regular(run_dir / "diagnostic_objects.json", limit=1 << 20)
    if _sha256(raw) != LEDGER_SHA256:
        raise DiagnosticValidationError("diagnostic ledger bytes drifted")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiagnosticValidationError("diagnostic ledger is not JSON") from error
    objects = value.get("objects") if isinstance(value, Mapping) else None
    if (
        _canonical_json(value) != raw
        or set(value) != {"artifact_kind", "objects", "run_tag"}
        or value.get("artifact_kind")
        != "gate_d_forced_round_pp16_numerical_diagnostic_ledger"
        or value.get("run_tag") != RUN_TAG
        or not isinstance(objects, list)
        or len(objects) != len(PAYLOAD_MEMBERS)
    ):
        raise DiagnosticValidationError("diagnostic ledger schema drifted")
    records: dict[str, dict[str, Any]] = {}
    for record in objects:
        if not isinstance(record, Mapping) or set(record) != {
            "crc32c",
            "generation",
            "path",
            "sha256",
            "size",
        }:
            raise DiagnosticValidationError("diagnostic ledger record drifted")
        relative = record.get("path")
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != relative
            or relative in records
            or type(record.get("generation")) is not str
            or not record["generation"].isdecimal()
            or type(record.get("size")) is not int
            or record["size"] < 0
            or type(record.get("crc32c")) is not str
            or type(record.get("sha256")) is not str
            or _HASH.fullmatch(record["sha256"]) is None
        ):
            raise DiagnosticValidationError("diagnostic ledger identity drifted")
        records[relative] = dict(record)
    if set(records) != PAYLOAD_MEMBERS:
        raise DiagnosticValidationError("diagnostic payload catalogue drifted")
    return records, raw


def _validate_local_members(
    run_dir: Path, records: Mapping[str, Mapping[str, Any]]
) -> dict[str, bytes]:
    snapshots: dict[str, bytes] = {}
    for relative, record in records.items():
        raw = _snapshot_regular(run_dir / relative)
        if (
            len(raw) != record["size"]
            or _sha256(raw) != record["sha256"]
            or _crc32c(raw) != record["crc32c"]
        ):
            raise DiagnosticValidationError(
                f"local diagnostic bytes drifted: {relative}"
            )
        snapshots[relative] = raw
    if _snapshot_regular(run_dir / "orchestrator.log", limit=1 << 20) != snapshots[
        "orchestrator.failure.log"
    ]:
        raise DiagnosticValidationError("orchestrator copies disagree")
    return snapshots


def _validate_receipt(run_dir: Path, ledger_raw: bytes) -> dict[str, Any]:
    raw = _snapshot_regular(
        run_dir / "diagnostic_upload_receipt.json", limit=1 << 20
    )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiagnosticValidationError("diagnostic receipt is not JSON") from error
    terminal = value.get("terminal") if isinstance(value, Mapping) else None
    if (
        _sha256(raw) != RECEIPT_SHA256
        or _canonical_json(value) != raw
        or set(value) != {"artifact_kind", "remote", "terminal"}
        or value.get("artifact_kind")
        != "gate_d_forced_round_pp16_numerical_diagnostic_receipt"
        or value.get("remote") != f"{REMOTE}/diagnostic_objects.json"
        or not isinstance(terminal, Mapping)
        or set(terminal) != {"crc32c", "generation", "path", "sha256", "size"}
        or terminal.get("path") != "diagnostic_objects.json"
        or terminal.get("sha256") != LEDGER_SHA256
        or terminal.get("size") != len(ledger_raw)
        or terminal.get("crc32c") != _crc32c(ledger_raw)
        or type(terminal.get("generation")) is not str
        or not terminal["generation"].isdecimal()
    ):
        raise DiagnosticValidationError("diagnostic receipt drifted")
    return dict(terminal)


def _strict_json(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiagnosticValidationError(f"{label} is not JSON") from error
    if not isinstance(value, Mapping) or _canonical_json(value) != raw:
        raise DiagnosticValidationError(f"{label} is not canonical JSON")
    return value


def _load_output_arrays(
    publisher: Any, raw: bytes, records: Mapping[str, Any], np: Any
) -> dict[str, Any]:
    publisher._validate_output_archive(raw, records)
    result: dict[str, Any] = {}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        for name, (shape, _semantic, storage) in publisher.EXPECTED_OUTPUT_SPEC.items():
            member = archive.read(f"{name}.npy")
            payload = publisher._npy_payload(member, shape, storage)
            dtype = np.dtype(storage)
            result[name] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    return result


def _validate_publication_runtime_authority(
    publisher: Any, primitives: Any, observed: Mapping[str, Any]
) -> None:
    if primitives._sha_file(primitives.PYTHON) != primitives.PYTHON_SHA256:
        raise DiagnosticValidationError("publisher interpreter bytes drifted")
    if (
        primitives._tree_sha256(
            primitives.PYTHON_RUNTIME_ROOT, require_sealed=True
        )
        != primitives.PYTHON_RUNTIME_TREE_SHA256
    ):
        raise DiagnosticValidationError("publisher interpreter tree drifted")
    manifest = primitives._validate_storage_site(
        primitives.STORAGE_SITE_ROOT,
        expected_tree_sha256=primitives.STORAGE_SITE_TREE_SHA256,
        expected_manifest_sha256=primitives.STORAGE_SITE_MANIFEST_SHA256,
        require_sealed=True,
    )
    expected = {
        "artifact_kind": "gate_d_forced_round_pp16_numerical_publisher_runtime",
        "dependency_allowlist_count": len(manifest["allowlist"]),
        "dependency_builder_source_path": (
            primitives.STORAGE_SITE_BUILDER_SOURCE_PATH
        ),
        "dependency_builder_source_sha256": (
            primitives.STORAGE_SITE_BUILDER_SOURCE_SHA256
        ),
        "dependency_manifest_sha256": primitives.STORAGE_SITE_MANIFEST_SHA256,
        "dependency_root": str(primitives.STORAGE_SITE_ROOT),
        "dependency_tree_sha256": primitives.STORAGE_SITE_TREE_SHA256,
        "publication_primitive_path": publisher.PRIMITIVE_PATH,
        "publication_primitive_sha256": publisher.PRIMITIVE_SHA256,
        "python_executable": str(primitives.PYTHON),
        "python_runtime_root": str(primitives.PYTHON_RUNTIME_ROOT),
        "python_runtime_tree_sha256": primitives.PYTHON_RUNTIME_TREE_SHA256,
        "python_sha256": primitives.PYTHON_SHA256,
    }
    if observed != expected:
        raise DiagnosticValidationError("publisher runtime authority drifted")


def _validate_dependencies_with_exact_helper(
    publisher: Any,
    primitives: Any,
    runner: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    dependencies_raw: bytes,
) -> None:
    original = primitives._validate_dependency_records

    def validate(
        records: Any, category: str, *, verify_live: bool
    ) -> None:
        if category != "python_modules":
            original(records, category, verify_live=verify_live)
            return
        if not isinstance(records, list) or not records:
            raise DiagnosticValidationError("Python dependencies are absent")
        helper_path = str(FORCED_HLO_HELPER_PATH)
        matches = [
            (index, record)
            for index, record in enumerate(records)
            if isinstance(record, Mapping) and record.get("path") == helper_path
        ]
        if len(matches) != 1:
            raise DiagnosticValidationError("forced-HLO helper count drifted")
        helper_index, helper = matches[0]
        if set(helper) != {"bytes", "device", "inode", "path", "sha256"}:
            raise DiagnosticValidationError("forced-HLO helper schema drifted")
        raw = _snapshot_regular(FORCED_HLO_HELPER_PATH, limit=4 << 20)
        if (
            _sha256(raw) != FORCED_HLO_HELPER_SHA256
            or helper.get("bytes") != len(raw)
            or helper.get("sha256") != FORCED_HLO_HELPER_SHA256
            or raw
            != _git("show", f"{RUN_PIN}:{FORCED_HLO_HELPER_SOURCE_PATH}")
            or raw
            != _git(
                "show",
                f"{HLO_ACQUISITION_PIN}:{FORCED_HLO_HELPER_SOURCE_PATH}",
            )
        ):
            raise DiagnosticValidationError("forced-HLO helper bytes drifted")
        primitives._verify_dependency_record_live(helper)
        original(
            [record for index, record in enumerate(records) if index != helper_index],
            category,
            verify_live=verify_live,
        )

    primitives._validate_dependency_records = validate
    try:
        publisher._validate_dependencies(
            primitives, runner, dependencies, dependencies_raw, RUN_PIN
        )
    finally:
        primitives._validate_dependency_records = original


def _validate_scientific_payload(
    snapshots: Mapping[str, bytes], publisher: Any, driver: Any
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    import ml_dtypes
    import numpy as np

    runner = _strict_json(snapshots["runner.json"], "runner")
    dependencies = _strict_json(snapshots["dependencies.json"], "dependencies")
    failure = _strict_json(snapshots["failure_status.json"], "failure status")
    primitives = publisher._load_primitives(RUN_PIN)
    publication_runtime = _strict_json(
        snapshots["publisher_runtime.json"], "publisher runtime"
    )
    _validate_publication_runtime_authority(
        publisher, primitives, publication_runtime
    )
    if publisher._validate_runner(runner, RUN_PIN) != "NUMERICAL_REJECTED":
        raise DiagnosticValidationError("runner is not the bounded rejection")
    _validate_dependencies_with_exact_helper(
        publisher, primitives, runner, dependencies, snapshots["dependencies.json"]
    )
    if failure != {
        "artifact_kind": "gate_d_forced_round_pp16_numerical_failure",
        "code_hash": RUN_PIN,
        "exit_status": 1,
        "gate_d_closed": False,
        "performance_claim": False,
        "run_tag": RUN_TAG,
        "terminal_payload_present": False,
    }:
        raise DiagnosticValidationError("bounded publication failure status drifted")
    output_raw = snapshots["outputs.npz"]
    if runner.get("output_artifact") != {
        "byte_count": len(output_raw),
        "filename": "outputs.npz",
        "sha256": _sha256(output_raw),
    }:
        raise DiagnosticValidationError("output archive identity drifted")
    host = _load_output_arrays(publisher, output_raw, runner["output_arrays"], np)
    capsule = driver._load_bound_json(
        CAPSULE_PATH, driver.CAPSULE_SHA256, "capsule"
    )
    authority = driver._load_bound_json(
        CAPSULE_AUTHORITY_PATH,
        driver.CAPSULE_EXECUTION_AUTHORITY_SHA256,
        "capsule execution authority",
    )
    input_raw = _snapshot_regular(CAPSULE_INPUTS_PATH)
    if _sha256(input_raw) != driver.CAPSULE_INPUT_SHA256:
        raise DiagnosticValidationError("capsule inputs drifted")
    inputs = driver._load_npz_arrays(input_raw, authority["input_arrays"], np)
    rms_input = inputs["rms_hidden_update_bf16_bits"].view(
        ml_dtypes.bfloat16
    ).astype(np.float32) + inputs["rms_residual_bf16_bits"].view(
        ml_dtypes.bfloat16
    ).astype(np.float32)
    independently_classified = driver.classify_outputs(
        host, capsule, inputs, rms_input, np
    )
    if independently_classified != runner.get("numerical"):
        raise DiagnosticValidationError("independent numerical classification drifted")
    watchpoints = independently_classified["candidate_watchpoint_matches"]
    exact_prefix = {
        "layer1.rms_operands_bf16:hidden_update",
        "layer1.rms_operands_bf16:residual",
        "layer1.rms_input_fp32:value",
        "layer1.normalized:owner0",
        "layer1.normalized:owner1",
    }
    divergent_projection = {
        "layer1.query:owner0",
        "layer1.query:owner1",
        "layer1.current_key:value",
    }
    if (
        runner.get("classification") != EXPECTED_CLASSIFICATION
        or any(watchpoints.get(label) is not True for label in exact_prefix)
        or any(watchpoints.get(label) is not False for label in divergent_projection)
    ):
        raise DiagnosticValidationError("projection frontier is not localized")
    for kind, relative in (
        (
            publisher.EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256,
            "hlo/acquired_preimage.optimized_hlo.txt",
        ),
        (
            publisher.EXPECTED_STABLEHLO_SHA256,
            "hlo/acquired_preimage.stablehlo.mlir",
        ),
        (
            publisher.EXPECTED_OPTIMIZED_HLO_SHA256,
            "hlo/forced_round_pp16_stage0.optimized_hlo.txt",
        ),
        (
            publisher.EXPECTED_STABLEHLO_SHA256,
            "hlo/forced_round_pp16_stage0.stablehlo.mlir",
        ),
    ):
        if _sha256(snapshots[relative]) != kind:
            raise DiagnosticValidationError(f"HLO bytes drifted: {relative}")
    for name in ("census_pre.txt", "census_post.txt"):
        raw = snapshots[name]
        primitives._require_census(raw, "CENSUS_OK")
        if b"CENSUS_BAD" in raw or b"CENSUS_BUSY" in raw:
            raise DiagnosticValidationError(f"zero-work census failed: {name}")
    if snapshots["runner.log"] != (
        b"GATE_D_FORCED_ROUND_PP16_NUMERICAL_REJECTED execution_count=1 "
        b"accepted_event=false gate_d_open=true\n"
    ):
        raise DiagnosticValidationError("runner log drifted")
    publisher._validate_remote_vacancy_evidence(
        snapshots["remote_vacancy.raw.txt"],
        snapshots["remote_vacancy.txt"],
        BASE_REMOTE,
    )
    mirror = _strict_json(snapshots["mirror.sha256"], "same-region mirror replay")
    if (
        mirror.get("artifact_kind") != "gate_d_same_region_git_mirror_replay"
        or mirror.get("commit") != RUN_PIN
        or mirror.get("branch") != "tooling/gate-d-compensated-pp16-numerical"
        or mirror.get("mirror_uri") != "gs://driftbench-dsv4-uc/repos/glm-tpu/.git"
        or mirror.get("origin") != "git@github.com:GianluigiVitale/glm-tpu.git"
        or mirror.get("origin_remote_exact_record") is not True
        or mirror.get("commit_connectivity_fsck") is not True
        or mirror.get("checkout_archive_sha256")
        != "eea9b8250a7f95e896f0f30a26c5624402ee854386bb93675d10b45244083086"
    ):
        raise DiagnosticValidationError("same-region mirror authority drifted")
    if snapshots["sync.txt"] != (
        b"SYNC_OK t1v-n-ae271d05-w-0 "
        + RUN_PIN.encode("ascii")
        + b" origin_and_same_region_mirror\n"
    ):
        raise DiagnosticValidationError("run sync authority drifted")
    return runner, independently_classified


def _parse_listing(
    raw: bytes, expected_prefixes: set[str]
) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiagnosticValidationError("remote listing is not JSON") from error
    if not isinstance(value, list):
        raise DiagnosticValidationError("remote listing schema drifted")
    prefixes: set[str] = set()
    objects: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise DiagnosticValidationError("remote listing record drifted")
        if item.get("type") == "prefix":
            if (
                set(item) != {"type", "url"}
                or item.get("url") not in expected_prefixes
                or item["url"] in prefixes
            ):
                raise DiagnosticValidationError("remote prefix record drifted")
            prefixes.add(item["url"])
        elif item.get("type") == "cloud_object":
            if (
                set(item) != {"type", "url", "metadata"}
                or not isinstance(item.get("url"), str)
                or not isinstance(item.get("metadata"), Mapping)
            ):
                raise DiagnosticValidationError("remote object record drifted")
            objects.append(dict(item))
        else:
            raise DiagnosticValidationError("unknown remote listing record")
    if prefixes != (expected_prefixes if objects else set()):
        raise DiagnosticValidationError("remote prefix/object relationship drifted")
    return objects


def _list_remote(*, soft_deleted: bool) -> list[dict[str, Any]]:
    command = [
        "/snap/bin/gcloud",
        "storage",
        "ls",
        "--json",
        "--recursive",
        "--soft-deleted" if soft_deleted else "--all-versions",
    ]
    if soft_deleted:
        command.append("--exhaustive")
    command.append(REMOTE.rstrip("/") + "/")
    result = _run(command, environment=_GCLOUD_ENV)
    if result.returncode:
        if (
            soft_deleted
            and result.stdout == b""
            and result.stderr
            == b"ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n"
        ):
            return []
        raise DiagnosticValidationError("remote history listing failed")
    prefix = REMOTE.rstrip("/") + "/"
    return _parse_listing(result.stdout, {prefix, prefix + "hlo/"})


def _cat_generation(url: str) -> bytes:
    result = _run(
        ["/snap/bin/gcloud", "storage", "cat", url], environment=_GCLOUD_ENV
    )
    if result.returncode:
        raise DiagnosticValidationError("generation-qualified replay failed")
    return result.stdout


def _validate_remote(
    records: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
    snapshots: Mapping[str, bytes],
    ledger_raw: bytes,
) -> list[dict[str, Any]]:
    objects = _list_remote(soft_deleted=False)
    if _list_remote(soft_deleted=True):
        raise DiagnosticValidationError("diagnostic has soft-deleted ambiguity")
    expected: dict[str, tuple[Mapping[str, Any], bytes]] = {
        path: (record, snapshots[path]) for path, record in records.items()
    }
    expected["diagnostic_objects.json"] = (terminal, ledger_raw)
    if len(objects) != len(expected):
        raise DiagnosticValidationError("remote object count drifted")
    prefix = REMOTE.rstrip("/") + "/"
    seen: set[str] = set()
    manifest: list[dict[str, Any]] = []
    for item in objects:
        metadata = item["metadata"]
        generation = str(metadata.get("generation", ""))
        suffix = f"#{generation}"
        url = item["url"]
        if not generation.isdecimal() or not url.startswith(prefix) or not url.endswith(
            suffix
        ):
            raise DiagnosticValidationError("remote generation URL drifted")
        relative = url[len(prefix) : -len(suffix)]
        if relative in seen or relative not in expected:
            raise DiagnosticValidationError("remote catalogue drifted")
        wanted, local_raw = expected[relative]
        remote_raw = _cat_generation(url)
        metadata_size = metadata.get("size")
        if (
            type(metadata_size) not in {int, str}
            or not str(metadata_size).isdecimal()
        ):
            raise DiagnosticValidationError("remote size metadata drifted")
        if (
            remote_raw != local_raw
            or generation != wanted["generation"]
            or str(metadata.get("crc32c", "")) != wanted["crc32c"]
            or int(metadata_size) != wanted["size"]
            or _crc32c(remote_raw) != wanted["crc32c"]
            or _sha256(remote_raw) != wanted["sha256"]
        ):
            raise DiagnosticValidationError(f"remote bytes drifted: {relative}")
        manifest.append(
            {
                "crc32c": wanted["crc32c"],
                "generation": generation,
                "path": relative,
                "sha256": wanted["sha256"],
                "size": wanted["size"],
            }
        )
        seen.add(relative)
    if seen != set(expected):
        raise DiagnosticValidationError("remote catalogue is incomplete")
    terminal_generation = int(terminal["generation"])
    if terminal_generation <= max(
        int(record["generation"]) for record in records.values()
    ):
        raise DiagnosticValidationError("diagnostic terminal was not uploaded last")
    return sorted(manifest, key=lambda value: value["path"])


def adjudicate(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    _validate_local_catalogue(run_dir)
    records, ledger_raw = _validate_ledger(run_dir)
    snapshots = _validate_local_members(run_dir, records)
    terminal = _validate_receipt(run_dir, ledger_raw)
    publisher = _load_committed_module(
        PUBLISHER_PATH, _git("rev-parse", "HEAD").decode("ascii").strip(), "_publisher"
    )
    driver = _load_committed_module(DRIVER_PATH, RUN_PIN, "_historical_driver")
    runner, numerical = _validate_scientific_payload(snapshots, publisher, driver)
    remote_manifest = _validate_remote(records, terminal, snapshots, ledger_raw)
    return {
        "artifact_kind": (
            "gate_d_forced_round_pp16_numerical_diagnostic_adjudication"
        ),
        "claim_scope": (
            "Read-only authentication and independent CPU reclassification of one "
            "already-completed bounded TPU discriminator. No TPU rerun, decoder, "
            "token, performance, or Gate-D closure claim."
        ),
        "classification": FINAL_CLASSIFICATION,
        "compiled_executable_invocation_count": 1,
        "diagnostic_ledger_sha256": LEDGER_SHA256,
        "exact_remote_object_count": len(remote_manifest),
        "first_observed_divergence": "layer1.query projection output",
        "forced_round_mechanism_accepted": False,
        "gate_d_closed": False,
        "host_transfer_count": 1,
        "hlo": runner["hlo"],
        "independent_numerical_classification": numerical,
        "performance_claim": False,
        "publication_failure_root_cause": (
            "The success publisher omitted one exact committed forced-HLO helper "
            "from its caller-specific Python dependency policy."
        ),
        "remote": REMOTE,
        "remote_manifest": remote_manifest,
        "run_code_hash": RUN_PIN,
        "run_tag": RUN_TAG,
        "soft_deleted_object_count": 0,
        "terminal_generation": terminal["generation"],
        "tpu_numerical_execution_performed": True,
        "tpu_rerun_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _verify_running_source(args.expected_code_hash, args.expected_source_sha256)
    report = adjudicate()
    report["adjudicator_code_hash"] = args.expected_code_hash
    report["adjudicator_source_sha256"] = args.expected_source_sha256
    print(_canonical_json(report).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
