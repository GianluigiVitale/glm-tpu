#!/usr/bin/env python3
"""Append-only publisher for the exact-M2048 StrategyND discriminator."""

from __future__ import annotations

import argparse
import base64
import fcntl
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Mapping
import zipfile


REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
INSTALL_PATH = Path(
    "/usr/local/libexec/glm-tpu/gate-d-m2048-strategy-nd-v3/"
    "publish_gate_d_m2048_strategy_nd_association.py"
)
SOURCE_PATH = (
    "scripts/greenfield/publish_gate_d_m2048_strategy_nd_association.py"
)
DRIVER_SOURCE_PATH = (
    "scripts/greenfield/probe_m2048_strategy_nd_association.py"
)
BASE_PATH = "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
BASE_PIN = "986378238ac6458307aea69ef1f5e12bf82bc020"
BASE_SHA256 = "f3f20a01fd37bb82988cd77f69fa7b0a780d120568bab0db4162f42e7855bc97"
BUCKET_NAME = "driftbench-dsv4-uc"
BUCKET_LOCATION = "US-CENTRAL2"
REMOTE_ROOT = "results/greenfield/glm52/m2048_strategy_nd/"
TAG_PATTERN = re.compile(
    r"greenfield_m2048_strategy_nd_[0-9]{8}T[0-9]{15}Z"
)
EXPECTED_TOPOLOGY_HASH = (
    "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
)
EXPECTED_LAUNCH_TO_JAX_PROCESS = (3, 5, 1, 2, 0, 6, 7, 4)
JAX_SITE_ROOT = "/opt/glm-tpu/gate-d-jax-site-55233c63939e"
LIBTPU_SITE_ROOT = "/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3"
PYTHON_RUNTIME_ROOT = "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95"
EXPECTED_RUNTIME_PATH = (
    f"{PYTHON_RUNTIME_ROOT}/lib/python312.zip",
    f"{PYTHON_RUNTIME_ROOT}/lib/python3.12",
    f"{PYTHON_RUNTIME_ROOT}/lib/python3.12/lib-dynload",
)
EXPECTED_PYTHON_PROVENANCE = {
    "python_executable": f"{PYTHON_RUNTIME_ROOT}/bin/python3.12",
    "python_sha256": "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7",
    "python_runtime_tree_sha256": "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616",
}
EXPECTED_SITE_PROVENANCE = {
    "jax": {
        "root": JAX_SITE_ROOT,
        "tree_sha256": "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df",
        "manifest_sha256": "ef454caafd2e4ba5da4bc7f7f73bef4e8f157afd6c795a91b09e319961941eff",
    },
    "libtpu": {
        "root": LIBTPU_SITE_ROOT,
        "tree_sha256": "db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca",
        "manifest_sha256": "d34064f4a0dfcdcd9ec13288ce967ae060a4650b3e86e53bc47e49756c0e97fa",
    },
}
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
WRAPPER_WRITE_MEMBERS = {
    "census_failure_exit.txt",
    "census_post.txt",
    "census_pre.txt",
    "distributed.raw.log",
    "mirror.sha256",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.log",
    "sync.txt",
}
SUCCESS_PAYLOAD = {
    "analysis.json",
    "census_post.txt",
    "census_pre.txt",
    "distributed.raw.log",
    "evidence.json",
    "fleet_records.json",
    "input_row0_bits.npy",
    "m2048.optimized_hlo.txt",
    "m2048.stablehlo.mlir",
    "mirror.sha256",
    "orchestrator.sealed.log",
    "output_row0_bits.npy",
    "publisher_runtime.json",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.log",
    "summary.json",
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
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError(f"publisher source changed: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_root_boundary(path: Path) -> None:
    for parent in (path.parent, *path.parent.parents):
        metadata = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError(f"publisher installation parent is mutable: {parent}")


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    _require_root_boundary(INSTALL_PATH)
    raw = _snapshot(INSTALL_PATH)
    metadata = os.stat(INSTALL_PATH, follow_symlinks=False)
    if (
        Path(__file__) != INSTALL_PATH
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or metadata.st_nlink != 1
        or os.listxattr(INSTALL_PATH, follow_symlinks=False)
        or _git_bytes("rev-parse", "HEAD").decode().strip() != code_pin
        or _git_bytes("status", "--porcelain=v1", "--untracked-files=all")
        or _git_bytes("for-each-ref", "--format=%(refname)", "refs/replace")
        or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("M2048 publisher source/repository identity drifted")


def _load_base(code_pin: str) -> types.ModuleType:
    path = REPO / BASE_PATH
    raw = _snapshot(path)
    if (
        raw != _git_bytes("show", f"{code_pin}:{BASE_PATH}")
        or raw != _git_bytes("show", f"{BASE_PIN}:{BASE_PATH}")
        or sha256(raw).hexdigest() != BASE_SHA256
    ):
        raise RuntimeError("M2048 publisher base bytes drifted")
    module = types.ModuleType("_gate_d_m2048_publisher_base")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    module.REPO = REPO
    module.RUN_ROOT = RUN_ROOT
    module.BUCKET_NAME = BUCKET_NAME
    module.REMOTE_ROOT = REMOTE_ROOT
    module.TAG_PATTERN = TAG_PATTERN
    module._WRAPPER_WRITE_MEMBERS = set(WRAPPER_WRITE_MEMBERS)
    module._EXPECTED_ENVIRONMENT = {
        "HOME": "/home/gianl",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    module._git_bytes = lambda *args: _git_bytes(*args)
    module._validate_remote_vacancy_evidence = (
        _validate_zero_retention_vacancy_evidence
    )
    module._require_never_used_prefix = lambda bucket, prefix: (
        _require_zero_retention_unused_prefix(module, bucket, prefix)
    )
    return module


def _require_zero_retention_policy(bucket: Any) -> None:
    bucket.reload()
    if (
        bucket.name != BUCKET_NAME
        or bucket.location != BUCKET_LOCATION
        or bucket.soft_delete_policy.retention_duration_seconds != 0
    ):
        raise RuntimeError("M2048 publication bucket zero-retention policy drifted")


def _require_zero_retention_unused_prefix(
    base: Any, bucket: Any, prefix: str
) -> None:
    _require_zero_retention_policy(bucket)
    observations = {
        "live": base._observed_names(bucket, prefix),
        "all_versions": base._observed_names(bucket, prefix, versions=True),
    }
    occupied = {scope: names for scope, names in observations.items() if names}
    if occupied:
        raise RuntimeError(
            "M2048 zero-retention prefix has prior live/versioned history: "
            + ",".join(sorted(occupied))
        )


def _validate_zero_retention_vacancy_evidence(
    raw: bytes, summary: bytes, remote: str
) -> None:
    no_objects = "ERROR: (gcloud.storage.ls) One or more URLs matched no objects."
    policy_disabled = (
        "ERROR: (gcloud.storage.ls) HTTPError 400: Soft delete policy is required "
        "to list soft-deleted versions"
    )
    expected_raw = (
        "soft_delete_retention_seconds=0\n"
        "scope=live flags=none returncode=1\n"
        f"{no_objects}\n"
        "scope=all_versions flags=--all-versions returncode=1\n"
        f"{no_objects}\n"
        "scope=soft_deleted flags=--soft-deleted,--exhaustive returncode=1 "
        "availability=policy_disabled\n"
        f"{policy_disabled}\n"
    ).encode("ascii")
    expected_summary = (
        "SOFT_DELETE_RETENTION_SECONDS 0\n"
        f"VACANT live {remote}\n"
        f"VACANT all_versions {remote}\n"
        f"UNAVAILABLE_POLICY_DISABLED soft_deleted {remote}\n"
    ).encode("ascii")
    if raw != expected_raw or summary != expected_summary:
        raise RuntimeError("M2048 zero-retention vacancy evidence drifted")


def _sealed_source_archive(code_pin: str) -> tuple[str, Mapping[str, Any]]:
    tree = _git_bytes("ls-tree", "-r", "-z", code_pin, "--", "glm_tpu")
    buffer = BytesIO()
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for raw_entry in tree.split(b"\0"):
            if not raw_entry:
                continue
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8", errors="strict")
            if (
                kind != b"blob"
                or mode not in {b"100644", b"100755"}
                or not path.startswith("glm_tpu/")
            ):
                raise RuntimeError(f"unsupported committed source entry: {path}")
            payload = _git_bytes(
                "cat-file", "blob", object_id.decode("ascii")
            )
            records.append(
                {
                    "git_object_id": object_id.decode("ascii"),
                    "path": path,
                    "sha256": sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (
                0o100555 if mode == b"100755" else 0o100444
            ) << 16
            archive.writestr(info, payload)
    if not records or records != sorted(records, key=lambda item: item["path"]):
        raise RuntimeError("committed source archive order drifted")
    raw = buffer.getvalue()
    descriptor = os.memfd_create(
        "gate-d-m2048-publisher-source.zip",
        os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC,
    )
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("publisher source archive write stalled")
        view = view[written:]
    fcntl.fcntl(descriptor, _F_ADD_SEALS, _MEMFD_SEALS)
    if (
        fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEALS
        or os.pread(descriptor, len(raw) + 1, 0) != raw
    ):
        raise RuntimeError("publisher source archive seal/replay drifted")
    manifest = _canonical(records)
    return f"/proc/self/fd/{descriptor}", {
        "archive_sha256": sha256(raw).hexdigest(),
        "file_manifest_count": len(records),
        "file_manifest_sha256": sha256(manifest).hexdigest(),
    }


def _load_project_api(code_pin: str) -> Mapping[str, Any]:
    archive, archive_identity = _sealed_source_archive(code_pin)
    original_path = list(sys.path)
    if any(name == "glm_tpu" or name.startswith("glm_tpu.") for name in sys.modules):
        raise RuntimeError("project module was loaded before sealed source installation")
    try:
        sys.path[:] = [
            archive,
            JAX_SITE_ROOT,
            LIBTPU_SITE_ROOT,
            *EXPECTED_RUNTIME_PATH,
        ]
        from glm_tpu.greenfield.benchmarking import (
            M2048StrategyNdFingerprintConfig,
            analyze_m2048_row0_association,
            array_sha256,
            generate_m2048_row0_input_bits,
            m2048_source_identity,
            validate_m2048_strategy_nd_fingerprint_hlo,
        )
        from glm_tpu.greenfield.topology import validate_target_v4_64
        from glm_tpu.greenfield.types import PhysicalTopology
        escaped = {
            name: getattr(module, "__file__", None)
            for name, module in sys.modules.items()
            if (name == "glm_tpu" or name.startswith("glm_tpu."))
            and not str(getattr(module, "__file__", "")).startswith(archive + "/")
        }
        if escaped:
            raise RuntimeError(f"project import escaped sealed source: {escaped}")
    finally:
        sys.path[:] = original_path
    return {
        "PhysicalTopology": PhysicalTopology,
        "analyze": analyze_m2048_row0_association,
        "array_sha256": array_sha256,
        "config": M2048StrategyNdFingerprintConfig,
        "generate": generate_m2048_row0_input_bits,
        "source_identity": m2048_source_identity,
        "source_archive_identity": archive_identity,
        "validate_hlo": validate_m2048_strategy_nd_fingerprint_hlo,
        "validate_topology": validate_target_v4_64,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _decode_json_b64(value: str, *, label: str) -> Mapping[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        raw = base64.b64decode(value, validate=True)
        parsed = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label} base64/JSON") from error
    if not isinstance(parsed, dict) or raw != _canonical(parsed):
        raise RuntimeError(f"noncanonical {label} JSON")
    return parsed


def _single_marker(raw: bytes, name: str) -> Mapping[str, Any]:
    text = raw.decode("utf-8", errors="strict")
    matches = re.findall(
        rf"(?m)^{name} process=0 json_b64=([A-Za-z0-9+/]+={{0,2}})$", text
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name} marker")
    return _decode_json_b64(matches[0], label=name)


def _record_markers(raw: bytes) -> list[Mapping[str, Any]]:
    text = raw.decode("utf-8", errors="strict")
    matches = re.findall(
        r"(?m)^M2048_RECORD process=([0-7]) "
        r"json_b64=([A-Za-z0-9+/]+={0,2})$",
        text,
    )
    if len(matches) != 8 or {int(rank) for rank, _ in matches} != set(range(8)):
        raise RuntimeError("M2048 stdout does not contain one record per process")
    records = [
        _decode_json_b64(payload, label=f"record {rank}")
        for rank, payload in sorted(matches, key=lambda item: int(item[0]))
    ]
    if any(
        record.get("launch_process_id") != index
        or record.get("jax_process_index")
        != EXPECTED_LAUNCH_TO_JAX_PROCESS[index]
        for index, record in enumerate(records)
    ):
        raise RuntimeError("M2048 record marker/process mapping drifted")
    return records


def _npy(value: Any) -> bytes:
    stream = BytesIO()
    import numpy as np

    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _require_census(raw: bytes, marker: bytes) -> None:
    lines = [line.split() for line in raw.splitlines() if line.startswith(marker)]
    if (
        len(lines) != 8
        or any(len(line) != 2 for line in lines)
        or len({line[1] for line in lines}) != 8
    ):
        raise RuntimeError(f"census does not prove eight unique {marker.decode()}")


def _normalized_provenance(
    value: Any,
    *,
    expected_probe_sha256: str,
    expected_source_archive: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "import_closure",
        "probe_sha256",
        "python",
        "sites",
        "source_archive",
        "sys_path",
    }:
        raise RuntimeError("M2048 provenance shape drifted")
    if (
        value["probe_sha256"] != expected_probe_sha256
        or value["python"] != EXPECTED_PYTHON_PROVENANCE
        or value["sites"] != EXPECTED_SITE_PROVENANCE
        or value["source_archive"] != expected_source_archive
    ):
        raise RuntimeError("M2048 sealed runtime provenance drifted")
    paths = value["sys_path"]
    if (
        not isinstance(paths, list)
        or len(paths) != 6
        or not isinstance(paths[0], str)
        or re.fullmatch(r"/proc/self/fd/[0-9]+", paths[0]) is None
        or paths[1:] != [
            JAX_SITE_ROOT,
            LIBTPU_SITE_ROOT,
            *EXPECTED_RUNTIME_PATH,
        ]
    ):
        raise RuntimeError("M2048 sealed sys.path drifted")
    closure = value["import_closure"]
    if not isinstance(closure, list) or not closure:
        raise RuntimeError("M2048 project import closure is empty")
    normalized_closure = []
    seen_modules: set[str] = set()
    for item in closure:
        if not isinstance(item, dict) or set(item) != {"module", "path"}:
            raise RuntimeError("M2048 import closure shape drifted")
        module = item["module"]
        path = item["path"]
        if (
            not isinstance(module, str)
            or (module != "glm_tpu" and not module.startswith("glm_tpu."))
            or module in seen_modules
            or not isinstance(path, str)
            or not path.startswith(paths[0] + "/glm_tpu/")
        ):
            raise RuntimeError("M2048 import closure escaped sealed source")
        seen_modules.add(module)
        normalized_closure.append(
            {"module": module, "path": "<SEALED_SOURCE>" + path[len(paths[0]) :]}
        )
    if normalized_closure != sorted(
        normalized_closure, key=lambda item: item["module"]
    ):
        raise RuntimeError("M2048 import closure order drifted")
    return {
        **value,
        "import_closure": normalized_closure,
        "sys_path": ["<SEALED_SOURCE>", *paths[1:]],
    }


def _prepare_success(
    base: Any,
    run_fd: int,
    *,
    code_pin: str,
    run_tag: str,
    remote: str,
    elapsed: int,
    api: Mapping[str, Any],
) -> dict[str, bytes]:
    import numpy as np

    raw_log = base.snapshot_member(run_fd, "distributed.raw.log", limit=32 << 20)
    acquisition = _single_marker(raw_log, "M2048_ACQUISITION")
    output_marker = _single_marker(raw_log, "M2048_OUTPUT")
    records = _record_markers(raw_log)
    if set(acquisition) != {
        "optimized_hlo_b64",
        "optimized_hlo_sha256",
        "stablehlo_b64",
        "stablehlo_sha256",
    } or set(output_marker) != {
        "output_bits_b64",
        "output_bits_sha256",
    }:
        raise RuntimeError("M2048 stdout marker schema drifted")
    optimized_hlo = base64.b64decode(
        acquisition["optimized_hlo_b64"], validate=True
    ).decode("utf-8", errors="strict")
    stablehlo = base64.b64decode(
        acquisition["stablehlo_b64"], validate=True
    ).decode("utf-8", errors="strict")
    if (
        sha256(optimized_hlo.encode()).hexdigest()
        != acquisition["optimized_hlo_sha256"]
        or sha256(stablehlo.encode()).hexdigest()
        != acquisition["stablehlo_sha256"]
    ):
        raise RuntimeError("M2048 acquisition HLO digest drifted")
    config = api["config"]()
    input_bits = api["generate"](config)
    output_raw = base64.b64decode(output_marker["output_bits_b64"], validate=True)
    if len(output_raw) != 32 * 6144 * 2:
        raise RuntimeError("M2048 output byte count drifted")
    output_bits = np.frombuffer(output_raw, dtype=np.uint16).reshape(32, 6144)
    input_sha = api["array_sha256"](input_bits)
    output_sha = api["array_sha256"](output_bits)
    if output_marker["output_bits_sha256"] != output_sha:
        raise RuntimeError("M2048 output marker hash drifted")
    topology_raw = _canonical(records[0]["topology"])
    topology = api["PhysicalTopology"].from_dict(records[0]["topology"])
    api["validate_topology"](topology)
    if topology.topology_hash != EXPECTED_TOPOLOGY_HASH:
        raise RuntimeError("M2048 topology differs from accepted physical authority")
    coordinates = {
        device.device_id: device.coordinates for device in topology.devices
    }
    analysis = api["analyze"](
        input_bits, output_bits, tuple(range(32)), coordinates
    )
    analysis_raw = _canonical(analysis)
    report, algorithm = api["validate_hlo"](optimized_hlo, tuple(range(32)))
    expected_probe_sha256 = sha256(
        _git_bytes("show", f"{code_pin}:{DRIVER_SOURCE_PATH}")
    ).hexdigest()
    normalized_provenance = [
        _normalized_provenance(
            record.get("provenance"),
            expected_probe_sha256=expected_probe_sha256,
            expected_source_archive=api["source_archive_identity"],
        )
        for record in records
    ]
    if any(
        value != normalized_provenance[0] for value in normalized_provenance[1:]
    ):
        raise RuntimeError("M2048 fleet normalized provenance differs")
    stable_fields = (
        "analysis_sha256",
        "capture",
        "code_hash",
        "collective_algorithm",
        "config",
        "diagnostic_only",
        "fleet_analysis_sha256s",
        "fleet_hlo_sha256s",
        "fleet_input_sha256s",
        "fleet_local_device_ids_in_runtime_order",
        "fleet_output_sha256s",
        "fleet_stablehlo_sha256s",
        "gate_d_closed",
        "hlo",
        "jax_version",
        "member_device_ids",
        "optimized_hlo_sha256",
        "performance_claim",
        "run_tag",
        "schema_version",
        "source_identity",
        "stablehlo_sha256",
        "topology",
        "topology_hash",
    )
    reference = records[0]
    expected_record_fields = set(stable_fields) | {
        "captured_utc",
        "hostname",
        "jax_process_index",
        "launch_process_id",
        "provenance",
    }
    if any(set(record) != expected_record_fields for record in records):
        raise RuntimeError("M2048 fleet record schema drifted")
    for field in stable_fields:
        if any(record[field] != reference[field] for record in records[1:]):
            raise RuntimeError(f"M2048 fleet field differs: {field}")
    hostname_matches = [
        re.fullmatch(r"([a-z0-9][a-z0-9-]*)-w-([0-7])", record["hostname"])
        if isinstance(record.get("hostname"), str)
        else None
        for record in records
    ]
    if (
        any(match is None for match in hostname_matches)
        or len({match.group(1) for match in hostname_matches if match}) != 1
        or any(
            int(match.group(2)) != index
            for index, match in enumerate(hostname_matches)
            if match
        )
        or any(record["code_hash"] != code_pin for record in records)
        or any(record["run_tag"] != run_tag for record in records)
        or any(_canonical(record["topology"]) != topology_raw for record in records)
        or reference["topology_hash"] != EXPECTED_TOPOLOGY_HASH
        or reference["config"] != config.to_dict()
        or reference["member_device_ids"] != list(range(32))
        or reference["source_identity"] != api["source_identity"]()
        or reference["jax_version"] != "0.10.1"
        or reference["optimized_hlo_sha256"]
        != acquisition["optimized_hlo_sha256"]
        or reference["stablehlo_sha256"] != acquisition["stablehlo_sha256"]
        or reference["analysis_sha256"] != sha256(analysis_raw).hexdigest()
        or reference["collective_algorithm"] != dict(algorithm)
        or reference["hlo"] != report.to_dict()
        or reference["diagnostic_only"] is not True
        or reference["gate_d_closed"] is not False
        or reference["performance_claim"] is not False
    ):
        raise RuntimeError("M2048 fleet record contract drifted")
    for field, expected in (
        ("fleet_hlo_sha256s", reference["optimized_hlo_sha256"]),
        ("fleet_stablehlo_sha256s", reference["stablehlo_sha256"]),
        ("fleet_input_sha256s", input_sha),
        ("fleet_output_sha256s", output_sha),
        ("fleet_analysis_sha256s", reference["analysis_sha256"]),
    ):
        if reference[field] != [expected] * 8:
            raise RuntimeError(f"M2048 fleet digest agreement drifted: {field}")
    fleet_ids = reference["fleet_local_device_ids_in_runtime_order"]
    if (
        len(fleet_ids) != 8
        or any(len(row) != 4 for row in fleet_ids)
        or sorted(item for row in fleet_ids for item in row) != list(range(32))
    ):
        raise RuntimeError("M2048 local device census drifted")
    capture = reference["capture"]
    expected_capture = {
        "collective_input_shape": [2048, 6144],
        "determinism_repeat_invocations": 32,
        "fleet_host_transfer_bytes": 25165824,
        "full_output_device_resident": True,
        "input_rows_1_through_2047_zero": True,
        "invocation_count": 64,
        "measured_trial_invocations": 32,
        "retained_output_artifact_bytes": 393216,
        "row0_slice_after_collective": True,
        "this_process_host_transfer_bytes": 3145728,
    }
    expected_capture_fields = set(expected_capture) | {
        "input_row0_bits_sha256",
        "local_replica_row0_sha256_by_trial",
        "output_row0_bits_sha256",
        "repeated_local_replica_row0_sha256_by_trial",
        "repeated_output_row0_bits_sha256",
    }
    if set(capture) != expected_capture_fields:
        raise RuntimeError("M2048 capture schema drifted")
    if any(capture.get(key) != value for key, value in expected_capture.items()):
        raise RuntimeError("M2048 capture contract drifted")
    if (
        capture["input_row0_bits_sha256"] != input_sha
        or capture["output_row0_bits_sha256"] != output_sha
        or capture["repeated_output_row0_bits_sha256"] != output_sha
    ):
        raise RuntimeError("M2048 capture digest linkage drifted")
    for field in (
        "local_replica_row0_sha256_by_trial",
        "repeated_local_replica_row0_sha256_by_trial",
    ):
        rows = capture[field]
        expected_rows = [
            [api["array_sha256"](output_bits[trial])] * 4
            for trial in range(32)
        ]
        if rows != expected_rows:
            raise RuntimeError(f"M2048 local replica agreement drifted: {field}")
    if (
        analysis["column_candidate_count_histogram"] != {"1": 6144}
        or analysis["every_lane_has_exactly_one_candidate"] is not True
        or analysis["uncovered_column_count"] != 0
    ):
        raise RuntimeError("M2048 association is not unique for every lane")
    pre = base.snapshot_member(run_fd, "census_pre.txt", limit=2 << 20)
    post = base.snapshot_member(run_fd, "census_post.txt", limit=2 << 20)
    _require_census(pre, b"CENSUS_OK")
    _require_census(post, b"CENSUS_OK")
    vacancy_raw = base.snapshot_member(
        run_fd, "remote_vacancy.raw.txt", limit=1 << 20
    )
    vacancy = base.snapshot_member(run_fd, "remote_vacancy.txt", limit=1 << 20)
    base._validate_remote_vacancy_evidence(vacancy_raw, vacancy, remote)
    orchestrator = base.snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
    base.write_member_exclusive(run_fd, "orchestrator.sealed.log", orchestrator)
    summary = {
        "analysis_sha256": sha256(analysis_raw).hexdigest(),
        "classification": "M2048_STRATEGY_ND_ASSOCIATION_UNIQUE;DIAGNOSTIC_ONLY;GATE_D_OPEN",
        "code_hash": code_pin,
        "elapsed_seconds": elapsed,
        "gate_d_closed": False,
        "input_row0_bits_sha256": input_sha,
        "launch_to_jax_process": list(EXPECTED_LAUNCH_TO_JAX_PROCESS),
        "optimized_hlo_sha256": reference["optimized_hlo_sha256"],
        "output_row0_bits_sha256": output_sha,
        "performance_claim": False,
        "run_tag": run_tag,
        "stablehlo_sha256": reference["stablehlo_sha256"],
        "topology_hash": EXPECTED_TOPOLOGY_HASH,
    }
    evidence = {
        "artifact_kind": "gate_d_m2048_strategy_nd_association",
        "claim_scope": (
            "Exact-shape model-free row-zero association only; no model, "
            "decoder, performance, locality-admissibility or Gate-D claim."
        ),
        "source_identity": api["source_identity"](),
        "summary": summary,
    }
    base.write_member_exclusive(run_fd, "analysis.json", analysis_raw + b"\n")
    base.write_member_exclusive(
        run_fd, "fleet_records.json", _canonical(records) + b"\n"
    )
    base.write_member_exclusive(
        run_fd, "input_row0_bits.npy", _npy(input_bits)
    )
    base.write_member_exclusive(
        run_fd, "output_row0_bits.npy", _npy(output_bits)
    )
    base.write_member_exclusive(
        run_fd, "m2048.optimized_hlo.txt", optimized_hlo.encode()
    )
    base.write_member_exclusive(
        run_fd, "m2048.stablehlo.mlir", stablehlo.encode()
    )
    base.write_member_exclusive(run_fd, "summary.json", _canonical(summary) + b"\n")
    base.write_member_exclusive(run_fd, "evidence.json", _canonical(evidence) + b"\n")
    observed = base.local_members(run_fd)
    if observed != SUCCESS_PAYLOAD | {"orchestrator.log"}:
        raise RuntimeError(f"M2048 local success inventory drifted: {sorted(observed)}")
    return {
        name: base.snapshot_member(run_fd, name, limit=64 << 20)
        for name in SUCCESS_PAYLOAD
    }


def _require_preterminal(base: Any, run_fd: int) -> None:
    base.require_preterminal(run_fd)
    for name in ("M2048_RESULT", "terminal_upload_receipt.json"):
        try:
            os.stat(name, dir_fd=run_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise RuntimeError("local M2048 terminal publication already began")


def _initialize(
    base: Any, run_dir: Path, run_dir_fd: int, runtime_raw: bytes
) -> str:
    run_fd = base._run_fd(run_dir, run_dir_fd)
    try:
        with os.scandir(run_fd) as entries:
            if any(True for _ in entries):
                raise RuntimeError("retained M2048 run directory is not pristine")
        base.write_member_exclusive(run_fd, "publisher_runtime.json", runtime_raw)
        metadata = os.fstat(run_fd)
        return f"{metadata.st_dev}:{metadata.st_ino}"
    finally:
        os.close(run_fd)


def _publish_success(
    base: Any,
    api: Mapping[str, Any],
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    elapsed: int,
    runtime_raw: bytes,
    run_dir_fd: int,
) -> str:
    run_tag = base.validate_run_dir(run_dir)
    run_fd = base._run_fd(run_dir, run_dir_fd)
    try:
        _require_preterminal(base, run_fd)
        base.require_publication_runtime(run_fd, runtime_raw)
        payload = _prepare_success(
            base,
            run_fd,
            code_pin=code_pin,
            run_tag=run_tag,
            remote=remote,
            elapsed=elapsed,
            api=api,
        )
        bucket, prefix = base._bucket_and_prefix(remote, None)
        base._require_never_used_prefix(bucket, prefix)
        records = []
        for relative in sorted(payload):
            record = base._upload_bound(bucket, prefix + relative, payload[relative])
            record["path"] = relative
            records.append(record)
        ledger_raw = _canonical(
            {
                "artifact_kind": "gate_d_m2048_remote_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "remote_objects.json", ledger_raw)
        ledger = base._upload_bound(bucket, prefix + "remote_objects.json", ledger_raw)
        ledger["path"] = "remote_objects.json"
        for record in records:
            base._replay_bound(bucket, prefix + record["path"], record)
        base._replay_bound(bucket, prefix + "remote_objects.json", ledger)
        expected_remote = set(payload) | {"remote_objects.json"}
        if base._observed_names(bucket, prefix) != expected_remote:
            raise RuntimeError("M2048 preterminal remote set drifted")
        marker = {
            "artifact_kind": "gate_d_m2048_strategy_nd_result",
            "classification": "M2048_STRATEGY_ND_ASSOCIATION_UNIQUE;DIAGNOSTIC_ONLY;GATE_D_OPEN",
            "evidence_sha256": sha256(payload["evidence.json"]).hexdigest(),
            "gate_d_closed": False,
            "performance_claim": False,
            "remote_ledger": ledger,
            "run_tag": run_tag,
            "status": "M2048_ASSOCIATION_UNIQUE",
            "summary_sha256": sha256(payload["summary.json"]).hexdigest(),
        }
        marker["marker_payload_sha256"] = sha256(_canonical(marker)).hexdigest()
        terminal_raw = _canonical(marker)
        base.write_member_exclusive(run_fd, "M2048_RESULT", terminal_raw)
        terminal = base._upload_bound(bucket, prefix + "M2048_RESULT", terminal_raw)
        terminal["path"] = "M2048_RESULT"
        base._replay_bound(bucket, prefix + "M2048_RESULT", terminal)
        if base._observed_names(bucket, prefix) != expected_remote | {"M2048_RESULT"}:
            raise RuntimeError("M2048 terminal remote set drifted")
        receipt = {
            "artifact_kind": "gate_d_m2048_terminal_receipt",
            "remote": remote + "/M2048_RESULT",
            "terminal": terminal,
        }
        base.write_member_exclusive(
            run_fd, "terminal_upload_receipt.json", _canonical(receipt)
        )
        return (
            "M2048_RESULT status=M2048_ASSOCIATION_UNIQUE "
            f"marker_sha256={marker['marker_payload_sha256']} "
            f"terminal_generation={terminal['generation']} "
            f"terminal_sha256={terminal['sha256']}"
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
    runtime_raw: bytes,
    run_dir_fd: int,
) -> None:
    run_tag = base.validate_run_dir(run_dir)
    run_fd = base._run_fd(run_dir, run_dir_fd)
    try:
        _require_preterminal(base, run_fd)
        base.require_publication_runtime(run_fd, runtime_raw)
        base.write_member_exclusive(
            run_fd,
            "failure_status.json",
            _canonical(
                {
                    "artifact_kind": "gate_d_m2048_failure",
                    "code_hash": code_pin,
                    "exit_status": status,
                    "gate_d_closed": False,
                    "performance_claim": False,
                    "run_tag": run_tag,
                }
            ),
        )
        orchestrator = base.snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
        base.write_member_exclusive(run_fd, "orchestrator.failure.log", orchestrator)
        vacancy_raw = base.snapshot_member(
            run_fd, "remote_vacancy.raw.txt", limit=1 << 20
        )
        vacancy = base.snapshot_member(run_fd, "remote_vacancy.txt", limit=1 << 20)
        base._validate_remote_vacancy_evidence(vacancy_raw, vacancy, remote)
        names = sorted(
            name
            for name in base.local_members(run_fd)
            if name
            not in {
                "orchestrator.log",
                "diagnostic_objects.json",
                "diagnostic_upload_receipt.json",
            }
        )
        if not names or len(names) > 64:
            raise RuntimeError("M2048 diagnostic inventory is unsafe")
        payload = {name: base.snapshot_member(run_fd, name) for name in names}
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
            base._replay_bound(
                bucket, diagnostic_prefix + record["path"], record
            )
        if base._observed_names(bucket, diagnostic_prefix) != set(payload):
            raise RuntimeError("M2048 diagnostic object set drifted")
        ledger_raw = _canonical(
            {
                "artifact_kind": "gate_d_m2048_diagnostic_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "diagnostic_objects.json", ledger_raw)
        terminal = base._upload_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_raw
        )
        terminal["path"] = "diagnostic_objects.json"
        base._replay_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", terminal
        )
        if base._observed_names(bucket, diagnostic_prefix) != set(payload) | {
            "diagnostic_objects.json"
        }:
            raise RuntimeError("M2048 diagnostic terminal set drifted")
        base.write_member_exclusive(
            run_fd,
            "diagnostic_upload_receipt.json",
            _canonical(
                {
                    "artifact_kind": "gate_d_m2048_diagnostic_receipt",
                    "remote": remote + "/diagnostic/diagnostic_objects.json",
                    "terminal": terminal,
                }
            ),
        )
    finally:
        os.close(run_fd)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--run-dir-fd", type=int, choices=(7,), required=True)
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
    modes.add_parser("append-log").add_argument(
        "--run-dir", type=Path, required=True
    )
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    arguments = _arguments()
    base = _load_base(arguments.expected_code_hash)
    base.validate_environment()
    runtime = base.validate_publication_runtime()
    runtime["artifact_kind"] = "gate_d_m2048_publisher_runtime"
    runtime_raw = _canonical(runtime)
    _verify_running_source(
        arguments.expected_code_hash, arguments.expected_source_sha256
    )
    if arguments.mode == "init":
        print(
            "RUN_IDENTITY "
            + _initialize(base, arguments.run_dir, arguments.run_dir_fd, runtime_raw),
            flush=True,
        )
    elif arguments.mode == "success":
        api = _load_project_api(arguments.expected_code_hash)
        print(
            _publish_success(
                base,
                api,
                arguments.run_dir,
                arguments.remote_prefix,
                code_pin=arguments.expected_code_hash,
                elapsed=arguments.elapsed,
                runtime_raw=runtime_raw,
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
            runtime_raw=runtime_raw,
            run_dir_fd=arguments.run_dir_fd,
        )
    elif arguments.mode == "write":
        run_fd = base._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            _require_preterminal(base, run_fd)
            base.require_publication_runtime(run_fd, runtime_raw)
            base.write_preterminal_member(
                run_fd, arguments.member, sys.stdin.buffer, limit=32 << 20
            )
        finally:
            os.close(run_fd)
    else:
        run_fd = base._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            _require_preterminal(base, run_fd)
            base.require_publication_runtime(run_fd, runtime_raw)
            raw = sys.stdin.buffer.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise RuntimeError("M2048 log append exceeds limit")
            base.append_preterminal_log(run_fd, raw)
        finally:
            os.close(run_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
