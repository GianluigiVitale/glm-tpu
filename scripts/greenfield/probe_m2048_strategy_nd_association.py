#!/usr/bin/env python3
"""Protected exact-M2048 StrategyND association discriminator.

All eight TPU hosts execute the same sealed committed source.  Results are
emitted as bounded base64 records on stdout so the controller can retain them
through its append-only publisher without trusting mutable worker pathnames.
The deterministic input bank is not transported: the publisher regenerates
and hash-checks it independently from the committed generator.

This is a model-free numerical diagnostic.  It makes no decoder, Gate-D,
performance, or admissibility claim for full-pod layer collectives.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
from typing import Any
import zipfile


SOURCE_PATH = "scripts/greenfield/probe_m2048_strategy_nd_association.py"
INSTALL_PATH = Path(
    "/usr/local/libexec/glm-tpu/gate-d-m2048-strategy-nd-v4/"
    "probe_m2048_strategy_nd_association.py"
)
REPOSITORY = Path("/home/gianl/glm-tpu-topology-rewrite")
JAX_SITE_ROOT = "/opt/glm-tpu/gate-d-jax-site-55233c63939e"
LIBTPU_SITE_ROOT = "/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3"
PYTHON_RUNTIME_ROOT = "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95"
EXPECTED_RUNTIME_PATH = (
    f"{PYTHON_RUNTIME_ROOT}/lib/python312.zip",
    f"{PYTHON_RUNTIME_ROOT}/lib/python3.12",
    f"{PYTHON_RUNTIME_ROOT}/lib/python3.12/lib-dynload",
)
TAG_PATTERN = re.compile(
    r"greenfield_m2048_strategy_nd_[0-9]{8}T[0-9]{15}Z"
)
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_STATIC_ENVIRONMENT = {
    "HOME": "/home/gianl",
    "JAX_ENABLE_COMPILATION_CACHE": "0",
    "JAX_PLATFORMS": "tpu",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": ".20",
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


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.attributesFile=/dev/null",
            "-C",
            str(REPOSITORY),
            *arguments,
        ],
        env=_GIT_ENVIRONMENT,
    )


def _snapshot_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe probe file: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after) or (
            before.st_dev,
            before.st_ino,
        ) != (named.st_dev, named.st_ino):
            raise RuntimeError(f"probe file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_root_boundary(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError("probe installation path is unsafe")
    for parent in (path.parent, *path.parent.parents):
        metadata = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError(f"probe installation parent is mutable: {parent}")


def _verify_running_source(code_pin: str, expected_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", code_pin):
        raise RuntimeError("code pin must be a full commit id")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("probe source SHA-256 is invalid")
    if Path(__file__) != INSTALL_PATH:
        raise RuntimeError("probe is not executing from its immutable capsule")
    _require_root_boundary(INSTALL_PATH)
    metadata = os.stat(INSTALL_PATH, follow_symlinks=False)
    raw = _snapshot_regular(INSTALL_PATH)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or os.listxattr(INSTALL_PATH, follow_symlinks=False)
        or _git_bytes("rev-parse", "HEAD").decode().strip() != code_pin
        or _git_bytes("status", "--porcelain=v1", "--untracked-files=all")
        or _git_bytes("for-each-ref", "--format=%(refname)", "refs/replace")
        or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("probe source/repository identity drifted")
    return expected_sha256


def _sealed_git_source_archive(code_pin: str) -> tuple[str, dict[str, Any]]:
    tree = _git_bytes("ls-tree", "-r", "-z", code_pin, "--", "glm_tpu")
    records = []
    buffer = BytesIO()
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
        "gate-d-m2048-source.zip", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC
    )
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("source archive write stalled")
        view = view[written:]
    fcntl.fcntl(descriptor, _F_ADD_SEALS, _MEMFD_SEALS)
    if (
        fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEALS
        or os.pread(descriptor, len(raw) + 1, 0) != raw
    ):
        raise RuntimeError("source archive seal/replay drifted")
    manifest = _canonical(records)
    return f"/proc/self/fd/{descriptor}", {
        "archive_sha256": sha256(raw).hexdigest(),
        "file_manifest_count": len(records),
        "file_manifest_sha256": sha256(manifest).hexdigest(),
    }


def _fleet_digest(
    multihost_utils: Any,
    digest_hex: str,
    *,
    label: str,
    process_count: int,
) -> list[str]:
    import numpy as np

    digest = np.frombuffer(bytes.fromhex(digest_hex), dtype=np.uint8)
    fleet = np.asarray(multihost_utils.process_allgather(digest)).reshape(
        process_count, len(digest)
    )
    values = [row.tobytes().hex() for row in fleet]
    if len(set(values)) != 1:
        raise RuntimeError(f"hosts disagree on {label}: {values}")
    return values


def _runtime_topology(jax: Any, multihost_utils: Any, process_count: int) -> Any:
    import numpy as np

    from glm_tpu.greenfield.topology import (
        discover_physical_topology,
        validate_target_v4_64,
    )

    local_ids = np.asarray(
        [device.id for device in jax.local_devices()], dtype=np.int32
    )
    fleet_local_ids = np.asarray(
        multihost_utils.process_allgather(local_ids)
    ).reshape(process_count, jax.local_device_count())
    observed_order = {
        int(device_id): local_index
        for process_row in fleet_local_ids
        for local_index, device_id in enumerate(process_row.tolist())
    }
    topology = discover_physical_topology(
        jax.devices(),
        slice_name="db-v4-64-od",
        observed_local_order=observed_order,
    )
    validate_target_v4_64(topology)
    return topology, fleet_local_ids


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-pin", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, choices=(8,), required=True)
    parser.add_argument("--process-id", type=int, choices=range(8), required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if TAG_PATTERN.fullmatch(arguments.run_tag) is None:
        raise RuntimeError("M2048 run tag is invalid")
    expected_environment = {
        **_STATIC_ENVIRONMENT,
        "GLM_GATE_D_M2048_TAG": arguments.run_tag,
    }
    if dict(os.environ) != expected_environment:
        raise RuntimeError(
            "probe environment drifted: "
            + ",".join(sorted(set(os.environ) ^ set(expected_environment)))
        )
    probe_sha = _verify_running_source(
        arguments.code_pin, arguments.expected_source_sha256
    )
    archive_path, archive_identity = _sealed_git_source_archive(
        arguments.code_pin
    )
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        raise RuntimeError("probe must run under Python -I -S")
    sys.path[:] = [
        archive_path,
        JAX_SITE_ROOT,
        LIBTPU_SITE_ROOT,
        *EXPECTED_RUNTIME_PATH,
    ]

    from glm_tpu.greenfield.benchmarking import sealed_runtime

    provenance = {
        "probe_sha256": probe_sha,
        "source_archive": archive_identity,
        "python": sealed_runtime.validate_python_runtime(),
        "sites": sealed_runtime.validate_dependency_sites(),
        "sys_path": list(sys.path),
    }
    import jax
    from jax.experimental import multihost_utils
    import numpy as np

    from glm_tpu.greenfield.benchmarking import (
        M2048StrategyNdFingerprintConfig,
        analyze_m2048_row0_association,
        array_sha256,
        build_m2048_strategy_nd_fingerprint,
        execute_m2048_strategy_nd_fingerprint,
        generate_m2048_row0_input_bits,
        m2048_source_identity,
        validate_m2048_strategy_nd_fingerprint_hlo,
    )

    jax.distributed.initialize(
        coordinator_address=arguments.coordinator_address,
        num_processes=arguments.num_processes,
        process_id=arguments.process_id,
        local_device_ids=(0, 1, 2, 3),
        cluster_detection_method="deactivate",
    )
    try:
        if (
            jax.default_backend() != "tpu"
            or jax.process_count() != 8
            or jax.local_device_count() != 4
            or jax.device_count() != 32
        ):
            raise RuntimeError(
                "protected M2048 fleet geometry drifted: "
                f"backend={jax.default_backend()} "
                f"process_count={jax.process_count()} "
                f"local_device_count={jax.local_device_count()} "
                f"device_count={jax.device_count()} "
                f"jax_process_index={jax.process_index()} "
                f"launch_process_id={arguments.process_id}"
            )
        # TPU JAX topology-orders processes independently of TPU-VM worker
        # suffixes.  The launch id is the distributed-client identity; the JAX
        # process index is physical-topology identity.  Preserve both and let
        # the publisher authenticate the complete DB555 fleet permutation.
        topology, fleet_local_ids = _runtime_topology(
            jax, multihost_utils, arguments.num_processes
        )
        members = tuple(sorted(device.device_id for device in topology.devices))
        if members != tuple(range(32)):
            raise RuntimeError("M2048 members are not global ids 0..31")
        coordinates = {
            device.device_id: device.coordinates for device in topology.devices
        }
        config = M2048StrategyNdFingerprintConfig()
        input_bits = generate_m2048_row0_input_bits(config)
        multihost_utils.sync_global_devices("gate-d-m2048-before-compile")
        compiled = build_m2048_strategy_nd_fingerprint(
            config,
            members,
            devices=jax.devices(),
            enforce_hlo_contract=False,
        )
        optimized_hlo_sha = sha256(compiled.optimized_hlo.encode()).hexdigest()
        stablehlo_sha = sha256(compiled.stablehlo.encode()).hexdigest()
        if jax.process_index() == 0:
            acquisition = {
                "optimized_hlo_b64": base64.b64encode(
                    compiled.optimized_hlo.encode()
                ).decode("ascii"),
                "optimized_hlo_sha256": optimized_hlo_sha,
                "stablehlo_b64": base64.b64encode(
                    compiled.stablehlo.encode()
                ).decode("ascii"),
                "stablehlo_sha256": stablehlo_sha,
            }
            print(
                "M2048_ACQUISITION process=0 json_b64="
                + base64.b64encode(_canonical(acquisition)).decode("ascii"),
                flush=True,
            )
        multihost_utils.sync_global_devices("gate-d-m2048-graphs-acquired")
        hlo_report, algorithm = validate_m2048_strategy_nd_fingerprint_hlo(
            compiled.optimized_hlo, members
        )
        fleet_hlo = _fleet_digest(
            multihost_utils,
            optimized_hlo_sha,
            label="optimized HLO",
            process_count=arguments.num_processes,
        )
        fleet_stablehlo = _fleet_digest(
            multihost_utils,
            stablehlo_sha,
            label="StableHLO",
            process_count=arguments.num_processes,
        )
        output_bits, capture = execute_m2048_strategy_nd_fingerprint(
            compiled, input_bits
        )
        input_sha = array_sha256(input_bits)
        output_sha = array_sha256(output_bits)
        if jax.process_index() == 0:
            output_bundle = {
                "output_bits_b64": base64.b64encode(
                    np.ascontiguousarray(output_bits).tobytes(order="C")
                ).decode("ascii"),
                "output_bits_sha256": output_sha,
            }
            print(
                "M2048_OUTPUT process=0 json_b64="
                + base64.b64encode(_canonical(output_bundle)).decode("ascii"),
                flush=True,
            )
        multihost_utils.sync_global_devices("gate-d-m2048-output-acquired")
        fleet_input = _fleet_digest(
            multihost_utils,
            input_sha,
            label="input bits",
            process_count=arguments.num_processes,
        )
        fleet_output = _fleet_digest(
            multihost_utils,
            output_sha,
            label="output bits",
            process_count=arguments.num_processes,
        )
        analysis = analyze_m2048_row0_association(
            input_bits, output_bits, members, coordinates
        )
        analysis_raw = _canonical(analysis)
        fleet_analysis = _fleet_digest(
            multihost_utils,
            sha256(analysis_raw).hexdigest(),
            label="association analysis",
            process_count=arguments.num_processes,
        )
        provenance["import_closure"] = sealed_runtime.verify_import_closure(
            Path(archive_path)
        )
        record = {
            "analysis_sha256": sha256(analysis_raw).hexdigest(),
            "capture": capture,
            "captured_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "code_hash": arguments.code_pin,
            "collective_algorithm": dict(algorithm),
            "config": config.to_dict(),
            "diagnostic_only": True,
            "fleet_analysis_sha256s": fleet_analysis,
            "fleet_hlo_sha256s": fleet_hlo,
            "fleet_input_sha256s": fleet_input,
            "fleet_local_device_ids_in_runtime_order": fleet_local_ids.tolist(),
            "fleet_output_sha256s": fleet_output,
            "fleet_stablehlo_sha256s": fleet_stablehlo,
            "gate_d_closed": False,
            "hlo": hlo_report.to_dict(),
            "hostname": socket.gethostname(),
            "jax_process_index": jax.process_index(),
            "launch_process_id": arguments.process_id,
            "jax_version": jax.__version__,
            "member_device_ids": list(members),
            "optimized_hlo_sha256": optimized_hlo_sha,
            "performance_claim": False,
            "provenance": provenance,
            "run_tag": arguments.run_tag,
            "schema_version": 1,
            "source_identity": m2048_source_identity(),
            "stablehlo_sha256": stablehlo_sha,
            "topology": topology.to_dict(),
            "topology_hash": topology.topology_hash,
        }
        encoded_record = base64.b64encode(_canonical(record)).decode("ascii")
        print(
            f"M2048_RECORD process={arguments.process_id} json_b64={encoded_record}",
            flush=True,
        )
        multihost_utils.sync_global_devices("gate-d-m2048-complete")
        print(
            f"M2048_OK process={arguments.process_id} host={socket.gethostname()} "
            f"hlo={optimized_hlo_sha} output={output_sha}",
            flush=True,
        )
        return 0
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
