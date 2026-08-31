#!/usr/bin/env -S /opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12 -I -S
"""Produce source-bound compensated Gate-D RMS StableHLO without executing it.

This explicit, default-off tool must run from a clean reviewed Git pin under the
exact sealed interpreter and JAX dependency capsule. It hashes the committed
source, uses only ``ShapeDtypeStruct`` abstract arguments, and calls
``lower().compiler_ir(dialect="stablehlo")``. This is a compiler lowering stage,
but it never performs executable compilation or JAX array/numerical execution.
The output directory is append-only and a SUCCESS receipt is written last.
"""

from __future__ import annotations

import argparse
import fcntl
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Any, Callable


REPO_ROOT = Path("/home/gianl/glm-tpu-topology-rewrite")
PRODUCER_INSTALLED = Path(
    "/opt/glm-tpu/bin/produce_gate_d_compensated_auxiliary_stablehlo.py"
)
PRODUCER_SOURCE = (
    REPO_ROOT / "scripts/greenfield/produce_gate_d_compensated_auxiliary_stablehlo.py"
)
SITE_ROOT = Path("/opt/glm-tpu/gate-d-jax-site-55233c63939e")
PYTHON_RUNTIME_ROOT = Path("/opt/glm-tpu/gate-d-python-3.12.13-021044895e95")
PYTHON = PYTHON_RUNTIME_ROOT / "bin/python3.12"
SOURCE_CERTIFICATE = (
    REPO_ROOT / "docs/artifacts/gate-d-compensated-auxiliary-source-authority.json"
)
PLAN_AUTHORITY = (
    REPO_ROOT / "docs/artifacts/gate-d-compensated-auxiliary-pp16-plan-authority.json"
)
EXPECTED_SOURCE_CERTIFICATE_SHA256 = (
    "237095c7ac9acdd7a37383b951b061752ee83f591722c3a2e2d3bede59326fb0"
)
EXPECTED_PLAN_AUTHORITY_SHA256 = (
    "7d0a5615ff4744801ac6a6598a52ea80e17d431788e4ef21a9d889e81772dbdf"
)
EXPECTED_SOURCE_FILES = {
    "glm_tpu/greenfield/kernels/layer.py": (
        "47c48b20564c0338de2bca70c5e01a9305748010e0f28f94e525176c1b882a0c"
    ),
    "glm_tpu/greenfield/kernels/reference/rmsnorm.py": (
        "b707ddaa4208a58c7a1a999fc0d40b8a2b460d64c15943569c03444a71e3edfd"
    ),
}
EXPECTED_SOURCE_CODE_PIN = "e16d74fcc025f5ffb910d9cbab1c4fa06df887ad"
EXPECTED_SOURCE_SET_SHA256 = (
    "5434644b42325393df94169755ad9bf510b95749e907668006e50375bf73feb5"
)
EXPECTED_CANDIDATE_AST_SHA256 = (
    "451f4fe010ffd38046ae5716742cb756ce54853bb46b8d6199af2e854946df2a"
)
EXPECTED_CALLSITE_AST_SHA256 = (
    "ac08b3c776e80c1c0510401dbd6cb58f2fdc8c008df2b8605c707a5e954e1eec"
)
EXPECTED_PYTHON_SHA256 = (
    "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
)
EXPECTED_PYTHON_RUNTIME_TREE_SHA256 = (
    "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616"
)
EXPECTED_SITE_TREE_SHA256 = (
    "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df"
)
RISKY_ENVIRONMENT_PREFIXES = (
    "CUDA_",
    "HIP_",
    "HSA_",
    "JAX_",
    "LD_",
    "LIBTPU_",
    "NCCL_",
    "PJRT_",
    "PYTHON",
    "ROCM_",
    "TF_XLA_",
    "TPU_",
    "XLA_",
)
HIDDEN_SIZE = 6144
EPSILON = 1e-5
# Linux UAPI values from <linux/fcntl.h>. CPython exposes ``fcntl.fcntl`` on
# this host but omits the GNU-gated symbolic constants from its ``fcntl``
# module, so source-level attribute lookup is not portable across the sealed
# runtime. The post-add exact mask check below remains the runtime authority.
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _runtime_tree_sha256(root: Path) -> str:
    digest = sha256()
    entries = [
        root,
        *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()),
    ]
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if metadata.st_uid != 0 or metadata.st_gid != 0 or (
            not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_mode
            & (stat.S_IWGRP | stat.S_IWOTH | stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)
        ):
            raise SystemExit(f"Python runtime entry is not sealed root:root: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"D"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
            payload = struct.pack(">Q", metadata.st_size) + bytes.fromhex(_sha_file(entry))
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            target = os.readlink(entry).encode("utf-8")
            try:
                Path(os.path.realpath(entry)).relative_to(root)
            except ValueError as error:
                raise SystemExit(f"Python runtime symlink escapes root: {entry}") from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise SystemExit(f"unsupported Python runtime entry: {entry}")
        if os.listxattr(entry, follow_symlinks=False):
            raise SystemExit(f"Python runtime entry has extended attributes: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", stat.S_IMODE(metadata.st_mode) & ~0o7022))
        digest.update(payload)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        text=True,
    )
    return completed.stdout.strip()


def _sealed_source_snapshot(path: Path, expected_sha256: str) -> bytes:
    descriptor = os.memfd_create(
        f"gate-d-source:{path.name}", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC
    )
    digest = sha256()
    observed = 0
    try:
        source_fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            while block := os.read(source_fd, 1024 * 1024):
                observed += len(block)
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        finally:
            os.close(source_fd)
        if digest.hexdigest() != expected_sha256:
            raise SystemExit(f"committed source drifted: {path}")
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _MEMFD_SEALS)
        if fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEALS:
            raise SystemExit(f"source memfd seal drifted: {path}")
        payload = bytearray()
        offset = 0
        while offset < observed:
            block = os.pread(descriptor, min(1024 * 1024, observed - offset), offset)
            if not block:
                break
            payload.extend(block)
            offset += len(block)
        if (
            offset != observed
            or os.pread(descriptor, 1, observed)
            or sha256(payload).hexdigest() != expected_sha256
        ):
            raise SystemExit(f"sealed source revalidation drifted: {path}")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _write_exclusive(directory_fd: int, name: str, payload: bytes) -> None:
    if not name or "/" in name or name in {".", ".."}:
        raise SystemExit("output member name is invalid")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
    )
    descriptor = os.open(name, flags, 0o400, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory_chain(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in Path(os.path.abspath(path)).parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise SystemExit(f"output parent chain is group/world writable: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_output_directory(
    output: Path,
) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    parent_fd = _open_directory_chain(output.parent)
    parent_metadata = os.fstat(parent_fd)
    parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
    try:
        os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        output_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except BaseException:
        os.close(parent_fd)
        raise
    metadata = os.fstat(output_fd)
    return parent_fd, output_fd, parent_identity, (metadata.st_dev, metadata.st_ino)


def _verify_output_identity(
    parent_fd: int,
    output_fd: int,
    output_parent: Path,
    output_name: str,
    parent_identity: tuple[int, int],
    identity: tuple[int, int],
) -> None:
    parent_descriptor_metadata = os.fstat(parent_fd)
    parent_path_metadata = os.stat(output_parent, follow_symlinks=False)
    descriptor_metadata = os.fstat(output_fd)
    path_metadata = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
    observed = (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
    if (
        (parent_descriptor_metadata.st_dev, parent_descriptor_metadata.st_ino)
        != parent_identity
        or (parent_path_metadata.st_dev, parent_path_metadata.st_ino) != parent_identity
        or not stat.S_ISDIR(parent_path_metadata.st_mode)
        or observed != identity
        or (path_metadata.st_dev, path_metadata.st_ino) != identity
        or not stat.S_ISDIR(path_metadata.st_mode)
    ):
        raise SystemExit("append-only output directory identity drifted")


def _beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _immutable_file_record(path: Path) -> dict[str, Any]:
    resolved = Path(os.path.realpath(path))
    current = Path("/")
    for component in resolved.parts[1:]:
        current /= component
        metadata = current.lstat()
        if metadata.st_uid != 0 or metadata.st_gid != 0 or metadata.st_mode & (
            stat.S_IWGRP | stat.S_IWOTH
        ):
            raise SystemExit(f"loaded dependency is not immutable root:root: {resolved}")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & (
        stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX
    ):
        raise SystemExit(f"loaded dependency type/mode is unsafe: {resolved}")
    if os.listxattr(resolved, follow_symlinks=False):
        raise SystemExit(f"loaded dependency has extended attributes: {resolved}")
    return {
        "bytes": metadata.st_size,
        "path": str(resolved),
        "sha256": _sha_file(resolved),
    }


def _loaded_dependency_records(producer: Path) -> dict[str, list[dict[str, Any]]]:
    python_paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path.startswith("/"):
            continue
        resolved = Path(os.path.realpath(raw_path))
        if not (
            _beneath(resolved, PYTHON_RUNTIME_ROOT) or _beneath(resolved, SITE_ROOT)
        ) and resolved != producer:
            raise SystemExit(f"loaded Python module escaped immutable roots: {resolved}")
        python_paths.add(resolved)
    native_paths: set[Path] = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            raise SystemExit(f"loaded native dependency was deleted: {raw_path}")
        native_paths.add(Path(raw_path))
    return {
        "native_mappings": [_immutable_file_record(path) for path in sorted(native_paths)],
        "python_modules": [_immutable_file_record(path) for path in sorted(python_paths)],
    }


def _annotate_candidate(raw: str) -> tuple[str, str]:
    metadata = {
        "gate_d.callsite_ast_sha256": EXPECTED_CALLSITE_AST_SHA256,
        "gate_d.candidate_ast_sha256": EXPECTED_CANDIDATE_AST_SHA256,
        "gate_d.source_set_sha256": EXPECTED_SOURCE_SET_SHA256,
    }
    fragment = ", ".join(f'{key} = "{value}"' for key, value in sorted(metadata.items()))
    first_line, separator, remainder = raw.partition("\n")
    if not separator or not first_line.startswith("module"):
        raise SystemExit("candidate lowering has no canonical module header")
    marker = " attributes {"
    if marker in first_line:
        annotated_first = first_line.replace(marker, f"{marker}{fragment}, ", 1)
        inverse_first = annotated_first.replace(f"{marker}{fragment}, ", marker, 1)
    else:
        brace = first_line.find(" {")
        if brace < 0:
            raise SystemExit("candidate lowering module header is unsupported")
        annotated_first = first_line[:brace] + f" attributes {{{fragment}}}" + first_line[brace:]
        inverse_first = annotated_first.replace(f" attributes {{{fragment}}}", "", 1)
    if inverse_first != first_line:
        raise SystemExit("candidate metadata transform is not exactly reversible")
    annotated = annotated_first + separator + remainder
    return annotated, sha256(fragment.encode("ascii")).hexdigest()


def _lower(
    jax: Any,
    jnp: Any,
    function: Callable[..., Any],
    *,
    name: str,
) -> str:
    function.__name__ = name
    activation = jax.ShapeDtypeStruct((1, HIDDEN_SIZE), jnp.bfloat16)
    weight = jax.ShapeDtypeStruct((HIDDEN_SIZE,), jnp.bfloat16)
    lowered = jax.jit(function).lower(activation, activation, weight)
    return str(lowered.compiler_ir(dialect="stablehlo")) + "\n"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-pin", required=True)
    parser.add_argument("--expected-producer-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if os.geteuid() == 0:
        raise SystemExit("lowering producer must run unprivileged")
    if (
        Path(sys.executable) != PYTHON
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        raise SystemExit("invoke with exact venv Python -I -S")
    inherited = sorted(
        name
        for name in os.environ
        if any(name.upper().startswith(prefix) for prefix in RISKY_ENVIRONMENT_PREFIXES)
    )
    if inherited:
        raise SystemExit(f"unsafe inherited environment: {inherited}")
    if _sha_file(PYTHON) != EXPECTED_PYTHON_SHA256:
        raise SystemExit("lowering Python drifted")
    if _runtime_tree_sha256(PYTHON_RUNTIME_ROOT) != EXPECTED_PYTHON_RUNTIME_TREE_SHA256:
        raise SystemExit("lowering Python runtime tree drifted")
    if _runtime_tree_sha256(SITE_ROOT) != EXPECTED_SITE_TREE_SHA256:
        raise SystemExit("lowering JAX site capsule drifted")
    producer = Path(__file__).resolve(strict=True)
    if producer != PRODUCER_INSTALLED:
        raise SystemExit("producer path drifted")
    producer_metadata = producer.lstat()
    if (
        producer_metadata.st_uid != 0
        or producer_metadata.st_gid != 0
        or stat.S_IMODE(producer_metadata.st_mode) != 0o555
        or os.listxattr(producer, follow_symlinks=False)
    ):
        raise SystemExit("installed producer ownership/mode drifted")
    if (
        _sha_file(producer) != arguments.expected_producer_sha256
        or _sha_file(PRODUCER_SOURCE) != arguments.expected_producer_sha256
    ):
        raise SystemExit("producer bytes drifted")
    code_pin = _git("rev-parse", "HEAD")
    if code_pin != arguments.expected_code_pin or _git("status", "--porcelain"):
        raise SystemExit("producer Git pin is dirty or unexpected")
    if _sha_file(SOURCE_CERTIFICATE) != EXPECTED_SOURCE_CERTIFICATE_SHA256:
        raise SystemExit("source certificate drifted")
    if _sha_file(PLAN_AUTHORITY) != EXPECTED_PLAN_AUTHORITY_SHA256:
        raise SystemExit("plan authority drifted")
    source_certificate = json.loads(SOURCE_CERTIFICATE.read_text())
    if (
        source_certificate.get("code_pin") != EXPECTED_SOURCE_CODE_PIN
        or source_certificate.get("source_set_sha256") != EXPECTED_SOURCE_SET_SHA256
        or source_certificate.get("candidate_id") != "compensated_auxiliary_dependency"
    ):
        raise SystemExit("source certificate compensated tuple drifted")
    source_payloads = {
        relative: _sealed_source_snapshot(REPO_ROOT / relative, expected)
        for relative, expected in EXPECTED_SOURCE_FILES.items()
    }
    output = Path(os.path.abspath(arguments.output))
    if output.name in {"", ".", ".."}:
        raise SystemExit("output directory name is invalid")
    parent_fd, output_fd, parent_identity, output_identity = _create_output_directory(
        output
    )

    sys.dont_write_bytecode = True
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(SITE_ROOT))
    if importlib.util.find_spec("libtpu") is not None:
        raise SystemExit("immutable lowering capsule unexpectedly exposes libtpu")
    if importlib.util.find_spec("jax_plugins") is not None:
        raise SystemExit("immutable lowering capsule unexpectedly exposes jax_plugins")
    if importlib.metadata.entry_points(group="jax_plugins"):
        raise SystemExit("immutable lowering capsule exposes a JAX plugin entry point")
    import jax
    import jax.numpy as jnp
    import jaxlib

    rms_path = "glm_tpu/greenfield/kernels/reference/rmsnorm.py"
    rms_namespace = {
        "__file__": str(REPO_ROOT / rms_path),
        "__name__": "gate_d_exact_rmsnorm_source",
        "__package__": None,
    }
    exec(
        compile(source_payloads[rms_path], str(REPO_ROOT / rms_path), "exec"),
        rms_namespace,
    )
    fused_add_rms_norm = rms_namespace["fused_add_rms_norm"]
    fused_add_rms_norm_with_compensated_auxiliary = rms_namespace[
        "fused_add_rms_norm_with_compensated_auxiliary"
    ]

    def accepted(hidden: Any, residual: Any, weight: Any) -> Any:
        return fused_add_rms_norm(hidden, residual, weight, epsilon=EPSILON)

    def candidate(hidden: Any, residual: Any, weight: Any) -> Any:
        return fused_add_rms_norm_with_compensated_auxiliary(
            hidden, residual, weight, epsilon=EPSILON
        )

    accepted_raw = _lower(jax, jnp, accepted, name="gate_d_accepted_rms")
    candidate_raw = _lower(
        jax, jnp, candidate, name="gate_d_compensated_auxiliary_rms"
    )
    candidate_annotated, annotation_sha = _annotate_candidate(candidate_raw)
    backend = jax.default_backend()
    devices = jax.devices("cpu")
    if backend != "cpu" or not devices or any(device.platform != "cpu" for device in devices):
        raise SystemExit("lowering escaped the forced CPU backend")
    if "libtpu" in sys.modules or any(
        name == "jax_plugins" or name.startswith("jax_plugins.") for name in sys.modules
    ):
        raise SystemExit("forbidden TPU/JAX plugin module loaded")
    loaded_dependencies = _loaded_dependency_records(producer)
    for relative, expected in EXPECTED_SOURCE_FILES.items():
        if _sha_file(REPO_ROOT / relative) != expected:
            raise SystemExit(f"committed source changed during lowering: {relative}")
    if _git("rev-parse", "HEAD") != code_pin or _git("status", "--porcelain"):
        raise SystemExit("repository changed during lowering")
    if _sha_file(producer) != arguments.expected_producer_sha256:
        raise SystemExit("producer changed during lowering")

    artifacts = {
        "accepted.raw.stablehlo": accepted_raw.encode("utf-8"),
        "candidate.raw.stablehlo": candidate_raw.encode("utf-8"),
        "candidate.stablehlo": candidate_annotated.encode("utf-8"),
    }
    for name, payload in artifacts.items():
        _write_exclusive(output_fd, name, payload)
    receipt = {
        "artifacts": {
            name: {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
            for name, payload in sorted(artifacts.items())
        },
        "backend": {
            "device_count": len(devices),
            "platform": backend,
        },
        "claim_scope": (
            "Forced-CPU abstract compiler lowering only; no executable compilation, JAX array/"
            "numerical execution, model, cloud workflow, TPU backend initialization, performance "
            "or Gate-D closure claim. JAX import may perform read-only host TPU PCI discovery."
        ),
        "loaded_dependencies": loaded_dependencies,
        "environment": {
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "python_executable": str(PYTHON),
            "python_runtime_root": str(PYTHON_RUNTIME_ROOT),
            "python_runtime_tree_sha256": EXPECTED_PYTHON_RUNTIME_TREE_SHA256,
            "python_sha256": EXPECTED_PYTHON_SHA256,
            "python_version": sys.version,
            "site_root": str(SITE_ROOT),
            "site_tree_sha256": EXPECTED_SITE_TREE_SHA256,
        },
        "inputs": {
            "activation_dtype": "bf16",
            "activation_shape": [1, HIDDEN_SIZE],
            "epsilon": EPSILON,
            "weight_dtype": "bf16",
            "weight_shape": [HIDDEN_SIZE],
        },
        "metadata_annotation_sha256": annotation_sha,
        "plan_authority_sha256": EXPECTED_PLAN_AUTHORITY_SHA256,
        "producer": {
            "code_pin": code_pin,
            "installed_path": str(producer),
            "sha256": arguments.expected_producer_sha256,
            "source_path": PRODUCER_SOURCE.relative_to(REPO_ROOT).as_posix(),
        },
        "schema_version": 1,
        "source": {
            "callsite_ast_sha256": EXPECTED_CALLSITE_AST_SHA256,
            "candidate_ast_sha256": EXPECTED_CANDIDATE_AST_SHA256,
            "certificate_sha256": EXPECTED_SOURCE_CERTIFICATE_SHA256,
            "code_pin": EXPECTED_SOURCE_CODE_PIN,
            "files": EXPECTED_SOURCE_FILES,
            "source_set_sha256": EXPECTED_SOURCE_SET_SHA256,
        },
    }
    receipt_payload = (_canonical(receipt) + "\n").encode("ascii")
    _write_exclusive(output_fd, "producer_receipt.json", receipt_payload)
    success = {
        "producer_receipt_sha256": sha256(receipt_payload).hexdigest(),
        "schema_version": 1,
    }
    _write_exclusive(output_fd, "SUCCESS", (_canonical(success) + "\n").encode("ascii"))
    os.fsync(output_fd)
    os.fsync(parent_fd)
    _verify_output_identity(
        parent_fd,
        output_fd,
        output.parent,
        output.name,
        parent_identity,
        output_identity,
    )
    os.close(output_fd)
    os.close(parent_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
