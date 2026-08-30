#!/usr/bin/env python3
"""Parse and validate one Gate-D causal StableHLO authority offline.

The parent admission process is stdlib-only and sends exact SHA-authenticated
bytes over stdin.  This helper imports jaxlib's MLIR bindings only; it never
imports JAX, lowers, compiles, initializes a backend, or touches a device.
"""

from __future__ import annotations

import base64
import ctypes
import fcntl
from hashlib import sha256
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_MEMFD_SEAL_MASK = 1 | 2 | 4 | 8
_SEALED_BINDINGS: dict[str, dict[str, Any]] = {}
_SEALED_SOURCE_MODULES: dict[str, tuple[str, bool]] = {}
_SEALED_EXTENSION_MODULES: dict[str, str] = {}
_SEALED_NAMESPACE_MODULES: set[str] = set()
_LOADED_SOURCE_PATHS: set[str] = set()
_LOADED_NATIVE_PATHS: set[str] = set()
_PROC_MAP_ESCAPE = re.compile(r"\\([0-7]{3})")
_FORBIDDEN_RUNTIME_MODE = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX


def _read_sealed_fd(descriptor: int, expected_bytes: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < expected_bytes:
        block = os.pread(descriptor, min(1024 * 1024, expected_bytes - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    if offset != expected_bytes or os.pread(descriptor, 1, offset):
        raise ImportError("sealed parser byte count drifted")
    return b"".join(chunks)


def _sealed_fd_sha256(descriptor: int, expected_bytes: int) -> str:
    digest = sha256()
    offset = 0
    while offset < expected_bytes:
        block = os.pread(descriptor, min(1024 * 1024, expected_bytes - offset), offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    if offset != expected_bytes or os.pread(descriptor, 1, offset):
        raise ImportError("sealed parser byte count drifted")
    return digest.hexdigest()


def _mapped_file_records() -> dict[tuple[int, int, int], set[str]]:
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError as error:
        raise ImportError("cannot inspect process mappings") from error
    records: dict[tuple[int, int, int], set[str]] = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 5 or ":" not in fields[3]:
            continue
        try:
            major_text, minor_text = fields[3].split(":", 1)
            identity = (int(major_text, 16), int(minor_text, 16), int(fields[4]))
        except ValueError:
            continue
        if identity[2] == 0:
            continue
        raw_path = fields[5] if len(fields) == 6 else ""
        mapped_path = _PROC_MAP_ESCAPE.sub(
            lambda match: chr(int(match.group(1), 8)), raw_path
        )
        records.setdefault(identity, set()).add(mapped_path)
    return records


def _immutable_file_record(
    path: str, *, expected_identity: tuple[int, int, int] | None = None
) -> dict[str, Any]:
    if not path.startswith("/") or path.endswith(" (deleted)"):
        raise ImportError(f"mapped dependency path is not immutable: {path}")
    resolved = os.path.realpath(path)
    if not resolved.startswith("/") or not os.path.exists(resolved):
        raise ImportError(f"mapped dependency path is unavailable: {path}")
    current = Path("/")
    for component in Path(resolved).parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise ImportError(f"cannot inspect immutable path: {resolved}") from error
        if (
            metadata.st_uid != 0
            or metadata.st_gid != 0
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ImportError(f"path remains writable outside root: {resolved}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise ImportError(f"cannot open immutable mapped file: {resolved}") from error
    try:
        metadata = os.fstat(descriptor)
        identity = (
            os.major(metadata.st_dev),
            os.minor(metadata.st_dev),
            metadata.st_ino,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or metadata.st_mode
            & (stat.S_IWGRP | stat.S_IWOTH | _FORBIDDEN_RUNTIME_MODE)
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise ImportError(f"immutable mapped file identity drifted: {resolved}")
        try:
            attributes = os.listxattr(resolved, follow_symlinks=False)
        except OSError as error:
            raise ImportError(
                f"cannot inspect immutable mapped file attributes: {resolved}"
            ) from error
        if attributes:
            raise ImportError(f"immutable mapped file has extended attributes: {resolved}")
        return {
            "bytes": metadata.st_size,
            "device_major": identity[0],
            "device_minor": identity[1],
            "inode": identity[2],
            "path": resolved,
            "sha256": _sealed_fd_sha256(descriptor, metadata.st_size),
        }
    finally:
        os.close(descriptor)


def _runtime_root() -> tuple[str, str, dict[str, Any]]:
    raw = os.environ.pop("GATE_D_VALIDATOR_RUNTIME_ROOT", None)
    raw_fd = os.environ.pop("GATE_D_VALIDATOR_PYTHON_FD", None)
    expected_sha = os.environ.pop("GATE_D_VALIDATOR_PYTHON_SHA256", None)
    expected_runtime_sha = os.environ.pop("GATE_D_VALIDATOR_RUNTIME_SHA256", None)
    expected_uid = os.environ.pop("GATE_D_VALIDATOR_UID", None)
    if (
        raw is None
        or raw_fd is None
        or expected_sha is None
        or expected_runtime_sha is None
        or expected_uid is None
        or not re.fullmatch(r"[0-9a-f]{64}", expected_runtime_sha)
    ):
        raise ImportError("immutable Python runtime binding is absent")
    try:
        bound_uid = int(expected_uid)
    except ValueError as error:
        raise ImportError("validator process UID binding is invalid") from error
    if bound_uid <= 0 or os.geteuid() != bound_uid:
        raise ImportError("validator must retain the bound unprivileged UID")
    root = os.path.realpath(raw)
    if not root.startswith("/") or root == "/":
        raise ImportError("immutable Python runtime root is invalid")
    root_record = os.stat(root)
    if (
        not stat.S_ISDIR(root_record.st_mode)
        or root_record.st_uid != 0
        or root_record.st_gid != 0
        or root_record.st_mode
        & (stat.S_IWGRP | stat.S_IWOTH | _FORBIDDEN_RUNTIME_MODE)
    ):
        raise ImportError("immutable Python runtime root remains writable")
    try:
        descriptor = int(raw_fd)
    except ValueError as error:
        raise ImportError("immutable Python descriptor is invalid") from error
    executable = os.path.realpath(f"/proc/self/fd/{descriptor}")
    if os.path.commonpath((root, executable)) != root:
        raise ImportError("immutable Python executable is outside runtime root")
    executable_record = _immutable_file_record(executable)
    if executable_record["sha256"] != expected_sha:
        raise ImportError("immutable Python executable SHA-256 drifted")
    if sys.flags.isolated != 1 or sys.flags.no_site != 1 or os.getcwd() != "/":
        raise ImportError("Python validator process is not isolated")
    for entry in sys.path:
        if not entry or not os.path.isabs(entry):
            raise ImportError("Python validator search path is not isolated")
        resolved = os.path.realpath(entry)
        parent = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)
        if os.path.commonpath((root, parent)) != root:
            raise ImportError("Python validator search path escapes runtime root")
        if os.path.exists(resolved):
            metadata = os.stat(resolved)
            if (
                metadata.st_uid != 0
                or metadata.st_gid != 0
                or metadata.st_mode
                & (stat.S_IWGRP | stat.S_IWOTH | _FORBIDDEN_RUNTIME_MODE)
            ):
                raise ImportError("Python validator search path remains writable")
    return root, expected_runtime_sha, executable_record


(
    _PYTHON_RUNTIME_ROOT,
    _PYTHON_RUNTIME_TREE_SHA256,
    _PYTHON_EXECUTABLE_RECORD,
) = _runtime_root()
_NATIVE_MAPPING_BASELINE = _mapped_file_records()


class _SealedSourceLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, logical_path: str, is_package: bool) -> None:
        self.fullname = fullname
        self.logical_path = logical_path
        self.is_package = is_package

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        binding = _SEALED_BINDINGS[self.logical_path]
        raw = _read_sealed_fd(binding["fd"], binding["bytes"])
        if sha256(raw).hexdigest() != binding["sha256"]:
            raise ImportError(f"sealed parser source drifted: {self.logical_path}")
        module.__file__ = f"<sealed-memfd:{self.logical_path}>"
        if self.is_package:
            module.__path__ = []
        code = compile(raw, module.__file__, "exec", dont_inherit=True)
        exec(code, module.__dict__)
        _LOADED_SOURCE_PATHS.add(self.logical_path)


class _SealedExtensionLoader(importlib.machinery.ExtensionFileLoader):
    def __init__(self, fullname: str, logical_path: str, descriptor: int) -> None:
        super().__init__(fullname, f"/proc/self/fd/{descriptor}")
        self.logical_path = logical_path

    def exec_module(self, module: Any) -> None:
        super().exec_module(module)
        _LOADED_NATIVE_PATHS.add(self.logical_path)


class _SealedParserFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        if fullname in _SEALED_SOURCE_MODULES:
            logical_path, is_package = _SEALED_SOURCE_MODULES[fullname]
            loader = _SealedSourceLoader(fullname, logical_path, is_package)
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=f"sealed-memfd:{logical_path}",
                is_package=is_package,
            )
        if fullname in _SEALED_EXTENSION_MODULES:
            logical_path = _SEALED_EXTENSION_MODULES[fullname]
            descriptor = _SEALED_BINDINGS[logical_path]["fd"]
            loader = _SealedExtensionLoader(fullname, logical_path, descriptor)
            return importlib.util.spec_from_file_location(
                fullname,
                f"/proc/self/fd/{descriptor}",
                loader=loader,
            )
        if fullname in _SEALED_NAMESPACE_MODULES:
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        if fullname == "jaxlib" or fullname.startswith("jaxlib."):
            raise ModuleNotFoundError(f"unsealed parser import refused: {fullname}")
        return None


def _module_binding(logical_path: str) -> tuple[str, bool] | None:
    if logical_path.endswith("/__init__.py"):
        return logical_path[: -len("/__init__.py")].replace("/", "."), True
    if logical_path.endswith(".py"):
        return logical_path[:-3].replace("/", "."), False
    if logical_path.endswith(".so") and not logical_path.endswith("/libjax_common.so"):
        return logical_path[:-3].replace("/", "."), False
    return None


def _install_sealed_parser_importer() -> None:
    raw = os.environ.pop("GATE_D_VALIDATOR_FDS", None)
    if raw is None:
        raise ImportError("sealed parser descriptor manifest is absent")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ImportError("sealed parser descriptor manifest is invalid") from error
    if not isinstance(manifest, dict) or not manifest:
        raise ImportError("sealed parser descriptor manifest is invalid")
    for logical_path, binding in manifest.items():
        if (
            not isinstance(logical_path, str)
            or not logical_path.startswith("jaxlib/")
            or not isinstance(binding, dict)
            or set(binding) != {"bytes", "fd", "sha256"}
            or not isinstance(binding["bytes"], int)
            or isinstance(binding["bytes"], bool)
            or binding["bytes"] <= 0
            or not isinstance(binding["fd"], int)
            or isinstance(binding["fd"], bool)
            or binding["fd"] < 0
            or not isinstance(binding["sha256"], str)
            or len(binding["sha256"]) != 64
        ):
            raise ImportError("sealed parser descriptor binding is invalid")
        descriptor = binding["fd"]
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_size != binding["bytes"]
            or fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEAL_MASK
        ):
            raise ImportError("sealed parser descriptor identity drifted")
        if _sealed_fd_sha256(descriptor, binding["bytes"]) != binding["sha256"]:
            raise ImportError("sealed parser descriptor content drifted")
        _SEALED_BINDINGS[logical_path] = dict(binding)
        module = _module_binding(logical_path)
        if module is None:
            continue
        fullname, is_package = module
        if logical_path.endswith(".py"):
            _SEALED_SOURCE_MODULES[fullname] = (logical_path, is_package)
        else:
            _SEALED_EXTENSION_MODULES[fullname] = logical_path
        components = fullname.split(".")
        for index in range(1, len(components)):
            parent = ".".join(components[:index])
            if parent not in _SEALED_SOURCE_MODULES:
                _SEALED_NAMESPACE_MODULES.add(parent)
    common_path = "jaxlib/libjax_common.so"
    if common_path not in _SEALED_BINDINGS:
        raise ImportError("sealed libjax_common binding is absent")
    common_fd = _SEALED_BINDINGS[common_path]["fd"]
    ctypes.CDLL(f"/proc/self/fd/{common_fd}", mode=ctypes.RTLD_GLOBAL)
    _LOADED_NATIVE_PATHS.add(common_path)
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, _SealedParserFinder())


_install_sealed_parser_importer()

import jaxlib
from jaxlib.mlir import ir
from jaxlib.mlir._mlir_libs import _jax_mlir_ext
from jaxlib.mlir.dialects import stablehlo


_COLLECTIVES_WITH_GROUPS = {
    "stablehlo.all_gather",
    "stablehlo.all_reduce",
    "stablehlo.all_to_all",
    "stablehlo.collective_broadcast",
    "stablehlo.reduce_scatter",
}
_COLLECTIVE_PERMUTE = "stablehlo.collective_permute"
_FORBIDDEN_OPERATIONS = {
    "func.call",
    "stablehlo.custom_call",
    "stablehlo.infeed",
    "stablehlo.outfeed",
    "stablehlo.recv",
    "stablehlo.send",
}


class ValidationError(ValueError):
    """Raised when the parsed module does not satisfy the causal contract."""


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_request() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(96 * 1024 * 1024 + 1)
    if len(raw) > 96 * 1024 * 1024:
        raise ValidationError("validator request is too large")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError("validator request is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValidationError("validator request must be an object")
    expected = {
        "accepted_stablehlo_base64",
        "accepted_stablehlo_sha256",
        "auxiliary_result_index",
        "callsite_ast_sha256",
        "candidate_ast_sha256",
        "candidate_stablehlo_base64",
        "candidate_stablehlo_sha256",
        "carried_residual_result_index",
        "expected_parser_files",
        "local_device_groups",
        "source_set_sha256",
        "weighted_output_result_index",
    }
    if set(value) != expected:
        raise ValidationError("validator request schema drifted")
    return value


def _bytes(value: Any, expected_sha: Any, label: str) -> bytes:
    if not isinstance(value, str) or not isinstance(expected_sha, str):
        raise ValidationError(f"{label} binding is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValidationError(f"{label} is not canonical base64") from error
    if sha256(raw).hexdigest() != expected_sha:
        raise ValidationError(f"{label} SHA-256 drifted")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError(f"{label} is not UTF-8") from error
    if any(marker in text for marker in ("//", "/*", "*/")):
        raise ValidationError(f"{label} contains comments")
    return raw


def _positive_index(value: Any, label: str, *, allow_zero: bool = True) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValidationError(f"{label} is invalid")
    return value


def _local_groups(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError("local device groups are absent")
    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for raw_group in value:
        if not isinstance(raw_group, list) or not 1 <= len(raw_group) <= 4:
            raise ValidationError("local device group size is invalid")
        group = tuple(_positive_index(rank, "local device rank") for rank in raw_group)
        if len(set(group)) != len(group) or any(rank > 31 for rank in group):
            raise ValidationError("local device group ranks are invalid")
        if seen.intersection(group):
            raise ValidationError("local device groups overlap")
        seen.update(group)
        groups.append(group)
    return tuple(groups)


def _loaded_python_runtime_files() -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    sealed_identities = {
        (
            os.major(os.fstat(binding["fd"]).st_dev),
            os.minor(os.fstat(binding["fd"]).st_dev),
            os.fstat(binding["fd"]).st_ino,
        )
        for binding in _SEALED_BINDINGS.values()
    }
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or raw_path.startswith("<sealed-memfd:"):
            continue
        if raw_path.startswith("/proc/self/fd/"):
            try:
                metadata = os.stat(raw_path)
            except OSError as error:
                raise ValidationError("loaded descriptor-backed module vanished") from error
            identity = (
                os.major(metadata.st_dev),
                os.minor(metadata.st_dev),
                metadata.st_ino,
            )
            if identity in sealed_identities:
                continue
            raise ValidationError("unrecognized descriptor-backed module is loaded")
        resolved = os.path.realpath(raw_path)
        if os.path.commonpath((_PYTHON_RUNTIME_ROOT, resolved)) != _PYTHON_RUNTIME_ROOT:
            raise ValidationError(f"Python module escaped immutable runtime: {raw_path}")
        if resolved not in records:
            try:
                records[resolved] = _immutable_file_record(resolved)
            except ImportError as error:
                raise ValidationError(str(error)) from error
    return [records[path] for path in sorted(records)]


def _loaded_parser_files(
    expected: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(expected, dict) or not expected:
        raise ValidationError("parser file manifest is absent")
    if any(
        not isinstance(path, str)
        or not path.startswith("jaxlib/")
        or not isinstance(digest, str)
        or len(digest) != 64
        for path, digest in expected.items()
    ):
        raise ValidationError("parser file manifest is invalid")
    bound = {
        path: binding["sha256"] for path, binding in _SEALED_BINDINGS.items()
    }
    if bound != expected:
        raise ValidationError("sealed parser bindings differ from requested manifest")
    expected_sources = {path for path in expected if path.endswith(".py")}
    expected_native = {path for path in expected if path.endswith(".so")}
    if _LOADED_SOURCE_PATHS != expected_sources:
        raise ValidationError("sealed parser source load set drifted")
    if _LOADED_NATIVE_PATHS != expected_native:
        raise ValidationError("sealed parser native load set drifted")
    try:
        mapped_records = _mapped_file_records()
    except ImportError as error:
        raise ValidationError(str(error)) from error
    mapped_identities = set(mapped_records)
    native_identities: set[tuple[int, int, int]] = set()
    for logical_path, binding in _SEALED_BINDINGS.items():
        descriptor = binding["fd"]
        status = os.fstat(descriptor)
        if (
            status.st_size != binding["bytes"]
            or _sealed_fd_sha256(descriptor, binding["bytes"])
            != binding["sha256"]
            or fcntl.fcntl(descriptor, _F_GET_SEALS) != _MEMFD_SEAL_MASK
        ):
            raise ValidationError("loaded sealed parser descriptor drifted")
        if logical_path.endswith(".so"):
            identity = (os.major(status.st_dev), os.minor(status.st_dev), status.st_ino)
            native_identities.add(identity)
            if identity not in mapped_identities:
                raise ValidationError(
                    f"parser native library is not mapped from its sealed inode: {logical_path}"
                )
    immutable_dependencies: dict[str, dict[str, Any]] = {}
    for identity, paths in mapped_records.items():
        if identity in native_identities:
            continue
        if not paths or any(not path for path in paths):
            raise ValidationError("a pathname-free native file mapping is present")
        for mapped_path in sorted(paths):
            try:
                record = _immutable_file_record(
                    mapped_path, expected_identity=identity
                )
            except ImportError as error:
                raise ValidationError(str(error)) from error
            immutable_dependencies[record["path"]] = record
    newly_mapped = set(mapped_records) - set(_NATIVE_MAPPING_BASELINE)
    return dict(sorted(bound.items())), {
        "immutable_native_dependencies": [
            immutable_dependencies[path] for path in sorted(immutable_dependencies)
        ],
        "memfd_seal_mask": _MEMFD_SEAL_MASK,
        "native_mapped_paths": sorted(_LOADED_NATIVE_PATHS),
        "new_native_mapping_count": len(newly_mapped),
        "python_executable": _PYTHON_EXECUTABLE_RECORD,
        "python_loaded_files": _loaded_python_runtime_files(),
        "process_uid": os.geteuid(),
        "python_runtime_root": _PYTHON_RUNTIME_ROOT,
        "python_runtime_tree_sha256": _PYTHON_RUNTIME_TREE_SHA256,
        "python_search_path": list(sys.path),
        "sealed_native_mapping_count": len(native_identities),
        "source_loaded_paths": sorted(_LOADED_SOURCE_PATHS),
    }


def _operation_name(operation: Any) -> str:
    raw = getattr(operation, "operation", operation)
    return str(raw.name)


def _walk(operation: Any) -> list[Any]:
    result = [operation]
    raw = getattr(operation, "operation", operation)
    for region in raw.regions:
        for block in region.blocks:
            for child in block.operations:
                result.extend(_walk(child))
    return result


def _function(module: ir.Module, label: str) -> tuple[Any, Any, list[Any]]:
    top = list(module.body.operations)
    functions = [item for item in top if _operation_name(item) == "func.func"]
    if len(top) != 1 or len(functions) != 1:
        raise ValidationError(f"{label} must contain exactly one function")
    function = functions[0]
    raw = getattr(function, "operation", function)
    if len(raw.regions) != 1 or len(raw.regions[0].blocks) != 1:
        raise ValidationError(f"{label} function body is not canonical")
    block = raw.regions[0].blocks[0]
    operations = list(block.operations)
    if not operations or _operation_name(operations[-1]) != "func.return":
        raise ValidationError(f"{label} function return is absent")
    return function, block, operations


def _dense_matrix(attribute: Any, label: str) -> list[list[int]]:
    try:
        dense = ir.DenseIntElementsAttr(attribute)
        shape = tuple(ir.RankedTensorType(dense.type).shape)
        values = [int(item) for item in dense]
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} is not a dense integer matrix") from error
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValidationError(f"{label} shape is invalid")
    return [values[index * shape[1] : (index + 1) * shape[1]] for index in range(shape[0])]


def _within_one_local_group(ranks: Sequence[int], groups: Sequence[Sequence[int]]) -> bool:
    requested = set(ranks)
    return bool(requested) and any(requested.issubset(set(group)) for group in groups)


def _permute_components(pairs: Sequence[Sequence[int]]) -> list[list[int]]:
    graph: dict[int, set[int]] = {}
    for source, target in pairs:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set()).add(source)
    components: list[list[int]] = []
    remaining = set(graph)
    while remaining:
        root = min(remaining)
        pending = [root]
        component: set[int] = set()
        while pending:
            rank = pending.pop()
            if rank in component:
                continue
            component.add(rank)
            pending.extend(graph[rank] - component)
        remaining -= component
        components.append(sorted(component))
    return components


def _collectives(
    module: ir.Module,
    groups: Sequence[Sequence[int]],
    label: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _walk(module.operation):
        name = _operation_name(item)
        dialect = name.split(".", 1)[0]
        if dialect not in {"builtin", "func", "stablehlo"}:
            raise ValidationError(f"{label} contains an unknown dialect: {name}")
        if name in _FORBIDDEN_OPERATIONS:
            raise ValidationError(f"{label} contains a forbidden operation: {name}")
        collective_tokens = (
            "collective",
            "all_gather",
            "all_reduce",
            "all_to_all",
            "reduce_scatter",
        )
        if (
            any(token in name for token in collective_tokens)
            and name not in _COLLECTIVES_WITH_GROUPS | {_COLLECTIVE_PERMUTE}
        ):
            raise ValidationError(f"{label} contains an unsupported collective: {name}")
        raw = getattr(item, "operation", item)
        if name in _COLLECTIVES_WITH_GROUPS:
            if "replica_groups" not in raw.attributes:
                raise ValidationError(f"{label} collective groups are absent: {name}")
            replica_groups = _dense_matrix(raw.attributes["replica_groups"], f"{name} groups")
            for group in replica_groups:
                if (
                    len(set(group)) != len(group)
                    or any(rank < 0 or rank > 31 for rank in group)
                    or not _within_one_local_group(group, groups)
                ):
                    raise ValidationError(f"{label} collective leaves a plan-local group: {name}")
            records.append({"groups": replica_groups, "operation": name})
        elif name == _COLLECTIVE_PERMUTE:
            if "source_target_pairs" not in raw.attributes:
                raise ValidationError(f"{label} collective-permute pairs are absent")
            pairs = _dense_matrix(raw.attributes["source_target_pairs"], "collective-permute pairs")
            if any(
                len(pair) != 2
                or pair[0] == pair[1]
                or any(rank < 0 or rank > 31 for rank in pair)
                for pair in pairs
            ):
                raise ValidationError(f"{label} collective-permute pairs are invalid")
            components = _permute_components(pairs)
            if any(not _within_one_local_group(component, groups) for component in components):
                raise ValidationError(f"{label} collective-permute leaves a plan-local group")
            records.append({"components": components, "operation": name, "pairs": pairs})
    return records


def _result_index(value: Any) -> int:
    owner = value.owner
    for index, result in enumerate(owner.results):
        if result == value:
            return index
    raise ValidationError("operation result identity is inconsistent")


def _region_signature(operation: Any) -> list[Any]:
    raw = getattr(operation, "operation", operation)
    regions: list[Any] = []
    for region in raw.regions:
        blocks: list[Any] = []
        for block in region.blocks:
            operations = list(block.operations)

            def local_reference(value: Any) -> Any:
                if isinstance(value, ir.BlockArgument):
                    if value.owner != block:
                        raise ValidationError("nested region captures an external block argument")
                    return ["block_argument", int(value.arg_number), str(value.type)]
                owner = value.owner
                if owner not in operations:
                    raise ValidationError("nested region captures an external operation result")
                return [
                    "local_result",
                    operations.index(owner),
                    _result_index(value),
                    str(value.type),
                ]

            blocks.append(
                {
                    "arguments": [str(argument.type) for argument in block.arguments],
                    "operations": [
                        {
                            "attributes": {
                                str(key): str(child.attributes[key])
                                for key in sorted(child.attributes)
                            },
                            "name": _operation_name(child),
                            "operands": [local_reference(item) for item in child.operands],
                            "results": [str(result.type) for result in child.results],
                        }
                        for child in operations
                    ],
                }
            )
        regions.append(blocks)
    return regions


def _slice_signature(value: Any, root_block: Any, memo: dict[Any, Any]) -> Any:
    if value in memo:
        return memo[value]
    if isinstance(value, ir.BlockArgument):
        if value.owner != root_block:
            raise ValidationError("primary slice crosses a nested-region argument")
        signature: Any = ["argument", int(value.arg_number), str(value.type)]
    else:
        owner = value.owner
        name = _operation_name(owner)
        attributes = {
            str(key): str(owner.attributes[key]) for key in sorted(owner.attributes)
        }
        signature = [
            "result",
            name,
            _result_index(value),
            str(value.type),
            attributes,
            [_slice_signature(item, root_block, memo) for item in owner.operands],
            _region_signature(owner),
        ]
    memo[value] = signature
    return signature


def _auxiliary_slice_signature(
    value: Any,
    source: Any,
    root_block: Any,
    memo: dict[Any, Any],
) -> Any:
    if value == source:
        return ["rms_frontier_fp32_sum", str(value.type)]
    if value in memo:
        return memo[value]
    if isinstance(value, ir.BlockArgument):
        if value.owner != root_block:
            raise ValidationError("auxiliary slice crosses a nested-region argument")
        signature: Any = ["argument", int(value.arg_number), str(value.type)]
    else:
        owner = value.owner
        if any(True for _ in owner.regions):
            raise ValidationError("auxiliary slice contains a region operation")
        signature = [
            "result",
            _operation_name(owner),
            _result_index(value),
            str(value.type),
            {str(key): str(owner.attributes[key]) for key in sorted(owner.attributes)},
            [
                _auxiliary_slice_signature(item, source, root_block, memo)
                for item in owner.operands
            ],
        ]
    memo[value] = signature
    return signature


def _depends_on(value: Any, source: Any, visited: set[Any]) -> bool:
    if value == source:
        return True
    if value in visited or isinstance(value, ir.BlockArgument):
        return False
    visited.add(value)
    return any(_depends_on(item, source, visited) for item in value.owner.operands)


def _reachable_operations(values: Sequence[Any], root_block: Any) -> set[Any]:
    reachable: set[Any] = set()
    visited: set[Any] = set()

    def visit(value: Any) -> None:
        if value in visited:
            return
        visited.add(value)
        if isinstance(value, ir.BlockArgument):
            if value.owner != root_block:
                raise ValidationError("root slice crosses a nested-region argument")
            return
        owner = value.owner
        reachable.add(owner)
        for operand in owner.operands:
            visit(operand)

    for value in values:
        visit(value)
    return reachable


def _candidate_module_metadata(module: ir.Module, request: Mapping[str, Any]) -> None:
    expected = {
        "gate_d.callsite_ast_sha256": request["callsite_ast_sha256"],
        "gate_d.candidate_ast_sha256": request["candidate_ast_sha256"],
        "gate_d.source_set_sha256": request["source_set_sha256"],
    }
    attributes = module.operation.attributes
    observed: dict[str, str] = {}
    for key in attributes:
        name = str(key)
        if name.startswith("gate_d."):
            try:
                observed[name] = ir.StringAttr(attributes[key]).value
            except (TypeError, ValueError) as error:
                raise ValidationError("candidate source metadata is not string-valued") from error
    if observed != expected:
        raise ValidationError("candidate source metadata drifted")


def _causal_operations(value: Any, source: Any, visited: set[Any]) -> list[str]:
    if value == source or value in visited or isinstance(value, ir.BlockArgument):
        return []
    visited.add(value)
    owner = value.owner
    causal_operands = [
        item for item in owner.operands if _depends_on(item, source, set())
    ]
    if not causal_operands:
        return []
    result = [_operation_name(owner)]
    for operand in causal_operands:
        result.extend(_causal_operations(operand, source, visited))
    return result


def _parse(raw: bytes, label: str) -> tuple[ir.Context, ir.Module]:
    registry = ir.DialectRegistry()
    _jax_mlir_ext.register_dialects(registry)
    context = ir.Context()
    context.append_dialect_registry(registry)
    context.load_all_available_dialects()
    stablehlo.register_dialect(context)
    try:
        module = ir.Module.parse(raw.decode("utf-8"), context=context)
        if module.operation.verify() is False:
            raise ValidationError(f"{label} MLIR verification failed")
    except (ValueError, ir.MLIRError) as error:
        raise ValidationError(f"{label} is not valid MLIR/StableHLO") from error
    return context, module


def validate(request: Mapping[str, Any]) -> dict[str, Any]:
    candidate_raw = _bytes(
        request["candidate_stablehlo_base64"],
        request["candidate_stablehlo_sha256"],
        "candidate StableHLO",
    )
    accepted_raw = _bytes(
        request["accepted_stablehlo_base64"],
        request["accepted_stablehlo_sha256"],
        "accepted-primary StableHLO",
    )
    groups = _local_groups(request["local_device_groups"])
    weighted_output_index = _positive_index(
        request["weighted_output_result_index"], "weighted-output result index"
    )
    carried_residual_index = _positive_index(
        request["carried_residual_result_index"], "carried-residual result index"
    )
    auxiliary_index = _positive_index(
        request["auxiliary_result_index"], "auxiliary result index"
    )
    candidate_context, candidate = _parse(candidate_raw, "candidate StableHLO")
    accepted_context, accepted = _parse(accepted_raw, "accepted-primary StableHLO")
    try:
        _candidate_module_metadata(candidate, request)
        _, candidate_block, candidate_ops = _function(candidate, "candidate StableHLO")
        _, accepted_block, accepted_ops = _function(accepted, "accepted-primary StableHLO")
        candidate_returns = list(candidate_ops[-1].operands)
        accepted_returns = list(accepted_ops[-1].operands)
        expected_arguments = ["tensor<1x6144xbf16>"] * 3
        if (
            [str(argument.type) for argument in candidate_block.arguments]
            != expected_arguments
            or [str(argument.type) for argument in accepted_block.arguments]
            != expected_arguments
            or len(candidate_returns) != 3
            or len(accepted_returns) != 2
            or (
                weighted_output_index,
                carried_residual_index,
                auxiliary_index,
            )
            != (0, 1, 2)
        ):
            raise ValidationError("function argument/result contract is invalid")
        candidate_body = candidate_ops[:-1]
        accepted_body = accepted_ops[:-1]
        weighted_output = candidate_returns[weighted_output_index]
        carried_residual = candidate_returns[carried_residual_index]
        carried_owner = carried_residual.owner
        if (
            _operation_name(carried_owner) != "stablehlo.convert"
            or len(carried_owner.operands) != 1
            or str(carried_residual.type) != "tensor<1x6144xbf16>"
            or str(weighted_output.type) != "tensor<1x6144xbf16>"
        ):
            raise ValidationError("accepted RMS primary result types drifted")
        source = carried_owner.operands[0]
        source_operation = source.owner
        if (
            _operation_name(source_operation) != "stablehlo.add"
            or str(source.type) != "tensor<1x6144xf32>"
            or source_operation not in candidate_body
        ):
            raise ValidationError("auxiliary source is not the exact RMS transient FP32 sum")
        if not _depends_on(weighted_output, source, set()):
            raise ValidationError("weighted RMS output does not consume the transient FP32 sum")
        source_operation_index = candidate_body.index(source_operation)
        source_result_index = _result_index(source)
        if not _depends_on(candidate_returns[auxiliary_index], source, set()):
            raise ValidationError("auxiliary result is not causally dependent on its source")
        auxiliary_operations = _causal_operations(
            candidate_returns[auxiliary_index], source, set()
        )
        auxiliary_signature = _auxiliary_slice_signature(
            candidate_returns[auxiliary_index], source, candidate_block, {}
        )
        auxiliary_signature_sha = sha256(
            _canonical({"auxiliary": auxiliary_signature}).encode("ascii")
        ).hexdigest()
        candidate_signatures = {
            "carried_residual": _slice_signature(carried_residual, candidate_block, {}),
            "weighted_output": _slice_signature(weighted_output, candidate_block, {}),
        }
        accepted_signatures = {
            "carried_residual": _slice_signature(accepted_returns[1], accepted_block, {}),
            "weighted_output": _slice_signature(accepted_returns[0], accepted_block, {}),
        }
        candidate_signature_shas = {
            key: sha256(_canonical({key: value}).encode("ascii")).hexdigest()
            for key, value in candidate_signatures.items()
        }
        accepted_signature_shas = {
            key: sha256(_canonical({key: value}).encode("ascii")).hexdigest()
            for key, value in accepted_signatures.items()
        }
        if candidate_signature_shas != accepted_signature_shas:
            raise ValidationError("candidate RMS primary slices differ from accepted primary")
        candidate_signature_sha = sha256(
            _canonical(candidate_signature_shas).encode("ascii")
        ).hexdigest()
        accepted_signature_sha = sha256(
            _canonical(accepted_signature_shas).encode("ascii")
        ).hexdigest()
        if candidate_signature_sha != accepted_signature_sha:
            raise ValidationError("candidate RMS primary authority drifted")
        if _reachable_operations(candidate_returns, candidate_block) != set(candidate_body):
            raise ValidationError("candidate StableHLO contains dead or unrooted operations")
        if _reachable_operations(accepted_returns, accepted_block) != set(accepted_body):
            raise ValidationError("accepted StableHLO contains dead or unrooted operations")
        candidate_collectives = _collectives(candidate, groups, "candidate StableHLO")
        accepted_collectives = _collectives(accepted, groups, "accepted-primary StableHLO")
        loaded_parser_files, parser_runtime_authority = _loaded_parser_files(
            request["expected_parser_files"]
        )
        return {
            "accepted_primary_slice_sha256": accepted_signature_sha,
            "accepted_primary_slice_sha256s": accepted_signature_shas,
            "accepted_stablehlo_sha256": request["accepted_stablehlo_sha256"],
            "auxiliary_result_index": auxiliary_index,
            "auxiliary_path_operations": auxiliary_operations,
            "auxiliary_slice_sha256": auxiliary_signature_sha,
            "auxiliary_source_operation_index": source_operation_index,
            "auxiliary_source_result_index": source_result_index,
            "candidate_primary_slice_sha256": candidate_signature_sha,
            "candidate_primary_slice_sha256s": candidate_signature_shas,
            "candidate_stablehlo_sha256": request["candidate_stablehlo_sha256"],
            "collectives": candidate_collectives,
            "jax_imported": "jax" in sys.modules,
            "jaxlib_version": jaxlib.__version__,
            "loaded_parser_files": loaded_parser_files,
            "immutable_parser_authority": True,
            "local_device_groups": [list(group) for group in groups],
            "parser": "jaxlib.mlir.ir",
            "parser_authority_scope": (
                "root-owned isolated Python plus sealed-memfd exact-fd parser "
                "and mapped-inode native authority"
            ),
            "parser_runtime_authority": parser_runtime_authority,
            "carried_residual_result_index": carried_residual_index,
            "weighted_output_result_index": weighted_output_index,
            "stablehlo_version": stablehlo.get_current_version(),
            "accepted_collectives": accepted_collectives,
        }
    finally:
        del candidate
        del accepted
        del candidate_context
        del accepted_context


def main() -> int:
    try:
        report = validate(_load_request())
    except (ValidationError, KeyError, TypeError) as error:
        print(f"StableHLO validation refused: {error}", file=sys.stderr)
        return 1
    print(_canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
