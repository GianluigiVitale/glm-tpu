"""Append-only precompile admission v2 for Gate-D mechanisms.

Version 1 remains the authority for candidates that already have an executable
identity.  This separate path keeps the parent auditor stdlib-only and delegates
parsing to a sealed jaxlib/MLIR child.  It represents the earlier, precompile
state honestly: exact source AST, plan authority, causal StableHLO, and one
candidate-coherent typed capsule.  Admission never authorizes compilation or
TPU execution.
"""

from __future__ import annotations

import ast
import base64
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import struct
import subprocess
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
import zipfile

from .errors import BenchmarkValidationError


__all__ = (
    "GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION",
    "admit_gate_d_precompile_candidates",
    "write_gate_d_precompile_admission_report",
)


GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_PIN = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_STABLEHLO_BYTES = 32 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_PARSER_TOTAL_BYTES = 512 * 1024 * 1024
_EXPECTED_JAXLIB_VERSION = "0.10.1"
_EXPECTED_STABLEHLO_VERSION = "1.17.0"
_EXPECTED_ACCEPTED_PRIMARY_SLICE_SHA256 = (
    "91c947fc934ab00b44be4c4b7ece62c2a7905b337e05ee78012047ba2b007737"
)
_EXPECTED_TOPOLOGY_HASH = (
    "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
)
_EXPECTED_PP16_LP2_HASH = (
    "6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21"
)
_EXPECTED_PP8_LP4_HASH = (
    "d5943ab8d7a074677d82f8e823c8bc983847f8df1deefdee1fbda8da98923c14"
)
_EXPECTED_TOPOLOGY_SOURCE_CONTRACT_HASH = (
    "74e5eafaf264b2d12663f653df469cf3feb8ead1f31be6217797c96e99864ad6"
)
_EXPECTED_ACCEPTED_SOURCE = {
    "ast_sha256": "490175a1e0732b4967b3e53a8b88df081c20855bae009425d6ba81c48422ed59",
    "file_sha256": "d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910",
    "git_commit": "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c",
    "path": "vllm/ir/ops/layernorm.py",
    "symbol": "fused_add_rms_norm",
}
_EXPECTED_VALIDATOR_IMPORTS = {
    "jaxlib/__init__.py": (629, "2a5b37b2bc9802769f45ca43de7cdc1a8b532f0afd2bf9542131ab68f649c4a3"),
    "jaxlib/libjax_common.so": (
        348039992,
        "b836d25d48b50c3e32f8344f19a3cbff25b3e5840d1500d85024ef3dfaf55fbd",
    ),
    "jaxlib/mlir/_mlir_libs/__init__.py": (
        8512,
        "1e053fc5d003af7296a29f55ae8079f10a81c36b35819e03b283fe5dbb8e6dea",
    ),
    "jaxlib/mlir/_mlir_libs/_jax_mlir_ext.so": (
        3632,
        "7a92d593271ab74d921de638bc2ee50dd90d9f191fc42cfe927e5d34321ff9c8",
    ),
    "jaxlib/mlir/_mlir_libs/_mlir.so": (
        3592,
        "ee42e2f7d805b9cbed32cf002320092cfa75a95f901694274ce1f41a76c5b929",
    ),
    "jaxlib/mlir/_mlir_libs/_stablehlo.so": (
        3624,
        "e21b420f7a3d58f8594ac4ab23f59ce3bbd9442aea7b775640e33064fe7a0ffd",
    ),
    "jaxlib/mlir/dialects/_ods_common.py": (
        10887,
        "52f38ee6701961f4840d52958feb128f3fcc31a0c41ef586bd70a6e63d95b7a7",
    ),
    "jaxlib/mlir/dialects/_stablehlo_ops_gen.py": (
        412837,
        "4a4a1ac5b161a5d65862838fc2bc9d63b5d7db554e82a3ae377313bc95b007dd",
    ),
    "jaxlib/mlir/dialects/stablehlo.py": (
        1133,
        "912840177a1c2945155e0f0841b4fde386315a1bca83d0f63cfc0439a10a21e7",
    ),
    "jaxlib/mlir/ir.py": (
        12758,
        "d4fb18407e74c6d496eeeaeda99a9be354f5ffbd9edad41b8498525c4ebfdaf9",
    ),
    "jaxlib/version.py": (6848, "6cecd708aacaa994c28a11cfcdcc56e9e67e64e4d1dcfc3ac7b85b87585c1bc0"),
}
_EXPECTED_SOURCE_SEMANTIC_SHA256 = {
    "auxiliary_device_tuple_dependency": (
        "fe303c7aeb51032f23a90ad61fcf3cb72030a376bc6f9bf20f7be794ce3b7534"
    ),
    "compensated_auxiliary_dependency": (
        "46654fcde1c8e9e6e19ef3bd530db3d6e085c0e5951cbb8c34997a9fafec6dc0"
    ),
}
_EXPECTED_AUXILIARY_SLICE_SHA256 = {
    "auxiliary_device_tuple_dependency": (
        "02eee1d7a84fd447396577c21c1d2988f23d9b6312b8fa0df4838a9b41642f7c"
    ),
    "compensated_auxiliary_dependency": (
        "d55e5f45120080dd0a6f81d3f3dddb9c8a738b59e2454ec1f9d87b9bce5d9a43"
    ),
}
_DTYPE_BYTES = {
    "|b1": 1,
    "<u2": 2,
    "<u4": 4,
    "<i4": 4,
    "<f4": 4,
}
_SEMANTIC_DTYPES = {
    "bf16_bits",
    "float32",
    "int32",
    "uint32",
}
_FINGERPRINT_KEYS = {
    "association",
    "consumer_boundary",
    "reduction",
    "representation",
    "transport",
}
_FRONTIER_ACTIONS = {"expose", "resolve"}
_EXPECTED_EVIDENCE_IDS = {
    "constructability",
    "direct_shadow_rejection",
    "observability_frontier",
    "shadow_adjudication",
}
_REQUIRED_WATCHPOINT_SCHEMA: dict[str, dict[str, Any]] = {
    "layer1.rms_operands_bf16": {
        "arrays": {
            "hidden_update": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "bf16_bits",
                "shape": [6144],
                "storage_dtype": "<u2",
            },
            "residual": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "bf16_bits",
                "shape": [6144],
                "storage_dtype": "<u2",
            },
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.rms_input_fp32": {
        "arrays": {
            "value": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "float32",
                "shape": [6144],
                "storage_dtype": "<f4",
            }
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.normalized": {
        "arrays": {
            "owner0": {
                "index_prefix": [0, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 0,
                "semantic_dtype": "bf16_bits",
                "shape": [6144],
                "storage_dtype": "<u2",
            },
            "owner1": {
                "index_prefix": [1, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 1,
                "semantic_dtype": "bf16_bits",
                "shape": [6144],
                "storage_dtype": "<u2",
            },
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.cache_history": {
        "arrays": {
            "value": {
                "index_prefix": [],
                "owner_axis": 0,
                "owner_axis_slots": [0, 1],
                "owner_slot": None,
                "semantic_dtype": "bf16_bits",
                "shape": [2, 16, 256, 128],
                "storage_dtype": "<u2",
            }
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.query": {
        "arrays": {
            "owner0": {
                "index_prefix": [0, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 0,
                "semantic_dtype": "float32",
                "shape": [32, 128],
                "storage_dtype": "<f4",
            },
            "owner1": {
                "index_prefix": [1, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 1,
                "semantic_dtype": "float32",
                "shape": [32, 128],
                "storage_dtype": "<f4",
            },
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.head_weights": {
        "arrays": {
            "owner0": {
                "index_prefix": [0, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 0,
                "semantic_dtype": "float32",
                "shape": [32],
                "storage_dtype": "<f4",
            },
            "owner1": {
                "index_prefix": [1, 0],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": 1,
                "semantic_dtype": "float32",
                "shape": [32],
                "storage_dtype": "<f4",
            },
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.current_key": {
        "arrays": {
            "value": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "float32",
                "shape": [128],
                "storage_dtype": "<f4",
            }
        },
        "layer": 1,
        "position": 8155,
    },
    "layer1.scorer_event1": {
        "arrays": {
            "positions": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "int32",
                "shape": [1, 2048],
                "storage_dtype": "<i4",
            },
            "scores": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "float32",
                "shape": [1, 2048],
                "storage_dtype": "<f4",
            },
            "valid_count": {
                "index_prefix": [],
                "owner_axis": None,
                "owner_axis_slots": [],
                "owner_slot": None,
                "semantic_dtype": "int32",
                "shape": [1],
                "storage_dtype": "<i4",
            },
        },
        "layer": 1,
        "position": 8155,
    },
}


def _required_watchpoint_schema(owner_count: int) -> dict[str, dict[str, Any]]:
    """Return the exact capsule schema for one complete LP2/LP4 owner group."""

    if owner_count not in (2, 4):
        raise BenchmarkValidationError("watchpoint owner count is not LP2/LP4")
    schema = json.loads(
        json.dumps(_REQUIRED_WATCHPOINT_SCHEMA, allow_nan=False, ensure_ascii=True)
    )
    for watchpoint_id in (
        "layer1.normalized",
        "layer1.query",
        "layer1.head_weights",
    ):
        prototype = schema[watchpoint_id]["arrays"]["owner0"]
        schema[watchpoint_id]["arrays"] = {
            f"owner{slot}": {
                **prototype,
                "index_prefix": [slot, 0],
                "owner_slot": slot,
            }
            for slot in range(owner_count)
        }
    cache = schema["layer1.cache_history"]["arrays"]["value"]
    cache["owner_axis_slots"] = list(range(owner_count))
    cache["shape"] = [owner_count, 16, 256, 128]
    return schema


def _serialized_watchpoint_schema(owner_count: int) -> list[dict[str, Any]]:
    schema = _required_watchpoint_schema(owner_count)
    return [
        {
            "arrays": [
                {"role": role, **specification}
                for role, specification in sorted(item["arrays"].items())
            ],
            "id": watchpoint_id,
            "layer": item["layer"],
            "position": item["position"],
        }
        for watchpoint_id, item in schema.items()
    ]
_EXPECTED_SURVIVORS: dict[str, dict[str, Any]] = {
    "auxiliary_device_tuple_dependency": {
        "mechanism_fingerprint": {
            "association": "plan.local.shadow",
            "consumer_boundary": "device.auxiliary",
            "reduction": "local.fp32",
            "representation": "tuple.bf16.fp32",
            "transport": "stage.local",
        },
        "normal_form": {
            "compensation": "none",
            "dependency": "device.auxiliary",
            "normalization_input": "persistent.shadow",
            "primary_recurrence": "bf16.rounded",
        },
    },
    "compensated_auxiliary_dependency": {
        "mechanism_fingerprint": {
            "association": "plan.local.compensated",
            "consumer_boundary": "device.auxiliary",
            "reduction": "cancelled.local.fp32",
            "representation": "compensated.bf16.fp32",
            "transport": "stage.local",
        },
        "normal_form": {
            "compensation": "cancelled",
            "dependency": "device.auxiliary",
            "normalization_input": "persistent.shadow",
            "primary_recurrence": "bf16.rounded",
        },
    },
}


def _canonical_json(value: Mapping[str, Any]) -> str:
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
            raise BenchmarkValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BenchmarkValidationError(f"non-finite JSON value: {value}")


@contextmanager
def _open_directory_no_symlinks(path: Path, label: str) -> Iterator[int]:
    normalized = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open("/", flags)
    except OSError as error:
        raise BenchmarkValidationError(
            f"cannot safely open {label}: {normalized}"
        ) from error
    try:
        for component in normalized.parts:
            if component in ("", ".", "/"):
                continue
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise BenchmarkValidationError(
                    f"cannot safely open {label}: {normalized}"
                ) from error
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _open_regular_file(path: Path, label: str) -> Iterator[BinaryIO]:
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not normalized.name or normalized.name in (".", ".."):
        raise BenchmarkValidationError(f"{label} path is invalid: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    with _open_directory_no_symlinks(normalized.parent, f"{label} parent") as parent:
        try:
            descriptor = os.open(normalized.name, flags, dir_fd=parent)
        except OSError as error:
            raise BenchmarkValidationError(
                f"cannot open {label}: {normalized}"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise BenchmarkValidationError(
                    f"{label} is not a regular file: {normalized}"
                )
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                yield stream
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _snapshot(
    path: Path, expected_sha256: str, label: str, *, limit: int
) -> bytes:
    digest = sha256()
    blocks: list[bytes] = []
    total = 0
    with _open_regular_file(path, label) as stream:
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > limit:
                raise BenchmarkValidationError(f"{label} is too large: {path}")
            digest.update(block)
            blocks.append(block)
    if digest.hexdigest() != expected_sha256:
        raise BenchmarkValidationError(f"{label} SHA-256 drifted: {path}")
    return b"".join(blocks)


def _load_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkValidationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise BenchmarkValidationError(
            f"{label} keys drifted: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _code_pin(value: Any, label: str) -> str:
    if not isinstance(value, str) or _CODE_PIN.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a full Git SHA")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} is not a canonical identifier")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkValidationError(f"{label} must be boolean")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BenchmarkValidationError(f"{label} must be a non-negative integer")
    return value


def _resolve(base: Path, value: Any, label: str) -> Path:
    raw = _string(value, label)
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _binding(
    base: Path, value: Any, label: str, *, limit: int
) -> tuple[Path, str, bytes]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    _exact_keys(value, {"path", "sha256"}, label)
    expected = _sha(value["sha256"], f"{label} SHA-256")
    path = _resolve(base, value["path"], f"{label} path")
    return path, expected, _snapshot(path, expected, label, limit=limit)


def _mechanism_fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    _exact_keys(value, _FINGERPRINT_KEYS, label)
    canonical = {
        key: _identifier(value[key], f"{label}.{key}")
        for key in sorted(_FINGERPRINT_KEYS)
    }
    return sha256(_canonical_json(canonical).encode("ascii")).hexdigest()


def _verify_locality(value: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    keys = {
        "logical_rows",
        "max_collective_group_size",
        "no_full_pod_hidden_reconstruction",
        "no_host_effects",
    }
    _exact_keys(value, keys, label)
    if {
        "logical_rows": _positive_int(value["logical_rows"], f"{label} rows"),
        "max_collective_group_size": _positive_int(
            value["max_collective_group_size"], f"{label} group size"
        ),
        "no_full_pod_hidden_reconstruction": _boolean(
            value["no_full_pod_hidden_reconstruction"], f"{label} full-pod rule"
        ),
        "no_host_effects": _boolean(
            value["no_host_effects"], f"{label} host rule"
        ),
    } != dict(expected):
        raise BenchmarkValidationError(f"{label} disagrees with contract")


def _device_groups(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty list")
    groups: list[list[int]] = []
    seen: set[int] = set()
    for index, raw_group in enumerate(value):
        if not isinstance(raw_group, list) or not 1 <= len(raw_group) <= 4:
            raise BenchmarkValidationError(f"{label}[{index}] size is invalid")
        group = [
            _nonnegative_int(rank, f"{label}[{index}] rank") for rank in raw_group
        ]
        if (
            len(set(group)) != len(group)
            or any(rank > 31 for rank in group)
            or seen.intersection(group)
        ):
            raise BenchmarkValidationError(f"{label}[{index}] ranks are invalid")
        seen.update(group)
        groups.append(group)
    return groups


def _topology_devices(value: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("runtime topology must be an object")
    _exact_keys(
        value,
        {"devices", "slice_name", "topology_shape"},
        "runtime topology",
    )
    if value["slice_name"] != "db-v4-64-od" or value["topology_shape"] != [2, 4, 4]:
        raise BenchmarkValidationError("runtime topology identity drifted")
    raw_devices = value["devices"]
    if not isinstance(raw_devices, list) or len(raw_devices) != 32:
        raise BenchmarkValidationError("runtime topology device catalogue drifted")
    devices: dict[int, dict[str, Any]] = {}
    coordinates: set[tuple[int, int, int]] = set()
    process_slots: dict[int, set[int]] = {}
    for index, item in enumerate(raw_devices):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"runtime topology device {index} is invalid")
        _exact_keys(
            item,
            {
                "coordinates",
                "core_on_chip",
                "device_id",
                "device_kind",
                "local_device_id",
                "platform",
                "process_index",
            },
            f"runtime topology device {index}",
        )
        device_id = _nonnegative_int(item["device_id"], f"device {index} id")
        process_index = _nonnegative_int(
            item["process_index"], f"device {index} process"
        )
        local_device_id = _nonnegative_int(
            item["local_device_id"], f"device {index} local id"
        )
        raw_coordinates = item["coordinates"]
        if (
            not isinstance(raw_coordinates, list)
            or len(raw_coordinates) != 3
            or any(
                not isinstance(coordinate, int)
                or isinstance(coordinate, bool)
                or coordinate < 0
                or coordinate >= value["topology_shape"][axis]
                for axis, coordinate in enumerate(raw_coordinates)
            )
        ):
            raise BenchmarkValidationError(f"device {index} coordinates are invalid")
        coordinate_tuple = tuple(raw_coordinates)
        if (
            device_id in devices
            or coordinate_tuple in coordinates
            or device_id > 31
            or process_index > 7
            or local_device_id > 3
            or item["core_on_chip"] != 0
            or item["device_kind"] != "TPU v4"
            or item["platform"] != "tpu"
        ):
            raise BenchmarkValidationError(f"runtime topology device {index} drifted")
        coordinates.add(coordinate_tuple)
        process_slots.setdefault(process_index, set()).add(local_device_id)
        devices[device_id] = {
            **item,
            "coordinates": list(raw_coordinates),
        }
    if (
        set(devices) != set(range(32))
        or set(process_slots) != set(range(8))
        or any(slots != set(range(4)) for slots in process_slots.values())
    ):
        raise BenchmarkValidationError("runtime topology process ownership drifted")
    return devices


def _stage_groups(
    value: Any,
    *,
    plan: str,
    group_size: int,
    devices: Mapping[int, Mapping[str, Any]],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{plan} stage authority must be an object")
    _exact_keys(value, {"groups", "plan"}, f"{plan} stage authority")
    raw_groups = value["groups"]
    expected_count = 32 // group_size
    if value["plan"] != plan or not isinstance(raw_groups, list) or len(raw_groups) != expected_count:
        raise BenchmarkValidationError(f"{plan} stage catalogue drifted")
    records: list[dict[str, Any]] = []
    flat_devices: list[int] = []
    for stage_id, record in enumerate(raw_groups):
        if not isinstance(record, dict):
            raise BenchmarkValidationError(f"{plan} stage {stage_id} is invalid")
        _exact_keys(
            record,
            {"coordinates", "device_ids", "process_index", "stage_id"},
            f"{plan} stage {stage_id}",
        )
        device_ids = record["device_ids"]
        coordinates = record["coordinates"]
        process_index = _nonnegative_int(
            record["process_index"], f"{plan} stage {stage_id} process"
        )
        if (
            record["stage_id"] != stage_id
            or not isinstance(device_ids, list)
            or len(device_ids) != group_size
            or any(
                not isinstance(device_id, int)
                or isinstance(device_id, bool)
                or device_id not in devices
                for device_id in device_ids
            )
            or len(set(device_ids)) != group_size
            or not isinstance(coordinates, list)
            or coordinates != [devices[device_id]["coordinates"] for device_id in device_ids]
            or any(devices[device_id]["process_index"] != process_index for device_id in device_ids)
        ):
            raise BenchmarkValidationError(f"{plan} stage {stage_id} ownership drifted")
        coordinate_tuples = [tuple(item) for item in coordinates]
        if plan == "PP16_LP2":
            first, second = coordinate_tuples
            coordinate_local = (
                first[1:] == second[1:] and {first[0], second[0]} == {0, 1}
            )
        else:
            coordinate_local = (
                len(set(coordinate_tuples)) == 4
                and len({item[2] for item in coordinate_tuples}) == 1
                and {item[0] for item in coordinate_tuples} == {0, 1}
                and len({item[1] for item in coordinate_tuples}) == 2
                and max(item[1] for item in coordinate_tuples)
                - min(item[1] for item in coordinate_tuples)
                == 1
            )
        if not coordinate_local:
            raise BenchmarkValidationError(f"{plan} stage {stage_id} is not coordinate-local")
        flat_devices.extend(device_ids)
        records.append(
            {
                "coordinates": coordinates,
                "device_ids": device_ids,
                "process_index": process_index,
                "stage_id": stage_id,
            }
        )
    if sorted(flat_devices) != list(range(32)):
        raise BenchmarkValidationError(f"{plan} stage ownership is incomplete")
    return [record["device_ids"] for record in records], records


def _verify_physical_locality(value: Any, base: Path) -> dict[str, Any]:
    path, file_sha, raw = _binding(
        base,
        value,
        "runtime physical locality authority",
        limit=_MAX_JSON_BYTES,
    )
    document = _load_json(raw, "runtime physical locality authority")
    _exact_keys(
        document,
        {
            "artifact_kind",
            "claim_scope",
            "db_run_id",
            "pp16_lp2",
            "pp16_lp2_hash",
            "pp8_lp4",
            "pp8_lp4_hash",
            "schema_version",
            "source_archive",
            "source_contract_hash",
            "source_host_record_rank0_sha256",
            "source_summary_sha256",
            "topology",
            "topology_hash",
            "tpu_successor_authorized",
        },
        "runtime physical locality authority",
    )
    topology = document["topology"]
    topology_hash = sha256(_canonical_json(topology).encode("ascii")).hexdigest()
    devices = _topology_devices(topology)
    pp16_groups, pp16_records = _stage_groups(
        document["pp16_lp2"],
        plan="PP16_LP2",
        group_size=2,
        devices=devices,
    )
    pp8_groups, pp8_records = _stage_groups(
        document["pp8_lp4"],
        plan="PP8_LP4",
        group_size=4,
        devices=devices,
    )
    pp16_hash = sha256(
        _canonical_json(document["pp16_lp2"]).encode("ascii")
    ).hexdigest()
    pp8_hash = sha256(
        _canonical_json(document["pp8_lp4"]).encode("ascii")
    ).hexdigest()
    if (
        document["artifact_kind"]
        != "gate_d_runtime_physical_locality_authority"
        or document["schema_version"] != 2
        or document["db_run_id"] != 555
        or topology_hash != _EXPECTED_TOPOLOGY_HASH
        or document["topology_hash"] != topology_hash
        or pp16_hash != _EXPECTED_PP16_LP2_HASH
        or document["pp16_lp2_hash"] != pp16_hash
        or pp8_hash != _EXPECTED_PP8_LP4_HASH
        or document["pp8_lp4_hash"] != pp8_hash
        or document["source_contract_hash"]
        != _EXPECTED_TOPOLOGY_SOURCE_CONTRACT_HASH
        or document["source_host_record_rank0_sha256"]
        != "71a4d01e3d266ba2a080c0634a3878aae40fcf11b1b6bf4f863f0cb1379afc7d"
        or document["source_summary_sha256"]
        != "1631eeb0102036319eef19af4be4f5cc165b889a70b51b36fe41b4996d7462d2"
        or document["tpu_successor_authorized"] is not False
    ):
        raise BenchmarkValidationError("runtime physical locality authority drifted")
    _string(document["claim_scope"], "runtime locality claim scope")
    _string(document["source_archive"], "runtime locality source archive")
    return {
        "authority_path": str(path),
        "authority_sha256": file_sha,
        "plans": {"PP16_LP2": pp16_groups, "PP8_LP4": pp8_groups},
        "plan_records": {
            "PP16_LP2": pp16_records,
            "PP8_LP4": pp8_records,
        },
        "topology_hash": topology_hash,
    }


def _verify_inherited_v1(
    value: Any, base: Path
) -> tuple[dict[str, Any], frozenset[str]]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("inherited v1 authority must be an object")
    _exact_keys(
        value,
        {"contract", "core", "frontier", "expected_classification"},
        "inherited v1 authority",
    )
    contract_path, contract_sha, contract_raw = _binding(
        base, value["contract"], "inherited v1 contract", limit=_MAX_JSON_BYTES
    )
    core_path, core_sha, _ = _binding(
        base, value["core"], "inherited v1 core", limit=_MAX_SOURCE_BYTES
    )
    _, frontier_sha, frontier_raw = _binding(
        base, value["frontier"], "inherited v1 frontier", limit=_MAX_JSON_BYTES
    )
    contract = _load_json(contract_raw, "inherited v1 contract")
    frontier = _load_json(frontier_raw, "inherited v1 frontier")
    expected_classification = _string(
        value["expected_classification"], "inherited v1 expected classification"
    )
    if (
        frontier.get("classification") != expected_classification
        or frontier.get("admitted_candidate_ids") != []
        or frontier.get("tpu_successor_authorized") is not False
        or frontier.get("jax_or_tpu_work_performed") is not False
    ):
        raise BenchmarkValidationError("inherited v1 frontier claim drifted")
    reproduction = frontier.get("reproduction")
    if (
        not isinstance(reproduction, dict)
        or reproduction.get("contract_sha256") != contract_sha
        or reproduction.get("core_sha256") != core_sha
    ):
        raise BenchmarkValidationError("inherited v1 reproduction binding drifted")
    closed = contract.get("closed_families")
    if not isinstance(closed, list) or not closed:
        raise BenchmarkValidationError("inherited v1 closed families are absent")
    fingerprints: set[str] = set()
    for family in closed:
        if not isinstance(family, dict):
            raise BenchmarkValidationError("inherited v1 closed family is invalid")
        values = family.get("mechanism_fingerprint_sha256s")
        if not isinstance(values, list) or not values:
            raise BenchmarkValidationError(
                "inherited v1 closed fingerprints are absent"
            )
        for item in values:
            fingerprint = _sha(item, "inherited v1 closed fingerprint")
            if fingerprint in fingerprints:
                raise BenchmarkValidationError(
                    "inherited v1 closed fingerprint is duplicated"
                )
            fingerprints.add(fingerprint)
    return (
        {
            "contract_path": str(contract_path),
            "contract_sha256": contract_sha,
            "core_path": str(core_path),
            "core_sha256": core_sha,
            "frontier_sha256": frontier_sha,
            "classification": expected_classification,
            "closed_fingerprint_count": len(fingerprints),
        },
        frozenset(fingerprints),
    )


def _verify_implementation(value: Any, base: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("precompile implementation must be an object")
    _exact_keys(
        value,
        {
            "cli",
            "core",
            "git",
            "stablehlo_validator",
            "validator_imports",
            "validator_python",
            "validator_pythonpath",
        },
        "precompile implementation",
    )
    core_path, core_sha, _ = _binding(
        base, value["core"], "precompile admission core", limit=_MAX_SOURCE_BYTES
    )
    cli_path, cli_sha, _ = _binding(
        base, value["cli"], "precompile admission CLI", limit=_MAX_SOURCE_BYTES
    )
    git_path, git_sha, _ = _binding(
        base, value["git"], "Git executable", limit=_MAX_SOURCE_BYTES
    )
    validator_path, validator_sha, _ = _binding(
        base,
        value["stablehlo_validator"],
        "StableHLO validator",
        limit=_MAX_SOURCE_BYTES,
    )
    python_path, python_sha, _ = _binding(
        base,
        value["validator_python"],
        "StableHLO validator Python",
        limit=64 * 1024 * 1024,
    )
    pythonpath = _resolve(
        base, value["validator_pythonpath"], "StableHLO validator PYTHONPATH"
    )
    with _open_directory_no_symlinks(
        pythonpath, "StableHLO validator PYTHONPATH"
    ):
        pass
    raw_imports = value["validator_imports"]
    if not isinstance(raw_imports, list) or not raw_imports:
        raise BenchmarkValidationError("StableHLO validator imports are absent")
    parser_imports: list[dict[str, Any]] = []
    import_paths: set[str] = set()
    total_bytes = 0
    for index, item in enumerate(raw_imports):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(
                f"StableHLO validator import {index} is invalid"
            )
        _exact_keys(
            item,
            {"bytes", "path", "sha256"},
            f"StableHLO validator import {index}",
        )
        relative = _canonical_repo_path(
            item["path"], f"StableHLO validator import {index} path"
        )
        if relative in import_paths or not relative.startswith("jaxlib/"):
            raise BenchmarkValidationError(
                "StableHLO validator import path is duplicated or outside jaxlib"
            )
        import_paths.add(relative)
        size = _positive_int(item["bytes"], f"parser import {relative} bytes")
        total_bytes += size
        if total_bytes > _MAX_PARSER_TOTAL_BYTES:
            raise BenchmarkValidationError("StableHLO validator imports exceed limit")
        parser_imports.append(
            {"bytes": size, "path": relative, "sha256": _sha(item["sha256"], relative)}
        )
    observed_imports = {
        item["path"]: (item["bytes"], item["sha256"]) for item in parser_imports
    }
    if observed_imports != _EXPECTED_VALIDATOR_IMPORTS:
        raise BenchmarkValidationError("StableHLO validator import manifest drifted")
    if Path(os.path.abspath(os.fspath(core_path))) != Path(
        os.path.abspath(__file__)
    ):
        raise BenchmarkValidationError(
            "precompile core binding is not the active implementation"
        )
    return {
        "cli_path": str(cli_path),
        "cli_sha256": cli_sha,
        "core_path": str(core_path),
        "core_sha256": core_sha,
        "git_path": str(git_path),
        "git_sha256": git_sha,
        "stablehlo_validator_path": str(validator_path),
        "stablehlo_validator_sha256": validator_sha,
        "validator_imports": sorted(parser_imports, key=lambda item: item["path"]),
        "validator_python_path": str(python_path),
        "validator_python_sha256": python_sha,
        "validator_pythonpath": str(pythonpath),
    }


def _qualified_ast_node(tree: ast.Module, qualified_name: str) -> ast.AST:
    components = qualified_name.split(".")
    if not components or any(not component.isidentifier() for component in components):
        raise BenchmarkValidationError(
            f"source symbol is not a qualified Python name: {qualified_name}"
        )
    body: Sequence[ast.stmt] = tree.body
    node: ast.AST | None = None
    for index, component in enumerate(components):
        matches = [
            item
            for item in body
            if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == component
        ]
        if len(matches) != 1:
            raise BenchmarkValidationError(
                f"source symbol is absent or ambiguous: {qualified_name}"
            )
        node = matches[0]
        if index + 1 < len(components):
            if not isinstance(node, ast.ClassDef):
                raise BenchmarkValidationError(
                    f"source symbol parent is not a class: {qualified_name}"
                )
            body = node.body
    assert node is not None
    return node


def _call_of(value: ast.AST, name: str, arguments: Sequence[ast.AST]) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == name
        and not value.keywords
        and len(value.args) == len(arguments)
        and all(ast.dump(left) == ast.dump(right) for left, right in zip(value.args, arguments))
    )


def _named_assignment(statement: ast.stmt, name: str) -> ast.AST | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    ):
        return statement.value
    return None


def _candidate_source_text(candidate_id: str) -> str:
    """Return the declarative semantic DSL used by hostile fixture validation.

    The names in this snippet are intentionally abstract operators.  This is not
    executable candidate source authority: a future candidate must separately
    bind concrete operator definitions and its real integration caller.
    """
    auxiliary = "    auxiliary = transient\n"
    if candidate_id == "compensated_auxiliary_dependency":
        auxiliary = (
            "    rounded = float32(carried_residual)\n"
            "    correction = transient - rounded\n"
            "    restored = rounded + correction\n"
            "    auxiliary = restored - correction\n"
        )
    return (
        "def candidate_dependency(hidden_update, residual, weight):\n"
        "    transient = float32(hidden_update) + float32(residual)\n"
        "    carried_residual = round_to_bf16(transient)\n"
        "    variance = mean(square(transient))\n"
        "    normalized = transient * rsqrt(variance + epsilon())\n"
        "    weighted_output = round_to_bf16(round_to_bf16(normalized) * weight)\n"
        f"{auxiliary}"
        "    return weighted_output, carried_residual, auxiliary\n"
        "\n"
        "def candidate_callsite(hidden_update, residual, weight):\n"
        "    return candidate_dependency(hidden_update, residual, weight)\n"
    )


def _candidate_source_semantics(
    node: ast.AST,
    callsite: ast.AST,
    candidate_id: str,
) -> dict[str, Any]:
    if candidate_id not in _EXPECTED_SURVIVORS:
        raise BenchmarkValidationError("candidate source identity is unsupported")
    expected = ast.parse(_candidate_source_text(candidate_id))
    expected_node, expected_callsite = expected.body
    if ast.dump(node) != ast.dump(expected_node):
        raise BenchmarkValidationError("candidate source RMS semantics drifted")
    if ast.dump(callsite) != ast.dump(expected_callsite):
        raise BenchmarkValidationError("candidate source callsite semantics drifted")
    return {
        "authority_scope": "declarative.semantic.dsl.only",
        "auxiliary": (
            "transient.fp32.sum"
            if candidate_id == "auxiliary_device_tuple_dependency"
            else "graph.identity.only;numeric.cancellation.unproven"
        ),
        "callsite": "declarative.consumer(candidate_dependency(hidden_update,residual,weight))",
        "carried_residual": "bf16.round(transient.fp32.sum)",
        "frontier": "layer1.rms_input_fp32",
        "normalization_input": "transient.fp32.sum",
        "weighted_output": (
            "bf16.round(bf16.round(transient*rsqrt(mean(square(transient))+epsilon))*weight)"
        ),
    }


def _accepted_semantics_from_evidence(document: Mapping[str, Any]) -> dict[str, Any]:
    authority = document.get("accepted_authority")
    logical_hlo = document.get("accepted_logical_hlo_authority")
    semantic_contract = document.get("sealed_semantic_contract")
    if (
        not isinstance(authority, dict)
        or not isinstance(logical_hlo, dict)
        or not isinstance(semantic_contract, dict)
        or authority.get("git_commit") != _EXPECTED_ACCEPTED_SOURCE["git_commit"]
        or logical_hlo.get("sha256")
        != "a8c9577d63909e647cd1b1a6d63210a818c51fc7efe7b305e385bad59189ba21"
        or semantic_contract.get("boundary_sum")
        != "float32(hidden_update) + float32(carried_residual)"
        or semantic_contract.get("normalization_input")
        != "the unrounded boundary_sum"
        or semantic_contract.get("greenfield_activation_dtype") != "bfloat16"
    ):
        raise BenchmarkValidationError("sealed accepted source semantics drifted")
    files = authority.get("files")
    if not isinstance(files, list):
        raise BenchmarkValidationError("sealed accepted source files are absent")
    matching = [
        item
        for item in files
        if isinstance(item, dict)
        and item.get("path") == _EXPECTED_ACCEPTED_SOURCE["path"]
        and item.get("symbol") == _EXPECTED_ACCEPTED_SOURCE["symbol"]
    ]
    if len(matching) != 1 or any(
        matching[0].get(key) != _EXPECTED_ACCEPTED_SOURCE[key]
        for key in ("ast_sha256", "file_sha256")
    ):
        raise BenchmarkValidationError("sealed accepted source symbol drifted")
    payload = {
        "accepted_source": _EXPECTED_ACCEPTED_SOURCE,
        "logical_hlo_sha256": logical_hlo["sha256"],
        "semantic_contract": semantic_contract,
    }
    return {
        **payload,
        "authority_sha256": sha256(
            _canonical_json(payload).encode("ascii")
        ).hexdigest(),
    }


def _git_output(
    repository_descriptor: int,
    git_path: Path,
    git_sha256: str,
    arguments: Sequence[str],
    label: str,
    *,
    limit: int,
) -> bytes:
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        with _open_regular_file(git_path, "Git executable") as git_stream:
            digest = sha256()
            while block := git_stream.read(1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != git_sha256:
                raise BenchmarkValidationError("Git executable SHA-256 drifted")
            completed = subprocess.run(
                [
                    "git",
                    "--no-pager",
                    "--no-replace-objects",
                    "-c",
                    "core.attributesfile=/dev/null",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-C",
                    f"/proc/self/fd/{repository_descriptor}",
                    *arguments,
                ],
                check=False,
                capture_output=True,
                env=environment,
                executable=f"/proc/self/fd/{git_stream.fileno()}",
                pass_fds=(repository_descriptor, git_stream.fileno()),
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkValidationError(f"cannot query {label}") from error
    if completed.returncode != 0:
        raise BenchmarkValidationError(f"{label} is not available from Git")
    if len(completed.stdout) > limit:
        raise BenchmarkValidationError(f"{label} exceeds the size limit")
    return completed.stdout


def _canonical_repo_path(value: Any, label: str) -> str:
    raw = _string(value, label)
    path = PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix() or any(
        component in ("", ".", "..") for component in path.parts
    ):
        raise BenchmarkValidationError(f"{label} is not a canonical repo path")
    return raw


def _read_source_records(
    repository_root: Path,
    code_pin: str,
    files: Any,
    *,
    git_path: Path,
    git_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, ast.AST]]:
    if not isinstance(files, list) or not files:
        raise BenchmarkValidationError("source authority files must be non-empty")
    records: list[dict[str, Any]] = []
    file_ids: set[str] = set()
    symbol_ids: set[str] = set()
    symbol_nodes: dict[str, ast.AST] = {}
    with _open_directory_no_symlinks(repository_root, "source repository") as repository:
        resolved_commit = _git_output(
            repository,
            git_path,
            git_sha256,
            ["rev-parse", "--verify", f"{code_pin}^{{commit}}"],
            "source commit",
            limit=1024,
        ).decode("ascii", errors="strict").strip()
        if resolved_commit != code_pin:
            raise BenchmarkValidationError("source commit authority drifted")
        for index, item in enumerate(files):
            if not isinstance(item, dict):
                raise BenchmarkValidationError(f"source file {index} must be an object")
            _exact_keys(
                item,
                {"id", "repo_path", "sha256", "symbols"},
                f"source file {index}",
            )
            source_id = _identifier(item["id"], f"source file {index} id")
            if source_id in file_ids:
                raise BenchmarkValidationError("source file id is duplicated")
            file_ids.add(source_id)
            expected_sha = _sha(item["sha256"], f"source file {source_id} SHA-256")
            repo_path = _canonical_repo_path(
                item["repo_path"], f"source file {source_id} repo path"
            )
            tree_entry = _git_output(
                repository,
                git_path,
                git_sha256,
                ["ls-tree", "-z", code_pin, "--", repo_path],
                f"source file {source_id} tree entry",
                limit=4096,
            )
            try:
                metadata, listed_path = tree_entry[:-1].split(b"\t", 1)
                mode, object_type, raw_object_id = metadata.split(b" ", 2)
                listed = listed_path.decode("utf-8", errors="strict")
                object_id = raw_object_id.decode("ascii", errors="strict")
            except (UnicodeDecodeError, ValueError) as error:
                raise BenchmarkValidationError(
                    f"source file tree entry is invalid: {source_id}"
                ) from error
            if (
                not tree_entry.endswith(b"\0")
                or tree_entry.count(b"\0") != 1
                or listed != repo_path
                or mode not in {b"100644", b"100755"}
                or object_type != b"blob"
                or _GIT_OBJECT_ID.fullmatch(object_id) is None
            ):
                raise BenchmarkValidationError(
                    f"source file is not a committed regular blob: {source_id}"
                )
            raw_size = _git_output(
                repository,
                git_path,
                git_sha256,
                ["cat-file", "-s", object_id],
                f"source file {source_id} size",
                limit=1024,
            )
            try:
                size = int(raw_size.decode("ascii", errors="strict").strip())
            except (UnicodeDecodeError, ValueError) as error:
                raise BenchmarkValidationError(
                    f"source file size is invalid: {source_id}"
                ) from error
            if size < 0 or size > _MAX_SOURCE_BYTES:
                raise BenchmarkValidationError(f"source file is too large: {source_id}")
            raw = _git_output(
                repository,
                git_path,
                git_sha256,
                ["cat-file", "blob", object_id],
                f"source file {source_id}",
                limit=_MAX_SOURCE_BYTES,
            )
            if len(raw) != size or sha256(raw).hexdigest() != expected_sha:
                raise BenchmarkValidationError(
                    f"source file authority drifted: {source_id}"
                )
            try:
                tree = ast.parse(raw.decode("utf-8"))
            except (UnicodeDecodeError, SyntaxError) as error:
                raise BenchmarkValidationError(
                    f"source file is not valid Python: {source_id}"
                ) from error
            symbols = item["symbols"]
            if not isinstance(symbols, list) or not symbols:
                raise BenchmarkValidationError(f"source symbols are absent: {source_id}")
            symbol_records: list[dict[str, str]] = []
            for symbol_index, symbol in enumerate(symbols):
                if not isinstance(symbol, dict):
                    raise BenchmarkValidationError("source symbol must be an object")
                _exact_keys(
                    symbol,
                    {"ast_sha256", "qualified_name"},
                    f"source symbol {source_id}[{symbol_index}]",
                )
                qualified_name = _string(
                    symbol["qualified_name"],
                    f"source symbol {source_id}[{symbol_index}] name",
                )
                canonical_id = f"{source_id}:{qualified_name}"
                if canonical_id in symbol_ids:
                    raise BenchmarkValidationError("source symbol is duplicated")
                symbol_ids.add(canonical_id)
                expected_ast = _sha(
                    symbol["ast_sha256"],
                    f"source symbol {canonical_id} AST SHA-256",
                )
                node = _qualified_ast_node(tree, qualified_name)
                observed_ast = sha256(
                    ast.dump(
                        node, annotate_fields=True, include_attributes=False
                    ).encode("utf-8")
                ).hexdigest()
                if observed_ast != expected_ast:
                    raise BenchmarkValidationError(
                        f"source symbol AST drifted: {canonical_id}"
                    )
                symbol_nodes[canonical_id] = node
                symbol_records.append(
                    {"ast_sha256": observed_ast, "qualified_name": qualified_name}
                )
            records.append(
                {
                    "git_object_id": object_id,
                    "id": source_id,
                    "repo_path": repo_path,
                    "sha256": expected_sha,
                    "symbols": sorted(
                        symbol_records, key=lambda item: item["qualified_name"]
                    ),
                }
            )
    return sorted(records, key=lambda item: item["id"]), symbol_nodes


def _verify_source_authority(
    value: Any,
    base: Path,
    *,
    candidate_id: str,
    frontier_action: str,
    mechanism_fingerprint_sha256: str,
    normal_form: Mapping[str, Any],
    locality: Mapping[str, Any],
    implementation: Mapping[str, Any],
    accepted_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("source authority must be an object")
    _exact_keys(
        value,
        {"files", "repository", "semantics_certificate"},
        "source authority",
    )
    repository = value["repository"]
    if not isinstance(repository, dict):
        raise BenchmarkValidationError("source repository authority must be an object")
    _exact_keys(repository, {"commit", "root"}, "source repository authority")
    code_pin = _code_pin(repository["commit"], "source authority code pin")
    repository_root = _resolve(base, repository["root"], "source repository root")
    files = value["files"]
    records, symbol_nodes = _read_source_records(
        repository_root,
        code_pin,
        files,
        git_path=Path(implementation["git_path"]),
        git_sha256=implementation["git_sha256"],
    )
    source_set_payload = {"code_pin": code_pin, "files": records}
    source_set_sha = sha256(
        _canonical_json(source_set_payload).encode("ascii")
    ).hexdigest()

    certificate_path, certificate_sha, certificate_raw = _binding(
        base,
        value["semantics_certificate"],
        "source semantics certificate",
        limit=_MAX_JSON_BYTES,
    )
    certificate = _load_json(certificate_raw, "source semantics certificate")
    _exact_keys(
        certificate,
        {
            "accepted_authority_sha256",
            "arithmetic_contract",
            "candidate_id",
            "candidate_callsite_symbol",
            "candidate_semantic_sha256",
            "candidate_symbol",
            "causal_frontier_action",
            "claim_scope",
            "code_pin",
            "compensation_claim",
            "locality",
            "mechanism_fingerprint_sha256",
            "normal_form",
            "schema_version",
            "source_set_sha256",
        },
        "source semantics certificate",
    )
    if certificate["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("source semantics certificate schema drifted")
    if (
        certificate["candidate_id"] != candidate_id
        or certificate["causal_frontier_action"] != frontier_action
        or certificate["code_pin"] != code_pin
        or certificate["mechanism_fingerprint_sha256"]
        != mechanism_fingerprint_sha256
        or certificate["source_set_sha256"] != source_set_sha
        or certificate["normal_form"] != dict(normal_form)
        or certificate["accepted_authority_sha256"]
        != accepted_semantics["authority_sha256"]
    ):
        raise BenchmarkValidationError("source semantics certificate authority drifted")
    available_symbols = {
        f"{record['id']}:{symbol['qualified_name']}": symbol["ast_sha256"]
        for record in records
        for symbol in record["symbols"]
    }
    candidate_symbol = _string(
        certificate["candidate_symbol"], "candidate source symbol"
    )
    callsite_symbol = _string(
        certificate["candidate_callsite_symbol"], "candidate callsite symbol"
    )
    if (
        candidate_symbol not in available_symbols
        or callsite_symbol not in available_symbols
        or callsite_symbol == candidate_symbol
    ):
        raise BenchmarkValidationError("source symbol authority is incomplete")
    source_semantics = _candidate_source_semantics(
        symbol_nodes[candidate_symbol], symbol_nodes[callsite_symbol], candidate_id
    )
    source_semantic_sha = sha256(
        _canonical_json(source_semantics).encode("ascii")
    ).hexdigest()
    if certificate["candidate_semantic_sha256"] != source_semantic_sha:
        raise BenchmarkValidationError("candidate source/HLO semantics binding drifted")
    claim_scope = _string(
        certificate["claim_scope"], "source semantics certificate claim scope"
    )
    expected_compensation_claim = (
        "graph.identity.only;numeric.cancellation.unproven"
        if candidate_id == "compensated_auxiliary_dependency"
        else "none"
    )
    if certificate["compensation_claim"] != expected_compensation_claim:
        raise BenchmarkValidationError("source compensation claim is overstated")
    _verify_locality(certificate["locality"], locality, "source semantics locality")
    arithmetic = certificate["arithmetic_contract"]
    if not isinstance(arithmetic, dict):
        raise BenchmarkValidationError("source arithmetic contract must be an object")
    _exact_keys(
        arithmetic,
        {
            "auxiliary_affects_primary_arithmetic",
            "normalization_input",
            "primary_outputs_bitwise_identical_by_construction",
            "primary_recurrence",
        },
        "source arithmetic contract",
    )
    if (
        _boolean(
            arithmetic["auxiliary_affects_primary_arithmetic"],
            "auxiliary arithmetic effect",
        )
        is not False
        or _boolean(
            arithmetic["primary_outputs_bitwise_identical_by_construction"],
            "primary bitwise construction",
        )
        is not True
        or arithmetic["normalization_input"] != "transient.fp32.sum"
        or arithmetic["primary_recurrence"] != "bf16.rounded"
    ):
        raise BenchmarkValidationError("source arithmetic contract is not accepted-preserving")
    report = {
        "certificate_path": str(certificate_path),
        "certificate_sha256": certificate_sha,
        "code_pin": code_pin,
        "accepted_authority_sha256": accepted_semantics["authority_sha256"],
        "candidate_symbol": {
            "ast_sha256": available_symbols[candidate_symbol],
            "id": candidate_symbol,
        },
        "callsite_symbol": {
            "ast_sha256": available_symbols[callsite_symbol],
            "id": callsite_symbol,
        },
        "certificate_claim_scope": claim_scope,
        "compensation_claim": expected_compensation_claim,
        "executable_source_authority": False,
        "files": records,
        "mechanism_fingerprint_sha256": mechanism_fingerprint_sha256,
        "repository_root": str(repository_root),
        "source_semantic_sha256": source_semantic_sha,
        "source_set_sha256": source_set_sha,
        "validation_scope": (
            "declarative semantic DSL and consumer-edge fixture only; "
            "abstract operators and the real integration caller are not bound"
        ),
    }
    report["authority_sha256"] = sha256(
        _canonical_json(
            {
                "certificate_sha256": certificate_sha,
                "code_pin": code_pin,
                "accepted_authority_sha256": accepted_semantics[
                    "authority_sha256"
                ],
                "candidate_ast_sha256": available_symbols[candidate_symbol],
                "callsite_ast_sha256": available_symbols[callsite_symbol],
                "compensation_claim": expected_compensation_claim,
                "mechanism_fingerprint_sha256": mechanism_fingerprint_sha256,
                "source_semantic_sha256": source_semantic_sha,
                "source_set_sha256": source_set_sha,
            }
        ).encode("ascii")
    ).hexdigest()
    return report


def _verify_plan_authority(
    value: Any,
    base: Path,
    physical_locality: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("plan authority must be an object")
    path, expected_file, raw = _binding(
        base, value, "plan authority", limit=_MAX_JSON_BYTES
    )
    document = _load_json(
        raw, "plan authority"
    )
    _exact_keys(
        document,
        {
            "local_device_groups",
            "plan",
            "plan_sha256",
            "schema_version",
            "topology_hash",
            "watchpoints",
        },
        "plan authority document",
    )
    if document["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("plan authority schema drifted")
    expected_plan = _sha(document["plan_sha256"], "plan SHA-256")
    plan_name = _string(document["plan"], "plan name")
    local_groups = _device_groups(document["local_device_groups"], "plan local groups")
    if (
        document["topology_hash"] != physical_locality["topology_hash"]
        or plan_name not in physical_locality["plans"]
        or local_groups != physical_locality["plans"][plan_name]
    ):
        raise BenchmarkValidationError(
            "plan groups are not the sealed runtime physical allowlist"
        )
    local_group_size = 2 if plan_name == "PP16_LP2" else 4
    if any(len(group) != local_group_size for group in local_groups):
        raise BenchmarkValidationError("plan local group size drifted")
    watchpoints = document["watchpoints"]
    if not isinstance(watchpoints, dict) or set(watchpoints) != set(
        _REQUIRED_WATCHPOINT_SCHEMA
    ):
        raise BenchmarkValidationError("plan watchpoint catalogue drifted")
    normalized_watchpoints: dict[str, dict[str, Any]] = {}
    sealed_owner_group: list[int] | None = None
    for watchpoint_id, item in watchpoints.items():
        if not isinstance(item, dict):
            raise BenchmarkValidationError(
                f"plan watchpoint is invalid: {watchpoint_id}"
            )
        _exact_keys(item, {"layout", "owner_ids"}, f"plan watchpoint {watchpoint_id}")
        layout = _identifier(item["layout"], f"plan watchpoint {watchpoint_id} layout")
        owner_ids = item["owner_ids"]
        if (
            not isinstance(owner_ids, list)
            or len(owner_ids) != local_group_size
            or any(
                not isinstance(owner, int)
                or isinstance(owner, bool)
                or owner < 0
                or owner > 31
                for owner in owner_ids
            )
            or len(set(owner_ids)) != local_group_size
            or owner_ids not in local_groups
        ):
            raise BenchmarkValidationError(
                f"plan watchpoint owners are not one exact {plan_name} group: {watchpoint_id}"
            )
        if sealed_owner_group is None:
            sealed_owner_group = owner_ids
        elif owner_ids != sealed_owner_group:
            raise BenchmarkValidationError(
                "plan watchpoints do not share one exact local owner group"
            )
        normalized_watchpoints[watchpoint_id] = {
            "layout": layout,
            "owner_ids": owner_ids,
        }
    payload = {
        "local_device_groups": local_groups,
        "plan": plan_name,
        "topology_hash": physical_locality["topology_hash"],
        "watchpoints": normalized_watchpoints,
    }
    if sha256(_canonical_json(payload).encode("ascii")).hexdigest() != expected_plan:
        raise BenchmarkValidationError("plan SHA-256 is not content-derived")
    return {
        "authority_file_sha256": expected_file,
        "authority_path": str(path),
        "local_device_groups": local_groups,
        "local_group_size": local_group_size,
        "owner_group": sealed_owner_group,
        "plan": plan_name,
        "plan_sha256": expected_plan,
        "topology_hash": physical_locality["topology_hash"],
        "watchpoints": normalized_watchpoints,
        "watchpoint_schema": _required_watchpoint_schema(local_group_size),
    }


def _copy_bound_parser_file(
    source: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    digest = sha256()
    observed = 0
    with _open_regular_file(source, f"parser import {source.name}") as reader:
        descriptor = os.open(destination, flags, 0o400)
        try:
            while block := reader.read(1024 * 1024):
                observed += len(block)
                if observed > expected_bytes:
                    raise BenchmarkValidationError("parser import byte count drifted")
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if observed != expected_bytes or digest.hexdigest() != expected_sha256:
        raise BenchmarkValidationError("parser import manifest drifted")


@contextmanager
def _sealed_parser_pythonpath(
    implementation: Mapping[str, Any],
) -> Iterator[tuple[int, str]]:
    with tempfile.TemporaryDirectory(prefix="glm-gate-d-parser-") as temporary:
        root = Path(temporary)
        source_root = Path(implementation["validator_pythonpath"])
        manifest = implementation["validator_imports"]
        for item in manifest:
            relative = PurePosixPath(item["path"])
            _copy_bound_parser_file(
                source_root.joinpath(*relative.parts),
                root.joinpath(*relative.parts),
                expected_bytes=item["bytes"],
                expected_sha256=item["sha256"],
            )
        with _open_directory_no_symlinks(root, "sealed parser root") as descriptor:
            yield descriptor, f"/proc/self/fd/{descriptor}"


def _run_stablehlo_validator(
    request: Mapping[str, Any], implementation: Mapping[str, Any]
) -> dict[str, Any]:
    validator_raw = _snapshot(
        Path(implementation["stablehlo_validator_path"]),
        implementation["stablehlo_validator_sha256"],
        "StableHLO validator",
        limit=_MAX_SOURCE_BYTES,
    )
    try:
        validator_source = validator_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BenchmarkValidationError("StableHLO validator is not UTF-8") from error
    payload = (_canonical_json(request) + "\n").encode("ascii")
    python_path = Path(implementation["validator_python_path"])
    try:
        with _sealed_parser_pythonpath(implementation) as (
            parser_descriptor,
            sealed_pythonpath,
        ), _open_regular_file(
            python_path, "StableHLO validator Python"
        ) as stream:
            digest = sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != implementation["validator_python_sha256"]:
                raise BenchmarkValidationError(
                    "StableHLO validator Python SHA-256 drifted"
                )
            environment = {
                "GATE_D_VALIDATOR_ROOT": sealed_pythonpath,
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONHASHSEED": "0",
                "PYTHONPATH": sealed_pythonpath,
            }
            completed = subprocess.run(
                ["python", "-S", "-c", validator_source],
                check=False,
                capture_output=True,
                env=environment,
                executable=f"/proc/self/fd/{stream.fileno()}",
                input=payload,
                pass_fds=(
                    parser_descriptor,
                    stream.fileno(),
                ),
                timeout=30,
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkValidationError("cannot run StableHLO validator") from error
    if len(completed.stdout) > _MAX_JSON_BYTES or len(completed.stderr) > _MAX_JSON_BYTES:
        raise BenchmarkValidationError("StableHLO validator output is too large")
    if completed.returncode != 0:
        refusal = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BenchmarkValidationError(
            f"StableHLO parser/causal validation failed: {refusal}"
        )
    report = _load_json(completed.stdout, "StableHLO validator output")
    if (
        report.get("parser") != "jaxlib.mlir.ir"
        or report.get("jaxlib_version") != _EXPECTED_JAXLIB_VERSION
        or report.get("stablehlo_version") != _EXPECTED_STABLEHLO_VERSION
        or report.get("jax_imported") is not False
        or report.get("loaded_parser_files")
        != {
            item["path"]: item["sha256"]
            for item in implementation["validator_imports"]
        }
    ):
        raise BenchmarkValidationError("StableHLO parser authority drifted")
    # The copied import tree is hash checked, but it is mutable by the invoking
    # UID while the child imports it.  Until loading is bound to immutable FDs
    # and mapped inode identities, this is a structural parser fixture rather
    # than admissible StableHLO authority.
    report["immutable_parser_authority"] = False
    report["parser_authority_scope"] = "mutable-copy structural fixture only"
    return report


def _verify_stablehlo_authority(
    value: Any,
    base: Path,
    *,
    candidate_id: str,
    mechanism_fingerprint_sha256: str,
    normal_form: Mapping[str, Any],
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    frontier_id: str,
    frontier_action: str,
    locality: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError("StableHLO authority must be an object")
    _exact_keys(
        value,
        {"accepted_primary", "candidate", "certificate"},
        "StableHLO authority",
    )
    candidate_path, candidate_sha, candidate_raw = _binding(
        base,
        value["candidate"],
        "candidate StableHLO",
        limit=_MAX_STABLEHLO_BYTES,
    )
    accepted_path, accepted_sha, accepted_raw = _binding(
        base,
        value["accepted_primary"],
        "accepted-primary StableHLO",
        limit=_MAX_STABLEHLO_BYTES,
    )
    certificate_path, certificate_sha, certificate_raw = _binding(
        base,
        value["certificate"],
        "StableHLO causal certificate",
        limit=_MAX_JSON_BYTES,
    )
    certificate = _load_json(certificate_raw, "StableHLO causal certificate")
    _exact_keys(
        certificate,
        {
            "accepted_primary_stablehlo_sha256",
            "candidate_id",
            "candidate_source_semantic_sha256",
            "candidate_stablehlo_sha256",
            "causal_frontier",
            "claim_scope",
            "code_pin",
            "locality",
            "mechanism_fingerprint_sha256",
            "normal_form",
            "plan_sha256",
            "schema_version",
            "source_set_sha256",
            "validator_contract",
        },
        "StableHLO causal certificate",
    )
    if certificate["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("StableHLO causal certificate schema drifted")
    if (
        certificate["accepted_primary_stablehlo_sha256"] != accepted_sha
        or certificate["candidate_id"] != candidate_id
        or certificate["candidate_stablehlo_sha256"] != candidate_sha
        or certificate["candidate_source_semantic_sha256"]
        != source["source_semantic_sha256"]
        or certificate["code_pin"] != source["code_pin"]
        or certificate["mechanism_fingerprint_sha256"]
        != mechanism_fingerprint_sha256
        or certificate["normal_form"] != dict(normal_form)
        or certificate["plan_sha256"] != plan["plan_sha256"]
        or certificate["source_set_sha256"] != source["source_set_sha256"]
    ):
        raise BenchmarkValidationError("StableHLO authority tuple drifted")
    _string(certificate["claim_scope"], "StableHLO certificate claim scope")
    _verify_locality(certificate["locality"], locality, "StableHLO locality")
    frontier = certificate["causal_frontier"]
    if not isinstance(frontier, dict):
        raise BenchmarkValidationError("StableHLO causal frontier must be an object")
    _exact_keys(frontier, {"action", "id"}, "StableHLO causal frontier")
    if frontier != {"action": frontier_action, "id": frontier_id}:
        raise BenchmarkValidationError("StableHLO causal frontier drifted")
    validator_contract = certificate["validator_contract"]
    if not isinstance(validator_contract, dict):
        raise BenchmarkValidationError("StableHLO validator contract must be an object")
    _exact_keys(
        validator_contract,
        {
            "auxiliary_result_index",
            "carried_residual_result_index",
            "weighted_output_result_index",
        },
        "StableHLO validator contract",
    )
    indices = {
        key: _nonnegative_int(value, f"StableHLO {key}")
        for key, value in validator_contract.items()
    }
    request = {
        "accepted_stablehlo_base64": base64.b64encode(accepted_raw).decode("ascii"),
        "accepted_stablehlo_sha256": accepted_sha,
        "auxiliary_result_index": indices["auxiliary_result_index"],
        "callsite_ast_sha256": source["callsite_symbol"]["ast_sha256"],
        "candidate_ast_sha256": source["candidate_symbol"]["ast_sha256"],
        "candidate_stablehlo_base64": base64.b64encode(candidate_raw).decode("ascii"),
        "candidate_stablehlo_sha256": candidate_sha,
        "carried_residual_result_index": indices[
            "carried_residual_result_index"
        ],
        "expected_parser_files": {
            item["path"]: item["sha256"]
            for item in implementation["validator_imports"]
        },
        "local_device_groups": plan["local_device_groups"],
        "source_set_sha256": source["source_set_sha256"],
        "weighted_output_result_index": indices["weighted_output_result_index"],
    }
    validator = _run_stablehlo_validator(request, implementation)
    auxiliary_operations = validator.get("auxiliary_path_operations")
    if (
        validator.get("accepted_stablehlo_sha256") != accepted_sha
        or validator.get("candidate_stablehlo_sha256") != candidate_sha
        or validator.get("local_device_groups") != plan["local_device_groups"]
        or validator.get("weighted_output_result_index")
        != indices["weighted_output_result_index"]
        or validator.get("carried_residual_result_index")
        != indices["carried_residual_result_index"]
        or validator.get("auxiliary_result_index")
        != indices["auxiliary_result_index"]
        or validator.get("accepted_primary_slice_sha256")
        != validator.get("candidate_primary_slice_sha256")
        or validator.get("accepted_primary_slice_sha256")
        != _EXPECTED_ACCEPTED_PRIMARY_SLICE_SHA256
        or source["source_semantic_sha256"]
        != _EXPECTED_SOURCE_SEMANTIC_SHA256[candidate_id]
        or validator.get("auxiliary_slice_sha256")
        != _EXPECTED_AUXILIARY_SLICE_SHA256[candidate_id]
        or not isinstance(auxiliary_operations, list)
        or any(not isinstance(item, str) for item in auxiliary_operations)
    ):
        raise BenchmarkValidationError("StableHLO validator result drifted")
    validator_sha = sha256(_canonical_json(validator).encode("ascii")).hexdigest()
    report = {
        "accepted_primary_path": str(accepted_path),
        "accepted_primary_sha256": accepted_sha,
        "accepted_primary_slice_sha256": validator[
            "accepted_primary_slice_sha256"
        ],
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "certificate_path": str(certificate_path),
        "certificate_sha256": certificate_sha,
        "collectives": validator["collectives"],
        "auxiliary_path_operations": auxiliary_operations,
        "auxiliary_slice_sha256": validator["auxiliary_slice_sha256"],
        "local_device_groups": plan["local_device_groups"],
        "immutable_parser_authority": validator["immutable_parser_authority"],
        "parser_authority_scope": validator["parser_authority_scope"],
        "validator_report_sha256": validator_sha,
    }
    report["authority_sha256"] = sha256(
        _canonical_json(
            {
                "accepted_primary_sha256": accepted_sha,
                "accepted_primary_slice_sha256": report[
                    "accepted_primary_slice_sha256"
                ],
                "candidate_sha256": candidate_sha,
                "certificate_sha256": certificate_sha,
                "mechanism_fingerprint_sha256": mechanism_fingerprint_sha256,
                "plan_sha256": plan["plan_sha256"],
                "source_semantic_sha256": source["source_semantic_sha256"],
                "source_set_sha256": source["source_set_sha256"],
                "validator_report_sha256": validator_sha,
            }
        ).encode("ascii")
    ).hexdigest()
    return report


def _product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _read_npy_header(stream: BinaryIO, label: str) -> tuple[str, tuple[int, ...]]:
    if stream.read(6) != b"\x93NUMPY":
        raise BenchmarkValidationError(f"{label} has invalid NPY magic")
    version = stream.read(2)
    if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise BenchmarkValidationError(f"{label} has unsupported NPY version")
    length_size = 2 if version[0] == 1 else 4
    raw_length = stream.read(length_size)
    if len(raw_length) != length_size:
        raise BenchmarkValidationError(f"{label} has truncated NPY header")
    header_size = struct.unpack("<H" if length_size == 2 else "<I", raw_length)[0]
    header_raw = stream.read(header_size)
    if len(header_raw) != header_size:
        raise BenchmarkValidationError(f"{label} has truncated NPY metadata")
    try:
        header = ast.literal_eval(header_raw.decode("latin1").strip())
    except (SyntaxError, ValueError) as error:
        raise BenchmarkValidationError(f"{label} has invalid NPY metadata") from error
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise BenchmarkValidationError(f"{label} NPY metadata schema drifted")
    dtype = header["descr"]
    shape = header["shape"]
    if (
        header["fortran_order"] is not False
        or not isinstance(dtype, str)
        or dtype not in _DTYPE_BYTES
    ):
        raise BenchmarkValidationError(f"{label} dtype/layout is unsupported")
    if not isinstance(shape, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in shape
    ):
        raise BenchmarkValidationError(f"{label} shape is invalid")
    return dtype, shape


def _inspect_npz(raw: bytes) -> dict[str, dict[str, Any]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (OSError, zipfile.BadZipFile) as error:
        raise BenchmarkValidationError("candidate capsule artifact is not NPZ") from error
    arrays: dict[str, dict[str, Any]] = {}
    with archive:
        names = archive.namelist()
        if not names or len(names) > 4096 or len(names) != len(set(names)):
            raise BenchmarkValidationError("candidate NPZ member catalogue is invalid")
        try:
            infos = [archive.getinfo(name) for name in names]
        except (KeyError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            raise BenchmarkValidationError("cannot inspect candidate NPZ") from error
        if any(
            info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            for info in infos
        ):
            raise BenchmarkValidationError("candidate NPZ compression is unsupported")
        if sum(info.file_size for info in infos) > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise BenchmarkValidationError("candidate NPZ uncompressed bytes exceed limit")
        for info in infos:
            name = info.filename
            if "/" in name or not name.endswith(".npy") or info.flag_bits & 0x1:
                raise BenchmarkValidationError(f"candidate NPZ member is invalid: {name}")
            if info.file_size > _MAX_ARRAY_BYTES + 64 * 1024:
                raise BenchmarkValidationError(f"candidate NPZ member is too large: {name}")
            try:
                with archive.open(info) as stream:
                    dtype, shape = _read_npy_header(stream, name)
                    expected_bytes = _product(shape) * _DTYPE_BYTES[dtype]
                    if expected_bytes > _MAX_ARRAY_BYTES:
                        raise BenchmarkValidationError(
                            f"candidate NPZ array is too large: {name}"
                        )
                    value = stream.read(expected_bytes + 1)
            except BenchmarkValidationError:
                raise
            except (NotImplementedError, OSError, zipfile.BadZipFile, RuntimeError) as error:
                raise BenchmarkValidationError(
                    f"cannot read candidate NPZ member: {name}"
                ) from error
            if len(value) != expected_bytes:
                raise BenchmarkValidationError(
                    f"candidate NPZ byte count drifted: {name}"
                )
            arrays[name[:-4]] = {
                "raw": value,
                "shape": shape,
                "storage_dtype": dtype,
            }
    return arrays


def _selected_array(
    array: Mapping[str, Any], index_prefix: list[int], label: str
) -> tuple[bytes, tuple[int, ...]]:
    shape = array["shape"]
    if not isinstance(index_prefix, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in index_prefix
    ):
        raise BenchmarkValidationError(f"{label} index prefix is invalid")
    if len(index_prefix) > len(shape):
        raise BenchmarkValidationError(f"{label} index prefix rank is invalid")
    flat_offset = 0
    for axis, index in enumerate(index_prefix):
        if index >= shape[axis]:
            raise BenchmarkValidationError(f"{label} index prefix is out of bounds")
        flat_offset += index * _product(shape[axis + 1 :])
    selected_shape = tuple(shape[len(index_prefix) :])
    item_bytes = _DTYPE_BYTES[array["storage_dtype"]]
    byte_offset = flat_offset * item_bytes
    byte_count = _product(selected_shape) * item_bytes
    return array["raw"][byte_offset : byte_offset + byte_count], selected_shape


def _derive_fp32_sum_from_bf16_bits(hidden: bytes, residual: bytes) -> bytes:
    """Derive the unique IEEE binary32 add from two finite BF16 bit rows."""

    expected_bytes = 6144 * 2
    if len(hidden) != expected_bytes or len(residual) != expected_bytes:
        raise BenchmarkValidationError("RMS BF16 operand byte count drifted")
    result = bytearray(6144 * 4)
    for index, (hidden_bits, residual_bits) in enumerate(
        zip(struct.iter_unpack("<H", hidden), struct.iter_unpack("<H", residual))
    ):
        left_bits = hidden_bits[0]
        right_bits = residual_bits[0]
        if (left_bits & 0x7F80) == 0x7F80 or (right_bits & 0x7F80) == 0x7F80:
            raise BenchmarkValidationError("RMS BF16 operands must be finite")
        left = struct.unpack("<f", struct.pack("<I", left_bits << 16))[0]
        right = struct.unpack("<f", struct.pack("<I", right_bits << 16))[0]
        try:
            encoded = struct.pack("<f", left + right)
        except OverflowError as error:
            raise BenchmarkValidationError("RMS FP32 operand sum overflowed") from error
        if struct.unpack("<I", encoded)[0] & 0x7F800000 == 0x7F800000:
            raise BenchmarkValidationError("RMS FP32 operand sum must be finite")
        result[index * 4 : (index + 1) * 4] = encoded
    return bytes(result)


def _verify_capsule(
    value: Any,
    base: Path,
    *,
    candidate_id: str,
    source: Mapping[str, Any],
    stablehlo: Mapping[str, Any],
    plan: Mapping[str, Any],
    required_watchpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path, capsule_sha, raw = _binding(
        base, value, "candidate coherent capsule", limit=_MAX_JSON_BYTES
    )
    capsule = _load_json(raw, "candidate coherent capsule")
    _exact_keys(
        capsule,
        {
            "artifact",
            "candidate_id",
            "claim_scope",
            "code_pin",
            "coherence_id",
            "plan_sha256",
            "schema_version",
            "source_authority_sha256",
            "stablehlo_authority_sha256",
            "watchpoints",
        },
        "candidate coherent capsule",
    )
    if capsule["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("candidate capsule schema drifted")
    if (
        capsule["candidate_id"] != candidate_id
        or capsule["code_pin"] != source["code_pin"]
        or capsule["plan_sha256"] != plan["plan_sha256"]
        or capsule["source_authority_sha256"] != source["authority_sha256"]
        or capsule["stablehlo_authority_sha256"] != stablehlo["authority_sha256"]
    ):
        raise BenchmarkValidationError("candidate capsule authority tuple drifted")
    coherence_id = _identifier(capsule["coherence_id"], "capsule coherence id")
    _string(capsule["claim_scope"], "capsule claim scope")
    artifact_path, artifact_sha, artifact_raw = _binding(
        path.parent,
        capsule["artifact"],
        "candidate capsule artifact",
        limit=_MAX_ARTIFACT_BYTES,
    )
    arrays = _inspect_npz(artifact_raw)
    watchpoints = capsule["watchpoints"]
    if not isinstance(watchpoints, list) or not watchpoints:
        raise BenchmarkValidationError("candidate capsule watchpoints must be non-empty")
    seen: dict[str, set[str]] = {}
    selected_references: set[tuple[str, tuple[int, ...]]] = set()
    selected_values: dict[tuple[str, str], bytes] = {}
    referenced_array_keys: set[str] = set()
    records: list[dict[str, Any]] = []
    for index, watchpoint in enumerate(watchpoints):
        if not isinstance(watchpoint, dict):
            raise BenchmarkValidationError(f"capsule watchpoint {index} is invalid")
        _exact_keys(
            watchpoint,
            {
                "arrays",
                "id",
                "layer",
                "layout",
                "owner_ids",
                "position",
            },
            f"capsule watchpoint {index}",
        )
        watchpoint_id = _identifier(watchpoint["id"], f"capsule watchpoint {index} id")
        if watchpoint_id in seen:
            raise BenchmarkValidationError("candidate capsule watchpoint is duplicated")
        if watchpoint_id not in required_watchpoints:
            raise BenchmarkValidationError(
                f"candidate capsule watchpoint is unexpected: {watchpoint_id}"
            )
        layer = _nonnegative_int(watchpoint["layer"], f"watchpoint {watchpoint_id} layer")
        position = _nonnegative_int(
            watchpoint["position"], f"watchpoint {watchpoint_id} position"
        )
        expected_watchpoint = required_watchpoints[watchpoint_id]
        if (
            layer != expected_watchpoint["layer"]
            or position != expected_watchpoint["position"]
        ):
            raise BenchmarkValidationError(
                f"watchpoint layer/position drifted: {watchpoint_id}"
            )
        layout = _identifier(watchpoint["layout"], f"watchpoint {watchpoint_id} layout")
        owner_ids = watchpoint["owner_ids"]
        if (
            not isinstance(owner_ids, list)
            or not owner_ids
            or len(owner_ids) > 4
            or any(
                not isinstance(owner, int) or isinstance(owner, bool) or owner < 0
                for owner in owner_ids
            )
            or len(set(owner_ids)) != len(owner_ids)
        ):
            raise BenchmarkValidationError(f"watchpoint owners are invalid: {watchpoint_id}")
        plan_watchpoint = plan["watchpoints"].get(watchpoint_id)
        if plan_watchpoint != {"layout": layout, "owner_ids": owner_ids}:
            raise BenchmarkValidationError(
                f"watchpoint layout/owners are not sealed plan authority: {watchpoint_id}"
            )
        array_specs = watchpoint["arrays"]
        if not isinstance(array_specs, list) or not array_specs:
            raise BenchmarkValidationError(f"watchpoint arrays are absent: {watchpoint_id}")
        roles: set[str] = set()
        array_records: list[dict[str, Any]] = []
        for array_index, array_spec in enumerate(array_specs):
            if not isinstance(array_spec, dict):
                raise BenchmarkValidationError("watchpoint array must be an object")
            _exact_keys(
                array_spec,
                {
                    "array_key",
                    "array_sha256",
                    "index_prefix",
                    "owner_axis",
                    "owner_axis_ids",
                    "owner_id",
                    "role",
                    "semantic_dtype",
                    "shape",
                    "storage_dtype",
                },
                f"watchpoint {watchpoint_id} array {array_index}",
            )
            role = _identifier(array_spec["role"], f"watchpoint {watchpoint_id} role")
            if role in roles:
                raise BenchmarkValidationError(
                    f"watchpoint array role is duplicated: {watchpoint_id}"
                )
            roles.add(role)
            expected_array = expected_watchpoint["arrays"].get(role)
            if expected_array is None:
                raise BenchmarkValidationError(
                    f"watchpoint array role is unexpected: {watchpoint_id}.{role}"
                )
            key = _string(array_spec["array_key"], f"watchpoint {watchpoint_id} key")
            if key not in arrays:
                raise BenchmarkValidationError(f"watchpoint array is absent: {key}")
            referenced_array_keys.add(key)
            index_prefix = array_spec["index_prefix"]
            if not isinstance(index_prefix, list) or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in index_prefix
            ):
                raise BenchmarkValidationError(
                    f"watchpoint index prefix is invalid: {watchpoint_id}.{role}"
                )
            if index_prefix != expected_array["index_prefix"]:
                raise BenchmarkValidationError(
                    f"watchpoint owner-axis prefix drifted: {watchpoint_id}.{role}"
                )
            reference = (key, tuple(index_prefix))
            if reference in selected_references:
                raise BenchmarkValidationError(
                    f"watchpoint selected array is reused: {watchpoint_id}.{role}"
                )
            selected_references.add(reference)
            expected_array_sha = _sha(
                array_spec["array_sha256"], f"watchpoint {watchpoint_id} array SHA-256"
            )
            storage_dtype = _string(
                array_spec["storage_dtype"], f"watchpoint {watchpoint_id} storage dtype"
            )
            semantic_dtype = _identifier(
                array_spec["semantic_dtype"], f"watchpoint {watchpoint_id} semantic dtype"
            )
            if semantic_dtype not in _SEMANTIC_DTYPES:
                raise BenchmarkValidationError(
                    f"watchpoint semantic dtype is unsupported: {watchpoint_id}"
                )
            shape = array_spec["shape"]
            if not isinstance(shape, list) or any(
                not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape
            ):
                raise BenchmarkValidationError(f"watchpoint shape is invalid: {watchpoint_id}")
            owner_id = array_spec["owner_id"]
            if owner_id is not None and (
                not isinstance(owner_id, int)
                or isinstance(owner_id, bool)
                or owner_id < 0
                or owner_id > 31
            ):
                raise BenchmarkValidationError(
                    f"watchpoint array owner is invalid: {watchpoint_id}.{role}"
                )
            expected_owner_slot = expected_array["owner_slot"]
            expected_owner_id = (
                None if expected_owner_slot is None else owner_ids[expected_owner_slot]
            )
            if owner_id != expected_owner_id:
                raise BenchmarkValidationError(
                    f"watchpoint array owner drifted: {watchpoint_id}.{role}"
                )
            owner_axis = array_spec["owner_axis"]
            owner_axis_ids = array_spec["owner_axis_ids"]
            if not isinstance(owner_axis_ids, list) or any(
                not isinstance(owner, int)
                or isinstance(owner, bool)
                or owner < 0
                or owner > 31
                for owner in owner_axis_ids
            ):
                raise BenchmarkValidationError(
                    f"watchpoint owner-axis ids are invalid: {watchpoint_id}.{role}"
                )
            expected_owner_axis = expected_array["owner_axis"]
            expected_axis_ids = [
                owner_ids[slot] for slot in expected_array["owner_axis_slots"]
            ]
            if (
                owner_axis != expected_owner_axis
                or owner_axis_ids != expected_axis_ids
                or (
                    owner_axis is not None
                    and (
                        not isinstance(owner_axis, int)
                        or isinstance(owner_axis, bool)
                        or owner_axis < 0
                    )
                )
            ):
                raise BenchmarkValidationError(
                    f"watchpoint owner-axis mapping drifted: {watchpoint_id}.{role}"
                )
            selected, selected_shape = _selected_array(
                arrays[key], index_prefix, f"watchpoint {watchpoint_id}"
            )
            selected_values[(watchpoint_id, role)] = selected
            expected_storage_shape = (
                tuple([len(owner_ids), 1, *shape])
                if index_prefix
                else tuple(shape)
            )
            if arrays[key]["shape"] != expected_storage_shape:
                raise BenchmarkValidationError(
                    f"watchpoint storage shape has unreferenced rows: {watchpoint_id}.{role}"
                )
            if owner_axis is not None and (
                owner_axis >= len(selected_shape)
                or selected_shape[owner_axis] != len(owner_axis_ids)
            ):
                raise BenchmarkValidationError(
                    f"watchpoint owner axis does not match array shape: {watchpoint_id}.{role}"
                )
            if (
                storage_dtype != arrays[key]["storage_dtype"]
                or list(selected_shape) != shape
                or sha256(selected).hexdigest() != expected_array_sha
                or storage_dtype != expected_array["storage_dtype"]
                or semantic_dtype != expected_array["semantic_dtype"]
                or shape != expected_array["shape"]
            ):
                raise BenchmarkValidationError(
                    f"watchpoint array metadata drifted: {watchpoint_id}"
                )
            array_records.append(
                {
                    "array_key": key,
                    "array_sha256": expected_array_sha,
                    "index_prefix": index_prefix,
                    "owner_axis": owner_axis,
                    "owner_axis_ids": owner_axis_ids,
                    "owner_id": owner_id,
                    "role": role,
                    "semantic_dtype": semantic_dtype,
                    "shape": shape,
                    "storage_dtype": storage_dtype,
                }
            )
        if roles != set(expected_watchpoint["arrays"]):
            raise BenchmarkValidationError(f"watchpoint array roles drifted: {watchpoint_id}")
        if watchpoint_id in {
            "layer1.normalized",
            "layer1.query",
            "layer1.head_weights",
        } and len({item["array_key"] for item in array_records}) != 1:
            raise BenchmarkValidationError(
                f"watchpoint owner slices do not share one owner axis: {watchpoint_id}"
            )
        seen[watchpoint_id] = roles
        records.append(
            {
                "arrays": sorted(array_records, key=lambda item: item["role"]),
                "id": watchpoint_id,
                "layer": layer,
                "layout": layout,
                "owner_ids": owner_ids,
                "position": position,
            }
        )
    if set(seen) != set(required_watchpoints):
        missing_watchpoints = sorted(set(required_watchpoints) - set(seen))
        raise BenchmarkValidationError(
            "candidate capsule watchpoint set drifted: "
            f"missing={missing_watchpoints}"
        )
    if referenced_array_keys != set(arrays):
        raise BenchmarkValidationError(
            "candidate NPZ contains unreferenced arrays: "
            f"{sorted(set(arrays) - referenced_array_keys)}"
        )
    derived_rms_input = _derive_fp32_sum_from_bf16_bits(
        selected_values[("layer1.rms_operands_bf16", "hidden_update")],
        selected_values[("layer1.rms_operands_bf16", "residual")],
    )
    if derived_rms_input != selected_values[("layer1.rms_input_fp32", "value")]:
        raise BenchmarkValidationError(
            "candidate RMS FP32 input is not derived from its sealed BF16 operands"
        )
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha,
        "capsule_path": str(path),
        "capsule_sha256": capsule_sha,
        "coherence_id": coherence_id,
        "derived_rms_input_sha256": sha256(derived_rms_input).hexdigest(),
        "producer_provenance_verified": False,
        "validation_scope": (
            "typed capsule schema and BF16-derived FP32 invariant only; "
            "no pinned producer or candidate-execution receipt"
        ),
        "watchpoints": sorted(records, key=lambda item: item["id"]),
    }


def admit_gate_d_precompile_candidates(
    contract_path: Path | str, expected_contract_sha256: str
) -> dict[str, Any]:
    """Authenticate and classify precompile candidates without importing JAX."""

    contract_path = Path(contract_path)
    expected_contract_sha256 = _sha(expected_contract_sha256, "contract SHA-256")
    raw = _snapshot(
        contract_path,
        expected_contract_sha256,
        "Gate-D precompile admission contract",
        limit=_MAX_JSON_BYTES,
    )
    contract = _load_json(raw, "Gate-D precompile admission contract")
    _exact_keys(
        contract,
        {
            "candidates",
            "causal_frontier",
            "claim_scope",
            "contract_id",
            "evidence",
            "implementation",
            "inherited_v1",
            "locality_contract",
            "physical_locality_authority",
            "required_coherent_watchpoints",
            "schema_version",
        },
        "Gate-D precompile admission contract",
    )
    if contract["schema_version"] != GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("Gate-D precompile admission schema drifted")
    contract_id = _identifier(contract["contract_id"], "contract id")
    claim_scope = _string(contract["claim_scope"], "claim scope")
    base = contract_path.parent
    implementation = _verify_implementation(contract["implementation"], base)
    inherited, closed_fingerprints = _verify_inherited_v1(contract["inherited_v1"], base)
    physical_locality = _verify_physical_locality(
        contract["physical_locality_authority"], base
    )

    frontier = contract["causal_frontier"]
    if not isinstance(frontier, dict):
        raise BenchmarkValidationError("causal frontier must be an object")
    _exact_keys(frontier, {"evidence_id", "id", "order"}, "causal frontier")
    frontier_id = _identifier(frontier["id"], "causal frontier id")
    if frontier["order"] != 0:
        raise BenchmarkValidationError("causal frontier order must remain zero")

    locality = contract["locality_contract"]
    expected_locality = {
        "logical_rows": 1,
        "max_collective_group_size": 4,
        "no_full_pod_hidden_reconstruction": True,
        "no_host_effects": True,
    }
    _verify_locality(locality, expected_locality, "locality contract")

    required_raw = contract["required_coherent_watchpoints"]
    expected_required = {
        "PP16_LP2": _serialized_watchpoint_schema(2),
        "PP8_LP4": _serialized_watchpoint_schema(4),
    }
    if required_raw != expected_required:
        raise BenchmarkValidationError("required coherent watchpoint schema drifted")
    if any(
        frontier_id not in _required_watchpoint_schema(owner_count)
        for owner_count in (2, 4)
    ):
        raise BenchmarkValidationError("causal frontier is absent from required watchpoints")

    evidence = contract["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise BenchmarkValidationError("precompile evidence must be non-empty")
    evidence_shas: dict[str, str] = {}
    evidence_documents: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"evidence {index} must be an object")
        _exact_keys(item, {"claim_scope", "id", "path", "sha256"}, f"evidence {index}")
        evidence_id = _identifier(item["id"], f"evidence {index} id")
        if evidence_id in evidence_shas:
            raise BenchmarkValidationError("evidence id is duplicated")
        expected = _sha(item["sha256"], f"evidence {evidence_id} SHA-256")
        path = _resolve(base, item["path"], f"evidence {evidence_id} path")
        evidence_raw = _snapshot(
            path,
            expected,
            f"precompile evidence {evidence_id}",
            limit=_MAX_ARTIFACT_BYTES,
        )
        _string(item["claim_scope"], f"evidence {evidence_id} claim scope")
        evidence_shas[evidence_id] = expected
        evidence_documents[evidence_id] = _load_json(
            evidence_raw, f"precompile evidence {evidence_id}"
        )
    if set(evidence_shas) != _EXPECTED_EVIDENCE_IDS:
        raise BenchmarkValidationError("precompile evidence catalogue drifted")
    frontier_evidence = _identifier(frontier["evidence_id"], "frontier evidence id")
    if frontier_evidence not in evidence_shas:
        raise BenchmarkValidationError("causal frontier evidence is absent")
    frontier_document = evidence_documents["observability_frontier"]
    if (
        frontier_document.get("classification")
        != "OBSERVABILITY_GAP;GATE_D_OPEN;NO_TPU_SUCCESSOR"
        or frontier_document.get("causal_frontier")
        != {
            "id": frontier_id,
            "missing_accepted": True,
            "missing_candidate": True,
            "order": 0,
        }
    ):
        raise BenchmarkValidationError("observability frontier evidence drifted")
    constructability = evidence_documents["constructability"]
    if (
        constructability.get("new_variant_capsule_constructable") is not False
        or constructability.get("variant_source_bound") is not False
        or constructability.get("tpu_successor_authorized") is not False
        or "append_only_precompile_admission_v2_with_source_and_stablehlo_authority"
        not in constructability.get("required_next", [])
    ):
        raise BenchmarkValidationError("constructability evidence drifted")
    direct_rejection = evidence_documents["direct_shadow_rejection"]
    if (
        direct_rejection.get("classification")
        != "DIRECT_UNROUNDED_FP32_SHADOW_SUBSTITUTION_REJECTED;"
        "OTHER_SHADOW_FORMS_UNADJUDICATED;NO_TPU_SUCCESSOR"
        or direct_rejection.get("causal_frontier") != frontier_id
        or direct_rejection.get("tpu_successor_authorized") is not False
    ):
        raise BenchmarkValidationError("direct-shadow rejection evidence drifted")
    accepted_semantics = _accepted_semantics_from_evidence(direct_rejection)
    shadow = evidence_documents["shadow_adjudication"]
    unresolved = shadow.get("unresolved_variant_ids")
    variant_results = shadow.get("variant_results")
    if (
        shadow.get("classification")
        != "NO_OFFLINE_COMPLETE_SHADOW_VARIANT;"
        "AUXILIARY_DEVICE_VARIANTS_REMAIN_UNADJUDICATED;"
        "GATE_D_OPEN;NO_TPU_SUCCESSOR"
        or shadow.get("tpu_successor_authorized") is not False
        or not isinstance(unresolved, list)
        or not isinstance(variant_results, list)
    ):
        raise BenchmarkValidationError("shadow adjudication evidence drifted")
    expected_normal_forms = {
        item.get("id"): item.get("normal_form")
        for item in variant_results
        if isinstance(item, dict) and item.get("id") in unresolved
    }
    if set(expected_normal_forms) != set(unresolved):
        raise BenchmarkValidationError("shadow unresolved variant evidence drifted")
    if set(unresolved) != set(_EXPECTED_SURVIVORS) or any(
        expected_normal_forms[candidate_id]
        != _EXPECTED_SURVIVORS[candidate_id]["normal_form"]
        for candidate_id in _EXPECTED_SURVIVORS
    ):
        raise BenchmarkValidationError("sealed survivor identities drifted")

    candidates = contract["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise BenchmarkValidationError("precompile candidates must be non-empty")
    candidate_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise BenchmarkValidationError(f"candidate {index} must be an object")
        _exact_keys(
            candidate,
            {
                "causal_frontier_action",
                "coherent_state_capsule",
                "evidence_ids",
                "host_effect",
                "id",
                "logical_rows",
                "max_collective_group_size",
                "mechanism_fingerprint",
                "normal_form",
                "plan_authority",
                "reconstructs_full_pod_hidden",
                "source_authority",
                "stablehlo_authority",
                "summary",
            },
            f"candidate {index}",
        )
        candidate_id = _identifier(candidate["id"], f"candidate {index} id")
        if candidate_id in candidate_ids:
            raise BenchmarkValidationError("candidate id is duplicated")
        candidate_ids.add(candidate_id)
        _string(candidate["summary"], f"candidate {candidate_id} summary")
        evidence_ids = candidate["evidence_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise BenchmarkValidationError(f"candidate evidence is absent: {candidate_id}")
        candidate_evidence = [
            _identifier(item, f"candidate {candidate_id} evidence") for item in evidence_ids
        ]
        if set(candidate_evidence) != _EXPECTED_EVIDENCE_IDS or len(
            candidate_evidence
        ) != len(_EXPECTED_EVIDENCE_IDS):
            raise BenchmarkValidationError(f"candidate evidence drifted: {candidate_id}")
        normal_form = candidate["normal_form"]
        if not isinstance(normal_form, dict):
            raise BenchmarkValidationError(f"candidate normal form is invalid: {candidate_id}")
        _exact_keys(
            normal_form,
            {"compensation", "dependency", "normalization_input", "primary_recurrence"},
            f"candidate {candidate_id} normal form",
        )
        for key, item in normal_form.items():
            _identifier(item, f"candidate {candidate_id} normal form {key}")
        if expected_normal_forms.get(candidate_id) != normal_form:
            raise BenchmarkValidationError(
                f"candidate is not the sealed unresolved normal form: {candidate_id}"
            )
        if candidate["mechanism_fingerprint"] != _EXPECTED_SURVIVORS.get(
            candidate_id, {}
        ).get("mechanism_fingerprint"):
            raise BenchmarkValidationError(
                f"candidate fingerprint is not the sealed survivor: {candidate_id}"
            )
        fingerprint = _mechanism_fingerprint(
            candidate["mechanism_fingerprint"], f"candidate {candidate_id} fingerprint"
        )
        reasons: list[str] = []
        if fingerprint in closed_fingerprints:
            reasons.append("DUPLICATES_V1_CLOSED_FAMILY")
        if _positive_int(candidate["logical_rows"], f"candidate {candidate_id} rows") != 1:
            reasons.append("NOT_TRUE_ONE_ROW")
        if _positive_int(
            candidate["max_collective_group_size"], f"candidate {candidate_id} group size"
        ) > 4:
            reasons.append("NONLOCAL_COLLECTIVE_GROUP")
        if _boolean(
            candidate["reconstructs_full_pod_hidden"],
            f"candidate {candidate_id} full pod",
        ):
            reasons.append("FULL_POD_HIDDEN_RECONSTRUCTION")
        if _boolean(candidate["host_effect"], f"candidate {candidate_id} host effect"):
            reasons.append("HOST_EFFECT_OR_DISPATCH")
        action = candidate["causal_frontier_action"]
        if action not in _FRONTIER_ACTIONS:
            raise BenchmarkValidationError(f"candidate frontier action is invalid: {candidate_id}")

        source_report: dict[str, Any] | None = None
        plan_report: dict[str, Any] | None = None
        stablehlo_report: dict[str, Any] | None = None
        capsule_report: dict[str, Any] | None = None
        if candidate["source_authority"] is None:
            reasons.append("MISSING_SOURCE_AST_AUTHORITY")
        else:
            try:
                source_report = _verify_source_authority(
                    candidate["source_authority"],
                    base,
                    candidate_id=candidate_id,
                    frontier_action=action,
                    mechanism_fingerprint_sha256=fingerprint,
                    normal_form=normal_form,
                    locality=locality,
                    implementation=implementation,
                    accepted_semantics=accepted_semantics,
                )
                reasons.append("MISSING_EXECUTABLE_SOURCE_AUTHORITY")
            except BenchmarkValidationError as error:
                reasons.append("INVALID_SOURCE_AST_AUTHORITY")
                source_report = {"refusal": str(error)}
        if candidate["plan_authority"] is None:
            reasons.append("MISSING_PLAN_AUTHORITY")
        else:
            try:
                plan_report = _verify_plan_authority(
                    candidate["plan_authority"], base, physical_locality
                )
            except BenchmarkValidationError as error:
                reasons.append("INVALID_PLAN_AUTHORITY")
                plan_report = {"refusal": str(error)}
        if candidate["stablehlo_authority"] is None:
            reasons.append("MISSING_CAUSAL_STABLEHLO_AUTHORITY")
        elif (
            source_report is None
            or "refusal" in source_report
            or plan_report is None
            or "refusal" in plan_report
        ):
            reasons.append("UNVERIFIABLE_CAUSAL_STABLEHLO_AUTHORITY")
        else:
            try:
                stablehlo_report = _verify_stablehlo_authority(
                    candidate["stablehlo_authority"],
                    base,
                    candidate_id=candidate_id,
                    mechanism_fingerprint_sha256=fingerprint,
                    normal_form=normal_form,
                    source=source_report,
                    plan=plan_report,
                    frontier_id=frontier_id,
                    frontier_action=action,
                    locality=locality,
                    implementation=implementation,
                )
                reasons.append("MISSING_IMMUTABLE_STABLEHLO_PARSER_AUTHORITY")
            except BenchmarkValidationError as error:
                reasons.append("INVALID_CAUSAL_STABLEHLO_AUTHORITY")
                stablehlo_report = {"refusal": str(error)}
        if candidate["coherent_state_capsule"] is None:
            reasons.append("MISSING_CANDIDATE_COHERENT_CAPSULE")
        elif (
            source_report is None
            or "refusal" in source_report
            or plan_report is None
            or "refusal" in plan_report
            or stablehlo_report is None
            or "refusal" in stablehlo_report
        ):
            reasons.append("UNVERIFIABLE_CANDIDATE_COHERENT_CAPSULE")
        else:
            try:
                capsule_report = _verify_capsule(
                    candidate["coherent_state_capsule"],
                    base,
                    candidate_id=candidate_id,
                    source=source_report,
                    stablehlo=stablehlo_report,
                    plan=plan_report,
                    required_watchpoints=plan_report["watchpoint_schema"],
                )
                reasons.append("MISSING_PINNED_COHERENT_CAPSULE_PRODUCER")
            except BenchmarkValidationError as error:
                reasons.append("INVALID_CANDIDATE_COHERENT_CAPSULE")
                capsule_report = {"refusal": str(error)}
        results.append(
            {
                "admitted_precompile": not reasons,
                "capsule": capsule_report,
                "evidence_sha256s": {
                    item: evidence_shas[item] for item in candidate_evidence
                },
                "id": candidate_id,
                "mechanism_fingerprint_sha256": fingerprint,
                "plan_authority": plan_report,
                "reasons": reasons,
                "source_authority": source_report,
                "stablehlo_authority": stablehlo_report,
            }
        )
    if candidate_ids != set(unresolved):
        raise BenchmarkValidationError("precompile candidate catalogue drifted")
    valid_stablehlo_shas = [
        item["stablehlo_authority"]["candidate_sha256"]
        for item in results
        if isinstance(item["stablehlo_authority"], dict)
        and "refusal" not in item["stablehlo_authority"]
    ]
    if len(valid_stablehlo_shas) != len(set(valid_stablehlo_shas)):
        raise BenchmarkValidationError(
            "distinct survivors reuse one candidate StableHLO implementation"
        )
    admitted = [item["id"] for item in results if item["admitted_precompile"]]
    if admitted:
        raise BenchmarkValidationError(
            "schema-fixture admission v2 cannot authorize a precompile candidate"
        )
    return {
        "admitted_candidate_ids": admitted,
        "candidate_results": results,
        "claim_scope": claim_scope,
        "classification": "NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;"
        "NO_JAX_OR_TPU_SUCCESSOR",
        "compile_only_review_required": True,
        "contract_id": contract_id,
        "contract_sha256": expected_contract_sha256,
        "evidence_sha256s": dict(sorted(evidence_shas.items())),
        "gate_d_closed": False,
        "inherited_v1": inherited,
        "implementation": implementation,
        "jax_compile_or_tpu_work_performed": False,
        "physical_locality_authority": physical_locality,
        "required_coherent_watchpoints": expected_required,
        "schema_version": GATE_D_PRECOMPILE_ADMISSION_SCHEMA_VERSION,
        "tpu_successor_authorized": False,
    }


def write_gate_d_precompile_admission_report(
    path: Path | str, report: Mapping[str, Any]
) -> None:
    """Write one canonical append-only report without following symlinks."""

    path = Path(path)
    if not path.name or path.name in (".", ".."):
        raise BenchmarkValidationError(f"output path is invalid: {path}")
    payload = (_canonical_json(report) + "\n").encode("ascii")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    with _open_directory_no_symlinks(path.parent, "output parent") as parent:
        try:
            os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise BenchmarkValidationError(f"cannot inspect output path: {path}") from error
        else:
            raise BenchmarkValidationError(f"output path is occupied: {path}")
        try:
            descriptor = os.open(path.name, flags, 0o644, dir_fd=parent)
        except OSError as error:
            raise BenchmarkValidationError(f"cannot create output: {path}") from error
        created = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(parent)
        except BaseException:
            try:
                current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            except OSError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (
                created.st_dev,
                created.st_ino,
            ):
                os.unlink(path.name, dir_fd=parent)
                os.fsync(parent)
            raise
