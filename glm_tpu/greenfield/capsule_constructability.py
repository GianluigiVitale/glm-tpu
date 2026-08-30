"""Offline constructability audit for Gate-D coherent-state capsules.

The audit answers a deliberately narrow question: can either surviving shadow
declaration be admitted from the sealed numerical state already on disk?  It
authenticates the old candidate artifact, inventories the seven required
watchpoints, and inspects the current observability/admission schemas from
SHA-bound source AST.  It does not
import JAX, derive new model values, or authorize compilation/TPU execution.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, BinaryIO, Iterator, Mapping
import zipfile

from .errors import BenchmarkValidationError


__all__ = (
    "CAPSULE_CONSTRUCTABILITY_SCHEMA_VERSION",
    "audit_capsule_constructability",
    "write_capsule_constructability_report",
)


CAPSULE_CONSTRUCTABILITY_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024 * 1024
_MAX_CANDIDATE_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_DTYPE_BYTES = {"|b1": 1, "<u2": 2, "<u4": 4, "<i4": 4, "<f4": 4}
_REQUIRED_WATCHPOINTS = (
    "layer1.rms_input_fp32",
    "layer1.normalized",
    "layer1.cache_history",
    "layer1.query",
    "layer1.head_weights",
    "layer1.current_key",
    "layer1.scorer_event1",
)
_VARIANT_IDS = (
    "auxiliary_device_tuple_dependency",
    "compensated_auxiliary_dependency",
)
_EXPECTED_EVIDENCE_IDS = {
    "admission_core",
    "constructability_cli",
    "constructability_core",
    "observability_contract",
    "observability_core",
    "observability_frontier",
    "observer_rejection",
    "shadow_contract",
    "shadow_report",
}
_EXPECTED_INVENTORY: dict[str, tuple[str, tuple[str, ...]]] = {
    "layer1.rms_input_fp32": (
        "absent_frontier",
        ("layer1_rms_input_fp32", "current_rms_input_fp32", "fused_add_float32"),
    ),
    "layer1.normalized": (
        "present_direct_old_authority",
        ("current_normalized_hidden_owners_bfloat16_bits",),
    ),
    "layer1.cache_history": (
        "present_post_event_only_old_authority",
        ("layer1_index_cache_owners_bfloat16_bits",),
    ),
    "layer1.query": (
        "present_direct_old_authority",
        ("current_dsa_query_owners",),
    ),
    "layer1.head_weights": (
        "present_direct_old_authority",
        ("current_dsa_head_weights_owners",),
    ),
    "layer1.current_key": (
        "absent_current_event",
        ("current_key", "current_key_owners", "current_key_owners_float32"),
    ),
    "layer1.scorer_event1": (
        "present_direct_old_authority",
        ("event1_positions", "event1_scores", "event1_valid_counts"),
    ),
}
_EXPECTED_CANDIDATE_BINDINGS: dict[str, tuple[str, ...]] = {
    "current_normalized_hidden_owners_bfloat16_bits": (
        "layer1.normalized.owner0",
        "layer1.normalized.owner1",
    ),
    "layer1_index_cache_owners_bfloat16_bits": ("layer1.post_event_cache",),
    "current_dsa_query_owners": ("layer1.query.owner0", "layer1.query.owner1"),
    "current_dsa_head_weights_owners": ("layer1.head.owner0", "layer1.head.owner1"),
    "event1_positions": ("layer1.event1.positions",),
    "event1_scores": ("layer1.event1.scores",),
    "event1_valid_counts": ("layer1.event1.valid_count",),
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
        raise BenchmarkValidationError(f"cannot safely open {label}") from error
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
                raise BenchmarkValidationError(f"{label} is not a regular file")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                yield stream
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} is not a canonical identifier")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise BenchmarkValidationError(
            f"{label} keys drifted: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _snapshot(path: Path, expected_sha256: str, label: str, limit: int) -> bytes:
    digest = sha256()
    blocks: list[bytes] = []
    total = 0
    with _open_regular_file(path, label) as stream:
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > limit:
                raise BenchmarkValidationError(f"{label} is too large")
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


def _product(shape: tuple[int, ...]) -> int:
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
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
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
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in shape
    ):
        raise BenchmarkValidationError(f"{label} shape is invalid")
    return dtype, shape


def _inspect_npz_bytes(
    raw: bytes, artifact_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (OSError, zipfile.BadZipFile) as error:
        raise BenchmarkValidationError(
            "candidate artifact is not a valid NPZ"
        ) from error
    with archive:
        names = archive.namelist()
        if not names or len(names) > 4096 or len(names) != len(set(names)):
            raise BenchmarkValidationError("candidate NPZ member catalogue is invalid")
        allowed_compressions = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        try:
            infos = [archive.getinfo(name) for name in names]
        except (KeyError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            raise BenchmarkValidationError(
                "cannot inspect candidate NPZ catalogue"
            ) from error
        if any(info.compress_type not in allowed_compressions for info in infos):
            raise BenchmarkValidationError(
                "candidate NPZ uses unsupported compression"
            )
        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise BenchmarkValidationError(
                "candidate NPZ aggregate uncompressed bytes exceed the limit"
            )
        members: list[dict[str, Any]] = []
        arrays: dict[str, dict[str, Any]] = {}
        for name in sorted(names):
            if "/" in name or not name.endswith(".npy"):
                raise BenchmarkValidationError(
                    f"candidate NPZ member is noncanonical: {name}"
                )
            info = archive.getinfo(name)
            if info.flag_bits & 0x1 or info.file_size > _MAX_ARRAY_BYTES + 64 * 1024:
                raise BenchmarkValidationError(
                    f"candidate NPZ member is invalid: {name}"
                )
            try:
                with archive.open(info) as stream:
                    dtype, shape = _read_npy_header(stream, name)
                    expected_bytes = _product(shape) * _DTYPE_BYTES[dtype]
                    if expected_bytes > _MAX_ARRAY_BYTES:
                        raise BenchmarkValidationError(
                            f"candidate NPZ member is too large: {name}"
                        )
                    value = stream.read(expected_bytes + 1)
            except (
                NotImplementedError,
                OSError,
                zipfile.BadZipFile,
                RuntimeError,
            ) as error:
                raise BenchmarkValidationError(
                    f"cannot read candidate NPZ member: {name}"
                ) from error
            if len(value) != expected_bytes:
                raise BenchmarkValidationError(
                    f"candidate NPZ member byte count drifted: {name}"
                )
            members.append(
                {
                    "key": name[:-4],
                    "raw_bytes": len(value),
                    "raw_sha256": sha256(value).hexdigest(),
                    "shape": list(shape),
                    "storage_dtype": dtype,
                }
            )
            arrays[name[:-4]] = {
                "raw": value,
                "shape": shape,
                "storage_dtype": dtype,
            }
    return (
        {
            "artifact_sha256": artifact_sha256,
            "member_count": len(members),
            "members": members,
        },
        arrays,
    )


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _inspect_observability_schema(raw: bytes) -> dict[str, Any]:
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise BenchmarkValidationError(
            "observability core is not valid Python"
        ) from error
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "audit_observability_contract"
        ),
        None,
    )
    if function is None:
        raise BenchmarkValidationError("observability contract auditor is absent")
    constants = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    allowed = {
        "unavailable",
        "optimized_hlo",
        "executable_fingerprint",
        "executable_triple",
    }
    if not allowed.issubset(constants) or "stablehlo" in constants:
        raise BenchmarkValidationError("observability identity-kind AST drifted")
    candidate_rejection = "candidate source requires an executable SHA-256"
    if candidate_rejection not in constants:
        raise BenchmarkValidationError("uncompiled-candidate rejection AST drifted")
    return {
        "allowed_executable_identity_kinds": sorted(allowed),
        "candidate_requires_nonnull_executable_identity": True,
        "explicit_source_ast_authority": False,
        "stablehlo_identity_kind_supported": False,
    }


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
    selected_shape = shape[len(index_prefix) :]
    item_size = _DTYPE_BYTES[array["storage_dtype"]]
    selected_bytes = _product(selected_shape) * item_size
    start = flat_offset * item_size
    return array["raw"][start : start + selected_bytes], selected_shape


def _verify_candidate_authority(
    document: Mapping[str, Any],
    *,
    artifact_path: Path,
    artifact_sha256: str,
    arrays: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    sources = document.get("sources")
    if not isinstance(sources, list):
        raise BenchmarkValidationError("observability sources are invalid")
    candidates = [
        source
        for source in sources
        if isinstance(source, dict) and source.get("role") == "candidate"
    ]
    if len(candidates) != 1:
        raise BenchmarkValidationError("candidate source authority is ambiguous")
    candidate = candidates[0]
    if candidate.get("artifact_sha256") != artifact_sha256 or Path(
        os.path.abspath(candidate.get("artifact_path", ""))
    ) != Path(os.path.abspath(artifact_path)):
        raise BenchmarkValidationError(
            "candidate artifact and observability authority disagree"
        )
    authority_fields = {
        "code_pin",
        "coherence_id",
        "executable_identity_kind",
        "executable_identity_sha256",
        "plan_id",
        "plan_sha256",
    }
    authority = {field: candidate.get(field) for field in sorted(authority_fields)}
    if any(value is None for value in authority.values()):
        raise BenchmarkValidationError("candidate authority tuple is incomplete")
    observations = candidate.get("observations")
    if not isinstance(observations, list):
        raise BenchmarkValidationError("candidate observations are invalid")
    by_key: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        if not isinstance(observation, dict):
            raise BenchmarkValidationError("candidate observation must be an object")
        key = observation.get("key")
        if key in _EXPECTED_CANDIDATE_BINDINGS:
            by_key.setdefault(key, []).append(observation)
    bound: dict[str, list[dict[str, Any]]] = {}
    for key, expected_watchpoints in _EXPECTED_CANDIDATE_BINDINGS.items():
        bindings = by_key.get(key, [])
        observed_watchpoints = tuple(
            binding.get("watchpoint_id") for binding in bindings
        )
        if observed_watchpoints != expected_watchpoints:
            raise BenchmarkValidationError(
                f"candidate observation bindings drifted: {key}"
            )
        array = arrays.get(key)
        if array is None:
            raise BenchmarkValidationError(f"candidate array is absent: {key}")
        records: list[dict[str, Any]] = []
        for binding in bindings:
            _exact_keys(
                binding,
                {
                    "array_sha256",
                    "index_prefix",
                    "key",
                    "shape",
                    "storage_dtype",
                    "watchpoint_id",
                },
                f"candidate observation {binding.get('watchpoint_id')}",
            )
            selected, selected_shape = _selected_array(
                array,
                binding["index_prefix"],
                f"candidate observation {binding['watchpoint_id']}",
            )
            if (
                binding["storage_dtype"] != array["storage_dtype"]
                or binding["shape"] != list(selected_shape)
                or binding["array_sha256"] != sha256(selected).hexdigest()
            ):
                raise BenchmarkValidationError(
                    f"candidate observation bytes drifted: {binding['watchpoint_id']}"
                )
            records.append(
                {
                    "array_sha256": binding["array_sha256"],
                    "index_prefix": binding["index_prefix"],
                    "watchpoint_id": binding["watchpoint_id"],
                }
            )
        bound[key] = records
    return authority, bound


def _verify_admission_ast(path: Path, expected_sha256: str) -> dict[str, bool]:
    raw = _snapshot(path, expected_sha256, "Gate-D admission core", _MAX_JSON_BYTES)
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise BenchmarkValidationError(
            "Gate-D admission core is not valid Python"
        ) from error
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_verify_capsule"
        ),
        None,
    )
    if function is None:
        raise BenchmarkValidationError("Gate-D capsule verifier is absent")
    requires_sha = False
    binds_authority = False
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_require_sha256" and node.args:
                requires_sha |= "executable_identity_sha256" in ast.dump(node.args[0])
        if isinstance(node, ast.Dict):
            keys = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if {
                "code_pin",
                "coherence_id",
                "executable_identity_sha256",
                "plan_sha256",
            }.issubset(keys):
                binds_authority = True
    if not requires_sha or not binds_authority:
        raise BenchmarkValidationError("Gate-D capsule authority AST drifted")
    return {
        "capsule_executable_sha256_required": requires_sha,
        "capsule_authority_tuple_bound": binds_authority,
    }


def audit_capsule_constructability(
    contract_path: Path | str, expected_contract_sha256: str
) -> dict[str, Any]:
    """Authenticate sealed state and classify precompile capsule feasibility."""

    contract_path = Path(contract_path)
    expected_contract_sha256 = _sha(expected_contract_sha256, "contract SHA-256")
    raw = _snapshot(
        contract_path,
        expected_contract_sha256,
        "capsule constructability contract",
        _MAX_JSON_BYTES,
    )
    contract = _load_json(raw, "capsule constructability contract")
    _exact_keys(
        contract,
        {
            "candidate_artifact",
            "claim_scope",
            "contract_id",
            "evidence",
            "required_watchpoints",
            "schema_version",
            "variant_ids",
            "watchpoint_inventory",
        },
        "capsule constructability contract",
    )
    if (
        not isinstance(contract["schema_version"], int)
        or isinstance(contract["schema_version"], bool)
        or contract["schema_version"] != CAPSULE_CONSTRUCTABILITY_SCHEMA_VERSION
    ):
        raise BenchmarkValidationError("capsule constructability schema drifted")
    contract_id = _identifier(contract["contract_id"], "contract id")
    if not isinstance(contract["claim_scope"], str) or not contract["claim_scope"]:
        raise BenchmarkValidationError("claim scope must be a non-empty string")
    if not isinstance(contract["required_watchpoints"], list) or tuple(
        contract["required_watchpoints"]
    ) != _REQUIRED_WATCHPOINTS:
        raise BenchmarkValidationError("seven-watchpoint contract drifted")
    if not isinstance(contract["variant_ids"], list) or tuple(
        contract["variant_ids"]
    ) != _VARIANT_IDS:
        raise BenchmarkValidationError("surviving variant set drifted")

    base = contract_path.parent
    evidence: dict[str, dict[str, Any]] = {}
    if not isinstance(contract["evidence"], list):
        raise BenchmarkValidationError("evidence must be a list")
    for index, item in enumerate(contract["evidence"]):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"evidence {index} must be an object")
        _exact_keys(item, {"id", "path", "sha256"}, f"evidence {index}")
        identifier = _identifier(item["id"], f"evidence {index} id")
        if identifier in evidence:
            raise BenchmarkValidationError("evidence id duplicated")
        digest = _sha(item["sha256"], f"evidence {identifier} SHA-256")
        path = _resolve(base, item["path"], f"evidence {identifier} path")
        limit = (
            _MAX_JSON_BYTES
            if path.suffix in {".json", ".py"}
            else _MAX_EVIDENCE_BYTES
        )
        payload = _snapshot(path, digest, f"evidence {identifier}", limit)
        evidence[identifier] = {"path": path, "sha256": digest, "raw": payload}
    if set(evidence) != _EXPECTED_EVIDENCE_IDS:
        raise BenchmarkValidationError("evidence catalogue drifted")

    active_core = Path(os.path.abspath(__file__))
    declared_core = Path(os.path.abspath(evidence["constructability_core"]["path"]))
    if active_core != declared_core:
        raise BenchmarkValidationError("active constructability core path drifted")
    _snapshot(
        active_core,
        evidence["constructability_core"]["sha256"],
        "active constructability core",
        _MAX_JSON_BYTES,
    )

    frontier = _load_json(evidence["observability_frontier"]["raw"], "frontier")
    if frontier.get("classification") != (
        "OBSERVABILITY_GAP;GATE_D_OPEN;NO_TPU_SUCCESSOR"
    ):
        raise BenchmarkValidationError("observability frontier classification drifted")
    if frontier.get("contract") != {
        "path": "configs/greenfield-gate-d-observability.json",
        "sha256": evidence["observability_contract"]["sha256"],
    }:
        raise BenchmarkValidationError("observability contract/frontier binding drifted")
    shadow = _load_json(evidence["shadow_report"]["raw"], "shadow report")
    if (
        shadow.get("contract_sha256") != evidence["shadow_contract"]["sha256"]
        or shadow.get("unresolved_variant_ids") != list(_VARIANT_IDS)
        or shadow.get("required_before_compile_only_acquisition_review")
        != [
            "declared_source_semantics_certificate",
            "causal_stablehlo_contract",
            "offline_candidate_coherent_seven_watchpoint_capsule",
            "separate_gate_d_admission",
        ]
    ):
        raise BenchmarkValidationError("shadow precompile requirements drifted")
    observer = _load_json(evidence["observer_rejection"]["raw"], "observer rejection")
    if observer.get("classification") != "REJECTED_OBSERVER_PERTURBATION":
        raise BenchmarkValidationError("observer rejection classification drifted")

    artifact = contract["candidate_artifact"]
    if not isinstance(artifact, dict):
        raise BenchmarkValidationError("candidate artifact must be an object")
    _exact_keys(artifact, {"path", "sha256"}, "candidate artifact")
    artifact_path = _resolve(base, artifact["path"], "candidate artifact path")
    artifact_sha256 = _sha(artifact["sha256"], "candidate artifact SHA-256")
    artifact_raw = _snapshot(
        artifact_path,
        artifact_sha256,
        "candidate artifact",
        _MAX_CANDIDATE_BYTES,
    )
    inventory, arrays = _inspect_npz_bytes(artifact_raw, artifact_sha256)
    members = {item["key"]: item for item in inventory["members"]}
    observability_document = _load_json(
        evidence["observability_contract"]["raw"], "observability contract"
    )
    candidate_authority, candidate_bindings = _verify_candidate_authority(
        observability_document,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        arrays=arrays,
    )

    declarations = contract["watchpoint_inventory"]
    if not isinstance(declarations, list) or len(declarations) != len(
        _REQUIRED_WATCHPOINTS
    ):
        raise BenchmarkValidationError("watchpoint inventory cardinality drifted")
    results: list[dict[str, Any]] = []
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, dict):
            raise BenchmarkValidationError(f"watchpoint inventory {index} is invalid")
        _exact_keys(
            declaration,
            {"id", "keys", "status"},
            f"watchpoint inventory {index}",
        )
        identifier = _identifier(declaration["id"], f"watchpoint inventory {index} id")
        expected = _EXPECTED_INVENTORY.get(identifier)
        if (
            expected is None
            or not isinstance(declaration["keys"], list)
            or declaration["status"] != expected[0]
            or tuple(declaration["keys"]) != expected[1]
        ):
            raise BenchmarkValidationError(
                f"watchpoint inventory drifted: {identifier}"
            )
        status, keys = expected
        present = [key for key in keys if key in members]
        expects_absent = status.startswith("absent_")
        if (expects_absent and present) or (
            not expects_absent and len(present) != len(keys)
        ):
            raise BenchmarkValidationError(
                f"NPZ watchpoint availability drifted: {identifier}"
            )
        results.append(
            {
                "id": identifier,
                "keys": list(keys),
                "member_metadata": [members[key] for key in present],
                "status": status,
                "usable_for_new_variant_authority": False,
            }
        )
    if [item["id"] for item in results] != list(_REQUIRED_WATCHPOINTS):
        raise BenchmarkValidationError("watchpoint inventory order drifted")

    schema_findings = _inspect_observability_schema(
        evidence["observability_core"]["raw"]
    )
    admission_ast = _verify_admission_ast(
        evidence["admission_core"]["path"], evidence["admission_core"]["sha256"]
    )
    missing = [
        item["id"] for item in results if item["status"].startswith("absent_")
    ]
    return {
        "artifact_kind": "greenfield_gate_d_capsule_constructability_audit",
        "candidate_artifact": {
            "member_count": inventory["member_count"],
            "sha256": artifact_sha256,
        },
        "candidate_authority": candidate_authority,
        "candidate_observation_bindings": candidate_bindings,
        "claim_scope": contract["claim_scope"],
        "classification": (
            "SEALED_STATE_INCOMPLETE;"
            "PRECOMPILE_SOURCE_STABLEHLO_AUTHORITY_UNREPRESENTED;"
            "GATE_D_OPEN;NO_JAX_OR_TPU_SUCCESSOR"
        ),
        "contract_id": contract_id,
        "contract_sha256": expected_contract_sha256,
        "gate_d_closed": False,
        "missing_old_artifact_watchpoints": missing,
        "new_variant_capsule_constructable": False,
        "observability_classification": "OBSERVABILITY_GAP",
        "observer_artifact_authoritative": False,
        "precompile_authority_audit": {
            **admission_ast,
            "observability_schema_findings": schema_findings,
        },
        "required_next": [
            "append_only_precompile_admission_v2_with_source_and_stablehlo_authority",
            "source_bound_variant_and_causal_stablehlo",
            "candidate_coherent_offline_seven_watchpoint_capsule",
            "separate_review_before_compile_only_acquisition",
        ],
        "schema_version": CAPSULE_CONSTRUCTABILITY_SCHEMA_VERSION,
        "tpu_successor_authorized": False,
        "variant_ids": list(_VARIANT_IDS),
        "variant_source_bound": False,
        "watchpoint_inventory": results,
    }


def write_capsule_constructability_report(
    path: Path, report: Mapping[str, Any]
) -> None:
    """Write one canonical append-only report without following symlinks."""

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
            raise BenchmarkValidationError(
                f"cannot inspect output path: {path}"
            ) from error
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
                current = os.stat(
                    path.name, dir_fd=parent, follow_symlinks=False
                )
            except OSError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (
                created.st_dev,
                created.st_ino,
            ):
                os.unlink(path.name, dir_fd=parent)
                os.fsync(parent)
            raise
