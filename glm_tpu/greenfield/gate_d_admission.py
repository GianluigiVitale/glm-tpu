"""Fail-closed offline admission for new Gate-D mechanism proposals.

The accepted and greenfield executions currently diverge before the layer-1
normalized row, while neither protected artifact exposes the unrounded FP32
RMS input.  This module prevents a proposed successor from reaching JAX or a
TPU merely because it has a new name.  It authenticates the historical closure
evidence, rejects duplicate/illegal mechanisms, and requires one coherent,
SHA-bound state capsule before reporting offline readiness.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

from .errors import BenchmarkValidationError
from .observability import audit_observability_contract


__all__ = (
    "GATE_D_ADMISSION_SCHEMA_VERSION",
    "admit_gate_d_mechanisms",
    "write_gate_d_admission_report",
)


GATE_D_ADMISSION_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_PIN = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024 * 1024
_FRONTIER_ACTIONS = frozenset(("expose", "resolve", "downstream", "none"))
_FINGERPRINT_KEYS = {
    "association",
    "consumer_boundary",
    "reduction",
    "representation",
    "transport",
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise BenchmarkValidationError(f"non-finite JSON value: {value}")


@contextmanager
def _open_directory_no_symlinks(path: Path, label: str) -> Iterator[int]:
    normalized = Path(os.path.abspath(os.fspath(path)))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open("/", directory_flags)
    except OSError as error:
        raise BenchmarkValidationError(
            f"cannot safely open {label}: {normalized}"
        ) from error
    try:
        for component in normalized.parts:
            if component in ("", ".", "/"):
                continue
            try:
                child = os.open(component, directory_flags, dir_fd=descriptor)
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


def _snapshot_bytes(
    path: Path,
    label: str,
    expected_sha256: str,
    *,
    limit: int,
) -> bytes:
    digest = sha256()
    blocks: list[bytes] = []
    total = 0
    with _open_regular_file(path, label) as stream:
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > limit:
                raise BenchmarkValidationError(
                    f"{label} is too large: {path}"
                )
            digest.update(block)
            blocks.append(block)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise BenchmarkValidationError(f"{label} SHA-256 drifted: {path}")
    return b"".join(blocks)


def _verify_regular_file(
    path: Path,
    label: str,
    expected_sha256: str,
    *,
    limit: int,
) -> None:
    """Stream-authenticate a bounded file without retaining its payload."""

    digest = sha256()
    total = 0
    with _open_regular_file(path, label) as stream:
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > limit:
                raise BenchmarkValidationError(
                    f"{label} is too large: {path}"
                )
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise BenchmarkValidationError(f"{label} SHA-256 drifted: {path}")


def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkValidationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise BenchmarkValidationError(
            f"{label} keys drifted: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} is not a canonical identifier")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_code_pin(value: Any, label: str) -> str:
    if not isinstance(value, str) or _CODE_PIN.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a full Git SHA")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkValidationError(f"{label} must be boolean")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkValidationError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BenchmarkValidationError(f"{label} must be a non-negative integer")
    return value


def _require_identifier_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty list")
    result = tuple(
        _require_identifier(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise BenchmarkValidationError(f"{label} contains duplicates")
    return result


def _resolve_path(base: Path, value: Any, label: str) -> Path:
    raw = _require_string(value, label)
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _read_json_path(document: Any, path: Any, label: str) -> Any:
    if not isinstance(path, list) or not path:
        raise BenchmarkValidationError(f"{label} JSON path must be non-empty")
    value = document
    for component in path:
        if isinstance(value, dict):
            if not isinstance(component, str) or component not in value:
                raise BenchmarkValidationError(f"{label} JSON path is absent")
            value = value[component]
        elif isinstance(value, list):
            if (
                not isinstance(component, int)
                or isinstance(component, bool)
                or component < 0
                or component >= len(value)
            ):
                raise BenchmarkValidationError(f"{label} JSON index is invalid")
            value = value[component]
        else:
            raise BenchmarkValidationError(f"{label} JSON path is not traversable")
    return value


def _mechanism_fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    _require_exact_keys(value, _FINGERPRINT_KEYS, label)
    canonical = {
        key: _require_identifier(value[key], f"{label}.{key}")
        for key in sorted(_FINGERPRINT_KEYS)
    }
    return sha256(_canonical_json(canonical).encode("ascii")).hexdigest()


def _verify_capsule(
    *,
    binding: Mapping[str, Any],
    contract_base: Path,
    mechanism_id: str,
    frontier_id: str,
    expected_action: str,
    required_watchpoints: frozenset[str],
) -> dict[str, Any]:
    _require_exact_keys(binding, {"path", "sha256"}, "capsule binding")
    expected_sha = _require_sha256(binding["sha256"], "capsule SHA-256")
    path = _resolve_path(contract_base, binding["path"], "capsule path")
    raw = _snapshot_bytes(
        path,
        "candidate coherence capsule",
        expected_sha,
        limit=_MAX_JSON_BYTES,
    )
    capsule = _load_json_bytes(raw, "candidate coherence capsule")
    _require_exact_keys(
        capsule,
        {
            "causal_frontier",
            "code_pin",
            "coherence_id",
            "executable_identity_sha256",
            "mechanism_id",
            "observability_contract",
            "plan_sha256",
            "schema_version",
            "watchpoint_metadata",
        },
        "candidate coherence capsule",
    )
    if capsule["schema_version"] != GATE_D_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("candidate capsule schema drifted")
    if capsule["mechanism_id"] != mechanism_id:
        raise BenchmarkValidationError("candidate capsule mechanism drifted")
    coherence_id = _require_identifier(
        capsule["coherence_id"], "capsule coherence id"
    )
    code_pin = _require_code_pin(capsule["code_pin"], "capsule code pin")
    plan_sha256 = _require_sha256(
        capsule["plan_sha256"], "capsule plan SHA-256"
    )
    executable_sha256 = _require_sha256(
        capsule["executable_identity_sha256"],
        "capsule executable identity SHA-256",
    )

    frontier = capsule["causal_frontier"]
    if not isinstance(frontier, dict):
        raise BenchmarkValidationError("capsule causal frontier must be an object")
    _require_exact_keys(frontier, {"action", "id"}, "capsule causal frontier")
    if frontier["id"] != frontier_id or frontier["action"] != expected_action:
        raise BenchmarkValidationError("candidate capsule causal frontier drifted")

    observability_binding = capsule["observability_contract"]
    if not isinstance(observability_binding, dict):
        raise BenchmarkValidationError(
            "capsule observability contract binding must be an object"
        )
    _require_exact_keys(
        observability_binding,
        {"path", "sha256"},
        "capsule observability contract binding",
    )
    observability_sha256 = _require_sha256(
        observability_binding["sha256"],
        "capsule observability contract SHA-256",
    )
    observability_path = _resolve_path(
        path.parent,
        observability_binding["path"],
        "capsule observability contract path",
    )
    observability_raw = _snapshot_bytes(
        observability_path,
        "candidate observability contract",
        observability_sha256,
        limit=_MAX_JSON_BYTES,
    )
    observability_document = _load_json_bytes(
        observability_raw, "candidate observability contract"
    )
    observability_report = audit_observability_contract(
        observability_path, observability_sha256
    )
    authority = observability_report.get("authority", {}).get("candidate")
    expected_authority = {
        "code_pin": code_pin,
        "coherence_id": coherence_id,
        "executable_identity_sha256": executable_sha256,
        "plan_sha256": plan_sha256,
    }
    if not isinstance(authority, dict) or any(
        authority.get(key) != value for key, value in expected_authority.items()
    ):
        raise BenchmarkValidationError(
            "candidate observability authority disagrees with capsule"
        )

    watchpoint_specs = {
        item.get("id"): item
        for item in observability_document.get("watchpoints", [])
        if isinstance(item, dict)
    }
    candidate_observations: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for source in observability_document.get("sources", []):
        if not isinstance(source, dict) or source.get("role") != "candidate":
            continue
        for observation in source.get("observations", []):
            if not isinstance(observation, dict):
                continue
            watchpoint_id = observation.get("watchpoint_id")
            if watchpoint_id in candidate_observations:
                raise BenchmarkValidationError(
                    "candidate watchpoint has multiple source observations"
                )
            candidate_observations[watchpoint_id] = (source, observation)

    coverage = {
        item.get("id"): item
        for item in observability_report.get("coverage", [])
        if isinstance(item, dict)
    }
    metadata = capsule["watchpoint_metadata"]
    if not isinstance(metadata, list) or not metadata:
        raise BenchmarkValidationError(
            "candidate capsule watchpoint metadata must be non-empty"
        )
    observed_watchpoints: set[str] = set()
    metadata_report: list[dict[str, Any]] = []
    for index, watchpoint in enumerate(metadata):
        if not isinstance(watchpoint, dict):
            raise BenchmarkValidationError(
                f"candidate capsule watchpoint metadata {index} must be an object"
            )
        _require_exact_keys(
            watchpoint,
            {
                "array_key",
                "array_sha256",
                "id",
                "index_prefix",
                "layer",
                "layout",
                "layout_evidence",
                "owner_ids",
                "position",
                "semantic_dtype",
                "shape",
                "source_id",
            },
            f"candidate capsule watchpoint metadata {index}",
        )
        watchpoint_id = _require_identifier(
            watchpoint["id"],
            f"candidate capsule watchpoint metadata {index} id",
        )
        if watchpoint_id in observed_watchpoints:
            raise BenchmarkValidationError("candidate capsule watchpoint duplicated")
        observed_watchpoints.add(watchpoint_id)
        source_id = _require_identifier(
            watchpoint["source_id"],
            f"candidate capsule watchpoint {watchpoint_id} source",
        )
        array_key = _require_string(
            watchpoint["array_key"],
            f"candidate capsule watchpoint {watchpoint_id} array key",
        )
        array_sha256 = _require_sha256(
            watchpoint["array_sha256"],
            f"candidate capsule watchpoint {watchpoint_id} array SHA-256",
        )
        layer = _require_nonnegative_int(
            watchpoint["layer"],
            f"candidate capsule watchpoint {watchpoint_id} layer",
        )
        position = _require_nonnegative_int(
            watchpoint["position"],
            f"candidate capsule watchpoint {watchpoint_id} position",
        )
        semantic_dtype = _require_identifier(
            watchpoint["semantic_dtype"],
            f"candidate capsule watchpoint {watchpoint_id} semantic dtype",
        )
        shape = watchpoint["shape"]
        if not isinstance(shape, list) or not shape or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in shape
        ):
            raise BenchmarkValidationError(
                f"candidate capsule watchpoint {watchpoint_id} shape is invalid"
            )
        index_prefix = watchpoint["index_prefix"]
        if not isinstance(index_prefix, list) or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in index_prefix
        ):
            raise BenchmarkValidationError(
                f"candidate capsule watchpoint {watchpoint_id} index is invalid"
            )
        layout = _require_identifier(
            watchpoint["layout"],
            f"candidate capsule watchpoint {watchpoint_id} layout",
        )
        owner_ids = watchpoint["owner_ids"]
        if (
            not isinstance(owner_ids, list)
            or not owner_ids
            or len(owner_ids) > 4
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in owner_ids
            )
            or len(set(owner_ids)) != len(owner_ids)
        ):
            raise BenchmarkValidationError(
                f"candidate capsule watchpoint {watchpoint_id} owners are invalid"
            )
        spec = watchpoint_specs.get(watchpoint_id)
        source_observation = candidate_observations.get(watchpoint_id)
        coverage_item = coverage.get(watchpoint_id)
        if (
            not isinstance(spec, dict)
            or source_observation is None
            or not isinstance(coverage_item, dict)
            or len(coverage_item.get("candidate_sources", [])) != 1
        ):
            raise BenchmarkValidationError(
                f"candidate typed watchpoint is absent: {watchpoint_id}"
            )
        source, observation = source_observation
        if (
            source.get("id") != source_id
            or observation.get("key") != array_key
            or observation.get("array_sha256") != array_sha256
            or observation.get("index_prefix") != index_prefix
            or observation.get("shape") != shape
            or spec.get("layer") != layer
            or spec.get("position") != position
            or spec.get("semantic_dtype") != semantic_dtype
            or spec.get("shape") != shape
            or coverage_item.get("semantic_dtype") != semantic_dtype
            or coverage_item.get("shape") != shape
        ):
            raise BenchmarkValidationError(
                f"candidate typed watchpoint metadata drifted: {watchpoint_id}"
            )
        layout_binding = watchpoint["layout_evidence"]
        if not isinstance(layout_binding, dict):
            raise BenchmarkValidationError(
                f"candidate watchpoint layout evidence is invalid: {watchpoint_id}"
            )
        _require_exact_keys(
            layout_binding,
            {"json_path", "path", "sha256"},
            f"candidate watchpoint {watchpoint_id} layout evidence",
        )
        layout_path = _resolve_path(
            path.parent,
            layout_binding["path"],
            f"candidate watchpoint {watchpoint_id} layout evidence path",
        )
        layout_sha256 = _require_sha256(
            layout_binding["sha256"],
            f"candidate watchpoint {watchpoint_id} layout evidence SHA-256",
        )
        layout_document = _load_json_bytes(
            _snapshot_bytes(
                layout_path,
                f"candidate watchpoint {watchpoint_id} layout evidence",
                layout_sha256,
                limit=_MAX_JSON_BYTES,
            ),
            f"candidate watchpoint {watchpoint_id} layout evidence",
        )
        if _read_json_path(
            layout_document,
            layout_binding["json_path"],
            f"candidate watchpoint {watchpoint_id} layout evidence",
        ) != {"layout": layout, "owner_ids": owner_ids}:
            raise BenchmarkValidationError(
                f"candidate watchpoint layout/owner evidence drifted: {watchpoint_id}"
            )
        metadata_report.append(
            {
                "array_sha256": array_sha256,
                "id": watchpoint_id,
                "layout": layout,
                "layout_evidence_sha256": layout_sha256,
                "owner_ids": owner_ids,
                "semantic_dtype": semantic_dtype,
                "shape": shape,
                "source_id": source_id,
            }
        )
    missing = required_watchpoints - observed_watchpoints
    if missing:
        raise BenchmarkValidationError(
            f"candidate capsule coherent watchpoints missing: {sorted(missing)}"
        )
    if frontier_id not in observed_watchpoints:
        raise BenchmarkValidationError("candidate capsule causal frontier is absent")
    return {
        "capsule_sha256": expected_sha,
        "candidate_authority": expected_authority,
        "coherence_id": coherence_id,
        "observability_classification": observability_report["classification"],
        "observability_contract_sha256": observability_sha256,
        "watchpoint_metadata": sorted(metadata_report, key=lambda item: item["id"]),
        "watchpoint_ids": sorted(observed_watchpoints),
    }


def admit_gate_d_mechanisms(
    contract_path: Path | str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Authenticate and classify every candidate without importing JAX."""

    contract_path = Path(contract_path)
    expected_contract_sha256 = _require_sha256(
        expected_contract_sha256, "contract SHA-256"
    )
    raw = _snapshot_bytes(
        contract_path,
        "Gate-D admission contract",
        expected_contract_sha256,
        limit=_MAX_JSON_BYTES,
    )
    contract = _load_json_bytes(raw, "Gate-D admission contract")
    _require_exact_keys(
        contract,
        {
            "candidates",
            "causal_frontier",
            "closed_families",
            "contract_id",
            "evidence",
            "locality_contract",
            "required_coherent_watchpoints",
            "schema_version",
            "upstream_snapshots",
        },
        "Gate-D admission contract",
    )
    if contract["schema_version"] != GATE_D_ADMISSION_SCHEMA_VERSION:
        raise BenchmarkValidationError("Gate-D admission schema drifted")
    contract_id = _require_identifier(contract["contract_id"], "contract id")
    base = contract_path.parent

    frontier = contract["causal_frontier"]
    if not isinstance(frontier, dict):
        raise BenchmarkValidationError("causal frontier must be an object")
    _require_exact_keys(
        frontier, {"evidence_id", "id", "order"}, "causal frontier"
    )
    frontier_id = _require_identifier(frontier["id"], "causal frontier id")
    if frontier["order"] != 0:
        raise BenchmarkValidationError("causal frontier order must remain zero")

    locality = contract["locality_contract"]
    if not isinstance(locality, dict):
        raise BenchmarkValidationError("locality contract must be an object")
    _require_exact_keys(
        locality,
        {
            "logical_rows",
            "max_collective_group_size",
            "no_full_pod_hidden_reconstruction",
            "no_host_effects",
        },
        "locality contract",
    )
    required_rows = _require_positive_int(
        locality["logical_rows"], "locality logical rows"
    )
    max_group = _require_positive_int(
        locality["max_collective_group_size"], "locality group size"
    )
    if required_rows != 1 or max_group != 4:
        raise BenchmarkValidationError("Gate-D locality contract drifted")
    if (
        _require_bool(
            locality["no_full_pod_hidden_reconstruction"],
            "locality full-pod rule",
        )
        is not True
        or _require_bool(locality["no_host_effects"], "locality host rule")
        is not True
    ):
        raise BenchmarkValidationError("Gate-D locality prohibition weakened")

    required_watchpoints = frozenset(
        _require_identifier_list(
            contract["required_coherent_watchpoints"],
            "required coherent watchpoints",
        )
    )
    if frontier_id not in required_watchpoints:
        raise BenchmarkValidationError("causal frontier is absent from coherent state")

    evidence = contract["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise BenchmarkValidationError("admission evidence must be non-empty")
    evidence_shas: dict[str, str] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"evidence {index} must be an object")
        _require_exact_keys(
            item,
            {"claim_scope", "id", "path", "sha256"},
            f"evidence {index}",
        )
        evidence_id = _require_identifier(item["id"], f"evidence {index} id")
        if evidence_id in evidence_shas:
            raise BenchmarkValidationError("evidence id duplicated")
        evidence_sha = _require_sha256(
            item["sha256"], f"evidence {evidence_id} SHA-256"
        )
        evidence_path = _resolve_path(base, item["path"], "evidence path")
        _verify_regular_file(
            evidence_path,
            f"admission evidence {evidence_id}",
            evidence_sha,
            limit=_MAX_EVIDENCE_BYTES,
        )
        _require_string(item["claim_scope"], f"evidence {evidence_id} scope")
        evidence_shas[evidence_id] = evidence_sha
    frontier_evidence = _require_identifier(
        frontier["evidence_id"], "causal frontier evidence id"
    )
    if frontier_evidence not in evidence_shas:
        raise BenchmarkValidationError("causal frontier evidence is absent")

    closed = contract["closed_families"]
    if not isinstance(closed, list) or not closed:
        raise BenchmarkValidationError("closed families must be non-empty")
    closed_families: dict[str, tuple[str, ...]] = {}
    closed_fingerprints: dict[str, str] = {}
    for index, family in enumerate(closed):
        if not isinstance(family, dict):
            raise BenchmarkValidationError(f"closed family {index} must be an object")
        _require_exact_keys(
            family,
            {
                "evidence_ids",
                "id",
                "mechanism_fingerprint_sha256s",
                "reason",
            },
            f"closed family {index}",
        )
        family_id = _require_identifier(family["id"], f"closed family {index} id")
        if family_id in closed_families:
            raise BenchmarkValidationError("closed family id duplicated")
        family_evidence = _require_identifier_list(
            family["evidence_ids"], f"closed family {family_id} evidence"
        )
        if any(item not in evidence_shas for item in family_evidence):
            raise BenchmarkValidationError(
                f"closed family evidence is absent: {family_id}"
            )
        _require_string(family["reason"], f"closed family {family_id} reason")
        fingerprint_values = family["mechanism_fingerprint_sha256s"]
        if not isinstance(fingerprint_values, list) or not fingerprint_values:
            raise BenchmarkValidationError(
                f"closed family fingerprints are absent: {family_id}"
            )
        for fingerprint_index, value in enumerate(fingerprint_values):
            fingerprint = _require_sha256(
                value,
                f"closed family {family_id} fingerprint {fingerprint_index}",
            )
            if fingerprint in closed_fingerprints:
                raise BenchmarkValidationError(
                    "closed mechanism fingerprint is assigned more than once"
                )
            closed_fingerprints[fingerprint] = family_id
        closed_families[family_id] = family_evidence

    snapshots = contract["upstream_snapshots"]
    if not isinstance(snapshots, list) or not snapshots:
        raise BenchmarkValidationError("upstream snapshots must be non-empty")
    snapshot_pins: dict[str, str] = {}
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict):
            raise BenchmarkValidationError(f"upstream snapshot {index} is invalid")
        _require_exact_keys(
            snapshot,
            {"commit", "id", "scope"},
            f"upstream snapshot {index}",
        )
        snapshot_id = _require_identifier(
            snapshot["id"], f"upstream snapshot {index} id"
        )
        if snapshot_id in snapshot_pins:
            raise BenchmarkValidationError("upstream snapshot id duplicated")
        snapshot_pins[snapshot_id] = _require_code_pin(
            snapshot["commit"], f"upstream snapshot {snapshot_id} commit"
        )
        _require_string(snapshot["scope"], f"upstream snapshot {snapshot_id} scope")

    candidates = contract["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise BenchmarkValidationError("candidate matrix must be non-empty")
    candidate_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise BenchmarkValidationError(f"candidate {index} must be an object")
        _require_exact_keys(
            candidate,
            {
                "causal_frontier_action",
                "coherent_state_capsule",
                "evidence_ids",
                "family_id",
                "host_effect",
                "id",
                "logical_rows",
                "max_collective_group_size",
                "mechanism_fingerprint",
                "reconstructs_full_pod_hidden",
                "summary",
            },
            f"candidate {index}",
        )
        candidate_id = _require_identifier(candidate["id"], f"candidate {index} id")
        if candidate_id in candidate_ids:
            raise BenchmarkValidationError("candidate id duplicated")
        candidate_ids.add(candidate_id)
        _require_string(candidate["summary"], f"candidate {candidate_id} summary")
        candidate_evidence = _require_identifier_list(
            candidate["evidence_ids"], f"candidate {candidate_id} evidence"
        )
        if any(item not in evidence_shas for item in candidate_evidence):
            raise BenchmarkValidationError(
                f"candidate evidence is absent: {candidate_id}"
            )
        reasons: list[str] = []
        fingerprint_sha256 = _mechanism_fingerprint(
            candidate["mechanism_fingerprint"],
            f"candidate {candidate_id} mechanism fingerprint",
        )
        family_id = candidate["family_id"]
        if family_id is not None:
            family_id = _require_identifier(
                family_id, f"candidate {candidate_id} family"
            )
            if family_id not in closed_families:
                raise BenchmarkValidationError(
                    f"candidate names unknown closed family: {candidate_id}"
                )
            if closed_fingerprints.get(fingerprint_sha256) != family_id:
                raise BenchmarkValidationError(
                    f"candidate family/fingerprint disagrees: {candidate_id}"
                )
        if fingerprint_sha256 in closed_fingerprints:
            reasons.append("DUPLICATES_CLOSED_FAMILY")
        logical_rows = _require_positive_int(
            candidate["logical_rows"], f"candidate {candidate_id} logical rows"
        )
        if logical_rows != required_rows:
            reasons.append("NOT_TRUE_ONE_ROW")
        group_size = _require_positive_int(
            candidate["max_collective_group_size"],
            f"candidate {candidate_id} group size",
        )
        if group_size > max_group:
            reasons.append("NONLOCAL_COLLECTIVE_GROUP")
        if _require_bool(
            candidate["reconstructs_full_pod_hidden"],
            f"candidate {candidate_id} full-pod hidden",
        ):
            reasons.append("FULL_POD_HIDDEN_RECONSTRUCTION")
        if _require_bool(
            candidate["host_effect"], f"candidate {candidate_id} host effect"
        ):
            reasons.append("HOST_EFFECT_OR_DISPATCH")
        action = candidate["causal_frontier_action"]
        if not isinstance(action, str) or action not in _FRONTIER_ACTIONS:
            raise BenchmarkValidationError(
                f"candidate {candidate_id} frontier action is invalid"
            )
        if action not in ("expose", "resolve"):
            reasons.append("DOES_NOT_MOVE_CAUSAL_FRONTIER")
        capsule_binding = candidate["coherent_state_capsule"]
        capsule_report: dict[str, Any] | None = None
        if capsule_binding is None:
            reasons.append("MISSING_CANDIDATE_COHERENT_CAPSULE")
        else:
            if not isinstance(capsule_binding, dict):
                raise BenchmarkValidationError(
                    f"candidate {candidate_id} capsule binding must be an object"
                )
            try:
                capsule_report = _verify_capsule(
                    binding=capsule_binding,
                    contract_base=base,
                    mechanism_id=candidate_id,
                    frontier_id=frontier_id,
                    expected_action=action,
                    required_watchpoints=required_watchpoints,
                )
            except BenchmarkValidationError as error:
                reasons.append("INVALID_CANDIDATE_COHERENT_CAPSULE")
                capsule_report = {"refusal": str(error)}
        results.append(
            {
                "admitted_offline": not reasons,
                "causal_frontier_action": action,
                "coherent_state_capsule": capsule_report,
                "evidence_sha256s": {
                    item: evidence_shas[item] for item in candidate_evidence
                },
                "family_id": family_id,
                "id": candidate_id,
                "mechanism_fingerprint_sha256": fingerprint_sha256,
                "reasons": reasons,
            }
        )

    admitted = [item["id"] for item in results if item["admitted_offline"]]
    return {
        "admitted_candidate_ids": admitted,
        "candidate_results": results,
        "causal_frontier_id": frontier_id,
        "classification": (
            "OFFLINE_CANDIDATE_ADMITTED"
            if admitted
            else "NO_ADMISSIBLE_MECHANISM"
        ),
        "contract_id": contract_id,
        "contract_sha256": expected_contract_sha256,
        "evidence_sha256s": dict(sorted(evidence_shas.items())),
        "jax_or_tpu_work_performed": False,
        "required_coherent_watchpoints": sorted(required_watchpoints),
        "schema_version": GATE_D_ADMISSION_SCHEMA_VERSION,
        "tpu_successor_authorized": False,
        "upstream_snapshot_pins": dict(sorted(snapshot_pins.items())),
    }


def write_gate_d_admission_report(
    path: Path | str, report: Mapping[str, Any]
) -> None:
    """Create one canonical report without replacing any occupied path."""

    path = Path(path)
    if not path.name or path.name in (".", ".."):
        raise BenchmarkValidationError("report output path is invalid")
    normalized = Path(os.path.abspath(os.fspath(path)))
    with _open_directory_no_symlinks(
        normalized.parent, "report output directory"
    ) as directory_fd:
        output_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                normalized.name, output_flags, 0o644, dir_fd=directory_fd
            )
        except OSError as error:
            raise BenchmarkValidationError(
                f"report output path is occupied: {path}"
            ) from error
        try:
            payload = (_canonical_json(report) + "\n").encode("ascii")
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.fsync(directory_fd)
