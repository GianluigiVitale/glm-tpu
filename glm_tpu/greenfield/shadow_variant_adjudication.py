"""Fail-closed normal-form adjudication for Gate-D shadow variants.

This module deliberately has no JAX or NumPy dependency.  It answers one
offline question: does a proposed rounded, compensated, or auxiliary shadow
escape the mechanisms already rejected by evidence?  It never authorizes TPU
work; the separate Gate-D admission contract remains authoritative.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Iterator, Mapping

from .errors import BenchmarkValidationError


__all__ = (
    "SHADOW_VARIANT_SCHEMA_VERSION",
    "adjudicate_shadow_variants",
    "write_shadow_variant_report",
)


SHADOW_VARIANT_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024 * 1024
_NORMAL_FORM_KEYS = {
    "compensation",
    "dependency",
    "normalization_input",
    "primary_recurrence",
}
_ALLOWED_NORMAL_FORMS = {
    "compensation": {"cancelled", "none", "restore.unrounded"},
    "dependency": {"device.auxiliary", "host.callback", "none", "primary"},
    "normalization_input": {
        "bf16.rounded.widened",
        "persistent.shadow",
        "transient.fp32.sum",
    },
    "primary_recurrence": {"bf16.rounded", "fp32.unrounded"},
}
_REQUIRED_WATCHPOINTS = {
    "layer1.rms_input_fp32",
    "layer1.normalized",
    "layer1.cache_history",
    "layer1.query",
    "layer1.head_weights",
    "layer1.current_key",
    "layer1.scorer_event1",
}
_EXPECTED_VARIANTS = {
    "auxiliary_device_tuple_dependency": {
        "compensation": "none",
        "dependency": "device.auxiliary",
        "normalization_input": "persistent.shadow",
        "primary_recurrence": "bf16.rounded",
    },
    "compensated_auxiliary_dependency": {
        "compensation": "cancelled",
        "dependency": "device.auxiliary",
        "normalization_input": "persistent.shadow",
        "primary_recurrence": "bf16.rounded",
    },
    "compensated_cancelled_no_dependency": {
        "compensation": "cancelled",
        "dependency": "none",
        "normalization_input": "persistent.shadow",
        "primary_recurrence": "bf16.rounded",
    },
    "compensated_restore_unrounded": {
        "compensation": "restore.unrounded",
        "dependency": "primary",
        "normalization_input": "persistent.shadow",
        "primary_recurrence": "bf16.rounded",
    },
    "host_shadow_observer": {
        "compensation": "none",
        "dependency": "host.callback",
        "normalization_input": "persistent.shadow",
        "primary_recurrence": "bf16.rounded",
    },
    "rounded_widened_primary": {
        "compensation": "none",
        "dependency": "primary",
        "normalization_input": "bf16.rounded.widened",
        "primary_recurrence": "bf16.rounded",
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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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


def _snapshot(path: Path, label: str, expected_sha256: str, limit: int) -> bytes:
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
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
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


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} is not a canonical identifier")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkValidationError(f"{label} must be boolean")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkValidationError(f"{label} must be a positive integer")
    return value


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkValidationError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _normal_form(value: Any, label: str) -> tuple[dict[str, str], str]:
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{label} must be an object")
    _exact_keys(value, _NORMAL_FORM_KEYS, label)
    result: dict[str, str] = {}
    for key in sorted(_NORMAL_FORM_KEYS):
        item = _identifier(value[key], f"{label}.{key}")
        if item not in _ALLOWED_NORMAL_FORMS[key]:
            raise BenchmarkValidationError(f"{label}.{key} is unsupported")
        result[key] = item
    fingerprint = sha256(_canonical_json(result).encode("ascii")).hexdigest()
    return result, fingerprint


def _classify(form: Mapping[str, str]) -> tuple[str, str, list[str]]:
    primary = form["primary_recurrence"]
    normalization = form["normalization_input"]
    compensation = form["compensation"]
    dependency = form["dependency"]
    if dependency == "host.callback":
        return (
            "CLOSED_HOST_OBSERVER",
            "callback_observation",
            ["HOST_EFFECT_FORBIDDEN", "EXECUTABLE_CLASS_PERTURBATION"],
        )
    if primary == "fp32.unrounded" or compensation == "restore.unrounded":
        return (
            "REJECTED_SEMANTIC_DRIFT",
            "direct_unrounded_shadow_substitution",
            ["REMOVES_ACCEPTED_BF16_RECURRENCE"],
        )
    if dependency == "none":
        return (
            "REJECTED_NONCAUSAL",
            "no_device_dependency",
            ["NO_LATER_DEVICE_CONSUMER"],
        )
    if dependency == "primary" and normalization == "bf16.rounded.widened":
        return (
            "DUPLICATE_CLOSED",
            "rounded_bf16_recurrence",
            ["DUPLICATES_ACCEPTED_PRIMARY_STATE", "SPLIT_RMS_NORMAL_FORM_REJECTED"],
        )
    if dependency == "primary" and normalization == "transient.fp32.sum":
        return (
            "DUPLICATE_CLOSED",
            "transient_unrounded_sum",
            ["DUPLICATES_EXISTING_RMS_INPUT"],
        )
    if dependency == "device.auxiliary" and primary == "bf16.rounded":
        canonical_class = (
            "compensated_auxiliary_dependency"
            if compensation == "cancelled"
            else "auxiliary_device_tuple_dependency"
        )
        return (
            "UNADJUDICATED",
            canonical_class,
            [
                "MISSING_DECLARED_SOURCE_STABLEHLO_CERTIFICATE",
                "MISSING_OFFLINE_COHERENT_STATE_CAPSULE",
                "MISSING_SEPARATE_GATE_D_ADMISSION",
            ],
        )
    raise BenchmarkValidationError("shadow normal form has no fail-closed rule")


def adjudicate_shadow_variants(
    contract_path: Path | str, expected_contract_sha256: str
) -> dict[str, Any]:
    """Authenticate and canonicalize the complete configured variant set."""

    contract_path = Path(contract_path)
    expected_contract_sha256 = _sha(
        expected_contract_sha256, "contract SHA-256"
    )
    contract = _load_json(
        _snapshot(
            contract_path,
            "shadow variant contract",
            expected_contract_sha256,
            _MAX_JSON_BYTES,
        ),
        "shadow variant contract",
    )
    _exact_keys(
        contract,
        {
            "accepted_semantics",
            "claim_scope",
            "contract_id",
            "evidence",
            "locality_contract",
            "required_coherent_watchpoints",
            "schema_version",
            "variants",
        },
        "shadow variant contract",
    )
    if (
        not isinstance(contract["schema_version"], int)
        or isinstance(contract["schema_version"], bool)
        or contract["schema_version"] != SHADOW_VARIANT_SCHEMA_VERSION
    ):
        raise BenchmarkValidationError("shadow variant schema drifted")
    contract_id = _identifier(contract["contract_id"], "contract id")
    claim_scope = contract["claim_scope"]
    if not isinstance(claim_scope, str) or not claim_scope:
        raise BenchmarkValidationError("claim scope must be a non-empty string")
    base = contract_path.parent

    semantics = contract["accepted_semantics"]
    if not isinstance(semantics, dict):
        raise BenchmarkValidationError("accepted semantics must be an object")
    _exact_keys(
        semantics,
        {"normalization_input", "returned_recurrence"},
        "accepted semantics",
    )
    if semantics != {
        "normalization_input": "transient.fp32.sum",
        "returned_recurrence": "bf16.rounded",
    }:
        raise BenchmarkValidationError("accepted BF16 recurrence was weakened")

    locality = contract["locality_contract"]
    if not isinstance(locality, dict):
        raise BenchmarkValidationError("locality contract must be an object")
    _exact_keys(
        locality,
        {
            "logical_rows",
            "max_collective_group_size",
            "no_full_pod_hidden_reconstruction",
            "no_host_effects",
        },
        "locality contract",
    )
    if (
        _positive_int(locality["logical_rows"], "logical rows") != 1
        or _positive_int(locality["max_collective_group_size"], "group size") != 4
        or not _boolean(locality["no_full_pod_hidden_reconstruction"], "full-pod rule")
        or not _boolean(locality["no_host_effects"], "host rule")
    ):
        raise BenchmarkValidationError("Gate-D locality contract was weakened")

    watchpoints = contract["required_coherent_watchpoints"]
    if not isinstance(watchpoints, list):
        raise BenchmarkValidationError("seven-watchpoint coherence contract drifted")
    checked_watchpoints = [
        _identifier(item, f"watchpoint {index}")
        for index, item in enumerate(watchpoints)
    ]
    if set(checked_watchpoints) != _REQUIRED_WATCHPOINTS:
        raise BenchmarkValidationError("seven-watchpoint coherence contract drifted")
    if len(checked_watchpoints) != len(set(checked_watchpoints)):
        raise BenchmarkValidationError("coherent watchpoint duplicated")

    evidence = contract["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise BenchmarkValidationError("evidence must be a non-empty list")
    evidence_shas: dict[str, str] = {}
    evidence_report: list[dict[str, str]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(f"evidence {index} must be an object")
        _exact_keys(item, {"claim_scope", "id", "path", "sha256"}, f"evidence {index}")
        evidence_id = _identifier(item["id"], f"evidence {index} id")
        if evidence_id in evidence_shas:
            raise BenchmarkValidationError("evidence id duplicated")
        evidence_sha = _sha(item["sha256"], f"evidence {evidence_id} SHA-256")
        evidence_path = _resolve(base, item["path"], f"evidence {evidence_id} path")
        _snapshot(
            evidence_path,
            f"shadow evidence {evidence_id}",
            evidence_sha,
            _MAX_EVIDENCE_BYTES,
        )
        if not isinstance(item["claim_scope"], str) or not item["claim_scope"]:
            raise BenchmarkValidationError(f"evidence {evidence_id} scope is empty")
        evidence_shas[evidence_id] = evidence_sha
        evidence_report.append({"id": evidence_id, "sha256": evidence_sha})

    variants = contract["variants"]
    if not isinstance(variants, list) or not variants:
        raise BenchmarkValidationError("variants must be a non-empty list")
    preflight_ids: list[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise BenchmarkValidationError(f"variant {index} must be an object")
        preflight_ids.append(_identifier(variant.get("id"), f"variant {index} id"))
    if len(preflight_ids) != len(set(preflight_ids)):
        raise BenchmarkValidationError("variant id duplicated")
    if set(preflight_ids) != set(_EXPECTED_VARIANTS):
        raise BenchmarkValidationError("versioned shadow variant catalogue drifted")
    ids: set[str] = set()
    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise BenchmarkValidationError(f"variant {index} must be an object")
        _exact_keys(
            variant,
            {
                "evidence_ids",
                "host_effect",
                "id",
                "logical_rows",
                "max_collective_group_size",
                "normal_form",
                "reconstructs_full_pod_hidden",
                "summary",
            },
            f"variant {index}",
        )
        variant_id = _identifier(variant["id"], f"variant {index} id")
        if variant_id in ids:
            raise BenchmarkValidationError("variant id duplicated")
        ids.add(variant_id)
        if not isinstance(variant["summary"], str) or not variant["summary"]:
            raise BenchmarkValidationError(f"variant {variant_id} summary is empty")
        cited = variant["evidence_ids"]
        if not isinstance(cited, list) or not cited:
            raise BenchmarkValidationError(f"variant {variant_id} evidence is invalid")
        checked_citations = [
            _identifier(item, f"variant {variant_id} evidence {citation_index}")
            for citation_index, item in enumerate(cited)
        ]
        if (
            len(checked_citations) != len(set(checked_citations))
            or any(item not in evidence_shas for item in checked_citations)
        ):
            raise BenchmarkValidationError(f"variant {variant_id} evidence is invalid")
        form, fingerprint = _normal_form(
            variant["normal_form"], f"variant {variant_id} normal form"
        )
        if form != _EXPECTED_VARIANTS[variant_id]:
            raise BenchmarkValidationError(
                f"versioned shadow variant normal form drifted: {variant_id}"
            )
        classification, canonical_class, reasons = _classify(form)
        locality_reasons: list[str] = []
        if _positive_int(variant["logical_rows"], f"variant {variant_id} rows") != 1:
            locality_reasons.append("DEAD_ROWS_FORBIDDEN")
        if _positive_int(
            variant["max_collective_group_size"], f"variant {variant_id} group"
        ) > 4:
            locality_reasons.append("FULL_POD_OR_OVERSIZED_GROUP_FORBIDDEN")
        if _boolean(variant["host_effect"], f"variant {variant_id} host effect"):
            locality_reasons.append("HOST_EFFECT_FORBIDDEN")
        if _boolean(
            variant["reconstructs_full_pod_hidden"],
            f"variant {variant_id} full-pod reconstruction",
        ):
            locality_reasons.append("FULL_POD_HIDDEN_RECONSTRUCTION_FORBIDDEN")
        if locality_reasons:
            classification = "ILLEGAL_LOCALITY"
            canonical_class = "illegal_execution_shape"
            reasons = sorted(set(locality_reasons + reasons))
        if classification == "UNADJUDICATED":
            unresolved.append(variant_id)
        results.append(
            {
                "canonical_class": canonical_class,
                "classification": classification,
                "evidence_sha256s": {
                    item: evidence_shas[item] for item in sorted(checked_citations)
                },
                "id": variant_id,
                "normal_form": form,
                "normal_form_sha256": fingerprint,
                "reasons": reasons,
            }
        )

    expected_unresolved = {
        "auxiliary_device_tuple_dependency",
        "compensated_auxiliary_dependency",
    }
    if set(unresolved) != expected_unresolved:
        raise BenchmarkValidationError("unresolved shadow variant set drifted")
    classification = (
        "NO_OFFLINE_COMPLETE_SHADOW_VARIANT;"
        "AUXILIARY_DEVICE_VARIANTS_REMAIN_UNADJUDICATED;"
        "GATE_D_OPEN;NO_TPU_SUCCESSOR"
    )
    return {
        "classification": classification,
        "claim_scope": claim_scope,
        "contract_id": contract_id,
        "contract_sha256": expected_contract_sha256,
        "evidence": sorted(evidence_report, key=lambda item: item["id"]),
        "gate_d_closed": False,
        "required_before_compile_only_acquisition_review": [
            "declared_source_semantics_certificate",
            "causal_stablehlo_contract",
            "offline_candidate_coherent_seven_watchpoint_capsule",
            "separate_gate_d_admission",
        ],
        "required_before_numerical_run_review": [
            "preserved_exact_optimized_tpu_hlo_value_flow",
            "compile_only_acquisition_integrity_and_cleanup",
        ],
        "schema_version": SHADOW_VARIANT_SCHEMA_VERSION,
        "tpu_successor_authorized": False,
        "unresolved_variant_ids": sorted(unresolved),
        "variant_results": sorted(results, key=lambda item: item["id"]),
    }


def write_shadow_variant_report(path: Path | str, report: Mapping[str, Any]) -> None:
    """Write canonical JSON once, refusing files and symlinked path components."""

    path = Path(path)
    normalized = Path(os.path.abspath(os.fspath(path)))
    payload = (_canonical_json(report) + "\n").encode("ascii")
    with _open_directory_no_symlinks(normalized.parent, "report parent") as parent:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(normalized.name, flags, 0o644, dir_fd=parent)
        except FileExistsError as error:
            raise BenchmarkValidationError(
                f"report path is occupied: {path}"
            ) from error
        except OSError as error:
            raise BenchmarkValidationError(f"cannot create report: {path}") from error
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.fsync(parent)
