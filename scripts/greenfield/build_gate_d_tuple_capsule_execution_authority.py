#!/usr/bin/env python3
"""Derive the canonical outer authority for one sealed Gate-D tuple capsule."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.errors import BenchmarkValidationError  # noqa: E402
from glm_tpu.greenfield.gate_d_precompile_admission import (  # noqa: E402
    _binding,
    _canonical_json,
    _exact_keys,
    _inspect_npz,
    _load_json,
    write_gate_d_precompile_admission_report,
)


_CANDIDATE_ID = "auxiliary_device_tuple_dependency"
_MAX_JSON_BYTES = 16 << 20
_MAX_ARTIFACT_BYTES = 512 << 20
_CAPSULE_KEYS = {
    "artifact",
    "candidate_id",
    "claim_scope",
    "code_pin",
    "coherence_id",
    "plan_sha256",
    "producer_device_evidence",
    "producer_input_artifact",
    "producer_receipt",
    "producer_success",
    "schema_version",
    "source_authority_sha256",
    "stablehlo_authority_sha256",
    "watchpoints",
}
_RECEIPT_KEYS = {
    "artifact",
    "backend",
    "candidate",
    "claim_scope",
    "coherence_id",
    "device_evidence",
    "environment",
    "execution",
    "input_arrays",
    "installed_producer",
    "loaded_dependencies",
    "producer",
    "schema_version",
    "source_snapshot",
    "tensor_receipts",
    "upstream_inputs",
    "watchpoint_manifest_sha256",
}
_EXPECTED_ARRAYS = {
    "event1_positions": ("<i4", (1, 2048)),
    "event1_scores": ("<f4", (1, 2048)),
    "event1_valid_count": ("<i4", (1,)),
    "rms_hidden_update": ("<u2", (6144,)),
    "rms_residual": ("<u2", (6144,)),
}


def _canonical_document(raw: bytes, label: str) -> dict[str, Any]:
    document = _load_json(raw, label)
    if raw != (_canonical_json(document) + "\n").encode("ascii"):
        raise BenchmarkValidationError(f"{label} is not canonical JSON")
    return document


def _bound_document(
    base: Path, binding: Any, label: str
) -> tuple[Path, str, dict[str, Any]]:
    path, digest, raw = _binding(
        base, binding, label, limit=_MAX_JSON_BYTES
    )
    return path, digest, _canonical_document(raw, label)


def _bound_artifact(
    base: Path, binding: Any, label: str
) -> tuple[Path, str, bytes]:
    return _binding(base, binding, label, limit=_MAX_ARTIFACT_BYTES)


def build_execution_authority(
    capsule_path: Path, expected_capsule_sha256: str
) -> dict[str, Any]:
    """Derive authority fields without importing JAX or trusting array values."""

    capsule_path = Path(capsule_path)
    capsule_path, capsule_sha, capsule_raw = _binding(
        capsule_path.parent,
        {"path": str(capsule_path), "sha256": expected_capsule_sha256},
        "tuple capsule",
        limit=_MAX_JSON_BYTES,
    )
    capsule = _canonical_document(capsule_raw, "tuple capsule")
    _exact_keys(capsule, _CAPSULE_KEYS, "tuple capsule")
    if (
        capsule["schema_version"] != 2
        or capsule["candidate_id"] != _CANDIDATE_ID
    ):
        raise BenchmarkValidationError("tuple capsule identity drifted")

    _, artifact_sha, artifact_raw = _bound_artifact(
        capsule_path.parent, capsule["artifact"], "tuple capsule state"
    )
    _, input_sha, input_raw = _bound_artifact(
        capsule_path.parent,
        capsule["producer_input_artifact"],
        "tuple capsule inputs",
    )
    _, device_sha, device_raw = _bound_artifact(
        capsule_path.parent,
        capsule["producer_device_evidence"],
        "tuple capsule device evidence",
    )
    _, receipt_sha, receipt = _bound_document(
        capsule_path.parent,
        capsule["producer_receipt"],
        "tuple capsule producer receipt",
    )
    _, _, success = _bound_document(
        capsule_path.parent,
        capsule["producer_success"],
        "tuple capsule SUCCESS",
    )
    _exact_keys(receipt, _RECEIPT_KEYS, "tuple capsule producer receipt")
    _exact_keys(
        success,
        {
            "artifact_sha256",
            "device_evidence_sha256",
            "input_artifact_sha256",
            "producer_receipt_sha256",
            "schema_version",
        },
        "tuple capsule SUCCESS",
    )
    if (
        receipt["schema_version"] != 1
        or success["schema_version"] != 1
        or receipt["candidate"].get("id") != _CANDIDATE_ID
        or receipt["coherence_id"] != capsule["coherence_id"]
        or receipt["artifact"]
        != {"bytes": len(artifact_raw), "sha256": artifact_sha}
        or receipt["device_evidence"]
        != {"bytes": len(device_raw), "sha256": device_sha}
        or success
        != {
            "artifact_sha256": artifact_sha,
            "device_evidence_sha256": device_sha,
            "input_artifact_sha256": input_sha,
            "producer_receipt_sha256": receipt_sha,
            "schema_version": 1,
        }
    ):
        raise BenchmarkValidationError("tuple capsule publication tuple drifted")

    arrays = _inspect_npz(artifact_raw)
    selected: dict[str, bytes] = {}
    for name, (dtype, shape) in _EXPECTED_ARRAYS.items():
        array = arrays.get(name)
        if (
            not isinstance(array, dict)
            or array.get("storage_dtype") != dtype
            or array.get("shape") != shape
            or not isinstance(array.get("raw"), bytes)
        ):
            raise BenchmarkValidationError(
                f"tuple capsule authority array drifted: {name}"
            )
        selected[name] = array["raw"]
    valid_count = struct.unpack("<i", selected["event1_valid_count"])[0]
    if valid_count <= 0 or valid_count > 2048:
        raise BenchmarkValidationError("tuple capsule valid count is invalid")

    authority: dict[str, Any] = {
        "authority_kind": "gate.d.capsule.execution.v1",
        "candidate_id": _CANDIDATE_ID,
        "environment": receipt["environment"],
        "expected_outputs": {
            "event1_positions_sha256": sha256(
                selected["event1_positions"]
            ).hexdigest(),
            "event1_scores_sha256": sha256(
                selected["event1_scores"]
            ).hexdigest(),
            "event1_valid_count": valid_count,
            "rms_hidden_update_sha256": sha256(
                selected["rms_hidden_update"]
            ).hexdigest(),
            "rms_residual_sha256": sha256(
                selected["rms_residual"]
            ).hexdigest(),
        },
        "input_arrays": receipt["input_arrays"],
        "installed_producer": receipt["installed_producer"],
        "loaded_dependencies": receipt["loaded_dependencies"],
        "producer": receipt["producer"],
        "schema_version": 1,
        "source_snapshot": receipt["source_snapshot"],
        "tensor_receipts": receipt["tensor_receipts"],
        "upstream_inputs": receipt["upstream_inputs"],
    }
    return authority


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--expected-capsule-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if "jax" in sys.modules:
        raise BenchmarkValidationError("authority builder must start without JAX")
    authority = build_execution_authority(
        arguments.capsule, arguments.expected_capsule_sha256
    )
    if "jax" in sys.modules:
        raise BenchmarkValidationError("authority builder imported JAX")
    write_gate_d_precompile_admission_report(arguments.output, authority)
    print(
        json.dumps(
            {
                "authority_sha256": sha256(
                    (_canonical_json(authority) + "\n").encode("ascii")
                ).hexdigest(),
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BenchmarkValidationError as error:
        raise SystemExit(
            f"Gate-D tuple capsule authority refused: {error}"
        ) from error
