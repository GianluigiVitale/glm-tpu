"""Fail-closed validation for the bounded Gate-D projection TPU replay.

This module is deliberately NumPy-agnostic at import time.  The protected
runner supplies its sealed NumPy module only after the TPU runtime and source
closure have been authenticated.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any

CAPSULE_SHA256 = "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4"
CAPSULE_INPUT_SHA256 = (
    "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
)
CAPSULE_STATE_SHA256 = (
    "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236"
)
CAPSULE_EXECUTION_AUTHORITY_SHA256 = (
    "8b8c9cc79a18679628582a66c418a7f63e06553091928df58defbdde3971a660"
)
HOST_MATERIALIZATION_AUTHORITY_SHA256 = (
    "a7e5b393f7c61181f6b2aab9531d788fb27594d9cf980055c165b6f278ba4f99"
)
FRONTIER_AUTHORITY_SHA256 = (
    "4a6be0f33f221df46f384cd2047da1aa664ae4c45763e68f0f1df05f8644e20c"
)
HLO_SUCCESS_AUTHORITY_SHA256 = (
    "54eb6105b81d9ffdbdd3b90735059fb324701509fe8efd003033bb3cff7e0ad7"
)
EXPECTED_NORMALIZED_OWNER_SHA256 = (
    "28b7db466b74f462b5cd3109cb052bcce0c8be185d3f24199239f75136b22c20"
)
EXPECTED_WK_WEIGHT_FP32_SHA256 = (
    "b2c67e0fdf4d7292494778233e9813d8256f2fb88b6f7376870379832fbab24b"
)
EXPECTED_PROJECTED_KEY_OWNER_SHA256 = (
    "f16ad9903c6d219c9796f2a1aac32b0fd7b09135ecdfadb9a19dbef5cc1a9aa6"
)
EXPECTED_CURRENT_KEY_OWNER_SHA256 = (
    "5006ad4f7224047652bbc46e6275b4c82bec0dead13f5f2b5f91494a2415329b"
)
EXPECTED_STABLEHLO_SHA256 = (
    "4b3fa252e837d381208453b06d7e369947fa4c6c58c795edc8ff5a8ea8c9e2e3"
)
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "817ba2ed87c33ec928f834fcf3a003ce63d3dedf4061a7c64fe354a26ac498ea"
)

_INPUT_RECORDS = {
    "key_norm_bias_bf16_bits": {
        "array_sha256": "0ba787bd79d8778c57377d734fc7b9aa88cdd298a445b66edf96b5628a3d24f9",
        "shape": [2, 128],
        "storage_dtype": "<u2",
    },
    "key_norm_weight_bf16_bits": {
        "array_sha256": "4be4de2c68782a10e01cf60cc3ca25106a6740e1be09450c4767e0a3525df43b",
        "shape": [2, 128],
        "storage_dtype": "<u2",
    },
    "wk_scale_inv": {
        "array_sha256": "c5153a4656801de98a1031e859b74c325e5f77905eb4b8d9428593791b7c5a4b",
        "shape": [2, 1, 48],
        "storage_dtype": "<f4",
    },
    "wk_weight_bits": {
        "array_sha256": "e39bcdad6b37018cf5e0fd7c21e1a771abbabbaa85645f2fbc02db998476b4ba",
        "shape": [2, 128, 6144],
        "storage_dtype": "|u1",
    },
}

_STATE_RECORDS = {
    "normalized": {
        "array_sha256": "03adfcc99391aec20c569e1cbbc82847e157e4331b3c70f8b8981f60d1fec5fb",
        "shape": [2, 1, 6144],
        "storage_dtype": "<u2",
    },
}


def _array_sha256(value: Any, np: Any) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def projection_input_records(
    execution_authority: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the exact four input records after authenticating the catalogue."""

    records = execution_authority.get("input_arrays")
    if not isinstance(records, Mapping):
        raise TypeError("Gate-D capsule input catalogue is absent")
    selected = {name: records.get(name) for name in _INPUT_RECORDS}
    if selected != _INPUT_RECORDS:
        raise RuntimeError("Gate-D projection input authority drifted")
    return {name: dict(record) for name, record in _INPUT_RECORDS.items()}


def projection_state_records(capsule: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Authenticate the captured normalized witness and return its NPZ record."""

    if (
        capsule.get("schema_version") != 2
        or capsule.get("candidate_id") != "compensated_auxiliary_dependency"
        or capsule.get("coherence_id")
        != "greenfield_gate_d_compensated_capsule_20260831T124838Z"
        or capsule.get("producer_input_artifact")
        != {"path": "candidate-inputs.npz", "sha256": CAPSULE_INPUT_SHA256}
        or capsule.get("artifact")
        != {"path": "candidate-state.npz", "sha256": CAPSULE_STATE_SHA256}
    ):
        raise RuntimeError("Gate-D projection capsule binding drifted")
    watchpoints = capsule.get("watchpoints")
    matches = (
        [
            item
            for item in watchpoints
            if isinstance(item, Mapping) and item.get("id") == "layer1.normalized"
        ]
        if isinstance(watchpoints, list)
        else []
    )
    expected_arrays = [
        {
            "array_key": "normalized",
            "array_sha256": EXPECTED_NORMALIZED_OWNER_SHA256,
            "index_prefix": [owner, 0],
            "owner_axis": None,
            "owner_axis_ids": [],
            "owner_id": owner,
            "role": f"owner{owner}",
            "semantic_dtype": "bf16_bits",
            "shape": [6144],
            "storage_dtype": "<u2",
        }
        for owner in (0, 1)
    ]
    if (
        len(matches) != 1
        or matches[0].get("layer") != 1
        or matches[0].get("position") != 8155
        or matches[0].get("layout") != "lp2.local"
        or matches[0].get("owner_ids") != [0, 1]
        or matches[0].get("arrays") != expected_arrays
    ):
        raise RuntimeError("Gate-D normalized watchpoint authority drifted")
    return {name: dict(record) for name, record in _STATE_RECORDS.items()}


def validate_predecessors(
    *,
    capsule: Mapping[str, Any],
    execution_authority: Mapping[str, Any],
    host_materialization: Mapping[str, Any],
    frontier: Mapping[str, Any],
    hlo_success: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on every predecessor that authorizes this bounded replay."""

    projection_input_records(execution_authority)
    projection_state_records(capsule)
    comparison = host_materialization.get("comparison")
    cpu_control = frontier.get("cpu_jax_dot_control")
    optimized_hlo = hlo_success.get("optimized_hlo")
    stablehlo = hlo_success.get("stablehlo")
    authorization = hlo_success.get("authorization")
    if (
        execution_authority.get("schema_version") != 1
        or execution_authority.get("authority_kind") != "gate.d.capsule.execution.v1"
        or execution_authority.get("candidate_id") != "compensated_auxiliary_dependency"
        or not isinstance(comparison, Mapping)
        or host_materialization.get("gate_d_closed") is not False
        or host_materialization.get("performance_claim") is not False
        or comparison.get("wk_weight_fp32_sha256") != EXPECTED_WK_WEIGHT_FP32_SHA256
        or frontier.get("classification")
        != "CPU_F32_DOT_CONTROL_EXACT_ACCEPTED_KEY_CAPTURED_INPUT;ENUMERATED_REDUCTION_PROBES_REJECTED;TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN"
        or frontier.get("gate_d_closed") is not False
        or frontier.get("root_cause_proven") is not False
        or frontier.get("tpu_compile_or_execution_performed") is not False
        or not isinstance(cpu_control, Mapping)
        or cpu_control.get("projection_sha256") != EXPECTED_PROJECTED_KEY_OWNER_SHA256
        or not isinstance(cpu_control.get("accepted"), Mapping)
        or cpu_control["accepted"].get("sha256") != EXPECTED_CURRENT_KEY_OWNER_SHA256
        or hlo_success.get("classification")
        != "HLO_CAUSAL_STRUCTURE_ACCEPTED;PP16_OWNER_LOCALITY_ACCEPTED;TPU_NUMERICAL_UNPROVEN;HLO_ACQUIRED_TERMINAL_VERIFIED;GATE_D_OPEN"
        or hlo_success.get("gate_d_closed") is not False
        or not isinstance(authorization, Mapping)
        or authorization
        != {
            "full_8k": False,
            "numerical_execution": False,
            "performance_claim": False,
            "persistence_only": True,
        }
        or not isinstance(optimized_hlo, Mapping)
        or optimized_hlo.get("sha256") != EXPECTED_OPTIMIZED_HLO_SHA256
        or not isinstance(stablehlo, Mapping)
        or stablehlo.get("sha256") != EXPECTED_STABLEHLO_SHA256
    ):
        raise RuntimeError("Gate-D projection numerical predecessor drifted")
    return {
        "capsule_input_sha256": CAPSULE_INPUT_SHA256,
        "capsule_sha256": CAPSULE_SHA256,
        "capsule_state_sha256": CAPSULE_STATE_SHA256,
        "execution_authority_sha256": CAPSULE_EXECUTION_AUTHORITY_SHA256,
        "frontier_authority_sha256": FRONTIER_AUTHORITY_SHA256,
        "hlo_success_authority_sha256": HLO_SUCCESS_AUTHORITY_SHA256,
        "host_materialization_authority_sha256": HOST_MATERIALIZATION_AUTHORITY_SHA256,
    }


def validate_wk_weight(value: Any, np: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    identity = {
        "array_sha256": _array_sha256(array, np),
        "semantic_dtype": str(array.dtype),
        "shape": list(array.shape),
    }
    if (
        array.dtype != np.dtype(np.float32)
        or array.shape != (2, 128, 6144)
        or identity["array_sha256"] != EXPECTED_WK_WEIGHT_FP32_SHA256
        or not bool(np.all(np.isfinite(array)))
    ):
        raise RuntimeError("Gate-D projection FP32 key weight drifted")
    return identity


def classify_projection_outputs(outputs: Mapping[str, Any], np: Any) -> dict[str, Any]:
    """Classify exactly three bounded outputs; acceptance never closes Gate D."""

    expected = {
        "normalized_hidden_owners": (
            (2, 1, 6144),
            "bfloat16",
            EXPECTED_NORMALIZED_OWNER_SHA256,
        ),
        "projected_key_owners": (
            (2, 1, 128),
            "float32",
            EXPECTED_PROJECTED_KEY_OWNER_SHA256,
        ),
        "current_key_owners": (
            (2, 1, 128),
            "float32",
            EXPECTED_CURRENT_KEY_OWNER_SHA256,
        ),
    }
    if set(outputs) != set(expected):
        raise RuntimeError("Gate-D projection output catalogue drifted")
    records: dict[str, Any] = {}
    accepted = True
    for name, (shape, dtype, expected_owner_hash) in expected.items():
        value = np.ascontiguousarray(outputs[name])
        shape_exact = value.shape == shape
        dtype_exact = str(value.dtype) == dtype
        finite = bool(np.all(np.isfinite(value)))
        owners_equal = shape_exact and value[0].tobytes(order="C") == value[1].tobytes(
            order="C"
        )
        owner_hashes = [
            _array_sha256(value[index, 0], np) if shape_exact else None
            for index in range(2)
        ]
        witness_exact = owner_hashes == [expected_owner_hash, expected_owner_hash]
        record = {
            "dtype_exact": dtype_exact,
            "finite": finite,
            "owner_sha256": owner_hashes,
            "owners_equal": owners_equal,
            "shape_exact": shape_exact,
            "witness_exact": witness_exact,
        }
        records[name] = record
        accepted = accepted and all(
            (shape_exact, dtype_exact, finite, owners_equal, witness_exact)
        )
    return {
        "accepted_tpu_projection_match": accepted,
        "outputs": records,
    }
