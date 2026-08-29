"""Offline adjudication of the sealed PP16 layer-1 BF16 straddler.

This module intentionally makes no numerical-correctness or Gate-D claim.  It
answers one smaller question: can the retained BF16 carried/normalized rows
identify the accepted FP32 value entering layer-1 RMSNorm at hidden index
2795?  All public entry points fail closed on the exact protected artifacts.
"""

from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from ..errors import BenchmarkValidationError
from ..kernels.stage_local import STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE
from .pp16_dense_boundary import derive_expected_dense_boundary_bits

SOURCE_CAPTURE_SHA256 = (
    "534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0"
)
SOURCE_COMPARISON_SHA256 = (
    "06ee82b9d487e3fdf8f9f19d4e824e33f1f9d453738a0090ace5e2cac7272a4d"
)
SOURCE_SUMMARY_SHA256 = (
    "b01a5ac10a080da0b4d2ad034e6932b778574e718625662d91b92b69f8350d2c"
)
SOURCE_OPTIMIZED_HLO_SHA256 = (
    "634cf81a31a89aaa704a1715ae7d2cb4fd96b588463d76c6061f5c90ab4607ca"
)
ACCEPTED_LAYER1_SHA256 = (
    "79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054"
)
DB550_BOUNDARY_SHA256 = (
    "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
)
SOURCE_RUNNER_SHA256 = (
    "fd51aacb2426a2e9ae5e45425253c5952203310c6864a0866b34675322947a80"
)
RUNTIME_LAYER1_INPUT_NORM_SHA256 = (
    "10e34f4f99c638b29557526283205071c1ac8f81f168f4a6817e7e1def4b6c87"
)
CURRENT_CARRIED_SHA256 = (
    "3f6c86ed6e96a59adfe706a522297bf83c2ed0802a36ede9f06a88cf6f3f53d2"
)
ACCEPTED_CARRIED_SHA256 = (
    "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c"
)
ACCEPTED_NORMALIZED_SHA256 = (
    "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
)
HIDDEN_INDEX = 2795
CURRENT_NORMALIZED_BITS = 48_422
ACCEPTED_NORMALIZED_BITS = 48_423
CARRIED_BITS_AT_INDEX = 47_953
NORM_WEIGHT_BITS_AT_INDEX = 15_762
EXPECTED_CARRIED_MISMATCHES = 968
CLASSIFICATION = "BF16_BOUNDARY_INSUFFICIENT_FOR_FP32_CAUSAL_ADJUDICATION"

_FLOAT_TYPE_CORRECTION_MARKER = '"float_type_correction_info"'
_BF16_ORIGINAL_TYPE_MARKER = '"original_type":"BF16"'
_RMS_SCOPE = "greenfield_pp16_feature2_rms/mul"
_RUNTIME_LAYER1_INPUT_NORM_NAME = "attention.slot_01.input_norm"
_RUNTIME_FEATURE_FILENAME = (
    "base_decoder_runtime_feature/stage_00/device_slot_{device_slot:02d}.safetensors"
)
_RUNTIME_LAYER1_INPUT_NORM_OFFSET = 91_144_776


def _read_exact_bytes(path: Path, expected_sha256: str) -> bytes:
    """Read once and bind every later parser to the authenticated buffer."""

    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise BenchmarkValidationError(f"PP16 straddler sealed input drifted: {path}")
    return raw


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(bits)
    if value.dtype != np.uint16:
        raise ValueError("BF16 bit storage must be uint16")
    return value.view(ml_dtypes.bfloat16).astype(np.float32)


def _float32_to_bf16_bits(value: np.ndarray | np.float32) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=ml_dtypes.bfloat16)).view(
        np.uint16
    )


def _finite_bf16_preimages(bits: int) -> np.ndarray:
    """Enumerate every finite FP32 value that rounds to one BF16 value."""

    if not isinstance(bits, int) or isinstance(bits, bool) or not 0 <= bits <= 0xFFFF:
        raise ValueError("BF16 bits must be one uint16 integer")
    stored = np.asarray([bits], dtype=np.uint16)
    center = _bf16_bits_to_float32(stored)[0]
    if not np.isfinite(center):
        raise ValueError("BF16 preimage target must be finite")
    center_code = int(np.asarray(center, dtype=np.float32).view(np.uint32))
    lower_code = max(0, center_code - 65_536)
    upper_code = min(0xFFFFFFFF, center_code + 65_536)
    codes = np.arange(lower_code, upper_code + 1, dtype=np.uint32)
    values = codes.view(np.float32)
    selected = values[_float32_to_bf16_bits(values) == np.uint16(bits)]
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("finite BF16 preimage enumeration failed")
    return np.ascontiguousarray(selected)


def _literal_double_round_outputs(norm_weight_bits: int) -> set[int]:
    """Return every finite output of BF16(BF16(x) * BF16(weight))."""

    weight = _bf16_bits_to_float32(np.asarray([norm_weight_bits], dtype=np.uint16))[0]
    all_bits = np.arange(65_536, dtype=np.uint16)
    all_values = _bf16_bits_to_float32(all_bits)
    finite = np.isfinite(all_values)
    outputs = _float32_to_bf16_bits(all_values[finite] * weight)
    return {int(value) for value in np.unique(outputs)}


def _single_round_witness(
    carried_bits: np.ndarray,
    *,
    norm_weight_bits: int,
    target_normalized_bits: int,
    hidden_index: int,
) -> dict[str, Any]:
    """Construct one deterministic FP32 row for a single-round target.

    Every coordinate except ``hidden_index`` uses its BF16 center.  The target
    coordinate is exhaustively searched over all FP32 values with the same
    BF16 round.  The reduction is explicitly the NumPy FP32 reference model;
    this is a non-injectivity witness, not a claim about TPU association.
    """

    bits = np.ascontiguousarray(carried_bits)
    if bits.shape != (6144,) or bits.dtype != np.uint16:
        raise ValueError("straddler carried row must be uint16[6144]")
    if not 0 <= hidden_index < bits.size:
        raise ValueError("straddler hidden index is out of bounds")
    center = _bf16_bits_to_float32(bits)
    candidates = _finite_bf16_preimages(int(bits[hidden_index]))
    gamma = _bf16_bits_to_float32(np.asarray([norm_weight_bits], dtype=np.uint16))[0]
    center_square = np.float32(center[hidden_index] * center[hidden_index])
    other_square_sum = np.float32(
        np.sum(center * center, dtype=np.float32) - center_square
    )
    square_sum = np.asarray(
        other_square_sum + candidates * candidates,
        dtype=np.float32,
    )
    mean = np.asarray(square_sum / np.float32(bits.size), dtype=np.float32)
    inverse = np.asarray(
        np.float32(1.0) / np.sqrt(mean + np.float32(1e-5), dtype=np.float32),
        dtype=np.float32,
    )
    outputs = _float32_to_bf16_bits(candidates * inverse * gamma)
    matches = np.flatnonzero(outputs == np.uint16(target_normalized_bits))
    if matches.size == 0:
        raise BenchmarkValidationError(
            f"single-round target {target_normalized_bits} has no BF16-preimage witness"
        )
    selected = int(matches[matches.size // 2])
    row = center.copy()
    row[hidden_index] = candidates[selected]
    rounded = _float32_to_bf16_bits(row)
    if not np.array_equal(rounded, bits):
        raise BenchmarkValidationError(
            "single-round witness changed the BF16 carried row"
        )
    scalar = np.asarray(candidates[selected], dtype=np.float32)
    selected_inverse = np.asarray(inverse[selected], dtype=np.float32)
    return {
        "bf16_preimage_count": int(candidates.size),
        "fp32_row_sha256": _sha256_array(row),
        "inverse_float32_bits": int(selected_inverse.view(np.uint32)),
        "inverse_float32_value": float(selected_inverse),
        "normalized_bits": int(outputs[selected]),
        "rounded_carried_sha256": _sha256_array(rounded),
        "scalar_float32_bits": int(scalar.view(np.uint32)),
        "scalar_float32_value": float(scalar),
        "target_match_count": int(matches.size),
    }


def _validate_hlo_float_type_correction(hlo: str) -> dict[str, Any]:
    marker_count = hlo.count(_FLOAT_TYPE_CORRECTION_MARKER)
    bf16_marker_count = hlo.count(_BF16_ORIGINAL_TYPE_MARKER)
    scoped_lines = [
        line
        for line in hlo.splitlines()
        if _RMS_SCOPE in line
        and _FLOAT_TYPE_CORRECTION_MARKER in line
        and _BF16_ORIGINAL_TYPE_MARKER in line
    ]
    if marker_count <= 0 or bf16_marker_count <= 0 or not scoped_lines:
        raise BenchmarkValidationError(
            "sealed feature2 HLO lacks the scoped BF16 float-type correction marker"
        )
    return {
        "bf16_original_type_marker_count": bf16_marker_count,
        "float_type_correction_marker_count": marker_count,
        "scoped_rms_marker_count": len(scoped_lines),
        "scope": _RMS_SCOPE,
    }


def _validate_runtime_norm_weight_receipts(
    runner: Any, norm_weight: np.ndarray
) -> dict[str, Any]:
    """Bind the DB550 gamma bytes to both protected runtime selected reads."""

    value = np.ascontiguousarray(norm_weight)
    if value.shape != (6144,) or value.dtype != np.uint16:
        raise BenchmarkValidationError("PP16 runtime norm-weight schema drifted")
    value_sha256 = _sha256_array(value)
    if value_sha256 != RUNTIME_LAYER1_INPUT_NORM_SHA256:
        raise BenchmarkValidationError("PP16 DB550 norm-weight bytes drifted")
    if not isinstance(runner, dict):
        raise BenchmarkValidationError("PP16 runner receipt schema drifted")
    selective_load = runner.get("selective_load")
    if not isinstance(selective_load, dict):
        raise BenchmarkValidationError("PP16 runner selective-load receipt is missing")
    selected_reads = selective_load.get("selected_reads")
    if not isinstance(selected_reads, list):
        raise BenchmarkValidationError("PP16 runner selected-read ledger is missing")
    receipts = [
        receipt
        for receipt in selected_reads
        if isinstance(receipt, dict)
        and receipt.get("name") == _RUNTIME_LAYER1_INPUT_NORM_NAME
    ]
    expected = [
        {
            "byte_count": 12_288,
            "device_slot": device_slot,
            "dtype": "BF16",
            "filename": _RUNTIME_FEATURE_FILENAME.format(device_slot=device_slot),
            "name": _RUNTIME_LAYER1_INPUT_NORM_NAME,
            "offset": _RUNTIME_LAYER1_INPUT_NORM_OFFSET,
            "sha256": RUNTIME_LAYER1_INPUT_NORM_SHA256,
            "shape": [6144],
        }
        for device_slot in (0, 1)
    ]
    if receipts != expected:
        raise BenchmarkValidationError(
            "PP16 runtime layer-1 norm-weight selected-read receipts drifted"
        )
    return {
        "db550_norm_weight_sha256": value_sha256,
        "device_slots": [0, 1],
        "runtime_selected_read_count": len(receipts),
        "runtime_selected_read_sha256": RUNTIME_LAYER1_INPUT_NORM_SHA256,
        "tensor_name": _RUNTIME_LAYER1_INPUT_NORM_NAME,
    }


def classify_pp16_feature2_straddler(
    capture_path: Path,
    *,
    comparison_path: Path,
    summary_path: Path,
    optimized_hlo_path: Path,
    accepted_layer1_path: Path,
    db550_boundary_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    """Authenticate and classify the sealed DB518 PP16 rejection offline."""

    pinned_paths = {
        Path(capture_path): SOURCE_CAPTURE_SHA256,
        Path(comparison_path): SOURCE_COMPARISON_SHA256,
        Path(summary_path): SOURCE_SUMMARY_SHA256,
        Path(optimized_hlo_path): SOURCE_OPTIMIZED_HLO_SHA256,
        Path(accepted_layer1_path): ACCEPTED_LAYER1_SHA256,
        Path(db550_boundary_path): DB550_BOUNDARY_SHA256,
        Path(runner_path): SOURCE_RUNNER_SHA256,
    }
    pinned = {
        path: _read_exact_bytes(path, expected)
        for path, expected in pinned_paths.items()
    }

    comparison = json.loads(pinned[Path(comparison_path)])
    summary = json.loads(pinned[Path(summary_path)])
    runner = json.loads(pinned[Path(runner_path)])
    full = comparison.get("full_width_comparison", {})
    position113 = comparison.get("position113_comparison", {})
    expected_mismatches = {
        "carried_bfloat16_bits": 968,
        "contract_valid": 0,
        "event1_positions": 1852,
        "event1_scores": 2048,
        "event1_valid_counts": 0,
        "layer1_current_key_bfloat16_bits": 0,
        "layer1_dsa_head_weights_float32": 64,
        "layer1_dsa_query_float32": 8192,
        "layer1_normalized_hidden_bfloat16_bits": 2,
        "layer1_q_a_state_bfloat16_bits": 94,
    }
    if (
        comparison.get("status") != "NUMERICAL_REJECTED"
        or comparison.get("exact") is not False
        or full.get("mismatch_counts") != expected_mismatches
        or position113.get("boundary_exact") is not True
        or position113.get("prompt_cache_diff", {}).get("mismatch_count") != 0
        or position113.get("prompt_cache_matches_accepted") is not True
        or summary.get("status") != "NUMERICAL_REJECTED"
        or summary.get("exact") is not False
        or summary.get("main_execution_count") != 1
        or summary.get("performance_claim") is not False
    ):
        raise BenchmarkValidationError("PP16 straddler rejection contract drifted")

    with (
        np.load(BytesIO(pinned[Path(capture_path)]), allow_pickle=False) as capture,
        np.load(
            BytesIO(pinned[Path(accepted_layer1_path)]), allow_pickle=False
        ) as accepted,
        np.load(
            BytesIO(pinned[Path(db550_boundary_path)]), allow_pickle=False
        ) as db550,
    ):
        current_halves = np.ascontiguousarray(
            capture["current_carried_halves_bfloat16_bits"]
        )
        current_owners = np.ascontiguousarray(
            capture["current_normalized_hidden_owners_bfloat16_bits"]
        )
        accepted_normalized = np.ascontiguousarray(
            accepted["accepted__normalized_hidden"]
        )
        norm_weight = np.ascontiguousarray(db550["layer1_input_norm_bfloat16_bits"])
        model_axis_device_ids = tuple(
            int(item)
            for item in np.argsort(
                np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
            )
        )
        _, expected_carried_2d = derive_expected_dense_boundary_bits(
            db550["dense_virtual_partials_bfloat16_bits"],
            db550["post_attention_residual_bfloat16_bits"],
            model_axis_device_ids,
        )

    if (
        current_halves.shape != (2, 1, 3072)
        or current_halves.dtype != np.uint16
        or current_owners.shape != (2, 1, 6144)
        or current_owners.dtype != np.uint16
        or accepted_normalized.shape != (6144,)
        or accepted_normalized.dtype != np.uint16
        or norm_weight.shape != (6144,)
        or norm_weight.dtype != np.uint16
    ):
        raise BenchmarkValidationError("PP16 straddler tensor schema drifted")
    current_carried = np.ascontiguousarray(
        np.concatenate(tuple(current_halves), axis=1).reshape(6144)
    )
    accepted_carried = np.ascontiguousarray(expected_carried_2d.reshape(6144))
    current_normalized = np.ascontiguousarray(current_owners[0, 0])
    mismatch_indices = np.flatnonzero(current_normalized != accepted_normalized)
    carried_mismatches = int(np.count_nonzero(current_carried != accepted_carried))
    if (
        not np.array_equal(current_owners[0], current_owners[1])
        or mismatch_indices.tolist() != [HIDDEN_INDEX]
        or carried_mismatches != EXPECTED_CARRIED_MISMATCHES
        or _sha256_array(current_carried) != CURRENT_CARRIED_SHA256
        or _sha256_array(accepted_carried) != ACCEPTED_CARRIED_SHA256
        or _sha256_array(accepted_normalized) != ACCEPTED_NORMALIZED_SHA256
        or int(current_normalized[HIDDEN_INDEX]) != CURRENT_NORMALIZED_BITS
        or int(accepted_normalized[HIDDEN_INDEX]) != ACCEPTED_NORMALIZED_BITS
        or int(current_carried[HIDDEN_INDEX]) != CARRIED_BITS_AT_INDEX
        or int(accepted_carried[HIDDEN_INDEX]) != CARRIED_BITS_AT_INDEX
        or int(norm_weight[HIDDEN_INDEX]) != NORM_WEIGHT_BITS_AT_INDEX
    ):
        raise BenchmarkValidationError("PP16 straddler localized boundary drifted")

    runtime_norm_weight = _validate_runtime_norm_weight_receipts(runner, norm_weight)

    hlo = pinned[Path(optimized_hlo_path)].decode("utf-8")
    hlo_markers = _validate_hlo_float_type_correction(hlo)
    double_round_outputs = _literal_double_round_outputs(NORM_WEIGHT_BITS_AT_INDEX)
    if (
        CURRENT_NORMALIZED_BITS in double_round_outputs
        or ACCEPTED_NORMALIZED_BITS not in double_round_outputs
    ):
        raise BenchmarkValidationError(
            "PP16 literal double-round classification drifted"
        )

    witness_rows: dict[str, dict[str, Any]] = {}
    for row_name, row in (
        ("current_carried", current_carried),
        ("accepted_carried", accepted_carried),
    ):
        observed_witness = _single_round_witness(
            row,
            norm_weight_bits=NORM_WEIGHT_BITS_AT_INDEX,
            target_normalized_bits=CURRENT_NORMALIZED_BITS,
            hidden_index=HIDDEN_INDEX,
        )
        accepted_witness = _single_round_witness(
            row,
            norm_weight_bits=NORM_WEIGHT_BITS_AT_INDEX,
            target_normalized_bits=ACCEPTED_NORMALIZED_BITS,
            hidden_index=HIDDEN_INDEX,
        )
        if (
            observed_witness["rounded_carried_sha256"]
            != accepted_witness["rounded_carried_sha256"]
            or observed_witness["fp32_row_sha256"]
            == accepted_witness["fp32_row_sha256"]
        ):
            raise BenchmarkValidationError("PP16 single-round witness identity drifted")
        witness_rows[row_name] = {
            "accepted_output_witness": accepted_witness,
            "observed_output_witness": observed_witness,
            "same_bf16_carried_row": True,
        }

    return {
        "artifact_kind": "greenfield_pp16_feature2_layer1_straddler_classification",
        "classification": CLASSIFICATION,
        "claim_scope": (
            "offline sealed-artifact evidence insufficiency only; no numerical "
            "acceptance, Gate-D, DB, token, TPU-run, latency or performance claim"
        ),
        "double_round": {
            "accepted_output_globally_reachable": True,
            "literal_formula": "BF16(BF16(x * inverse) * BF16(weight))",
            "observed_output_globally_reachable": False,
            "protected_output_falsifies_literal_double_round": True,
        },
        "hlo": {
            "optimized_hlo_sha256": SOURCE_OPTIMIZED_HLO_SHA256,
            **hlo_markers,
        },
        "localized_boundary": {
            "accepted_normalized_bits": ACCEPTED_NORMALIZED_BITS,
            "carried_bits_both_rows": CARRIED_BITS_AT_INDEX,
            "carried_mismatch_count": carried_mismatches,
            "hidden_index": HIDDEN_INDEX,
            "norm_weight_bits": NORM_WEIGHT_BITS_AT_INDEX,
            "normalized_mismatch_count_per_owner": 1,
            "observed_normalized_bits": CURRENT_NORMALIZED_BITS,
            "owners_bitwise_equal": True,
        },
        "runtime_norm_weight": runtime_norm_weight,
        "next_evidence": (
            "acquire the accepted FP32 layer-1 RMS input boundary with a separately "
            "reviewed non-perturbing oracle path; do not rerun unchanged BF16-only "
            "comparison or accept bounded error"
        ),
        "sealed_inputs": {
            "accepted_layer1_sha256": ACCEPTED_LAYER1_SHA256,
            "capture_sha256": SOURCE_CAPTURE_SHA256,
            "comparison_sha256": SOURCE_COMPARISON_SHA256,
            "db550_boundary_sha256": DB550_BOUNDARY_SHA256,
            "runner_sha256": SOURCE_RUNNER_SHA256,
            "summary_sha256": SOURCE_SUMMARY_SHA256,
        },
        "single_round_noninjectivity": {
            "formula": "BF16(fp32_source * fp32_inverse * BF16(weight))",
            "interpretation": (
                "two distinct FP32 rows can round to the same retained BF16 carried "
                "row while producing the protected observed and accepted BF16 output "
                "bits at the mismatch index"
            ),
            "reference_model": "deterministic NumPy FP32, not TPU association proof",
            "witness_rows": witness_rows,
        },
        "status": "CLASSIFIED",
    }


def serialize_straddler_classification(report: dict[str, Any]) -> bytes:
    """Serialize one deterministic classification report."""

    if report.get("classification") != CLASSIFICATION:
        raise BenchmarkValidationError("PP16 straddler report is not admissible")
    return (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
