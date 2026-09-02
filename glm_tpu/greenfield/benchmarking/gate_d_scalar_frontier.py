"""Offline proof for the Gate-D layer-1 RMS conversion frontier."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import ml_dtypes
import numpy as np

EXPECTED_INPUT_SHA256 = (
    "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
)
EXPECTED_STATE_SHA256 = (
    "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236"
)
EXPECTED_OUTPUT_SHA256 = (
    "bd017d4c7e3f42b3f60fb4b918811ea028447656227c361c84fffae2544dc1a3"
)
_POSITIVE_FINITE_F32_END = 0x7F7FFFFF + 1


class ScalarFrontierError(RuntimeError):
    """Raised when the exact scalar-frontier authority does not hold."""


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    identity = (
        str(array.dtype).encode("ascii")
        + b"\0"
        + ",".join(str(item) for item in array.shape).encode("ascii")
        + b"\0"
        + array.tobytes(order="C")
    )
    return _sha256(identity)


def _widen_bf16_bits(bits: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(bits, dtype=np.uint16)
    return np.left_shift(value.astype(np.uint32), np.uint32(16)).view(np.float32)


def _bf16_bits(value: np.ndarray | np.float32) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=ml_dtypes.bfloat16).view(np.uint16)


def _strict_double_round(
    rms_input: np.ndarray, weight_fp32: np.ndarray, scale: np.float32
) -> np.ndarray:
    normalized_bits = _bf16_bits(np.multiply(rms_input, scale, dtype=np.float32))
    normalized_fp32 = _widen_bf16_bits(normalized_bits)
    weighted_fp32 = np.multiply(normalized_fp32, weight_fp32, dtype=np.float32)
    return _bf16_bits(weighted_fp32)


def _single_round(
    rms_input: np.ndarray, weight_fp32: np.ndarray, scale: np.float32
) -> np.ndarray:
    normalized_fp32 = np.multiply(rms_input, scale, dtype=np.float32)
    weighted_fp32 = np.multiply(normalized_fp32, weight_fp32, dtype=np.float32)
    return _bf16_bits(weighted_fp32)


def _strict_output_magnitude_at(
    rms_value: np.float32,
    weight_value: np.float32,
    scale_bits: int,
) -> int:
    scale = np.asarray(scale_bits, dtype=np.uint32).view(np.float32)
    normalized_bits = int(_bf16_bits(np.float32(rms_value * scale)).reshape(-1)[0])
    normalized = np.asarray(normalized_bits << 16, dtype=np.uint32).view(np.float32)
    output_bits = int(_bf16_bits(np.float32(normalized * weight_value)).reshape(-1)[0])
    return output_bits & 0x7FFF


def _first_scale_with_magnitude(
    rms_value: np.float32,
    weight_value: np.float32,
    target: int,
    *,
    strict_greater: bool,
    lower: int = 1,
) -> int:
    low = lower
    high = _POSITIVE_FINITE_F32_END
    while low < high:
        middle = (low + high) // 2
        observed = _strict_output_magnitude_at(rms_value, weight_value, middle)
        satisfies = observed > target if strict_greater else observed >= target
        if satisfies:
            high = middle
        else:
            low = middle + 1
    return low


def strict_scalar_preimage(
    rms_input: np.ndarray, weight_fp32: np.ndarray, target_bits: np.ndarray
) -> dict[str, Any]:
    """Intersect exact positive-f32 scale preimages under strict double rounding."""

    rms = np.ascontiguousarray(rms_input, dtype=np.float32)
    weight = np.ascontiguousarray(weight_fp32, dtype=np.float32)
    target = np.ascontiguousarray(target_bits, dtype=np.uint16)
    if rms.shape != (6144,) or weight.shape != rms.shape or target.shape != rms.shape:
        raise ScalarFrontierError("scalar-frontier rows must have shape [6144]")
    if (
        not np.all(np.isfinite(rms))
        or not np.all(np.isfinite(weight))
        or np.any(np.bitwise_and(target, np.uint16(0x7F80)) == np.uint16(0x7F80))
    ):
        raise ScalarFrontierError(
            "scalar-frontier preimage requires finite factors and BF16 targets"
        )
    lower = 1
    upper = _POSITIVE_FINITE_F32_END
    constrained = 0
    for index in range(rms.size):
        target_magnitude = int(target[index]) & 0x7FFF
        if rms[index] == 0 or weight[index] == 0:
            if target_magnitude != 0:
                return {
                    "constrained_elements": constrained,
                    "empty_at_index": index,
                    "reason": "zero_factor_nonzero_target",
                    "status": "EMPTY",
                }
            continue
        expected_sign = (
            0x8000 if np.signbit(rms[index]) != np.signbit(weight[index]) else 0
        )
        if int(target[index]) & 0x8000 != expected_sign:
            return {
                "constrained_elements": constrained,
                "empty_at_index": index,
                "reason": "sign_mismatch",
                "status": "EMPTY",
            }
        first = _first_scale_with_magnitude(
            rms[index], weight[index], target_magnitude, strict_greater=False
        )
        after = _first_scale_with_magnitude(
            rms[index],
            weight[index],
            target_magnitude,
            strict_greater=True,
            lower=first,
        )
        if first == after:
            return {
                "constrained_elements": constrained,
                "empty_at_index": index,
                "reason": "target_skipped_by_double_round",
                "status": "EMPTY",
                "target_bf16_bits": f"0x{int(target[index]):04x}",
            }
        lower = max(lower, first)
        upper = min(upper, after)
        constrained += 1
        if lower >= upper:
            return {
                "constrained_elements": constrained,
                "empty_at_index": index,
                "lower_bits": f"0x{lower:08x}",
                "reason": "disjoint_element_preimages",
                "status": "EMPTY",
                "upper_exclusive_bits": f"0x{upper:08x}",
            }
    return {
        "constrained_elements": constrained,
        "lower": float(np.asarray(lower, dtype=np.uint32).view(np.float32)),
        "lower_bits": f"0x{lower:08x}",
        "status": "NONEMPTY",
        "upper_inclusive": float(
            np.asarray(upper - 1, dtype=np.uint32).view(np.float32)
        ),
        "upper_inclusive_bits": f"0x{upper - 1:08x}",
        "width_f32_values": upper - lower,
    }


def analyze_scalar_frontier(
    *,
    hidden_update_bits: np.ndarray,
    residual_bits: np.ndarray,
    weight_bits: np.ndarray,
    accepted_rms_input: np.ndarray,
    observed_rms_input_owners: np.ndarray,
    accepted_output_owners: np.ndarray,
    observed_output_owners: np.ndarray,
) -> dict[str, Any]:
    """Prove the first differing physical operation from authenticated arrays."""

    hidden = np.ascontiguousarray(hidden_update_bits, dtype=np.uint16)
    residual = np.ascontiguousarray(residual_bits, dtype=np.uint16)
    weight = np.ascontiguousarray(weight_bits, dtype=np.uint16)
    if (
        hidden.shape != (6144,)
        or residual.shape != hidden.shape
        or weight.shape != hidden.shape
    ):
        raise ScalarFrontierError("RMS operand/weight shape drifted")
    derived_rms_input = np.add(
        _widen_bf16_bits(hidden), _widen_bf16_bits(residual), dtype=np.float32
    )
    accepted_input = np.ascontiguousarray(accepted_rms_input, dtype=np.float32)
    observed_inputs = np.ascontiguousarray(observed_rms_input_owners, dtype=np.float32)
    accepted_outputs = np.ascontiguousarray(accepted_output_owners, dtype=np.uint16)
    observed_outputs = np.ascontiguousarray(observed_output_owners, dtype=np.uint16)
    if (
        accepted_input.shape != (6144,)
        or observed_inputs.shape != (2, 6144)
        or accepted_outputs.shape != (2, 6144)
        or observed_outputs.shape != (2, 6144)
    ):
        raise ScalarFrontierError("scalar-frontier accepted/observed shape drifted")
    if (
        derived_rms_input.tobytes() != accepted_input.tobytes()
        or any(row.tobytes() != derived_rms_input.tobytes() for row in observed_inputs)
        or accepted_outputs[0].tobytes() != accepted_outputs[1].tobytes()
        or observed_outputs[0].tobytes() != observed_outputs[1].tobytes()
    ):
        raise ScalarFrontierError("scalar-frontier owner/input authority drifted")

    weight_fp32 = _widen_bf16_bits(weight)
    square = np.multiply(derived_rms_input, derived_rms_input, dtype=np.float32)
    variance = np.mean(square, dtype=np.float32)
    denominator = np.add(variance, np.float32(1e-5), dtype=np.float32)
    scale = np.divide(
        np.float32(1), np.sqrt(denominator, dtype=np.float32), dtype=np.float32
    )
    strict = _strict_double_round(derived_rms_input, weight_fp32, scale)
    single = _single_round(derived_rms_input, weight_fp32, scale)
    accepted = accepted_outputs[0]
    observed = observed_outputs[0]
    accepted_vs_observed = np.flatnonzero(accepted != observed)
    strict_vs_single = np.flatnonzero(strict != single)
    if (
        strict.tobytes() != accepted.tobytes()
        or single.tobytes() != observed.tobytes()
        or accepted_vs_observed.tolist() != strict_vs_single.tolist()
        or accepted_vs_observed.size != 1622
    ):
        raise ScalarFrontierError(
            "conversion-placement identity did not reproduce evidence"
        )
    accepted_preimage = strict_scalar_preimage(derived_rms_input, weight_fp32, accepted)
    observed_preimage = strict_scalar_preimage(derived_rms_input, weight_fp32, observed)
    scale_bits = int(np.asarray(scale).view(np.uint32))
    if (
        accepted_preimage.get("status") != "NONEMPTY"
        or not (
            int(accepted_preimage["lower_bits"], 16)
            <= scale_bits
            <= int(accepted_preimage["upper_inclusive_bits"], 16)
        )
        or observed_preimage
        != {
            "constrained_elements": 0,
            "empty_at_index": 0,
            "reason": "target_skipped_by_double_round",
            "status": "EMPTY",
            "target_bf16_bits": "0x3c18",
        }
    ):
        raise ScalarFrontierError("strict scalar preimage classification drifted")
    mismatch_sha = _array_sha256(accepted_vs_observed.astype(np.int32))
    return {
        "artifact_kind": "gate_d_scalar_frontier_conversion_placement",
        "classification": (
            "RMS_INPUT_EXACT;ACCEPTED_STRICT_DOUBLE_ROUND_EXACT;"
            "TPU_RETAINED_FP32_EXPLANATION_EXACT;UNIFORM_STRICT_SCALAR_IMPOSSIBLE;"
            "PHYSICAL_CAUSE_UNPROVEN;GATE_D_OPEN"
        ),
        "claim_scope": (
            "Offline exact adjudication of the already-completed protected PP16 "
            "layer-1 RMS output; no JAX, TPU rerun, performance, DB, token, or Gate-D "
            "closure claim."
        ),
        "first_divergence": {
            "accepted_vs_observed_mismatch_count": int(accepted_vs_observed.size),
            "mismatch_indices_int32_sha256": mismatch_sha,
            "rms_input_fp32_exact": True,
            "strict_double_round_matches_accepted": True,
            "retained_fp32_round_matches_observed_tpu": True,
        },
        "next_mechanism": (
            "Causally test the exact conversion-placement hypothesis by forcing the "
            "existing normalized-f32 to BF16 rounding edge before the RMS weight "
            "multiply (local reduce_precision first; excess-precision-disabled "
            "comparator); inspect optimized HLO, then run only the bounded PP16 layer."
        ),
        "observed_tpu_strict_scalar_preimage": observed_preimage,
        "performance_claim": False,
        "reference_scale": {
            "accepted_strict_preimage": accepted_preimage,
            "bits": f"0x{scale_bits:08x}",
            "float32": float(scale),
            "variance_bits": f"0x{int(np.asarray(variance).view(np.uint32)):08x}",
        },
        "source_arrays": {
            "accepted_output_sha256": _array_sha256(accepted),
            "derived_rms_input_sha256": _array_sha256(derived_rms_input),
            "observed_output_sha256": _array_sha256(observed),
            "rms_weight_bits_sha256": _array_sha256(weight),
        },
        "status": "EXACT_EXPLANATION_PROVED_PHYSICAL_CAUSE_UNPROVEN",
        "tpu_rerun_performed": False,
    }
