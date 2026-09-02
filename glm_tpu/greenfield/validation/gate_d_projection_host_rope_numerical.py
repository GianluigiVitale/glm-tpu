"""Fail-closed validation for the bounded Gate-D host-rope projection replay (V3).

The V2 replay proved the TPU projection FP32-accurate (sub-ulp) and the key
LayerNorm bit-exact, and localized the accepted-key divergence to on-device
``cos``/``sin`` at large rotary angles.  V3 feeds one host-evaluated FP32
``cos|sin`` row and must show that the whole key path is FP32-faithful:
projection within tolerance of an F64-accumulated reference and current key
within tolerance of an F64 reference built from the TPU's own projection and
the same row.  Byte identity with the accepted legacy key is not a criterion
(even a pure CPU replica differs from it by one ulp on ~45 dimensions);
DSA-selection exactness is judged later on the full decoder.

This module imports no JAX at module import time; the protected runner supplies
its sealed NumPy module.  HLO audits use the repository's textual HLO parser.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from ..sharding.hlo_contract import parse_hlo_module
from .gate_d_projection_contraction_numerical import (
    EXPECTED_NORMALIZED_OWNER_SHA256,
    EXPECTED_WK_WEIGHT_FP32_SHA256,
)

DSA_ROPE_POSITION = 8155
DSA_ROTARY_DIM = 64
DSA_ROTARY_THETA = 8_000_000.0
EXPECTED_DSA_ROPE_ROW_SHA256 = (
    "748aa6122b8d83cfcf65928d33d1ac10392b7cc968617f324a75a62168d3a9c0"
)
HLO_MODULE_NAME = "jit__projection_host_rope_local"
PROJECTION_TOLERANCE = 1e-6
KEY_TOLERANCE = 1e-6
IMPLIED_COS_SIN_TOLERANCE = 1e-6
ENTRY_PARAMETER_SHAPES = (
    ("bf16", (1, 1, 6144)),
    ("f32", (1, 128, 6144)),
    ("bf16", (1, 128)),
    ("bf16", (1, 128)),
    ("f32", (1, 64)),
)
ROOT_SHAPES = (("bf16", (1, 1, 6144)), ("f32", (1, 1, 128)), ("f32", (1, 1, 128)))
_FORBIDDEN_OPCODES = frozenset(
    {
        "all-gather",
        "all-reduce",
        "all-to-all",
        "collective-broadcast",
        "collective-permute",
        "infeed",
        "outfeed",
        "recv",
        "recv-done",
        "reduce-scatter",
        "send",
        "send-done",
        "custom-call",
    }
)
_TRANSCENDENTAL_OPCODES = frozenset(
    {"cosine", "sine", "tan", "power", "exponential", "log", "atan2"}
)
_FORBIDDEN_TEXT = ("host_callback", "xla_python_cpu_callback", "outside_compilation")


class HostRopeValidationError(RuntimeError):
    """Raised when the V3 replay evidence drifts from its contract."""


def _array_sha256(value: Any, np: Any) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def materialize_dsa_rope_row(np: Any) -> Any:
    """Host FP32 ``cos|sin`` row for both owners, shaped ``(2, 64)``.

    Uses the same arithmetic as ``rotary_table.dsa_rotary_row_host`` without
    importing JAX: negative-power FP32 inverse frequencies, FP32 angle
    products, FP32 libm cos/sin.
    """

    dimensions = np.arange(0, DSA_ROTARY_DIM, 2, dtype=np.float32)
    frequencies = np.power(
        np.float32(DSA_ROTARY_THETA),
        -dimensions / np.float32(DSA_ROTARY_DIM),
        dtype=np.float32,
    )
    angles = np.multiply(np.float32(DSA_ROPE_POSITION), frequencies, dtype=np.float32)
    row = np.concatenate(
        (np.cos(angles, dtype=np.float32), np.sin(angles, dtype=np.float32))
    ).astype(np.float32)
    owners = np.ascontiguousarray(np.stack((row, row)), dtype=np.float32)
    if (
        owners.shape != (2, DSA_ROTARY_DIM)
        or _array_sha256(owners[0], np) != EXPECTED_DSA_ROPE_ROW_SHA256
        or not bool(np.all(np.isfinite(owners)))
    ):
        raise HostRopeValidationError("Gate-D host rotary row drifted")
    return owners


def dsa_rope_row_identity(row_owners: Any, np: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(row_owners)
    identity = {
        "array_sha256": _array_sha256(array, np),
        "position": DSA_ROPE_POSITION,
        "row_sha256": _array_sha256(array[0], np),
        "rotary_dim": DSA_ROTARY_DIM,
        "shape": list(array.shape),
        "storage_dtype": array.dtype.str,
        "theta": DSA_ROTARY_THETA,
    }
    if (
        array.dtype != np.dtype(np.float32)
        or array.shape != (2, DSA_ROTARY_DIM)
        or identity["row_sha256"] != EXPECTED_DSA_ROPE_ROW_SHA256
        or array[0].tobytes() != array[1].tobytes()
    ):
        raise HostRopeValidationError("Gate-D host rotary row identity drifted")
    return identity


def _f32_bits(values: Any) -> list[float]:
    """Exact float values of an FP32 buffer without NumPy (publisher path)."""

    raw = bytes(values)
    if len(raw) % 4:
        raise HostRopeValidationError("FP32 buffer length is not a multiple of four")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def reference_key_f64(
    projected: list[float],
    key_norm_weight: list[float],
    key_norm_bias: list[float],
    rope_row: list[float],
    *,
    epsilon: float = 1e-6,
) -> list[float]:
    """F64 reference key from an FP32 projection and FP32 host row (pure Python).

    Reproduces the graph's semantics (biased LayerNorm with divide-by-sqrt, GLM
    interleaved rotary on the first 64 dimensions) in F64 so that the comparison
    is independent of reduction order and of FP32 rounding details.
    """

    width = len(projected)
    if (
        width != 128
        or len(key_norm_weight) != width
        or len(key_norm_bias) != width
        or len(rope_row) != DSA_ROTARY_DIM
    ):
        raise HostRopeValidationError("reference key operand widths drifted")
    mean = math.fsum(projected) / width
    variance = math.fsum((value - mean) ** 2 for value in projected) / width
    scale = 1.0 / math.sqrt(variance + epsilon)
    normalized = [
        (value - mean) * scale * weight + bias
        for value, weight, bias in zip(
            projected, key_norm_weight, key_norm_bias, strict=True
        )
    ]
    half = DSA_ROTARY_DIM // 2
    cos, sin = rope_row[:half], rope_row[half:]
    rotated = list(normalized)
    for pair in range(half):
        a, b = normalized[2 * pair], normalized[2 * pair + 1]
        rotated[2 * pair] = a * cos[pair] - b * sin[pair]
        rotated[2 * pair + 1] = a * sin[pair] + b * cos[pair]
    return rotated


def implied_cos_sin_error(
    pre_rotation: list[float], post_rotation: list[float], rope_row: list[float]
) -> dict[str, float]:
    """Recover the cos/sin actually applied per pair and compare to the row."""

    half = DSA_ROTARY_DIM // 2
    worst_cos = worst_sin = 0.0
    for pair in range(half):
        a, b = pre_rotation[2 * pair], pre_rotation[2 * pair + 1]
        ap, bp = post_rotation[2 * pair], post_rotation[2 * pair + 1]
        radius = a * a + b * b
        if radius <= 0.0:
            continue
        implied_cos = (a * ap + b * bp) / radius
        implied_sin = (a * bp - b * ap) / radius
        worst_cos = max(worst_cos, abs(implied_cos - rope_row[pair]))
        worst_sin = max(worst_sin, abs(implied_sin - rope_row[half + pair]))
    return {"max_abs_cos_error": worst_cos, "max_abs_sin_error": worst_sin}


def classify_host_rope_outputs(
    outputs: Mapping[str, Any],
    np: Any,
    *,
    wk_weight: Any,
    key_norm_weight_bits: Any,
    key_norm_bias_bits: Any,
    rope_row: Any,
    ml_dtypes: Any,
) -> dict[str, Any]:
    """Classify the three bounded outputs; acceptance never closes Gate D."""

    # The normalized row is BF16 in process and archived as its uint16 bits;
    # both carry identical bytes, so either storage form is accepted.
    expected_shapes = {
        "normalized_hidden_owners": ((2, 1, 6144), ("bfloat16", "uint16")),
        "projected_key_owners": ((2, 1, 128), ("float32",)),
        "current_key_owners": ((2, 1, 128), ("float32",)),
    }
    if set(outputs) != set(expected_shapes):
        raise HostRopeValidationError("Gate-D host-rope output catalogue drifted")
    records: dict[str, Any] = {}
    structural = True
    for name, (shape, dtypes) in expected_shapes.items():
        value = np.ascontiguousarray(outputs[name])
        shape_exact = value.shape == shape
        dtype_exact = str(value.dtype) in dtypes
        finite = bool(
            np.all(np.isfinite(value.view(ml_dtypes.bfloat16)))
            if str(value.dtype) == "uint16"
            else np.all(np.isfinite(value))
        )
        owners_equal = shape_exact and value[0].tobytes(order="C") == value[1].tobytes(
            order="C"
        )
        records[name] = {
            "dtype_exact": dtype_exact,
            "finite": finite,
            "owner_sha256": [
                _array_sha256(value[index, 0], np) if shape_exact else None
                for index in range(2)
            ],
            "owners_equal": owners_equal,
            "shape_exact": shape_exact,
        }
        structural = (
            structural and shape_exact and dtype_exact and finite and owners_equal
        )
    if not structural:
        return {
            "accepted_tpu_host_rope_faithful": False,
            "outputs": records,
            "structural": False,
        }
    normalized_exact = (
        records["normalized_hidden_owners"]["owner_sha256"][0]
        == EXPECTED_NORMALIZED_OWNER_SHA256
    )
    records["normalized_hidden_owners"]["witness_exact"] = normalized_exact
    wk = np.ascontiguousarray(wk_weight, dtype=np.float32)
    if (
        wk.shape != (2, 128, 6144)
        or _array_sha256(wk, np) != EXPECTED_WK_WEIGHT_FP32_SHA256
    ):
        raise HostRopeValidationError("Gate-D host-rope FP32 key weight drifted")
    normalized_bits = np.ascontiguousarray(outputs["normalized_hidden_owners"])[
        0, 0
    ].view(np.uint16)
    hidden = normalized_bits.view(ml_dtypes.bfloat16).astype(np.float64)
    reference_projection = wk[0].astype(np.float64) @ hidden
    tpu_projection = np.ascontiguousarray(outputs["projected_key_owners"])[0, 0].astype(
        np.float64
    )
    projection_error = float(np.abs(tpu_projection - reference_projection).max())
    projection_ok = projection_error <= PROJECTION_TOLERANCE
    records["projected_key_owners"].update(
        {
            "max_abs_error_vs_f64_reference": projection_error,
            "tolerance": PROJECTION_TOLERANCE,
            "within_tolerance": projection_ok,
        }
    )
    weight = [
        float(v)
        for v in np.ascontiguousarray(key_norm_weight_bits)[0]
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    ]
    bias = [
        float(v)
        for v in np.ascontiguousarray(key_norm_bias_bits)[0]
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    ]
    row = [float(v) for v in np.ascontiguousarray(rope_row, dtype=np.float32)[0]]
    projected = [
        float(v) for v in np.ascontiguousarray(outputs["projected_key_owners"])[0, 0]
    ]
    reference_key = reference_key_f64(projected, weight, bias, row)
    tpu_key = [
        float(v) for v in np.ascontiguousarray(outputs["current_key_owners"])[0, 0]
    ]
    key_errors = [abs(a - b) for a, b in zip(tpu_key, reference_key, strict=True)]
    key_error = max(key_errors)
    rotary_error = max(key_errors[:DSA_ROTARY_DIM])
    nonrotary_error = max(key_errors[DSA_ROTARY_DIM:])
    key_ok = key_error <= KEY_TOLERANCE
    pre_rotation = reference_key_f64(projected, weight, bias, [1.0] * 32 + [0.0] * 32)
    implied = implied_cos_sin_error(pre_rotation, tpu_key, row)
    implied_ok = (
        implied["max_abs_cos_error"] <= IMPLIED_COS_SIN_TOLERANCE
        and implied["max_abs_sin_error"] <= IMPLIED_COS_SIN_TOLERANCE
    )
    records["current_key_owners"].update(
        {
            "implied_rotary": implied,
            "implied_rotary_within_tolerance": implied_ok,
            "max_abs_error_vs_f64_reference": key_error,
            "nonrotary_max_abs_error": nonrotary_error,
            "rotary_max_abs_error": rotary_error,
            "tolerance": KEY_TOLERANCE,
            "within_tolerance": key_ok,
        }
    )
    accepted = bool(normalized_exact and projection_ok and key_ok and implied_ok)
    return {
        "accepted_tpu_host_rope_faithful": accepted,
        "outputs": records,
        "structural": True,
    }


def _shape_tuple(shape: Any) -> tuple[str, tuple[int, ...]]:
    return (shape.dtype, tuple(shape.dimensions))


def audit_host_rope_optimized_hlo(optimized_hlo: str) -> dict[str, Any]:
    """Structural contract: no communication, no device transcendentals, exact I/O."""

    try:
        module = parse_hlo_module(optimized_hlo)
    except (TypeError, ValueError) as error:
        raise HostRopeValidationError("optimized HLO is not parseable") from error
    if module.name != HLO_MODULE_NAME or module.num_partitions != 2:
        raise HostRopeValidationError("optimized HLO module identity drifted")
    if tuple(module.collectives) or any(
        item.opcode in _FORBIDDEN_OPCODES for item in module.instructions
    ):
        raise HostRopeValidationError(
            "host-rope HLO contains communication or host calls"
        )
    lowered = optimized_hlo.lower()
    if any(token in lowered for token in _FORBIDDEN_TEXT):
        raise HostRopeValidationError("host-rope HLO contains a host effect")
    transcendental = [
        item.name
        for item in module.instructions
        if item.opcode in _TRANSCENDENTAL_OPCODES
    ]
    if transcendental:
        raise HostRopeValidationError(
            f"host-rope HLO still evaluates transcendentals on device: {transcendental}"
        )
    entry = [
        item for item in module.instructions if item.computation.startswith("ENTRY ")
    ]
    parameters = tuple(
        _shape_tuple(item.result_shapes[0])
        for item in sorted(
            (item for item in entry if item.opcode == "parameter"),
            key=lambda item: int(item.operand_names[0]) if item.operand_names else -1,
        )
    )
    if parameters != ENTRY_PARAMETER_SHAPES:
        raise HostRopeValidationError(
            f"host-rope HLO entry parameters drifted: {parameters}"
        )
    roots = [item for item in entry if item.raw_line.strip().startswith("ROOT")]
    if (
        len(roots) != 1
        or tuple(_shape_tuple(s) for s in roots[0].result_shapes) != ROOT_SHAPES
    ):
        raise HostRopeValidationError("host-rope HLO root drifted")
    contraction_operands = [
        item
        for item in module.instructions
        if any(
            shape.dtype == "f32" and shape.dimensions in ((1, 128, 6144), (128, 6144))
            for shape in item.operand_shapes
        )
    ]
    f32_reduces = [
        item
        for item in module.instructions
        if item.opcode == "reduce"
        and all(shape.dtype == "f32" for shape in item.result_shapes)
    ]
    if not contraction_operands or not f32_reduces:
        raise HostRopeValidationError("host-rope HLO lacks the FP32 6144 contraction")
    return {
        "collective_count": 0,
        "contraction_accumulation_dtype": "f32",
        "contraction_input_width": 6144,
        "entry_instruction_count": len(entry),
        "entry_parameter_count": len(parameters),
        "host_effect_count": 0,
        "instruction_count": len(module.instructions),
        "live_rows_per_owner": 1,
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "output_shapes": [list(shape) for _, shape in ROOT_SHAPES],
        "owner_count": 2,
        "transcendental_count": 0,
    }


def audit_host_rope_stablehlo(stablehlo: str) -> dict[str, Any]:
    required = (
        "mhlo.num_partitions = 2",
        'sdy.mesh @mesh = <["feature"=2]>',
        "tensor<2x1x6144xbf16>",
        "tensor<2x128x6144xf32>",
        "tensor<2x64xf32>",
        "stablehlo.dot_general",
    )
    lowered = stablehlo.lower()
    forbidden = (
        *_FORBIDDEN_TEXT,
        "stablehlo.all_",
        "stablehlo.collective_",
        "stablehlo.infeed",
        "stablehlo.outfeed",
        "stablehlo.cosine",
        "stablehlo.sine",
        "stablehlo.power",
        "stablehlo.custom_call",
    )
    if any(item not in stablehlo for item in required) or any(
        token in lowered for token in forbidden
    ):
        raise HostRopeValidationError("StableHLO host-rope contract drifted")
    return {
        "collective_count": 0,
        "contraction_dtype": "f32",
        "contraction_width": 6144,
        "live_rows_per_owner": 1,
        "manual_axis": {"feature": 2},
        "owner_count": 2,
        "transcendental_count": 0,
    }
