"""CPU-only source proof for the Gate-D forced BF16 RMS boundary."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax

from ..kernels.reference.rmsnorm import (
    fused_add_rms_norm_with_forced_bf16_boundary,
)


class ForcedRoundSourceError(RuntimeError):
    """Raised when the forced-round source proof does not hold exactly."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    identity = (
        str(array.dtype).encode("ascii")
        + b"\0"
        + ",".join(str(item) for item in array.shape).encode("ascii")
        + b"\0"
        + array.tobytes(order="C")
    )
    return sha256(identity).hexdigest()


def _bf16_from_bits(bits: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bits, dtype=np.uint16).view(ml_dtypes.bfloat16)


def _validate_forced_round_closed_jaxpr(closed: Any) -> dict[str, Any]:
    """Fail closed unless the precision edge has the exact weight lineage."""

    jaxpr = closed.jaxpr
    equations = jaxpr.eqns
    if len(jaxpr.invars) != 3 or len(jaxpr.outvars) != 1:
        raise ForcedRoundSourceError("forced-round JAXPR signature drifted")
    precision_equations = [
        equation
        for equation in equations
        if equation.primitive.name == "reduce_precision"
    ]
    if len(precision_equations) != 1:
        raise ForcedRoundSourceError("forced-round JAXPR precision edge drifted")
    precision = precision_equations[0]
    if precision.params != {"exponent_bits": 8, "mantissa_bits": 7}:
        raise ForcedRoundSourceError("forced-round JAXPR is not exact e8m7")
    precision_consumers = [
        equation for equation in equations if precision.outvars[0] in equation.invars
    ]
    if len(precision_consumers) != 1 or (
        precision_consumers[0].primitive.name != "mul"
    ):
        raise ForcedRoundSourceError(
            "forced-round JAXPR precision edge does not solely feed weight multiply"
        )
    weighted = precision_consumers[0]

    weight_input = jaxpr.invars[2]
    weight_consumers = [
        equation for equation in equations if weight_input in equation.invars
    ]
    if len(weight_consumers) != 1:
        raise ForcedRoundSourceError("forced-round JAXPR weight input lineage drifted")
    weight_convert = weight_consumers[0]
    if (
        weight_convert.primitive.name != "convert_element_type"
        or weight_convert.invars != [weight_input]
        or weight_convert.params.get("new_dtype") != np.dtype(np.float32)
        or len(weight_convert.outvars) != 1
    ):
        raise ForcedRoundSourceError("forced-round JAXPR weight widening drifted")
    if (
        len(weighted.invars) != 2
        or precision.outvars[0] not in weighted.invars
        or weight_convert.outvars[0] not in weighted.invars
        or precision.outvars[0] == weight_convert.outvars[0]
    ):
        raise ForcedRoundSourceError(
            "forced-round JAXPR multiply does not use the widened BF16 weight"
        )

    weighted_consumers = [
        equation for equation in equations if weighted.outvars[0] in equation.invars
    ]
    if len(weighted_consumers) != 1:
        raise ForcedRoundSourceError(
            "forced-round JAXPR weighted value has extra consumers"
        )
    final_convert = weighted_consumers[0]
    if (
        final_convert.primitive.name != "convert_element_type"
        or final_convert.invars != [weighted.outvars[0]]
        or final_convert.params.get("new_dtype") != np.dtype(ml_dtypes.bfloat16)
        or final_convert.outvars != jaxpr.outvars
    ):
        raise ForcedRoundSourceError(
            "forced-round JAXPR weight multiply does not solely feed final BF16 output"
        )
    text = str(closed)
    return {
        "explicit_hlo_lowering_performed": False,
        "final_bf16_convert_count": 1,
        "jaxpr_sha256": sha256(text.encode("utf-8")).hexdigest(),
        "reduce_precision_count": 1,
        "reduce_precision_exponent_bits": 8,
        "reduce_precision_feeds_weight_multiply": True,
        "reduce_precision_mantissa_bits": 7,
        "trace_kind": "jax.make_jaxpr",
        "weight_input_bf16_to_fp32_lineage_exact": True,
        "weighted_value_single_final_consumer": True,
    }


def inspect_forced_round_jaxpr(
    hidden_bits: np.ndarray,
    residual_bits: np.ndarray,
    weight_bits: np.ndarray,
) -> dict[str, Any]:
    """Trace without lowering and prove the explicit e8m7 data dependency."""

    hidden = jnp.asarray(_bf16_from_bits(hidden_bits))
    residual = jnp.asarray(_bf16_from_bits(residual_bits))
    weight = jnp.asarray(_bf16_from_bits(weight_bits))

    def output_only(
        hidden_value: jax.Array,
        residual_value: jax.Array,
        weight_value: jax.Array,
    ) -> jax.Array:
        output, _ = fused_add_rms_norm_with_forced_bf16_boundary(
            hidden_value,
            residual_value,
            weight_value,
            epsilon=1e-5,
        )
        return output

    return _validate_forced_round_closed_jaxpr(
        jax.make_jaxpr(output_only)(hidden, residual, weight)
    )


def analyze_bf16_precision_boundaries() -> dict[str, Any]:
    """Compare e8m7 reduction with explicit BF16 conversion at exact edges."""

    input_bits = np.asarray(
        [
            0x00000000,
            0x80000000,
            0x00008000,
            0x00008001,
            0x00010000,
            0x007F0000,
            0x007F8000,
            0x00800000,
            0x3F808000,
            0x3F818000,
            0xBF808000,
            0x7F7F0000,
            0x7F7F8000,
            0xFF7F8000,
        ],
        dtype=np.uint32,
    )
    expected_bf16_bits = np.asarray(
        [
            0x0000,
            0x8000,
            0x0000,
            0x0001,
            0x0001,
            0x007F,
            0x0080,
            0x0080,
            0x3F80,
            0x3F82,
            0xBF80,
            0x7F7F,
            0x7F80,
            0xFF80,
        ],
        dtype=np.uint16,
    )
    values = jnp.asarray(input_bits.view(np.float32))
    explicit_bf16 = values.astype(jnp.bfloat16)
    explicit_bf16_bits = np.ascontiguousarray(np.asarray(explicit_bf16)).view(np.uint16)
    reduced_fp32 = lax.reduce_precision(values, exponent_bits=8, mantissa_bits=7)
    reduced_fp32_bits = np.ascontiguousarray(np.asarray(reduced_fp32)).view(np.uint32)
    weight_bits = np.resize(
        np.asarray([0x3F80, 0xBF80, 0x4000, 0x3F00], dtype=np.uint16),
        input_bits.size,
    )
    weights = jnp.asarray(_bf16_from_bits(weight_bits))
    explicit_weighted = (
        explicit_bf16.astype(jnp.float32) * weights.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    reduced_weighted = (reduced_fp32 * weights.astype(jnp.float32)).astype(jnp.bfloat16)
    explicit_weighted_bits = np.ascontiguousarray(np.asarray(explicit_weighted)).view(
        np.uint16
    )
    reduced_weighted_bits = np.ascontiguousarray(np.asarray(reduced_weighted)).view(
        np.uint16
    )
    if (
        explicit_bf16_bits.tobytes() != expected_bf16_bits.tobytes()
        or reduced_fp32_bits.tobytes()
        != np.left_shift(expected_bf16_bits.astype(np.uint32), 16).tobytes()
        or explicit_weighted_bits.tobytes() != reduced_weighted_bits.tobytes()
    ):
        raise ForcedRoundSourceError("forced-round BF16 boundary semantics drifted")
    return {
        "case_count": int(input_bits.size),
        "explicit_cast_widen_matches_e8m7": True,
        "input_f32_bits_sha256": _array_sha256(input_bits),
        "includes_max_finite_and_overflow_to_infinity": True,
        "includes_signed_zero_subnormal_normal_and_ties": True,
        "weighted_bf16_outputs_exact": True,
        "weighted_output_bits_sha256": _array_sha256(reduced_weighted_bits),
    }


def analyze_forced_round_source(
    *,
    hidden_update_bits: np.ndarray,
    residual_bits: np.ndarray,
    weight_bits: np.ndarray,
    accepted_output_owners: np.ndarray,
    observed_output_owners: np.ndarray,
) -> dict[str, Any]:
    """Prove source exactness against sealed accepted and protected TPU rows."""

    hidden_bits = np.ascontiguousarray(hidden_update_bits, dtype=np.uint16)
    residual_value_bits = np.ascontiguousarray(residual_bits, dtype=np.uint16)
    weight_value_bits = np.ascontiguousarray(weight_bits, dtype=np.uint16)
    accepted_owners = np.ascontiguousarray(accepted_output_owners, dtype=np.uint16)
    observed_owners = np.ascontiguousarray(observed_output_owners, dtype=np.uint16)
    if (
        hidden_bits.shape != (6144,)
        or residual_value_bits.shape != hidden_bits.shape
        or weight_value_bits.shape != hidden_bits.shape
        or accepted_owners.shape != (2, 6144)
        or observed_owners.shape != (2, 6144)
    ):
        raise ForcedRoundSourceError("forced-round source authority shape drifted")
    if (
        accepted_owners[0].tobytes() != accepted_owners[1].tobytes()
        or observed_owners[0].tobytes() != observed_owners[1].tobytes()
    ):
        raise ForcedRoundSourceError("forced-round owner agreement drifted")

    hidden = jnp.asarray(_bf16_from_bits(hidden_bits))
    residual = jnp.asarray(_bf16_from_bits(residual_value_bits))
    weight = jnp.asarray(_bf16_from_bits(weight_value_bits))
    output, carried = fused_add_rms_norm_with_forced_bf16_boundary(
        hidden,
        residual,
        weight,
        epsilon=1e-5,
    )
    output_bits = np.ascontiguousarray(np.asarray(output)).view(np.uint16)
    carried_bits = np.ascontiguousarray(np.asarray(carried)).view(np.uint16)
    expected_carried_bits = np.ascontiguousarray(
        _bf16_from_bits(hidden_bits).astype(np.float32)
        + _bf16_from_bits(residual_value_bits).astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    ).view(np.uint16)
    accepted = accepted_owners[0]
    observed = observed_owners[0]
    accepted_vs_observed = np.flatnonzero(accepted != observed).astype(np.int32)
    candidate_vs_observed = np.flatnonzero(output_bits != observed).astype(np.int32)
    if (
        output_bits.tobytes() != accepted.tobytes()
        or carried_bits.tobytes() != expected_carried_bits.tobytes()
        or accepted_vs_observed.tolist() != candidate_vs_observed.tolist()
        or accepted_vs_observed.size != 1622
    ):
        raise ForcedRoundSourceError("forced-round CPU exactness proof drifted")

    return {
        "artifact_kind": "gate_d_forced_normalized_bf16_source_design",
        "authorization": {
            "full_8k": False,
            "persistence_only": True,
            "tpu_compile_or_hlo_acquisition": False,
            "tpu_execution": False,
        },
        "candidate_id": "forced_normalized_bf16_boundary",
        "classification": (
            "DEFAULT_OFF;SOURCE_EXACT;CPU_ACCEPTED_BITS_EXACT;"
            "EXPLICIT_E8M7_DATA_DEPENDENCY;PHYSICAL_CAUSE_UNPROVEN;GATE_D_OPEN"
        ),
        "cpu_exactness": {
            "accepted_output_sha256": _array_sha256(accepted),
            "candidate_matches_accepted": True,
            "candidate_output_sha256": _array_sha256(output_bits),
            "candidate_vs_observed_mismatch_count": int(candidate_vs_observed.size),
            "carried_residual_matches_strict_source": True,
            "mismatch_indices_int32_sha256": _array_sha256(candidate_vs_observed),
        },
        "bf16_boundary_semantics": analyze_bf16_precision_boundaries(),
        "default_off": True,
        "jaxpr": inspect_forced_round_jaxpr(
            hidden_bits,
            residual_value_bits,
            weight_value_bits,
        ),
        "next_authorized_action": (
            "Persist this source design after review. Compile/HLO acquisition and "
            "any bounded PP16 numerical start each require a separate fresh review, "
            "tag, and exact authority."
        ),
        "performance_claim": False,
        "physical_cause_claim": False,
        "status": "SOURCE_DESIGN_CPU_EXACT_PERSISTENCE_ONLY",
        "tpu_rerun_performed": False,
    }
