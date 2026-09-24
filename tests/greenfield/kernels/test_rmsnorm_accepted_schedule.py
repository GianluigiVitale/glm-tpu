"""CPU contract of the accepted decode-step RMS variance schedule (default off)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from glm_tpu.optimized.reference.rmsnorm import (
    ACCEPTED_SCHEDULE_ROWS,
    final_norm,
    fused_add_rms_norm,
    rms_norm,
)


@pytest.mark.parametrize("rows", (1, 3, 40))
def test_accepted_schedule_matches_default_within_bf16_rounding(rows: int) -> None:
    rng = np.random.default_rng(2795)
    hidden = (rng.standard_normal((rows, 512)) * 0.05).astype(ml_dtypes.bfloat16)
    residual = (rng.standard_normal((rows, 512)) * 0.05).astype(ml_dtypes.bfloat16)
    weight = (1.0 + rng.standard_normal(512) * 0.1).astype(ml_dtypes.bfloat16)
    default_out, default_carry = fused_add_rms_norm(
        jnp.asarray(hidden), jnp.asarray(residual), jnp.asarray(weight), epsilon=1e-5
    )
    schedule_out, schedule_carry = fused_add_rms_norm(
        jnp.asarray(hidden),
        jnp.asarray(residual),
        jnp.asarray(weight),
        epsilon=1e-5,
        accepted_schedule=True,
    )
    assert np.array_equal(np.asarray(default_carry), np.asarray(schedule_carry))
    assert default_out.shape == schedule_out.shape == (rows, 512)
    delta = np.abs(
        np.asarray(default_out).astype(np.float32) - np.asarray(schedule_out).astype(np.float32)
    )
    assert float(delta.max()) <= 2.0**-6
    plain = rms_norm(jnp.asarray(hidden), jnp.asarray(weight), epsilon=1e-5)
    plain_schedule = rms_norm(
        jnp.asarray(hidden), jnp.asarray(weight), epsilon=1e-5, accepted_schedule=True
    )
    assert float(
        np.abs(np.asarray(plain).astype(np.float32) - np.asarray(plain_schedule).astype(np.float32)).max()
    ) <= 2.0**-6
    final = final_norm(jnp.asarray(hidden), jnp.asarray(weight), epsilon=1e-5, accepted_schedule=True)
    assert np.array_equal(np.asarray(final), np.asarray(plain_schedule))


def test_accepted_schedule_lowers_to_a_32_row_barrier_carried_reduce() -> None:
    hidden = jnp.ones((1, 6144), dtype=jnp.bfloat16)
    weight = jnp.ones((6144,), dtype=jnp.bfloat16)
    text = jax.jit(
        lambda h, w: rms_norm(h, w, epsilon=1e-5, accepted_schedule=True)
    ).lower(hidden, weight).as_text()
    assert f"tensor<{ACCEPTED_SCHEDULE_ROWS}x6144xf32>" in text
    assert "stablehlo.optimization_barrier" in text
    assert "stablehlo.rsqrt" in text
    assert f"tensor<{ACCEPTED_SCHEDULE_ROWS}x1xf32>" in text
    default_text = jax.jit(lambda h, w: rms_norm(h, w, epsilon=1e-5)).lower(hidden, weight).as_text()
    assert "stablehlo.optimization_barrier" not in default_text
    assert f"tensor<{ACCEPTED_SCHEDULE_ROWS}x6144xf32>" not in default_text


def test_accepted_schedule_flag_must_be_boolean() -> None:
    hidden = jnp.ones((1, 8), dtype=jnp.bfloat16)
    weight = jnp.ones((8,), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="accepted-schedule flag must be boolean"):
        rms_norm(hidden, weight, epsilon=1e-5, accepted_schedule=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="accepted-schedule flag must be boolean"):
        fused_add_rms_norm(hidden, hidden, weight, epsilon=1e-5, accepted_schedule="yes")  # type: ignore[arg-type]
