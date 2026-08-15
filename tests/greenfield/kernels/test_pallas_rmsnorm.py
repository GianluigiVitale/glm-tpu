from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas.rmsnorm import fused_add_rms_norm_m1
from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm


def test_fused_add_rms_norm_m1_interpret_matches_reference() -> None:
    hidden = jnp.linspace(-2.0, 3.0, 128, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    residual = jnp.linspace(0.5, -1.5, 128, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    weight = jnp.linspace(0.25, 1.25, 128, dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    expected_output, expected_carried = fused_add_rms_norm(
        hidden, residual, weight, epsilon=1e-5
    )
    output, carried = fused_add_rms_norm_m1(
        hidden, residual, weight, epsilon=1e-5, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(output), np.asarray(expected_output))
    np.testing.assert_array_equal(
        np.asarray(carried), np.asarray(expected_carried)
    )


@pytest.mark.parametrize(
    ("hidden_shape", "residual_shape", "weight_shape"),
    [
        ((2, 128), (2, 128), (128,)),
        ((1, 127), (1, 127), (127,)),
        ((1, 128), (1, 64), (128,)),
        ((1, 128), (1, 128), (64,)),
    ],
)
def test_fused_add_rms_norm_m1_refuses_geometry_drift(
    hidden_shape: tuple[int, ...],
    residual_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        fused_add_rms_norm_m1(
            jnp.ones(hidden_shape, jnp.bfloat16),
            jnp.ones(residual_shape, jnp.bfloat16),
            jnp.ones(weight_shape, jnp.bfloat16),
            epsilon=1e-5,
            interpret=True,
        )


def test_fused_add_rms_norm_m1_refuses_dtype_and_epsilon_drift() -> None:
    hidden = jnp.ones((1, 128), jnp.bfloat16)
    residual = jnp.ones((1, 128), jnp.bfloat16)
    weight = jnp.ones((128,), jnp.bfloat16)
    with pytest.raises(ValueError):
        fused_add_rms_norm_m1(
            hidden.astype(jnp.float32),
            residual,
            weight,
            epsilon=1e-5,
            interpret=True,
        )
    with pytest.raises(ValueError):
        fused_add_rms_norm_m1(
            hidden,
            residual,
            weight,
            epsilon=0.0,
            interpret=True,
        )
