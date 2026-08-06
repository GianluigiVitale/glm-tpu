from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
)
from glm_tpu.greenfield.kernels.reference.fp8 import (
    dequantize_fp8_bits_block_weight,
)


def _bits(values: jax.Array) -> jax.Array:
    quantized = values.astype(jnp.float8_e4m3fn)
    return lax.bitcast_convert_type(quantized, jnp.uint8)


@pytest.mark.parametrize("shape", [(1, 128, 128), (3, 130, 135)])
def test_fp8_block_matmul_interpret_matches_reference(
    shape: tuple[int, int, int],
) -> None:
    rows, contraction, output = shape
    lhs = jnp.asarray(
        np.linspace(-0.75, 0.75, rows * contraction, dtype=np.float32).reshape(
            rows, contraction
        ),
        dtype=jnp.bfloat16,
    )
    weight = jnp.asarray(
        np.sin(np.arange(output * contraction, dtype=np.float32) * 0.013).reshape(
            output, contraction
        )
        * np.float32(0.5),
        dtype=jnp.float32,
    )
    weight_bits = _bits(weight)
    scale = jnp.asarray(
        np.linspace(
            0.25,
            1.0,
            ((output + 127) // 128) * ((contraction + 127) // 128),
            dtype=np.float32,
        ).reshape((output + 127) // 128, (contraction + 127) // 128)
    )
    decoded = dequantize_fp8_bits_block_weight(weight_bits, scale)
    expected = lax.dot_general(
        lhs,
        decoded,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    actual = fp8_block_matmul(lhs, weight_bits, scale, interpret=True)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_fp8_block_matmul_rejects_shape_and_dtype_drift() -> None:
    lhs = jnp.ones((1, 128), dtype=jnp.bfloat16)
    bits = jnp.zeros((128, 128), dtype=jnp.uint8)
    scale = jnp.ones((1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="lhs must be BF16"):
        fp8_block_matmul(lhs.astype(jnp.float32), bits, scale, interpret=True)
    with pytest.raises(ValueError, match="uint8 bits"):
        fp8_block_matmul(lhs, bits.astype(jnp.int8), scale, interpret=True)
    with pytest.raises(ValueError, match="scale shape"):
        fp8_block_matmul(lhs, bits, jnp.ones((1, 2), jnp.float32), interpret=True)


def test_fp8_block_matmul_config_is_v4_numerically_pinned() -> None:
    with pytest.raises(ValueError, match="one scale per Pallas weight tile"):
        Fp8BlockMatmulConfig(output_tile=256)
    with pytest.raises(ValueError, match="accumulator must be FP32"):
        Fp8BlockMatmulConfig(accumulator_dtype=jnp.bfloat16)
