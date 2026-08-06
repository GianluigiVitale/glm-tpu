from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_block_up_gate,
    fp8_selected_up_gate,
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


@pytest.mark.parametrize("shape", [(1, 128, 128), (3, 130, 135)])
def test_fp8_block_up_gate_interpret_matches_two_references(
    shape: tuple[int, int, int],
) -> None:
    rows, contraction, output = shape
    lhs = jnp.asarray(
        np.linspace(-0.5, 0.5, rows * contraction, dtype=np.float32).reshape(
            rows, contraction
        ),
        dtype=jnp.bfloat16,
    )
    linear = np.arange(output * contraction, dtype=np.float32).reshape(
        output, contraction
    )
    gate_bits = _bits(jnp.asarray(np.sin(linear * 0.017) * 0.5))
    up_bits = _bits(jnp.asarray(np.cos(linear * 0.011) * 0.375))
    scale_shape = ((output + 127) // 128, (contraction + 127) // 128)
    gate_scale = jnp.asarray(
        np.linspace(0.25, 0.75, np.prod(scale_shape), dtype=np.float32).reshape(
            scale_shape
        )
    )
    up_scale = jnp.asarray(
        np.linspace(0.5, 1.0, np.prod(scale_shape), dtype=np.float32).reshape(
            scale_shape
        )
    )

    def reference(bits: jax.Array, scale: jax.Array) -> jax.Array:
        decoded = dequantize_fp8_bits_block_weight(bits, scale)
        return lax.dot_general(
            lhs,
            decoded,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)

    actual_gate, actual_up = fp8_block_up_gate(
        lhs,
        gate_bits,
        gate_scale,
        up_bits,
        up_scale,
        interpret=True,
    )
    np.testing.assert_array_equal(
        np.asarray(actual_gate), np.asarray(reference(gate_bits, gate_scale))
    )
    np.testing.assert_array_equal(
        np.asarray(actual_up), np.asarray(reference(up_bits, up_scale))
    )


def test_fp8_block_matmul_config_is_v4_numerically_pinned() -> None:
    with pytest.raises(ValueError, match="one output scale block"):
        Fp8BlockMatmulConfig(output_tile=256)
    with pytest.raises(ValueError, match="integral number"):
        Fp8BlockMatmulConfig(contraction_tile=192)
    with pytest.raises(ValueError, match="accumulator must be FP32"):
        Fp8BlockMatmulConfig(accumulator_dtype=jnp.bfloat16)


def test_fp8_selected_up_gate_interpret_uses_distinct_owned_experts() -> None:
    # emit_pipeline needs physical tiling metadata even under the HLO
    # interpreter. Register v4's public JAX hardware description for this
    # forced-CPU test; the kernel still executes on the CPU interpreter.
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()
    routes, experts, contraction, output = 4, 3, 130, 135
    hidden = jnp.asarray(
        np.linspace(-0.5, 0.5, contraction, dtype=np.float32)[None, :],
        dtype=jnp.bfloat16,
    )
    linear = np.arange(
        experts * output * contraction, dtype=np.float32
    ).reshape(experts, output, contraction)
    gate_bits = _bits(jnp.asarray(np.sin(linear * 0.013) * 0.5))
    up_bits = _bits(jnp.asarray(np.cos(linear * 0.019) * 0.375))
    scale_shape = (experts, (output + 127) // 128, (contraction + 127) // 128)
    gate_scale = jnp.asarray(
        np.linspace(0.25, 0.75, np.prod(scale_shape), dtype=np.float32).reshape(
            scale_shape
        )
    )
    up_scale = jnp.asarray(
        np.linspace(0.5, 1.0, np.prod(scale_shape), dtype=np.float32).reshape(
            scale_shape
        )
    )
    route_indices = jnp.asarray([11, 500, 10, 12], dtype=jnp.int32)
    expert_start = jnp.asarray(10, dtype=jnp.int32)
    actual_gate, actual_up = fp8_selected_up_gate(
        hidden,
        route_indices,
        expert_start,
        gate_bits,
        gate_scale,
        up_bits,
        up_scale,
        interpret=True,
    )

    def reference(bits: jax.Array, scale: jax.Array) -> np.ndarray:
        values = []
        for global_expert in np.asarray(route_indices):
            local_expert = int(global_expert) - int(expert_start)
            if not 0 <= local_expert < experts:
                values.append(np.zeros((output,), dtype=np.float32))
                continue
            decoded = dequantize_fp8_bits_block_weight(
                bits[local_expert], scale[local_expert]
            )
            value = lax.dot_general(
                hidden,
                decoded,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            values.append(np.asarray(value[0]))
        return np.stack(values)

    np.testing.assert_array_equal(
        np.asarray(actual_gate), reference(gate_bits, gate_scale)
    )
    np.testing.assert_array_equal(
        np.asarray(actual_up), reference(up_bits, up_scale)
    )
