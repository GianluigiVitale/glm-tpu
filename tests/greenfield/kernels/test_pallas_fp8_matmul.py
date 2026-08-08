from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from glm_tpu.greenfield.kernels.pallas import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_block_matmul_f32,
    fp8_block_vector_matmul_f32,
    fp8_block_up_gate,
    fp8_fused_block_swiglu,
    fp8_fused_selected_moe,
    fp8_fused_structured_kv_b_value_output,
    fp32_to_bf16_pallas_boundary,
    fp8_rmsnorm_block_matmul,
    fp8_selected_swiglu_down,
    fp8_selected_up_gate,
    fp8_structured_kv_b_q_absorb,
    fp8_structured_kv_b_value,
)
from glm_tpu.greenfield.kernels.reference.fp8 import (
    dequantize_fp8_bits_block_weight,
)
from glm_tpu.greenfield.kernels.reference.rmsnorm import rms_norm


def _bits(values: jax.Array) -> jax.Array:
    quantized = values.astype(jnp.float8_e4m3fn)
    return lax.bitcast_convert_type(quantized, jnp.uint8)


@pytest.mark.parametrize("shape", ((8, 256), (3, 135)))
def test_fp32_to_bf16_pallas_boundary_interpret_matches_cast(
    shape: tuple[int, int],
) -> None:
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()
    value = jnp.asarray(
        np.linspace(-3.0, 3.0, np.prod(shape), dtype=np.float32).reshape(shape)
    )
    actual = fp32_to_bf16_pallas_boundary(value, interpret=True)
    assert actual.shape == shape
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(value.astype(jnp.bfloat16))
    )


def test_fp32_to_bf16_pallas_boundary_rejects_contract_drift() -> None:
    with pytest.raises(ValueError, match="rank-two FP32"):
        fp32_to_bf16_pallas_boundary(
            jnp.ones((8, 128), dtype=jnp.bfloat16), interpret=True
        )
    with pytest.raises(ValueError, match="rank-two FP32"):
        fp32_to_bf16_pallas_boundary(
            jnp.ones((1, 8, 128), dtype=jnp.float32), interpret=True
        )
    with pytest.raises(ValueError, match="tiles must be positive"):
        fp32_to_bf16_pallas_boundary(
            jnp.ones((8, 128), dtype=jnp.float32),
            output_tile=0,
            interpret=True,
        )


@pytest.mark.parametrize(
    ("shape", "output_tile"),
    (
        ((1, 128, 128), 128),
        ((3, 130, 135), 128),
        ((1, 128, 256), 256),
        ((3, 130, 257), 256),
    ),
)
def test_fp8_block_matmul_interpret_matches_reference(
    shape: tuple[int, int, int],
    output_tile: int,
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
    actual = fp8_block_matmul(
        lhs,
        weight_bits,
        scale,
        config=Fp8BlockMatmulConfig(output_tile=output_tile),
        interpret=True,
    )
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
    with pytest.raises(ValueError, match="contraction tiles"):
        fp8_block_matmul(
            lhs,
            bits,
            scale,
            config=Fp8BlockMatmulConfig(contraction_tile=512),
            interpret=True,
        )


@pytest.mark.parametrize("shape", [(1, 128, 128), (1, 130, 135)])
def test_fp8_block_matmul_f32_interpret_matches_dsa_reference(
    shape: tuple[int, int, int],
) -> None:
    rows, contraction, output = shape
    lhs = jnp.asarray(
        np.linspace(-0.375, 0.625, rows * contraction, dtype=np.float32).reshape(
            rows, contraction
        ),
        dtype=jnp.bfloat16,
    )
    weight_values = np.arange(output * contraction, dtype=np.float32).reshape(
        output, contraction
    )
    weight_bits = _bits(jnp.asarray(np.sin(weight_values * 0.009) * 0.375))
    scale_shape = (
        (output + 127) // 128,
        (contraction + 127) // 128,
    )
    scale = jnp.asarray(
        np.linspace(0.375, 1.0, np.prod(scale_shape), dtype=np.float32).reshape(
            scale_shape
        )
    )
    decoded = dequantize_fp8_bits_block_weight(weight_bits, scale)
    expected = lax.dot_general(
        lhs.astype(jnp.float32),
        decoded.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    actual = fp8_block_matmul_f32(lhs, weight_bits, scale, interpret=True)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=2e-3, atol=2e-3
    )


def test_fp8_block_vector_matmul_f32_interpret_matches_reference() -> None:
    lhs = jnp.asarray(
        np.linspace(-0.375, 0.625, 128, dtype=np.float32)[None, :],
        dtype=jnp.bfloat16,
    )
    weight_values = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    weight_bits = _bits(jnp.asarray(np.sin(weight_values * 0.009) * 0.375))
    scale = jnp.asarray([[0.625]], dtype=jnp.float32)
    decoded = dequantize_fp8_bits_block_weight(
        weight_bits, scale, output_dtype=jnp.float32
    )
    expected = lax.dot_general(
        lhs.astype(jnp.float32),
        decoded.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    actual = fp8_block_vector_matmul_f32(
        lhs, weight_bits, scale, interpret=True
    )
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=0, atol=5e-7
    )


def test_fp8_block_vector_matmul_f32_rejects_non_dsa_shapes() -> None:
    lhs = jnp.ones((1, 128), dtype=jnp.bfloat16)
    bits = jnp.zeros((128, 128), dtype=jnp.uint8)
    scale = jnp.ones((1, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="one exact decode row"):
        fp8_block_vector_matmul_f32(
            jnp.ones((2, 128), dtype=jnp.bfloat16),
            bits,
            scale,
            interpret=True,
        )
    with pytest.raises(ValueError, match="block-aligned"):
        fp8_block_vector_matmul_f32(
            lhs[:, :127],
            bits[:, :127],
            scale,
            interpret=True,
        )


@pytest.mark.parametrize("shape", [(1, 128, 128), (1, 130, 135)])
def test_fp8_rmsnorm_block_matmul_interpret_matches_exact_reference(
    shape: tuple[int, int, int],
) -> None:
    rows, contraction, output = shape
    hidden = jnp.asarray(
        np.linspace(-0.875, 0.625, rows * contraction, dtype=np.float32).reshape(
            rows, contraction
        ),
        dtype=jnp.bfloat16,
    )
    norm_weight = jnp.asarray(
        np.linspace(0.5, 1.5, contraction, dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    linear_values = np.arange(
        output * contraction, dtype=np.float32
    ).reshape(output, contraction)
    weight_bits = _bits(jnp.asarray(np.sin(linear_values * 0.013) * 0.5))
    scale_shape = (
        (output + 127) // 128,
        (contraction + 127) // 128,
    )
    scale = jnp.asarray(
        np.linspace(0.25, 0.875, np.prod(scale_shape), dtype=np.float32).reshape(
            scale_shape
        )
    )
    normalized = rms_norm(hidden, norm_weight, epsilon=1e-5)
    decoded = dequantize_fp8_bits_block_weight(weight_bits, scale)
    expected = lax.dot_general(
        normalized,
        decoded,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    actual = fp8_rmsnorm_block_matmul(
        hidden,
        norm_weight,
        weight_bits,
        scale,
        epsilon=1e-5,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_fp8_rmsnorm_block_matmul_rejects_contract_drift() -> None:
    hidden = jnp.ones((1, 128), dtype=jnp.bfloat16)
    norm_weight = jnp.ones((128,), dtype=jnp.bfloat16)
    bits = jnp.zeros((128, 128), dtype=jnp.uint8)
    scale = jnp.ones((1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="one decode row"):
        fp8_rmsnorm_block_matmul(
            jnp.ones((2, 128), dtype=jnp.bfloat16),
            norm_weight,
            bits,
            scale,
            epsilon=1e-5,
            interpret=True,
        )
    with pytest.raises(ValueError, match="contraction width"):
        fp8_rmsnorm_block_matmul(
            hidden,
            jnp.ones((127,), dtype=jnp.bfloat16),
            bits,
            scale,
            epsilon=1e-5,
            interpret=True,
        )
    with pytest.raises(ValueError, match="weight must be BF16"):
        fp8_rmsnorm_block_matmul(
            hidden,
            norm_weight.astype(jnp.float32),
            bits,
            scale,
            epsilon=1e-5,
            interpret=True,
        )
    with pytest.raises(ValueError, match="epsilon must be positive"):
        fp8_rmsnorm_block_matmul(
            hidden,
            norm_weight,
            bits,
            scale,
            epsilon=0.0,
            interpret=True,
        )


def _structured_kv_b_case() -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    tuple[jax.Array, jax.Array],
]:
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()
    # Two heads are the minimum exact layout covering both the aligned and
    # 64-row-offset checkpoint cases. The protected TPU proof uses all 16.
    heads, qk_nope, value_width, latent = 2, 192, 256, 512
    combined = qk_nope + value_width
    q_nope = jnp.asarray(
        np.linspace(-0.25, 0.375, heads * qk_nope, dtype=np.float32).reshape(
            1, heads, qk_nope
        ),
        dtype=jnp.bfloat16,
    )
    attended = jnp.asarray(
        np.linspace(-0.375, 0.25, heads * latent, dtype=np.float32).reshape(
            1, heads, latent
        ),
        dtype=jnp.bfloat16,
    )
    # E4M3FN 0x38 is exactly +1.0. Distinct scales in all 28 source blocks
    # isolate row-offset/scale selection without compiling a second JAX oracle.
    weight_bits = jnp.full((heads * combined, latent), 0x38, dtype=jnp.uint8)
    scale_values = np.linspace(0.125, 1.0, 7 * 4, dtype=np.float32).reshape(7, 4)
    scale = jnp.asarray(scale_values)
    decoded = np.repeat(np.repeat(scale_values, 128, axis=0), 128, axis=1)
    decoded = decoded[: heads * combined, :latent]
    decoded = np.asarray(jnp.asarray(decoded, dtype=jnp.bfloat16)).astype(np.float32)
    decoded = decoded.reshape(heads, combined, latent)
    expected_absorbed = jnp.asarray(
        np.einsum(
            "rhp,hpl->rhl",
            np.asarray(q_nope).astype(np.float32),
            decoded[:, :qk_nope],
            dtype=np.float32,
        ),
        dtype=jnp.bfloat16,
    )
    expected_value = jnp.asarray(
        np.einsum(
            "rhl,hvl->rhv",
            np.asarray(attended).astype(np.float32),
            decoded[:, qk_nope:],
            dtype=np.float32,
        ),
        dtype=jnp.bfloat16,
    )

    return q_nope, attended, weight_bits, scale, (expected_absorbed, expected_value)


def test_fp8_structured_kv_b_q_absorb_interpret_matches_reference() -> None:
    q_nope, _, weight_bits, scale, expected = _structured_kv_b_case()
    actual = fp8_structured_kv_b_q_absorb(
        q_nope, weight_bits, scale, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected[0]))


def test_fp8_structured_kv_b_value_interpret_matches_reference() -> None:
    _, attended, weight_bits, scale, expected = _structured_kv_b_case()
    actual = fp8_structured_kv_b_value(
        attended, weight_bits, scale, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected[1]))


def test_fp8_fused_structured_value_output_interpret_matches_reference() -> None:
    _, attended, kv_b_bits, kv_b_scale, expected = _structured_kv_b_case()
    value_states = expected[1].reshape(1, 512)
    output_bits = jnp.full((256, 512), 0x38, dtype=jnp.uint8)
    output_scale = jnp.asarray(
        np.linspace(0.0625, 0.5, 8, dtype=np.float32).reshape(2, 4)
    )
    decoded_output = dequantize_fp8_bits_block_weight(
        output_bits, output_scale
    )
    expected_output = lax.dot_general(
        value_states,
        decoded_output,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    actual = fp8_fused_structured_kv_b_value_output(
        attended,
        kv_b_bits,
        kv_b_scale,
        output_bits,
        output_scale,
        interpret=True,
    )
    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(expected_output)
    )


def test_fp8_structured_kv_b_rejects_contract_drift() -> None:
    bits = jnp.zeros((2 * 448, 512), dtype=jnp.uint8)
    scale = jnp.ones((7, 4), dtype=jnp.float32)
    q_nope = jnp.zeros((1, 2, 192), dtype=jnp.bfloat16)
    attended = jnp.zeros((1, 2, 512), dtype=jnp.bfloat16)

    with pytest.raises(ValueError, match="exact GLM"):
        fp8_structured_kv_b_q_absorb(
            q_nope[:, :, :128], bits, scale, interpret=True
        )
    with pytest.raises(ValueError, match="activation must be BF16"):
        fp8_structured_kv_b_value(
            attended.astype(jnp.float32), bits, scale, interpret=True
        )
    with pytest.raises(ValueError, match="scale shape"):
        fp8_structured_kv_b_value(
            attended, bits, jnp.ones((7, 3), dtype=jnp.float32), interpret=True
        )
    with pytest.raises(ValueError, match="contract all value heads"):
        fp8_fused_structured_kv_b_value_output(
            attended,
            bits,
            scale,
            jnp.zeros((256, 384), dtype=jnp.uint8),
            jnp.ones((2, 3), dtype=jnp.float32),
            interpret=True,
        )


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
    unsupported_config = Fp8BlockMatmulConfig(output_tile=384)
    with pytest.raises(ValueError, match="one or two scale blocks"):
        fp8_block_matmul(
            jnp.ones((1, 128), dtype=jnp.bfloat16),
            jnp.zeros((128, 128), dtype=jnp.uint8),
            jnp.ones((1, 1), dtype=jnp.float32),
            config=unsupported_config,
            interpret=True,
        )
    with pytest.raises(
        ValueError, match="integral output and contraction scale blocks"
    ):
        Fp8BlockMatmulConfig(output_tile=192)
    with pytest.raises(
        ValueError, match="integral output and contraction scale blocks"
    ):
        Fp8BlockMatmulConfig(contraction_tile=192)
    with pytest.raises(ValueError, match="accumulator must be FP32"):
        Fp8BlockMatmulConfig(accumulator_dtype=jnp.bfloat16)


def test_fp8_fused_block_swiglu_interpret_matches_exact_reference() -> None:
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()
    rows, hidden_size, intermediate, output = 1, 135, 130, 137
    lhs = jnp.asarray(
        np.linspace(-0.5, 0.5, rows * hidden_size, dtype=np.float32).reshape(
            rows, hidden_size
        ),
        dtype=jnp.bfloat16,
    )
    projection_linear = np.arange(
        intermediate * hidden_size, dtype=np.float32
    ).reshape(intermediate, hidden_size)
    gate_bits = _bits(jnp.asarray(np.sin(projection_linear * 0.013) * 0.5))
    up_bits = _bits(jnp.asarray(np.cos(projection_linear * 0.019) * 0.375))
    projection_scale_shape = (
        (intermediate + 127) // 128,
        (hidden_size + 127) // 128,
    )
    gate_scale = jnp.asarray(
        np.linspace(
            0.25, 0.75, np.prod(projection_scale_shape), dtype=np.float32
        ).reshape(projection_scale_shape)
    )
    up_scale = jnp.asarray(
        np.linspace(
            0.5, 1.0, np.prod(projection_scale_shape), dtype=np.float32
        ).reshape(projection_scale_shape)
    )
    down_linear = np.arange(output * intermediate, dtype=np.float32).reshape(
        output, intermediate
    )
    down_bits = _bits(jnp.asarray(np.sin(down_linear * 0.017) * 0.375))
    down_scale_shape = (
        (output + 127) // 128,
        (intermediate + 127) // 128,
    )
    down_scale = jnp.asarray(
        np.linspace(
            0.375, 0.875, np.prod(down_scale_shape), dtype=np.float32
        ).reshape(down_scale_shape)
    )

    actual = fp8_fused_block_swiglu(
        lhs,
        gate_bits,
        gate_scale,
        up_bits,
        up_scale,
        down_bits,
        down_scale,
        interpret=True,
    )
    decoded_gate = dequantize_fp8_bits_block_weight(gate_bits, gate_scale)
    decoded_up = dequantize_fp8_bits_block_weight(up_bits, up_scale)
    gate = lax.dot_general(
        lhs,
        decoded_gate,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    up = lax.dot_general(
        lhs,
        decoded_up,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    activated = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
    decoded_down = dequantize_fp8_bits_block_weight(down_bits, down_scale)
    expected = lax.dot_general(
        activated,
        decoded_down,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_fp8_fused_block_swiglu_rejects_down_contract_drift() -> None:
    lhs = jnp.ones((1, 128), dtype=jnp.bfloat16)
    bits = jnp.zeros((128, 128), dtype=jnp.uint8)
    scale = jnp.ones((1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="contract the gate/up output"):
        fp8_fused_block_swiglu(
            lhs,
            bits,
            scale,
            bits,
            scale,
            jnp.zeros((128, 127), dtype=jnp.uint8),
            scale,
            interpret=True,
        )
    with pytest.raises(ValueError, match="down FP32 scale shape"):
        fp8_fused_block_swiglu(
            lhs,
            bits,
            scale,
            bits,
            scale,
            bits,
            jnp.ones((1, 2), dtype=jnp.float32),
            interpret=True,
        )


@pytest.mark.parametrize(
    "route_values",
    ([11, 500, 10, 12], [12, 10, 11, 12]),
)
def test_fp8_selected_up_gate_interpret_uses_distinct_owned_experts(
    route_values: list[int],
) -> None:
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
    gate_bits_nk = _bits(jnp.asarray(np.sin(linear * 0.013) * 0.5))
    up_bits_nk = _bits(jnp.asarray(np.cos(linear * 0.019) * 0.375))
    gate_bits = jnp.transpose(gate_bits_nk, (0, 2, 1))
    up_bits = jnp.transpose(up_bits_nk, (0, 2, 1))
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
    route_indices = jnp.asarray(route_values, dtype=jnp.int32)
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
                jnp.transpose(bits[local_expert]), scale[local_expert]
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


@pytest.mark.parametrize(
    "route_values",
    ([11, 500, 10, 12], [12, 10, 11, 12]),
)
def test_fp8_selected_swiglu_down_interpret_uses_distinct_owned_experts(
    route_values: list[int],
) -> None:
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()
    routes, experts, intermediate, hidden = 4, 3, 130, 135
    gate = jnp.asarray(
        np.sin(
            np.arange(routes * intermediate, dtype=np.float32).reshape(
                routes, intermediate
            )
            * 0.013
        ),
        dtype=jnp.bfloat16,
    )
    up = jnp.asarray(
        np.cos(
            np.arange(routes * intermediate, dtype=np.float32).reshape(
                routes, intermediate
            )
            * 0.017
        ),
        dtype=jnp.bfloat16,
    )
    linear = np.arange(
        experts * hidden * intermediate, dtype=np.float32
    ).reshape(experts, hidden, intermediate)
    down_bits_nk = _bits(jnp.asarray(np.sin(linear * 0.019) * 0.375))
    down_bits = jnp.transpose(down_bits_nk, (0, 2, 1))
    scale_shape = (
        experts,
        (hidden + 127) // 128,
        (intermediate + 127) // 128,
    )
    down_scale = jnp.asarray(
        np.linspace(0.25, 0.75, np.prod(scale_shape), dtype=np.float32).reshape(
            scale_shape
        )
    )
    route_indices = jnp.asarray(route_values, dtype=jnp.int32)
    expert_start = jnp.asarray(10, dtype=jnp.int32)
    actual = fp8_selected_swiglu_down(
        gate,
        up,
        route_indices,
        expert_start,
        down_bits,
        down_scale,
        interpret=True,
    )

    expected = []
    for route_slot, global_expert in enumerate(np.asarray(route_indices)):
        local_expert = int(global_expert) - int(expert_start)
        if not 0 <= local_expert < experts:
            expected.append(np.zeros((hidden,), dtype=np.float32))
            continue
        activated = (
            gate[route_slot]
            * jax.nn.sigmoid(gate[route_slot])
            * up[route_slot]
        ).astype(jnp.bfloat16)
        decoded = dequantize_fp8_bits_block_weight(
            down_bits_nk[local_expert], down_scale[local_expert]
        )
        value = lax.dot_general(
            activated[None, :],
            decoded,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        expected.append(np.asarray(value[0]))
    np.testing.assert_array_equal(np.asarray(actual), np.stack(expected))


def test_fp8_selected_swiglu_down_rejects_contract_drift() -> None:
    gate = jnp.ones((2, 128), dtype=jnp.bfloat16)
    up = jnp.ones_like(gate)
    routes = jnp.asarray([0, 1], dtype=jnp.int32)
    expert_start = jnp.asarray(0, dtype=jnp.int32)
    down_bits = jnp.zeros((2, 128, 128), dtype=jnp.uint8)
    scale = jnp.ones((2, 1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="gate/up shapes"):
        fp8_selected_swiglu_down(
            gate,
            up[:, :-1],
            routes,
            expert_start,
            down_bits,
            scale,
            interpret=True,
        )
    with pytest.raises(ValueError, match="inputs must be BF16"):
        fp8_selected_swiglu_down(
            gate.astype(jnp.float32),
            up,
            routes,
            expert_start,
            down_bits,
            scale,
            interpret=True,
        )
    with pytest.raises(ValueError, match="scale shape"):
        fp8_selected_swiglu_down(
            gate,
            up,
            routes,
            expert_start,
            down_bits,
            jnp.ones((2, 1, 2), dtype=jnp.float32),
            interpret=True,
        )


@pytest.mark.parametrize(
    "route_values",
    ([11, 500, 10, 12], [12, 10, 11, 12]),
)
@pytest.mark.parametrize("output_tile", (128, 256))
@pytest.mark.parametrize("retain_down_f32", (False, True))
def test_fp8_fused_selected_moe_interpret_matches_exact_reference(
    route_values: list[int],
    output_tile: int,
    retain_down_f32: bool,
) -> None:
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()
    routes, experts, hidden_size, intermediate = 4, 3, 135, 130
    hidden = jnp.asarray(
        np.linspace(-0.5, 0.5, hidden_size, dtype=np.float32)[None, :],
        dtype=jnp.bfloat16,
    )
    up_linear = np.arange(
        experts * intermediate * hidden_size, dtype=np.float32
    ).reshape(experts, intermediate, hidden_size)
    gate_bits_nk = _bits(jnp.asarray(np.sin(up_linear * 0.013) * 0.5))
    up_bits_nk = _bits(jnp.asarray(np.cos(up_linear * 0.019) * 0.375))
    gate_bits = jnp.transpose(gate_bits_nk, (0, 2, 1))
    up_bits = jnp.transpose(up_bits_nk, (0, 2, 1))
    up_scale_shape = (
        experts,
        (intermediate + 127) // 128,
        (hidden_size + 127) // 128,
    )
    gate_scale = jnp.asarray(
        np.linspace(
            0.25, 0.75, np.prod(up_scale_shape), dtype=np.float32
        ).reshape(up_scale_shape)
    )
    up_scale = jnp.asarray(
        np.linspace(
            0.5, 1.0, np.prod(up_scale_shape), dtype=np.float32
        ).reshape(up_scale_shape)
    )
    down_linear = np.arange(
        experts * hidden_size * intermediate, dtype=np.float32
    ).reshape(experts, hidden_size, intermediate)
    down_bits_nk = _bits(jnp.asarray(np.sin(down_linear * 0.017) * 0.375))
    down_bits = jnp.transpose(down_bits_nk, (0, 2, 1))
    down_scale_shape = (
        experts,
        (hidden_size + 127) // 128,
        (intermediate + 127) // 128,
    )
    down_scale = jnp.asarray(
        np.linspace(
            0.375, 0.875, np.prod(down_scale_shape), dtype=np.float32
        ).reshape(down_scale_shape)
    )
    route_indices = jnp.asarray(route_values, dtype=jnp.int32)
    expert_start = jnp.asarray(10, dtype=jnp.int32)

    actual = fp8_fused_selected_moe(
        hidden,
        route_indices,
        expert_start,
        gate_bits,
        gate_scale,
        up_bits,
        up_scale,
        down_bits,
        down_scale,
        down_result_dtype=(jnp.float32 if retain_down_f32 else jnp.bfloat16),
        config=Fp8BlockMatmulConfig(
            contraction_tile=512,
            output_tile=output_tile,
        ),
        interpret=True,
    )

    expected = []
    for global_expert in np.asarray(route_indices):
        local_expert = int(global_expert) - int(expert_start)
        if not 0 <= local_expert < experts:
            expected.append(np.zeros((hidden_size,), dtype=np.float32))
            continue
        decoded_gate = dequantize_fp8_bits_block_weight(
            gate_bits_nk[local_expert], gate_scale[local_expert]
        )
        decoded_up = dequantize_fp8_bits_block_weight(
            up_bits_nk[local_expert], up_scale[local_expert]
        )
        gate = lax.dot_general(
            hidden,
            decoded_gate,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        up = lax.dot_general(
            hidden,
            decoded_up,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        activated = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)
        decoded_down = dequantize_fp8_bits_block_weight(
            down_bits_nk[local_expert], down_scale[local_expert]
        )
        value = lax.dot_general(
            activated,
            decoded_down,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.float32 if retain_down_f32 else jnp.bfloat16)
        expected.append(np.asarray(value[0]))
    assert actual.dtype == (jnp.float32 if retain_down_f32 else jnp.bfloat16)
    if retain_down_f32:
        # The interpreted tiled accumulator and the monolithic reference dot
        # associate FP32 adds differently. Pin the observed FP32-only bound;
        # the promoted contract still rounds just once after reconstruction.
        np.testing.assert_allclose(
            np.asarray(actual),
            np.stack(expected),
            rtol=2e-5,
            atol=1e-5,
        )
    else:
        np.testing.assert_array_equal(np.asarray(actual), np.stack(expected))


def test_fp8_fused_selected_moe_rejects_down_contract_drift() -> None:
    hidden = jnp.ones((1, 128), dtype=jnp.bfloat16)
    routes = jnp.asarray([0, 1], dtype=jnp.int32)
    expert_start = jnp.asarray(0, dtype=jnp.int32)
    up_bits = jnp.zeros((2, 128, 128), dtype=jnp.uint8)
    scale = jnp.ones((2, 1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="down weights"):
        fp8_fused_selected_moe(
            hidden,
            routes,
            expert_start,
            up_bits,
            scale,
            up_bits,
            scale,
            jnp.zeros((2, 127, 128), dtype=jnp.uint8),
            scale,
            interpret=True,
        )
    with pytest.raises(ValueError, match="down FP32 scale shape"):
        fp8_fused_selected_moe(
            hidden,
            routes,
            expert_start,
            up_bits,
            scale,
            up_bits,
            scale,
            up_bits,
            jnp.ones((2, 1, 2), dtype=jnp.float32),
            interpret=True,
        )
    with pytest.raises(ValueError, match="requires BF16 down results"):
        fp8_fused_selected_moe(
            hidden,
            routes,
            expert_start,
            up_bits,
            scale,
            up_bits,
            scale,
            up_bits,
            scale,
            route_weights=jnp.ones((2,), dtype=jnp.float32),
            down_result_dtype=jnp.float32,
            interpret=True,
        )
    with pytest.raises(ValueError, match="must be BF16 or FP32"):
        fp8_fused_selected_moe(
            hidden,
            routes,
            expert_start,
            up_bits,
            scale,
            up_bits,
            scale,
            up_bits,
            scale,
            down_result_dtype=jnp.float16,
            interpret=True,
        )
