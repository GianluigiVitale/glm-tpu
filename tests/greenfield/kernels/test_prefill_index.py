from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
from glm_tpu.greenfield.kernels.reference.dsa_association import (
    Layer0DsaProbeGeometry,
    layer0_prompt_index_key_chunk,
)
from glm_tpu.greenfield.kernels.reference.prefill_index import (
    decode_stage_local_prefill_index_wk_bf16,
    materialize_stage_local_prefill_index_wk,
    physical_m64_prompt_index_key_chunk,
    promote_stage_local_prefill_index_wk,
    repair_stage_local_prompt_index_cache,
)
from glm_tpu.greenfield.kernels.reference.rmsnorm import rms_norm


def _contract() -> DsaNumericalContract:
    return DsaNumericalContract(
        hidden_size=8,
        q_lora_rank=4,
        num_heads=1,
        head_dim=4,
        rotary_dim=2,
        top_k=4,
    )


def _chunk_arguments() -> tuple[jax.Array, ...]:
    hidden = jnp.asarray(
        np.arange(8 * 8, dtype=np.float32).reshape(8, 8) / 64,
        dtype=jnp.bfloat16,
    )
    positions = jnp.arange(8, dtype=jnp.int32)
    input_norm = jnp.ones((8,), dtype=jnp.bfloat16)
    wk = jnp.asarray(
        np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 32,
        dtype=jnp.float32,
    )
    key_norm = jnp.ones((4,), dtype=jnp.bfloat16)
    key_bias = jnp.zeros((4,), dtype=jnp.bfloat16)
    normalized = rms_norm(hidden, input_norm, epsilon=1e-5)
    return (
        hidden,
        normalized,
        positions,
        input_norm,
        wk,
        key_norm,
        key_bias,
    )


def test_physical_m64_prompt_key_chunk_is_semantic_and_lowering_exact() -> None:
    (
        hidden,
        normalized,
        positions,
        input_norm,
        wk,
        key_norm,
        key_bias,
    ) = _chunk_arguments()
    arguments = (normalized, positions, wk, key_norm, key_bias)
    function = partial(
        physical_m64_prompt_index_key_chunk,
        contract=_contract(),
        physical_rows=4,
    )
    actual = function(*arguments)
    proven_hidden = jnp.tile(hidden, (16, 1))
    proven_normalized = rms_norm(
        proven_hidden, input_norm, epsilon=1e-5
    )
    proven_positions = jnp.arange(128, dtype=jnp.int32)
    proven_arguments = (
        proven_normalized,
        proven_positions,
        wk,
        key_norm,
        key_bias,
    )
    production_shape = partial(
        physical_m64_prompt_index_key_chunk,
        contract=_contract(),
        physical_rows=64,
    )
    production_value = production_shape(*proven_arguments)
    proven = layer0_prompt_index_key_chunk(
        proven_hidden,
        proven_positions,
        input_norm,
        wk,
        key_norm,
        key_bias,
        geometry=Layer0DsaProbeGeometry(
            prompt_tokens=127,
            prompt_chunk=128,
            decode_rows=1,
            hidden_size=8,
            q_lora_rank=4,
            qkv_a_companion_rank=2,
            legacy_tensor_shards=2,
            heads=1,
            head_dim=4,
            rotary_dim=2,
            theta=_contract().theta,
            key_norm_epsilon=_contract().key_layer_norm_epsilon,
        ),
        projection_weight_mode="adapted_fp32",
        projection_mapping_mode="physical_m64_projection_keynorm_lax_map",
    )
    np.testing.assert_array_equal(
        np.asarray(production_value.astype(jnp.bfloat16)), np.asarray(proven)
    )
    first = function(
        arguments[0][:4],
        arguments[1][:4],
        *arguments[2:],
    )
    second = function(
        arguments[0][4:],
        arguments[1][4:],
        *arguments[2:],
    )
    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(jnp.concatenate((first, second)))
    )

    stablehlo = str(
        jax.jit(function)
        .lower(*arguments)
        .compiler_ir(dialect="stablehlo")
    )
    assert stablehlo.count("stablehlo.while") == 1
    assert stablehlo.count("precision = [DEFAULT, HIGHEST]") == 1
    assert "tensor<4x4xf32>" in stablehlo
    assert "tensor<4x1xf32>" in stablehlo


def test_prompt_index_repair_writes_only_each_lp4_owner() -> None:
    contract = _contract()
    history = jnp.asarray(
        np.arange(10 * 8, dtype=np.float32).reshape(10, 8) / 128,
        dtype=jnp.bfloat16,
    )
    initial = jnp.zeros((2, 2, 4), dtype=jnp.bfloat16)
    block_tables = jnp.asarray([[0, 1]], dtype=jnp.int32)
    # E4M3FN 0x38 is exactly 1.0; scales are one for the 2x2 blocks.
    wk_bits = jnp.full((4, 8), 0x38, dtype=jnp.uint8)
    wk_scale = jnp.ones((2, 4), dtype=jnp.float32)
    wk_weight = materialize_stage_local_prefill_index_wk(
        wk_bits,
        wk_scale,
        contract=contract,
        fp8_block_shape=(2, 2),
    )
    decoded_wk = decode_stage_local_prefill_index_wk_bf16(
        wk_bits,
        wk_scale,
        contract=contract,
        fp8_block_shape=(2, 2),
    )
    promoted_wk = promote_stage_local_prefill_index_wk(
        decoded_wk, contract=contract
    )
    assert decoded_wk.dtype == jnp.bfloat16
    assert promoted_wk.dtype == jnp.float32
    np.testing.assert_array_equal(promoted_wk, wk_weight)
    key_norm = jnp.ones((4,), dtype=jnp.bfloat16)
    key_bias = jnp.asarray([0.0, 0.25, -0.5, 1.0], dtype=jnp.bfloat16)
    repaired = [
        repair_stage_local_prompt_index_cache(
            initial,
            history,
            block_tables,
            wk_weight,
            key_norm,
            key_bias,
            jnp.int32(owner),
            contract=contract,
            logical_page_size=8,
            local_rows_per_page=2,
            prompt_chunk=8,
            physical_rows=4,
        )
        for owner in range(4)
    ]
    repair = partial(
        repair_stage_local_prompt_index_cache,
        contract=contract,
        logical_page_size=8,
        local_rows_per_page=2,
        prompt_chunk=8,
        physical_rows=4,
    )
    stablehlo = str(
        jax.jit(repair)
        .lower(
            initial,
            history,
            block_tables,
            wk_weight,
            key_norm,
            key_bias,
            jnp.int32(0),
        )
        .compiler_ir(dialect="stablehlo")
    )
    assert "tensor<4x8xui8>" not in stablehlo
    assert "tensor<2x4xf32>" not in stablehlo

    materialize = partial(
        materialize_stage_local_prefill_index_wk,
        contract=contract,
        fp8_block_shape=(2, 2),
    )
    materialize_stablehlo = str(
        jax.jit(materialize)
        .lower(wk_bits, wk_scale)
        .compiler_ir(dialect="stablehlo")
    )
    bf16_origin = materialize_stablehlo.index(
        ": (tensor<4x8xf32>) -> tensor<4x8xbf16>"
    )
    fp32_adapter = materialize_stablehlo.index(
        ": (tensor<4x8xbf16>) -> tensor<4x8xf32>", bf16_origin
    )
    assert fp32_adapter > bf16_origin

    logical = np.zeros((10, 4), dtype=np.dtype(jnp.bfloat16))
    for position in range(10):
        owner = (position % 8) // 2
        page = position // 8
        local_row = position % 2
        logical[position] = np.asarray(repaired[owner])[page, local_row]
    assert np.all(np.isfinite(logical.astype(np.float32)))
    assert all(
        np.count_nonzero(np.asarray(value).view(np.uint16)) > 0
        for value in repaired
    )
