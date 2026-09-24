from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from glm_tpu.optimized.reference.dsa import DsaNumericalContract
from glm_tpu.greenfield.kernels.reference.dsa_association import (
    Layer0DsaProbeGeometry,
    layer0_prompt_index_key_chunk,
)
from glm_tpu.optimized.reference.prefill_index import (
    decode_stage_local_prefill_index_wk_bf16,
    materialize_stage_local_prefill_index_wk,
    physical_m64_prompt_index_key_chunk,
    promote_stage_local_prefill_index_wk,
    repair_stage_local_prompt_index_cache,
)
from glm_tpu.optimized.reference.rmsnorm import rms_norm


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


def test_prompt_index_repair_supports_lp8_offset_and_valid_tail() -> None:
    contract = _contract()
    history = jnp.asarray(
        np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 64,
        dtype=jnp.bfloat16,
    )
    initial = jnp.zeros((2, 1, 4), dtype=jnp.bfloat16)
    block_tables = jnp.asarray([[0, 1]], dtype=jnp.int32)
    wk_weight = jnp.ones((4, 8), dtype=jnp.float32)
    key_norm = jnp.ones((4,), dtype=jnp.bfloat16)
    key_bias = jnp.zeros((4,), dtype=jnp.bfloat16)
    repaired = tuple(
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
            local_rows_per_page=1,
            prompt_chunk=4,
            physical_rows=2,
            local_parallel_size=8,
            position_offset=8,
            valid_rows=3,
        )
        for owner in range(8)
    )
    assert all(
        np.count_nonzero(np.asarray(repaired[owner])[1].view(np.uint16)) > 0
        for owner in range(3)
    )
    assert all(
        np.count_nonzero(np.asarray(repaired[owner]).view(np.uint16)) == 0
        for owner in range(3, 8)
    )


def test_prompt_index_repair_supports_lp2_page_ownership() -> None:
    contract = _contract()
    prompt_chunk = 2048
    valid_rows = 2012
    position_offset = 4096
    history = jnp.asarray(
        np.sin(
            np.arange(prompt_chunk * 8, dtype=np.float32).reshape(
                prompt_chunk, 8
            )
            / 67.0
        ),
        dtype=jnp.bfloat16,
    )
    initial = jnp.zeros((12, 256, 4), dtype=jnp.bfloat16)
    block_tables = jnp.arange(12, dtype=jnp.int32)[None, :]
    wk_weight = jnp.asarray(
        np.cos(np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 11.0),
        dtype=jnp.float32,
    )
    key_norm = jnp.asarray([1.0, 0.75, -0.5, 1.25], dtype=jnp.bfloat16)
    key_bias = jnp.asarray([0.0, 0.25, -0.125, 0.5], dtype=jnp.bfloat16)
    repaired = tuple(
        repair_stage_local_prompt_index_cache(
            initial,
            history,
            block_tables,
            wk_weight,
            key_norm,
            key_bias,
            jnp.int32(owner),
            contract=contract,
            logical_page_size=512,
            local_rows_per_page=256,
            prompt_chunk=prompt_chunk,
            physical_rows=64,
            local_parallel_size=2,
            position_offset=position_offset,
            valid_rows=valid_rows,
        )
        for owner in range(2)
    )
    direct = physical_m64_prompt_index_key_chunk(
        history,
        jnp.arange(
            position_offset,
            position_offset + prompt_chunk,
            dtype=jnp.int32,
        ),
        wk_weight,
        key_norm,
        key_bias,
        contract=contract,
        physical_rows=64,
    ).astype(jnp.bfloat16)

    logical = np.stack(
        tuple(
            np.asarray(repaired[(position % 512) // 256])[
                position // 512, position % 256
            ]
            for position in range(
                position_offset, position_offset + valid_rows
            )
        ),
        axis=0,
    )
    np.testing.assert_array_equal(
        logical.view(np.uint16),
        np.asarray(direct[:valid_rows]).view(np.uint16),
    )
    masked_tail = np.stack(
        tuple(
            np.asarray(repaired[(position % 512) // 256])[
                position // 512, position % 256
            ]
            for position in range(
                position_offset + valid_rows,
                position_offset + prompt_chunk,
            )
        ),
        axis=0,
    )
    assert masked_tail.shape == (36, 4)
    assert np.count_nonzero(masked_tail.view(np.uint16)) == 0
    assert all(
        np.count_nonzero(np.asarray(value)[:8].view(np.uint16)) == 0
        for value in repaired
    )


def test_prompt_index_repair_traced_offset_matches_static_offset() -> None:
    """Chunked prefill passes the chunk start as a traced int32 scalar; every
    owner's repaired rows must be bit-identical to the static-offset call and
    the owners together must write exactly the chunk's valid positions."""
    import jax
    import pytest

    contract = _contract()
    prompt_chunk = 64
    valid_rows = 37
    offset = 128
    logical_page_size, local_rows_per_page = 128, 16
    history = jnp.asarray(
        np.cos(np.arange(prompt_chunk * 8, dtype=np.float32).reshape(prompt_chunk, 8) / 13.0),
        dtype=jnp.bfloat16,
    )
    initial = jnp.zeros((4, local_rows_per_page, 4), dtype=jnp.bfloat16)
    block_tables = jnp.arange(4, dtype=jnp.int32)[None, :]
    wk_weight = jnp.asarray(
        np.sin(np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 7.0), dtype=jnp.float32
    )
    key_norm = jnp.asarray([1.0, 0.5, -0.75, 1.5], dtype=jnp.bfloat16)
    key_bias = jnp.asarray([0.0, 0.125, -0.25, 0.5], dtype=jnp.bfloat16)

    def repair(position_offset, owner):
        return repair_stage_local_prompt_index_cache(
            initial,
            history,
            block_tables,
            wk_weight,
            key_norm,
            key_bias,
            jnp.int32(owner),
            contract=contract,
            logical_page_size=logical_page_size,
            local_rows_per_page=local_rows_per_page,
            prompt_chunk=prompt_chunk,
            physical_rows=64,
            local_parallel_size=8,
            position_offset=position_offset,
            valid_rows=valid_rows,
        )

    written: set[int] = set()
    for owner in range(8):
        static = np.asarray(repair(offset, owner)).view(np.uint16)
        traced = np.asarray(
            jax.jit(lambda o, owner=owner: repair(o, owner))(jnp.int32(offset))
        ).view(np.uint16)
        assert np.array_equal(static, traced), owner
        for page, row in zip(*np.nonzero(static.any(axis=-1))):
            position = int(page) * logical_page_size + owner * local_rows_per_page + int(row)
            assert position not in written
            written.add(position)
    assert written == set(range(offset, offset + valid_rows))
    with pytest.raises(ValueError, match="static int or int32 scalar"):
        repair(jnp.asarray([offset], dtype=jnp.int32), 0)
    with pytest.raises(ValueError, match="nonnegative"):
        repair(-1, 0)


def test_chunked_prompt_index_repair_equals_monolithic_repair() -> None:
    """Spec §23.2 dual buffer: repairing a prompt in fixed chunks (traced
    offsets, static valid rows) into a zero buffer must reproduce the single
    full-prompt repair bit-for-bit, for a chunk that divides the prompt and one
    that leaves a short tail."""
    import jax

    contract = _contract()
    prompt_length = 300
    logical_page_size, local_rows_per_page = 128, 16
    pages = 4
    history = jnp.asarray(
        np.cos(np.arange(prompt_length * 8, dtype=np.float32).reshape(prompt_length, 8) / 17.0),
        dtype=jnp.bfloat16,
    )
    zeros = jnp.zeros((pages, local_rows_per_page, 4), dtype=jnp.bfloat16)
    block_tables = jnp.arange(pages, dtype=jnp.int32)[None, :]
    wk_weight = jnp.asarray(
        np.sin(np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 5.0), dtype=jnp.float32
    )
    key_norm = jnp.asarray([1.0, 0.5, -0.75, 1.5], dtype=jnp.bfloat16)
    key_bias = jnp.asarray([0.0, 0.125, -0.25, 0.5], dtype=jnp.bfloat16)

    def repair(cache, inputs, owner, offset, valid_rows, prompt_chunk):
        return repair_stage_local_prompt_index_cache(
            cache,
            inputs,
            block_tables,
            wk_weight,
            key_norm,
            key_bias,
            jnp.int32(owner),
            contract=contract,
            logical_page_size=logical_page_size,
            local_rows_per_page=local_rows_per_page,
            prompt_chunk=prompt_chunk,
            physical_rows=64,
            local_parallel_size=8,
            position_offset=offset,
            valid_rows=valid_rows,
        )

    for chunk in (100, 64, 128):  # 3 chunks exact; 4 + tail 44; 2 + tail 44
        full_chunks = (prompt_length - 1) // chunk
        tail = prompt_length - full_chunks * chunk
        assert 1 <= tail <= chunk
        for owner in range(8):
            monolithic = np.asarray(
                repair(zeros, history, owner, 0, prompt_length, 2048)
            ).view(np.uint16)
            buffer = zeros
            for index in range(full_chunks):
                start = index * chunk
                buffer = jax.jit(
                    lambda b, h, o, owner=owner, chunk=chunk: repair(
                        b, h, owner, o, chunk, min(2048, (chunk + 63) // 64 * 64)
                    )
                )(buffer, history[start : start + chunk], jnp.int32(start))
            start = full_chunks * chunk
            buffer = jax.jit(
                lambda b, h, o, owner=owner, tail=tail: repair(
                    b, h, owner, o, tail, min(2048, (tail + 63) // 64 * 64)
                )
            )(buffer, history[start : start + tail], jnp.int32(start))
            assert np.array_equal(np.asarray(buffer).view(np.uint16), monolithic), (chunk, owner)
        assert np.count_nonzero(monolithic) > 0
