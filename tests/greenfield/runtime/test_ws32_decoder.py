from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.runtime.ws32_decoder import (
    _pack_ws32_fused_qkv_a,
    Ws32DecoderConfig,
    Ws32DecoderWeights,
    bind_ws32_decoder_weights,
    build_ws32_decoder_program,
    build_ws32_teacher_forced_prefill_program,
    make_ws32_initial_state,
    ws32_dsa_observation_specs,
    ws32_decode_result_specs,
    ws32_decoder_state_specs,
    ws32_decoder_weight_names,
    ws32_decoder_weight_specs,
    ws32_observed_decode_result_specs,
    ws32_prefill_result_specs,
)
from glm_tpu.greenfield.kernels.ws32_layer import (
    Ws32StrategyNdDenseWeights,
)
from glm_tpu.greenfield.types import ModelGeometry


ROOT = Path(__file__).resolve().parents[3]


def _geometry() -> ModelGeometry:
    config = json.loads(
        (ROOT / "configs" / "glm-5.2-fp8-config.json").read_text(
            encoding="utf-8"
        )
    )
    return ModelGeometry.from_hf_config(config)


def test_ws32_decoder_contract_covers_exact_78_layer_model() -> None:
    geometry = _geometry()
    config = Ws32DecoderConfig(
        geometry=geometry, context_capacity=262_144
    )
    weights = ws32_decoder_weight_specs(config)
    state = ws32_decoder_state_specs()
    result = ws32_decode_result_specs()
    observation = ws32_dsa_observation_specs()
    observed_result = ws32_observed_decode_result_specs()
    prefill_result = ws32_prefill_result_specs()

    assert config.page_count == 512
    assert config.local_rows_per_page == 64
    assert config.kv_cache_shape == (78, 512, 512, 640)
    assert config.index_cache_shape == (21, 512, 512, 128)
    assert config.full_index_slots == (0, 1, 2, *range(6, 75, 4))
    assert config.full_index_slot_by_layer[:11] == (
        0,
        1,
        2,
        None,
        None,
        None,
        3,
        None,
        None,
        None,
        4,
    )
    assert len(weights.layers) == 78
    assert sum(layer.dense is not None for layer in weights.layers) == 3
    assert sum(layer.moe is not None for layer in weights.layers) == 75
    assert sum(layer.dsa is not None for layer in weights.layers) == 21
    assert isinstance(weights, Ws32DecoderWeights)
    assert str(weights.embedding_local) == "P('expert', 'feature')"
    assert str(weights.final_norm_weight_local) == "P('feature',)"
    assert str(weights.lm_head_local) == "P('expert', 'feature')"
    assert str(state.kv_cache_local) == "P(None, None, 'expert', None)"
    assert str(state.index_cache_local) == "P(None, None, 'expert', None)"
    assert str(result.final_residual_local) == "P(None, 'feature')"
    assert all(str(value) == "P()" for value in observation)
    assert observed_result.result == result
    assert observed_result.dsa == observation
    assert prefill_result.state == state
    assert str(prefill_result.next_token) == "P()"

    local_kv_bytes = 78 * 512 * 64 * 640 * 2
    local_index_bytes = 21 * 512 * 64 * 128 * 2
    assert local_kv_bytes == 3_271_557_120
    assert local_index_bytes == 176_160_768
    assert local_kv_bytes + local_index_bytes == 3_447_717_888


def test_ws32_strategy_nd_dense_contract_is_default_off_and_final_layout() -> None:
    geometry = _geometry()
    default = Ws32DecoderConfig(geometry=geometry, context_capacity=8192)
    assert not default.strategy_nd_dense
    config = Ws32DecoderConfig(
        geometry=geometry,
        context_capacity=8192,
        strategy_nd_dense=True,
    )
    specs = ws32_decoder_weight_specs(config)
    names = ws32_decoder_weight_names(config)
    for layer_id in range(3):
        dense_specs = specs.layers[layer_id].dense
        dense_names = names.layers[layer_id].dense
        assert isinstance(dense_specs, Ws32StrategyNdDenseWeights)
        assert isinstance(dense_names, Ws32StrategyNdDenseWeights)
        assert tuple(map(str, dense_specs)) == (
            "P('expert', None, None)",
            "P('expert', None, None)",
            "P('expert', None, 'feature')",
            "P('expert', None, 'feature')",
        )
        assert dense_names.merged_bits_in_out_local == (
            f"model.layers.{layer_id}.mlp.strategy_nd."
            "merged_gate_up.weight_bits_in_out"
        )
    assert all(layer.dense is None for layer in specs.layers[3:])
    with pytest.raises(PlanValidationError, match="exact GLM-5.2 geometry"):
        Ws32DecoderConfig(
            geometry=replace(geometry, hidden_size=3072),
            context_capacity=8192,
            strategy_nd_dense=True,
        )


def test_ws32_decoder_contract_refuses_schedule_and_cache_drift() -> None:
    geometry = _geometry()
    with pytest.raises(PlanValidationError, match="layer zero"):
        Ws32DecoderConfig(
            geometry=replace(
                geometry,
                indexer_types=("shared", *geometry.indexer_types[1:]),
            ),
            context_capacity=8192,
        )
    with pytest.raises(PlanValidationError, match="packed cache"):
        Ws32DecoderConfig(
            geometry=geometry,
            context_capacity=8192,
            packed_cache_width=576,
        )
    with pytest.raises(PlanValidationError, match="context capacity"):
        Ws32DecoderConfig(geometry=geometry, context_capacity=0)
    with pytest.raises(PlanValidationError, match="exact DSA flag"):
        Ws32DecoderConfig(
            geometry=geometry,
            context_capacity=8192,
            exact_dsa=1,  # type: ignore[arg-type]
        )


def test_ws32_exact_dsa_materializer_has_production_global_shapes() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig,
    Ws32ExactDsaRawLayerWeights,
    build_ws32_exact_dsa_materializer_program,
)
from glm_tpu.greenfield.types import ModelGeometry

geometry = ModelGeometry.from_hf_config(
    json.loads(open("configs/glm-5.2-fp8-config.json").read())
)
config = Ws32DecoderConfig(
    geometry=geometry, context_capacity=8192, exact_dsa=True
)
mesh = Mesh(
    np.asarray(jax.devices(), dtype=object).reshape(8, 4),
    ("expert", "feature"),
)
raw = tuple(
    Ws32ExactDsaRawLayerWeights(
        jax.ShapeDtypeStruct((2048, 6144), jnp.uint8),
        jax.ShapeDtypeStruct((16, 48), jnp.float32),
        jax.ShapeDtypeStruct((576, 6144), jnp.uint8),
        jax.ShapeDtypeStruct((5, 48), jnp.float32),
        jax.ShapeDtypeStruct((4096, 2048), jnp.uint8),
        jax.ShapeDtypeStruct((32, 16), jnp.float32),
        jax.ShapeDtypeStruct((128, 6144), jnp.uint8),
        jax.ShapeDtypeStruct((1, 48), jnp.float32),
        jax.ShapeDtypeStruct((32, 6144), jnp.bfloat16),
    )
    for _ in config.full_index_slots
)
materializer = build_ws32_exact_dsa_materializer_program(mesh, config)
decoded = jax.eval_shape(materializer.decode, raw)
promoted = jax.eval_shape(materializer.promote, decoded)
first = decoded[0]
final = promoted[0]
print(json.dumps({
    "layers": len(decoded),
    "qkv_bits": list(first.qkv_a_bits.shape),
    "qkv_scale": list(first.qkv_a_scale.shape),
    "wq": list(first.wq_b_weight_local.shape),
    "wk_bf16": [list(first.wk_weight_bf16.shape), first.wk_weight_bf16.dtype.name],
    "wq_aliases": [list(value.shape) for value in final.wq_b_weight_aliases],
    "wk_f32": [list(final.wk_weight.shape), final.wk_weight.dtype.name],
    "head": list(first.head_weight_local.shape),
}))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "layers": 21,
        "qkv_bits": [32, 6144, 82],
        "qkv_scale": [32, 48, 82],
        "wq": [4096, 2048],
        "wq_aliases": [[4096, 2048]] * 4,
        "wk_bf16": [[128, 6144], "bfloat16"],
        "wk_f32": [[128, 6144], "float32"],
        "head": [32, 6144],
    }


def test_ws32_exact_qkv_pack_preserves_virtual_shard_part_order() -> None:
    geometry = SimpleNamespace(
        hidden_size=128,
        q_lora_rank=64,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
    )
    config = SimpleNamespace(geometry=geometry)
    q_bits = np.arange(64 * 128, dtype=np.uint16).reshape(64, 128).astype(
        np.uint8
    )
    kv_bits = (
        np.arange(32 * 128, dtype=np.uint16).reshape(32, 128) + 17
    ).astype(np.uint8)
    packed_bits, packed_scale = _pack_ws32_fused_qkv_a(
        jnp.asarray(q_bits),
        jnp.asarray([[2.0]], dtype=jnp.float32),
        jnp.asarray(kv_bits),
        jnp.asarray([[3.0]], dtype=jnp.float32),
        config=config,  # type: ignore[arg-type]
    )
    expected_bits = np.stack(
        [
            np.concatenate(
                (
                    q_bits[shard * 2 : (shard + 1) * 2].T,
                    kv_bits[shard : shard + 1].T,
                ),
                axis=1,
            )
            for shard in range(32)
        ]
    )
    assert np.array_equal(np.asarray(packed_bits), expected_bits)
    assert np.array_equal(
        np.asarray(packed_scale),
        np.broadcast_to(
            np.asarray([2.0, 2.0, 3.0], dtype=np.float32),
            (32, 1, 3),
        ),
    )


def test_ws32_decoder_names_bind_every_exact_final_layout_tensor() -> None:
    config = Ws32DecoderConfig(
        geometry=_geometry(), context_capacity=8192
    )
    names = ws32_decoder_weight_names(config)

    def leaves(value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        assert isinstance(value, tuple)
        return tuple(name for item in value for name in leaves(item))

    exact_names = leaves(names)
    assert len(exact_names) == len(set(exact_names)) == 2310
    arrays = {name: object() for name in exact_names}
    bound = bind_ws32_decoder_weights(arrays, config)
    assert bound.embedding_local is arrays["model.embed_tokens.weight"]
    assert bound.layers[0].qkv_a.q_a_bits_local is arrays[
        "model.layers.0.self_attn.q_a_proj.weight_bits"
    ]
    assert bound.layers[77].moe is not None
    assert bound.layers[77].moe.expert_down_bits_local is arrays[
        "model.layers.77.mlp.experts.down_proj.weight_bits"
    ]
    assert bound.final_norm_weight_local is arrays["model.norm.weight"]
    assert bound.lm_head_local is arrays["lm_head.weight"]

    missing = dict(arrays)
    missing.pop("model.norm.weight")
    with pytest.raises(ValueError, match="tensor set drifted"):
        bind_ws32_decoder_weights(missing, config)
    with pytest.raises(ValueError, match="tensor set drifted"):
        bind_ws32_decoder_weights({**arrays, "rogue": object()}, config)


def test_ws32_program_builders_require_exact_mesh_and_prompt_contract() -> None:
    program = r'''
import json
from dataclasses import replace

import jax
import numpy as np
from jax.sharding import Mesh

from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig,
    build_ws32_decoder_program,
    build_ws32_teacher_forced_prefill_program,
    make_ws32_initial_state,
)
from glm_tpu.greenfield.types import ModelGeometry

source = json.loads(open("configs/glm-5.2-fp8-config.json").read())
base = ModelGeometry.from_hf_config(source)
geometry = replace(
    base,
    num_layers=1,
    first_dense_layers=1,
    hidden_size=128,
    dense_intermediate_size=256,
    num_routed_experts=16,
    routed_top_k=4,
    moe_intermediate_size=128,
    dsa_top_k=128,
    dsa_indexer_heads=8,
    dsa_indexer_head_dim=16,
    attention_heads=8,
    kv_heads=1,
    kv_lora_rank=32,
    q_lora_rank=64,
    qk_nope_head_dim=16,
    qk_rope_head_dim=16,
    v_head_dim=16,
    max_position_embeddings=256,
    vocab_size=64,
    fp8_block_shape=(16, 16),
    mlp_layer_types=("dense",),
    indexer_types=("full",),
)
config = Ws32DecoderConfig(
    geometry=geometry,
    context_capacity=256,
    logical_page_size=128,
    packed_cache_width=112,
    sparse_segment_block=128,
)
mesh = Mesh(
    np.asarray(jax.devices(), dtype=object).reshape(8, 4),
    ("expert", "feature"),
)
decoder = build_ws32_decoder_program(
    mesh,
    config,
    sparse_attention_interpret=True,
    linear_interpret=True,
)
prefill = build_ws32_teacher_forced_prefill_program(
    mesh,
    config,
    prompt_length=2,
    sparse_attention_interpret=True,
    linear_interpret=True,
)
state = make_ws32_initial_state(mesh, config)
print(json.dumps({
    "decoder": callable(decoder.execute) and callable(decoder.observe)
        and callable(decoder.probe_cache_write),
    "prefill": callable(prefill.execute),
    "prompt_length": prefill.prompt_length,
    "state": {
        "context_lengths": np.asarray(state.context_lengths).tolist(),
        "index_shape": list(state.index_cache_local.shape),
        "kv_shape": list(state.kv_cache_local.shape),
        "position": np.asarray(state.position).tolist(),
    },
}))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "decoder": True,
        "prefill": True,
        "prompt_length": 2,
        "state": {
            "context_lengths": [1],
            "index_shape": [1, 2, 128, 16],
            "kv_shape": [1, 2, 128, 112],
            "position": [0],
        },
    }

    config = Ws32DecoderConfig(
        geometry=_geometry(), context_capacity=2048
    )
    with pytest.raises(PlanValidationError, match="leave decode capacity"):
        build_ws32_teacher_forced_prefill_program(
            object(), config, prompt_length=2048
        )


def test_ws32_prefill_chunk_plan_always_has_a_tail_and_no_padding() -> None:
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        ws32_prefill_chunk_plan,
        ws32_repair_prompt_chunk,
    )

    assert ws32_prefill_chunk_plan(8155, 2048) == (3, 2011)
    assert ws32_prefill_chunk_plan(2034, 2048) == (0, 2034)  # 2K profile: tail only
    assert ws32_prefill_chunk_plan(8155, 512) == (15, 475)
    assert ws32_prefill_chunk_plan(127363, 2048) == (62, 387)
    assert ws32_prefill_chunk_plan(262144, 2048) == (127, 2048)
    assert ws32_prefill_chunk_plan(5, 5) == (0, 5)
    assert ws32_prefill_chunk_plan(1, 2048) == (0, 1)
    for prompt_length in (1, 2, 511, 512, 513, 8155, 127363, 262144):
        for chunk in (1, 64, 512, 2048):
            full, tail = ws32_prefill_chunk_plan(prompt_length, chunk)
            assert 1 <= tail <= chunk and full * chunk + tail == prompt_length
    with pytest.raises(PlanValidationError):
        ws32_prefill_chunk_plan(0, 2048)
    with pytest.raises(PlanValidationError):
        ws32_prefill_chunk_plan(2048, 0)
    assert ws32_repair_prompt_chunk(2048) == 2048
    assert ws32_repair_prompt_chunk(512) == 512
    assert ws32_repair_prompt_chunk(387) == 448
    assert ws32_repair_prompt_chunk(2011) == 2048
    assert ws32_repair_prompt_chunk(4096) == 2048


def test_ws32_chunked_prefill_builder_requires_capacity_and_mesh() -> None:
    from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_chunked_prefill_program

    config = Ws32DecoderConfig(geometry=_geometry(), context_capacity=2048)
    with pytest.raises(PlanValidationError, match="leave decode capacity"):
        build_ws32_chunked_prefill_program(object(), config, chunk_length=2048)
    with pytest.raises(PlanValidationError, match="leave decode capacity"):
        build_ws32_chunked_prefill_program(object(), config, chunk_length=0)


def test_ws32_main_rope_table_is_the_accepted_legacy_construction() -> None:
    """Spec §23.8: the host main-attention rotary table is the accepted GLM
    runtime's own construction, sized by the run's context capacity."""
    import ml_dtypes
    import numpy as np

    from glm_tpu.greenfield.kernels.reference.rotary import (
        build_rotary_table_host,
        rotary_table_sha256,
    )
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        WS32_MAIN_ROPE_THETA,
        build_ws32_main_rope_table,
    )

    assert WS32_MAIN_ROPE_THETA == 8_000_000.0
    config = Ws32DecoderConfig(geometry=_geometry(), context_capacity=8192)
    assert config.main_rope_table_shape == (8192, 64)
    assert config.host_main_rope_table is False
    table = build_ws32_main_rope_table(config)
    assert table.shape == (8192, 64) and table.dtype == ml_dtypes.bfloat16
    assert rotary_table_sha256(table) == rotary_table_sha256(
        build_rotary_table_host(8192, rotary_dim=64, theta=8_000_000.0)
    )
    # Row 0 is cos=1, sin=0 for every pair; rows are within BF16 of FP64 truth.
    rows = np.asarray(table, dtype=np.float32)
    assert np.array_equal(rows[0], np.concatenate([np.ones(32), np.zeros(32)]).astype(np.float32))
    frequencies = np.power(
        np.float64(8_000_000.0), -np.arange(0, 64, 2, dtype=np.float64) / np.float64(64)
    )
    positions = np.arange(8192, dtype=np.float64)
    angles = positions[:, None] * frequencies[None, :]
    truth = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
    assert np.max(np.abs(rows - truth)) <= 2 ** -8

    capacity = Ws32DecoderConfig(
        geometry=_geometry(), context_capacity=262_656, host_main_rope_table=True
    )
    assert capacity.main_rope_table_shape == (262_656, 64)
    assert capacity.host_main_rope_table is True
    with pytest.raises(PlanValidationError, match="host main-rotary table flag"):
        Ws32DecoderConfig(
            geometry=_geometry(), context_capacity=8192, host_main_rope_table=1
        )


def test_ws32_programs_accept_the_optional_main_rope_table_input() -> None:
    """The table is one extra replicated trailing input; every program builder
    accepts it only when the config declares it (spec §23.8)."""
    program = r'''
import json
from dataclasses import replace

import jax
import numpy as np
from jax.sharding import Mesh

from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig,
    build_ws32_chunked_prefill_program,
    build_ws32_decoder_program,
    build_ws32_main_rope_table,
    build_ws32_teacher_forced_prefill_program,
)
from glm_tpu.greenfield.types import ModelGeometry

source = json.loads(open("configs/glm-5.2-fp8-config.json").read())
base = ModelGeometry.from_hf_config(source)
geometry = replace(
    base,
    num_layers=1,
    first_dense_layers=1,
    hidden_size=128,
    dense_intermediate_size=256,
    num_routed_experts=16,
    routed_top_k=4,
    moe_intermediate_size=128,
    dsa_top_k=128,
    dsa_indexer_heads=8,
    dsa_indexer_head_dim=16,
    attention_heads=8,
    kv_heads=1,
    kv_lora_rank=32,
    q_lora_rank=64,
    qk_nope_head_dim=16,
    qk_rope_head_dim=16,
    v_head_dim=16,
    max_position_embeddings=256,
    vocab_size=64,
    fp8_block_shape=(16, 16),
    mlp_layer_types=("dense",),
    indexer_types=("full",),
)
mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(8, 4), ("expert", "feature"))
out = {}
for enabled in (False, True):
    config = Ws32DecoderConfig(
        geometry=geometry,
        context_capacity=256,
        logical_page_size=128,
        packed_cache_width=112,
        sparse_segment_block=128,
        host_main_rope_table=enabled,
    )
    decoder = build_ws32_decoder_program(mesh, config, sparse_attention_interpret=True, linear_interpret=True)
    prefill = build_ws32_teacher_forced_prefill_program(mesh, config, prompt_length=2, sparse_attention_interpret=True, linear_interpret=True)
    chunk = build_ws32_chunked_prefill_program(mesh, config, chunk_length=2, sparse_attention_interpret=True, linear_interpret=True)
    table = build_ws32_main_rope_table(config)
    out[str(enabled)] = {
        "callables": all(callable(f) for f in (decoder.execute, decoder.observe, prefill.execute, chunk.execute)),
        "table_shape": list(table.shape),
        "table_dtype": str(table.dtype),
    }
print(json.dumps(out))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "False": {"callables": True, "table_shape": [256, 16], "table_dtype": "bfloat16"},
        "True": {"callables": True, "table_shape": [256, 16], "table_dtype": "bfloat16"},
    }
