from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference import (
    IndexShareSchedule,
    MlaNumericalContract,
    SelectedKvSegment,
    SelectedPositions,
    dsa_scores,
    exact_topk,
    produce_index_share_state,
    residual_add,
    resolve_index_share_layer,
    silu,
    sparse_mla_attention,
)
from glm_tpu.greenfield.validation import (
    GateCOracleConfig,
    capture_gate_c_oracle,
    inspect_gate_c_oracle,
)


HASH = "a" * 40


def tiny_config(source_root: Path, output_dir: Path) -> GateCOracleConfig:
    return GateCOracleConfig(
        source_root=source_root,
        output_dir=output_dir,
        source_uri="gs://driftbench-dsv4-uc/models/tiny-gate-c",
        source_revision="tiny-gate-c-v1",
        code_hash=HASH,
        legacy_code_hash="b" * 40,
        vllm_code_hash="c" * 40,
        reference_source_hashes=(("reference.py", "d" * 64),),
        allowed_source_shards=(
            "model-00001.safetensors",
            "model-00002.safetensors",
            "model-00003.safetensors",
        ),
        producer_layer=0,
        consumer_layer=1,
        hidden_size=8,
        dense_intermediate_size=4,
        q_lora_rank=4,
        num_attention_heads=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        qk_head_dim=4,
        kv_lora_rank=4,
        v_head_dim=2,
        indexer_heads=2,
        indexer_head_dim=4,
        indexer_rotary_dim=2,
        top_k=4,
        context_length=6,
        packed_cache_width=8,
        fp8_block_shape=(2, 2),
        require_current_selected=False,
    )


def write_tiny_source(config: GateCOracleConfig) -> None:
    import torch
    from safetensors.torch import save_file

    config.source_root.mkdir()
    producer = config.producer_prefix
    consumer = config.consumer_prefix
    tensors: dict[str, torch.Tensor] = {}

    def bf16(name: str, shape: tuple[int, ...], *, bias: bool = False) -> None:
        if bias:
            tensor = torch.zeros(shape, dtype=torch.bfloat16)
        elif name.endswith("weight") and len(shape) == 1:
            tensor = torch.linspace(0.75, 1.25, shape[0], dtype=torch.bfloat16)
        else:
            values = torch.arange(np.prod(shape), dtype=torch.float32)
            tensor = ((values.reshape(shape) % 7) - 3).to(torch.bfloat16) / 8
        tensors[name] = tensor

    def fp8_pair(name: str, shape: tuple[int, int], salt: int) -> None:
        values = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        tensors[f"{name}.weight"] = (
            ((values + salt) % 5) - 2
        ).to(torch.float8_e4m3fn)
        scale_shape = tuple(
            (dimension + block - 1) // block
            for dimension, block in zip(
                shape, config.fp8_block_shape, strict=True
            )
        )
        tensors[f"{name}.weight_scale_inv"] = torch.full(
            scale_shape, 0.125 + salt / 128, dtype=torch.float32
        )

    bf16(f"{producer}.input_layernorm.weight", (config.hidden_size,))
    bf16(f"{producer}.post_attention_layernorm.weight", (config.hidden_size,))
    fp8_pair(
        f"{producer}.mlp.gate_proj",
        (config.dense_intermediate_size, config.hidden_size),
        1,
    )
    fp8_pair(
        f"{producer}.mlp.up_proj",
        (config.dense_intermediate_size, config.hidden_size),
        2,
    )
    fp8_pair(
        f"{producer}.mlp.down_proj",
        (config.hidden_size, config.dense_intermediate_size),
        3,
    )
    fp8_pair(
        f"{producer}.self_attn.q_a_proj",
        (config.q_lora_rank, config.hidden_size),
        4,
    )
    bf16(
        f"{producer}.self_attn.q_a_layernorm.weight",
        (config.q_lora_rank,),
    )
    indexer = f"{producer}.self_attn.indexer"
    fp8_pair(
        f"{indexer}.wq_b",
        (config.indexer_heads * config.indexer_head_dim, config.q_lora_rank),
        5,
    )
    fp8_pair(
        f"{indexer}.wk",
        (config.indexer_head_dim, config.hidden_size),
        6,
    )
    bf16(f"{indexer}.k_norm.weight", (config.indexer_head_dim,))
    bf16(
        f"{indexer}.k_norm.bias", (config.indexer_head_dim,), bias=True
    )
    bf16(
        f"{indexer}.weights_proj.weight",
        (config.indexer_heads, config.hidden_size),
    )

    bf16(f"{consumer}.input_layernorm.weight", (config.hidden_size,))
    attention = f"{consumer}.self_attn"
    fp8_pair(
        f"{attention}.q_a_proj",
        (config.q_lora_rank, config.hidden_size),
        7,
    )
    bf16(f"{attention}.q_a_layernorm.weight", (config.q_lora_rank,))
    fp8_pair(
        f"{attention}.q_b_proj",
        (
            config.num_attention_heads * config.qk_head_dim,
            config.q_lora_rank,
        ),
        8,
    )
    fp8_pair(
        f"{attention}.kv_a_proj_with_mqa",
        (
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
        ),
        9,
    )
    bf16(f"{attention}.kv_a_layernorm.weight", (config.kv_lora_rank,))
    fp8_pair(
        f"{attention}.kv_b_proj",
        (
            config.num_attention_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
        ),
        10,
    )
    fp8_pair(
        f"{attention}.o_proj",
        (config.hidden_size, config.num_attention_heads * config.v_head_dim),
        11,
    )

    names = sorted(tensors)
    weight_map = {}
    for shard_index, filename in enumerate(config.allowed_source_shards):
        shard = {
            name: tensors[name]
            for index, name in enumerate(names)
            if index % len(config.allowed_source_shards) == shard_index
        }
        save_file(shard, config.source_root / filename)
        weight_map.update({name: filename for name in shard})
    (config.source_root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, sort_keys=True)
    )


def test_capture_is_independent_exact_and_reinspectable(tmp_path: Path) -> None:
    from safetensors import safe_open
    import torch

    config = tiny_config(tmp_path / "source", tmp_path / "oracle")
    write_tiny_source(config)
    manifest = capture_gate_c_oracle(config)
    assert inspect_gate_c_oracle(config.output_dir) == manifest
    assert len(manifest["source_tensors"]) == 31
    assert manifest["cases"]["full_dsa"]["selected_count"] == 4
    assert manifest["cases"]["index_share"]["payload_shape"] == [1, 4]
    assert manifest["cases"]["index_share"]["payload_byte_count"] == 16
    assert manifest["cases"]["index_share"]["producer_layer"] == 0
    assert manifest["cases"]["index_share"]["consumer_layer"] == 1
    with safe_open(
        config.output_dir / "oracle.safetensors", framework="pt", device="cpu"
    ) as handle:
        producer = handle.get_tensor("producer_selected_positions")
        consumer = handle.get_tensor("consumer_selected_positions")
        canonical = handle.get_tensor("consumer_attention_positions")
        cache = handle.get_tensor("consumer_cache")
        current = handle.get_tensor("consumer_current_cache_row")
        assert producer.dtype == consumer.dtype
        assert producer.shape == (1, 4)
        assert np.array_equal(producer.numpy(), consumer.numpy())
        assert np.array_equal(
            canonical.numpy(), np.sort(producer.numpy(), axis=-1)
        )
        assert np.array_equal(
            cache[-1:].view(torch.uint16).numpy(),
            current.view(torch.uint16).numpy(),
        )
        assert handle.get_slice("producer_dense_output").get_shape() == [1, 8]
        assert handle.get_slice("consumer_attention_output").get_dtype() == "BF16"


def test_captured_arithmetic_matches_greenfield_reference_contract(
    tmp_path: Path,
) -> None:
    import jax.numpy as jnp
    from safetensors import safe_open
    import torch

    config = tiny_config(tmp_path / "source", tmp_path / "oracle")
    write_tiny_source(config)
    capture_gate_c_oracle(config)
    with safe_open(
        config.output_dir / "oracle.safetensors", framework="pt", device="cpu"
    ) as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}

    scores = dsa_scores(
        jnp.asarray(tensors["producer_index_query"].numpy()),
        jnp.asarray(tensors["producer_index_keys"].numpy()),
        jnp.asarray(tensors["producer_index_head_weights"].numpy()),
    )
    np.testing.assert_allclose(
        np.asarray(scores),
        tensors["producer_index_scores"].numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    selected = exact_topk(
        scores,
        jnp.asarray([config.context_length], dtype=jnp.int32),
        top_k=config.top_k,
    )
    np.testing.assert_array_equal(
        np.asarray(selected.positions),
        tensors["producer_selected_positions"].numpy(),
    )

    attention_contract = MlaNumericalContract(
        num_heads=config.num_attention_heads,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        qk_head_dim=config.qk_head_dim,
        packed_cache_width=config.packed_cache_width,
        top_k=config.top_k,
    )
    attention = sparse_mla_attention(
        jnp.asarray(tensors["consumer_q_absorbed"].float().numpy()).astype(
            jnp.bfloat16
        ),
        jnp.asarray(tensors["consumer_q_rope"].float().numpy()).astype(
            jnp.bfloat16
        ),
        SelectedKvSegment(
            jnp.asarray(tensors["consumer_selected_cache"].float().numpy()).astype(
                jnp.bfloat16
            ),
            jnp.asarray(tensors["consumer_attention_positions"].numpy()),
            jnp.asarray([config.top_k], dtype=jnp.int32),
            jnp.asarray([True]),
        ),
        contract=attention_contract,
    )
    np.testing.assert_array_equal(
        np.asarray(attention.output).view(np.uint16),
        tensors["consumer_attended_latent"].view(torch.uint16).numpy(),
    )
    np.testing.assert_allclose(
        np.asarray(attention.logsumexp),
        tensors["consumer_attention_lse"].numpy(),
        rtol=2e-6,
        atol=2e-6,
    )

    activated = silu(
        jnp.asarray(tensors["producer_dense_gate"].float().numpy()).astype(
            jnp.bfloat16
        )
    ) * jnp.asarray(tensors["producer_dense_up"].float().numpy()).astype(
        jnp.bfloat16
    )
    np.testing.assert_allclose(
        np.asarray(activated, dtype=np.float32),
        tensors["producer_dense_activated"].float().numpy(),
        rtol=1e-2,
        atol=1e-3,
    )
    dense_output = residual_add(
        jnp.asarray(tensors["producer_dense_residual"].float().numpy()).astype(
            jnp.bfloat16
        ),
        jnp.asarray(tensors["producer_dense_update"].float().numpy()).astype(
            jnp.bfloat16
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(dense_output).view(np.uint16),
        tensors["producer_dense_output"].view(torch.uint16).numpy(),
    )

    score_ordered = SelectedPositions(
        jnp.asarray(tensors["producer_selected_positions"].numpy()),
        jnp.asarray([config.top_k], dtype=jnp.int32),
    )
    state = produce_index_share_state(score_ordered, source_layer=0)
    reused, _ = resolve_index_share_layer(
        IndexShareSchedule(("full", "shared"), 2),
        1,
        fresh_selection=None,
        shared_state=state,
    )
    np.testing.assert_array_equal(
        np.asarray(reused.positions),
        tensors["consumer_selected_positions"].numpy(),
    )


def test_capture_refuses_existing_destination_or_wrong_source_shard(
    tmp_path: Path,
) -> None:
    config = tiny_config(tmp_path / "source", tmp_path / "oracle")
    write_tiny_source(config)
    capture_gate_c_oracle(config)
    with pytest.raises(FileExistsError, match="append-only"):
        capture_gate_c_oracle(config)
    wrong = replace(
        config,
        output_dir=tmp_path / "wrong",
        allowed_source_shards=("missing.safetensors",),
    )
    with pytest.raises(ValueError, match="escaped allowed shards"):
        capture_gate_c_oracle(wrong)


def test_inspector_refuses_manifest_or_file_corruption(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "source", tmp_path / "oracle")
    write_tiny_source(config)
    capture_gate_c_oracle(config)
    manifest_path = config.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["producer_layer"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest checksum"):
        inspect_gate_c_oracle(config.output_dir)

    second = replace(config, output_dir=tmp_path / "oracle-two")
    capture_gate_c_oracle(second)
    with (second.output_dir / "oracle.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="file size"):
        inspect_gate_c_oracle(second.output_dir)


def test_config_refuses_nonreal_geometry_or_bad_provenance(
    tmp_path: Path,
) -> None:
    config = tiny_config(tmp_path / "source", tmp_path / "oracle")
    with pytest.raises(ValueError, match="approved"):
        replace(config, source_uri="gs://wrong/model")
    with pytest.raises(ValueError, match="greater than top_k"):
        replace(config, context_length=config.top_k)
    with pytest.raises(ValueError, match="immediately follow"):
        replace(config, consumer_layer=3)
    with pytest.raises(ValueError, match="reference_source_hashes"):
        replace(config, reference_source_hashes=())
