from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.checkpoint.one_layer import (
    OneLayerPackConfig,
    expected_source_names,
    inspect_one_layer_artifact,
    pack_one_layer_moe,
)


HASH = "a" * 40


def tiny_config(source_root: Path, output_dir: Path) -> OneLayerPackConfig:
    return OneLayerPackConfig(
        source_root=source_root,
        output_dir=output_dir,
        source_uri="gs://driftbench-dsv4-uc/models/tiny",
        source_revision="test-fixture-v1",
        code_hash=HASH,
        topology_hash="b" * 64,
        plan_group_hash="c" * 64,
        layer=3,
        hidden_size=8,
        intermediate_size=4,
        num_experts=8,
        top_k=2,
        stage_size=4,
        fp8_block_shape=(2, 2),
    )


def write_tiny_source(config: OneLayerPackConfig) -> None:
    import torch
    from safetensors.torch import save_file

    config.source_root.mkdir()
    prefix = config.layer_prefix
    tensors = {}
    for expert in range(config.num_experts):
        for projection in ("gate_proj", "up_proj"):
            base = f"{prefix}.experts.{expert}.{projection}"
            tensors[f"{base}.weight"] = torch.full(
                (config.intermediate_size, config.hidden_size),
                float(expert + 1),
                dtype=torch.float8_e4m3fn,
            )
            tensors[f"{base}.weight_scale_inv"] = torch.full(
                (
                    config.intermediate_size // config.fp8_block_shape[0],
                    config.hidden_size // config.fp8_block_shape[1],
                ),
                float(expert + 1),
                dtype=torch.float32,
            )
        base = f"{prefix}.experts.{expert}.down_proj"
        tensors[f"{base}.weight"] = torch.full(
            (config.hidden_size, config.intermediate_size),
            float(expert + 1),
            dtype=torch.float8_e4m3fn,
        )
        tensors[f"{base}.weight_scale_inv"] = torch.full(
            (
                config.hidden_size // config.fp8_block_shape[0],
                config.intermediate_size // config.fp8_block_shape[1],
            ),
            float(expert + 1),
            dtype=torch.float32,
        )
    tensors[f"{prefix}.gate.weight"] = torch.arange(
        config.num_experts * config.hidden_size, dtype=torch.bfloat16
    ).reshape(config.num_experts, config.hidden_size)
    tensors[f"{prefix}.gate.e_score_correction_bias"] = torch.arange(
        config.num_experts, dtype=torch.float32
    )
    for projection in ("gate_proj", "up_proj"):
        base = f"{prefix}.shared_experts.{projection}"
        tensors[f"{base}.weight"] = torch.arange(
            config.intermediate_size * config.hidden_size,
            dtype=torch.float32,
        ).reshape(config.intermediate_size, config.hidden_size).to(
            torch.float8_e4m3fn
        )
        tensors[f"{base}.weight_scale_inv"] = torch.arange(
            (
                config.intermediate_size
                // config.fp8_block_shape[0]
                * config.hidden_size
                // config.fp8_block_shape[1]
            ),
            dtype=torch.float32,
        ).reshape(
            config.intermediate_size // config.fp8_block_shape[0],
            config.hidden_size // config.fp8_block_shape[1],
        )
    base = f"{prefix}.shared_experts.down_proj"
    tensors[f"{base}.weight"] = torch.arange(
        config.hidden_size * config.intermediate_size,
        dtype=torch.float32,
    ).reshape(config.hidden_size, config.intermediate_size).to(
        torch.float8_e4m3fn
    )
    tensors[f"{base}.weight_scale_inv"] = torch.arange(
        (
            config.hidden_size
            // config.fp8_block_shape[0]
            * config.intermediate_size
            // config.fp8_block_shape[1]
        ),
        dtype=torch.float32,
    ).reshape(
        config.hidden_size // config.fp8_block_shape[0],
        config.intermediate_size // config.fp8_block_shape[1],
    )

    names = sorted(tensors)
    first = {name: tensors[name] for index, name in enumerate(names) if index % 2 == 0}
    second = {name: tensors[name] for index, name in enumerate(names) if index % 2 == 1}
    save_file(first, config.source_root / "model-00001.safetensors")
    save_file(second, config.source_root / "model-00002.safetensors")
    weight_map = {
        name: (
            "model-00001.safetensors"
            if index % 2 == 0
            else "model-00002.safetensors"
        )
        for index, name in enumerate(names)
    }
    (config.source_root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, sort_keys=True)
    )


def test_tiny_one_layer_pack_reconciles_and_shards(tmp_path: Path) -> None:
    from safetensors import safe_open

    config = tiny_config(tmp_path / "source", tmp_path / "packed")
    write_tiny_source(config)
    manifest = pack_one_layer_moe(config)
    assert len(manifest["source_leaves"]) == 8 * 6 + 2 + 6
    assert len(manifest["files"]) == 4
    assert inspect_one_layer_artifact(config.output_dir) == manifest
    assert [record["tensors"][0]["ownership"]["device_slot"] for record in manifest["files"]] == [0, 1, 2, 3]
    with safe_open(
        config.output_dir / "device_slot_02.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        assert handle.get_slice("expert_gate").get_shape() == [2, 4, 8]
        assert handle.get_slice("shared_gate").get_shape() == [1, 8]
        assert handle.get_slice("shared_down").get_shape() == [8, 1]
        assert handle.get_slice("router_weight").get_shape() == [8, 8]
        expert_gate = handle.get_tensor("expert_gate").float()
        assert expert_gate[:, 0, 0].tolist() == [5.0, 6.0]


def test_pack_refuses_incomplete_source_leaf_set(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "source", tmp_path / "packed")
    write_tiny_source(config)
    index_path = config.source_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop(expected_source_names(config)[0])
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="leaf set mismatch"):
        pack_one_layer_moe(config)


def test_inspector_refuses_packed_file_corruption(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "source", tmp_path / "packed")
    write_tiny_source(config)
    manifest = pack_one_layer_moe(config)
    path = config.output_dir / manifest["files"][0]["filename"]
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        inspect_one_layer_artifact(config.output_dir)


def test_config_refuses_wrong_bucket_or_nondivisible_layout(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "source", tmp_path / "packed")
    with pytest.raises(ValueError, match="approved"):
        replace(config, source_uri="gs://wrong-bucket/model")
    with pytest.raises(ValueError, match="divide evenly"):
        replace(config, num_experts=7)
    with pytest.raises(ValueError, match="Git object"):
        replace(config, code_hash="not-a-commit")
