from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.validation.one_layer_oracle import (
    OneLayerOracleConfig,
    capture_one_layer_oracle,
    inspect_one_layer_oracle,
)
from tests.greenfield.checkpoint.test_one_layer import (
    tiny_config,
    write_tiny_source,
)


HASH = "a" * 40


def oracle_config(source_root: Path, output_dir: Path) -> OneLayerOracleConfig:
    return OneLayerOracleConfig(
        source_root=source_root,
        output_dir=output_dir,
        source_uri="gs://driftbench-dsv4-uc/models/tiny",
        source_revision="test-source-v1",
        code_hash=HASH,
        legacy_code_hash="b" * 40,
        legacy_source_hashes=(("legacy/moe.py", "c" * 64),),
        vllm_code_hash="d" * 40,
        vllm_source_hashes=(("vllm/router.py", "e" * 64),),
        allowed_source_shards=("model-00001.safetensors", "model-00002.safetensors"),
        layer=3,
        hidden_size=8,
        intermediate_size=4,
        num_experts=8,
        top_k=2,
        stage_size=4,
        concentrated_slot=2,
        min_normal_stage_slots=1,
        fp8_block_shape=(2, 2),
    )


def write_source(source_root: Path) -> None:
    pack_config = tiny_config(source_root, source_root.parent / "unused-pack")
    write_tiny_source(pack_config)


def test_capture_is_independent_bounded_and_reinspectable(tmp_path: Path) -> None:
    from safetensors import safe_open

    source = tmp_path / "source"
    write_source(source)
    config = oracle_config(source, tmp_path / "oracle")
    manifest = capture_one_layer_oracle(config)
    assert inspect_one_layer_oracle(config.output_dir) == manifest
    assert set(manifest["cases"]["concentrated"]["route_indices"]) == {4, 5}
    assert manifest["cases"]["concentrated"]["stage_slot"] == 2
    assert len(manifest["cases"]["normal"]["route_indices"]) == 2
    assert len(manifest["source_tensors"]) < 8 * 6 + 8
    assert {record["source_shard"] for record in manifest["source_tensors"]} <= {
        "model-00001.safetensors",
        "model-00002.safetensors",
    }
    with safe_open(
        config.output_dir / "oracle.safetensors", framework="pt", device="cpu"
    ) as handle:
        assert handle.get_slice("hidden_states").get_shape() == [1, 8]
        assert handle.get_slice("normal_expert_outputs").get_shape() == [2, 8]
        assert handle.get_slice("concentrated_output").get_dtype() == "BF16"


def test_capture_refuses_existing_destination_and_wrong_shard(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    config = oracle_config(source, tmp_path / "oracle")
    capture_one_layer_oracle(config)
    with pytest.raises(FileExistsError, match="append-only"):
        capture_one_layer_oracle(config)
    wrong = replace(
        config,
        output_dir=tmp_path / "wrong-shard",
        allowed_source_shards=("not-the-source.safetensors",),
    )
    with pytest.raises(ValueError, match="escaped allowed shards"):
        capture_one_layer_oracle(wrong)


def test_inspector_refuses_manifest_or_file_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source(source)
    config = oracle_config(source, tmp_path / "oracle")
    capture_one_layer_oracle(config)
    manifest_path = config.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layer"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest checksum"):
        inspect_one_layer_oracle(config.output_dir)

    second = oracle_config(source, tmp_path / "oracle-two")
    capture_one_layer_oracle(second)
    with (second.output_dir / "oracle.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="file size"):
        inspect_one_layer_oracle(second.output_dir)


def test_config_refuses_unapproved_source_or_bad_provenance(tmp_path: Path) -> None:
    config = oracle_config(tmp_path / "source", tmp_path / "output")
    with pytest.raises(ValueError, match="approved"):
        replace(config, source_uri="gs://wrong/model")
    with pytest.raises(ValueError, match="legacy_source_hashes"):
        replace(config, legacy_source_hashes=())
