from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.partitioning.source_inventory import (
    inspect_source_inventory,
    read_source_inventory,
    write_source_inventory,
)


def write_source(root: Path) -> None:
    import torch
    from safetensors.torch import save_file

    root.mkdir()
    first = {
        "model.layers.0.proj.weight": torch.arange(
            6, dtype=torch.bfloat16
        ).reshape(2, 3),
    }
    second = {
        "lm_head.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
    }
    save_file(first, root / "model-00001.safetensors")
    save_file(second, root / "model-00002.safetensors")
    (root / "config.json").write_text(json.dumps({"model_type": "tiny"}))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 28},
                "weight_map": {
                    "lm_head.weight": "model-00002.safetensors",
                    "model.layers.0.proj.weight": "model-00001.safetensors",
                },
            },
            sort_keys=True,
        )
    )


def inventory(root: Path):
    return read_source_inventory(
        root,
        model_id="tiny/model",
        source_revision="fixture-v1",
    )


def test_inventory_reconciles_headers_index_and_payload_without_loading(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    write_source(root)
    result = inventory(root)
    assert len(result.files) == 2
    assert len(result.tensors) == 2
    assert result.payload_bytes == 28
    assert result.header_bytes == result.file_bytes - 28
    assert result.tensors[0].name == "lm_head.weight"
    assert result.tensors[1].layer_id == 0
    assert result.summary_dict()["layer_bytes"] == {"0": 12}
    assert result.summary_dict()["non_layer_bytes"] == 16
    assert len(result.inventory_sha256) == 64


def test_inventory_manifest_is_append_only_and_self_authenticating(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    output = tmp_path / "inventory.json"
    write_source(root)
    result = inventory(root)
    write_source_inventory(result, output)
    assert inspect_source_inventory(output) == result
    with pytest.raises(CheckpointValidationError, match="overwrite"):
        write_source_inventory(result, output)

    decoded = json.loads(output.read_text())
    decoded["source_revision"] = "tampered"
    output.write_text(json.dumps(decoded))
    with pytest.raises(CheckpointValidationError, match="SHA-256 mismatch"):
        inspect_source_inventory(output)


def test_layer_worker_authenticates_canonical_inventory_not_file_digest(
    tmp_path: Path,
) -> None:
    from scripts.greenfield.ws32_compile_originals import authenticated_inventory

    root = tmp_path / "source"
    output = tmp_path / "inventory.json"
    write_source(root)
    original = inventory(root)
    for indent in (None, 2):
        output.write_text(json.dumps(original.to_dict(), indent=indent))
        raw_digest = sha256(output.read_bytes()).hexdigest()
        assert raw_digest != original.inventory_sha256
        assert authenticated_inventory(output, original.inventory_sha256) == original
        with pytest.raises(ValueError, match="canonical hash drifted"):
            authenticated_inventory(output, raw_digest)

    edited = original.to_dict()
    edited["source_revision"] = "tampered"
    output.write_text(json.dumps(edited))
    with pytest.raises(CheckpointValidationError, match="SHA-256 mismatch"):
        authenticated_inventory(output, original.inventory_sha256)

    # A self-consistent replacement must still be refused against the original pin.
    replacement = replace(original, source_revision="replacement")
    output.write_text(json.dumps(replacement.to_dict()))
    with pytest.raises(ValueError, match="canonical hash drifted"):
        authenticated_inventory(output, original.inventory_sha256)


def test_inventory_refuses_index_header_file_disagreement(tmp_path: Path) -> None:
    root = tmp_path / "source"
    write_source(root)
    path = root / "model.safetensors.index.json"
    decoded = json.loads(path.read_text())
    decoded["weight_map"]["lm_head.weight"] = "model-00001.safetensors"
    path.write_text(json.dumps(decoded))
    with pytest.raises(CheckpointValidationError, match="disagrees with weight_map"):
        inventory(root)


def test_inventory_refuses_declared_byte_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "source"
    write_source(root)
    path = root / "model.safetensors.index.json"
    decoded = json.loads(path.read_text())
    decoded["metadata"]["total_size"] += 1
    path.write_text(json.dumps(decoded))
    with pytest.raises(CheckpointValidationError, match="payload totals"):
        inventory(root)


def test_inventory_refuses_trailing_or_unsafe_source_file(tmp_path: Path) -> None:
    root = tmp_path / "source"
    write_source(root)
    with (root / "model-00001.safetensors").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(CheckpointValidationError, match="size does not match"):
        inventory(root)

    root = tmp_path / "unsafe"
    write_source(root)
    path = root / "model.safetensors.index.json"
    decoded = json.loads(path.read_text())
    decoded["weight_map"]["lm_head.weight"] = "../escape.safetensors"
    path.write_text(json.dumps(decoded))
    with pytest.raises(CheckpointValidationError, match="unsafe"):
        inventory(root)
