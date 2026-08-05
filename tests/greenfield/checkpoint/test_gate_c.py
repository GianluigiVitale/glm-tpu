from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.checkpoint import (
    build_gate_c_layout,
    inspect_gate_c_checkpoint,
    pack_gate_c_checkpoint,
    validate_gate_c_layout,
    validate_gate_c_layout_bindings,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from tests.greenfield.checkpoint.test_stream_pack import fixture


def _mapping_hash(value: dict, field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(field, None)
    raw = json.dumps(
        unhashed,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(raw).hexdigest()


def _oracle(source_root: Path, parent: dict) -> dict:
    from safetensors import safe_open

    name = "model.layers.0.synthetic.weight"
    placement = next(
        record
        for record in parent["placements"]
        if record["source"]["name"] == name
    )
    source = placement["source"]
    with safe_open(
        source_root / source["filename"], framework="pt", device="cpu"
    ) as handle:
        digest = sha256(handle.get_tensor(name).numpy().tobytes()).hexdigest()
    value = {
        "artifact_kind": "greenfield_gate_c_oracle",
        "consumer_layer": 1,
        "format_version": 1,
        "model_id": parent["source"]["model_id"],
        "producer_layer": 0,
        "source_revision": parent["source"]["revision"],
        "source_tensors": [
            {
                "byte_count": source["byte_count"],
                "dtype": source["dtype"],
                "name": name,
                "sha256": digest,
                "shape": source["shape"],
                "source_shard": source["filename"],
            }
        ],
        "source_uri": parent["source"]["uri"],
    }
    value["manifest_sha256"] = _mapping_hash(value, "manifest_sha256")
    return value


def test_gate_c_pack_is_exact_hashed_parent_subset(tmp_path: Path) -> None:
    import torch
    from safetensors import safe_open

    source_root, parent = fixture(tmp_path)
    oracle = _oracle(source_root, parent)
    layout = build_gate_c_layout(
        parent_layout=parent,
        oracle_manifest=oracle,
        code_hash="c" * 40,
    )
    validate_gate_c_layout_bindings(
        layout,
        parent_layout=parent,
        oracle_manifest=oracle,
    )
    assert layout["parent_layout_manifest_sha256"] == parent["manifest_sha256"]
    assert layout["oracle_manifest_sha256"] == oracle["manifest_sha256"]
    assert layout["source"]["payload_bytes"] == 128
    assert layout["packed_payload_bytes"] == 128
    assert len(layout["destination_files"]) == 4
    assert layout["placements"] == [
        next(
            placement
            for placement in parent["placements"]
            if placement["source"]["name"]
            == "model.layers.0.synthetic.weight"
        )
    ]

    output = tmp_path / "gate-c-pack"
    manifest = pack_gate_c_checkpoint(
        layout=layout,
        source_root=source_root,
        output_dir=output,
        chunk_bytes=64,
    )
    assert inspect_gate_c_checkpoint(
        output,
        parent_layout=parent,
        oracle_manifest=oracle,
    ) == manifest
    assert manifest["file_count"] == 4
    for slot in range(4):
        path = output / (
            f"base_decoder/stage_00/device_slot_{slot:02d}.safetensors"
        )
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual = handle.get_tensor("model.layers.0.synthetic.weight")
        expected = torch.arange(32, dtype=torch.float32).reshape(4, 8)[
            :, slot * 2 : (slot + 1) * 2
        ]
        assert torch.equal(actual, expected)


def test_gate_c_layout_refuses_oracle_or_ownership_drift(tmp_path: Path) -> None:
    source_root, parent = fixture(tmp_path)
    oracle = _oracle(source_root, parent)
    drifted_oracle = deepcopy(oracle)
    drifted_oracle["source_tensors"][0]["shape"] = [8, 4]
    drifted_oracle["manifest_sha256"] = _mapping_hash(
        drifted_oracle, "manifest_sha256"
    )
    with pytest.raises(CheckpointValidationError, match="metadata disagrees"):
        build_gate_c_layout(
            parent_layout=parent,
            oracle_manifest=drifted_oracle,
            code_hash="c" * 40,
        )

    layout = build_gate_c_layout(
        parent_layout=parent,
        oracle_manifest=oracle,
        code_hash="c" * 40,
    )
    layout["placements"][0]["destinations"][0]["device_slot"] = 3
    layout["manifest_sha256"] = _mapping_hash(layout, "manifest_sha256")
    with pytest.raises(CheckpointValidationError, match="destination identity"):
        validate_gate_c_layout(layout)


def test_gate_c_inspector_refuses_file_corruption(tmp_path: Path) -> None:
    source_root, parent = fixture(tmp_path)
    oracle = _oracle(source_root, parent)
    layout = build_gate_c_layout(
        parent_layout=parent,
        oracle_manifest=oracle,
        code_hash="c" * 40,
    )
    output = tmp_path / "gate-c-pack"
    manifest = pack_gate_c_checkpoint(
        layout=layout,
        source_root=source_root,
        output_dir=output,
        chunk_bytes=64,
    )
    path = output / manifest["files"][0]["filename"]
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(CheckpointValidationError, match="size mismatch"):
        inspect_gate_c_checkpoint(output)
