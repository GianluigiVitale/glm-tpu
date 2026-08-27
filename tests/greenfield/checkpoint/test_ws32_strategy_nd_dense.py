from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import (
    strategy_nd_dense_tensor_names,
    verify_ws32_strategy_nd_dense_overlay,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from scripts.greenfield.pack_ws32_strategy_nd_dense_overlay import (
    _model_rank_values,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_overlay(root: Path, *, replica_drift: bool = False) -> tuple[str, str, str]:
    contracts = (
        ((4, 6144, 768), "U8"),
        ((4, 48, 768), "F32"),
        ((4, 384, 1536), "U8"),
        ((4, 3, 1536), "F32"),
    )
    files = []
    for layer in range(3):
        names = strategy_nd_dense_tensor_names(layer)
        for expert in range(8):
            for feature in range(4):
                tensors = {}
                for index, (name, (shape, dtype)) in enumerate(
                    zip(names, contracts, strict=True)
                ):
                    replica_feature = (
                        feature if replica_drift and index == 0 else 0
                    )
                    tensors[name] = {
                        "byte_count": 1,
                        "dtype": dtype,
                        "sha256": sha256(
                            f"{layer}:{expert}:{replica_feature}:{index}".encode()
                        ).hexdigest(),
                        "shape": list(shape),
                    }
                files.append(
                    {
                        "byte_count": 1,
                        "expert_coordinate": expert,
                        "feature_coordinate": feature,
                        "filename": (
                            f"layer_{layer:02d}/expert_{expert:02d}/"
                            f"feature_{feature:02d}.safetensors"
                        ),
                        "layer_id": layer,
                        "model_ranks": list(
                            range(expert * 4, expert * 4 + 4)
                        ),
                        "sha256": "1" * 64,
                        "tensors": tensors,
                    }
                )
    manifest = {
        "artifact_kind": "greenfield_ws32_strategy_nd_dense_overlay",
        "code_hash": "2" * 40,
        "dense_layer_ids": [0, 1, 2],
        "file_count": 96,
        "files": files,
        "format_version": 1,
        "plan_id": "WS32_2D",
        "total_bytes": 96,
    }
    manifest["manifest_sha256"] = sha256(_canonical(manifest)).hexdigest()
    root.mkdir()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success = {
        "artifact_kind": "greenfield_ws32_strategy_nd_dense_overlay_success",
        "manifest_sha256": manifest["manifest_sha256"],
    }
    success["success_sha256"] = sha256(_canonical(success)).hexdigest()
    success_path = root / "SUCCESS"
    success_path.write_text(
        json.dumps(success, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest["manifest_sha256"], _sha(manifest_path), _sha(success_path)


def test_ws32_strategy_nd_dense_names_and_model_rank_order() -> None:
    assert strategy_nd_dense_tensor_names(2)[0] == (
        "model.layers.2.mlp.strategy_nd.merged_gate_up.weight_bits_in_out"
    )
    with pytest.raises(CheckpointValidationError, match="one of 0, 1, 2"):
        strategy_nd_dense_tensor_names(3)

    sources = {
        (owner, layer, "tensor"): np.arange(
            owner * 80 + layer * 1000,
            owner * 80 + layer * 1000 + 80,
        ).reshape(8, 10)
        for owner in range(4)
        for layer in range(3)
    }
    selected = _model_rank_values(
        sources,
        layer=2,
        suffix="tensor",
        ranks=(7, 8, 15, 16, 31),
    )
    assert selected[:, 0].tolist() == [2070, 2080, 2150, 2160, 2310]


def test_ws32_strategy_nd_dense_overlay_manifest_is_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "valid"
    manifest_sha, manifest_file_sha, success_file_sha = _write_overlay(root)
    verified = verify_ws32_strategy_nd_dense_overlay(
        root,
        expected_manifest_sha256=manifest_sha,
        expected_manifest_file_sha256=manifest_file_sha,
        expected_success_file_sha256=success_file_sha,
    )
    assert len(verified.records) == 96

    drifted = tmp_path / "drifted"
    drift_manifest, drift_file, drift_success = _write_overlay(
        drifted, replica_drift=True
    )
    with pytest.raises(CheckpointValidationError, match="feature replicas"):
        verify_ws32_strategy_nd_dense_overlay(
            drifted,
            expected_manifest_sha256=drift_manifest,
            expected_manifest_file_sha256=drift_file,
            expected_success_file_sha256=drift_success,
        )
