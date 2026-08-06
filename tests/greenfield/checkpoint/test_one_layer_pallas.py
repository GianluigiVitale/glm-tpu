from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from glm_tpu.greenfield.checkpoint.one_layer import pack_one_layer_moe
from glm_tpu.greenfield.checkpoint.one_layer_pallas import (
    PALLAS_ONE_LAYER_LAYOUT_ID,
    PallasOneLayerPackConfig,
    inspect_pallas_one_layer_artifact,
    pack_pallas_one_layer,
)
from tests.greenfield.checkpoint.test_one_layer import (
    tiny_config,
    write_tiny_source,
)


def _pack_config(
    source_dir: Path,
    output_dir: Path,
    source_manifest_sha256: str,
) -> PallasOneLayerPackConfig:
    return PallasOneLayerPackConfig(
        source_artifact_dir=source_dir,
        source_artifact_uri=(
            "gs://driftbench-dsv4-uc/checkpoints/greenfield/test-source"
        ),
        source_manifest_sha256=source_manifest_sha256,
        output_dir=output_dir,
        code_hash="d" * 40,
    )


def test_pallas_derivative_transposes_only_routed_weights(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors import safe_open

    source_config = replace(
        tiny_config(
            tmp_path / "source_checkpoint", tmp_path / "source_artifact"
        ),
        intermediate_size=16,
    )
    write_tiny_source(source_config)
    source_manifest = pack_one_layer_moe(source_config)
    config = _pack_config(
        source_config.output_dir,
        tmp_path / "pallas_artifact",
        source_manifest["manifest_sha256"],
    )
    manifest = pack_pallas_one_layer(config)

    assert inspect_pallas_one_layer_artifact(
        config.output_dir,
        source_artifact_dir=source_config.output_dir,
    ) == manifest
    assert manifest["layout"]["layout_id"] == PALLAS_ONE_LAYER_LAYOUT_ID
    assert manifest["packed_payload_byte_count"] == source_manifest[
        "packed_payload_byte_count"
    ]
    assert manifest["source_manifest_sha256"] == source_manifest[
        "manifest_sha256"
    ]
    source_path = source_config.output_dir / "device_slot_02.safetensors"
    destination_path = config.output_dir / "device_slot_02.safetensors"
    with safe_open(
        source_path, framework="pt", device="cpu"
    ) as source, safe_open(
        destination_path, framework="pt", device="cpu"
    ) as destination:
        assert source.get_slice("expert_gate").get_shape() == [2, 16, 8]
        assert destination.get_slice("expert_gate").get_shape() == [2, 8, 16]
        assert torch.equal(
            destination.get_tensor("expert_gate"),
            source.get_tensor("expert_gate").transpose(-2, -1),
        )
        for name in (
            "expert_gate_scale",
            "shared_gate",
            "shared_down",
            "router_weight",
            "correction_bias",
        ):
            assert torch.equal(
                destination.get_tensor(name), source.get_tensor(name)
            )

    with pytest.raises(FileExistsError, match="append-only"):
        pack_pallas_one_layer(config)


def test_pallas_inspector_refuses_file_corruption(tmp_path: Path) -> None:
    source_config = replace(
        tiny_config(
            tmp_path / "source_checkpoint", tmp_path / "source_artifact"
        ),
        intermediate_size=16,
    )
    write_tiny_source(source_config)
    source_manifest = pack_one_layer_moe(source_config)
    config = _pack_config(
        source_config.output_dir,
        tmp_path / "pallas_artifact",
        source_manifest["manifest_sha256"],
    )
    manifest = pack_pallas_one_layer(config)
    path = config.output_dir / manifest["files"][0]["filename"]
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        inspect_pallas_one_layer_artifact(config.output_dir)


def _run_forced_cpu_raw_loader(artifact: Path) -> None:
    import jax
    import ml_dtypes
    import numpy as np

    from glm_tpu.greenfield.checkpoint.one_layer_loader import (
        StageDeviceResolution,
    )
    from glm_tpu.greenfield.checkpoint.one_layer_pallas_loader import (
        PallasOneLayerLoadExpectation,
        load_pallas_one_layer,
        verify_pallas_one_layer_load_contract,
    )

    manifest = json.loads((artifact / "manifest.json").read_text())
    expectation = PallasOneLayerLoadExpectation(
        manifest_sha256=manifest["manifest_sha256"],
        source_manifest_sha256=manifest["source_manifest_sha256"],
        code_hash=manifest["code_hash"],
        source_revision=manifest["source_revision"],
        topology_hash=manifest["topology_hash"],
        plan_group_hash=manifest["plan_group_hash"],
        plan_id=manifest["plan_id"],
    )
    assert verify_pallas_one_layer_load_contract(
        artifact, expectation
    ) == manifest
    devices = tuple(jax.devices())
    resolution = StageDeviceResolution(
        devices=(devices[0], devices[2], devices[1], devices[3]),
        coordinates=((0,), (2,), (1,), (3,)),
        captured_device_ids=(0, 2, 1, 3),
        captured_process_index=0,
        stage_id=0,
    )
    loaded = load_pallas_one_layer(
        artifact, expectation, resolution, expert_chunk_size=1
    )
    assert loaded.expert_gate_bits.shape == (8, 8, 8)
    assert loaded.expert_down_bits.shape == (8, 8, 8)
    assert loaded.expert_gate_bits.dtype == np.dtype("uint8")
    assert loaded.expert_gate_scale.dtype == np.dtype("float32")
    assert loaded.shared_gate_bits.shape == (8, 8)
    assert loaded.shared_down_bits.shape == (8, 8)
    assert loaded.router_weight.shape == (8, 8)
    expected_bits = np.asarray(
        [1.0, 5.0, 8.0], dtype=ml_dtypes.float8_e4m3fn
    ).view(np.uint8)
    observed = np.asarray(loaded.expert_gate_bits)
    np.testing.assert_array_equal(
        observed[[0, 4, 7], 0, 0], expected_bits
    )
    assert len(loaded.kernel_weights) == 14
    assert loaded.load_record["packed_single_device_transfers"] == 56
    assert loaded.load_record["device_fp8_dequantizations"] == 0
    assert loaded.load_record["host_fp8_dequantizations"] == 0
    assert loaded.load_record["host_global_concatenations"] == 0
    assert loaded.load_record["runtime_routed_weight_transposes"] == 0
    assert set(loaded.load_record["storage_dtypes"].values()) == {
        "BF16",
        "F32",
        "U8_E4M3FN_BITS",
    }
    loaded.close()
    assert loaded.closed


def test_raw_loader_places_final_layout_without_dequant_or_transpose(
    tmp_path: Path,
) -> None:
    source_config = replace(
        tiny_config(
            tmp_path / "source_checkpoint", tmp_path / "source_artifact"
        ),
        intermediate_size=8,
    )
    write_tiny_source(source_config)
    source_manifest = pack_one_layer_moe(source_config)
    config = _pack_config(
        source_config.output_dir,
        tmp_path / "pallas_artifact",
        source_manifest["manifest_sha256"],
    )
    pack_pallas_one_layer(config)

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    code = (
        "from pathlib import Path; "
        "from tests.greenfield.checkpoint.test_one_layer_pallas import "
        "_run_forced_cpu_raw_loader; "
        f"_run_forced_cpu_raw_loader(Path({str(config.output_dir)!r}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
