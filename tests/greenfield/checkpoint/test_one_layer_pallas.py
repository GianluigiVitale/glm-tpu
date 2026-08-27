from __future__ import annotations

from contextlib import ExitStack
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
from glm_tpu.greenfield.checkpoint.one_layer_pallas_feature import (
    PALLAS_FEATURE_LAYOUT_ID,
    PallasFeaturePackConfig,
    inspect_pallas_feature_one_layer_artifact,
    pack_pallas_feature_one_layer,
)
from tests.greenfield.checkpoint.test_one_layer import (
    tiny_config,
    write_tiny_source,
)


REPO = Path(__file__).resolve().parents[3]


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


def test_pallas_pack_wrappers_admit_pp16_only_in_approved_region() -> None:
    pallas = REPO / "scripts/greenfield/run_pack_one_layer_pallas.sh"
    feature = REPO / "scripts/greenfield/run_pack_one_layer_pallas_feature.sh"
    for path in (pallas, feature):
        source = path.read_text()
        assert "PP16_LP2" in source
        assert "US-CENTRAL2" in source
        assert "driftbench-storage" not in source
        subprocess.run(["bash", "-n", str(path)], check=True)
    runner = REPO / "scripts/greenfield/run_real_one_layer_pp8.sh"
    runner_source = runner.read_text()
    assert "GLM_GREENFIELD_PP16_PALLAS_FEATURE_PACK_RUN" in runner_source
    assert "PP16 Pallas feature run requires exact artifact identities" in (
        runner_source
    )
    subprocess.run(["bash", "-n", str(runner)], check=True)


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


def test_pallas_feature_derivative_balances_all_experts_without_new_bytes(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors import safe_open

    source_config = replace(
        tiny_config(
            tmp_path / "source_checkpoint", tmp_path / "source_artifact"
        ),
        intermediate_size=8,
    )
    write_tiny_source(source_config)
    source_manifest = pack_one_layer_moe(source_config)
    pallas_config = _pack_config(
        source_config.output_dir,
        tmp_path / "pallas_artifact",
        source_manifest["manifest_sha256"],
    )
    pallas_manifest = pack_pallas_one_layer(pallas_config)
    feature_config = PallasFeaturePackConfig(
        source_artifact_dir=pallas_config.output_dir,
        source_artifact_uri=(
            "gs://driftbench-dsv4-uc/checkpoints/greenfield/test-pallas"
        ),
        source_manifest_sha256=pallas_manifest["manifest_sha256"],
        output_dir=tmp_path / "feature_artifact",
        code_hash="e" * 40,
    )
    feature = pack_pallas_feature_one_layer(feature_config)

    assert inspect_pallas_feature_one_layer_artifact(
        feature_config.output_dir,
        source_artifact_dir=pallas_config.output_dir,
    ) == feature
    assert feature["layout"]["layout_id"] == PALLAS_FEATURE_LAYOUT_ID
    assert feature["packed_payload_byte_count"] == pallas_manifest[
        "packed_payload_byte_count"
    ]
    destination_path = feature_config.output_dir / "device_slot_02.safetensors"
    with ExitStack() as stack:
        sources = [
            stack.enter_context(
                safe_open(
                    pallas_config.output_dir
                    / f"device_slot_{slot:02d}.safetensors",
                    framework="pt",
                    device="cpu",
                )
            )
            for slot in range(4)
        ]
        destination = stack.enter_context(
            safe_open(destination_path, framework="pt", device="cpu")
        )
        assert destination.get_slice("expert_gate").get_shape() == [8, 8, 2]
        assert destination.get_slice("expert_down").get_shape() == [8, 2, 8]
        assert destination.get_slice("expert_gate_scale").get_shape() == [
            8,
            1,
            4,
        ]
        expected_gate = torch.cat(
            [source.get_tensor("expert_gate")[:, :, 4:6] for source in sources]
        )
        expected_down = torch.cat(
            [source.get_tensor("expert_down")[:, 4:6, :] for source in sources]
        )
        assert torch.equal(destination.get_tensor("expert_gate"), expected_gate)
        assert torch.equal(destination.get_tensor("expert_down"), expected_down)
        assert torch.equal(
            destination.get_tensor("shared_gate"),
            sources[2].get_tensor("shared_gate"),
        )

    with pytest.raises(FileExistsError, match="append-only"):
        pack_pallas_feature_one_layer(feature_config)


def test_pp16_pallas_feature_derivative_writes_two_exact_owners(
    tmp_path: Path,
) -> None:
    from safetensors import safe_open

    source_config = replace(
        tiny_config(
            tmp_path / "source_checkpoint", tmp_path / "source_artifact"
        ),
        intermediate_size=8,
        plan_id="PP16_LP2",
        stage_size=2,
    )
    write_tiny_source(source_config)
    source_manifest = pack_one_layer_moe(source_config)
    pallas_config = _pack_config(
        source_config.output_dir,
        tmp_path / "pallas_artifact",
        source_manifest["manifest_sha256"],
    )
    pallas_manifest = pack_pallas_one_layer(pallas_config)
    feature_config = PallasFeaturePackConfig(
        source_artifact_dir=pallas_config.output_dir,
        source_artifact_uri=(
            "gs://driftbench-dsv4-uc/checkpoints/greenfield/test-pallas-pp16"
        ),
        source_manifest_sha256=pallas_manifest["manifest_sha256"],
        output_dir=tmp_path / "feature_artifact",
        code_hash="e" * 40,
    )
    feature = pack_pallas_feature_one_layer(feature_config)

    assert feature["plan_id"] == "PP16_LP2"
    assert feature["geometry"]["stage_size"] == 2
    assert len(feature["files"]) == 2
    assert feature["packed_payload_byte_count"] == pallas_manifest[
        "packed_payload_byte_count"
    ]
    assert inspect_pallas_feature_one_layer_artifact(
        feature_config.output_dir,
        source_artifact_dir=pallas_config.output_dir,
    ) == feature
    with safe_open(
        feature_config.output_dir / "device_slot_01.safetensors",
        framework="pt",
        device="cpu",
    ) as destination:
        assert destination.get_slice("expert_gate").get_shape() == [8, 8, 4]
        assert destination.get_slice("expert_down").get_shape() == [8, 4, 8]


def _run_forced_cpu_feature_loader(artifact: Path) -> None:
    import jax
    import numpy as np

    from glm_tpu.greenfield.checkpoint.one_layer_loader import (
        StageDeviceResolution,
    )
    from glm_tpu.greenfield.checkpoint.one_layer_pallas_feature_loader import (
        PallasFeatureLoadExpectation,
        load_pallas_feature_one_layer,
        verify_pallas_feature_load_contract,
    )

    manifest = json.loads((artifact / "manifest.json").read_text())
    expectation = PallasFeatureLoadExpectation(
        manifest_sha256=manifest["manifest_sha256"],
        source_manifest_sha256=manifest["source_manifest_sha256"],
        code_hash=manifest["code_hash"],
        source_revision=manifest["source_revision"],
        topology_hash=manifest["topology_hash"],
        plan_group_hash=manifest["plan_group_hash"],
        plan_id=manifest["plan_id"],
    )
    assert verify_pallas_feature_load_contract(
        artifact, expectation
    ) == manifest
    devices = tuple(jax.devices())
    resolution_devices = (
        (devices[0], devices[2], devices[1], devices[3])
        if expectation.stage_size == 4
        else (devices[0], devices[1])
    )
    resolution = StageDeviceResolution(
        devices=resolution_devices,
        coordinates=tuple((index,) for index in range(expectation.stage_size)),
        captured_device_ids=tuple(range(expectation.stage_size)),
        captured_process_index=0,
        stage_id=0,
    )
    loaded = load_pallas_feature_one_layer(
        artifact, expectation, resolution, expert_chunk_size=1
    )
    assert loaded.expert_gate_bits.shape == (8, 8, 8)
    assert loaded.expert_down_bits.shape == (8, 8, 8)
    assert {shard.data.shape for shard in loaded.expert_gate_bits.addressable_shards} == {
        (8, 8, 8 // expectation.stage_size)
    }
    assert {shard.data.shape for shard in loaded.expert_down_bits.addressable_shards} == {
        (8, 8 // expectation.stage_size, 8)
    }
    assert loaded.load_record["routed_layout"] == "expert_intermediate_shard"
    assert loaded.load_record["packed_single_device_transfers"] == (
        14 * expectation.stage_size
    )
    assert loaded.load_record["host_global_concatenations"] == 0
    assert loaded.load_record["runtime_routed_weight_transposes"] == 0
    assert np.asarray(loaded.expert_gate_bits).shape == (8, 8, 8)
    loaded.close()


def test_feature_loader_places_final_owners_without_runtime_reshard(
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
    pallas_config = _pack_config(
        source_config.output_dir,
        tmp_path / "pallas_artifact",
        source_manifest["manifest_sha256"],
    )
    pallas_manifest = pack_pallas_one_layer(pallas_config)
    feature_config = PallasFeaturePackConfig(
        source_artifact_dir=pallas_config.output_dir,
        source_artifact_uri=(
            "gs://driftbench-dsv4-uc/checkpoints/greenfield/test-pallas"
        ),
        source_manifest_sha256=pallas_manifest["manifest_sha256"],
        output_dir=tmp_path / "feature_artifact",
        code_hash="e" * 40,
    )
    pack_pallas_feature_one_layer(feature_config)

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    code = (
        "from pathlib import Path; "
        "from tests.greenfield.checkpoint.test_one_layer_pallas import "
        "_run_forced_cpu_feature_loader; "
        f"_run_forced_cpu_feature_loader(Path({str(feature_config.output_dir)!r}))"
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


def test_pp16_feature_loader_places_two_final_owners(
    tmp_path: Path,
) -> None:
    source_config = replace(
        tiny_config(
            tmp_path / "source_checkpoint", tmp_path / "source_artifact"
        ),
        intermediate_size=8,
        plan_id="PP16_LP2",
        stage_size=2,
    )
    write_tiny_source(source_config)
    source_manifest = pack_one_layer_moe(source_config)
    pallas_config = _pack_config(
        source_config.output_dir,
        tmp_path / "pallas_artifact",
        source_manifest["manifest_sha256"],
    )
    pallas_manifest = pack_pallas_one_layer(pallas_config)
    feature_config = PallasFeaturePackConfig(
        source_artifact_dir=pallas_config.output_dir,
        source_artifact_uri=(
            "gs://driftbench-dsv4-uc/checkpoints/greenfield/test-pallas-pp16"
        ),
        source_manifest_sha256=pallas_manifest["manifest_sha256"],
        output_dir=tmp_path / "feature_artifact",
        code_hash="e" * 40,
    )
    pack_pallas_feature_one_layer(feature_config)

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    code = (
        "from pathlib import Path; "
        "from tests.greenfield.checkpoint.test_one_layer_pallas import "
        "_run_forced_cpu_feature_loader; "
        f"_run_forced_cpu_feature_loader(Path({str(feature_config.output_dir)!r}))"
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
