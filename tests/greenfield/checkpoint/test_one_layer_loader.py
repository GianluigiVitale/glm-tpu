from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint.one_layer import pack_one_layer_moe
from glm_tpu.greenfield.checkpoint.one_layer_loader import (
    OneLayerLoadExpectation,
    StageDeviceResolution,
    _fp8_e4m3fn_lookup,
    dequantize_packed_fp8,
    load_one_layer,
    load_pp8_one_layer,
    resolve_pp16_stage_devices,
    resolve_pp8_stage_devices,
    verify_one_layer_load_contract,
)
from tests.greenfield.checkpoint.test_one_layer import (
    tiny_config,
    write_tiny_source,
)


TOPOLOGY_CAPTURE = Path(
    "/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z/"
    "topology.rank0.json"
)


def _expectation(manifest: dict[str, object]) -> OneLayerLoadExpectation:
    return OneLayerLoadExpectation(
        manifest_sha256=str(manifest["manifest_sha256"]),
        source_revision=str(manifest["source_revision"]),
        topology_hash=str(manifest["topology_hash"]),
        plan_group_hash=str(manifest["plan_group_hash"]),
        plan_id=str(manifest.get("plan_id", "PP8_LP4")),
    )


def test_chunked_dequant_folds_rank_three_blocks() -> None:
    import torch

    weight = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    weight = weight.to(torch.float8_e4m3fn)
    scale = torch.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    got = dequantize_packed_fp8(
        weight,
        scale,
        block_shape=(2, 2),
        expert_chunk_size=1,
    )
    expanded = scale.repeat_interleave(2, -2).repeat_interleave(2, -1)
    expected = (weight.float() * expanded).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)


def test_device_lookup_is_bit_exact_for_every_finite_e4m3fn_value() -> None:
    import torch

    bits = torch.arange(256, dtype=torch.uint8)
    reference = bits.view(torch.float8_e4m3fn).float().numpy()
    observed = np.asarray(_fp8_e4m3fn_lookup(), dtype=np.float32)
    finite = np.isfinite(reference)
    np.testing.assert_array_equal(observed[finite], reference[finite])
    assert np.flatnonzero(~finite).tolist() == [127, 255]
    assert np.flatnonzero(~np.isfinite(observed)).tolist() == [127, 255]


class _FakeDevice:
    platform = "tpu"
    device_kind = "TPU v4"
    core_on_chip = 0

    def __init__(self, coordinate: tuple[int, ...], label: str) -> None:
        self.coords = coordinate
        self.label = label


@pytest.mark.skipif(
    not TOPOLOGY_CAPTURE.is_file(),
    reason="protected topology capture is not present",
)
def test_resolution_uses_physical_group_order_not_runtime_order() -> None:
    capture = json.loads(TOPOLOGY_CAPTURE.read_text())
    contract = capture["contract"]
    manifest = {
        "manifest_sha256": "a" * 64,
        "source_revision": "fixture",
        "topology_hash": contract["topology_hash"],
        "plan_group_hash": contract["pp8_lp4_hash"],
    }
    # This host capture is JAX process 1: runtime ids [4,5,6,7], while the
    # physical PP8 order is [4,6,5,7]. The local-subcube coordinates below
    # are the exact normalized form of its global physical coordinates.
    runtime = [
        _FakeDevice((0, 0, 0), "runtime-0"),
        _FakeDevice((1, 0, 0), "runtime-1"),
        _FakeDevice((0, 1, 0), "runtime-2"),
        _FakeDevice((1, 1, 0), "runtime-3"),
    ]
    resolved = resolve_pp8_stage_devices(
        runtime,
        capture,
        _expectation(manifest),
    )
    assert [device.label for device in resolved.devices] == [
        "runtime-0",
        "runtime-2",
        "runtime-1",
        "runtime-3",
    ]
    assert resolved.captured_device_ids == (4, 6, 5, 7)
    assert resolved.stage_id == 7


@pytest.mark.skipif(
    not TOPOLOGY_CAPTURE.is_file(),
    reason="protected topology capture is not present",
)
def test_pp16_resolution_selects_explicit_adjacent_pair() -> None:
    capture = json.loads(TOPOLOGY_CAPTURE.read_text())
    contract = capture["contract"]
    manifest = {
        "manifest_sha256": "a" * 64,
        "source_revision": "fixture",
        "topology_hash": contract["topology_hash"],
        "plan_group_hash": contract["pp16_lp2_hash"],
        "plan_id": "PP16_LP2",
    }
    runtime = [
        _FakeDevice((0, 0, 0), "runtime-0"),
        _FakeDevice((1, 0, 0), "runtime-1"),
    ]
    resolved = resolve_pp16_stage_devices(
        runtime,
        capture,
        _expectation(manifest),
        stage_id=10,
        visible_device_indices=(0, 1),
    )
    assert [device.label for device in resolved.devices] == [
        "runtime-0",
        "runtime-1",
    ]
    assert resolved.captured_device_ids == (4, 5)
    assert resolved.coordinates == ((0, 2, 0), (1, 2, 0))
    assert resolved.stage_id == 10


@pytest.mark.skipif(
    not TOPOLOGY_CAPTURE.is_file(),
    reason="protected topology capture is not present",
)
def test_pp16_resolution_selects_pair_from_full_host_runtime() -> None:
    capture = json.loads(TOPOLOGY_CAPTURE.read_text())
    contract = capture["contract"]
    manifest = {
        "manifest_sha256": "a" * 64,
        "source_revision": "fixture",
        "topology_hash": contract["topology_hash"],
        "plan_group_hash": contract["pp16_lp2_hash"],
        "plan_id": "PP16_LP2",
    }
    runtime = [
        _FakeDevice((0, 0, 0), "runtime-0"),
        _FakeDevice((1, 0, 0), "runtime-1"),
        _FakeDevice((0, 1, 0), "runtime-2"),
        _FakeDevice((1, 1, 0), "runtime-3"),
    ]
    resolved = resolve_pp16_stage_devices(
        runtime,
        capture,
        _expectation(manifest),
        stage_id=10,
        visible_device_indices=(0, 1, 2, 3),
    )
    assert [device.label for device in resolved.devices] == [
        "runtime-0",
        "runtime-1",
    ]
    assert resolved.captured_device_ids == (4, 5)


def _run_forced_cpu_loader(artifact: Path) -> None:
    import jax

    if len(jax.devices()) != 4:
        raise AssertionError(f"expected four CPU devices, got {jax.devices()}")
    manifest = json.loads((artifact / "manifest.json").read_text())
    devices = tuple(jax.devices())
    resolution = StageDeviceResolution(
        # Exercise a physical slot order that differs from JAX runtime order.
        devices=(devices[0], devices[2], devices[1], devices[3]),
        coordinates=((0,), (2,), (1,), (3,)),
        captured_device_ids=(0, 2, 1, 3),
        captured_process_index=0,
        stage_id=0,
    )
    loaded = load_pp8_one_layer(
        artifact,
        _expectation(manifest),
        resolution,
        expert_chunk_size=1,
    )
    assert loaded.expert_gate.shape == (8, 8, 8)
    assert loaded.expert_up.shape == (8, 8, 8)
    assert loaded.expert_down.shape == (8, 8, 8)
    assert loaded.shared_gate.shape == (8, 8)
    assert loaded.shared_down.shape == (8, 8)
    assert loaded.router_weight.shape == (8, 8)
    assert loaded.correction_bias.shape == (8,)
    gate = np.asarray(loaded.expert_gate, dtype=np.float32)
    assert gate[0, 0, 0] == 1.0
    assert gate[4, 0, 0] == 25.0
    assert gate[7, 0, 0] == 64.0
    np.testing.assert_array_equal(
        np.asarray(loaded.correction_bias), np.arange(8, dtype=np.float32)
    )
    assert loaded.load_record["host_global_concatenations"] == 0
    assert loaded.load_record["packed_single_device_transfers"] == 56
    assert loaded.load_record["device_dequantizations"] == 24
    assert loaded.load_record["host_fp8_dequantizations"] == 0


def test_full_loader_directly_builds_global_arrays_from_local_shards(
    tmp_path: Path,
) -> None:
    config = replace(
        tiny_config(tmp_path / "source", tmp_path / "packed"),
        intermediate_size=8,
    )
    write_tiny_source(config)
    manifest = pack_one_layer_moe(config)
    assert verify_one_layer_load_contract(
        config.output_dir, _expectation(manifest)
    ) == manifest
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    code = (
        "from pathlib import Path; "
        "from tests.greenfield.checkpoint.test_one_layer_loader import "
        "_run_forced_cpu_loader; "
        f"_run_forced_cpu_loader(Path({str(config.output_dir)!r}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _run_forced_cpu_pp16_loader(artifact: Path) -> None:
    import jax

    if len(jax.devices()) != 2:
        raise AssertionError(f"expected two CPU devices, got {jax.devices()}")
    manifest = json.loads((artifact / "manifest.json").read_text())
    devices = tuple(jax.devices())
    resolution = StageDeviceResolution(
        devices=devices,
        coordinates=((0,), (1,)),
        captured_device_ids=(4, 5),
        captured_process_index=1,
        stage_id=10,
    )
    loaded = load_one_layer(
        artifact,
        _expectation(manifest),
        resolution,
        expert_chunk_size=1,
    )
    assert loaded.mesh.devices.size == 2
    assert loaded.expert_gate.shape == (8, 8, 8)
    assert loaded.shared_gate.shape == (8, 8)
    gate = np.asarray(loaded.expert_gate, dtype=np.float32)
    assert gate[0, 0, 0] == 1.0
    assert gate[4, 0, 0] == 25.0
    assert gate[7, 0, 0] == 64.0
    assert loaded.load_record["local_experts_per_device"] == 4
    assert loaded.load_record["local_shared_intermediate_per_device"] == 4
    assert loaded.load_record["packed_single_device_transfers"] == 28
    assert loaded.load_record["device_dequantizations"] == 12
    assert loaded.load_record["host_global_concatenations"] == 0
    assert loaded.load_record["host_fp8_dequantizations"] == 0


def test_pp16_loader_consumes_final_two_file_layout(tmp_path: Path) -> None:
    config = replace(
        tiny_config(tmp_path / "source", tmp_path / "packed"),
        intermediate_size=8,
        plan_id="PP16_LP2",
        stage_size=2,
    )
    write_tiny_source(config)
    manifest = pack_one_layer_moe(config)
    assert verify_one_layer_load_contract(
        config.output_dir, _expectation(manifest)
    ) == manifest
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=2".strip()
    )
    code = (
        "from pathlib import Path; "
        "from tests.greenfield.checkpoint.test_one_layer_loader import "
        "_run_forced_cpu_pp16_loader; "
        f"_run_forced_cpu_pp16_loader(Path({str(config.output_dir)!r}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_verifier_refuses_wrong_manifest_identity(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "source", tmp_path / "packed")
    write_tiny_source(config)
    manifest = pack_one_layer_moe(config)
    wrong = OneLayerLoadExpectation(
        manifest_sha256="d" * 64,
        source_revision=str(manifest["source_revision"]),
        topology_hash=str(manifest["topology_hash"]),
        plan_group_hash=str(manifest["plan_group_hash"]),
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        verify_one_layer_load_contract(config.output_dir, wrong)
