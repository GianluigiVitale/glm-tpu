from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint import (
    GateCLoadExpectation,
    build_gate_c_layout,
    pack_gate_c_checkpoint,
    verify_gate_c_load_contract,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from tests.greenfield.checkpoint.test_gate_c import _oracle
from tests.greenfield.checkpoint.test_stream_pack import fixture


def _expectation(artifact: Path) -> GateCLoadExpectation:
    manifest = json.loads((artifact / "manifest.json").read_text())
    layout = json.loads((artifact / "layout_manifest.json").read_text())
    return GateCLoadExpectation(
        packed_manifest_sha256=manifest["manifest_sha256"],
        layout_manifest_sha256=layout["manifest_sha256"],
        parent_layout_manifest_sha256=layout[
            "parent_layout_manifest_sha256"
        ],
        oracle_manifest_sha256=layout["oracle_manifest_sha256"],
        source_revision=layout["source"]["revision"],
        topology_hash=layout["topology_hash"],
        plan_group_hash=layout["plan_group_hash"],
        packed_code_hash=manifest["code_hash"],
    )


def _make_artifact(tmp_path: Path) -> Path:
    source_root, parent = fixture(tmp_path)
    oracle = _oracle(source_root, parent)
    layout = build_gate_c_layout(
        parent_layout=parent,
        oracle_manifest=oracle,
        code_hash="c" * 40,
    )
    artifact = tmp_path / "gate-c-pack"
    pack_gate_c_checkpoint(
        layout=layout,
        source_root=source_root,
        output_dir=artifact,
        chunk_bytes=64,
    )
    return artifact


def test_gate_c_load_contract_refuses_identity_drift(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    expectation = _expectation(artifact)
    assert verify_gate_c_load_contract(artifact, expectation)[0][
        "manifest_sha256"
    ] == expectation.packed_manifest_sha256
    drifted = GateCLoadExpectation(
        packed_manifest_sha256="d" * 64,
        layout_manifest_sha256=expectation.layout_manifest_sha256,
        parent_layout_manifest_sha256=expectation.parent_layout_manifest_sha256,
        oracle_manifest_sha256=expectation.oracle_manifest_sha256,
        source_revision=expectation.source_revision,
        topology_hash=expectation.topology_hash,
        plan_group_hash=expectation.plan_group_hash,
        packed_code_hash=expectation.packed_code_hash,
    )
    with pytest.raises(CheckpointValidationError, match="load contract mismatch"):
        verify_gate_c_load_contract(artifact, drifted)


def _run_forced_cpu_gate_c_loader(artifact: Path) -> None:
    import jax

    from glm_tpu.greenfield.checkpoint import load_gate_c_checkpoint
    from glm_tpu.greenfield.checkpoint.one_layer_loader import (
        StageDeviceResolution,
    )

    if len(jax.devices()) != 4:
        raise AssertionError(f"expected four CPU devices, got {jax.devices()}")
    devices = tuple(jax.devices())
    layout = json.loads((artifact / "layout_manifest.json").read_text())
    captured_ids = tuple(
        record["device_id"]
        for record in sorted(
            layout["destination_files"],
            key=lambda item: item["device_slot"],
        )
    )
    resolution = StageDeviceResolution(
        devices=devices,
        coordinates=((0,), (1,), (2,), (3,)),
        captured_device_ids=captured_ids,
        captured_process_index=0,
        stage_id=0,
    )
    loaded = load_gate_c_checkpoint(
        artifact,
        _expectation(artifact),
        resolution,
    )
    name = "model.layers.0.synthetic.weight"
    assert set(loaded.weights) == {name}
    np.testing.assert_array_equal(
        np.asarray(loaded.weights[name]),
        np.arange(32, dtype=np.float32).reshape(4, 8),
    )
    assert loaded.weights[name].sharding.spec == jax.sharding.PartitionSpec(
        None, "stage"
    )
    assert loaded.load_record["packed_single_device_transfers"] == 4
    assert loaded.load_record["device_dequantizations"] == 0
    assert loaded.load_record["host_fp8_dequantizations"] == 0
    assert loaded.load_record["host_global_concatenations"] == 0
    assert loaded.load_record["runtime_checkpoint_reshards"] == 0
    assert loaded.state_manifest["loaded_final_shard_count"] == 4
    loaded.delete()


def test_gate_c_loader_builds_global_array_from_four_final_owners(
    tmp_path: Path,
) -> None:
    artifact = _make_artifact(tmp_path)
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    code = (
        "from pathlib import Path; "
        "from tests.greenfield.checkpoint.test_gate_c_loader import "
        "_run_forced_cpu_gate_c_loader; "
        f"_run_forced_cpu_gate_c_loader(Path({str(artifact)!r}))"
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
