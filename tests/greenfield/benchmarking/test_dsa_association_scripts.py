from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[3]
DISTRIBUTED_PROBE = (
    REPO / "scripts/greenfield/probe_layer0_distributed_q_a_norm.py"
)
STATE_PROBE = REPO / "scripts/greenfield/probe_layer0_dsa_association.py"
WRAPPER = REPO / "scripts/greenfield/run_layer0_dsa_association_probe.sh"


def test_distributed_q_a_probe_is_exact_pin_and_collective_bound() -> None:
    source = DISTRIBUTED_PROBE.read_text()
    for required in (
        "jax.distributed.initialize",
        "jax.process_count() != 8",
        "jax.local_device_count() != 4",
        "jax.device_count() != 32",
        '"launch_process_id": args.process_id',
        '"jax_process_index": jax.process_index()',
        "if args.process_id == 0:",
        "jax.make_array_from_callback",
        "legacy_tp32_distributed_q_a_norm",
        "distributed_collective_violations",
        "diagnostic_only_full_pod_collectives",
        "profiler_free_timing\": False",
        "jax.distributed.shutdown",
    ):
        assert required in source
    assert "jax.process_index() != args.process_id" not in source
    assert "if jax.process_index() == 0:" not in source
    for forbidden in ("import tpu_inference", "from tpu_inference", "import vllm"):
        assert forbidden not in source
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(DISTRIBUTED_PROBE)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_state_probe_requires_checksum_bound_distributed_result() -> None:
    source = STATE_PROBE.read_text()
    for required in (
        "--distributed-q-a-norm-dir",
        "--distributed-q-a-norm-manifest-sha256",
        "_inspect_distributed_q_a_norm_artifact",
        "layer0_dsa_state_from_q_residual",
        "legacy_tp32_distributed_q_a_norm_fp32_divsqrt",
    ):
        assert required in source
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(STATE_PROBE)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_distributed_result_artifact_round_trips_and_refuses_drift(
    tmp_path: Path,
) -> None:
    from safetensors.numpy import save_file

    from scripts.greenfield.probe_layer0_dsa_association import (
        _inspect_distributed_q_a_norm_artifact,
        _manifest_hash,
        _sha256_file,
    )

    q_bits = np.arange(32 * 2048, dtype=np.uint16).reshape(32, 2048)
    tensor_path = tmp_path / "distributed_q_a_norm.safetensors"
    save_file(
        {"q_residual_bfloat16_bits": q_bits},
        tensor_path,
        metadata={
            "artifact_kind": "greenfield_distributed_q_a_norm",
            "format_version": "1",
        },
    )
    code_hash = "1" * 40
    input_hash = "2" * 64
    manifest = {
        "artifact_kind": "greenfield_distributed_q_a_norm",
        "format_version": 1,
        "diagnostic_only": True,
        "code_hash": code_hash,
        "input_manifest_sha256": input_hash,
        "hlo_sha256": "3" * 64,
        "q_residual": {
            "shape": [32, 2048],
            "dtype": "bfloat16",
            "byte_count": q_bits.nbytes,
            "sha256": sha256(q_bits.view(np.uint8)).hexdigest(),
        },
        "file": {
            "filename": tensor_path.name,
            "byte_count": tensor_path.stat().st_size,
            "sha256": _sha256_file(tensor_path),
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    inspected, actual = _inspect_distributed_q_a_norm_artifact(
        tmp_path,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_code_hash=code_hash,
        expected_input_manifest_sha256=input_hash,
    )
    assert inspected == manifest
    np.testing.assert_array_equal(actual, q_bits)


def test_protected_wrapper_serializes_and_archives_both_phases() -> None:
    source = WRAPPER.read_text()
    for required in (
        ".glm_pod_workload.lock",
        "strict_census pre",
        "strict_census distributed_post",
        "strict_census post",
        "probe_layer0_distributed_q_a_norm.py",
        "probe_layer0_dsa_association.py",
        "DISTRIBUTED_Q_A_NORM_OK",
        "distributed_q_a_norm_manifest_sha256",
        "bench/results.db",
        "--no-clobber",
        "REMOTE_PREFIX/SUCCESS",
    ):
        assert required in source
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
