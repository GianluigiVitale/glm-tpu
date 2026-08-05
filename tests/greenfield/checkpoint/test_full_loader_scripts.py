from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
STAGE_RUNNER = REPO / "scripts/greenfield/load_full_checkpoint_stage.py"
FLEET_RUNNER = REPO / "scripts/greenfield/run_full_checkpoint_load_pp8.sh"


def test_stage_runner_is_raw_fp8_roundtrip_and_append_only() -> None:
    source = STAGE_RUNNER.read_text()
    for required in (
        "--verify-device-roundtrip",
        "fp8_host_dequantizations",
        "runtime_checkpoint_reshards",
        "device_memory_after_load",
        "state_manifest_output",
        "TPU_VISIBLE_DEVICES",
    ):
        assert required in source
    assert "jax.distributed.initialize" not in source
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(STAGE_RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_fleet_runner_probes_then_covers_and_cleans_all_hosts() -> None:
    source = FLEET_RUNNER.read_text()
    for required in (
        "strict_census pre",
        "strict_census post_probe",
        "strict_census post",
        "--worker=0",
        "--worker=all",
        "--verify-device-roundtrip",
        "loaded_files != {plan.filename for plan in base_plans}",
        "results_db_run_id",
        "remote_success",
        "sealing append-only evidence",
        "sha256sum -c evidence.sha256",
    ):
        assert required in source
    assert 'say "SUCCESS DB=' not in source
    assert "Do not append to any sealed evidence file" in source
    completed = subprocess.run(
        ["bash", "-n", str(FLEET_RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
