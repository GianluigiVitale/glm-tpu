from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
STAGE_RUNNER = REPO / "scripts/greenfield/load_full_checkpoint_stage.py"
FLEET_RUNNER = REPO / "scripts/greenfield/run_full_checkpoint_load_pp8.sh"
PP16_FLEET_RUNNER = REPO / "scripts/greenfield/run_full_checkpoint_load_pp16.sh"
PP16_VALIDATOR = REPO / "scripts/greenfield/validate_full_checkpoint_load_pp16.py"
PP16_SEALER = REPO / "scripts/greenfield/seal_full_checkpoint_load_pp16.py"


def test_stage_runner_is_raw_fp8_roundtrip_and_append_only() -> None:
    source = STAGE_RUNNER.read_text()
    for required in (
        "--verify-device-roundtrip",
        "fp8_host_dequantizations",
        "runtime_checkpoint_reshards",
        "device_memory_after_load",
        "state_manifest_output",
        "TPU_VISIBLE_DEVICES",
        "--stage-id",
        "stage_id=args.stage_id",
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


def test_pp16_fleet_runner_probes_maps_sixteen_stages_and_seals_last() -> None:
    source = PP16_FLEET_RUNNER.read_text()
    for required in (
        "GLM_GREENFIELD_PP16_FULL_LOAD",
        "APPROVED_BUCKET=gs://driftbench-dsv4-uc",
        "APPROVED_LOCATION=US-CENTRAL2",
        "strict_census pre",
        "strict_census post_probe",
        "strict_census post",
        "--worker=4",
        "--worker=all",
        '--stage-id "$stage"',
        '0) stages="9 14"',
        '1) stages="8 13"',
        '2) stages="10 15"',
        '3) stages="1 6"',
        '4) stages="0 5"',
        '5) stages="3 4"',
        '6) stages="11 12"',
        '7) stages="2 7"',
        "validate_full_checkpoint_load_pp16.py",
        "seal_full_checkpoint_load_pp16.py",
    ):
        assert required in source
    for forbidden in (
        "driftbench-storage",
        "EUROPE-WEST4",
        "jax.distributed.initialize",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        ["bash", "-n", str(PP16_FLEET_RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pp16_validator_and_sealer_cover_bytes_hbm_db_and_terminal_set() -> None:
    validator = PP16_VALIDATOR.read_text()
    for required in (
        "set(range(16))",
        "list(range(32))",
        "loaded_files",
        "device_roundtrip_verified",
        "fp8_device_dequantizations",
        "minimum_largest_free_block_bytes",
        "greenfield_full_checkpoint_load_pp16",
        "results_db_run_id",
        "PRAGMA integrity_check",
        '"performance_claim": False',
    ):
        assert required in validator
    sealer = PP16_SEALER.read_text()
    for required in (
        "gs://driftbench-dsv4-uc/results/",
        "if_generation_match=0",
        "crc32c",
        "remote_objects.json",
        "Terminal marker is intentionally the final remote write",
        "SUCCESS-to-ledger binding",
    ):
        assert required in sealer
    for script in (PP16_VALIDATOR, PP16_SEALER):
        completed = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
