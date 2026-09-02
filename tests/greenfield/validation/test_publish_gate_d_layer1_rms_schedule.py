from __future__ import annotations

import importlib.util
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/greenfield/publish_gate_d_layer1_rms_schedule.py"
SPEC = importlib.util.spec_from_file_location("publish_layer1_rms_schedule", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CAPTURE = Path(
    "/home/gianl/glm-run/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
ENVELOPE = Path(
    "/home/gianl/glm-run/greenfield_layer0_dense_envelope_cross_layer_20260813T120703034434907Z/"
    "dense_envelope_cross_layer.npz"
)


def _rows() -> tuple[np.ndarray, np.ndarray]:
    with np.load(CAPTURE, allow_pickle=False) as payload:
        accepted = np.ascontiguousarray(payload["accepted_layer1_normalized_bfloat16_bits"])
    with np.load(ENVELOPE, allow_pickle=False) as payload:
        db548 = np.ascontiguousarray(payload["layer1_normalized_bfloat16_bits"])
    return db548, accepted


DRIVER = REPO / "scripts/greenfield/run_gate_d_layer1_rms_schedule.py"
DRIVER_SPEC = importlib.util.spec_from_file_location("run_layer1_rms_schedule", DRIVER)
DRIVER_MODULE = importlib.util.module_from_spec(DRIVER_SPEC)
assert DRIVER_SPEC.loader is not None
DRIVER_SPEC.loader.exec_module(DRIVER_MODULE)


def _npz(arrays: dict[str, np.ndarray]) -> bytes:
    """Use the production driver writer so the publisher is tested against real bytes."""

    return DRIVER_MODULE._deterministic_npz(arrays, np)


def _npz_stored(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            member = BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100400 << 16
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def _records(arrays: dict[str, np.ndarray], *, exact: bool) -> tuple[dict, dict]:
    runner = {
        "output_arrays": {
            name: {
                "array_sha256": sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                "shape": list(value.shape),
                "storage_dtype": value.dtype.str,
            }
            for name, value in arrays.items()
        }
    }
    control = arrays["control_layer1_normalized_bfloat16_bits"]
    schedule = arrays["schedule_layer1_normalized_bfloat16_bits"]
    numerical = {
        "harness_admissible": True,
        "schedule_arm_exact": exact,
        "control_vs_db548": {
            "elementwise_exact": True,
            "mismatch_count": 0,
            "observed_sha256": sha256(control.tobytes()).hexdigest(),
        },
        "schedule_vs_accepted": {
            "elementwise_exact": exact,
            "observed_sha256": sha256(schedule.tobytes()).hexdigest(),
            "expected_sha256": MODULE.ACCEPTED_ROW_SHA256,
        },
    }
    return runner, numerical


@pytest.mark.skipif(not (CAPTURE.exists() and ENVELOPE.exists()), reason="sealed DB548 rows unavailable")
def test_output_npz_rederives_exact_and_nonexact_from_bytes() -> None:
    db548, accepted = _rows()
    exact_arrays = {
        "accepted_layer1_normalized_bfloat16_bits": accepted,
        "control_layer1_normalized_bfloat16_bits": db548,
        "db548_layer1_normalized_bfloat16_bits": db548,
        "schedule_layer1_normalized_bfloat16_bits": accepted,
    }
    runner, numerical = _records(exact_arrays, exact=True)
    assert MODULE._validate_output_npz(_npz(exact_arrays), runner, numerical) is True
    nonexact_arrays = dict(exact_arrays, schedule_layer1_normalized_bfloat16_bits=db548)
    runner, numerical = _records(nonexact_arrays, exact=False)
    assert MODULE._validate_output_npz(_npz(nonexact_arrays), runner, numerical) is False
    # A runner flag that disagrees with the bytes is refused.
    runner, numerical = _records(nonexact_arrays, exact=True)
    with pytest.raises(RuntimeError, match="disagree with bytes"):
        MODULE._validate_output_npz(_npz(nonexact_arrays), runner, numerical)
    # The publisher binds the driver's exact deterministic container: a stored
    # (or otherwise re-encoded) archive of the same arrays is refused.
    runner, numerical = _records(exact_arrays, exact=True)
    with pytest.raises(RuntimeError, match="output member is unsafe"):
        MODULE._validate_output_npz(_npz_stored(exact_arrays), runner, numerical)
    # A control arm that does not reproduce DB548 is refused.
    refused_arrays = dict(exact_arrays, control_layer1_normalized_bfloat16_bits=accepted)
    runner, numerical = _records(refused_arrays, exact=True)
    with pytest.raises(RuntimeError, match="control arm did not reproduce DB548"):
        MODULE._validate_output_npz(_npz(refused_arrays), runner, numerical)


@pytest.mark.parametrize(
    ("exact", "status"),
    ((True, "SCHEDULE_ARM_EXACT"), (False, "SCHEDULE_ARM_NONEXACT")),
)
def test_status_is_derived_from_boolean_result(exact: bool, status: str) -> None:
    runner = {"numerical": {"schedule_arm_exact": exact, "harness_admissible": True}}
    assert MODULE._expected_status(runner)[1] == status


def test_status_refuses_inadmissible_harness_and_non_boolean() -> None:
    with pytest.raises(RuntimeError):
        MODULE._expected_status({"numerical": {"schedule_arm_exact": True, "harness_admissible": False}})
    with pytest.raises(RuntimeError):
        MODULE._expected_status({"numerical": {"schedule_arm_exact": 1, "harness_admissible": True}})


def test_source_keeps_terminal_upload_last_and_claims_open() -> None:
    source = SCRIPT.read_text()
    assert source.index("This must remain the final remote mutation") < source.index(
        'prefix + "NUMERICAL_RESULT", terminal_raw'
    )
    assert '"gate_d_closed": False' in source
    assert '"performance_claim": False' in source
    assert "SCHEDULE_ARM_EXACT" in source and "SCHEDULE_ARM_NONEXACT" in source
    assert "HARNESS_REFUSED" not in source.replace("harness_admissible", "")
    assert set(MODULE.SUCCESS_PAYLOAD) >= {
        "hlo/layer1_rms_schedule_control.optimized_hlo.txt",
        "hlo/layer1_rms_schedule_schedule.stablehlo.mlir",
        "outputs.npz",
        "runner.json",
        "mirror.sha256",
    }


def test_result_authority_line_binds_status_marker_and_terminal_identity() -> None:
    line = MODULE._result_authority_line(
        "SCHEDULE_ARM_EXACT", "a" * 64, {"generation": "123", "sha256": "b" * 64}
    )
    assert line.startswith("NUMERICAL_RESULT status=SCHEDULE_ARM_EXACT marker_sha256=")
    with pytest.raises(RuntimeError):
        MODULE._result_authority_line("NUMERICAL_ACCEPTED", "a" * 64, {"generation": "1", "sha256": "b" * 64})


def test_publisher_compiler_environment_equals_driver_environment() -> None:
    assert MODULE.EXPECTED_COMPILER_ENVIRONMENT == DRIVER_MODULE._EXPECTED_ENVIRONMENT


REAL_RUN = Path(
    "/home/gianl/gate-d-runs/gate_d_layer1_rms_schedule_20260902T061305905714981Z"
)


@pytest.mark.skipif(not REAL_RUN.exists(), reason="archived real run records unavailable")
def test_real_run_records_satisfy_dependency_authority_and_output_rederivation() -> None:
    """The 2026-09-02 TPU run's archived records must pass the publisher checks that refused them."""

    import json

    runner = json.loads((REAL_RUN / "runner.json").read_text())
    dependencies_raw = (REAL_RUN / "dependencies.json").read_bytes()
    dependencies = json.loads(dependencies_raw)
    assert dependencies["environment"] == MODULE.EXPECTED_COMPILER_ENVIRONMENT
    assert dependencies["artifact_kind"] == "gate_d_layer1_rms_schedule_dependencies"
    identity = runner["compiler_dependency_manifest"]
    assert identity["sha256"] == sha256(dependencies_raw).hexdigest()
    assert identity["byte_count"] == len(dependencies_raw)
    exact, status, classification = MODULE._expected_status(runner)
    assert exact and status == "SCHEDULE_ARM_EXACT" and runner["status"] == status
    assert runner["classification"] == classification
    output_raw = (REAL_RUN / "outputs.npz").read_bytes()
    assert MODULE._validate_output_npz(output_raw, runner, runner["numerical"]) is True
    assert runner["physical_group"] == MODULE.EXPECTED_PHYSICAL_GROUP
    assert runner["schedule_source_sha256s"] == MODULE.SCHEDULE_SOURCE_SHA256S
    assert runner["input_arrays"] == MODULE.INPUT_ARRAY_SHA256S
    assert runner["claim_scope"] == MODULE.CLAIM_SCOPE
