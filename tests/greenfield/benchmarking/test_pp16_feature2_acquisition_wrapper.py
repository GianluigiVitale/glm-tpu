from __future__ import annotations

import os
import subprocess
from pathlib import Path

WRAPPER = Path("scripts/greenfield/run_pp16_feature2_prefill_acquisition.sh")


def test_feature2_acquisition_wrapper_is_default_off_and_serialized() -> None:
    source = WRAPPER.read_text()
    for marker in (
        "GLM_GREENFIELD_PP16_FEATURE2_ACQUIRE:-0",
        "GLM_GREENFIELD_PP16_FEATURE2_MODE:-off",
        "GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE:-0",
        "compile_only",
        "/home/gianl/glm-run/.glm_pod_workload.lock",
        "/home/gianl/.glm-tpu-rsync.lock",
        "flock -n 9",
        "flock 8",
        "strict_census pre",
        "strict_census post",
        "ray_enum=",
        "--worker=all",
        "TPU_PROCESS_BOUNDS=1,1,1",
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
        "--compile-only 1",
        "--full-width-rounded-then-slice",
        "full_width_rounded_then_slice",
    ):
        assert marker in source


def test_feature2_acquisition_wrapper_refuses_the_rejected_variant() -> None:
    source = WRAPPER.read_text()
    assert "[[ $FULL_WIDTH_ROUNDED_THEN_SLICE == 1 ]]" in source
    assert "== 0 ||" not in source
    assert "readonly runner_variant_args=(--full-width-rounded-then-slice)" in source
    environment = os.environ.copy()
    environment.update(
        {
            "GLM_GREENFIELD_PP16_FEATURE2_ACQUIRE": "1",
            "GLM_GREENFIELD_PP16_FEATURE2_MODE": "compile_only",
            "GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE": "0",
        }
    )
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "admitted successor" in completed.stderr


def test_feature2_acquisition_wrapper_has_no_numerical_or_db_success_path() -> None:
    source = WRAPPER.read_text()
    assert "'main_executed':False" in source
    assert "'numerical_claim':False" in source
    assert "'performance_claim':False" in source
    assert "HLO_ACQUIRED" in source
    assert "results.db" not in source
    assert "/SUCCESS" not in source
    assert "--warmup" not in source
    assert "--iterations" not in source


def test_feature2_acquisition_wrapper_pins_region_sources_and_remote_hashes() -> None:
    source = WRAPPER.read_text()
    for marker in (
        "APPROVED_LOCATION=US-CENTRAL2",
        "b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab",
        "e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2",
        "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da",
        "79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054",
        "ab5be45aecf3b0b5d87ad76af8076bc9351823529a08c0eadb414b072b31cb2d",
        'gcloud storage cat "$REMOTE_PREFIX/$relative"',
        "upload_failure_diagnostic",
        "verify_failure_diagnostic",
        "verify_success_evidence",
        "DIAGNOSTIC_UPLOAD_FAILED after two attempts",
        'gcloud storage ls --recursive "$REMOTE_PREFIX/**"',
        "--no-clobber",
    ):
        assert marker in source


def test_feature2_acquisition_wrapper_is_bash_syntax_valid() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
