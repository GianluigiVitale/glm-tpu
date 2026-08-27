from __future__ import annotations

from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "scripts/greenfield/run_feature_fused_qkv_derivative_pp16.sh"


def test_pp16_direct_fused_qkv_wrapper_is_bounded_and_default_off() -> None:
    source = WRAPPER.read_text()
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    for required in (
        "APPROVED_BUCKET=gs://driftbench-dsv4-uc",
        "APPROVED_LOCATION=US-CENTRAL2",
        "SOURCE_FEATURE_MANIFEST_SHA=0f1bb271",
        "--source-feature-runtime-root",
        "--source-feature-runtime-manifest-sha256",
        "--source-metadata-only --fused-qkv-a",
        "running one exact stage-0 probe before fleet fanout",
        'if [[ "$jax" != 0 ]]',
        "strict_census pre",
        "strict_census probe_post",
        "strict_census post",
        "--worker=all",
        "stage_assignments",
        "process_index",
        "for stage in $stages",
        "--stage-id \"$stage\" --resume",
        "inspect_feature_runtime_checkpoint.py",
        'gcloud storage cp "$RUN_DIR/SUCCESS"',
    ):
        assert required in source
    for forbidden in (
        "driftbench-storage",
        "EUROPE-WEST4",
        "gcloud compute tpus tpu-vm create",
        "jax.distributed.initialize",
        "rm -",
    ):
        assert forbidden not in source
    syntax = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stdout + syntax.stderr
