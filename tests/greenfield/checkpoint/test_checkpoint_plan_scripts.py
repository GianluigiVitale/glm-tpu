from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
BUILDER = REPO / "scripts/greenfield/build_checkpoint_plan.py"
PP16_WRAPPER = REPO / "scripts/greenfield/run_checkpoint_plan_pp16.sh"
PP16_PACK_WRAPPER = REPO / "scripts/greenfield/run_checkpoint_pack_pp16.sh"


def test_checkpoint_plan_builder_is_metadata_only_and_hash_bound() -> None:
    source = BUILDER.read_text()
    for required in (
        "read_source_inventory",
        "build_placement_ledger",
        "build_pipeline_plan",
        "build_layout_manifest",
        "expected_code_hash",
        "requires a clean worktree",
        "gs://driftbench-dsv4-uc/",
    ):
        assert required in source
    for forbidden in (
        "jax.distributed.initialize",
        "safetensors.numpy.load_file",
        "TPU_VISIBLE_DEVICES",
        "gcloud compute tpus",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(BUILDER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pp16_plan_wrapper_is_same_region_append_only_and_terminal_last() -> None:
    source = PP16_WRAPPER.read_text()
    for required in (
        "GLM_GREENFIELD_PP16_PLAN",
        "APPROVED_BUCKET=gs://driftbench-dsv4-uc",
        "APPROVED_LOCATION=US-CENTRAL2",
        "bucket_location",
        ".glm-tpu-rsync.lock",
        "JAX_PLATFORMS=cpu",
        "--plan PP16_LP2",
        "--topology-capture",
        "EXECUTION_PLAN_HASH=079cefe6",
        "PLAN_MANIFEST_SHA=3c3ea07b",
        "remote_vacancy.txt",
        "--no-clobber",
        "remote_objects.json",
        "generation",
        "crc32c",
        "terminal marker is intentionally the final remote write",
        '"$REMOTE_PREFIX/SUCCESS"',
        "performance_claim\": False",
    ):
        assert required in source
    for forbidden in (
        "driftbench-storage",
        "EUROPE-WEST4",
        "jax.distributed.initialize",
        "TPU_VISIBLE_DEVICES",
        "gcloud compute tpus",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        ["bash", "-n", str(PP16_WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pp16_full_pack_wrapper_is_resumable_same_region_and_terminal_last() -> None:
    source = PP16_PACK_WRAPPER.read_text()
    for required in (
        "GLM_GREENFIELD_PP16_FULL_PACK",
        "GLM_GREENFIELD_PP16_FULL_PACK_RESUME",
        "APPROVED_BUCKET=gs://driftbench-dsv4-uc",
        "APPROVED_LOCATION=US-CENTRAL2",
        ".glm_checkpoint_pack.lock",
        ".glm-tpu-rsync.lock",
        "JAX_PLATFORMS=cpu",
        "pack_checkpoint_streaming.py",
        "inspect_packed_checkpoint.py",
        "--plan-id PP16_LP2",
        "757149950848",
        "remote_objects.json",
        "generation",
        "crc32c",
        "terminal marker is intentionally the final remote write",
        '"$REMOTE_PREFIX/SUCCESS"',
        "performance_claim':False",
    ):
        assert required in source
    for forbidden in (
        "driftbench-storage",
        "EUROPE-WEST4",
        "jax.distributed.initialize",
        "TPU_VISIBLE_DEVICES",
        "gcloud compute tpus",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        ["bash", "-n", str(PP16_PACK_WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
