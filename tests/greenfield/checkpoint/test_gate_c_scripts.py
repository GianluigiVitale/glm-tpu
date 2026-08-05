from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
PACKER = REPO / "scripts/greenfield/pack_gate_c_checkpoint.py"
WRAPPER = REPO / "scripts/greenfield/run_pack_gate_c_checkpoint.sh"


def test_gate_c_packer_is_pinned_to_clean_isolated_branch() -> None:
    source = PACKER.read_text()
    for required in (
        "rewrite/topology-first-decode",
        "stale code hash",
        "requires a clean worktree",
        "inspect_gate_c_oracle",
        "inspect_layout_manifest",
        "inspect_gate_c_checkpoint",
    ):
        assert required in source
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(PACKER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_gate_c_pack_wrapper_is_append_only_cpu_and_hash_bound() -> None:
    source = WRAPPER.read_text()
    for required in (
        ".glm_pod_workload.lock",
        "JAX_PLATFORMS=cpu",
        "PARENT_LAYOUT_HASH=aca0eb6d",
        "ORACLE_HASH=54262529",
        "--no-clobber",
        "crc32c_hash",
        "evidence.sha256",
        "SUCCESS",
    ):
        assert required in source
    for forbidden in (
        "jax.distributed.initialize",
        "TPU_VISIBLE_DEVICES",
        "gcloud compute tpus",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
