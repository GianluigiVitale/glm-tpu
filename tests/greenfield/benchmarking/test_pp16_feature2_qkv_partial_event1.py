from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature2_qkv_partial_event1 import (
    derive_db518_prompt_only_cache_bits,
)
from glm_tpu.greenfield.errors import PlanValidationError
from scripts.greenfield.classify_pp16_feature2_qkv_partial_event1 import (
    EXPECTED_ARMS_SHA256,
    _canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/greenfield/classify_pp16_feature2_qkv_partial_event1.py"
ARTIFACT = ROOT / "docs/artifacts/pp16-feature2-qkv-khalf-event1-cpu-rejection.json"
ADMISSION = ROOT / "docs/artifacts/pp16-feature2-qkv-khalf-cpu-admission.json"
RUNTIME = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/"
    "PP16_LP2/greenfield_runtime_feature_qkv_direct_pp16_"
    "20260827T164842844148623Z"
)
DB518 = Path(
    "/home/gianl/glm-run/"
    "greenfield_pp16_feature2_layer0_db518_numerical_"
    "20260829T115022665987633Z/result.npz"
)
DB518_COMPARISON = DB518.with_name("comparison.json")
INTERNALS = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/"
    "layer1/greenfield_layer1_dsa_internal_comparison_"
    "20260808T115135394251231Z/internals.npz"
)
DSA_ORACLE = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_"
    "20260807T174904381704076Z/oracle"
)


def test_prompt_only_cache_removes_exactly_the_current_row() -> None:
    cache = np.arange(2 * 16 * 256 * 128, dtype=np.uint16).reshape(2, 16, 256, 128)
    current = cache[1, 15, 219].copy()
    prompt, digest = derive_db518_prompt_only_cache_bits(cache, current)
    expected = cache.copy()
    expected[1, 15, 219] = np.uint16(0)
    assert np.array_equal(prompt, expected)
    assert len(digest) == 64
    assert np.array_equal(cache[1, 15, 219], current)


def test_prompt_only_cache_refuses_wrong_current_key() -> None:
    cache = np.zeros((2, 16, 256, 128), dtype=np.uint16)
    with pytest.raises(PlanValidationError, match="exact current key"):
        derive_db518_prompt_only_cache_bits(cache, np.ones((128,), dtype=np.uint16))


def test_tracked_event1_result_rejects_tpu_successor() -> None:
    report = json.loads(ARTIFACT.read_text())
    assert report["status"] == "SUCCESS"
    assert report["classification"] == (
        "CPU_EVENT1_ADMISSION_REJECTED;NO_TPU_SUCCESSOR"
    )
    assert report["cpu_control_exact"] is False
    assert report["tpu_execution_authorized"] is False
    assert report["tpus_used"] == 0
    assert report["arms_sha256"] == EXPECTED_ARMS_SHA256
    assert _canonical_json_sha256(report["arms"]) == EXPECTED_ARMS_SHA256
    assert (
        report["historical_cache_provenance"][
            "layer1_prompt_history_accepted_exactness"
        ]
        == "UNKNOWN_NO_ACCEPTED_LAYER1_CACHE"
    )
    intended = report["arms"]["intended_split_k_association"]
    assert intended["accepted_event"]["positions"]["mismatch_count"] == 1892
    assert intended["accepted_event"]["symmetric_set_difference_count"] == 18
    control = report["arms"]["db518_captured_q_control"]
    assert control["db518_event"]["positions"]["mismatch_count"] == 1723
    assert len(report["runtime_tensor_receipts"]) == 22


def test_complete_arm_digest_catches_decisive_drift() -> None:
    arms = json.loads(ARTIFACT.read_text())["arms"]
    mutations = []
    changed = copy.deepcopy(arms)
    changed["intended_split_k_association"]["accepted_event"]["scores"][
        "mismatch_count"
    ] -= 1
    mutations.append(changed)
    changed = copy.deepcopy(arms)
    changed["db518_captured_q_control"]["db518_event"][
        "symmetric_set_difference_count"
    ] += 1
    mutations.append(changed)
    changed = copy.deepcopy(arms)
    changed["accepted_current_row_sensitivity"]["accepted_internals"]["query"][
        "observed_sha256"
    ] = "0" * 64
    mutations.append(changed)
    changed = copy.deepcopy(arms)
    changed["intended_split_k_association"]["current_cache_key_vs_replay_round"][
        "mismatch_count"
    ] = 1
    mutations.append(changed)
    assert all(
        _canonical_json_sha256(value) != EXPECTED_ARMS_SHA256 for value in mutations
    )


def _command(output: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        str(SCRIPT),
        "--runtime-root",
        str(RUNTIME),
        "--db518-result",
        str(DB518),
        "--db518-comparison",
        str(DB518_COMPARISON),
        "--accepted-internals",
        str(INTERNALS),
        "--dsa-oracle",
        str(DSA_ORACLE),
        "--admission",
        str(ADMISSION),
        "--output",
        str(output),
    )


def _jax_import_trap(tmp_path: Path) -> dict[str, str]:
    trap = tmp_path / "import_trap"
    trap.mkdir()
    (trap / "jax.py").write_text(
        'raise RuntimeError("JAX_IMPORTED_BEFORE_PREFLIGHT")\n'
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{trap}:{ROOT}"
    return environment


def test_unforced_classifier_refuses_before_jax_import(tmp_path: Path) -> None:
    output = tmp_path / "unforced.json"
    environment = _jax_import_trap(tmp_path)
    environment.pop("JAX_PLATFORMS", None)
    completed = subprocess.run(
        _command(output),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "requires JAX_PLATFORMS=cpu" in completed.stderr
    assert "JAX_IMPORTED_BEFORE_PREFLIGHT" not in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("dangling_symlink", [False, True])
def test_occupied_output_refuses_before_jax_import(
    tmp_path: Path, dangling_symlink: bool
) -> None:
    output = tmp_path / "occupied.json"
    if dangling_symlink:
        output.symlink_to(tmp_path / "missing.json")
    else:
        output.write_text("owner evidence\n")
    environment = _jax_import_trap(tmp_path)
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        _command(output),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "classifier output already exists" in completed.stderr
    assert "JAX_IMPORTED_BEFORE_PREFLIGHT" not in completed.stderr
    if dangling_symlink:
        assert output.is_symlink()
    else:
        assert output.read_text() == "owner evidence\n"


@pytest.mark.skipif(
    not all(
        value.exists()
        for value in (
            RUNTIME,
            DB518,
            DB518_COMPARISON,
            INTERNALS,
            DSA_ORACLE,
            ADMISSION,
        )
    ),
    reason="protected local event-1 sources are unavailable",
)
def test_real_sources_regenerate_event1_rejection(tmp_path: Path) -> None:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        pytest.skip("event-1 regeneration requires forced CPU JAX")
    output = tmp_path / "event1.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    subprocess.run(_command(output), cwd=ROOT, env=environment, check=True)
    assert output.read_bytes() == ARTIFACT.read_bytes()
