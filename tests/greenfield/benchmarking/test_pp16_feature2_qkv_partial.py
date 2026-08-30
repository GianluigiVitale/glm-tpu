from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature2_qkv_partial import (
    one_row_lp2_feature_partial_qkv_a_convolution,
    validate_lp2_feature_partial_stablehlo,
)
from glm_tpu.greenfield.errors import HloContractViolationError

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "docs/artifacts/pp16-feature2-qkv-khalf-cpu-admission.json"
RUNTIME = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/"
    "PP16_LP2/greenfield_runtime_feature_qkv_direct_pp16_20260827T164842844148623Z"
)
ORACLE = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/layer1/"
    "greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z/internals.npz"
)
DB518 = Path(
    "/home/gianl/glm-run/"
    "greenfield_pp16_feature2_layer0_db518_numerical_20260829T115022665987633Z/"
    "result.npz"
)
SCRIPT = ROOT / "scripts/greenfield/classify_pp16_feature2_qkv_partial.py"


_STABLEHLO = """
func.func public @main(%arg0: tensor<1x6144xbf16>) -> tensor<1x2048xbf16> {
  %0 = stablehlo.dynamic_slice %arg0, ... : (tensor<1x6144xbf16>) -> tensor<1x3072xbf16>
  %1 = stablehlo.dynamic_slice %weight, ... : (tensor<32x6144x82xui8>) -> tensor<32x3072x82xui8>
  %2 = stablehlo.dynamic_slice %scale, ... : (tensor<32x48x82xf32>) -> tensor<32x24x82xf32>
  %3 = "stablehlo.all_reduce"(%partial) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<32x1x82xf32>) -> tensor<32x1x82xf32>
  %4 = stablehlo.convert %3 : (tensor<32x1x82xf32>) -> tensor<32x1x82xbf16>
  return %out : tensor<1x2048xbf16>
}
"""


def test_source_contract_requires_exact_lp2_group() -> None:
    with pytest.raises(ValueError, match="exact local group"):
        one_row_lp2_feature_partial_qkv_a_convolution(
            object(),
            object(),
            object(),
            object(),
            axis_name="feature",
            groups=((0, 2),),
        )


def test_stablehlo_contract_accepts_only_compact_f32_lp2_reduction() -> None:
    report = validate_lp2_feature_partial_stablehlo(_STABLEHLO)
    assert report["passed"] is True
    assert report["all_reduce_count"] == 1
    assert report["partial_shape"] == [32, 1, 82]
    hostile = (
        _STABLEHLO.replace("[[0, 1]]", "[[0, 2]]"),
        _STABLEHLO.replace("32x1x82xf32", "32x1x81xf32"),
        _STABLEHLO.replace('"stablehlo.all_reduce"', '"stablehlo.add"'),
        _STABLEHLO.replace(
            "return %out",
            '%dead = "stablehlo.all_gather"(%out) : '
            "(tensor<1x2048xbf16>) -> tensor<1x2048xbf16>\n  return %out",
        ),
        _STABLEHLO.replace(
            "return %out",
            "%dead = stablehlo.reshape %out : "
            "(tensor<1x2048xbf16>) -> tensor<32x6144xbf16>\n  return %out",
        ),
    )
    for value in hostile:
        with pytest.raises(HloContractViolationError):
            validate_lp2_feature_partial_stablehlo(value)


def test_tracked_admission_is_fail_closed() -> None:
    report = json.loads(ARTIFACT.read_text())
    assert report["status"] == "SUCCESS"
    assert report["gate_d_closed"] is False
    assert report["protected_tpu_evidence"] is False
    assert report["tpus_used"] == 0
    assert report["stablehlo"]["collective_group"] == [[0, 1]]
    assert (
        report["comparisons"]["accepted_f32_partial_vs_accepted"]["mismatch_count"] == 0
    )
    assert (
        report["comparisons"]["accepted_bf16_partial_vs_accepted"]["mismatch_count"]
        == 656
    )
    assert report["state_coherence"]["candidate_coherent_history_reusable"] is True


def _classifier_command(output: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        str(SCRIPT),
        "--runtime-root",
        "/missing/runtime",
        "--accepted-oracle",
        "/missing/oracle.npz",
        "--db518-result",
        "/missing/db518.npz",
        "--feature2-program",
        "/missing/program.py",
        "--output",
        str(output),
    )


def _environment_with_jax_import_trap(tmp_path: Path) -> dict[str, str]:
    trap = tmp_path / "import_trap"
    trap.mkdir()
    (trap / "jax.py").write_text(
        'raise RuntimeError("JAX_IMPORTED_BEFORE_PREFLIGHT")\n'
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{trap}:{ROOT}"
    return environment


def test_unforced_subprocess_refuses_before_importing_jax(tmp_path: Path) -> None:
    output = tmp_path / "unforced.json"
    environment = _environment_with_jax_import_trap(tmp_path)
    environment.pop("JAX_PLATFORMS", None)
    completed = subprocess.run(
        _classifier_command(output),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "explicit JAX_PLATFORMS=cpu before JAX import" in completed.stderr
    assert "JAX_IMPORTED_BEFORE_PREFLIGHT" not in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("dangling_symlink", [False, True])
def test_existing_output_refuses_before_importing_jax(
    tmp_path: Path, dangling_symlink: bool
) -> None:
    output = tmp_path / "occupied.json"
    if dangling_symlink:
        output.symlink_to(tmp_path / "missing-target.json")
    else:
        output.write_text("owner evidence\n")
    environment = _environment_with_jax_import_trap(tmp_path)
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        _classifier_command(output),
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
    not (RUNTIME.is_dir() and ORACLE.is_file() and DB518.is_file()),
    reason="protected local admission sources are unavailable",
)
def test_real_sources_regenerate_tracked_admission(tmp_path: Path) -> None:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        pytest.skip("real admission regeneration requires forced CPU JAX")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("historical admission regeneration requires uv")
    output = tmp_path / "admission.json"
    command = (
        uv,
        "run",
        "--with",
        "jax[cpu]==0.11.1",
        "--with",
        "numpy==2.5.2",
        "--with",
        "ml-dtypes==0.6.0",
        "python",
        str(SCRIPT),
        "--runtime-root",
        str(RUNTIME),
        "--accepted-oracle",
        str(ORACLE),
        "--db518-result",
        str(DB518),
        "--feature2-program",
        str(ROOT / "glm_tpu/greenfield/benchmarking/pp16_feature2_program.py"),
        "--output",
        str(output),
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    assert output.read_bytes() == ARTIFACT.read_bytes()
