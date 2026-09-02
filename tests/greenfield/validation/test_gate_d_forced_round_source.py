from __future__ import annotations

import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax import lax

from glm_tpu.greenfield.benchmarking.gate_d_forced_round_source import (
    ForcedRoundSourceError,
    _validate_forced_round_closed_jaxpr,
    analyze_bf16_precision_boundaries,
    analyze_forced_round_source,
)
from glm_tpu.greenfield.kernels.reference.rmsnorm import (
    fused_add_rms_norm,
    fused_add_rms_norm_with_forced_bf16_boundary,
)

ROOT = Path(__file__).parents[3]
RUN_ROOT = Path("/home/gianl/gate-d-runs")
CAPSULE = RUN_ROOT / "greenfield_gate_d_compensated_capsule_20260831T124838Z"
SOURCE = RUN_ROOT / "gate_d_compensated_pp16_numerical_20260901T094622505067868Z"
SCRIPT = ROOT / "scripts/greenfield/analyze_gate_d_forced_round_source.py"
ARTIFACT = ROOT / "docs/artifacts/gate-d-forced-normalized-bf16-source-design.json"


def _real_arrays() -> dict[str, np.ndarray]:
    with np.load(CAPSULE / "candidate-inputs.npz", allow_pickle=False) as archive:
        inputs = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(CAPSULE / "candidate-state.npz", allow_pickle=False) as archive:
        state = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(SOURCE / "outputs.npz", allow_pickle=False) as archive:
        outputs = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    return {**inputs, **state, **outputs}


@pytest.mark.skipif(not SOURCE.is_dir(), reason="protected source evidence absent")
def test_real_forced_round_source_matches_accepted_exactly() -> None:
    arrays = _real_arrays()
    report = analyze_forced_round_source(
        hidden_update_bits=arrays["rms_hidden_update_bf16_bits"],
        residual_bits=arrays["rms_residual_bf16_bits"],
        weight_bits=arrays["rms_weight_bf16_bits"],
        accepted_output_owners=arrays["normalized"][:, 0],
        observed_output_owners=arrays["normalized_hidden_owners"][:, 0],
    )
    assert report["status"] == "SOURCE_DESIGN_CPU_EXACT_PERSISTENCE_ONLY"
    assert report["cpu_exactness"]["candidate_matches_accepted"] is True
    assert report["cpu_exactness"]["candidate_vs_observed_mismatch_count"] == 1622
    assert report["jaxpr"]["reduce_precision_count"] == 1
    assert report["jaxpr"]["reduce_precision_exponent_bits"] == 8
    assert report["jaxpr"]["reduce_precision_mantissa_bits"] == 7
    assert report["jaxpr"]["reduce_precision_feeds_weight_multiply"] is True
    assert report["jaxpr"]["weight_input_bf16_to_fp32_lineage_exact"] is True
    assert report["jaxpr"]["weighted_value_single_final_consumer"] is True
    assert report["jaxpr"]["explicit_hlo_lowering_performed"] is False
    assert report["authorization"] == {
        "full_8k": False,
        "persistence_only": True,
        "tpu_compile_or_hlo_acquisition": False,
        "tpu_execution": False,
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forced_round_is_bit_exact_to_reference_on_bf16_rows(seed: int) -> None:
    generator = np.random.default_rng(seed)
    hidden = jnp.asarray(generator.normal(size=(3, 64)).astype(ml_dtypes.bfloat16))
    residual = jnp.asarray(generator.normal(size=(3, 64)).astype(ml_dtypes.bfloat16))
    weight = jnp.asarray(generator.normal(size=(64,)).astype(ml_dtypes.bfloat16))
    expected_output, expected_carried = fused_add_rms_norm(
        hidden,
        residual,
        weight,
        epsilon=1e-5,
    )
    output, carried = fused_add_rms_norm_with_forced_bf16_boundary(
        hidden,
        residual,
        weight,
        epsilon=1e-5,
    )
    np.testing.assert_array_equal(np.asarray(output), np.asarray(expected_output))
    np.testing.assert_array_equal(np.asarray(carried), np.asarray(expected_carried))


def test_exact_bf16_boundary_semantics_cover_subnormal_ties_and_overflow() -> None:
    result = analyze_bf16_precision_boundaries()
    assert result["case_count"] == 14
    assert result["explicit_cast_widen_matches_e8m7"] is True
    assert result["includes_signed_zero_subnormal_normal_and_ties"] is True
    assert result["includes_max_finite_and_overflow_to_infinity"] is True
    assert result["weighted_bf16_outputs_exact"] is True


def test_jaxpr_validator_rejects_wrong_weight_lineage() -> None:
    def wrong_weight(
        hidden: jax.Array, residual: jax.Array, weight: jax.Array
    ) -> jax.Array:
        del weight
        rounded = lax.reduce_precision(
            hidden.astype(jnp.float32), exponent_bits=8, mantissa_bits=7
        )
        return (rounded * residual.astype(jnp.float32)).astype(jnp.bfloat16)

    values = jnp.ones((8,), dtype=jnp.bfloat16)
    hostile = jax.make_jaxpr(wrong_weight)(values, values, values)
    with pytest.raises(ForcedRoundSourceError, match="weight input lineage drifted"):
        _validate_forced_round_closed_jaxpr(hostile)


def test_jaxpr_validator_rejects_extra_weighted_consumer() -> None:
    def extra_consumer(
        hidden: jax.Array, residual: jax.Array, weight: jax.Array
    ) -> jax.Array:
        del residual
        rounded = lax.reduce_precision(
            hidden.astype(jnp.float32), exponent_bits=8, mantissa_bits=7
        )
        weighted = rounded * weight.astype(jnp.float32)
        final = weighted.astype(jnp.bfloat16)
        extra = (weighted + jnp.float32(1)).astype(jnp.bfloat16)
        return final + extra * jnp.bfloat16(0)

    values = jnp.ones((8,), dtype=jnp.bfloat16)
    hostile = jax.make_jaxpr(extra_consumer)(values, values, values)
    with pytest.raises(ForcedRoundSourceError, match="extra consumers"):
        _validate_forced_round_closed_jaxpr(hostile)


def test_forced_round_rejects_contract_drift() -> None:
    hidden = jnp.ones((1, 8), dtype=jnp.bfloat16)
    residual = jnp.ones((1, 8), dtype=jnp.bfloat16)
    weight = jnp.ones((8,), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="shapes must match"):
        fused_add_rms_norm_with_forced_bf16_boundary(
            hidden,
            residual[:, :7],
            weight,
            epsilon=1e-5,
        )
    with pytest.raises(ValueError, match="weight must match"):
        fused_add_rms_norm_with_forced_bf16_boundary(
            hidden,
            residual,
            weight[:7],
            epsilon=1e-5,
        )
    with pytest.raises(ValueError, match="requires BF16"):
        fused_add_rms_norm_with_forced_bf16_boundary(
            hidden.astype(jnp.float32),
            residual.astype(jnp.float32),
            weight.astype(jnp.float32),
            epsilon=1e-5,
        )


@pytest.mark.parametrize(
    "epsilon",
    [True, False, float("nan"), float("inf"), float("-inf"), 0.0, -1.0],
)
def test_forced_round_rejects_invalid_epsilon(epsilon: float) -> None:
    hidden = jnp.ones((1, 8), dtype=jnp.bfloat16)
    weight = jnp.ones((8,), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        fused_add_rms_norm_with_forced_bf16_boundary(
            hidden,
            hidden,
            weight,
            epsilon=epsilon,
        )


def test_forced_round_owner_or_shape_drift_fails_closed() -> None:
    arrays = _real_arrays()
    hostile = arrays["normalized_hidden_owners"][:, 0].copy()
    hostile[1, 0] ^= np.uint16(1)
    with pytest.raises(ForcedRoundSourceError, match="owner agreement drifted"):
        analyze_forced_round_source(
            hidden_update_bits=arrays["rms_hidden_update_bf16_bits"],
            residual_bits=arrays["rms_residual_bf16_bits"],
            weight_bits=arrays["rms_weight_bf16_bits"],
            accepted_output_owners=arrays["normalized"][:, 0],
            observed_output_owners=hostile,
        )
    with pytest.raises(ForcedRoundSourceError, match="shape drifted"):
        analyze_forced_round_source(
            hidden_update_bits=arrays["rms_hidden_update_bf16_bits"][:-1],
            residual_bits=arrays["rms_residual_bf16_bits"][:-1],
            weight_bits=arrays["rms_weight_bf16_bits"][:-1],
            accepted_output_owners=arrays["normalized"][:, 0],
            observed_output_owners=arrays["normalized_hidden_owners"][:, 0],
        )


def test_cli_is_default_off_and_cpu_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "gcloud" not in source
    assert "jax.distributed" not in source
    assert "libtpu" not in source.lower()
    assert "import ray" not in source
    assert "from ray" not in source
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "default-off" in result.stderr

    environment = {**os.environ, "GLM_GATE_D_FORCED_ROUND_SOURCE": "1"}
    environment.pop("JAX_PLATFORMS", None)
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "CPU-pinned" in result.stderr


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="tracked certificate absent")
def test_cli_regenerates_tracked_certificate_exactly() -> None:
    environment = {
        **os.environ,
        "GLM_GATE_D_FORCED_ROUND_SOURCE": "1",
        "JAX_PLATFORMS": "cpu",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stdout == ARTIFACT.read_bytes()
    assert result.stderr == b""


def test_cpu_backend_is_active() -> None:
    assert jax.default_backend() == "cpu"
