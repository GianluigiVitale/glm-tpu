from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation import (
    gate_d_projection_host_rope_numerical as module,
)

ROOT = Path(__file__).parents[3]
V2_RUN = Path(
    "/home/gianl/gate-d-runs/gate_d_projection_contraction_pp16_numerical_20260901T235818944668679Z"
)
CAPSULE = Path(
    "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z"
)
ACCEPTED_HLO = Path(
    "/home/gianl/gate-d-runs/gate_d_projection_contraction_pp16_hlo_20260901T213605719107105Z/hlo/"
    "projection_contraction_pp16_stage0.optimized_hlo.txt"
)


def _protected_bytes_present() -> bool:
    return (V2_RUN / "outputs.npz").exists() and (
        CAPSULE / "candidate-inputs.npz"
    ).exists()


def _materialize_wk(inputs: dict[str, np.ndarray]) -> np.ndarray:
    values = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        if exponent == 0:
            value = mantissa * (2.0**-9)
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(sign * value)
    lookup = np.asarray(values, dtype=np.float32)
    bits = inputs["wk_weight_bits"]
    scales = inputs["wk_scale_inv"]
    rows = np.arange(bits.shape[-2]) // 128
    columns = np.arange(bits.shape[-1]) // 128
    expanded = scales[..., rows[:, None], columns[None, :]]
    decoded = lookup[bits.astype(np.int32)] * expanded.astype(np.float32)
    return np.ascontiguousarray(
        np.ascontiguousarray(decoded, dtype=ml_dtypes.bfloat16).astype(np.float32)
    )


def test_host_rope_row_is_pinned_and_matches_kernel_helper() -> None:
    from glm_tpu.greenfield.kernels.reference.rotary_table import dsa_rotary_row_host

    owners = module.materialize_dsa_rope_row(np)
    assert owners.shape == (2, 64) and owners.dtype == np.float32
    assert np.array_equal(
        owners[0], dsa_rotary_row_host(8155, rotary_dim=64, theta=8_000_000.0)
    )
    identity = module.dsa_rope_row_identity(owners, np)
    assert identity["row_sha256"] == module.EXPECTED_DSA_ROPE_ROW_SHA256
    hostile = owners.copy()
    hostile[1, 3] += np.float32(1e-3)
    with pytest.raises(module.HostRopeValidationError, match="identity drifted"):
        module.dsa_rope_row_identity(hostile, np)


def test_reference_key_and_implied_rotary_are_self_consistent() -> None:
    generator = np.random.default_rng(1)
    projected = [float(v) for v in generator.standard_normal(128) * 2]
    weight = [float(v) for v in 1 + 0.1 * generator.standard_normal(128)]
    bias = [float(v) for v in 0.05 * generator.standard_normal(128)]
    row = [float(v) for v in module.materialize_dsa_rope_row(np)[0]]
    key = module.reference_key_f64(projected, weight, bias, row)
    pre = module.reference_key_f64(projected, weight, bias, [1.0] * 32 + [0.0] * 32)
    implied = module.implied_cos_sin_error(pre, key, row)
    assert implied["max_abs_cos_error"] < 1e-12 and implied["max_abs_sin_error"] < 1e-12
    assert key[64:] == pre[64:]
    with pytest.raises(module.HostRopeValidationError):
        module.reference_key_f64(projected[:-1], weight, bias, row)


@pytest.mark.skipif(not _protected_bytes_present(), reason="protected run bytes absent")
def test_v2_archived_outputs_are_rejected_for_rotary_only_and_fixed_key_is_accepted() -> (
    None
):
    import jax.numpy as jnp

    from glm_tpu.greenfield.kernels.reference.dsa_host_rope import (
        dsa_index_keys_from_projection_host_rope,
    )

    inputs = {k: v for k, v in np.load(CAPSULE / "candidate-inputs.npz").items()}
    outputs = {k: v for k, v in np.load(V2_RUN / "outputs.npz").items()}
    wk = _materialize_wk(inputs)
    row = module.materialize_dsa_rope_row(np)
    verdict = module.classify_host_rope_outputs(
        outputs,
        np,
        wk_weight=wk,
        key_norm_weight_bits=inputs["key_norm_weight_bf16_bits"],
        key_norm_bias_bits=inputs["key_norm_bias_bf16_bits"],
        rope_row=row,
        ml_dtypes=ml_dtypes,
    )
    assert verdict["structural"] is True
    assert verdict["accepted_tpu_host_rope_faithful"] is False
    projected = verdict["outputs"]["projected_key_owners"]
    key = verdict["outputs"]["current_key_owners"]
    assert verdict["outputs"]["normalized_hidden_owners"]["witness_exact"] is True
    assert (
        projected["within_tolerance"] is True
        and projected["max_abs_error_vs_f64_reference"] < 1e-6
    )
    assert key["within_tolerance"] is False
    assert key["nonrotary_max_abs_error"] < 1e-6
    assert key["rotary_max_abs_error"] > 1e-3
    assert key["implied_rotary"]["max_abs_cos_error"] > 5e-3
    # The same TPU projection rotated on CPU with the host row is faithful.
    fixed = np.asarray(
        dsa_index_keys_from_projection_host_rope(
            jnp.asarray(outputs["projected_key_owners"][:, 0]),
            jnp.asarray(
                inputs["key_norm_weight_bf16_bits"][0].view(ml_dtypes.bfloat16)
            ),
            jnp.asarray(inputs["key_norm_bias_bf16_bits"][0].view(ml_dtypes.bfloat16)),
            jnp.asarray([8155, 8155], dtype=jnp.int32),
            jnp.asarray(row),
            key_norm_mode="divide_sqrt",
        )
    )[:, None, :]
    fixed_outputs = {
        **outputs,
        "current_key_owners": np.ascontiguousarray(fixed, dtype=np.float32),
    }
    fixed_verdict = module.classify_host_rope_outputs(
        fixed_outputs,
        np,
        wk_weight=wk,
        key_norm_weight_bits=inputs["key_norm_weight_bf16_bits"],
        key_norm_bias_bits=inputs["key_norm_bias_bf16_bits"],
        rope_row=row,
        ml_dtypes=ml_dtypes,
    )
    assert fixed_verdict["accepted_tpu_host_rope_faithful"] is True
    fixed_key = fixed_verdict["outputs"]["current_key_owners"]
    assert fixed_key["max_abs_error_vs_f64_reference"] < 1e-6
    assert fixed_key["implied_rotary"]["max_abs_cos_error"] < 1e-6
    # Owner disagreement or non-finite values fail structurally.
    hostile = {**outputs, "current_key_owners": outputs["current_key_owners"].copy()}
    hostile["current_key_owners"][1, 0, 0] += 1.0
    hostile_verdict = module.classify_host_rope_outputs(
        hostile,
        np,
        wk_weight=wk,
        key_norm_weight_bits=inputs["key_norm_weight_bf16_bits"],
        key_norm_bias_bits=inputs["key_norm_bias_bf16_bits"],
        rope_row=row,
        ml_dtypes=ml_dtypes,
    )
    assert hostile_verdict["structural"] is False
    assert hostile_verdict["accepted_tpu_host_rope_faithful"] is False


def _cpu_host_rope_hlo() -> dict[str, str]:
    code = """
import json
import jax
import jax.numpy as jnp
from glm_tpu.greenfield.benchmarking.gate_d_projection_contraction_pp16_host_rope import _build_shard_map
devices = tuple(jax.devices())
replay = _build_shard_map(devices=devices, axis_name='feature')
lowered = replay.lower(
    jax.ShapeDtypeStruct((2, 1, 6144), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 128, 6144), jnp.float32),
    jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 64), jnp.float32),
)
print(json.dumps({'stablehlo': lowered.as_text(), 'optimized': lowered.compile().as_text()}))
"""
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_hlo_audits_accept_host_rope_graph_and_reject_transcendental_predecessor() -> (
    None
):
    texts = _cpu_host_rope_hlo()
    stable = module.audit_host_rope_stablehlo(texts["stablehlo"])
    assert stable["transcendental_count"] == 0 and stable["collective_count"] == 0
    optimized = module.audit_host_rope_optimized_hlo(texts["optimized"])
    assert optimized["module_name"] == module.HLO_MODULE_NAME
    assert optimized["entry_parameter_count"] == 5
    assert optimized["transcendental_count"] == 0
    with pytest.raises(module.HostRopeValidationError):
        module.audit_host_rope_stablehlo(
            texts["stablehlo"].replace("tensor<2x64xf32>", "tensor<2x65xf32>")
        )
    with pytest.raises(module.HostRopeValidationError, match="transcendental|identity"):
        module.audit_host_rope_optimized_hlo(
            texts["optimized"].replace(
                "HloModule jit__projection_host_rope_local", "HloModule other", 1
            )
        )
    if ACCEPTED_HLO.exists():
        with pytest.raises(module.HostRopeValidationError):
            module.audit_host_rope_optimized_hlo(ACCEPTED_HLO.read_text())
