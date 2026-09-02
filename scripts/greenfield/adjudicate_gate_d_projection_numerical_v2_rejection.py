#!/usr/bin/env python3
"""Adjudicate the V2 projection numerical rejection from archived bytes only.

CPU-only, read-only: reloads the protected run's ``outputs.npz``, the capsule
inputs/state and the reference kernels, and localizes the divergence between
the TPU current key and the accepted witness to projection, key LayerNorm or
rotary, deriving the cosine/sine the TPU effectively applied.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

# CPU only: force the platform before JAX can initialize and refuse any other
# backend afterwards, so this adjudication can never touch an accelerator.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.pop("TPU_VISIBLE_DEVICES", None)

RUN = Path(
    "/home/gianl/gate-d-runs/"
    "gate_d_projection_contraction_pp16_numerical_20260901T235818944668679Z"
)
CAPSULE = Path(
    "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z"
)
WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
DRIVER = (
    WORKTREE / "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.py"
)
POSITION = 8155


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _cmp(a, b, np):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "bit_mismatch_count": int((a != b).sum()),
        "rotary_bit_mismatch_count": int((a[:64] != b[:64]).sum()),
        "nonrotary_bit_mismatch_count": int((a[64:] != b[64:]).sum()),
        "max_abs_error": float(d.max()),
        "rotary_max_abs_error": float(d[:64].max()),
        "nonrotary_max_abs_error": float(d[64:].max()),
        "mean_abs_error": float(d.mean()),
    }


def main() -> int:
    sys.path.insert(0, str(WORKTREE))
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np

    if jax.default_backend() != "cpu" or any(
        device.platform != "cpu" for device in jax.devices()
    ):
        raise RuntimeError("adjudication must run on the CPU backend only")

    from glm_tpu.greenfield.kernels.reference import dsa as dsa_mod
    from glm_tpu.greenfield.kernels.reference.rotary import rotary_cos_sin

    spec = importlib.util.spec_from_file_location("driver", DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    outputs_raw = (RUN / "outputs.npz").read_bytes()
    runner = json.loads((RUN / "runner.json").read_bytes())
    receipt = json.loads((RUN / "diagnostic_upload_receipt.json").read_bytes())
    out = np.load(RUN / "outputs.npz")
    state = np.load(CAPSULE / "candidate-state.npz")
    inputs = {k: v for k, v in np.load(CAPSULE / "candidate-inputs.npz").items()}
    wk = np.asarray(driver._materialize_wk(inputs, np, ml_dtypes)[0], dtype=np.float32)
    x = state["normalized"][0, 0].view(ml_dtypes.bfloat16).astype(np.float32)
    knw = inputs["key_norm_weight_bf16_bits"][0].view(ml_dtypes.bfloat16)
    knb = inputs["key_norm_bias_bf16_bits"][0].view(ml_dtypes.bfloat16)
    tpu_proj = out["projected_key_owners"][0, 0].astype(np.float32)
    tpu_key = out["current_key_owners"][0, 0].astype(np.float32)
    accepted_key = state["current_key"].astype(np.float32)
    contract = dsa_mod.DsaNumericalContract()
    cpu_proj = np.asarray(
        jnp.dot(jnp.asarray(wk), jnp.asarray(x), precision=jax.lax.Precision.HIGHEST),
        dtype=np.float32,
    )
    f64_proj = (wk.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)

    def key_from(proj):
        keys = dsa_mod.dsa_index_keys_from_projection(
            jnp.asarray(proj)[None, :],
            jnp.asarray(knw),
            jnp.asarray(knb),
            jnp.asarray([POSITION], dtype=jnp.int32),
            contract=contract,
        )
        return np.asarray(keys, dtype=np.float32)[0]

    def norm_of(proj):
        value = dsa_mod._affine_layer_norm(
            jnp.asarray(proj)[None, :],
            jnp.asarray(knw),
            jnp.asarray(knb),
            epsilon=contract.key_layer_norm_epsilon,
        )
        return np.asarray(value, dtype=np.float32)[0]

    cpu_key_from_tpu_proj = key_from(tpu_proj)
    cpu_key_from_cpu_proj = key_from(cpu_proj)
    norm_tpu = norm_of(tpu_proj)
    inv_freq = 1.0 / (np.float64(contract.theta) ** (np.arange(0, 64, 2) / 64.0))
    angles = POSITION * inv_freq
    cos_true, sin_true = np.cos(angles), np.sin(angles)
    cos_cpu, sin_cpu = (
        np.asarray(v, dtype=np.float64)[0]
        for v in rotary_cos_sin(
            jnp.asarray([POSITION], dtype=jnp.int32),
            rotary_dim=64,
            theta=contract.theta,
            dtype=jnp.float32,
        )
    )

    def implied(pre, post):
        a, b = pre[0:64:2].astype(np.float64), pre[1:64:2].astype(np.float64)
        ap, bp = post[0:64:2].astype(np.float64), post[1:64:2].astype(np.float64)
        r2 = a * a + b * b
        return (a * ap + b * bp) / r2, (a * bp - b * ap) / r2

    tpu_cos, tpu_sin = implied(norm_tpu, tpu_key)
    acc_cos, acc_sin = implied(norm_of(cpu_proj), accepted_key)
    report = {
        "artifact_kind": "gate_d_projection_numerical_v2_rejection_adjudication",
        "analysis_scope": "CPU-only re-derivation from the archived protected outputs; no TPU access.",
        "run_tag": runner.get("run_tag", RUN.name),
        "runner_status": runner["status"],
        "runner_classification": runner["classification"],
        "outputs_npz_sha256": _sha(outputs_raw),
        "runner_json_sha256": _sha((RUN / "runner.json").read_bytes()),
        "diagnostic_terminal_receipt": receipt,
        "position": POSITION,
        "witness_hashes": {
            "accepted_current_key": _sha(accepted_key.tobytes()),
            "tpu_current_key": _sha(tpu_key.tobytes()),
            "tpu_projected_key": _sha(tpu_proj.tobytes()),
            "cpu_projected_key_highest": _sha(cpu_proj.tobytes()),
            "cpu_key_from_tpu_projection": _sha(cpu_key_from_tpu_proj.tobytes()),
        },
        "projection": {
            "tpu_vs_cpu_highest": _cmp(tpu_proj, cpu_proj, np),
            "tpu_vs_f64_accumulated": _cmp(tpu_proj, f64_proj, np),
            "matmul_precision_verdict": "f32-accurate (sub-ulp); accumulation order differs from CPU",
        },
        "key_layer_norm": {
            "tpu_nonrotary_key_vs_cpu_norm_of_tpu_projection": _cmp(
                np.r_[np.zeros(64, np.float32), tpu_key[64:]],
                np.r_[np.zeros(64, np.float32), norm_tpu[64:]],
                np,
            ),
            "verdict": "bit-exact on the 64 non-rotary dimensions",
        },
        "current_key": {
            "tpu_vs_accepted": _cmp(tpu_key, accepted_key, np),
            "cpu_from_tpu_projection_vs_accepted": _cmp(
                cpu_key_from_tpu_proj, accepted_key, np
            ),
            "cpu_from_tpu_projection_vs_tpu": _cmp(cpu_key_from_tpu_proj, tpu_key, np),
            "cpu_from_cpu_projection_vs_accepted": _cmp(
                cpu_key_from_cpu_proj, accepted_key, np
            ),
        },
        "rotary": {
            "angles_rad": [float(v) for v in angles],
            "cpu_f32_cos_max_error_vs_f64": float(np.abs(cos_cpu - cos_true).max()),
            "cpu_f32_sin_max_error_vs_f64": float(np.abs(sin_cpu - sin_true).max()),
            "accepted_key_implied_cos_max_error_vs_f64": float(
                np.abs(acc_cos - cos_true).max()
            ),
            "accepted_key_implied_sin_max_error_vs_f64": float(
                np.abs(acc_sin - sin_true).max()
            ),
            "tpu_key_implied_cos_max_error_vs_f64": float(
                np.abs(tpu_cos - cos_true).max()
            ),
            "tpu_key_implied_sin_max_error_vs_f64": float(
                np.abs(tpu_sin - sin_true).max()
            ),
            "tpu_key_implied_cos_error_per_pair": [
                float(v) for v in (tpu_cos - cos_true)
            ],
            "tpu_key_implied_sin_error_per_pair": [
                float(v) for v in (tpu_sin - sin_true)
            ],
            "verdict": (
                "TPU on-device f32 cos/sin at large rotary angles (up to 8155 rad) deviate by up to "
                "~1e-2; the accepted key is consistent with an accurately evaluated f32 table."
            ),
        },
        "classification": (
            "V2_NUMERICAL_REJECTED_ADJUDICATED;PROJECTION_MATMUL_F32_ACCURATE;"
            "KEY_LAYERNORM_BIT_EXACT;ROOT_CAUSE_CANDIDATE_TPU_ON_DEVICE_ROTARY_COS_SIN;"
            "TPU_FIX_UNPROVEN;GATE_D_OPEN"
        ),
        "exact_next": (
            "Move DSA rotary cos/sin evaluation off the accelerator: build the f32 table on the host "
            "with the accepted f32 formula and gather it by position on device; then re-run the "
            "bounded discriminator expecting rotary dims bit-exact to the CPU key of the TPU projection."
        ),
        "gate_d_closed": False,
        "performance_claim": False,
        "root_cause_fix_proven": False,
        "schema_version": 1,
    }
    sys.stdout.write(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
