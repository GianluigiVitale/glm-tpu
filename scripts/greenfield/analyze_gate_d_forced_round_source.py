#!/usr/bin/env python3
"""Emit the CPU-only Gate-D forced-round source-design certificate."""

from __future__ import annotations

import inspect
import json
import os
import stat
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

RUN_ROOT = Path("/home/gianl/gate-d-runs")
WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
CAPSULE = RUN_ROOT / "greenfield_gate_d_compensated_capsule_20260831T124838Z"
SOURCE = RUN_ROOT / "gate_d_compensated_pp16_numerical_20260901T094622505067868Z"
SCALAR_ARTIFACT = (
    WORKTREE / "docs/artifacts/gate-d-scalar-frontier-conversion-placement.json"
)
ANALYSIS_MODULE = (
    WORKTREE / "glm_tpu/greenfield/benchmarking/gate_d_forced_round_source.py"
)
KERNEL_MODULE = WORKTREE / "glm_tpu/greenfield/kernels/reference/rmsnorm.py"
CAPSULE_INPUT_SHA256 = (
    "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
)
CAPSULE_STATE_SHA256 = (
    "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236"
)
PROTECTED_OUTPUT_SHA256 = (
    "bd017d4c7e3f42b3f60fb4b918811ea028447656227c361c84fffae2544dc1a3"
)
SCALAR_ARTIFACT_SHA256 = (
    "8ee468ebddd8fc0f7c594e90fd463a6582379aea5969f59859900e2ccec170ce"
)


def _snapshot(path: Path, *, limit: int = 64 << 20) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise RuntimeError(f"unsafe forced-round source input: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 8 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_uid,
                value.st_gid,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if (
            len(raw) != before.st_size
            or identity(before) != identity(after)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError(f"forced-round source input changed: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _require_hash(raw: bytes, expected: str, label: str) -> None:
    if sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"{label} bytes drifted")


def _archive(raw: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def main() -> int:
    if os.environ.get("GLM_GATE_D_FORCED_ROUND_SOURCE") != "1":
        raise RuntimeError("forced-round source analysis is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("forced-round source analysis must be CPU-pinned")

    input_raw = _snapshot(CAPSULE / "candidate-inputs.npz")
    state_raw = _snapshot(CAPSULE / "candidate-state.npz")
    output_raw = _snapshot(SOURCE / "outputs.npz")
    scalar_raw = _snapshot(SCALAR_ARTIFACT, limit=1 << 20)
    analysis_source_raw = _snapshot(ANALYSIS_MODULE, limit=1 << 20)
    kernel_source_raw = _snapshot(KERNEL_MODULE, limit=1 << 20)
    _require_hash(input_raw, CAPSULE_INPUT_SHA256, "capsule inputs")
    _require_hash(state_raw, CAPSULE_STATE_SHA256, "capsule state")
    _require_hash(output_raw, PROTECTED_OUTPUT_SHA256, "protected outputs")
    _require_hash(scalar_raw, SCALAR_ARTIFACT_SHA256, "scalar-frontier artifact")
    scalar = json.loads(scalar_raw)
    if (
        scalar.get("status") != "EXACT_EXPLANATION_PROVED_PHYSICAL_CAUSE_UNPROVEN"
        or scalar.get("tpu_rerun_performed") is not False
    ):
        raise RuntimeError("scalar-frontier authority drifted")

    import jax

    from glm_tpu.greenfield.benchmarking.gate_d_forced_round_source import (
        analyze_forced_round_source,
    )
    from glm_tpu.greenfield.kernels.reference.rmsnorm import (
        fused_add_rms_norm_with_forced_bf16_boundary,
    )

    if jax.default_backend() != "cpu":
        raise RuntimeError("forced-round source analysis initialized a non-CPU backend")
    inputs = _archive(input_raw)
    state = _archive(state_raw)
    outputs = _archive(output_raw)
    report = analyze_forced_round_source(
        hidden_update_bits=inputs["rms_hidden_update_bf16_bits"],
        residual_bits=inputs["rms_residual_bf16_bits"],
        weight_bits=inputs["rms_weight_bf16_bits"],
        accepted_output_owners=state["normalized"][:, 0],
        observed_output_owners=outputs["normalized_hidden_owners"][:, 0],
    )
    function_raw = inspect.getsource(
        fused_add_rms_norm_with_forced_bf16_boundary
    ).encode("utf-8")
    report["evidence_authority"] = {
        "capsule_input_sha256": CAPSULE_INPUT_SHA256,
        "capsule_state_sha256": CAPSULE_STATE_SHA256,
        "protected_output_sha256": PROTECTED_OUTPUT_SHA256,
        "scalar_frontier_artifact_sha256": SCALAR_ARTIFACT_SHA256,
    }
    report["source_authority"] = {
        "analysis_module_path": str(ANALYSIS_MODULE.relative_to(WORKTREE)),
        "analysis_module_sha256": sha256(analysis_source_raw).hexdigest(),
        "function": "fused_add_rms_norm_with_forced_bf16_boundary",
        "function_module_path": str(KERNEL_MODULE.relative_to(WORKTREE)),
        "function_module_sha256": sha256(kernel_source_raw).hexdigest(),
        "function_source_sha256": sha256(function_raw).hexdigest(),
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
