#!/usr/bin/env python3
"""Emit the exact CPU-only Gate-D RMS conversion-placement certificate."""

from __future__ import annotations

import json
import os
import stat
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

from glm_tpu.greenfield.benchmarking.gate_d_scalar_frontier import (
    EXPECTED_INPUT_SHA256,
    EXPECTED_OUTPUT_SHA256,
    EXPECTED_STATE_SHA256,
    ScalarFrontierError,
    analyze_scalar_frontier,
)

RUN_ROOT = Path("/home/gianl/gate-d-runs")
CAPSULE = RUN_ROOT / "greenfield_gate_d_compensated_capsule_20260831T124838Z"
SOURCE = RUN_ROOT / "gate_d_compensated_pp16_numerical_20260901T094622505067868Z"
RECOVERY = RUN_ROOT / ("gate_d_compensated_pp16_recovery_20260901T102415964684137Z")
RECOVERY_TERMINAL_SHA256 = (
    "a3a4a0ec674eb3cd9f74abff72c3a4f1b8803b5c51726c267b98482943aad959"
)
RECOVERY_AUTHENTICATION_SHA256 = (
    "679e4a7354aa3a0afe716a5e183afb7a93a5cbfe27beed16f27e1aba0aa51433"
)
RECOVERY_REMOTE_OBJECTS_SHA256 = (
    "0fda2ed70ee7257ba33f5bf49c557f3b45a59a846febb4349826c06fdc8a8adc"
)
RECOVERY_TERMINAL_GENERATION = "1788258353080631"


def _snapshot(path: Path, *, limit: int = 64 << 20) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise ScalarFrontierError(f"unsafe scalar-frontier input: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 8 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        stable = lambda value: (
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
            or stable(before) != stable(after)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ScalarFrontierError(f"scalar-frontier input changed: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _require_hash(raw: bytes, expected: str, label: str) -> None:
    if sha256(raw).hexdigest() != expected:
        raise ScalarFrontierError(f"{label} bytes drifted")


def _archive(raw: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def main() -> int:
    if os.environ.get("GLM_GATE_D_SCALAR_FRONTIER") != "1":
        raise ScalarFrontierError("scalar-frontier analysis is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ScalarFrontierError("scalar-frontier analysis must be CPU-pinned")
    terminal_raw = _snapshot(
        RECOVERY / "NUMERICAL_REJECTED_RECOVERED.json", limit=1 << 20
    )
    authentication_raw = _snapshot(
        RECOVERY / "source_authentication.json", limit=1 << 20
    )
    remote_objects_raw = _snapshot(RECOVERY / "remote_objects.json", limit=1 << 20)
    _require_hash(terminal_raw, RECOVERY_TERMINAL_SHA256, "recovery terminal")
    _require_hash(
        authentication_raw,
        RECOVERY_AUTHENTICATION_SHA256,
        "recovery authentication",
    )
    _require_hash(
        remote_objects_raw, RECOVERY_REMOTE_OBJECTS_SHA256, "recovery remote ledger"
    )
    terminal = json.loads(terminal_raw)
    preterminal_generation = terminal.get("preterminal_remote_ledger", {}).get(
        "generation"
    )
    if (
        terminal.get("status") != "NUMERICAL_REJECTED_RECOVERED"
        or terminal.get("preterminal_remote_ledger_sha256")
        != RECOVERY_REMOTE_OBJECTS_SHA256
        or not isinstance(preterminal_generation, str)
        or not preterminal_generation.isdecimal()
        or int(preterminal_generation) >= int(RECOVERY_TERMINAL_GENERATION)
    ):
        raise ScalarFrontierError("recovery terminal authority drifted")

    input_raw = _snapshot(CAPSULE / "candidate-inputs.npz")
    state_raw = _snapshot(CAPSULE / "candidate-state.npz")
    output_raw = _snapshot(SOURCE / "outputs.npz")
    _require_hash(input_raw, EXPECTED_INPUT_SHA256, "capsule inputs")
    _require_hash(state_raw, EXPECTED_STATE_SHA256, "capsule state")
    _require_hash(output_raw, EXPECTED_OUTPUT_SHA256, "protected outputs")
    inputs = _archive(input_raw)
    state = _archive(state_raw)
    outputs = _archive(output_raw)
    report = analyze_scalar_frontier(
        hidden_update_bits=inputs["rms_hidden_update_bf16_bits"],
        residual_bits=inputs["rms_residual_bf16_bits"],
        weight_bits=inputs["rms_weight_bf16_bits"],
        accepted_rms_input=state["rms_input"],
        observed_rms_input_owners=outputs["rms_input_fp32_owners"][:, 0],
        accepted_output_owners=state["normalized"][:, 0],
        observed_output_owners=outputs["normalized_hidden_owners"][:, 0],
    )
    report["evidence_authority"] = {
        "capsule_input_sha256": EXPECTED_INPUT_SHA256,
        "capsule_state_sha256": EXPECTED_STATE_SHA256,
        "protected_output_sha256": EXPECTED_OUTPUT_SHA256,
        "recovery_authentication_sha256": RECOVERY_AUTHENTICATION_SHA256,
        "recovery_remote_objects_sha256": RECOVERY_REMOTE_OBJECTS_SHA256,
        "recovery_terminal_generation": RECOVERY_TERMINAL_GENERATION,
        "recovery_terminal_sha256": RECOVERY_TERMINAL_SHA256,
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
