#!/usr/bin/env python3
"""Run one protected two-chip Gate-D compensated numerical discriminator.

This default-off driver is intentionally narrower than a decoder run.  It
authenticates the accepted compile-only HLO boundary, reuses the exact sealed
capsule inputs, recompiles the unchanged PP16 stage-zero graph, requires the
previously adjudicated StableHLO and optimized-HLO hashes, and only then
invokes the compiled executable exactly once.  A bounded result tree is copied
to the host once and classified against the independent accepted TPU event-1
authority.  Either acceptance or rejection keeps Gate D open and makes no
performance claim.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import subprocess
import sys
import time
import types
import zipfile
from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
DRIVER_REPOSITORY_PATH = "scripts/greenfield/run_gate_d_compensated_pp16_numerical.py"
COMPILE_HELPER_REPOSITORY_PATH = (
    "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py"
)
COMPILE_HELPER_SHA256 = (
    "a4599e0d88a1ef8a2df897d1677196e8cd0db5009f12b866e7316ecaadb5dd15"
)
HLO_SOURCE_CODE_HASH = "94518b7d4ce788157afa98b7bc1f8144613852d5"
HLO_ADJUDICATION_SHA256 = (
    "bb04e959fd752fed8c5e8befd94875d5f81086dba5820d43212e87e3628cac04"
)
ADMISSION_SHA256 = "7cd7e569ed9ed5fd978d933efd4229d65906ae28312e264863b8336e4cc6b37d"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
CAPSULE_SHA256 = "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4"
CAPSULE_INPUT_SHA256 = (
    "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
)
CAPSULE_STATE_SHA256 = (
    "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236"
)
CAPSULE_EXECUTION_AUTHORITY_SHA256 = (
    "8b8c9cc79a18679628582a66c418a7f63e06553091928df58defbdde3971a660"
)
HOST_MATERIALIZATION_AUTHORITY_SHA256 = (
    "a7e5b393f7c61181f6b2aab9531d788fb27594d9cf980055c165b6f278ba4f99"
)
EXPECTED_QUERY_WEIGHT_FP32_SHA256 = (
    "e460e727c123856b92fe45f63c2ca2f332a988965324cc824cda2c0fcd51338f"
)
EXPECTED_WK_WEIGHT_FP32_SHA256 = (
    "b2c67e0fdf4d7292494778233e9813d8256f2fb88b6f7376870379832fbab24b"
)
EXPECTED_QUERY_WEIGHT_SHAPE = (2, 2048, 2048)
EXPECTED_WK_WEIGHT_SHAPE = (2, 128, 6144)
EXPECTED_HOST_MATERIALIZATION_RECORDED_AT_UTC = "2026-09-01T08:20:15.649219275Z"
EXPECTED_STABLEHLO_SHA256 = (
    "55d7940c2aa9f0f7cda463cb816f3a5475985aab89050a5aafc22c2c88792a83"
)
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "b63623498d82f67824b3be8998c09448440870b753cc1aa95d7a7765422c692b"
)
EXPECTED_ACCEPTED_POSITIONS_SHA256 = (
    "e55e66c6dcb35de94b9dce54d8fff602704cf26ae92d4afb333bd2501ab88ad7"
)
EXPECTED_ACCEPTED_SCORES_SHA256 = (
    "a61587a9d18bd169c1697ecb0060b39e8f1aad1caf532bca835b15a426c0b0e7"
)
EXPECTED_VALID_COUNT = 2048
TAG_PATTERN = r"gate_d_compensated_pp16_numerical_[0-9]{8}T[0-9]{15}Z"
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_F_GET_SEALS = 1034
_MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES = 32 * 1024 * 1024


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _array_sha256(value: Any) -> str:
    import numpy as np

    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _snapshot_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe regular-file identity: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"regular file changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _git_bytes(*arguments: str, repo: Path = WORKTREE) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )


def _git_text(*arguments: str, repo: Path = WORKTREE) -> str:
    return _git_bytes(*arguments, repo=repo).decode("ascii", errors="strict").strip()


def _load_compile_helper(code_pin: str) -> types.ModuleType:
    """Execute the exact audited helper bytes without a pathname import race."""

    path = WORKTREE / COMPILE_HELPER_REPOSITORY_PATH
    raw = _snapshot_regular(path)
    committed = _git_bytes("show", f"{code_pin}:{COMPILE_HELPER_REPOSITORY_PATH}")
    historical = _git_bytes(
        "show", f"{HLO_SOURCE_CODE_HASH}:{COMPILE_HELPER_REPOSITORY_PATH}"
    )
    if (
        raw != committed
        or raw != historical
        or sha256(raw).hexdigest() != COMPILE_HELPER_SHA256
    ):
        raise RuntimeError("audited Gate-D compile helper bytes drifted")
    module = types.ModuleType("_gate_d_exact_compile_helper")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    module.REPO = WORKTREE
    module.RUN_ROOT = RUN_ROOT
    module.TAG_PATTERN = TAG_PATTERN
    return module


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    if _git_text("rev-parse", "HEAD") != code_pin:
        raise RuntimeError("Gate-D numerical code pin drifted")
    raw = _snapshot_regular(Path(__file__))
    if (
        raw != _git_bytes("show", f"{code_pin}:{DRIVER_REPOSITORY_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
        or _git_text("rev-parse", f"{code_pin}^{{commit}}") != code_pin
    ):
        raise RuntimeError("Gate-D numerical driver is not the committed blob")
    if _git_text("rev-parse", f"{code_pin}:glm_tpu") != _git_text(
        "rev-parse", f"{HLO_SOURCE_CODE_HASH}:glm_tpu"
    ):
        raise RuntimeError(
            "numerical graph source tree differs from acquired HLO source"
        )


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = _snapshot_regular(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"Gate-D {label} bytes drifted")
    value = json.loads(raw.decode("ascii", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError(f"Gate-D {label} must be one JSON object")
    return value


def validate_hlo_adjudication(report: Mapping[str, Any]) -> dict[str, str]:
    """Fail closed unless the exact non-numerical HLO boundary was accepted."""

    expected_classification = (
        "HLO_LOCALITY_POLICY_ACCEPTED;TPU_NUMERICAL_UNPROVEN;"
        "NO_PERFORMANCE_CLAIM;GATE_D_OPEN"
    )
    hlo = report.get("hlo")
    optimized = hlo.get("optimized") if isinstance(hlo, Mapping) else None
    stablehlo = hlo.get("stablehlo") if isinstance(hlo, Mapping) else None
    if (
        set(report)
        != {
            "artifact_kind",
            "claim_scope",
            "classification",
            "code_hash",
            "gate_d_closed",
            "hlo",
            "numerical_claim",
            "optimized_hlo_locality_policy_accepted",
            "performance_claim",
            "provenance",
            "remote_archive",
            "run_tag",
            "schema_version",
            "tpu_numerical_execution_performed",
            "tpu_numerical_proven",
            "tpu_successor_authorized",
            "validator",
        }
        or report.get("artifact_kind") != "gate_d_compensated_pp16_hlo_adjudication"
        or report.get("classification") != expected_classification
        or report.get("code_hash") != HLO_SOURCE_CODE_HASH
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("gate_d_closed") is not False
        or report.get("numerical_claim") is not False
        or report.get("performance_claim") is not False
        or report.get("optimized_hlo_locality_policy_accepted") is not True
        or report.get("tpu_numerical_execution_performed") is not False
        or report.get("tpu_numerical_proven") is not False
        or report.get("tpu_successor_authorized") is not False
        or not isinstance(optimized, Mapping)
        or not isinstance(stablehlo, Mapping)
        or optimized.get("sha256") != EXPECTED_OPTIMIZED_HLO_SHA256
        or stablehlo.get("sha256") != EXPECTED_STABLEHLO_SHA256
        or type(optimized.get("collective_count")) is not int
        or optimized.get("collective_count") != 3
        or type(optimized.get("num_partitions")) is not int
        or optimized.get("num_partitions") != 2
        or type(optimized.get("logical_row_axis")) is not int
        or optimized.get("logical_row_axis") != 1
        or optimized.get("rooted_compensated_witness") is not True
        or optimized.get("primary_noninterference") is not True
        or type(stablehlo.get("logical_rows")) is not int
        or stablehlo.get("logical_rows") != 1
        or type(stablehlo.get("mesh_size")) is not int
        or stablehlo.get("mesh_size") != 2
    ):
        raise RuntimeError("Gate-D PP16 HLO adjudication boundary drifted")
    collectives = optimized.get("collectives")
    if not isinstance(collectives, list) or any(
        not isinstance(item, Mapping)
        or item.get("opcode") != "all-gather"
        or item.get("replica_groups") != [[0, 1]]
        or item.get("use_global_device_ids") is not True
        for item in collectives
    ):
        raise RuntimeError("Gate-D PP16 HLO locality catalogue drifted")
    return {
        "optimized_hlo_sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
        "stablehlo_sha256": EXPECTED_STABLEHLO_SHA256,
    }


def validate_numerical_policy(admission: Mapping[str, Any]) -> dict[str, Any]:
    policy = admission.get("precompile_numerical_policy")
    accepted = (
        policy.get("accepted_tpu_equality") if isinstance(policy, Mapping) else None
    )
    outputs = accepted.get("event1_outputs") if isinstance(accepted, Mapping) else None
    checks = (
        accepted.get("required_future_checks")
        if isinstance(accepted, Mapping)
        else None
    )
    required = {
        "identical_protected_input_and_event_identity",
        "exact_event1_selected_positions",
        "exact_event1_selected_scores",
        "exact_event1_valid_count",
        "descending_score_then_lowest_global_position_tie_order",
    }
    if (
        not isinstance(outputs, Mapping)
        or outputs.get("positions_sha256") != EXPECTED_ACCEPTED_POSITIONS_SHA256
        or outputs.get("scores_sha256") != EXPECTED_ACCEPTED_SCORES_SHA256
        or type(outputs.get("valid_count")) is not int
        or outputs.get("valid_count") != EXPECTED_VALID_COUNT
        or accepted.get("status") != "deferred.mandatory.protected_tpu_numerical_replay"
        or not isinstance(checks, list)
        or not required.issubset(set(checks))
    ):
        raise RuntimeError("Gate-D accepted TPU numerical policy drifted")
    return {
        "positions_sha256": EXPECTED_ACCEPTED_POSITIONS_SHA256,
        "scores_sha256": EXPECTED_ACCEPTED_SCORES_SHA256,
        "valid_count": EXPECTED_VALID_COUNT,
    }


def validate_host_materialization_authority(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the CPU-only proof for the two host-derived graph inputs."""

    backend = report.get("backend")
    comparison = report.get("comparison")
    expected_classification = (
        "HOST_DERIVED_INPUTS_BYTE_EXACT;TPU_NUMERICAL_UNPROVEN;GATE_D_OPEN"
    )
    if (
        set(report)
        != {
            "artifact_kind",
            "backend",
            "capsule_input_sha256",
            "classification",
            "comparison",
            "gate_d_closed",
            "libtpu_lock_holder_after_test",
            "performance_claim",
            "recorded_at_utc",
            "schema_version",
            "tpu_backend_initialized",
            "tpu_execution_performed",
        }
        or report.get("artifact_kind")
        != "gate_d_pp16_numerical_host_materialization_equivalence"
        or report.get("capsule_input_sha256") != CAPSULE_INPUT_SHA256
        or report.get("classification") != expected_classification
        or report.get("recorded_at_utc")
        != EXPECTED_HOST_MATERIALIZATION_RECORDED_AT_UTC
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("gate_d_closed") is not False
        or report.get("libtpu_lock_holder_after_test") is not False
        or report.get("performance_claim") is not False
        or report.get("tpu_backend_initialized") is not False
        or report.get("tpu_execution_performed") is not False
        or not isinstance(backend, Mapping)
        or set(backend) != {"device_count", "jax_platform", "xla_flags"}
        or type(backend.get("device_count")) is not int
        or backend.get("device_count") != 2
        or backend.get("jax_platform") != "cpu"
        or backend.get("xla_flags") != "--xla_force_host_platform_device_count=2"
        or not isinstance(comparison, Mapping)
        or set(comparison)
        != {
            "host_query_matches_sealed_jax_reference",
            "host_wk_bf16_then_fp32_matches_sealed_jax_reference",
            "query_weight_fp32_sha256",
            "wk_weight_fp32_sha256",
        }
        or comparison.get("host_query_matches_sealed_jax_reference") is not True
        or comparison.get("host_wk_bf16_then_fp32_matches_sealed_jax_reference")
        is not True
        or comparison.get("query_weight_fp32_sha256")
        != EXPECTED_QUERY_WEIGHT_FP32_SHA256
        or comparison.get("wk_weight_fp32_sha256") != EXPECTED_WK_WEIGHT_FP32_SHA256
    ):
        raise RuntimeError("Gate-D host-materialization authority drifted")
    return {
        "authority_sha256": HOST_MATERIALIZATION_AUTHORITY_SHA256,
        "backend": "cpu",
        "capsule_input_sha256": CAPSULE_INPUT_SHA256,
        "query_weight": {
            "array_sha256": EXPECTED_QUERY_WEIGHT_FP32_SHA256,
            "semantic_dtype": "float32",
            "shape": list(EXPECTED_QUERY_WEIGHT_SHAPE),
        },
        "wk_weight": {
            "array_sha256": EXPECTED_WK_WEIGHT_FP32_SHA256,
            "semantic_dtype": "float32",
            "shape": list(EXPECTED_WK_WEIGHT_SHAPE),
        },
    }


def _validate_capsule_bindings(
    capsule: Mapping[str, Any],
    authority: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = admission.get("candidate_results")
    if not isinstance(candidates, list):
        raise TypeError("Gate-D candidate catalogue is absent")
    matching = [
        item
        for item in candidates
        if isinstance(item, Mapping)
        and item.get("id") == "compensated_auxiliary_dependency"
    ]
    admitted_input_arrays = (
        matching[0].get("capsule", {}).get("producer", {}).get("input_arrays")
        if len(matching) == 1
        else None
    )
    if (
        type(capsule.get("schema_version")) is not int
        or capsule.get("schema_version") != 2
        or capsule.get("candidate_id") != "compensated_auxiliary_dependency"
        or capsule.get("coherence_id")
        != "greenfield_gate_d_compensated_capsule_20260831T124838Z"
        or capsule.get("producer_input_artifact")
        != {"path": "candidate-inputs.npz", "sha256": CAPSULE_INPUT_SHA256}
        or capsule.get("artifact")
        != {"path": "candidate-state.npz", "sha256": CAPSULE_STATE_SHA256}
        or type(authority.get("schema_version")) is not int
        or authority.get("schema_version") != 1
        or authority.get("authority_kind") != "gate.d.capsule.execution.v1"
        or authority.get("candidate_id") != "compensated_auxiliary_dependency"
        or authority.get("input_arrays") is None
        or authority.get("input_arrays") != admitted_input_arrays
    ):
        raise RuntimeError("Gate-D compensated capsule binding drifted")
    return {
        "capsule_sha256": CAPSULE_SHA256,
        "execution_authority_sha256": CAPSULE_EXECUTION_AUTHORITY_SHA256,
        "input_sha256": CAPSULE_INPUT_SHA256,
        "state_sha256": CAPSULE_STATE_SHA256,
    }


def _load_npz_arrays(raw: bytes, records: Mapping[str, Any], np: Any) -> dict[str, Any]:
    expected_members = {f"{name}.npy" for name in records}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        infos = archive.infolist()
        if (
            len(names) != len(set(names))
            or set(names) != expected_members
            or sum(info.file_size for info in infos) > _MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES
            or any(
                info.is_dir()
                or info.flag_bits & 0x1
                or info.file_size > _MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES
                for info in infos
            )
        ):
            raise RuntimeError("Gate-D capsule NPZ member catalogue drifted")
    result: dict[str, Any] = {}
    with np.load(BytesIO(raw), allow_pickle=False) as values:
        if set(values.files) != set(records):
            raise RuntimeError("Gate-D capsule NPZ array catalogue drifted")
        for name in sorted(records):
            record = records[name]
            value = np.ascontiguousarray(values[name])
            if (
                not isinstance(record, Mapping)
                or list(value.shape) != record.get("shape")
                or value.dtype.str != record.get("storage_dtype")
                or _array_sha256(value) != record.get("array_sha256")
            ):
                raise RuntimeError(f"Gate-D capsule array drifted: {name}")
            result[name] = value
    return result


def _fp8_lookup(np: Any) -> Any:
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
    return np.asarray(values, dtype=np.float32)


def materialize_derived_weights(
    inputs: Mapping[str, Any], np: Any, ml_dtypes: Any
) -> tuple[Any, Any]:
    """Materialize the two compile inputs on host; this performs no TPU work."""

    lookup = _fp8_lookup(np)

    def decode(bits: Any, scales: Any) -> Any:
        rows = np.arange(bits.shape[-2]) // 128
        columns = np.arange(bits.shape[-1]) // 128
        expanded = scales[..., rows[:, None], columns[None, :]]
        return lookup[bits.astype(np.int32)] * expanded.astype(np.float32)

    query = np.ascontiguousarray(
        decode(inputs["wq_b_weight_bits"], inputs["wq_b_scale_inv"]),
        dtype=np.float32,
    )
    wk_bf16 = np.ascontiguousarray(
        decode(inputs["wk_weight_bits"], inputs["wk_scale_inv"]),
        dtype=ml_dtypes.bfloat16,
    )
    wk = np.ascontiguousarray(wk_bf16.astype(np.float32), dtype=np.float32)
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(wk)):
        raise RuntimeError("Gate-D materialized FP8 input is non-finite")
    return query, wk


def validate_derived_weights(query: Any, wk: Any, np: Any) -> dict[str, Any]:
    """Require exact real-capsule derived bytes before graph lowering."""

    if not isinstance(query, np.ndarray) or not isinstance(wk, np.ndarray):
        raise TypeError("Gate-D host-derived graph inputs must be NumPy arrays")
    identities = {
        "query_weight": {
            "array_sha256": _array_sha256(query),
            "semantic_dtype": str(query.dtype),
            "shape": list(query.shape),
        },
        "wk_weight": {
            "array_sha256": _array_sha256(wk),
            "semantic_dtype": str(wk.dtype),
            "shape": list(wk.shape),
        },
    }
    if (
        query.dtype != np.dtype(np.float32)
        or wk.dtype != np.dtype(np.float32)
        or query.shape != EXPECTED_QUERY_WEIGHT_SHAPE
        or wk.shape != EXPECTED_WK_WEIGHT_SHAPE
        or identities["query_weight"]["array_sha256"]
        != EXPECTED_QUERY_WEIGHT_FP32_SHA256
        or identities["wk_weight"]["array_sha256"] != EXPECTED_WK_WEIGHT_FP32_SHA256
        or not np.all(np.isfinite(query))
        or not np.all(np.isfinite(wk))
    ):
        raise RuntimeError("Gate-D host-derived graph input identity drifted")
    return identities


def _execution_arguments(
    inputs: Mapping[str, Any], query: Any, wk: Any, ml_dtypes: Any
) -> tuple[Any, ...]:
    bf16 = ml_dtypes.bfloat16
    return (
        inputs["rms_hidden_update_bf16_bits"].view(bf16)[None, :],
        inputs["rms_residual_bf16_bits"].view(bf16)[None, :],
        inputs["rms_weight_bf16_bits"].view(bf16),
        inputs["prompt_cache_bf16_bits"].view(bf16),
        inputs["qkv_a_weight_bits"],
        inputs["qkv_a_scale_inv"],
        inputs["q_a_norm_bf16_bits"].view(bf16),
        inputs["wq_b_weight_bits"],
        inputs["wq_b_scale_inv"],
        query,
        inputs["wk_weight_bits"],
        inputs["wk_scale_inv"],
        wk,
        inputs["key_norm_weight_bf16_bits"].view(bf16),
        inputs["key_norm_bias_bf16_bits"].view(bf16),
        inputs["head_weight_bf16_bits"].view(bf16),
    )


def _deterministic_npz(values: Mapping[str, Any], np: Any) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(values):
            payload = BytesIO()
            np.lib.format.write_array(
                payload, np.ascontiguousarray(values[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100400 << 16
            archive.writestr(info, payload.getvalue())
    return output.getvalue()


def _owner_equal(value: Any, np: Any) -> bool:
    array = np.ascontiguousarray(value)
    return array.shape[0] == 2 and array[0].tobytes(order="C") == array[1].tobytes(
        order="C"
    )


def _validate_output_spec(host: Mapping[str, Any], output_spec: Any) -> None:
    expected = {name: (tuple(shape), dtype) for name, shape, dtype in output_spec}
    if set(host) != set(expected):
        raise RuntimeError("Gate-D numerical output catalogue drifted")
    for name, value in host.items():
        shape, dtype = expected[name]
        if tuple(value.shape) != shape or str(value.dtype) != dtype:
            raise RuntimeError(
                f"Gate-D numerical output spec drifted: {name} "
                f"{value.shape} {value.dtype}"
            )


def _candidate_watchpoint_matches(
    capsule: Mapping[str, Any],
    host: Mapping[str, Any],
    inputs: Mapping[str, Any],
    np: Any,
) -> dict[str, bool]:
    state = {
        "cache_history": np.ascontiguousarray(host["index_cache_owners"]).view(
            np.uint16
        ),
        "current_key": np.ascontiguousarray(
            host["current_key_owners"][0, 0], dtype=np.float32
        ),
        "event1_positions": np.ascontiguousarray(
            host["selected_positions_owners"][0], dtype=np.int32
        ),
        "event1_scores": np.ascontiguousarray(
            host["selected_scores_owners"][0], dtype=np.float32
        ),
        "event1_valid_count": np.ascontiguousarray(
            host["valid_counts_owners"][0], dtype=np.int32
        ),
        "head_weights": np.ascontiguousarray(
            host["head_weights_owners"], dtype=np.float32
        ),
        "normalized": np.ascontiguousarray(host["normalized_hidden_owners"]).view(
            np.uint16
        ),
        "query": np.ascontiguousarray(host["query_owners"], dtype=np.float32),
        "rms_hidden_update": inputs["rms_hidden_update_bf16_bits"],
        "rms_input": np.ascontiguousarray(
            host["rms_input_fp32_owners"][0, 0], dtype=np.float32
        ),
        "rms_residual": inputs["rms_residual_bf16_bits"],
    }
    watchpoints = capsule.get("watchpoints")
    if not isinstance(watchpoints, list) or len(watchpoints) != 8:
        raise RuntimeError("Gate-D capsule watchpoint catalogue drifted")
    matches: dict[str, bool] = {}
    for watchpoint in watchpoints:
        if not isinstance(watchpoint, Mapping):
            raise TypeError("Gate-D capsule watchpoint record drifted")
        watchpoint_id = watchpoint.get("id")
        arrays = watchpoint.get("arrays")
        if not isinstance(watchpoint_id, str) or not isinstance(arrays, list):
            raise TypeError("Gate-D capsule watchpoint schema drifted")
        for record in arrays:
            if not isinstance(record, Mapping):
                raise TypeError("Gate-D capsule watchpoint array drifted")
            key = record.get("array_key")
            prefix = record.get("index_prefix")
            role = record.get("role")
            if (
                not isinstance(key, str)
                or key not in state
                or not isinstance(prefix, list)
                or not all(type(item) is int and item >= 0 for item in prefix)
                or not isinstance(role, str)
            ):
                raise RuntimeError("Gate-D capsule watchpoint selector drifted")
            value = state[key][tuple(prefix)] if prefix else state[key]
            observed = _array_sha256(value)
            label = f"{watchpoint_id}:{role}"
            if label in matches:
                raise RuntimeError("Gate-D capsule watchpoint label is duplicated")
            matches[label] = observed == record.get("array_sha256")
    if len(matches) != 14:
        raise RuntimeError("Gate-D capsule watchpoint array count drifted")
    return dict(sorted(matches.items()))


def _tie_order_is_exact(positions: Any, scores: Any, valid_count: int, np: Any) -> bool:
    position = np.ascontiguousarray(positions).reshape(-1)
    score = np.ascontiguousarray(scores).reshape(-1)
    if (
        valid_count != EXPECTED_VALID_COUNT
        or position.shape != (EXPECTED_VALID_COUNT,)
        or score.shape != (EXPECTED_VALID_COUNT,)
        or not np.all(np.isfinite(score))
        or np.any(position < 0)
        or np.any(position >= 8156)
        or np.unique(position).size != position.size
    ):
        return False
    for index in range(position.size - 1):
        if score[index] < score[index + 1]:
            return False
        if score[index] == score[index + 1] and position[index] > position[index + 1]:
            return False
    return True


def classify_outputs(
    host: Mapping[str, Any],
    capsule: Mapping[str, Any],
    inputs: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    owner_fields = (
        "rms_input_fp32_owners",
        "normalized_hidden_owners",
        "query_owners",
        "head_weights_owners",
        "current_key_owners",
        "selected_positions_owners",
        "valid_counts_owners",
        "selected_scores_owners",
    )
    owner_agreement = {name: _owner_equal(host[name], np) for name in owner_fields}
    contract = np.ascontiguousarray(host["contract_valid_owners"], dtype=np.uint8)
    positions = np.ascontiguousarray(
        host["selected_positions_owners"][0], dtype=np.int32
    )
    scores = np.ascontiguousarray(host["selected_scores_owners"][0], dtype=np.float32)
    counts = np.ascontiguousarray(host["valid_counts_owners"][0], dtype=np.int32)
    count = int(counts.reshape(-1)[0]) if counts.size == 1 else -1
    observed = {
        "positions_sha256": _array_sha256(positions),
        "scores_sha256": _array_sha256(scores),
        "valid_count": count,
    }
    tie_order = _tie_order_is_exact(positions, scores, count, np)
    accepted = (
        all(owner_agreement.values())
        and contract.shape == (2,)
        and contract.tobytes(order="C") == b"\x01\x01"
        and tie_order
        and observed
        == {
            "positions_sha256": EXPECTED_ACCEPTED_POSITIONS_SHA256,
            "scores_sha256": EXPECTED_ACCEPTED_SCORES_SHA256,
            "valid_count": EXPECTED_VALID_COUNT,
        }
    )
    return {
        "accepted_tpu_event_match": accepted,
        "contract_valid": contract.tolist(),
        "cpu_candidate_watchpoint_matches_diagnostic_only": (
            _candidate_watchpoint_matches(capsule, host, inputs, np)
        ),
        "observed_event1": observed,
        "owner_agreement": owner_agreement,
        "tie_order_exact": tie_order,
    }


def _host_outputs(result: Any, jax: Any, np: Any) -> dict[str, Any]:
    transferred = jax.device_get(result)
    names = tuple(result._fields)
    host = {
        name: np.ascontiguousarray(value)
        for name, value in zip(names, transferred, strict=True)
    }
    return host


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--hlo-adjudication", type=Path, required=True)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--topology-authority", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--capsule-inputs", type=Path, required=True)
    parser.add_argument("--capsule-state", type=Path, required=True)
    parser.add_argument("--capsule-execution-authority", type=Path, required=True)
    parser.add_argument("--host-materialization-authority", type=Path, required=True)
    parser.add_argument("--execute-once", type=int, choices=(1,), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-dir-fd", type=int, choices=(7,), required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        raise RuntimeError("Gate-D numerical replay requires Python -I -S")
    helper = _load_compile_helper(args.expected_code_hash)
    python_runtime = helper._validate_python_runtime()
    dependency_sites = helper._validate_dependency_sites()
    compiler_environment = helper._validate_environment()
    _verify_running_source(args.expected_code_hash, args.expected_driver_sha256)

    admission = _load_bound_json(
        args.admission_report, ADMISSION_SHA256, "admission report"
    )
    topology = _load_bound_json(
        args.topology_authority, TOPOLOGY_SHA256, "topology authority"
    )
    stage_zero = helper.validate_admission_and_topology(admission, topology)
    helper.validate_capsule_input_spec(admission)
    policy = validate_numerical_policy(admission)
    hlo_report = _load_bound_json(
        args.hlo_adjudication, HLO_ADJUDICATION_SHA256, "HLO adjudication"
    )
    hlo_authority = validate_hlo_adjudication(hlo_report)
    capsule = _load_bound_json(args.capsule, CAPSULE_SHA256, "capsule")
    authority = _load_bound_json(
        args.capsule_execution_authority,
        CAPSULE_EXECUTION_AUTHORITY_SHA256,
        "capsule execution authority",
    )
    host_materialization_report = _load_bound_json(
        args.host_materialization_authority,
        HOST_MATERIALIZATION_AUTHORITY_SHA256,
        "host-materialization authority",
    )
    host_materialization_authority = validate_host_materialization_authority(
        host_materialization_report
    )
    capsule_binding = _validate_capsule_bindings(capsule, authority, admission)
    input_raw = _snapshot_regular(args.capsule_inputs)
    state_raw = _snapshot_regular(args.capsule_state)
    if sha256(input_raw).hexdigest() != CAPSULE_INPUT_SHA256:
        raise RuntimeError("Gate-D capsule input bytes drifted")
    if sha256(state_raw).hexdigest() != CAPSULE_STATE_SHA256:
        raise RuntimeError("Gate-D capsule state bytes drifted")
    run_fd = helper._open_inherited_run_dir(args.run_dir, args.run_dir_fd)

    source_fd, source_path, source_snapshot = helper._sealed_git_source_archive(
        args.expected_code_hash, repo=WORKTREE
    )
    helper._install_sealed_source_path(source_path)
    helper._validate_compiler_import_path(source_path)

    import jax
    import ml_dtypes
    import numpy as np

    from glm_tpu.greenfield.benchmarking.gate_d_compensated_pp16_hlo import (
        build_gate_d_compensated_pp16_hlo_replay,
    )

    project_modules = helper._verify_project_imports(source_path)
    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("Gate-D numerical replay requires one four-chip TPU-v4 host")
    if bool(jax.config.jax_enable_compilation_cache):
        raise RuntimeError("Gate-D numerical replay requires disabled JAX cache")
    devices = tuple(jax.local_devices()[:2])
    observed_group = {
        "coordinates": [list(device.coords) for device in devices],
        "device_ids": [int(device.id) for device in devices],
        "process_index": int(devices[0].process_index),
        "stage_id": 0,
    }
    if observed_group != stage_zero or any(
        int(device.process_index) != 0 or str(device.device_kind) != "TPU v4"
        for device in devices
    ):
        raise RuntimeError(f"Gate-D live PP16 topology drifted: {observed_group}")

    inputs = _load_npz_arrays(input_raw, authority["input_arrays"], np)
    query_weight, wk_weight = materialize_derived_weights(inputs, np, ml_dtypes)
    derived_weights = validate_derived_weights(query_weight, wk_weight, np)
    if derived_weights != {
        "query_weight": host_materialization_authority["query_weight"],
        "wk_weight": host_materialization_authority["wk_weight"],
    }:
        raise RuntimeError("Gate-D derived inputs differ from bound CPU authority")
    arguments = _execution_arguments(inputs, query_weight, wk_weight, ml_dtypes)
    replay = build_gate_d_compensated_pp16_hlo_replay(devices=devices)
    dependencies_before = helper._compiler_dependency_records(source_path)
    if not dependencies_before["accelerator_device_nodes_observed_mapped"]:
        raise RuntimeError("TPU accelerator device mapping is absent before compile")
    memory_before = [helper._memory_stats(device) for device in devices]
    lowered = replay.lower(*arguments)
    stablehlo = lowered.as_text()
    compiled = lowered.compile()
    optimized_hlo = compiled.as_text()
    observed_hlo = {
        "stablehlo_sha256": sha256(stablehlo.encode("utf-8")).hexdigest(),
        "optimized_hlo_sha256": sha256(optimized_hlo.encode("utf-8")).hexdigest(),
    }
    if observed_hlo != hlo_authority:
        raise RuntimeError(f"Gate-D executable HLO identity drifted: {observed_hlo}")
    helper._write_run_member_exclusive(
        run_fd, "hlo/compensated_pp16_stage0.stablehlo.mlir", stablehlo.encode("utf-8")
    )
    helper._write_run_member_exclusive(
        run_fd,
        "hlo/compensated_pp16_stage0.optimized_hlo.txt",
        optimized_hlo.encode("utf-8"),
    )
    memory_after_compile = [helper._memory_stats(device) for device in devices]

    invocation_count = 0
    started = time.monotonic_ns()
    invocation_count += 1
    result = compiled(*arguments)
    jax.block_until_ready(result)
    completed = time.monotonic_ns()
    host = _host_outputs(result, jax, np)
    _validate_output_spec(host, helper.OUTPUT_SPEC)
    if invocation_count != 1:
        raise AssertionError("Gate-D executable invocation count drifted")
    classification = classify_outputs(host, capsule, inputs, np)
    archived_host = {
        name: (
            np.ascontiguousarray(value).view(np.uint16)
            if str(value.dtype) == "bfloat16"
            else np.ascontiguousarray(value)
        )
        for name, value in host.items()
    }
    expected_semantic_dtypes = {
        name: dtype for name, _shape, dtype in helper.OUTPUT_SPEC
    }
    output_hashes = {
        name: {
            "array_sha256": _array_sha256(value),
            "semantic_dtype": expected_semantic_dtypes[name],
            "shape": list(value.shape),
            "storage_dtype": value.dtype.str,
        }
        for name, value in sorted(archived_host.items())
    }
    outputs_raw = _deterministic_npz(archived_host, np)
    output_identity = helper._write_run_member_exclusive(
        run_fd, "outputs.npz", outputs_raw
    )
    dependencies_after = helper._compiler_dependency_records(source_path)
    if not dependencies_after["accelerator_device_nodes_observed_mapped"]:
        raise RuntimeError("TPU accelerator device mapping is absent after execution")
    helper._require_dependency_prefix_stable(dependencies_before, dependencies_after)
    if helper._verify_project_imports(source_path) != project_modules:
        raise RuntimeError("Gate-D sealed project import closure drifted")
    if fcntl.fcntl(source_fd, _F_GET_SEALS) != _MEMFD_SEALS:
        raise RuntimeError("Gate-D sealed source archive drifted")
    helper._validate_compiler_import_path(source_path)
    if helper._validate_python_runtime_storage() != python_runtime:
        raise RuntimeError("Gate-D sealed Python runtime changed")
    if helper._validate_dependency_sites() != dependency_sites:
        raise RuntimeError("Gate-D dependency sites changed")

    dependency_manifest = {
        "accelerator_device_observation_scope": (
            helper._ACCELERATOR_DEVICE_OBSERVATION_SCOPE
        ),
        "accelerator_device_nodes_observed_mapped": dependencies_after[
            "accelerator_device_nodes_observed_mapped"
        ],
        "artifact_kind": "gate_d_compensated_pp16_numerical_dependencies",
        "code_hash": args.expected_code_hash,
        "dependency_sites": dependency_sites,
        "environment": compiler_environment,
        "native_mappings": dependencies_after["native_mappings"],
        "python_modules": dependencies_after["python_modules"],
        "python_runtime": python_runtime,
        "sealed_project_source": source_snapshot,
    }
    dependency_raw = (_canonical(dependency_manifest) + "\n").encode("ascii")
    dependency_identity = helper._write_run_member_exclusive(
        run_fd, "dependencies.json", dependency_raw
    )
    accepted = classification["accepted_tpu_event_match"]
    status = "NUMERICAL_ACCEPTED" if accepted else "NUMERICAL_REJECTED"
    claim = (
        "BOUNDED_TPU_NUMERICAL_ACCEPTED;FULL_DECODER_UNPROVEN;GATE_D_OPEN;"
        "NO_PERFORMANCE_CLAIM"
        if accepted
        else "BOUNDED_TPU_NUMERICAL_REJECTED;COMPENSATED_MECHANISM_TOMBSTONED;"
        "GATE_D_OPEN;NO_PERFORMANCE_CLAIM"
    )
    report = {
        "artifact_kind": "gate_d_compensated_pp16_numerical_replay",
        "capsule": capsule_binding,
        "claim_scope": (
            "One exact-input, one-invocation, two-chip PP16 stage-zero numerical "
            "discriminator. No decoder, token, performance or Gate-D closure claim."
        ),
        "classification": claim,
        "code_hash": args.expected_code_hash,
        "compiler_dependency_manifest": {
            **dependency_identity,
            "accelerator_device_node_count": len(
                dependencies_after["accelerator_device_nodes_observed_mapped"]
            ),
            "accelerator_device_nodes_sha256": (
                helper._accelerator_device_nodes_sha256(
                    dependencies_after["accelerator_device_nodes_observed_mapped"]
                )
            ),
            "filename": "dependencies.json",
            "native_mapping_count": len(dependencies_after["native_mappings"]),
            "python_module_count": len(dependencies_after["python_modules"]),
        },
        "compiled_executable_invocation_count": invocation_count,
        "gate_d_closed": False,
        "hlo": observed_hlo,
        "host_materialization": {
            **host_materialization_authority,
            "observed_derived_weights": derived_weights,
        },
        "host_transfer_count": 1,
        "memory_after_compile": memory_after_compile,
        "memory_after_execute": [helper._memory_stats(device) for device in devices],
        "memory_before_compile": memory_before,
        "numerical": classification,
        "numerical_claim_scope": "bounded_event1_only",
        "output_artifact": {**output_identity, "filename": "outputs.npz"},
        "output_arrays": output_hashes,
        "performance_claim": False,
        "persistent_compilation_cache_enabled": False,
        "physical_group": observed_group,
        "policy": policy,
        "runtime": {
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "libtpu": version("libtpu"),
            "ml_dtypes": version("ml_dtypes"),
            "numpy": version("numpy"),
        },
        "sealed_project_source": {
            **source_snapshot,
            "loaded_module_count": len(project_modules),
        },
        "schema_version": 1,
        "status": status,
        "tpu_execution_elapsed_ns_diagnostic_only": completed - started,
        "tpu_numerical_execution_performed": True,
    }
    helper._write_run_member_exclusive(
        run_fd, "runner.json", (_canonical(report) + "\n").encode("ascii")
    )
    os.close(run_fd)
    os.close(source_fd)
    print(
        f"GATE_D_COMPENSATED_PP16_{status} execution_count=1 "
        f"accepted_event={str(accepted).lower()} gate_d_open=true",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
