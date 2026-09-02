#!/usr/bin/env python3
"""Replay sealed layer-0 dense partials through two isolated RMS programs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking.gate_d_layer1_rms_schedule import (  # noqa: E402
    build_layer1_rms_schedule_arm,
)
from glm_tpu.greenfield.validation.gate_d_layer1_rms_schedule import (  # noqa: E402
    _compare_bits,
    _exact_accepted_rms_schedule,
    audit_layer1_rms_schedule_optimized_hlo,
)
from scripts.greenfield.probe_layer0_projection_reduction import (  # noqa: E402
    _file_sha256,
)


def _validate_captured_rms_optimized_hlo(
    optimized_hlo: str,
    *,
    split_layer1_rms: bool,
) -> dict[str, Any]:
    """Shared implementation: the accepted-schedule flag is the split arm."""

    result = audit_layer1_rms_schedule_optimized_hlo(
        optimized_hlo, fp32_carry_schedule=split_layer1_rms
    )
    result["split_layer1_rms"] = result.pop("fp32_carry_schedule")
    return result


def _build_arm(mesh: Any, *, split_layer1_rms: bool) -> Any:
    return build_layer1_rms_schedule_arm(mesh, fp32_carry_schedule=split_layer1_rms)


_POSITION = 8155
_CAPTURE_CODE_HASH = "c0ef9525bcd893bdad377af2b56e5ebd5fb13b34"
_DB548_CODE_HASH = "3d5b2ee840c41bcd86a1e1936a95db2d4a8e73f9"
_DB549_CODE_HASH = "57f6a052214e3394c86c2b2ebe0073f9989aca3b"
_CAPTURE_PARTIALS_SHA = (
    "9d9f65dddc7b622875872a33a6522c330c8fb5490c8cba14526553c211516e35"
)
_CAPTURE_RESIDUAL_SHA = (
    "f583581fe6cdd8f1cb437b7864070de9fdd39b83be01d2d4afe072a997042dc0"
)
_ATTENTION_UPDATE_SHA = (
    "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7"
)
_COMBINED_RESIDUAL_SHA = (
    "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3"
)
_LAYER1_NORM_SHA = (
    "10e34f4f99c638b29557526283205071c1ac8f81f168f4a6817e7e1def4b6c87"
)
_ACCEPTED_LAYER1_SHA = (
    "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
)
_DB548_OBSERVED_SHA = (
    "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
)
_DB549_OBSERVED_SHA = (
    "229dc8ace9bfa31fce6d6ccabc9fca49ccc55f30b9d1dd6f97a032f5117b812f"
)
_DENSE_UPDATE_SHA = (
    "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
)
_DB548_OPTIMIZED_HLO_SHA = (
    "5f4dd83793da67be6a8c580949920e93f8c64fe8205816738e7d04890640a877"
)
_DB549_OPTIMIZED_HLO_SHA = (
    "faf38fa987d174421d2631bc4921deee88bc30b50d557fbb02046c50fd9741c9"
)


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_file(path: Path, expected_sha: str, label: str) -> None:
    if not path.is_file() or _file_sha256(path) != expected_sha:
        raise RuntimeError(f"{label} SHA-256 drifted")


def _terminal_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or "=" not in line:
            raise RuntimeError(f"malformed terminal record: {path}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise RuntimeError(f"duplicate terminal field: {path}")
        values[key] = value
    return values


def _reproduce_dense_update(partial_bits: np.ndarray) -> np.ndarray:
    """Reduce the captured model-ordered BF16 partials with DB533's tree."""

    from glm_tpu.greenfield.kernels.stage_local import (
        STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
    )

    if partial_bits.dtype != np.uint16 or partial_bits.shape != (
        4,
        8,
        1,
        6144,
    ):
        raise RuntimeError("captured dense partial geometry drifted")
    model_partials = partial_bits.view(ml_dtypes.bfloat16).reshape(
        32, 1, 6144
    )

    def add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray(left + right, dtype=ml_dtypes.bfloat16)

    def reduce_four(values: np.ndarray, *, cross: bool) -> np.ndarray:
        if cross:
            return add(add(values[0], values[3]), add(values[1], values[2]))
        return add(add(values[0], values[1]), add(values[2], values[3]))

    physical = np.stack(
        tuple(
            model_partials[index]
            for index in STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE
        ),
        axis=0,
    )
    physical_y_x_z = np.transpose(
        physical.reshape(4, 4, 2, 1, 6144), (1, 2, 0, 3, 4)
    )
    y_reduced = np.concatenate(
        (
            reduce_four(physical_y_x_z[..., :2048], cross=False),
            reduce_four(
                physical_y_x_z[..., 2048:4096], cross=True
            ),
            reduce_four(physical_y_x_z[..., 4096:], cross=False),
        ),
        axis=-1,
    )
    x_reduced = add(y_reduced[0], y_reduced[1])
    reduced = np.concatenate(
        tuple(
            reduce_four(
                x_reduced[..., start : start + 256],
                cross=bool((start // 256) % 2),
            )
            for start in range(0, 6144, 256)
        ),
        axis=-1,
    )
    return np.ascontiguousarray(reduced).view(np.uint16)


def _load_sources(args: argparse.Namespace) -> tuple[np.ndarray, ...]:
    files = (
        (args.capture_tensor, args.capture_tensor_sha256, "capture tensor"),
        (args.capture_runner, args.capture_runner_sha256, "capture runner"),
        (args.capture_summary, args.capture_summary_sha256, "capture summary"),
        (args.capture_success, args.capture_success_sha256, "capture SUCCESS"),
        (args.db548_tensor, args.db548_tensor_sha256, "DB548 tensor"),
        (args.db548_runner, args.db548_runner_sha256, "DB548 runner"),
        (args.db548_summary, args.db548_summary_sha256, "DB548 summary"),
        (args.db548_success, args.db548_success_sha256, "DB548 SUCCESS"),
        (args.db548_hlo, args.db548_hlo_sha256, "DB548 optimized HLO"),
        (args.db549_runner, args.db549_runner_sha256, "DB549 runner"),
        (args.db549_tensor, args.db549_tensor_sha256, "DB549 tensor"),
        (args.db549_summary, args.db549_summary_sha256, "DB549 summary"),
        (args.db549_success, args.db549_success_sha256, "DB549 SUCCESS"),
        (args.db549_hlo, args.db549_hlo_sha256, "DB549 optimized HLO"),
    )
    for path, expected, label in files:
        _require_file(path, expected, label)

    capture = json.loads(args.capture_runner.read_text())
    capture_summary = json.loads(args.capture_summary.read_text())
    capture_success = _terminal_values(args.capture_success)
    capture_records = {
        "dense_virtual_partials_bfloat16_bits": {
            "axis_order": ["pp8_owner", "virtual_rank", "row", "hidden"],
            "dtype": "uint16",
            "sha256": _CAPTURE_PARTIALS_SHA,
            "shape": [4, 8, 1, 6144],
        },
        "post_attention_m32_bfloat16_bits": {
            "axis_order": ["row", "hidden"],
            "dtype": "uint16",
            "sha256": _CAPTURE_RESIDUAL_SHA,
            "shape": [32, 6144],
        },
    }
    capture_records_sha = sha256(
        json.dumps(
            capture_records, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if not (
        capture.get("artifact_kind") == "glm52_layer0_dense_partial_capture"
        and capture.get("classification")
        == "accepted_m32_dense_partials_captured"
        and capture.get("code_hash") == _CAPTURE_CODE_HASH
        and capture.get("compile_rows") == 32
        and capture.get("capture_partials") is True
        and capture.get("capture_records") == capture_records
        and capture.get("dense_envelope") is True
        and capture.get("final_dense_layout") is True
        and capture.get("split_layer1_rms") is False
        and capture.get("status") == "SUCCESS"
        and capture.get("performance_claim") is False
        and capture.get("hlo", {}).get("optimized_contract", {}).get("passed")
        is True
        and capture.get("hlo", {}).get("stablehlo_contract", {}).get("passed")
        is True
        and capture_summary.get("artifact_kind")
        == "glm52_layer0_dense_partial_capture"
        and capture_summary.get("capture_records") == capture_records
        and capture_summary.get("classification")
        == "accepted_m32_dense_partials_captured"
        and capture_summary.get("code_hash") == _CAPTURE_CODE_HASH
        and capture_summary.get("results_db_run_id") is None
        and capture_summary.get("runner_sha256")
        == args.capture_runner_sha256
        and capture_summary.get("tensor_sha256")
        == args.capture_tensor_sha256
        and capture_summary.get("status") == "SUCCESS"
        and capture_success.get("artifact_kind")
        == "glm52_layer0_dense_partial_capture"
        and capture_success.get("capture_records_sha256")
        == capture_records_sha
        and capture_success.get("classification")
        == "accepted_m32_dense_partials_captured"
        and capture_success.get("code_hash") == _CAPTURE_CODE_HASH
        and capture_success.get("results_db_run_id") == "none"
        and capture_success.get("runner_sha256") == args.capture_runner_sha256
        and capture_success.get("tensor_sha256") == args.capture_tensor_sha256
        and capture_success.get("performance_claim") == "false"
    ):
        raise RuntimeError("captured-partial source contract drifted")

    db548 = json.loads(args.db548_runner.read_text())
    db548_summary = json.loads(args.db548_summary.read_text())
    db548_success = _terminal_values(args.db548_success)
    comparison = db548.get("layer1_comparison", {})
    if not (
        db548.get("artifact_kind") == "glm52_layer0_dense_convolution_probe"
        and db548.get("classification")
        == "accepted_m32_dense_envelope_cross_layer_nonexact"
        and db548.get("code_hash") == _DB548_CODE_HASH
        and db548.get("dense_envelope") is True
        and db548.get("final_dense_layout") is True
        and db548.get("split_layer1_rms", False) is False
        and db548.get("exact") is False
        and db548.get("exact_arms") == []
        and comparison.get("elementwise_exact") is False
        and comparison.get("mismatch_count") == 1
        and comparison.get("first_mismatch_index") == 2795
        and db548.get("hlo", {}).get("optimized_sha256")
        == _DB548_OPTIMIZED_HLO_SHA
        and db548.get("hlo", {}).get("optimized_contract", {}).get("passed")
        is True
        and db548_summary.get("classification") == db548["classification"]
        and db548_summary.get("code_hash") == _DB548_CODE_HASH
        and db548_summary.get("results_db_run_id") == 548
        and db548_summary.get("status") == "SUCCESS"
        and db548_success.get("code_hash") == _DB548_CODE_HASH
        and db548_success.get("artifact_kind")
        == "glm52_layer0_dense_convolution_probe"
        and db548_success.get("results_db_run_id") == "548"
        and db548_success.get("classification") == db548["classification"]
        and db548_success.get("exact_arms") == "none"
        and db548_success.get("performance_claim") == "false"
    ):
        raise RuntimeError("DB548 control source contract drifted")

    db549 = json.loads(args.db549_runner.read_text())
    db549_summary = json.loads(args.db549_summary.read_text())
    db549_success = _terminal_values(args.db549_success)
    rms = db549.get("hlo", {}).get("optimized_contract", {}).get(
        "lineage", {}
    ).get("rmsnorm_contract", {})
    db549_comparison = db549.get("layer1_comparison", {})
    if not (
        db549.get("artifact_kind") == "glm52_layer0_dense_convolution_probe"
        and db549.get("classification")
        == "accepted_m32_dense_envelope_split_rms_nonexact"
        and db549.get("code_hash") == _DB549_CODE_HASH
        and db549.get("split_layer1_rms") is True
        and db549.get("exact") is False
        and db549.get("exact_arms") == []
        and db549_comparison.get("elementwise_exact") is False
        and db549_comparison.get("mismatch_count") == 1073
        and db549_comparison.get("first_mismatch_index") == 1
        and db549_comparison.get("expected_sha256") == _ACCEPTED_LAYER1_SHA
        and db549_comparison.get("observed_sha256") == _DB549_OBSERVED_SHA
        and db549.get("hlo", {}).get("optimized_sha256")
        == _DB549_OPTIMIZED_HLO_SHA
        and db549.get("hlo", {}).get("optimized_contract", {}).get("passed")
        is True
        and rms.get("exact_accepted_scheduled_reduction") is True
        and rms.get("split_recompute_exact") is True
        and rms.get("split_output_fusion_exact") is True
        and db549_summary.get("classification") == db549["classification"]
        and db549_summary.get("code_hash") == _DB549_CODE_HASH
        and db549_summary.get("results_db_run_id") == 549
        and db549_summary.get("exact_accepted_scheduled_reduction") is True
        and db549_summary.get("status") == "SUCCESS"
        and db549_success.get("code_hash") == _DB549_CODE_HASH
        and db549_success.get("artifact_kind")
        == "glm52_layer0_dense_convolution_probe"
        and db549_success.get("results_db_run_id") == "549"
        and db549_success.get("classification") == db549["classification"]
        and db549_success.get("exact_arms") == "none"
        and db549_success.get("exact_accepted_scheduled_reduction") == "true"
        and db549_success.get("split_recompute_exact") == "true"
        and db549_success.get("split_output_fusion_exact") == "true"
        and db549_success.get("performance_claim") == "false"
    ):
        raise RuntimeError("DB549 accepted-schedule source contract drifted")

    with np.load(args.capture_tensor, allow_pickle=False) as payload:
        expected_keys = {
            "accepted_layer1_normalized_bfloat16_bits",
            "attention_update_bfloat16_bits",
            "combined_residual_bfloat16_bits",
            "compile_rows",
            "dense_virtual_partials_bfloat16_bits",
            "layer1_input_norm_bfloat16_bits",
            "normalized_mlp_bfloat16_bits",
            "post_attention_m32_bfloat16_bits",
            "post_attention_residual_bfloat16_bits",
        }
        if set(payload.files) != expected_keys:
            raise RuntimeError("captured-partial NPZ key set drifted")
        partial_bits = np.ascontiguousarray(
            payload["dense_virtual_partials_bfloat16_bits"]
        )
        residual_bits = np.ascontiguousarray(
            payload["post_attention_m32_bfloat16_bits"]
        )
        attention_bits = np.ascontiguousarray(
            payload["attention_update_bfloat16_bits"]
        )
        combined_bits = np.ascontiguousarray(
            payload["combined_residual_bfloat16_bits"]
        )
        norm_bits = np.ascontiguousarray(
            payload["layer1_input_norm_bfloat16_bits"]
        )
        accepted_bits = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
    expected_arrays = (
        (partial_bits, (4, 8, 1, 6144), _CAPTURE_PARTIALS_SHA),
        (residual_bits, (32, 6144), _CAPTURE_RESIDUAL_SHA),
        (attention_bits, (1, 6144), _ATTENTION_UPDATE_SHA),
        (combined_bits, (1, 6144), _COMBINED_RESIDUAL_SHA),
        (norm_bits, (6144,), _LAYER1_NORM_SHA),
        (accepted_bits, (6144,), _ACCEPTED_LAYER1_SHA),
    )
    if any(
        value.dtype != np.uint16
        or value.shape != shape
        or _array_sha256(value) != expected_sha
        for value, shape, expected_sha in expected_arrays
    ):
        raise RuntimeError("captured-partial tensor record drifted")

    with np.load(args.db548_tensor, allow_pickle=False) as payload:
        db548_observed = np.ascontiguousarray(
            payload["layer1_normalized_bfloat16_bits"]
        )
        db548_accepted = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
    if (
        db548_observed.dtype != np.uint16
        or db548_observed.shape != (6144,)
        or _array_sha256(db548_observed) != _DB548_OBSERVED_SHA
        or not np.array_equal(db548_accepted, accepted_bits)
    ):
        raise RuntimeError("DB548 tensor source drifted")
    with np.load(args.db549_tensor, allow_pickle=False) as payload:
        if set(payload.files) != {
            "accepted_layer1_normalized_bfloat16_bits",
            "attention_update_bfloat16_bits",
            "combined_residual_bfloat16_bits",
            "compile_rows",
            "layer1_normalized_bfloat16_bits",
            "normalized_mlp_bfloat16_bits",
            "post_attention_residual_bfloat16_bits",
        }:
            raise RuntimeError("DB549 tensor key set drifted")
        db549_observed = np.ascontiguousarray(
            payload["layer1_normalized_bfloat16_bits"]
        )
        db549_accepted = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
    if (
        db549_observed.dtype != np.uint16
        or db549_observed.shape != (6144,)
        or _array_sha256(db549_observed) != _DB549_OBSERVED_SHA
        or not np.array_equal(db549_accepted, accepted_bits)
    ):
        raise RuntimeError("DB549 tensor source drifted")
    dense_update_bits = _reproduce_dense_update(partial_bits)
    if (
        dense_update_bits.shape != (1, 6144)
        or _array_sha256(dense_update_bits) != _DENSE_UPDATE_SHA
    ):
        raise RuntimeError("captured partials do not reproduce DB533 StrategyND")
    return (
        partial_bits,
        attention_bits,
        combined_bits,
        norm_bits,
        accepted_bits,
        db548_observed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-worktree", type=Path, required=True)
    for prefix in ("capture", "db548"):
        parser.add_argument(f"--{prefix}-tensor", type=Path, required=True)
        parser.add_argument(f"--{prefix}-tensor-sha256", required=True)
        parser.add_argument(f"--{prefix}-runner", type=Path, required=True)
        parser.add_argument(f"--{prefix}-runner-sha256", required=True)
        parser.add_argument(f"--{prefix}-summary", type=Path, required=True)
        parser.add_argument(f"--{prefix}-summary-sha256", required=True)
        parser.add_argument(f"--{prefix}-success", type=Path, required=True)
        parser.add_argument(f"--{prefix}-success-sha256", required=True)
    parser.add_argument("--db548-hlo", type=Path, required=True)
    parser.add_argument("--db548-hlo-sha256", required=True)
    parser.add_argument("--db549-tensor", type=Path, required=True)
    parser.add_argument("--db549-tensor-sha256", required=True)
    parser.add_argument("--db549-runner", type=Path, required=True)
    parser.add_argument("--db549-runner-sha256", required=True)
    parser.add_argument("--db549-summary", type=Path, required=True)
    parser.add_argument("--db549-summary-sha256", required=True)
    parser.add_argument("--db549-success", type=Path, required=True)
    parser.add_argument("--db549-success-sha256", required=True)
    parser.add_argument("--db549-hlo", type=Path, required=True)
    parser.add_argument("--db549-hlo-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed_worktrees = (
        Path("/home/gianl/glm-tpu-topology-rewrite"),
        Path("/home/gianl/glm-tpu-gate-d-pp16-numerical"),
    )
    if args.expected_worktree not in allowed_worktrees or REPO != args.expected_worktree:
        raise RuntimeError(
            f"wrong greenfield worktree: running={REPO} expected={args.expected_worktree}"
        )
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    (
        partial_bits,
        attention_bits,
        combined_bits,
        norm_bits,
        accepted_bits,
        db548_observed_bits,
    ) = _load_sources(args)

    import jax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_captured_dense_rms_stablehlo,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("captured RMS replay requires one four-chip TPU host")
    mesh = Mesh(np.asarray(jax.local_devices()), ("lp4",))
    replicated = NamedSharding(mesh, P())
    partial_sharding = NamedSharding(mesh, P("lp4", None, None, None))
    arguments = (
        jax.device_put(
            partial_bits.view(ml_dtypes.bfloat16), partial_sharding
        ),
        jax.device_put(
            attention_bits.view(ml_dtypes.bfloat16), replicated
        ),
        jax.device_put(combined_bits.view(ml_dtypes.bfloat16), replicated),
        jax.device_put(norm_bits.view(ml_dtypes.bfloat16), replicated),
    )
    args.hlo_dir.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, Any]] = {}
    outputs: dict[str, np.ndarray] = {}
    for arm_name, split in (("control", False), ("accepted_split", True)):
        mapped = _build_arm(mesh, split_layer1_rms=split)
        lowered = jax.jit(mapped).lower(*arguments)
        stablehlo = lowered.as_text()
        stable_contract = validate_captured_dense_rms_stablehlo(
            stablehlo, split_layer1_rms=split
        )
        stable_path = args.hlo_dir / f"{arm_name}.stablehlo.mlir"
        stable_path.write_text(stablehlo)
        if not stable_contract["passed"]:
            raise RuntimeError(
                f"{arm_name} captured RMS StableHLO failed: {stable_contract}"
            )
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text()
        optimized_contract = _validate_captured_rms_optimized_hlo(
            optimized_hlo, split_layer1_rms=split
        )
        optimized_path = args.hlo_dir / f"{arm_name}.optimized_hlo.txt"
        optimized_path.write_text(optimized_hlo)
        if not optimized_contract["passed"]:
            raise RuntimeError(
                f"{arm_name} captured RMS optimized HLO failed: "
                f"{optimized_contract}"
            )
        value = compiled(*arguments)
        jax.block_until_ready(value)
        bits = np.ascontiguousarray(np.asarray(value).reshape(6144)).view(
            np.uint16
        )
        outputs[arm_name] = bits
        reference = db548_observed_bits if arm_name == "control" else accepted_bits
        arms[arm_name] = {
            "comparison": _compare_bits(reference, bits),
            "output_sha256": _array_sha256(bits),
            "split_layer1_rms": split,
            "hlo": {
                "optimized_contract": optimized_contract,
                "optimized_sha256": sha256(optimized_hlo.encode()).hexdigest(),
                "stablehlo_contract": stable_contract,
                "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
            },
        }

    control_admissible = bool(
        arms["control"]["comparison"]["elementwise_exact"]
        and arms["control"]["output_sha256"] == _DB548_OBSERVED_SHA
    )
    split_exact = bool(
        arms["accepted_split"]["comparison"]["elementwise_exact"]
        and arms["accepted_split"]["output_sha256"] == _ACCEPTED_LAYER1_SHA
    )
    diagnostic_json = args.hlo_dir / "arithmetic_diagnostic.json"
    diagnostic_npz = args.hlo_dir / "arithmetic_diagnostic.npz"
    diagnostic_json.write_text(
        json.dumps(
            {
                "arms": arms,
                "control_admissible": control_admissible,
                "split_exact": split_exact,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    np.savez(
        diagnostic_npz,
        accepted_layer1_normalized_bfloat16_bits=accepted_bits,
        control_layer1_normalized_bfloat16_bits=outputs["control"],
        db548_layer1_normalized_bfloat16_bits=db548_observed_bits,
        split_layer1_normalized_bfloat16_bits=outputs["accepted_split"],
    )
    if not control_admissible:
        raise RuntimeError(
            "captured RMS control did not reproduce the protected DB548 output"
        )
    diagnostic_json.unlink()
    diagnostic_npz.unlink()
    classification = (
        "captured_partials_split_rms_exact"
        if split_exact
        else "captured_partials_split_rms_nonexact"
    )
    result = {
        "artifact_kind": "glm52_layer0_captured_rms_replay",
        "arms": arms,
        "classification": classification,
        "code_hash": code_hash,
        "control_admissible": control_admissible,
        "dense_update_sha256": _DENSE_UPDATE_SHA,
        "exact": split_exact,
        "exact_arms": ["accepted_split"] if split_exact else [],
        "live_rows": 1,
        "performance_claim": False,
        "position": _POSITION,
        "source": {
            "capture_runner_sha256": args.capture_runner_sha256,
            "capture_summary_sha256": args.capture_summary_sha256,
            "capture_success_sha256": args.capture_success_sha256,
            "capture_tensor_sha256": args.capture_tensor_sha256,
            "db548_hlo_sha256": args.db548_hlo_sha256,
            "db548_runner_sha256": args.db548_runner_sha256,
            "db548_summary_sha256": args.db548_summary_sha256,
            "db548_success_sha256": args.db548_success_sha256,
            "db548_tensor_sha256": args.db548_tensor_sha256,
            "db549_hlo_sha256": args.db549_hlo_sha256,
            "db549_runner_sha256": args.db549_runner_sha256,
            "db549_summary_sha256": args.db549_summary_sha256,
            "db549_success_sha256": args.db549_success_sha256,
            "db549_tensor_sha256": args.db549_tensor_sha256,
        },
        "status": "SUCCESS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.tensor_output,
        accepted_layer1_normalized_bfloat16_bits=accepted_bits,
        control_layer1_normalized_bfloat16_bits=outputs["control"],
        db548_layer1_normalized_bfloat16_bits=db548_observed_bits,
        split_layer1_normalized_bfloat16_bits=outputs["accepted_split"],
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
