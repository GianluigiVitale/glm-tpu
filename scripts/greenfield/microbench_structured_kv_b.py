#!/usr/bin/env python3
"""Verify and time production-shaped raw-FP8 structured MLA kv_b kernels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _percentiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _comparison(name: str, actual: Any, expected: np.ndarray) -> dict[str, Any]:
    actual_host = np.asarray(actual).astype(np.float32)
    expected_host = expected.astype(np.float32)
    difference = np.abs(actual_host - expected_host)
    record = {
        "name": name,
        "all_finite": bool(np.isfinite(actual_host).all()),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.percentile(difference, 99)),
    }
    record["passed"] = bool(
        record["all_finite"]
        and record["max_abs"] <= 0.125
        and record["p99_abs"] <= 0.0625
        and record["mean_abs"] <= 0.01
    )
    return record


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    if args.warmup < 200 or args.iterations < 1000:
        raise ValueError("protected structured kv_b proof requires 200/1000 samples")

    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.kernels.pallas import (
        fp8_structured_kv_b_q_absorb,
        fp8_structured_kv_b_value,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import fp8_e4m3fn_lookup

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("structured kv_b proof requires one four-chip TPU-v4 host")
    device = jax.local_devices()[0]
    heads, qk_nope, value_width, latent = 16, 192, 256, 512
    combined = qk_nope + value_width

    q_host = (
        np.sin(np.arange(heads * qk_nope, dtype=np.float32) * 0.0037)
        .reshape(1, heads, qk_nope)
        .astype(ml_dtypes.bfloat16)
    )
    attended_host = (
        np.cos(np.arange(heads * latent, dtype=np.float32) * 0.0029)
        .reshape(1, heads, latent)
        .astype(ml_dtypes.bfloat16)
    )
    linear = np.arange(heads * combined * latent, dtype=np.uint64)
    weight_host = (
        (linear % np.uint64(120))
        + ((linear // np.uint64(120)) % np.uint64(2)) * np.uint64(128)
    ).astype(np.uint8).reshape(heads * combined, latent)
    scale_host = np.linspace(
        0.0001, 0.0003, 56 * 4, dtype=np.float32
    ).reshape(56, 4)

    lookup = np.asarray(fp8_e4m3fn_lookup(), dtype=np.float32)
    expanded_scale = np.repeat(
        np.repeat(scale_host, 128, axis=0), 128, axis=1
    )[: heads * combined, :latent]
    decoded = (lookup[weight_host] * expanded_scale).astype(ml_dtypes.bfloat16)
    decoded_f32 = decoded.astype(np.float32).reshape(
        heads, combined, latent
    )
    expected_q = np.einsum(
        "rhp,hpl->rhl",
        q_host.astype(np.float32),
        decoded_f32[:, :qk_nope],
        dtype=np.float32,
    ).astype(ml_dtypes.bfloat16)
    expected_value = np.einsum(
        "rhl,hvl->rhv",
        attended_host.astype(np.float32),
        decoded_f32[:, qk_nope:],
        dtype=np.float32,
    ).astype(ml_dtypes.bfloat16)
    del decoded, decoded_f32, expanded_scale

    with jax.default_device(device):
        q = jax.device_put(q_host, device)
        attended = jax.device_put(attended_host, device)
        weight_bits = jax.device_put(weight_host, device)
        scale = jax.device_put(scale_host, device)

        compile_started = time.monotonic()
        q_compiled = jax.jit(fp8_structured_kv_b_q_absorb).lower(
            q, weight_bits, scale
        ).compile()
        q_compile_seconds = time.monotonic() - compile_started
        compile_started = time.monotonic()
        value_compiled = jax.jit(fp8_structured_kv_b_value).lower(
            attended, weight_bits, scale
        ).compile()
        value_compile_seconds = time.monotonic() - compile_started

        q_hlo = q_compiled.as_text()
        value_hlo = value_compiled.as_text()
        args.hlo_dir.mkdir(parents=True, exist_ok=True)
        q_path = args.hlo_dir / "structured_kv_b_q_absorb.optimized_hlo.txt"
        value_path = args.hlo_dir / "structured_kv_b_value.optimized_hlo.txt"
        q_path.write_text(q_hlo)
        value_path.write_text(value_hlo)
        q_sha = sha256(q_hlo.encode()).hexdigest()
        value_sha = sha256(value_hlo.encode()).hexdigest()

        expected_names = {
            "q_absorb": "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512",
            "value": "greenfield_fp8_structured_kv_b_value_h16_l512_v256",
        }
        hlo_by_name = {"q_absorb": q_hlo, "value": value_hlo}
        kernel_counts = {
            name: sum(
                expected_names[name] in line
                and 'custom_call_target="tpu_custom_call"' in line
                for line in hlo.splitlines()
            )
            for name, hlo in hlo_by_name.items()
        }
        kernel_calls = {
            name: [
                line.strip()
                for line in hlo.splitlines()
                if expected_names[name] in line
                and 'custom_call_target="tpu_custom_call"' in line
            ]
            for name, hlo in hlo_by_name.items()
        }
        forbidden_overlays = {
            name: [
                shape
                for shape in ("bf16[7168,512]", "f32[7168,512]")
                if shape in hlo
            ]
            for name, hlo in hlo_by_name.items()
        }
        violations = []
        if kernel_counts != {"q_absorb": 1, "value": 1}:
            violations.append(f"structured kv_b kernel counts drifted: {kernel_counts}")
        if any(forbidden_overlays.values()):
            violations.append(
                f"structured kv_b decoded overlay exists: {forbidden_overlays}"
            )
        malformed_raw_calls = [
            name
            for name, calls in kernel_calls.items()
            if len(calls) != 1 or "u8[7168,512]" not in calls[0]
        ]
        forbidden_formatted_overlays = {
            name: "f8e4m3fn[7168,512]" in hlo
            for name, hlo in hlo_by_name.items()
        }
        if malformed_raw_calls:
            violations.append(
                "structured kv_b calls do not consume raw U8 weights: "
                f"{malformed_raw_calls}"
            )
        if any(forbidden_formatted_overlays.values()):
            violations.append(
                "structured kv_b performs whole-table FP8 formatting: "
                f"{forbidden_formatted_overlays}"
            )

        actual_q = q_compiled(q, weight_bits, scale)
        actual_value = value_compiled(attended, weight_bits, scale)
        actual_q.block_until_ready()
        actual_value.block_until_ready()
        comparisons = (
            _comparison("q_absorb", actual_q, expected_q),
            _comparison("value", actual_value, expected_value),
        )
        comparison = {
            "all_finite": all(item["all_finite"] for item in comparisons),
            "max_abs": max(item["max_abs"] for item in comparisons),
            "mean_abs": float(np.mean([item["mean_abs"] for item in comparisons])),
            "p99_abs": max(item["p99_abs"] for item in comparisons),
            "outputs": comparisons,
            "passed": all(item["passed"] for item in comparisons),
        }
        if violations or not comparison["passed"]:
            raise RuntimeError(
                f"structured kv_b proof failed: violations={violations} "
                f"comparison={comparison}"
            )

        for _ in range(args.warmup):
            q_compiled(q, weight_bits, scale).block_until_ready()
            value_compiled(attended, weight_bits, scale).block_until_ready()
        q_samples: list[float] = []
        value_samples: list[float] = []
        combined_samples: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            q_compiled(q, weight_bits, scale).block_until_ready()
            q_elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            started = time.perf_counter_ns()
            value_compiled(attended, weight_bits, scale).block_until_ready()
            value_elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            q_samples.append(q_elapsed)
            value_samples.append(value_elapsed)
            combined_samples.append(q_elapsed + value_elapsed)

        aggregate_sha = sha256(f"{q_sha}:{value_sha}".encode()).hexdigest()
        result = {
            "status": "SUCCESS",
            "code_hash": code_hash,
            "backend": jax.default_backend(),
            "device": str(device),
            "device_kind": device.device_kind,
            "kernel": "structured_kv_b",
            "selected_route_case": None,
            "shape": {
                "q_nope": [1, heads, qk_nope],
                "attended_latent": [1, heads, latent],
                "weight_bits": [heads * combined, latent],
                "q_absorbed": [1, heads, latent],
                "value": [1, heads, value_width],
            },
            "dtype_contract": {
                "activation": "bfloat16",
                "weight_storage": "uint8:e4m3fn-bits",
                "scale": "float32",
                "tile_dequant": "float32-product-to-bfloat16",
                "accumulator": "float32",
                "output": "bfloat16",
            },
            "compile_seconds": {
                "q_absorb": q_compile_seconds,
                "value": value_compile_seconds,
                "maximum": max(q_compile_seconds, value_compile_seconds),
            },
            "comparison": comparison,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profiler_free_timing": True,
            "latency": _percentiles(combined_samples),
            "latencies": {
                "q_absorb": _percentiles(q_samples),
                "value": _percentiles(value_samples),
            },
            "checksum": float(
                np.asarray(actual_q).astype(np.float32).sum()
                + np.asarray(actual_value).astype(np.float32).sum()
            ),
            "memory_stats": _memory_stats(device),
            "hlo": {
                "sha256": aggregate_sha,
                "artifacts": {
                    "q_absorb": {"path": str(q_path), "sha256": q_sha},
                    "value": {"path": str(value_path), "sha256": value_sha},
                },
                "contract": {
                    "expected_kernel_names": expected_names,
                    "kernel_counts": kernel_counts,
                    "forbidden_decoded_weight_overlays": forbidden_overlays,
                    "forbidden_formatted_weight_overlays": (
                        forbidden_formatted_overlays
                    ),
                    "passed": not violations,
                    "violations": violations,
                },
            },
        }
        _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
