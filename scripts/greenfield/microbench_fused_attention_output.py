#!/usr/bin/env python3
"""Protected A/B for fused structured-value and attention-output projection."""

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
            f"stale code hash: expected={args.expected_code_hash} "
            f"found={code_hash}"
        )
    if args.warmup < 200 or args.iterations < 1000:
        raise ValueError("protected fused attention proof requires 200/1000")

    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.kernels.pallas import (
        fp8_block_matmul,
        fp8_fused_structured_kv_b_value_output,
        fp8_structured_kv_b_value,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError(
            "fused attention-output proof requires one four-chip TPU-v4 host"
        )
    device = jax.local_devices()[0]
    heads, qk_nope, value_width, latent = 16, 192, 256, 512
    combined = qk_nope + value_width
    output_width = 6144
    value_contraction = heads * value_width

    attended_host = (
        np.cos(np.arange(heads * latent, dtype=np.float32) * 0.0029)
        .reshape(1, heads, latent)
        .astype(ml_dtypes.bfloat16)
    )
    kv_linear = np.arange(heads * combined * latent, dtype=np.uint64)
    kv_bits_host = (
        (kv_linear % np.uint64(120))
        + ((kv_linear // np.uint64(120)) % np.uint64(2)) * np.uint64(128)
    ).astype(np.uint8).reshape(heads * combined, latent)
    kv_scale_host = np.linspace(
        0.0001, 0.0003, 56 * 4, dtype=np.float32
    ).reshape(56, 4)
    output_linear = np.arange(
        output_width * value_contraction, dtype=np.uint64
    )
    output_bits_host = (
        (output_linear % np.uint64(120))
        + ((output_linear // np.uint64(120)) % np.uint64(2))
        * np.uint64(128)
    ).astype(np.uint8).reshape(output_width, value_contraction)
    output_scale_host = np.linspace(
        0.0002,
        0.0008,
        (output_width // 128) * (value_contraction // 128),
        dtype=np.float32,
    ).reshape(output_width // 128, value_contraction // 128)

    def baseline(
        attended: Any,
        kv_bits: Any,
        kv_scale: Any,
        output_bits: Any,
        output_scale: Any,
    ) -> Any:
        values = fp8_structured_kv_b_value(
            attended, kv_bits, kv_scale
        )
        return fp8_block_matmul(
            values.reshape(1, value_contraction),
            output_bits,
            output_scale,
        )

    def candidate(
        attended: Any,
        kv_bits: Any,
        kv_scale: Any,
        output_bits: Any,
        output_scale: Any,
    ) -> Any:
        return fp8_fused_structured_kv_b_value_output(
            attended,
            kv_bits,
            kv_scale,
            output_bits,
            output_scale,
        )

    with jax.default_device(device):
        inputs = (
            jax.device_put(attended_host, device),
            jax.device_put(kv_bits_host, device),
            jax.device_put(kv_scale_host, device),
            jax.device_put(output_bits_host, device),
            jax.device_put(output_scale_host, device),
        )
        compile_started = time.monotonic()
        baseline_compiled = jax.jit(baseline).lower(*inputs).compile()
        baseline_compile_seconds = time.monotonic() - compile_started
        compile_started = time.monotonic()
        candidate_compiled = jax.jit(candidate).lower(*inputs).compile()
        candidate_compile_seconds = time.monotonic() - compile_started

        baseline_hlo = baseline_compiled.as_text()
        candidate_hlo = candidate_compiled.as_text()
        args.hlo_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = args.hlo_dir / "value_output_baseline.optimized_hlo.txt"
        candidate_path = args.hlo_dir / "value_output_fused.optimized_hlo.txt"
        baseline_path.write_text(baseline_hlo)
        candidate_path.write_text(candidate_hlo)
        baseline_sha = sha256(baseline_hlo.encode()).hexdigest()
        candidate_sha = sha256(candidate_hlo.encode()).hexdigest()

        baseline_names = (
            "greenfield_fp8_structured_kv_b_value_h16_l512_v256",
            "greenfield_fp8_block_matmul_m8_k4096_n6144",
        )
        candidate_name = (
            "greenfield_fp8_fused_structured_value_output_"
            "h16_l512_v256_o6144"
        )

        def kernel_calls(hlo: str, name: str) -> list[str]:
            return [
                line.strip()
                for line in hlo.splitlines()
                if name in line
                and 'custom_call_target="tpu_custom_call"' in line
            ]

        baseline_calls = {
            name: kernel_calls(baseline_hlo, name)
            for name in baseline_names
        }
        candidate_calls = kernel_calls(candidate_hlo, candidate_name)
        forbidden_overlays = [
            shape
            for shape in (
                "bf16[7168,512]",
                "f32[7168,512]",
                "bf16[6144,4096]",
                "f32[6144,4096]",
                "f8e4m3fn[7168,512]",
                "f8e4m3fn[6144,4096]",
            )
            if shape in candidate_hlo
        ]
        violations: list[str] = []
        if any(len(calls) != 1 for calls in baseline_calls.values()):
            violations.append(
                f"baseline kernel counts drifted: {baseline_calls}"
            )
        if len(candidate_calls) != 1:
            violations.append(
                f"candidate kernel count drifted: {len(candidate_calls)}"
            )
        if candidate_calls and (
            "u8[7168,512]" not in candidate_calls[0]
            or "u8[6144,4096]" not in candidate_calls[0]
            or "bf16[8,6144]" not in candidate_calls[0]
        ):
            violations.append("candidate raw operands/result shape drifted")
        if any(name in candidate_hlo for name in baseline_names):
            violations.append("candidate retained a baseline kernel boundary")
        if "bf16[1,16,256]" in candidate_hlo:
            violations.append("candidate retained the value-state HBM result")
        if forbidden_overlays:
            violations.append(
                f"candidate decoded a complete weight: {forbidden_overlays}"
            )

        expected = baseline_compiled(*inputs)
        actual = candidate_compiled(*inputs)
        expected.block_until_ready()
        actual.block_until_ready()
        expected_host = np.asarray(expected, dtype=np.float32)
        actual_host = np.asarray(actual, dtype=np.float32)
        difference = np.abs(actual_host - expected_host)
        comparison = {
            "all_finite": bool(np.isfinite(actual_host).all()),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "p99_abs": float(np.percentile(difference, 99)),
            "elementwise_exact": bool(np.array_equal(actual_host, expected_host)),
        }
        comparison["passed"] = bool(
            comparison["all_finite"]
            and comparison["elementwise_exact"]
        )
        if violations or not comparison["passed"]:
            raise RuntimeError(
                f"fused attention-output proof failed: {violations}; "
                f"comparison={comparison}"
            )

        for _ in range(args.warmup):
            baseline_compiled(*inputs).block_until_ready()
            candidate_compiled(*inputs).block_until_ready()
        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            baseline_compiled(*inputs).block_until_ready()
            baseline_samples.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
            started = time.perf_counter_ns()
            candidate_compiled(*inputs).block_until_ready()
            candidate_samples.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )

        baseline_latency = _percentiles(baseline_samples)
        candidate_latency = _percentiles(candidate_samples)
        aggregate_sha = sha256(
            f"{baseline_sha}:{candidate_sha}".encode()
        ).hexdigest()
        result = {
            "status": "SUCCESS",
            "code_hash": code_hash,
            "backend": jax.default_backend(),
            "device": str(device),
            "device_kind": device.device_kind,
            "kernel": "fused_attention_output",
            "output_tile": 128,
            "selected_route_case": None,
            "shape": {
                "attended_latent": [1, heads, latent],
                "kv_b_bits": [heads * combined, latent],
                "output_bits": [output_width, value_contraction],
                "output": [1, output_width],
            },
            "dtype_contract": {
                "activation": "bfloat16",
                "weight_storage": "uint8:e4m3fn-bits",
                "scale": "float32",
                "tile_dequant": "float32-product-to-bfloat16",
                "intermediate": "bfloat16-vmem",
                "accumulator": "float32",
                "output": "bfloat16",
            },
            "compile_seconds": {
                "baseline": baseline_compile_seconds,
                "candidate": candidate_compile_seconds,
                "maximum": max(
                    baseline_compile_seconds, candidate_compile_seconds
                ),
            },
            "comparison": comparison,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profiler_free_timing": True,
            "latency": candidate_latency,
            "latencies": {
                "baseline": baseline_latency,
                "candidate": candidate_latency,
            },
            "checksum": float(actual_host.sum()),
            "memory_stats": _memory_stats(device),
            "hlo": {
                "sha256": aggregate_sha,
                "artifacts": {
                    "baseline": {
                        "path": str(baseline_path),
                        "sha256": baseline_sha,
                    },
                    "candidate": {
                        "path": str(candidate_path),
                        "sha256": candidate_sha,
                    },
                },
                "contract": {
                    "baseline_kernel_calls": baseline_calls,
                    "candidate_kernel_name": candidate_name,
                    "candidate_kernel_calls": candidate_calls,
                    "forbidden_decoded_weight_overlays": forbidden_overlays,
                    "passed": not violations,
                    "violations": violations,
                },
            },
        }
        _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
