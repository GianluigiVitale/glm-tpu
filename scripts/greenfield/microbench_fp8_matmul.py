#!/usr/bin/env python3
"""Compile, verify, and time one production-shaped greenfield FP8 kernel."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--contraction", type=int, default=6144)
    parser.add_argument("--output-width", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    if args.rows != 8 or args.contraction != 6144 or args.output_width != 2048:
        raise ValueError("first metal proof is fixed to the GLM expert up projection")
    if args.warmup < 200 or args.iterations < 1000:
        raise ValueError("protected FP8 microbenchmark requires 200/1000 samples")

    import jax
    from jax import lax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.kernels.pallas import fp8_block_matmul
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"FP8 metal proof requires TPU, got {jax.default_backend()}")
    if jax.local_device_count() != 4:
        raise RuntimeError(
            f"FP8 metal proof requires one four-chip host, got {jax.local_device_count()}"
        )
    device = jax.local_devices()[0]
    rows = args.rows
    contraction = args.contraction
    output = args.output_width

    lhs_host = (
        np.sin(np.arange(rows * contraction, dtype=np.float32) * 0.0037)
        .reshape(rows, contraction)
        .astype(ml_dtypes.bfloat16)
    )
    linear = np.arange(output * contraction, dtype=np.uint64)
    # Codes 0..119 and 128..247 are finite E4M3FN values.  Avoid the two NaN
    # encodings 127/255 while exercising both signs and every finite exponent.
    weight_host = (
        (linear % np.uint64(120))
        + ((linear // np.uint64(120)) % np.uint64(2)) * np.uint64(128)
    ).astype(np.uint8).reshape(output, contraction)
    scale_host = np.linspace(
        0.0005,
        0.0015,
        (output // 128) * (contraction // 128),
        dtype=np.float32,
    ).reshape(output // 128, contraction // 128)

    with jax.default_device(device):
        lhs = jax.device_put(lhs_host, device)
        weight_bits = jax.device_put(weight_host, device)
        scale = jax.device_put(scale_host, device)
        lower_started = time.monotonic()
        compiled = jax.jit(fp8_block_matmul).lower(
            lhs, weight_bits, scale
        ).compile()
        compile_seconds = time.monotonic() - lower_started
        hlo = compiled.as_text()
        hlo_sha256 = sha256(hlo.encode()).hexdigest()

        custom_calls = [
            line.strip()
            for line in hlo.splitlines()
            if "custom-call(" in line
        ]
        kernel_calls = [
            line
            for line in custom_calls
            if "greenfield_fp8_block_matmul" in line
            or "tpu_custom_call" in line
        ]
        forbidden_full_overlays = [
            shape
            for shape in (
                f"bf16[{output},{contraction}]",
                f"f32[{output},{contraction}]",
                f"bf16[{contraction},{output}]",
                f"f32[{contraction},{output}]",
            )
            if shape in hlo
        ]
        hlo_contract = {
            "custom_call_count": len(custom_calls),
            "kernel_custom_call_count": len(kernel_calls),
            "kernel_custom_calls": kernel_calls,
            "forbidden_full_weight_overlays": forbidden_full_overlays,
            "passed": len(kernel_calls) == 1 and not forbidden_full_overlays,
        }
        if not hlo_contract["passed"]:
            raise RuntimeError(f"FP8 Pallas HLO contract failed: {hlo_contract}")

        actual = compiled(lhs, weight_bits, scale)
        actual.block_until_ready()
        decoded = dequantize_fp8_bits_block_weight(weight_bits, scale)
        expected = lax.dot_general(
            lhs,
            decoded,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        expected.block_until_ready()
        actual_host = np.asarray(actual, dtype=np.float32)
        expected_host = np.asarray(expected, dtype=np.float32)
        difference = np.abs(actual_host - expected_host)
        comparison = {
            "all_finite": bool(np.isfinite(actual_host).all()),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "p99_abs": float(np.percentile(difference, 99)),
            "passed": bool(
                np.isfinite(actual_host).all()
                and np.allclose(actual_host, expected_host, rtol=0.02, atol=0.0625)
            ),
        }
        if not comparison["passed"]:
            raise RuntimeError(f"FP8 Pallas/reference comparison failed: {comparison}")

        for _ in range(args.warmup):
            compiled(lhs, weight_bits, scale).block_until_ready()
        samples = []
        checksum = 0.0
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            value = compiled(lhs, weight_bits, scale)
            value.block_until_ready()
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
            checksum += float(np.asarray(value[0, 0], dtype=np.float32))

    args.hlo_output.parent.mkdir(parents=True, exist_ok=True)
    args.hlo_output.write_text(hlo)
    record = {
        "status": "SUCCESS",
        "code_hash": code_hash,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "shape": {
            "lhs": [rows, contraction],
            "weight_bits": [output, contraction],
            "scale": [output // 128, contraction // 128],
            "output": [rows, output],
        },
        "dtype_contract": {
            "lhs": "bfloat16",
            "weight_storage": "uint8:e4m3fn-bits",
            "scale": "float32",
            "tile_dequant": "float32-product-to-bfloat16",
            "accumulator": "float32",
            "output": "bfloat16",
        },
        "compile_seconds": compile_seconds,
        "hlo": {
            "path": str(args.hlo_output),
            "sha256": hlo_sha256,
            "contract": hlo_contract,
        },
        "comparison": comparison,
        "profiler_free_timing": True,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency": _percentiles(samples),
        "checksum": checksum,
        "memory_stats": _memory_stats(device),
    }
    _atomic_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
