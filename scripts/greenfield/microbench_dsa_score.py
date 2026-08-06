#!/usr/bin/env python3
"""Compile, verify, and time the production-shaped Pallas DSA scorer."""

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
    parser.add_argument("--local-context", type=int, default=65_536)
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
    if args.local_context != 65_536:
        raise ValueError("protected DSA proof requires one 256K/LP4 context shard")
    if args.warmup < 200 or args.iterations < 1000:
        raise ValueError("protected DSA proof requires 200/1000 timing samples")

    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking.dsa import validate_dsa_score_hlo
    from glm_tpu.greenfield.kernels.pallas import dsa_scores_pallas
    from glm_tpu.greenfield.kernels.reference.dsa import dsa_scores

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"DSA metal proof requires TPU, got {jax.default_backend()}")
    if jax.local_device_count() != 4:
        raise RuntimeError(
            f"DSA metal proof requires one four-chip host, got {jax.local_device_count()}"
        )
    device = jax.local_devices()[0]
    context = args.local_context
    query_host = np.sin(
        np.arange(32 * 128, dtype=np.float32) * np.float32(0.0137)
    ).reshape(1, 32, 128)
    key_linear = np.arange(context * 128, dtype=np.float32).reshape(context, 128)
    keys_host = (
        np.sin(key_linear * np.float32(0.000173))
        * np.cos(key_linear * np.float32(0.000071))
    ).astype(ml_dtypes.bfloat16)
    del key_linear
    head_weights_host = np.linspace(
        -0.875, 1.0, 32, dtype=np.float32
    ).reshape(1, 32)
    query = jax.device_put(query_host, device)
    keys = jax.device_put(keys_host, device)
    head_weights = jax.device_put(head_weights_host, device)

    lower_started = time.monotonic()
    compiled = jax.jit(dsa_scores_pallas).lower(
        query, keys, head_weights
    ).compile()
    compile_seconds = time.monotonic() - lower_started
    hlo = compiled.as_text()
    hlo_sha256 = sha256(hlo.encode()).hexdigest()
    args.hlo_output.parent.mkdir(parents=True, exist_ok=True)
    args.hlo_output.write_text(hlo)
    hlo_contract = validate_dsa_score_hlo(hlo, local_context=context)
    if not hlo_contract["passed"]:
        raise RuntimeError(f"Pallas DSA HLO contract failed: {hlo_contract}")

    actual = compiled(query, keys, head_weights)
    expected = jax.jit(dsa_scores)(query, keys, head_weights)
    jax.block_until_ready((actual, expected))
    actual_host = np.asarray(actual, dtype=np.float32)
    expected_host = np.asarray(expected, dtype=np.float32)
    difference = np.abs(actual_host - expected_host)
    comparison = {
        "all_finite": bool(np.isfinite(actual_host).all()),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.percentile(difference, 99)),
        "rtol": 2e-5,
        "atol": 2e-4,
        "passed": bool(
            np.isfinite(actual_host).all()
            and np.allclose(actual_host, expected_host, rtol=2e-5, atol=2e-4)
        ),
    }
    if not comparison["passed"]:
        raise RuntimeError(f"Pallas DSA/reference comparison failed: {comparison}")

    for _ in range(args.warmup):
        jax.block_until_ready(compiled(query, keys, head_weights))
    samples: list[float] = []
    checksum = 0.0
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        value = compiled(query, keys, head_weights)
        jax.block_until_ready(value)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        checksum += float(np.asarray(value[0, 0], dtype=np.float32))

    record = {
        "status": "SUCCESS",
        "code_hash": code_hash,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "kernel": "dsa_score",
        "shape": {
            "query": [1, 32, 128],
            "index_keys": [context, 128],
            "head_weights": [1, 32],
            "scores": [1, context],
        },
        "dtype_contract": {
            "query": "float32",
            "index_keys": "bfloat16",
            "head_weights": "float32",
            "per_head_relu": "float32",
            "head_reduction": "deterministic-head-order-float32",
            "scores": "float32",
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
