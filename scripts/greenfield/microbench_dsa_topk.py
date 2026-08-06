#!/usr/bin/env python3
"""Compile, verify, and time the production-shaped exact DSA selector."""

from __future__ import annotations

import argparse
from functools import partial
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


def _owner_positions(owner: int, *, local_context: int) -> np.ndarray:
    rows_per_page = 128
    logical_page_size = 512
    if local_context % rows_per_page:
        raise ValueError("protected local context must contain complete cache stripes")
    pages = local_context // rows_per_page
    return (
        np.arange(pages, dtype=np.int32)[:, None] * np.int32(logical_page_size)
        + np.int32(owner * rows_per_page)
        + np.arange(rows_per_page, dtype=np.int32)[None, :]
    ).reshape(-1)


def _scores_for_positions(positions: np.ndarray) -> np.ndarray:
    source = positions.astype(np.float32)
    scores = (
        np.sin(source * np.float32(0.000913))
        + np.cos(source * np.float32(0.000371))
        - np.sin(source * np.float32(0.000117)) * np.float32(0.25)
    ).astype(np.float32)
    # An exact high-score tie band makes lowest-global-position ordering a
    # correctness discriminator instead of relying on accidental unique data.
    scores[positions % np.int32(4093) == 17] = np.float32(8.0)
    return scores


def _host_local_topk(
    scores: np.ndarray,
    positions: np.ndarray,
    *,
    valid_length: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (positions >= 0) & (positions < valid_length)
    live_scores = scores[valid]
    live_positions = positions[valid]
    order = np.lexsort((live_positions, -live_scores))
    count = min(top_k, order.size)
    output_scores = np.full((top_k,), -np.inf, dtype=np.float32)
    output_positions = np.full((top_k,), -1, dtype=np.int32)
    output_scores[:count] = live_scores[order[:count]]
    output_positions[:count] = live_positions[order[:count]]
    return output_scores, output_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    parser.add_argument("--local-context", type=int, default=65_536)
    parser.add_argument("--top-k", type=int, default=2048)
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
    if args.local_context != 65_536 or args.top_k != 2048:
        raise ValueError("protected top-k proof requires the 256K/LP4 65536x2048 shape")
    if args.warmup < 200 or args.iterations < 1000:
        raise ValueError("protected top-k proof requires 200/1000 timing samples")

    import jax
    import jax.numpy as jnp

    from glm_tpu.greenfield.benchmarking.topk import validate_dsa_topk_hlo
    from glm_tpu.greenfield.kernels.pallas import (
        DsaTopKConfig,
        local_topk_candidates_pallas,
        merge_topk_candidates_pallas,
    )
    from glm_tpu.greenfield.kernels.reference.dsa import (
        local_topk_candidates,
        merge_topk_candidates,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"top-k metal proof requires TPU, got {jax.default_backend()}")
    if jax.local_device_count() != 4:
        raise RuntimeError(
            f"top-k metal proof requires one four-chip host, got {jax.local_device_count()}"
        )
    device = jax.local_devices()[0]
    local_context = args.local_context
    top_k = args.top_k
    global_context = local_context * 4
    valid_length = global_context - 37
    valid_lengths_host = np.asarray([valid_length], dtype=np.int32)
    valid_lengths = jax.device_put(valid_lengths_host, device)
    config = DsaTopKConfig(
        selection_width=top_k,
        local_block_size=top_k,
    )

    owner = 3
    local_positions_host = _owner_positions(owner, local_context=local_context)
    local_scores_host = _scores_for_positions(local_positions_host)[None, :]
    local_scores = jax.device_put(local_scores_host, device)
    local_positions = jax.device_put(local_positions_host, device)
    local_fn = partial(local_topk_candidates_pallas, config=config)
    local_compile_started = time.monotonic()
    compiled_local = jax.jit(local_fn).lower(
        local_scores, local_positions, valid_lengths
    ).compile()
    local_compile_seconds = time.monotonic() - local_compile_started
    local_hlo = compiled_local.as_text()
    local_hlo_contract = validate_dsa_topk_hlo(local_hlo, phase="local")
    if not local_hlo_contract["passed"]:
        raise RuntimeError(
            f"Pallas local top-k HLO contract failed: {local_hlo_contract}"
        )

    actual_local = compiled_local(local_scores, local_positions, valid_lengths)
    expected_local = jax.jit(
        partial(local_topk_candidates, top_k=top_k)
    )(local_scores, local_positions, valid_lengths)
    jax.block_until_ready((actual_local, expected_local))
    actual_local_scores = np.asarray(actual_local[0], dtype=np.float32)
    actual_local_positions = np.asarray(actual_local[1], dtype=np.int32)
    expected_local_scores = np.asarray(expected_local[0], dtype=np.float32)
    expected_local_positions = np.asarray(expected_local[1], dtype=np.int32)
    host_local_scores, host_local_positions = _host_local_topk(
        local_scores_host[0],
        local_positions_host,
        valid_length=valid_length,
        top_k=top_k,
    )
    local_comparison = {
        "scores_elementwise_exact": bool(
            np.array_equal(actual_local_scores, expected_local_scores)
        ),
        "positions_elementwise_exact": bool(
            np.array_equal(actual_local_positions, expected_local_positions)
        ),
        "host_scores_elementwise_exact": bool(
            np.array_equal(actual_local_scores[0], host_local_scores)
        ),
        "host_positions_elementwise_exact": bool(
            np.array_equal(actual_local_positions[0], host_local_positions)
        ),
    }
    local_comparison["passed"] = bool(all(local_comparison.values()))
    if not local_comparison["passed"]:
        raise RuntimeError(f"Pallas local top-k comparison failed: {local_comparison}")

    owner_scores: list[np.ndarray] = []
    owner_positions: list[np.ndarray] = []
    for slot in range(4):
        positions = _owner_positions(slot, local_context=local_context)
        scores = _scores_for_positions(positions)
        selected_scores, selected_positions = _host_local_topk(
            scores,
            positions,
            valid_length=valid_length,
            top_k=top_k,
        )
        owner_scores.append(selected_scores)
        owner_positions.append(selected_positions)
    owner_order = np.asarray([2, 0, 3, 1], dtype=np.int32)
    candidate_scores_host = np.stack(owner_scores, axis=0)[owner_order, None, :]
    candidate_positions_host = np.stack(owner_positions, axis=0)[
        owner_order, None, :
    ]
    candidate_scores = jax.device_put(candidate_scores_host, device)
    candidate_positions = jax.device_put(candidate_positions_host, device)
    merge_fn = partial(
        merge_topk_candidates_pallas,
        global_context_size=global_context,
        config=config,
    )
    merge_compile_started = time.monotonic()
    compiled_merge = jax.jit(merge_fn).lower(
        candidate_scores, candidate_positions, valid_lengths
    ).compile()
    merge_compile_seconds = time.monotonic() - merge_compile_started
    merge_hlo = compiled_merge.as_text()
    merge_hlo_contract = validate_dsa_topk_hlo(merge_hlo, phase="merge")
    if not merge_hlo_contract["passed"]:
        raise RuntimeError(
            f"Pallas candidate-merge HLO contract failed: {merge_hlo_contract}"
        )

    actual_merge = compiled_merge(
        candidate_scores, candidate_positions, valid_lengths
    )
    expected_merge = jax.jit(
        partial(
            merge_topk_candidates,
            top_k=top_k,
            global_context_size=global_context,
        )
    )(candidate_scores, candidate_positions, valid_lengths)
    jax.block_until_ready((actual_merge, expected_merge))
    actual_global_positions = np.asarray(actual_merge.positions, dtype=np.int32)
    expected_global_positions = np.asarray(expected_merge.positions, dtype=np.int32)
    actual_valid_counts = np.asarray(actual_merge.valid_counts, dtype=np.int32)
    expected_valid_counts = np.asarray(expected_merge.valid_counts, dtype=np.int32)
    all_scores = np.concatenate(owner_scores)
    all_positions = np.concatenate(owner_positions)
    host_order = np.lexsort((all_positions, -all_scores))[:top_k]
    host_global_positions = all_positions[host_order][None, :]
    merge_comparison = {
        "positions_elementwise_exact": bool(
            np.array_equal(actual_global_positions, expected_global_positions)
        ),
        "valid_counts_elementwise_exact": bool(
            np.array_equal(actual_valid_counts, expected_valid_counts)
        ),
        "host_positions_elementwise_exact": bool(
            np.array_equal(actual_global_positions, host_global_positions)
        ),
        "owner_order": owner_order.tolist(),
    }
    merge_comparison["passed"] = bool(
        merge_comparison["positions_elementwise_exact"]
        and merge_comparison["valid_counts_elementwise_exact"]
        and merge_comparison["host_positions_elementwise_exact"]
    )
    if not merge_comparison["passed"]:
        raise RuntimeError(
            f"Pallas candidate-merge comparison failed: {merge_comparison}"
        )

    def time_compiled(
        compiled: Any,
        inputs: tuple[Any, ...],
    ) -> tuple[dict[str, float | int], int]:
        for _ in range(args.warmup):
            jax.block_until_ready(compiled(*inputs))
        samples: list[float] = []
        checksum = 0
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            value = compiled(*inputs)
            jax.block_until_ready(value)
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
            leaves = jax.tree.leaves(value)
            checksum += int(np.asarray(leaves[-1]).reshape(-1)[0])
        return _percentiles(samples), checksum

    local_latency, local_checksum = time_compiled(
        compiled_local, (local_scores, local_positions, valid_lengths)
    )
    merge_latency, merge_checksum = time_compiled(
        compiled_merge, (candidate_scores, candidate_positions, valid_lengths)
    )

    args.hlo_dir.mkdir(parents=True, exist_ok=True)
    local_hlo_path = args.hlo_dir / "dsa_topk_local.optimized_hlo.txt"
    merge_hlo_path = args.hlo_dir / "dsa_topk_merge.optimized_hlo.txt"
    local_hlo_path.write_text(local_hlo)
    merge_hlo_path.write_text(merge_hlo)
    record = {
        "status": "SUCCESS",
        "code_hash": code_hash,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "kernel": "dsa_exact_topk",
        "shape": {
            "local_scores": [1, local_context],
            "local_positions": [local_context],
            "local_candidates": [1, top_k],
            "merge_candidates": [4, 1, top_k],
            "selected_positions": [1, top_k],
            "global_context": global_context,
            "valid_length": valid_length,
        },
        "dtype_contract": {
            "scores": "float32",
            "positions": "int32",
            "valid_lengths": "int32",
            "tie_order": "descending_score_then_lowest_global_position",
            "padding_sentinel": -1,
        },
        "algorithm": {
            "hardware": "TPU v4 TensorCore",
            "sparse_core_available": False,
            "local_block_size": config.local_block_size,
            "local_tree_custom_calls": 6,
            "merge_tree_custom_calls": 2,
            "selection_width": config.selection_width,
        },
        "compile_seconds": {
            "local": local_compile_seconds,
            "merge": merge_compile_seconds,
        },
        "hlo": {
            "local": {
                "path": str(local_hlo_path),
                "sha256": sha256(local_hlo.encode()).hexdigest(),
                "contract": local_hlo_contract,
            },
            "merge": {
                "path": str(merge_hlo_path),
                "sha256": sha256(merge_hlo.encode()).hexdigest(),
                "contract": merge_hlo_contract,
            },
        },
        "comparison": {
            "local": local_comparison,
            "merge": merge_comparison,
            "passed": bool(
                local_comparison["passed"] and merge_comparison["passed"]
            ),
        },
        "profiler_free_timing": True,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency": {
            "local": local_latency,
            "merge": merge_latency,
        },
        "checksum": {
            "local": local_checksum,
            "merge": merge_checksum,
        },
        "memory_stats": _memory_stats(device),
    }
    _atomic_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
