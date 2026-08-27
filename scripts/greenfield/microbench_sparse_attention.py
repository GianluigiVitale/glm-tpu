#!/usr/bin/env python3
"""Compile, verify, and time production-shaped fused selected-KV sparse MLA."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

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


def _validate_sampling_contract(
    *, geometry: str, diagnostic_reference_timing: bool, warmup: int, iterations: int
) -> None:
    if diagnostic_reference_timing:
        if geometry != "pp16_lp2_2k":
            raise ValueError(
                "reference timing is admitted only for the PP16 LP2 2K diagnostic"
            )
        if warmup < 1 or iterations < 3:
            raise ValueError("reference diagnostic requires warmup>=1 and iterations>=3")
        return
    if geometry != "pp8_lp4_256k":
        raise ValueError("PP16 LP2 2K is diagnostic-only")
    if warmup < 200 or iterations < 1000:
        raise ValueError("protected sparse MLA proof requires 200/1000 samples")


def _reference_hlo_contract(hlo: str) -> dict[str, Any]:
    qk_count = sum(
        " convolution(" in line and "rhd,rkd->rhk/dot_general" in line
        for line in hlo.splitlines()
    )
    pv_count = sum(
        " convolution(" in line and "rhk,rkd->rhd/dot_general" in line
        for line in hlo.splitlines()
    )
    selected_overlay = "bf16[2048,640]" in hlo
    violations = []
    if qk_count != 2:
        violations.append(f"reference qk convolution count is {qk_count}, expected 2")
    if pv_count != 1:
        violations.append(f"reference pv convolution count is {pv_count}, expected 1")
    if not selected_overlay:
        violations.append("reference selected-KV overlay is absent")
    return {
        "qk_convolution_count": qk_count,
        "pv_convolution_count": pv_count,
        "selected_kv_overlay": selected_overlay,
        "violations": violations,
        "passed": not violations,
    }


def _owner_positions(
    owner: int,
    count: int,
    *,
    valid_length: int,
) -> np.ndarray:
    """Return a deterministic unique sample spanning one 65,536-row owner."""

    local = (
        np.arange(65_536, dtype=np.int64) * np.int64(257)
        + np.int64(17 + owner * 29)
    ) % np.int64(65_536)
    pages = local // np.int64(128)
    rows = local % np.int64(128)
    positions = pages * np.int64(512) + np.int64(owner * 128) + rows
    positions = positions[positions < valid_length]
    if positions.size < count:
        raise ValueError("not enough valid positions for protected owner case")
    return positions[:count].astype(np.int32)


def _selection_cases(valid_length: int) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(70_207)
    balanced_live = np.concatenate(
        [
            _owner_positions(owner, 512, valid_length=valid_length)
            for owner in range(4)
        ]
    )
    balanced_live = balanced_live[rng.permutation(balanced_live.size)]
    concentrated_live = _owner_positions(0, 2048, valid_length=valid_length)
    concentrated_live = concentrated_live[rng.permutation(concentrated_live.size)]
    tail_live = balanced_live[:2037]
    tail = np.full((2048,), -1, dtype=np.int32)
    tail[: tail_live.size] = tail_live
    return {
        "balanced_owner0": {
            "positions": balanced_live[None, :],
            "valid_count": np.asarray([2048], dtype=np.int32),
            "owner": np.asarray(0, dtype=np.int32),
            "expected_owner_rows": 512,
        },
        "concentrated_owner0": {
            "positions": concentrated_live[None, :],
            "valid_count": np.asarray([2048], dtype=np.int32),
            "owner": np.asarray(0, dtype=np.int32),
            "expected_owner_rows": 2048,
        },
        "tail2037_owner2": {
            "positions": tail[None, :],
            "valid_count": np.asarray([2037], dtype=np.int32),
            "owner": np.asarray(2, dtype=np.int32),
            "expected_owner_rows": int(
                np.count_nonzero((tail_live % 512) // 128 == 2)
            ),
        },
        "empty_owner3": {
            "positions": concentrated_live[None, :],
            "valid_count": np.asarray([2048], dtype=np.int32),
            "owner": np.asarray(3, dtype=np.int32),
            "expected_owner_rows": 0,
        },
    }


def _selection_cases_lp2_2k(valid_length: int) -> dict[str, dict[str, Any]]:
    if valid_length != 2035:
        raise ValueError("PP16 LP2 diagnostic requires exactly 2,035 live positions")
    rng = np.random.default_rng(70_216)
    live = np.arange(valid_length, dtype=np.int32)
    balanced_live = live[rng.permutation(live.size)]
    balanced = np.full((2048,), -1, dtype=np.int32)
    balanced[: balanced_live.size] = balanced_live
    owner0_live = live[((live % 512) // 256) == 0]
    owner0_live = owner0_live[rng.permutation(owner0_live.size)]
    concentrated = np.full((2048,), -1, dtype=np.int32)
    concentrated[: owner0_live.size] = owner0_live
    tail_live = balanced_live[:-11]
    tail = np.full((2048,), -1, dtype=np.int32)
    tail[: tail_live.size] = tail_live
    return {
        "balanced_owner0": {
            "positions": balanced[None, :],
            "valid_count": np.asarray([balanced_live.size], dtype=np.int32),
            "owner": np.asarray(0, dtype=np.int32),
            "expected_owner_rows": int(
                np.count_nonzero(((balanced_live % 512) // 256) == 0)
            ),
        },
        "concentrated_owner0": {
            "positions": concentrated[None, :],
            "valid_count": np.asarray([owner0_live.size], dtype=np.int32),
            "owner": np.asarray(0, dtype=np.int32),
            "expected_owner_rows": int(owner0_live.size),
        },
        "tail2037_owner1": {
            "positions": tail[None, :],
            "valid_count": np.asarray([tail_live.size], dtype=np.int32),
            "owner": np.asarray(1, dtype=np.int32),
            "expected_owner_rows": int(
                np.count_nonzero(((tail_live % 512) // 256) == 1)
            ),
        },
        "empty_owner1": {
            "positions": concentrated[None, :],
            "valid_count": np.asarray([owner0_live.size], dtype=np.int32),
            "owner": np.asarray(1, dtype=np.int32),
            "expected_owner_rows": 0,
        },
    }


def _compare_results(actual: Any, expected: Any) -> dict[str, Any]:
    actual_output = np.asarray(actual.output, dtype=np.float32)
    expected_output = np.asarray(expected.output, dtype=np.float32)
    output_difference = np.abs(actual_output - expected_output)
    actual_lse = np.asarray(actual.logsumexp, dtype=np.float32)
    expected_lse = np.asarray(expected.logsumexp, dtype=np.float32)
    finite_pattern_exact = bool(
        np.array_equal(np.isfinite(actual_lse), np.isfinite(expected_lse))
    )
    finite = np.isfinite(actual_lse) & np.isfinite(expected_lse)
    lse_difference = (
        np.abs(actual_lse[finite] - expected_lse[finite])
        if np.any(finite)
        else np.zeros((1,), dtype=np.float32)
    )
    actual_valid = np.asarray(actual.contract_valid, dtype=np.bool_)
    expected_valid = np.asarray(expected.contract_valid, dtype=np.bool_)
    record = {
        "output_all_finite": bool(np.isfinite(actual_output).all()),
        "output_max_abs": float(output_difference.max()),
        "output_mean_abs": float(output_difference.mean()),
        "output_p99_abs": float(np.percentile(output_difference, 99)),
        "output_max_abs_limit": 0.03125,
        "output_mean_abs_limit": 0.002,
        "lse_finite_pattern_exact": finite_pattern_exact,
        "lse_max_abs": float(lse_difference.max()),
        "lse_max_abs_limit": 0.01,
        "contract_valid_exact": bool(np.array_equal(actual_valid, expected_valid)),
        "contract_valid": actual_valid.tolist(),
    }
    record["passed"] = bool(
        record["output_all_finite"]
        and record["output_max_abs"] <= record["output_max_abs_limit"]
        and record["output_mean_abs"] <= record["output_mean_abs_limit"]
        and record["lse_finite_pattern_exact"]
        and record["lse_max_abs"] <= record["lse_max_abs_limit"]
        and record["contract_valid_exact"]
        and bool(actual_valid.all())
    )
    return record


def _time_compiled(
    compiled: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float | int], float]:
    import jax

    for _ in range(warmup):
        jax.block_until_ready(compiled(*inputs))
    samples: list[float] = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = compiled(*inputs)
        jax.block_until_ready(result)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    if result is None:
        raise RuntimeError("protected sparse-attention timing produced no result")
    checksum = float(
        np.asarray(result.output[0, 0, 0], dtype=np.float32)
        + np.asarray(result.logsumexp[0, 0], dtype=np.float32)
    )
    return _percentiles(samples), checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-output", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        choices=("pp8_lp4_256k", "pp16_lp2_2k"),
        default="pp8_lp4_256k",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--diagnostic-reference-timing", action="store_true")
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
    _validate_sampling_contract(
        geometry=args.geometry,
        diagnostic_reference_timing=args.diagnostic_reference_timing,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking.sparse_attention import (
        validate_sparse_attention_hlo,
    )
    from glm_tpu.greenfield.kernels.pallas import (
        SparseMlaConfig,
        stage_local_sparse_mla_pallas,
    )
    from glm_tpu.greenfield.kernels.reference.attention import (
        MlaNumericalContract,
        StageLocalKvLayout,
        gather_stage_local_selected_kv,
        sparse_mla_attention,
    )
    from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions

    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"sparse-attention metal proof requires TPU, got {jax.default_backend()}"
        )
    if jax.local_device_count() != 4:
        raise RuntimeError(
            "sparse-attention proof requires one four-chip host, got "
            f"{jax.local_device_count()}"
        )
    device = jax.local_devices()[0]
    contract = MlaNumericalContract()
    local_parallel_size = 2 if args.geometry == "pp16_lp2_2k" else 4
    layout = StageLocalKvLayout(local_parallel_size=local_parallel_size)
    config = SparseMlaConfig(segment_block=128)
    pages = 4 if args.geometry == "pp16_lp2_2k" else 512
    valid_length = (
        2035
        if args.geometry == "pp16_lp2_2k"
        else pages * layout.logical_page_size - 37
    )

    rows = np.arange(
        pages * layout.local_rows_per_page, dtype=np.float32
    )[:, None]
    columns = np.arange(contract.packed_cache_width, dtype=np.float32)[None, :]
    cache_host = (
        np.sin(rows * np.float32(0.00137) + columns * np.float32(0.00311))
        * np.cos(rows * np.float32(0.00029) - columns * np.float32(0.00173))
    ).astype(ml_dtypes.bfloat16).reshape(
        pages, layout.local_rows_per_page, 640
    )
    query_nope_host = np.sin(
        np.arange(64 * 512, dtype=np.float32) * np.float32(0.00713)
    ).astype(ml_dtypes.bfloat16).reshape(1, 64, 512)
    query_rope_host = np.cos(
        np.arange(64 * 64, dtype=np.float32) * np.float32(0.01171)
    ).astype(ml_dtypes.bfloat16).reshape(1, 64, 64)
    block_tables_host = (
        (np.arange(pages, dtype=np.int32) * np.int32(197) + np.int32(17))
        % np.int32(pages)
    )[None, :]
    lengths_host = np.asarray([valid_length], dtype=np.int32)
    del rows, columns

    query_nope = jax.device_put(query_nope_host, device)
    query_rope = jax.device_put(query_rope_host, device)
    cache = jax.device_put(cache_host, device)
    block_tables = jax.device_put(block_tables_host, device)
    lengths = jax.device_put(lengths_host, device)

    def pallas_fn(
        q_nope: Any,
        q_rope: Any,
        cache_value: Any,
        tables: Any,
        positions: Any,
        counts: Any,
        context_lengths: Any,
        owner: Any,
    ) -> Any:
        return stage_local_sparse_mla_pallas(
            q_nope,
            q_rope,
            cache_value,
            tables,
            SelectedPositions(positions, counts),
            context_lengths,
            layout=layout,
            owner_index=owner,
            contract=contract,
            config=config,
        )

    def reference_fn(
        q_nope: Any,
        q_rope: Any,
        cache_value: Any,
        tables: Any,
        positions: Any,
        counts: Any,
        context_lengths: Any,
        owner: Any,
    ) -> Any:
        segment = gather_stage_local_selected_kv(
            cache_value,
            tables,
            SelectedPositions(positions, counts),
            context_lengths,
            layout=layout,
            owner_index=owner,
        )
        return sparse_mla_attention(q_nope, q_rope, segment, contract=contract)

    host_cases = (
        _selection_cases_lp2_2k(valid_length)
        if args.geometry == "pp16_lp2_2k"
        else _selection_cases(valid_length)
    )
    device_cases: dict[str, tuple[Any, ...]] = {}
    for name, case in host_cases.items():
        device_cases[name] = (
            query_nope,
            query_rope,
            cache,
            block_tables,
            jax.device_put(case["positions"], device),
            jax.device_put(case["valid_count"], device),
            lengths,
            jax.device_put(case["owner"], device),
        )

    compile_inputs = device_cases["balanced_owner0"]
    compile_started = time.monotonic()
    compiled = jax.jit(pallas_fn).lower(*compile_inputs).compile()
    pallas_compile_seconds = time.monotonic() - compile_started
    hlo = compiled.as_text()
    args.hlo_output.parent.mkdir(parents=True, exist_ok=True)
    args.hlo_output.write_text(hlo)
    hlo_contract = validate_sparse_attention_hlo(
        hlo,
        cache_rows=pages * layout.local_rows_per_page,
        expected_metadata_gather_count=(
            0 if args.geometry == "pp16_lp2_2k" else 1
        ),
    )
    if not hlo_contract["passed"]:
        raise RuntimeError(
            f"fused sparse-attention HLO contract failed: {hlo_contract}"
        )

    reference_compile_started = time.monotonic()
    compiled_reference = jax.jit(reference_fn).lower(*compile_inputs).compile()
    reference_compile_seconds = time.monotonic() - reference_compile_started
    reference_hlo = compiled_reference.as_text()
    reference_hlo_path = args.hlo_output.with_name(
        args.hlo_output.name.replace(
            ".optimized_hlo.txt", ".reference.optimized_hlo.txt"
        )
    )
    if reference_hlo_path == args.hlo_output:
        raise ValueError("diagnostic HLO output lacks the expected suffix")
    reference_hlo_record = None
    if args.diagnostic_reference_timing:
        reference_hlo_path.write_text(reference_hlo)
        reference_contract = _reference_hlo_contract(reference_hlo)
        if not reference_contract["passed"]:
            raise RuntimeError(
                f"reference sparse-attention HLO contract failed: {reference_contract}"
            )
        reference_hlo_record = {
            "path": str(reference_hlo_path),
            "sha256": sha256(reference_hlo.encode()).hexdigest(),
            "contract": reference_contract,
        }

    comparisons: dict[str, dict[str, Any]] = {}
    for name, inputs in device_cases.items():
        actual = compiled(*inputs)
        expected = compiled_reference(*inputs)
        jax.block_until_ready((actual, expected))
        comparison = _compare_results(actual, expected)
        comparison["selected_owner_rows"] = host_cases[name][
            "expected_owner_rows"
        ]
        comparisons[name] = comparison
        if not comparison["passed"]:
            raise RuntimeError(
                f"fused sparse-attention comparison failed for {name}: "
                f"{comparison}"
            )

    bad_tables_host = block_tables_host.copy()
    balanced_positions = host_cases["balanced_owner0"]["positions"][0]
    bad_position = int(
        balanced_positions[((balanced_positions % 512) // 128) == 0][0]
    )
    bad_tables_host[0, bad_position // 512] = pages + 31
    bad_inputs = list(device_cases["balanced_owner0"])
    bad_inputs[3] = jax.device_put(bad_tables_host, device)
    bad_result = compiled(*tuple(bad_inputs))
    jax.block_until_ready(bad_result)
    invalid_page_safety = {
        "contract_valid": np.asarray(
            bad_result.contract_valid, dtype=np.bool_
        ).tolist(),
        "output_all_finite": bool(
            np.isfinite(np.asarray(bad_result.output, dtype=np.float32)).all()
        ),
        "lse_all_finite": bool(
            np.isfinite(np.asarray(bad_result.logsumexp, dtype=np.float32)).all()
        ),
    }
    invalid_page_safety["passed"] = bool(
        not any(invalid_page_safety["contract_valid"])
        and invalid_page_safety["output_all_finite"]
        and invalid_page_safety["lse_all_finite"]
    )
    if not invalid_page_safety["passed"]:
        raise RuntimeError(
            f"invalid-page fail-closed check failed: {invalid_page_safety}"
        )

    latency: dict[str, dict[str, float | int]] = {}
    reference_latency: dict[str, dict[str, float | int]] = {}
    checksums: dict[str, float] = {}
    reference_checksums: dict[str, float] = {}
    for name in ("balanced_owner0", "concentrated_owner0"):
        latency[name], checksums[name] = _time_compiled(
            compiled,
            device_cases[name],
            warmup=args.warmup,
            iterations=args.iterations,
        )
        if args.diagnostic_reference_timing:
            reference_latency[name], reference_checksums[name] = _time_compiled(
                compiled_reference,
                device_cases[name],
                warmup=args.warmup,
                iterations=args.iterations,
            )

    record = {
        "status": "SUCCESS",
        "code_hash": code_hash,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "kernel": "fused_selected_kv_sparse_mla",
        "geometry": args.geometry,
        "shape": {
            "query_nope": [1, 64, 512],
            "query_rope": [1, 64, 64],
            "local_cache": [pages, layout.local_rows_per_page, 640],
            "selected_positions": [1, 2048],
            "global_context_capacity": pages * layout.logical_page_size,
            "valid_length": valid_length,
            "local_parallel_size": local_parallel_size,
            "segment_block": 128,
        },
        "dtype_contract": {
            "query": "bfloat16",
            "cache": "bfloat16",
            "scores_softmax_lse": "float32",
            "probability_before_pv": "bfloat16",
            "output": "bfloat16",
            "position_order": "ascending_global_position_private_copy",
        },
        "algorithm": {
            "hardware": "TPU v4 TensorCore",
            "selected_kv_hbm_tensor": False,
            "cache_fetch": "scalar-dynamic HBM-to-VMEM DMA per selected row",
            "dma_rows_per_selected_row": config.dma_rows,
            "dma_overfetch": "8x TPU-v4 VMEM tile granularity; one-hot lane selection",
            "external_lane_metadata": "bfloat16[2048,1,8] one-hot (32 KiB)",
            "tile_consumption": "immediate online softmax and PV",
            "returned_state": "owner partial output plus additive LSE",
        },
        "compile_seconds": {
            "pallas": pallas_compile_seconds,
            "reference": reference_compile_seconds,
        },
        "hlo": {
            "path": str(args.hlo_output),
            "sha256": sha256(hlo.encode()).hexdigest(),
            "contract": hlo_contract,
        },
        "reference_hlo": reference_hlo_record,
        "comparison": {
            "cases": comparisons,
            "invalid_page_safety": invalid_page_safety,
            "passed": bool(
                all(item["passed"] for item in comparisons.values())
                and invalid_page_safety["passed"]
            ),
        },
        "profiler_free_timing": True,
        "diagnostic_only": args.diagnostic_reference_timing,
        "performance_claim": False,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency": latency,
        "reference_latency": reference_latency or None,
        "reference_checksum": reference_checksums or None,
        "checksum": checksums,
        "memory_stats": _memory_stats(device),
    }
    _atomic_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
