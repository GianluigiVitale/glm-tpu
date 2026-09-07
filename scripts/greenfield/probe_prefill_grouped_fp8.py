#!/usr/bin/env python3
"""Bounded TPU arithmetic admission, NOT a prefill performance experiment.

Synthetic local WS32 routed-up or down shape; three routing distributions, one compiled
candidate, exact old M1 projection reference. No model checkpoint or timed loop.
Launch only through run_fp8_matmul_microbench.sh's ws32_grouped_admission mode.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield.microbench_fp8_matmul import (
    _atomic_json,
    _compiled_memory,
    _git_head,
    _memory_stats,
)

KERNEL = "ws32_grouped_admission"
PROTOCOL = "ws32-grouped-fp8-up-arithmetic-v1"
M, K, N, LOCAL_GROUPS, GLOBAL_GROUPS, OFFSET = 136, 1536, 2048, 32, 256, 64
CASES = ("distributed", "concentrated_owner", "empty_owner")


def projection_contract(direction: str) -> tuple[str, str, int, int, str]:
    if direction == "up":
        return KERNEL, PROTOCOL, K, N, "float32"
    if direction == "down":
        return (
            "ws32_grouped_down_admission",
            "ws32-grouped-fp8-down-arithmetic-v1",
            N,
            K,
            "bfloat16",
        )
    raise ValueError("grouped admission direction must be up or down")


def counts_for_case(name: str) -> np.ndarray:
    counts = np.zeros(GLOBAL_GROUPS, np.int32)
    if name == "distributed":
        counts[:M] = 1
    elif name == "concentrated_owner":
        counts[OFFSET : OFFSET + 8] = 17
    elif name == "empty_owner":
        counts[:8] = 17
    else:
        raise ValueError(f"unknown admission case {name}")
    return counts


def check_hlo(hlo: str, *, direction: str = "up") -> dict[str, Any]:
    _, _, contraction, output, _ = projection_contract(direction)
    calls = [line.strip() for line in hlo.splitlines() if "custom-call(" in line]
    kernels = [line for line in calls if 'custom_call_target="tpu_custom_call"' in line]
    collectives = re.findall(
        r"\b(?:all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all)(?:-start)?\(",
        hlo,
    )
    # Also reject flattened/transposed full decoded tables, not only [G,N,K].
    full_overlays = []
    for match in re.finditer(r"\b(?:bf16|f32)\[([0-9,]+)\]", hlo):
        if np.prod([int(v) for v in match[1].split(",")]) >= LOCAL_GROUPS * N * K:
            full_overlays.append(match[0])
    raw_shape = f"u8[{LOCAL_GROUPS},{output},{contraction}]"
    return {
        "passed": len(kernels) == 1
        and "greenfield_prefill_grouped_raw_fp8" in kernels[0]
        and raw_shape in kernels[0]
        and not collectives
        and not full_overlays,
        "kernel_calls": kernels,
        "auxiliary_custom_calls": [line for line in calls if line not in kernels],
        "collectives": collectives,
        "full_weight_overlays": full_overlays,
        "scope": "single-chip compilation admission; metadata calls inventoried, not a performance linter",
    }


def validate_record(record: dict[str, Any]) -> None:
    """Reject performance classification and protocol drift before DB success."""
    kernel, protocol, k, n, dtype = projection_contract(record.get("direction", "up"))
    if not (
        record["kernel"] == kernel
        and record["protocol"] == protocol
        and record["admission_only"] is True
        and record["baseline_only"] is False
        and record["performance_claim"] is False
        and record["latency"] is None
        and record["warmup"] == 0
        and record["iterations"] == 0
        and record["shape"]
        == {"lhs": [M, k], "weights": [LOCAL_GROUPS, n, k], "output": [M, n]}
        and record["dtype_contract"]["output"] == dtype
        and record["compiled_memory_estimate"]
        and record["comparison"]["passed"] is True
        and [v["case"] for v in record["comparison"]["cases"]] == list(CASES)
        and all(
            v["passed"] is True and v["bit_mismatches"] == 0
            for v in record["comparison"]["cases"]
        )
    ):
        raise ValueError("grouped arithmetic admission protocol/classification drift")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-output", type=Path, required=True)
    parser.add_argument("--direction", choices=("up", "down"), default="up")
    args = parser.parse_args()
    kernel_name, protocol, k, n, dtype_name = projection_contract(args.direction)
    if (
        REPO != Path("/home/gianl/glm-tpu-topology-rewrite")
        or _git_head() != args.expected_code_hash
    ):
        raise RuntimeError("wrong worktree/code pin")
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from glm_tpu.greenfield.kernels.pallas.fp8_matmul import (
        fp8_block_matmul,
        fp8_block_matmul_f32,
    )
    from glm_tpu.greenfield.kernels.pallas.prefill_grouped_fp8 import (
        prefill_grouped_fp8_matmul,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("requires one four-chip TPU host")
    device = jax.local_devices()[0]
    if device.device_kind != "TPU v4":
        raise RuntimeError("requires TPU v4")
    record = dict(
        status="RUNNING",
        code_hash=args.expected_code_hash,
        kernel=kernel_name,
        protocol=protocol,
        direction=args.direction,
        admission_only=True,
        baseline_only=False,
        performance_claim=False,
        diagnostic_only=False,
        latency=None,
        profiler_free_timing=False,
        warmup=0,
        iterations=0,
        selected_route_case=None,
        device_kind=device.device_kind,
        versions={"jax": version("jax"), "libtpu": version("libtpu")},
        shape={"lhs": [M, k], "weights": [LOCAL_GROUPS, n, k], "output": [M, n]},
        dtype_contract={
            "lhs": "bfloat16",
            "weights": "uint8",
            "scales": "float32",
            "output": dtype_name,
        },
    )
    try:
        rng = np.random.default_rng(243091)
        lhs_host = rng.normal(0, 0.25, (M, k)).astype(ml_dtypes.bfloat16)
        lhs_host[7] = 0
        # Quantize on CPU; no full BF16/FP32 expert table is sent to the TPU.
        bits_host = (
            rng.normal(0, 0.15, (LOCAL_GROUPS, n, k))
            .astype(ml_dtypes.float8_e4m3fn)
            .view(np.uint8)
        )
        scales_host = rng.uniform(0.25, 2.0, (LOCAL_GROUPS, n // 128, k // 128)).astype(
            np.float32
        )
        scales_host[1, 0, 8] = 0
        record["inputs"] = {
            name: dict(sha256=sha256(a.tobytes()).hexdigest(), bytes=a.nbytes)
            for name, a in (
                ("lhs", lhs_host),
                ("weights", bits_host),
                ("scales", scales_host),
            )
        }
        with jax.default_device(device):
            x, w, s = [
                jax.device_put(a, device) for a in (lhs_host, bits_host, scales_host)
            ]
            counts = jax.device_put(counts_for_case(CASES[0]), device)
            offset = jax.device_put(np.asarray(OFFSET, np.int32), device)
            started = time.monotonic()
            dtype = jnp.dtype(dtype_name)

            def candidate(x, w, s, counts, offset):
                return prefill_grouped_fp8_matmul(
                    x, w, s, counts, offset, result_dtype=dtype
                )

            compiled = jax.jit(candidate).lower(x, w, s, counts, offset).compile()
            record["compile_seconds"] = time.monotonic() - started
            hlo = compiled.as_text()
            args.hlo_output.parent.mkdir(parents=True, exist_ok=True)
            args.hlo_output.write_text(hlo)
            record["hlo"] = dict(
                sha256=sha256(hlo.encode()).hexdigest(),
                contract=check_hlo(hlo, direction=args.direction),
            )
            record["compiled_memory_estimate"] = memory = _compiled_memory(compiled)
            _atomic_json(args.output, record)
            if not record["hlo"]["contract"]["passed"]:
                raise RuntimeError(
                    "grouped HLO admission failed; inspect saved exact HLO"
                )
            if (
                sum(
                    memory[name]
                    for name in (
                        "argument_size_in_bytes",
                        "output_size_in_bytes",
                        "temp_size_in_bytes",
                    )
                )
                > 512 * 1024**2
            ):
                raise RuntimeError(
                    "grouped admission compiled allocation exceeds 512MiB"
                )
            reference = (
                jax.jit(
                    fp8_block_matmul_f32 if args.direction == "up" else fp8_block_matmul
                )
                .lower(
                    jax.ShapeDtypeStruct((1, k), jnp.bfloat16),
                    jax.ShapeDtypeStruct((n, k), jnp.uint8),
                    jax.ShapeDtypeStruct((n // 128, k // 128), jnp.float32),
                )
                .compile()
            )
            reference_hlo = reference.as_text()
            reference_path = args.hlo_output.with_name("reference_m1.optimized_hlo.txt")
            reference_path.write_text(reference_hlo)
            record["reference_hlo_sha256"] = sha256(reference_hlo.encode()).hexdigest()
            comparisons = []
            record["comparison"] = {"passed": False, "cases": comparisons}
            for case in CASES:
                counts_host = counts_for_case(case)
                actual, valid = compiled(
                    x, w, s, jax.device_put(counts_host, device), offset
                )
                jax.block_until_ready((actual, valid))
                actual_host = np.asarray(actual)
                expected = np.zeros(
                    (M, n), np.float32 if args.direction == "up" else ml_dtypes.bfloat16
                )
                owners = np.repeat(np.arange(GLOBAL_GROUPS), counts_host)
                for row, group in enumerate(owners):
                    if OFFSET <= group < OFFSET + LOCAL_GROUPS:
                        local = group - OFFSET
                        # Host dispatch here is the deliberately slow reference ONLY.
                        expected[row] = np.asarray(
                            reference(
                                jax.device_put(lhs_host[row : row + 1], device),
                                jax.device_put(bits_host[local], device),
                                jax.device_put(scales_host[local], device),
                            )
                        )[0]
                mismatches = int(
                    np.count_nonzero(
                        actual_host.view(
                            np.uint32 if args.direction == "up" else np.uint16
                        )
                        != expected.view(
                            np.uint32 if args.direction == "up" else np.uint16
                        )
                    )
                )
                passed = (
                    bool(valid)
                    and bool(np.isfinite(actual_host).all())
                    and mismatches == 0
                )
                comparisons.append(
                    dict(
                        case=case,
                        passed=passed,
                        bit_mismatches=mismatches,
                        max_abs=float(
                            np.max(
                                np.abs(
                                    actual_host.astype(np.float32)
                                    - expected.astype(np.float32)
                                )
                            )
                        ),
                        output_sha256=sha256(actual_host.tobytes()).hexdigest(),
                        expected_sha256=sha256(expected.tobytes()).hexdigest(),
                        counts=counts_host.tolist(),
                        offset=OFFSET,
                    )
                )
                _atomic_json(args.output, record)
                print(
                    f"GROUPED_ADMISSION {case} passed={passed} bit_mismatches={mismatches}",
                    flush=True,
                )
                if not passed:
                    raise RuntimeError(f"first arithmetic failure: {case}")
            record["comparison"]["passed"] = True
            record["device_memory_stats_including_reference"] = _memory_stats(device)
            record["checksum"] = comparisons[-1]["output_sha256"]
        validate_record(record)
        record["status"] = "SUCCESS"
        _atomic_json(args.output, record)
        return 0
    except Exception as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        _atomic_json(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
