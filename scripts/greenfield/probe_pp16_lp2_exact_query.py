#!/usr/bin/env python3
"""Bounded real-weight PP16 LP2 exact-DSA-query discriminator."""

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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _comparison(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.ascontiguousarray(expected)
    actual = np.ascontiguousarray(actual)
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(
            "PP16 query comparison geometry drifted: "
            f"expected={expected.shape}/{expected.dtype} "
            f"actual={actual.shape}/{actual.dtype}"
        )
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "actual_sha256": _sha256_array(actual),
        "elementwise_exact": bool(np.array_equal(expected, actual)),
        "expected_sha256": _sha256_array(expected),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
        "mismatch_count": int(np.count_nonzero(actual != expected)),
        "shape": list(expected.shape),
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "samples_ms": [float(value) for value in array],
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


def _record_hlo(
    hlo_dir: Path, name: str, lowered: Any, compiled: Any
) -> tuple[str, str, dict[str, Any]]:
    stablehlo = lowered.as_text()
    optimized_hlo = compiled.as_text()
    stable_path = hlo_dir / f"{name}.stablehlo.mlir"
    optimized_path = hlo_dir / f"{name}.optimized_hlo.txt"
    _atomic_text(stable_path, stablehlo)
    _atomic_text(optimized_path, optimized_hlo)
    return stablehlo, optimized_hlo, {
        "optimized_hlo": {
            "filename": optimized_path.name,
            "sha256": sha256(optimized_hlo.encode()).hexdigest(),
        },
        "stablehlo": {
            "filename": stable_path.name,
            "sha256": sha256(stablehlo.encode()).hexdigest(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--association-input-dir", type=Path, required=True)
    parser.add_argument("--association-input-manifest-sha256", required=True)
    parser.add_argument("--association-input-file-sha256", required=True)
    parser.add_argument("--internal-observer-dir", type=Path, required=True)
    parser.add_argument("--internal-contract-sha256", required=True)
    parser.add_argument("--internal-tensor-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
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
    if (args.warmup, args.iterations) != (1, 3):
        raise ValueError("bounded LP2 discriminator requires exactly 1/3 samples")
    association_file = args.association_input_dir / "layer0_dsa_input.safetensors"
    if _sha256_file(association_file) != args.association_input_file_sha256:
        raise RuntimeError("association input file hash drifted")

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.kernels.stage_local import (
        _local_dsa_query_tuple4_exact,
    )
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_dsa_query_decoder_association,
        validate_dsa_query_weight_materializer_hlo,
    )
    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        inspect_greenfield_layer0_dsa_internal_observation,
        inspect_layer0_dsa_association_input,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError(
            "LP2 query discriminator requires one four-chip TPU host, got "
            f"backend={jax.default_backend()} local={jax.local_device_count()}"
        )
    devices = tuple(jax.local_devices()[:2])
    coordinates = [list(device.coords) for device in devices]
    if [device.id for device in devices] != [0, 1] or coordinates != [
        [0, 0, 0],
        [1, 0, 0],
    ]:
        raise RuntimeError(
            "LP2 query discriminator lost the protected adjacent stage-0 pair: "
            f"ids={[device.id for device in devices]} coords={coordinates}"
        )
    mesh = Mesh(np.asarray(devices, dtype=object), ("stage",))
    replicated = NamedSharding(mesh, P())
    weight_sharding = NamedSharding(mesh, P("stage", None))
    head_sharding = NamedSharding(mesh, P("stage", None))
    args.hlo_dir.mkdir(parents=True, exist_ok=True)

    _, association = inspect_layer0_dsa_association_input(
        args.association_input_dir,
        expected_manifest_sha256=args.association_input_manifest_sha256,
    )
    _, internals = inspect_greenfield_layer0_dsa_internal_observation(
        args.internal_observer_dir,
        expected_contract_sha256=args.internal_contract_sha256,
        expected_tensor_sha256=args.internal_tensor_sha256,
    )
    contract = DsaNumericalContract()
    q_residual_host = np.ascontiguousarray(
        internals["q_a_state_bfloat16_bits"][0:1]
    ).view(ml_dtypes.bfloat16)
    normalized_host = np.ascontiguousarray(
        internals["normalized_hidden_bfloat16_bits"][0:1]
    ).view(ml_dtypes.bfloat16)
    query_expected = np.ascontiguousarray(internals["query"][0:1])
    head_expected = np.ascontiguousarray(internals["head_weights"][0:1])
    position_host = np.asarray([8155], dtype=np.int32)
    wq_bits_host = np.ascontiguousarray(
        association["self_attn__indexer__wq_b__weight"]
    )
    wq_scale_host = np.ascontiguousarray(
        association["self_attn__indexer__wq_b__weight_scale_inv"]
    )
    head_weight_host = np.ascontiguousarray(
        association["self_attn__indexer__weights_proj__weight"]
    ).view(ml_dtypes.bfloat16)

    q_residual = jax.device_put(q_residual_host, replicated)
    normalized = jax.device_put(normalized_host, replicated)
    position = jax.device_put(position_host, replicated)
    wq_bits = jax.device_put(wq_bits_host, weight_sharding)
    wq_scale = jax.device_put(wq_scale_host, weight_sharding)
    head_weight = jax.device_put(head_weight_host, head_sharding)

    materializer_mapped = jax.shard_map(
        lambda bits, scale: dequantize_fp8_bits_block_weight(
            bits, scale, output_dtype=jnp.float32
        ),
        mesh=mesh,
        in_specs=(P("stage", None), P("stage", None)),
        out_specs=P("stage", None),
        check_vma=False,
    )
    materializer_started = time.monotonic()
    materializer_lowered = jax.jit(materializer_mapped).lower(
        wq_bits, wq_scale
    )
    materializer_compiled = materializer_lowered.compile()
    materializer_compile_seconds = time.monotonic() - materializer_started
    materializer_stablehlo, materializer_hlo, materializer_record = _record_hlo(
        args.hlo_dir,
        "pp16_lp2_query_weight_materializer",
        materializer_lowered,
        materializer_compiled,
    )
    materializer_contract = validate_dsa_query_weight_materializer_hlo(
        materializer_hlo,
        full_indexer_slots=1,
        local_output_width=2048,
        q_lora_rank=2048,
        total_devices=2,
    )
    if not materializer_contract["passed"]:
        raise RuntimeError(
            f"LP2 query materializer HLO failed: {materializer_contract}"
        )
    materialized = materializer_compiled(wq_bits, wq_scale)
    jax.block_until_ready(materialized)

    query_mapped = jax.shard_map(
        lambda q, n, w0, w1, w2, w3, h, p: _local_dsa_query_tuple4_exact(
            q, n, (w0, w1, w2, w3), h, p, contract=contract
        ),
        mesh=mesh,
        in_specs=(
            P(),
            P(),
            P("stage", None),
            P("stage", None),
            P("stage", None),
            P("stage", None),
            P("stage", None),
            P(),
        ),
        out_specs=(P(None, "stage", None), P(None, "stage")),
        check_vma=False,
    )
    query_started = time.monotonic()
    query_lowered = jax.jit(query_mapped).lower(
        q_residual,
        normalized,
        materialized,
        materialized,
        materialized,
        materialized,
        head_weight,
        position,
    )
    query_compiled = query_lowered.compile()
    query_compile_seconds = time.monotonic() - query_started
    query_stablehlo, query_hlo, query_record = _record_hlo(
        args.hlo_dir,
        "pp16_lp2_exact_query_head",
        query_lowered,
        query_compiled,
    )
    query_contract = _validate_dsa_query_decoder_association(
        query_hlo,
        full_indexer_layers=1,
        local_parallel_size=2,
        dsa_indexer_heads=32,
        index_key_width=128,
        backend="reference",
        exact_association=True,
    )
    stablehlo_contract = {
        "dot_general_count": query_stablehlo.count("stablehlo.dot_general"),
        "expected_dot_general_count": 9,
        "optimization_barrier_count": query_stablehlo.count(
            "stablehlo.optimization_barrier"
        ),
        "expected_optimization_barrier_count": 3,
        "all_gather_count": query_stablehlo.count("stablehlo.all_gather"),
        "host_callback_count": sum(
            query_stablehlo.lower().count(marker)
            for marker in (
                "host_callback",
                "outside_compilation",
                "xla_ffi_python_cpu_callback",
                "xla_python_cpu_callback",
            )
        ),
    }
    stablehlo_contract["passed"] = (
        stablehlo_contract["dot_general_count"]
        == stablehlo_contract["expected_dot_general_count"]
        and stablehlo_contract["optimization_barrier_count"]
        == stablehlo_contract["expected_optimization_barrier_count"]
        and stablehlo_contract["all_gather_count"] == 0
        and stablehlo_contract["host_callback_count"] == 0
    )
    if not query_contract["passed"] or not stablehlo_contract["passed"]:
        raise RuntimeError(
            "LP2 exact query HLO failed: "
            f"optimized={query_contract} stable={stablehlo_contract}"
        )

    query_args = (
        q_residual,
        normalized,
        materialized,
        materialized,
        materialized,
        materialized,
        head_weight,
        position,
    )
    for _ in range(args.warmup):
        jax.block_until_ready(query_compiled(*query_args))
    samples_ms = []
    final_output = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        final_output = query_compiled(*query_args)
        jax.block_until_ready(final_output)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    assert final_output is not None
    query_actual = np.ascontiguousarray(np.asarray(final_output[0]))
    head_actual = np.ascontiguousarray(np.asarray(final_output[1]))
    query_comparison = _comparison(query_expected, query_actual)
    head_comparison = _comparison(head_expected, head_actual)
    if not query_comparison["elementwise_exact"] or not head_comparison[
        "elementwise_exact"
    ]:
        raise RuntimeError(
            "LP2 exact query arithmetic drifted: "
            f"query={query_comparison} head={head_comparison}"
        )

    materializer_record["contract"] = materializer_contract
    materializer_record["stablehlo_sha256"] = sha256(
        materializer_stablehlo.encode()
    ).hexdigest()
    query_record["contract"] = query_contract
    query_record["stablehlo_contract"] = stablehlo_contract
    output = {
        "artifact_kind": "greenfield_pp16_lp2_exact_dsa_query_discriminator",
        "claim_scope": (
            "bounded real layer-0 arithmetic/HLO diagnostic; no decoder, "
            "Gate-D, token-rate, or performance claim"
        ),
        "code_hash": code_hash,
        "compile_seconds": {
            "materializer": materializer_compile_seconds,
            "query_head": query_compile_seconds,
        },
        "device_kind": devices[0].device_kind,
        "device_memory": [_memory_stats(device) for device in devices],
        "head_comparison": head_comparison,
        "hlo": {
            "materializer": materializer_record,
            "query_head": query_record,
        },
        "iterations": args.iterations,
        "physical_group": {
            "coordinates": coordinates,
            "device_ids": [device.id for device in devices],
            "local_device_count_visible": jax.local_device_count(),
            "mesh_device_count": 2,
        },
        "profiler_free_timing": True,
        "query_comparison": query_comparison,
        "source": {
            "association_input_file_sha256": args.association_input_file_sha256,
            "association_input_manifest_sha256": (
                args.association_input_manifest_sha256
            ),
            "internal_contract_sha256": args.internal_contract_sha256,
            "internal_tensor_sha256": args.internal_tensor_sha256,
        },
        "status": "SUCCESS",
        "timing": _distribution(samples_ms),
        "warmup": args.warmup,
    }
    _atomic_json(args.output, output)
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
