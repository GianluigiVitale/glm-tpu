#!/usr/bin/env python3
"""Measure the dependent small-collective floor on the live 4x8 TPU mesh.

This is intentionally a no-model discriminator.  It reproduces the dominant
MoE combine payload (bf16[2, 6144]) and unrolls exactly 75 dependent stages,
then compares the current physical 32-way group with dcp-8, model-4, and a
mathematically global sequential dcp-8 + model-4 schedule.

Eight copies of this program must be launched together, one per TPU VM.  The
launcher supplies GLM_MB_COORDINATOR and GLM_MB_PROCESS_ID.  Every process
executes and validates the same global program; process zero writes the JSON
and optimized HLO artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import statistics
import time
from typing import Any, Callable

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np


_PROCESS_COUNT = 8
_LOCAL_DEVICE_COUNT = 4
_GLOBAL_DEVICE_COUNT = _PROCESS_COUNT * _LOCAL_DEVICE_COUNT
_LAYERS = 75
_PAYLOAD_SHAPE = (2, 6144)
_MESH_SHAPE = (1, 1, 1, 1, 4, 8)
_MESH_AXES = ("data", "attn_dp", "attn_dp_expert", "expert", "model",
              "dcp")
_WARMUP = 5
_SAMPLES = 40
_COMPILER_OPTIONS = {
    "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative",
    "xla_tpu_reduce_scatter_collective_matmul_mode":
    "post_spmd_conservative",
    "xla_tpu_use_minor_sharding_for_major_trivial_input": "true",
}


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _device_record(device: jax.Device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": int(device.id),
        "process_index": int(device.process_index),
        "platform": device.platform,
        "repr": str(device),
    }
    for field in ("coords", "core_on_chip", "device_kind"):
        if hasattr(device, field):
            value = getattr(device, field)
            if isinstance(value, tuple):
                value = list(value)
            record[field] = value
    return record


def _hlo_collective_counts(hlo: str) -> dict[str, int]:
    """Count scheduled HLO collective instructions, not textual mentions."""
    patterns = {
        "all_reduce_sync": r"\sall-reduce\(",
        "all_reduce_start": r"\sall-reduce-start\(",
        "all_reduce_done": r"\sall-reduce-done\(",
        "all_gather_sync": r"\sall-gather\(",
        "all_gather_start": r"\sall-gather-start\(",
        "all_gather_done": r"\sall-gather-done\(",
        "reduce_scatter_sync": r"\sreduce-scatter\(",
        "reduce_scatter_start": r"\sreduce-scatter-start\(",
        "reduce_scatter_done": r"\sreduce-scatter-done\(",
        "collective_permute_sync": r"\scollective-permute\(",
        "collective_permute_start": r"\scollective-permute-start\(",
        "collective_permute_done": r"\scollective-permute-done\(",
        "optimization_barrier": r"\soptimization-barrier\(",
    }
    counts = {
        name: sum(bool(re.search(pattern, line)) for line in hlo.splitlines())
        for name, pattern in patterns.items()
    }
    counts["logical_all_reduce"] = (counts["all_reduce_sync"] +
                                     counts["all_reduce_start"])
    return counts


def _nonlinear_step(value: jax.Array, previous: jax.Array,
                    layer: int) -> jax.Array:
    # A nonlinear, layer-dependent recurrence plus an explicit optimization
    # barrier prevents XLA from folding or reassociating adjacent collectives.
    layer_bias = jnp.asarray((layer + 1) / 32768.0, dtype=jnp.bfloat16)
    mixed = jnp.tanh(value + layer_bias) + previous * jnp.asarray(
        1.0 / 64.0, dtype=jnp.bfloat16)
    return lax.optimization_barrier(mixed)


def _chain_body(kind: str) -> Callable[[jax.Array], jax.Array]:
    if kind not in {"full32", "dcp8", "model4", "sequential8x4"}:
        raise ValueError(kind)

    def chain(value: jax.Array) -> jax.Array:
        model_rank = lax.axis_index("model").astype(jnp.bfloat16)
        dcp_rank = lax.axis_index("dcp").astype(jnp.bfloat16)
        epsilon = jnp.asarray(1.0 / 4096.0, dtype=jnp.bfloat16)
        current = value
        for layer in range(_LAYERS):
            if kind == "full32":
                rank = model_rank * jnp.asarray(8, jnp.bfloat16) + dcp_rank
                reduced = lax.psum(current + (rank + 1) * epsilon,
                                   ("model", "dcp"))
                reduced /= jnp.asarray(32, jnp.bfloat16)
            elif kind == "dcp8":
                reduced = lax.psum(current + (dcp_rank + 1) * epsilon,
                                   "dcp")
                reduced /= jnp.asarray(8, jnp.bfloat16)
            elif kind == "model4":
                reduced = lax.psum(current + (model_rank + 1) * epsilon,
                                   "model")
                reduced /= jnp.asarray(4, jnp.bfloat16)
            else:
                reduced_dcp = lax.psum(
                    current + (dcp_rank + 1) * epsilon, "dcp")
                reduced_dcp /= jnp.asarray(8, jnp.bfloat16)
                reduced_dcp = lax.optimization_barrier(reduced_dcp)
                reduced = lax.psum(
                    reduced_dcp + (model_rank + 1) * epsilon, "model")
                reduced /= jnp.asarray(4, jnp.bfloat16)
            current = _nonlinear_step(reduced, current, layer)
        return current

    return chain


def _build_variant(mesh: Mesh, kind: str) -> Callable[[jax.Array], jax.Array]:
    mapped = jax.shard_map(_chain_body(kind),
                           mesh=mesh,
                           in_specs=P(),
                           out_specs=P(),
                           check_vma=False)
    return jax.jit(mapped, compiler_options=_COMPILER_OPTIONS)


def _benchmark_variant(mesh: Mesh, value: jax.Array, kind: str,
                       artifact_dir: Path, process_id: int) -> dict[str, Any]:
    started = time.perf_counter()
    lowered = _build_variant(mesh, kind).lower(value)
    compiled = lowered.compile()
    compile_s = time.perf_counter() - started
    hlo = compiled.as_text()
    counts = _hlo_collective_counts(hlo)
    expected = _LAYERS * (2 if kind == "sequential8x4" else 1)
    if counts["logical_all_reduce"] != expected:
        raise AssertionError(
            f"{kind}: expected {expected} scheduled all-reduces, got "
            f"{counts['logical_all_reduce']}; counts={counts}")

    output = compiled(value)
    output.block_until_ready()
    for _ in range(_WARMUP - 1):
        output = compiled(value)
        output.block_until_ready()

    samples_ms: list[float] = []
    for _ in range(_SAMPLES):
        before = time.perf_counter_ns()
        output = compiled(value)
        output.block_until_ready()
        samples_ms.append((time.perf_counter_ns() - before) / 1_000_000.0)

    local_output = np.asarray(jax.device_get(output.addressable_data(0)),
                              dtype=np.float32)
    if not np.isfinite(local_output).all():
        raise AssertionError(f"{kind}: non-finite output")

    if process_id == 0:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"{kind}.optimized_hlo.txt").write_text(hlo)

    result = {
        "kind": kind,
        "layers": _LAYERS,
        "payload_shape": list(_PAYLOAD_SHAPE),
        "payload_dtype": "bfloat16",
        "expected_collectives": expected,
        "hlo_counts": counts,
        "compile_s": compile_s,
        "warmup": _WARMUP,
        "samples": _SAMPLES,
        "latency_ms": {
            "mean": statistics.fmean(samples_ms),
            "median": statistics.median(samples_ms),
            "p10": _percentile(samples_ms, 0.10),
            "p90": _percentile(samples_ms, 0.90),
            "min": min(samples_ms),
            "max": max(samples_ms),
        },
        "per_collective_median_ms": statistics.median(samples_ms) / expected,
        "output_first": float(local_output.flat[0]),
        "output_sum": float(local_output.sum(dtype=np.float64)),
    }
    print("GLM_MB_VARIANT " + json.dumps({
        "host": socket.gethostname(),
        "process_id": process_id,
        "kind": kind,
        "median_ms": result["latency_ms"]["median"],
        "logical_all_reduce": counts["logical_all_reduce"],
        "output_first": result["output_first"],
    }, sort_keys=True), flush=True)
    return result


def main() -> None:
    coordinator = _required_env("GLM_MB_COORDINATOR")
    process_id = int(_required_env("GLM_MB_PROCESS_ID"))
    if not 0 <= process_id < _PROCESS_COUNT:
        raise RuntimeError(f"invalid GLM_MB_PROCESS_ID={process_id}")
    result_path = Path(_required_env("GLM_MB_RESULT_PATH"))
    artifact_dir = result_path.parent / "hlo"

    jax.distributed.initialize(coordinator_address=coordinator,
                               num_processes=_PROCESS_COUNT,
                               process_id=process_id)
    try:
        if jax.process_count() != _PROCESS_COUNT:
            raise AssertionError(jax.process_count())
        if jax.local_device_count() != _LOCAL_DEVICE_COUNT:
            raise AssertionError(jax.local_device_count())
        if jax.device_count() != _GLOBAL_DEVICE_COUNT:
            raise AssertionError(jax.device_count())

        devices = list(jax.devices())
        devices_array = mesh_utils.create_device_mesh(
            _MESH_SHAPE,
            devices,
            allow_split_physical_axes=True,
        )
        mesh = Mesh(devices_array, _MESH_AXES)
        replicated = NamedSharding(mesh, P())
        host_value = np.linspace(-0.25, 0.25,
                                 num=np.prod(_PAYLOAD_SHAPE),
                                 dtype=np.float32).reshape(_PAYLOAD_SHAPE)
        value = jax.device_put(host_value.astype(jnp.bfloat16), replicated)

        variants = []
        for kind in ("full32", "dcp8", "model4", "sequential8x4"):
            variants.append(
                _benchmark_variant(mesh, value, kind, artifact_dir,
                                   process_id))

        medians = {
            result["kind"]: result["latency_ms"]["median"]
            for result in variants
        }
        summary = {
            "schema": 1,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
            "host": socket.gethostname(),
            "process_id": process_id,
            "jax_version": jax.__version__,
            "process_count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "global_device_count": jax.device_count(),
            "mesh_shape": list(_MESH_SHAPE),
            "mesh_axes": list(_MESH_AXES),
            "mesh_devices": [
                _device_record(device)
                for device in np.asarray(devices_array).reshape(-1)
            ],
            "compiler_options": _COMPILER_OPTIONS,
            "variants": variants,
            "speedup_vs_full32": {
                name: medians["full32"] / latency
                for name, latency in medians.items()
            },
        }
        if process_id == 0:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(summary, indent=2,
                                              sort_keys=True) + "\n")
            print("GLM_MB_RESULT " + json.dumps(summary,
                                                sort_keys=True), flush=True)
        print(f"GLM_MB_HOST_OK {socket.gethostname()} process_id={process_id}",
              flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
