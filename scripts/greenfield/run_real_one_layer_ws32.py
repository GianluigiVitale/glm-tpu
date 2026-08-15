#!/usr/bin/env python3
"""Distributed compile/correctness runner for the bounded real WS32 layer."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking import (  # noqa: E402
    REAL_LAYER_OUTPUT_TOLERANCE,
    build_ws32_one_layer_mapped,
    compare_bounded_tensor,
    latency_distribution,
    validate_ws32_one_layer_hlo,
    validate_ws32_topology_fleet,
    ws32_one_layer_inputs,
)
from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    load_ws32_one_layer_global,
)
from glm_tpu.greenfield.kernels.reference import (  # noqa: E402
    GlmMoeNumericalContract,
)
from glm_tpu.greenfield.sharding.ws32 import (  # noqa: E402
    build_ws32_physical_mesh,
)
from glm_tpu.greenfield.validation import inspect_one_layer_oracle  # noqa: E402


EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--topology-capture-root", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--packed-manifest-sha256", required=True)
    parser.add_argument("--oracle-manifest-sha256", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--topology-fleet-sha256", required=True)
    parser.add_argument("--mesh-sha256", required=True)
    parser.add_argument("--expected-stablehlo-sha256", required=True)
    parser.add_argument("--expected-optimized-hlo-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stablehlo-output", type=Path, required=True)
    parser.add_argument("--optimized-hlo-output", type=Path, required=True)
    parser.add_argument("--prevalidation-output", type=Path, required=True)
    parser.add_argument("--compile-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            allow_nan=False,
            default=lambda item: item.item()
            if isinstance(item, np.generic)
            else str(item),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_record(
    device: object, *, observed_local_device_id: int
) -> dict[str, Any]:
    runtime_local_device_id = device.local_hardware_id
    if (
        runtime_local_device_id is not None
        and int(runtime_local_device_id) != observed_local_device_id
    ):
        raise ValueError(
            "runtime and fleet-observed local device ids disagree: "
            f"{runtime_local_device_id} != {observed_local_device_id}"
        )
    return {
        "coordinates": [int(value) for value in device.coords],
        "core_on_chip": int(device.core_on_chip),
        "device_id": int(device.id),
        "device_kind": str(device.device_kind),
        "local_device_id": observed_local_device_id,
        "platform": str(device.platform),
        "process_index": int(device.process_index),
    }


def _memory_stats(device: object) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _compiled_memory_analysis(compiled: Any) -> dict[str, int | None]:
    analysis = compiled.memory_analysis()
    fields = (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    return {
        field: None
        if getattr(analysis, field, None) is None
        else int(getattr(analysis, field))
        for field in fields
    }


def _bfloat16_numpy(tensor: Any) -> np.ndarray:
    import ml_dtypes
    import torch

    if tensor.dtype != torch.bfloat16:
        raise ValueError("oracle tensor is not BF16")
    return (
        tensor.contiguous().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    )


def _load_oracle(
    oracle_dir: Path,
    *,
    expected_manifest_sha256: str,
    pack_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    manifest = inspect_one_layer_oracle(oracle_dir)
    if manifest.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("WS32 oracle manifest identity drifted")
    if manifest.get("model_id") != pack_manifest.get("model_id") or (
        manifest.get("layer") != pack_manifest.get("layer")
    ):
        raise ValueError("WS32 oracle model/layer identity drifted")
    for name in (
        "fp8_block_shape",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
    ):
        if manifest["geometry"].get(name) != pack_manifest["geometry"].get(name):
            raise ValueError(f"WS32 oracle geometry drifted for {name}")
    path = oracle_dir / manifest["file"]["filename"]
    if _sha256_file(path) != manifest["file"]["sha256"]:
        raise ValueError("WS32 oracle file checksum drifted")
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    return manifest, tensors


def _case_inputs(
    jax: Any,
    mesh: Any,
    loaded: Any,
    oracle: Mapping[str, Any],
    case: str,
) -> tuple[Any, ...]:
    from jax.sharding import NamedSharding, PartitionSpec as P

    hidden = jax.device_put(
        _bfloat16_numpy(oracle["hidden_states"]),
        NamedSharding(mesh, P(None, "feature")),
    )
    route_indices = jax.device_put(
        oracle[f"{case}_route_indices"].numpy(), NamedSharding(mesh, P())
    )
    route_weights = jax.device_put(
        oracle[f"{case}_route_weights"].numpy(), NamedSharding(mesh, P())
    )
    return ws32_one_layer_inputs(
        hidden, route_indices, route_weights, loaded.arrays
    )


def _case_result(
    jax: Any,
    result: Any,
    oracle: Mapping[str, Any],
    case: str,
    slot_by_device_id: Mapping[int, int],
) -> dict[str, Any]:
    expected = _bfloat16_numpy(oracle[f"{case}_output"])
    records = []
    for shard in result.addressable_shards:
        device_id = int(shard.device.id)
        slot = slot_by_device_id[device_id]
        expert_coordinate, feature_coordinate = divmod(slot, 4)
        observed = np.asarray(jax.device_get(shard.data))
        start = feature_coordinate * 1536
        expected_shard = expected[:, start : start + 1536]
        comparison = compare_bounded_tensor(
            observed, expected_shard, REAL_LAYER_OUTPUT_TOLERANCE
        )
        observed_bits = np.ascontiguousarray(observed).view(np.uint16)
        expected_bits = np.ascontiguousarray(expected_shard).view(np.uint16)
        records.append(
            {
                "device_id": device_id,
                "device_slot": slot,
                "expert_coordinate": expert_coordinate,
                "feature_coordinate": feature_coordinate,
                "observed_bf16_sha256": sha256(observed_bits.tobytes()).hexdigest(),
                "oracle_bf16_sha256": sha256(expected_bits.tobytes()).hexdigest(),
                "bitwise_mismatch_count": int(np.count_nonzero(observed_bits != expected_bits)),
                "comparison": comparison,
            }
        )
    records.sort(key=lambda item: item["device_slot"])
    return {
        "case": case,
        "local_shards": records,
        "passed": len(records) == 4
        and all(item["comparison"]["passed"] for item in records),
    }


def main() -> int:
    args = parse_args()
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise ValueError("WS32 runner requires exactly eight launch processes")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("WS32 runner warmup/iterations must be positive")
    for path in (
        args.output,
        args.stablehlo_output,
        args.optimized_hlo_output,
        args.prevalidation_output,
    ):
        if path.exists():
            raise FileExistsError(f"WS32 output is append-only: {path}")
    if REPO != EXPECTED_WORKTREE or _git_head() != args.expected_code_hash:
        raise RuntimeError("WS32 runner code/worktree identity drifted")

    import jax
    from jax.sharding import Mesh

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    if (
        jax.default_backend() != "tpu"
        or jax.device_count() != 32
        or len(jax.local_devices()) != 4
        or jax.process_count() != 8
    ):
        raise RuntimeError("WS32 runner did not initialize the exact 8x4 TPU runtime")

    captures = tuple(
        json.loads(
            (
                args.topology_capture_root
                / f"topology.rank{launch_process_id}.json"
            ).read_text()
        )
        for launch_process_id in range(8)
    )
    topology, ordered_captures, topology_fleet_sha256 = (
        validate_ws32_topology_fleet(
            captures,
            expected_topology_sha256=args.topology_sha256,
            expected_fleet_sha256=args.topology_fleet_sha256,
            slice_name=args.slice_name,
        )
    )
    launch_capture = ordered_captures[args.process_id]
    if (
        launch_capture["hostname"] != socket.gethostname()
        or launch_capture["jax_process_index"] != jax.process_index()
        or launch_capture["local_device_ids"]
        != [int(device.id) for device in jax.local_devices()]
    ):
        raise RuntimeError("WS32 launch/JAX/topology fleet mapping drifted")
    physical_mesh = build_ws32_physical_mesh(topology)
    if physical_mesh.mesh_hash != args.mesh_sha256:
        raise ValueError("WS32 physical mesh hash drifted")
    runtime_by_id = {int(device.id): device for device in jax.devices()}
    if set(runtime_by_id) != set(physical_mesh.flattened_device_ids):
        raise ValueError("WS32 runtime device ids differ from physical mesh")
    for captured in topology.devices:
        runtime = runtime_by_id[captured.device_id]
        if (
            _device_record(
                runtime,
                observed_local_device_id=captured.local_device_id,
            )
            != captured.to_dict()
        ):
            raise ValueError(f"WS32 runtime topology drifted at {captured.device_id}")
    topology_by_id = {item.device_id: item for item in topology.devices}
    mesh = Mesh(
        np.asarray(
            [runtime_by_id[item] for item in physical_mesh.flattened_device_ids],
            dtype=object,
        ).reshape(8, 4),
        ("expert", "feature"),
    )

    load_started = time.perf_counter()
    loaded = load_ws32_one_layer_global(
        args.artifact_dir,
        expected_manifest_sha256=args.packed_manifest_sha256,
        mesh=mesh,
        physical_mesh=physical_mesh,
        payload_subdirectory="packed",
    )
    load_seconds = time.perf_counter() - load_started
    oracle_manifest, oracle = _load_oracle(
        args.oracle_dir,
        expected_manifest_sha256=args.oracle_manifest_sha256,
        pack_manifest=loaded.manifest,
    )
    contract = GlmMoeNumericalContract(stage_size=8)
    mapped = build_ws32_one_layer_mapped(mesh, contract=contract)
    normal_inputs = _case_inputs(jax, mesh, loaded, oracle, "normal")
    lowered = jax.jit(mapped).lower(*normal_inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    _atomic_text(args.stablehlo_output, stablehlo)
    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    optimized_hlo = compiled.as_text()
    _atomic_text(args.optimized_hlo_output, optimized_hlo)
    report = validate_ws32_one_layer_hlo(
        stablehlo,
        optimized_hlo,
        expected_stablehlo_sha256=args.expected_stablehlo_sha256,
        expected_optimized_hlo_sha256=args.expected_optimized_hlo_sha256,
    )
    prevalidation = {
        "code_hash": args.expected_code_hash,
        "compiled_memory_analysis": _compiled_memory_analysis(compiled),
        "compile_seconds": compile_seconds,
        "device_memory_after_load": list(loaded.device_memory_after),
        "device_memory_before_load": list(loaded.device_memory_before),
        "hlo": report.to_dict(),
        "hostname": socket.gethostname(),
        "jax_devices": [
            _device_record(
                device,
                observed_local_device_id=topology_by_id[int(device.id)].local_device_id,
            )
            for device in jax.local_devices()
        ],
        "jax_process_index": jax.process_index(),
        "launch_process_id": args.process_id,
        "load_seconds": load_seconds,
        "local_device_slots": list(loaded.local_device_slots),
        "mesh_sha256": physical_mesh.mesh_hash,
        "oracle_manifest_sha256": oracle_manifest["manifest_sha256"],
        "packed_manifest_sha256": loaded.manifest["manifest_sha256"],
        "topology_sha256": topology.topology_hash,
        "topology_fleet_sha256": topology_fleet_sha256,
    }
    _atomic_json(args.prevalidation_output, prevalidation)
    vacant_violations = {
        "StableHLO pin is intentionally vacant",
        "optimized HLO pin is intentionally vacant",
    }
    if args.compile_only:
        if (
            args.expected_stablehlo_sha256 != "0" * 64
            or args.expected_optimized_hlo_sha256 != "0" * 64
            or set(report.violations) != vacant_violations
        ):
            raise RuntimeError(
                "WS32 compile acquisition has non-vacant or structural failures: "
                + "; ".join(report.violations)
            )
        _atomic_json(
            args.output,
            {
                **prevalidation,
                "artifact_kind": "greenfield_ws32_real_layer3_hlo_acquisition",
                "performance_claim": False,
                "schema_version": 1,
                "status": "HLO_ACQUIRED",
            },
        )
        print(
            "GREENFIELD_WS32_HLO_ACQUIRED "
            f"rank={args.process_id} stable={report.stablehlo_sha256} "
            f"optimized={report.optimized_hlo_sha256}",
            flush=True,
        )
        return 0
    if not report.passed:
        raise RuntimeError("WS32 pre-execution HLO contract failed: " + "; ".join(report.violations))

    slot_by_device_id = {
        device_id: slot
        for slot, device_id in enumerate(physical_mesh.flattened_device_ids)
    }
    cases: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    for case in ("normal", "concentrated"):
        inputs = normal_inputs if case == "normal" else _case_inputs(
            jax, mesh, loaded, oracle, case
        )
        first = compiled(*inputs)
        jax.block_until_ready(first)
        cases[case] = _case_result(
            jax, first, oracle, case, slot_by_device_id
        )
        for _ in range(args.warmup - 1):
            jax.block_until_ready(compiled(*inputs))
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            jax.block_until_ready(compiled(*inputs))
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        timing[case] = {
            "iterations": args.iterations,
            "latency": latency_distribution(samples).to_dict(),
            "performance_claim": False,
            "profiler_active": False,
            "samples_ms": samples,
            "warmup": args.warmup,
        }
    if not all(value["passed"] for value in cases.values()):
        raise RuntimeError("WS32 real-layer oracle comparison failed")
    record = {
        **prevalidation,
        "artifact_kind": "greenfield_ws32_real_layer3",
        "correctness": cases,
        "device_memory_after_execute": [
            _memory_stats(device) for device in jax.local_devices()
        ],
        "performance_claim": False,
        "schema_version": 1,
        "status": "SUCCESS",
        "timing": timing,
    }
    _atomic_json(args.output, record)
    print(
        "GREENFIELD_WS32_REAL_LAYER_OK "
        f"rank={args.process_id} stable={report.stablehlo_sha256} "
        f"optimized={report.optimized_hlo_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
