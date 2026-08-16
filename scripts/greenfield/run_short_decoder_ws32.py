#!/usr/bin/env python3
"""Protected-worker runner for the complete WS32 2K/8K decoder.

The shell orchestrator owns fleet locking, append-only remote publication,
results.db, and cleanup.  This process owns only one launch rank's exact JAX
runtime, checkpoint load, HLO proof, oracle comparison, timing, trace, and
compact cache witness.
"""

from __future__ import annotations

import argparse
import gc
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
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking import (  # noqa: E402
    validate_ws32_decoder_hlo,
    validate_ws32_topology_fleet,
)
from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    load_ws32_runtime_checkpoint,
    verify_ws32_runtime_checkpoint,
)
from glm_tpu.greenfield.partitioning import inspect_source_inventory  # noqa: E402
from glm_tpu.greenfield.runtime import (  # noqa: E402
    Ws32DecoderConfig,
    bind_ws32_decoder_weights,
    build_ws32_decoder_program,
    build_ws32_teacher_forced_prefill_program,
    make_ws32_initial_state,
)
from glm_tpu.greenfield.sharding.ws32 import (  # noqa: E402
    build_ws32_physical_mesh,
)
from glm_tpu.greenfield.types import ModelGeometry  # noqa: E402
from glm_tpu.greenfield.validation import (  # noqa: E402
    compare_ws32_dsa_step,
    compare_ws32_raw_tokens,
    load_ws32_short_context_oracle,
    validate_ws32_cache_probe,
)


_ZERO_SHA = "0" * 64
_XLA_MEMORY_FRACTION = ".95"
_VACANT_HLO_VIOLATIONS = {
    "StableHLO identity drifted",
    "optimized HLO identity drifted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", required=True, type=int)
    parser.add_argument("--process-id", required=True, type=int)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--topology-capture-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    parser.add_argument("--token-oracle-dir", required=True, type=Path)
    parser.add_argument("--dsa-oracle-dir", required=True, type=Path)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-success-sha256", required=True)
    parser.add_argument("--token-oracle-manifest-sha256", required=True)
    parser.add_argument("--dsa-oracle-manifest-sha256", required=True)
    parser.add_argument("--token-oracle-success-sha256", required=True)
    parser.add_argument("--dsa-oracle-success-sha256", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--topology-fleet-sha256", required=True)
    parser.add_argument("--mesh-sha256", required=True)
    for graph in ("prefill", "observer", "decode", "cache-probe"):
        parser.add_argument(
            f"--expected-{graph}-stablehlo-sha256", required=True
        )
        parser.add_argument(
            f"--expected-{graph}-optimized-hlo-sha256", required=True
        )
    parser.add_argument("--context-capacity", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-output", required=True, type=Path)
    parser.add_argument("--hlo-dir", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--compile-only", choices=(0, 1), default=0, type=int)
    parser.add_argument("--observer-steps", default=14, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument("--trace-steps", default=2, type=int)
    return parser.parse_args()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_clean_code(expected: str) -> None:
    if REPO != EXPECTED_WORKTREE or _git_head() != expected:
        raise RuntimeError("WS32 short-decoder worktree/code identity drifted")
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("WS32 protected short decoder requires clean code")


def _atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"append-only WS32 evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"append-only WS32 evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    records = {}
    for name, value in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(value)
        records[name] = {
            "dtype": contiguous.dtype.name,
            "sha256": sha256(contiguous.view(np.uint8).tobytes()).hexdigest(),
            "shape": list(contiguous.shape),
        }
    return {
        "arrays": records,
        "byte_count": path.stat().st_size,
        "filename": path.name,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_record(device: object, *, local_device_id: int) -> dict[str, Any]:
    runtime_local_id = device.local_hardware_id
    if runtime_local_id is not None and int(runtime_local_id) != local_device_id:
        raise ValueError("runtime and captured local device ids disagree")
    return {
        "coordinates": [int(value) for value in device.coords],
        "core_on_chip": int(device.core_on_chip),
        "device_id": int(device.id),
        "device_kind": str(device.device_kind),
        "local_device_id": int(local_device_id),
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


def _compiled_memory(compiled: Any) -> dict[str, int | None]:
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


def _distribution(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("latency distribution requires samples")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(values.size),
        "maximum_ms": float(values.max()),
        "mean_ms": float(values.mean()),
        "minimum_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _geometry() -> ModelGeometry:
    value = json.loads(
        (REPO / "configs/glm-5.2-fp8-config.json").read_text(
            encoding="utf-8"
        )
    )
    return ModelGeometry.from_hf_config(value)


def _replicated(jax: Any, mesh: Any, value: np.ndarray) -> Any:
    from jax.sharding import NamedSharding, PartitionSpec as P

    host = np.ascontiguousarray(value)
    return jax.make_array_from_callback(
        host.shape,
        NamedSharding(mesh, P()),
        lambda _: host.copy(),
    )


def _scalar_token(jax: Any, value: Any) -> int:
    host = np.asarray(jax.device_get(value), dtype=np.int32)
    if host.shape != (1,):
        raise RuntimeError("WS32 token result geometry drifted")
    return int(host[0])


def _write_graph(
    *,
    graph: str,
    lowered: Any,
    compiled: Any,
    hlo_dir: Path,
    expected_stable: str,
    expected_optimized: str,
    hidden_size: int,
) -> tuple[dict[str, Any], str, str]:
    stable = str(lowered.compiler_ir(dialect="stablehlo"))
    optimized = compiled.as_text()
    stable_path = hlo_dir / f"{graph}.stablehlo.mlir"
    optimized_path = hlo_dir / f"{graph}.optimized_hlo.txt"
    _atomic_text(stable_path, stable)
    _atomic_text(optimized_path, optimized)
    report = validate_ws32_decoder_hlo(
        stable,
        optimized,
        expected_stablehlo_sha256=expected_stable,
        expected_optimized_hlo_sha256=expected_optimized,
        hidden_size=hidden_size,
        kind=graph,
    )
    return report.to_dict(), stable, optimized


def _require_graph_authorized(
    report: Mapping[str, Any], *, compile_only: bool
) -> None:
    if compile_only:
        # Acquisition never executes a model graph.  Preserve and validate all
        # four compiler products before reporting any structural refusal so a
        # single protected load cannot fail one graph at a time.
        return
    elif not report["passed"]:
        raise RuntimeError(
            "WS32 numerical graph failed before execution: "
            + "; ".join(report["violations"])
        )


def _require_acquisition_authorized(
    graphs: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {"cache_probe", "decode", "observer", "prefill"}
    if set(graphs) != expected:
        raise RuntimeError("WS32 acquisition did not preserve all four graphs")
    if all(value["passed"] for value in graphs.values()) or any(
        set(value["violations"]) != _VACANT_HLO_VIOLATIONS
        for value in graphs.values()
    ):
        raise RuntimeError("WS32 HLO acquisition found structural violations")


def _trace_files(trace_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(trace_dir.rglob("*.xplane.pb")):
        records.append(
            {
                "byte_count": path.stat().st_size,
                "relative_path": path.relative_to(trace_dir).as_posix(),
                "sha256": _sha256_file(path),
            }
        )
    if len(records) != 1:
        raise RuntimeError(
            f"WS32 profiler produced {len(records)} XPlane files, expected one"
        )
    return records


def _initialize_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
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
            ).read_text(encoding="utf-8")
        )
        for launch_process_id in range(8)
    )
    topology, ordered_captures, fleet_sha = validate_ws32_topology_fleet(
        captures,
        expected_topology_sha256=args.topology_sha256,
        expected_fleet_sha256=args.topology_fleet_sha256,
        slice_name=args.slice_name,
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
        if _device_record(
            runtime_by_id[captured.device_id],
            local_device_id=captured.local_device_id,
        ) != captured.to_dict():
            raise ValueError(
                f"WS32 runtime topology drifted at {captured.device_id}"
            )
    mesh = Mesh(
        np.asarray(
            [runtime_by_id[item] for item in physical_mesh.flattened_device_ids],
            dtype=object,
        ).reshape(8, 4),
        ("expert", "feature"),
    )
    return jax, mesh, physical_mesh, topology, fleet_sha


def main() -> int:
    args = parse_args()
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise ValueError("WS32 short decoder requires eight launch processes")
    if min(args.observer_steps, args.warmup, args.iterations, args.trace_steps) < 1:
        raise ValueError("WS32 execution counts must all be positive")
    if (
        args.output.exists()
        or args.tensor_output.exists()
        or args.hlo_dir.exists()
        or args.trace_dir.exists()
    ):
        raise FileExistsError("WS32 short-decoder evidence is append-only")
    if args.compile_only and any(
        value != _ZERO_SHA
        for name, value in vars(args).items()
        if name.startswith("expected_") and "hlo_sha256" in name
    ):
        raise ValueError("WS32 acquisition requires eight vacant HLO pins")
    if not args.compile_only and any(
        value == _ZERO_SHA
        for name, value in vars(args).items()
        if name.startswith("expected_") and "hlo_sha256" in name
    ):
        raise ValueError("WS32 numerical execution requires eight acquired HLO pins")
    _require_clean_code(args.expected_code_hash)
    if os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") != _XLA_MEMORY_FRACTION:
        raise RuntimeError("WS32 XLA allocator fraction is not pinned to .95")

    geometry = _geometry()
    config = Ws32DecoderConfig(
        geometry=geometry,
        context_capacity=args.context_capacity,
    )
    oracle = load_ws32_short_context_oracle(
        args.token_oracle_dir,
        args.dsa_oracle_dir,
        expected_token_manifest_sha256=args.token_oracle_manifest_sha256,
        expected_dsa_manifest_sha256=args.dsa_oracle_manifest_sha256,
        expected_token_success_sha256=args.token_oracle_success_sha256,
        expected_dsa_success_sha256=args.dsa_oracle_success_sha256,
    )
    required_capacity = (
        oracle.prompt_token_ids.size
        + args.observer_steps
        + args.warmup
        + args.iterations
        + args.trace_steps
    )
    if args.context_capacity < required_capacity + 1:
        raise ValueError(
            "WS32 context capacity does not cover prompt plus proof/timing steps"
        )
    if args.observer_steps > oracle.decode_positions.size:
        raise ValueError("WS32 observer steps exceed the sealed DSA oracle")

    jax, mesh, physical_mesh, topology, fleet_sha = _initialize_runtime(args)
    inventory = inspect_source_inventory(args.source_inventory)
    local_device_ids = {int(device.id) for device in jax.local_devices()}
    local_hash_slots = tuple(
        slot
        for slot, device_id in enumerate(physical_mesh.flattened_device_ids)
        if device_id in local_device_ids
    )
    if len(local_hash_slots) != 4:
        raise RuntimeError("WS32 host does not own exactly four checkpoint slots")
    checkpoint = verify_ws32_runtime_checkpoint(
        args.checkpoint_root,
        expected_manifest_sha256=args.checkpoint_manifest_sha256,
        expected_success_sha256=args.checkpoint_success_sha256,
        expected_mesh_hash=args.mesh_sha256,
        expected_topology_hash=args.topology_sha256,
        inventory=inventory,
        geometry=geometry,
        verify_file_hashes=True,
        verify_file_hash_slots=local_hash_slots,
    )
    load_started = time.perf_counter()
    loaded = load_ws32_runtime_checkpoint(
        checkpoint,
        mesh=mesh,
        physical_mesh=physical_mesh,
    )
    load_seconds = time.perf_counter() - load_started
    weights = bind_ws32_decoder_weights(loaded.arrays, config)
    program = build_ws32_decoder_program(mesh, config)
    prefill_program = build_ws32_teacher_forced_prefill_program(
        mesh,
        config,
        prompt_length=int(oracle.prompt_token_ids.size),
    )
    prompt = _replicated(jax, mesh, oracle.prompt_token_ids)
    state = make_ws32_initial_state(mesh, config)
    initial_token = _replicated(jax, mesh, np.asarray([-1], dtype=np.int32))
    args.hlo_dir.mkdir(parents=True, exist_ok=False)

    graphs: dict[str, Any] = {}
    compile_seconds: dict[str, float] = {}
    compiled_memory: dict[str, Any] = {}

    prefill_jit = jax.jit(prefill_program.execute, donate_argnums=(1,))
    prefill_lowered = prefill_jit.lower(prompt, state, weights)
    started = time.perf_counter()
    prefill_compiled = prefill_lowered.compile()
    compile_seconds["prefill"] = time.perf_counter() - started
    compiled_memory["prefill"] = _compiled_memory(prefill_compiled)
    graphs["prefill"], _, _ = _write_graph(
        graph="prefill",
        lowered=prefill_lowered,
        compiled=prefill_compiled,
        hlo_dir=args.hlo_dir,
        expected_stable=args.expected_prefill_stablehlo_sha256,
        expected_optimized=args.expected_prefill_optimized_hlo_sha256,
        hidden_size=geometry.hidden_size,
    )
    _require_graph_authorized(
        graphs["prefill"], compile_only=bool(args.compile_only)
    )

    if args.compile_only:
        observer_state = state
        observer_token = initial_token
    else:
        prefill_result = prefill_compiled(prompt, state, weights)
        jax.block_until_ready(prefill_result)
        observer_state = prefill_result.state
        observer_token = prefill_result.next_token
        state = None
        del prefill_result

    # The complete prefill executable is large and is never used again.  Do
    # not retain three complete-model executables concurrently in 32 GiB HBM.
    prefill_jit.clear_cache()
    del prefill_compiled
    del prefill_lowered
    gc.collect()

    observer_jit = jax.jit(program.observe, donate_argnums=(1,))
    observer_lowered = observer_jit.lower(observer_token, observer_state, weights)
    started = time.perf_counter()
    observer_compiled = observer_lowered.compile()
    compile_seconds["observer"] = time.perf_counter() - started
    compiled_memory["observer"] = _compiled_memory(observer_compiled)
    graphs["observer"], _, _ = _write_graph(
        graph="observer",
        lowered=observer_lowered,
        compiled=observer_compiled,
        hlo_dir=args.hlo_dir,
        expected_stable=args.expected_observer_stablehlo_sha256,
        expected_optimized=args.expected_observer_optimized_hlo_sha256,
        hidden_size=geometry.hidden_size,
    )
    _require_graph_authorized(
        graphs["observer"], compile_only=bool(args.compile_only)
    )

    observed_tokens: list[int] = []
    dsa_steps: list[dict[str, Any]] = []
    observed_dsa_positions: list[np.ndarray] = []
    observed_dsa_counts: list[np.ndarray] = []
    observed_dsa_scores: list[np.ndarray] = []
    observed_dsa_producers: np.ndarray | None = None
    current_state = observer_state
    current_token = observer_token
    if not args.compile_only:
        observed_tokens.append(_scalar_token(jax, observer_token))
        for step in range(args.observer_steps):
            observed = observer_compiled(current_token, current_state, weights)
            jax.block_until_ready(observed)
            host_dsa = jax.device_get(observed.dsa)
            producers = np.asarray(host_dsa.producer_layer_ids, dtype=np.int32)
            positions = np.asarray(host_dsa.selected_positions, dtype=np.int32)
            counts = np.asarray(host_dsa.selected_valid_counts, dtype=np.int32)
            scores = np.asarray(host_dsa.selected_scores, dtype=np.float32)
            if observed_dsa_producers is None:
                observed_dsa_producers = producers
            elif not np.array_equal(observed_dsa_producers, producers):
                raise RuntimeError("WS32 DSA producer identities changed by step")
            comparison = compare_ws32_dsa_step(
                producer_layer_ids=producers,
                selected_positions=positions,
                selected_valid_counts=counts,
                selected_scores=scores,
                oracle=oracle,
                step=step,
            )
            dsa_steps.append(comparison)
            observed_dsa_positions.append(positions)
            observed_dsa_counts.append(counts)
            observed_dsa_scores.append(scores)
            current_state = observed.result.state
            current_token = observed.result.next_token
            observed_tokens.append(_scalar_token(jax, current_token))

    # The proof-only observer is also single-use.  Release it before compiling
    # the production decoder so only one complete-model executable is live
    # during timing and tracing.
    observer_jit.clear_cache()
    del observer_compiled
    del observer_lowered
    gc.collect()

    decode_jit = jax.jit(program.execute, donate_argnums=(1,))
    decode_lowered = decode_jit.lower(current_token, current_state, weights)
    started = time.perf_counter()
    decode_compiled = decode_lowered.compile()
    compile_seconds["decode"] = time.perf_counter() - started
    compiled_memory["decode"] = _compiled_memory(decode_compiled)
    graphs["decode"], _, _ = _write_graph(
        graph="decode",
        lowered=decode_lowered,
        compiled=decode_compiled,
        hlo_dir=args.hlo_dir,
        expected_stable=args.expected_decode_stablehlo_sha256,
        expected_optimized=args.expected_decode_optimized_hlo_sha256,
        hidden_size=geometry.hidden_size,
    )
    _require_graph_authorized(
        graphs["decode"], compile_only=bool(args.compile_only)
    )
    probe_jit = jax.jit(program.probe_cache_write)
    probe_lowered = probe_jit.lower(current_state)
    started = time.perf_counter()
    probe_compiled = probe_lowered.compile()
    compile_seconds["cache_probe"] = time.perf_counter() - started
    compiled_memory["cache_probe"] = _compiled_memory(probe_compiled)
    graphs["cache_probe"], _, _ = _write_graph(
        graph="cache_probe",
        lowered=probe_lowered,
        compiled=probe_compiled,
        hlo_dir=args.hlo_dir,
        expected_stable=args.expected_cache_probe_stablehlo_sha256,
        expected_optimized=args.expected_cache_probe_optimized_hlo_sha256,
        hidden_size=geometry.hidden_size,
    )
    _require_graph_authorized(
        graphs["cache_probe"], compile_only=bool(args.compile_only)
    )

    prevalidation: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_short_decoder_prevalidation",
        "checkpoint_manifest_sha256": checkpoint.manifest["manifest_sha256"],
        "checkpoint_success_sha256": checkpoint.success["success_sha256"],
        "checkpoint_verified_device_slots": list(local_hash_slots),
        "code_hash": args.expected_code_hash,
        "compile_only": bool(args.compile_only),
        "compile_seconds": compile_seconds,
        "compiled_memory_analysis": compiled_memory,
        "context_capacity": args.context_capacity,
        "device_memory_after_compile": [
            _memory_stats(device) for device in jax.local_devices()
        ],
        "device_memory_after_load": list(loaded.device_memory_after),
        "device_memory_before_load": list(loaded.device_memory_before),
        "dsa_oracle_manifest_sha256": oracle.dsa_manifest["manifest_sha256"],
        "dsa_oracle_success_sha256": oracle.dsa_success_sha256,
        "graphs": graphs,
        "hostname": socket.gethostname(),
        "jax_process_index": int(jax.process_index()),
        "launch_process_id": args.process_id,
        "load_seconds": load_seconds,
        "local_device_slots": list(loaded.local_device_slots),
        "mesh_sha256": physical_mesh.mesh_hash,
        "prompt_length": int(oracle.prompt_token_ids.size),
        "source_inventory_sha256": inventory.inventory_sha256,
        "token_oracle_manifest_sha256": oracle.token_manifest["manifest_sha256"],
        "token_oracle_success_sha256": oracle.token_success_sha256,
        "topology_fleet_sha256": fleet_sha,
        "topology_sha256": topology.topology_hash,
        "xla_python_client_mem_fraction": _XLA_MEMORY_FRACTION,
    }
    _atomic_json(args.hlo_dir / "prevalidation.json", prevalidation)
    graph_passed = all(value["passed"] for value in graphs.values())
    if args.compile_only:
        _require_acquisition_authorized(graphs)
        result = {
            **prevalidation,
            "performance_claim": False,
            "schema_version": 1,
            "status": "HLO_ACQUIRED",
        }
        _atomic_json(args.output, result)
        print(
            f"GREENFIELD_WS32_SHORT_HLO_ACQUIRED rank={args.process_id}",
            flush=True,
        )
        return 0
    if not graph_passed:
        raise RuntimeError("WS32 complete pre-execution HLO contract failed")

    for _ in range(args.warmup):
        result = decode_compiled(current_token, current_state, weights)
        jax.block_until_ready(result)
        current_state = result.state
        current_token = result.next_token
        observed_tokens.append(_scalar_token(jax, current_token))

    samples_ms = []
    for _ in range(args.iterations):
        started_ns = time.perf_counter_ns()
        result = decode_compiled(current_token, current_state, weights)
        jax.block_until_ready(result)
        samples_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
        current_state = result.state
        current_token = result.next_token
        observed_tokens.append(_scalar_token(jax, current_token))

    args.trace_dir.mkdir(parents=True, exist_ok=False)
    with jax.profiler.trace(str(args.trace_dir), create_perfetto_link=False):
        for _ in range(args.trace_steps):
            result = decode_compiled(current_token, current_state, weights)
            jax.block_until_ready(result)
            current_state = result.state
            current_token = result.next_token
            observed_tokens.append(_scalar_token(jax, current_token))

    probe = probe_compiled(current_state)
    jax.block_until_ready(probe)
    host_probe = jax.device_get(probe)
    expected_probe_position = required_capacity - 1
    cache = validate_ws32_cache_probe(
        position=np.asarray(host_probe.position),
        kv_rows=np.asarray(host_probe.kv_rows),
        index_rows=np.asarray(host_probe.index_rows),
        contract_valid=np.asarray(host_probe.contract_valid),
        expected_position=expected_probe_position,
        num_layers=geometry.num_layers,
        full_indexer_count=len(config.full_index_slots),
        packed_cache_width=config.packed_cache_width,
        index_width=geometry.dsa_indexer_head_dim,
    )
    if observed_dsa_producers is None:
        raise RuntimeError("WS32 numerical run produced no DSA observations")
    tensor_record = _atomic_npz(
        args.tensor_output,
        cache_contract_valid=np.asarray(
            host_probe.contract_valid, dtype=np.bool_
        ),
        cache_index_bfloat16_bits=np.ascontiguousarray(
            np.asarray(host_probe.index_rows)
        ).view(np.uint16),
        cache_kv_bfloat16_bits=np.ascontiguousarray(
            np.asarray(host_probe.kv_rows)
        ).view(np.uint16),
        cache_position=np.asarray(host_probe.position, dtype=np.int32),
        dsa_producer_layer_ids=observed_dsa_producers,
        dsa_selected_positions=np.stack(observed_dsa_positions, axis=0),
        dsa_selected_scores=np.stack(observed_dsa_scores, axis=0),
        dsa_selected_valid_counts=np.stack(observed_dsa_counts, axis=0),
    )
    oracle_token_count = min(
        len(observed_tokens), int(oracle.generated_token_ids.size)
    )
    tokens = compare_ws32_raw_tokens(
        observed_tokens[:oracle_token_count], oracle
    )
    state_position = np.asarray(
        jax.device_get(current_state.position), dtype=np.int32
    ).tolist()
    state_context = np.asarray(
        jax.device_get(current_state.context_lengths), dtype=np.int32
    ).tolist()
    state_health = np.asarray(
        jax.device_get(current_state.contract_valid), dtype=np.bool_
    ).tolist()
    correctness_passed = bool(
        tokens["exact_prefix_match"]
        and all(item["passed"] for item in dsa_steps)
        and cache["passed"]
        and state_position == [required_capacity]
        and state_context == [required_capacity + 1]
        and state_health == [True]
    )
    record = {
        **prevalidation,
        "artifact_kind": "greenfield_ws32_short_decoder",
        "cache_write_probe": cache,
        "correctness_passed": correctness_passed,
        "device_memory_after_execute": [
            _memory_stats(device) for device in jax.local_devices()
        ],
        "dsa_steps": dsa_steps,
        "observed_generated_token_ids": observed_tokens,
        "numerical_tensors": tensor_record,
        "performance_claim": False,
        "profiler_free_timing": {
            "distribution": _distribution(samples_ms),
            "iterations": args.iterations,
            "profiler_active": False,
            "samples_ms": samples_ms,
            "warmup": args.warmup,
        },
        "schema_version": 1,
        "state": {
            "context_lengths": state_context,
            "contract_valid": state_health,
            "position": state_position,
        },
        "status": "SUCCESS" if correctness_passed else "ORACLE_MISMATCH",
        "token_comparison": tokens,
        "trace": {
            "files": _trace_files(args.trace_dir),
            "steps": args.trace_steps,
        },
    }
    _atomic_json(args.output, record)
    if not correctness_passed:
        raise RuntimeError("WS32 complete decoder correctness contract failed")
    print(
        "GREENFIELD_WS32_SHORT_DECODER_OK "
        f"rank={args.process_id} p50_ms="
        f"{record['profiler_free_timing']['distribution']['p50_ms']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
