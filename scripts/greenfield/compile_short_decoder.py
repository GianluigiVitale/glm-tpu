#!/usr/bin/env python3
"""Bind real runtime weights and compile/run the 78-layer PP8 decoder body."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    FeatureRuntimeCheckpointLoadExpectation,
    RuntimeCheckpointLoadExpectation,
    load_runtime_checkpoint,
    verify_feature_runtime_packed_checkpoint,
    verify_runtime_packed_checkpoint,
)
from glm_tpu.greenfield.model import (  # noqa: E402
    build_decoder_state_layout,
    build_pipeline_schedule,
)
from glm_tpu.greenfield.runtime import (  # noqa: E402
    build_decoder_step_program,
    validate_decoder_step_hlo,
)
from glm_tpu.greenfield.topology import (  # noqa: E402
    discover_physical_topology,
    validate_target_v4_64,
)
from scripts.greenfield.pack_feature_runtime_checkpoint import (  # noqa: E402
    _build_context as _build_feature_context,
)
from scripts.greenfield.pack_runtime_checkpoint import (  # noqa: E402
    _build_context as _build_reference_context,
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _fleet_digest(
    multihost_utils: Any,
    digest_hex: str,
    *,
    num_processes: int,
) -> list[str]:
    digest = np.frombuffer(bytes.fromhex(digest_hex), dtype=np.uint8)
    fleet = np.asarray(multihost_utils.process_allgather(digest)).reshape(
        num_processes,
        len(digest),
    )
    values = [row.tobytes().hex() for row in fleet]
    if len(set(values)) != 1:
        raise RuntimeError(f"hosts disagree on optimized HLO: {values}")
    return values


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _materialize_global_array(
    jax: Any,
    multihost_utils: Any,
    value: Any,
) -> np.ndarray:
    """Materialize a global array without fetching non-addressable shards."""
    if value.is_fully_addressable:
        materialized = jax.device_get(value)
    else:
        materialized = multihost_utils.process_allgather(value, tiled=True)
    host = np.asarray(materialized)
    if host.shape != tuple(value.shape):
        raise RuntimeError(
            "global array gather changed shape: "
            f"expected={tuple(value.shape)} actual={host.shape}"
        )
    return host


def _make_global_array(
    jax: Any,
    mesh: Any,
    partition_spec: Any,
    global_shape: tuple[int, ...],
    builder: Callable[[int, tuple[int, ...]], np.ndarray],
) -> Any:
    from jax.sharding import NamedSharding

    sharding = NamedSharding(mesh, partition_spec)
    indices = sharding.addressable_devices_indices_map(global_shape)
    arrays = []
    for device, index in indices.items():
        shard_shape = tuple(
            (
                (dimension if item.stop is None else item.stop)
                - (0 if item.start is None else item.start)
            )
            if isinstance(item, slice)
            else 1
            for dimension, item in zip(global_shape, index, strict=True)
        )
        rank = (
            0 if index[0].start is None else index[0].start
        ) if isinstance(index[0], slice) else int(index[0])
        host = builder(rank, shard_shape)
        if tuple(host.shape) != shard_shape:
            raise RuntimeError(
                f"state builder shape drift: expected={shard_shape} "
                f"observed={host.shape}"
            )
        array = jax.device_put(host, device)
        array.block_until_ready()
        arrays.append(array)
    return jax.make_array_from_single_device_arrays(
        global_shape,
        sharding,
        arrays,
    )


def _delete_arrays(values: tuple[Any, ...]) -> None:
    for value in values:
        try:
            value.delete()
        except (AttributeError, RuntimeError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument(
        "--runtime-kind",
        choices=("reference", "pallas_feature", "pallas_feature_linear"),
        default="reference",
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--source-runtime-root", type=Path)
    parser.add_argument("--source-runtime-manifest-sha256")
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-packed-manifest-sha256", required=True)
    parser.add_argument("--context-capacity", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--trace-steps", type=int, default=0)
    parser.add_argument(
        "--feature-output-tile",
        type=int,
        choices=(128, 256),
        default=128,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise ValueError("protected decoder compile requires process ids 0..7")
    if args.context_capacity != 2048:
        raise ValueError("first production compile is fixed to 2K")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("decoder warmup/iterations must be positive")
    if args.trace_steps not in (0, 2):
        raise ValueError("decoder diagnostic trace requires zero or two steps")
    if (args.trace_root is None) != (args.trace_steps == 0):
        raise ValueError("decoder trace root and trace steps must be enabled together")
    if args.runtime_kind in ("pallas_feature", "pallas_feature_linear") and (
        args.source_runtime_root is None
        or args.source_runtime_manifest_sha256 is None
    ):
        raise ValueError(
            "feature runtime requires its source runtime root and manifest"
        )
    if args.runtime_kind == "reference" and args.feature_output_tile != 128:
        raise ValueError(
            "a non-default feature output tile requires a feature runtime"
        )
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    runtime_manifest = json.loads(
        (args.runtime_root / "runtime_manifest.json").read_text()
    )
    if not isinstance(runtime_manifest, dict):
        raise RuntimeError("runtime manifest is not an object")
    if args.runtime_kind == "reference":
        context_args = SimpleNamespace(
            source_checkpoint_root=args.source_checkpoint_root,
            source_packed_manifest_sha256=args.source_packed_manifest_sha256,
            destination=runtime_manifest["destination"],
        )
        pack_context = _build_reference_context(
            context_args,
            runtime_manifest["pack_code_hash"],
        )
        from glm_tpu.greenfield.types import ExecutionPlan

        execution_plan = ExecutionPlan.from_dict(
            pack_context.source_checkpoint.layout["plan_manifest"][
                "execution_plan"
            ]
        )
        expectation = RuntimeCheckpointLoadExpectation(
            runtime_manifest_sha256=args.runtime_manifest_sha256,
            runtime_layout_manifest_sha256=runtime_manifest[
                "runtime_layout_manifest_sha256"
            ],
            runtime_layout_hash=runtime_manifest["runtime_layout_hash"],
            source_packed_manifest_sha256=runtime_manifest[
                "source_packed_manifest_sha256"
            ],
            source_layout_manifest_sha256=runtime_manifest[
                "source_layout_manifest_sha256"
            ],
            plan_hash=runtime_manifest["plan_hash"],
            schedule_hash=runtime_manifest["schedule_hash"],
            pack_code_hash=runtime_manifest["pack_code_hash"],
            destination=runtime_manifest["destination"],
            source_destination=runtime_manifest[
                "source_checkpoint_destination"
            ],
            plan_id=runtime_manifest["plan_id"],
            model_id=runtime_manifest["model_id"],
        )
        verified_runtime = verify_runtime_packed_checkpoint(
            args.runtime_root,
            expectation,
            pack_context.layout,
            pack_context.source_checkpoint,
        )
        sparse_moe_backend = "reference"
        linear_backend = "reference"
        hlo_backend_contract = "tpu_v4_pp8_reference"
    else:
        context_args = SimpleNamespace(
            source_checkpoint_root=args.source_checkpoint_root,
            source_packed_manifest_sha256=args.source_packed_manifest_sha256,
            source_runtime_root=args.source_runtime_root,
            source_runtime_manifest_sha256=(
                args.source_runtime_manifest_sha256
            ),
            destination=runtime_manifest["destination"],
        )
        pack_context = _build_feature_context(
            context_args,
            runtime_manifest["pack_code_hash"],
        )
        execution_plan = pack_context.target_plan
        expectation = FeatureRuntimeCheckpointLoadExpectation(
            runtime_manifest_sha256=args.runtime_manifest_sha256,
            runtime_layout_manifest_sha256=runtime_manifest[
                "runtime_layout_manifest_sha256"
            ],
            runtime_layout_hash=runtime_manifest["runtime_layout_hash"],
            source_runtime_manifest_sha256=runtime_manifest[
                "source_runtime_manifest_sha256"
            ],
            source_runtime_layout_manifest_sha256=runtime_manifest[
                "source_runtime_layout_manifest_sha256"
            ],
            source_runtime_layout_hash=runtime_manifest[
                "source_runtime_layout_hash"
            ],
            plan_hash=runtime_manifest["plan_hash"],
            schedule_hash=runtime_manifest["schedule_hash"],
            pack_code_hash=runtime_manifest["pack_code_hash"],
            destination=runtime_manifest["destination"],
            source_destination=runtime_manifest[
                "source_checkpoint_destination"
            ],
            plan_id=runtime_manifest["plan_id"],
            model_id=runtime_manifest["model_id"],
        )
        verified_runtime = verify_feature_runtime_packed_checkpoint(
            args.runtime_root,
            expectation,
            pack_context.layout,
            pack_context.source_runtime_checkpoint,
        )
        sparse_moe_backend = "pallas_feature"
        linear_backend = (
            "pallas" if args.runtime_kind == "pallas_feature_linear" else "reference"
        )
        hlo_backend_contract = (
            "tpu_v4_pp8_pallas_feature_linear"
            if linear_backend == "pallas"
            else "tpu_v4_pp8_pallas_feature"
        )
    schedule = build_pipeline_schedule(execution_plan)
    state_layout = build_decoder_state_layout(
        execution_plan,
        schedule,
        context_capacity=args.context_capacity,
    )
    import jax
    import ml_dtypes
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    loaded = None
    state_values: tuple[Any, ...] = ()
    try:
        if (
            jax.process_count() != 8
            or jax.local_device_count() != 4
            or jax.device_count() != 32
        ):
            raise RuntimeError(
                "protected decoder requires 8 processes, 4 local chips, and 32 chips"
            )
        local_ids = np.asarray(
            [device.id for device in jax.local_devices()],
            dtype=np.int32,
        )
        fleet_local_ids = np.asarray(
            multihost_utils.process_allgather(local_ids)
        ).reshape(args.num_processes, jax.local_device_count())
        observed_local_order = {
            int(device_id): local_index
            for process_row in fleet_local_ids
            for local_index, device_id in enumerate(process_row.tolist())
        }
        live_topology = discover_physical_topology(
            jax.devices(),
            slice_name="db-v4-64-od",
            observed_local_order=observed_local_order,
        )
        validate_target_v4_64(live_topology)
        if live_topology.topology_hash != execution_plan.topology.topology_hash:
            raise RuntimeError("live topology differs from the runtime checkpoint plan")
        by_id = {int(device.id): device for device in jax.devices()}
        runtime_devices = tuple(
            by_id[device.device_id] for device in pack_context.layout.devices
        )
        groups = tuple(
            tuple(range(stage * 4, stage * 4 + 4)) for stage in range(8)
        )
        pairs = tuple(
            (groups[stage][slot], groups[(stage + 1) % 8][slot])
            for stage in range(8)
            for slot in range(4)
        )
        decoder = build_decoder_step_program(
            execution_plan,
            schedule,
            state_layout,
            pack_context.layout,
            groups,
            pairs,
            devices=runtime_devices,
            sparse_moe_backend=sparse_moe_backend,
            feature_output_tile=args.feature_output_tile,
            linear_backend=linear_backend,
        )
        multihost_utils.sync_global_devices("greenfield-short-decoder-load-start")
        load_started = time.monotonic()
        loaded = load_runtime_checkpoint(
            verified_runtime,
            expectation,
            pack_context.layout,
            decoder.mesh,
            verify_device_roundtrip=False,
        )
        load_seconds = time.monotonic() - load_started
        multihost_utils.sync_global_devices("greenfield-short-decoder-load-end")

        total_devices = decoder.config.total_devices
        kv_shape = state_layout.stages[0].padded_kv_cache_shape
        index_shape = state_layout.stages[0].padded_indexer_cache_shape
        residual_shape = (total_devices, 1, execution_plan.geometry.hidden_size)
        kv_global_shape = (total_devices, *kv_shape)
        index_global_shape = (total_devices, *index_shape)
        metadata_shape = (total_devices, 1, decoder.config.metadata_width)
        initial_row = np.linspace(
            -0.5,
            0.5,
            execution_plan.geometry.hidden_size,
            dtype=np.float32,
        ).astype(ml_dtypes.bfloat16)[None, None, :]

        def residual_builder(rank: int, shape: tuple[int, ...]) -> np.ndarray:
            value = np.zeros(shape, dtype=ml_dtypes.bfloat16)
            if rank in groups[0]:
                value[...] = initial_row
            return value

        def zero_bf16(_: int, shape: tuple[int, ...]) -> np.ndarray:
            return np.zeros(shape, dtype=ml_dtypes.bfloat16)

        def metadata_builder(rank: int, shape: tuple[int, ...]) -> np.ndarray:
            value = np.full(shape, -1, dtype=np.int32)
            value[..., decoder.config.count_index] = 0
            value[..., decoder.config.producer_index] = -1
            value[..., decoder.config.visited_index] = 0
            value[..., decoder.config.health_index] = 1
            value[..., decoder.config.active_index] = int(rank in groups[0])
            return value

        residual = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[1],
            residual_shape,
            residual_builder,
        )
        kv = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[2],
            kv_global_shape,
            zero_bf16,
        )
        index = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[3],
            index_global_shape,
            zero_bf16,
        )
        metadata = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[4],
            metadata_shape,
            metadata_builder,
        )
        position = jax.device_put(
            np.asarray([0], dtype=np.int32),
            NamedSharding(decoder.mesh, P()),
        )
        pages = state_layout.stages[0].physical_page_count
        block_tables = jax.device_put(
            np.arange(pages, dtype=np.int32)[None, :],
            NamedSharding(decoder.mesh, P()),
        )
        context_lengths = jax.device_put(
            np.asarray([1], dtype=np.int32),
            NamedSharding(decoder.mesh, P()),
        )
        state_values = (
            residual,
            kv,
            index,
            metadata,
            position,
            block_tables,
            context_lengths,
        )
        inputs = (
            loaded.weights,
            residual,
            kv,
            index,
            metadata,
            position,
            block_tables,
            context_lengths,
        )
        multihost_utils.sync_global_devices("greenfield-short-decoder-compile-start")
        compile_started = time.monotonic()
        lowered = jax.jit(
            decoder.execute,
            donate_argnums=(1, 2, 3, 4),
        ).lower(*inputs)
        compiled = lowered.compile()
        compile_seconds = time.monotonic() - compile_started
        multihost_utils.sync_global_devices("greenfield-short-decoder-compile-end")
        optimized_hlo = compiled.as_text()
        hlo_sha256 = sha256(optimized_hlo.encode("utf-8")).hexdigest()
        fleet_hlo_hashes = _fleet_digest(
            multihost_utils,
            hlo_sha256,
            num_processes=args.num_processes,
        )
        hlo_contract = validate_decoder_step_hlo(
            optimized_hlo,
            config=decoder.config,
            schedule=schedule,
            groups=groups,
            pairs=pairs,
            backend_contract=hlo_backend_contract,
            feature_output_tile=decoder.feature_output_tile,
        )
        if jax.process_index() == 0:
            hlo_dir = args.output.parent / "hlo"
            hlo_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(
                hlo_dir / "decoder_78layer_2k.optimized_hlo.txt.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                stream.write(optimized_hlo)
            _atomic_json(
                hlo_dir / "decoder_78layer_2k.hlo_contract.json",
                hlo_contract,
            )
        del optimized_hlo
        if not hlo_contract["passed"]:
            raise RuntimeError(
                "decoder HLO contract failed before execution: "
                f"{hlo_contract['violations']}"
            )

        output = compiled(*inputs)
        output[3].block_until_ready()
        state_values = (*output, position, block_tables, context_lengths)
        current = output
        for _ in range(args.warmup):
            current = compiled(
                loaded.weights,
                *current,
                position,
                block_tables,
                context_lengths,
            )
            current[3].block_until_ready()
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            current = compiled(
                loaded.weights,
                *current,
                position,
                block_tables,
                context_lengths,
            )
            current[3].block_until_ready()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
        trace_record = None
        if args.trace_root is not None:
            args.trace_root.mkdir(parents=True, exist_ok=False)
            options = jax.profiler.ProfileOptions()
            options.python_tracer_level = 0
            tracing = False
            multihost_utils.sync_global_devices(
                "greenfield-short-decoder-trace-ready"
            )
            try:
                jax.profiler.start_trace(
                    str(args.trace_root),
                    profiler_options=options,
                )
                tracing = True
                multihost_utils.sync_global_devices(
                    "greenfield-short-decoder-trace-started"
                )
                for step in range(args.trace_steps):
                    with jax.profiler.TraceAnnotation(
                        "greenfield_short_decoder_body_step",
                        step_num=step,
                    ):
                        current = compiled(
                            loaded.weights,
                            *current,
                            position,
                            block_tables,
                            context_lengths,
                        )
                        current[3].block_until_ready()
                jax.profiler.stop_trace()
                tracing = False
            finally:
                if tracing:
                    jax.profiler.stop_trace()
            multihost_utils.sync_global_devices(
                "greenfield-short-decoder-trace-stopped"
            )
            xplanes = tuple(sorted(args.trace_root.rglob("*.xplane.pb")))
            if len(xplanes) != 1:
                raise RuntimeError(
                    "expected one local decoder XPlane, "
                    f"found {len(xplanes)}"
                )
            trace_record = {
                "files": [
                    {
                        "path": str(xplanes[0]),
                        "sha256": _sha256_file(xplanes[0]),
                        "size_bytes": xplanes[0].stat().st_size,
                    }
                ],
                "profiler_started_after_profiler_free_timing": True,
                "steps": args.trace_steps,
            }
        state_values = (*current, position, block_tables, context_lengths)
        metadata_host = _materialize_global_array(
            jax,
            multihost_utils,
            current[3],
        )
        active = np.flatnonzero(
            metadata_host[:, 0, decoder.config.active_index] == 1
        )
        expected_producer = schedule.stages[-1].layers[-1].index_state_producer_layer
        active_metadata = metadata_host[active, 0]
        metadata_contract = {
            "active_ranks": active.tolist(),
            "health": sorted(
                set(active_metadata[:, decoder.config.health_index].tolist())
            ),
            "producer": sorted(
                set(active_metadata[:, decoder.config.producer_index].tolist())
            ),
            "selected_prefix": sorted(
                set(
                    tuple(row)
                    for row in active_metadata[:, :8].tolist()
                )
            ),
            "valid_counts": sorted(
                set(active_metadata[:, decoder.config.count_index].tolist())
            ),
            "visited": sorted(
                set(active_metadata[:, decoder.config.visited_index].tolist())
            ),
        }
        metadata_passed = (
            metadata_contract["active_ranks"] == list(groups[0])
            and metadata_contract["health"] == [1]
            and metadata_contract["producer"] == [expected_producer]
            and metadata_contract["valid_counts"] == [1]
            and metadata_contract["visited"] == [255]
            and all(row[0] == 0 for row in metadata_contract["selected_prefix"])
        )
        local_kv_nonzero = []
        local_index_nonzero = []
        for shard in current[1].addressable_shards:
            host = np.asarray(jax.device_get(shard.data))
            local_kv_nonzero.append(int(np.count_nonzero(host[:, 0, 0, 0])))
        for shard in current[2].addressable_shards:
            host = np.asarray(jax.device_get(shard.data))
            local_index_nonzero.append(int(np.count_nonzero(host[:, 0, 0, 0])))
        record = {
            "artifact_kind": (
                "greenfield_real_78layer_2k_decoder_body_"
                f"{args.runtime_kind}"
            ),
            "body_only": True,
            "code_hash": code_hash,
            "compile_seconds": compile_seconds,
            "context_capacity": args.context_capacity,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device_memory_after_execute": [
                _memory_stats(device) for device in jax.local_devices()
            ],
            "fleet_hlo_hashes": fleet_hlo_hashes,
            "fleet_local_device_ids_in_runtime_order": fleet_local_ids.tolist(),
            "hlo_contract": hlo_contract,
            "hostname": socket.gethostname(),
            "iterations": args.iterations,
            "jax_process_index": jax.process_index(),
            "launch_process_id": args.process_id,
            "load_record": loaded.load_record,
            "load_seconds": load_seconds,
            "local_index_nonzero_counts": local_index_nonzero,
            "local_kv_nonzero_counts": local_kv_nonzero,
            "metadata_contract": metadata_contract,
            "metadata_passed": metadata_passed,
            "optimized_hlo_sha256": hlo_sha256,
            "plan_hash": execution_plan.plan_hash,
            "profiler_free_body_wall": _percentiles(samples),
            "raw_token_claim": False,
            "runtime_layout_hash": pack_context.layout.layout_hash,
            "runtime_manifest_sha256": expectation.runtime_manifest_sha256,
            "runtime_kind": args.runtime_kind,
            "schedule_hash": schedule.schedule_hash,
            "schema_version": 1,
            "state_layout": state_layout.to_dict(),
            "state_layout_hash": state_layout.state_layout_hash,
            "sparse_moe_backend": decoder.sparse_moe_backend,
            "feature_output_tile": decoder.feature_output_tile,
            "linear_backend": decoder.linear_backend,
            "topology_hash": live_topology.topology_hash,
            "trace": trace_record,
            "transformer_body_timing_only": True,
            "warmup": args.warmup,
        }
        _atomic_json(args.output, record)
        if not metadata_passed:
            raise RuntimeError(f"decoder metadata contract failed: {metadata_contract}")
        print(
            "GREENFIELD_SHORT_DECODER_HOST_OK "
            f"launch_process={args.process_id} jax_process={jax.process_index()} "
            f"compile_s={compile_seconds:.3f} "
            f"p50_body_ms={record['profiler_free_body_wall']['p50_ms']:.6f} "
            f"hlo={hlo_sha256}",
            flush=True,
        )
        return 0
    finally:
        _delete_arrays(state_values)
        if loaded is not None:
            loaded.close()
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
