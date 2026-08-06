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
    build_teacher_forced_prefill_program,
    validate_decoder_step_hlo,
    validate_teacher_forced_prefill_hlo,
)
from glm_tpu.greenfield.topology import (  # noqa: E402
    discover_physical_topology,
    validate_target_v4_64,
)
from glm_tpu.greenfield.validation import (  # noqa: E402
    inspect_short_context_oracle,
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


def _validate_completed_step_selected_states(
    active_metadata: np.ndarray,
    *,
    selected_width: int,
    count_index: int,
    next_position: int,
    next_context_length: int,
) -> dict[str, Any]:
    """Validate DSA state against the just-completed, not next, position."""

    if selected_width <= 0 or count_index < selected_width:
        raise ValueError("selected-state geometry is invalid")
    if active_metadata.ndim != 2 or active_metadata.shape[1] <= count_index:
        raise ValueError("active selected-state metadata shape is invalid")
    position_context_aligned = (
        next_position >= 0 and next_context_length == next_position + 1
    )
    expected_valid_count = min(next_position, selected_width)
    rows_valid = []
    for row in active_metadata:
        count = int(row[count_index])
        positions = row[:selected_width]
        valid = positions[:count]
        tail = positions[count:]
        rows_valid.append(
            position_context_aligned
            and count == expected_valid_count
            and bool(np.all(valid >= 0))
            # The output position is the exclusive bound for the DSA state
            # produced during this step. The incremented context length also
            # includes the slot to be consumed by the following step.
            and bool(np.all(valid < next_position))
            and len(set(valid.tolist())) == count
            and bool(np.all(tail == -1))
        )
    return {
        "expected_valid_count": expected_valid_count,
        "next_context_length": next_context_length,
        "next_position": next_position,
        "position_context_aligned": position_context_aligned,
        "rows_valid": rows_valid,
    }


def _load_short_context_oracle(
    oracle_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Load exact prompt/output IDs only after the sealed oracle passes."""

    from safetensors import safe_open

    manifest = inspect_short_context_oracle(oracle_dir)
    if manifest["manifest_sha256"] != expected_manifest_sha256:
        raise RuntimeError(
            "short-context oracle manifest differs from the protected pin"
        )
    tensor_path = oracle_dir / manifest["files"]["tokens"]["filename"]
    with safe_open(tensor_path, framework="np") as handle:
        prompt = np.asarray(
            handle.get_tensor("prompt_token_ids"), dtype=np.int32
        ).copy()
        generated = np.asarray(
            handle.get_tensor("generated_token_ids"), dtype=np.int32
        ).copy()
    return manifest, prompt, generated


def _raw_token_sequence_contract(
    observed: list[int],
    expected: np.ndarray,
) -> dict[str, Any]:
    """Compare one autoregressive prefix without decoding or normalization."""

    actual = np.asarray(observed, dtype=np.int32)
    reference = np.asarray(expected, dtype=np.int32)
    if actual.ndim != 1 or reference.ndim != 1:
        raise ValueError("raw token sequences must be one-dimensional")
    if actual.size <= 0 or actual.size > reference.size:
        raise ValueError("raw token comparison length exceeds the oracle")
    expected_prefix = reference[: actual.size]
    mismatches = np.flatnonzero(actual != expected_prefix)
    return {
        "compared_token_count": int(actual.size),
        "exact_prefix_match": bool(mismatches.size == 0),
        "expected_token_ids": expected_prefix.tolist(),
        "first_mismatch_index": (
            None if mismatches.size == 0 else int(mismatches[0])
        ),
        "observed_token_ids": actual.tolist(),
        "oracle_token_count": int(reference.size),
    }


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
        default=None,
    )
    parser.add_argument(
        "--feature-fuse-route-weighting",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--complete-token-path",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--short-context-oracle-dir", type=Path)
    parser.add_argument("--short-context-oracle-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.feature_output_tile is None:
        args.feature_output_tile = (
            128 if args.runtime_kind == "reference" else 256
        )
    args.feature_fuse_route_weighting = bool(
        args.feature_fuse_route_weighting
    )
    args.complete_token_path = bool(args.complete_token_path)
    oracle_mode = args.short_context_oracle_dir is not None
    if oracle_mode != (
        args.short_context_oracle_manifest_sha256 is not None
    ):
        raise ValueError(
            "short-context oracle directory and manifest pin are required together"
        )
    if oracle_mode and not args.complete_token_path:
        raise ValueError("short-context oracle requires the complete-token path")
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
    oracle_manifest = None
    prompt_token_ids = None
    oracle_generated_token_ids = None
    if oracle_mode:
        assert args.short_context_oracle_dir is not None
        assert args.short_context_oracle_manifest_sha256 is not None
        (
            oracle_manifest,
            prompt_token_ids,
            oracle_generated_token_ids,
        ) = _load_short_context_oracle(
            args.short_context_oracle_dir,
            expected_manifest_sha256=(
                args.short_context_oracle_manifest_sha256
            ),
        )
        compared_tokens = 1 + args.warmup + args.iterations
        if compared_tokens > oracle_generated_token_ids.size:
            raise ValueError(
                "decoder correctness window exceeds the sealed token oracle"
            )
        recurrent_steps = args.warmup + args.iterations + args.trace_steps
        if prompt_token_ids.size + recurrent_steps > args.context_capacity:
            raise ValueError(
                "prompt plus recurrent/trace steps exceeds context capacity"
            )
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
    if args.runtime_kind == "reference" and args.feature_fuse_route_weighting:
        raise ValueError(
            "feature route-weight fusion requires a feature runtime"
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
    auxiliary_values: tuple[Any, ...] = ()
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
            feature_fuse_route_weighting=(
                args.feature_fuse_route_weighting
            ),
            linear_backend=linear_backend,
            complete_token_path=args.complete_token_path,
        )
        prefill = None
        if oracle_mode:
            assert prompt_token_ids is not None
            prefill = build_teacher_forced_prefill_program(
                decoder,
                prompt_length=int(prompt_token_ids.size),
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
        token_shape = (total_devices, 1)
        initial_row = np.linspace(
            -0.5,
            0.5,
            execution_plan.geometry.hidden_size,
            dtype=np.float32,
        ).astype(ml_dtypes.bfloat16)[None, None, :]

        def residual_builder(rank: int, shape: tuple[int, ...]) -> np.ndarray:
            value = np.zeros(shape, dtype=ml_dtypes.bfloat16)
            if not oracle_mode and rank in groups[0]:
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

        def token_builder(rank: int, shape: tuple[int, ...]) -> np.ndarray:
            value = np.full(shape, -1, dtype=np.int32)
            if rank in groups[0]:
                value[...] = np.int32(
                    prompt_token_ids[0]
                    if oracle_mode and prompt_token_ids is not None
                    else 1
                )
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
        token = None
        if args.complete_token_path:
            token = _make_global_array(
                jax,
                decoder.mesh,
                decoder.input_specs[5],
                token_shape,
                token_builder,
            )
        prompt = None
        if oracle_mode:
            assert prompt_token_ids is not None
            prompt = jax.device_put(
                prompt_token_ids,
                NamedSharding(decoder.mesh, P()),
            )
        auxiliary_values = tuple(
            value for value in (token, prompt) if value is not None
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
        common_inputs = (
            loaded.weights,
            residual,
            kv,
            index,
            metadata,
        )
        if args.complete_token_path:
            assert token is not None
            inputs = (
                *common_inputs,
                token,
                position,
                block_tables,
                context_lengths,
            )
            state_values = (
                residual,
                kv,
                index,
                metadata,
                token,
                position,
                block_tables,
                context_lengths,
            )
            donate_argnums = (1, 2, 3, 4, 5)
        else:
            inputs = (
                *common_inputs,
                position,
                block_tables,
                context_lengths,
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
            donate_argnums = (1, 2, 3, 4)
        prefill_inputs = None
        if oracle_mode:
            assert prefill is not None and prompt is not None
            prefill_inputs = (
                loaded.weights,
                residual,
                kv,
                index,
                metadata,
                prompt,
                position,
                block_tables,
                context_lengths,
            )
        multihost_utils.sync_global_devices("greenfield-short-decoder-compile-start")
        compile_started = time.monotonic()
        lowered = jax.jit(
            decoder.execute,
            donate_argnums=donate_argnums,
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
            feature_fuse_route_weighting=(
                decoder.feature_fuse_route_weighting
            ),
            complete_token_path=decoder.complete_token_path,
        )
        if jax.process_index() == 0:
            hlo_dir = args.output.parent / "hlo"
            hlo_dir.mkdir(parents=True, exist_ok=True)
            hlo_stem = (
                "decoder_78layer_2k_token"
                if args.complete_token_path
                else "decoder_78layer_2k"
            )
            with gzip.open(
                hlo_dir / f"{hlo_stem}.optimized_hlo.txt.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                stream.write(optimized_hlo)
            _atomic_json(
                hlo_dir / f"{hlo_stem}.hlo_contract.json",
                hlo_contract,
            )
        del optimized_hlo
        if not hlo_contract["passed"]:
            raise RuntimeError(
                "decoder HLO contract failed before execution: "
                f"{hlo_contract['violations']}"
            )
        compiled_prefill = None
        prefill_compile_seconds = None
        prefill_hlo_sha256 = None
        fleet_prefill_hlo_hashes = None
        prefill_hlo_contract = None
        if oracle_mode:
            assert prefill is not None and prefill_inputs is not None
            multihost_utils.sync_global_devices(
                "greenfield-short-prefill-compile-start"
            )
            prefill_compile_started = time.monotonic()
            lowered_prefill = jax.jit(
                prefill.execute,
                donate_argnums=(1, 2, 3, 4),
            ).lower(*prefill_inputs)
            compiled_prefill = lowered_prefill.compile()
            prefill_compile_seconds = (
                time.monotonic() - prefill_compile_started
            )
            multihost_utils.sync_global_devices(
                "greenfield-short-prefill-compile-end"
            )
            optimized_prefill_hlo = compiled_prefill.as_text()
            prefill_hlo_sha256 = sha256(
                optimized_prefill_hlo.encode("utf-8")
            ).hexdigest()
            fleet_prefill_hlo_hashes = _fleet_digest(
                multihost_utils,
                prefill_hlo_sha256,
                num_processes=args.num_processes,
            )
            prefill_hlo_contract = validate_teacher_forced_prefill_hlo(
                optimized_prefill_hlo,
                program=prefill,
                schedule=schedule,
                backend_contract=hlo_backend_contract,
            )
            if jax.process_index() == 0:
                hlo_dir = args.output.parent / "hlo"
                with gzip.open(
                    hlo_dir / "prefill_78layer_2k.optimized_hlo.txt.gz",
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(optimized_prefill_hlo)
                _atomic_json(
                    hlo_dir / "prefill_78layer_2k.hlo_contract.json",
                    prefill_hlo_contract,
                )
            del optimized_prefill_hlo
            if not prefill_hlo_contract["passed"]:
                raise RuntimeError(
                    "prefill HLO contract failed before execution: "
                    f"{prefill_hlo_contract['violations']}"
                )

        generated_tokens: list[int] = []
        prefill_wall_ms = None
        if oracle_mode:
            assert compiled_prefill is not None and prefill_inputs is not None
            prefill_started = time.perf_counter_ns()
            output = compiled_prefill(*prefill_inputs)
            output[3].block_until_ready()
            prefill_wall_ms = (
                time.perf_counter_ns() - prefill_started
            ) / 1_000_000
            first_token = _materialize_global_array(
                jax,
                multihost_utils,
                output[4],
            )[list(groups[0]), 0]
            if not np.all(first_token == first_token[0]):
                raise RuntimeError("prefill token lanes disagree")
            generated_tokens.append(int(first_token[0]))
        else:
            output = compiled(*inputs)
            output[3].block_until_ready()
        state_values = (
            tuple(output)
            if args.complete_token_path
            else (*output, position, block_tables, context_lengths)
        )
        current = output

        def run_step(values: tuple[Any, ...]) -> tuple[Any, ...]:
            if args.complete_token_path:
                return compiled(loaded.weights, *values)
            return compiled(
                loaded.weights,
                *values,
                position,
                block_tables,
                context_lengths,
            )

        def materialize_active_token(
            values: tuple[Any, ...], *, phase: str
        ) -> int:
            token_values = _materialize_global_array(
                jax,
                multihost_utils,
                values[4],
            )[list(groups[0]), 0]
            if not np.all(token_values == token_values[0]):
                raise RuntimeError(
                    f"complete token lanes disagree after {phase} step"
                )
            return int(token_values[0])

        for _ in range(args.warmup):
            current = run_step(current)
            current[3].block_until_ready()
            if oracle_mode:
                generated_tokens.append(
                    materialize_active_token(current, phase="warmup")
                )
        samples = []
        if not oracle_mode:
            generated_tokens = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            current = run_step(current)
            current[3].block_until_ready()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            if args.complete_token_path:
                generated_tokens.append(
                    materialize_active_token(current, phase="timed")
                )
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
                        current = run_step(current)
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
        state_values = (
            tuple(current)
            if args.complete_token_path
            else (*current, position, block_tables, context_lengths)
        )
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
        base_metadata_passed = (
            metadata_contract["active_ranks"] == list(groups[0])
            and metadata_contract["health"] == [1]
            and metadata_contract["producer"] == [expected_producer]
            and metadata_contract["visited"] == [255]
        )
        if args.complete_token_path:
            next_position = np.asarray(jax.device_get(current[5]))
            next_context_lengths = np.asarray(
                jax.device_get(current[7])
            )
            selected_state_contract = (
                _validate_completed_step_selected_states(
                    active_metadata,
                    selected_width=decoder.config.selected_width,
                    count_index=decoder.config.count_index,
                    next_position=int(next_position[0]),
                    next_context_length=int(next_context_lengths[0]),
                )
            )
            metadata_contract["selected_state_contract"] = (
                selected_state_contract
            )
            metadata_passed = bool(
                base_metadata_passed
                and all(selected_state_contract["rows_valid"])
            )
        else:
            metadata_passed = bool(
                base_metadata_passed
                and metadata_contract["valid_counts"] == [1]
                and all(
                    row[0] == 0
                    for row in metadata_contract["selected_prefix"]
                )
            )
        token_contract = None
        token_passed = True
        if args.complete_token_path:
            token_host = _materialize_global_array(
                jax,
                multihost_utils,
                current[4],
            )
            active_tokens = token_host[active, 0]
            raw_sequence = None
            if oracle_mode:
                assert oracle_manifest is not None
                assert oracle_generated_token_ids is not None
                assert prompt_token_ids is not None
                raw_sequence = _raw_token_sequence_contract(
                    generated_tokens,
                    oracle_generated_token_ids,
                )
            expected_next_position = (
                int(prompt_token_ids.size)
                + args.warmup
                + args.iterations
                + args.trace_steps
                if oracle_mode and prompt_token_ids is not None
                else None
            )
            token_contract = {
                "active_tokens": sorted(set(active_tokens.tolist())),
                "correctness_window_tokens": (
                    generated_tokens if oracle_mode else None
                ),
                "profiler_free_window_tokens": generated_tokens[
                    -args.iterations:
                ],
                "all_active_lanes_equal": bool(
                    active_tokens.size == len(groups[0])
                    and np.all(active_tokens == active_tokens[0])
                ),
                "all_in_vocabulary": bool(
                    np.all(active_tokens >= 0)
                    and np.all(
                        active_tokens < execution_plan.geometry.vocab_size
                    )
                ),
                "next_position": next_position.tolist(),
                "next_context_lengths": next_context_lengths.tolist(),
                "expected_next_position": expected_next_position,
                "position_contract_passed": bool(
                    expected_next_position is None
                    or (
                        next_position.tolist() == [expected_next_position]
                        and next_context_lengths.tolist()
                        == [expected_next_position + 1]
                    )
                ),
                "prefill_used": oracle_mode,
                "raw_token_sequence": raw_sequence,
                "short_context_oracle": (
                    {
                        "generated_token_ids_sha256": oracle_manifest[
                            "generated_token_ids_sha256"
                        ],
                        "manifest_sha256": oracle_manifest[
                            "manifest_sha256"
                        ],
                        "prompt_token_count": int(prompt_token_ids.size),
                        "prompt_token_ids_sha256": oracle_manifest[
                            "prompt_token_ids_sha256"
                        ],
                        "source": oracle_manifest["source"],
                    }
                    if oracle_mode
                    and oracle_manifest is not None
                    and prompt_token_ids is not None
                    else None
                ),
                "synthetic_initial_state": not oracle_mode,
            }
            token_passed = bool(
                token_contract["all_active_lanes_equal"]
                and token_contract["all_in_vocabulary"]
                and token_contract["position_contract_passed"]
                and (
                    raw_sequence is None
                    or raw_sequence["exact_prefix_match"]
                )
            )
            metadata_passed = bool(metadata_passed and token_passed)
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
                "greenfield_real_78layer_2k_decoder_"
                + (
                    "token_oracle_"
                    if oracle_mode
                    else ("token_" if args.complete_token_path else "body_")
                )
                + f"{args.runtime_kind}"
            ),
            "body_only": not args.complete_token_path,
            "code_hash": code_hash,
            "compile_seconds": compile_seconds,
            "decoder_compile_seconds": compile_seconds,
            "context_capacity": args.context_capacity,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device_memory_after_execute": [
                _memory_stats(device) for device in jax.local_devices()
            ],
            "fleet_hlo_hashes": fleet_hlo_hashes,
            "fleet_prefill_hlo_hashes": fleet_prefill_hlo_hashes,
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
            "profiler_free_complete_step_wall": (
                _percentiles(samples) if args.complete_token_path else None
            ),
            "prefill_compile_seconds": prefill_compile_seconds,
            "prefill_hlo_contract": prefill_hlo_contract,
            "prefill_hlo_sha256": prefill_hlo_sha256,
            "prefill_wall_ms": prefill_wall_ms,
            "prefill_used": oracle_mode,
            "raw_token_claim": bool(
                oracle_mode
                and token_contract is not None
                and token_contract["raw_token_sequence"] is not None
                and token_contract["raw_token_sequence"]["exact_prefix_match"]
            ),
            "runtime_layout_hash": pack_context.layout.layout_hash,
            "runtime_manifest_sha256": expectation.runtime_manifest_sha256,
            "runtime_kind": args.runtime_kind,
            "schedule_hash": schedule.schedule_hash,
            "schema_version": 2,
            "state_layout": state_layout.to_dict(),
            "state_layout_hash": state_layout.state_layout_hash,
            "sparse_moe_backend": decoder.sparse_moe_backend,
            "feature_output_tile": decoder.feature_output_tile,
            "feature_fuse_route_weighting": (
                decoder.feature_fuse_route_weighting
            ),
            "linear_backend": decoder.linear_backend,
            "complete_token_path": decoder.complete_token_path,
            "token_contract": token_contract,
            "token_passed": token_passed,
            "topology_hash": live_topology.topology_hash,
            "trace": trace_record,
            "transformer_body_timing_only": not args.complete_token_path,
            "warmup": args.warmup,
        }
        _atomic_json(args.output, record)
        if not token_passed:
            raise RuntimeError(
                f"decoder raw-token contract failed: {token_contract}"
            )
        if not metadata_passed:
            raise RuntimeError(f"decoder metadata contract failed: {metadata_contract}")
        print(
            "GREENFIELD_SHORT_DECODER_HOST_OK "
            f"launch_process={args.process_id} jax_process={jax.process_index()} "
            f"compile_s={compile_seconds:.3f} "
            f"p50_step_ms={record['profiler_free_body_wall']['p50_ms']:.6f} "
            f"hlo={hlo_sha256}",
            flush=True,
        )
        return 0
    finally:
        _delete_arrays(state_values)
        _delete_arrays(auxiliary_values)
        if loaded is not None:
            loaded.close()
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
