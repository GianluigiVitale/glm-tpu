"""Asynchronous stage-to-stage Pallas transport for the greenfield decoder.

The readable path keeps the accepted pair of ``lax.ppermute`` operations.
The default-off Pallas path starts independent remote DMAs for the one live
BF16 residual row and compact int32 IndexShare metadata, then waits for both.
Every device participates and has exactly one source and destination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@dataclass(frozen=True, slots=True)
class StageRemoteCopyConfig:
    """Static production payload and distributed-kernel contract."""

    total_devices: int = 32
    hidden_width: int = 6144
    metadata_width: int = 2052
    collective_id: int = 8

    def __post_init__(self) -> None:
        for name in (
            "total_devices",
            "hidden_width",
            "metadata_width",
            "collective_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.total_devices <= 1:
            raise ValueError("remote copy requires at least two devices")
        if self.hidden_width <= 0 or self.metadata_width <= 0:
            raise ValueError("remote-copy payload widths must be positive")
        if self.collective_id < 0:
            raise ValueError("collective_id must be nonnegative")


def stage_value_remote_copy_kernel_name(value: Any) -> str:
    """Return the stable custom-call name for a single transported payload."""

    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("remote-copy value must be a nonempty rank-two array")
    dtype_names = {
        jnp.dtype(jnp.bfloat16): "bf16",
        jnp.dtype(jnp.float32): "f32",
        jnp.dtype(jnp.int32): "s32",
    }
    dtype = dtype_names.get(jnp.dtype(value.dtype))
    if dtype is None:
        raise ValueError("remote-copy value must be BF16, FP32, or int32")
    return f"greenfield_stage_remote_copy_{dtype}_{value.shape[0]}x{value.shape[1]}"


def stage_value_remote_copy_pallas(
    value: Any,
    destination_rank: Any,
    *,
    collective_id: int = 8,
    interpret: bool = False,
) -> Any:
    """Asynchronously DMA one rank-two payload to its destination device."""

    kernel_name = stage_value_remote_copy_kernel_name(value)
    if destination_rank.shape != () or not jnp.issubdtype(
        destination_rank.dtype, jnp.integer
    ):
        raise ValueError("destination_rank must be an integer scalar")
    if not isinstance(collective_id, int) or isinstance(collective_id, bool) or (
        collective_id < 0
    ):
        raise ValueError("collective_id must be a nonnegative integer")
    destination = jnp.reshape(destination_rank.astype(jnp.int32), (1,))

    def kernel(
        destination_ref: Any,
        source_ref: Any,
        output_ref: Any,
        send_semaphore: Any,
        receive_semaphore: Any,
    ) -> None:
        remote_dma = pltpu.make_async_remote_copy(
            source_ref,
            output_ref,
            send_semaphore,
            receive_semaphore,
            destination_ref[0],
        )
        remote_dma.start()
        remote_dma.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(value.shape, value.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),),
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            dimension_semantics=(),
        ),
        interpret=interpret,
        name=kernel_name,
        cost_estimate=pl.CostEstimate(
            flops=0,
            bytes_accessed=value.size * value.dtype.itemsize,
            transcendentals=0,
        ),
    )(destination, value)


def _validate_payloads(
    residual: Any,
    metadata: Any,
    destination_rank: Any,
    config: StageRemoteCopyConfig,
) -> None:
    if residual.shape != (1, config.hidden_width):
        raise ValueError("stage residual must be one BF16 live row")
    if residual.dtype != jnp.bfloat16:
        raise ValueError("stage residual must remain BF16")
    if metadata.shape != (1, config.metadata_width):
        raise ValueError("stage metadata has an invalid compact shape")
    if metadata.dtype != jnp.int32:
        raise ValueError("stage metadata must remain int32")
    if destination_rank.shape != () or not jnp.issubdtype(
        destination_rank.dtype, jnp.integer
    ):
        raise ValueError("destination_rank must be an integer scalar")


def _canonical_targets(
    pairs: Sequence[Sequence[int]], total_devices: int
) -> tuple[int, ...]:
    canonical = tuple(
        (int(pair[0]), int(pair[1])) for pair in pairs if len(pair) == 2
    )
    if len(canonical) != len(pairs) or len(canonical) != total_devices:
        raise ValueError("remote copy requires one source-target pair per device")
    sources = tuple(source for source, _ in canonical)
    targets = tuple(target for _, target in canonical)
    expected = tuple(range(total_devices))
    if tuple(sorted(sources)) != expected or tuple(sorted(targets)) != expected:
        raise ValueError("remote-copy pairs must permute every device exactly once")
    if any(source == target for source, target in canonical):
        raise ValueError("remote-copy pairs cannot contain self edges")
    by_source = dict(canonical)
    return tuple(by_source[source] for source in expected)


def stage_remote_copy_pallas(
    residual: Any,
    metadata: Any,
    destination_rank: Any,
    *,
    config: StageRemoteCopyConfig = StageRemoteCopyConfig(),
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Send both live payloads with two overlapping TPU remote DMAs."""

    _validate_payloads(residual, metadata, destination_rank, config)
    destination = jnp.reshape(destination_rank.astype(jnp.int32), (1,))

    def kernel(
        destination_ref: Any,
        residual_ref: Any,
        metadata_ref: Any,
        residual_out_ref: Any,
        metadata_out_ref: Any,
        send_semaphores: Any,
        receive_semaphores: Any,
    ) -> None:
        residual_dma = pltpu.make_async_remote_copy(
            residual_ref,
            residual_out_ref,
            send_semaphores[0],
            receive_semaphores[0],
            destination_ref[0],
        )
        metadata_dma = pltpu.make_async_remote_copy(
            metadata_ref,
            metadata_out_ref,
            send_semaphores[1],
            receive_semaphores[1],
            destination_ref[0],
        )
        with jax.named_scope("start_remote_residual"):
            residual_dma.start()
        with jax.named_scope("start_remote_metadata"):
            metadata_dma.start()
        with jax.named_scope("wait_remote_residual"):
            residual_dma.wait()
        with jax.named_scope("wait_remote_metadata"):
            metadata_dma.wait()

    result_shapes = (
        jax.ShapeDtypeStruct(residual.shape, residual.dtype),
        jax.ShapeDtypeStruct(metadata.shape, metadata.dtype),
    )
    call = pl.pallas_call(
        kernel,
        out_shape=result_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ),
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ),
            scratch_shapes=(
                (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
                (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=config.collective_id,
            dimension_semantics=(),
        ),
        interpret=interpret,
        name=(
            "greenfield_stage_remote_copy_"
            f"bf16_{config.hidden_width}_s32_{config.metadata_width}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=0,
            bytes_accessed=(
                2 * config.hidden_width + 4 * config.metadata_width
            ),
            transcendentals=0,
        ),
    )
    return call(destination, residual, metadata)


def stage_remote_copy_kernel(
    residual: Any,
    metadata: Any,
    *,
    axis_name: str,
    pairs: Sequence[Sequence[int]],
    backend: Literal["reference", "pallas"] = "reference",
    config: StageRemoteCopyConfig = StageRemoteCopyConfig(),
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Dispatch the accepted fallback or default-off asynchronous DMA path."""

    targets = _canonical_targets(pairs, config.total_devices)
    if backend == "reference":
        canonical_pairs = tuple(
            (source, target) for source, target in enumerate(targets)
        )
        return (
            lax.ppermute(residual, axis_name, canonical_pairs),
            lax.ppermute(metadata, axis_name, canonical_pairs),
        )
    if backend == "pallas":
        rank = lax.axis_index(axis_name)
        destination_rank = jnp.asarray(targets, dtype=jnp.int32)[rank]
        return stage_remote_copy_pallas(
            residual,
            metadata,
            destination_rank,
            config=config,
            interpret=interpret,
        )
    raise ValueError(f"unknown stage remote-copy backend {backend!r}")
