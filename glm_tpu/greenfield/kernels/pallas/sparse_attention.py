"""Fused stage-local selected-KV gather and sparse MLA for TPU v4.

The readable fallback first materializes ``[1, top_k, cache_width]`` in HBM.
The Pallas path never creates that tensor. A small exact TensorCore network
orders one owner's selected positions, ordinary JAX resolves only compact
page-row metadata, and the attention kernel dynamically DMAs an aligned
eight-row TPU-v4 tile around each live cache row into VMEM. It masks the seven
overfetch lanes before immediately consuming the selected lane in online
softmax. Only the attended latent and additive LSE leave the kernel.

This module is deliberately default-off and independent of legacy execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..reference.attention import (
    MlaNumericalContract,
    SparseAttentionResult,
    StageLocalKvLayout,
    gather_stage_local_selected_kv,
    sparse_mla_attention,
)
from ..reference.dsa import SelectedPositions
from .topk import bitonic_sort_pairs


_NEGATIVE_INFINITY = float("-inf")
_NO_POSITION = -1
_MAX_EXACT_FP32_INTEGER = 1 << 24
_TPU_V4_DMA_ROWS = 8


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("power-of-two input must be positive")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True, slots=True)
class SparseMlaConfig:
    """Static TensorCore/DMA geometry for one local-owner partial."""

    segment_block: int = 128
    sort_tile: int = 128
    dma_rows: int = _TPU_V4_DMA_ROWS
    vmem_limit_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("segment_block", "sort_tile", "dma_rows"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sort_tile != 128:
            raise ValueError("TPU-v4 selected-position sort tiles must be 128")
        if self.dma_rows != _TPU_V4_DMA_ROWS:
            raise ValueError("TPU-v4 BF16 VMEM DMA rows must be 8")
        if self.vmem_limit_bytes is not None and (
            not isinstance(self.vmem_limit_bytes, int)
            or isinstance(self.vmem_limit_bytes, bool)
            or self.vmem_limit_bytes <= 0
        ):
            raise ValueError("vmem_limit_bytes must be a positive integer")


def _validate_owner_index(owner_index: Any, layout: StageLocalKvLayout) -> Any:
    if isinstance(owner_index, int) and not isinstance(owner_index, bool):
        if not 0 <= owner_index < layout.local_parallel_size:
            raise ValueError("owner_index is outside the local stage group")
        return jnp.asarray(owner_index, dtype=jnp.int32)
    if (
        not hasattr(owner_index, "shape")
        or owner_index.shape != ()
        or not jnp.issubdtype(owner_index.dtype, jnp.integer)
    ):
        raise ValueError("owner_index must be an integer scalar")
    return owner_index.astype(jnp.int32)


def _validate_inputs(
    query_nope_absorbed: Any,
    query_rope: Any,
    cache_local: Any,
    block_tables: Any,
    selected: SelectedPositions,
    context_lengths: Any,
    *,
    layout: StageLocalKvLayout,
    contract: MlaNumericalContract,
    config: SparseMlaConfig,
    owner_index: Any,
    interpret: bool,
) -> Any:
    if query_nope_absorbed.shape != (
        1,
        contract.num_heads,
        contract.kv_lora_rank,
    ):
        raise ValueError("absorbed sparse-MLA query has an invalid shape")
    if query_rope.shape != (
        1,
        contract.num_heads,
        contract.qk_rope_head_dim,
    ):
        raise ValueError("RoPE sparse-MLA query has an invalid shape")
    if query_nope_absorbed.dtype != query_rope.dtype or (
        query_nope_absorbed.dtype not in (jnp.bfloat16, jnp.float32)
    ):
        raise ValueError("sparse-MLA queries must share BF16 or FP32 dtype")
    if cache_local.ndim != 3 or cache_local.shape[1:] != (
        layout.local_rows_per_page,
        contract.packed_cache_width,
    ):
        raise ValueError("local sparse-MLA cache has an invalid shape")
    if cache_local.shape[0] <= 0:
        raise ValueError("local sparse-MLA cache must expose physical pages")
    if cache_local.shape[0] * layout.local_rows_per_page < config.dma_rows:
        raise ValueError("local sparse-MLA cache is smaller than one DMA tile")
    if cache_local.dtype != query_nope_absorbed.dtype:
        raise ValueError("query and local sparse-MLA cache dtypes must match")
    if layout.packed_cache_width != contract.packed_cache_width:
        raise ValueError("KV layout and sparse-MLA contract widths disagree")
    if block_tables.ndim != 2 or block_tables.shape[0] != 1 or (
        block_tables.shape[1] <= 0 or block_tables.dtype != jnp.int32
    ):
        raise ValueError("block_tables must be nonempty int32[1,max_blocks]")
    if selected.positions.shape != (1, contract.top_k) or (
        selected.positions.dtype != jnp.int32
    ):
        raise ValueError("selected positions must be int32[1,top_k]")
    if selected.valid_counts.shape != (1,) or (
        selected.valid_counts.dtype != jnp.int32
    ):
        raise ValueError("selected valid counts must be int32[1]")
    if context_lengths.shape != (1,) or context_lengths.dtype != jnp.int32:
        raise ValueError("context lengths must be int32[1]")
    capacity = block_tables.shape[1] * layout.logical_page_size
    if capacity >= _MAX_EXACT_FP32_INTEGER:
        raise ValueError("selected positions exceed exact FP32 integer range")
    if not interpret:
        if config.segment_block % 128:
            raise ValueError("compiled TPU-v4 segment blocks must divide into 128")
        if contract.top_k % config.segment_block:
            raise ValueError("compiled top_k must divide exactly into segment blocks")
        if contract.packed_cache_width % 128:
            raise ValueError("compiled TPU-v4 packed cache width must divide into 128")
        if contract.kv_lora_rank % 128:
            raise ValueError("compiled TPU-v4 latent width must divide into 128")
    return _validate_owner_index(owner_index, layout)


def _owner_position_order_pallas(
    positions: Any,
    valid_counts: Any,
    owner_index: Any,
    *,
    layout: StageLocalKvLayout,
    global_context_size: int,
    config: SparseMlaConfig,
    interpret: bool,
) -> tuple[Any, Any, Any]:
    """Return one owner's exact ascending positions and metadata predicates."""

    width = positions.shape[1]
    padded_width = max(config.sort_tile, _next_power_of_two(width))
    padded = jnp.pad(
        positions, ((0, 0), (0, padded_width - width)), constant_values=-1
    )
    safe_count = jnp.clip(valid_counts, jnp.int32(0), jnp.int32(width))
    slots = jnp.arange(width, dtype=jnp.int32)[None, :]
    live = slots < safe_count[:, None]
    input_well_formed = (
        (valid_counts >= 0)
        & (valid_counts <= width)
        & jnp.all(
            jnp.where(
                live,
                (positions >= 0) & (positions < global_context_size),
                positions == _NO_POSITION,
            ),
            axis=1,
        )
    )
    position_owner = layout.owner(jnp.maximum(positions, jnp.int32(0)))
    owned = live & (positions >= 0) & (positions < global_context_size) & (
        position_owner == owner_index
    )
    owned_counts = jnp.sum(owned, axis=1, dtype=jnp.int32)

    def kernel(
        position_ref: Any,
        valid_count_ref: Any,
        owner_ref: Any,
        output_ref: Any,
    ) -> None:
        position_f32 = position_ref[...].astype(jnp.float32)
        position_slots = lax.broadcasted_iota(
            jnp.int32, (1, 1, padded_width), 2
        )
        position_live = position_slots < valid_count_ref[0]
        exact_position = (
            (position_f32 >= jnp.float32(0.0))
            & (position_f32 < jnp.float32(global_context_size))
        )
        in_page = position_f32.astype(jnp.int32) % jnp.int32(
            layout.logical_page_size
        )
        actual_owner = in_page // jnp.int32(layout.local_rows_per_page)
        position_owned = position_live & exact_position & (
            actual_owner == owner_ref[0]
        )
        ordering_scores = jnp.where(
            position_owned, -position_f32, _NEGATIVE_INFINITY
        )
        ordered_scores, ordered_positions = bitonic_sort_pairs(
            ordering_scores, position_f32, width=padded_width
        )
        output_ref[...] = jnp.where(
            ordered_scores[..., :width] == _NEGATIVE_INFINITY,
            jnp.float32(_NO_POSITION),
            ordered_positions[..., :width],
        ).astype(jnp.int32)

    def position_index() -> tuple[int, int, int]:
        return 0, 0, 0

    ordered = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, 1, width), jnp.int32),
        grid=(),
        in_specs=(
            pl.BlockSpec((1, 1, padded_width), position_index),
            pl.BlockSpec((1,), lambda: (0,)),
            pl.BlockSpec((1,), lambda: (0,)),
        ),
        out_specs=pl.BlockSpec((1, 1, width), position_index),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=interpret,
        name=f"greenfield_owner_position_order_k{width}",
    )(
        padded[:, None, :],
        safe_count,
        jnp.reshape(owner_index, (1,)),
    )[:, 0, :]
    ordered_slots = jnp.arange(width, dtype=jnp.int32)[None, :]
    ordered_live = ordered_slots < owned_counts[:, None]
    ordered_distinct = jnp.all(
        jnp.where(
            ordered_slots[:, 1:] < owned_counts[:, None],
            ordered[:, 1:] > ordered[:, :-1],
            True,
        ),
        axis=1,
    )
    output_well_formed = jnp.all(
        jnp.where(ordered_live, ordered >= 0, ordered == _NO_POSITION), axis=1
    )
    return ordered, owned_counts, (
        input_well_formed & output_well_formed & ordered_distinct
    )


def _resolve_owner_flat_rows(
    ordered_positions: Any,
    owned_counts: Any,
    block_tables: Any,
    context_lengths: Any,
    *,
    cache_local: Any,
    layout: StageLocalKvLayout,
    owner_index: Any,
    selection_valid: Any,
) -> tuple[Any, Any]:
    """Resolve compact positions to safe local flat-row ids and health."""

    width = ordered_positions.shape[1]
    slots = jnp.arange(width, dtype=jnp.int32)[None, :]
    live = slots < owned_counts[:, None]
    position_ok = (
        (ordered_positions >= 0)
        & (ordered_positions < context_lengths[:, None])
    )
    safe_positions = jnp.where(live & position_ok, ordered_positions, 0)
    logical_blocks = safe_positions // jnp.int32(layout.logical_page_size)
    block_ok = logical_blocks < block_tables.shape[1]
    safe_blocks = jnp.clip(logical_blocks, 0, block_tables.shape[1] - 1)
    page_ids = jnp.take_along_axis(block_tables, safe_blocks, axis=1)
    page_ok = (page_ids >= 0) & (page_ids < cache_local.shape[0])
    safe_pages = jnp.clip(page_ids, 0, cache_local.shape[0] - 1)
    within_page = safe_positions % jnp.int32(layout.logical_page_size)
    actual_owner = within_page // jnp.int32(layout.local_rows_per_page)
    owner_ok = actual_owner == owner_index
    local_row = within_page % jnp.int32(layout.local_rows_per_page)
    flat_rows = safe_pages * jnp.int32(layout.local_rows_per_page) + local_row
    slot_ok = live & position_ok & block_ok & page_ok & owner_ok
    flat_rows = jnp.where(slot_ok, flat_rows, jnp.int32(0))
    capacity = block_tables.shape[1] * layout.logical_page_size
    length_ok = (context_lengths >= 0) & (context_lengths <= capacity)
    valid = (
        selection_valid
        & length_ok
        & jnp.all(jnp.where(live, slot_ok, True), axis=1)
    )
    return flat_rows.astype(jnp.int32), valid


def _fused_selected_kv_attention_pallas(
    query_nope_absorbed: Any,
    query_rope: Any,
    cache_flat: Any,
    flat_rows: Any,
    valid_counts: Any,
    *,
    contract: MlaNumericalContract,
    config: SparseMlaConfig,
    interpret: bool,
) -> tuple[Any, Any]:
    """DMA selected rows into VMEM and consume them in online softmax."""

    heads = contract.num_heads
    latent = contract.kv_lora_rank
    cache_width = contract.packed_cache_width
    segment_width = contract.top_k
    segment_block = min(config.segment_block, segment_width)
    segment_blocks = _ceil_div(segment_width, segment_block)
    if segment_blocks * segment_block != segment_width:
        pad = segment_blocks * segment_block - segment_width
        flat_rows = jnp.pad(flat_rows, ((0, 0), (0, pad)))
    physical_segment_width = flat_rows.shape[1]
    dma_rows = config.dma_rows
    dma_starts = jnp.minimum(
        flat_rows, jnp.int32(cache_flat.shape[0] - dma_rows)
    )
    dma_lanes = flat_rows - dma_starts
    precision = (
        lax.Precision.HIGHEST
        if query_nope_absorbed.dtype == jnp.float32
        else lax.Precision.DEFAULT
    )

    def kernel(
        valid_count_ref: Any,
        query_nope_ref: Any,
        query_rope_ref: Any,
        cache_ref: Any,
        dma_start_ref: Any,
        dma_lane_ref: Any,
        output_ref: Any,
        lse_ref: Any,
        cache_tile_ref: Any,
        maximum_ref: Any,
        denominator_ref: Any,
        accumulator_ref: Any,
        dma_sem: Any,
    ) -> None:
        block = pl.program_id(0)
        block_start = block * segment_block
        valid_count = valid_count_ref[0]

        @pl.when(block == 0)
        def initialize() -> None:
            maximum_ref[...] = jnp.full(
                maximum_ref.shape, _NEGATIVE_INFINITY, jnp.float32
            )
            denominator_ref[...] = jnp.zeros(
                denominator_ref.shape, jnp.float32
            )
            accumulator_ref[...] = jnp.zeros(
                accumulator_ref.shape, jnp.float32
            )

        @pl.when((block == 0) & (valid_count == 0))
        def write_empty() -> None:
            output_ref[0] = jnp.zeros(output_ref.shape[1:], output_ref.dtype)
            lse_ref[0] = jnp.full(lse_ref.shape[1:], -jnp.inf, jnp.float32)

        @pl.when(block_start < valid_count)
        def attend_tile() -> None:
            descriptors = []
            for row in range(segment_block):
                descriptor = pltpu.make_async_copy(
                    cache_ref.at[
                        pl.ds(dma_start_ref[0, row], dma_rows), :
                    ],
                    cache_tile_ref.at[row],
                    dma_sem,
                )
                descriptors.append(descriptor)
            for descriptor in descriptors:
                descriptor.start()
            for descriptor in descriptors:
                descriptor.wait()

            group_slots = (
                jnp.arange(segment_block, dtype=jnp.int32)
                + jnp.int32(block_start)
            )
            dma_lanes_iota = jnp.arange(dma_rows, dtype=jnp.int32)[None, :]
            selected_lanes = dma_lanes_iota == jnp.reshape(
                dma_lane_ref[...], (segment_block, 1)
            )
            live_groups = group_slots[:, None] < valid_count
            selected_and_live = (selected_lanes & live_groups).reshape(
                segment_block * dma_rows, 1
            )
            # The TPU-v4 VMEM tile requires eight-row DMA slices. Keep only the
            # selected dynamic lane from each group and zero all overfetch and
            # tail lanes before both QK and PV, including poisoned padding.
            cache_tile = jnp.where(
                selected_and_live,
                cache_tile_ref[...].reshape(
                    segment_block * dma_rows, cache_width
                ),
                jnp.zeros((), cache_tile_ref.dtype),
            )
            query_packed = jnp.concatenate(
                (
                    query_nope_ref[...],
                    query_rope_ref[...],
                    jnp.zeros(
                        (
                            1,
                            heads,
                            cache_width
                            - latent
                            - contract.qk_rope_head_dim,
                        ),
                        dtype=query_nope_ref.dtype,
                    ),
                ),
                axis=-1,
            )
            scores = lax.dot_general(
                query_packed[0],
                cache_tile,
                dimension_numbers=(((1,), (1,)), ((), ())),
                precision=precision,
                preferred_element_type=jnp.float32,
            ) * jnp.float32(contract.softmax_scale)
            absolute_slots = jnp.broadcast_to(
                group_slots[:, None], (segment_block, dma_rows)
            ).reshape(1, segment_block * dma_rows)
            scores = jnp.where(
                (absolute_slots < valid_count)
                & jnp.transpose(selected_and_live),
                scores,
                _NEGATIVE_INFINITY,
            )
            block_maximum = jnp.max(scores, axis=1, keepdims=True)
            maximum = jnp.maximum(maximum_ref[...], block_maximum)
            correction = jnp.exp(maximum_ref[...] - maximum)
            probabilities = jnp.exp(scores - maximum)
            denominator = (
                denominator_ref[...] * correction
                + jnp.sum(probabilities, axis=1, keepdims=True)
            )
            partial = lax.dot_general(
                probabilities.astype(cache_tile.dtype),
                cache_tile[:, :latent],
                dimension_numbers=(((1,), (0,)), ((), ())),
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            accumulator = accumulator_ref[...] * correction + partial
            maximum_ref[...] = maximum
            denominator_ref[...] = denominator
            accumulator_ref[...] = accumulator

            last_live_block = (valid_count - jnp.int32(1)) // jnp.int32(
                segment_block
            )

            @pl.when(block == last_live_block)
            def finalize() -> None:
                output_ref[0] = (
                    accumulator / denominator
                ).astype(output_ref.dtype)
                lse = maximum + jnp.log(denominator)
                lse_ref[0] = jnp.broadcast_to(lse, lse_ref.shape[1:])

    def query_index(block: Any, valid_count: Any) -> tuple[int, int, int]:
        del block, valid_count
        return 0, 0, 0

    def compact_row_index(block: Any, valid_count: Any) -> tuple[int, Any]:
        del valid_count
        return 0, block

    def output_index(block: Any, valid_count: Any) -> tuple[int, int, int]:
        del block, valid_count
        return 0, 0, 0

    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(segment_blocks,),
            in_specs=(
                pl.BlockSpec((1, heads, latent), query_index),
                pl.BlockSpec(
                    (1, heads, contract.qk_rope_head_dim), query_index
                ),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec((1, segment_block), compact_row_index),
                pl.BlockSpec((1, segment_block), compact_row_index),
            ),
            out_specs=(
                pl.BlockSpec((1, heads, latent), output_index),
                pl.BlockSpec((1, heads, 128), output_index),
            ),
            scratch_shapes=(
                pltpu.VMEM(
                    (segment_block, dma_rows, cache_width),
                    query_nope_absorbed.dtype,
                ),
                pltpu.VMEM((heads, 1), jnp.float32),
                pltpu.VMEM((heads, 1), jnp.float32),
                pltpu.VMEM((heads, latent), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (1, heads, latent), query_nope_absorbed.dtype
            ),
            jax.ShapeDtypeStruct((1, heads, 128), jnp.float32),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        interpret=interpret,
        name=(
            "greenfield_fused_selected_kv_sparse_mla_"
            f"h{heads}_k{segment_width}_b{segment_block}_w{cache_width}_d{dma_rows}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=(
                heads
                * physical_segment_width
                * dma_rows
                * (2 * cache_width + 2 * latent + 8)
            ),
            bytes_accessed=(
                physical_segment_width
                * dma_rows
                * cache_width
                * cache_flat.dtype.itemsize
                + heads
                * (latent + contract.qk_rope_head_dim)
                * query_nope_absorbed.dtype.itemsize
                + heads * latent * query_nope_absorbed.dtype.itemsize
            ),
            transcendentals=heads * physical_segment_width * dma_rows,
        ),
    )
    output, lse_lanes = call(
        valid_counts,
        query_nope_absorbed,
        query_rope,
        cache_flat,
        dma_starts,
        dma_lanes,
    )
    return output, lse_lanes[:, :, 0]


def stage_local_sparse_mla_pallas(
    query_nope_absorbed: Any,
    query_rope: Any,
    cache_local: Any,
    block_tables: Any,
    selected: SelectedPositions,
    context_lengths: Any,
    *,
    layout: StageLocalKvLayout,
    owner_index: int | Any,
    contract: MlaNumericalContract = MlaNumericalContract(),
    config: SparseMlaConfig = SparseMlaConfig(),
    interpret: bool = False,
) -> SparseAttentionResult:
    """Return one owner's fused selected-KV/sparse-attention partial."""

    owner = _validate_inputs(
        query_nope_absorbed,
        query_rope,
        cache_local,
        block_tables,
        selected,
        context_lengths,
        layout=layout,
        contract=contract,
        config=config,
        owner_index=owner_index,
        interpret=interpret,
    )
    capacity = block_tables.shape[1] * layout.logical_page_size
    ordered, owned_counts, selection_valid = _owner_position_order_pallas(
        selected.positions,
        selected.valid_counts,
        owner,
        layout=layout,
        global_context_size=capacity,
        config=config,
        interpret=interpret,
    )
    flat_rows, metadata_valid = _resolve_owner_flat_rows(
        ordered,
        owned_counts,
        block_tables,
        context_lengths,
        cache_local=cache_local,
        layout=layout,
        owner_index=owner,
        selection_valid=selection_valid,
    )
    cache_flat = cache_local.reshape(
        cache_local.shape[0] * layout.local_rows_per_page,
        contract.packed_cache_width,
    )
    output, lse = _fused_selected_kv_attention_pallas(
        query_nope_absorbed,
        query_rope,
        cache_flat,
        flat_rows,
        owned_counts,
        contract=contract,
        config=config,
        interpret=interpret,
    )
    return SparseAttentionResult(output, lse, metadata_valid)


def stage_local_sparse_mla_kernel(
    query_nope_absorbed: Any,
    query_rope: Any,
    cache_local: Any,
    block_tables: Any,
    selected: SelectedPositions,
    context_lengths: Any,
    *,
    layout: StageLocalKvLayout,
    owner_index: int | Any,
    contract: MlaNumericalContract = MlaNumericalContract(),
    backend: Literal["reference", "pallas"] = "reference",
    config: SparseMlaConfig = SparseMlaConfig(),
    interpret: bool = False,
) -> SparseAttentionResult:
    """Dispatch the default-off fused kernel or exact readable fallback."""

    if backend == "reference":
        segment = gather_stage_local_selected_kv(
            cache_local,
            block_tables,
            selected,
            context_lengths,
            layout=layout,
            owner_index=owner_index,
        )
        return sparse_mla_attention(
            query_nope_absorbed, query_rope, segment, contract=contract
        )
    if backend == "pallas":
        return stage_local_sparse_mla_pallas(
            query_nope_absorbed,
            query_rope,
            cache_local,
            block_tables,
            selected,
            context_lengths,
            layout=layout,
            owner_index=owner_index,
            contract=contract,
            config=config,
            interpret=interpret,
        )
    raise ValueError(f"unsupported sparse-MLA backend {backend!r}")
