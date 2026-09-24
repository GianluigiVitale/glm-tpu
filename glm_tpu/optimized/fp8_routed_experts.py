"""Route-grouped raw-FP8 projections for the WS32 one-row MoE (opt-in challenger).

The frozen ``ws32_moe_pallas_from_routes_mapped`` executes the eight routed
experts of one decode row as eight ``lax.cond`` blocks.  Each owned block
launches three ``pallas_call``s whose grid walks 128x128 scale blocks one at
a time (``(2048/128) x (1536/128) = 192`` grid steps per projection) and ends
with its own feature-4 ``psum``.  A sparse layer therefore issues up to 27
kernel launches, 8 conditionals and 10 collectives for a single row.  The
reference GLM-5.3-Flash engine (``glm53/pallas_moe.py``) instead runs ONE
kernel whose grid is the route slots, takes the selected expert ids as a
scalar-prefetch operand, and DMAs exactly the chosen expert per slot.

``fp8_routed_projection`` is that shape for the existing raw-U8 ``[E, N, K]``
checkpoint layout and the tile-local dequantization the frozen kernel uses:

* grid ``(owned slots, N/output_tile, K/contraction_tile)`` with a traced
  leading size: owned slots are compacted to the front through scalar
  prefetch, so unowned slots cost no grid iteration and no DMA (measured on
  the pod: iterating skipped slots at 128x128 tiles cost more than the
  frozen conditionals saved);
* ``output_tile``/``contraction_tile`` may span several 128x128 scale blocks;
  every block is still decoded and accumulated exactly as the frozen kernel
  does (bf16 decode per block, the same 8-row MXU tile, FP32 accumulation in
  ascending K order), so owned rows are bitwise equal to ``fp8_block_matmul``
  / ``fp8_block_matmul_f32`` with the default 128x128 tiles;
* gate and up projections of one slot share a grid step, halving DMA/steps.

``ws32_moe_grouped_routes_mapped`` composes this into the frozen layer's exact
arithmetic boundaries (FP32 feature reduction of stacked gate/up partials,
BF16 activation, BF16 route weighting, BF16 route sum, FP32 expert-8
reduction, ``routed * 2.5 + shared``) with two collectives per layer instead
of ten and four kernel launches instead of up to 27.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .reference.moe import GlmMoeNumericalContract


@dataclass(frozen=True, slots=True)
class RoutedProjectionConfig:
    """Static tile contract for :func:`fp8_routed_projection`.

    ``output_tile`` and ``contraction_tile`` are multiples of the checkpoint
    scale block.  ``contraction_tile`` may cover 1, 2, 4 or 8 scale blocks so
    that one aligned ``[8, 128]`` VMEM scale slab serves a whole grid step.
    """

    block_shape: tuple[int, int] = (128, 128)
    output_tile: int = 512
    contraction_tile: int = 512
    # TPU-proven correctness requirement: the forced empty-owner grid row
    # must write a defined output window. False reproduces the old bug only.
    write_empty_slot: bool = True

    def __post_init__(self) -> None:
        if self.write_empty_slot is not True:
            raise ValueError('optimized release always initializes empty route slots')
        if len(self.block_shape) != 2 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0
            for v in self.block_shape
        ):
            raise ValueError("routed FP8 block shape must contain two positive integers")
        for name in ("output_tile", "contraction_tile"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.output_tile % self.block_shape[0]:
            raise ValueError("output_tile must be a multiple of the output scale block")
        if self.contraction_tile % self.block_shape[1]:
            raise ValueError(
                "contraction_tile must be a multiple of the contraction scale block"
            )
        if 8 % self.blocks_per_contraction_tile:
            raise ValueError(
                "contraction_tile may span 1, 2, 4 or 8 scale blocks (one VMEM slab)"
            )
        if self.blocks_per_output_tile > 128:
            raise ValueError("output_tile may span at most 128 scale blocks")

    @property
    def blocks_per_output_tile(self) -> int:
        return self.output_tile // self.block_shape[0]

    @property
    def blocks_per_contraction_tile(self) -> int:
        return self.contraction_tile // self.block_shape[1]

    @classmethod
    def frozen_tiles(cls, block_shape: tuple[int, int]) -> "RoutedProjectionConfig":
        """The frozen kernel's geometry: one scale block per grid step."""

        return cls(block_shape=block_shape, output_tile=block_shape[0],
                   contraction_tile=block_shape[1])


def _scale_entry(scale_slab: Any, block_row: Any, block_column: Any) -> Any:
    """Select one FP32 scale from an aligned ``[8, 128]`` VMEM slab in registers.

    Same masked reduction as the frozen ``_scale_value``: Mosaic on TPU v4 has
    no dynamically indexed scalar VMEM load, so the other entries are zeroed.
    """

    rows = lax.broadcasted_iota(jnp.int32, scale_slab.shape, 0)
    columns = lax.broadcasted_iota(jnp.int32, scale_slab.shape, 1)
    mask = (rows == block_row % jnp.int32(8)) & (columns == block_column)
    return jnp.sum(jnp.where(mask, scale_slab, jnp.float32(0.0)), dtype=jnp.float32)


def _routed_scale_table(scale: Any, *, contraction_blocks: int) -> Any:
    """``[E, N/bn, K/bk]`` -> ``[E, ceil8(K/bk), 128]`` (K-block rows, N-block lanes)."""

    experts, output_blocks, k_blocks = scale.shape
    if k_blocks != contraction_blocks:
        raise ValueError("routed scale contraction blocks disagree with the weights")
    if output_blocks > 128:
        raise ValueError("routed FP8 supports at most 128 output-scale blocks")
    padded_k = ((k_blocks + 7) // 8) * 8
    return jnp.pad(
        jnp.swapaxes(scale, 1, 2),
        ((0, 0), (0, padded_k - k_blocks), (0, 128 - output_blocks)),
    )


def fp8_routed_projection(
    lhs: Any,
    weights: tuple[tuple[Any, Any], ...],
    local_expert_ids: Any,
    owned: Any,
    *,
    config: RoutedProjectionConfig = RoutedProjectionConfig(),
    result_dtype: Any = jnp.float32,
    interpret: bool = False,
) -> Any:
    """Project one BF16 row per route slot against that slot's selected expert.

    ``lhs`` is ``[slots, K]``; every ``weights`` entry is a
    ``(bits [E, N, K] uint8, scale [E, N/bn, K/bk] float32)`` table pair sharing
    ``N`` and ``K`` (one pair for a down projection, two for fused gate/up).
    ``local_expert_ids [slots]`` selects the expert of each slot (clipped to
    ``[0, E)``); ``owned [slots]`` marks the slots this chip computes.  The
    result is ``[slots, len(weights), N]``.

    The grid's leading dimension is the NUMBER OF OWNED SLOTS (a traced size,
    as in the repo's Megablox-style prefill kernel): owned slots are compacted
    to the front through scalar prefetch, so unowned slots cost no grid
    iteration and no DMA at all.  Their output rows come from the aliased
    zero-initialised output buffer and are exact zeros.
    """

    if not isinstance(weights, tuple) or not 1 <= len(weights) <= 2:
        raise ValueError("routed FP8 projection takes one or two weight table pairs")
    if lhs.ndim != 2 or lhs.dtype != jnp.bfloat16:
        raise ValueError("routed FP8 lhs must be BF16 [slots, K]")
    slots, contraction = lhs.shape
    if slots <= 0 or contraction <= 0:
        raise ValueError("routed FP8 lhs dimensions must be positive")
    first_bits = weights[0][0]
    if first_bits.ndim != 3:
        raise ValueError("routed FP8 weights must be [E, N, K] uint8")
    experts, output, weight_contraction = first_bits.shape
    if min(experts, output) <= 0 or weight_contraction != contraction:
        raise ValueError("routed FP8 weight/input geometry disagrees")
    bn, bk = config.block_shape
    if output % config.output_tile or contraction % config.contraction_tile:
        raise ValueError(
            "routed FP8 requires N/K divisible by the output/contraction tiles"
        )
    output_blocks, k_blocks = output // bn, contraction // bk
    for bits, scale in weights:
        if bits.shape != (experts, output, contraction) or bits.dtype != jnp.uint8:
            raise ValueError("routed FP8 weight tables must share [E, N, K] uint8")
        if scale.shape != (experts, output_blocks, k_blocks) or scale.dtype != jnp.float32:
            raise ValueError("routed FP8 scales must be [E, N/bn, K/bk] float32")
    if local_expert_ids.shape != (slots,) or not jnp.issubdtype(
        local_expert_ids.dtype, jnp.integer
    ):
        raise ValueError("routed FP8 expert ids must be one integer per slot")
    if owned.shape != (slots,) or owned.dtype != jnp.bool_:
        raise ValueError("routed FP8 ownership must be one boolean per slot")
    dtype = jnp.dtype(result_dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16)):
        raise ValueError("routed FP8 result must be FP32 or BF16")

    tables = len(weights)
    tn, tk = config.output_tile, config.contraction_tile
    bpn, bpk = config.blocks_per_output_tile, config.blocks_per_contraction_tile
    n_tiles, k_tiles = output // tn, contraction // tk
    ids = jnp.clip(local_expert_ids.astype(jnp.int32), 0, experts - 1)
    # Owned slots first (stable), then the count of active grid rows.
    slot_of_step = jnp.argsort(jnp.where(owned, jnp.int32(0), jnp.int32(1)), stable=True).astype(jnp.int32)
    expert_of_step = jnp.take(ids, slot_of_step)
    owned_count = jnp.sum(owned.astype(jnp.int32))
    active = jnp.maximum(owned_count, 1)
    metadata = jnp.concatenate((slot_of_step, expert_of_step, owned_count[None]))
    scale_tables = tuple(
        _routed_scale_table(scale, contraction_blocks=k_blocks) for _, scale in weights
    )

    def kernel(meta_ref, lhs_ref, *refs):
        weight_refs = refs[:tables]
        scale_refs = refs[tables : 2 * tables]
        initial_ref = refs[2 * tables]
        out_ref = refs[2 * tables + 1]
        acc_refs = refs[2 * tables + 2 :]
        del initial_ref  # aliased zero buffer: unvisited slots stay zero
        step, ni, ki = pl.program_id(0), pl.program_id(1), pl.program_id(2)
        live = step < meta_ref[2 * slots]

        @pl.when(ki == 0)
        def initialize() -> None:
            for acc_ref in acc_refs:
                acc_ref[...] = jnp.zeros_like(acc_ref)

        @pl.when(live)
        def compute() -> None:
            # The frozen kernel pads its one live row to an 8-row MXU tile;
            # rows 1..7 are zero here too so both dots are the same operation.
            x = lhs_ref[...]
            for j in range(bpk):
                block_row = ki * jnp.int32(bpk) + jnp.int32(j)
                x_block = x[:, j * bk : (j + 1) * bk]
                for t in range(tables):
                    slab = scale_refs[t][...]
                    for i in range(bpn):
                        block_column = ni * jnp.int32(bpn) + jnp.int32(i)
                        scale_value = _scale_entry(slab, block_row, block_column)
                        # Identical to the frozen kernel: reinterpret only the
                        # resident VMEM block, decode to FP32, apply the block
                        # scale, round to BF16, MXU dot with FP32 accumulation.
                        bits = weight_refs[t][pl.ds(i * bn, bn), pl.ds(j * bk, bk)]
                        decoded = (
                            lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(
                                jnp.float32
                            )
                            * scale_value
                        ).astype(jnp.bfloat16)
                        update = lax.dot_general(
                            x_block,
                            decoded,
                            dimension_numbers=(((1,), (1,)), ((), ())),
                            preferred_element_type=jnp.float32,
                        )
                        acc_refs[t][:, i * bn : (i + 1) * bn] += update

        @pl.when((live | jnp.bool_(config.write_empty_slot)) & (ki == k_tiles - 1))
        def store() -> None:
            for t in range(tables):
                out_ref[t : t + 1, :] = acc_refs[t][0:1, :].astype(dtype)

    def slot_index(step, meta):
        return meta[jnp.minimum(step, slots - 1)]

    def expert_index(step, meta):
        return meta[slots + jnp.minimum(step, slots - 1)]

    def lhs_index(step, ni, ki, meta):
        del ni
        return slot_index(step, meta), 0, ki

    def weight_index(step, ni, ki, meta):
        return expert_index(step, meta), ni, ki

    def scale_index(step, ni, ki, meta):
        del ni
        return expert_index(step, meta), (ki * jnp.int32(bpk)) // jnp.int32(8), 0

    def out_index(step, ni, ki, meta):
        del ki
        return slot_index(step, meta), 0, ni

    out_spec = pl.BlockSpec((None, tables, tn), out_index)
    in_specs = [pl.BlockSpec((None, 8, tk), lhs_index)]
    in_specs += [pl.BlockSpec((None, tn, tk), weight_index) for _ in range(tables)]
    in_specs += [pl.BlockSpec((None, 8, 128), scale_index) for _ in range(tables)]
    in_specs += [out_spec]
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((slots, tables, output), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=tuple(in_specs),
            out_specs=out_spec,
            grid=(active, n_tiles, k_tiles),
            scratch_shapes=tuple(
                pltpu.VMEM((8, tn), jnp.float32) for _ in range(tables)
            ),
        ),
        # metadata, lhs, weights..., scales..., initial -> the initial zero
        # buffer is donated to the output so unvisited slots read as zeros.
        input_output_aliases={2 + 2 * tables: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "parallel", "arbitrary")
        ),
        interpret=interpret,
        name=(
            f"glm_perf_fp8_routed_projection_s{slots}_t{tables}"
            f"_k{contraction}_n{output}_tn{tn}_tk{tk}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=2 * slots * tables * contraction * output,
            bytes_accessed=(
                slots * 8 * contraction * 2
                + slots * tables * output * contraction
                + slots * tables * output * dtype.itemsize
            ),
            transcendentals=0,
        ),
    )
    padded_lhs = jnp.zeros((slots, 8, contraction), jnp.bfloat16).at[:, 0, :].set(lhs)
    operands = [padded_lhs]
    operands += [bits for bits, _ in weights]
    operands += list(scale_tables)
    operands += [jnp.zeros((slots, tables, output), dtype)]
    return call(metadata, *operands)


def ws32_moe_grouped_routes_mapped(
    hidden_local: Any,
    route_indices: Any,
    route_weights: Any,
    expert_gate_bits_local: Any,
    expert_gate_scale_local: Any,
    expert_up_bits_local: Any,
    expert_up_scale_local: Any,
    expert_down_bits_local: Any,
    expert_down_scale_local: Any,
    shared_gate_bits_local: Any,
    shared_gate_scale_local: Any,
    shared_up_bits_local: Any,
    shared_up_scale_local: Any,
    shared_down_bits_local: Any,
    shared_down_scale_local: Any,
    *,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    config: RoutedProjectionConfig | None = None,
    expert_axis: str = "expert",
    feature_axis: str = "feature",
    interpret: bool = False,
    shared_bf16: tuple[Any, Any, Any] | None = None,
) -> Any:
    """Route-grouped challenger for ``ws32_moe_pallas_from_routes_mapped``.

    ``shared_bf16 = (gate, up, down)`` supplies pre-decoded BF16 shared-expert
    tables (``glm_tpu.perf.bf16_resident``); the shared FP8 operands are then
    ignored (pass None) and the shared projections are plain dots at the same
    FP32-accumulate boundaries.

    Same inputs, ownership rule, arithmetic boundaries and output as the frozen
    body.  Structural differences only: all routed gate/up partials and the
    shared expert's partials cross the feature axis in ONE stacked FP32
    ``psum``; routed down projections run in one grouped kernel; no per-route
    ``lax.cond``.  ``config`` defaults to 512x512 tiles when the checkpoint
    block is 128x128 and to the frozen one-block tiles otherwise.
    """

    if contract.stage_size != 8:
        raise ValueError("WS32 grouped MoE requires stage_size=8")
    if hidden_local.ndim != 2 or hidden_local.shape[0] != 1:
        raise ValueError("WS32 grouped MoE input must contain one live row")
    if hidden_local.dtype != jnp.bfloat16:
        raise ValueError("WS32 grouped MoE input must be bfloat16")
    if route_indices.shape != (1, contract.top_k) or route_indices.dtype != jnp.int32:
        raise ValueError("WS32 grouped MoE routes must contain one exact int32 row")
    if route_weights.shape != (1, contract.top_k) or route_weights.dtype != jnp.float32:
        raise ValueError("WS32 grouped MoE route weights must contain one FP32 row")
    local_hidden = hidden_local.shape[-1]
    local_experts = expert_gate_bits_local.shape[0]
    if local_experts != contract.local_experts:
        raise ValueError("WS32 grouped MoE local expert ownership drifted")
    expected_gate = (local_experts, contract.intermediate_size, local_hidden)
    expected_down = (local_experts, local_hidden, contract.intermediate_size)
    if expert_gate_bits_local.shape != expected_gate or (
        expert_up_bits_local.shape != expected_gate
    ):
        raise ValueError("WS32 grouped MoE routed gate/up shapes drifted")
    if expert_down_bits_local.shape != expected_down:
        raise ValueError("WS32 grouped MoE routed down shape drifted")
    if shared_bf16 is None:
        if shared_gate_bits_local.shape != expected_gate[1:] or (
            shared_up_bits_local.shape != expected_gate[1:]
            or shared_down_bits_local.shape != expected_down[1:]
        ):
            raise ValueError("WS32 grouped MoE shared expert shapes drifted")
    else:
        if len(shared_bf16) != 3 or any(
            t.dtype != jnp.bfloat16 for t in shared_bf16
        ) or shared_bf16[0].shape != expected_gate[1:] or (
            shared_bf16[1].shape != expected_gate[1:] or shared_bf16[2].shape != expected_down[1:]
        ):
            raise ValueError("WS32 grouped MoE BF16 shared tables drifted")
    if config is None:
        block = tuple(contract.fp8_block_shape)
        config = (
            RoutedProjectionConfig(block_shape=block)
            if block == (128, 128)
            else RoutedProjectionConfig.frozen_tiles(block)
        )
    elif tuple(config.block_shape) != tuple(contract.fp8_block_shape):
        raise ValueError("WS32 grouped MoE tile config block differs from the contract")

    top_k = contract.top_k
    expert_start = lax.axis_index(expert_axis).astype(jnp.int32) * local_experts
    routes = route_indices[0]
    owned = (routes >= expert_start) & (routes < expert_start + local_experts)
    local_ids = jnp.clip(routes - expert_start, jnp.int32(0), jnp.int32(local_experts - 1))
    always = jnp.ones((1,), jnp.bool_)
    zero_id = jnp.zeros((1,), jnp.int32)

    with jax.named_scope("glm_perf_moe_grouped/routed_gate_up"):
        routed_gate_up = fp8_routed_projection(
            jnp.broadcast_to(hidden_local, (top_k, local_hidden)),
            (
                (expert_gate_bits_local, expert_gate_scale_local),
                (expert_up_bits_local, expert_up_scale_local),
            ),
            local_ids,
            owned,
            config=config,
            result_dtype=jnp.float32,
            interpret=interpret,
        )
    with jax.named_scope("glm_perf_moe_grouped/shared_gate_up"):
        if shared_bf16 is None:
            shared_gate_up = fp8_routed_projection(
                hidden_local,
                (
                    (shared_gate_bits_local[None], shared_gate_scale_local[None]),
                    (shared_up_bits_local[None], shared_up_scale_local[None]),
                ),
                zero_id,
                always,
                config=config,
                result_dtype=jnp.float32,
                interpret=interpret,
            )
        else:
            dims = (((1,), (1,)), ((), ()))
            shared_gate_up = jnp.stack(
                (
                    lax.dot_general(hidden_local, shared_bf16[0], dims, preferred_element_type=jnp.float32),
                    lax.dot_general(hidden_local, shared_bf16[1], dims, preferred_element_type=jnp.float32),
                ),
                axis=1,
            )
    stacked = jnp.concatenate((routed_gate_up, shared_gate_up), axis=0)
    with jax.named_scope("glm_perf_moe_grouped/gate_up_feature_reduce"):
        gate_up = lax.psum(stacked, axis_name=feature_axis).astype(jnp.bfloat16)
    activated = (gate_up[:, 0] * jax.nn.sigmoid(gate_up[:, 0]) * gate_up[:, 1]).astype(
        jnp.bfloat16
    )
    with jax.named_scope("glm_perf_moe_grouped/routed_down"):
        down = fp8_routed_projection(
            activated[:top_k],
            ((expert_down_bits_local, expert_down_scale_local),),
            local_ids,
            owned,
            config=config,
            result_dtype=jnp.bfloat16,
            interpret=interpret,
        )[:, 0]
    weighted = (down * route_weights[0][:, None].astype(jnp.bfloat16)).astype(jnp.bfloat16)
    # Same [route, 1, hidden] BF16 reduction shape as the frozen route stack.
    local_routed = jnp.sum(weighted[:, None, :], axis=0, dtype=jnp.bfloat16)
    with jax.named_scope("glm_perf_moe_grouped/routed_expert_reduce"):
        routed = lax.psum(local_routed.astype(jnp.float32), axis_name=expert_axis).astype(
            jnp.bfloat16
        )
    with jax.named_scope("glm_perf_moe_grouped/shared_down"):
        if shared_bf16 is None:
            shared = fp8_routed_projection(
                activated[top_k:],
                ((shared_down_bits_local[None], shared_down_scale_local[None]),),
                zero_id,
                always,
                config=config,
                result_dtype=jnp.bfloat16,
                interpret=interpret,
            )[:, 0]
        else:
            shared = lax.dot_general(
                activated[top_k:], shared_bf16[2], (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
    routed_scale = jnp.asarray(contract.routed_scaling_factor, dtype=jnp.bfloat16)
    return (routed * routed_scale + shared).astype(jnp.bfloat16)
