"""Exact one-row DSA top-k for TPU v4 TensorCores.

TPU v4 has no SparseCore and Mosaic exposes no TensorCore sort primitive.
This implementation therefore uses exact bitonic sorting networks. Local
score rows are split into independent blocks, each block emits its exact best
``top_k`` pairs, and a tree of pairwise networks reduces those candidates.
Every comparison is the semantic pair

``(score descending, global position ascending)``.

The implementation is deliberately default-off. The readable JAX functions
remain the fallback and oracle until the protected TPU/HLO/latency gate passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ....optimized.reference.dsa import SelectedPositions, local_topk_candidates, merge_topk_candidates

# Keep these as Python literals so a Pallas kernel does not capture JAX arrays.
_NEGATIVE_INFINITY = float("-inf")
_NO_POSITION = -1


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("power-of-two input must be positive")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True, slots=True)
class DsaTopKConfig:
    """Static TensorCore selector geometry."""

    selection_width: int = 2048
    local_block_size: int = 2048
    tile_size: int = 128

    def __post_init__(self) -> None:
        for name in ("selection_width", "local_block_size", "tile_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.tile_size != 128:
            raise ValueError("TPU-v4 top-k tiles must contain 128 elements")
        if self.local_block_size % self.tile_size:
            raise ValueError("local_block_size must be a multiple of 128")
        if self.local_block_size & (self.local_block_size - 1):
            raise ValueError("local_block_size must be a power of two")
        if self.local_block_size < self.padded_selection_width:
            raise ValueError("local_block_size must cover the padded top-k width")

    @property
    def padded_selection_width(self) -> int:
        return _next_power_of_two(max(self.selection_width, self.tile_size))


def bitonic_sort_pairs(
    scores: Any,
    positions: Any,
    *,
    width: int,
) -> tuple[Any, Any]:
    """Sort a power-of-two row by score descending then position ascending."""

    slots = lax.broadcasted_iota(jnp.int32, (1, 1, width), 2)
    merge_width = 2
    while merge_width <= width:
        distance = merge_width // 2
        while distance:
            upper_half = (slots & jnp.int32(distance)) != jnp.int32(0)
            partner_scores = jnp.where(
                upper_half,
                pltpu.roll(scores, distance, 2),
                pltpu.roll(scores, width - distance, 2),
            )
            partner_positions = jnp.where(
                upper_half,
                pltpu.roll(positions, distance, 2),
                pltpu.roll(positions, width - distance, 2),
            )
            self_better_or_equal = (
                (scores > partner_scores)
                | (
                    (scores == partner_scores)
                    & (positions <= partner_positions)
                )
            )
            descending = (
                slots & jnp.int32(merge_width)
            ) == jnp.int32(0)
            lower_half = ~upper_half
            take_better = lower_half == descending
            choose_self = take_better == self_better_or_equal
            scores = jnp.where(choose_self, scores, partner_scores)
            positions = jnp.where(
                choose_self, positions, partner_positions
            )
            distance //= 2
        merge_width *= 2
    return scores, positions


def _validate_local_inputs(
    local_scores: Any,
    global_positions: Any,
    valid_lengths: Any,
) -> int:
    if local_scores.ndim != 2 or local_scores.shape[0] != 1:
        raise ValueError("Pallas DSA top-k requires one exact score row")
    context = local_scores.shape[1]
    if context <= 0:
        raise ValueError("Pallas DSA top-k requires nonempty local context")
    if global_positions.shape != (context,):
        raise ValueError("global positions must match the local context")
    if valid_lengths.shape != (1,):
        raise ValueError("valid lengths must contain one exact row")
    if local_scores.dtype != jnp.float32:
        raise ValueError("Pallas DSA top-k scores must remain FP32")
    if global_positions.dtype != jnp.int32 or valid_lengths.dtype != jnp.int32:
        raise ValueError("Pallas DSA positions and valid lengths must be int32")
    return context


def _select_blocks_pallas(
    scores: Any,
    positions: Any,
    valid_lengths: Any,
    *,
    input_width: int,
    output_width: int,
    selection_width: int,
    programs: int,
    name: str,
    interpret: bool,
) -> tuple[Any, Any]:
    """Select exact score/position pairs independently in ``programs`` blocks."""

    if scores.shape != (programs, input_width):
        raise ValueError("selector score blocks disagree with the static geometry")
    if positions.shape != scores.shape:
        raise ValueError("selector score/position blocks must have identical shape")
    if output_width < selection_width:
        raise ValueError("physical selector output is narrower than top-k")
    if input_width & (input_width - 1):
        raise ValueError("bitonic selector input width must be a power of two")
    if output_width > input_width:
        raise ValueError("bitonic selector output cannot exceed its input")

    def kernel(
        score_ref: Any,
        position_ref: Any,
        valid_length_ref: Any,
        output_score_ref: Any,
        output_position_ref: Any,
    ) -> None:
        block_scores = score_ref[...]
        # TPU-v4 Mosaic cannot legalize masked selects between wide int32
        # position vectors. DSA positions are below 2**24, so FP32 preserves
        # every integer exactly through comparison/permutation; cast back only
        # at the kernel boundary.
        block_positions = position_ref[...].astype(jnp.float32)
        valid = (
            (block_positions >= jnp.float32(0.0))
            & (block_positions < valid_length_ref[0].astype(jnp.float32))
        )
        block_scores = jnp.where(
            valid, block_scores, _NEGATIVE_INFINITY
        )
        sorted_scores, sorted_positions = bitonic_sort_pairs(
            block_scores,
            block_positions,
            width=input_width,
        )
        output_score_ref[...] = sorted_scores[:, :, :output_width]
        output_position_ref[...] = jnp.where(
            sorted_scores[:, :, :output_width] == _NEGATIVE_INFINITY,
            jnp.float32(_NO_POSITION),
            sorted_positions[:, :, :output_width],
        ).astype(jnp.int32)

    def input_index(program: Any) -> tuple[Any, int, int]:
        return program, 0, 0

    def valid_length_index(program: Any) -> tuple[int]:
        del program
        return (0,)

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((programs, 1, output_width), jnp.float32),
            jax.ShapeDtypeStruct((programs, 1, output_width), jnp.int32),
        ),
        grid=(programs,),
        in_specs=(
            pl.BlockSpec((1, 1, input_width), input_index),
            pl.BlockSpec((1, 1, input_width), input_index),
            pl.BlockSpec((1,), valid_length_index),
        ),
        out_specs=(
            pl.BlockSpec((1, 1, output_width), input_index),
            pl.BlockSpec((1, 1, output_width), input_index),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=name,
        cost_estimate=pl.CostEstimate(
            flops=(
                programs
                * input_width
                * input_width.bit_length()
                * (input_width.bit_length() - 1)
                * 6
            ),
            bytes_accessed=(
                programs * input_width * 8
                + programs * output_width * 8
                + 4
            ),
            transcendentals=0,
        ),
    )
    output_scores, output_positions = call(
        scores[:, None, :], positions[:, None, :], valid_lengths
    )
    return output_scores[:, 0, :], output_positions[:, 0, :]


def _merge_candidate_tree_pallas(
    candidate_scores: Any,
    candidate_positions: Any,
    valid_lengths: Any,
    *,
    selection_width: int,
    output_width: int,
    name_prefix: str,
    interpret: bool,
) -> tuple[Any, Any]:
    """Reduce a power-of-two candidate stack to one exact ordered row."""

    groups, candidate_width = candidate_scores.shape
    if groups <= 0 or groups & (groups - 1):
        raise ValueError("candidate tree requires a positive power-of-two group count")
    if candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate score/position trees must have identical shape")
    level = 0
    while groups > 1:
        programs = groups // 2
        pair_scores = candidate_scores.reshape(programs, 2 * candidate_width)
        pair_positions = candidate_positions.reshape(programs, 2 * candidate_width)
        candidate_scores, candidate_positions = _select_blocks_pallas(
            pair_scores,
            pair_positions,
            valid_lengths,
            input_width=2 * candidate_width,
            output_width=output_width,
            selection_width=selection_width,
            programs=programs,
            name=(
                f"{name_prefix}_bitonic_merge_"
                f"l{level}_g{groups}_k{selection_width}"
            ),
            interpret=interpret,
        )
        groups = programs
        candidate_width = output_width
        level += 1
    return candidate_scores, candidate_positions


def local_topk_candidates_pallas(
    local_scores: Any,
    global_positions: Any,
    valid_lengths: Any,
    *,
    config: DsaTopKConfig = DsaTopKConfig(),
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Return exact one-row local candidates with lowest-position ties."""

    context = _validate_local_inputs(local_scores, global_positions, valid_lengths)
    block_count = _ceil_div(context, config.local_block_size)
    programs = _next_power_of_two(block_count)
    padded_context = programs * config.local_block_size
    score_padding = padded_context - context
    scores = jnp.pad(
        local_scores,
        ((0, 0), (0, score_padding)),
        constant_values=_NEGATIVE_INFINITY,
    ).reshape(programs, config.local_block_size)
    positions = jnp.pad(
        global_positions,
        ((0, score_padding),),
        constant_values=_NO_POSITION,
    ).reshape(programs, config.local_block_size)
    output_width = config.padded_selection_width
    candidate_scores, candidate_positions = _select_blocks_pallas(
        scores,
        positions,
        valid_lengths,
        input_width=config.local_block_size,
        output_width=output_width,
        selection_width=config.selection_width,
        programs=programs,
        name=(
            "greenfield_dsa_topk_local_bitonic_select_"
            f"n{padded_context}_k{config.selection_width}_"
            f"b{config.local_block_size}_g{programs}"
        ),
        interpret=interpret,
    )
    candidate_scores, candidate_positions = _merge_candidate_tree_pallas(
        candidate_scores,
        candidate_positions,
        valid_lengths,
        selection_width=config.selection_width,
        output_width=output_width,
        name_prefix="greenfield_dsa_topk_local",
        interpret=interpret,
    )
    return (
        candidate_scores[:, : config.selection_width],
        candidate_positions[:, : config.selection_width],
    )


def merge_topk_candidates_pallas(
    candidate_scores: Any,
    candidate_positions: Any,
    valid_lengths: Any,
    *,
    global_context_size: int,
    config: DsaTopKConfig = DsaTopKConfig(),
    interpret: bool = False,
) -> SelectedPositions:
    """Merge local candidate rows independent of collective concatenation order."""

    if candidate_scores.ndim != 3 or candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate scores/positions must share rank-three shape")
    groups, rows, candidates = candidate_scores.shape
    if groups <= 0 or rows != 1 or candidates <= 0:
        raise ValueError("Pallas candidate merge requires [groups,1,candidates]")
    if groups * candidates < config.selection_width:
        raise ValueError("candidate union is narrower than top-k")
    if candidate_scores.dtype != jnp.float32:
        raise ValueError("Pallas candidate scores must remain FP32")
    if candidate_positions.dtype != jnp.int32 or valid_lengths.dtype != jnp.int32:
        raise ValueError("Pallas candidate positions and lengths must be int32")
    if valid_lengths.shape != (1,):
        raise ValueError("valid lengths must contain one exact row")
    if not isinstance(global_context_size, int) or global_context_size < 0:
        raise ValueError("global_context_size must be a non-negative integer")

    input_width = _next_power_of_two(max(candidates, config.tile_size))
    output_width = config.padded_selection_width
    power_groups = _next_power_of_two(groups)
    score_padding = input_width - candidates
    scores = jnp.pad(
        candidate_scores[:, 0, :],
        ((0, power_groups - groups), (0, score_padding)),
        constant_values=_NEGATIVE_INFINITY,
    )
    positions = jnp.pad(
        candidate_positions[:, 0, :],
        ((0, power_groups - groups), (0, score_padding)),
        constant_values=_NO_POSITION,
    )
    if power_groups == 1:
        scores, positions = _select_blocks_pallas(
            scores,
            positions,
            valid_lengths,
            input_width=input_width,
            output_width=output_width,
            selection_width=config.selection_width,
            programs=1,
            name=(
                "greenfield_dsa_topk_global_bitonic_select_"
                f"n{input_width}_k{config.selection_width}"
            ),
            interpret=interpret,
        )
    else:
        scores, positions = _merge_candidate_tree_pallas(
            scores,
            positions,
            valid_lengths,
            selection_width=config.selection_width,
            output_width=output_width,
            name_prefix="greenfield_dsa_topk_global",
            interpret=interpret,
        )
    del scores
    selected = positions[:, : config.selection_width]
    valid_counts = jnp.clip(
        valid_lengths,
        jnp.int32(0),
        jnp.int32(min(config.selection_width, global_context_size)),
    )
    slots = lax.broadcasted_iota(jnp.int32, (1, config.selection_width), 1)
    selected = jnp.where(
        slots < valid_counts[:, None], selected, _NO_POSITION
    )
    return SelectedPositions(selected.astype(jnp.int32), valid_counts)


def local_topk_candidates_kernel(
    local_scores: Any,
    global_positions: Any,
    valid_lengths: Any,
    *,
    top_k: int,
    backend: Literal["reference", "pallas"] = "reference",
    config: DsaTopKConfig | None = None,
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Dispatch the default-off exact local selector or readable fallback."""

    if backend == "reference":
        return local_topk_candidates(
            local_scores, global_positions, valid_lengths, top_k=top_k
        )
    if backend == "pallas":
        selected_config = config or DsaTopKConfig(selection_width=top_k)
        if selected_config.selection_width != top_k:
            raise ValueError("top_k must match the Pallas selection width")
        return local_topk_candidates_pallas(
            local_scores,
            global_positions,
            valid_lengths,
            config=selected_config,
            interpret=interpret,
        )
    raise ValueError(f"unsupported DSA top-k backend {backend!r}")


def merge_topk_candidates_kernel(
    candidate_scores: Any,
    candidate_positions: Any,
    valid_lengths: Any,
    *,
    top_k: int,
    global_context_size: int,
    backend: Literal["reference", "pallas"] = "reference",
    config: DsaTopKConfig | None = None,
    interpret: bool = False,
) -> SelectedPositions:
    """Dispatch the default-off exact candidate merge or readable fallback."""

    if backend == "reference":
        return merge_topk_candidates(
            candidate_scores,
            candidate_positions,
            valid_lengths,
            top_k=top_k,
            global_context_size=global_context_size,
        )
    if backend == "pallas":
        selected_config = config or DsaTopKConfig(selection_width=top_k)
        if selected_config.selection_width != top_k:
            raise ValueError("top_k must match the Pallas selection width")
        return merge_topk_candidates_pallas(
            candidate_scores,
            candidate_positions,
            valid_lengths,
            global_context_size=global_context_size,
            config=selected_config,
            interpret=interpret,
        )
    raise ValueError(f"unsupported DSA top-k backend {backend!r}")
