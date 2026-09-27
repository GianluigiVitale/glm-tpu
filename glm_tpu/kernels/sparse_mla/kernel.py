"""Fused stage-local selected-KV gather and sparse MLA for TPU v4.

The readable fallback first materializes ``[1, top_k, cache_width]`` in HBM.
The Pallas path never creates that tensor. A small exact TensorCore network
orders one owner's selected positions, ordinary JAX resolves only compact
page-row metadata, and the attention kernel dynamically DMAs an aligned
eight-row TPU-v4 tile around each live cache row into VMEM. It masks the seven
overfetch lanes before immediately consuming the selected lane in online
softmax. Only the attended latent and additive LSE leave the kernel.

Moved verbatim at S2f out of the research package (its production definitions; the research
remainder, and the module these definitions came from, are at ``archive/research-20260922``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from glm_tpu.kernels.names import KERNEL_NAMES
from glm_tpu.layers.contracts import MlaNumericalContract, require_int32, require_shape
from glm_tpu.layers.attention.kv_cache import SelectedKvSegment


_FINITE_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


_TPU_V4_DMA_ROWS = 8


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


def pregathered_sparse_mla_pallas(
    query_nope_absorbed: Any,
    query_rope: Any,
    selected_cache: Any,
    valid_counts: Any,
    *,
    contract: MlaNumericalContract = MlaNumericalContract(),
    config: SparseMlaConfig = SparseMlaConfig(segment_block=512),
    interpret: bool = False,
    prefill: bool = False,
) -> Any:
    """Attend one live row over an already ordered, topology-local segment.

    This is the arithmetic half of the selected-row exchange.  It accepts no
    global cache or position table: callers must provide the
    exact ascending 2,048-row BF16 segment already resident on the local stage.  The
    finite mask, online update, BF16 probability boundary and final BF16 round mirror
    the accepted flash association while remaining independent of legacy execution.
    prefill=True enables up to 32 independent causal query rows in one call.
    Caller validates counts/ascending causal selection and sanitizes cache tails;
    a masked score alone cannot protect the value dot from nonfinite padding.
    """

    heads = contract.num_heads
    latent = contract.kv_lora_rank
    rope_width = contract.qk_rope_head_dim
    segment_width = contract.top_k
    cache_width = contract.packed_cache_width
    if type(prefill) is not bool:
        raise ValueError("pregathered prefill flag must be boolean")
    rows = query_nope_absorbed.shape[0] if query_nope_absorbed.ndim == 3 else 0
    if not (1 <= rows <= 32 if prefill else rows == 1):
        raise ValueError("pregathered rows require explicit prefill=True for1..32")
    if query_nope_absorbed.shape != (rows, heads, latent):
        raise ValueError("pregathered sparse-MLA absorbed query shape drifted")
    if query_rope.shape != (rows, heads, rope_width):
        raise ValueError("pregathered sparse-MLA RoPE query shape drifted")
    if selected_cache.shape != (rows, segment_width, cache_width):
        raise ValueError("pregathered sparse-MLA selected cache shape drifted")
    if valid_counts.shape != (rows,) or valid_counts.dtype != jnp.int32:
        raise ValueError("pregathered sparse-MLA valid counts must be int32[rows]")
    if (
        query_nope_absorbed.dtype != jnp.bfloat16
        or query_rope.dtype != jnp.bfloat16
        or selected_cache.dtype != jnp.bfloat16
    ):
        raise ValueError("pregathered sparse-MLA operands must be BF16")
    segment_block = min(config.segment_block, segment_width)
    if segment_width % segment_block:
        raise ValueError("pregathered sparse-MLA segment blocks must divide top_k")
    if not interpret and segment_block % 128:
        raise ValueError("compiled pregathered segment blocks must divide into 128")
    block_count = segment_width // segment_block
    padding = cache_width - latent - rope_width
    if padding < 0:
        raise ValueError("pregathered sparse-MLA cache truncates the packed query")
    packed_query = jnp.concatenate(
        (
            query_nope_absorbed,
            query_rope,
            jnp.zeros((rows, heads, padding), dtype=jnp.bfloat16),
        ),
        axis=-1,
    )
    blocked_cache = selected_cache.reshape(rows, block_count, segment_block, cache_width)

    def kernel(
        valid_count_ref: Any,
        query_ref: Any,
        cache_ref: Any,
        output_ref: Any,
        maximum_ref: Any,
        denominator_ref: Any,
        accumulator_ref: Any,
    ) -> None:
        block = pl.program_id(1)

        @pl.when(block == 0)
        def initialize() -> None:
            maximum_ref[...] = jnp.full(maximum_ref.shape, _FINITE_MASK_VALUE, jnp.float32)
            denominator_ref[...] = jnp.zeros(denominator_ref.shape, jnp.float32)
            accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)

        query = query_ref[0]
        cache = cache_ref[0, 0]
        slots = lax.broadcasted_iota(jnp.int32, (1, segment_block), 1) + block * segment_block
        scores = lax.dot_general(
            query,
            cache,
            dimension_numbers=(((1,), (1,)), ((), ())),
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        ) * jnp.float32(contract.softmax_scale)
        scores = jnp.where(
            slots < valid_count_ref[pl.program_id(0) if prefill else 0],
            scores,
            _FINITE_MASK_VALUE,
        ).astype(jnp.float32)
        block_maximum = jnp.max(scores, axis=1, keepdims=True)
        maximum = jnp.maximum(maximum_ref[...], block_maximum)
        correction = jnp.exp(maximum_ref[...] - maximum)
        probabilities = jnp.exp(scores - maximum)
        denominator = denominator_ref[...] * correction + jnp.sum(probabilities, axis=1, keepdims=True)
        partial = lax.dot_general(
            probabilities.astype(cache.dtype),
            cache[:, :latent],
            dimension_numbers=(((1,), (0,)), ((), ())),
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        accumulator = accumulator_ref[...] * correction + partial
        maximum_ref[...] = maximum
        denominator_ref[...] = denominator
        accumulator_ref[...] = accumulator

        @pl.when(block == block_count - 1)
        def finalize() -> None:
            maximum_final = jnp.maximum(maximum, _FINITE_MASK_VALUE)
            final_correction = jnp.exp(maximum - maximum_final)
            denominator_final = denominator * final_correction + jnp.exp(_FINITE_MASK_VALUE - maximum_final)
            normalized = accumulator * final_correction / denominator_final
            output_ref[0] = jnp.where(maximum > _FINITE_MASK_VALUE, normalized, 0.0).astype(jnp.bfloat16)

    def query_map(row: Any, block: Any, valid_count: Any) -> tuple[Any, int, int]:
        del block, valid_count
        return row, 0, 0

    def cache_map(row: Any, block: Any, valid_count: Any) -> tuple[Any, Any, int, int]:
        del valid_count
        return row, block, 0, 0

    def output_map(row: Any, block: Any, valid_count: Any) -> tuple[Any, int, int]:
        del block, valid_count
        return row, 0, 0

    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(rows, block_count),
            in_specs=(
                pl.BlockSpec((1, heads, cache_width), query_map),
                pl.BlockSpec((1, 1, segment_block, cache_width), cache_map),
            ),
            out_specs=pl.BlockSpec((1, heads, latent), output_map),
            scratch_shapes=(
                pltpu.VMEM((heads, 1), jnp.float32),
                pltpu.VMEM((heads, 1), jnp.float32),
                pltpu.VMEM((heads, latent), jnp.float32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((rows, heads, latent), query_nope_absorbed.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel" if prefill else "arbitrary", "arbitrary"),
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        interpret=interpret,
        name=(
            f"{KERNEL_NAMES['sparse_mla']}_"
            f"h{heads}_k{segment_width}_b{segment_block}_w{cache_width}" + (f"_prefill_m{rows}" if prefill else "")
        ),
    )
    return call(valid_counts, packed_query, blocked_cache)


class SparseAttentionResult(NamedTuple):
    """Attended latent, additive LSE, and the propagated health predicate."""

    output: jax.Array
    logsumexp: jax.Array
    contract_valid: jax.Array


def sparse_mla_attention(
    query_nope_absorbed: jax.Array,
    query_rope: jax.Array,
    segment: SelectedKvSegment,
    *,
    contract: MlaNumericalContract = MlaNumericalContract(),
) -> SparseAttentionResult:
    """FP32 softmax attention over only the gathered selected latent rows."""

    if query_nope_absorbed.ndim != 3 or query_rope.ndim != 3:
        raise ValueError("sparse MLA queries must have rank three")
    rows = query_nope_absorbed.shape[0]
    require_shape(
        "query_nope_absorbed",
        query_nope_absorbed,
        (rows, contract.num_heads, contract.kv_lora_rank),
    )
    require_shape(
        "query_rope",
        query_rope,
        (rows, contract.num_heads, contract.qk_rope_head_dim),
    )
    if query_nope_absorbed.dtype != query_rope.dtype:
        raise ValueError("sparse MLA query components must share one dtype")
    if query_nope_absorbed.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError("sparse MLA queries must be BF16 or FP32")
    if segment.positions.ndim != 2:
        raise ValueError("selected KV positions must have shape [rows,top_k]")
    width = segment.positions.shape[1]
    if width != contract.top_k:
        raise ValueError(f"selected KV width must equal contract top_k={contract.top_k}, got {width}")
    require_shape(
        "selected KV segment",
        segment.values,
        (rows, width, contract.packed_cache_width),
    )
    require_shape("segment valid_counts", segment.valid_counts, (rows,))
    require_shape("segment contract_valid", segment.contract_valid, (rows,))
    require_int32("segment positions", segment.positions)
    require_int32("segment valid_counts", segment.valid_counts)
    if segment.contract_valid.dtype != jnp.bool_:
        raise ValueError("segment contract_valid must have dtype bool")
    if segment.values.dtype != query_nope_absorbed.dtype:
        raise ValueError("queries and selected KV segment must share the cache dtype")

    safe_counts = jnp.clip(segment.valid_counts, jnp.int32(0), jnp.int32(width))
    position_slots = jnp.arange(width, dtype=jnp.int32)[None, :]
    position_live = position_slots < safe_counts[:, None]
    positions_well_formed = jnp.all(
        jnp.where(
            position_live,
            segment.positions >= 0,
            segment.positions == -1,
        ),
        axis=1,
    )
    positions_ascending = jnp.all(
        jnp.where(
            position_slots[:, 1:] < safe_counts[:, None],
            segment.positions[:, 1:] > segment.positions[:, :-1],
            True,
        ),
        axis=1,
    )
    metadata_valid = (
        (segment.valid_counts >= 0) & (segment.valid_counts <= width) & positions_well_formed & positions_ascending
    )

    precision = lax.Precision.HIGHEST if query_nope_absorbed.dtype == jnp.float32 else lax.Precision.DEFAULT
    q_nope = query_nope_absorbed
    q_rope = query_rope
    kv_latent = segment.values[..., : contract.kv_lora_rank]
    kv_rope = segment.values[
        ...,
        contract.kv_lora_rank : contract.kv_lora_rank + contract.qk_rope_head_dim,
    ]
    scores = (
        jnp.einsum(
            "rhd,rkd->rhk",
            q_nope,
            kv_latent,
            preferred_element_type=jnp.float32,
            precision=precision,
        )
        + jnp.einsum(
            "rhd,rkd->rhk",
            q_rope,
            kv_rope,
            preferred_element_type=jnp.float32,
            precision=precision,
        )
    ) * jnp.float32(contract.softmax_scale)
    slots = jnp.arange(width, dtype=jnp.int32)[None, None, :]
    live = slots < safe_counts[:, None, None]
    masked_scores = jnp.where(live, scores, -jnp.inf)
    has_live = safe_counts > 0
    maximum = jnp.max(masked_scores, axis=-1, keepdims=True)
    safe_maximum = jnp.where(has_live[:, None, None], maximum, 0.0)
    unnormalized = jnp.where(live, jnp.exp(masked_scores - safe_maximum), 0.0)
    denominator = jnp.sum(unnormalized, axis=-1, keepdims=True)
    safe_denominator = jnp.where(has_live[:, None, None], denominator, 1.0)
    weighted_values = jnp.einsum(
        "rhk,rkd->rhd",
        unnormalized.astype(segment.values.dtype),
        segment.values[..., : contract.kv_lora_rank],
        preferred_element_type=jnp.float32,
        precision=precision,
    )
    output = weighted_values / safe_denominator
    output = jnp.where(has_live[:, None, None], output, 0.0)
    logsumexp = jnp.where(
        has_live[:, None],
        safe_maximum[..., 0] + jnp.log(safe_denominator[..., 0]),
        -jnp.inf,
    )
    return SparseAttentionResult(
        output.astype(query_nope_absorbed.dtype),
        logsumexp.astype(jnp.float32),
        segment.contract_valid & metadata_valid,
    )
