"""Bounded layer-0 DSA arithmetic used to diagnose the sealed 8K drift.

This module does not import or execute the legacy engine.  It independently
reconstructs the exact source-level arithmetic and static shapes used by the
sealed oracle so individual associations can be changed one at a time.  The
32-row functions are diagnostic-only; production ``decode_batch1`` remains a
true one-row executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from .rmsnorm import rms_norm
from .rotary import apply_rotary, rotary_cos_sin


KeyNormMode = Literal["divide_sqrt", "multiply_rsqrt"]
QaProjectionMode = Literal[
    "separate_q_a",
    "legacy_fused_qkv_a",
    "legacy_runtime_fused_qkv_a_global",
    "legacy_runtime_fused_qkv_a_sharded",
]
VirtualQaProjectionMode = Literal[
    "lax_map",
    "vmap",
    "unrolled",
    "lax_map_convolution",
]
VirtualQaNormMode = Literal[
    "logical_mean",
    "shard_sum",
    "left_fold",
    "topology_tree",
]


@dataclass(frozen=True, slots=True)
class Layer0DsaProbeGeometry:
    """Static geometry of the paired protected 8K layer-0 event."""

    prompt_tokens: int = 8155
    prompt_chunk: int = 2048
    decode_rows: int = 32
    hidden_size: int = 6144
    q_lora_rank: int = 2048
    qkv_a_companion_rank: int = 576
    legacy_tensor_shards: int = 32
    heads: int = 32
    head_dim: int = 128
    rotary_dim: int = 64
    theta: float = 8_000_000.0
    rms_norm_epsilon: float = 1e-5
    # The accepted GLM-5.2 config pins rms_norm_eps=1e-5 and vLLM passes
    # that value to q_a_layernorm.  This bounded diagnostic geometry must
    # reproduce the model value; key_norm below is a distinct LayerNorm
    # whose source contract remains 1e-6.
    q_norm_epsilon: float = 1e-5
    key_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        integer_fields = (
            "prompt_tokens",
            "prompt_chunk",
            "decode_rows",
            "hidden_size",
            "q_lora_rank",
            "qkv_a_companion_rank",
            "legacy_tensor_shards",
            "heads",
            "head_dim",
            "rotary_dim",
        )
        if any(
            not isinstance(getattr(self, name), int)
            or isinstance(getattr(self, name), bool)
            or getattr(self, name) <= 0
            for name in integer_fields
        ):
            raise ValueError("layer-0 DSA geometry dimensions must be positive")
        if self.prompt_tokens % self.prompt_chunk == 0:
            raise ValueError("the protected prompt must exercise a padded tail chunk")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("layer-0 DSA rotary geometry drifted")


@dataclass(frozen=True, slots=True)
class LegacyScoreGeometry:
    """Static legacy page/DCP geometry for an isolated score-row probe."""

    dcp_size: int = 8
    local_page_size: int = 512
    local_score_width: int = 43_008

    def __post_init__(self) -> None:
        for value in (
            self.dcp_size,
            self.local_page_size,
            self.local_score_width,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("legacy scorer geometry must be positive")
        if self.local_score_width % self.local_page_size:
            raise ValueError("local score width must contain complete pages")


@dataclass(frozen=True, slots=True)
class LegacyDcpXlaScoreGeometry:
    """Exact static local scorer geometry of the sealed 8K oracle.

    The legacy cache uses a 4,096-token logical block split over DCP8, so
    each physical scorer sees 512 keys per block-table entry.  At the sealed
    ``max_model_len=8704`` the owned-width gate retains exactly three block
    entries.  ``cache_pages=24`` is the protected run's explicit cache-block
    override.  This geometry is diagnostic-only; greenfield decode remains
    one row and never adopts the legacy batch bucket.
    """

    dcp_size: int = 8
    local_page_size: int = 512
    max_model_len: int = 8704
    cache_pages: int = 24
    decode_rows: int = 32
    heads: int = 32
    head_dim: int = 128

    def __post_init__(self) -> None:
        values = (
            self.dcp_size,
            self.local_page_size,
            self.max_model_len,
            self.cache_pages,
            self.decode_rows,
            self.heads,
            self.head_dim,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in values
        ):
            raise ValueError("legacy DCP XLA scorer geometry must be positive")
        if self.cache_pages < self.owned_block_count:
            raise ValueError("legacy DCP XLA cache cannot hold its owned blocks")

    @property
    def global_page_size(self) -> int:
        return self.dcp_size * self.local_page_size

    @property
    def owned_block_count(self) -> int:
        return (
            self.max_model_len + self.global_page_size - 1
        ) // self.global_page_size

    @property
    def local_score_width(self) -> int:
        return self.owned_block_count * self.local_page_size


class Layer0DsaState(NamedTuple):
    """Live fused-projection output plus DSA state for one event."""

    query: Any
    index_keys: Any
    head_weights: Any
    qkv_a_companion: Any


class Layer0DsaScorerInternals(NamedTuple):
    """Actual source-level tensors entering the layer-0 DSA scorer."""

    query: Any
    head_weights: Any
    current_keys: Any


class LegacyFusedQkvRuntimeWeights(NamedTuple):
    """Raw-FP8 runtime layouts produced by the sealed TP32 loader."""

    global_weight: Any
    global_scale: Any
    sharded_weight: Any
    sharded_scale: Any


class LegacyTp32QaNormOutput(NamedTuple):
    """Replicated q-a norm result plus the still-sharded fused companion."""

    q_residual: Any
    qkv_a_companion_shard: Any


class LegacyVirtualQaNormOutput(NamedTuple):
    """One-row q-a norm plus the reconstructed fused companion."""

    q_residual: Any
    qkv_a_companion: Any


class LegacyLocalDcpScoreInputs(NamedTuple):
    """One DCP shard's cache and metadata for the exact local XLA scorer."""

    cache: Any
    block_tables: Any
    kv_lens: Any
    local_positions: Any


def bfloat16_from_uint16_bits(value: Any) -> Any:
    """Reinterpret portable little-endian BF16 payload bits on device."""

    if value.dtype != jnp.uint16:
        raise ValueError("BF16 artifact payload must use uint16 bits")
    return lax.bitcast_convert_type(value, jnp.bfloat16)


def pack_legacy_fused_qkv_runtime_weights(
    q_a_weight_bits: Any,
    q_a_scale: Any,
    kv_a_weight_bits: Any,
    kv_a_scale: Any,
    *,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
    quant_block: int = 128,
) -> LegacyFusedQkvRuntimeWeights:
    """Reconstruct the sealed loader's raw-FP8 fused-output packing.

    The legacy layer is declared ``disable_tp=True``, but the TPU linear
    adapter classifies its ``MergedColumnParallelLinear`` by type and shards
    the fused output over all 32 ``ATTN_HEAD`` shards.  Loading therefore
    reorders each fused part independently: every shard owns 64 q-a columns
    followed by 18 kv-a columns.  The 2-D block scales retain their 48-way
    contracting axis and expand only the output axis before the same reorder.

    Both the global reordered tensors and the exact per-shard tensors remain
    live so a bounded probe can distinguish a global-width dot from the real
    local ``N=82`` dot without importing the legacy execution path.
    """

    if not isinstance(quant_block, int) or isinstance(quant_block, bool) or (
        quant_block <= 0
    ):
        raise ValueError("legacy FP8 quant block must be a positive integer")
    expected = {
        "q_a_weight_bits": (geometry.q_lora_rank, geometry.hidden_size),
        "q_a_scale": (
            (geometry.q_lora_rank + quant_block - 1) // quant_block,
            (geometry.hidden_size + quant_block - 1) // quant_block,
        ),
        "kv_a_weight_bits": (
            geometry.qkv_a_companion_rank,
            geometry.hidden_size,
        ),
        "kv_a_scale": (
            (geometry.qkv_a_companion_rank + quant_block - 1) // quant_block,
            (geometry.hidden_size + quant_block - 1) // quant_block,
        ),
    }
    values = {
        "q_a_weight_bits": q_a_weight_bits,
        "q_a_scale": q_a_scale,
        "kv_a_weight_bits": kv_a_weight_bits,
        "kv_a_scale": kv_a_scale,
    }
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise ValueError(
                f"legacy fused qkv {name} shape drifted: "
                f"expected={shape} found={values[name].shape}"
            )
    if q_a_weight_bits.dtype != jnp.uint8 or (
        kv_a_weight_bits.dtype != jnp.uint8
    ):
        raise ValueError("legacy fused qkv source weights must be raw uint8 bits")
    if q_a_scale.dtype != jnp.float32 or kv_a_scale.dtype != jnp.float32:
        raise ValueError("legacy fused qkv source scales must be FP32")

    shards = geometry.legacy_tensor_shards
    if geometry.q_lora_rank % shards or (
        geometry.qkv_a_companion_rank % shards
    ):
        raise ValueError("legacy fused qkv outputs must divide the tensor shards")
    q_per_shard = geometry.q_lora_rank // shards
    kv_per_shard = geometry.qkv_a_companion_rank // shards
    q_fp8 = lax.bitcast_convert_type(q_a_weight_bits, jnp.float8_e4m3fn)
    kv_fp8 = lax.bitcast_convert_type(kv_a_weight_bits, jnp.float8_e4m3fn)
    q_sharded = q_fp8.reshape(
        shards, q_per_shard, geometry.hidden_size
    ).transpose(0, 2, 1)
    kv_sharded = kv_fp8.reshape(
        shards, kv_per_shard, geometry.hidden_size
    ).transpose(0, 2, 1)
    sharded_weight = jnp.concatenate((q_sharded, kv_sharded), axis=-1)

    def expanded_output_scale(scale: Any, output_size: int) -> Any:
        return jnp.repeat(scale, quant_block, axis=0)[:output_size]

    q_scale_full = expanded_output_scale(q_a_scale, geometry.q_lora_rank)
    kv_scale_full = expanded_output_scale(
        kv_a_scale, geometry.qkv_a_companion_rank
    )
    q_scale_sharded = q_scale_full.reshape(
        shards, q_per_shard, -1
    ).transpose(0, 2, 1)
    kv_scale_sharded = kv_scale_full.reshape(
        shards, kv_per_shard, -1
    ).transpose(0, 2, 1)
    sharded_scale = jnp.concatenate(
        (q_scale_sharded, kv_scale_sharded), axis=-1
    )

    local_output = q_per_shard + kv_per_shard
    with jax.named_scope("legacy_fused_qkv_runtime_pack_tp32_n82"):
        global_weight = sharded_weight.transpose(1, 0, 2).reshape(
            geometry.hidden_size, shards * local_output
        )
        global_scale = sharded_scale.transpose(1, 0, 2).reshape(
            sharded_scale.shape[1], shards * local_output
        )
    return LegacyFusedQkvRuntimeWeights(
        global_weight=global_weight,
        global_scale=global_scale,
        sharded_weight=sharded_weight,
        sharded_scale=sharded_scale,
    )


def _legacy_runtime_fp8_dot(lhs: Any, weight: Any, scale: Any) -> Any:
    """Independently reproduce the disabled-requant XLA matmul body."""

    if lhs.ndim != 2 or weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("legacy runtime FP8 dot requires rank-2 operands")
    if weight.shape[0] != lhs.shape[1] or scale.shape[1] != weight.shape[1]:
        raise ValueError("legacy runtime FP8 dot dimensions drifted")
    if weight.shape[0] % scale.shape[0]:
        raise ValueError("legacy runtime FP8 contracting scale is ragged")
    if weight.dtype != jnp.float8_e4m3fn or scale.dtype != jnp.float32:
        raise ValueError("legacy runtime FP8 dot dtype contract drifted")
    block = weight.shape[0] // scale.shape[0]
    expanded_scale = jnp.repeat(scale, block, axis=0)[: weight.shape[0]]
    decoded = (
        weight.astype(jnp.float32) * expanded_scale.astype(jnp.float32)
    ).astype(lhs.dtype)
    output = lax.dot_general(
        lhs,
        decoded,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return output.astype(lhs.dtype)


def _legacy_runtime_fp8_convolution(
    lhs: Any, weight: Any, scale: Any
) -> Any:
    """Keep the accepted zero-spatial convolution primitive for one row."""

    if lhs.ndim != 2 or weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("legacy FP8 convolution requires rank-2 operands")
    if weight.shape[0] != lhs.shape[1] or scale.shape[1] != weight.shape[1]:
        raise ValueError("legacy FP8 convolution dimensions drifted")
    if weight.shape[0] % scale.shape[0]:
        raise ValueError("legacy FP8 convolution scale is ragged")
    if weight.dtype != jnp.float8_e4m3fn or scale.dtype != jnp.float32:
        raise ValueError("legacy FP8 convolution dtype contract drifted")
    block = weight.shape[0] // scale.shape[0]
    expanded_scale = jnp.repeat(scale, block, axis=0)[: weight.shape[0]]
    decoded = (
        weight.astype(jnp.float32) * expanded_scale.astype(jnp.float32)
    ).astype(lhs.dtype)
    output = lax.conv_general_dilated(
        lhs,
        decoded,
        window_strides=(),
        padding=(),
        dimension_numbers=("NC", "IO", "NC"),
        preferred_element_type=jnp.float32,
    )
    return output.astype(lhs.dtype)


def one_row_virtual_tp32_fused_qkv_a_rms_norm(
    normalized_hidden: Any,
    sharded_qkv_weight: Any,
    sharded_qkv_scale: Any,
    q_a_norm_weight: Any,
    *,
    projection_mode: VirtualQaProjectionMode,
    norm_mode: VirtualQaNormMode,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
) -> LegacyVirtualQaNormOutput:
    """Emulate the accepted shard-major q-a association with one live row.

    The accepted TP32 loader forms 32 independent fused output shards, each
    containing 64 q-a columns followed by 18 kv-a companion columns.  This
    function preserves that physical ``N=82`` dot boundary while virtualizing
    the 32 shards on one stage-local device.  It has no collective and never
    creates the legacy 32-token decode bucket.

    The projection and norm association are explicit discriminator axes.  No
    mode is a production default until a protected real layer-0 capture proves
    its BF16 q-a state bitwise and its optimized HLO passes the local-only,
    one-row contract.
    """

    shards = geometry.legacy_tensor_shards
    if geometry.q_lora_rank % shards or (
        geometry.qkv_a_companion_rank % shards
    ):
        raise ValueError("virtual TP32 fused outputs must divide the shards")
    q_local = geometry.q_lora_rank // shards
    companion_local = geometry.qkv_a_companion_rank // shards
    local_output = q_local + companion_local
    expected_shapes = {
        "normalized_hidden": (1, geometry.hidden_size),
        "sharded_qkv_weight": (
            shards,
            geometry.hidden_size,
            local_output,
        ),
        "sharded_qkv_scale": (
            shards,
            geometry.hidden_size // 128,
            local_output,
        ),
        "q_a_norm_weight": (geometry.q_lora_rank,),
    }
    values = {
        "normalized_hidden": normalized_hidden,
        "sharded_qkv_weight": sharded_qkv_weight,
        "sharded_qkv_scale": sharded_qkv_scale,
        "q_a_norm_weight": q_a_norm_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"one-row virtual q-a {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if normalized_hidden.dtype != jnp.bfloat16 or (
        sharded_qkv_weight.dtype != jnp.float8_e4m3fn
    ) or sharded_qkv_scale.dtype != jnp.float32 or (
        q_a_norm_weight.dtype != jnp.bfloat16
    ):
        raise ValueError("one-row virtual q-a dtype contract drifted")
    if projection_mode not in (
        "lax_map",
        "vmap",
        "unrolled",
        "lax_map_convolution",
    ):
        raise ValueError("one-row virtual q-a projection mode is unknown")
    if norm_mode not in (
        "logical_mean",
        "shard_sum",
        "left_fold",
        "topology_tree",
    ):
        raise ValueError("one-row virtual q-a norm mode is unknown")
    if norm_mode == "topology_tree" and shards != 32:
        raise ValueError("topology-tree q-a norm requires 32 virtual shards")

    def project(weight_scale: tuple[Any, Any]) -> Any:
        weight, scale = weight_scale
        projection = (
            _legacy_runtime_fp8_convolution
            if projection_mode == "lax_map_convolution"
            else _legacy_runtime_fp8_dot
        )
        return projection(
            normalized_hidden,
            weight,
            scale,
        )

    with jax.named_scope(
        f"one_row_virtual_tp32_qkv_n{local_output}_{projection_mode}"
    ):
        operands = (sharded_qkv_weight, sharded_qkv_scale)
        if projection_mode in ("lax_map", "lax_map_convolution"):
            projected = lax.map(project, operands)
        elif projection_mode == "vmap":
            projected = jax.vmap(project)(operands)
        else:
            projected = jnp.stack(
                tuple(
                    project(
                        (
                            sharded_qkv_weight[index],
                            sharded_qkv_scale[index],
                        )
                    )
                    for index in range(shards)
                )
            )

    local_q = projected[:, :, :q_local]
    companion = jnp.transpose(
        projected[:, :, q_local:], (1, 0, 2)
    ).reshape(1, geometry.qkv_a_companion_rank)
    logical_q = jnp.transpose(local_q, (1, 0, 2)).reshape(
        1, geometry.q_lora_rank
    )
    if norm_mode == "logical_mean":
        q_residual = rms_norm(
            logical_q,
            q_a_norm_weight,
            epsilon=geometry.q_norm_epsilon,
        )
    else:
        local_square_sum = jnp.sum(
            jnp.square(local_q.astype(jnp.float32)), axis=-1
        )
        if norm_mode == "shard_sum":
            global_square_sum = jnp.sum(local_square_sum, axis=0)
        elif norm_mode == "left_fold":
            global_square_sum = jnp.zeros_like(local_square_sum[0])
            for index in range(shards):
                global_square_sum = (
                    global_square_sum + local_square_sum[index]
                )
        else:
            topology = local_square_sum.reshape(2, 4, 4, 1)
            global_square_sum = jnp.sum(
                jnp.sum(jnp.sum(topology, axis=2), axis=1), axis=0
            )
        inverse_rms = lax.rsqrt(
            global_square_sum / jnp.float32(geometry.q_lora_rank)
            + jnp.float32(geometry.q_norm_epsilon)
        )
        local_normalized = (
            local_q.astype(jnp.float32) * inverse_rms[None, :, None]
        ).astype(jnp.bfloat16)
        full_normalized = jnp.transpose(
            local_normalized, (1, 0, 2)
        ).reshape(1, geometry.q_lora_rank)
        q_residual = (
            full_normalized * q_a_norm_weight
        ).astype(jnp.bfloat16)
    return LegacyVirtualQaNormOutput(q_residual, companion)


def legacy_tp32_sharded_rms_norm(
    local_value: Any,
    weight: Any,
    *,
    axis_name: str,
    tensor_shards: int,
    logical_width: int,
    epsilon: float,
) -> Any:
    """Reproduce the sealed feature-sharded RMSNorm association.

    The legacy fused q-a projection leaves 64 of 2,048 q-a columns on each
    member of the 32-way ``model`` group.  Torchax lowers the logical mean to
    a local FP32 reduce followed by one 32-way all-reduce.  The normalized
    BF16 shards are then all-gathered before multiplication by the replicated
    BF16 RMSNorm weight.  The accepted legacy XPlane attributes exactly those
    two collectives to ``q_a_layernorm``.

    This function is diagnostic-only and must be called inside a ``shard_map``
    that binds ``axis_name``.  It deliberately uses the observed rank-3
    all-gather result ``[rows, local_width, tensor_shards]`` before restoring
    shard-major logical feature order.
    """

    if not isinstance(axis_name, str) or not axis_name:
        raise ValueError("legacy TP32 RMSNorm axis name must be non-empty")
    for name, value in (
        ("tensor_shards", tensor_shards),
        ("logical_width", logical_width),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"legacy TP32 RMSNorm {name} must be positive")
    if local_value.ndim != 2 or weight.shape != (logical_width,):
        raise ValueError("legacy TP32 RMSNorm shapes drifted")
    if local_value.shape[-1] * tensor_shards != logical_width:
        raise ValueError("legacy TP32 RMSNorm local width does not reconstruct")
    if local_value.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        raise ValueError("legacy TP32 RMSNorm values and weight must be BF16")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or (
        epsilon <= 0
    ):
        raise ValueError("legacy TP32 RMSNorm epsilon must be positive")

    value_f32 = local_value.astype(jnp.float32)
    with jax.named_scope("legacy_tp32_q_a_rms_norm_variance_psum"):
        local_square_sum = jnp.sum(jnp.square(value_f32), axis=-1)
        global_square_sum = lax.psum(local_square_sum, axis_name)
        inverse_rms = lax.rsqrt(
            global_square_sum / jnp.float32(logical_width)
            + jnp.float32(epsilon)
        )
    local_normalized = (value_f32 * inverse_rms[:, None]).astype(
        jnp.bfloat16
    )
    with jax.named_scope("legacy_tp32_q_a_rms_norm_bf16_all_gather"):
        gathered = lax.all_gather(
            local_normalized,
            axis_name,
            axis=local_normalized.ndim,
            tiled=False,
        )
        full_normalized = gathered.transpose(0, 2, 1).reshape(
            local_value.shape[0], logical_width
        )
    return (full_normalized * weight).astype(local_value.dtype)


def legacy_tp32_fused_qkv_a_rms_norm(
    normalized_hidden: Any,
    local_qkv_weight: Any,
    local_qkv_scale: Any,
    q_a_norm_weight: Any,
    *,
    axis_name: str,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
) -> LegacyTp32QaNormOutput:
    """Run the real local-N82 projection and distributed q-a RMSNorm.

    ``local_qkv_weight`` and ``local_qkv_scale`` are one physical shard of the
    packed tensors returned by :func:`pack_legacy_fused_qkv_runtime_weights`.
    The q-a result becomes replicated only at the exact legacy norm boundary;
    the 18-column kv-a companion remains sharded and live.
    """

    shards = geometry.legacy_tensor_shards
    if geometry.hidden_size % 128 or geometry.q_lora_rank % shards or (
        geometry.qkv_a_companion_rank % shards
    ):
        raise ValueError("legacy TP32 fused qkv geometry does not divide")
    q_local = geometry.q_lora_rank // shards
    companion_local = geometry.qkv_a_companion_rank // shards
    local_output = q_local + companion_local
    expected_shapes = {
        "normalized_hidden": (geometry.decode_rows, geometry.hidden_size),
        "local_qkv_weight": (geometry.hidden_size, local_output),
        "local_qkv_scale": (geometry.hidden_size // 128, local_output),
        "q_a_norm_weight": (geometry.q_lora_rank,),
    }
    values = {
        "normalized_hidden": normalized_hidden,
        "local_qkv_weight": local_qkv_weight,
        "local_qkv_scale": local_qkv_scale,
        "q_a_norm_weight": q_a_norm_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"legacy TP32 fused qkv {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if normalized_hidden.dtype != jnp.bfloat16 or (
        local_qkv_weight.dtype != jnp.float8_e4m3fn
    ) or local_qkv_scale.dtype != jnp.float32 or (
        q_a_norm_weight.dtype != jnp.bfloat16
    ):
        raise ValueError("legacy TP32 fused qkv dtype contract drifted")

    with jax.named_scope(
        "legacy_runtime_fused_qkv_a_m32_tp32_distributed_norm"
    ):
        local_qkv_a = _legacy_runtime_fp8_dot(
            normalized_hidden,
            local_qkv_weight,
            local_qkv_scale,
        )
        local_q_a = local_qkv_a[:, :q_local]
        companion = local_qkv_a[:, q_local:]
        q_residual = legacy_tp32_sharded_rms_norm(
            local_q_a,
            q_a_norm_weight,
            axis_name=axis_name,
            tensor_shards=shards,
            logical_width=geometry.q_lora_rank,
            epsilon=geometry.q_norm_epsilon,
        )
    return LegacyTp32QaNormOutput(q_residual, companion)


def legacy_tp32_gspmd_fused_qkv_a_rms_norm(
    normalized_hidden: Any,
    global_qkv_weight: Any,
    global_qkv_scale: Any,
    q_a_norm_weight: Any,
    *,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
) -> LegacyTp32QaNormOutput:
    """Reproduce the source-level fused projection and RMSNorm for GSPMD.

    The accepted Torchax path expresses q-a RMSNorm as a logical-width
    ``mean`` and lets GSPMD partition the feature-sharded tensor.  This
    diagnostic tests whether that source form changes reduction association
    relative to manually spelling ``local sum -> psum -> divide`` inside
    :func:`jax.shard_map`.  It keeps the same shard-major fused ``N=82``
    runtime layout, but presents the complete logical operation to ``jax.jit``
    so the partitioner chooses the collective and division placement.

    Callers must provide explicit global ``NamedSharding`` values: hidden and
    norm weight replicated, fused weight/scale sharded over their output axis,
    q-a output replicated, and companion output-sharded.  The function has no
    explicit collective and is never a production greenfield layer.
    """

    shards = geometry.legacy_tensor_shards
    if geometry.q_lora_rank % shards or (
        geometry.qkv_a_companion_rank % shards
    ) or geometry.hidden_size % 128:
        raise ValueError("legacy TP32 GSPMD fused qkv geometry does not divide")
    q_local = geometry.q_lora_rank // shards
    companion_local = geometry.qkv_a_companion_rank // shards
    local_output = q_local + companion_local
    fused_output = geometry.q_lora_rank + geometry.qkv_a_companion_rank
    expected_shapes = {
        "normalized_hidden": (geometry.decode_rows, geometry.hidden_size),
        "global_qkv_weight": (geometry.hidden_size, fused_output),
        "global_qkv_scale": (geometry.hidden_size // 128, fused_output),
        "q_a_norm_weight": (geometry.q_lora_rank,),
    }
    values = {
        "normalized_hidden": normalized_hidden,
        "global_qkv_weight": global_qkv_weight,
        "global_qkv_scale": global_qkv_scale,
        "q_a_norm_weight": q_a_norm_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"legacy TP32 GSPMD fused qkv {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if normalized_hidden.dtype != jnp.bfloat16 or (
        global_qkv_weight.dtype != jnp.float8_e4m3fn
    ) or global_qkv_scale.dtype != jnp.float32 or (
        q_a_norm_weight.dtype != jnp.bfloat16
    ):
        raise ValueError("legacy TP32 GSPMD fused qkv dtype contract drifted")

    with jax.named_scope("legacy_runtime_fused_qkv_a_m32_tp32_gspmd_norm"):
        fused = _legacy_runtime_fp8_dot(
            normalized_hidden,
            global_qkv_weight,
            global_qkv_scale,
        ).reshape(geometry.decode_rows, shards, local_output)
        q_a = fused[:, :, :q_local].reshape(
            geometry.decode_rows, geometry.q_lora_rank
        )
        companion = fused[:, :, q_local:].reshape(
            geometry.decode_rows, geometry.qkv_a_companion_rank
        )
        with jax.named_scope("legacy_tp32_q_a_rms_norm_source_mean"):
            q_residual = rms_norm(
                q_a,
                q_a_norm_weight,
                epsilon=geometry.q_norm_epsilon,
            )
    return LegacyTp32QaNormOutput(q_residual, companion)


def layer0_dsa_state_from_q_residual(
    q_residual: Any,
    index_keys: Any,
    head_weights: Any,
    qkv_a_companion: Any,
    wq_b_weight: Any,
    *,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
) -> Layer0DsaState:
    """Replace only the q-a norm association in an otherwise sealed state.

    Prompt keys and head weights are independent of q-a.  Accepting them as
    explicit inputs lets a bounded multi-host diagnostic execute the physical
    distributed norm once and reuse the already-proven one-device state for
    every unaffected quantity.
    """

    expected_shapes = {
        "q_residual": (geometry.decode_rows, geometry.q_lora_rank),
        "index_keys": (geometry.prompt_tokens + 1, geometry.head_dim),
        "head_weights": (geometry.decode_rows, geometry.heads),
        "qkv_a_companion": (
            geometry.decode_rows,
            geometry.qkv_a_companion_rank,
        ),
        "wq_b_weight": (
            geometry.heads * geometry.head_dim,
            geometry.q_lora_rank,
        ),
    }
    values = {
        "q_residual": q_residual,
        "index_keys": index_keys,
        "head_weights": head_weights,
        "qkv_a_companion": qkv_a_companion,
        "wq_b_weight": wq_b_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"layer-0 DSA replacement {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if q_residual.dtype != jnp.bfloat16 or (
        index_keys.dtype != jnp.bfloat16
    ) or head_weights.dtype != jnp.float32 or (
        qkv_a_companion.dtype != jnp.bfloat16
    ) or wq_b_weight.dtype != jnp.float32:
        raise ValueError("layer-0 DSA replacement dtype contract drifted")

    query = _dot_out_in(
        q_residual.astype(jnp.float32),
        wq_b_weight,
        output_dtype=jnp.float32,
    ).reshape(geometry.decode_rows, geometry.heads, geometry.head_dim)
    decode_positions = jnp.zeros(
        (geometry.decode_rows,), dtype=jnp.int32
    ).at[0].set(jnp.int32(geometry.prompt_tokens))
    cos, sin = rotary_cos_sin(
        decode_positions,
        rotary_dim=geometry.rotary_dim,
        theta=geometry.theta,
        dtype=jnp.float32,
    )
    rotated_query = apply_rotary(
        query[:, :, : geometry.rotary_dim],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=True,
    )
    query = jnp.concatenate(
        (rotated_query, query[:, :, geometry.rotary_dim :]), axis=-1
    ).astype(jnp.float32)
    return Layer0DsaState(
        query=query,
        index_keys=index_keys,
        head_weights=head_weights,
        qkv_a_companion=qkv_a_companion,
    )


def _unpack_legacy_fused_qkv_output(
    packed: Any,
    *,
    geometry: Layer0DsaProbeGeometry,
) -> tuple[Any, Any]:
    """Undo the fused-part-per-shard output reorder after the runtime dot."""

    shards = geometry.legacy_tensor_shards
    q_per_shard = geometry.q_lora_rank // shards
    kv_per_shard = geometry.qkv_a_companion_rank // shards
    local_output = q_per_shard + kv_per_shard
    if packed.shape != (geometry.decode_rows, shards * local_output):
        raise ValueError("legacy fused qkv packed output geometry drifted")
    packed = packed.reshape(geometry.decode_rows, shards, local_output)
    q_a = packed[:, :, :q_per_shard].reshape(
        geometry.decode_rows, geometry.q_lora_rank
    )
    companion = packed[:, :, q_per_shard:].reshape(
        geometry.decode_rows, geometry.qkv_a_companion_rank
    )
    return q_a, companion


def _dot_out_in(lhs: Any, weight_out_in: Any, *, output_dtype: Any) -> Any:
    """Match the legacy ``lhs @ weight.T`` source association."""

    result = lax.dot_general(
        lhs,
        weight_out_in,
        dimension_numbers=(((lhs.ndim - 1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return result.astype(output_dtype)


def affine_key_layer_norm(
    value: Any,
    weight: Any,
    bias: Any,
    *,
    epsilon: float,
    mode: KeyNormMode,
) -> Any:
    """Apply one selectable FP32 index-key LayerNorm association."""

    if value.ndim != 2 or weight.shape != (value.shape[1],) or (
        bias.shape != weight.shape
    ):
        raise ValueError("index-key LayerNorm shapes drifted")
    value_f32 = value.astype(jnp.float32)
    mean = jnp.mean(value_f32, axis=-1, keepdims=True)
    centered = value_f32 - mean
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    denominator = variance + jnp.float32(epsilon)
    if mode == "divide_sqrt":
        normalized = centered / jnp.sqrt(denominator)
    elif mode == "multiply_rsqrt":
        normalized = centered * lax.rsqrt(denominator)
    else:
        raise ValueError(f"unsupported index-key LayerNorm mode {mode!r}")
    return (
        normalized * weight.astype(jnp.float32)
        + bias.astype(jnp.float32)
    ).astype(jnp.float32)


def _project_keys(
    hidden: Any,
    positions: Any,
    wk_weight: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    *,
    geometry: Layer0DsaProbeGeometry,
    key_norm_mode: KeyNormMode,
) -> Any:
    projected = _dot_out_in(
        hidden.astype(jnp.float32),
        wk_weight.astype(jnp.float32),
        output_dtype=jnp.float32,
    )
    keys = affine_key_layer_norm(
        projected,
        key_norm_weight,
        key_norm_bias,
        epsilon=geometry.key_norm_epsilon,
        mode=key_norm_mode,
    )
    cos, sin = rotary_cos_sin(
        positions,
        rotary_dim=geometry.rotary_dim,
        theta=geometry.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        keys[:, : geometry.rotary_dim],
        cos,
        sin,
        interleaved=True,
    )
    return jnp.concatenate(
        (rotated, keys[:, geometry.rotary_dim :]), axis=-1
    ).astype(jnp.bfloat16)


def layer0_dsa_scorer_internals(
    normalized_hidden: Any,
    q_residual: Any,
    positions: Any,
    wq_b_weight: Any,
    wk_weight: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    head_weight: Any,
    *,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
) -> Layer0DsaScorerInternals:
    """Reproduce the three adapted tensors at the accepted scorer boundary.

    Unlike :func:`_project_keys`, ``current_keys`` intentionally remains
    FP32.  The accepted callback observes the post-RoPE value before its
    separate BF16 cache write, which makes this a narrow arithmetic
    discriminator rather than a reconstructed prompt-cache claim.
    """

    expected_shapes = {
        "normalized_hidden": (geometry.decode_rows, geometry.hidden_size),
        "q_residual": (geometry.decode_rows, geometry.q_lora_rank),
        "positions": (geometry.decode_rows,),
        "wq_b_weight": (
            geometry.heads * geometry.head_dim,
            geometry.q_lora_rank,
        ),
        "wk_weight": (geometry.head_dim, geometry.hidden_size),
        "key_norm_weight": (geometry.head_dim,),
        "key_norm_bias": (geometry.head_dim,),
        "head_weight": (geometry.heads, geometry.hidden_size),
    }
    values = {
        "normalized_hidden": normalized_hidden,
        "q_residual": q_residual,
        "positions": positions,
        "wq_b_weight": wq_b_weight,
        "wk_weight": wk_weight,
        "key_norm_weight": key_norm_weight,
        "key_norm_bias": key_norm_bias,
        "head_weight": head_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"layer-0 scorer internal {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if normalized_hidden.dtype != jnp.bfloat16 or (
        q_residual.dtype != jnp.bfloat16
    ):
        raise ValueError("layer-0 scorer hidden/q state must be BF16")
    if positions.dtype != jnp.int32:
        raise ValueError("layer-0 scorer positions must be int32")
    for name in (
        "wq_b_weight",
        "wk_weight",
        "key_norm_weight",
        "key_norm_bias",
        "head_weight",
    ):
        if values[name].dtype != jnp.float32:
            raise ValueError(f"layer-0 scorer {name} must be FP32")

    query = _dot_out_in(
        q_residual.astype(jnp.float32),
        wq_b_weight,
        output_dtype=jnp.float32,
    ).reshape(geometry.decode_rows, geometry.heads, geometry.head_dim)
    query_cos, query_sin = rotary_cos_sin(
        positions,
        rotary_dim=geometry.rotary_dim,
        theta=geometry.theta,
        dtype=jnp.float32,
    )
    rotated_query = apply_rotary(
        query[:, :, : geometry.rotary_dim],
        query_cos[:, None, :],
        query_sin[:, None, :],
        interleaved=True,
    )
    query = jnp.concatenate(
        (rotated_query, query[:, :, geometry.rotary_dim :]), axis=-1
    ).astype(jnp.float32)

    hidden_f32 = normalized_hidden.astype(jnp.float32)
    head_weights = _dot_out_in(
        hidden_f32,
        head_weight,
        output_dtype=jnp.float32,
    ) * jnp.float32(geometry.heads**-0.5)
    key_pre = _dot_out_in(
        hidden_f32,
        wk_weight,
        output_dtype=jnp.float32,
    )
    current_keys = affine_key_layer_norm(
        key_pre,
        key_norm_weight,
        key_norm_bias,
        epsilon=geometry.key_norm_epsilon,
        mode="divide_sqrt",
    )
    key_cos, key_sin = rotary_cos_sin(
        positions,
        rotary_dim=geometry.rotary_dim,
        theta=geometry.theta,
        dtype=jnp.float32,
    )
    rotated_keys = apply_rotary(
        current_keys[:, : geometry.rotary_dim],
        key_cos,
        key_sin,
        interleaved=True,
    )
    current_keys = jnp.concatenate(
        (rotated_keys, current_keys[:, geometry.rotary_dim :]), axis=-1
    ).astype(jnp.float32)
    return Layer0DsaScorerInternals(query, head_weights, current_keys)


def layer0_decode_normalized_hidden(
    unique_embeddings: Any,
    current_embedding_row: Any,
    input_norm_weight: Any,
    *,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
) -> Any:
    """Build the sealed M32 decode input and apply its ordinary input norm."""

    expected_shapes = {
        "unique_embeddings": (
            unique_embeddings.shape[0],
            geometry.hidden_size,
        ),
        "current_embedding_row": (1,),
        "input_norm_weight": (geometry.hidden_size,),
    }
    values = {
        "unique_embeddings": unique_embeddings,
        "current_embedding_row": current_embedding_row,
        "input_norm_weight": input_norm_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"layer-0 decode normalization {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if unique_embeddings.dtype != jnp.bfloat16 or (
        input_norm_weight.dtype != jnp.bfloat16
    ):
        raise ValueError("layer-0 decode normalization inputs must be BF16")
    if current_embedding_row.dtype != jnp.int32:
        raise ValueError("layer-0 decode embedding row must be int32")

    current = jnp.take(unique_embeddings, current_embedding_row, axis=0)
    decode_hidden = jnp.zeros(
        (geometry.decode_rows, geometry.hidden_size), dtype=jnp.bfloat16
    ).at[0].set(current[0])
    return rms_norm(
        decode_hidden,
        input_norm_weight,
        epsilon=geometry.rms_norm_epsilon,
    )


def layer0_prompt_index_keys_chunked(
    unique_embeddings: Any,
    prompt_embedding_rows: Any,
    input_norm_weight: Any,
    wk_weight: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    *,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
    key_norm_mode: KeyNormMode = "divide_sqrt",
) -> Any:
    """Build layer-0 prompt keys with the accepted 2,048-row association.

    Only row indices are padded before the four-chunk map.  Each map body
    gathers and normalizes one prompt chunk, so this helper never creates a
    full ``[prompt, hidden]`` materialization.  It is a bounded prefill
    discriminator; decode remains a true one-row program.
    """

    expected_shapes = {
        "unique_embeddings": (
            unique_embeddings.shape[0],
            geometry.hidden_size,
        ),
        "prompt_embedding_rows": (geometry.prompt_tokens,),
        "input_norm_weight": (geometry.hidden_size,),
        "wk_weight": (geometry.head_dim, geometry.hidden_size),
        "key_norm_weight": (geometry.head_dim,),
        "key_norm_bias": (geometry.head_dim,),
    }
    values = {
        "unique_embeddings": unique_embeddings,
        "prompt_embedding_rows": prompt_embedding_rows,
        "input_norm_weight": input_norm_weight,
        "wk_weight": wk_weight,
        "key_norm_weight": key_norm_weight,
        "key_norm_bias": key_norm_bias,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"layer-0 chunked prompt-key {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if unique_embeddings.dtype != jnp.bfloat16 or (
        input_norm_weight.dtype != jnp.bfloat16
    ) or key_norm_weight.dtype != jnp.bfloat16 or (
        key_norm_bias.dtype != jnp.bfloat16
    ):
        raise ValueError("layer-0 chunked embeddings/norms must remain BF16")
    if prompt_embedding_rows.dtype != jnp.int32:
        raise ValueError("layer-0 chunked prompt rows must remain int32")
    if wk_weight.dtype != jnp.float32:
        raise ValueError("layer-0 accepted adapted wk must remain FP32")

    padded_tokens = (
        (geometry.prompt_tokens + geometry.prompt_chunk - 1)
        // geometry.prompt_chunk
        * geometry.prompt_chunk
    )
    padded_rows = jnp.pad(
        prompt_embedding_rows,
        (0, padded_tokens - geometry.prompt_tokens),
    ).reshape(-1, geometry.prompt_chunk)
    position_chunks = jnp.arange(padded_tokens, dtype=jnp.int32).reshape(
        -1, geometry.prompt_chunk
    )

    def prompt_key_chunk(inputs: tuple[Any, Any]) -> Any:
        embedding_rows, positions = inputs
        hidden = jnp.take(unique_embeddings, embedding_rows, axis=0)
        normalized = rms_norm(
            hidden,
            input_norm_weight,
            epsilon=geometry.rms_norm_epsilon,
        )
        return _project_keys(
            normalized,
            positions,
            wk_weight,
            key_norm_weight,
            key_norm_bias,
            geometry=geometry,
            key_norm_mode=key_norm_mode,
        )

    return lax.map(
        prompt_key_chunk,
        (padded_rows, position_chunks),
    ).reshape(padded_tokens, geometry.head_dim)[: geometry.prompt_tokens]


def layer0_dsa_state(
    unique_embeddings: Any,
    prompt_embedding_rows: Any,
    current_embedding_row: Any,
    input_norm_weight: Any,
    q_a_weight: Any,
    q_a_norm_weight: Any,
    wq_b_weight: Any,
    wk_weight: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    head_weight: Any,
    q_a_projection_scale: Any | None = None,
    *,
    geometry: Layer0DsaProbeGeometry = Layer0DsaProbeGeometry(),
    key_norm_mode: KeyNormMode = "divide_sqrt",
    q_a_projection_mode: QaProjectionMode = "separate_q_a",
) -> Layer0DsaState:
    """Build one event with legacy prompt-chunk and decode-row geometry.

    ``wq_b_weight`` and ``wk_weight`` are supplied already dequantized.  The
    caller can therefore compare FP32 legacy adaptation with BF16 production
    adaptation without changing any other operation or compiler shape.

    Under ``legacy_fused_qkv_a``, ``q_a_weight`` is the already dequantized
    logical ``q_a + kv_a`` weight.  The two ``legacy_runtime_*`` modes instead
    consume the raw-FP8 reordered weight and its separate FP32 scale exactly
    as the sealed disabled-requant path presents them to its XLA matmul.  The
    companion output remains live in every fused mode so XLA cannot prune the
    fused output width.
    """

    expected_shapes = {
        "unique_embeddings": (unique_embeddings.shape[0], geometry.hidden_size),
        "prompt_embedding_rows": (geometry.prompt_tokens,),
        "current_embedding_row": (1,),
        "input_norm_weight": (geometry.hidden_size,),
        "q_a_norm_weight": (geometry.q_lora_rank,),
        "wq_b_weight": (geometry.heads * geometry.head_dim, geometry.q_lora_rank),
        "wk_weight": (geometry.head_dim, geometry.hidden_size),
        "key_norm_weight": (geometry.head_dim,),
        "key_norm_bias": (geometry.head_dim,),
        "head_weight": (geometry.heads, geometry.hidden_size),
    }
    values = {
        "unique_embeddings": unique_embeddings,
        "prompt_embedding_rows": prompt_embedding_rows,
        "current_embedding_row": current_embedding_row,
        "input_norm_weight": input_norm_weight,
        "q_a_norm_weight": q_a_norm_weight,
        "wq_b_weight": wq_b_weight,
        "wk_weight": wk_weight,
        "key_norm_weight": key_norm_weight,
        "key_norm_bias": key_norm_bias,
        "head_weight": head_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"layer-0 DSA {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    q_a_output = geometry.q_lora_rank
    if q_a_projection_mode == "legacy_fused_qkv_a":
        q_a_output += geometry.qkv_a_companion_rank
        expected_q_a_weight = (q_a_output, geometry.hidden_size)
    elif q_a_projection_mode == "separate_q_a":
        expected_q_a_weight = (q_a_output, geometry.hidden_size)
    elif q_a_projection_mode == "legacy_runtime_fused_qkv_a_global":
        q_a_output += geometry.qkv_a_companion_rank
        if geometry.q_lora_rank % geometry.legacy_tensor_shards or (
            geometry.qkv_a_companion_rank % geometry.legacy_tensor_shards
        ):
            raise ValueError(
                "legacy fused qkv outputs must divide the tensor shards"
            )
        expected_q_a_weight = (geometry.hidden_size, q_a_output)
    elif q_a_projection_mode == "legacy_runtime_fused_qkv_a_sharded":
        q_a_output += geometry.qkv_a_companion_rank
        if geometry.q_lora_rank % geometry.legacy_tensor_shards or (
            geometry.qkv_a_companion_rank % geometry.legacy_tensor_shards
        ):
            raise ValueError(
                "legacy fused qkv outputs must divide the tensor shards"
            )
        local_output = q_a_output // geometry.legacy_tensor_shards
        expected_q_a_weight = (
            geometry.legacy_tensor_shards,
            geometry.hidden_size,
            local_output,
        )
    else:
        raise ValueError(
            f"unsupported q_a projection mode {q_a_projection_mode!r}"
        )
    if q_a_weight.shape != expected_q_a_weight:
        raise ValueError(
            "layer-0 DSA q_a weight shape drifted: "
            f"expected={expected_q_a_weight} found={q_a_weight.shape}"
        )
    runtime_mode = q_a_projection_mode.startswith("legacy_runtime_")
    if runtime_mode:
        local_output = q_a_output // geometry.legacy_tensor_shards
        expected_scale = (
            (
                geometry.hidden_size // 128,
                q_a_output,
            )
            if q_a_projection_mode == "legacy_runtime_fused_qkv_a_global"
            else (
                geometry.legacy_tensor_shards,
                geometry.hidden_size // 128,
                local_output,
            )
        )
        if q_a_projection_scale is None or (
            q_a_projection_scale.shape != expected_scale
        ):
            found = (
                None
                if q_a_projection_scale is None
                else q_a_projection_scale.shape
            )
            raise ValueError(
                "legacy runtime fused qkv scale shape drifted: "
                f"expected={expected_scale} found={found}"
            )
        if q_a_weight.dtype != jnp.float8_e4m3fn or (
            q_a_projection_scale.dtype != jnp.float32
        ):
            raise ValueError("legacy runtime fused qkv dtype contract drifted")
    elif q_a_projection_scale is not None:
        raise ValueError("non-runtime q_a projection must not receive a scale")
    if unique_embeddings.dtype != jnp.bfloat16:
        raise ValueError("layer-0 DSA embeddings must remain BF16")
    if prompt_embedding_rows.dtype != jnp.int32 or (
        current_embedding_row.dtype != jnp.int32
    ):
        raise ValueError("layer-0 DSA embedding rows must be int32")

    prompt = jnp.take(unique_embeddings, prompt_embedding_rows, axis=0)
    padded_tokens = (
        (geometry.prompt_tokens + geometry.prompt_chunk - 1)
        // geometry.prompt_chunk
        * geometry.prompt_chunk
    )
    prompt = jnp.pad(prompt, ((0, padded_tokens - geometry.prompt_tokens), (0, 0)))
    prompt_chunks = prompt.reshape(
        padded_tokens // geometry.prompt_chunk,
        geometry.prompt_chunk,
        geometry.hidden_size,
    )
    position_chunks = jnp.arange(padded_tokens, dtype=jnp.int32).reshape(
        padded_tokens // geometry.prompt_chunk,
        geometry.prompt_chunk,
    )

    def prompt_key_chunk(inputs: tuple[Any, Any]) -> Any:
        hidden, positions = inputs
        normalized = rms_norm(
            hidden,
            input_norm_weight,
            epsilon=geometry.rms_norm_epsilon,
        )
        return _project_keys(
            normalized,
            positions,
            wk_weight,
            key_norm_weight,
            key_norm_bias,
            geometry=geometry,
            key_norm_mode=key_norm_mode,
        )

    prompt_keys = lax.map(
        prompt_key_chunk,
        (prompt_chunks, position_chunks),
    ).reshape(padded_tokens, geometry.head_dim)[: geometry.prompt_tokens]

    normalized = layer0_decode_normalized_hidden(
        unique_embeddings,
        current_embedding_row,
        input_norm_weight,
        geometry=geometry,
    )
    if q_a_projection_mode == "legacy_fused_qkv_a":
        with jax.named_scope("legacy_fused_qkv_a_m32_n2624"):
            fused_qkv_a = _dot_out_in(
                normalized,
                q_a_weight.astype(jnp.bfloat16),
                output_dtype=jnp.bfloat16,
            )
            q_a = fused_qkv_a[:, : geometry.q_lora_rank]
            qkv_a_companion = fused_qkv_a[:, geometry.q_lora_rank :]
    elif q_a_projection_mode == "legacy_runtime_fused_qkv_a_global":
        with jax.named_scope("legacy_runtime_fused_qkv_a_m32_global_n2624"):
            packed_qkv_a = _legacy_runtime_fp8_dot(
                normalized,
                q_a_weight,
                q_a_projection_scale,
            )
            q_a, qkv_a_companion = _unpack_legacy_fused_qkv_output(
                packed_qkv_a,
                geometry=geometry,
            )
    elif q_a_projection_mode == "legacy_runtime_fused_qkv_a_sharded":
        with jax.named_scope("legacy_runtime_fused_qkv_a_m32_tp32_n82"):
            local_qkv_a = lax.map(
                lambda operands: _legacy_runtime_fp8_dot(
                    normalized,
                    operands[0],
                    operands[1],
                ),
                (q_a_weight, q_a_projection_scale),
            )
            packed_qkv_a = local_qkv_a.transpose(1, 0, 2).reshape(
                geometry.decode_rows,
                geometry.q_lora_rank + geometry.qkv_a_companion_rank,
            )
            q_a, qkv_a_companion = _unpack_legacy_fused_qkv_output(
                packed_qkv_a,
                geometry=geometry,
            )
    else:
        q_a = _dot_out_in(
            normalized,
            q_a_weight.astype(jnp.bfloat16),
            output_dtype=jnp.bfloat16,
        )
        qkv_a_companion = jnp.zeros(
            (geometry.decode_rows, geometry.qkv_a_companion_rank),
            dtype=jnp.bfloat16,
        )
    q_residual = rms_norm(
        q_a,
        q_a_norm_weight,
        epsilon=geometry.q_norm_epsilon,
    )
    query = _dot_out_in(
        q_residual.astype(jnp.float32),
        wq_b_weight.astype(jnp.float32),
        output_dtype=jnp.float32,
    ).reshape(geometry.decode_rows, geometry.heads, geometry.head_dim)
    decode_positions = jnp.zeros((geometry.decode_rows,), dtype=jnp.int32).at[
        0
    ].set(jnp.int32(geometry.prompt_tokens))
    cos, sin = rotary_cos_sin(
        decode_positions,
        rotary_dim=geometry.rotary_dim,
        theta=geometry.theta,
        dtype=jnp.float32,
    )
    rotated_query = apply_rotary(
        query[:, :, : geometry.rotary_dim],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=True,
    )
    query = jnp.concatenate(
        (rotated_query, query[:, :, geometry.rotary_dim :]), axis=-1
    ).astype(jnp.float32)
    head_weights = _dot_out_in(
        normalized.astype(jnp.float32),
        head_weight.astype(jnp.float32),
        output_dtype=jnp.float32,
    ) * jnp.float32(geometry.heads**-0.5)
    current_key = _project_keys(
        normalized,
        decode_positions,
        wk_weight,
        key_norm_weight,
        key_norm_bias,
        geometry=geometry,
        key_norm_mode=key_norm_mode,
    )[0:1]
    return Layer0DsaState(
        query=query,
        index_keys=jnp.concatenate((prompt_keys, current_key), axis=0),
        head_weights=head_weights.astype(jnp.float32),
        qkv_a_companion=qkv_a_companion,
    )


def legacy_pagewise_dcp_scores(
    query: Any,
    index_keys: Any,
    head_weights: Any,
    *,
    geometry: LegacyScoreGeometry = LegacyScoreGeometry(),
) -> Any:
    """Reconstruct row zero with the sealed batch/page/DCP score shapes."""

    if query.ndim != 3 or index_keys.ndim != 2 or head_weights.ndim != 2:
        raise ValueError("legacy DSA scorer ranks must be 3/2/2")
    rows, heads, head_dim = query.shape
    if rows <= 1 or head_weights.shape != (rows, heads) or (
        index_keys.shape[1] != head_dim
    ):
        raise ValueError("legacy DSA scorer shapes drifted")
    if query.dtype != jnp.float32 or head_weights.dtype != jnp.float32 or (
        index_keys.dtype != jnp.bfloat16
    ):
        raise ValueError("legacy DSA scorer dtype contract drifted")

    context = index_keys.shape[0]
    pages = geometry.local_score_width // geometry.local_page_size
    local_columns = jnp.arange(geometry.local_score_width, dtype=jnp.int32)
    page_size = jnp.int32(geometry.local_page_size)
    global_page_size = jnp.int32(
        geometry.local_page_size * geometry.dcp_size
    )
    scale = jnp.float32(head_dim**-0.5)

    def score_shard(shard: Any) -> tuple[Any, Any]:
        local_positions = (
            (local_columns // page_size) * global_page_size
            + shard.astype(jnp.int32) * page_size
            + local_columns % page_size
        )
        valid = local_positions < jnp.int32(context)
        safe_positions = jnp.clip(local_positions, 0, context - 1)
        local_keys = jnp.where(
            valid[:, None],
            jnp.take(index_keys, safe_positions, axis=0),
            jnp.zeros((1, head_dim), dtype=jnp.bfloat16),
        )
        cache = jnp.concatenate(
            (
                local_keys.reshape(pages, geometry.local_page_size, head_dim),
                jnp.zeros(
                    (1, geometry.local_page_size, head_dim), dtype=jnp.bfloat16
                ),
            ),
            axis=0,
        )
        zero_page = jnp.int32(pages)

        def score_page(page: Any) -> Any:
            page_ids = jnp.full((rows,), zero_page, dtype=jnp.int32).at[0].set(
                page.astype(jnp.int32)
            )
            key_block = jnp.take(cache, page_ids, axis=0).astype(jnp.float32)
            per_head = jnp.maximum(
                jnp.einsum(
                    "thd,tpd->thp",
                    query,
                    key_block,
                    preferred_element_type=jnp.float32,
                )
                * scale,
                jnp.float32(0.0),
            )
            scores = jnp.einsum(
                "th,thp->tp",
                head_weights,
                per_head,
                preferred_element_type=jnp.float32,
            )
            return scores[0]

        local_scores = lax.map(
            score_page, jnp.arange(pages, dtype=jnp.int32)
        ).reshape(geometry.local_score_width)
        return local_positions, jnp.where(valid, local_scores, -jnp.inf)

    positions, scores = lax.map(
        score_shard, jnp.arange(geometry.dcp_size, dtype=jnp.int32)
    )
    valid = positions < jnp.int32(context)
    scatter_positions = jnp.where(valid, positions, jnp.int32(context))
    return jnp.full((context,), -jnp.inf, dtype=jnp.float32).at[
        scatter_positions.reshape(-1)
    ].set(scores.reshape(-1), mode="drop")


def legacy_local_dcp_score_inputs(
    index_keys: Any,
    shard: int,
    *,
    geometry: LegacyDcpXlaScoreGeometry = LegacyDcpXlaScoreGeometry(),
) -> LegacyLocalDcpScoreInputs:
    """Pack one protected DCP8 stripe into the sealed 8K local shapes.

    Packing is intentionally outside :func:`legacy_local_dcp_xla_scores` so
    its compiled HLO sees the same cache, block-table, and local-length
    operands as the accepted shard-map body.  Only row zero is live in this
    bounded event; the other 31 rows preserve the legacy static bucket solely
    to reproduce its numerical association.
    """

    if index_keys.ndim != 2 or index_keys.shape[1] != geometry.head_dim:
        raise ValueError("legacy local DCP index-key geometry drifted")
    if index_keys.dtype != jnp.bfloat16:
        raise ValueError("legacy local DCP index keys must remain BF16")
    if not isinstance(shard, int) or isinstance(shard, bool) or not (
        0 <= shard < geometry.dcp_size
    ):
        raise ValueError("legacy local DCP shard is out of range")
    context = index_keys.shape[0]
    if context > geometry.max_model_len:
        raise ValueError("legacy local DCP context exceeds max_model_len")

    local_columns = jnp.arange(
        geometry.local_score_width, dtype=jnp.int32
    )
    local_page = jnp.int32(geometry.local_page_size)
    global_page = jnp.int32(geometry.global_page_size)
    local_positions = (
        (local_columns // local_page) * global_page
        + jnp.int32(shard * geometry.local_page_size)
        + local_columns % local_page
    )
    valid = local_positions < jnp.int32(context)
    safe_positions = jnp.clip(local_positions, 0, context - 1)
    local_keys = jnp.where(
        valid[:, None],
        jnp.take(index_keys, safe_positions, axis=0),
        jnp.zeros((1, geometry.head_dim), dtype=jnp.bfloat16),
    ).reshape(
        geometry.owned_block_count,
        geometry.local_page_size,
        geometry.head_dim,
    )
    cache = jnp.pad(
        local_keys,
        (
            (0, geometry.cache_pages - geometry.owned_block_count),
            (0, 0),
            (0, 0),
        ),
    )
    block_tables = jnp.zeros(
        (geometry.decode_rows, geometry.owned_block_count),
        dtype=jnp.int32,
    ).at[0].set(
        jnp.arange(geometry.owned_block_count, dtype=jnp.int32)
    )
    kv_lens = jnp.zeros((geometry.decode_rows,), dtype=jnp.int32).at[0].set(
        jnp.sum(valid, dtype=jnp.int32)
    )
    return LegacyLocalDcpScoreInputs(
        cache=cache,
        block_tables=block_tables,
        kv_lens=kv_lens,
        local_positions=local_positions,
    )


def legacy_local_dcp_xla_scores(
    query: Any,
    index_cache: Any,
    head_weights: Any,
    block_tables: Any,
    kv_lens: Any,
    *,
    geometry: LegacyDcpXlaScoreGeometry = LegacyDcpXlaScoreGeometry(),
) -> Any:
    """Reproduce one accepted 8K XLA scorer shard with exact static shapes."""

    if query.shape != (
        geometry.decode_rows,
        geometry.heads,
        geometry.head_dim,
    ) or head_weights.shape != (geometry.decode_rows, geometry.heads):
        raise ValueError("legacy local DCP query/head-weight geometry drifted")
    if index_cache.shape != (
        geometry.cache_pages,
        geometry.local_page_size,
        geometry.head_dim,
    ):
        raise ValueError("legacy local DCP cache geometry drifted")
    if block_tables.shape != (
        geometry.decode_rows,
        geometry.owned_block_count,
    ) or kv_lens.shape != (geometry.decode_rows,):
        raise ValueError("legacy local DCP score metadata geometry drifted")
    if query.dtype != jnp.float32 or head_weights.dtype != jnp.float32:
        raise ValueError("legacy local DCP query/head weights must remain FP32")
    if index_cache.dtype != jnp.bfloat16:
        raise ValueError("legacy local DCP cache must remain BF16")
    if block_tables.dtype != jnp.int32 or kv_lens.dtype != jnp.int32:
        raise ValueError("legacy local DCP metadata must remain int32")

    scale = geometry.head_dim**-0.5

    def block_scores(block: Any) -> Any:
        page_ids = jnp.clip(
            block_tables[:, block], 0, geometry.cache_pages - 1
        )
        key_block = index_cache[page_ids].astype(jnp.float32)
        per_head = jnp.maximum(
            jnp.einsum(
                "thd,tpd->thp",
                query,
                key_block,
                preferred_element_type=jnp.float32,
            )
            * scale,
            jnp.float32(0.0),
        )
        scores = jnp.einsum(
            "th,thp->tp",
            head_weights,
            per_head,
            preferred_element_type=jnp.float32,
        )
        local_columns = (
            block * geometry.local_page_size
            + lax.broadcasted_iota(
                jnp.int32, (1, geometry.local_page_size), 1
            )
        )
        return jnp.where(
            local_columns < kv_lens[:, None], scores, -jnp.inf
        )

    blocks = lax.map(
        block_scores,
        jnp.arange(geometry.owned_block_count, dtype=jnp.int32),
    )
    return blocks.transpose(1, 0, 2).reshape(
        geometry.decode_rows, geometry.local_score_width
    )


def one_row_pagewise_scores(
    query: Any,
    index_keys: Any,
    head_weights: Any,
    *,
    page_size: int = 512,
) -> Any:
    """One-row XLA challenger with the same 512-key score association."""

    if query.ndim != 3 or query.shape[0] != 1:
        raise ValueError("one-row DSA challenger requires query shape [1,H,D]")
    if index_keys.ndim != 2 or index_keys.shape[1] != query.shape[2] or (
        head_weights.shape != query.shape[:2]
    ):
        raise ValueError("one-row DSA challenger shapes drifted")
    if not isinstance(page_size, int) or isinstance(page_size, bool) or (
        page_size <= 0
    ):
        raise ValueError("one-row DSA page size must be positive")
    context, head_dim = index_keys.shape
    padded = (context + page_size - 1) // page_size * page_size
    keys = jnp.pad(index_keys, ((0, padded - context), (0, 0))).reshape(
        padded // page_size, page_size, head_dim
    )
    scale = jnp.float32(head_dim**-0.5)

    def score_page(key_block: Any) -> Any:
        per_head = jnp.maximum(
            jnp.einsum(
                "hd,pd->hp",
                query[0],
                key_block.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
            * scale,
            jnp.float32(0.0),
        )
        return jnp.einsum(
            "h,hp->p",
            head_weights[0],
            per_head,
            preferred_element_type=jnp.float32,
        )

    return lax.map(score_page, keys).reshape(padded)[:context]
