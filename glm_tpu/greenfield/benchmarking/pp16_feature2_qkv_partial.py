"""Bounded LP2 hidden-feature partial reduction for the real N82 projection.

This module is deliberately diagnostic-only.  It keeps one live token row,
contracts one 3,072-feature half on each member of the stage-local LP2 group,
and reduces only the compact FP32 ``[32, 1, 82]`` partial.  It does not change
the default decoder or authorize a TPU acquisition.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..errors import HloContractViolationError
from ..kernels.reference.qkv_a import (
    FusedQkvAContract,
    FusedQkvAProjection,
)


def one_row_lp2_feature_partial_qkv_a_convolution(
    normalized_hidden: Any,
    packed_weight_bits: Any,
    packed_scale: Any,
    q_a_norm_weight: Any,
    *,
    axis_name: str,
    groups: Sequence[Sequence[int]],
    contract: FusedQkvAContract | None = None,
) -> FusedQkvAProjection:
    """Project one row using two FP32 K-half partials and one local reduction."""

    import jax
    import jax.numpy as jnp
    from jax import lax

    if contract is None:
        contract = FusedQkvAContract()
    if not isinstance(axis_name, str) or not axis_name:
        raise ValueError("LP2 qkv-a axis name must be nonempty")
    canonical_groups = tuple(tuple(int(rank) for rank in group) for group in groups)
    if canonical_groups != ((0, 1),):
        raise ValueError("LP2 qkv-a requires the exact local group {0,1}")
    if contract.hidden_size % 2 or contract.scale_rows % 2:
        raise ValueError("LP2 qkv-a hidden and scale rows must divide in two")

    expected_shapes = {
        "normalized_hidden": (1, contract.hidden_size),
        "packed_weight_bits": (
            contract.virtual_shards,
            contract.hidden_size,
            contract.packed_width_per_shard,
        ),
        "packed_scale": (
            contract.virtual_shards,
            contract.scale_rows,
            contract.packed_width_per_shard,
        ),
        "q_a_norm_weight": (contract.q_lora_rank,),
    }
    values = {
        "normalized_hidden": normalized_hidden,
        "packed_weight_bits": packed_weight_bits,
        "packed_scale": packed_scale,
        "q_a_norm_weight": q_a_norm_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"LP2 qkv-a {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if normalized_hidden.dtype != jnp.bfloat16:
        raise ValueError("LP2 qkv-a hidden row must be BF16")
    if packed_weight_bits.dtype != jnp.uint8:
        raise ValueError("LP2 qkv-a weights must be raw uint8 FP8 bits")
    if packed_scale.dtype != jnp.float32:
        raise ValueError("LP2 qkv-a scales must be FP32")
    if q_a_norm_weight.dtype != jnp.bfloat16:
        raise ValueError("LP2 qkv-a norm weight must be BF16")

    local_slot = lax.axis_index(axis_name).astype(jnp.int32)
    hidden_half = contract.hidden_size // 2
    scale_half = contract.scale_rows // 2
    hidden_start = local_slot * jnp.int32(hidden_half)
    scale_start = local_slot * jnp.int32(scale_half)
    local_hidden = lax.dynamic_slice(
        normalized_hidden,
        (jnp.int32(0), hidden_start),
        (1, hidden_half),
    )
    local_weight_bits = lax.dynamic_slice(
        packed_weight_bits,
        (jnp.int32(0), hidden_start, jnp.int32(0)),
        (
            contract.virtual_shards,
            hidden_half,
            contract.packed_width_per_shard,
        ),
    )
    local_scale = lax.dynamic_slice(
        packed_scale,
        (jnp.int32(0), scale_start, jnp.int32(0)),
        (
            contract.virtual_shards,
            scale_half,
            contract.packed_width_per_shard,
        ),
    )

    def project(weight_scale: tuple[Any, Any]) -> Any:
        weight_bits, scale = weight_scale
        weight = lax.bitcast_convert_type(weight_bits, jnp.float8_e4m3fn)
        expanded_scale = jnp.repeat(scale, contract.quant_block, axis=0)[
            :hidden_half
        ]
        decoded = (
            weight.astype(jnp.float32) * expanded_scale.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        return lax.conv_general_dilated(
            local_hidden,
            decoded,
            window_strides=(),
            padding=(),
            dimension_numbers=("NC", "IO", "NC"),
            preferred_element_type=jnp.float32,
        )

    with jax.named_scope("greenfield_pp16_lp2_khalf_n82_partial"):
        local_partial = lax.map(project, (local_weight_bits, local_scale))
    with jax.named_scope("greenfield_pp16_lp2_khalf_n82_f32_reduce"):
        projected_f32 = lax.psum(
            local_partial,
            axis_name,
            axis_index_groups=canonical_groups,
        )
    projected = projected_f32.astype(jnp.bfloat16)

    local_q = projected[:, :, : contract.q_width_per_shard]
    local_square_sum = jnp.sum(
        jnp.square(local_q.astype(jnp.float32)), axis=-1
    )
    global_square_sum = jnp.sum(local_square_sum, axis=0)
    inverse_rms = lax.rsqrt(
        global_square_sum / jnp.float32(contract.q_lora_rank)
        + jnp.float32(contract.epsilon)
    )
    local_normalized = (
        local_q.astype(jnp.float32) * inverse_rms[None, :, None]
    ).astype(jnp.bfloat16)
    logical_q = jnp.transpose(local_normalized, (1, 0, 2)).reshape(
        1, contract.q_lora_rank
    )
    q_residual = (logical_q * q_a_norm_weight).astype(jnp.bfloat16)
    kv_a_projection = jnp.transpose(
        projected[:, :, contract.q_width_per_shard :], (1, 0, 2)
    ).reshape(1, contract.kv_a_width)
    return FusedQkvAProjection(q_residual, kv_a_projection)


def validate_lp2_feature_partial_stablehlo(stablehlo: str) -> dict[str, Any]:
    """Fail closed on the bounded source-level communication contract."""

    lowered = stablehlo.lower()
    violations: list[str] = []
    lines = stablehlo.splitlines()
    all_reduce_indices = [
        index
        for index, line in enumerate(lines)
        if '"stablehlo.all_reduce"' in line
    ]
    all_reduce_lines = [lines[index].strip() for index in all_reduce_indices]
    if len(all_reduce_lines) != 1:
        violations.append(
            f"expected one LP2 all-reduce, found {len(all_reduce_lines)}"
        )
    else:
        all_reduce_block = "\n".join(
            lines[all_reduce_indices[0] : all_reduce_indices[0] + 7]
        )
        if (
            "(tensor<32x1x82xf32>) -> tensor<32x1x82xf32>"
            not in all_reduce_block
            or "dense<[[0, 1]]>" not in all_reduce_block
        ):
            violations.append("LP2 all-reduce shape or replica group drifted")
    for operation in (
        "all_gather",
        "all_to_all",
        "collective_broadcast",
        "collective_permute",
        "reduce_scatter",
    ):
        if f'"stablehlo.{operation}"' in stablehlo:
            violations.append(f"unexpected {operation} in LP2 qkv-a helper")
    for marker in (
        "host_callback",
        "outside_compilation",
        "xla_ffi_python_cpu_callback",
        "xla_python_cpu_callback",
    ):
        if marker in lowered:
            violations.append("LP2 qkv-a helper contains a host callback")
            break
    if "tensor<1x6144xbf16>" not in stablehlo:
        violations.append("LP2 qkv-a lost the one-row hidden signature")
    if "tensor<32x3072x82xui8>" not in stablehlo:
        violations.append("LP2 qkv-a lost the selected K-half weight")
    if "tensor<32x24x82xf32>" not in stablehlo:
        violations.append("LP2 qkv-a lost the selected K-half scale")
    if "tensor<32x1x82xbf16>" not in stablehlo:
        violations.append("LP2 qkv-a lost the post-reduction BF16 boundary")
    for dead_rows in ("tensor<32x6144xbf16>", "tensor<32x1x6144xbf16>"):
        if dead_rows in stablehlo:
            violations.append(f"LP2 qkv-a contains forbidden dead rows {dead_rows}")
    report = {
        "all_reduce_count": len(all_reduce_lines),
        "collective_group": [[0, 1]],
        "one_live_row": True,
        "partial_shape": [32, 1, 82],
        "partial_dtype": "float32",
        "passed": not violations,
        "violations": violations,
    }
    if violations:
        raise HloContractViolationError(
            f"LP2 feature-partial StableHLO rejected: {violations}"
        )
    return report
