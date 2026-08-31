"""TPU compile-only form of the admitted Gate-D compensated replay.

The computational body is intentionally identical to the sealed two-CPU
capsule replay.  Only the device-platform admission changes: this builder
accepts exactly the adjacent TPU-v4 devices used by PP16 stage zero.  Callers
must lower and compile it without invoking the returned executable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import PlanValidationError
from ..kernels.reference.attention import StageLocalKvLayout
from ..kernels.reference.dsa import DsaNumericalContract
from ..kernels.reference.qkv_a import (
    FusedQkvAContract,
    one_row_fused_qkv_a_convolution,
)
from ..kernels.reference.rmsnorm import (
    fused_add_rms_norm_with_compensated_auxiliary,
)
from ..kernels.stage_local import stage_local_dsa_fp8_mapped
from .gate_d_compensated_capsule import (
    GATE_D_CAPSULE_CONTEXT_LENGTH,
    GATE_D_CAPSULE_LOGICAL_PAGE_SIZE,
    GATE_D_CAPSULE_PAGE_COUNT,
    GATE_D_CAPSULE_POSITION,
    GateDCompensatedCapsuleReplayResult,
)


def build_gate_d_compensated_pp16_hlo_replay(
    *, devices: Sequence[Any], axis_name: str = "feature"
) -> Any:
    """Build the exact two-owner TPU compile-only replay."""

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    runtime_devices = tuple(devices)
    if len(runtime_devices) != 2 or len({device.id for device in runtime_devices}) != 2:
        raise PlanValidationError(
            "Gate-D compensated capsule replay requires two distinct devices"
        )
    if any(device.platform != "tpu" for device in runtime_devices):
        raise PlanValidationError(
            "Gate-D compensated capsule replay is TPU compile-only"
        )
    if not isinstance(axis_name, str) or not axis_name:
        raise PlanValidationError(
            "Gate-D compensated capsule replay axis name must be nonempty"
        )

    mesh = Mesh(np.asarray(runtime_devices, dtype=object), (axis_name,))
    dsa_contract = DsaNumericalContract()
    qkv_contract = FusedQkvAContract()
    cache_layout = StageLocalKvLayout(
        logical_page_size=GATE_D_CAPSULE_LOGICAL_PAGE_SIZE,
        local_parallel_size=2,
        packed_cache_width=640,
    )
    groups = ((0, 1),)

    def mapped(
        hidden_update: Any,
        residual: Any,
        rms_weight: Any,
        prompt_cache: Any,
        qkv_a_weight_bits: Any,
        qkv_a_scale_inv: Any,
        q_a_norm_weight: Any,
        wq_b_bits: Any,
        wq_b_scale: Any,
        wq_b_weight: Any,
        wk_bits: Any,
        wk_scale: Any,
        wk_weight: Any,
        key_norm_weight: Any,
        key_norm_bias: Any,
        head_weight: Any,
    ) -> GateDCompensatedCapsuleReplayResult:
        local_slot = lax.axis_index(axis_name)
        rms = fused_add_rms_norm_with_compensated_auxiliary(
            hidden_update,
            residual,
            rms_weight,
            epsilon=1e-5,
        )
        qkv_a = one_row_fused_qkv_a_convolution(
            rms.output,
            qkv_a_weight_bits,
            qkv_a_scale_inv,
            q_a_norm_weight,
            contract=qkv_contract,
        )
        dsa = stage_local_dsa_fp8_mapped(
            None,
            prompt_cache[0],
            jnp.asarray([GATE_D_CAPSULE_POSITION], dtype=jnp.int32),
            jnp.arange(GATE_D_CAPSULE_PAGE_COUNT, dtype=jnp.int32)[None, :],
            jnp.asarray([GATE_D_CAPSULE_CONTEXT_LENGTH], dtype=jnp.int32),
            rms_weight,
            None,
            None,
            q_a_norm_weight,
            wq_b_bits[0],
            wq_b_scale[0],
            wk_bits[0],
            wk_scale[0],
            key_norm_weight[0],
            key_norm_bias[0],
            head_weight[0],
            local_slot,
            axis_name=axis_name,
            contract=dsa_contract,
            cache_layout=cache_layout,
            axis_index_groups=groups,
            precomputed_normalized=rms.output,
            precomputed_q_residual=qkv_a.q_residual,
            linear_backend="pallas",
            dsa_query_backend="reference",
            dsa_query_weight_aliases=(wq_b_weight[0],) * 4,
            precomputed_wk_weight=wk_weight[0],
            dsa_head_key_exact_association=True,
            dsa_score_precision="default",
        )
        return GateDCompensatedCapsuleReplayResult(
            rms.restored_rms_input_fp32[None, ...],
            rms.output[None, ...],
            dsa.internals.query[None, ...],
            dsa.internals.head_weights[None, ...],
            dsa.internals.current_key[None, ...],
            dsa.index_cache[None, ...],
            dsa.selected_positions[None, ...],
            dsa.valid_counts[None, ...],
            dsa.selected_scores[None, ...],
            jnp.asarray(dsa.contract_valid, dtype=jnp.uint8).reshape(1),
        )

    owner_matrix = P(axis_name, None, None)
    owner_vector = P(axis_name, None)
    return jax.jit(
        jax.shard_map(
            mapped,
            mesh=mesh,
            in_specs=(
                P(),
                P(),
                P(),
                P(axis_name, None, None, None),
                P(),
                P(),
                P(),
                owner_matrix,
                owner_matrix,
                owner_matrix,
                owner_matrix,
                owner_matrix,
                owner_matrix,
                owner_vector,
                owner_vector,
                owner_matrix,
            ),
            out_specs=GateDCompensatedCapsuleReplayResult(
                P(axis_name, None, None),
                P(axis_name, None, None),
                P(axis_name, None, None, None),
                P(axis_name, None, None),
                P(axis_name, None, None),
                P(axis_name, None, None, None),
                P(axis_name, None, None),
                P(axis_name, None),
                P(axis_name, None, None),
                P(axis_name),
            ),
            check_vma=False,
        )
    )
