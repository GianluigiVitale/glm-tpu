"""Bounded layer-1 RMS schedule discriminator arms (four-chip, sealed inputs).

``build_layer1_rms_schedule_arm(mesh, fp32_carry_schedule=False)`` is the
control arm that reproduced the protected DB548 row; ``True`` is the
accepted-schedule arm: the three sealed BF16 rows are summed in FP32 with no
BF16 rounding of the post-attention residual, one FP32 ``optimization_barrier``
keeps the ``[32,6144]`` sum alive so the variance reduce takes the accepted
``f32[32,6144] -> f32[32]`` schedule, and both the reduce and the weighted
output consume that barrier.
"""

from __future__ import annotations

from typing import Any


def build_layer1_rms_schedule_arm(mesh: Any, *, fp32_carry_schedule: bool) -> Any:
    import jax
    from jax import lax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm
    from glm_tpu.greenfield.kernels.stage_local import (
        _strategy_nd_row0_bf16_reduce,
    )

    groups = ((0, 1, 2, 3),)

    def local(
        partial_slot: Any,
        attention_value: Any,
        residual_value: Any,
        layer1_norm: Any,
    ) -> Any:
        with jax.named_scope("greenfield_captured_rms_strategy_gather"):
            gathered = lax.all_gather(
                partial_slot[0],
                axis_name="lp4",
                axis=0,
                tiled=False,
                axis_index_groups=groups,
            )
        with jax.named_scope("greenfield_captured_rms_strategy_tree"):
            dense_update = _strategy_nd_row0_bf16_reduce(
                gathered.reshape(32, 1, 6144)
            )
        with jax.named_scope("greenfield_captured_rms_dense_m32"):
            dense_m32 = jnp.pad(
                dense_update,
                ((0, 31), (0, 0)),
                mode="constant",
                constant_values=jnp.bfloat16(0),
            )
        with jax.named_scope("greenfield_captured_rms_predense_m32"):
            attention_m32 = jnp.pad(
                attention_value,
                ((0, 31), (0, 0)),
                mode="constant",
                constant_values=jnp.bfloat16(0),
            )
            residual_source_m32 = jnp.pad(
                residual_value,
                ((0, 31), (0, 0)),
                mode="constant",
                constant_values=jnp.bfloat16(0),
            )
        with jax.named_scope("greenfield_captured_rms_carried_residual"):
            residual_m32 = (
                attention_m32.astype(jnp.float32)
                + residual_source_m32.astype(jnp.float32)
            ).astype(jnp.bfloat16)
        with jax.named_scope("greenfield_captured_rms_layer1"):
            if fp32_carry_schedule:
                # Accepted-schedule arm with an FP32 carry.  The certificate
                # `gate-d-layer1-scale-frontier-certificate.json` proved that
                # the DB548 and accepted rows are exact functions of the same
                # FP32 sum dense + attention + combined_residual (no BF16
                # rounding of the post-attention residual, which XLA elides in
                # the compiled decoder and in the legacy) and differ only in
                # the FP32 rsqrt(mean+eps) scale.  The FP32 barrier keeps the
                # [32,6144] operand alive so the variance reduce takes the
                # accepted f32[32,6144] -> f32[32] schedule instead of the
                # folded single-row reduce; it introduces no rounding.
                with jax.named_scope("split_reduction"):
                    summed = (
                        dense_m32.astype(jnp.float32)
                        + attention_m32.astype(jnp.float32)
                        + residual_source_m32.astype(jnp.float32)
                    )
                    reduction_sum = lax.optimization_barrier(summed)
                    inverse = lax.rsqrt(
                        jnp.mean(
                            lax.square(reduction_sum), axis=-1, keepdims=True
                        )
                        + jnp.float32(1e-5)
                    )
                with jax.named_scope("split_recompute"):
                    layer1 = (
                        (reduction_sum * inverse).astype(layer1_norm.dtype)
                        * layer1_norm
                    ).astype(dense_m32.dtype)
            else:
                layer1 = fused_add_rms_norm(
                    dense_m32,
                    residual_m32,
                    layer1_norm,
                    epsilon=1e-5,
                )[0]
        with jax.named_scope("greenfield_captured_rms_live_row"):
            return layer1[:1, :]

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("lp4", None, None, None), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
