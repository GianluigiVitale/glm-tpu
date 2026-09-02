"""Default-off PP16 projection discriminator with a host FP32 rotary row.

Identical to :mod:`gate_d_projection_contraction_pp16` except that the DSA
key rotation consumes one host-evaluated FP32 ``cos|sin`` row instead of
evaluating ``cos``/``sin`` on the accelerator.  The V2 protected replay proved
the projection FP32-accurate and the key LayerNorm bit-exact while the
on-device rotary deviated by up to ~1e-2; this callable isolates that fix.
Building the callable does not lower, compile, or execute it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from ..errors import PlanValidationError
from ..kernels.reference.dsa import DsaNumericalContract
from ..kernels.reference.dsa_host_rope import (
    dsa_index_keys_from_projection_host_rope,
)
from ..kernels.reference.linear import linear
from .gate_d_compensated_capsule import GATE_D_CAPSULE_POSITION


class GateDProjectionHostRopePp16Result(NamedTuple):
    """Three rooted values sufficient to adjudicate the rotary fix."""

    normalized_hidden_owners: Any
    projected_key_owners: Any
    current_key_owners: Any


def _projection_host_rope_local(
    normalized_hidden_owner: Any,
    wk_weight_owner: Any,
    key_norm_weight_owner: Any,
    key_norm_bias_owner: Any,
    dsa_rope_row_owner: Any,
) -> GateDProjectionHostRopePp16Result:
    import jax
    import jax.numpy as jnp
    from jax import lax

    contract = DsaNumericalContract()
    if (
        normalized_hidden_owner.shape != (1, 1, contract.hidden_size)
        or normalized_hidden_owner.dtype != jnp.bfloat16
        or wk_weight_owner.shape != (1, contract.head_dim, contract.hidden_size)
        or wk_weight_owner.dtype != jnp.float32
        or key_norm_weight_owner.shape != (1, contract.head_dim)
        or key_norm_weight_owner.dtype != jnp.bfloat16
        or key_norm_bias_owner.shape != (1, contract.head_dim)
        or key_norm_bias_owner.dtype != jnp.bfloat16
        or dsa_rope_row_owner.shape != (1, contract.rotary_dim)
        or dsa_rope_row_owner.dtype != jnp.float32
    ):
        raise ValueError("Gate-D host-rope discriminator local input contract drifted")
    normalized = lax.optimization_barrier(normalized_hidden_owner[0])
    wk_weight = wk_weight_owner[0]
    with jax.default_matmul_precision("highest"):
        projected_key = linear(
            normalized,
            wk_weight,
            output_dtype=jnp.float32,
        )
    current_key = dsa_index_keys_from_projection_host_rope(
        projected_key,
        key_norm_weight_owner[0],
        key_norm_bias_owner[0],
        jnp.asarray([GATE_D_CAPSULE_POSITION], dtype=jnp.int32),
        dsa_rope_row_owner,
        contract=contract,
        key_norm_mode="divide_sqrt",
    ).astype(jnp.float32)
    return GateDProjectionHostRopePp16Result(
        normalized[None, ...],
        projected_key[None, ...],
        current_key[None, ...],
    )


def _build_shard_map(*, devices: Sequence[Any], axis_name: str) -> Any:
    import jax
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.asarray(tuple(devices), dtype=object), (axis_name,))
    owner_matrix = P(axis_name, None, None)
    owner_vector = P(axis_name, None)
    return jax.jit(
        jax.shard_map(
            _projection_host_rope_local,
            mesh=mesh,
            in_specs=(
                owner_matrix,
                owner_matrix,
                owner_vector,
                owner_vector,
                owner_vector,
            ),
            out_specs=GateDProjectionHostRopePp16Result(
                owner_matrix,
                owner_matrix,
                owner_matrix,
            ),
            check_vma=True,
        )
    )


def build_gate_d_projection_host_rope_pp16(
    *, devices: Sequence[Any], axis_name: str = "feature"
) -> Any:
    """Build, but do not lower, compile, or invoke, the two-owner replay."""

    runtime_devices = tuple(devices)
    if len(runtime_devices) != 2 or len({device.id for device in runtime_devices}) != 2:
        raise PlanValidationError(
            "Gate-D host-rope discriminator requires two distinct devices"
        )
    expected_stage_zero = (
        (0, "tpu", "TPU v4", 0, (0, 0, 0), 0),
        (1, "tpu", "TPU v4", 0, (1, 0, 0), 0),
    )
    observed_stage_zero = tuple(
        (
            int(device.id),
            str(device.platform),
            str(device.device_kind),
            int(device.process_index),
            tuple(int(coordinate) for coordinate in device.coords),
            int(device.core_on_chip),
        )
        for device in runtime_devices
    )
    if observed_stage_zero != expected_stage_zero:
        raise PlanValidationError(
            "Gate-D host-rope discriminator requires exact PP16 stage-zero TPU-v4 devices"
        )
    if axis_name != "feature":
        raise PlanValidationError(
            "Gate-D host-rope discriminator axis name must be exactly 'feature'"
        )
    return _build_shard_map(devices=runtime_devices, axis_name=axis_name)
