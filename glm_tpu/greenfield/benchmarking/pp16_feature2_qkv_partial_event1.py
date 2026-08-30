"""Forced-CPU Gate-D admission replay for the LP2 split-K qkv-a arm.

The replay deliberately stops at layer-1/event-1 DSA.  It consumes DB518's
candidate-coherent prompt-key history, recomputes the current key, and exposes
all live DSA intermediates.  It is an offline admission gate, not TPU or Gate-D
evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from ..errors import PlanValidationError
from ..kernels.reference.attention import StageLocalKvLayout
from ..kernels.reference.dsa import DsaNumericalContract
from ..kernels.stage_local import stage_local_dsa_fp8_mapped


PP16_EVENT1_POSITION = 8155
PP16_EVENT1_CONTEXT_LENGTH = 8156
PP16_EVENT1_PAGE_COUNT = 16
PP16_EVENT1_LOGICAL_PAGE_SIZE = 512
PP16_EVENT1_LOCAL_ROWS_PER_PAGE = 256
PP16_EVENT1_CURRENT_PAGE = PP16_EVENT1_POSITION // PP16_EVENT1_LOGICAL_PAGE_SIZE
_PP16_EVENT1_PAGE_ROW = PP16_EVENT1_POSITION % PP16_EVENT1_LOGICAL_PAGE_SIZE
PP16_EVENT1_CURRENT_OWNER = _PP16_EVENT1_PAGE_ROW // PP16_EVENT1_LOCAL_ROWS_PER_PAGE
PP16_EVENT1_CURRENT_LOCAL_ROW = _PP16_EVENT1_PAGE_ROW % PP16_EVENT1_LOCAL_ROWS_PER_PAGE


class Pp16Event1ReplayResult(NamedTuple):
    """Rooted values needed to classify one candidate-coherent event."""

    selected_positions: Any
    valid_counts: Any
    selected_scores: Any
    contract_valid: Any
    normalized_hidden: Any
    q_a_state: Any
    query: Any
    head_weights: Any
    current_key: Any
    index_cache: Any


def derive_db518_prompt_only_cache_bits(
    post_event_cache_bits: np.ndarray,
    expected_current_key_bits: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Remove only DB518's already-written position-8155 cache row."""

    from hashlib import sha256

    cache = np.ascontiguousarray(post_event_cache_bits)
    expected_shape = (2, 16, 256, 128)
    if cache.shape != expected_shape or cache.dtype != np.uint16:
        raise PlanValidationError(
            "DB518 layer-1 post-event cache shape or dtype drifted"
        )
    current = np.ascontiguousarray(expected_current_key_bits).reshape(-1)
    if current.shape != (128,) or current.dtype != np.uint16:
        raise PlanValidationError("accepted current-key BF16 bits drifted")
    observed = cache[
        PP16_EVENT1_CURRENT_OWNER,
        PP16_EVENT1_CURRENT_PAGE,
        PP16_EVENT1_CURRENT_LOCAL_ROW,
    ]
    if not np.array_equal(observed, current):
        raise PlanValidationError(
            "DB518 post-event cache does not contain the exact current key"
        )
    prompt_only = cache.copy()
    prompt_only[
        PP16_EVENT1_CURRENT_OWNER,
        PP16_EVENT1_CURRENT_PAGE,
        PP16_EVENT1_CURRENT_LOCAL_ROW,
    ] = np.uint16(0)
    return prompt_only, sha256(prompt_only.tobytes(order="C")).hexdigest()


def build_pp16_event1_cpu_replay(
    *, devices: Sequence[Any], axis_name: str = "feature"
) -> Any:
    """Build the exact existing DSA composition on two forced CPU devices."""

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    runtime_devices = tuple(devices)
    if len(runtime_devices) != 2 or len({device.id for device in runtime_devices}) != 2:
        raise PlanValidationError("event-1 replay requires two distinct devices")
    if any(device.platform != "cpu" for device in runtime_devices):
        raise PlanValidationError("event-1 admission replay is CPU-only")
    if not isinstance(axis_name, str) or not axis_name:
        raise PlanValidationError("event-1 replay axis name must be nonempty")

    mesh = Mesh(np.asarray(runtime_devices, dtype=object), (axis_name,))
    contract = DsaNumericalContract()
    cache_layout = StageLocalKvLayout(
        logical_page_size=512,
        local_parallel_size=2,
        packed_cache_width=640,
    )
    groups = ((0, 1),)

    def mapped(
        normalized_hidden: Any,
        q_a_state: Any,
        prompt_cache: Any,
        wq_b_bits: Any,
        wq_b_scale: Any,
        wq_b_weight: Any,
        wk_bits: Any,
        wk_scale: Any,
        wk_weight: Any,
        head_weight: Any,
        input_norm_weight: Any,
        q_a_norm_weight: Any,
        key_norm_weight: Any,
        key_norm_bias: Any,
    ) -> Pp16Event1ReplayResult:
        local_slot = lax.axis_index(axis_name)
        result = stage_local_dsa_fp8_mapped(
            None,
            prompt_cache[0],
            jnp.asarray([PP16_EVENT1_POSITION], dtype=jnp.int32),
            jnp.arange(PP16_EVENT1_PAGE_COUNT, dtype=jnp.int32)[None, :],
            jnp.asarray([PP16_EVENT1_CONTEXT_LENGTH], dtype=jnp.int32),
            input_norm_weight,
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
            contract=contract,
            cache_layout=cache_layout,
            axis_index_groups=groups,
            precomputed_normalized=normalized_hidden,
            precomputed_q_residual=q_a_state,
            linear_backend="pallas",
            dsa_query_backend="reference",
            dsa_query_weight_aliases=(wq_b_weight[0],) * 4,
            precomputed_wk_weight=wk_weight[0],
            dsa_head_key_exact_association=True,
            dsa_score_precision="default",
        )
        return Pp16Event1ReplayResult(
            result.selected_positions,
            result.valid_counts,
            result.selected_scores,
            result.contract_valid,
            result.internals.normalized_hidden,
            result.internals.q_a_state,
            result.internals.query,
            result.internals.head_weights,
            result.internals.current_key,
            result.index_cache[None, ...],
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
                P(axis_name, None, None, None),
                owner_matrix,
                owner_matrix,
                owner_matrix,
                owner_matrix,
                owner_matrix,
                owner_matrix,
                owner_matrix,
                P(),
                P(),
                owner_vector,
                owner_vector,
            ),
            out_specs=Pp16Event1ReplayResult(
                P(),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(axis_name, None, None, None),
            ),
            check_vma=False,
        )
    )
