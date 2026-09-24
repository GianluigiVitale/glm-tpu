"""The fresh KV-cache initializer: the device program that allocates one request's state.

``build_cache_initializer`` returns the jitted program the runtime compiles as ``cache_init``
(and vmaps into ``batch_cache_init``): zeroed MLA and indexer caches, empty DSA frontiers and the
identity page table in the prefill-state shardings, with health set from the prompt length. It
runs no model computation, so its compiled memory is admitted before any model graph. Moved
verbatim in S2a out of the research native-benchmark program script (``archive/research-20260922``).
"""

from __future__ import annotations

from typing import Any


def build_cache_initializer(mesh: Any, config: Any) -> Any:
    """Compile fresh-cache allocation so its full device output is budgeted first.

    Same values/sharding as make_ws32_batched_prefill_state; no model execution,
    host-sized cache, existing-state donation or prompt-specific compilation.
    The protected caller inspects its actual memory analysis BEFORE invocation.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.models.glm_moe_dsa.state import (
        Ws32BatchedPrefillState,
        ws32_batched_prefill_state_specs,
        _require_config,
    )
    from glm_tpu.models.glm_moe_dsa.state import Ws32DecoderState
    _require_config(config)
    if tuple(mesh.axis_names) != ('expert','feature') or mesh.devices.shape != (8,4):
        raise ValueError('native cache initializer requires the original expert8/feature4 mesh')
    specs = ws32_batched_prefill_state_specs()
    shardings = jax.tree.map(lambda spec: NamedSharding(mesh, spec), specs)

    def initialize(prompt_length: Any) -> Any:
        if prompt_length.shape != () or prompt_length.dtype != jnp.int32:
            raise ValueError('native cache initializer requires scalar int32 prompt length')
        healthy = (prompt_length > 0) & (prompt_length < config.context_capacity)
        decoder = Ws32DecoderState(
            jnp.zeros(config.kv_cache_shape, jnp.bfloat16),
            jnp.zeros(config.index_cache_shape, jnp.bfloat16),
            jnp.full((1, config.geometry.dsa_top_k), -1, jnp.int32),
            jnp.zeros((1,), jnp.int32),
            jnp.full((1, config.geometry.dsa_top_k), -jnp.inf, jnp.float32),
            jnp.zeros((1,), jnp.int32),
            jnp.arange(config.page_count, dtype=jnp.int32)[None, :],
            jnp.ones((1,), jnp.int32), healthy[None])
        return Ws32BatchedPrefillState(decoder,
            jnp.zeros(config.index_cache_shape, jnp.bfloat16), prompt_length, jnp.bool_(False))

    return jax.jit(initialize, in_shardings=(NamedSharding(mesh, P()),),
                   out_shardings=shardings)
