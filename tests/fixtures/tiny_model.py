"""The frozen tiny GLM checkpoint the CPU tests run on (fixture v1 of ``tools/equivalence/fixture.py``).

``fixture(mesh, panel_geometry=...)`` returns ``(config, weights, wk_extra)`` leaf for leaf equal to
the historical research CPU fixture it replaces (``archive/research-20260922``; proved at S0 for
both geometries, ``tests/golden/data/fixture.json``), so a switched test keeps its inputs
byte for byte. ``engine_inputs`` derives what ``OrdinaryRuntime._load`` derives from those weights
on the fleet: the BF16-resident weight tree, the promoted indexer ``wk`` tables and the main RoPE
table. Everything runs on 32 forced CPU devices (``expert=8 x feature=4``; run the test body in a
child with ``XLA_FLAGS=--xla_force_host_platform_device_count=32``).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from tools.equivalence import fixture as frozen


def cpu_mesh() -> Any:
    """The fixture's ``expert=8 x feature=4`` mesh over the 32 forced CPU devices."""
    return frozen.cpu_mesh()


def fixture(mesh: Any, *, panel_geometry: bool = False) -> tuple[Any, Any, tuple[np.ndarray, ...]]:
    """Decoder config, checkpoint-form weights bound by name, and the four trailing ``wk`` draws."""
    value = frozen.fixture_v1(panel_geometry=panel_geometry)
    return value.config, frozen.bind(mesh, value), value.wk_extra


class EngineInputs(NamedTuple):
    config: Any    # the decoder config ``_load`` builds (fixture geometry)
    raw: Any       # checkpoint-form weights (FP8 bits + scales)
    weights: Any   # BF16-resident weight tree (``bf16_resident_weights``)
    wk: tuple      # promoted FP32 indexer ``wk`` tables, one per full index slot
    rope: Any      # main RoPE table, replicated


def engine_inputs(mesh: Any, *, panel_geometry: bool = True) -> EngineInputs:
    """The fixture's engine inputs exactly as ``OrdinaryRuntime._load`` computes them."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.models.glm_moe_dsa.weights import bf16_resident_weights
    from glm_tpu.layers.rope import build_main_rope_table
    from glm_tpu.runner.programs import build_program_set

    config, raw, _ = fixture(mesh, panel_geometry=panel_geometry)
    decode, promote = (spec.fn for spec in build_program_set(mesh, config).wk)
    wk = []
    for layer_id in config.full_index_slots:
        dsa = raw.layers[layer_id].dsa
        wk.append(jax.block_until_ready(promote(decode(dsa.wk_bits_local, dsa.wk_scale_local))))
    rope = jax.device_put(np.asarray(build_main_rope_table(config)), NamedSharding(mesh, P()))
    return EngineInputs(config, raw, bf16_resident_weights(mesh, config, raw), tuple(wk), rope)


def prefill(mesh: Any, inputs: EngineInputs, programs: Any, prompt: list[int]) -> tuple[Any, Any]:
    """One prompt of at most 128 tokens through the production program set's cache initializer and
    one B114/B128 prefill block (``OrdinaryRuntime.generate``'s block rule); returns the decode
    state and the first token (``finish_ws32_batched_prefill``)."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.models.glm_moe_dsa.state import finish_batched_prefill

    if not 0 < len(prompt) <= 128:
        raise ValueError("one prefill block holds 1..128 prompt tokens")
    rows = 114 if len(prompt) <= 114 else 128

    def put(value: Any) -> Any:
        return jax.device_put(value, NamedSharding(mesh, P()))

    state = programs.cache_init.fn(put(np.int32(len(prompt))))
    tokens = put(np.asarray(list(prompt) + [-1] * (rows - len(prompt)), np.int32))
    result = programs.prefill[rows].fn(tokens, put(np.int32(len(prompt))), state, inputs.weights, inputs.wk,
                                       inputs.rope)
    return finish_batched_prefill(jax.block_until_ready(result))
