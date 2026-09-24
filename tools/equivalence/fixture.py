"""Frozen fixture v1: the small real-schema GLM checkpoint used by every CPU gate.

It reproduces the random-number call order of the historical test fixture
``tests/greenfield/runtime/ws32_prefill_cpu_fixture.fixture(mesh, panel_geometry=...)`` exactly
(generator seed 921; per layer q/kv-a, attention, indexer, dense, MoE; then embedding, LM head and
four trailing WK draws), but returns ``{checkpoint tensor name: numpy array}`` and binds it
through the production name-based binder, so it does not depend on pytree field names.

Geometry comes from the pinned GLM-5.3 ``config.json`` (identical to the GLM-5.2 file the
historical fixture read, except ``transformers_version``). ``model_id`` is pinned to the value the
181c013e parser produces, so the fixture geometry is byte-identical to the historical fixture's
and independent of later changes to the parser's model identity. ``tests/golden/data/fixture.json``
records that every leaf equals the historical fixture's (panel and non-panel geometry).

Frozen: never edit the generator; a new fixture is a new version with a new name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Any

import numpy as np

from .common import REPO

VERSION = "frozen_fixture_v1"
SEED = 921
CAPACITY = 1536
MODEL_ID = "zai-org/GLM-5.2-FP8"
CONFIG_SHA256 = "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69"
# Candidate locations of the pinned GLM-5.3 config.json, newest layout first.
CONFIG_CANDIDATES = (
    "glm_tpu/models/glm_moe_dsa/hf_config/config.json",
    "reference/hf-glm53/config.json",
)
INDEXER_TYPES = ("full", "full", "full", "shared", "shared", "shared", "full", "shared")
MLP_TYPES = ("dense",) * 3 + ("sparse",) * 5


@dataclass(frozen=True)
class FrozenFixture:
    config: Any                        # decoder config (production class)
    arrays: dict[str, np.ndarray]      # checkpoint tensor name -> host array
    wk_extra: tuple[np.ndarray, ...]   # the historical fixture's four trailing draws (not model weights)
    panel_geometry: bool


def config_json() -> dict[str, Any]:
    for candidate in CONFIG_CANDIDATES:
        path = REPO / candidate
        if path.is_file():
            raw = path.read_bytes()
            if sha256(raw).hexdigest() != CONFIG_SHA256:
                raise ValueError(f"{candidate} differs from the pinned GLM-5.3 config")
            return json.loads(raw)
    raise FileNotFoundError("pinned GLM-5.3 config.json not found")


def geometry(*, panel_geometry: bool = True) -> Any:
    from glm_tpu.optimized.geometry import ModelGeometry

    hidden = 1024 if panel_geometry else 512
    intermediate = 256 if panel_geometry else 128
    return replace(
        ModelGeometry.from_hf_config(config_json()),
        model_id=MODEL_ID, num_layers=8, hidden_size=hidden, attention_heads=16, kv_heads=16,
        q_lora_rank=128, num_routed_experts=64, moe_intermediate_size=intermediate,
        dense_intermediate_size=1024, vocab_size=256, dsa_top_k=128, mlp_layer_types=MLP_TYPES,
        indexer_types=INDEXER_TYPES,
    )


def decoder_config(*, panel_geometry: bool = True, capacity: int = CAPACITY) -> Any:
    from glm_tpu.optimized.ws32_decoder import Ws32DecoderConfig

    return Ws32DecoderConfig(geometry(panel_geometry=panel_geometry), capacity,
                             sparse_segment_block=128, host_main_rope_table=True)


def fixture_v1(*, panel_geometry: bool = True) -> FrozenFixture:
    """Build the frozen fixture on the host. Conversions use the historical fixture's own calls
    (NumPy/ml_dtypes for FP8 bits, ``jnp.asarray`` for BF16/FP32) so every byte is identical."""
    import jax.numpy as jnp
    import ml_dtypes

    config = decoder_config(panel_geometry=panel_geometry)
    hidden = config.geometry.hidden_size
    intermediate = config.geometry.moe_intermediate_size
    hb, ib = hidden // 128, intermediate // 128
    rng = np.random.default_rng(SEED)

    def bf(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(jnp.asarray(rng.normal(0, 0.1, shape), jnp.bfloat16))

    def bits(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(rng.normal(0, 0.02, shape), ml_dtypes.float8_e4m3fn).view(np.uint8)

    def scale(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(jnp.asarray(rng.uniform(0.5, 1.5, shape), jnp.float32))

    def ones(n: int) -> np.ndarray:
        return np.asarray(jnp.ones(n, jnp.bfloat16))

    arrays: dict[str, np.ndarray] = {}

    def put(name: str, value: np.ndarray) -> None:
        if name in arrays:
            raise AssertionError("duplicate fixture tensor " + name)
        arrays[name] = np.ascontiguousarray(value)

    for i in range(8):
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        # q/kv-a, in the historical draw order
        put(f"{p}.input_layernorm.weight", ones(hidden))
        put(f"{a}.q_a_proj.weight_bits", bits((128, hidden)))
        put(f"{a}.q_a_proj.scale_inv", scale((1, hb)))
        put(f"{a}.q_a_layernorm.weight", ones(128))
        put(f"{a}.kv_a_proj_with_mqa.weight_bits", bits((576, hidden)))
        put(f"{a}.kv_a_proj_with_mqa.scale_inv", scale((5, hb)))
        put(f"{a}.kv_a_layernorm.weight", ones(512))
        # attention
        put(f"{a}.q_b_proj.weight_bits", bits((4096, 128)))
        put(f"{a}.q_b_proj.scale_inv", scale((32, 1)))
        put(f"{a}.kv_b_proj.weight_bits", bits((7168, 512)))
        put(f"{a}.kv_b_proj.scale_inv", scale((56, 4)))
        put(f"{a}.o_proj.weight_bits", bits((hidden, 4096)))
        put(f"{a}.o_proj.scale_inv", scale((hb, 32)))
        # indexer (full layers only)
        if INDEXER_TYPES[i] == "full":
            put(f"{a}.indexer.wq_b.weight_bits", bits((4096, 128)))
            put(f"{a}.indexer.wq_b.scale_inv", scale((32, 1)))
            put(f"{a}.indexer.wk.weight_bits", bits((128, hidden)))
            put(f"{a}.indexer.wk.scale_inv", scale((1, hb)))
            put(f"{a}.indexer.k_norm.weight", ones(128))
            put(f"{a}.indexer.k_norm.bias", bf((128,)))
            put(f"{a}.indexer.weights_proj.weight", bf((32, hidden)))
        if MLP_TYPES[i] == "dense":
            put(f"{p}.mlp.gate_proj.weight_bits", bits((1024, hidden)))
            put(f"{p}.mlp.gate_proj.scale_inv", scale((8, hb)))
            put(f"{p}.mlp.up_proj.weight_bits", bits((1024, hidden)))
            put(f"{p}.mlp.up_proj.scale_inv", scale((8, hb)))
            put(f"{p}.mlp.down_proj.weight_bits", bits((hidden, 1024)))
            put(f"{p}.mlp.down_proj.scale_inv", scale((hb, 8)))
        else:
            put(f"{p}.mlp.gate.weight", bf((64, hidden)))
            put(f"{p}.mlp.gate.e_score_correction_bias", np.zeros(64, np.float32))
            put(f"{p}.mlp.experts.gate_proj.weight_bits", bits((64, intermediate, hidden)))
            put(f"{p}.mlp.experts.gate_proj.scale_inv", scale((64, ib, hb)))
            put(f"{p}.mlp.experts.up_proj.weight_bits", bits((64, intermediate, hidden)))
            put(f"{p}.mlp.experts.up_proj.scale_inv", scale((64, ib, hb)))
            put(f"{p}.mlp.experts.down_proj.weight_bits", bits((64, hidden, intermediate)))
            put(f"{p}.mlp.experts.down_proj.scale_inv", scale((64, hb, ib)))
            put(f"{p}.mlp.shared_experts.gate_proj.weight_bits", bits((intermediate, hidden)))
            put(f"{p}.mlp.shared_experts.gate_proj.scale_inv", scale((ib, hb)))
            put(f"{p}.mlp.shared_experts.up_proj.weight_bits", bits((intermediate, hidden)))
            put(f"{p}.mlp.shared_experts.up_proj.scale_inv", scale((ib, hb)))
            put(f"{p}.mlp.shared_experts.down_proj.weight_bits", bits((hidden, intermediate)))
            put(f"{p}.mlp.shared_experts.down_proj.scale_inv", scale((hb, ib)))
        put(f"{p}.post_attention_layernorm.weight", ones(hidden))
    put("model.embed_tokens.weight", bf((256, hidden)))
    put("lm_head.weight", bf((256, hidden)))
    put("model.norm.weight", ones(hidden))
    wk_extra = tuple(np.asarray(jnp.asarray(bf((128, hidden))).astype(jnp.float32)) for _ in range(4))
    return FrozenFixture(config, arrays, wk_extra, panel_geometry)


def name_spec_pairs(config: Any) -> list[tuple[str, Any]]:
    """Every production checkpoint tensor name with its partition spec, in binder tree order."""
    import jax

    from glm_tpu.optimized.ws32_decoder import ws32_decoder_weight_names, ws32_decoder_weight_specs

    pairs = jax.tree.map(lambda name, spec: (name, spec), ws32_decoder_weight_names(config),
                         ws32_decoder_weight_specs(config))
    return list(jax.tree.leaves(pairs, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2
                                and isinstance(x[0], str)))


def bind(mesh: Any, fixture: FrozenFixture) -> Any:
    """Place every named tensor with its production partition spec and bind by name."""
    import jax
    from jax.sharding import NamedSharding

    from glm_tpu.optimized.ws32_decoder import bind_ws32_decoder_weights

    pairs = name_spec_pairs(fixture.config)
    if {name for name, _ in pairs} != set(fixture.arrays):
        raise ValueError("frozen fixture tensor set differs from the production name tree")
    placed = {name: jax.device_put(fixture.arrays[name], NamedSharding(mesh, spec)) for name, spec in pairs}
    return bind_ws32_decoder_weights(placed, fixture.config)


def cpu_mesh() -> Any:
    import jax
    from jax.sharding import Mesh

    devices = jax.devices()
    if len(devices) != 32 or jax.default_backend() != "cpu":
        raise RuntimeError("the frozen fixture needs 32 forced CPU devices "
                           "(XLA_FLAGS=--xla_force_host_platform_device_count=32, JAX_PLATFORMS=cpu)")
    return Mesh(np.asarray(devices, object).reshape(8, 4), ("expert", "feature"))


def historical_equivalence() -> dict[str, Any]:
    """S0 proof that fixture v1 equals the historical fixture leaf for leaf (both geometries).

    Needs the historical fixture module (removed from main later); the result is recorded in
    ``tests/golden/data/fixture.json`` and only re-checked while that module exists.
    """
    import jax

    from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture as historical

    from .common import digest_json, leaf_digest, sha256_hex

    mesh = cpu_mesh()
    result: dict[str, Any] = {}
    for panel in (True, False):
        frozen = fixture_v1(panel_geometry=panel)
        bound = bind(mesh, frozen)
        config, weights, wk = historical(mesh, panel_geometry=panel)
        new = [leaf_digest(x) for x in jax.tree.leaves(bound)] + [leaf_digest(x) for x in frozen.wk_extra]
        old = [leaf_digest(x) for x in jax.tree.leaves(weights)] + [leaf_digest(x) for x in wk]
        new_shardings = [str(x.sharding.spec) for x in jax.tree.leaves(bound)]
        old_shardings = [str(x.sharding.spec) for x in jax.tree.leaves(weights)]
        result["panel" if panel else "non_panel"] = dict(
            leaves=len(new),
            digest=sha256_hex("\n".join(new)),
            historical_digest=sha256_hex("\n".join(old)),
            identical=new == old and new_shardings == old_shardings,
            geometry_equal=frozen.config == config,
            geometry_sha256=digest_json(frozen.config.geometry.to_dict()),
            config_fields=digest_json({k: getattr(frozen.config, k) for k in (
                "context_capacity", "logical_page_size", "packed_cache_width", "sparse_segment_block",
                "rms_norm_epsilon", "exact_dsa", "strategy_nd_dense", "host_main_rope_table")}),
        )
    return result


def main() -> int:
    from .common import emit, environment, require_cpu

    require_cpu()
    emit(dict(fixture=historical_equivalence(), environment=environment(), version=VERSION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
