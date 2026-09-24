"""BF16-resident non-routed weights and the decode layer bodies of the production engine.

Measured on the pod (docs/perf, phase 4): a one-row ``[1,1536] x [2048,1536]``
projection costs 43-60 us in every FP8-decoding form (Pallas or XLA) but
10 us from a resident BF16 table.  TPU v4 has no FP8 datapath, so the frozen
step spends ~40 ms of its 121 ms decoding e4m3 on the vector units.

The frozen kernels compute, per element, ``bf16(f32(bits) * scale_block)``
and feed that BF16 value to the MXU with FP32 accumulation.  Decoding the
same expression once at load time and keeping the BF16 table resident gives
bitwise-identical MXU operands, so this is an exact transformation of the
non-routed weights (attention q_a/kv_a/q_b/kv_b/o, DSA wq_b/wk, shared
experts, dense layers).  Routed experts (22 GB/chip) stay FP8 and go through
the route-grouped kernel.  Cost: about +1.8 GB HBM per chip.

Every function here mirrors one frozen body in ``kernels/ws32_layer.py`` /
``kernels/ws32.py`` with the FP8 Pallas projection replaced by ``dot_general``
at the same FP32-accumulate / BF16-round boundary.  The only numerical
difference is the MXU accumulation order inside one contraction, which the CPU
tests bound; the TPU pod run reports token/state agreement.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Mapping

from jax.sharding import PartitionSpec as P

from glm_tpu.config.cache import CacheConfig
from glm_tpu.layers.fp8 import _decode_program


# ----------------------------------------------------------------------------- weights
class Bf16QkvAWeights(NamedTuple):
    input_norm_weight_local: Any
    q_a_local: Any          # bf16 [q_lora, local_hidden]
    q_a_norm_weight: Any
    kv_a_local: Any         # bf16 [kv_lora + rope, local_hidden]
    kv_a_norm_weight: Any


class Bf16AttentionWeights(NamedTuple):
    q_b_local: Any          # bf16 [local_heads * qk_head, q_lora]
    kv_b_local: Any         # bf16 [local_heads * 448, 512]
    o_local: Any            # bf16 [local_hidden, local_heads * v_head]


class Bf16DsaWeights(NamedTuple):
    wq_b_local: Any         # bf16 [local_dsa_heads * 128, q_lora]
    wk_local: Any           # bf16 [128, local_hidden]
    key_norm_weight: Any
    key_norm_bias: Any
    head_weight_local: Any


class Bf16DenseWeights(NamedTuple):
    gate_local: Any
    up_local: Any
    down_local: Any


class Bf16MoeWeights(NamedTuple):
    router_weight_local: Any
    correction_bias_local: Any
    expert_gate_bits_local: Any
    expert_gate_scale_local: Any
    expert_up_bits_local: Any
    expert_up_scale_local: Any
    expert_down_bits_local: Any
    expert_down_scale_local: Any
    shared_gate_local: Any  # bf16
    shared_up_local: Any    # bf16
    shared_down_local: Any  # bf16


class Bf16LayerWeights(NamedTuple):
    qkv_a: Bf16QkvAWeights
    attention: Bf16AttentionWeights
    dsa: Bf16DsaWeights | None
    post_attention_norm_weight_local: Any
    dense: Bf16DenseWeights | None
    moe: Bf16MoeWeights | None


class Bf16DecoderWeights(NamedTuple):
    embedding_local: Any
    layers: tuple[Bf16LayerWeights, ...]
    final_norm_weight_local: Any
    lm_head_local: Any


def bf16_weight_specs(config: CacheConfig) -> Bf16DecoderWeights:
    """The checkpoint's partition specs, field by field: a resident table keeps its FP8 table's spec."""
    frozen = decoder_weight_specs(config)
    layers = []
    for spec in frozen.layers:
        q, a, d, dn, m = spec.qkv_a, spec.attention, spec.dsa, spec.dense, spec.moe
        layers.append(Bf16LayerWeights(
            Bf16QkvAWeights(q.input_norm_weight_local, q.q_a_bits_local, q.q_a_norm_weight,
                            q.kv_a_bits_local, q.kv_a_norm_weight),
            Bf16AttentionWeights(a.q_b_bits_local, a.kv_b_bits_local, a.o_bits_local),
            None if d is None else Bf16DsaWeights(d.wq_b_bits_local, d.wk_bits_local, d.key_norm_weight,
                                                  d.key_norm_bias, d.head_weight_local),
            spec.post_attention_norm_weight_local,
            None if dn is None else Bf16DenseWeights(dn.gate_bits_local, dn.up_bits_local, dn.down_bits_local),
            None if m is None else Bf16MoeWeights(
                m.router_weight_local, m.correction_bias_local,
                m.expert_gate_bits_local, m.expert_gate_scale_local, m.expert_up_bits_local,
                m.expert_up_scale_local, m.expert_down_bits_local, m.expert_down_scale_local,
                m.shared_gate_bits_local, m.shared_up_bits_local, m.shared_down_bits_local),
        ))
    return Bf16DecoderWeights(frozen.embedding_local, tuple(layers), frozen.final_norm_weight_local, frozen.lm_head_local)


def bf16_resident_weights(mesh: Any, config: CacheConfig, weights: Fp8DecoderWeights) -> Bf16DecoderWeights:
    """Decode every non-routed FP8 shard on its owning chip; routed experts pass through.

    Shards of every affected table start on 128-block boundaries in both
    dimensions, so decoding a local shard with its local scale shard equals
    decoding the global table.  One small jitted ``shard_map`` per table (a
    single program over all layers ran out of HBM at compile time on the pod:
    XLA kept every FP32 intermediate of 78 layers live at once).
    """

    frozen_specs = decoder_weight_specs(config)
    block = tuple(config.geometry.fp8_block_shape)

    def dec(bits: Any, scale: Any, spec: Any) -> Any:
        return _decode_program(mesh, bits, scale, spec, block)(bits, scale)

    layers = []
    for layer, spec in zip(weights.layers, frozen_specs.layers, strict=True):
        q, qs = layer.qkv_a, spec.qkv_a
        a, as_ = layer.attention, spec.attention
        qkv = Bf16QkvAWeights(
            q.input_norm_weight_local,
            dec(q.q_a_bits_local, q.q_a_scale_local, qs.q_a_bits_local),
            q.q_a_norm_weight,
            dec(q.kv_a_bits_local, q.kv_a_scale_local, qs.kv_a_bits_local),
            q.kv_a_norm_weight,
        )
        att = Bf16AttentionWeights(
            dec(a.q_b_bits_local, a.q_b_scale_local, as_.q_b_bits_local),
            dec(a.kv_b_bits_local, a.kv_b_scale_local, as_.kv_b_bits_local),
            dec(a.o_bits_local, a.o_scale_local, as_.o_bits_local),
        )
        dsa = None
        if layer.dsa is not None:
            d, ds = layer.dsa, spec.dsa
            dsa = Bf16DsaWeights(
                dec(d.wq_b_bits_local, d.wq_b_scale_local, ds.wq_b_bits_local),
                dec(d.wk_bits_local, d.wk_scale_local, ds.wk_bits_local),
                d.key_norm_weight, d.key_norm_bias, d.head_weight_local,
            )
        dense = None
        if layer.dense is not None:
            dn, dns = layer.dense, spec.dense
            dense = Bf16DenseWeights(
                dec(dn.gate_bits_local, dn.gate_scale_local, dns.gate_bits_local),
                dec(dn.up_bits_local, dn.up_scale_local, dns.up_bits_local),
                dec(dn.down_bits_local, dn.down_scale_local, dns.down_bits_local),
            )
        moe = None
        if layer.moe is not None:
            m, ms = layer.moe, spec.moe
            moe = Bf16MoeWeights(
                m.router_weight_local, m.correction_bias_local,
                m.expert_gate_bits_local, m.expert_gate_scale_local, m.expert_up_bits_local,
                m.expert_up_scale_local, m.expert_down_bits_local, m.expert_down_scale_local,
                dec(m.shared_gate_bits_local, m.shared_gate_scale_local, ms.shared_gate_bits_local),
                dec(m.shared_up_bits_local, m.shared_up_scale_local, ms.shared_up_bits_local),
                dec(m.shared_down_bits_local, m.shared_down_scale_local, ms.shared_down_bits_local),
            )
        layers.append(Bf16LayerWeights(qkv, att, dsa, layer.post_attention_norm_weight_local, dense, moe))
    return Bf16DecoderWeights(weights.embedding_local, tuple(layers), weights.final_norm_weight_local, weights.lm_head_local)


# ----------------------------------------------------------------------------- indexer WK programs
def build_wk_programs(
    mesh: Any, bits_spec: Any, scale_spec: Any, *, contract: Any
) -> tuple[Any, Any]:
    """Reuse the mandatory COMPLETED BF16 decode -> separate FP32 promotion."""
    import jax
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.layers.fp8 import decode_stage_local_prefill_index_wk_bf16, promote_stage_local_prefill_index_wk

    def decode(bits, scales):
        bits = lax.all_gather(bits, "feature", axis=1, tiled=True)
        scales = lax.all_gather(scales, "feature", axis=1, tiled=True)
        return decode_stage_local_prefill_index_wk_bf16(bits, scales, contract=contract)

    return (
        jax.jit(
            jax.shard_map(
                decode,
                mesh=mesh,
                in_specs=(bits_spec, scale_spec),
                out_specs=P(),
                check_vma=False,
            )
        ),
        jax.jit(
            jax.shard_map(
                lambda x: promote_stage_local_prefill_index_wk(x, contract=contract),
                mesh=mesh,
                in_specs=(P(),),
                out_specs=P(),
                check_vma=False,
            )
        ),
    )


class Fp8LayerWeights(NamedTuple):
    qkv_a: Fp8QkvAWeights
    attention: Fp8AttentionWeights
    dsa: Fp8DsaWeights | None
    post_attention_norm_weight_local: Any
    dense: Fp8DenseWeights | Fp8StrategyNdDenseWeights | None
    moe: Fp8MoeWeights | None


class Fp8DecoderWeights(NamedTuple):
    embedding_local: Any
    layers: tuple[Fp8LayerWeights, ...]
    final_norm_weight_local: Any
    lm_head_local: Any


def _qkv_specs() -> Fp8QkvAWeights:
    return Fp8QkvAWeights(
        P("feature"),
        P(None, "feature"),
        P(None, "feature"),
        P(),
        P(None, "feature"),
        P(None, "feature"),
        P(),
    )


def _attention_specs() -> Fp8AttentionWeights:
    return Fp8AttentionWeights(
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("feature", "expert"),
        P("feature", "expert"),
    )


def _dsa_specs() -> Fp8DsaWeights:
    return Fp8DsaWeights(
        P("expert", None),
        P("expert", None),
        P(None, "feature"),
        P(None, "feature"),
        P(),
        P(),
        P("expert", "feature"),
    )


def _dense_specs(
    *, strategy_nd: bool = False
) -> Fp8DenseWeights | Fp8StrategyNdDenseWeights:
    if strategy_nd:
        return Fp8StrategyNdDenseWeights(
            P("expert", None, None),
            P("expert", None, None),
            P("expert", None, "feature"),
            P("expert", None, "feature"),
        )
    return Fp8DenseWeights(
        P("expert", "feature"),
        P("expert", "feature"),
        P("expert", "feature"),
        P("expert", "feature"),
        P("feature", "expert"),
        P("feature", "expert"),
    )


def _moe_specs() -> Fp8MoeWeights:
    return Fp8MoeWeights(
        P("expert", "feature"),
        P("expert"),
        P("expert", None, "feature"),
        P("expert", None, "feature"),
        P("expert", None, "feature"),
        P("expert", None, "feature"),
        P("expert", "feature", None),
        P("expert", "feature", None),
        P(None, "feature"),
        P(None, "feature"),
        P(None, "feature"),
        P(None, "feature"),
        P("feature", None),
        P("feature", None),
    )


def decoder_weight_specs(config: CacheConfig) -> Fp8DecoderWeights:
    """Return the exact pytree of global partition specifications."""

    layers = []
    for indexer_kind, mlp_kind in zip(
        config.geometry.indexer_types,
        config.geometry.mlp_layer_types,
        strict=True,
    ):
        layers.append(
            Fp8LayerWeights(
                _qkv_specs(),
                _attention_specs(),
                _dsa_specs() if indexer_kind == "full" else None,
                P("feature"),
                (
                    _dense_specs(strategy_nd=config.strategy_nd_dense)
                    if mlp_kind == "dense"
                    else None
                ),
                _moe_specs() if mlp_kind == "sparse" else None,
            )
        )
    return Fp8DecoderWeights(
        P("expert", "feature"),
        tuple(layers),
        P("feature"),
        P("expert", "feature"),
    )


def _fp8_names(prefix: str) -> tuple[str, str]:
    return f"{prefix}.weight_bits", f"{prefix}.scale_inv"


def decoder_weight_names(config: CacheConfig) -> Fp8DecoderWeights:
    """Return the exact final-layout tensor name for every decoder input."""

    layers = []
    for layer_id, (indexer_kind, mlp_kind) in enumerate(
        zip(
            config.geometry.indexer_types,
            config.geometry.mlp_layer_types,
            strict=True,
        )
    ):
        prefix = f"model.layers.{layer_id}"
        attention_prefix = f"{prefix}.self_attn"
        q_a_bits, q_a_scale = _fp8_names(
            f"{attention_prefix}.q_a_proj"
        )
        kv_a_bits, kv_a_scale = _fp8_names(
            f"{attention_prefix}.kv_a_proj_with_mqa"
        )
        q_b_bits, q_b_scale = _fp8_names(
            f"{attention_prefix}.q_b_proj"
        )
        kv_b_bits, kv_b_scale = _fp8_names(
            f"{attention_prefix}.kv_b_proj"
        )
        o_bits, o_scale = _fp8_names(f"{attention_prefix}.o_proj")
        dsa = None
        if indexer_kind == "full":
            wq_b_bits, wq_b_scale = _fp8_names(
                f"{attention_prefix}.indexer.wq_b"
            )
            wk_bits, wk_scale = _fp8_names(
                f"{attention_prefix}.indexer.wk"
            )
            dsa = Fp8DsaWeights(
                wq_b_bits,
                wq_b_scale,
                wk_bits,
                wk_scale,
                f"{attention_prefix}.indexer.k_norm.weight",
                f"{attention_prefix}.indexer.k_norm.bias",
                f"{attention_prefix}.indexer.weights_proj.weight",
            )
        dense = None
        moe = None
        if mlp_kind == "dense":
            if config.strategy_nd_dense:
                strategy_prefix = f"{prefix}.mlp.strategy_nd"
                dense = Fp8StrategyNdDenseWeights(
                    f"{strategy_prefix}.merged_gate_up.weight_bits_in_out",
                    f"{strategy_prefix}.merged_gate_up.scale_inv_in_out",
                    f"{strategy_prefix}.down.weight_bits_in_out",
                    f"{strategy_prefix}.down.scale_inv_in_out",
                )
            else:
                gate_bits, gate_scale = _fp8_names(f"{prefix}.mlp.gate_proj")
                up_bits, up_scale = _fp8_names(f"{prefix}.mlp.up_proj")
                down_bits, down_scale = _fp8_names(f"{prefix}.mlp.down_proj")
                dense = Fp8DenseWeights(
                    gate_bits,
                    gate_scale,
                    up_bits,
                    up_scale,
                    down_bits,
                    down_scale,
                )
        else:
            expert_gate_bits, expert_gate_scale = _fp8_names(
                f"{prefix}.mlp.experts.gate_proj"
            )
            expert_up_bits, expert_up_scale = _fp8_names(
                f"{prefix}.mlp.experts.up_proj"
            )
            expert_down_bits, expert_down_scale = _fp8_names(
                f"{prefix}.mlp.experts.down_proj"
            )
            shared_gate_bits, shared_gate_scale = _fp8_names(
                f"{prefix}.mlp.shared_experts.gate_proj"
            )
            shared_up_bits, shared_up_scale = _fp8_names(
                f"{prefix}.mlp.shared_experts.up_proj"
            )
            shared_down_bits, shared_down_scale = _fp8_names(
                f"{prefix}.mlp.shared_experts.down_proj"
            )
            moe = Fp8MoeWeights(
                f"{prefix}.mlp.gate.weight",
                f"{prefix}.mlp.gate.e_score_correction_bias",
                expert_gate_bits,
                expert_gate_scale,
                expert_up_bits,
                expert_up_scale,
                expert_down_bits,
                expert_down_scale,
                shared_gate_bits,
                shared_gate_scale,
                shared_up_bits,
                shared_up_scale,
                shared_down_bits,
                shared_down_scale,
            )
        layers.append(
            Fp8LayerWeights(
                Fp8QkvAWeights(
                    f"{prefix}.input_layernorm.weight",
                    q_a_bits,
                    q_a_scale,
                    f"{attention_prefix}.q_a_layernorm.weight",
                    kv_a_bits,
                    kv_a_scale,
                    f"{attention_prefix}.kv_a_layernorm.weight",
                ),
                Fp8AttentionWeights(
                    q_b_bits,
                    q_b_scale,
                    kv_b_bits,
                    kv_b_scale,
                    o_bits,
                    o_scale,
                ),
                dsa,
                f"{prefix}.post_attention_layernorm.weight",
                dense,
                moe,
            )
        )
    return Fp8DecoderWeights(
        "model.embed_tokens.weight",
        tuple(layers),
        "model.norm.weight",
        "lm_head.weight",
    )


def _weight_name_leaves(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return tuple(
            name for item in value for name in _weight_name_leaves(item)
        )
    raise TypeError("WS32 weight-name tree contains a non-string leaf")


def _bind_weight_name_tree(value: object, arrays: Mapping[str, Any]) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return arrays[value]
    if isinstance(value, tuple):
        children = tuple(_bind_weight_name_tree(item, arrays) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*children)
        return children
    raise TypeError("WS32 weight-name tree contains a non-string leaf")


def bind_decoder_weights(
    arrays: Mapping[str, Any],
    config: CacheConfig,
) -> Fp8DecoderWeights:
    """Bind one exact manifest tensor map to the typed 78-layer decoder."""

    names = decoder_weight_names(config)
    leaves = _weight_name_leaves(names)
    expected = set(leaves)
    if len(expected) != len(leaves):
        raise ValueError("WS32 decoder weight names are not bijective")
    observed = set(arrays)
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        unexpected = sorted(observed - expected)[:5]
        raise ValueError(
            "WS32 decoder tensor set drifted: "
            f"missing={missing}, unexpected={unexpected}"
        )
    result = _bind_weight_name_tree(names, arrays)
    if not isinstance(result, Fp8DecoderWeights):
        raise AssertionError("WS32 decoder binding lost its typed root")
    return result


class Fp8QkvAWeights(NamedTuple):
    input_norm_weight_local: Any
    q_a_bits_local: Any
    q_a_scale_local: Any
    q_a_norm_weight: Any
    kv_a_bits_local: Any
    kv_a_scale_local: Any
    kv_a_norm_weight: Any


class Fp8DsaWeights(NamedTuple):
    wq_b_bits_local: Any
    wq_b_scale_local: Any
    wk_bits_local: Any
    wk_scale_local: Any
    key_norm_weight: Any
    key_norm_bias: Any
    head_weight_local: Any


class Fp8AttentionWeights(NamedTuple):
    q_b_bits_local: Any
    q_b_scale_local: Any
    kv_b_bits_local: Any
    kv_b_scale_local: Any
    o_bits_local: Any
    o_scale_local: Any


class Fp8DenseWeights(NamedTuple):
    gate_bits_local: Any
    gate_scale_local: Any
    up_bits_local: Any
    up_scale_local: Any
    down_bits_local: Any
    down_scale_local: Any


class Fp8StrategyNdDenseWeights(NamedTuple):
    """Four ordered legacy-rank shards in their final WS32 ownership."""

    merged_bits_in_out_local: Any
    merged_scale_in_out_local: Any
    down_bits_in_out_local: Any
    down_scale_in_out_local: Any


class Fp8MoeWeights(NamedTuple):
    router_weight_local: Any
    correction_bias_local: Any
    expert_gate_bits_local: Any
    expert_gate_scale_local: Any
    expert_up_bits_local: Any
    expert_up_scale_local: Any
    expert_down_bits_local: Any
    expert_down_scale_local: Any
    shared_gate_bits_local: Any
    shared_gate_scale_local: Any
    shared_up_bits_local: Any
    shared_up_scale_local: Any
    shared_down_bits_local: Any
    shared_down_scale_local: Any
