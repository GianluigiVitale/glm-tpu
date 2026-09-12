"""Opt-in sampled heads on the existing native prefill and decode bodies.

No alternative layer implementation or serial teacher forcing. Each invocation
receives one replicated FP32 random draw as DATA, so changing request/token does
not recompile the model. Only the final prefill block consumes its draw. The
caller must validate all-owner health before emission, stop at EOS/cap, and
preserve the whole cache plus request RNG identity when resuming.

These builders do not authorize a protected launch or inherit old HLO hashes.
The sampled path requires its own HLO/memory admission and TPU validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
from jax.sharding import PartitionSpec as P

from ..kernels.ws32_sampling import NucleusConfig, ws32_split_nucleus_sample_mapped
from . import ws32_batched_prefill as prefill
from . import ws32_decoder as decoder


@dataclass(frozen=True, slots=True)
class Ws32SampledPrefillProgram:
    # The template is used for the original option validation/metadata only;
    # its greedy executable is never invoked by this program.
    template: prefill.Ws32BatchedPrefillProgram
    sampling: NucleusConfig
    execute: Any


@dataclass(frozen=True, slots=True)
class Ws32SampledDecoderProgram:
    config: decoder.Ws32DecoderConfig
    sampling: NucleusConfig
    execute: Any
    observe: Any
    probe_cache_write: Any


def _require_sampling(sampling: NucleusConfig) -> None:
    if not isinstance(sampling, NucleusConfig):
        raise ValueError("explicit NucleusConfig required for sampled requests")


def build_ws32_sampled_prefill_program(
    mesh: Any,
    config: decoder.Ws32DecoderConfig,
    *,
    sampling: NucleusConfig,
    block_rows: int,
    **prefill_options: Any,
) -> Ws32SampledPrefillProgram:
    """Original prefill arguments, followed by one scalar FP32 uniform.

    All prefill options are checked by the ORIGINAL builder. Its program is
    only constructed, not lowered, compiled, or run. No additional model copy.
    """
    _require_sampling(sampling)
    template = prefill.build_ws32_batched_prefill_program(
        mesh, config, block_rows=block_rows, **prefill_options
    )

    def body(tokens: Any, count: Any, state: Any, weights: Any,
             wk: Any, rope: Any, uniform: Any) -> prefill.Ws32BatchedPrefillResult:
        if tokens.shape != (block_rows,):
            raise ValueError("sampled prefill static row count drifted")
        return prefill.ws32_batched_prefill_mapped(
            tokens, count, state, weights, wk, rope, config=config,
            final_sample=partial(ws32_split_nucleus_sample_mapped,
                                 uniform=uniform, config=sampling),
            **prefill_options,
        )

    specs = prefill.ws32_batched_prefill_state_specs()
    execute = jax.jit(jax.shard_map(
        body, mesh=mesh,
        in_specs=(P(), P(), specs, decoder.ws32_decoder_weight_specs(config),
                  tuple(P() for _ in config.full_index_slots), P(), P()),
        out_specs=prefill.Ws32BatchedPrefillResult(specs, P()), check_vma=False,
    ))
    return Ws32SampledPrefillProgram(template, sampling, execute)


def build_ws32_sampled_decoder_program(
    mesh: Any,
    config: decoder.Ws32DecoderConfig,
    *,
    sampling: NucleusConfig,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32SampledDecoderProgram:
    """Original decode arguments, followed by one scalar FP32 uniform.

    Exact-DSA weights and host rotary table are present precisely when the
    original config requires them. Observer uses the SAME sampled head.
    """
    _require_sampling(sampling)
    template = decoder.build_ws32_decoder_program(
        mesh, config, sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
    )
    specs = (P(), decoder.ws32_decoder_state_specs(),
             decoder.ws32_decoder_weight_specs(config))
    if config.exact_dsa:
        specs += (decoder.ws32_exact_dsa_specs(config),)
    if config.host_main_rope_table:
        specs += (P(),)
    specs += (P(),)

    def body(tokens: Any, state: Any, weights: Any, *extra: Any,
             observe: bool) -> Any:
        expected = int(config.exact_dsa) + int(config.host_main_rope_table) + 1
        if len(extra) != expected:
            raise ValueError("sampled decoder input/config presence drifted")
        exact = extra[0] if config.exact_dsa else None
        rope = extra[int(config.exact_dsa)] if config.host_main_rope_table else None
        scope = ("greenfield_ws32_complete_decoder_dsa_observer" if observe else
                 "greenfield_ws32_complete_decoder")
        with jax.named_scope(scope):
            result, observation, _ = decoder._ws32_decode_impl(
                tokens, state, weights, config=config, exact_dsa_weights=exact,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret, observe_dsa=observe,
                main_rope_table=rope,
                final_sample=partial(ws32_split_nucleus_sample_mapped,
                                     uniform=extra[-1], config=sampling),
            )
        return decoder.Ws32ObservedDecodeStepResult(result, observation) if observe else result

    execute = jax.jit(jax.shard_map(
        partial(body, observe=False), mesh=mesh, in_specs=specs,
        out_specs=decoder.ws32_decode_result_specs(), check_vma=False,
    ))
    observe = jax.jit(jax.shard_map(
        partial(body, observe=True), mesh=mesh, in_specs=specs,
        out_specs=decoder.ws32_observed_decode_result_specs(), check_vma=False,
    ))
    return Ws32SampledDecoderProgram(config, sampling, execute, observe,
                                    template.probe_cache_write)
