"""Retained D8/P1/P2 prefill composition with frozen cache/repair control flow."""
from ..greenfield.runtime import ws32_batched_prefill as runtime
from .function_bindings import bind_dependencies
from .dsa_candidates import prefill_dsa_one_pass_mapped


def _one_pass_selector(*args, key_tile=4096, paired_position_sort=False,
                       sorted_local_merge=False, **kwargs):
    if type(key_tile) is not int or not 128 <= key_tile <= 4096 or key_tile % 128:
        raise ValueError('prefill key tile must be a multiple of 128 up to 4096')
    if type(paired_position_sort) is not bool or type(sorted_local_merge) is not bool:
        raise ValueError('prefill sort options must be static booleans')
    if sorted_local_merge and kwargs.get('top_k', 2048) & (kwargs.get('top_k', 2048)-1):
        raise ValueError('sorted local merge requires power-of-two top_k')
    return prefill_dsa_one_pass_mapped(*args, **kwargs)


def build_ws32_prefill_challenger_program(mesh, config, **options):
    """Build the retained resident-BF16, local-LSE, one-pass-selection program.

    Rebind private function globals; never modify frozen modules. The release
    entry fixes the admitted profile and does not expose experimental variants.
    """
    from .bf16_resident import bf16_weight_specs
    from .prefill_bf16 import bind_bf16_prefill, _adapt_weights
    from .prefill_attention import prefill_index_share_lse_mapped

    layer, window = bind_bf16_prefill(
        bind_dependencies, _one_pass_selector, prefill_index_share_lse_mapped)
    body = bind_dependencies(runtime.ws32_batched_prefill_mapped,
        ws32_prefill_transformer_layer_mapped=layer,
        ws32_prefill_layer_window_mapped=window)

    def resident_body(tokens, count, state, weights, wk, rope, **kwargs):
        return body(tokens, count, state, _adapt_weights(weights), wk, rope, **kwargs)

    builder = bind_dependencies(runtime.build_ws32_batched_prefill_program,
        ws32_batched_prefill_mapped=resident_body,
        ws32_decoder_weight_specs=bf16_weight_specs)
    return builder(mesh, config, **options)
