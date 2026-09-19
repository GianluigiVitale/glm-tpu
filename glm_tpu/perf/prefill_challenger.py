"""Opt-in P2 prefill composition with unchanged cache/commit/repair control flow.

The frozen composition functions have no dependency-injection hooks. Bind fresh
function objects to private copies of their globals instead of patching modules
or duplicating their 1,000+ lines of state management. Only the DSA selector is
replaced. The original functions, bytecode, defaults and module globals remain
unchanged, and both programs can be traced/executed in the same process.

This explicitly follows the pinned frozen call graph (selector -> DSA -> layer
-> window -> runtime -> builder). A changed call graph needs review and tests.
"""
from types import FunctionType

from ..greenfield.kernels import ws32_prefill_dsa as dsa
from ..greenfield.kernels import ws32_prefill_layer as layer
from ..greenfield.kernels import ws32_prefill_window as window
from ..greenfield.runtime import ws32_batched_prefill as runtime
from .dsa_candidates import prefill_dsa_one_pass_mapped


def _bind_dependencies(function, **replacements):
    if function.__closure__ is not None:
        raise ValueError("prefill composition bindings must be module-level functions")
    if not replacements.keys() <= function.__globals__.keys():
        raise ValueError("frozen prefill dependency name drifted")
    namespace = {**function.__globals__, **replacements}
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    bound.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    bound.__annotations__ = dict(function.__annotations__)
    bound.__module__ = __name__
    bound.__doc__ = function.__doc__
    return bound


def _one_pass_selector(*args, key_tile=4096, paired_position_sort=False,
                       sorted_local_merge=False, **kwargs):
    # These control only the frozen tile chain that P2 replaces. Precision,
    # causal bounds, geometry and top-k still pass through unchanged.
    if type(key_tile) is not int or not 128 <= key_tile <= 4096 or key_tile % 128:
        raise ValueError("prefill key tile must be a multiple of 128 up to 4096")
    if type(paired_position_sort) is not bool or type(sorted_local_merge) is not bool:
        raise ValueError("prefill sort options must be static booleans")
    if sorted_local_merge and kwargs.get("top_k", 2048) & (kwargs.get("top_k", 2048)-1):
        raise ValueError("sorted local merge requires power-of-two top_k")
    del key_tile, paired_position_sort, sorted_local_merge
    return prefill_dsa_one_pass_mapped(*args, **kwargs)


def build_ws32_prefill_challenger_program(mesh, config, **options):
    """Frozen greedy prefill API with the P2 one-pass exact selector.

    Raw FP8 weights, M64 repaired-key production, frozen attention/MLP and
    all-owner atomic state admission are unchanged. D8/P1 are not enabled here.
    """
    dsa_body = _bind_dependencies(dsa.ws32_prefill_dsa_mapped,
                                 ws32_prefill_dsa_from_query_mapped=_one_pass_selector)
    layer_body = _bind_dependencies(layer.ws32_prefill_transformer_layer_mapped,
                                   ws32_prefill_dsa_mapped=dsa_body)
    window_body = _bind_dependencies(window.ws32_prefill_layer_window_mapped,
                                    ws32_prefill_transformer_layer_mapped=layer_body)
    runtime_body = _bind_dependencies(runtime.ws32_batched_prefill_mapped,
                                     ws32_prefill_transformer_layer_mapped=layer_body,
                                     ws32_prefill_layer_window_mapped=window_body)
    builder = _bind_dependencies(runtime.build_ws32_batched_prefill_program,
                                 ws32_batched_prefill_mapped=runtime_body)
    return builder(mesh, config, **options)
