"""Explicit, on-demand benchmark/inspection exports.

Importing one supported inspector must not eagerly load every historical
experiment. Retained names keep their original module targets; retired
diagnostics (PP16 feature2, M2048, transport, Gate C, StrategyND replay,
layer-0 association, live-SSA and sealed-runtime helpers) left with their
verified consumers and are recoverable at b667f00f. No model function bodies
or numerical source registrations are changed here.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    "WS32_ONE_LAYER_INPUT_SPECS": "ws32_one_layer",
    "WS32_ONE_LAYER_OUTPUT_SPEC": "ws32_one_layer",
    "WS32_PALLAS_REAL_LAYER_LIVE_CLOSURE_SHA256": "ws32_pallas_one_layer",
    "WS32_PALLAS_REAL_LAYER_OPTIMIZED_HLO_SHA256": "ws32_pallas_one_layer",
    "WS32_PALLAS_REAL_LAYER_STABLEHLO_SHA256": "ws32_pallas_one_layer",
    "Ws32DecoderHloReport": "ws32_decoder",
    "Ws32ExactDsaMaterializerHloReport": "ws32_decoder",
    "Ws32OneLayerHloReport": "ws32_one_layer",
    "Ws32PallasOneLayerHloReport": "ws32_pallas_one_layer",
    "build_ws32_one_layer_mapped": "ws32_one_layer",
    "build_ws32_pallas_one_layer_mapped": "ws32_pallas_one_layer",
    "validate_ws32_decoder_hlo": "ws32_decoder",
    "validate_ws32_exact_dsa_materializer_hlo": "ws32_decoder",
    "validate_ws32_one_layer_hlo": "ws32_one_layer",
    "validate_ws32_pallas_one_layer_hlo": "ws32_pallas_one_layer",
    "validate_ws32_topology_fleet": "ws32_one_layer",
    "ws32_one_layer_inputs": "ws32_one_layer",
}

__all__ = [
    "WS32_ONE_LAYER_INPUT_SPECS",
    "WS32_ONE_LAYER_OUTPUT_SPEC",
    "WS32_PALLAS_REAL_LAYER_LIVE_CLOSURE_SHA256",
    "WS32_PALLAS_REAL_LAYER_OPTIMIZED_HLO_SHA256",
    "WS32_PALLAS_REAL_LAYER_STABLEHLO_SHA256",
    "Ws32PallasOneLayerHloReport",
    "Ws32DecoderHloReport",
    "Ws32ExactDsaMaterializerHloReport",
    "Ws32OneLayerHloReport",
    "build_ws32_one_layer_mapped",
    "build_ws32_pallas_one_layer_mapped",
    "validate_ws32_one_layer_hlo",
    "validate_ws32_pallas_one_layer_hlo",
    "validate_ws32_decoder_hlo",
    "validate_ws32_exact_dsa_materializer_hlo",
    "validate_ws32_topology_fleet",
    "ws32_one_layer_inputs",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
