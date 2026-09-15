"""On-demand validation exports for the native engine and retained inspectors.

Importing one validator must not eagerly import every legacy capture workflow.
Retained names keep their original module targets. The Gate-D layer-0
legacy-capture validators, StrategyND replays, Gate C oracle and logprob
oracle were retired with their scripts and tests (recoverable at b667f00f).
The old __all__ also advertised an unbound Ws32LongContextOracle; that broken
package alias is omitted (the class remains in its own module). No validator
bodies, numerical source or admission identities change here.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    "LongContextOracleConfig": "long_context_oracle",
    "OneLayerOracleConfig": "one_layer_oracle",
    "ShortContextDsaOracleConfig": "short_context_dsa_oracle",
    "ShortContextOracleConfig": "short_context_oracle",
    "Ws32AdjudicatedDivergence": "ws32_short_context",
    "Ws32ShortContextOracle": "ws32_short_context",
    "bind_ws32_adjudication": "ws32_short_context",
    "capture_long_context_oracle": "long_context_oracle",
    "capture_one_layer_oracle": "one_layer_oracle",
    "capture_short_context_dsa_oracle": "short_context_dsa_oracle",
    "capture_short_context_oracle": "short_context_oracle",
    "compare_short_context_dsa_oracles": "short_context_dsa_oracle",
    "compare_ws32_dsa_step": "ws32_short_context",
    "compare_ws32_dsa_within_engine": "ws32_short_context",
    "compare_ws32_raw_tokens": "ws32_short_context",
    "inspect_long_context_oracle": "long_context_oracle",
    "inspect_one_layer_oracle": "one_layer_oracle",
    "inspect_short_context_dsa_oracle": "short_context_dsa_oracle",
    "inspect_short_context_oracle": "short_context_oracle",
    "load_legacy_bench_module": "long_context_oracle",
    "load_ws32_adjudicated_divergence": "ws32_short_context",
    "load_ws32_long_context_oracle": "long_context_oracle",
    "load_ws32_short_context_oracle": "ws32_short_context",
    "validate_ws32_cache_probe": "ws32_short_context",
}

__all__ = (
    "OneLayerOracleConfig",
    "LongContextOracleConfig",
    "ShortContextOracleConfig",
    "capture_long_context_oracle",
    "inspect_long_context_oracle",
    "ShortContextDsaOracleConfig",
    "Ws32AdjudicatedDivergence",
    "bind_ws32_adjudication",
    "Ws32ShortContextOracle",
    "capture_one_layer_oracle",
    "capture_short_context_oracle",
    "capture_short_context_dsa_oracle",
    "compare_short_context_dsa_oracles",
    "compare_ws32_dsa_step",
    "compare_ws32_dsa_within_engine",
    "load_legacy_bench_module",
    "load_ws32_long_context_oracle",
    "compare_ws32_raw_tokens",
    "inspect_one_layer_oracle",
    "inspect_short_context_oracle",
    "inspect_short_context_dsa_oracle",
    "load_ws32_adjudicated_divergence",
    "load_ws32_short_context_oracle",
    "validate_ws32_cache_probe",
)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
