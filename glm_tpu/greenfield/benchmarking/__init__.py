"""Synthetic mechanism benchmarks for the topology-first engine."""

from .collective_chain import (
    CollectiveChainConfig,
    CollectiveKind,
    LatencyDistribution,
    addressable_checksum,
    benchmark_collective_chain,
    build_collective_chain,
    collective_chain_hlo_policy,
    jax_dtype,
    latency_distribution,
)
from .transport_chain import (
    TransportChainConfig,
    TransportKind,
    benchmark_transport_chain,
    build_transport_chain,
    transport_chain_hlo_policy,
    validate_compiled_transport,
    validate_transport_pairs,
)
from .one_layer import (
    REAL_LAYER_OUTPUT_TOLERANCE,
    ROUTE_WEIGHT_TOLERANCE,
    TensorTolerance,
    compare_bounded_tensor,
    validate_real_layer_hlo,
)

__all__ = [
    "CollectiveChainConfig",
    "CollectiveKind",
    "LatencyDistribution",
    "REAL_LAYER_OUTPUT_TOLERANCE",
    "ROUTE_WEIGHT_TOLERANCE",
    "TensorTolerance",
    "addressable_checksum",
    "benchmark_collective_chain",
    "build_collective_chain",
    "collective_chain_hlo_policy",
    "compare_bounded_tensor",
    "jax_dtype",
    "latency_distribution",
    "TransportChainConfig",
    "TransportKind",
    "benchmark_transport_chain",
    "build_transport_chain",
    "transport_chain_hlo_policy",
    "validate_compiled_transport",
    "validate_real_layer_hlo",
    "validate_transport_pairs",
]
