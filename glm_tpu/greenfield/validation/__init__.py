"""Independent greenfield correctness artifacts and comparisons."""

from .gate_c_oracle import (
    GateCOracleConfig,
    capture_gate_c_oracle,
    inspect_gate_c_oracle,
)
from .one_layer_oracle import (
    OneLayerOracleConfig,
    capture_one_layer_oracle,
    inspect_one_layer_oracle,
)

__all__ = (
    "GateCOracleConfig",
    "OneLayerOracleConfig",
    "capture_gate_c_oracle",
    "capture_one_layer_oracle",
    "inspect_gate_c_oracle",
    "inspect_one_layer_oracle",
)
