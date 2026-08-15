"""Distinct raw-FP8 Pallas challenger for the bounded WS32 real layer."""

from __future__ import annotations

from typing import Any

import jax

from .ws32_one_layer import (
    WS32_ONE_LAYER_INPUT_SPECS,
    WS32_ONE_LAYER_OUTPUT_SPEC,
)
from ..kernels.reference.moe import GlmMoeNumericalContract
from ..kernels.ws32 import ws32_moe_pallas_from_routes_mapped


def build_ws32_pallas_one_layer_mapped(
    mesh: Any,
    *,
    contract: GlmMoeNumericalContract,
    interpret: bool = False,
) -> Any:
    """Return the default-off one-row Pallas WS32 shard-map body."""

    if contract.stage_size != 8 or contract.local_experts != 32:
        raise ValueError("real WS32 Pallas layer requires expert8 ownership")

    def body(*values: Any) -> Any:
        with jax.named_scope("greenfield_ws32_pallas_real_layer3"):
            return ws32_moe_pallas_from_routes_mapped(
                *values,
                contract=contract,
                expert_axis="expert",
                feature_axis="feature",
                interpret=interpret,
            )

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=WS32_ONE_LAYER_INPUT_SPECS,
        out_specs=WS32_ONE_LAYER_OUTPUT_SPEC,
        check_vma=False,
    )
