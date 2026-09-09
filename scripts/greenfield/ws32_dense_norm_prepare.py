"""Four abstract compiler jobs using the existing selected dense metadata loader.

No checkpoint payload read, runtime initialization or execution authorization.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.greenfield import ws32_dense_frontier_prepare as original
from scripts.greenfield.ws32_dense_norm_boundary import (
    build_capture_program,
    build_owner_packet_suffix,
)
from scripts.greenfield.ws32_dense_norm_protocol import PROGRAMS


def prepare(mesh: Any, *, repo: Path) -> original.PreparedDenseFrontier:
    prepared = original.prepare(mesh, repo=repo)
    return replace(prepared, program=build_capture_program(mesh, prepared.config))


def compiler_programs(prepared: original.PreparedDenseFrontier, mesh: Any) -> tuple:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    jobs = original.compiler_programs(prepared, mesh)
    packet = jax.ShapeDtypeStruct(
        (8, 4, 128, prepared.config.geometry.hidden_size // 4),
        jnp.bfloat16,
        sharding=NamedSharding(mesh, P("expert", "feature")),
    )
    scalar = jax.ShapeDtypeStruct((), jnp.int32, sharding=NamedSharding(mesh, P()))
    result = (
        *jobs[:2],
        ("dense01_norm", prepared.program, prepared.inputs),
        (
            "dense_suffix",
            build_owner_packet_suffix(mesh, prepared.config),
            (packet, scalar, scalar, prepared.inputs[6][0].dense),
        ),
    )
    if tuple(name for name, _, _ in result) != PROGRAMS:
        raise ValueError("norm compiler inventory differs")
    return result
