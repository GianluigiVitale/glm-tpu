"""Unwired completed-prefix/suffix programs for the new prefill window.

Calls must be completed outside an enclosing JIT; inlining these builders does
not preserve the proposed materialization contract. No worker/launch admission.
Reuses real prefix/MLP arithmetic and keeps all intermediate arrays on device.
"""

from __future__ import annotations

from typing import Any, Sequence


def build_completed_window_programs(
    mesh: Any, input_specs: tuple[Any, ...], **options: Any
) -> tuple[Any, Any]:
    """One narrow prefix and one suffix callable compiled at narrow/wide rows."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
    from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_mlp_mapped
    from scripts.greenfield.prefill_layer_programs import build_layer_programs

    if any(
        k in options
        for k in (
            "candidate_window",
            "capture_boundaries",
            "completed_prefix",
            "prefix_only",
        )
    ):
        raise ValueError("completed window owns its prefix/capture mode")
    prefix, _ = build_layer_programs(
        mesh, input_specs, completed_prefix=True, **options
    )
    contract = options.get("moe_contract", GlmMoeNumericalContract(stage_size=8))
    interpret = options.get("linear_interpret", False)

    def suffix(normalized: Any, live: Any, dense: Any, moe: Any, health: Any) -> tuple:
        rows = normalized.shape[0]
        if health.shape != (1, 1, rows) or health.dtype != jnp.bool_:
            raise ValueError("suffix requires actual per-owner prefix health")
        if live.shape != (rows,) or live.dtype != jnp.bool_:
            raise ValueError("suffix requires row-aligned live mask")
        output, ids, weights, valid = ws32_prefill_mlp_mapped(
            normalized,
            live,
            dense,
            moe,
            moe_contract=contract,
            linear_interpret=interpret,
        )
        output = jnp.where(live[:, None], output, 0)
        valid = health[0, 0] & valid & (~live | jnp.all(jnp.isfinite(output), axis=1))
        return output, ids, weights, valid[None, None]

    mapped = jax.jit(
        jax.shard_map(
            suffix,
            mesh=mesh,
            in_specs=(
                P(None, "feature"),
                P(),
                input_specs[16],
                input_specs[17],
                P("expert", "feature", None),
            ),
            out_specs=(P(None, "feature"), P(), P(), P("expert", "feature", None)),
            check_vma=False,
        )
    )
    return prefix, mapped


def prefix_inputs(values: tuple, tile: int, previous: tuple | None = None) -> tuple:
    """Adapt the existing causal window slicing to the exact10-output prefix."""
    import jax.numpy as jnp

    if (
        len(values) != 20
        or values[0].shape[0] != 128
        or type(tile) is not int
        or not 0 <= tile < 4
    ):
        raise ValueError("completed window requires128 rows/20 inputs/tile0..3")
    if (previous is None) != (tile == 0) or (
        previous is not None and len(previous) != 10
    ):
        raise ValueError("later prefix needs actual prior10-output cache proposal")
    if any(
        getattr(values[i], "shape", None) != ()
        or getattr(values[i], "dtype", None) != jnp.int32
        for i in (8, 9)
    ):
        raise ValueError("prefix offset and valid_rows must be scalar int32")
    capacity = values[2].shape[1] * values[2].shape[2] * 8
    start = jnp.clip(values[8], 0, capacity - 1)
    count = jnp.clip(values[9], 0, 128)
    span = (values[8] == start) & (values[9] == count) & (count <= capacity - start)
    first, last = tile * 32, (tile + 1) * 32
    result = list(values)
    for i in (0, 1, 5, 6, 7, 19):
        result[i] = values[i][first:last]
    result[8] = start + jnp.minimum(jnp.int32(first), capacity - 1 - start)
    result[9] = jnp.clip(count - first, 0, 32)
    result[18] = values[18][:, :, first:last] & span
    if previous is not None:
        result[2:5] = previous[2:5]
    return tuple(result)


def suffix_inputs(
    prefixes: Sequence[tuple], valid_rows: Any, dense: Any, moe: Any
) -> tuple:
    """Device-only concatenation; assembly/dispatch costs must remain in wall."""
    import jax.numpy as jnp

    if not 1 <= len(prefixes) <= 4 or any(len(p) != 10 for p in prefixes):
        raise ValueError("one to four completed prefix outputs required")
    if (
        getattr(valid_rows, "shape", None) != ()
        or getattr(valid_rows, "dtype", None) != jnp.int32
    ):
        raise ValueError("suffix valid_rows must be scalar int32")
    normalized = jnp.concatenate([p[0] for p in prefixes], axis=0)
    health = jnp.concatenate([p[8] for p in prefixes], axis=2)
    rows = normalized.shape[0]
    valid = (valid_rows >= 0) & (valid_rows <= rows)
    live = jnp.arange(rows, dtype=jnp.int32) < jnp.clip(valid_rows, 0, rows)
    return normalized, live, dense, moe, health & valid


def assemble_result(prefixes: Sequence[tuple], suffix: tuple) -> tuple:
    """Return12 existing fields; these remain proposals, not committed state."""
    import jax.numpy as jnp

    if len(prefixes) != 4 or any(len(p) != 10 for p in prefixes) or len(suffix) != 4:
        raise ValueError("complete128-row prefix/suffix result required")

    def joined(i: int) -> Any:
        return jnp.concatenate([p[i] for p in prefixes], axis=0)

    return (
        suffix[0],
        joined(1),
        *prefixes[-1][2:5],
        joined(5),
        joined(6),
        joined(7),
        suffix[1],
        suffix[2],
        suffix[3],
        joined(9),
    )
