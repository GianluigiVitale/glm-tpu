"""Committed native history and disposable IndexShare continuation.

Refresh always starts at the committed draft root and uses target hidden rows.
The recurrent proposal is only a guess; it never becomes accepted history.
"""
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from ..greenfield.runtime import ws32_decoder as decoder
from .mtp_draft import draft_mapped, mtp_weight_specs
from .speculative_verify import commit_prefix_mapped, proposal_specs


class NativeState(NamedTuple):
    cache: Any
    next_token: Any
    normalized_hidden_local: Any


class TargetHistory(NamedTuple):
    position: Any
    shifted_tokens: Any
    normalized_hidden_local: Any
    contract_valid: Any


def native_specs():
    return NativeState(decoder.ws32_decoder_state_specs(), P(), P(None, 'feature'))


def history_specs():
    return TargetHistory(P(), P(), P(None, 'feature'), P())


def make_native_state(mesh, config):
    return NativeState(decoder.make_ws32_initial_state(mesh, config),
        jax.device_put(jnp.array([-1], jnp.int32), NamedSharding(mesh, P())),
        jax.device_put(jnp.zeros((1, config.geometry.hidden_size), jnp.bfloat16),
                       NamedSharding(mesh, P(None, 'feature'))))


def commit_native_mapped(original, proposal, count, *, config):
    cache = commit_prefix_mapped(original.cache, proposal, count, config=config)
    changed = cache.contract_valid[0] & (count > 0)
    last = jnp.clip(count - 1, 0, proposal.predictions.size - 1)
    return NativeState(cache,
        jnp.where(changed, proposal.predictions[last][None], original.next_token),
        jnp.where(changed, proposal.normalized_hidden_local[last][None],
                  original.normalized_hidden_local))


def refresh_mapped(original, history, count, weights, rope, *, config, **options):
    """Bootstrap or refresh from aligned target rows, committing only count."""
    if (history.position.shape != (1,) or history.position.dtype != jnp.int32
            or history.contract_valid.shape != history.shifted_tokens.shape
            or history.contract_valid.dtype != jnp.bool_):
        raise ValueError('target history requires an int32 frontier and one health bit per row')
    proposal = draft_mapped(history.shifted_tokens, history.normalized_hidden_local,
        original.cache, weights, rope, config=config, **options)
    valid = jnp.all(history.contract_valid) & (history.position[0] == original.cache.position[0])
    valid = lax.pmin(valid.astype(jnp.int32), ('expert', 'feature')) != 0
    proposal = proposal._replace(contract_valid=proposal.contract_valid & valid)
    return commit_native_mapped(original, proposal, count, config=config)


def extend_mapped(original, weights, rope, *, config, **options):
    """Return an uncommitted second guess; do not change the native root."""
    return draft_mapped(original.next_token, original.normalized_hidden_local,
        original.cache, weights, rope, config=config, index_share=True, **options)


def _mesh(mesh):
    if tuple(mesh.axis_names) != ('expert', 'feature') or tuple(mesh.devices.shape) != (8, 4):
        raise ValueError('native state requires expert8 by feature4 mesh')


def build_native_refresh(mesh, config, **options):
    _mesh(mesh)
    return jax.jit(jax.shard_map(partial(refresh_mapped, config=config, **options), mesh=mesh,
        in_specs=(native_specs(), history_specs(), P(), mtp_weight_specs(config), P()),
        out_specs=native_specs(), check_vma=False))


def build_native_extend(mesh, config, **options):
    _mesh(mesh)
    return jax.jit(jax.shard_map(partial(extend_mapped, config=config, **options), mesh=mesh,
        in_specs=(native_specs(), mtp_weight_specs(config), P()),
        out_specs=proposal_specs(), check_vma=False))


def inputs_mapped(pending, original, weights, rope, *, config, rows, **options):
    """Device-only proposal IDs; unhealthy recurrence poisons target admission."""
    if rows not in (1, 2, 3) or pending.shape != (1,) or pending.dtype != jnp.int32:
        raise ValueError('native speculation requires one pending token and 1..3 rows')
    valid = original.cache.contract_valid[0]
    parts = [pending]
    if rows >= 2:
        parts.append(original.next_token)
    if rows == 3:
        second = extend_mapped(original, weights, rope, config=config, **options)
        parts.append(second.predictions)
        valid &= jnp.all(second.contract_valid)
    ids = jnp.concatenate(parts)
    valid &= jnp.all((ids >= 0) & (ids < config.geometry.vocab_size))
    valid = lax.pmin(valid.astype(jnp.int32), ('expert', 'feature')) != 0
    return jnp.where(valid, ids, jnp.int32(-1))


def build_native_inputs(mesh, config, *, rows, **options):
    _mesh(mesh)
    return jax.jit(jax.shard_map(partial(inputs_mapped, config=config, rows=rows, **options),
        mesh=mesh, in_specs=(P(), native_specs(), mtp_weight_specs(config), P()),
        out_specs=P(), check_vma=False))
