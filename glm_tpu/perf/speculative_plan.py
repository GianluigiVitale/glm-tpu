"""Fuse greedy acceptance with verification; no host cache commit or delivery.

The upstream TPU device rejection sampler motivates this boundary. This uses
our existing greedy rule, not upstream sampling probabilities or cache layout.
The host still authenticates the compact plan across the fleet before either
committed root changes. Experimental builders do not select a serving default.
"""
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ..greenfield.runtime import ws32_decoder as decoder
from .bf16_resident import bf16_weight_specs
from .speculative_accept import greedy_acceptance
from .speculative_verify import proposal_specs, verify_mapped


class PlannedVerification(NamedTuple):
    proposal: Any
    metadata: Any
    count: Any


def plan_verification(tokens, proposal, position, remaining, *, vocab_size, eos_ids):
    """Return one compact host/fleet vector and the device-resident commit count.

    Metadata: position, budget, count, accepted drafts, reason, health, inputs,
    predictions. Reason is 0=continue, 1=EOS, 2=budget. Failed health authorizes
    zero cache rows, even if the token comparison itself would accept a draft.
    """
    if (position.shape != (1,) or position.dtype != jnp.int32
            or proposal.contract_valid.shape != tokens.shape
            or proposal.contract_valid.dtype != jnp.bool_
            or type(vocab_size) is not int or not 0 < vocab_size < 2**31
            or not isinstance(eos_ids, tuple)
            or any(type(x) is not int or not 0 <= x < vocab_size for x in eos_ids)):
        raise ValueError('device plan requires scalar frontier, row health and valid vocabulary/EOS')
    acceptance = greedy_acceptance(tokens, proposal.predictions, remaining,
                                   eos_token_ids=eos_ids)
    valid = (acceptance.valid & jnp.all(proposal.contract_valid)
             & (position[0] >= 0) & (remaining > 0)
             & jnp.all(tokens < vocab_size) & jnp.all(proposal.predictions < vocab_size))
    count = jnp.where(valid, acceptance.emitted_count, jnp.int32(0))
    accepted = jnp.where(valid, acceptance.accepted_draft_count, jnp.int32(0))
    reason = jnp.where(acceptance.stopped_on_eos, 1,
                       jnp.where(acceptance.stopped_on_length, 2, 0)).astype(jnp.int32)
    header = jnp.stack((position[0], remaining, count, accepted,
                        jnp.where(valid, reason, 0), valid.astype(jnp.int32)))
    return PlannedVerification(proposal, jnp.concatenate((header, tokens, proposal.predictions)), count)


def planned_verify_mapped(tokens, state, weights, rope, remaining, *, config, eos_ids, **options):
    proposal = verify_mapped(tokens, state, weights, rope, config=config, **options)
    return plan_verification(tokens, proposal, state.position, remaining,
                             vocab_size=config.geometry.vocab_size, eos_ids=eos_ids)


def build_planned_verifier(mesh, config, *, eos_ids, **options):
    """Compile target forward and acceptance together; fresh numerical gate needed."""
    if tuple(mesh.axis_names) != ('expert', 'feature') or tuple(mesh.devices.shape) != (8, 4):
        raise ValueError('planned verifier requires expert8 by feature4 mesh')
    return jax.jit(jax.shard_map(
        partial(planned_verify_mapped, config=config, eos_ids=eos_ids, **options), mesh=mesh,
        in_specs=(P(), decoder.ws32_decoder_state_specs(), bf16_weight_specs(config), P(), P()),
        out_specs=PlannedVerification(proposal_specs(), P(), P()), check_vma=False))
