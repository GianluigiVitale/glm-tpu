"""Acceptance covers rejection, bonus, EOS and budgets independently of drafting."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.perf.speculative_accept import greedy_acceptance


def test_every_rejection_eos_and_budget_boundary():
    for rows in (1, 2, 3, 5):
        fn = jax.jit(partial(greedy_acceptance, eos_token_ids=(90, 91)))
        for rejection in range(rows):
            for eos_at in range(rows + 1):
                targets = np.arange(10, 10 + rows, dtype=np.int32)
                if eos_at < rows:
                    targets[eos_at] = 90
                inputs = np.concatenate((np.array([7], np.int32), targets[:-1])).copy()
                if rejection < rows - 1:
                    inputs[rejection + 1] = 200
                for budget in range(rows + 2):
                    expected = list(targets[:rejection + 1])[:budget]
                    if 90 in expected:
                        expected = expected[:expected.index(90) + 1]
                    actual = fn(jnp.asarray(inputs), jnp.asarray(targets), jnp.int32(budget))
                    assert bool(actual.valid)
                    assert int(actual.emitted_count) == len(expected)
                    assert int(actual.accepted_draft_count) == min(rejection, len(expected))
                    np.testing.assert_array_equal(actual.accepted_by_position,
                        np.arange(rows - 1) < min(rejection, len(expected)))
                    assert bool(actual.stopped_on_eos) == (bool(expected) and expected[-1] == 90)
                    assert bool(actual.stopped_on_length) == (len(expected) == budget and 90 not in expected)


def test_invalid_inputs_never_authorize_output():
    tokens = jnp.array([4, 5, 6], jnp.int32)
    for inputs, targets, budget in ((tokens.at[1].set(-1), tokens, 3),
                                     (tokens, tokens.at[2].set(-1), 3), (tokens, tokens, -1)):
        result = greedy_acceptance(inputs, targets, jnp.int32(budget))
        assert not result.valid and result.emitted_count == 0
        assert not np.asarray(result.accepted_by_position).any()
    with pytest.raises(ValueError):
        greedy_acceptance(tokens, tokens[:1], jnp.int32(2))
