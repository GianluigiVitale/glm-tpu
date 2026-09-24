from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas.dsa import DsaScoreConfig, dsa_scores_kernel, dsa_scores_pallas
from glm_tpu.optimized.reference.dsa import dsa_scores


@pytest.mark.parametrize("context", [128, 130, 257])
def test_dsa_scores_pallas_interpret_matches_reference(context: int) -> None:
    query = jnp.asarray(
        np.sin(np.arange(32 * 128, dtype=np.float32) * np.float32(0.013))
        .reshape(1, 32, 128),
        dtype=jnp.float32,
    )
    keys = jnp.asarray(
        np.cos(
            np.arange(context * 128, dtype=np.float32) * np.float32(0.007)
        ).reshape(context, 128),
        dtype=jnp.bfloat16,
    )
    head_weights = jnp.asarray(
        np.linspace(-0.75, 0.875, 32, dtype=np.float32).reshape(1, 32)
    )

    expected = dsa_scores(query, keys, head_weights)
    actual = dsa_scores_pallas(
        query, keys, head_weights, interpret=True
    )
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=2e-6, atol=2e-5
    )


def test_dsa_score_dispatch_defaults_to_reference() -> None:
    query = jnp.ones((1, 2, 128), dtype=jnp.float32)
    keys = jnp.ones((3, 128), dtype=jnp.bfloat16)
    head_weights = jnp.asarray([[1.0, -0.5]], dtype=jnp.float32)
    expected = dsa_scores(query, keys, head_weights)

    np.testing.assert_array_equal(
        np.asarray(dsa_scores_kernel(query, keys, head_weights)),
        np.asarray(expected),
    )
    with pytest.raises(ValueError, match="unsupported DSA score backend"):
        dsa_scores_kernel(query, keys, head_weights, backend="unknown")  # type: ignore[arg-type]


def test_dsa_scores_pallas_rejects_contract_drift() -> None:
    query = jnp.ones((1, 2, 128), dtype=jnp.float32)
    keys = jnp.ones((128, 128), dtype=jnp.bfloat16)
    head_weights = jnp.ones((1, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="exactly one row"):
        dsa_scores_pallas(
            jnp.ones((2, 2, 128), dtype=jnp.float32),
            keys,
            jnp.ones((2, 2), dtype=jnp.float32),
            interpret=True,
        )
    with pytest.raises(ValueError, match="query and head weights must remain FP32"):
        dsa_scores_pallas(
            query.astype(jnp.bfloat16), keys, head_weights, interpret=True
        )
    with pytest.raises(ValueError, match="cache-resident.*BF16"):
        dsa_scores_pallas(
            query, keys.astype(jnp.float32), head_weights, interpret=True
        )
    with pytest.raises(ValueError, match="head dimensions disagree"):
        dsa_scores_pallas(
            query[..., :64], keys[:, :64], head_weights, interpret=True
        )

    with pytest.raises(ValueError, match="context tiles"):
        DsaScoreConfig(context_tile=64)
    with pytest.raises(ValueError, match="head dimension"):
        DsaScoreConfig(head_dim=64)
    with pytest.raises(ValueError, match="score dtype"):
        DsaScoreConfig(score_dtype=jnp.bfloat16)
