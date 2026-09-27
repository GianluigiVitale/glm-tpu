"""Tests of :mod:`glm_tpu.layers.sampler`."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.cpu32
def test_ws32_io_matches_forced_32_reference_without_vocab_gather() -> None:
    program = r"""
import json
import math

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tests.reference.norm import fused_add_rms_norm
from glm_tpu.layers.norm import sharded_fused_add_rms_norm
from glm_tpu.layers.embed import EmbeddingResult, embed_tokens
from glm_tpu.layers.sampler import SplitGreedySampleResult, compute_logits, split_final_sample
from glm_tpu.runner.hlo_utils import parse_hlo_module

devices = np.asarray(jax.devices(), dtype=object).reshape(8, 4)
mesh = Mesh(devices, ("expert", "feature"))
hidden_size = 128
vocab_size = 64
rng = np.random.default_rng(1)
embedding = np.asarray(
    rng.normal(0, 0.125, (vocab_size, hidden_size)),
    dtype=ml_dtypes.bfloat16,
)
lm_head = np.asarray(
    rng.normal(0, 0.125, (vocab_size, hidden_size)),
    dtype=ml_dtypes.bfloat16,
)
hidden = np.asarray(
    rng.normal(0, 0.25, (1, hidden_size)), dtype=ml_dtypes.bfloat16
)
carried = np.asarray(
    rng.normal(0, 0.25, (1, hidden_size)), dtype=ml_dtypes.bfloat16
)
norm = np.asarray(
    rng.uniform(0.75, 1.25, (hidden_size,)), dtype=ml_dtypes.bfloat16
)
token = np.asarray([37], dtype=np.int32)

def put(value, spec):
    return jax.device_put(value, NamedSharding(mesh, spec))

embedding_program = jax.shard_map(
    lambda ids, table: embed_tokens(
        ids, table, vocab_size=vocab_size
    ),
    mesh=mesh,
    in_specs=(P(), P("expert", "feature")),
    out_specs=EmbeddingResult(P(None, "feature"), P()),
    check_vma=False,
)
logits_program = jax.shard_map(
    lambda value, table: compute_logits(
        value, table, vocab_size=vocab_size
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), P("expert", "feature")),
    out_specs=P(None, "expert"),
    check_vma=False,
)
split_norm_program = jax.shard_map(
    lambda update, residual, weight: sharded_fused_add_rms_norm(
        update,
        residual,
        weight,
        global_hidden_size=hidden_size,
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), P(None, "feature"), P("feature")),
    out_specs=(P(None, "feature"), P(None, "feature")),
    check_vma=False,
)
split_sample_program = jax.shard_map(
    lambda update, residual, weight, table: split_final_sample(
        update,
        residual,
        weight,
        table,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    ),
    mesh=mesh,
    in_specs=(
        P(None, "feature"), P(None, "feature"), P("feature"),
        P("expert", "feature"),
    ),
    out_specs=SplitGreedySampleResult(P(), P(), P(None, "feature")),
    check_vma=False,
)

embedding_args = (put(token, P()), put(embedding, P("expert", "feature")))
logits_args = (
    put(hidden, P(None, "feature")),
    put(lm_head, P("expert", "feature")),
)
split_args = (
    put(hidden, P(None, "feature")),
    put(carried, P(None, "feature")),
    put(norm, P("feature")),
)
split_sample_args = split_args + (logits_args[1],)
embedding_compiled = jax.jit(embedding_program).lower(*embedding_args).compile()
logits_compiled = jax.jit(logits_program).lower(*logits_args).compile()
split_norm_compiled = jax.jit(split_norm_program).lower(*split_args).compile()
split_sample_compiled = jax.jit(split_sample_program).lower(
    *split_sample_args
).compile()
embedded = embedding_compiled(*embedding_args)
logits = logits_compiled(*logits_args)
split_normalized, split_carried = split_norm_compiled(*split_args)
split_sample = split_sample_compiled(*split_sample_args)

expected_logits = lax.dot_general(
    jnp.asarray(hidden).astype(jnp.float32),
    jnp.asarray(lm_head).astype(jnp.float32),
    dimension_numbers=(((1,), (1,)), ((), ())),
    preferred_element_type=jnp.float32,
).astype(jnp.bfloat16)
expected_split_normalized, expected_split_carried = fused_add_rms_norm(
    jnp.asarray(hidden),
    jnp.asarray(carried),
    jnp.asarray(norm),
    epsilon=1e-5,
)
expected_split_logits = lax.dot_general(
    expected_split_normalized.astype(jnp.float32),
    jnp.asarray(lm_head).astype(jnp.float32),
    dimension_numbers=(((1,), (1,)), ((), ())),
    preferred_element_type=jnp.float32,
).astype(jnp.bfloat16)
expected_split_token = int(jnp.argmax(expected_split_logits[0]))

zero_head = put(
    np.zeros_like(lm_head), P("expert", "feature")
)
tied = split_sample_compiled(*split_args, zero_head)

def collectives(compiled):
    module = parse_hlo_module(compiled.as_text())
    return [
        {
            "opcode": item.raw_opcode,
            "group": item.maximum_group_size,
            "operand_dims": [list(shape.dimensions) for shape in item.operand_shapes],
        }
        for item in module.collectives
    ]

print(json.dumps({
    "embedding_bitwise": bool(np.array_equal(
        np.asarray(embedded.residual_local).view(np.uint16),
        np.asarray(embedding[37:38]).view(np.uint16),
    )),
    "embedding_valid": np.asarray(embedded.contract_valid).tolist(),
    "embedding_sharding": str(embedded.residual_local.sharding.spec),
    "logits_max_abs": float(jnp.max(jnp.abs(
        logits.astype(jnp.float32) - expected_logits.astype(jnp.float32)
    ))),
    "logits_sharding": str(logits.sharding.spec),
    "split_normalized_bitwise": bool(np.array_equal(
        np.asarray(split_normalized).view(np.uint16),
        np.asarray(expected_split_normalized).view(np.uint16),
    )),
    "split_carried_bitwise": bool(np.array_equal(
        np.asarray(split_carried).view(np.uint16),
        np.asarray(expected_split_carried).view(np.uint16),
    )),
    "split_sample_carried_bitwise": bool(np.array_equal(
        np.asarray(split_sample.final_residual_local).view(np.uint16),
        np.asarray(expected_split_carried).view(np.uint16),
    )),
    "split_sample_token": np.asarray(split_sample.token_id).tolist(),
    "expected_split_token": expected_split_token,
    "split_sample_valid": np.asarray(split_sample.contract_valid).tolist(),
    "split_collectives": collectives(split_sample_compiled),
    "tie_token": np.asarray(tied.token_id).tolist(),
    "tie_valid": np.asarray(tied.contract_valid).tolist(),
    "embedding_collectives": collectives(embedding_compiled),
    "logits_collectives": collectives(logits_compiled),
    "split_sample_gathered_values": [
        math.prod(shape.dimensions)
        for item in parse_hlo_module(split_sample_compiled.as_text()).collectives
        if item.raw_opcode == "all-gather"
        for shape in item.result_shapes
    ],
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = f"{existing} --xla_force_host_platform_device_count=32".strip()
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["embedding_bitwise"]
    assert result["embedding_valid"] == [True]
    assert result["embedding_sharding"] == "P(None, 'feature')"
    assert result["logits_max_abs"] <= 0.015625
    assert result["logits_sharding"] == "P(None, 'expert')"
    assert result["split_normalized_bitwise"]
    assert result["split_carried_bitwise"]
    assert result["split_sample_carried_bitwise"]
    assert result["split_sample_token"] == [result["expected_split_token"]]
    assert result["split_sample_valid"] == [True]
    assert max(item["group"] for item in result["split_collectives"]) <= 8
    assert result["tie_token"] == [0] and result["tie_valid"] == [True]  # all logits equal: the lowest id
    assert [item["group"] for item in result["embedding_collectives"]] == [8]
    assert [item["group"] for item in result["logits_collectives"]] == [4]
    # no full-vocabulary gather: the owners exchange one candidate each, no all-gather assembles 64 logits
    assert result["split_sample_gathered_values"] and max(result["split_sample_gathered_values"]) < 64
