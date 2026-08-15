from __future__ import annotations

import json
import os
import subprocess
import sys


def test_ws32_io_matches_forced_32_reference_without_vocab_gather() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.ws32_io import (
    Ws32EmbeddingResult,
    Ws32GreedySampleResult,
    ws32_embedding_mapped,
    ws32_final_sample_mapped,
    ws32_logits_mapped,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

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
norm = np.asarray(
    rng.uniform(0.75, 1.25, (hidden_size,)), dtype=ml_dtypes.bfloat16
)
token = np.asarray([37], dtype=np.int32)

def put(value, spec):
    return jax.device_put(value, NamedSharding(mesh, spec))

embedding_program = jax.shard_map(
    lambda ids, table: ws32_embedding_mapped(
        ids, table, vocab_size=vocab_size
    ),
    mesh=mesh,
    in_specs=(P(), P("expert", "feature")),
    out_specs=Ws32EmbeddingResult(P(None, "feature"), P()),
    check_vma=False,
)
logits_program = jax.shard_map(
    lambda value, table: ws32_logits_mapped(
        value, table, vocab_size=vocab_size
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), P("expert", "feature")),
    out_specs=P(None, "expert"),
    check_vma=False,
)
sample_program = jax.shard_map(
    lambda value, weight, table: ws32_final_sample_mapped(
        value,
        weight,
        table,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    ),
    mesh=mesh,
    in_specs=(
        P(None, "feature"), P("feature"), P("expert", "feature")
    ),
    out_specs=Ws32GreedySampleResult(P(), P()),
    check_vma=False,
)

embedding_args = (put(token, P()), put(embedding, P("expert", "feature")))
logits_args = (
    put(hidden, P(None, "feature")),
    put(lm_head, P("expert", "feature")),
)
sample_args = logits_args[:1] + (
    put(norm, P("feature")), logits_args[1]
)
embedding_compiled = jax.jit(embedding_program).lower(*embedding_args).compile()
logits_compiled = jax.jit(logits_program).lower(*logits_args).compile()
sample_compiled = jax.jit(sample_program).lower(*sample_args).compile()
embedded = embedding_compiled(*embedding_args)
logits = logits_compiled(*logits_args)
sample = sample_compiled(*sample_args)

hidden_f32 = jnp.asarray(hidden).astype(jnp.float32)
inverse = lax.rsqrt(
    jnp.sum(lax.square(hidden_f32), axis=-1, keepdims=True)
    / jnp.float32(hidden_size)
    + jnp.float32(1e-5)
)
normalized = (
    (hidden_f32 * inverse).astype(jnp.bfloat16) * jnp.asarray(norm)
).astype(jnp.bfloat16)
expected_logits = lax.dot_general(
    jnp.asarray(hidden).astype(jnp.float32),
    jnp.asarray(lm_head).astype(jnp.float32),
    dimension_numbers=(((1,), (1,)), ((), ())),
    preferred_element_type=jnp.float32,
).astype(jnp.bfloat16)
expected_sample_logits = lax.dot_general(
    normalized.astype(jnp.float32),
    jnp.asarray(lm_head).astype(jnp.float32),
    dimension_numbers=(((1,), (1,)), ((), ())),
    preferred_element_type=jnp.float32,
).astype(jnp.bfloat16)
expected_token = int(jnp.argmax(expected_sample_logits[0]))

zero_head = put(
    np.zeros_like(lm_head), P("expert", "feature")
)
tied = sample_compiled(sample_args[0], sample_args[1], zero_head)

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
    "sample_token": np.asarray(sample.token_id).tolist(),
    "expected_token": expected_token,
    "sample_valid": np.asarray(sample.contract_valid).tolist(),
    "tie_token": np.asarray(tied.token_id).tolist(),
    "embedding_collectives": collectives(embedding_compiled),
    "logits_collectives": collectives(logits_compiled),
    "sample_collectives": collectives(sample_compiled),
    "sample_has_full_vocab_gather": any(
        item.raw_opcode == "all-gather"
        and any(shape.dimensions == (8,) for shape in item.operand_shapes)
        for item in parse_hlo_module(sample_compiled.as_text()).collectives
    ),
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
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
    assert result["sample_token"] == [result["expected_token"]]
    assert result["sample_valid"] == [True]
    assert result["tie_token"] == [0]
    assert [item["group"] for item in result["embedding_collectives"]] == [8]
    assert [item["group"] for item in result["logits_collectives"]] == [4]
    assert max(
        item["group"] for item in result["sample_collectives"]
    ) <= 8
    assert not result["sample_has_full_vocab_gather"]
