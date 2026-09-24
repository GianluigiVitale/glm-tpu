from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from glm_tpu.optimized.reference.moe import (
    GlmMoeNumericalContract,
    dequantize_fp8_block_weight,
    reference_moe_from_routes,
    route_glm_noaux_tc_logits,
    stage_local_moe_from_routes,
)


def small_contract() -> GlmMoeNumericalContract:
    return GlmMoeNumericalContract(
        hidden_size=8,
        intermediate_size=8,
        num_experts=16,
        top_k=4,
        stage_size=4,
        fp8_block_shape=(2, 2),
    )


def small_weights(contract: GlmMoeNumericalContract):
    key = jax.random.key(7)
    keys = jax.random.split(key, 6)

    def draw(k, shape):
        return (
            jax.random.normal(k, shape, dtype=jnp.float32) * 0.125
        ).astype(jnp.bfloat16)

    return (
        draw(
            keys[0],
            (contract.num_experts, contract.intermediate_size, contract.hidden_size),
        ),
        draw(
            keys[1],
            (contract.num_experts, contract.intermediate_size, contract.hidden_size),
        ),
        draw(
            keys[2],
            (contract.num_experts, contract.hidden_size, contract.intermediate_size),
        ),
        draw(keys[3], (contract.intermediate_size, contract.hidden_size)),
        draw(keys[4], (contract.intermediate_size, contract.hidden_size)),
        draw(keys[5], (contract.hidden_size, contract.intermediate_size)),
    )


def test_glm_52_contract_is_exact() -> None:
    contract = GlmMoeNumericalContract()
    assert contract.hidden_size == 6144
    assert contract.intermediate_size == 2048
    assert contract.num_experts == 256
    assert contract.top_k == 8
    assert contract.stage_size == 4
    assert contract.local_experts == 64
    assert contract.local_shared_intermediate == 512
    assert contract.routed_scaling_factor == 2.5
    assert contract.fp8_block_shape == (128, 128)
    assert contract.router_dtype == "float32"


def test_contract_refuses_nonlocal_expert_partition() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        GlmMoeNumericalContract(num_experts=255)


def test_block_dequant_uses_checkpoint_out_in_blocks() -> None:
    weight = jnp.arange(1, 25, dtype=jnp.float32).reshape(4, 6)
    scale = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    got = dequantize_fp8_block_weight(
        weight, scale, block_shape=(2, 3), output_dtype=jnp.float32
    )
    expanded = np.asarray(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(weight) * expanded)


def test_noaux_bias_selects_but_does_not_weight() -> None:
    logits = jnp.asarray([[3.0, 2.0, 1.0, 0.0]], dtype=jnp.float32)
    bias = jnp.asarray([[0.0, 0.0, 100.0, 99.0]], dtype=jnp.float32)[0]
    indices, weights = route_glm_noaux_tc_logits(logits, bias, top_k=2)
    np.testing.assert_array_equal(np.asarray(indices), [[2, 3]])
    unbiased = jax.nn.sigmoid(logits)[0, jnp.asarray([2, 3])]
    expected = unbiased / unbiased.sum()
    np.testing.assert_allclose(np.asarray(weights[0]), np.asarray(expected), rtol=0, atol=0)


def test_router_ties_choose_lowest_expert_ids() -> None:
    logits = jnp.zeros((1, 16), dtype=jnp.float32)
    indices, weights = route_glm_noaux_tc_logits(
        logits, jnp.zeros((16,), dtype=jnp.float32), top_k=8
    )
    np.testing.assert_array_equal(np.asarray(indices), [list(range(8))])
    np.testing.assert_array_equal(
        np.asarray(weights), np.full((1, 8), 0.125, dtype=np.float32)
    )


def _run_forced_cpu_case(concentrated: bool, stage_size: int = 4) -> None:
    if len(jax.devices()) != stage_size:
        raise AssertionError(
            f"expected {stage_size} forced CPU devices, got {jax.devices()}"
        )
    contract = replace(small_contract(), stage_size=stage_size)
    mesh = jax.make_mesh(
        (stage_size,), ("expert",), devices=np.asarray(jax.devices())
    )
    hidden = jnp.asarray(
        [[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]],
        dtype=jnp.bfloat16,
    )
    indices = (
        jnp.asarray([[8, 9, 10, 11]], dtype=jnp.int32)
        if concentrated
        else jnp.asarray([[0, 5, 10, 15]], dtype=jnp.int32)
    )
    weights = jnp.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=jnp.float32)
    expert_gate, expert_up, expert_down, shared_gate, shared_up, shared_down = (
        small_weights(contract)
    )
    args = (
        hidden,
        indices,
        weights,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
    )
    expected = reference_moe_from_routes(*args, contract=contract)
    replicated = NamedSharding(mesh, P())
    expert_sharded = NamedSharding(mesh, P("expert"))
    shared_down_sharded = NamedSharding(mesh, P(None, "expert"))
    input_shardings = (
        replicated,
        replicated,
        replicated,
        expert_sharded,
        expert_sharded,
        expert_sharded,
        expert_sharded,
        expert_sharded,
        shared_down_sharded,
    )
    sharded_args = tuple(
        jax.device_put(value, sharding)
        for value, sharding in zip(args, input_shardings, strict=True)
    )
    compiled = jax.jit(
        lambda *values: stage_local_moe_from_routes(
            *values, mesh=mesh, contract=contract
        )
    )
    got = compiled(*sharded_args)
    # Sharding the shared intermediate changes its dot/reduction association.
    # The public contract therefore requires bounded internal tensor error,
    # while routes remain elementwise exact.
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=0,
        # Two-way and four-way shared-intermediate reductions have different
        # legal BF16 association. This narrow synthetic fixture differs by at
        # most one/two local ULPs; the real protected tensor bars are unchanged.
        atol=2**-13 if stage_size == 2 else 2**-14,
    )

    hlo = compiled.lower(*sharded_args).compile().as_text()
    collective_lines = [
        line
        for line in hlo.splitlines()
        if "all-reduce(" in line or "all-gather(" in line or "all-to-all(" in line
        or "reduce-scatter(" in line or "collective-permute(" in line
    ]
    assert len(collective_lines) == 1, "\n".join(collective_lines)
    assert "all-reduce(" in collective_lines[0]
    expected_group = ",".join(str(rank) for rank in range(stage_size))
    assert (
        "replica_groups={{" + expected_group + "}}" in collective_lines[0]
    )
    # CPU XLA promotes bfloat16 psum to f32; the protected TPU contract will
    # separately require bf16[2,1,6144]. The logical payload and group are
    # already fixed here.
    assert (
        "bf16[2,1,8]" in collective_lines[0]
        or "f32[2,1,8]" in collective_lines[0]
    ), collective_lines[0]


@pytest.mark.parametrize("concentrated", [False, True])
def test_stage_local_exactness_and_hlo_in_forced_cpu_subprocess(
    concentrated: bool,
) -> None:
    if os.environ.get("GLM_GREENFIELD_MOE_SUBPROCESS") == "1":
        _run_forced_cpu_case(concentrated)
        return
    env = dict(os.environ)
    env["GLM_GREENFIELD_MOE_SUBPROCESS"] = "1"
    env["GLM_GREENFIELD_MOE_CONCENTRATED"] = "1" if concentrated else "0"
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    code = (
        "from tests.greenfield.kernels.test_reference_moe import "
        "_run_forced_cpu_case; import os; "
        "_run_forced_cpu_case(os.environ['GLM_GREENFIELD_MOE_CONCENTRATED']=='1')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("concentrated", [False, True])
def test_pp16_stage_local_exactness_and_hlo_in_forced_cpu_subprocess(
    concentrated: bool,
) -> None:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=2".strip()
    )
    code = (
        "from tests.greenfield.kernels.test_reference_moe import "
        "_run_forced_cpu_case; "
        f"_run_forced_cpu_case({concentrated!r}, 2)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
