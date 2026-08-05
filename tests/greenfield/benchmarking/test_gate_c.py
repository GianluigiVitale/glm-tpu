from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _run_forced_cpu_gate_c_kernels() -> None:
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.benchmarking.gate_c import (
        stage_local_dense_gate_c,
        stage_local_dsa_gate_c,
        stage_local_index_share_gate_c,
        validate_gate_c_hlo,
    )
    from glm_tpu.greenfield.kernels.reference import (
        DsaNumericalContract,
        MlaNumericalContract,
        SelectedPositions,
        StageLocalKvLayout,
        dsa_index_keys,
        dsa_query_and_head_weights,
        dsa_scores,
        exact_topk,
        linear,
        residual_add,
        rms_norm,
        stage_local_sparse_mla_reference,
    )
    from glm_tpu.greenfield.kernels.reference.rotary import (
        apply_rotary,
        rotary_cos_sin,
    )

    if len(jax.devices()) != 4:
        raise AssertionError(f"expected four CPU devices, got {jax.devices()}")
    mesh = Mesh(np.asarray(jax.devices()), ("stage",))
    replicated = NamedSharding(mesh, P())

    def bf16(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=ml_dtypes.bfloat16)

    def put(values: np.ndarray, spec: P) -> jax.Array:
        return jax.device_put(values, NamedSharding(mesh, spec))

    def values(shape: tuple[int, ...], salt: int) -> np.ndarray:
        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        return bf16((((raw * 17 + salt) % 61) - 30) / 32)

    # Dense: output-feature shards for gate/up, matching input shards for down.
    hidden = 8
    intermediate = 16
    dense_residual_host = values((1, hidden), 3)
    dense_norm_host = values((hidden,), 5)
    gate_host = values((intermediate, hidden), 7)
    up_host = values((intermediate, hidden), 11)
    down_host = values((hidden, intermediate), 13)
    dense_inputs = (
        put(dense_residual_host, P()),
        put(dense_norm_host, P()),
        put(gate_host, P("stage", None)),
        put(up_host, P("stage", None)),
        put(down_host, P(None, "stage")),
    )
    dense_executable = jax.jit(
        lambda *args: stage_local_dense_gate_c(*args, mesh=mesh)
    ).lower(*dense_inputs).compile()
    assert validate_gate_c_hlo(
        dense_executable.as_text(),
        case="dense",
        stage_size=4,
        hidden_size=hidden,
    )["passed"]
    dense = dense_executable(*dense_inputs)
    dense_normalized = rms_norm(
        jnp.asarray(dense_residual_host),
        jnp.asarray(dense_norm_host),
        epsilon=1e-5,
    )
    dense_gate = linear(dense_normalized, jnp.asarray(gate_host))
    dense_up = linear(dense_normalized, jnp.asarray(up_host))
    dense_activated = (
        dense_gate * jax.nn.sigmoid(dense_gate) * dense_up
    ).astype(jnp.bfloat16)
    dense_update = linear(dense_activated, jnp.asarray(down_host))
    dense_output = residual_add(jnp.asarray(dense_residual_host), dense_update)
    np.testing.assert_allclose(
        np.asarray(dense.normalized, dtype=np.float32),
        np.asarray(dense_normalized, dtype=np.float32),
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(dense.output, dtype=np.float32),
        np.asarray(dense_output, dtype=np.float32),
        atol=0.03125,
        rtol=0,
    )

    # DSA: context is striped independently of the eight local head shards.
    dsa_contract = DsaNumericalContract(
        hidden_size=hidden,
        q_lora_rank=4,
        num_heads=4,
        head_dim=4,
        rotary_dim=2,
        top_k=8,
        theta=128.0,
    )
    context = 12
    local_context = context // 4
    position_owners = np.asarray(
        [[owner + 4 * row for row in range(local_context)] for owner in range(4)],
        dtype=np.int32,
    )
    history_all = values((context, hidden), 17)
    history_by_owner = np.stack(
        [history_all[row] for row in position_owners], axis=0
    )
    dsa_residual_host = values((1, hidden), 19)
    input_norm_host = values((hidden,), 23)
    qa_host = values((4, hidden), 29)
    qa_norm_host = values((4,), 31)
    wq_host = values((16, 4), 37)
    wk_host = values((4, hidden), 41)
    key_norm_host = values((4,), 43)
    key_bias_host = values((4,), 47)
    head_weight_host = values((4, hidden), 53)
    decode_position_host = np.asarray([context - 1], dtype=np.int32)
    dsa_inputs = (
        put(dsa_residual_host, P()),
        put(history_by_owner, P("stage", None, None)),
        put(position_owners, P("stage", None)),
        put(decode_position_host, P()),
        put(np.asarray([context], dtype=np.int32), P()),
        put(input_norm_host, P()),
        put(qa_host, P()),
        put(qa_norm_host, P()),
        put(wq_host, P("stage", None)),
        put(wk_host, P()),
        put(key_norm_host, P()),
        put(key_bias_host, P()),
        put(head_weight_host, P("stage", None)),
    )
    dsa_executable = jax.jit(
        lambda *args: stage_local_dsa_gate_c(
            *args,
            mesh=mesh,
            contract=dsa_contract,
        )
    ).lower(*dsa_inputs).compile()
    dsa_hlo = validate_gate_c_hlo(
        dsa_executable.as_text(),
        case="dsa",
        stage_size=4,
        hidden_size=hidden,
    )
    assert dsa_hlo["passed"], dsa_hlo
    dsa = dsa_executable(*dsa_inputs)
    dsa_normalized = rms_norm(
        jnp.asarray(dsa_residual_host),
        jnp.asarray(input_norm_host),
        epsilon=1e-5,
    )
    q_residual = rms_norm(
        linear(dsa_normalized, jnp.asarray(qa_host)),
        jnp.asarray(qa_norm_host),
        epsilon=1e-5,
    )
    query, head_weights = dsa_query_and_head_weights(
        dsa_normalized,
        q_residual,
        jnp.asarray(wq_host),
        jnp.asarray(head_weight_host),
        jnp.asarray(decode_position_host),
        contract=dsa_contract,
    )
    index_keys = dsa_index_keys(
        jnp.asarray(history_all),
        jnp.asarray(wk_host),
        jnp.asarray(key_norm_host),
        jnp.asarray(key_bias_host),
        jnp.arange(context, dtype=jnp.int32),
        contract=dsa_contract,
    )
    scores = dsa_scores(query, index_keys, head_weights)
    selected = exact_topk(
        scores,
        jnp.asarray([context], dtype=jnp.int32),
        top_k=dsa_contract.top_k,
    )
    np.testing.assert_array_equal(
        np.asarray(dsa.selected_positions), np.asarray(selected.positions)
    )
    np.testing.assert_allclose(
        np.asarray(dsa.query), np.asarray(query), atol=0.002, rtol=0
    )
    expected_owner_scores = np.stack(
        [np.asarray(scores)[:, row] for row in position_owners], axis=0
    )
    np.testing.assert_allclose(
        np.asarray(dsa.scores_by_owner),
        expected_owner_scores,
        atol=0.01,
        rtol=0,
    )

    # IndexShare: score-ordered state remains unchanged while attention uses
    # its own ascending copy and writes the current KV row to its sole owner.
    mla_contract = MlaNumericalContract(
        num_heads=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        qk_head_dim=4,
        v_head_dim=2,
        packed_cache_width=6,
        top_k=8,
    )
    cache_layout = StageLocalKvLayout(
        logical_page_size=8,
        local_parallel_size=4,
        packed_cache_width=6,
    )
    pages = 2
    full_cache = values((pages * cache_layout.logical_page_size, 6), 59)
    cache_by_owner = np.zeros(
        (4, pages, cache_layout.local_rows_per_page, 6),
        dtype=ml_dtypes.bfloat16,
    )
    for position in range(full_cache.shape[0]):
        page = position // cache_layout.logical_page_size
        within = position % cache_layout.logical_page_size
        owner = within // cache_layout.local_rows_per_page
        local_row = within % cache_layout.local_rows_per_page
        cache_by_owner[owner, page, local_row] = full_cache[position]
    index_residual_host = values((1, hidden), 61)
    index_norm_host = values((hidden,), 67)
    index_qa_host = values((4, hidden), 71)
    index_qa_norm_host = values((4,), 73)
    qb_host = values((16, 4), 79)
    kva_host = values((6, hidden), 83)
    kva_norm_host = values((4,), 89)
    kvb_host = values((16, 4), 97)
    o_host = values((hidden, 8), 101)
    selected_host = np.asarray([[11, 0, 7, 3, 9, 2, 5, 1]], dtype=np.int32)
    selected_counts_host = np.asarray([8], dtype=np.int32)
    tables_host = np.asarray([[0, 1]], dtype=np.int32)
    lengths_host = np.asarray([context], dtype=np.int32)
    index_inputs = (
        put(index_residual_host, P()),
        put(cache_by_owner, P("stage", None, None, None)),
        put(selected_host, P()),
        put(selected_counts_host, P()),
        put(decode_position_host, P()),
        put(tables_host, P()),
        put(lengths_host, P()),
        put(index_norm_host, P()),
        put(index_qa_host, P()),
        put(index_qa_norm_host, P()),
        put(qb_host, P("stage", None)),
        put(kva_host, P()),
        put(kva_norm_host, P()),
        put(kvb_host, P("stage", None)),
        put(o_host, P(None, "stage")),
    )
    index_executable = jax.jit(
        lambda *args: stage_local_index_share_gate_c(
            *args,
            mesh=mesh,
            contract=mla_contract,
            cache_layout=cache_layout,
            rope_theta=128.0,
        )
    ).lower(*index_inputs).compile()
    index_hlo = validate_gate_c_hlo(
        index_executable.as_text(),
        case="index_share",
        stage_size=4,
        hidden_size=hidden,
    )
    assert index_hlo["passed"], index_hlo
    index_result = index_executable(*index_inputs)

    index_normalized = rms_norm(
        jnp.asarray(index_residual_host),
        jnp.asarray(index_norm_host),
        epsilon=1e-5,
    )
    index_q_residual = rms_norm(
        linear(index_normalized, jnp.asarray(index_qa_host)),
        jnp.asarray(index_qa_norm_host),
        epsilon=1e-5,
    )
    q_states = linear(index_q_residual, jnp.asarray(qb_host)).reshape(
        1, 4, 4
    )
    q_nope = q_states[..., :2]
    cos, sin = rotary_cos_sin(
        jnp.asarray(decode_position_host),
        rotary_dim=2,
        theta=128.0,
        dtype=q_states.dtype,
    )
    q_rope = apply_rotary(
        q_states[..., 2:],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=True,
    )
    current_kv = linear(index_normalized, jnp.asarray(kva_host))
    current_latent = rms_norm(
        current_kv[..., :4], jnp.asarray(kva_norm_host), epsilon=1e-5
    )
    current_rope = apply_rotary(
        current_kv[..., 4:][:, None, :],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=True,
    )[:, 0, :]
    current_row = jnp.concatenate((current_latent, current_rope), axis=-1)
    updated_cache = np.array(cache_by_owner, copy=True)
    within = (context - 1) % cache_layout.logical_page_size
    current_owner = within // cache_layout.local_rows_per_page
    current_local_row = within % cache_layout.local_rows_per_page
    updated_cache[current_owner, 1, current_local_row] = np.asarray(current_row)
    kvb = jnp.asarray(kvb_host).reshape(4, 4, 4)
    weight_uk = kvb[:, :2, :]
    weight_uv = jnp.transpose(kvb[:, 2:, :], (0, 2, 1))
    q_absorbed = jnp.einsum(
        "rhp,hpl->rhl",
        q_nope.astype(jnp.float32),
        weight_uk.astype(jnp.float32),
    ).astype(jnp.bfloat16)
    attention = stage_local_sparse_mla_reference(
        q_absorbed,
        q_rope,
        jnp.asarray(updated_cache),
        jnp.asarray(tables_host),
        SelectedPositions(
            jnp.asarray(selected_host), jnp.asarray(selected_counts_host)
        ),
        jnp.asarray(lengths_host),
        layout=cache_layout,
        contract=mla_contract,
    )
    value_states = jnp.einsum(
        "rhl,hlv->rhv",
        attention.output.astype(jnp.float32),
        weight_uv.astype(jnp.float32),
    ).astype(jnp.bfloat16)
    attention_update = linear(
        value_states.reshape(1, 8), jnp.asarray(o_host)
    )
    index_output = residual_add(
        jnp.asarray(index_residual_host), attention_update
    )
    np.testing.assert_array_equal(
        np.asarray(index_result.selected_positions), selected_host
    )
    np.testing.assert_array_equal(
        np.asarray(index_result.attention_positions),
        np.sort(selected_host, axis=1),
    )
    np.testing.assert_array_equal(
        np.asarray(index_result.current_cache_row), np.asarray(current_row)
    )
    np.testing.assert_array_equal(
        np.asarray(index_result.cache_by_owner), updated_cache
    )
    gathered_positions = []
    gathered_values = []
    for owner in range(4):
        count = int(index_result.selected_cache_counts_by_owner[owner, 0])
        gathered_positions.append(
            np.asarray(
                index_result.selected_cache_positions_by_owner[owner, 0, :count]
            )
        )
        gathered_values.append(
            np.asarray(index_result.selected_cache_by_owner[owner, 0, :count])
        )
    gathered_positions_host = np.concatenate(gathered_positions)
    gathered_values_host = np.concatenate(gathered_values)
    gather_order = np.argsort(gathered_positions_host, stable=True)
    np.testing.assert_array_equal(
        gathered_positions_host[gather_order], np.sort(selected_host[0])
    )
    full_updated_cache = np.array(full_cache, copy=True)
    full_updated_cache[context - 1] = np.asarray(current_row)
    np.testing.assert_array_equal(
        gathered_values_host[gather_order],
        full_updated_cache[np.sort(selected_host[0])],
    )
    np.testing.assert_allclose(
        np.asarray(index_result.attended_latent, dtype=np.float32),
        np.asarray(attention.output, dtype=np.float32),
        atol=0.03125,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(index_result.attention_lse),
        np.asarray(attention.logsumexp),
        atol=2e-5,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(index_result.output, dtype=np.float32),
        np.asarray(index_output, dtype=np.float32),
        atol=0.03125,
        rtol=0,
    )
    np.testing.assert_array_equal(
        np.asarray(index_result.contract_valid), np.asarray([True])
    )


def test_gate_c_stage_local_kernels_on_four_forced_devices() -> None:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    code = (
        "from tests.greenfield.benchmarking.test_gate_c import "
        "_run_forced_cpu_gate_c_kernels; "
        "_run_forced_cpu_gate_c_kernels()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_gate_c_dsa_hlo_distinguishes_head_weight_from_dead_rows() -> None:
    from glm_tpu.greenfield.benchmarking import validate_gate_c_hlo

    hlo = r'''HloModule jit_dsa_step, replica_count=1, num_partitions=4

ENTRY main {
  weight = bf16[32,6144]{1,0} parameter(0)
  query = f32[1,8]{1,0} parameter(1)
  gathered_query = f32[4,1,8]{2,1,0} all-gather(query), dimensions={0}, replica_groups={{0,1,2,3}}, use_global_device_ids=true
  gathered_scores = f32[4,1,8]{2,1,0} all-gather(query), dimensions={0}, replica_groups={{0,1,2,3}}, use_global_device_ids=true
  gathered_positions = f32[4,1,8]{2,1,0} all-gather(query), dimensions={0}, replica_groups={{0,1,2,3}}, use_global_device_ids=true
  ROOT result = (bf16[32,6144]{1,0}, f32[4,1,8]{2,1,0}, f32[4,1,8]{2,1,0}, f32[4,1,8]{2,1,0}) tuple(weight, gathered_query, gathered_scores, gathered_positions)
}
'''
    result = validate_gate_c_hlo(hlo, case="dsa")
    assert result["passed"], result
