"""Host adapter for the separate §24 prefill graph pair.

This module does not launch workers, load/repack weights, approve HLO, or seal
results. The protected worker must authorize BOTH compiled graphs and their
memory before calling execute. Serial graph contracts are not authorization.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from time import perf_counter
from typing import Any, Callable, Mapping

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
    Ws32BatchedPrefillState,
    build_ws32_batched_prefill_program,
    finish_ws32_batched_prefill,
    make_ws32_batched_prefill_state,
)
from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig,
    Ws32DecoderState,
    Ws32DecoderWeights,
    bind_ws32_decoder_weights,
    ws32_decoder_weight_names,
)


from glm_tpu.greenfield.validation.ws32_prefill import (
    BatchedPrefillPlan,
    GRAPHS,
    PREFILL_MODE,
    PREFILL_GRAPH_KIND,
    validate_execution_record,
)


def bind_raw_prefill_weights(
    arrays: Mapping[str, Any], decode_config: Ws32DecoderConfig
) -> tuple[Ws32DecoderConfig, Ws32DecoderWeights]:
    """Bind a raw view BEFORE the worker drops its loaded-array dictionary.

    The promoted decode view may select overlay arrays instead. Shared leaves
    remain the exact same array objects; there is no device copy or full repack.
    Only the original loader can certify these arrays' checkpoint provenance.
    """
    if not decode_config.exact_dsa or not decode_config.host_main_rope_table:
        raise ValueError(
            "batched adapter requires exact decode repair weights and host RoPE"
        )
    config = replace(decode_config, exact_dsa=False, strategy_nd_dense=False)
    names = jax.tree.leaves(ws32_decoder_weight_names(config))
    missing = sorted(set(names) - set(arrays))
    if missing:
        raise ValueError(f"raw prefill view missing retained tensors: {missing[:5]}")
    selected = {name: arrays[name] for name in names}
    return config, bind_ws32_decoder_weights(selected, config)


def completed_repair_weights(
    exact_dsa_weights: tuple[Any, ...], config: Ws32DecoderConfig
) -> tuple[Any, ...]:
    """Reuse existing completed wk arrays, not raw bits or per-block dequant."""
    if len(exact_dsa_weights) != len(config.full_index_slots):
        raise ValueError("batched repair producer cardinality drifted")
    values = tuple(value.wk_weight for value in exact_dsa_weights)
    for value in values:
        if (
            value.shape
            != (config.geometry.dsa_indexer_head_dim, config.geometry.hidden_size)
            or str(value.dtype) != "float32"
        ):
            raise ValueError("batched repair requires completed full FP32 wk owners")
    return values


def build_graph_pair(
    mesh: Any, config: Ws32DecoderConfig, plan: BatchedPrefillPlan
) -> dict[str, Any]:
    """Return uncompiled builders; outer worker owns HLO/memory authorization."""
    if config.context_capacity != plan.context_capacity:
        raise ValueError("batched plan/config capacity differs")
    return {
        name: build_ws32_batched_prefill_program(mesh, config, block_rows=rows)
        for name, rows in plan.graph_rows
    }


def replicated(mesh: Any, value: np.ndarray) -> Any:
    host = np.ascontiguousarray(value)
    # ascontiguousarray promotes a scalar to rank1; preserve the scalar count.
    if np.asarray(value).shape == ():
        host = np.asarray(value).copy()
    return jax.make_array_from_callback(
        host.shape, NamedSharding(mesh, P()), lambda _: host.copy()
    )


def graph_inputs(
    mesh: Any,
    tokens: np.ndarray,
    state: Ws32BatchedPrefillState,
    weights: Ws32DecoderWeights,
    wk: tuple[Any, ...],
    rope: Any,
) -> tuple[Any, ...]:
    """One narrow live block: no padded token and no serial donation indices."""
    if tokens.dtype != np.int32 or tokens.ndim != 1 or not 1 <= tokens.size <= 32:
        raise ValueError("batched host block requires1..32 int32 IDs")
    return (
        replicated(mesh, tokens),
        replicated(mesh, np.asarray(tokens.size, np.int32)),
        state,
        weights,
        wk,
        rope,
    )


def execute_graph_pair(
    mesh: Any,
    config: Ws32DecoderConfig,
    plan: BatchedPrefillPlan,
    prompt_tokens: np.ndarray,
    compiled: Mapping[str, Any],
    weights: Ws32DecoderWeights,
    wk: tuple[Any, ...],
    rope: Any,
    *,
    budget_seconds: float,
    progress: Callable[[dict[str, Any]], None],
    fleet_all: Callable[[bool], bool],
) -> tuple[Ws32DecoderState, Any, dict[str, Any]]:
    """Execute already-authorized blocks, refusing at the first unhealthy one.

    Includes transfer and cache-init in request wall; block wall starts after
    transfer completion. Host progress/health checks are intentionally disclosed
    in total wall. No new checkpoint/persistent resume is created here.
    fleet_all MUST return the AND of all hosts' predicates to every host.
    Every host calls it twice per healthy block, including the final block:
    structural health, then budget/logging continuation. Host-local clocks may
    never independently decide whether to dispatch another collective program.
    This harness consensus is outside model graphs and included in request wall.
    """
    if config.context_capacity != plan.context_capacity:
        raise ValueError("batched execution plan/config capacity differs")
    if prompt_tokens.shape != (plan.prompt_length,) or prompt_tokens.dtype != np.int32:
        raise ValueError("batched prompt IDs differ from declared plan")
    if set(compiled) != set(GRAPHS):
        raise ValueError("batched execution requires both authorized graph programs")
    if (
        type(budget_seconds) not in (int, float)
        or not isfinite(budget_seconds)
        or budget_seconds <= 0
    ):
        raise ValueError("batched prefill requires finite positive wall budget")
    started = perf_counter()
    init_started = perf_counter()
    current = make_ws32_batched_prefill_state(
        mesh, config, prompt_length=plan.prompt_length
    )
    jax.block_until_ready(current)
    init_seconds = perf_counter() - init_started
    full, tail = plan.split
    block_walls, transfer_walls = [], []
    projected_max = 0.0
    offset = 0
    final_result = None
    for block in range(full + 1):
        name = GRAPHS[0] if block < full else GRAPHS[1]
        rows = plan.block_rows if block < full else tail
        transfer_started = perf_counter()
        inputs = graph_inputs(
            mesh, prompt_tokens[offset : offset + rows], current, weights, wk, rope
        )
        jax.block_until_ready(inputs[:2])
        transfer_walls.append(perf_counter() - transfer_started)
        block_started = perf_counter()
        result = compiled[name](*inputs)
        jax.block_until_ready(result)
        block_walls.append(perf_counter() - block_started)
        offset += rows
        final = offset == plan.prompt_length
        state = result.state
        unhealthy = (
            not np.asarray(state.decoder.contract_valid).all()
            or not np.array_equal(np.asarray(state.decoder.position), [offset])
            or not np.array_equal(
                np.asarray(state.decoder.context_lengths), [offset + 1]
            )
            or int(np.asarray(state.prompt_length)) != plan.prompt_length
            or bool(np.asarray(state.finished)) != final
            or (not final and not np.array_equal(np.asarray(result.next_token), [-1]))
        )
        if not fleet_all(not bool(unhealthy)):
            raise RuntimeError(
                f"batched prefill unhealthy/wrong frontier at block{block}, offset{offset}"
            )
        current = state
        final_result = result
        elapsed = perf_counter() - started
        projected = (
            init_seconds
            + max(0.0, elapsed - init_seconds) * plan.prompt_length / offset
        )
        projected_max = max(projected_max, projected)
        progress_error = None
        try:
            progress(
                {
                    "block": block,
                    "graph": name,
                    "rows": rows,
                    "frontier": offset,
                    "block_seconds": block_walls[-1],
                    "projected_total_seconds": projected,
                    "finished": final,
                }
            )
        except Exception as exc:
            # A logging exception must not let peers dispatch the next block.
            progress_error = exc
        local_continue = (
            progress_error is None
            and projected <= budget_seconds
            and perf_counter() - started <= budget_seconds
        )
        if not fleet_all(local_continue):
            raise RuntimeError(
                f"batched prefill fleet refused projected/request wall budget "
                f"{budget_seconds}s or progress logging"
            ) from progress_error
    decoder, token = finish_ws32_batched_prefill(final_result)
    record = {
        "identity": plan.identity(),
        "budget_seconds": float(budget_seconds),
        "cache_initialization_seconds": init_seconds,
        "input_transfer_seconds": transfer_walls,
        "block_wall_seconds": block_walls,
        "projected_total_seconds_max": projected_max,
        "request_prefill_seconds": perf_counter() - started,
        "final_frontier": offset,
        "finished_healthy": True,
        "repaired_index_installed": True,
        "first_token_ready": int(np.asarray(token)[0]),
        "ttft_measured": False,
        "timing_scope": "input_ids_ready_to_prefill_token_ready_not_delivery",
    }
    return decoder, token, record
