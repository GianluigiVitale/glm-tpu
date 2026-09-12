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
from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    make_prefill_memory_record,
    validate_prefill_memory_record,
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
    mesh: Any,
    config: Ws32DecoderConfig,
    plan: BatchedPrefillPlan,
    *,
    paired_position_sort: bool = False,
    rolled_prefix: bool = False,
    expert_panels: bool = False,
    sorted_local_merge: bool = False,
    canonical_dense: bool = False,
    pending_cache_rows: bool = False,
    flat_pending_rows: bool = False,
    capture_barrier: bool = False,
    key_tile: int = 4096,
) -> dict[str, Any]:
    """Return uncompiled builders; outer worker owns HLO/memory authorization."""
    if type(pending_cache_rows) is not bool:
        raise ValueError("pending cache rows must be a static bool")
    if type(flat_pending_rows) is not bool or (flat_pending_rows and not pending_cache_rows):
        raise ValueError("flat pending rows require static bool and pending cache rows")
    if type(capture_barrier) is not bool or (capture_barrier and not flat_pending_rows):
        raise ValueError("capture barrier requires static bool and flat pending rows")
    if config.context_capacity != plan.context_capacity:
        raise ValueError("batched plan/config capacity differs")
    return {
        name: build_ws32_batched_prefill_program(
            mesh,
            config,
            block_rows=rows,
            paired_position_sort=paired_position_sort,
            mlp_window=plan.mlp_window,
            rolled_prefix=rolled_prefix,
            expert_panels=expert_panels,
            sorted_local_merge=sorted_local_merge,
            canonical_dense=canonical_dense,
            key_tile=key_tile,
            **({"pending_cache_rows": True} if pending_cache_rows else {}),
            **({"flat_pending_rows": True} if flat_pending_rows else {}),
            **({"capture_barrier": True} if capture_barrier else {}),
        )
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
    *,
    mlp_window: bool = False,
    physical_rows: int | None = None,
) -> tuple[Any, ...]:
    """Transfer live IDs with optional masked static padding; count stays live.

    Padding is host-created legal token0, never included in the prompt/frontier.
    The frozen model already masks rows beyond the dynamic valid-row count.
    Default None retains the historical unpadded input and graph shapes.
    """
    if type(mlp_window) is not bool:
        raise ValueError("batched input window option must be a bool")
    if (
        tokens.dtype != np.int32
        or tokens.ndim != 1
        or not 1 <= tokens.size <= (128 if mlp_window else 32)
    ):
        raise ValueError("batched host int32 IDs exceed explicit window mode")
    live_rows = tokens.size
    if physical_rows is not None:
        if (
            type(physical_rows) is not int
            or not mlp_window
            or not live_rows <= physical_rows <= 128
        ):
            raise ValueError("batched physical input rows must cover live window rows")
        tokens = np.pad(tokens, (0, physical_rows - live_rows), constant_values=0)
    return (
        replicated(mesh, tokens),
        replicated(mesh, np.asarray(live_rows, np.int32)),
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
    required_memory_reserve_bytes: int,
    progress: Callable[[dict[str, Any]], None],
    fleet_all: Callable[[bool], bool],
    additional_resident_executables: Mapping[str, Any] | None = None,
    memory_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Ws32DecoderState, Any, dict[str, Any]]:
    """Execute already-authorized blocks, refusing at the first unhealthy one.

    Includes transfer and cache-init in request wall; block wall starts after
    transfer completion. Host progress/health checks are intentionally disclosed
    in total wall. No new checkpoint/persistent resume is created here.
    fleet_all MUST return the AND of all hosts' predicates to every host.
    One initial all-live memory census budgets both resident executables before
    the first dispatch and is fleet-AND gated; it is NOT repeated for every
    block or every layer. The caller must release superseded state references
    and disclose any other resident model executable before using this adapter.
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
    if (
        type(required_memory_reserve_bytes) is not int
        or required_memory_reserve_bytes <= 0
    ):
        raise ValueError("batched prefill requires explicit positive memory reserve")
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
    memory_record = None
    memory_seconds = 0.0
    for block in range(full + 1):
        name = GRAPHS[0] if block < full else GRAPHS[1]
        rows = plan.stride_rows if block < full else tail
        transfer_started = perf_counter()
        inputs = graph_inputs(
            mesh,
            prompt_tokens[offset : offset + rows],
            current,
            weights,
            wk,
            rope,
            mlp_window=plan.mlp_window,
            physical_rows=(
                plan.tail_graph_rows
                if name == GRAPHS[1]
                else plan.block_rows if plan.live_block_rows is not None else None
            ),
        )
        jax.block_until_ready(inputs[:2])
        transfer_walls.append(perf_counter() - transfer_started)
        if block == 0:
            memory_started = perf_counter()
            memory_error = None
            try:
                memory_record = make_prefill_memory_record(
                    compiled,
                    {"active_inputs": inputs},
                    devices=jax.local_devices(),
                    required_reserve_bytes=required_memory_reserve_bytes,
                    additional_resident_executables=additional_resident_executables,
                )
                validate_prefill_memory_record(memory_record)
            except Exception as exc:
                memory_error = exc
            if memory_progress is not None:
                try:
                    # Preserve both successful admission and partial/refused
                    # evidence before first dispatch. A publication failure is
                    # itself voted false, so peers cannot continue alone.
                    memory_progress(
                        {
                            "memory_admission": memory_record,
                            "error": (
                                None if memory_error is None else str(memory_error)
                            ),
                        }
                    )
                except Exception as exc:
                    if memory_error is None:
                        memory_error = exc
            if not fleet_all(memory_error is None):
                raise RuntimeError(
                    "batched prefill fleet refused initial memory budget"
                ) from memory_error
            # A broken fleet callback cannot convert a local failure to a pass.
            if memory_error is not None:
                raise RuntimeError(
                    "batched prefill local memory refusal"
                ) from memory_error
            memory_seconds = perf_counter() - memory_started
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
        # Census is paid ONCE, not once per block. Extrapolating its time by
        # prompt_length/offset would cause a false budget refusal after block1.
        fixed_seconds = init_seconds + memory_seconds
        projected = (
            fixed_seconds
            + max(0.0, elapsed - fixed_seconds) * plan.prompt_length / offset
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
        "memory_admission_seconds": memory_seconds,
        "memory_admission": memory_record,
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
