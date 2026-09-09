"""Bounded fresh-cache/input/local-delivery budgets without model weights.

No launcher or transport implementation. Calls must be owned by the protected
budget campaign. Local callback receipt is NOT model TTFT or client-network ACK.
"""

from __future__ import annotations

from math import prod
from time import perf_counter
from typing import Any

import numpy as np

from scripts.greenfield import prefill_budget_probe as probe
from scripts.greenfield import prefill_budget_worker as worker
from scripts.greenfield.prefill_window_worker import validate_memory_owners
from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    capture_identified_device_memory,
)

MAX_SECONDS = 240
INPUT_ROWS = (11, 17, 32, 127363, 262144)


def state_bytes_per_chip(config: Any) -> int:
    """Logical local payload, not allocator/executable overhead or measured HBM."""
    return (
        2 * prod(config.kv_cache_shape) // 8
        + 4 * prod(config.index_cache_shape) // 8  # unrepaired AND repaired
        + 8 * config.geometry.dsa_top_k
        + 4 * config.page_count
        + 18  # counts,position,length,prompt int32; health,finished bool
    )


def configs() -> tuple:
    from scripts.greenfield.run_short_decoder_ws32 import _geometry
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig

    geometry = _geometry()
    return tuple(
        Ws32DecoderConfig(
            geometry=geometry, context_capacity=capacity, host_main_rope_table=True
        )
        for capacity, _ in probe.CAPACITIES
    )


def state_structure(state: Any, config: Any, slots: dict[int, int]) -> dict:
    """Validate local shapes/dtypes/metadata, without downloading whole caches."""
    import jax

    small = dict(
        selected_positions=(
            np.full((1, 2048), -1, np.int32),
            state.decoder.selected_positions,
        ),
        selected_valid_counts=(
            np.zeros((1,), np.int32),
            state.decoder.selected_valid_counts,
        ),
        selected_scores=(
            np.full((1, 2048), -np.inf, np.float32),
            state.decoder.selected_scores,
        ),
        position=(np.zeros((1,), np.int32), state.decoder.position),
        context_lengths=(np.ones((1,), np.int32), state.decoder.context_lengths),
        block_tables=(
            np.arange(config.page_count, dtype=np.int32)[None, :],
            state.decoder.block_tables,
        ),
        health=(np.ones((1,), np.bool_), state.decoder.contract_valid),
        prompt=(
            np.asarray(dict(probe.CAPACITIES)[config.context_capacity], np.int32),
            state.prompt_length,
        ),
        finished=(np.asarray(False, np.bool_), state.finished),
    )
    for expected, array in small.values():
        seen = set()
        for shard in array.addressable_shards:
            device = int(shard.device.id)
            actual = np.asarray(shard.data)
            if (
                device in seen
                or device not in slots
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.tobytes() != expected.tobytes()
            ):
                raise ValueError("fresh-cache metadata differs")
            seen.add(device)
        if seen != set(slots):
            raise ValueError("fresh-cache metadata owners differ")
    payloads = {str(d): 0 for d in slots}
    caches = (
        state.decoder.kv_cache_local,
        state.decoder.index_cache_local,
        state.repaired_index_local,
    )
    for array, shape in zip(
        caches,
        (config.kv_cache_shape, config.index_cache_shape, config.index_cache_shape),
        strict=True,
    ):
        local_shape = (shape[0], shape[1], shape[2] // 8, shape[3])
        if tuple(array.shape) != shape or str(array.dtype) != "bfloat16":
            raise ValueError("fresh-cache global geometry differs")
        if {int(s.device.id) for s in array.addressable_shards} != set(slots):
            raise ValueError("fresh-cache owners differ")
        if any(tuple(s.data.shape) != local_shape for s in array.addressable_shards):
            raise ValueError("fresh-cache local stripe shape differs")
    for leaf in jax.tree.leaves(state):
        for shard in leaf.addressable_shards:
            payloads[str(int(shard.device.id))] += (
                shard.data.size * shard.data.dtype.itemsize
            )
    if any(v != state_bytes_per_chip(config) for v in payloads.values()):
        raise ValueError("fresh-cache payload byte accounting differs")
    return dict(
        local_payload_bytes=payloads,
        metadata_passed=True,
        cache_content_checked=False,
        scope="FRESH_STATE_STRUCTURE_NOT_POPULATED_RESUME",
    )


def run(calls: Any, mesh: Any, *, clock=perf_counter) -> dict:
    """Four fresh allocations;35 placements; seven local-consumer deliveries.

    Each fresh state's leaves are deleted before another state is allocated.
    No state from a real request or model weight is accepted by this function.
    Everything is outside DSA sampling, with its own inclusive240s ceiling.
    """
    import jax
    from scripts.greenfield.ws32_batched_prefill_runner import graph_inputs, replicated

    configurations = calls.phase("overhead/config", configs)
    started = calls.phase("overhead/start", clock)
    report = dict(
        initialization={},
        input_placement={},
        delivery=[],
        complete=False,
        model_ttft_measured=False,
        budget_seconds=MAX_SECONDS,
    )
    calls.record["budget_overhead"] = report

    def deadline():
        elapsed = clock() - started
        if not np.isfinite(elapsed) or not 0 <= elapsed < MAX_SECONDS:
            raise ValueError("request-overhead budget exceeded")

    def counters():
        rows = capture_identified_device_memory(tuple(jax.local_devices()))
        validate_memory_owners(
            rows,
            local_slots=calls.local_slots,
            process_index=calls.record["jax_process_index"],
        )
        return rows

    for config, (capacity, prompt) in zip(
        configurations, probe.CAPACITIES, strict=True
    ):
        report["initialization"][str(capacity)] = []
        for iteration in range(2):

            def initialize():
                deadline()
                before = counters()
                required = state_bytes_per_chip(config) + worker.RESERVE_BYTES
                if any(r["bytes_limit"] - r["bytes_in_use"] < required for r in before):
                    raise ValueError("fresh-state allocation exceeds memory reserve")
                state = None
                try:
                    state, timing = probe.measure_initial_state(
                        mesh, config, prompt_length=prompt, clock=clock
                    )
                    after = counters()
                    if any(
                        r["bytes_in_use"] < state_bytes_per_chip(config) for r in after
                    ):
                        raise ValueError(
                            "fresh-state live HBM is smaller than its payload"
                        )
                    if any(
                        r["bytes_limit"] - r["peak_bytes_in_use"] < worker.RESERVE_BYTES
                        for r in after
                    ):
                        raise ValueError("fresh-state peak HBM exceeds reserve")
                    structure = state_structure(state, config, calls.local_slots)
                    timing.update(
                        iteration=iteration,
                        before_memory=before,
                        after_memory=after,
                        structure=structure,
                        first_allocation=iteration == 0,
                    )
                    report["initialization"][str(capacity)].append(timing)
                finally:
                    if state is not None:
                        for leaf in jax.tree.leaves(state):
                            leaf.delete()
                deadline()

            calls.phase(f"overhead/c{capacity}/initialize{iteration}", initialize)

    for rows in INPUT_ROWS:
        host = calls.phase(
            f"overhead/input{rows}/host",
            lambda: np.arange(rows, dtype=np.int32) % 154000 + 256,
        )
        samples = []
        report["input_placement"][str(rows)] = dict(
            rows=rows,
            samples_seconds=samples,
            warmup=probe.WARMUP,
            scope=(
                "CURRENT_BLOCK_GRAPH_INPUTS"
                if rows <= 32
                else "BULK_PLACEMENT_ALTERNATIVE_NOT_CURRENT_RUNNER"
            ),
        )
        for iteration in range(probe.WARMUP + probe.ITERATIONS):

            def place():
                deadline()
                values = None
                try:
                    begin = clock()
                    values = (
                        graph_inputs(mesh, host, None, None, (), None)[:2]
                        if rows <= 32
                        else (
                            replicated(mesh, host),
                            replicated(mesh, np.asarray(rows, np.int32)),
                        )
                    )
                    jax.block_until_ready(values)
                    elapsed = clock() - begin
                    if not np.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("input placement clock invalid")
                    # Force actual original-byte comparison outside timing.
                    for value, expected in zip(
                        values, (host, np.asarray(rows, np.int32)), strict=True
                    ):
                        if len(value.addressable_shards) != len(calls.local_slots) or {
                            int(s.device.id) for s in value.addressable_shards
                        } != set(calls.local_slots):
                            raise ValueError("input placement owners differ")
                        for shard in value.addressable_shards:
                            actual = np.asarray(shard.data)
                            if (
                                actual.shape != expected.shape
                                or actual.dtype != expected.dtype
                                or actual.tobytes() != expected.tobytes()
                            ):
                                raise ValueError("input placement bytes differ")
                    if iteration >= probe.WARMUP:
                        samples.append(elapsed)
                finally:
                    if values is not None:
                        for value in values:
                            value.delete()
                deadline()

            calls.phase(f"overhead/input{rows}/sample{iteration}", place)

    for iteration in range(probe.WARMUP + probe.ITERATIONS):

        def deliver():
            deadline()
            token = replicated(mesh, np.asarray([2048 + iteration], np.int32))
            try:
                jax.block_until_ready(token)  # placement is not device-to-host cost
                if calls.record["jax_process_index"] == 0:
                    received = []
                    result = probe.deliver_local_token(
                        token, received.append, clock=clock
                    )
                    if received != [2048 + iteration]:
                        raise ValueError("local token consumer acknowledgement differs")
                    if iteration >= probe.WARMUP:
                        report["delivery"].append(result)
            finally:
                token.delete()
            deadline()

        calls.phase(f"overhead/delivery{iteration}", deliver)

    def complete():
        deadline()
        report.update(
            complete=True,
            elapsed_seconds=clock() - started,
            delivery_process_index=0,
            delivery_scope=probe.DELIVERY_SCOPE,
        )

    calls.phase("overhead/complete", complete)
    return report


def expected_stages() -> list[str]:
    stages = ["overhead/config", "overhead/start"]
    stages.extend(
        f"overhead/c{c}/initialize{i}" for c, _ in probe.CAPACITIES for i in range(2)
    )
    for rows in INPUT_ROWS:
        stages.append(f"overhead/input{rows}/host")
        stages.extend(
            f"overhead/input{rows}/sample{i}"
            for i in range(probe.WARMUP + probe.ITERATIONS)
        )
    stages.extend(
        f"overhead/delivery{i}" for i in range(probe.WARMUP + probe.ITERATIONS)
    )
    return stages + ["overhead/complete"]


def validate_record(report: dict, *, slots: dict[int, int], process_index: int) -> None:
    """Recompute fixed geometry, counter/reserve and reporting contracts."""
    from scripts.greenfield.prefill_window_evidence import same_json

    fixed = dict(
        complete=True,
        model_ttft_measured=False,
        budget_seconds=MAX_SECONDS,
        delivery_process_index=0,
        delivery_scope=probe.DELIVERY_SCOPE,
    )
    same_json({k: report.get(k) for k in fixed}, fixed, "overhead scope")
    if set(report) != set(fixed) | {
        "initialization",
        "input_placement",
        "delivery",
        "elapsed_seconds",
    }:
        raise ValueError("overhead field inventory differs")

    def seconds(value):
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("overhead duration invalid")
        return value

    total = 0.0
    if set(report["initialization"]) != {str(c) for c, _ in probe.CAPACITIES}:
        raise ValueError("overhead capacities differ")
    prior_peak = {}
    for config, (capacity, prompt) in zip(configs(), probe.CAPACITIES, strict=True):
        samples = report["initialization"][str(capacity)]
        if len(samples) != 2:
            raise ValueError("overhead fresh/warm initialization count differs")
        payload = state_bytes_per_chip(config)
        for iteration, sample in enumerate(samples):
            fixed_init = dict(
                capacity=capacity,
                prompt_length=prompt,
                weights_loaded=False,
                initialized_prefix_length=0,
                model_ttft_measured=False,
                iteration=iteration,
                first_allocation=iteration == 0,
                structure=dict(
                    local_payload_bytes={str(d): payload for d in slots},
                    metadata_passed=True,
                    cache_content_checked=False,
                    scope="FRESH_STATE_STRUCTURE_NOT_POPULATED_RESUME",
                ),
            )
            same_json(
                {k: sample.get(k) for k in fixed_init},
                fixed_init,
                "initialization geometry",
            )
            if set(sample) != set(fixed_init) | {
                "before_memory",
                "after_memory",
                "cache_initialization_seconds",
            }:
                raise ValueError("initialization fields differ")
            total += seconds(sample["cache_initialization_seconds"])
            for key in ("before_memory", "after_memory"):
                rows = sample[key]
                validate_memory_owners(
                    rows, local_slots=slots, process_index=process_index
                )
                for row in rows:
                    d = row["device_id"]
                    if key == "after_memory" and row["bytes_in_use"] < payload:
                        raise ValueError(
                            "fresh-state live HBM is smaller than its payload"
                        )
                    if (
                        any(
                            type(row[k]) is not int
                            for k in (
                                "bytes_in_use",
                                "peak_bytes_in_use",
                                "bytes_limit",
                            )
                        )
                        or not 0
                        <= row["bytes_in_use"]
                        <= row["peak_bytes_in_use"]
                        <= row["bytes_limit"]
                        or row["bytes_limit"] - row["peak_bytes_in_use"]
                        < worker.RESERVE_BYTES
                        or d in prior_peak
                        and (
                            row["bytes_limit"] != prior_peak[d][0]
                            or row["peak_bytes_in_use"] < prior_peak[d][1]
                        )
                    ):
                        raise ValueError(
                            "initialization memory counter/reserve differs"
                        )
                    if (
                        key == "before_memory"
                        and row["bytes_limit"] - row["bytes_in_use"]
                        < payload + worker.RESERVE_BYTES
                    ):
                        raise ValueError("initialization allocation budget differs")
                    prior_peak[d] = row["bytes_limit"], row["peak_bytes_in_use"]
    if set(report["input_placement"]) != set(map(str, INPUT_ROWS)):
        raise ValueError("input placement geometries differ")
    for rows in INPUT_ROWS:
        sample = report["input_placement"][str(rows)]
        fixed_input = dict(
            rows=rows,
            warmup=probe.WARMUP,
            scope=(
                "CURRENT_BLOCK_GRAPH_INPUTS"
                if rows <= 32
                else "BULK_PLACEMENT_ALTERNATIVE_NOT_CURRENT_RUNNER"
            ),
        )
        same_json(
            {k: sample.get(k) for k in fixed_input},
            fixed_input,
            "input placement scope",
        )
        if (
            set(sample) != set(fixed_input) | {"samples_seconds"}
            or len(sample["samples_seconds"]) != probe.ITERATIONS
        ):
            raise ValueError("input placement samples differ")
        total += sum(seconds(s) for s in sample["samples_seconds"])
    deliveries = report["delivery"]
    if len(deliveries) != (probe.ITERATIONS if process_index == 0 else 0):
        raise ValueError("delivery consumer process/sample count differs")
    for index, sample in enumerate(deliveries):
        copy = seconds(sample["device_to_host_seconds"])
        callback = seconds(sample["callback_ack_seconds"])
        delivered = seconds(sample["delivery_seconds"])
        if not np.isclose(copy + callback, delivered, rtol=0, atol=1e-9):
            raise ValueError("delivery intervals differ")
        same_json(
            sample,
            dict(
                token_id=2048 + probe.WARMUP + index,
                device_to_host_seconds=copy,
                callback_ack_seconds=callback,
                delivery_seconds=delivered,
                delivery_scope=probe.DELIVERY_SCOPE,
                model_ttft_measured=False,
            ),
            "delivery receipt",
        )
        total += delivered
    elapsed = seconds(report["elapsed_seconds"])
    if not total <= elapsed < MAX_SECONDS:
        raise ValueError("inclusive overhead budget differs")
