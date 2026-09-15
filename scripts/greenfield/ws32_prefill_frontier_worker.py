"""Fixed five-call continuation using existing protected worker primitives.

No CLI, runtime initialization, compiler/profile admission, deployment or seal.
Outer worker must authenticate weights/prompt/source/graphs and physical owners.
This continuation cannot run a full prompt or replace the frozen baseline.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.ws32_budgeted_calls import BudgetedCalls, save_arrays
from scripts.greenfield.ws32_batched_prefill_runner import (
    graph_inputs, make_ws32_batched_prefill_state,
)
from scripts.greenfield.ws32_prefill_frontier import compare_cache
from scripts.greenfield.ws32_prefill_frontier_state import (
    PROMPT_LENGTH, capture_state, require_config,
)

GRAPH = "prefill_chunk"
ORIGINALS_LIMIT = 128 * 1024**2
RESERVE = 1024**3


def memory_budget(census: Any, analyses: Any, *, active_graph: str) -> dict[str, Any]:
    if active_graph != GRAPH:
        raise ValueError("first-window only dispatches original B128")
    return budget_resident_execution(
        census, analyses, active_graph=active_graph,
        resident_graphs=tuple(analyses), required_reserve_bytes=RESERVE,
    )


def _independent_caches(wide: Any, narrow: Any) -> None:
    """Check per-device allocation identities, not Python container identity."""
    seen = set()
    for state in (wide, narrow):
        for array in (state.decoder.kv_cache_local, state.decoder.index_cache_local,
                      state.repaired_index_local):
            for shard in array.addressable_shards:
                pointer = int(shard.data.unsafe_buffer_pointer())
                key = (shard.device.id, pointer)
                if pointer <= 0 or key in seen:
                    raise ValueError("first-window initial cache allocations alias")
                seen.add(key)


def execute_first_window(
    calls: BudgetedCalls, *, mesh: Any, config: Any, prompt_tokens: np.ndarray,
    weights: Any, wk: Any, rope: Any,
) -> None:
    """One128 versus four32 on unchanged B128; preserve before any refusal.

    All distributed calls use BudgetedCalls' preflight vote and all-live census.
    Local preparation/capture/publication uses its matched phases. Wide output
    stays live throughout the narrow branch; obsolete narrow generations are
    released. Byte inequality is a diagnostic outcome, not failed health.
    """
    def validate():
        require_config(config)
        if (calls.budgeter is not memory_budget or GRAPH not in calls.programs
                or calls.record["call_evidence"]
                or not isinstance(prompt_tokens, np.ndarray)
                or prompt_tokens.dtype != np.int32 or prompt_tokens.shape != (PROMPT_LENGTH,)
                or np.any(prompt_tokens < 0) or np.any(prompt_tokens >= config.geometry.vocab_size)):
            raise ValueError("first-window requires fresh fixed prompt/budget/call inventory")

    calls.phase("first_window/preflight", validate)
    capture = partial(capture_state, config=config, local_slots=calls.local_slots,
                      process_index=calls.record["jax_process_index"])
    records: dict[str, Any] = {}
    calls.record["first_window"] = dict(
        originals=records, numerical_promotion=False, performance_claim=False,
        complete=False, maximum_calls=5,
    )
    used_raw_bytes = 0
    used_file_bytes = 0

    def preserve(label, state, next_token, frontier, *, caches):
        nonlocal used_raw_bytes, used_file_bytes
        arrays, record = capture(state, next_token=next_token, frontier=frontier, caches=caches)
        raw_bytes = sum(value.nbytes for value in arrays.values())
        # Reserve1MiB overhead per original before writing. No full-size safety copy.
        if used_raw_bytes + raw_bytes + (len(records)+1)*1024**2 > ORIGINALS_LIMIT:
            raise ValueError("first-window original byte budget exceeded")
        path = calls.root / f"{label}.npz"
        record["npz_sha256"] = save_arrays(path, arrays)
        record["npz_bytes"] = path.stat().st_size
        record["raw_array_bytes"] = raw_bytes
        records[label] = record
        _atomic_json(calls.root / f"{label}.json", record)
        used_raw_bytes += raw_bytes
        used_file_bytes += record["npz_bytes"] + (calls.root / f"{label}.json").stat().st_size
        if used_file_bytes > ORIGINALS_LIMIT:
            raise ValueError("first-window serialized original budget exceeded")
        if not record["valid"]:
            raise ValueError(f"first-window {label} invalid; originals preserved: {record['errors']}")

    # Separate allocator invocations, same config and complete immutable prompt.
    wide = calls.phase("first_window/wide_initial", lambda: make_ws32_batched_prefill_state(
        mesh, config, prompt_length=PROMPT_LENGTH))
    narrow = calls.phase("first_window/narrow_initial", lambda: make_ws32_batched_prefill_state(
        mesh, config, prompt_length=PROMPT_LENGTH))
    calls.phase("first_window/independent", lambda: _independent_caches(wide, narrow))
    for label, state in (("wide_initial", wide), ("narrow_initial", narrow)):
        calls.phase(f"first_window/{label}_capture",
                    lambda: preserve(label, state, None, 0, caches=True))
    # The for-loop's state alias would otherwise retain a zero cache generation.
    del state

    def compare_initial():
        if records["wide_initial"]["owners"] != records["narrow_initial"]["owners"]:
            raise ValueError("first-window complete initial state bytes differ")
    calls.phase("first_window/initial_equality", compare_initial)

    def inputs(tokens, state):
        return graph_inputs(mesh, tokens, state, weights, wk, rope,
                            mlp_window=True, physical_rows=128)

    values = calls.phase("first_window/wide_inputs", lambda: inputs(prompt_tokens[:128], wide))
    result = calls.call("first_window/wide", GRAPH, values,
                        preserve=lambda r: preserve("wide_final", r.state, r.next_token, 128, caches=True))
    wide = result.state
    del result, values
    for tile in range(4):
        start, end = tile*32, (tile+1)*32
        values = calls.phase(f"first_window/narrow{tile}_inputs",
                             lambda: inputs(prompt_tokens[start:end], narrow))
        result = calls.call(
            f"first_window/narrow{tile}", GRAPH, values,
            preserve=lambda r: preserve(f"narrow_{end}", r.state, r.next_token,
                                         end, caches=end == 128),
        )
        narrow = result.state
        del result, values

    def comparison():
        result = {}
        with np.load(calls.root / "wide_final.npz", allow_pickle=False) as a, np.load(
            calls.root / "narrow_128.npz", allow_pickle=False
        ) as b:
            for slot in sorted(records["wide_final"]["owners"], key=int):
                left, right = (records[label]["owners"][slot]
                               for label in ("wide_final", "narrow_128"))
                result[slot] = dict(
                    metadata_bytes_equal=left["metadata_sha256"] == right["metadata_sha256"],
                    caches={name: compare_cache(
                        a[f"slot{slot}_{name}_rows"], left["caches"][name]["cache"],
                        b[f"slot{slot}_{name}_rows"], right["caches"][name]["cache"],
                    ) for name in ("kv", "index", "repair")},
                )
        report = dict(owners=result, calls=5, numerical_promotion=False,
                      performance_claim=False, scope="FIRST128_ONLY_NOT_ROOT_CAUSE")
        _atomic_json(calls.root / "comparison.json", report)
        if len(calls.record["call_evidence"]) != 5:
            raise ValueError("first-window completed call count differs")
        calls.record["first_window"].update(complete=True, comparison=report,
                                            original_file_bytes=used_file_bytes)
    calls.phase("first_window/comparison", comparison)
