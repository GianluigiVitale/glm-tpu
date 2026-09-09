"""Five-call dense continuation under the existing protected worker primitives.

No CLI, runtime initialization, checkpoint load, compile admission or launch.
The parent must bind real prompt/weights/WK/RoPE, sources and physical owners.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_worker import BudgetedCalls, save_arrays
from scripts.greenfield.ws32_dense_frontier_capture import cache_bits, capture
from scripts.greenfield.ws32_dense_frontier_witness import compare_owner
from scripts.greenfield.ws32_prefill_frontier_state import require_config

GRAPH = "dense01"
PROGRAMS = ("wk_decode", "wk_promote", GRAPH)
RESERVE = 1 << 30
ORIGINALS_LIMIT = 128 << 20
PROMPT_SHA = "d860b7f4be91608c86e0a629c4096fd7a95036287d0f8e31ea67b01475de0cc0"


def memory_budget(census: Any, analyses: Any, *, active_graph: str) -> dict:
    if set(analyses) != set(PROGRAMS) or active_graph not in PROGRAMS:
        raise ValueError("dense diagnostic requires its three resident programs")
    return budget_resident_execution(census, analyses, active_graph=active_graph,
                                    resident_graphs=PROGRAMS, required_reserve_bytes=RESERVE)


def fresh_caches(mesh: Any) -> tuple:
    """Allocate six independent owner caches; all initial bytes are explicit zero."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharding = NamedSharding(mesh, P("expert", None, None, None))
    return tuple(tuple(jax.make_array_from_callback(
        (8, 16, 64, width), sharding,
        lambda index, width=width: np.zeros((1, 16, 64, width), dtype=jnp.bfloat16))
        for width in (640, 128, 128)) for _ in range(2))


def independent_caches(wide: Any, narrow: Any) -> None:
    seen = set()
    for branch in (wide, narrow):
        for layer in branch:
            for array in layer:
                for shard in array.addressable_shards:
                    key = (shard.device.id, int(shard.data.unsafe_buffer_pointer()))
                    if key[1] <= 0 or key in seen:
                        raise ValueError("dense initial cache allocations alias")
                    seen.add(key)


def inputs(mesh: Any, tokens: np.ndarray, offset: int, caches: Any,
           embedding: Any, layers: Any, wk: Any, rope: Any) -> tuple:
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharding = NamedSharding(mesh, P())
    host = (np.pad(tokens, (0, 128-len(tokens))), np.asarray(len(tokens), np.int32),
            np.asarray(offset, np.int32), np.arange(16, dtype=np.int32)[None])
    values = tuple(jax.make_array_from_callback(v.shape, sharding,
                   lambda index, value=v: value[index]) for v in host)
    return (*values, caches, embedding, layers, wk, rope)


def execute_five_calls(
    calls: BudgetedCalls, *, mesh: Any, config: Any, prompt_tokens: np.ndarray,
    embedding: Any, layers: Any, wk: Any, rope: Any, witness: Mapping,
) -> None:
    """Preserve both branches before refusing failed reproduction; never promote.

    WK/WK for layer0 and WK/WK for layer1 must already have completed through
    these SAME BudgetedCalls. Both new cache states have separate allocations.
    Intermediate row outputs are retained, obsolete device caches are released.
    """
    def preflight():
        require_config(config)
        expected = [(f"layer{layer}/{name}", name) for layer in (0, 1) for name in PROGRAMS[:2]]
        if (calls.budgeter is not memory_budget or set(calls.programs) != set(PROGRAMS)
                or [(v["phase"], v["graph"]) for v in calls.record["call_evidence"]] != expected
                or not all(v["completed"] for v in calls.record["call_evidence"])
                or not isinstance(prompt_tokens, np.ndarray) or prompt_tokens.dtype != np.int32
                or prompt_tokens.shape != (8155,) or sha256(prompt_tokens.tobytes()).hexdigest() != PROMPT_SHA):
            raise ValueError("dense continuation prompt/budget/WK call inventory differs")
    calls.phase("dense/preflight", preflight)
    used = 0
    records = {}
    calls.record["dense_frontier"] = dict(complete=False, originals=records,
                                         numerical_promotion=False, performance_claim=False)

    def preserve(label, result, count, endpoint):
        nonlocal used
        arrays, report = capture(result, local_slots=calls.local_slots,
                                  process_index=calls.record["jax_process_index"],
                                  count=count, keep_caches=endpoint)
        amount = sum(v.nbytes for v in arrays.values())
        if used + amount + (len(records)+1)*(1 << 20) > ORIGINALS_LIMIT:
            raise ValueError("dense originals exceed fixed rank budget")
        path = calls.root / f"{label}.npz"
        report.update(npz_sha256=save_arrays(path, arrays), npz_bytes=path.stat().st_size,
                      raw_array_bytes=amount)
        used += amount
        records[label] = report
        _atomic_json(calls.root / f"{label}.json", report)
        if not report["valid"]:
            raise ValueError("dense output unhealthy; originals preserved")

    wide = calls.phase("dense/wide_initial", lambda: fresh_caches(mesh))
    narrow = calls.phase("dense/narrow_initial", lambda: fresh_caches(mesh))
    calls.phase("dense/independent", lambda: independent_caches(wide, narrow))
    values = calls.phase("dense/wide_inputs", lambda: inputs(
        mesh, prompt_tokens[:128], 0, wide, embedding, layers, wk, rope))
    result = calls.call("dense/wide", GRAPH, values,
                        preserve=lambda r: preserve("wide_final", r, 128, True))
    wide = tuple(tuple(layer[2:5]) for layer in result)
    del result, values
    for tile in range(4):
        start, end = tile*32, (tile+1)*32
        values = calls.phase(f"dense/narrow{tile}_inputs", lambda: inputs(
            mesh, prompt_tokens[start:end], start, narrow, embedding, layers, wk, rope))
        result = calls.call(f"dense/narrow{tile}", GRAPH, values,
                            preserve=lambda r: preserve(f"narrow_{end}", r, 32, end == 128))
        narrow = tuple(tuple(layer[2:5]) for layer in result)
        del result, values

    def compare():
        comparisons = []
        for branch in ("wide_final", "narrow_128"):
            with np.load(calls.root / f"{branch}.npz", allow_pickle=False) as arrays:
                comparisons.extend(compare_owner(witness, branch=branch, slot=slot,
                                    caches=cache_bits(arrays, slot)) for slot in calls.local_slots.values())
        report = dict(owners=comparisons, reproduced=all(v["reproduced"] for v in comparisons),
                      model_calls=5, wk_calls=4, numerical_promotion=False, performance_claim=False)
        _atomic_json(calls.root / "comparison.json", report)
        calls.record["dense_frontier"].update(comparison=report, complete=True)
        if not report["reproduced"]:
            raise ValueError("dense realization does not reproduce DB604; both branches preserved")
    calls.phase("dense/comparison", compare)
