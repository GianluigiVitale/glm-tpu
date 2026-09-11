"""Two-branch layers0..6 history driver under the budgeted-call contract.

No CLI, runtime initialization, checkpoint load, compile admission or launch.
The parent binds real prompt/weights/WK/exact/overlay/RoPE, physical owners and
the retained step0 originals before calling ``execute_history``. Every device
dispatch goes through ``calls.call``; every host step through ``calls.phase``.

Both branches keep independent caches, chain their own health scalar on device,
and are compared row-by-row at every layer boundary of every interleave group.
Exact operands are retained only for the FIRST differing boundary; afterwards
only compact statistics are kept. Nothing here promotes a result, samples a
token or explains the failure: reproduction of each branch's own retained
step0 observation is what makes a recorded difference eligible for attribution.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Callable, Mapping

import numpy as np

from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_worker import save_arrays
from scripts.greenfield.ws32_history_frontier import HistoryCaches, LAYERS, PRODUCERS
from scripts.greenfield.ws32_history_capture import (
    ReplicaMismatch, capture_boundaries as read_boundaries, replicated_host,
)

FIELDS = ("update", "residual", "normalized_input", "route_ids", "route_weights")
FLOAT_FIELDS = ("update", "residual", "normalized_input", "route_weights")
CACHE_FAMILIES = ("kv", "unrepaired", "repaired")
POISON = -2147483648
OBSERVER_FIELDS = ("positions", "counts", "scores")


def _replicated(mesh: Any, value: np.ndarray) -> Any:
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharding = NamedSharding(mesh, P())
    return jax.make_array_from_callback(value.shape, sharding, lambda index, v=value: v[index])


def fresh_caches(mesh: Any, config: Any) -> HistoryCaches:
    """Seven KV and four unrepaired/repaired owner caches, all explicit zeros."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharding = NamedSharding(mesh, P("expert", None, None, None))
    pages = config.page_count

    def cache(width: int) -> Any:
        return jax.make_array_from_callback(
            (8, pages, 64, width), sharding,
            lambda index, width=width: np.zeros((1, pages, 64, width), dtype=jnp.bfloat16))

    return HistoryCaches(tuple(cache(config.packed_cache_width) for _ in LAYERS),
                         tuple(cache(config.geometry.dsa_indexer_head_dim) for _ in PRODUCERS),
                         tuple(cache(config.geometry.dsa_indexer_head_dim) for _ in PRODUCERS))


def independent(*branches: HistoryCaches) -> None:
    """Every cache buffer of every branch is a distinct device allocation."""
    import jax
    seen = set()
    for caches in branches:
        for array in jax.tree.leaves(caches):
            for shard in array.addressable_shards:
                key = (int(shard.device.id), int(shard.data.unsafe_buffer_pointer()))
                if key[1] <= 0 or key in seen:
                    raise ValueError("history branch caches alias")
                seen.add(key)


def block_values(mesh: Any, tokens: np.ndarray, offset: int, rows: int, caches: HistoryCaches,
                 embedding: Any, layers: Any, wk: Any, rope: Any, table: Any, healthy: Any) -> tuple:
    """One frontier call's inputs; dead rows carry the poison token."""
    if (not isinstance(tokens, np.ndarray) or tokens.dtype != np.int32 or tokens.ndim != 1
            or not 0 < tokens.size <= rows or type(offset) is not int or offset < 0):
        raise ValueError("history block tokens/offset differ")
    padded = np.full(rows, POISON, np.int32)
    padded[: tokens.size] = tokens
    return (_replicated(mesh, padded), _replicated(mesh, np.asarray(tokens.size, np.int32)),
            _replicated(mesh, np.asarray(offset, np.int32)), table, caches, embedding, layers,
            wk, rope, healthy)


def _bits(value: np.ndarray) -> np.ndarray:
    return value.view(np.uint16) if value.dtype.name == "bfloat16" else value


def compare_rows(candidate: Mapping[int, Mapping[str, np.ndarray]],
                 control: Mapping[int, Mapping[str, np.ndarray]], *, offset: int) -> dict[str, Any]:
    """Row-aligned exact comparison; statistics only, no tolerance."""
    report: dict[str, Any] = dict(offset=offset, first=None, layers={})
    for layer in LAYERS:
        report["layers"][str(layer)] = {}
        for field in FIELDS:
            a, b = candidate[layer][field], control[layer][field]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"history comparison geometry differs at {layer}/{field}")
            # Exactness includes signed zero and floating-point payload bits.
            left = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape[0], -1)
            right = np.ascontiguousarray(b).view(np.uint8).reshape(b.shape[0], -1)
            differ = np.any(left != right, axis=1)
            entry: dict[str, Any] = dict(rows=int(a.shape[0]), differing_rows=int(differ.sum()))
            if differ.any():
                first_row = int(np.flatnonzero(differ)[0])
                entry["first_position"] = offset + first_row
                if field in FLOAT_FIELDS:
                    delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
                    entry["max_abs_difference"] = float(delta.max())
                if report["first"] is None or entry["first_position"] < report["first"]["position"]:
                    report["first"] = dict(layer=layer, field=field, position=entry["first_position"])
            report["layers"][str(layer)][field] = entry
    report["equal"] = report["first"] is None
    return report


def cache_digests(caches: HistoryCaches, *, local_slots: Mapping[int, int]) -> dict[str, Any]:
    """Per-owner digests of every cache family; each host digests its own owners."""
    digests: dict[str, Any] = {}
    for family, arrays in zip(CACHE_FAMILIES, caches, strict=True):
        digests[family] = {}
        for index, array in enumerate(arrays):
            owners = {}
            for shard in array.addressable_shards:
                device = int(shard.device.id)
                if device not in local_slots:
                    raise ValueError("history cache owner is not an authenticated local device")
                owners[str(local_slots[device])] = sha256(
                    np.ascontiguousarray(np.asarray(shard.data)).view(np.uint8)).hexdigest()
            if not owners:
                raise ValueError("history cache has no addressable owner")
            digests[family][str(index)] = dict(sorted(owners.items()))
    return digests


def observer_values(mesh: Any, caches: HistoryCaches, embedding: Any, observer_layers: Any,
                    exact: Any, rope: Any, table: Any) -> tuple:
    """The original witness step on one branch's completed REPAIRED history."""
    witness = protocol.WITNESS
    return (_replicated(mesh, np.asarray([witness["token"]], np.int32)),
            _replicated(mesh, np.asarray([witness["position"]], np.int32)),
            _replicated(mesh, np.asarray([witness["context_length"]], np.int32)),
            table, caches.kv, caches.repaired, embedding, observer_layers, exact, rope,
            _replicated(mesh, np.asarray(True, np.bool_)))


def capture_observation(result: Any, **owners: Any) -> tuple:
    producers = replicated_host(result.dsa.producer_layer_ids, context="observer/producers", **owners)
    if producers.tolist() != list(PRODUCERS):
        raise ValueError("history observer producers differ")
    observation = dict(positions=replicated_host(result.dsa.selected_positions, context="observer/positions", **owners),
                       counts=replicated_host(result.dsa.selected_valid_counts, context="observer/counts", **owners),
                       scores=replicated_host(result.dsa.selected_scores, context="observer/scores", **owners))
    rows, layout, errors = read_boundaries(result, 1, **owners)
    return observation, rows, layout, errors


def reproduce(observed: Mapping[str, np.ndarray], original: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Byte equality of step0 event0..3 arrays against the retained original."""
    arrays = {}
    for field in OBSERVER_FIELDS:
        a, b = np.asarray(observed[field]), np.asarray(original[field])
        arrays[field] = dict(shape=list(a.shape), dtype=str(a.dtype),
                             equal=a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())
    return dict(arrays=arrays, reproduced=all(v["equal"] for v in arrays.values()))


def execute_history(
    calls: Any, *, mesh: Any, config: Any, plan: tuple[protocol.Step, ...],
    prompt_tokens: np.ndarray, embedding: Any, layers: Any, observer_layers: Any,
    wk: Any, exact: Any, rope: Any, originals: Mapping[str, Mapping[str, np.ndarray]],
    expected_prompt_sha256: str = protocol.PROMPT_SHA,
) -> dict[str, Any]:
    """Run both histories, compare every boundary, observe both, require reproduction.

    Raises after persisting the report when a block is unhealthy, when a
    branch's observer fails to reproduce its retained original, or when the
    originals budget would be exceeded. A raised error never discards evidence.
    """
    import jax

    def preflight() -> None:
        if (not isinstance(prompt_tokens, np.ndarray) or prompt_tokens.dtype != np.int32
                or prompt_tokens.ndim != 1 or prompt_tokens.size == 0):
            raise ValueError("history prompt must be a non-empty int32 vector")
        if sha256(prompt_tokens.tobytes()).hexdigest() != expected_prompt_sha256:
            raise ValueError("history prompt bytes differ from the pinned original")
        if tuple(plan) != protocol.plan(int(prompt_tokens.size)):
            raise ValueError("history plan is not the protocol plan for this prompt")
        if not set(protocol.FRONTIER_PROGRAMS + ("observer",)) <= set(calls.programs):
            raise ValueError("history programs are not all compiled")
        if len(wk) != len(PRODUCERS) or len(exact) != len(PRODUCERS) or len(layers) != len(LAYERS) or len(observer_layers) != len(LAYERS):
            raise ValueError("history weight/WK/exact inventory differs")
        if set(originals) != set(protocol.BRANCHES) or any(set(v) != set(OBSERVER_FIELDS) for v in originals.values()):
            raise ValueError("history retained originals incomplete")
        if config.context_capacity != protocol.CAPACITY:
            raise ValueError("history requires the original8K capacity")
        evidence = calls.root / "history_report.json"
        if evidence.exists() or evidence.is_symlink():
            raise ValueError("history evidence root already contains a report; no replay/overwrite")
        for pattern in ("first_difference_group*.npz*", "observer_*.npz*",
                        "unhealthy_*.npz*", "refused_completed_*.npz*", "refused_replica_*.npz*"):
            if any(calls.root.glob(pattern)):
                raise ValueError("history evidence root contains prior originals; no overwrite")

    calls.phase("history/preflight", preflight)
    root = calls.root
    owners = dict(local_slots=calls.local_slots,
                  process_index=int(calls.record["jax_process_index"]),
                  platform=jax.default_backend())
    report: dict[str, Any] = dict(
        protocol=protocol.PROTOCOL, complete=False, prompt_length=int(prompt_tokens.size),
        steps=len(plan), groups=[], first_difference=None, retained={}, retained_bytes=0,
        cache_digests={}, observer={}, reproduction={}, numerical_promotion=False,
        performance_claim=False, cause_claim=False,
    )
    calls.record["history"] = report

    def persist() -> None:
        _atomic_json(root / "history_report.json", report)

    def bind_layout(layout: dict[str, Any]) -> None:
        if report.setdefault("boundary_layout", layout) != layout:
            raise ValueError("history local owner/column layout changed between calls")

    def retain(label: str, arrays: Mapping[str, np.ndarray]) -> None:
        amount = sum(int(v.nbytes) for v in arrays.values())
        if report["retained_bytes"] + amount + (len(report["retained"]) + 1) * (1 << 20) > protocol.ORIGINALS_LIMIT:
            raise ValueError("history retained originals exceed the fixed rank budget")
        path = root / f"{label}.npz"
        if any(p.exists() or p.is_symlink() for p in (path, path.with_suffix(".npz.pending"))):
            raise ValueError(f"history original already exists; refusing overwrite: {path}")
        digest = save_arrays(path, {k: _bits(v) for k, v in arrays.items()})
        report["retained"][label] = dict(npz_sha256=digest, bytes=path.stat().st_size, raw_array_bytes=amount)
        report["retained_bytes"] += amount

    table = calls.phase("history/table", lambda: _replicated(
        mesh, np.arange(config.page_count, dtype=np.int32)[None]))
    caches = {branch: calls.phase(f"history/{branch}/caches", lambda: fresh_caches(mesh, config))
              for branch in protocol.BRANCHES}
    calls.phase("history/independent", lambda: independent(*caches.values()))
    healthy = calls.phase("history/initial_health", lambda: {
        branch: _replicated(mesh, np.asarray(True, np.bool_)) for branch in protocol.BRANCHES})
    pending: dict[int, dict[str, Any]] = {}

    def finish_group(group: int) -> None:
        entry = pending.pop(group)
        candidate, control = entry["candidate"], entry["control"]
        if control["count"] != candidate["count"] or control["offset"] != candidate["offset"]:
            raise ValueError("history group rows are not aligned")
        joined = {layer: {field: np.concatenate([part[layer][field] for part in control["parts"]])
                          for field in FIELDS} for layer in LAYERS}
        comparison = compare_rows(candidate["rows"], joined, offset=candidate["offset"])
        comparison["group"] = group
        report["groups"].append(comparison)
        if comparison["first"] is not None and report["first_difference"] is None:
            report["first_difference"] = dict(comparison["first"], group=group)
            arrays = {}
            for branch, rows in (("candidate", candidate["rows"]), ("control", joined)):
                for layer in LAYERS:
                    for field in FIELDS:
                        arrays[f"{branch}_layer{layer}_{field}"] = rows[layer][field]
            retain(f"first_difference_group{group}", arrays)
        persist()

    for step in plan:
        branch = step.branch
        values = calls.phase(f"history/{step.index}/inputs", lambda: block_values(
            mesh, prompt_tokens[step.offset: step.offset + step.count], step.offset,
            128 if step.program.endswith("b128") else protocol.TAIL_ROWS, caches[branch],
            embedding, layers, wk, rope, table, healthy[branch]))
        captured: dict[str, Any] = {}

        def preserve(result: Any, step=step, captured=captured) -> None:
            rows, layout, errors = read_boundaries(result, step.count, **owners)
            captured["rows"] = rows
            bind_layout(layout)
            accepted = bool(replicated_host(result.healthy, context="history/healthy", **owners))
            if not accepted or errors:
                report["unhealthy"] = dict(step=step._asdict(), capture_errors=errors)
                retain(f"unhealthy_step{step.index}", {f"layer{l}_{f}": captured["rows"][l][f]
                                                      for l in LAYERS for f in (*FIELDS, "health")})
                persist()
                raise ValueError("history block unhealthy; originals preserved")

        try:
            result = calls.call(f"history/{step.index}/{branch}", step.program, values, preserve=preserve)
        except Exception as exc:
            # The callback has completed local readback, but the shared wrapper
            # can still refuse publication/consensus or its post-call HBM check.
            # Persist this completed step before propagating the terminal error;
            # successful steps need not generate hundreds of redundant dumps.
            report["refused_step"] = dict(step=step._asdict(), error=str(exc))
            try:
                if isinstance(exc, ReplicaMismatch):
                    report["replica_failure"] = exc.metadata
                    retain(f"refused_replica_step{step.index}", exc.originals)
                elif "rows" in captured and "unhealthy" not in report:
                    retain(f"refused_completed_step{step.index}", {
                        f"layer{l}_{f}": captured["rows"][l][f] for l in LAYERS for f in (*FIELDS, "health")})
            finally:
                persist()
            raise
        caches[branch], healthy[branch] = result.caches, result.healthy
        group = pending.setdefault(step.group, {"candidate": None, "control": dict(parts=[], count=0, offset=None)})
        if branch == "candidate":
            group["candidate"] = dict(rows=captured["rows"], count=step.count, offset=step.offset)
        else:
            control = group["control"]
            if control["offset"] is None:
                control["offset"] = step.offset
            elif control["offset"] + control["count"] != step.offset:
                raise ValueError("history control blocks are not contiguous within a group")
            control["parts"].append(captured["rows"])
            control["count"] += step.count
        del result, values
        if group["candidate"] is not None and group["control"]["count"] == group["candidate"]["count"]:
            calls.phase(f"history/group{step.group}/compare", lambda g=step.group: finish_group(g))
    if pending:
        raise ValueError("history plan left an unfinished group")

    def digests() -> None:
        report["cache_digests"] = {branch: cache_digests(caches[branch], local_slots=calls.local_slots)
                                   for branch in protocol.BRANCHES}
        report["cache_equality"] = {
            family: {index: report["cache_digests"]["candidate"][family][index]
                     == report["cache_digests"]["control"][family][index]
                     for index in report["cache_digests"]["candidate"][family]}
            for family in CACHE_FAMILIES}
        persist()

    calls.phase("history/cache_digests", digests)

    observations: dict[str, dict[str, np.ndarray]] = {}
    boundaries: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for branch in protocol.BRANCHES:
        values = calls.phase(f"history/observer/{branch}/inputs", lambda b=branch: observer_values(
            mesh, caches[b], embedding, observer_layers, exact, rope, table))

        def preserve_observation(result: Any, b=branch) -> None:
            accepted = bool(replicated_host(result.healthy, context="observer/healthy", **owners))
            observation, rows, layout, errors = capture_observation(result, **owners)
            bind_layout(layout)
            arrays = {**{k: v for k, v in observation.items()},
                      **{f"layer{l}_{f}": rows[l][f] for l in LAYERS for f in (*FIELDS, "health")}}
            report["observer"][b] = dict(healthy=accepted, capture_errors=errors)
            if not accepted or errors:
                # Never file a refused step under the reproduction-witness name.
                retain(f"unhealthy_observer_{b}", arrays)
                persist()
                raise ValueError(f"history observer unhealthy on {b}; originals preserved")
            observations[b], boundaries[b] = observation, rows
            retain(f"observer_{b}", arrays)
            persist()

        try:
            calls.call(f"history/observer/{branch}", "observer", values, preserve=preserve_observation)
        except Exception as exc:
            report["refused_observer"] = dict(branch=branch, error=str(exc))
            try:
                if isinstance(exc, ReplicaMismatch):
                    report["replica_failure"] = exc.metadata
                    retain(f"refused_replica_observer_{branch}", exc.originals)
            finally:
                persist()
            raise
        del values

    def conclude() -> None:
        report["reproduction"] = {branch: reproduce(observations[branch], originals[branch])
                                  for branch in protocol.BRANCHES}
        report["observer_comparison"] = compare_rows(boundaries["candidate"], boundaries["control"],
                                                     offset=protocol.WITNESS["position"])
        report["observations_equal"] = reproduce(observations["candidate"], observations["control"])["reproduced"]
        report["attribution_eligible"] = all(v["reproduced"] for v in report["reproduction"].values())
        report["complete"] = True
        persist()
        if not report["attribution_eligible"]:
            raise ValueError("history observer does not reproduce a retained original; attribution ineligible")

    calls.phase("history/conclude", conclude)
    return report
