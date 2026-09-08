"""Resident-buffer accounting and conservative pre-execution prefill budgeting.

No full-model allocation, host tensor download, launch authority or measured
execution-peak claim. The caller must supply every resident executable and use
fleet-wide agreement before dispatch. Actual numerical peaks remain mandatory.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import json


SCHEMA = "ws32_prefill_resident_buffers_v1"
MEMORY_FIELDS = (
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "temp_size_in_bytes",
    "generated_code_size_in_bytes",
    "alias_size_in_bytes",
)


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(
            f"{name} must be {'positive' if positive else 'nonnegative'} int"
        )
    return value


def capture_resident_buffers(
    named_trees: Mapping[str, Any], *, devices: Sequence[Any]
) -> dict[str, Any]:
    """Inventory named roots PLUS all JAX-live arrays, without copying their data.

    Equal nonzero buffer pointers on the same device deduplicate physical
    storage; different pointers are never merged from assumed overlap. Where
    a backend does not expose pointers, distinct array objects count separately
    (a conservative upper count, explicitly labelled, never guessed aliases).
    Local shard allocated sizes, not global logical model sizes, are counted.
    Captured pointers/object ids are not serialized. All arrays remain alive
    throughout the snapshot, preventing allocator address reuse during census.
    """
    import jax

    if not named_trees or any(
        type(name) is not str or not name for name in named_trees
    ):
        raise ValueError("resident roots need nonempty names")
    if "__all_live_arrays__" in named_trees:
        raise ValueError("reserved resident root name")
    device_ids = [int(device.id) for device in devices]
    if not devices or len(set(device_ids)) != len(device_ids):
        raise ValueError("resident census needs unique local devices")
    device_identity = {
        int(device.id): (str(device.platform), int(device.process_index))
        for device in devices
    }
    roots = dict(named_trees)
    roots["__all_live_arrays__"] = tuple(jax.live_arrays())
    # Keep the original roots AND all leaves pinned until the stats are read.
    arrays: dict[int, Any] = {}
    labels: dict[int, set[str]] = {}
    for label, tree in roots.items():
        for leaf in jax.tree.leaves(tree):
            if not isinstance(leaf, jax.Array):
                raise ValueError(f"non-array leaf in resident root {label}")
            arrays[id(leaf)] = leaf
            labels.setdefault(id(leaf), set()).add(label)
    buffers: dict[int, dict[tuple, dict[str, Any]]] = {i: {} for i in device_ids}
    for identity, array in arrays.items():
        if array.is_deleted():
            raise ValueError("deleted array in resident census")
        # Before dispatch, never in a timed model region. No host materialization.
        array.block_until_ready()
        for shard in array.addressable_shards:
            device_id = int(shard.device.id)
            if device_id not in buffers:
                raise ValueError("resident array on an undeclared local device")
            if (
                str(shard.device.platform),
                int(shard.device.process_index),
            ) != device_identity[device_id]:
                raise ValueError("resident array backend/process identity differs")
            local = shard.data
            size = _integer(
                int(local.on_device_size_in_bytes()), "local allocated bytes"
            )
            if size < int(local.nbytes):
                raise ValueError("local allocated size smaller than logical payload")
            if size == 0:
                continue
            try:
                pointer = local.unsafe_buffer_pointer()
            except (AttributeError, NotImplementedError, RuntimeError, ValueError):
                pointer = None
            if type(pointer) is int and pointer > 0:
                key, mode = ("pointer", pointer), "physical_pointer"
            else:
                # One global array has at most one shard per local device here.
                key, mode = ("object", identity), "distinct_object_upper_count"
            entry = buffers[device_id].setdefault(
                key, dict(bytes=0, groups=set(), identity_mode=mode)
            )
            # Equal base pointers may expose differently sized views. Count the
            # largest allocation, never the smallest view or a sum of aliases.
            entry["bytes"] = max(entry["bytes"], size)
            entry["groups"].update(labels[identity])
    records = []
    for device in devices:
        device_id = int(device.id)
        entries = list(buffers[device_id].values())
        if not entries:
            raise ValueError("local device has no resident array allocations")
        stats = device.memory_stats()
        if stats is None:
            raise ValueError("device memory counters unavailable")
        counters = {
            key: _integer(stats.get(key), key, positive=key == "bytes_limit")
            for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        }
        if (
            not counters["bytes_in_use"]
            <= counters["peak_bytes_in_use"]
            <= counters["bytes_limit"]
        ):
            raise ValueError("inconsistent device memory counters")
        records.append(
            dict(
                device_id=device_id,
                platform=str(device.platform),
                process_index=int(device.process_index),
                buffers=[
                    dict(
                        buffer_id=i,
                        bytes=e["bytes"],
                        groups=sorted(e["groups"]),
                        identity_mode=e["identity_mode"],
                    )
                    for i, e in enumerate(entries)
                ],
                accounted_resident_bytes=sum(e["bytes"] for e in entries),
                group_resident_bytes={
                    name: sum(e["bytes"] for e in entries if name in e["groups"])
                    for name in sorted(roots)
                },
                conservative_identity_fallback=any(
                    e["identity_mode"] != "physical_pointer" for e in entries
                ),
                memory_stats=counters,
            )
        )
    return dict(
        schema_version=SCHEMA,
        devices=records,
        roots=sorted(roots),
        array_object_count=len(arrays),
        includes_all_live_arrays=True,
        execution_peak_measured=False,
    )


def budget_prefill_execution(
    census: Mapping[str, Any],
    compiled_memory: Mapping[str, Mapping[str, Any]],
    *,
    active_graph: str,
    resident_graphs: Sequence[str],
    required_reserve_bytes: int,
) -> dict[str, Any]:
    """Conservative admission estimate from a current completed-buffer census.

    Resident baseline = max(census bytes, allocator current bytes, active graph
    argument allocation). Add every RESIDENT executable's generated-code bytes,
    then only the active graph's full output and scratch. No alias subtraction:
    the acquired prefill has no donation. Never sum seven graph argument trees.
    Generated code and allocator counters can overlap: double counting that
    component is intentional because residency attribution is not established.
    Reserve is explicit and must be preregistered by the protected workload.
    This estimate does not bound compiler/runtime allocations not reported by
    these APIs; measured numerical peaks and all other gate checks still apply.
    """
    reserve = _integer(required_reserve_bytes, "required reserve", positive=True)
    if (
        census.get("schema_version") != SCHEMA
        or census.get("includes_all_live_arrays") is not True
    ):
        raise ValueError("memory admission requires complete live-array census")
    if active_graph not in ("prefill_chunk", "prefill_tail"):
        raise ValueError("memory budget applies only to acquired prefill pair")
    names = list(resident_graphs)
    if len(names) != len(set(names)) or not {"prefill_chunk", "prefill_tail"}.issubset(
        names
    ):
        raise ValueError("both distinct prefill executables must be budgeted resident")
    analyses = {}
    for name in names:
        if name not in compiled_memory:
            raise ValueError(f"missing resident executable memory: {name}")
        analyses[name] = {
            key: _integer(compiled_memory[name].get(key), f"{name}.{key}")
            for key in MEMORY_FIELDS
        }
    current = analyses[active_graph]
    if current["alias_size_in_bytes"] != 0:
        raise ValueError("acquired no-donation prefill allocation changed")
    code = sum(a["generated_code_size_in_bytes"] for a in analyses.values())
    outputs, scratch = current["output_size_in_bytes"], current["temp_size_in_bytes"]
    devices = []
    seen = set()
    for device in census.get("devices", []):
        device_id = _integer(device["device_id"], "device id")
        if device_id in seen:
            raise ValueError("duplicate memory device")
        seen.add(device_id)
        entries = device["buffers"]
        resident = sum(
            _integer(e["bytes"], "buffer bytes", positive=True) for e in entries
        )
        if resident != device["accounted_resident_bytes"]:
            raise ValueError("resident buffer total drifted")
        stats = device["memory_stats"]
        used = _integer(stats["bytes_in_use"], "resident allocator bytes")
        peak = _integer(stats["peak_bytes_in_use"], "prior allocator peak")
        limit = _integer(stats["bytes_limit"], "device limit", positive=True)
        if not used <= peak <= limit:
            raise ValueError("memory counters inconsistent")
        baseline = max(resident, used, current["argument_size_in_bytes"])
        estimate = baseline + code + outputs + scratch
        headroom = limit - max(estimate, peak)
        devices.append(
            dict(
                device_id=device_id,
                resident_baseline_bytes=baseline,
                resident_code_bytes=code,
                active_output_bytes=outputs,
                active_temp_bytes=scratch,
                estimated_peak_bytes=estimate,
                previous_measured_peak_bytes=peak,
                limit_bytes=limit,
                estimated_headroom_bytes=headroom,
                estimate_fits=headroom >= reserve,
            )
        )
    if not devices:
        raise ValueError("memory census contains no devices")
    return dict(
        schema_version="ws32_prefill_memory_budget_v1",
        active_graph=active_graph,
        resident_graphs=names,
        required_reserve_bytes=reserve,
        devices=devices,
        estimate_fits=all(d["estimate_fits"] for d in devices),
        execution_peak_measured=False,
        numerical_admission=False,
        code_allocation_attribution="conservative_addition_not_measured_residency",
    )


def make_prefill_memory_record(
    compiled: Mapping[str, Any],
    named_trees: Mapping[str, Any],
    *,
    devices: Sequence[Any],
    required_reserve_bytes: int,
    additional_resident_executables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind both real compiled analyses to a single all-live resident census."""
    if set(compiled) != {"prefill_chunk", "prefill_tail"}:
        raise ValueError("prefill memory snapshot requires exactly the graph pair")
    additional = dict(additional_resident_executables or {})
    if set(additional) & set(compiled):
        raise ValueError("additional resident executable names overlap prefill pair")
    resident = {**compiled, **additional}
    if any(type(name) is not str or not name for name in resident):
        raise ValueError("resident executable needs a nonempty name")
    analyses = {}
    for name, executable in resident.items():
        analysis = executable.memory_analysis()
        analyses[name] = {key: getattr(analysis, key, None) for key in MEMORY_FIELDS}
    census = capture_resident_buffers(named_trees, devices=devices)
    return dict(
        schema_version="ws32_prefill_memory_record_v1",
        census=census,
        compiled_memory=analyses,
        required_reserve_bytes=required_reserve_bytes,
        budgets={
            name: budget_prefill_execution(
                census,
                analyses,
                active_graph=name,
                resident_graphs=tuple(resident),
                required_reserve_bytes=required_reserve_bytes,
            )
            for name in compiled
        },
    )


def validate_prefill_memory_record(record: Mapping[str, Any]) -> None:
    """Recompute both estimates; a reported pass never supplies the verdict."""
    if (
        set(record)
        != {
            "schema_version",
            "census",
            "compiled_memory",
            "required_reserve_bytes",
            "budgets",
        }
        or record["schema_version"] != "ws32_prefill_memory_record_v1"
    ):
        raise ValueError("prefill memory record schema drifted")
    analyses = record["compiled_memory"]
    pair = {"prefill_chunk", "prefill_tail"}
    if not pair.issubset(analyses) or set(record["budgets"]) != pair:
        raise ValueError("prefill memory record omits a graph")
    for name in pair:
        declared = record["budgets"][name]
        if set(declared["resident_graphs"]) != set(analyses):
            raise ValueError("prefill memory budget omits declared resident executable")
        recomputed = budget_prefill_execution(
            record["census"],
            analyses,
            active_graph=name,
            resident_graphs=declared["resident_graphs"],
            required_reserve_bytes=record["required_reserve_bytes"],
        )
        if (
            json.dumps(declared, sort_keys=True, allow_nan=False)
            != json.dumps(recomputed, sort_keys=True, allow_nan=False)
            or recomputed["estimate_fits"] is not True
        ):
            raise ValueError("prefill memory estimate fails or record drifted")
