"""Replay native cold readiness from ORIGINAL files and authenticated owners.

This proves the preparation and pre-cache admission only, NOT request execution,
actual cache/peak behavior, trace coverage, delivered TTFT or benchmark quality.
The outer final sealer must add those request records before any goal SUCCESS.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.greenfield import ws32_native_benchmark_memory as memory
from scripts.greenfield import ws32_native_benchmark_programs as programs
from scripts.greenfield import ws32_native_benchmark_transport as transport
from scripts.greenfield import ws32_delivery_phase_evidence as original
from scripts.greenfield import ws32_delivery_wk as wk
from scripts.greenfield import ws32_delivery_decode as decode
from scripts.greenfield import ws32_dense_frontier_admission as wk_admission
from scripts.greenfield.prefill_window_evidence import same_json, validate_call_sequence
from scripts.greenfield.ws32_history_call_evidence import load_calls
from scripts.greenfield.ws32_history_preflight import _plain_path
from scripts.greenfield.ws32_delivery_hlo import FRESH_POLICY


def _read(root: Path, relative: str) -> Any:
    path = root / relative
    _plain_path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= transport.file_limits()[relative]:
        raise ValueError("native cold replay original name/size differs")
    return json.loads(path.read_bytes())


def replay_preparation(root: Path, parent: Mapping, slots: Mapping[int, int],
                       full_index_layers: Sequence[int]) -> dict:
    """Use the original44-call/overlay/monotone-peak checks in native order."""
    phases = []
    for folder, field, kind, roles in (
        ("wk", "wk_preparation", "ws32_delivery_wk_phase_v1", wk.ROLES),
        ("exact", "exact_preparation", "ws32_delivery_decode_preparation_v1", decode.ROLES),
    ):
        record = _read(root, folder + "/runner.json")
        same_json(record, parent[field], "native nested preparation original")
        if (record.get("artifact_kind") != kind or record.get("complete") is not True
                or record.get("phase_contract") != transport.PROFILE
                or record.get("performance_claim") is not False
                or set(record["programs"]) != set(roles)):
            raise ValueError("native preparation scope/complete/graph inventory differs")
        for key in ("profile", "code_hash", "launch_process_id", "jax_process_index",
                    "context_capacity", "mesh_sha256", "topology_sha256", "topology_fleet_sha256", "local_slots"):
            same_json(record[key], parent[key], "native preparation parent identity")
        phases.append(record)
    wr, dr = phases
    if wr.get("model_math_changed") is not False or dr.get("numerical_value_replay") is not False:
        raise ValueError("native preparation overstates numerical proof")
    schedule = [(f"layer{layer}/{name}", name) for layer in full_index_layers for name in wk.ROLES]
    if len(schedule) != 42 or len(set(full_index_layers)) != 21:
        raise ValueError("native preparation requires all21 full-indexer producers")
    calls = load_calls(root / "wk", wr, expected_calls=42)
    validate_call_sequence(wr, calls, expected=schedule, local_slots=slots,
                           names=wk.ROLES, budgeter=wk.memory_budget)
    hashes = original.validate_wk_outputs(calls, slots)
    process = parent["jax_process_index"]
    before, after = original.overlay_replay(dr,
        {"strategy_nd_dense_overlay": parent["overlay"], "jax_process_index": process}, slots)
    original._advance(original._boundary(calls[-1]["post_memory"], slots, process, nested=False), before)
    exact_calls = dr["call_evidence"]
    if not isinstance(exact_calls, list) or len(exact_calls) != 2:
        raise ValueError("native exact preparation call count differs")
    for index, call in enumerate(exact_calls):
        name = decode.ROLES[index]
        validate_call_sequence(dr, [call], expected=[(f"delivery_decode/{name}", name)],
            local_slots=slots, names=decode.ROLES[:index + 1], budgeter=decode.memory_budget)
        original._advance(after, original._boundary(call["census"]["devices"], slots, process, nested=True))
        after = original._boundary(call["post_memory"], slots, process, nested=False)
        if not isinstance(call.get("output_schema"), list) or not call["output_schema"]:
            raise ValueError("native exact completed output schema missing")
    initial = _read(root, "initial_memory.json")
    same_json(initial["local_slots"], parent["local_slots"], "native initial memory owners")
    if initial["process_index"] != process or initial["phase"] != "before_cache":
        raise ValueError("native initial memory phase/process differs")
    memory.validate_record(initial)
    same_json(initial["compiled_memory"],
        {name: parent["programs"][name]["compiled_memory"] for name in memory.ROLES},
        "native initial resident compiler memory")
    original._advance(after, original._boundary(initial["census"]["devices"], slots, process, nested=True))
    return dict(wk_output_hashes=hashes, calls=44, initial_memory=initial)


def replay_cold_fleet(*, root: Path, repo: Path, pin: str,
                      captures: Sequence[Mapping[str, Any]], physical_mesh: Any,
                      topology_hash: str, fleet_hash: str,
                      full_index_layers: Sequence[int]) -> dict:
    """Join all32 actual owners and replay each unique actual graph once.

    Caller must authenticate captures using validate_ws32_topology_fleet and
    bind source/manifest/generations through the outer collector first. No
    acceptance is inherited from DB620's one-way memory result.
    """
    programs.require_source(repo)
    if len(captures) != 8 or len(set(physical_mesh.flattened_device_ids)) != 32:
        raise ValueError("native cold replay requires authenticated8x4 topology")
    pins = json.loads((repo / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    results, seen_devices, seen_processes, graph_cache = [], set(), set(), {}
    for rank, capture in enumerate(captures):
        directory = root / f"native.rank{rank}"
        record = _read(directory, "runner.json")
        process = capture["jax_process_index"]
        ids = set(capture["local_device_ids"])
        if (record.get("profile") != transport.PROFILE
                or record.get("artifact_kind") != "ws32_native_cold_load_v1"
                or record.get("complete") is not True or record.get("compile_only") is not False
                or record.get("performance_claim") is not False
                or record.get("code_hash") != pin or record.get("launch_process_id") != rank
                or record.get("jax_process_index") != process or process in seen_processes
                or record.get("context_capacity") != programs.PLAN.context_capacity
                or record.get("mesh_sha256") != physical_mesh.mesh_hash
                or record.get("topology_sha256") != topology_hash
                or record.get("topology_fleet_sha256") != fleet_hash
                or len(ids) != 4 or ids & seen_devices):
            raise ValueError("native cold identity/topology/complete scope differs")
        slots = {device: slot for slot, device in enumerate(physical_mesh.flattened_device_ids) if device in ids}
        same_json(record["local_slots"], [dict(device_id=d, slot=s) for d, s in sorted(slots.items())], "native physical slots")
        checkpoint = record["checkpoint"]
        inventory_sha = json.loads((repo / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())["source_inventory_sha256"]
        if (checkpoint["manifest_sha256"] != pins["GLM_GREENFIELD_WS32_CHECKPOINT_MANIFEST_SHA"]
                or checkpoint["success_sha256"] != pins["GLM_GREENFIELD_WS32_CHECKPOINT_SUCCESS_SHA"]
                or checkpoint["source_inventory_sha256"] != inventory_sha
                or checkpoint["verified_slots"] != sorted(slots.values())):
            raise ValueError("native checkpoint identity/verified owner set differs")
        for key, env in (("manifest_sha256", "MANIFEST_SHA"), ("manifest_file_sha256", "MANIFEST_FILE_SHA"),
                         ("success_file_sha256", "SUCCESS_FILE_SHA")):
            if record["overlay"][key] != pins["GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_" + env]:
                raise ValueError("native tested overlay identity differs")
        if set(record["programs"]) != set(memory.ROLES):
            raise ValueError("native resident role inventory differs")
        for graph, folder in transport.GRAPHS.items():
            meta = record["programs"][graph] if not folder else record[
                "wk_preparation" if folder == "wk" else "exact_preparation"]["programs"][graph]
            texts = []
            for form, field in (("stablehlo.mlir", "stablehlo_sha256"), ("optimized_hlo.txt", "optimized_hlo_sha256")):
                path = directory / folder / f"{graph}.{form}"
                _plain_path(path)
                cap = transport.file_limits()[str(Path(folder) / path.name)]
                if not path.is_file() or not 0 < path.stat().st_size <= cap:
                    raise ValueError("native compiler original missing or oversized")
                text = path.read_text()
                if sha256(text.encode()).hexdigest() != meta[field]:
                    raise ValueError("native compiler original differs from worker record")
                texts.append(text)
            key = (graph, meta["stablehlo_sha256"], meta["optimized_hlo_sha256"],
                   json.dumps(meta["compiled_memory"], sort_keys=True))
            if key not in graph_cache:
                graph_cache[key] = (wk_admission.inspect_program(graph, *texts, meta["compiled_memory"])
                    if folder == "wk" else programs.inspect_hlo(*texts, repo=repo, graph=graph,
                                                               expected_optimized=meta["optimized_hlo_sha256"]))
            report = graph_cache[key]
            # Worker used the explicit fresh-optimized mode; replay binds those
            # exact bytes, but retains that original identity-policy annotation.
            if folder != "wk":
                if meta["admission"].get("optimized_identity_policy") != FRESH_POLICY:
                    raise ValueError("native worker must inspect fresh actual optimized HLO")
                report = dict(report, optimized_identity_policy=FRESH_POLICY)
            same_json(meta["admission"], report, "native graph structural replay")
            seconds = meta["compile_seconds"]
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("native compile timing is invalid")
        if len(graph_cache) != 10:
            raise ValueError("native rank compiler identities differ from previous ranks")
        result = replay_preparation(directory, record, slots, full_index_layers)
        results.append(result)
        seen_devices.update(ids)
        seen_processes.add(process)
    if seen_devices != set(physical_mesh.flattened_device_ids) or seen_processes != set(range(8)):
        raise ValueError("native cold evidence lacks32 distinct physical owners")
    if any(result["wk_output_hashes"] != results[0]["wk_output_hashes"] for result in results[1:]):
        raise ValueError("native completed WK differs across hosts")
    return dict(schema="ws32_native_cold_replay_v1", owners=32, calls_per_rank=44,
        unique_graphs=10, original_preparation_replayed=True,
        cache_allocation_executed=False, request_execution_proven=False,
        benchmark_quality_proven=False)
