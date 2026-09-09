"""Independent original replay; outer collector owns generation/source/HLO binding.

No run is promoted here. Caller must authenticate expected host/device ownership,
model/checkpoint/prompt identities and original compiler artifacts independently.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from scripts.greenfield.prefill_window_evidence import same_json, validate_call_sequence
from scripts.greenfield.ws32_prefill_frontier import compare_cache, replay_cache
from scripts.greenfield.ws32_prefill_frontier_publish import FILES, LABELS, LIMIT, originals
from scripts.greenfield.ws32_prefill_frontier_state import INDEX_LAYERS, METADATA, metadata_errors
from scripts.greenfield.ws32_prefill_frontier_worker import GRAPH, memory_budget

FRONTIERS = dict(zip(LABELS, (0, 0, 128, 32, 64, 96, 128), strict=True))
SHAPES = {"kv": [78, 16, 64, 640], "index": [21, 16, 64, 128], "repair": [21, 16, 64, 128]}


def replay_graphs(record: Mapping[str, Any], texts: Mapping[str, tuple[str, str]], *, args: Any) -> None:
    """Reuse the actual short sealer's HLO/source checks, never a stored verdict."""
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        FROZEN_FIRST_WINDOW_PROFILE, require_acquired_model_source, validate_short_compiled_memory,
    )
    from scripts.greenfield.seal_short_decoder_ws32 import REPO, _replay_batched_graph

    if args.batched_prefill_profile != FROZEN_FIRST_WINDOW_PROFILE:
        raise ValueError("first-window graph replay requires its fixed profile")
    required = {"exact_materialize", "exact_promote", "prefill_chunk"}
    if set(texts) != required or set(record["graphs"]) != required or set(record["compiled_memory_analysis"]) != required:
        raise ValueError("first-window actual graph inventory differs")
    require_acquired_model_source(REPO, profile=FROZEN_FIRST_WINDOW_PROFILE)
    for graph, (stable, optimized) in texts.items():
        actual = _replay_batched_graph(stable, optimized, graph=graph, args=args)
        same_json(record["graphs"][graph], actual, "first-window actual HLO report")
        validate_short_compiled_memory(graph, record["compiled_memory_analysis"][graph],
                                      profile=FROZEN_FIRST_WINDOW_PROFILE, repo=REPO)


def replay_envelope(
    root: Path, record: Mapping[str, Any], journal: Path, *,
    local_slots: Mapping[int, int], expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind both runner copies and the complete original fsynced phase journal.

    This checks a completed diagnostic only, not an interrupted capture. Cloud
    generation binding, actual compiler text replay and authenticated terminal
    fleet cleanup remain the outer collector's obligations.
    """
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        FROZEN_FIRST_WINDOW_PROFILE, short_budget, short_numerical_identity,
    )
    same_json(json.loads((root / "runner.json").read_text()), record, "both runner copies")
    for key, value in {**short_numerical_identity(profile=FROZEN_FIRST_WINDOW_PROFILE),
                       **expected_identity}.items():
        same_json(record.get(key), value, f"diagnostic identity {key}")
    if (record.get("status") != "DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION"
            or record.get("current_phase") != "first_window/final_publication"
            or "phase_error" in record):
        raise ValueError("first-window final publication incomplete")
    raw = journal.read_bytes()
    if not raw or len(raw) > 8 * 1024**2 or not raw.endswith(b"\n"):
        raise ValueError("first-window journal size or final line invalid")
    rows = [json.loads(line) for line in raw.splitlines()]
    graphs = ("exact_materialize", "exact_promote", "prefill_chunk")
    phases = ["preflight", "wide_initial", "narrow_initial", "independent",
              "wide_initial_capture", "narrow_initial_capture", "initial_equality"]
    for name in ("wide", "narrow0", "narrow1", "narrow2", "narrow3"):
        phases += [name + "_inputs", name + "/memory", name + "/execute", name + "/memory_after"]
    phases += ["comparison", "final_publication"]
    expected = [(s, None) for s in ("identity", "runtime_initialize_started",
                "checkpoint_verify_started", "load_started", "load_completed")]
    expected += [(s, g) for g in graphs for s in ("lower_compile_started", "compiled", "raw_written", "inspected")]
    expected += [("first_window/" + s, "prefill_chunk") for s in phases]
    same_json([[r["stage"], r["graph"]] for r in rows], [list(v) for v in expected], "journal phase/graph sequence")
    times = [r["monotonic_seconds"] for r in rows]
    start = record.get("diagnostic_started_monotonic_seconds")
    end = record.get("diagnostic_closed_monotonic_seconds")
    if (any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in [*times, start, end])
            or times != sorted(times)
            or not times[16] <= start <= times[17] <= times[-1] <= end
            or end - start > short_budget(FROZEN_FIRST_WINDOW_PROFILE)):
        raise ValueError("first-window continuation/finalization timing invalid")
    for row in rows:
        for key, value in dict(schema_version=1, artifact_kind="greenfield_ws32_numerical_journal",
            status="NUMERICAL_EXECUTION_PARTIAL", performance_claim=False, numerical_claim=False).items():
            same_json(row.get(key), value, f"journal {key}")
        if row["stage"].startswith("first_window/") and row.get("passed") is not True:
            raise ValueError("first-window journal contains failed phase")
    identity = {**short_numerical_identity(profile=FROZEN_FIRST_WINDOW_PROFILE), "compile_only": False}
    for key in ("code_hash", "launch_process_id", "hostname", "prompt_ids_sha256",
                "checkpoint_manifest_sha256", "checkpoint_success_sha256"):
        identity[key] = record[key]
    same_json(rows[0]["identity"], identity, "journal original identity")
    for key in ("jax_process_index", "mesh_sha256", "topology_fleet_sha256"):
        same_json(rows[2][key], record[key], f"journal runtime {key}")
    ids = rows[2]["local_device_ids"]
    if len(ids) != 4 or any(type(d) is not int for d in ids) or set(ids) != set(local_slots):
        raise ValueError("first-window journal runtime owners differ")
    same_json(rows[3]["checkpoint_verified_device_slots"], record["checkpoint_verified_device_slots"], "journal verified slots")
    same_json(rows[4]["local_device_slots"], record["local_device_slots"], "journal loaded slots")
    same_json(rows[4]["seconds"], record["load_seconds"], "journal load seconds")
    for i, graph in enumerate(graphs):
        compiled, written, inspected = rows[6+4*i:9+4*i]
        same_json(compiled["compiled_memory"], record["compiled_memory_analysis"][graph], "journal compiled memory")
        same_json(compiled["seconds"], record["compile_seconds"][graph], "journal compile seconds")
        same_json(inspected["report"], record["graphs"][graph], "journal inspected report")
        for key in ("stablehlo_sha256", "optimized_hlo_sha256"):
            same_json(written[key], record["graphs"][graph][key], "journal raw compiler SHA")
    return dict(journal_sha256=sha256(raw).hexdigest(), journal_bytes=len(raw),
                continuation_seconds=end-start, completed_phases=len(phases),
                numerical_promotion=False, performance_claim=False)


def _digest(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def replay_host(
    root: Path, record: Mapping[str, Any], *, local_slots: Mapping[int, int],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay original metadata/cache bytes, both branch comparison and5call HBM.

    expected_identity must be independently obtained by the outer collector,
    never populated from the record being checked. This does not recompile or
    authenticate the three original HLO texts; that remains an outer obligation.
    """
    if {p.name for p in originals(root)} != FILES:
        raise ValueError("first-window completed original file set differs")
    if not expected_identity or "jax_process_index" not in expected_identity:
        raise ValueError("first-window needs independent runtime identity")
    for key, expected in expected_identity.items():
        same_json(record.get(key), expected, f"first-window {key}")
    same_json(record["local_slots"], {str(k): v for k, v in local_slots.items()}, "physical slots")
    if (record.get("status") != "DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION"
            or record.get("numerical_promotion") is not False
            or record.get("performance_claim") is not False
            or record["first_window"].get("complete") is not True):
        raise ValueError("first-window diagnostic is incomplete or misclassified")
    validate_call_sequence(
        {**record, "programs": {GRAPH: {"compiled_memory": record["compiled_memory_analysis"][GRAPH]}}},
        record["call_evidence"],
        expected=[("first_window/wide", GRAPH)] + [(f"first_window/narrow{i}", GRAPH) for i in range(4)],
        local_slots=local_slots, names=(GRAPH,), budgeter=memory_budget,
    )
    capsule_reports = {}
    final_rows = {}
    for label in LABELS:
        report = json.loads((root / f"{label}.json").read_text())
        same_json(report, record["first_window"]["originals"][label], "original report")
        frontier = FRONTIERS[label]
        caches = frontier in (0, 128)
        if (report.get("frontier") != frontier or report.get("prompt_length") != 8155
                or report.get("context_capacity") != 8192 or report.get("valid") is not True
                or report.get("errors") != [] or report.get("numerical_promotion") is not False
                or report.get("performance_claim") is not False):
            raise ValueError("first-window observed state contract differs")
        same_json(sorted(report["owners"]), sorted(str(s) for s in local_slots.values()), "owner set")
        path = root / f"{label}.npz"
        raw = path.read_bytes()
        if len(raw) != report["npz_bytes"] or sha256(raw).hexdigest() != report["npz_sha256"]:
            raise ValueError("first-window NPZ original bytes differ")
        with zipfile.ZipFile(path) as archive:
            if sum(v.file_size for v in archive.infolist()) > LIMIT:
                raise ValueError("first-window expanded originals exceed budget")
        with np.load(path, allow_pickle=False) as arrays:
            wanted = {f"slot{s}_{name}" for s in local_slots.values() for name in METADATA}
            if frontier == 128:
                wanted |= {f"slot{s}_{name}_rows" for s in local_slots.values() for name in SHAPES}
            if set(arrays.files) != wanted or len(arrays.files) != len(wanted):
                raise ValueError("first-window original array inventory differs")
            if sum(arrays[n].nbytes for n in arrays.files) != report["raw_array_bytes"]:
                raise ValueError("first-window original raw byte accounting differs")
            for device, slot in local_slots.items():
                owner = report["owners"][str(slot)]
                if owner["device_id"] != device or owner["slot"] != slot:
                    raise ValueError("first-window observed owner differs")
                values = {name: arrays[f"slot{slot}_{name}"] for name in METADATA}
                if metadata_errors(values, frontier=frontier):
                    raise ValueError("first-window original metadata invalid")
                same_json(owner["metadata_sha256"], {n: _digest(v) for n, v in values.items()}, "metadata SHA")
                same_json(sorted(owner["caches"]), sorted(SHAPES) if caches else [], "cache families")
                for name in owner["caches"]:
                    evidence = owner["caches"][name]
                    for key, expected in dict(initial=frontier == 0, valid=True,
                        violations=dict(nonfinite=0, outside_nonzero=0), samples=[],
                        samples_truncated=False, numerical_promotion=False).items():
                        same_json(evidence.get(key), expected, f"cache {key}")
                    capsule = evidence["cache"]
                    for key, expected in dict(shape=SHAPES[name], slot=slot,
                        layer_ids=list(range(78)) if name == "kv" else list(INDEX_LAYERS),
                        block_table=values["block_tables"].tolist()).items():
                        same_json(capsule[key], expected, f"cache {key}")
                    rows = (np.zeros((SHAPES[name][0], 64 if slot//4 < 2 else 0, SHAPES[name][3]), np.uint16)
                            if frontier == 0 else arrays[f"slot{slot}_{name}_rows"])
                    replay_cache(rows, capsule)
                    if frontier == 128:
                        final_rows[(label, slot, name)] = rows
        capsule_reports[label] = report
    same_json(capsule_reports["wide_initial"]["owners"], capsule_reports["narrow_initial"]["owners"], "complete initial state")
    comparisons = {}
    for slot in local_slots.values():
        left, right = (capsule_reports[label]["owners"][str(slot)] for label in ("wide_final", "narrow_128"))
        comparisons[str(slot)] = dict(
            metadata_bytes_equal=left["metadata_sha256"] == right["metadata_sha256"],
            caches={name: compare_cache(final_rows[("wide_final", slot, name)], left["caches"][name]["cache"],
                                        final_rows[("narrow_128", slot, name)], right["caches"][name]["cache"])
                    for name in SHAPES},
        )
    result = dict(owners=comparisons, calls=5, numerical_promotion=False,
                  performance_claim=False, scope="FIRST128_ONLY_NOT_ROOT_CAUSE")
    same_json(json.loads((root / "comparison.json").read_text()), result, "saved comparison")
    same_json(record["first_window"]["comparison"], result, "runner comparison")
    return dict(comparison=result, states=capsule_reports)


def replay_replicas(hosts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Join eight independently replayed hosts; no rank-order ownership shortcut."""
    if len(hosts) != 8:
        raise ValueError("first-window replica replay needs eight hosts")
    for label in LABELS:
        owners = {}
        for host in hosts:
            for slot, owner in host["states"][label]["owners"].items():
                if slot in owners:
                    raise ValueError("first-window duplicate physical slot")
                owners[slot] = owner
        if set(owners) != {str(s) for s in range(32)}:
            raise ValueError("first-window replica coverage differs")
        for slot, owner in owners.items():
            same_json(owner["metadata_sha256"], owners["0"]["metadata_sha256"], "fleet metadata replicas")
            for name, evidence in owner["caches"].items():
                reference = owners[str(int(slot)//4*4)]["caches"][name]
                same_json(evidence["cache"]["whole_cache_sha256"], reference["cache"]["whole_cache_sha256"], "fleet feature cache replicas")
    earliest = {}
    for name in SHAPES:
        changed = [value for host in hosts for owner in host["comparison"]["owners"].values()
                   if (value := owner["caches"][name]["earliest_differing_writer"]) is not None]
        earliest[name] = min(changed) if changed else None
    return dict(physical_owners=32, earliest_differing_writer=earliest,
                numerical_promotion=False, performance_claim=False,
                scope="FIRST128_CACHE_WRITERS_NOT_FIRST_ERRONEOUS_LAYER")
