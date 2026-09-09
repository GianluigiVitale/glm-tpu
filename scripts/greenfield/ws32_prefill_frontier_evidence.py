"""Independent original replay; outer collector owns generation/source/HLO binding.

No run is promoted here. Caller must authenticate expected host/device ownership,
model/checkpoint/prompt identities and original compiler artifacts independently.
"""

from __future__ import annotations

from hashlib import sha256
import json
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
