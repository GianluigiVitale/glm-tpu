"""Full local-original collector for the bounded two-branch history diagnostic.

The protected parent must generation-authenticate the collected files first.
This reuses the independent graph/call, materializer and boundary readers and
joins current owners to the retained original runtime. No launch or cleanup.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield import ws32_history_preflight as preflight
from scripts.greenfield import ws32_history_entry as entry
from scripts.greenfield import ws32_history_execution_evidence as execution
from scripts.greenfield import ws32_history_materializer_evidence as materializers
from scripts.greenfield import ws32_history_boundary_evidence as boundaries
from scripts.greenfield.prefill_window_evidence import same_json

PHASES = ("history/bind_runtime", "history/prepare_original_inputs", "history/load_selected",
          "history/load_overlay", "history/place_original_rope", "history/setup",
          *(f"history/compile/{n}" for n in protocol.PROGRAMS),
          "history/finalize", "history/terminal")


def _phase_identity(record: Mapping) -> None:
    phases = record.get("acquisition_phases", {})
    if set(phases) != set(PHASES):
        raise ValueError("history fleet phase inventory differs")
    for name, value in phases.items():
        seconds = value.get("seconds")
        if (value.get("status") != "COMPLETE" or value.get("error") is not None
                or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
            raise ValueError(f"history fleet phase failed: {name}")


def _overlay_identity(record: Mapping, prior: Mapping, slots: Mapping, overlay: Any) -> None:
    actual = record["strategy_nd_dense_overlay"]
    same_json(actual, prior["strategy_nd_dense_overlay"], "history original overlay")
    fixed = dict(manifest_sha256=overlay.manifest["manifest_sha256"],
                 manifest_file_sha256=overlay.manifest_file_sha256,
                 success_file_sha256=overlay.success_file_sha256)
    same_json({k: actual[k] for k in fixed}, fixed, "history authenticated overlay metadata")
    rows = []
    for device, slot in slots.items():
        expert, feature = divmod(slot, 4)
        amount = 0
        for layer in range(3):
            source = overlay.records[layer, expert, feature]
            if len(source["tensors"]) != 4:
                raise ValueError("history overlay tensor count differs")
            amount += sum(v["byte_count"] for v in source["tensors"].values())
            rows.append(dict(device_id=device, expert_coordinate=expert,
                             feature_coordinate=feature, layer_id=layer,
                             file_sha256=source["sha256"]))
        if amount != protocol.OVERLAY_BYTES:
            raise ValueError("history overlay payload bytes differ")
    key = lambda row: (row["device_id"], row["layer_id"])
    same_json(sorted(actual["local_records"], key=key), sorted(rows, key=key),
              "history overlay actual owners")


def validate_fleet(*, root: Path, records: list[dict], repo: Path, original_root: Path,
                   read_source: Callable[[int, str], Any]) -> dict:
    """All eight original-file replays; missing source reconstruction is fatal."""
    preflight._plain_path(root)
    preflight._plain_path(original_root)
    if (not callable(read_source) or not isinstance(records, list) or len(records) != 8
            or any(type(r.get("launch_rank")) is not int or r["launch_rank"] != i
                   for i, r in enumerate(records))):
        raise ValueError("history fleet rank/source-reader inventory differs")
    tag, pin = records[0].get("tag"), records[0].get("code_hash")
    if not protocol.is_tag(tag) or not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{40}", pin) is None:
        raise ValueError("history fleet tag/pin differs")
    # Same fixed repository receipt consumed by preflight.selected_metadata;
    # runner and retained-preflight agreement alone is not an authority.
    checkpoint_pins = json.loads((repo / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())
    if not isinstance(checkpoint_pins, dict) or not checkpoint_pins:
        raise ValueError("history fixed host-admission checkpoint pins invalid")
    for record in records:
        same_json(record.get("checkpoint_pins"), checkpoint_pins,
                  "history fixed host-admission checkpoint pins")
    bindings = materializers.checkpoint_bindings(repo)
    if set(bindings.owners) != set(range(32)):
        raise ValueError("history source metadata lacks all32owners")
    ids, processes, hosts = {}, set(), set()
    graph_signature, graph_cache = None, {}
    materializer_reports, boundary_reports, execution_reports = [], [], []
    for rank, record in enumerate(records):
        rankroot = root / f"rank{rank}"
        runners, observations, identity = preflight.load_originals(
            original_root / f"rank{rank}" / "retained_reference", repo=repo, rank=rank)
        prior = runners["candidate"]
        slots = {v["device_id"]: v["device_slot"] for v in prior["local_device_slots"]}
        fixed = dict(status=entry.STATUS, tag=tag, code_hash=pin, launch_rank=rank,
            kernel=protocol.KERNEL, protocol=protocol.PROTOCOL, profile=entry.admission.PROFILE,
            diagnostic_only=True, admission_only=False, compile_only=False,
            numerical_promotion=False, performance_claim=False, iterations=0, latency=None,
            selected_layer_ids=list(protocol.LAYERS), include_embedding=True,
            selected_leaf_count=protocol.SELECTED_LEAVES, payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
            integrity_scope="selected_layers_and_embedding_only_not_complete_checkpoint",
            overlay_tensor_count=protocol.OVERLAY_TENSORS, overlay_bytes_per_chip=protocol.OVERLAY_BYTES,
            prompt_ids_sha256=protocol.PROMPT_SHA, main_rope_table=prior["main_rope_table"],
            original_oracle_pins={k: prior[k] for k in preflight.ORACLE_KEYS},
            versions=dict(jax="0.10.1", libtpu="0.0.41"), **bindings.context, **identity)
        fixed.update({k: prior[k] for k in ("hostname", "jax_process_index", "topology_fleet_sha256")})
        same_json({k: record.get(k) for k in fixed}, fixed, "history fleet source/runtime identity")
        same_json({k: prior[k] for k in bindings.context}, dict(bindings.context), "history original checkpoint context")
        _phase_identity(record)
        # _complete checks the pre-terminal worker state; actual terminal status
        # is checked above and no original record is modified for this reuse.
        entry._complete({**record, "status": "RUNNING"})
        process, host = record["jax_process_index"], record["hostname"]
        if (type(process) is not int or not 0 <= process < 8 or process in processes
                or not isinstance(host, str) or not host or host in hosts
                or any(type(record.get(k)) is not int or record[k] <= 0 for k in ("pid", "start_ticks"))
                or not isinstance(record.get("boot_id"), str) or not record["boot_id"]):
            raise ValueError("history fleet process identity incomplete or duplicated")
        processes.add(process)
        hosts.add(host)
        expected_owners = []
        for device, slot in slots.items():
            if device in ids.values() or slot in ids:
                raise ValueError("history fleet duplicate physical owner")
            ids[slot] = device
            source = bindings.owners[slot]
            original_owner = next(v for v in prior["local_device_slots"] if v["device_slot"] == slot)
            if original_owner["file_sha256"] != source["full_file_sha256"]:
                raise ValueError("history original owner file differs")
            expected_owners.append(dict(device_id=device, device_slot=slot,
                expected_full_file_sha256_not_verified=source["full_file_sha256"],
                observed_selected_tensor_sha256=source["selected"],
                selected_payload_bytes=protocol.PAYLOAD_BYTES))
        same_json(sorted(record["local_device_slots"], key=lambda v: v["device_slot"]),
                  sorted(expected_owners, key=lambda v: v["device_slot"]), "history selected loaded owners")
        _overlay_identity(record, prior, slots, bindings.overlay)
        entry._retained_identity(rankroot, record, repo)
        path = rankroot / "retained_preflight.json"
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != record["retained_preflight_sha256"]:
            raise ValueError("history preflight original SHA differs")
        retained = json.loads(raw)
        expected_preflight = dict(**identity, checkpoint_pins=checkpoint_pins,
            original_oracle_pins=fixed["original_oracle_pins"], main_rope_table=prior["main_rope_table"],
            strategy_nd_dense_overlay=prior["strategy_nd_dense_overlay"],
            local_device_slots=prior["local_device_slots"],
            overlay_manifest_sha256=bindings.overlay.manifest["manifest_sha256"],
            headers=[bindings.owners[s]["header"] for s in sorted(slots.values())])
        same_json({k: retained.get(k) for k in expected_preflight}, expected_preflight,
                  "history retained preflight metadata")
        signature = {n: {k: record["programs"][n][k] for k in
                        ("stablehlo_sha256", "optimized_hlo_sha256", "compiled_memory")} for n in protocol.PROGRAMS}
        if graph_signature is None:
            graph_signature = signature
        same_json(signature, graph_signature, "history fleet actual graph identity")
        execution_reports.append(execution.replay(rankroot, record, slots, repo=repo, graph_cache=graph_cache))
        materializer_reports.append(materializers.replay_rank(rankroot, record, slots, bindings))
        boundary_reports.append(boundaries.replay(rankroot, record, slots, originals=observations))
    if set(ids) != set(range(32)) or processes != set(range(8)):
        raise ValueError("history fleet owner/process coverage incomplete")
    mesh = [[ids[e * 4 + f] for f in range(4)] for e in range(8)]
    for record in records:
        same_json(record["physical_device_ids"], mesh, "history fleet physical mesh")
    materializer_replay = materializers.replay_fleet(materializer_reports, bindings=bindings, read_source=read_source)
    replica_replay = boundaries.validate_replicas(boundary_reports, records)
    firsts = [dict(rank=rank, **r["first_difference"]) for rank, r in enumerate(boundary_reports)
              if r["first_difference"] is not None]
    earliest = min(firsts, key=lambda v: (v["position"], v["layer"],
        boundaries.worker.FIELDS.index(v["field"]), v["rank"])) if firsts else None
    return dict(protocol=protocol.PROTOCOL, hosts=8, owners=32, reproduced=True,
                calls_per_host=331, materializers=materializer_replay,
                replicas=replica_replay, first_retained_difference_by_rank=firsts,
                earliest_recorded_difference=earliest,
                execution=execution_reports,
                scope="FULL_HISTORY_ORIGINAL_EVENT_REPRODUCTION_NOT_8K_TOKENS_OR_CAUSE",
                unretained_groups="WORKER_OBSERVATIONS_NOT_RETAINED_ARRAY_REPLAY",
                cache_values_replayed=False, cleanup_checked=False,
                numerical_promotion=False, performance_claim=False, cause_claim=False)
