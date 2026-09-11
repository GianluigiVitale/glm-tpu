"""Fleet identity composition with explicit source/math/evidence stand-ins.

Each underlying reader has its own original-byte tests. This suite isolates the
joins and refusal ordering; it does not produce fake hardware admission.
"""

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import ws32_history_evidence as reader
from scripts.greenfield.ws32_history_execution import call_schedule
from scripts.greenfield.ws32_history_call_evidence import SCHEMA

TAG = "greenfield_fp8_ws32_history_frontier_l06_20260911T200000000000000Z"
PIN = "a" * 40
SHA = "b" * 64


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root, original_root = tmp_path / "fleet", tmp_path / "references"
    root.mkdir(); original_root.mkdir()
    checkpoint_pins = dict(fixture=SHA, checkpoint_root="/fixture/checkpoint",
                           expected_manifest_sha256=SHA, classification="FIXED_HOST_ADMISSION")
    receipt = tmp_path / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(checkpoint_pins))
    owners = {s: dict(full_file_sha256=SHA, selected={"fixture": SHA},
                     header=dict(device_slot=s, filename=f"slot{s}", file_bytes=1, header_sha256=SHA))
              for s in range(32)}
    context = dict.fromkeys(("checkpoint_manifest_sha256", "checkpoint_success_sha256",
                            "source_inventory_sha256", "mesh_sha256", "topology_sha256"), SHA)
    overlay = SimpleNamespace(manifest={"manifest_sha256": SHA}, manifest_file_sha256=SHA,
        success_file_sha256=SHA, records={(l, e, f): dict(sha256=SHA,
            tensors={str(i): dict(byte_count=reader.protocol.OVERLAY_BYTES // 12) for i in range(4)})
            for l in range(3) for e in range(8) for f in range(4)})
    bindings = SimpleNamespace(owners=owners, context=context, overlay=overlay)
    records, priors, identities, calls = [], [], [], []
    for rank in range(8):
        rankroot = root / f"rank{rank}"; rankroot.mkdir()
        slots = list(range(rank * 4, rank * 4 + 4))
        prior = dict(**context, launch_process_id=rank, hostname=f"host{rank}", jax_process_index=rank,
                     topology_fleet_sha256=SHA, main_rope_table={"fixture": SHA},
                     local_device_slots=[dict(device_id=s, device_slot=s, file_sha256=SHA,
                        expert_coordinate=s // 4, feature_coordinate=s % 4) for s in slots],
                     **dict.fromkeys(reader.preflight.ORACLE_KEYS, SHA))
        prior["strategy_nd_dense_overlay"] = dict(manifest_sha256=SHA, manifest_file_sha256=SHA,
            success_file_sha256=SHA, local_records=[dict(device_id=s, expert_coordinate=s // 4,
                feature_coordinate=s % 4, layer_id=l, file_sha256=SHA) for s in slots for l in range(3)])
        identity = dict(original_tags=reader.protocol.ORIGINAL_TAGS,
            original_pins={"fixture_rank": rank}, receipt_sha256={"fixture": SHA}, bytes=17,
            reproduction_row_sha256={"fixture": SHA})
        entries = [dict(phase=phase, graph=graph, completed=True,
                       original=dict(schema=SCHEMA, path=f"call_records/call{i:03d}.json",
                                     bytes=10, sha256=SHA, index=i))
                   for i, (phase, graph) in enumerate(call_schedule())]
        record = dict(**context, **identity, status=reader.entry.STATUS, tag=TAG, code_hash=PIN,
            launch_rank=rank, hostname=prior["hostname"], jax_process_index=rank,
            topology_fleet_sha256=SHA, pid=rank + 1, start_ticks=10, boot_id=f"boot{rank}",
            kernel=reader.protocol.KERNEL, protocol=reader.protocol.PROTOCOL, profile=reader.entry.admission.PROFILE,
            diagnostic_only=True, admission_only=False, compile_only=False,
            numerical_promotion=False, performance_claim=False, iterations=0, latency=None,
            selected_layer_ids=list(reader.protocol.LAYERS), include_embedding=True,
            selected_leaf_count=201, payload_bytes_per_chip=reader.protocol.PAYLOAD_BYTES,
            integrity_scope="selected_layers_and_embedding_only_not_complete_checkpoint",
            overlay_tensor_count=12, overlay_bytes_per_chip=reader.protocol.OVERLAY_BYTES,
            prompt_ids_sha256=reader.protocol.PROMPT_SHA, main_rope_table=prior["main_rope_table"],
            original_oracle_pins={k: prior[k] for k in reader.preflight.ORACLE_KEYS},
            versions=dict(jax="0.10.1", libtpu="0.0.41"), strategy_nd_dense_overlay=prior["strategy_nd_dense_overlay"],
            local_device_slots=[dict(device_id=s, device_slot=s, expected_full_file_sha256_not_verified=SHA,
                observed_selected_tensor_sha256=owners[s]["selected"], selected_payload_bytes=reader.protocol.PAYLOAD_BYTES)
                for s in slots], physical_device_ids=np.arange(32).reshape(8, 4).tolist(),
            programs={n: dict(stablehlo_sha256=SHA, optimized_hlo_sha256=SHA, compiled_memory={},
                             admission=dict(passed=True)) for n in reader.protocol.PROGRAMS},
            acquisition_phases={n: dict(status="COMPLETE", seconds=0., error=None) for n in reader.PHASES},
            current_phase="history/execution_complete", history_execution_complete=True, planned_call_count=331,
            call_evidence_layout=SCHEMA, call_evidence=entries, call_original_bytes=3310,
            compile_journal_sha256=SHA, checkpoint_pins=deepcopy(checkpoint_pins),
            history=dict(protocol=reader.protocol.PROTOCOL, complete=True, attribution_eligible=True,
                prompt_length=8155, steps=319, numerical_promotion=False, performance_claim=False, cause_claim=False,
                reproduction={b: dict(reproduced=True) for b in reader.protocol.BRANCHES}))
        retained = dict(**identity, checkpoint_pins=record["checkpoint_pins"],
            original_oracle_pins=record["original_oracle_pins"], main_rope_table=prior["main_rope_table"],
            strategy_nd_dense_overlay=prior["strategy_nd_dense_overlay"], local_device_slots=prior["local_device_slots"],
            overlay_manifest_sha256=SHA, headers=[owners[s]["header"] for s in slots])
        raw = json.dumps(retained).encode()
        (rankroot / "retained_preflight.json").write_bytes(raw)
        record["retained_preflight_sha256"] = sha256(raw).hexdigest()
        records.append(record); priors.append(prior); identities.append(identity)
    monkeypatch.setattr(reader.materializers, "checkpoint_bindings", lambda repo: bindings)
    monkeypatch.setattr(reader.preflight, "load_originals", lambda path, *, repo, rank:
        ({b: priors[rank] for b in reader.protocol.BRANCHES}, {"fixture": rank}, identities[rank]))
    monkeypatch.setattr(reader.entry, "_retained_identity", lambda *args: calls.append("entry_identity"))
    def execution(root, record, slots, **kwargs):
        calls.append(("execution", record["launch_rank"]))
        return {"fixture": True}
    def rank_materializers(root, record, slots, actual_bindings):
        assert actual_bindings is bindings
        calls.append(("materializer_rank", record["launch_rank"]))
        return record["launch_rank"]
    def boundary(root, record, slots, *, originals):
        rank = record["launch_rank"]
        assert originals == {"fixture": rank}
        calls.append(("boundary", rank))
        return dict(first_difference=dict(group=1, layer=6 - rank % 4, field="update", position=200 - rank))
    def materializer_fleet(reports, *, bindings, read_source):
        assert reports == list(range(8))
        calls.append("materializer_fleet")
        return dict(fixture=True, source=read_source(0, "fixture"))
    def replica(reports, records):
        assert len(reports) == len(records) == 8
        calls.append("boundary_replicas")
        return dict(fixture=True)
    monkeypatch.setattr(reader.execution, "replay", execution)
    monkeypatch.setattr(reader.materializers, "replay_rank", rank_materializers)
    monkeypatch.setattr(reader.materializers, "replay_fleet", materializer_fleet)
    monkeypatch.setattr(reader.boundaries, "replay", boundary)
    monkeypatch.setattr(reader.boundaries, "validate_replicas", replica)
    return dict(root=root, records=records, repo=tmp_path, original_root=original_root,
                read_source=lambda *args: "fixture-source"), bindings, calls


def test_all32_owner_composition_preserves_scope_and_earliest_is_not_rank0(bundle):
    kwargs, _, calls = bundle
    before = deepcopy(kwargs["records"])
    result = reader.validate_fleet(**kwargs)
    assert result["reproduced"] and result["owners"] == 32
    assert result["earliest_recorded_difference"]["rank"] == 7
    assert result["calls_per_host"] == 331
    assert result["numerical_promotion"] is False and result["cause_claim"] is False
    assert result["cleanup_checked"] is False and result["cache_values_replayed"] is False
    assert calls[-2:] == ["materializer_fleet", "boundary_replicas"]
    assert kwargs["records"] == before


@pytest.mark.parametrize("mutation", ["digest", "root", "classification", "missing", "extra"])
def test_dual_copy_checkpoint_pins_cannot_replace_fixed_host_admission(bundle, mutation):
    kwargs, _, calls = bundle
    record = kwargs["records"][3]
    altered = deepcopy(record["checkpoint_pins"])
    if mutation == "digest":
        altered["expected_manifest_sha256"] = "0" * 64
    elif mutation == "root":
        altered["checkpoint_root"] = "/different/checkpoint"
    elif mutation == "classification":
        altered["classification"] = "DIFFERENT_HOST_ADMISSION"
    elif mutation == "missing":
        altered.pop("fixture")
    else:
        altered["unregistered"] = True
    record["checkpoint_pins"] = altered
    path = kwargs["root"] / "rank3/retained_preflight.json"
    retained = json.loads(path.read_text())
    retained["checkpoint_pins"] = deepcopy(altered)
    raw = json.dumps(retained).encode()
    path.write_bytes(raw)
    record["retained_preflight_sha256"] = sha256(raw).hexdigest()
    # Both mutable copies and the enclosing file digest agree. Only the fixed
    # repository host-admission original can independently reject the change.
    assert record["checkpoint_pins"] == retained["checkpoint_pins"]
    with pytest.raises(ValueError, match="fixed host-admission checkpoint pins"):
        reader.validate_fleet(**kwargs)
    assert calls == []


@pytest.mark.parametrize("defect", ["rank", "bool_rank", "tag", "pin", "source_reader", "source_owners",
    "current_hash", "original_hash", "selected_owner", "selected_source", "duplicate_process",
    "pid", "boot", "version", "mesh", "overlay_hash", "overlay_owner", "overlay_bytes",
    "header", "preflight_sha", "graph", "phase_missing", "phase_failed", "phase_seconds",
    "terminal_call", "promotion"])
def test_fleet_identity_mismatch_refuses_before_source_payload(bundle, defect):
    kwargs, bindings, calls = bundle
    r = kwargs["records"][0]
    if defect == "rank": r["launch_rank"] = 1
    elif defect == "bool_rank": r["launch_rank"] = False
    elif defect == "tag": r["tag"] = "other"
    elif defect == "pin": r["code_hash"] = "bad"
    elif defect == "source_reader": kwargs["read_source"] = None
    elif defect == "source_owners": del bindings.owners[0]
    elif defect == "current_hash": r["checkpoint_manifest_sha256"] = "0" * 64
    elif defect == "original_hash": bindings.owners[0]["full_file_sha256"] = "0" * 64
    elif defect == "selected_owner": r["local_device_slots"][0]["device_id"] = 7
    elif defect == "selected_source": r["local_device_slots"][0]["observed_selected_tensor_sha256"] = {}
    elif defect == "duplicate_process": kwargs["records"][1]["jax_process_index"] = 0
    elif defect == "pid": r["pid"] = 0
    elif defect == "boot": r["boot_id"] = None
    elif defect == "version": r["versions"]["jax"] = "other"
    elif defect == "mesh": r["physical_device_ids"][0][0] = 7
    elif defect == "overlay_hash": bindings.overlay.manifest["manifest_sha256"] = "0" * 64
    elif defect == "overlay_owner": bindings.overlay.records[0, 0, 0]["sha256"] = "0" * 64
    elif defect == "overlay_bytes": bindings.overlay.records[0, 0, 0]["tensors"]["0"]["byte_count"] += 1
    elif defect == "header": bindings.owners[0]["header"]["header_sha256"] = "0" * 64
    elif defect == "preflight_sha": r["retained_preflight_sha256"] = "0" * 64
    elif defect == "graph": kwargs["records"][1]["programs"]["observer"]["optimized_hlo_sha256"] = "0" * 64
    elif defect == "phase_missing": del r["acquisition_phases"]["history/setup"]
    elif defect == "phase_failed": r["acquisition_phases"]["history/setup"]["status"] = "FAILED"
    elif defect == "phase_seconds": r["acquisition_phases"]["history/setup"]["seconds"] = float("nan")
    elif defect == "terminal_call": r["call_evidence"][1]["graph"] = "observer"
    else: r["performance_claim"] = True
    with pytest.raises((ValueError, KeyError)):
        reader.validate_fleet(**kwargs)
    assert "materializer_fleet" not in calls


@pytest.mark.parametrize("component", ["execution", "materializers", "boundaries"])
def test_independent_reader_failure_is_not_suppressed(bundle, monkeypatch, component):
    kwargs, _, calls = bundle
    module = getattr(reader, component)
    def refuse(*args, **kwargs): raise ValueError("fixture original refused")
    monkeypatch.setattr(module, "replay_rank" if component == "materializers" else "replay", refuse)
    with pytest.raises(ValueError, match="fixture original refused"):
        reader.validate_fleet(**kwargs)
    assert "materializer_fleet" not in calls
