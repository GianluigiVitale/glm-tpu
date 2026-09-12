"""Original phase replay and fleet identity controls, with explicit CPU fixtures."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.greenfield import ws32_native_benchmark_evidence as evidence
from scripts.greenfield import ws32_native_benchmark_memory as memory
from tests.greenfield.validation.test_ws32_delivery_phase_evidence import completed, originals
from tests.greenfield.validation.test_ws32_native_benchmark_memory import record as memory_record

ROOT = Path(__file__).resolve().parents[3]


def test_original44_calls_replayed_in_native_residency_order(completed):
    root, old = completed["root"], completed["parent"]
    (root / "delivery_wk.rank0").rename(root / "wk")
    (root / "delivery_decode.rank0").rename(root / "exact")
    slots = {r["device_id"]: r["device_slot"] for r in old["local_device_slots"]}
    identity = dict(profile=evidence.transport.PROFILE, context_capacity=262656,
        code_hash="a"*40, launch_process_id=0, jax_process_index=old["jax_process_index"],
        mesh_sha256="mesh", topology_sha256="topology", topology_fleet_sha256="fleet",
        local_slots=[dict(device_id=d, slot=s) for d, s in sorted(slots.items())])
    parent = dict(identity, overlay=old["strategy_nd_dense_overlay"])
    for folder, field, oldfield in (("wk", "wk_preparation", "delivery_wk_phase"),
                                   ("exact", "exact_preparation", "delivery_decode_preparation")):
        row = deepcopy(old[oldfield])
        row.update(identity, phase_contract=evidence.transport.PROFILE)
        (root / folder / "runner.json").write_text(json.dumps(row))
        parent[field] = row
    initial = memory_record()
    initial.update(phase="before_cache", cache_present=False, process_index=identity["jax_process_index"],
                   local_slots=identity["local_slots"])
    for row, d in zip(initial["census"]["devices"], sorted(slots), strict=True):
        row.update(device_id=d, process_index=identity["jax_process_index"])
        row["memory_stats"]["peak_bytes_in_use"] = 27_000_000_000
    initial["budgets"] = memory.budgets(initial)
    parent["programs"] = {n:dict(compiled_memory=m) for n,m in initial["compiled_memory"].items()}
    path = root / "initial_memory.json"
    path.write_text(json.dumps(initial))
    options = dict(root=root, parent=parent, slots=slots, full_index_layers=completed["full_index_layers"])
    result = evidence.replay_preparation(**options)
    assert result["calls"] == 44 and len(result["wk_output_hashes"]) == 42
    original = path.read_bytes()
    initial["budgets"]["cache_init"]["devices"][0]["estimated_peak_bytes"] = 0
    path.write_text(json.dumps(initial))
    with pytest.raises(ValueError): evidence.replay_preparation(**options)
    path.write_bytes(original)
    wr = parent["wk_preparation"]
    wr["phase_contract"] = "old"
    (root / "wk/runner.json").write_text(json.dumps(wr))
    with pytest.raises(ValueError, match="scope"): evidence.replay_preparation(**options)


def fleet(tmp_path, monkeypatch):
    inspected = []
    def inspect(stable, optimized, *, graph, expected_optimized, **kwargs):
        inspected.append(graph)
        assert sha256(optimized.encode()).hexdigest() == expected_optimized
        return dict(graph=graph, passed=True, optimized_identity_policy="LITERAL_OPTIMIZED_SHA256")
    monkeypatch.setattr(evidence.programs, "inspect_hlo", inspect)
    monkeypatch.setattr(evidence.wk_admission, "inspect_program", lambda graph, *a: dict(graph=graph, passed=True))
    monkeypatch.setattr(evidence, "replay_preparation", lambda *a: dict(wk_output_hashes=["same"]*42))
    pins = json.loads((ROOT / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    captures = []
    for rank in range(8):
        ids = list(range(rank*4, rank*4+4))
        captures.append(dict(jax_process_index=rank, local_device_ids=ids))
        row = dict(profile=evidence.transport.PROFILE, artifact_kind="ws32_native_cold_load_v1",
            complete=True, compile_only=False, performance_claim=False, code_hash="a"*40,
            launch_process_id=rank, jax_process_index=rank, context_capacity=262656,
            mesh_sha256="mesh", topology_sha256="topology", topology_fleet_sha256="fleet",
            local_slots=[dict(device_id=d, slot=d) for d in ids],
            checkpoint=dict(manifest_sha256=pins["GLM_GREENFIELD_WS32_CHECKPOINT_MANIFEST_SHA"],
                success_sha256=pins["GLM_GREENFIELD_WS32_CHECKPOINT_SUCCESS_SHA"], verified_slots=ids,
                source_inventory_sha256=json.loads((ROOT / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())["source_inventory_sha256"]),
            overlay={key:pins["GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_"+env] for key,env in
                     (("manifest_sha256", "MANIFEST_SHA"),("manifest_file_sha256", "MANIFEST_FILE_SHA"),
                      ("success_file_sha256", "SUCCESS_FILE_SHA"))},
            programs={}, wk_preparation=dict(programs={}), exact_preparation=dict(programs={}))
        root = tmp_path / f"native.rank{rank}"
        root.mkdir()
        for graph, folder in evidence.transport.GRAPHS.items():
            meta = dict(compiled_memory={"fixture": 1}, compile_seconds=1.0,
                        admission=dict(graph=graph, passed=True))
            if folder != "wk": meta["admission"]["optimized_identity_policy"] = evidence.FRESH_POLICY
            for form, field in (("stablehlo.mlir", "stablehlo_sha256"),("optimized_hlo.txt", "optimized_hlo_sha256")):
                text = graph + form
                path = root / folder / (graph + "." + form)
                path.parent.mkdir(exist_ok=True)
                path.write_text(text)
                meta[field] = sha256(text.encode()).hexdigest()
            target = row["programs"] if not folder else row["wk_preparation" if folder == "wk" else "exact_preparation"]["programs"]
            target[graph] = meta
        (root / "runner.json").write_text(json.dumps(row))
    return dict(root=tmp_path, repo=ROOT, pin="a"*40, captures=captures,
        physical_mesh=NS(flattened_device_ids=tuple(range(32)), mesh_hash="mesh"),
        topology_hash="topology", fleet_hash="fleet", full_index_layers=()), inspected


def test_all32_owners_same_graphs_and_explicit_limits(tmp_path, monkeypatch):
    options, inspected = fleet(tmp_path, monkeypatch)
    result = evidence.replay_cold_fleet(**options)
    assert result["owners"] == 32 and len(inspected) == 8  # two WK inspectors separate
    assert not result["benchmark_quality_proven"] and not result["request_execution_proven"]
    path = tmp_path / "native.rank7/runner.json"
    original = path.read_bytes()
    for mutation in ("policy", "compile_time", "checkpoint", "owner", "raw", "memory"):
        row = json.loads(original)
        if mutation == "policy": row["programs"]["decode"]["admission"]["optimized_identity_policy"] = "trust label"
        elif mutation == "compile_time": row["programs"]["decode"]["compile_seconds"] = 0
        elif mutation == "checkpoint": row["checkpoint"]["verified_slots"] = [0,1,2,3]
        elif mutation == "owner": row["local_slots"][0]["device_id"] = 0
        elif mutation == "raw": row["programs"]["decode"]["stablehlo_sha256"] = "0"*64
        else: row["programs"]["decode"]["compiled_memory"] = {"fixture": 2}
        path.write_text(json.dumps(row))
        with pytest.raises(ValueError): evidence.replay_cold_fleet(**options)
    path.write_bytes(original)
