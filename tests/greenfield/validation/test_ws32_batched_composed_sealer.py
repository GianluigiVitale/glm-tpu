"""Composed controller wiring, NOT numerical or TPU performance evidence.

Synthetic eight-rank records exercise the real schema, raw-file hashes, replay
cache, execution accounting, 32-owner memory join and summary construction.
Only external provenance/topology/oracle and expensive HLO/math/trace boundaries
are substituted. Their actual validators have separate admitted tests/evidence.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    short_acquisition,
    short_numerical_identity,
)
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from tests.greenfield.validation.test_ws32_prefill_admission import request_args
from tests.greenfield.validation.test_ws32_prefill_fleet_memory import (
    fixture as memory_fixture,
    execution_fixture,
)


ROOT = Path(__file__).resolve().parents[3]
ACQUIRED = Path(
    "/home/gianl/glm-run/greenfield_ws32_short_decoder_2k_acquire_c17_hrope_bp1_20260908T034734466845101Z"
)


def build(tmp_path, monkeypatch):
    # These two small immutable source records are inherited, never fabricated
    # or modified; this test intentionally fails if local evidence is missing.
    for name in ("exact_dsa_source_summary.json", "exact_dsa_source_SUCCESS"):
        shutil.copyfile(ACQUIRED / name, tmp_path / name)
    fleet = memory_fixture()
    args = request_args()
    args.__dict__.update(
        run_dir=tmp_path,
        output=tmp_path / "summary.json",
        topology_capture_root=tmp_path / "topology",
        mode="numerical",
        context_label="2k",
        tag="greenfield_ws32_short_decoder_2k_numerical_c17_hrope_bp1_20260908T000000000000000Z",
        code_hash="a" * 40,
        checkpoint_transport="shm",
        evidence_layout=sealer.EVIDENCE_LAYOUT_V2,
        observer_steps=1,
        warmup=2,
        iterations=10,
        trace_steps=2,
        token_oracle_dir=tmp_path / "oracle",
        dsa_oracle_dir=tmp_path / "dsa",
        recovery_code_hash="",
        reviewed_ref="refs/remotes/origin/rewrite/topology-first-decode",
        later_event_alarm_acknowledged=0,
        later_event_alarm_lessons_pin="",
        later_event_alarm_profile=None,
        later_event_alarm_profile_sha256="0" * 64,
    )
    for name in (
        "checkpoint_manifest",
        "checkpoint_success",
        "token_oracle_manifest",
        "token_oracle_success",
        "dsa_oracle_manifest",
        "dsa_oracle_success",
        "topology",
        "topology_fleet",
        "mesh",
        "source_inventory",
        "strategy_nd_dense_overlay_manifest",
        "strategy_nd_dense_overlay_manifest_file",
        "strategy_nd_dense_overlay_success_file",
    ):
        setattr(args, name + "_sha256", "b" * 64)
    args.dsa_association_summary_sha256 = sealer._DSA_ASSOCIATION_SUMMARY_SHA256
    args.dsa_association_success_sha256 = sealer._DSA_ASSOCIATION_SUCCESS_SHA256
    args.topology_capture_root.mkdir()
    captures = tuple(fleet["ordered_captures"])
    for rank, capture in enumerate(captures):
        (args.topology_capture_root / f"topology.rank{rank}.json").write_text(
            json.dumps(capture)
        )
    monkeypatch.setattr(
        sealer,
        "validate_ws32_topology_fleet",
        lambda *a, **k: (None, captures, args.topology_fleet_sha256),
    )
    mesh = SimpleNamespace(
        mesh_hash=args.mesh_sha256, flattened_device_ids=fleet["flattened_device_ids"]
    )
    monkeypatch.setattr(sealer, "build_ws32_physical_mesh", lambda _: mesh)
    oracle = SimpleNamespace(
        generated_token_ids=np.array([100], dtype=np.int32),
        producer_layer_ids=np.array([0], dtype=np.int32),
    )
    monkeypatch.setattr(
        sealer, "load_ws32_short_context_oracle", lambda *a, **k: oracle
    )
    for name in (
        "_require_clean_worktree",
        "_require_imports_come_from",
        "_require_reviewed_enforcement",
    ):
        monkeypatch.setattr(sealer, name, lambda *a, **k: None)
    monkeypatch.setattr(
        sealer, "_enforcement_surface_identity", lambda _: {"test": "synthetic surface"}
    )
    comparison = {"exact_prefix_match": True, "compared_count": 1}
    monkeypatch.setattr(sealer, "compare_ws32_raw_tokens", lambda *a, **k: comparison)
    monkeypatch.setattr(sealer, "compare_ws32_dsa_step", lambda **k: {"passed": True})
    monkeypatch.setattr(
        sealer, "validate_ws32_cache_probe", lambda **k: {"passed": True}
    )
    monkeypatch.setitem(
        sys.modules,
        "parse_xplane",
        SimpleNamespace(
            aggregate_fleet=lambda *a, **k: {
                "n_files": 8,
                "n_cores": 64,
                "steps_per_core": 2,
            }
        ),
    )

    graphs, calls = {}, []
    graph_names = tuple(short_acquisition(ROOT)["graphs"])
    for graph in graph_names:
        stable, optimized = f"synthetic stable {graph}", f"synthetic optimized {graph}"
        graphs[graph] = {
            "stablehlo_sha256": sha256(stable.encode()).hexdigest(),
            "optimized_hlo_sha256": sha256(optimized.encode()).hexdigest(),
            "source_location_identity": {"synthetic": graph},
        }
        for rank in range(8):
            folder = tmp_path / "fleet_hlo"
            folder.mkdir(exist_ok=True)
            (folder / f"{graph}.rank{rank}.stablehlo.mlir").write_text(stable)
            (folder / f"{graph}.rank{rank}.optimized_hlo.txt").write_text(optimized)

    def replay(stable, optimized, *, graph, args):
        assert (
            stable == f"synthetic stable {graph}"
            and optimized == f"synthetic optimized {graph}"
        )
        calls.append(graph)
        return graphs[graph]

    monkeypatch.setattr(sealer, "_replay_batched_graph", replay)
    original_memory = sealer._require_batched_fleet_memory
    memory_calls = []

    def check_memory(*a, **k):
        result = original_memory(*a, **k)
        memory_calls.append(result)
        return result

    monkeypatch.setattr(sealer, "_require_batched_fleet_memory", check_memory)
    base = json.loads((ACQUIRED / "runner.rank0.json").read_text())
    del base["failure"]
    base.update(short_numerical_identity())
    for key in list(base):
        if key.endswith("sha256") and hasattr(args, key):
            base[key] = getattr(args, key)
    base.update(
        artifact_kind="greenfield_ws32_short_decoder",
        code_hash=args.code_hash,
        compile_only=False,
        status="SUCCESS",
        correctness_passed=True,
        graphs=graphs,
        token_comparison=comparison,
        observed_generated_token_ids=[100] * 16,
        dsa_steps=[{"passed": True}],
        cache_write_probe={"passed": True},
        state={"position": [2049], "context_lengths": [2050], "contract_valid": [True]},
        profiler_free_timing={
            "profiler_active": False,
            "warmup": 2,
            "iterations": 10,
            "samples_ms": [100.0] * 10,
            "distribution": sealer._distribution([100.0] * 10),
        },
    )
    for rank, mr in enumerate(fleet["records"]):
        record = deepcopy(base)
        record.update(
            {
                k: v
                for k, v in mr.items()
                if k not in ("prefill_execution", "compiled_memory_analysis")
            }
        )
        for slot in record["local_device_slots"]:
            slot["file_sha256"] = "c" * 64
        record["checkpoint_verified_device_slots"] = [
            s["device_slot"] for s in record["local_device_slots"]
        ]
        record["strategy_nd_dense_overlay"] = {
            "manifest_sha256": args.strategy_nd_dense_overlay_manifest_sha256,
            "manifest_file_sha256": args.strategy_nd_dense_overlay_manifest_file_sha256,
            "success_file_sha256": args.strategy_nd_dense_overlay_success_file_sha256,
            "local_records": [
                {"device_id": device, "layer_id": layer}
                for device in captures[rank]["local_device_ids"]
                for layer in range(3)
            ],
        }
        record["prefill_execution"] = execution_fixture()["prefill_execution"]
        record["prefill_execution"]["memory_admission"] = mr["prefill_execution"][
            "memory_admission"
        ]
        record["device_memory_after_execute"] = deepcopy(
            base["device_memory_after_compile"]
        )
        arrays = {
            name: np.zeros((1,), dtype=np.uint16)
            for name in (
                "cache_contract_valid",
                "cache_index_bfloat16_bits",
                "cache_kv_bfloat16_bits",
                "cache_position",
                "dsa_producer_layer_ids",
                "dsa_selected_positions",
                "dsa_selected_scores",
                "dsa_selected_valid_counts",
            )
        }
        target = tmp_path / "fleet" / f"runner.rank{rank}.npz"
        target.parent.mkdir(exist_ok=True)
        np.savez(target, **arrays)
        record["numerical_tensors"] = {
            "filename": target.name,
            "byte_count": target.stat().st_size,
            "sha256": sealer._digest_file(target),
            "arrays": {
                name: {
                    "dtype": value.dtype.name,
                    "sha256": sha256(value.tobytes()).hexdigest(),
                    "shape": list(value.shape),
                }
                for name, value in arrays.items()
            },
        }
        trace = tmp_path / "traces" / f"trace.rank{rank}.xplane.pb"
        trace.parent.mkdir(exist_ok=True)
        trace.write_bytes(b"synthetic trace")
        record["trace"] = {
            "steps": 2,
            "files": [
                {
                    "byte_count": trace.stat().st_size,
                    "sha256": sealer._digest_file(trace),
                }
            ],
        }
        target.with_suffix(".json").write_text(json.dumps(record))
    return args, calls, memory_calls


@pytest.mark.parametrize(
    "mutation", [None, "identity", "frontier", "graph_bytes", "memory_owner"]
)
def test_actual_validate_batched_branch(tmp_path, monkeypatch, mutation):
    args, graphs, memories = build(tmp_path, monkeypatch)
    target = tmp_path / "fleet/runner.rank7.json"
    record = json.loads(target.read_text())
    if mutation == "identity":
        record["prefill_budget_seconds"] = 400.0
    elif mutation == "frontier":
        record["prefill_execution"]["final_frontier"] -= 1
    elif mutation == "graph_bytes":
        (tmp_path / "fleet_hlo/decode.rank7.stablehlo.mlir").write_text("changed")
    elif mutation == "memory_owner":
        record["batched_device_memory_after_execute"][0]["device_id"] = 999
    target.write_text(json.dumps(record))
    if mutation:
        reason = {
            "identity": "immutable identity drifted at rank 7",
            "frontier": "final frontier differs",
            "graph_bytes": "HLO artifact drifted at rank 7/decode",
            "memory_owner": "duplicate/foreign owner",
        }[mutation]
        with pytest.raises((SystemExit, ValueError), match=reason):
            sealer._validate(args)
        assert not args.output.exists()
    else:
        assert sealer._validate(args) == 0
        summary = json.loads(args.output.read_text())
        assert all(summary[k] == v for k, v in short_numerical_identity().items())
        assert len(graphs) == 7 and len(set(graphs)) == 7
        assert len(memories) == 1 and memories[0]["owner_count"] == 32
        assert summary["prefill_execution"]["fleet_total_seconds_max"] == 15.0
        assert "PREFILL_SPEEDUP_NOT_ESTABLISHED" in summary["classification"]
        assert "DELIVERED_TTFT_NOT_MEASURED" in summary["classification"]
        assert summary[
            "performance_claim"
        ]  # protected DECODE interface, not prefill speed
        assert set(summary["graph_source_location_identity"]) == set(graphs)
