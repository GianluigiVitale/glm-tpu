"""Full long host schedules, original allocations and fleet joins with fixture math.

No TPU, checkpoint values or runtime-memory claim. Original optimized text is
used only to authenticate the simulated compiler interface, never recompiled.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillResult, Ws32BatchedPrefillState
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState, ws32_decoder_weight_names
from glm_tpu.greenfield.validation import ws32_prefill_memory as memory
from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_batched_prefill_runner as adapter
from scripts.greenfield import ws32_delivery_runtime as runtime
from scripts.greenfield import ws32_owned_prefill_memory as owned
from scripts.greenfield.ws32_phase_weights import PhaseWeights
from tests.greenfield.runtime.test_ws32_phase_weights import config
from tests.greenfield.validation.test_ws32_owned_prefill_fleet_memory import fixture as owned_fixture
from tests.greenfield.validation.test_ws32_prefill_fleet_memory import fixture as plain_fixture

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def originals():
    base = Path("/home/gianl/glm-run")
    l7 = base / "greenfield_fp8_ws32_delivery_long_prefill_compile_20260911T232546214124179Z/rank0"
    e0 = base / "greenfield_fp8_ws32_capture_barrier_prefill_compile_20260912T025701169016006Z/fleet/rank0"
    return {
        ("128k_d1_0", role): (l7 / ("prefill_128k_" + role.removeprefix("prefill_") + ".optimized_hlo.txt")).read_text()
        for role in runtime.ROLES
    } | dict.fromkeys([("256k_e0", role) for role in runtime.ROLES],
        (e0 / "prefill_256k_capture_barrier.optimized_hlo.txt").read_text())


def fleet(label):
    fixture = owned_fixture() if label == "256k_e0" else plain_fixture()
    expected = runtime.registration(ROOT, label)
    analyses = {role: entry["memory"] for role, entry in expected.items()}
    fixture["expected_prefill_analyses"] = analyses
    for record in fixture["records"]:
        record["batched_prefill_profile"] = runtime.PROFILE
        record["context_capacity"] = runtime.programs.long_plan(label).context_capacity
        record["compiled_memory_analysis"] = deepcopy(analyses)
        if label != "256k_e0":
            admission = record["prefill_execution"]["memory_admission"]
            admission["compiled_memory"] = deepcopy(analyses)
            admission["budgets"] = {role: memory.budget_prefill_execution(
                admission["census"], analyses, active_graph=role,
                resident_graphs=runtime.ROLES, required_reserve_bytes=runtime.RESERVE,
            ) for role in runtime.ROLES}
    return fixture


class Compiler:
    def __init__(self, label, role, text, analysis, calls):
        self.label, self.role, self.text, self.analysis, self.calls = label, role, text, analysis, calls

    def as_text(self):
        return self.text

    def memory_analysis(self):
        return SimpleNamespace(**self.analysis)

    def __call__(self, tokens, count, state, weights, wk, rope):
        plan = runtime.programs.long_plan(self.label)
        begin = int(state.decoder.position[0])
        end = begin + int(count)
        expected_rows = 128 if self.label == "256k_e0" else dict(plan.graph_rows)[self.role]
        assert tokens.shape == (expected_rows,) and count.shape == ()
        np.testing.assert_array_equal(tokens[:int(count)], np.arange(begin, end, dtype=np.int32))
        assert not tokens[int(count):].any()
        self.calls.append((begin, end, expected_rows))
        final = end == plan.prompt_length
        decoder = state.decoder._replace(position=np.array([end], np.int32),
            context_lengths=np.array([end + 1], np.int32),
            index_cache_local=state.repaired_index_local if final else state.decoder.index_cache_local)
        return Ws32BatchedPrefillResult(state._replace(decoder=decoder, finished=np.bool_(final)),
                                       np.array([123 if final else -1], np.int32))


def setup(tmp_path, monkeypatch, originals, label):
    plan = runtime.programs.long_plan(label)
    cfg = replace(config(), context_capacity=plan.context_capacity)
    raw = replace(cfg, exact_dsa=False, strategy_nd_dense=False)
    owner = PhaseWeights({name: object() for name in jax.tree.leaves(ws32_decoder_weight_names(raw))}, cfg)
    owner.phase, owner.wk = "prefill", tuple(object() for _ in cfg.full_index_slots)
    expected = runtime.registration(ROOT, label)
    calls, votes = [], []
    compiled = {role: Compiler(label, role, originals[label, role], entry["memory"], calls)
                for role, entry in expected.items()}
    if label == "256k_e0":
        compiled["prefill_tail"] = compiled["prefill_chunk"]
    reports = {role: dict(schema="ws32_delivery_long_structural_v1", passed=True,
        context_capacity=plan.context_capacity, block_rows=dict(plan.graph_rows)[role],
        state_ownership_contract=runtime.programs.state_ownership(label), dispatch_authorized=False,
        **{key: entry[key] for key in ("stablehlo_sha256", "optimized_hlo_sha256")})
        for role, entry in expected.items()}
    args = SimpleNamespace(batched_prefill_profile=runtime.PROFILE,
        prefill_memory_reserve_bytes=runtime.RESERVE, prefill_budget_seconds=runtime.budget_seconds(label),
        expected_code_hash="a" * 40, process_id=0, output=tmp_path / "runner.rank0.json",
        **{f"expected_{role}_{form}": entry[form] for role, entry in expected.items()
           for form in ("stablehlo_sha256", "optimized_hlo_sha256")})
    def fresh(mesh, cfg, *, prompt_length):
        decoder = Ws32DecoderState(np.zeros(1), np.zeros(1), np.zeros((1, 1), np.int32),
            np.zeros(1, np.int32), np.zeros((1, 1)), np.array([0], np.int32),
            np.zeros((1, 1), np.int32), np.array([1], np.int32), np.array([True]))
        return Ws32BatchedPrefillState(decoder, np.ones(1), np.int32(prompt_length), np.bool_(False))
    fixture = fleet(label)
    admission = fixture["records"][0]["prefill_execution"]["memory_admission"]
    monkeypatch.setattr(adapter, "make_ws32_batched_prefill_state", fresh)
    monkeypatch.setattr(adapter, "replicated", lambda mesh, v: np.asarray(v).copy())
    monkeypatch.setattr(adapter, "make_prefill_memory_record", lambda *a, **k: deepcopy(admission))
    monkeypatch.setattr(owned, "make_record", lambda *a, **k: deepcopy(admission))
    monkeypatch.setattr(jax, "process_index", lambda: 3)
    stats = dict(bytes_in_use=25_000_000_000, peak_bytes_in_use=26_000_001_000, bytes_limit=runtime.DEVICE_LIMIT)
    devices = [SimpleNamespace(id=i, process_index=3, platform="tpu", memory_stats=lambda: stats) for i in range(12, 16)]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    monkeypatch.setattr(worker, "_batched_fleet_all", lambda value: votes.append(value) or value)
    # Real host loop, JSON publication and validators; fixture model/counters.
    kwargs = dict(args=args, mesh=None, config=owner.raw_config, plan=plan,
        prompt_tokens=np.arange(plan.prompt_length, dtype=np.int32), compiled=compiled,
        weights=owner.raw_weights, wk=owner.wk, rope=None, long_context_label=label,
        phase_owner=owner, graph_reports=reports)
    return kwargs, calls, votes, fixture


@pytest.mark.parametrize("label", ["128k_d1_0", "256k_e0"])
def test_full_host_schedule_to_execution_and_all32_memory_sealers(tmp_path, monkeypatch, originals, label):
    kwargs, calls, votes, fixture = setup(tmp_path, monkeypatch, originals, label)
    decoder, token, execution, after = worker._execute_batched_prefill(**kwargs)
    plan = kwargs["plan"]
    assert len(calls) == (996 if label != "256k_e0" else 2048)
    expected_tail = (127360, 127363, 114) if label != "256k_e0" else (262016, 262144, 128)
    assert calls[-1] == expected_tail
    assert decoder.position.tolist() == [plan.prompt_length] and token.tolist() == [123]
    assert len(votes) == 2 * len(calls) + 3 and all(votes)
    saved = json.loads((tmp_path / "batched_prefill_complete.rank0.json").read_text())
    assert saved["execution"] == execution and saved["device_memory_after_prefill"] == after
    record = dict(batched_prefill_profile=runtime.PROFILE, prefill_mode=adapter.PREFILL_MODE,
        batched_prefill_plan=runtime.identity(label), context_capacity=plan.context_capacity,
        prefill_chunk_length=128, prefill_execution=execution, observed_generated_token_ids=[123])
    sealer._require_batched_execution(record, mode="numerical", prompt_length=plan.prompt_length,
                                    expected_chunk=128, long_context_label=label)
    with pytest.raises(ValueError):
        sealer._require_batched_execution(record, mode="numerical", prompt_length=plan.prompt_length, expected_chunk=128)
    result = sealer._require_batched_fleet_memory(fixture["records"], fixture["ordered_captures"],
        SimpleNamespace(flattened_device_ids=fixture["flattened_device_ids"]), long_context_label=label)
    assert result["owner_count"] == 32
    if label == "256k_e0":
        separate = owned_fixture(shared=False)
        for item in separate["records"]:
            item["batched_prefill_profile"] = runtime.PROFILE
            item["context_capacity"] = plan.context_capacity
        with pytest.raises(ValueError, match="shared compiled"):
            sealer._require_batched_fleet_memory(separate["records"], separate["ordered_captures"],
                SimpleNamespace(flattened_device_ids=separate["flattened_device_ids"]), long_context_label=label)
    for changed in (None, "8k", "256k_e0" if label != "256k_e0" else "128k_d1_0"):
        with pytest.raises(ValueError):
            sealer._require_batched_fleet_memory(fixture["records"], fixture["ordered_captures"],
                SimpleNamespace(flattened_device_ids=fixture["flattened_device_ids"]), long_context_label=changed)


@pytest.mark.parametrize("mutation", ["owner", "wk", "decode_root", "phase", "reserve", "budget",
    "pin", "graph_text", "graph_report", "memory", "duplicate_code", "peer", "publication"])
def test_long_preflight_refuses_before_first_dispatch(tmp_path, monkeypatch, originals, mutation):
    kwargs, calls, votes, _ = setup(tmp_path, monkeypatch, originals, "256k_e0")
    if mutation == "owner": kwargs["phase_owner"] = None
    elif mutation == "wk": kwargs["wk"] = ()
    elif mutation == "decode_root": kwargs["phase_owner"].decode_weights = object()
    elif mutation == "phase": kwargs["phase_owner"].phase = "raw"
    elif mutation == "reserve": kwargs["args"].prefill_memory_reserve_bytes -= 1
    elif mutation == "budget": kwargs["args"].prefill_budget_seconds += 1
    elif mutation == "pin": kwargs["args"].expected_prefill_chunk_optimized_hlo_sha256 = "0" * 64
    elif mutation == "graph_text": kwargs["compiled"]["prefill_chunk"].text += "changed"
    elif mutation == "graph_report": kwargs["graph_reports"]["prefill_chunk"]["passed"] = False
    elif mutation == "memory": kwargs["compiled"]["prefill_chunk"].analysis["temp_size_in_bytes"] += 1
    elif mutation == "duplicate_code": kwargs["compiled"]["prefill_tail"] = deepcopy(kwargs["compiled"]["prefill_chunk"])
    elif mutation == "peer": monkeypatch.setattr(worker, "_batched_fleet_all", lambda value: False)
    else:
        original = worker._atomic_json
        def fail(path, payload):
            if "preflight" in path.name: raise OSError("fixture publication failure")
            return original(path, payload)
        monkeypatch.setattr(worker, "_atomic_json", fail)
    with pytest.raises(RuntimeError, match="numerical preflight"):
        worker._execute_batched_prefill(**kwargs)
    assert not calls
    assert (tmp_path / "batched_prefill_failure.rank0.json").exists()


def test_fixed_registration_retains_both_original_geometries():
    for label in ("128k_d0_0", "128k_d0_05", "128k_d0_95", "128k_d1_0", "256k_e0"):
        expected = runtime.registration(ROOT, label)
        assert set(expected) == set(runtime.ROLES)
        for role, value in expected.items():
            runtime.require_memory(value["memory"], value["memory"])
            assert value["stablehlo_sha256"] == runtime.programs.raw_registration(label)[role][1]
    for bad in (None, "8k", True):
        with pytest.raises(ValueError): runtime.budget_seconds(bad)


@pytest.mark.parametrize("mutation", ["token", "role_record", "limit", "peak", "peer", "publication"])
def test_completed_long_execution_cannot_hide_boundary_failure(tmp_path, monkeypatch, originals, mutation):
    kwargs, calls, votes, _ = setup(tmp_path, monkeypatch, originals, "256k_e0")
    execute = adapter.execute_graph_pair
    def changed(*a, **k):
        decoder, token, record = execute(*a, **k)
        if mutation == "token": token = np.array([124], np.int32)
        if mutation == "role_record":
            # Individually valid distinct-code memory is not the E0 profile.
            record["memory_admission"] = owned_fixture(shared=False)["records"][0]["prefill_execution"]["memory_admission"]
        if mutation in ("limit", "peak"):
            devices = jax.local_devices()
            stats = dict(devices[0].memory_stats())
            stats["bytes_limit" if mutation == "limit" else "peak_bytes_in_use"] = (
                runtime.DEVICE_LIMIT + 1 if mutation == "limit" else runtime.DEVICE_LIMIT)
            devices[0].memory_stats = lambda: stats
        if mutation == "peer":
            monkeypatch.setattr(worker, "_batched_fleet_all", lambda _: False)
        return decoder, token, record
    monkeypatch.setattr(adapter, "execute_graph_pair", changed)
    if mutation == "publication":
        original = worker._atomic_json
        def fail(path, payload):
            if "complete" in path.name: raise OSError("fixture final publication")
            return original(path, payload)
        monkeypatch.setattr(worker, "_atomic_json", fail)
    with pytest.raises(RuntimeError, match="completed prefill"):
        worker._execute_batched_prefill(**kwargs)
    assert len(calls) == 2048
    assert (tmp_path / "batched_prefill_memory.rank0.json").exists()
    assert (tmp_path / "batched_prefill_failure.rank0.json").exists()


def test_long_script_authorities_are_on_sealer_enforcement_surface():
    for name in ("ws32_delivery_runtime", "ws32_delivery_hlo", "ws32_delivery_programs",
                 "ws32_phase_weights", "ws32_owned_prefill_memory", "ws32_batched_prefill_runner",
                 "ws32_prefill_owned_state", "ws32_capture_barrier_compile", "ws32_flat_rows_compile",
                 "ws32_pending_rows_compile", "ws32_owned_state_compile", "ws32_delivery_compile",
                 "ws32_rolled_prefill_compile", "ws32_canonical_prefill_compile"):
        assert "scripts/greenfield/" + name + ".py" in sealer._ENFORCEMENT_SURFACE
