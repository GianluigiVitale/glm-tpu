"""CPU worker lifecycle and original NPZ replay; executable math is a fixture."""

from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_completed_window_admission as admission
from scripts.greenfield import prefill_completed_window_protocol as protocol
from scripts.greenfield import prefill_completed_window_worker as worker
from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield import prefill_window_worker as base
from scripts.greenfield import probe_ws32_prefill_layer as layer
from tests.greenfield.hlo.test_prefill_window_worker import fake_memory, fixture_output


@contextmanager
def run_cases(root, failure=None):
    patches = pytest.MonkeyPatch()
    record = dict(
        protocol=protocol.PROTOCOL,
        profile=admission.PROFILE,
        compile_only=False,
        code_hash="a" * 40,
        launch_rank=0,
        jax_process_index=3,
        cases={},
    )
    journal = worker.CompletedJournal(
        root / "compile_journal.jsonl",
        {
            k: record[k]
            for k in ("protocol", "profile", "compile_only", "code_hash", "launch_rank")
        },
    )
    calls = base.BudgetedCalls(
        root=root,
        record=record,
        consensus=lambda ok: ok,
        journal=journal,
        local_slots={9: 0, 13: 1, 25: 2, 29: 3},
        budgeter=admission.memory_budget,
    )
    state, observations, prefixes, suffix_results, sequence = {}, {}, [], [], []
    pins = admission.registered_programs()

    def inputs(host, *args):
        state.clear()
        state.update(host)
        prefixes.clear()
        suffix_results.clear()
        return (
            tuple(
                host[k]
                for k in (
                    "update",
                    "residual",
                    "kv",
                    "index",
                    "repair",
                    "positions",
                    "counts",
                    "scores",
                    "offset",
                    "count",
                    "table",
                )
            )
            + (None,) * 7
            + (host["health"], host["rope"])
        )

    patches.setattr(layer, "device_inputs", inputs)
    patches.setattr(base, "capture_resident_buffers", lambda *a, **k: fake_memory())
    patches.setattr(
        base,
        "capture_identified_device_memory",
        lambda *a: [
            dict(
                device_id=r["device_id"],
                process_index=3,
                platform="tpu",
                **r["memory_stats"],
            )
            for r in fake_memory()["devices"]
        ],
    )
    patches.setattr(protocol, "observe", lambda r, f: observations[id(r)])
    patches.setattr(worker, "local_observations", lambda r: observations[id(r)])

    def prefix_inputs(values, tile, previous=None):
        assert previous is (prefixes[-1] if tile else None)
        return (np.int32(tile),)

    def suffix_inputs(parts, valid, dense, moe):
        if len(parts) == 4:
            assert all(a is b for a, b in zip(parts, prefixes))
            return (np.int32(-1),)
        tile = next(i for i, p in enumerate(prefixes) if parts[0] is p)
        assert int(valid) == max(0, min(32, int(state["count"]) - tile * 32))
        return (np.int32(tile),)

    patches.setattr(worker.programs, "prefix_inputs", prefix_inputs)
    patches.setattr(worker.programs, "suffix_inputs", suffix_inputs)

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return SimpleNamespace(**pins[self.name]["compiled_memory"])

        def __call__(self, tile):
            tile = int(tile)
            sequence.append((self.name, tile))
            fields = (
                protocol.PREFIX_FIELDS
                if self.name == "prefix"
                else protocol.SUFFIX_FIELDS
            )
            result = tuple(np.zeros((1, 1, 1)) for _ in fields)
            observed = {}
            for device, slot in calls.local_slots.items():
                whole = fixture_output(state, slot)
                if self.name == "prefix":
                    part = {
                        n: (
                            np.zeros((32, 1536), window.BF16)
                            if n == "mlp_input"
                            else (
                                whole[n]
                                if n in ("kv", "index", "repair")
                                else whole[n][tile * 32 : (tile + 1) * 32]
                            )
                        )
                        for n in fields
                    }
                else:
                    part = {
                        n: (
                            whole[n]
                            if tile == -1
                            else whole[n][tile * 32 : (tile + 1) * 32]
                        )
                        for n in fields
                    }
                if failure == "prefix_health" and self.name == "prefix" and tile == 1:
                    part["health"][0] = False
                if failure == "prefix_nan" and self.name == "prefix" and tile == 1:
                    part["mlp_input"][0, 0] = np.nan
                if failure == "narrow_health" and self.name == "control" and tile == 3:
                    part["health"][0] = False
                observed[device] = part
            observations[id(result)] = observed
            if self.name == "prefix":
                prefixes.append(result)
            else:
                suffix_results.append(result)
            return result

    calls.programs = {n: Program(n) for n in admission.PROGRAMS}

    def assembled(parts, suffix):
        assert all(a is b for a, b in zip(parts, prefixes))
        result = tuple(np.zeros(1) for _ in range(12))
        wide = suffix is suffix_results[0]
        observed = {}
        for device in calls.local_slots:
            suffix_host = (
                observations[id(suffix_results[0])][device]
                if wide
                else {
                    n: np.concatenate(
                        [observations[id(s)][device][n] for s in suffix_results[1:]]
                    )
                    for n in protocol.SUFFIX_FIELDS
                }
            )
            observed[device] = protocol.assemble(
                [observations[id(p)][device] for p in parts], suffix_host
            )
        observations[id(result)] = observed
        return result

    patches.setattr(worker.programs, "assemble_result", assembled)
    try:
        yield calls, sequence
    finally:
        journal.close()
        patches.undo()


def test_actual_budgeted_worker_three_cases_original_replay(tmp_path):
    with run_cases(tmp_path) as (calls, sequence):
        worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
        assert len(sequence) == 27
        assert sequence[:9] == [("prefix", t) for t in range(4)] + [
            ("candidate", -1)
        ] + [("control", t) for t in range(4)]
        assert len(calls.record["call_evidence"]) == 27
        serialized = json.loads(json.dumps(calls.record))
        for case in protocol.CASES:
            replay = protocol.replay_case(
                tmp_path / f"{case}.npz", case=case, slots_by_device=calls.local_slots
            )
            assert replay == serialized["cases"][case]["replay"]
            assert replay["passed"] and not replay["independent_full_layer_admission"]
        assert calls.record["performance_claim"] is False


@pytest.mark.parametrize("failure", ["prefix_health", "prefix_nan", "narrow_health"])
def test_failure_preserves_completed_components_before_successor(tmp_path, failure):
    with run_cases(tmp_path, failure) as (calls, sequence):
        with pytest.raises(ValueError, match="originals preserved"):
            worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
        assert len(sequence) == (9 if failure == "narrow_health" else 2)
        assert set(calls.record["cases"]) == {"boundary"}
        with np.load(tmp_path / "boundary.npz", allow_pickle=False) as saved:
            if failure == "prefix_nan":
                assert np.isnan(saved["prefix1_9__mlp_input"].view(window.BF16)[0, 0])
            else:
                key = (
                    "narrow3_9__health"
                    if failure == "narrow_health"
                    else "prefix1_9__health"
                )
                assert not saved[key][0]


@pytest.mark.parametrize(
    "change", ["assembly", "component", "dtype", "missing", "extra", "nan"]
)
def test_consumer_refuses_component_or_assembly_changes(tmp_path, change):
    with run_cases(tmp_path) as (calls, _):
        # One fixture output is sufficient for every consumer mutation.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(protocol, "CASES", ("boundary",))
            worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
        path = tmp_path / "boundary.npz"
        with np.load(path, allow_pickle=False) as saved:
            values = {n: saved[n].copy() for n in saved.files}
        if change == "assembly":
            values["actual_9__output"][0, 0] += 1
        elif change == "component":
            values["wide_9__output"][0, 0] += 1
        elif change == "dtype":
            values["prefix0_9__mlp_input"] = values["prefix0_9__mlp_input"].astype(
                np.float32
            )
        elif change == "missing":
            values.pop("prefix1_9__mlp_input")
        elif change == "extra":
            values["unexpected"] = np.zeros(1)
        else:
            values["prefix1_9__mlp_input"].view(window.BF16)[0, 0] = np.nan
        base.save_arrays(path, values)
        with pytest.raises((ValueError, KeyError)):
            protocol.replay_case(
                path, case="boundary", slots_by_device=calls.local_slots
            )


def test_capture_uses_addressable_shards_and_strict_leaf_owners():
    def leaf(data, ids=(9, 13, 25, 29)):
        return SimpleNamespace(
            addressable_shards=[
                SimpleNamespace(device=SimpleNamespace(id=d), data=data) for d in ids
            ]
        )

    values = (
        leaf(np.zeros((32, 1536), window.BF16)),
        leaf(np.zeros((32, 8), np.int32)),
        leaf(np.zeros((32, 8), np.float32)),
        leaf(np.ones((1, 1, 32), bool)),
    )
    observed = protocol.observe(values, protocol.SUFFIX_FIELDS)
    assert set(observed) == {9, 13, 25, 29} and observed[9]["health"].shape == (32,)
    for ids in ((9, 9, 25, 29), (9, 13)):
        bad = (*values[:3], leaf(np.ones((1, 1, 32), bool), ids))
        with pytest.raises(ValueError, match="owner"):
            protocol.observe(bad, protocol.SUFFIX_FIELDS)


@pytest.mark.parametrize("change", ["profile", "prefix", "budget", "journal"])
def test_binding_refuses_before_model_dispatch(tmp_path, change):
    with run_cases(tmp_path) as (calls, sequence):
        if change == "profile":
            calls.record["profile"] = "old"
        elif change == "prefix":
            calls.programs.pop("prefix")
        elif change == "budget":
            calls.budgeter = lambda *a, **k: {}
        else:
            journal = calls.journal
            calls.journal = SimpleNamespace(phase=journal.phase)
        with pytest.raises(ValueError):
            worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
        assert not sequence
