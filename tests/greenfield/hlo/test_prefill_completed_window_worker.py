"""CPU worker lifecycle and original NPZ replay; executable math is a fixture."""

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import inspect
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
def run_cases(root, failure=None, *, create_journal=True):
    patches = pytest.MonkeyPatch()
    record = dict(
        protocol=protocol.PROTOCOL,
        profile=admission.PROFILE,
        compile_only=False,
        code_hash="a" * 40,
        launch_rank=0,
        jax_process_index=3,
        local_device_slots=[
            dict(device_id=d, device_slot=s)
            for d, s in {9: 0, 13: 1, 25: 2, 29: 3}.items()
        ],
        cases={},
    )
    journal = (
        worker.CompletedJournal(
            root / "compile_journal.jsonl",
            {
                k: record[k]
                for k in (
                    "protocol",
                    "profile",
                    "compile_only",
                    "code_hash",
                    "launch_rank",
                )
            },
        )
        if create_journal
        else None
    )
    calls = base.BudgetedCalls(
        root=root,
        record=record,
        consensus=lambda ok: ok,
        journal=journal,
        local_slots={9: 0, 13: 1, 25: 2, 29: 3},
        budgeter=worker.assembly.memory_budget,
    )
    state, observations, prefixes, suffix_results, sequence = {}, {}, [], [], []
    pins = admission.registered_programs()
    for name in worker.assembly.PROGRAMS:
        pins[name] = dict(
            compiled_memory={
                key: 0 if key == "alias_size_in_bytes" else 1024
                for key in worker.assembly.ALLOCATION_CAPS
            }
        )

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

    def suffix_inputs(parts, valid, tile=None):
        if len(parts) == 4:
            assert all(a is b for a, b in zip(parts, prefixes))
            return (np.int32(-1),)
        tile = int(tile)
        assert parts[0] is prefixes[tile]
        assert int(valid) == int(state["count"])
        return (np.int32(tile),)

    patches.setattr(
        worker.assembly,
        "place_tiles",
        lambda mesh: tuple(np.int32(i) for i in range(4)),
    )
    patches.setattr(worker.assembly, "prefix_arguments", lambda values, tile: (tile,))
    patches.setattr(
        worker.assembly,
        "attach_prefix",
        lambda values, prepared, previous: prefix_inputs(
            values, int(prepared), previous
        ),
    )
    patches.setattr(worker.assembly, "suffix_arguments", suffix_inputs)
    patches.setattr(
        worker.assembly, "attach_suffix", lambda values, prepared: (prepared,)
    )

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return SimpleNamespace(**pins[self.name]["compiled_memory"])

        def __call__(self, tile, *extra):
            if self.name == "assemble":
                sequence.append((self.name, -1))
                return assembled(tile, extra[0]), assembled(tile, None)
            tile = int(tile)
            sequence.append((self.name, tile))
            if self.name.startswith("prepare_"):
                return np.int32(tile)
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

    calls.programs = {
        n: Program(n) for n in (*admission.PROGRAMS, *worker.assembly.PROGRAMS)
    }

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

    patches.setattr(
        worker.assembly,
        "assembly_arguments",
        lambda prefixes, wide, narrow: (prefixes, wide, narrow),
    )
    patches.setattr(worker.assembly, "attach_result", lambda rows, last: rows)
    try:
        yield calls, sequence
    finally:
        if journal is not None:
            journal.close()
        patches.undo()


def test_actual_budgeted_worker_three_cases_original_replay(tmp_path):
    with run_cases(tmp_path, create_journal=False) as (calls, sequence):
        continue_from_acquisition(tmp_path, calls, sequence)
        assert len(sequence) == 59
        model_sequence = [
            entry
            for entry in sequence
            if entry[0] in ("prefix", "candidate", "control")
        ]
        assert model_sequence[:9] == [("prefix", t) for t in range(4)] + [
            ("candidate", -1)
        ] + [("control", t) for t in range(4)]
        assert len(calls.record["call_evidence"]) == 59
        assert calls.record["call_evidence"][-1]["graph"] == "assemble"
        assert len(calls.record["call_evidence"][-1]["post_memory"]) == 4
        serialized = json.loads(json.dumps(calls.record))
        for case in protocol.CASES:
            replay = protocol.replay_case(
                tmp_path / f"{case}.npz", case=case, slots_by_device=calls.local_slots
            )
            assert replay == serialized["cases"][case]["replay"]
            assert replay["passed"] and not replay["independent_full_layer_admission"]
        assert calls.record["performance_claim"] is False
        assert calls.record["model_executable_calls"] == 27
        assert calls.record["wk_executable_calls"] == 2
        assert calls.record["assembly_executable_calls"] == 30
        from scripts.greenfield import prefill_window_evidence as evidence

        evidence.validate_calls(
            serialized, local_slots=calls.local_slots, completed_numerical=True
        )
        evidence.validate_wk(tmp_path, serialized, calls.local_slots)
        evidence.validate_files(tmp_path, serialized, completed_numerical=True)
        for change in ("omitted_helper", "missing_code", "last_peak", "bool_count"):
            bad = deepcopy(serialized)
            if change == "omitted_helper":
                bad["call_evidence"].pop(2)
            elif change == "missing_code":
                del bad["call_evidence"][-1]["compiled_memory"]["prepare_wide"]
            elif change == "last_peak":
                bad["call_evidence"][-1]["post_memory"][0]["peak_bytes_in_use"] = 0
            else:
                bad["call_evidence"][-1]["completed_call_seconds"] = True
            with pytest.raises(ValueError):
                evidence.validate_calls(
                    bad, local_slots=calls.local_slots, completed_numerical=True
                )


def continue_from_acquisition(root, calls, sequence, *, failure=None):
    """Actual five-original + four helper compiler -> WK -> case continuation.

    Math/counters are fixtures; existing production compiler, admission, journal,
    nine-program live budget, phase votes and WK publication really execute.
    """
    import jax
    from scripts.greenfield import prefill_window_acquisition as acquisition
    from tests.greenfield.hlo.test_prefill_completed_window_admission import ORIGINAL

    programs = calls.programs.copy()
    wk_values = np.zeros((128, 6144), window.BF16)
    wk_values[3, 7] = 1.5

    class Distributed:
        def __init__(self, value):
            self.addressable_shards = [
                SimpleNamespace(device=SimpleNamespace(id=d), data=value.copy())
                for d in calls.local_slots
            ]

    class Function:
        def __init__(self, name):
            self.name = name

        def lower(self, *values):
            name = self.name
            if name in admission.PROGRAMS:
                functions = [f.function for f in inspect.stack()]
                assert functions[1:6] == [
                    "compile_program",
                    "<lambda>",
                    "fleet_step",
                    "acquire_programs",
                    "execute_acquisition",
                ]
                assert "execute_numerical" not in functions
            if failure == name:
                raise ValueError("injected helper compile failure")

            class Compiled:
                def memory_analysis(self):
                    return programs[name].memory_analysis()

                def as_text(self):
                    if name in admission.PROGRAMS:
                        return (ORIGINAL / f"{name}.optimized_hlo.txt").read_text()
                    return f"HloModule {name}\nENTRY %main {{\n%p = f32[2] parameter(0)\nROOT %out = f32[2] copy(%p)\n}}"

                def __call__(self, *values):
                    if name.startswith("wk_"):
                        sequence.append((name, -1))
                        if name == "wk_decode":
                            return Distributed(wk_values)
                        np.testing.assert_array_equal(
                            values[0].addressable_shards[0].data, wk_values
                        )
                        return Distributed(wk_values.astype(np.float32))
                    return programs[name](*values)

            return SimpleNamespace(
                compiler_ir=lambda **kwargs: (
                    (ORIGINAL / f"{name}.stablehlo.mlir").read_text()
                    if name in admission.PROGRAMS
                    else f"module @{name} {{}}"
                ),
                compile=Compiled,
            )

    with pytest.MonkeyPatch.context() as patch:
        original_ready = jax.block_until_ready
        patch.setattr(
            jax,
            "block_until_ready",
            lambda x: x if isinstance(x, Distributed) else original_ready(x),
        )
        patch.setattr(
            jax,
            "local_devices",
            lambda: [SimpleNamespace(id=d) for d in calls.local_slots],
        )
        patch.setattr(
            layer,
            "_memory_stats",
            lambda d: fake_memory()["devices"][0]["memory_stats"],
        )
        patch.setattr(layer, "input_specs", lambda *a: ())
        patch.setattr(
            acquisition,
            "prepare_programs",
            lambda **kwargs: tuple((n, Function(n), ()) for n in admission.PROGRAMS),
        )
        patch.setattr(
            worker.assembly,
            "prepare_programs",
            lambda mesh: tuple((n, Function(n), ()) for n in worker.assembly.PROGRAMS),
        )
        weights = SimpleNamespace(
            dsa=SimpleNamespace(wk_bits_local=np.zeros(1), wk_scale_local=np.zeros(1))
        )
        acquisition.execute_acquisition(
            args=SimpleNamespace(output_dir=root),
            record=calls.record,
            mesh=None,
            config=None,
            weights=weights,
            consensus=calls.consensus,
            local_slots=calls.local_slots,
            completed_window=True,
            completed_numerical=True,
        )
    assert (
        calls.record["compile_journal_sha256"]
        == sha256((root / "compile_journal.jsonl").read_bytes()).hexdigest()
    )
    assert json.loads((root / "runner.json").read_text()) == calls.record


@pytest.mark.parametrize("failure", worker.assembly.PROGRAMS)
def test_helper_compile_failure_keeps_originals_and_finalizes_before_wk(
    tmp_path, failure
):
    with run_cases(tmp_path, create_journal=False) as (calls, sequence):
        with pytest.raises(ValueError, match="injected helper compile failure"):
            continue_from_acquisition(tmp_path, calls, sequence, failure=failure)
        assert not sequence and not list(tmp_path.glob("*.npz"))
        for name in admission.PROGRAMS:
            assert (tmp_path / f"{name}.optimized_hlo.txt").is_file()
        assert (
            calls.record["compile_journal_sha256"]
            == sha256((tmp_path / "compile_journal.jsonl").read_bytes()).hexdigest()
        )
        assert (
            calls.record["acquisition_phases"]["close_compile_journal"]["status"]
            == "COMPLETE"
        )


@pytest.mark.parametrize("failure", ["prefix_health", "prefix_nan", "narrow_health"])
def test_failure_preserves_completed_components_before_successor(tmp_path, failure):
    with run_cases(tmp_path, failure) as (calls, sequence):
        with pytest.raises(ValueError, match="originals preserved"):
            worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
        assert len(sequence) == (18 if failure == "narrow_health" else 4)
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


def test_consumer_refuses_component_or_assembly_changes(tmp_path):
    with run_cases(tmp_path) as (calls, _):
        # One fixture output is sufficient for every consumer mutation.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(protocol, "CASES", ("boundary",))
            worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
        path = tmp_path / "boundary.npz"
        with np.load(path, allow_pickle=False) as saved:
            originals = {n: saved[n].copy() for n in saved.files}
        for change in ("assembly", "component", "dtype", "missing", "extra", "nan"):
            values = deepcopy(originals)
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


def test_last_assembly_originals_survive_postpeak_refusal(tmp_path, monkeypatch):
    with run_cases(tmp_path) as (calls, sequence):
        original = base.capture_identified_device_memory

        def post(*args):
            if calls.record["call_evidence"][-1]["graph"] == "assemble":
                with np.load(tmp_path / "boundary.npz", allow_pickle=False) as saved:
                    assert "actual_9__output" in saved and "control_9__output" in saved
                raise ValueError("final assembly peak refused")
            return original(*args)

        monkeypatch.setattr(base, "capture_identified_device_memory", post)
        with pytest.raises(ValueError, match="final assembly peak"):
            worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
        assert len(sequence) == 19 and sequence[-1][0] == "assemble"
        assert set(calls.record["cases"]) == {"boundary"}


def test_peer_memory_refusal_before_helper_prevents_dispatch(tmp_path):
    with run_cases(tmp_path) as (calls, sequence):
        calls.consensus = (
            lambda ok: ok
            and calls.record.get("current_phase") != "boundary/prepare_prefix0/memory"
        )
        with pytest.raises(RuntimeError, match="peer refused"):
            worker.execute_cases(calls, weights=None, wk=None, mesh=None, specs=())
        assert not sequence and calls.record["call_evidence"][0]["completed"] is False


@pytest.mark.parametrize(
    "change",
    [
        "protocol",
        "profile",
        "compile_only",
        "reference_scope",
        "independent_full_layer_admission",
        "performance_claim",
        "programs",
        "mixed",
    ],
)
def test_completed_continuation_wrong_scope_refuses_before_wk(tmp_path, change):
    with run_cases(tmp_path) as (calls, sequence):
        calls.record.update(
            programs={n: {} for n in calls.programs},
            reference_scope=protocol.REFERENCE_SCOPE,
            independent_full_layer_admission=False,
            performance_claim=False,
        )
        if change == "programs":
            del calls.record["programs"]["assemble"]
        elif change != "mixed":
            calls.record[change] = "wrong"
        with pytest.raises(ValueError):
            base.execute_numerical(
                args=SimpleNamespace(output_dir=tmp_path),
                record=calls.record,
                mesh=None,
                config=None,
                weights=None,
                local_slots=calls.local_slots,
                consensus=calls.consensus,
                compiled=tuple(calls.programs.values()),
                journal=calls.journal,
                completed_numerical=True,
                boundary_diagnostic=change == "mixed",
            )
        assert not sequence and not calls.record["call_evidence"]
        assert (
            calls.record["compile_journal_sha256"]
            == sha256((tmp_path / "compile_journal.jsonl").read_bytes()).hexdigest()
        )
