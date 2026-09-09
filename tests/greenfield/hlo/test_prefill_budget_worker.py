"""Actual budget lifecycle with fake device math/counters; no TPU evidence."""

import json
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_budget_probe as probe
from scripts.greenfield import prefill_budget_worker as worker
from scripts.greenfield import prefill_window_worker as shared


def memory():
    return dict(
        argument_size_in_bytes=10_000_000,
        output_size_in_bytes=524_544,
        temp_size_in_bytes=100_000_000,
        generated_code_size_in_bytes=100_000,
        alias_size_in_bytes=0,
    )


def hlo():
    groups = (
        "{"
        + ",".join("{" + ",".join(map(str, g)) + "}" for g in worker.EXPERT_GROUPS)
        + "}"
    )
    return f"""HloModule budget, num_partitions=32
ENTRY main {{
  p = s32[32,2048] parameter(0)
  s = f32[32,2048] parameter(1)
  pg = s32[256,2048] all-gather(p), dimensions={{0}}, replica_groups={groups}, channel_id=1, use_global_device_ids=true
  sg = f32[256,2048] all-gather(s), dimensions={{0}}, replica_groups={groups}, channel_id=2, use_global_device_ids=true
  ROOT output = (s32[256,2048], f32[256,2048]) tuple(pg, sg)
}}
"""


def test_fixed_graph_collective_and_memory_admission_json_replays():
    report = worker.inspect_program(worker.PROGRAMS[0], "stable", hlo(), memory())
    assert report == json.loads(json.dumps(report))
    assert report["exchanged_candidate_leaves"] == {"s32": 1, "f32": 1}
    assert report["model_performance_claim"] is False
    assert len(report["collective_groups"]) == 2


@pytest.mark.parametrize(
    "mutation",
    ["group", "payload", "opcode", "missing", "extra", "host", "memory", "alias"],
)
def test_graph_or_allocation_mutations_refuse(mutation):
    text, allocation = hlo(), memory()
    if mutation == "group":
        text = text.replace("0,4,8,12,16,20,24,28", "0,1,2,3,4,5,6,7")
    elif mutation == "payload":
        text = text.replace("s32[32,2048]", "s32[32,4096]")
    elif mutation == "opcode":
        text = text.replace("all-gather", "all-reduce")
    elif mutation == "missing":
        text = "\n".join(line for line in text.splitlines() if "sg =" not in line)
    elif mutation == "extra":
        line = next(line for line in text.splitlines() if "pg =" in line)
        text = text.replace("  ROOT", line.replace("pg =", "extra =") + "\n  ROOT")
    elif mutation == "host":
        text = text.replace(
            "  ROOT",
            '  host = f32[] custom-call(), custom_call_target="host_callback"\n  ROOT',
        )
    elif mutation == "memory":
        allocation["temp_size_in_bytes"] = (1 << 30) + 1
    else:
        allocation["alias_size_in_bytes"] = 1
    with pytest.raises(ValueError):
        worker.inspect_program(worker.PROGRAMS[0], "stable", text, allocation)


def make_worker(tmp_path, monkeypatch):
    import jax

    events = []
    record = dict(
        protocol=probe.PROTOCOL,
        profile=worker.PROFILE,
        compile_only=False,
        code_hash="a" * 40,
        launch_rank=0,
        jax_process_index=0,
        programs={},
    )
    journal = worker.BudgetJournal(
        tmp_path / "compile_journal.jsonl",
        {
            k: record[k]
            for k in ("protocol", "profile", "compile_only", "code_hash", "launch_rank")
        },
    )
    calls = shared.BudgetedCalls(
        root=tmp_path,
        record=record,
        consensus=lambda ok: ok,
        journal=journal,
        local_slots={i: i for i in range(4)},
    )

    def census(*args, **kwargs):
        events.append("census")
        return dict(
            schema_version="ws32_prefill_resident_buffers_v1",
            includes_all_live_arrays=True,
            devices=[
                dict(
                    device_id=i,
                    process_index=0,
                    platform="tpu",
                    buffers=[dict(bytes=10_000_000)],
                    accounted_resident_bytes=10_000_000,
                    memory_stats=dict(
                        bytes_in_use=10_000_000,
                        peak_bytes_in_use=4_000_000_000,
                        bytes_limit=33_014_398_976,
                    ),
                )
                for i in range(4)
            ],
        )

    def post(*args):
        return [
            dict(
                device_id=r["device_id"],
                process_index=0,
                platform="tpu",
                **r["memory_stats"],
            )
            for r in census()["devices"]
        ]

    monkeypatch.setattr(shared, "capture_resident_buffers", census)
    monkeypatch.setattr(shared, "capture_identified_device_memory", post)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        probe, "make_dsa_inputs", lambda mesh, case, tied=False: (case, tied)
    )
    monkeypatch.setattr(probe, "build_dsa_program", lambda mesh, capacity: capacity)

    class Program:
        def memory_analysis(self):
            return SimpleNamespace(**memory())

        def __call__(self, case, tied):
            events.append(("execute", case.name, tied))
            expected = probe.expected_selection(case.valid_lengths, tied=tied)

            def array(value):
                return SimpleNamespace(
                    addressable_shards=[
                        SimpleNamespace(device=SimpleNamespace(id=i), data=value.copy())
                        for i in range(4)
                    ]
                )

            return tuple(
                array(expected[name])
                for name in ("positions", "valid_counts", "scores")
            ), array(np.ones((1, 1), np.bool_))

    def compiler(fn, inputs, name, root, record, *, journal):
        events.append(("compile", name))
        journal.begin(name)
        stable, optimized = "fixture_stable", hlo()
        (root / f"{name}.stablehlo.mlir").write_text(stable)
        journal.compiled(name, seconds=0.1, memory=memory(), device_memory=[])
        (root / f"{name}.optimized_hlo.txt").write_text(optimized)
        record["programs"][name] = dict(
            compiled_memory=memory(),
            compile_seconds=0.1,
            stablehlo_sha256=sha256(stable.encode()).hexdigest(),
            optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
        )
        return Program()

    return calls, compiler, events


@pytest.mark.parametrize("include_overhead", [False, True])
def test_actual_prepare_sampler_journal_npz_lifecycle(
    tmp_path, monkeypatch, include_overhead
):
    calls, compiler, events = make_worker(tmp_path, monkeypatch)
    if include_overhead:
        from scripts.greenfield import prefill_budget_overhead as overhead
        from scripts.greenfield import ws32_batched_prefill_runner as runner
        from tests.greenfield.hlo.test_prefill_budget_overhead import Array, state

        def initialize(mesh, config, *, prompt_length, clock):
            return state(config, prompt_length), dict(
                capacity=config.context_capacity,
                prompt_length=prompt_length,
                cache_initialization_seconds=0.0,
                weights_loaded=False,
                initialized_prefix_length=0,
                model_ttft_measured=False,
            )

        monkeypatch.setattr(probe, "measure_initial_state", initialize)
        monkeypatch.setattr(
            overhead,
            "capture_identified_device_memory",
            lambda devices: [
                dict(
                    device_id=i,
                    process_index=0,
                    platform="tpu",
                    bytes_in_use=4_000_000_000,
                    peak_bytes_in_use=4_000_000_000,
                    bytes_limit=33_014_398_976,
                )
                for i in range(4)
            ],
        )
        monkeypatch.setattr(
            runner, "replicated", lambda mesh, value: Array(np.asarray(value))
        )
        report = worker.run_budget_campaign(calls, "mesh", compiler=compiler)
    else:
        prepared = worker.prepare(calls, "mesh", compiler=compiler)
        assert [v for v in events if isinstance(v, tuple)] == [
            ("compile", name) for name in worker.PROGRAMS
        ]
        report = worker.run_samples(calls, prepared)
    calls.journal.close()
    assert len(report) == 6 and all(
        len(r["samples_seconds"]) == 5 for r in report.values()
    )
    assert len(calls.record["call_evidence"]) == 44  # 2ties +6*(2warmups+5samples)
    assert len(calls.record["budget_originals"]) == 14  # 2ties +6*(first+last)
    assert all(
        e["completed"] and len(e["post_memory"]) == 4
        for e in calls.record["call_evidence"]
    )
    for case in probe.cases():
        expected = probe.expected_selection(case.valid_lengths)
        for sample in (0, 6):
            with np.load(tmp_path / f"{case.name}_sample{sample}.npz") as data:
                assert len(data.files) == 16
                for device in range(4):
                    for field, reference in expected.items():
                        np.testing.assert_array_equal(
                            data[f"device{device}_{field}"], reference
                        )
    journal = [
        json.loads(line)
        for line in (tmp_path / "compile_journal.jsonl").read_text().splitlines()
    ]
    start = next(
        i for i, r in enumerate(journal) if r["stage"] == "budget/sampling_started"
    )
    assert sum(r["stage"] == "inspected" for r in journal[:start]) == 2
    assert all(r["stage"] not in ("compiled", "inspected") for r in journal[start:])
    assert json.loads((tmp_path / "runner.json").read_text())["budget_cases"] == report
    from scripts.greenfield import prefill_budget_evidence as evidence

    calls.record["compile_journal_sha256"] = sha256(
        (tmp_path / "compile_journal.jsonl").read_bytes()
    ).hexdigest()
    record = json.loads(json.dumps(calls.record))
    replay = evidence.validate_files(
        tmp_path, record, slots=calls.local_slots, require_overhead=include_overhead
    )
    assert replay["replayed_originals"] == 14 and replay["replayed_calls"] == 44
    if not include_overhead:
        with pytest.raises(ValueError, match="missing overhead"):
            evidence.validate_files(
                tmp_path, record, slots=calls.local_slots, require_overhead=True
            )

    # Reuse the one completed fixture for negative collector tests.
    for kind in (
        "sample",
        "scope",
        "owner",
        "digest",
        "allocation",
        "missing_call",
        "duration",
        "flag",
    ):
        bad = deepcopy(record)
        case = probe.cases()[0].name
        if kind == "sample":
            bad["budget_cases"][case]["samples_seconds"][0] += 1
        elif kind == "scope":
            bad["budget_cases"][case]["model_ttft_measured"] = True
        elif kind == "owner":
            bad["call_evidence"][0]["post_memory"][0]["device_id"] = 31
        elif kind == "digest":
            bad["budget_originals"][case + "_sample0.npz"] = "0" * 64
        elif kind == "allocation":
            bad["call_evidence"][0]["compiled_memory"][worker.PROGRAMS[0]][
                "temp_size_in_bytes"
            ] += 1
        elif kind == "missing_call":
            bad["call_evidence"].pop()
        elif kind == "duration":
            bad["budget_sampling"]["elapsed_seconds"] = 121
        else:
            bad["budget_cases"][case]["checks"][0]["passed"] = 1
        with pytest.raises(ValueError):
            evidence.validate_files(tmp_path, bad, slots=calls.local_slots)

    # A new matching digest cannot bless changed original array values.
    path = tmp_path / (probe.cases()[0].name + "_sample0.npz")
    with np.load(path) as original:
        arrays = {k: original[k] for k in original.files}
    arrays["device0_positions"][0, 0] = -1
    bad = deepcopy(record)
    bad["budget_originals"][path.name] = shared.save_arrays(path, arrays)
    with pytest.raises(ValueError, match="analytic reference"):
        evidence.validate_files(tmp_path, bad, slots=calls.local_slots)


@pytest.mark.parametrize(
    "failure", ["pre_publication", "peer_preflight", "post_publication", "output"]
)
def test_actual_voted_failure_stops_without_successor_and_keeps_originals(
    tmp_path, monkeypatch, failure
):
    calls, compiler, events = make_worker(tmp_path, monkeypatch)
    prepared = worker.prepare(calls, "mesh", compiler=compiler)
    original = shared._atomic_json

    def publish(path, record):
        phase = record.get("current_phase", "")
        if failure == "pre_publication" and phase.endswith("/tie/memory"):
            raise OSError("preflight publication failed")
        if failure == "post_publication" and phase.endswith("/tie/execute"):
            raise OSError("completion publication failed")
        original(path, record)

    monkeypatch.setattr(shared, "_atomic_json", publish)
    if failure == "peer_preflight":
        calls.consensus = lambda ok: ok and not calls.record.get(
            "current_phase", ""
        ).endswith("/tie/memory")
    if failure == "output":

        def bad_output(*args):
            raise ValueError("wrong selected positions")

        monkeypatch.setattr(probe, "check_output", bad_output)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        worker.run_samples(calls, prepared)
    executed = [
        event for event in events if isinstance(event, tuple) and event[0] == "execute"
    ]
    assert len(executed) == (
        0 if failure in ("pre_publication", "peer_preflight") else 1
    )
    assert len(list(tmp_path.glob("*_tie.npz"))) == len(executed)
    calls.journal.close()


def test_prepared_geometry_drift_refuses_before_tied_dispatch(tmp_path, monkeypatch):
    calls, compiler, events = make_worker(tmp_path, monkeypatch)
    prepared = worker.prepare(calls, "mesh", compiler=compiler)
    prepared.pop(next(iter(prepared)))
    with pytest.raises(ValueError, match="prepared workload"):
        worker.run_samples(calls, prepared)
    assert not any(isinstance(e, tuple) and e[0] == "execute" for e in events)
    calls.journal.close()
