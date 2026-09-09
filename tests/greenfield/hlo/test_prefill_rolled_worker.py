"""Real journal/NPZ/budget/collector lifecycle, fixture device math/counters."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_rolled_window as protocol
from scripts.greenfield import prefill_rolled_admission as admission
from scripts.greenfield import prefill_rolled_worker as worker
from scripts.greenfield import prefill_rolled_evidence as evidence
from scripts.greenfield import prefill_window_worker as shared

ROOT = Path("/home/gianl/glm-run") / (
    "greenfield_fp8_ws32_prefill_expert_panel_phase_l6_20260909T033031628458338Z"
)


@pytest.fixture(scope="module")
def reference():
    return protocol.load_reference(ROOT, rank=0)


def setup_worker(root, monkeypatch, reference, *, defect=None):
    import jax
    from scripts.greenfield import probe_ws32_prefill_layer as entry

    record = deepcopy(reference.record)
    record.update(
        protocol=protocol.PROTOCOL,
        profile=admission.PROFILE,
        code_hash="a" * 40,
        programs={},
        cases={},
        compile_only=False,
        performance_claim=False,
    )
    dispatched = []
    allocations = {
        n: admission.registered_programs()[n]["compiled_memory"]
        for n in protocol.PROGRAMS[:2]
    }
    allocations["candidate"] = dict(
        argument_size_in_bytes=350_000_000,
        output_size_in_bytes=10_000_000,
        temp_size_in_bytes=100_000_000,
        generated_code_size_in_bytes=20_000_000,
        alias_size_in_bytes=0,
    )
    captures = {}
    for name in protocol.PROGRAMS[:2]:
        with np.load(ROOT / "fleet/rank0" / f"{name}.npz", allow_pickle=False) as saved:
            values = {int(d): saved[d] for d in saved.files}
        if name == "wk_decode":
            values = {d: v.view(protocol.window.BF16) for d, v in values.items()}
        captures[name] = SimpleNamespace(
            addressable_shards=[
                SimpleNamespace(device=SimpleNamespace(id=d), data=v)
                for d, v in values.items()
            ]
        )
    captures["candidate"] = {d: dict(v) for d, v in reference.controls.items()}
    if defect == "numerical":
        first = next(iter(reference.slots))
        captures["candidate"][first]["output"] = captures["candidate"][first][
            "output"
        ].copy()
        captures["candidate"][first]["output"][0, 0] = np.nan

    def census(*a, **kw):
        return dict(
            schema_version="ws32_prefill_resident_buffers_v1",
            includes_all_live_arrays=True,
            devices=[
                dict(
                    device_id=d,
                    process_index=record["jax_process_index"],
                    platform="tpu",
                    buffers=[dict(bytes=1_000_000_000)],
                    accounted_resident_bytes=1_000_000_000,
                    memory_stats=dict(
                        bytes_in_use=1_000_000_000,
                        peak_bytes_in_use=4_000_000_000,
                        bytes_limit=33_014_398_976,
                    ),
                )
                for d in reference.slots
            ],
        )

    def post(*a):
        return [
            dict(
                device_id=r["device_id"],
                process_index=r["process_index"],
                platform="tpu",
                **r["memory_stats"],
            )
            for r in census()["devices"]
        ]

    monkeypatch.setattr(shared, "capture_resident_buffers", census)
    monkeypatch.setattr(shared, "capture_identified_device_memory", post)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(worker, "local_observations", lambda value: value)
    monkeypatch.setattr(entry, "input_specs", lambda *a: ())
    monkeypatch.setattr(entry, "device_inputs", lambda *a: ("initial-state",))
    monkeypatch.setattr(
        protocol,
        "prepare_programs",
        lambda **kw: tuple((n, n, ()) for n in protocol.PROGRAMS),
    )

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return SimpleNamespace(**allocations[self.name])

        def __call__(self, *args):
            dispatched.append(self.name)
            return captures[self.name]

    def compile_program(fn, values, name, root, record, *, journal):
        journal.begin(name)
        stable, optimized = name + " stable fixture", name + " optimized fixture"
        (root / f"{name}.stablehlo.mlir").write_text(stable)
        (root / f"{name}.optimized_hlo.txt").write_text(optimized)
        memory = dict(allocations[name])
        record["programs"][name] = dict(
            stablehlo_sha256=sha256(stable.encode()).hexdigest(),
            optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
            compiled_memory=memory,
            compile_seconds=0.1,
        )
        journal.compiled(name, seconds=0.1, memory=memory, device_memory=post())
        return Program(name)

    monkeypatch.setattr(entry, "compile_program", compile_program)
    # This test covers actual persistence, budget and independent numerical replay,
    # not actual-TPU HLO validation; raw/structural admission is separately tested.
    monkeypatch.setattr(
        admission,
        "inspect_program",
        lambda n, s, h, m: dict(
            passed=True, name=n, stable=s, optimized=h, memory=dict(m)
        ),
    )

    def consensus(ok):
        if (
            defect == "peer_after_candidate"
            and record.get("current_phase") == "candidate/execute"
        ):
            return False
        return ok

    weights = SimpleNamespace(
        dsa=SimpleNamespace(wk_bits_local="bits", wk_scale_local="scale")
    )
    return (
        record,
        dispatched,
        dict(
            root=root,
            record=record,
            mesh=None,
            config=None,
            weights=weights,
            local_slots=reference.slots,
            consensus=consensus,
            reference=reference,
        ),
    )


def test_actual_three_call_worker_original_collector_and_mutations(
    tmp_path, monkeypatch, reference
):
    record, dispatched, kwargs = setup_worker(tmp_path, monkeypatch, reference)
    worker.execute(**kwargs)
    assert dispatched == list(protocol.PROGRAMS)
    record = json.loads(json.dumps(record))
    evidence.validate_files(tmp_path, record, reference)
    for field in ("candidate_comparison", "retained_original_comparison"):
        mutated = deepcopy(record)
        mutated[field]["report_bytes"] += 1
        with pytest.raises(ValueError, match="differs"):
            evidence.validate_files(tmp_path, mutated, reference)
    mutated = deepcopy(record)
    mutated["call_evidence"] = mutated["call_evidence"][:-1]
    with pytest.raises(ValueError):
        evidence.validate_files(tmp_path, mutated, reference)
    mutated = deepcopy(record)
    mutated["call_evidence"][-1]["post_memory"][0]["peak_bytes_in_use"] = 34_000_000_000
    with pytest.raises(ValueError):
        evidence.validate_files(tmp_path, mutated, reference)
    mutated = deepcopy(record)
    mutated["programs"]["candidate"]["compiled_memory"]["temp_size_in_bytes"] += 1
    with pytest.raises(ValueError):
        evidence.validate_files(tmp_path, mutated, reference)


@pytest.mark.parametrize("defect", ["numerical", "peer_after_candidate"])
def test_completed_candidate_preserved_before_local_or_peer_refusal(
    tmp_path, monkeypatch, reference, defect
):
    record, dispatched, kwargs = setup_worker(
        tmp_path, monkeypatch, reference, defect=defect
    )
    with pytest.raises((ValueError, RuntimeError)):
        worker.execute(**kwargs)
    assert dispatched == list(protocol.PROGRAMS)
    assert (tmp_path / "candidate.npz").is_file()
    assert record["call_evidence"][-1]["completed"]
    assert not record.get("integration_complete", False)
    assert record["acquisition_phases"]["rolled_finalize"]["status"] == "COMPLETE"


def test_live_owner_mismatch_refuses_before_journal_or_compile(
    tmp_path, monkeypatch, reference
):
    record, dispatched, kwargs = setup_worker(tmp_path, monkeypatch, reference)
    kwargs["local_slots"] = {d: (s + 4) % 32 for d, s in reference.slots.items()}
    with pytest.raises(ValueError, match="owners"):
        worker.execute(**kwargs)
    assert not dispatched and not (tmp_path / "compile_journal.jsonl").exists()
