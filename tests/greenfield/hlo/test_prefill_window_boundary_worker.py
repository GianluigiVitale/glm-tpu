"""Bounded capture lifecycle and original fingerprints; CPU only."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_window_boundary_worker as boundary
from scripts.greenfield import prefill_window_boundary_admission as admission
from scripts.greenfield import prefill_window_worker as worker
from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield import probe_ws32_prefill_layer as layer_worker
from scripts.greenfield.prefill_layer_evidence import decode_arrays, input_hashes
from scripts.greenfield.prefill_layer_numerical import FIELDS
from tests.greenfield.hlo.test_prefill_window_worker import fixture_output, fake_memory

ORIGINAL = Path(
    "/home/gianl/glm-run/greenfield_fp8_ws32_prefill_layer_window_numerical_l6_"
    "20260908T132240925858058Z/rank0"
)


def test_archived_outputs_reproduce_slot_bound_hashes_and_every_field_mutation():
    if not ORIGINAL.is_dir():
        pytest.skip("requires original failed boundary arrays")
    record = json.loads((ORIGINAL / "runner.json").read_text())
    receipt = boundary.original_receipt()
    source = next(r for r in receipt["original_sources"] if r["rank"] == 0)
    for filename in ("runner.json", "boundary.npz"):
        expected = next(
            r for r in source["originals"] if r["name"].endswith("/" + filename)
        )
        raw = (ORIGINAL / filename).read_bytes()
        assert len(raw) == expected["size"]
        assert sha256(raw).hexdigest() == expected["original_sha256"]
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
    binding = boundary.bind_originals(record, slots)
    assert binding["original_verdict"] == "FAILED"
    with np.load(ORIGINAL / "boundary.npz", allow_pickle=False) as arrays:
        for device, slot in slots.items():
            for kind in ("actual", "control"):
                observed = decode_arrays(arrays, f"{kind}_{device}", FIELDS)
                result = boundary.compare_fingerprints(observed, slot=slot, kind=kind)
                assert (
                    result["all_outputs_reproduced"] and result["signature_reproduced"]
                )
                assert not result["numerical_admission"]
                for field in FIELDS:
                    changed = dict(observed)
                    value = observed[field].copy()
                    value.view(np.uint8).flat[0] ^= 1
                    changed[field] = value
                    result = boundary.compare_fingerprints(
                        changed, slot=slot, kind=kind
                    )
                    assert not result["all_outputs_reproduced"]
                    assert not result["fields_byte_identical"][field]
                    assert result["signature_reproduced"] == (
                        field not in boundary.SIGNATURE_FIELDS
                    )


@pytest.mark.parametrize("change", ["duplicate", "identity", "mesh", "bool", "receipt"])
def test_bad_original_binding_refuses(tmp_path, monkeypatch, change):
    original = boundary.original_receipt()
    record = dict(physical_device_ids=original["physical_device_ids"])
    slots = {0: 0, 8: 1, 16: 2, 24: 3}
    if change == "duplicate":
        slots[8] = 0
    elif change == "identity":
        slots = {0: 0, 1: 1, 2: 2, 3: 3}
    elif change == "mesh":
        record["physical_device_ids"] = list(reversed(record["physical_device_ids"]))
    elif change == "bool":
        slots[0] = False
    else:
        path = tmp_path / "receipt.json"
        path.write_bytes(boundary.ORIGINAL_RECEIPT.read_bytes() + b"\n")
        monkeypatch.setattr(boundary, "ORIGINAL_RECEIPT", path)
    with pytest.raises(ValueError):
        boundary.bind_originals(record, slots)


def make_calls(tmp_path):
    record = dict(code_hash="a" * 40, launch_rank=0, jax_process_index=3, programs={})
    journal = boundary.BoundaryJournal(
        tmp_path / "compile_journal.jsonl",
        dict(protocol=boundary.PROTOCOL, profile=admission.PROFILE, compile_only=False),
    )
    return worker.BudgetedCalls(
        root=tmp_path,
        record=record,
        consensus=lambda ok: ok,
        journal=journal,
        local_slots={9: 0, 13: 1, 25: 2, 29: 3},
        budgeter=admission.memory_budget,
    )


@pytest.mark.parametrize(
    "late_failure", [None, "health", "schema", "post_memory", "nan", "inf"]
)
def test_boundary_only_causal_carry_captures_and_failure_preservation(
    tmp_path, monkeypatch, late_failure
):
    calls = make_calls(tmp_path)
    host_now, results, operands, sequence = {}, {}, {}, []
    last_control = [None]

    def inputs(host, *args):
        host_now.update(host)
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

    monkeypatch.setattr(layer_worker, "device_inputs", inputs)
    monkeypatch.setattr(boundary, "local_observations", lambda r: results[id(r)])
    monkeypatch.setattr(
        boundary, "capture_owner_arrays", lambda r, **kw: operands[id(r)]
    )
    for name, rows in (("candidate", 128), ("control", 32)):
        calls.record["programs"][name] = dict(
            compiler_output_schema=dict(
                captures={
                    "router/input": dict(shape=[8, 4, rows, 1536], dtype="bfloat16")
                }
            )
        )

    def call(phase, name, values, *, preserve):
        sequence.append(phase)
        tile = None if name == "candidate" else int(phase[-1])
        if tile is not None:
            for i, key in enumerate(("kv", "index", "repair"), 2):
                assert values[i] is (host_now[key] if tile == 0 else last_control[0][i])
            assert int(values[8]) == 505 + 32 * tile
            assert int(values[9]) == 32
        original = tuple(np.full(1, len(sequence)) for _ in range(12))
        captured = object()
        obs, cap = {}, {}
        for device, slot in calls.local_slots.items():
            value = fixture_output(host_now, slot)
            if tile is not None:
                value = {
                    k: (
                        v
                        if k in ("kv", "index", "repair")
                        else v[32 * tile : 32 * (tile + 1)]
                    )
                    for k, v in value.items()
                }
            if late_failure == "health" and tile == 3:
                value["health"][-1] = False
            obs[device] = value
            cap[device] = {
                "router/input": np.zeros(
                    (128 if tile is None else 32, 1536), window.BF16
                )
            }
        if late_failure == "schema" and tile == 3:
            cap[9]["router/input"] = np.zeros((31, 1536), window.BF16)
        if late_failure in ("nan", "inf") and tile == 0:
            cap[9]["router/input"][0, 0] = float(late_failure)
        results[id(original)], operands[id(captured)] = obs, cap
        if tile is not None:
            last_control[0] = original
        preserve((original, captured))
        if late_failure == "post_memory" and tile == 3:
            raise ValueError("post memory failed")
        return original, captured

    calls.call = call
    try:
        if late_failure:
            with pytest.raises(ValueError):
                boundary.execute_boundary_case(
                    calls, weights=None, wk=None, mesh=None, specs=()
                )
        else:
            boundary.execute_boundary_case(
                calls, weights=None, wk=None, mesh=None, specs=()
            )
            assert calls.record["boundary_capture_complete"]
            assert not calls.record[
                "original_signature_reproduced"
            ]  # synthetic outputs, not original
            assert (
                calls.record["classification"]
                == "INSTRUMENTATION_PERTURBED_ORIGINAL_SIGNATURE"
            )
            assert not calls.record["numerical_admission"]
        early = late_failure in ("nan", "inf")
        assert len(sequence) == (2 if early else 5)
        assert all(p.startswith("boundary/") for p in sequence)
        with np.load(tmp_path / "boundary.npz", allow_pickle=False) as arrays:
            for kind in (
                ("actual", "tile0")
                if early
                else ("actual", "tile0", "tile1", "tile2", "tile3")
            ):
                assert f"{kind}_9__output" in arrays
                key = f"capture_{kind}_9__router/input"
                assert key in arrays and arrays[key].dtype == np.uint16
            if late_failure == "health":
                assert not arrays["tile3_9__health"][-1]
            if late_failure == "schema":
                assert arrays["capture_tile3_9__router/input"].shape[0] == 31
            if late_failure in ("nan", "inf"):
                assert arrays["tile0_9__health"].all()
                assert not np.isfinite(
                    arrays["capture_tile0_9__router/input"].view(window.BF16)[0, 0]
                )
                assert "boundary_capture_complete" not in calls.record
    finally:
        calls.journal.close()


def test_budgeted_call_uses_db591_allocations(tmp_path, monkeypatch):
    calls = make_calls(tmp_path)
    pins = admission.registered_programs()
    invoked = []

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return SimpleNamespace(**pins[self.name]["compiled_memory"])

        def __call__(self, *args):
            invoked.append(self.name)
            return np.zeros(1)

    calls.programs = {name: Program(name) for name in admission.PROGRAMS}
    monkeypatch.setattr(
        worker, "capture_resident_buffers", lambda *a, **kw: fake_memory()
    )
    monkeypatch.setattr(
        worker,
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
    try:
        calls.call("boundary/candidate", "candidate", (), preserve=lambda r: None)
        assert invoked == ["candidate"]
        assert (
            calls.record["call_evidence"][0]["budget"]["devices"][0][
                "resident_code_bytes"
            ]
            == 67_827_200
        )
    finally:
        calls.journal.close()


@pytest.mark.parametrize("change", ["profile", "protocol", "scope", "mesh"])
def test_diagnostic_continuation_refuses_before_wk_on_wrong_identity(tmp_path, change):
    calls = make_calls(tmp_path)
    record = calls.record
    record.update(
        profile=admission.PROFILE,
        protocol=boundary.PROTOCOL,
        compile_only=False,
        diagnostic_only=True,
        reference_scope=boundary.REFERENCE_SCOPE,
        physical_device_ids=boundary.original_receipt()["physical_device_ids"],
        programs={n: {} for n in admission.PROGRAMS},
    )
    if change == "profile":
        record["profile"] = worker.admission.PROFILE
    elif change == "protocol":
        record["protocol"] = window.PROTOCOL
    elif change == "scope":
        record["diagnostic_only"] = False
    else:
        record["physical_device_ids"] = []
    with pytest.raises(ValueError):
        worker.execute_numerical(
            args=SimpleNamespace(output_dir=tmp_path),
            record=record,
            mesh=None,
            config=None,
            weights=None,
            local_slots={0: 0, 8: 1, 16: 2, 24: 3},
            consensus=lambda ok: ok,
            compiled=(object(),) * 4,
            journal=calls.journal,
            boundary_diagnostic=True,
        )
    assert record["call_evidence"] == []
    assert (tmp_path / "runner.json").exists()
    assert (
        record["compile_journal_sha256"]
        == sha256((tmp_path / "compile_journal.jsonl").read_bytes()).hexdigest()
    )
