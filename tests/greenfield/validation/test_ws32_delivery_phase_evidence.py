"""Actual preparation writers and reader; fixture compute/counters, not TPU proof."""
from copy import deepcopy
from hashlib import sha256
import json
import math

import pytest

from scripts.greenfield import ws32_delivery_phase_evidence as evidence
from scripts.greenfield import ws32_delivery_wk as wk
from scripts.greenfield import ws32_delivery_decode as decode
from scripts.greenfield import run_short_decoder_ws32 as worker
from tests.greenfield.validation.test_ws32_delivery_phase_loading import originals, stage
from tests.greenfield.validation.test_ws32_delivery_decode import setup, overlay_fixture
from tests.greenfield.validation.test_ws32_prefill_fleet_memory import fixture as memory_fixture


@pytest.fixture
def completed(tmp_path, monkeypatch, originals):
    # Use the real42-call append-once writer plus real two-call protected path.
    # The existing fixtures replace model compute/compiler/counters explicitly.
    kwargs, _, _ = stage(tmp_path, monkeypatch, originals)
    kwargs["root"] = tmp_path / "delivery_wk.rank0"
    identity = dict(kwargs["identity"], launch_process_id=0,
                    checkpoint_manifest_sha256="b" * 64, checkpoint_success_sha256="c" * 64)
    kwargs["identity"] = identity
    def preserve(value, entry, *, graph, slots):
        raw = value.tobytes()
        entry["output"] = [dict(device_id=d, slot=s, shape=list(value.shape),
            dtype=str(value.dtype), bytes=len(raw), sha256=sha256(raw).hexdigest(), finite=True)
            for d, s in slots.items()]
    monkeypatch.setattr(wk, "preserve_output", preserve)
    try:
        wr = wk.prepare(**kwargs)
    finally:
        kwargs["journal"].close()
    parent = deepcopy(memory_fixture()["records"][0])
    # The fixture writer's slots are deliberately not the production mesh.
    # Parent topology joins are separately enforced before this helper.
    parent["local_device_slots"] = [dict(device_id=d, device_slot=s)
        for d, s in kwargs["local_slots"].items()]
    parent.update(identity, batched_prefill_profile=evidence.runtime.PROFILE)
    for row in parent["batched_prefill_memory"]:
        row["peak_bytes_in_use"] = 26_000_000_000
    decodedir = tmp_path / "exact_fixture"
    decodedir.mkdir()
    options, calls, _, _, _ = setup(decodedir, monkeypatch)
    calls.root.rename(tmp_path / "delivery_decode.rank0")
    calls.root = tmp_path / "delivery_decode.rank0"
    calls.record.update(identity)
    overlay = overlay_fixture(calls)
    # Replace fixture byte counts with the actual verified local tensor contract.
    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype, _, _ in evidence._LOCAL_CONTRACT]
    for row in overlay.records.values():
        for tensor, size in zip(row["tensors"].values(), sizes, strict=True):
            tensor["byte_count"] = size
    census = parent["prefill_execution"]["memory_admission"]["census"]
    monkeypatch.setattr(decode, "capture_resident_buffers", lambda *a, **k: deepcopy(census))
    after = [dict(device_id=r["device_id"], platform="tpu", process_index=r["process_index"],
                  **r["memory_stats"]) for r in census["devices"]]
    monkeypatch.setattr(decode, "capture_identified_device_memory", lambda *a: deepcopy(after))
    try:
        decode.overlay_preflight(overlay, calls, {})
        decode.overlay_completed(calls)
        worker._materialize_exact_decode(**options, protected_calls=calls)
        dr = decode.finish(calls)
    finally:
        calls.journal.close()
    parent.update(delivery_wk_phase=wr, delivery_decode_preparation=dr,
                  load_seconds=2.0,
                  delivery_phase_timings=dict(base_checkpoint_load_seconds=1.0,
                      wk_prepare_wall_seconds=wr["phase_wall_seconds"],
                      decode_overlay_load_seconds=1.0, decode_materialization_wall_seconds=1.0),
                  strategy_nd_dense_overlay={key: dr["overlay_memory"][key] for key in (
                      "manifest_sha256", "manifest_file_sha256", "success_file_sha256")})
    parent["graphs"], parent["compiled_memory_analysis"], parent["compile_seconds"] = {}, {}, {}
    for record in (wr, dr):
        for name, program in record["programs"].items():
            parent["graphs"][name] = program["admission"]
            parent["compiled_memory_analysis"][name] = program["compiled_memory"]
            parent["compile_seconds"][name] = program["compile_seconds"]
    return dict(root=tmp_path, rank=0, parent=parent,
                full_index_layers=tuple(kwargs["owner"].raw_config.full_index_slots))


def test_actual_phase_originals_and_memory_boundaries(completed):
    result = evidence.validate_rank(**completed)
    assert result["call_count"] == 44 and len(result["wk_output_hashes"]) == 42
    assert result["numerical_value_replay"] is False
    # Reuse originals for mutations, never rerun42 unchanged model fixtures.
    parent = completed["parent"]
    old = parent["checkpoint_manifest_sha256"]
    parent["checkpoint_manifest_sha256"] = "d" * 64
    with pytest.raises(ValueError): evidence.validate_rank(**completed)
    parent["checkpoint_manifest_sha256"] = old
    old = parent["batched_prefill_memory"][0]["peak_bytes_in_use"]
    parent["batched_prefill_memory"][0]["peak_bytes_in_use"] += 1
    with pytest.raises(ValueError, match="peak regressed"): evidence.validate_rank(**completed)
    parent["batched_prefill_memory"][0]["peak_bytes_in_use"] = old
    path = completed["root"] / "delivery_wk.rank0/call_records/call041.json"
    raw = path.read_bytes()
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError): evidence.validate_rank(**completed)
    path.write_bytes(raw)
    assert evidence.validate_rank(**completed) == result


def test_overlay_refuses_lost_reserve_and_wrong_payload(completed):
    parent = completed["parent"]
    slots = {r["device_id"]: r["device_slot"] for r in parent["local_device_slots"]}
    original = parent["delivery_decode_preparation"]
    for mutation in ("payload", "census", "owner", "limit", "reserve", "manifest"):
        record = deepcopy(original)
        memory = record["overlay_memory"]
        if mutation == "payload": memory["devices"][0]["payload_bytes"] -= 1
        if mutation == "census": memory["census"]["includes_all_live_arrays"] = False
        if mutation == "owner": memory["census"]["devices"][0]["device_id"] = 99
        if mutation == "limit": memory["census"]["devices"][0]["memory_stats"]["bytes_limit"] += 1
        if mutation == "reserve": memory["reserve_bytes"] = 0
        if mutation == "manifest": memory["manifest_sha256"] = "e" * 64
        with pytest.raises(ValueError): evidence.overlay_replay(record, parent, slots)


def test_fleet_requires_all_eight_and_equal_wk(monkeypatch):
    def validate(**kwargs):
        return dict(rank=kwargs["rank"], wk_output_hashes=kwargs["parent"]["hashes"])
    monkeypatch.setattr(evidence, "validate_rank", validate)
    records = [dict(hashes=["a"] * 42) for _ in range(8)]
    assert evidence.validate_fleet(root=None, records=records, full_index_layers=())["owner_count"] == 32
    with pytest.raises(ValueError): evidence.validate_fleet(root=None, records=records[:7], full_index_layers=())
    records[-1]["hashes"][41] = "b"
    with pytest.raises(ValueError): evidence.validate_fleet(root=None, records=records, full_index_layers=())
