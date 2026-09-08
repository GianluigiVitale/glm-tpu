"""Staged worker boundary with fake executables; never starts a TPU backend."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import ws32_batched_prefill_runner as adapter


def workload(tmp_path, monkeypatch, *, failure=None):
    receipt = admission.short_acquisition(worker.REPO)
    args = SimpleNamespace(
        batched_prefill_profile=admission.SHORT_PROFILE,
        prefill_memory_reserve_bytes=admission.SHORT_RESERVE_BYTES,
        prefill_budget_seconds=admission.SHORT_BUDGET_SECONDS,
        expected_code_hash="a" * 40,
        process_id=0,
        output=tmp_path / "runner.rank0.json",
        **{
            f"expected_{graph}_{form}": sha
            for graph, pins in receipt["graphs"].items()
            for form, sha in pins.items()
        },
    )
    compiled = {}
    for graph in adapter.GRAPHS:
        fields = dict(receipt["fleet"][0]["compiled"][graph]["memory"])
        if failure == "compiled_memory" and graph == "prefill_chunk":
            fields["temp_size_in_bytes"] += 1
        analysis = SimpleNamespace(**fields)
        compiled[graph] = SimpleNamespace(memory_analysis=lambda a=analysis: a)
    stats = dict(bytes_in_use=1 << 30, peak_bytes_in_use=2 << 30, bytes_limit=4 << 30)
    if failure == "reserve":
        stats["peak_bytes_in_use"] = (
            stats["bytes_limit"] - admission.SHORT_RESERVE_BYTES + 1
        )
    devices = tuple(
        SimpleNamespace(
            id=i, process_index=3, platform="tpu", memory_stats=lambda: stats
        )
        for i in range(12, 16)
    )
    monkeypatch.setattr(jax, "process_index", lambda: 3)
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    calls, votes = [], []

    def vote(value):
        votes.append(value)
        return value and failure != "peer_preflight"

    monkeypatch.setattr(worker, "_batched_fleet_all", vote)
    monkeypatch.setattr(adapter, "validate_execution_record", lambda record, plan: None)

    def execute(*values, **kwargs):
        calls.append(values)
        assert kwargs["additional_resident_executables"] == {}
        assert kwargs["required_memory_reserve_bytes"] == admission.SHORT_RESERVE_BYTES
        assert kwargs["budget_seconds"] == admission.SHORT_BUDGET_SECONDS
        assert (tmp_path / "batched_prefill_preflight.rank0.json").exists()
        kwargs["memory_progress"]({"memory_admission": {"test": True}, "error": None})
        assert (tmp_path / "batched_prefill_memory.rank0.json").exists()
        if failure == "execution":
            raise RuntimeError("original model failure")
        token = np.asarray([123], np.int32)
        return (
            "decoder",
            token,
            {"first_token_ready": 124 if failure == "token" else 123},
        )

    monkeypatch.setattr(adapter, "execute_graph_pair", execute)
    values = dict(
        args=args,
        mesh="mesh",
        config="config",
        plan=admission.SHORT_PLAN,
        prompt_tokens=np.zeros(2034, np.int32),
        compiled=compiled,
        weights="weights",
        wk="wk",
        rope="rope",
    )
    return values, calls, votes


def test_actual_worker_boundary_publishes_before_dispatch_and_preserves_owners(
    tmp_path, monkeypatch
):
    values, calls, votes = workload(tmp_path, monkeypatch)
    decoder, token, execution, memory = worker._execute_batched_prefill(**values)
    assert decoder == "decoder" and int(token[0]) == execution["first_token_ready"]
    assert len(calls) == 1 and votes == [True, True]
    assert [(r["process_index"], r["device_id"]) for r in memory] == [
        (3, i) for i in range(12, 16)
    ]
    record = json.loads((tmp_path / "batched_prefill_complete.rank0.json").read_text())
    assert record["device_memory_after_prefill"] == memory
    assert not record["performance_claim"] and not record["numerical_claim"]
    assert not values["args"].output.exists()  # Phase record is not a completed run.


@pytest.mark.parametrize("failure", ["compiled_memory", "peer_preflight"])
def test_preflight_refusal_votes_before_any_adapter_dispatch(
    tmp_path, monkeypatch, failure
):
    values, calls, votes = workload(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match="numerical preflight"):
        worker._execute_batched_prefill(**values)
    assert not calls and votes == [failure != "compiled_memory"]
    record = json.loads((tmp_path / "batched_prefill_failure.rank0.json").read_text())
    assert record["exception_type"] == "RuntimeError"
    assert not (tmp_path / "batched_prefill_memory.rank0.json").exists()


@pytest.mark.parametrize("failure", ["execution", "reserve", "token"])
def test_failed_execution_retains_memory_and_original_failure(
    tmp_path, monkeypatch, failure
):
    values, calls, votes = workload(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError):
        worker._execute_batched_prefill(**values)
    assert len(calls) == 1
    assert (tmp_path / "batched_prefill_memory.rank0.json").exists()
    record = json.loads((tmp_path / "batched_prefill_failure.rank0.json").read_text())
    if failure == "execution":
        assert record["exception"] == "original model failure"
    else:
        assert votes == [True, False]
    assert not (tmp_path / "batched_prefill_complete.rank0.json").exists()


def test_real_worker_drops_all_compile_placeholders_before_adapter():
    source = Path(worker.__file__).read_text()
    boundary = source[
        source.index("elif batched_prefill:", source.index("prefill_execution: dict")) :
    ]
    assert boundary.index(
        "state = repaired_buffer = batched_state = None"
    ) < boundary.index("_execute_batched_prefill(")
    assert boundary.index("gc.collect()") < boundary.index("_execute_batched_prefill(")
    assert boundary.index("raw_prefill_weights = batched_wk = None") < boundary.index(
        "observer_jit ="
    )


def test_fleet_consensus_rejects_peer_and_bad_cardinality(monkeypatch):
    from jax.experimental import multihost_utils

    monkeypatch.setattr(
        multihost_utils, "process_allgather", lambda _: np.array([1] * 7 + [0])
    )
    assert not worker._batched_fleet_all(True)
    monkeypatch.setattr(
        multihost_utils, "process_allgather", lambda _: np.ones(4, np.int32)
    )
    with pytest.raises(ValueError, match="eight boolean"):
        worker._batched_fleet_all(True)
