"""Real BudgetedCalls/saving lifecycle, fixture model/math/memory only."""

import json
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_budgeted_calls as budgeted
from scripts.greenfield import ws32_prefill_frontier_worker as worker
from scripts.greenfield.ws32_prefill_frontier import capture_cache_evidence
from scripts.greenfield.ws32_prefill_frontier_state import INDEX_LAYERS


def setup(tmp_path, monkeypatch, fail=None):
    slots = {9: 0, 13: 1, 25: 2, 29: 3}
    events = []
    calls = budgeted.BudgetedCalls(
        root=tmp_path, record=dict(jax_process_index=3), local_slots=slots,
        consensus=lambda ok: events.append(("vote", ok)) or ok,
        journal=NS(phase=lambda *a, **k: None), budgeter=worker.memory_budget,
    )
    config = NS(context_capacity=8192, logical_page_size=512, local_rows_per_page=64,
                kv_cache_shape=(78, 16, 512, 640), index_cache_shape=(21, 16, 512, 128),
                full_index_slots=INDEX_LAYERS, exact_dsa=False, strategy_nd_dense=False,
                host_main_rope_table=True, geometry=NS(hidden_size=6144, dsa_top_k=2048, vocab_size=10000))
    allocated = []
    def fresh(*args, prompt_length):
        assert prompt_length == 8155
        state = NS(position=0, branch=len(allocated))
        allocated.append(state)
        return state
    monkeypatch.setattr(worker, "make_ws32_batched_prefill_state", fresh)
    monkeypatch.setattr(worker, "_independent_caches", lambda a, b: a is not b)
    def inputs(mesh, ids, state, weights, wk, rope, **options):
        assert options == dict(mlp_window=True, physical_rows=128)
        assert np.array_equal(ids, np.arange(state.position, state.position+len(ids)))
        return (np.pad(ids, (0, 128-len(ids))), np.asarray(len(ids), np.int32), state, weights, wk, rope)
    monkeypatch.setattr(worker, "graph_inputs", inputs)
    class Program:
        def memory_analysis(self):
            return NS(**{key: 0 for key in budgeted.MEMORY_FIELDS})
        def __call__(self, ids, count, state, *args):
            events.append(("dispatch", int(count), state.position, state.branch))
            assert ids.shape == (128,) and (ids[int(count):] == 0).all()
            if fail == "dispatch":
                raise RuntimeError("fixture dispatch failure")
            return NS(state=NS(position=state.position+int(count), branch=state.branch), next_token=np.asarray([-1]))
    calls.programs = {worker.GRAPH: Program()}
    def memory(*args, **kwargs):
        return dict(schema_version="ws32_prefill_resident_buffers_v1", includes_all_live_arrays=True,
                    devices=[dict(device_id=d, process_index=3, platform="tpu",
                                  buffers=[], accounted_resident_bytes=0,
                                  memory_stats=dict(bytes_in_use=0, peak_bytes_in_use=0, bytes_limit=33_014_398_976)) for d in slots])
    monkeypatch.setattr(budgeted, "capture_resident_buffers", memory)
    def after(*args):
        if fail == "post_memory":
            raise ValueError("fixture post memory failure")
        return [dict(device_id=d, process_index=3, platform="tpu", bytes_in_use=0,
                     peak_bytes_in_use=0, bytes_limit=33_014_398_976) for d in slots]
    monkeypatch.setattr(budgeted, "capture_identified_device_memory", after)
    def capture(state, *, next_token, frontier, caches, **kwargs):
        assert state.position == frontier
        arrays, owners = {}, {}
        for slot in slots.values():
            owner = dict(device_id=next(d for d, s in slots.items() if s == slot), slot=slot,
                         metadata_sha256={"position": str(frontier)}, caches={})
            if caches:
                for name in ("kv", "index", "repair"):
                    value = np.zeros((1, 1, 64, 128), ml_dtypes.bfloat16)
                    if frontier:
                        value[0, 0, 0, 0] = state.branch+1  # Legitimate diagnostic DIFFERENCE.
                    rows, evidence = capture_cache_evidence(value, slot=slot,
                        block_table=np.asarray([[0]], np.int32), layer_ids=(0,), initial=frontier == 0)
                    owner["caches"][name] = evidence
                    if frontier:
                        arrays[f"slot{slot}_{name}_rows"] = rows
            owners[str(slot)] = owner
        valid = not (fail == "capture_health" and frontier == 128)
        return arrays, dict(owners=owners, valid=valid, errors=[] if valid else ["health"])
    monkeypatch.setattr(worker, "capture_state", capture)
    return calls, config, events


def execute(calls, config):
    worker.execute_first_window(calls, mesh=None, config=config,
                                prompt_tokens=np.arange(8155, dtype=np.int32), weights=None, wk=None, rope=None)


def test_fixed_five_calls_and_differences_are_diagnostic(tmp_path, monkeypatch):
    calls, config, events = setup(tmp_path, monkeypatch)
    execute(calls, config)
    assert [e for e in events if e[0] == "dispatch"] == [
        ("dispatch", 128, 0, 0), ("dispatch", 32, 0, 1), ("dispatch", 32, 32, 1),
        ("dispatch", 32, 64, 1), ("dispatch", 32, 96, 1)]
    assert len(calls.record["call_evidence"]) == 5
    assert all(e["completed"] and e["budget"]["estimate_fits"] for e in calls.record["call_evidence"])
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert not report["owners"]["0"]["caches"]["kv"]["bytes_equal"]
    assert calls.record["first_window"]["complete"]
    assert not report["numerical_promotion"] and not report["performance_claim"]
    assert len(list(tmp_path.glob("*.npz"))) == 7


@pytest.mark.parametrize("failure", ["capture_health", "post_memory", "dispatch"])
def test_completed_output_preserved_before_refusal_no_successor(tmp_path, monkeypatch, failure):
    calls, config, events = setup(tmp_path, monkeypatch, failure)
    with pytest.raises((ValueError, RuntimeError)):
        execute(calls, config)
    assert len([e for e in events if e[0] == "dispatch"]) == 1
    assert (tmp_path / "wide_final.npz").exists() == (failure != "dispatch")
    assert events[-1] == ("vote", False)
    assert not (tmp_path / "comparison.json").exists()


def test_wrong_budgeter_refuses_before_allocating_or_dispatching(tmp_path, monkeypatch):
    calls, config, events = setup(tmp_path, monkeypatch)
    calls.budgeter = lambda *a, **k: dict(estimate_fits=True)
    with pytest.raises(ValueError, match="budget"):
        execute(calls, config)
    assert not any(e[0] == "dispatch" for e in events)


def test_cache_alias_refuses():
    shard = NS(device=NS(id=9), data=NS(unsafe_buffer_pointer=lambda: 123))
    array = NS(addressable_shards=[shard])
    state = NS(decoder=NS(kv_cache_local=array, index_cache_local=array), repaired_index_local=array)
    with pytest.raises(ValueError, match="alias"):
        worker._independent_caches(state, state)
