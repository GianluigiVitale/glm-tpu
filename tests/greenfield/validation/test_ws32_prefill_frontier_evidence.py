"""Production-shaped originals through actual producer/consumer; fixture compute."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_prefill_frontier_evidence as evidence
from scripts.greenfield import ws32_prefill_frontier_entry as entry
from scripts.greenfield import ws32_prefill_frontier_state as state_capture
from scripts.greenfield import ws32_prefill_frontier_worker as worker
from glm_tpu.greenfield.validation.ws32_prefill_admission import FROZEN_FIRST_WINDOW_PROFILE

ROOT = Path(__file__).resolve().parents[3]


def test_actual_entry_capture_and_independent_original_replay(tmp_path, monkeypatch):
    fixture = runpy.run_path(str(ROOT / "tests/greenfield/validation/test_ws32_prefill_frontier_worker.py"))
    meta_fixture = runpy.run_path(str(ROOT / "tests/greenfield/validation/test_ws32_prefill_frontier_state.py"))["metadata"]
    calls, config, events = fixture["setup"](tmp_path, monkeypatch)
    calls.journal.close = lambda: None
    slots = calls.local_slots
    def actual_capture(state, *, frontier, **kwargs):
        def distributed(value, shape, cache=False):
            index = [slice(None)] * len(shape)
            if cache:
                index[2] = slice(0, 64)
            return NS(shape=shape, dtype=value.dtype,
                addressable_shards=[NS(device=NS(id=d, platform="tpu", process_index=3),
                    index=tuple(index), data=value) for d in slots])
        values = meta_fixture(frontier)
        fields = {n: distributed(v, v.shape) for n, v in values.items()}
        def cache(shape):
            value = np.broadcast_to(np.asarray(0, ml_dtypes.bfloat16), (*shape[:2], 64, shape[3]))
            if frontier:
                value = value.copy()
                value[0, 0, 0, 0] = state.branch+1
            return distributed(value, shape, True)
        decoder = NS(**{n: v for n, v in fields.items() if n not in ("prompt_length", "finished", "next_token")},
                     kv_cache_local=cache(config.kv_cache_shape), index_cache_local=cache(config.index_cache_shape))
        actual = NS(decoder=decoder, repaired_index_local=cache(config.index_cache_shape),
                    prompt_length=fields["prompt_length"], finished=fields["finished"])
        kwargs["next_token"] = fields["next_token"] if frontier else None
        return state_capture.capture_state(actual, frontier=frontier, **kwargs)
    monkeypatch.setattr(worker, "capture_state", actual_capture)
    # This test exercises lifecycle/originals, not source/HLO compiler admission.
    monkeypatch.setattr(entry, "require_acquired_model_source", lambda *a, **k: None)
    monkeypatch.setattr(entry, "validate_short_compiled_memory", lambda *a, **k: None)
    args = NS(batched_prefill_profile=FROZEN_FIRST_WINDOW_PROFILE, expected_code_hash="a"*40,
              process_id=0, output=tmp_path / "runner.rank0.json")
    graphs = {n: {"passed": True} for n in ("exact_materialize", "exact_promote", "prefill_chunk")}
    memory = {n: dict(vars(calls.programs[worker.GRAPH].memory_analysis())) for n in graphs}
    entry.execute(args=args, repo=ROOT,
        jax=NS(process_index=lambda: 3, local_devices=lambda: [NS(id=d) for d in slots]),
        mesh=None, physical_mesh=NS(flattened_device_ids=list(slots)+list(range(100, 128))),
        config=config, prompt_tokens=np.arange(8155, dtype=np.int32), compiled=calls.programs,
        graphs=graphs, compiled_memory=memory, identity={}, journal=calls.journal,
        weights=None, wk=None, rope=None, consensus=calls.consensus)
    record = json.loads(args.output.read_text())
    root = tmp_path / "first_window.rank0"
    identity = dict(jax_process_index=3, code_hash="a"*40, launch_process_id=0)
    replay = evidence.replay_host(root, record, local_slots=slots, expected_identity=identity)
    assert replay["comparison"]["owners"]["0"]["caches"]["kv"]["earliest_differing_writer"] == 0
    assert not replay["comparison"]["numerical_promotion"]
    # Reuse original producer once; mutate several independent consumer edges.
    for mutation in ("call_count", "owner", "budget", "array_digest", "comparison"):
        changed = deepcopy(record)
        if mutation == "call_count":
            changed["call_evidence"].pop()
        if mutation == "owner":
            changed["local_slots"]["9"] = 4
        if mutation == "budget":
            changed["call_evidence"][0]["budget"]["required_reserve_bytes"] = 1
        if mutation == "array_digest":
            changed["first_window"]["originals"]["wide_initial"]["npz_sha256"] = "0"*64
        if mutation == "comparison":
            changed["first_window"]["comparison"]["owners"]["0"]["caches"]["kv"]["bytes_equal"] = True
        with pytest.raises(ValueError):
            evidence.replay_host(root, changed, local_slots=slots, expected_identity=identity)


def replicas():
    hosts = []
    # Deliberately interleaved features across hosts, not oneexpert perhost.
    for rank in range(8):
        slots = [rank, rank+8, rank+16, rank+24]
        states = {label: dict(owners={str(s): dict(
            metadata_sha256={"position": str(evidence.FRONTIERS[label])},
            caches={name: {"cache": {"whole_cache_sha256": f"{label}/{s//4}/{name}"}}
                    for name in evidence.SHAPES} if evidence.FRONTIERS[label] in (0,128) else {}) for s in slots})
                  for label in evidence.LABELS}
        comparison = dict(owners={str(s): dict(caches={name: dict(earliest_differing_writer=6 if s//4 == 1 else None)
                                                       for name in evidence.SHAPES}) for s in slots})
        hosts.append(dict(states=states, comparison=comparison))
    return hosts


@pytest.mark.parametrize("mutation", [None, "missing", "duplicate", "metadata", "cache"])
def test_cross_host_replica_join(mutation):
    hosts = replicas()
    if mutation == "missing":
        hosts.pop()
    if mutation == "duplicate":
        hosts[1] = deepcopy(hosts[0])
    if mutation == "metadata":
        hosts[1]["states"]["narrow_32"]["owners"]["1"]["metadata_sha256"]["position"] = "128"
    if mutation == "cache":
        hosts[1]["states"]["wide_final"]["owners"]["1"]["caches"]["kv"]["cache"]["whole_cache_sha256"] = "bad"
    if mutation:
        with pytest.raises(ValueError):
            evidence.replay_replicas(hosts)
    else:
        result = evidence.replay_replicas(hosts)
        assert result["physical_owners"] == 32
        assert result["earliest_differing_writer"]["kv"] == 6
