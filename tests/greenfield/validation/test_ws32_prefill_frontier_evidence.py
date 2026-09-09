"""Production-shaped originals through actual producer/consumer; fixture compute."""

from copy import deepcopy
from hashlib import sha256
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
from glm_tpu.greenfield.validation.ws32_prefill_admission import short_numerical_identity
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal

ROOT = Path(__file__).resolve().parents[3]


def test_actual_entry_capture_and_independent_original_replay(tmp_path, monkeypatch):
    fixture = runpy.run_path(str(ROOT / "tests/greenfield/validation/test_ws32_prefill_frontier_worker.py"))
    meta_fixture = runpy.run_path(str(ROOT / "tests/greenfield/validation/test_ws32_prefill_frontier_state.py"))["metadata"]
    calls, config, events = fixture["setup"](tmp_path, monkeypatch)
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
    graphs = {n: {"passed": True, "stablehlo_sha256": sha256(n.encode()).hexdigest(),
                  "optimized_hlo_sha256": sha256((n+" optimized").encode()).hexdigest()}
              for n in ("exact_materialize", "exact_promote", "prefill_chunk")}
    memory = {n: dict(vars(calls.programs[worker.GRAPH].memory_analysis())) for n in graphs}
    identity = dict(jax_process_index=3, code_hash="a"*40, launch_process_id=0,
                    hostname="fixture-host", prompt_ids_sha256="b"*64,
                    checkpoint_manifest_sha256="c"*64, checkpoint_success_sha256="d"*64,
                    mesh_sha256="e"*64, topology_fleet_sha256="f"*64,
                    checkpoint_verified_device_slots=list(slots.values()),
                    local_device_slots=list(slots.values()), load_seconds=1.0,
                    compile_seconds={n: 1.0 for n in graphs})
    journal_identity = dict(**short_numerical_identity(profile=FROZEN_FIRST_WINDOW_PROFILE), compile_only=False,
                           **{k: identity[k] for k in ("code_hash", "launch_process_id", "hostname", "prompt_ids_sha256",
                               "checkpoint_manifest_sha256", "checkpoint_success_sha256")})
    journal_path = tmp_path / "numerical_journal.rank0.jsonl"
    calls.journal = Ws32NumericalJournal(journal_path, journal_identity)
    calls.journal.phase("runtime_initialize_started")
    calls.journal.phase("checkpoint_verify_started", jax_process_index=3, local_device_ids=list(slots),
                        mesh_sha256=identity["mesh_sha256"], topology_fleet_sha256=identity["topology_fleet_sha256"])
    calls.journal.phase("load_started", checkpoint_verified_device_slots=list(slots.values()))
    calls.journal.phase("load_completed", seconds=1.0, local_device_slots=list(slots.values()))
    for graph in graphs:
        calls.journal.begin(graph)
        calls.journal.compiled(graph, seconds=1.0, memory=memory[graph], device_memory=[])
        calls.journal.inspect(graph, graph, graph+" optimized", lambda: graphs[graph])
    entry.execute(args=args, repo=ROOT,
        jax=NS(process_index=lambda: 3, local_devices=lambda: [NS(id=d) for d in slots]),
        mesh=None, physical_mesh=NS(flattened_device_ids=list(slots)+list(range(100, 128))),
        config=config, prompt_tokens=np.arange(8155, dtype=np.int32), compiled=calls.programs,
        graphs=graphs, compiled_memory=memory,
        identity={k:v for k,v in identity.items() if k not in ("jax_process_index", "code_hash", "launch_process_id")}, journal=calls.journal,
        weights=None, wk=None, rope=None, consensus=calls.consensus)
    record = json.loads(args.output.read_text())
    root = tmp_path / "first_window.rank0"
    envelope = evidence.replay_envelope(root, record, journal_path, local_slots=slots, expected_identity=identity)
    assert envelope["completed_phases"] == 29
    assert calls.journal._stream.closed
    journal_raw = journal_path.read_bytes()
    for mutation in ("second_runner", "missing_close", "deadline", "phase_missing", "failed_phase",
                     "raw_sha", "journal_identity", "runtime_owner", "memory", "unclosed_line"):
        changed = deepcopy(record)
        rows = [json.loads(line) for line in journal_raw.splitlines()]
        if mutation == "missing_close":
            changed.pop("diagnostic_closed_monotonic_seconds")
        if mutation == "deadline":
            changed["diagnostic_closed_monotonic_seconds"] = changed["diagnostic_started_monotonic_seconds"] + 301
        if mutation == "phase_missing":
            rows.pop(-2)
        if mutation == "failed_phase":
            rows[-1]["passed"] = False
        if mutation == "raw_sha":
            rows[7]["stablehlo_sha256"] = "0"*64
        if mutation == "journal_identity":
            rows[0]["identity"]["prompt_ids_sha256"] = "0"*64
        if mutation == "runtime_owner":
            rows[2]["local_device_ids"][0] = 99
        if mutation == "memory":
            rows[6]["compiled_memory"]["temp_size_in_bytes"] = 1
        (root / "runner.json").write_text(json.dumps(record if mutation == "second_runner" else changed))
        if mutation == "second_runner":
            changed["status"] = "DIAGNOSTIC_FAILED"
        raw = "".join(json.dumps(r)+"\n" for r in rows).encode()
        journal_path.write_bytes(raw.rstrip(b"\n") if mutation == "unclosed_line" else raw)
        with pytest.raises(ValueError):
            evidence.replay_envelope(root, changed, journal_path, local_slots=slots, expected_identity=identity)
    journal_path.write_bytes(journal_raw)
    (root / "runner.json").write_text(json.dumps(record))
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
