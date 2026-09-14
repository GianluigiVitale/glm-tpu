"""Actual user producer/observer/replay, with explicit fake math/HBM/XPlanes.

These tests do not exercise cold HLO admission or physical trace parsing and
cannot establish TPU performance, ownership, task quality or a success seal.
"""

from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS
from zipfile import BadZipFile

import ml_dtypes
import numpy as np
import pytest

from glm_tpu import user_request
from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DsaObservation,
    Ws32ObservedDecodeStepResult,
)
from scripts.greenfield import ws32_native_benchmark_memory as memory
from scripts.greenfield import ws32_native_benchmark_observability as obs
from scripts.greenfield.ws32_native_benchmark_requests import RequestStore
from scripts.release.ws32_user_request import execute_user_request
from scripts.release import ws32_user_result as result
from tests.greenfield.runtime.test_ws32_native_benchmark_runtime import build_runtime
from tests.greenfield.validation.test_ws32_native_benchmark_memory import record


def produce_rank(root, monkeypatch, request, rank):
    runtime, calls, *_ = build_runtime(monkeypatch)
    runtime.raw_config.context_capacity = runtime.decode_config.context_capacity = (
        user_request.CAPACITY
    )
    runtime.raw_config.geometry.vocab_size = user_request.VOCAB
    runtime.decode_config.full_index_slots = tuple(range(21))
    runtime.decode_config.geometry.num_layers = 78
    runtime.decode_config.geometry.dsa_indexer_head_dim = 128
    runtime.decode_config.packed_cache_width = 640
    store = RequestStore(root / f"sessions.rank{rank}", rank)
    census = record()
    slots = [dict(device_id=rank * 4 + i, slot=rank * 4 + i) for i in range(4)]
    census.update(local_slots=slots, process_index=rank)
    for row in census["census"]["devices"]:
        row.update(device_id=rank * 4 + row["device_id"], process_index=rank)

    def authorize(phase, roots, state):
        value = deepcopy(census)
        value.update(phase=phase, cache_present=phase != "before_cache")
        value["budgets"] = memory.budgets(value)
        store.preserve_memory(phase, value)

    runtime.authorize = authorize
    positions = np.full((21, 1, 2048), -1, np.int32)
    scores = np.full((21, 1, 2048), -np.inf, np.float32)
    positions[:, :, :4] = np.arange(4)
    scores[:, :, :4] = 1
    dsa = Ws32DsaObservation(
        np.arange(21, dtype=np.int32), positions, np.full((21, 1), 4, np.int32), scores
    )
    compiled = dict(
        observer=lambda *args: Ws32ObservedDecodeStepResult(
            runtime.decode_compiled(*args), dsa
        ),
        cache_probe=lambda state: NS(
            position=state.position - 1,
            kv_rows=np.ones((78, 640), ml_dtypes.bfloat16),
            index_rows=np.ones((21, 128), ml_dtypes.bfloat16),
            contract_valid=np.array([True]),
        ),
    )
    counters = [
        dict(
            device_id=s["device_id"],
            process_index=rank,
            platform="tpu",
            bytes_in_use=350,
            peak_bytes_in_use=360,
            bytes_limit=memory.DEVICE_LIMIT,
        )
        for s in slots
    ]
    monkeypatch.setattr(obs, "capture_identified_device_memory", lambda _: counters)
    monkeypatch.setattr(obs.jax, "device_get", lambda x: x)

    def start(path):
        path = Path(path)
        path.mkdir()
        (path / "host.xplane.pb").write_bytes(b"CPU fixture NOT an actual XPlane")

    monkeypatch.setattr(obs, "start_device_trace", start)
    monkeypatch.setattr(obs.jax.profiler, "stop_trace", lambda: None)
    parent = dict(
        local_slots=slots,
        jax_process_index=rank,
        cold_load_compile_seconds=1.0,
        programs={
            name: dict(compiled_memory=analysis)
            for name, analysis in census["compiled_memory"].items()
        },
    )
    loaded = NS(runtime=runtime, record=parent, compiled=compiled)
    tokenizer = NS(decode=lambda ids, **kw: str(ids))
    row = execute_user_request(
        loaded=loaded,
        request=request,
        store=store,
        tokenizer=tokenizer,
        observations=obs.NativeObservability(loaded, store),
        deadline=runtime.clock() + 120,
    )
    assert len(calls) == request["max_new_tokens"]  # one prefill, no extra model calls
    outer = dict(
        profile=result.WORKER_PROFILE,
        artifact_kind=result.WORKER_PROFILE,
        code_hash="a" * 40,
        request_file_sha256=sha256(user_request.canonical(request)).hexdigest(),
        request_sha256=request["request_sha256"],
        launch_process_id=rank,
        jax_process_index=rank,
        complete=True,
        benchmark=False,
        protected_result_sealed=False,
        generated_tokens=row["generated_tokens"],
        token_ids_sha256=row["token_ids_sha256"],
        finish_reason=row["finish_reason"],
        cold_load_compile_seconds=1.0,
        worker_wall_seconds=3.0,
    )
    (root / f"runner.rank{rank}.json").write_bytes(user_request.canonical(outer))
    return parent, tokenizer


def fleet(tmp_path, monkeypatch, count=4):
    request = user_request.from_token_ids(
        [1, 2, 3], request_id="user-example", seed=42, max_new_tokens=count
    )
    parents = []
    for rank in range(8):
        parent, tokenizer = produce_rank(tmp_path, monkeypatch, request, rank)
        parents.append(parent)
    return dict(
        root=tmp_path,
        request=request,
        request_file_sha256=sha256(user_request.canonical(request)).hexdigest(),
        pin="a" * 40,
        parents=parents,
        tokenizer=tokenizer,
        full_index_layers=tuple(range(21)),
    )


@pytest.mark.parametrize("count", [1, 2, 4])
def test_real_user_producer_replay_excludes_instrumentation_and_has_no_quality_claim(
    tmp_path, monkeypatch, count
):
    from scripts.greenfield import ws32_native_benchmark_protocol as protocol

    def forbidden(*a, **kw):
        raise AssertionError("user replay invoked benchmark registry")

    monkeypatch.setattr(protocol, "benchmark_registry", forbidden)
    args = fleet(tmp_path, monkeypatch, count)
    report = result.replay_user_request(**args)
    assert report["complete"] and report["generated_tokens"] == count
    assert not report["protected_result_sealed"] and not report["benchmark"]
    assert report["quality_score"] is None and not report["cold_admission_verified"]
    assert (
        not report["trace_physical_coverage_verified"]
        and not report["worker_ownership_verified"]
    )
    assert not report["durable_resume_verified"]
    assert report["ordinary_decode_samples"] == max(count - 2, 0)
    assert len(report["trace_originals"]) == (8 if count > 1 else 0)
    if count > 2:
        row = json.loads(
            gzip.decompress(
                (tmp_path / "sessions.rank0/item000/result.json.gz").read_bytes()
            )
        )
        assert report["ordinary_decode_seconds"] == sum(row["decode_seconds"][1:])
        assert report["decode_wall_tokens_per_second"] > 0
    else:
        assert (
            report["decode_wall_tokens_per_second"] is None
            and report["decode_p50_ms"] is None
        )


def test_tampered_rank_tokens_timings_scope_and_trace_refused(tmp_path, monkeypatch):
    args = fleet(tmp_path, monkeypatch)
    path = tmp_path / "sessions.rank7/item000/result.json.gz"
    original = path.read_bytes()
    mutations = [
        lambda r: r.update(token_ids_sha256="0" * 64),
        lambda r: r.update(rank=True),
        lambda r: r.update(complete=1),
        lambda r: r.update(quality_score=0.99),
        lambda r: r.update(correct=True),
        lambda r: r.update(live_session_resume=False),
        lambda r: r.update(ttft_seconds=0.1),
        lambda r: r.update(decode_seconds=[0.1]),
        lambda r: r["decode_seconds"].__setitem__(0, float("nan")),
        lambda r: r["phases"].update(input_transfer=True),
        lambda r: r.update(cold_load_compile_seconds=2),
        lambda r: r["observations"].update(traced_decode_indices=[]),
        lambda r: r["observations"].update(instrumented_decode_indices=[False]),
        lambda r: r["observations"].update(dsa_observed=False),
        lambda r: r["observations"]["trace"].update(
            path="native_trace.rank7/../native_trace.rank0/host.xplane.pb"
        ),
        lambda r: r["observations"]["trace"].update(actual_model_calls=True),
        lambda r: r["observations"]["trace"].update(sha256="0" * 64),
    ]
    for mutate in mutations:
        row = json.loads(gzip.decompress(original))
        mutate(row)
        path.write_bytes(gzip.compress(json.dumps(row).encode()))
        with pytest.raises(ValueError):
            result.replay_user_request(**args)
    path.write_bytes(original)
    assert result.replay_user_request(**args)["complete"]


def test_original_cache_answer_memory_and_worker_failures_refused(
    tmp_path, monkeypatch
):
    args = fleet(tmp_path, monkeypatch)
    item = tmp_path / "sessions.rank7/item000"
    cases = [
        (item / "cache.npz", b"invalid npz"),
        (item / "final_memory.json", b"[]"),
        (tmp_path / "sessions.rank0/item000/answer.txt.gz", gzip.compress(b"changed")),
        (tmp_path / "runner.rank7.json", b"{}"),
    ]
    for path, bad in cases:
        original = path.read_bytes()
        path.write_bytes(bad)
        with pytest.raises((ValueError, KeyError, BadZipFile)):
            result.replay_user_request(**args)
        path.write_bytes(original)
    (item / "failure.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="failure original"):
        result.replay_user_request(**args)


def test_raw_delivery_eos_precedence_and_partial_event_refusal(tmp_path):
    request = user_request.from_token_ids(
        [1], request_id="stop", seed=0, max_new_tokens=2
    )
    events = [
        dict(request_id="stop", index=0, token_id=5, finish_reason=None),
        dict(
            request_id="stop",
            index=1,
            token_id=user_request.EOS[0],
            finish_reason="eos",
        ),
    ]
    path = tmp_path / "tokens.jsonl"

    def write():
        path.write_bytes(b"".join(user_request.canonical(e) + b"\n" for e in events))

    write()
    assert result.delivered_tokens(tmp_path, request)[1] == "eos"
    events[-1]["finish_reason"] = "length"
    write()
    with pytest.raises(ValueError):
        result.delivered_tokens(tmp_path, request)
    events[-1]["finish_reason"] = "eos"
    write()
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="partial"):
        result.delivered_tokens(tmp_path, request)


def test_original_numerical_witnesses_are_recomputed_not_trusted(tmp_path, monkeypatch):
    args = fleet(tmp_path, monkeypatch)
    item = tmp_path / "sessions.rank7/item000"
    path = item / "dsa.npz"
    original = path.read_bytes()
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["selected_positions"][0, 0, 1] = arrays["selected_positions"][0, 0, 0]
    path.write_bytes(obs.npz_bytes(**arrays))
    # Result still says DSA passed; the original array contradicts that claim.
    with pytest.raises(ValueError, match="DSA"):
        result.replay_user_request(**args)
    path.write_bytes(original)
    path = item / "cache.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["contract_valid"][:] = False
    path.write_bytes(obs.npz_bytes(**arrays))
    with pytest.raises(ValueError, match="cache"):
        result.replay_user_request(**args)
