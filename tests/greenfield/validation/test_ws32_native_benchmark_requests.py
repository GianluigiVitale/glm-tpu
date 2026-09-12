"""Full registered request/control path with explicit fake model, never TPU proof."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import ws32_native_benchmark_protocol as protocol
from scripts.greenfield import ws32_native_benchmark_requests as requests
from scripts.greenfield import ws32_native_benchmark_entry as entry
from tests.greenfield.runtime.test_ws32_native_benchmark_runtime import build_runtime


def capsule():
    rows = []
    for name, count in protocol.COUNTS.items():
        for i in range(count):
            ident = ("gpqa" if name == "gpqa_diamond" else "aime") + f"_{i}"
            rows.append(dict(dataset=name, item=dict(item_id=ident, gold="A", n_choices=4),
                request_id=f"{name}/{ident}/sample0", seed=20260912+i, prompt_ids=[1,2,3],
                prompt_ids_sha256=sha256(np.array([1,2,3], dtype="<i4").tobytes()).hexdigest()))
    payload = dict(schema=protocol.SCHEMA, requests=rows)
    return payload, protocol.protocol(payload, {"tokenizer.json":"a"*64, "tokenizer_config.json":"b"*64})


@pytest.mark.parametrize("change", ["missing", "seed", "tokens", "hash", "cap", "sample", "target", "order", "duplicate", "budget"])
def test_protocol_refuses_subset_or_changed_identity_and_limits(change):
    payload, plan = capsule()
    if change == "missing": payload["requests"].pop()
    elif change == "seed": payload["requests"][0]["seed"] += 1
    elif change == "tokens": payload["requests"][0]["prompt_ids"][0] = True
    elif change == "hash": payload["requests"][0]["prompt_ids_sha256"] = "0"*64
    elif change == "order": payload["requests"].reverse()
    elif change == "duplicate": payload["requests"][1] = deepcopy(payload["requests"][0])
    elif change == "cap": plan["max_new_tokens"] = 1024
    elif change == "sample": plan["samples_per_item"] = 2
    elif change == "target": plan["targets"] = dict(gpqa_diamond=0.8, aime_2026=0.992)
    else: plan["tranche_wall_seconds"] = 1
    # Even if an altered payload is rehashed, policy/order checks still apply.
    plan["payload_sha256"] = sha256(protocol.canonical(payload)).hexdigest()
    with pytest.raises(ValueError): protocol.validate(payload, plan)


def test_original_registry_and_final_answer_only_no_unapproved_math_judge():
    registry = protocol.benchmark_registry()
    payload, _ = capsule()
    tok = NS(decode=lambda *a, **k: r"trying \boxed{B}</think>Final answer: A")
    result = requests.score_answer(payload["requests"][0], [1], tok, registry)
    assert result["correct"] is True and result["extracted"] == "A"
    tok.decode = lambda *a, **k: r"still thinking \boxed{A}"
    result = requests.score_answer(payload["requests"][0], [1], tok, registry)
    assert result["correct"] is False and result["extracted"] is None
    result = requests.score_answer(payload["requests"][-1], [1], tok, registry)
    assert result["correct"] is None and result["score_status"] == "AWAITING_APPROVED_JUDGE"
    def terminal_decode(ids, **kwargs):
        assert ids == [5,6] and kwargs["skip_special_tokens"] is False
        return "reasoning</think>\nA"
    tok.decode = terminal_decode
    assert requests.score_answer(payload["requests"][0], [5,6,protocol.EOS[0]], tok, registry)["correct"] is True
    low, high = protocol.wilson(180, 198)
    assert 0.85 < low < 180/198 < high < 0.95


def setup(tmp_path, monkeypatch, rank=0):
    runtime, calls, allocations, admissions, _ = build_runtime(monkeypatch)
    runtime.raw_config.geometry.vocab_size = protocol.VOCAB
    runtime.raw_config.context_capacity = runtime.decode_config.context_capacity = protocol.CAPACITY
    old = runtime.decode_compiled
    def decode(*args):
        value = old(*args)
        return value._replace(next_token=np.array([protocol.EOS[0]], np.int32))
    runtime.decode_compiled = decode
    store = requests.RequestStore(tmp_path / f"sessions.rank{rank}", rank)
    old_admit = runtime.authorize
    def authorize(stage, roots, state):
        old_admit(stage, roots, state)
        # Fixture memory only, the real callback has its own actual census tests.
        store.preserve_memory(stage, dict(phase=stage, fixture_not_TPU=True))
    runtime.authorize = authorize
    tok = NS(decode=lambda *a, **k: "reasoning</think>Final answer: A")
    return runtime, store, tok, calls, allocations


@pytest.mark.parametrize("rank", [0, 7])
def test_all228_requests_actual_host_loop_first_delivery_resume_distinct_caches(tmp_path, monkeypatch, rank):
    runtime, store, tok, calls, allocations = setup(tmp_path, monkeypatch, rank)
    payload, plan = capsule()
    rows = requests.execute_requests(loaded=NS(runtime=runtime), payload=payload, plan=plan,
        store=store, tokenizer=tok, deadline=runtime.clock()+60)
    assert len(rows) == 228 and runtime.active is None
    assert len(allocations) == 228 and len({id(x) for x in allocations}) == 228
    assert len([c for c in calls if c[0] == "decode"]) == 228
    for index, row in enumerate(rows):
        assert row["complete"] and row["first_token_delivered_before_decode"] and row["live_session_resume"]
        assert row["generated_tokens"] == 2 and row["finish_reason"] == "eos"
        directory = store.root / f"item{index:03d}"
        stored = json.loads(gzip.decompress((directory / "result.json.gz").read_bytes()))
        assert len(stored["decode_seconds"]) == 1 and "decode_seconds" not in row
        assert len(list(directory.glob("*.json.gz"))) == 4
        if rank == 0:
            events = [json.loads(line) for line in (directory / "tokens.jsonl").read_text().splitlines()]
            assert [r["token_id"] for r in events] == [7, protocol.EOS[0]]
            assert row["ttft_seconds"] >= 0 and row["delivered_request_seconds"] >= row["ttft_seconds"]
            assert row["correct"] is (True if index < 198 else None)
        else:
            assert not (directory / "tokens.jsonl").exists()
            assert row["ttft_seconds"] is None and "correct" not in row


@pytest.mark.parametrize("failure", ["deadline", "decode", "sink", "answer"])
def test_failure_preserves_partial_answer_never_starts_successor(tmp_path, monkeypatch, failure):
    runtime, store, tok, calls, allocations = setup(tmp_path, monkeypatch)
    payload, plan = capsule()
    if failure == "decode":
        runtime.decode_compiled = lambda *a: (_ for _ in ()).throw(ValueError("fixture dispatch"))
    if failure == "sink":
        monkeypatch.setattr(requests, "TOKEN_CAP", 1)
    if failure == "answer":
        tok.decode = lambda *a, **k: (_ for _ in ()).throw(ValueError("fixture tokenizer"))
    ticks = iter([0, 2])
    options = dict(clock=lambda: next(ticks), deadline=1) if failure == "deadline" else dict(deadline=runtime.clock()+60)
    with pytest.raises((ValueError, RuntimeError)):
        requests.execute_requests(loaded=NS(runtime=runtime), payload=payload, plan=plan,
            store=store, tokenizer=tok, **options)
    assert len(allocations) == 1 and not (store.root / "item001").exists()
    assert json.loads((store.root / "item000/failure.json").read_bytes())["complete"] is False
    tokens = (store.root / "item000/tokens.jsonl").read_text().splitlines()
    assert len(tokens) == (0 if failure == "sink" else 2 if failure == "answer" else 1)
    assert not (store.root / "item000/result.json.gz").exists()


def test_storage_limit_refuses_without_overwriting_original(tmp_path, monkeypatch):
    store = requests.RequestStore(tmp_path / "rank", 0)
    row = capsule()[0]["requests"][0]
    store.begin(0, row)
    store.save("original", b"old", 10)
    with pytest.raises(FileExistsError): store.save("original", b"new", 10)
    monkeypatch.setattr(requests, "RANK_CAP", store.used)
    with pytest.raises(ValueError): store.save("next", b"new", 10)
    assert (store.current / "original").read_bytes() == b"old"
    assert not (store.current / "next").exists()


def test_native_entry_default_off_no_backend(monkeypatch):
    from scripts.greenfield import run_short_decoder_ws32 as original
    monkeypatch.setattr(original, "_require_clean_code", lambda p: None)
    monkeypatch.delenv("GLM_GREENFIELD_NATIVE_BENCHMARK", raising=False)
    with pytest.raises(ValueError, match="default-off"):
        entry.preflight(NS(expected_code_hash="a"*40))


def test_entry_connects_original_initializer_loader_request_writer_and_complete_record(tmp_path, monkeypatch):
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield import ws32_native_benchmark_worker as worker
    from scripts.greenfield import ws32_native_benchmark_observability as observations
    from transformers import AutoTokenizer
    payload, plan = capsule()
    args = NS(output=tmp_path / "runner.rank0.json", process_id=0,
              expected_code_hash="a"*40, protocol_sha256="b"*64)
    monkeypatch.setattr(entry, "parse_args", lambda argv: args)
    monkeypatch.setattr(entry, "preflight", lambda value:(payload, plan))
    monkeypatch.setattr(original, "_initialize_runtime", lambda value:
        (NS(process_index=lambda:3), "mesh", "physical", "topology", "fleet"))
    monkeypatch.setattr(original, "_batched_fleet_all", lambda value: value)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: NS())
    seen = []
    def load(**kwargs):
        assert kwargs["mesh"] == "mesh" and kwargs["physical_mesh"] == "physical"
        assert kwargs["root"] == tmp_path / "native.rank0"
        seen.append(kwargs)
        return NS(record=dict(cold_load_compile_seconds=1.0))
    monkeypatch.setattr(worker, "load_runtime", load)
    observer = object()
    monkeypatch.setattr(observations, "NativeObservability", lambda loaded, store: observer)
    def execute(**kwargs):
        assert kwargs["observations"] is observer
        assert kwargs["payload"] is payload and kwargs["plan"] is plan
        assert kwargs["store"].preserve_memory == seen[0]["preserve_memory"]
        row = dict(index=227, request_id="last", generated_tokens=2)
        kwargs["after_request"](row)
        return [row]*228
    monkeypatch.setattr(requests, "execute_requests", execute)
    assert entry.main([]) == 0
    row = json.loads(args.output.read_bytes())
    assert row["complete"] and row["completed_requests"] == 228
    assert row["jax_process_index"] == 3 and row["launch_process_id"] == 0
    assert not row["benchmark_quality_proven"] and not row["protected_result_sealed"]
    assert int(row["owner"]["start_ticks"]) > 0
