"""User prompt boundaries and actual native host loop with CPU fake model math."""

import gzip
import json
import os
from pathlib import Path
import runpy
import stat
from types import SimpleNamespace

import pytest

from glm_tpu import user_request as request

REPO = Path(__file__).resolve().parents[2]


def example():
    return request.from_token_ids(
        [1, 2, 3], request_id="example/1", seed=42, max_new_tokens=2
    )


@pytest.mark.parametrize(
    "change",
    [
        "digest",
        "id",
        "bool_token",
        "negative",
        "float_token",
        "max_new",
        "capacity",
        "sampling",
        "seed",
        "benchmark",
        "extra",
    ],
)
def test_changed_or_invalid_request_refused(change):
    value = example()
    if change == "digest":
        value["request_sha256"] = "a" * 64
    elif change == "id":
        value["request_id"] = "bad id"
    elif change == "bool_token":
        value["prompt_ids"][0] = True
    elif change == "negative":
        value["prompt_ids"][0] = -1
    elif change == "float_token":
        value["prompt_ids"][0] = 1.0
    elif change == "max_new":
        value["max_new_tokens"] = 0
    elif change == "capacity":
        value["context_capacity"] = 262144
    elif change == "sampling":
        value["temperature"] = 0.7
    elif change == "seed":
        value["seed"] = 2**64
    elif change == "benchmark":
        value["benchmark"] = True
    elif change == "extra":
        value["gold"] = "must not become a benchmark"
    with pytest.raises(ValueError):
        request.validate(value)


def test_valid_request_roundtrip_and_full_budget_no_truncation():
    value = example()
    request.validate(json.loads(request.canonical(value)))
    with pytest.raises(ValueError):
        request.from_token_ids(
            [1] * 3073, request_id="long", seed=0, max_new_tokens=request.MAX_NEW
        )
    assert (
        len(
            request.from_token_ids(
                [1] * 3072, request_id="fits", seed=0, max_new_tokens=request.MAX_NEW
            )["prompt_ids"]
        )
        == 3072
    )


def test_frozen_template_is_used_without_shortening_messages():
    messages = [{"role": "user", "content": "Explain the sky."}]
    calls = []
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda value, **kwargs: calls.append((value, kwargs))
        or [1, 2, 3]
    )
    template = (REPO / "reference/hf-repo/chat_template.jinja").read_text()
    value = request.from_messages(
        messages,
        tokenizer=tokenizer,
        chat_template=template,
        request_id="chat",
        seed=0,
        max_new_tokens=32,
    )
    request.validate(value)
    assert calls[0][0] is messages
    assert calls[0][1]["reasoning_effort"] == "max"
    assert calls[0][1]["chat_template"] == template
    with pytest.raises(ValueError):
        request.from_messages(
            messages,
            tokenizer=tokenizer,
            chat_template=template + "changed",
            request_id="chat",
            seed=0,
            max_new_tokens=32,
        )


def test_private_output_is_exclusive_outside_repo_and_owner_only(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    path = tmp_path / "request.json"
    report = request.write_private(path, example(), repo=repo)
    assert report["model_executions"] == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    request.validate(json.loads(request.read_bounded(path, request.PAYLOAD_CAP)))
    with pytest.raises(FileExistsError):
        request.write_private(path, example(), repo=repo)
    with pytest.raises(ValueError):
        request.write_private(repo / "private.json", example(), repo=repo)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError):
        request.read_bounded(alias, request.PAYLOAD_CAP)
    with pytest.raises(ValueError):
        request.read_bounded(path, 1)


def setup_worker(monkeypatch, tmp_path):
    from scripts.greenfield.ws32_native_benchmark_requests import RequestStore

    fixture = runpy.run_path(
        str(REPO / "tests/greenfield/runtime/test_ws32_native_benchmark_runtime.py")
    )
    runtime, calls, _, admissions, _ = fixture["build_runtime"](monkeypatch)
    runtime.raw_config.context_capacity = runtime.decode_config.context_capacity = (
        request.CAPACITY
    )
    runtime.raw_config.geometry.vocab_size = request.VOCAB
    store = RequestStore(tmp_path / "sessions.rank0", 0)
    authorize = runtime.authorize

    def preserve(phase, roots, state):
        authorize(phase, roots, state)
        store.preserve_memory(phase, {"CPU_FIXTURE_NOT_HBM_EVIDENCE": True})

    runtime.authorize = preserve
    observations = SimpleNamespace(
        begin=lambda *args: None,
        finish=lambda session: {
            "CPU_FIXTURE_NOT_DSA_EVIDENCE": True,
            "instrumented_decode_indices": [],
        },
    )
    return (
        SimpleNamespace(runtime=runtime, record={"cold_load_compile_seconds": 1.0}),
        store,
        observations,
        calls,
        admissions,
    )


def test_user_inference_uses_actual_host_runtime_without_benchmark_registry(
    monkeypatch, tmp_path
):
    from scripts.release.ws32_user_request import execute_user_request
    from scripts.greenfield import ws32_native_benchmark_protocol as benchmark

    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark/scoring called for user prompt")

    monkeypatch.setattr(benchmark, "benchmark_registry", forbidden)
    monkeypatch.setattr(benchmark, "validate", forbidden)
    loaded, store, obs, calls, admissions = setup_worker(monkeypatch, tmp_path)
    result = execute_user_request(
        loaded=loaded,
        request=example(),
        store=store,
        tokenizer=SimpleNamespace(decode=lambda ids, **kw: str(ids)),
        observations=obs,
        deadline=loaded.runtime.clock() + 60,
    )
    assert result["complete"] and result["finish_reason"] == "length"
    assert result["first_token_delivered_before_decode"]
    assert result["generated_tokens"] == 2 and result["benchmark"] is False
    assert result["quality_score"] is None and not result["protected_result_sealed"]
    assert admissions == ["before_cache", "cache_ready", "prefill_done"]
    assert [row[0] for row in calls] == ["prefill", "decode"]
    assert loaded.runtime.active is None
    answer = gzip.decompress((store.current / "answer.txt.gz").read_bytes()).decode()
    assert answer == "[7, 10]"
    saved = json.loads(gzip.decompress((store.current / "result.json.gz").read_bytes()))
    assert saved == result


def test_failed_delivery_keeps_partial_and_never_retries(monkeypatch, tmp_path):
    from scripts.release.ws32_user_request import execute_user_request

    loaded, store, obs, calls, _ = setup_worker(monkeypatch, tmp_path)

    def broken(*args):
        raise RuntimeError("fake decode failed")

    loaded.runtime.decode_compiled = broken
    with pytest.raises(RuntimeError):
        execute_user_request(
            loaded=loaded,
            request=example(),
            store=store,
            tokenizer=None,
            observations=obs,
            deadline=loaded.runtime.clock() + 60,
        )
    assert len((store.current / "tokens.jsonl").read_text().splitlines()) == 1
    assert (store.current / "failure.json").exists()
    assert loaded.runtime.active.failed
    assert [row[0] for row in calls] == ["prefill"]


@pytest.mark.parametrize("problem", ["deadline", "observability", "digest"])
def test_preflight_failure_cannot_run_model(monkeypatch, tmp_path, problem):
    from scripts.release.ws32_user_request import execute_user_request

    loaded, store, obs, calls, _ = setup_worker(monkeypatch, tmp_path)
    value = example()
    if problem == "digest":
        value["request_sha256"] = "bad"
    with pytest.raises(RuntimeError):
        execute_user_request(
            loaded=loaded,
            request=value,
            store=store,
            tokenizer=None,
            observations=None if problem == "observability" else obs,
            deadline=0 if problem == "deadline" else loaded.runtime.clock() + 60,
        )
    assert calls == [] and store.current is None


@pytest.mark.skipif(
    not os.environ.get("GLM_RELEASE_LOCAL_TOKENIZER"),
    reason="explicit existing local tokenizer required; never download",
)
def test_actual_local_tokenizer_cli(tmp_path, capsys):
    """Real pinned tokenizer and CLI, no model weights, backend or cloud calls."""
    from glm_tpu.cli import main

    messages = tmp_path / "messages.json"
    messages.write_text(
        json.dumps(
            [
                {
                    "role": "user",
                    "content": "Explain why the sky is blue in one sentence.",
                }
            ]
        )
    )
    output = tmp_path / "request.json"
    code = main(
        [
            "prepare-request",
            "--messages",
            str(messages),
            "--output",
            str(output),
            "--repo",
            str(REPO),
            "--tokenizer-root",
            os.environ["GLM_RELEASE_LOCAL_TOKENIZER"],
            "--request-id",
            "release-example",
            "--max-new-tokens",
            "256",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    value = json.loads(output.read_bytes())
    request.validate(value)
    assert value["max_new_tokens"] == 256 and value["thinking"] == "on/max"
    assert report["prompt_tokens"] > 1 and report["model_executions"] == 0
    assert "sky" not in json.dumps(report)
