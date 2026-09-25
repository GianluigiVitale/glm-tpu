"""Tests of :mod:`glm_tpu.worker.tpu_worker`."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from glm_tpu.engine import request
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.engine.outputs import TokenEvent
from glm_tpu.worker import tpu_worker as worker
from glm_tpu.worker.tpu_worker import run_concurrent
from tests.executor.test_multihost_executor import rows
from tests.fixtures.site import example_site, installed_site


def test_worker_uses_one_batch_and_separate_deliveries(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens)),
    )
    pending = [
        request.from_token_ids([7] * 130, request_id=f"lane{i}", max_new_tokens=3, context_capacity=32768)
        for i in range(8)
    ]
    calls = []

    def generate(values, *, deliver, deadline):
        calls.append(values)
        for rnd in range(2):
            for lane, item in enumerate(values):
                deliver(lane, TokenEvent(item["request_id"], rnd, 10 + lane, "eos" if rnd else None), rnd)
        return [(np.array([10 + i, 10 + i]), dict(emitted=2, finish_reason="eos")) for i in range(8)], dict(
            batch_size=8
        )

    runtime = SimpleNamespace(phase=lambda name, fn: fn(), generate_concurrent=generate)
    with installed_site(example_site(tmp_path / "site")):  # the answer writer's tokenizer location
        reports, aggregate = run_concurrent(runtime, pending, tmp_path, 0, 100)
    assert len(calls) == 2 and calls[1] == pending  # Disposable warmup, then ONE measured batch.
    assert all(len(v["prompt_ids"]) == 128 and v["max_new_tokens"] == 2 for v in calls[0])
    assert len(reports) == 8 and aggregate["batch_size"] == 8
    assert all(r["stop_cause"] == "eos" and r["output_budget_tokens"] == 3 for r in reports)
    for lane in range(8):
        rows = [json.loads(line) for line in (tmp_path / f"item{lane:03d}" / "tokens.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and [row["batch_round"] for row in rows] == [0, 1]
        assert all(row["request_id"] == f"lane{lane}" for row in rows)
        assert (tmp_path / f"item{lane:03d}" / "answer.txt").read_text() == str([10 + lane] * 2)


def test_resident_reuses_runtime_and_has_explicit_stop(monkeypatch, tmp_path):
    import io
    from glm_tpu.engine import request

    value = request.from_token_ids([7], request_id="fixture", max_new_tokens=2, context_capacity=32768)
    runtime = SimpleNamespace(capacity=32768, phase=lambda name, action: action())
    seen = []

    def generate(actual, pending, value, root, rank, deadline, **kwargs):
        seen.append((actual, kwargs["warmup"], root.name))
        return [dict(emitted=2, token_sha256="c" * 64)]

    monkeypatch.setattr(worker, "run_queued", generate)
    commands = json.dumps(dict(sequence=1, request=value)) + "\n" + json.dumps(dict(stop=True)) + "\n"
    worker.resident_loop(runtime, rows()[0], tmp_path, 0, 3600, stream=io.StringIO(commands))
    assert seen == [(runtime, False, "resident-0001")]
    assert json.loads((tmp_path / "resident-ready.json").read_text()) == {"sequence": 1}
    assert (
        json.loads((tmp_path / "resident-0001/runner.rank0.json").read_text())["request_sha256"]
        == value["request_sha256"]
    )


@pytest.mark.parametrize("case", ["disconnect", "sequence", "capacity"])
def test_resident_refuses_ambiguous_or_incompatible_input(tmp_path, case):
    import io
    from glm_tpu.engine import request

    value = request.from_token_ids(
        [7], request_id="fixture", max_new_tokens=2, context_capacity=8192 if case == "capacity" else 32768
    )
    line = (
        "" if case == "disconnect" else json.dumps(dict(sequence=2 if case == "sequence" else 1, request=value)) + "\n"
    )
    runtime = SimpleNamespace(capacity=32768, phase=lambda name, action: action())
    with pytest.raises(ValueError):
        worker.resident_loop(runtime, rows()[0], tmp_path, 0, 3600, stream=io.StringIO(line))
    assert not (tmp_path / "resident-0001").exists()


def test_worker_default_off_before_model_import(monkeypatch, tmp_path):
    monkeypatch.delenv(protocol.WORKER_ENV_FLAG, raising=False)
    with pytest.raises(ValueError, match="protected"):
        worker.preflight(SimpleNamespace(output=tmp_path, code_hash="a" * 40, wall_seconds=100))
