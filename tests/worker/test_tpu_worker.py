from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np

from glm_tpu.engine import request
from glm_tpu.engine.outputs import TokenEvent
from glm_tpu.worker.tpu_worker import run_concurrent
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
