"""Tests of :mod:`glm_tpu.worker.tpu_worker`."""

import json
import sys
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

    engine = SimpleNamespace(runner=SimpleNamespace(phase=lambda name, fn: fn()), generate_concurrent=generate)
    with installed_site(example_site(tmp_path / "site")):  # the answer writer's tokenizer location
        reports, aggregate = run_concurrent(engine, pending, tmp_path, 0, 100)
    assert len(calls) == 2 and calls[1] == pending  # Disposable warmup, then ONE measured batch.
    assert all(len(v["prompt_ids"]) == 128 and v["max_new_tokens"] == 2 for v in calls[0])
    assert len(reports) == 8 and aggregate["batch_size"] == 8
    assert all(r["stop_cause"] == "eos" and r["output_budget_tokens"] == 3 for r in reports)
    for lane in range(8):
        records = [
            json.loads(line) for line in (tmp_path / f"item{lane:03d}" / "tokens.jsonl").read_text().splitlines()
        ]
        assert len(records) == 2 and [record["batch_round"] for record in records] == [0, 1]
        assert all(record["request_id"] == f"lane{lane}" for record in records)
        assert (tmp_path / f"item{lane:03d}" / "answer.txt").read_text() == str([10 + lane] * 2)


def test_resident_reuses_runtime_and_has_explicit_stop(monkeypatch, tmp_path):
    import io
    from glm_tpu.engine import request

    value = request.from_token_ids([7], request_id="fixture", max_new_tokens=2, context_capacity=32768)
    engine = SimpleNamespace(runner=SimpleNamespace(capacity=32768, phase=lambda name, action: action()))
    seen = []

    def generate(actual, pending, value, root, rank, deadline, **kwargs):
        seen.append((actual, kwargs["warmup"], root.name))
        return [dict(emitted=2, token_sha256="c" * 64)]

    monkeypatch.setattr(worker, "run_queued", generate)
    commands = json.dumps(dict(sequence=1, request=value)) + "\n" + json.dumps(dict(stop=True)) + "\n"
    worker.resident_loop(engine, rows()[0], tmp_path, 0, 3600, stream=io.StringIO(commands))
    assert seen == [(engine, False, "resident-0001")]
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
    engine = SimpleNamespace(runner=SimpleNamespace(capacity=32768, phase=lambda name, action: action()))
    with pytest.raises(ValueError):
        worker.resident_loop(engine, rows()[0], tmp_path, 0, 3600, stream=io.StringIO(line))
    assert not (tmp_path / "resident-0001").exists()


def test_worker_default_off_before_model_import(monkeypatch, tmp_path):
    monkeypatch.delenv(protocol.WORKER_ENV_FLAG, raising=False)
    with pytest.raises(ValueError, match="protected"):
        worker.preflight(SimpleNamespace(output=tmp_path, code_hash="a" * 40, wall_seconds=100))


# `python -m glm_tpu.worker.tpu_worker --help`, byte for byte: argparse wraps to the terminal width (pinned:
# COLUMNS=100) and its layout differs between Python versions (the literal is Python 3.12's). The description is
# the module docstring.
HELP = """\
usage: tpu_worker.py [-h] --output OUTPUT --code-hash CODE_HASH --source-manifest-sha256
                     SOURCE_MANIFEST_SHA256 --request-file-sha256 REQUEST_FILE_SHA256
                     --site-sha256 SITE_SHA256 --topology-rebinding-sha256
                     TOPOLOGY_REBINDING_SHA256 --coordinator-address COORDINATOR_ADDRESS
                     --wall-seconds WALL_SECONDS [--preflight-only] [--keep-loaded]

The per-host ordinary-inference worker; source staging and fleet leases belong to its controller.

options:
  -h, --help            show this help message and exit
  --output OUTPUT
  --code-hash CODE_HASH
  --source-manifest-sha256 SOURCE_MANIFEST_SHA256
  --request-file-sha256 REQUEST_FILE_SHA256
  --site-sha256 SITE_SHA256
  --topology-rebinding-sha256 TOPOLOGY_REBINDING_SHA256
  --coordinator-address COORDINATOR_ADDRESS
  --wall-seconds WALL_SECONDS
  --preflight-only
  --keep-loaded
"""


def test_help_text_is_unchanged(monkeypatch, capsys):
    # In this process, as `python -m` runs it (sys.argv[0] is the module's file): a child process whose argv names
    # the worker would make a concurrent cpu32 test or heavy gate see a live TPU run (tools.equivalence budget).
    # `python -m glm_tpu.worker.tpu_worker --help` itself: tests/engine/test_resident_protocol.py.
    assert sys.version_info[:2] == (3, 12), "the literal is Python 3.12 argparse output"
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setattr(sys, "argv", [worker.__file__, "--help"])
    with pytest.raises(SystemExit) as exited:
        worker.main()
    assert exited.value.code == 0
    assert capsys.readouterr().out == HELP
