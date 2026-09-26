"""Tests of :mod:`glm_tpu.entrypoints.cli.prepare`: ``glm-tpu prepare-request`` (CPU only, no tokenizer)."""

import json
from pathlib import Path

import pytest

from glm_tpu.engine import request
from glm_tpu.entrypoints.cli.main import main

ARGV = [
    "prepare-request",
    "--messages",
    "private/messages.json",
    "--output",
    "private/prepared.json",
    "--repo",
    "checkout",
    "--tokenizer-root",
    "model",
    "--request-id",
    "request-7",
    "--max-new-tokens",
    "64",
]
GIVEN = dict(
    messages_path=Path("private/messages.json"),
    output=Path("private/prepared.json"),
    repo=Path("checkout"),
    tokenizer_root=Path("model"),
    request_id="request-7",
    max_new_tokens=64,
)


@pytest.mark.parametrize(
    ("profile", "extra"),
    [
        ([], {}),
        (["--profile", "ordinary-greedy-8k"], {}),
        (["--profile", "ordinary-greedy-128k"], {"context_capacity": 166912}),
    ],
    ids=["default", "8k", "128k"],
)
def test_the_profile_sets_the_capacity_and_the_report_is_printed(monkeypatch, capsys, profile, extra):
    """8K passes no capacity (``prepare_file``'s default); 128K passes 166912 slots; the report goes to stdout."""
    seen = []
    report = {"request_id": "request-7", "prompt_tokens": 12, "b": [1, 2]}
    monkeypatch.setattr(request, "prepare_file", lambda **kwargs: seen.append(kwargs) or report)
    capsys.readouterr()
    assert main([*ARGV, *profile]) == 0
    assert seen == [{**GIVEN, **extra}]
    assert capsys.readouterr() == (json.dumps(report, indent=2, sort_keys=True) + "\n", "")


@pytest.mark.parametrize("error", [ValueError, OSError, FileNotFoundError, ImportError])
def test_a_refusal_prints_only_the_error_type(monkeypatch, capsys, error):
    """Private input, tokenizer exception text and file contents never reach the output."""

    def refuse(**kwargs):
        raise error("private chat text /private/path")

    monkeypatch.setattr(request, "prepare_file", refuse)
    capsys.readouterr()
    assert main(ARGV) == 1
    refusal = json.dumps(dict(error=error.__name__, status="request preparation refused")) + "\n"
    assert capsys.readouterr() == ("", refusal)


def test_other_errors_propagate(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("not a refusal")

    monkeypatch.setattr(request, "prepare_file", fail)
    with pytest.raises(RuntimeError, match="not a refusal"):
        main(ARGV)
