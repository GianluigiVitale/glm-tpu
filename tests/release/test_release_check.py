"""Release check orchestration is fail-closed without touching real services."""
import json
import subprocess

import pytest

from tools import check_release


@pytest.mark.parametrize("failure", [OSError("fixture"), subprocess.TimeoutExpired(["fixture"], 180)])
def test_launch_error_or_timeout_fails_without_continuing(monkeypatch, capsys, failure):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
        assert "GLM_RELEASE_LOCAL_TOKENIZER" not in kwargs["env"]
        raise failure

    monkeypatch.setenv("GLM_RELEASE_LOCAL_TOKENIZER", "/unused")
    monkeypatch.setattr(check_release.subprocess, "run", run)
    assert check_release.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["steps"][0]["error"] == type(failure).__name__
    assert len(calls) == 1


def test_success_never_grants_hardware_or_merge_admission(monkeypatch, capsys):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "2 passed\n", "")

    monkeypatch.setattr(check_release.subprocess, "run", run)
    assert check_release.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["hardware_execution"] is False
    assert report["full_dependency_install"] is False
    assert report["benchmark_completion"] is False
    assert report["release_merge_authorized"] is False
