"""Tests of :mod:`glm_tpu.executor.fleet`: the SSH command discovery and the fan-out, offline."""

from types import SimpleNamespace

import pytest

from glm_tpu.executor import multihost_executor as launch
from glm_tpu.executor.fleet import remote_all, ssh_commands
from tests.fixtures.site import example_site


def test_ssh_unknown_host_never_dispatches(monkeypatch, tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "gcloud":
            return SimpleNamespace(
                stdout="\n".join(
                    f"/usr/bin/ssh -o HostKeyAlias=host{i} -o StrictHostKeyChecking=no example -- true"
                    for i in range(8)
                )
            )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(launch.subprocess, "run", run)
    fleet = example_site(tmp_path).fleet
    with pytest.raises(ValueError, match="unknown"):
        ssh_commands(fleet)
    assert len(calls) == 2
    assert calls[0][5:7] == [fleet.tpu_name, "--zone=" + fleet.zone]  # the site's TPU VM and zone
    assert calls[1][-2:] == ["-f", str(fleet.known_hosts)]


def test_ssh_failure_is_not_retried(monkeypatch, tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(launch.subprocess, "run", run)
    commands = [["ssh", str(i), "--", "true"] for i in range(8)]
    with pytest.raises(ValueError):
        remote_all(commands, "command", tmp_path, "run")
    assert len(calls) == 8 and {argv[1] for argv in calls} == {str(i) for i in range(8)}
