"""Tests of :mod:`glm_tpu.entrypoints.cli.topology`: ``glm-tpu topology capture`` hands the loaded site to the capture
job (faked here; the job itself: tests/executor/test_topology_job.py) and ``topology bind`` writes the binding of a
capture run; each prints the report as JSON, or the refusal as JSON on standard error with exit 1."""

from __future__ import annotations

import json
from pathlib import Path


from glm_tpu.distributed.topology import CAPTURE_NAMES
from glm_tpu.entrypoints.cli.main import main
from glm_tpu.executor import topology_job
from tests.fixtures import topology
from tests.fixtures.site import example_mapping, example_site, write_example_site

PIN = "f" * 40


def run(capsys, argv):
    capsys.readouterr()
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def test_capture_runs_the_job_on_the_site_file(tmp_path: Path, monkeypatch, capsys):
    site_file = write_example_site(tmp_path / "site.toml")
    calls = []

    def capture(site, *, repo, wall_seconds):
        calls.append((site.resolved_sha256(), repo, wall_seconds))
        return dict(passed=True, run="r")

    monkeypatch.setattr(topology_job, "capture_topology", capture)
    code, out, err = run(capsys, ["topology", "capture", "--site", str(site_file), "--wall-seconds", "120"])
    assert (code, err) == (0, "") and json.loads(out) == dict(passed=True, run="r")
    assert calls == [(example_site(tmp_path).resolved_sha256(), None, 120)]


def test_capture_refusal_is_json_on_stderr(tmp_path: Path, monkeypatch, capsys):
    site_file = write_example_site(tmp_path / "site.toml")

    def capture(site, *, repo, wall_seconds):
        raise BlockingIOError(11, "Resource temporarily unavailable")

    monkeypatch.setattr(topology_job, "capture_topology", capture)
    code, out, err = run(capsys, ["topology", "capture", "--site", str(site_file)])
    assert (code, out) == (1, "")
    assert json.loads(err) == dict(
        error="BlockingIOError",
        message="[Errno 11] Resource temporarily unavailable",
        status="topology capture refused",
    )


def capture_run(tmp_path: Path) -> Path:
    """A finished capture run: the eight captures and a clean terminal record."""
    run_dir = tmp_path / "runs" / "topology_capture_20260101T000000000000Z"
    run_dir.mkdir(parents=True, mode=0o700)
    site = example_site(tmp_path, paths=dict(run_root=str(tmp_path / "runs")))
    topology.capture_all(run_dir, site, PIN)
    terminal = dict(
        codes=[0] * 8,
        failed=False,
        all_hosts_idle=True,
        stalled_ssh_clients=[],
        divergent_records=[],
        uncollected_ranks=[],
        missing=[],
        collect_error=None,
        code_hash=PIN,
        hosts=topology.HOSTS,
    )
    (run_dir / topology_job.TERMINAL_FILE).write_text(json.dumps(terminal))
    return run_dir


def test_bind_prints_the_site_values(tmp_path: Path, capsys):
    run_dir = capture_run(tmp_path)
    output = tmp_path / "binding"
    code, out, err = run(capsys, ["topology", "bind", str(run_dir), "--output", str(output), "--note", "test"])
    assert (code, err) == (0, "")
    report = json.loads(out)
    assert report == topology_job.bind_topology(run_dir, tmp_path / "again", note="test") | dict(
        binding_dir=str(output), capture_root=str(output / "captures")
    )
    assert sorted(p.name for p in (output / "captures").iterdir()) == sorted(CAPTURE_NAMES)
    # The printed values are what a site file's [topology] table takes.
    mapping = example_mapping(
        tmp_path,
        topology={
            k: report[k]
            for k in (
                "binding_dir",
                "binding_sha256",
                "capture_root",
                "topology_sha256",
                "topology_fleet_sha256",
                "mesh_sha256",
                "slice_name",
            )
        },
    )
    assert write_example_site(tmp_path / "bound.toml", mapping).is_file()


def test_bind_refuses_an_existing_output(tmp_path: Path, capsys):
    run_dir = capture_run(tmp_path)
    output = tmp_path / "binding"
    output.mkdir()
    code, out, err = run(capsys, ["topology", "bind", str(run_dir), "--output", str(output)])
    assert (code, out) == (1, "")
    assert json.loads(err)["error"] == "FileExistsError" and json.loads(err)["status"] == "topology bind refused"
    assert list(output.iterdir()) == []


def test_bind_refuses_a_mismatched_expected_topology(tmp_path: Path, capsys):
    run_dir = capture_run(tmp_path)
    argv = ["topology", "bind", str(run_dir), "--output", str(tmp_path / "b"), "--expected-topology-sha256", "0" * 64]
    code, out, err = run(capsys, argv)
    assert (code, out) == (1, "")
    assert json.loads(err) == dict(
        error="ValueError",
        message="the captured topology differs from the expected one",
        status="topology bind refused",
    )
    assert not (tmp_path / "b").exists()
