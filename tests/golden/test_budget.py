"""D25 live-run detection is fail-closed about the site file (read-only; nothing is signalled)."""

from __future__ import annotations

import fcntl

from tests.fixtures.site import example_mapping, write_example_site
from tools.equivalence import budget


def test_no_site_file_means_no_workload_locks(monkeypatch, tmp_path):
    monkeypatch.setenv("GLM_TPU_SITE_CONFIG", str(tmp_path / "absent.toml"))
    assert budget._workload_locks() == [] and not budget.indeterminate()


def test_an_unloadable_site_file_makes_detection_indeterminate(monkeypatch, tmp_path):
    path = write_example_site(tmp_path / "site.toml")
    path.chmod(0o644)  # not owner-only: the loader refuses it
    monkeypatch.setenv("GLM_TPU_SITE_CONFIG", str(path))
    assert budget.indeterminate() and budget.live_tpu_run()
    assert budget.refusal_reason() == budget.INDETERMINATE and budget.report()["indeterminate"] is True


def test_an_unresolvable_site_location_makes_detection_indeterminate(monkeypatch):
    monkeypatch.setenv("GLM_TPU_SITE_CONFIG", "relative/site.toml")
    assert budget.indeterminate() and budget.live_tpu_run()


def test_a_held_workload_lock_of_a_valid_site_is_seen(monkeypatch, tmp_path):
    locks = [tmp_path / "locks" / f"lock{i}" for i in range(4)]
    locks[0].parent.mkdir()
    path = write_example_site(
        tmp_path / "site.toml",
        example_mapping(tmp_path, locks=dict(workload=[str(p) for p in locks[:2]], sync=[str(p) for p in locks[2:]])),
    )
    monkeypatch.setenv("GLM_TPU_SITE_CONFIG", str(path))
    assert not budget.indeterminate() and budget.held_workload_locks() == []
    with open(locks[1], "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert budget.held_workload_locks() == [str(locks[1])] and budget.live_tpu_run()
    with open(locks[2], "a") as stream:  # a sync lock (cron backup) is not an indicator
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert budget.held_workload_locks() == []


def test_a_process_of_a_fleet_module_is_a_live_run(monkeypatch, tmp_path):
    """The process scan reads /proc/<pid>/cmdline (a fake /proc here: a real child naming a fleet module would make
    a concurrent heavy gate see a live run): the topology capture counts as the worker and the pack worker do."""
    proc = tmp_path / "proc"
    for pid, argv in (
        (101, [b"python3.12", b"-m", b"glm_tpu.distributed.topology_capture", b"--output", b"/runs/x"]),
        (102, [b"python3.12", b"-m", b"glm_tpu.model_loader.pack_worker", b"--output", b"/runs/y"]),
        (103, [b"python3.12", b"-m", b"glm_tpu.entrypoints.serve.server", b"--run", b"/runs/z"]),
        (104, [b"python3.12", b"-c", b"print('glm_tpu.distributed.topology_capture')"]),
    ):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(b"\0".join(argv) + b"\0")
    (proc / "self").mkdir()
    real = budget.Path
    monkeypatch.setattr(budget, "Path", lambda path: proc if str(path) == "/proc" else real(path))
    assert budget.live_processes() == [101, 102]  # an element naming the module, not a substring
