"""Fixed failure recovery must refuse before deployment when ownership differs."""
import sys

import pytest

from scripts.greenfield import recover_native_trace_originals as recovery


@pytest.mark.parametrize("fault", ["busy", "missing", "exit"])
def test_refuse_before_sync_or_publication(tmp_path, monkeypatch, fault):
    root = tmp_path/recovery.TAG
    root.mkdir()
    (root/"census_post.txt").write_text("fixture census")
    monkeypatch.setattr(sys, "argv", ["recover", "--recovery-pin", "a"*40])
    monkeypatch.setattr(recovery.launch, "source_preflight", lambda _: None)
    monkeypatch.setattr(recovery.launch.watch, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(recovery.launch.watch, "LOCKS", [tmp_path/"lock1", tmp_path/"lock2"])
    monkeypatch.setattr(recovery, "validate_fleet", lambda _: None)
    calls = []
    monkeypatch.setattr(recovery.launch, "ssh", lambda command: calls.append(command) or "fixture census")
    fleet = [dict(processes=[], holders=[]) for _ in range(8)]
    rows = [dict(ended=dict(worker_exit_code=1), published=dict(publish_exit_code=1)) for _ in range(8)]
    if fault == "busy": fleet[3]["holders"] = [42]
    if fault == "missing": rows[4]["ended"] = None
    if fault == "exit": rows[5]["ended"]["worker_exit_code"] = 0
    monkeypatch.setattr(recovery.launch.watch, "observe", lambda *a: fleet)
    monkeypatch.setattr(recovery.launch, "publication_state", lambda *a: rows)
    with pytest.raises(ValueError): recovery.main()
    assert len(calls) == 1  # fresh census only, never deploy/publish/run
