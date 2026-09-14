"""Protected user controller orchestration with isolated locks and fake hosts."""
import argparse
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu import user_request
from scripts.release import launch_ws32_user_request as launch

TAG = "greenfield_ws32_user_request_20260914T020000000000000Z"
PIN = "a"*40


def args(**kwargs):
    return argparse.Namespace(role="controller", tag=TAG, code_hash=PIN,
        reviewed_branch="release/fixture", request=None, attach=False, wall_seconds=3600,
        request_file_sha256="b"*64, request_generation=123456789, request_bytes=1000,
        coordinator="192.168.0.37:8476", **kwargs)


def test_default_off_before_any_controller_action(monkeypatch):
    monkeypatch.delenv("GLM_GREENFIELD_USER_REQUEST", raising=False)
    monkeypatch.setattr(launch, "controller", lambda _: pytest.fail("default-off dispatched"))
    with pytest.raises(ValueError, match="default-off"):
        launch.main(["--tag", TAG, "--code-hash", PIN])


@pytest.mark.parametrize("extra", [[], ["--role", "publish", "--attach"]])
def test_republication_only_allowed_for_controller_attach(monkeypatch, extra):
    monkeypatch.setenv("GLM_GREENFIELD_USER_REQUEST", "1")
    monkeypatch.setattr(launch, "controller", lambda _: pytest.fail("invalid recovery dispatched"))
    with pytest.raises(ValueError, match="requires controller attach"):
        launch.main(["--tag", TAG, "--code-hash", PIN, "--republish-originals", *extra])


@pytest.mark.parametrize("extra", [["--request", "/different"], ["--wall-seconds", "5"],
    ["--request-generation", "1"], ["--request-file-sha256", "c"*64], ["--coordinator", "192.168.0.1:8476"]])
def test_attach_cannot_replace_original_input_or_deadline(monkeypatch, extra):
    monkeypatch.setenv("GLM_GREENFIELD_USER_REQUEST", "1")
    monkeypatch.setattr(launch, "controller", lambda _: pytest.fail("invalid attach dispatched"))
    with pytest.raises(ValueError, match="no overrides"):
        launch.main(["--tag", TAG, "--code-hash", PIN, "--attach", *extra])


@pytest.mark.parametrize("field,value", [("tag", "bad;command"), ("code_hash", "--bad"),
    ("request_generation", True), ("request_bytes", 17<<20), ("request_file_sha256", "bad"),
    ("wall_seconds", 86401), ("coordinator", "192.168.0.1:9999")])
def test_transport_command_refuses_invalid_identity(field, value):
    value_args = args()
    setattr(value_args, field, value)
    with pytest.raises(ValueError):
        launch.remote_command(value_args, "worker")


def test_worker_command_is_private_detached_and_only_uses_user_identity():
    text = launch.remote_command(args(), "worker")
    assert "umask 077" in text and "nohup" in text and "GLM_GREENFIELD_USER_REQUEST=1" in text
    assert "JAX_PLATFORMS=cpu" in text and "--role worker" in text
    assert "--request-generation 123456789" in text
    assert "--native-benchmark-request" not in text


def locks(tmp_path, monkeypatch):
    paths = (tmp_path / "workload.lock", tmp_path / "mirror.lock")
    for path in paths: path.touch()
    monkeypatch.setattr(launch.watch, "LOCKS", paths)
    return paths


def test_second_lease_refusal_precedes_request_upload_or_dispatch(tmp_path, monkeypatch):
    from google.cloud import storage
    paths = locks(tmp_path, monkeypatch)
    monkeypatch.setattr(launch.socket, "gethostname", lambda: "fixture-w-0")
    monkeypatch.setattr(launch.shared, "source_preflight", lambda *a, **k: None)
    monkeypatch.setattr(storage, "Client", lambda: pytest.fail("cloud invoked under another lease owner"))
    monkeypatch.setattr(launch.shared, "ssh", lambda *a, **k: pytest.fail("remote work under another lease owner"))
    with paths[1].open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): launch.controller(args())
        with paths[0].open("a") as first:
            fcntl.flock(first, fcntl.LOCK_EX|fcntl.LOCK_NB)  # first lease released on refusal


@pytest.mark.parametrize("seal_failure", [False, True])
def test_ambiguous_dispatch_observes_once_with_both_leases_no_restart(tmp_path, monkeypatch, capsys, seal_failure):
    from google.cloud import storage
    paths = locks(tmp_path, monkeypatch)
    monkeypatch.setattr(launch.socket, "gethostname", lambda: "fixture-w-0")
    run_root = tmp_path / "runs"
    run_root.mkdir(mode=0o700)
    monkeypatch.setattr(launch.worker, "RUN_ROOT", run_root)
    monkeypatch.setattr(launch.shared, "source_preflight", lambda *a, **k: None)
    monkeypatch.setattr(launch.shared, "sync_command", lambda *a, **k: "sync")
    monkeypatch.setattr(launch, "census_command", lambda: "census")
    monkeypatch.setattr(launch, "validate_fleet", lambda value: value == "fixture-census" or pytest.fail("bad census"))
    monkeypatch.setattr(launch.shutil, "disk_usage", lambda _: SimpleNamespace(free=20<<30))
    bucket = SimpleNamespace(list_blobs=lambda **kw: iter([]),
                             soft_delete_policy=SimpleNamespace(retention_duration_seconds=0))
    monkeypatch.setattr(storage, "Client", lambda: None)
    monkeypatch.setattr(launch.transport.cold, "_bucket", lambda _: bucket)
    monkeypatch.setattr(launch, "publish_exact", lambda *a, **k: dict(generation="100", name=launch.request_object(TAG)))
    seen = []

    def ssh(command, **kwargs):
        seen.append(command)
        if command == "census": return "fixture-census"
        if command == "sync": return "\n".join(f"NATIVE_SYNC_OK {r}" for r in range(8))
        if command == "hostname -I": return "192.168.0.37"
        if "--role prepare" in command: return "\n".join(f"USER_PREPARED {r}" for r in range(8))
        assert "--role worker" in command
        raise TimeoutError("SSH observation lost after command delivery")

    monkeypatch.setattr(launch.shared, "ssh", ssh)

    def watch(root, value_args):
        assert (root / "dispatch_unknown.json").is_file()
        assert (root / "launch.json").is_file()
        for path in paths:
            with path.open("a") as other:
                with pytest.raises(BlockingIOError): fcntl.flock(other, fcntl.LOCK_EX|fcntl.LOCK_NB)
        return [dict(rank=r) for r in range(8)], [dict(published=dict(publish_exit_code=0)) for _ in range(8)]

    monkeypatch.setattr(launch, "watch_originals", watch)
    monkeypatch.setattr(launch.transport, "collect", lambda **kw:
        dict(benchmark=False, protected_result_sealed=False, fake_transport_for_orchestration_test=True))
    def replay(root, tag, pin):
        assert tag == TAG and pin == PIN and (root / "user_collection.json").is_file()
        for path in paths:
            with path.open("a") as other:
                with pytest.raises(BlockingIOError): fcntl.flock(other, fcntl.LOCK_EX|fcntl.LOCK_NB)
        return dict(fixture_not_evidence=True, protected_result_sealed=False)
    monkeypatch.setattr(launch, "replay_collected", replay)
    def seal(**kwargs):
        assert (kwargs["root"] / "user_replay.json").is_file()
        assert kwargs["report"]["fixture_not_evidence"]
        for path in paths:
            with path.open("a") as other:
                with pytest.raises(BlockingIOError): fcntl.flock(other, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if seal_failure:
            raise RuntimeError("fixture archive interrupted")
        return dict(fixture_only=True, protected_result_sealed=False, benchmark=False)
    monkeypatch.setattr(launch, "seal", seal)
    value_args = args()
    value_args.request = tmp_path / "prepared.json"
    value = user_request.from_token_ids([1, 2, 3], request_id="fixture", seed=1, max_new_tokens=2)
    value_args.request.write_bytes(user_request.canonical(value)+b"\n")
    if seal_failure:
        with pytest.raises(RuntimeError, match="archive interrupted"):
            launch.controller(value_args)
        assert not capsys.readouterr().out
        assert (run_root / TAG / "user_replay.json").exists()
    else:
        assert launch.controller(value_args) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["protected_result_sealed"] is False and report["benchmark"] is False
    assert sum("--role worker" in command for command in seen) == 1
    assert not (run_root / TAG / "SUCCESS").exists()


@pytest.mark.parametrize("unknown_wait", [False, True])
def test_supervisor_distinguishes_prelaunch_failure_from_unknown_exit(tmp_path, monkeypatch, unknown_wait):
    monkeypatch.setattr(launch.worker, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(launch.socket, "gethostname", lambda: "fixture-w-0")
    root = tmp_path / TAG
    root.mkdir(mode=0o700)
    if unknown_wait:
        monkeypatch.setattr(launch.worker, "preflight", lambda _: None)
        monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k:
            SimpleNamespace(wait=lambda: (_ for _ in ()).throw(RuntimeError("unknown wait"))))
    else:
        monkeypatch.setattr(launch.worker, "preflight", lambda _: (_ for _ in ()).throw(ValueError("preflight")))
        monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("refused worker started"))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    if unknown_wait:
        with pytest.raises(RuntimeError, match="unknown wait"): launch.worker_child(args())
        assert not (root / "ended.rank0.json").exists()
        assert not (root / "published.rank0.json").exists()
    else:
        assert launch.worker_child(args()) == 1
        ended = json.loads((root / "ended.rank0.json").read_bytes())
        assert ended["worker_started"] is False and ended["worker_exit_code"] == 1
        assert ended["worker_error_type"] == "ValueError"


def test_watch_never_adopts_replacement_pid_or_treats_unknown_as_exit(tmp_path, monkeypatch):
    from copy import deepcopy
    root = tmp_path / TAG
    root.mkdir(mode=0o700)
    initial = [dict(rank=r, host=f"fixture-w-{r}", boot_id="boot", tag=TAG, pin=PIN,
        processes=[dict(pid=100+r, start_ticks="1", argv_sha256="b"*64)], holders=[100+r]) for r in range(8)]
    changed = deepcopy(initial)
    changed[0]["processes"][0]["pid"] = 999
    idle = deepcopy(initial)
    for row in idle: row.update(processes=[], holders=[])
    observations = iter([initial, changed, idle, idle])
    monkeypatch.setattr(launch.watch, "observe", lambda *a: next(observations))
    states = [dict(ended=dict(request_file_sha256="b"*64),
                   published=dict(request_file_sha256="b"*64)) for _ in range(8)]
    monkeypatch.setattr(launch.shared, "publication_state", lambda *a: states)
    sleeps = []
    monkeypatch.setattr(launch.time, "sleep", sleeps.append)
    original, publication = launch.watch_originals(root, args())
    assert original == initial and publication == states
    rows = [json.loads(line) for line in (root / "user_watch.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["OBSERVED", "UNKNOWN_RETRYING", "OBSERVED", "OBSERVED"]
    assert "fleet" not in rows[1] and sleeps == [30, 30, 30]


def test_controller_refuses_nonzero_host_before_any_remote_work(monkeypatch):
    monkeypatch.setattr(launch.socket, "gethostname", lambda: "fixture-w-1")
    monkeypatch.setattr(launch.shared, "source_preflight", lambda *a, **k: pytest.fail("wrong controller admitted"))
    with pytest.raises(ValueError, match="worker0"):
        launch.controller(args())
