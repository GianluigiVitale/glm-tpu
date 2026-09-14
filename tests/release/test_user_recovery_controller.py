"""Actual attach/recovery control flow, isolated leases and fake transport/math."""

import fcntl
import json
from types import SimpleNamespace

import pytest

from glm_tpu import user_request
from scripts.release import launch_ws32_user_request as launch
from tests.release.test_user_launch import args, locks
from tests.release.test_user_publication_recovery import case as recovery_case


@pytest.mark.parametrize(
    "fault", [None, "not_enabled", "worker_failed", "upload_timeout"]
)
def test_attach_recovers_upload_only_under_both_leases(
    recovery_case, monkeypatch, tmp_path, capsys, fault
):
    from google.cloud import storage

    case = recovery_case
    paths = locks(tmp_path, monkeypatch)
    monkeypatch.setattr(launch.worker, "RUN_ROOT", case.root.parent)
    monkeypatch.setattr(launch.socket, "gethostname", lambda: "fixture-w-0")
    monkeypatch.setattr(launch.shared, "source_preflight", lambda *a, **k: None)
    value_args = args()
    value_args.attach = True
    value_args.republish_originals = fault != "not_enabled"
    raw = (case.root / "request.json").read_bytes()
    metadata = dict(
        tag=value_args.tag,
        code_hash=value_args.code_hash,
        reviewed_branch=value_args.reviewed_branch,
        benchmark=False,
        request_file_sha256=case.args.request_file_sha256,
        request_bytes=len(raw),
        request_generation=123456789,
        wall_seconds=3600,
        coordinator=value_args.coordinator,
    )
    (case.root / "launch.json").write_bytes(user_request.canonical(metadata))
    if fault == "worker_failed":
        case.publication[7]["ended"]["worker_exit_code"] = 1
    originals = {}
    for rank, row in enumerate(case.publication):
        path = case.root / f"published.rank{rank}.json"
        originals[path] = user_request.canonical(row["published"])
        path.write_bytes(originals[path])
    steps = []

    def held():
        for path in paths:
            with path.open("a") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def observed(root, received):
        held()
        steps.append("observe")
        assert received.request_file_sha256 == case.args.request_file_sha256
        return case.owners, case.publication

    def ssh(command, *, workers="all", timeout=55):
        held()
        if command == "fixture-census":
            steps.append("census")
            return command
        assert "--role publish" in command
        assert "--role worker" not in command and "--role prepare" not in command
        steps.append("publish" + workers)
        if fault == "upload_timeout":
            raise TimeoutError("ambiguous original upload; never restart worker")
        return case.ssh(
            "FIXTURE_PUBLISH_ORIGINALS_ONLY", workers=workers, timeout=timeout
        )

    monkeypatch.setattr(launch, "watch_originals", observed)
    monkeypatch.setattr(launch, "census_command", lambda: "fixture-census")
    monkeypatch.setattr(
        launch,
        "validate_fleet",
        lambda value: value == "fixture-census" or pytest.fail("bad census"),
    )
    monkeypatch.setattr(launch.shared, "ssh", ssh)
    bucket = SimpleNamespace(list_blobs=lambda **kwargs: [])
    monkeypatch.setattr(storage, "Client", lambda: None)
    monkeypatch.setattr(launch.transport, "approved_bucket", lambda _: bucket)

    def collect(**kwargs):
        held()
        steps.append("collect")
        return dict(fixture_only=True, protected_result_sealed=False)

    def replay(*values):
        held()
        steps.append("replay")
        return dict(fixture_only=True, protected_result_sealed=False)

    def seal(**kwargs):
        held()
        steps.append("seal")
        assert (case.root / "publication_recovery.json").is_file()
        return dict(fixture_only=True, protected_result_sealed=False)

    monkeypatch.setattr(launch.transport, "collect", collect)
    monkeypatch.setattr(launch, "replay_collected", replay)
    monkeypatch.setattr(launch, "seal", seal)
    if fault is None:
        assert launch.controller(value_args) == 0
        assert steps == [
            "observe",
            "census",
            "publish2",
            "publish7",
            "collect",
            "replay",
            "seal",
        ]
        assert json.loads(capsys.readouterr().out)["protected_result_sealed"] is False
    else:
        with pytest.raises((RuntimeError, ValueError, TimeoutError)):
            launch.controller(value_args)
        assert "collect" not in steps and "seal" not in steps
        assert not capsys.readouterr().out
        assert not (case.root / "publication_recovery.json").exists()
        if fault != "upload_timeout":
            assert steps == ["observe", "census"]
    for path, content in originals.items():
        assert path.read_bytes() == content
