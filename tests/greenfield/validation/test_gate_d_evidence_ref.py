from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/greenfield/validate_gate_d_evidence_ref.py"


def _load():
    specification = importlib.util.spec_from_file_location(
        "gate_d_evidence_ref_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


MODULE = _load()


def _git(repository: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Gate D Test",
        "GIT_AUTHOR_EMAIL": "gate-d@example.invalid",
        "GIT_COMMITTER_NAME": "Gate D Test",
        "GIT_COMMITTER_EMAIL": "gate-d@example.invalid",
    }
    return subprocess.run(
        ["/usr/bin/git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    ).stdout.strip()


def _bare_git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", "--git-dir", str(repository), *arguments],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
        text=True,
    ).stdout.strip()


def _repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str, str]:
    repository = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "switch", "-q", "-c", "rewrite/topology-first-decode")
    (repository / "HANDOFF.md").write_text("base\n")
    (repository / "docs/greenfield").mkdir(parents=True)
    (repository / "docs/greenfield/EVIDENCE_MAP.md").write_text("base\n")
    (repository / "scripts").mkdir()
    (repository / "scripts/runtime.py").write_text("VALUE = 1\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "source")
    pin = _git(repository, "rev-parse", "HEAD")

    subprocess.run(
        ["/usr/bin/git", "init", "--bare", "-q", str(remote)], check=True
    )
    origin = remote.as_uri()
    _git(repository, "remote", "add", "origin", origin)
    _git(
        repository,
        "push",
        "-q",
        "origin",
        "rewrite/topology-first-decode:rewrite/topology-first-decode",
    )
    _git(repository, "switch", "-q", "-c", "evidence/gate-d-topology-first")

    monkeypatch.setattr(MODULE, "EXPECTED_REPOSITORY", repository)
    monkeypatch.setattr(
        MODULE,
        "EXPECTED_COMMON_DIR",
        Path(
            _git(
                repository,
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            )
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "EXPECTED_GIT_DIR",
        Path(_git(repository, "rev-parse", "--path-format=absolute", "--git-dir")),
    )
    monkeypatch.setattr(MODULE, "EXPECTED_ORIGIN", origin)
    monkeypatch.setattr(MODULE, "MIN_TRACKED_FILES", 1)
    return repository, remote, pin, origin


def _stage_root(repository: Path, pin: str, origin: str) -> None:
    path = repository / MODULE.ROOT_AUTHORITY
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "gate_d_evidence_ref_root_v1",
                "evidence_branch": "evidence/gate-d-topology-first",
                "execution_branch": "rewrite/topology-first-decode",
                "execution_pin": pin,
                "hlo_work": False,
                "origin": origin,
                "schema_version": 1,
                "tpu_work": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    _git(repository, "add", MODULE.ROOT_AUTHORITY)


def _rewrite_staged_root(repository: Path, **updates: object) -> None:
    path = repository / MODULE.ROOT_AUTHORITY
    value = json.loads(path.read_text())
    value.update(updates)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    _git(repository, "add", MODULE.ROOT_AUTHORITY)


def _stage_artifact(repository: Path, name: str = "gate-d-run.json") -> None:
    path = repository / "docs/artifacts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    _git(repository, "add", str(path.relative_to(repository)))


def _commit_root(repository: Path, pin: str, origin: str) -> str:
    _stage_root(repository, pin, origin)
    _git(repository, "commit", "-q", "-m", "root evidence authority")
    return _git(repository, "rev-parse", "HEAD")


def _push_evidence(repository: Path) -> None:
    _git(
        repository,
        "push",
        "-q",
        "origin",
        "HEAD:evidence/gate-d-topology-first",
    )


def test_staged_prefix_committed_and_actual_remote_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _stage_root(repository, pin, origin)
    staged_root = MODULE.validate(
        repository, pin, mode="staged", require_remote_equal=False
    )
    assert staged_root["changed_paths"] == [MODULE.ROOT_AUTHORITY]
    _git(repository, "commit", "-q", "-m", "root evidence authority")

    _stage_artifact(repository)
    staged = MODULE.validate(
        repository, pin, mode="staged", require_remote_equal=False
    )
    assert set(staged["changed_paths"]) == {
        MODULE.ROOT_AUTHORITY,
        "docs/artifacts/gate-d-run.json",
    }
    _git(repository, "commit", "-q", "-m", "evidence")
    committed = MODULE.validate(
        repository, None, mode="committed", require_remote_equal=False
    )
    assert committed["remote_evidence_head"] is None
    _push_evidence(repository)
    post_push = MODULE.validate(
        repository, None, mode="committed", require_remote_equal=True
    )
    assert post_push["evidence_head"] == post_push["remote_evidence_head"]


@pytest.mark.parametrize("hostile_version", [True, 1.0])
def test_root_schema_version_requires_an_exact_integer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile_version: object,
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _stage_root(repository, pin, origin)
    _rewrite_staged_root(repository, schema_version=hostile_version)
    with pytest.raises(MODULE.EvidenceRefError, match="root authority contract"):
        MODULE.validate(repository, pin, mode="staged", require_remote_equal=False)


def test_remote_heads_require_exact_ascii_lf_and_records() -> None:
    execution = b"a" * 40 + b"\t" + MODULE.REMOTE_EXECUTION_BRANCH.encode()
    evidence = b"b" * 40 + b"\t" + MODULE.REMOTE_EVIDENCE_BRANCH.encode()
    assert MODULE._parse_remote_heads(execution + b"\n" + evidence + b"\n") == {
        MODULE.REMOTE_EXECUTION_BRANCH: "a" * 40,
        MODULE.REMOTE_EVIDENCE_BRANCH: "b" * 40,
    }
    hostile = (
        execution + b"\r\n",
        execution,
        execution + b"\n\n",
        execution + b"\n" + execution + b"\n",
        execution + b" extra\n",
        execution + b"\n" + b"c" * 40 + b"\trefs/heads/unexpected\n",
        execution + b"\n\xff",
    )
    for raw in hostile:
        with pytest.raises(MODULE.EvidenceRefError):
            MODULE._parse_remote_heads(raw)


def test_ambient_repository_index_and_object_redirects_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _stage_root(repository, pin, origin)
    for name in (
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        monkeypatch.setenv(name, str(tmp_path / "forged"))
        with pytest.raises(MODULE.EvidenceRefError, match="ambient Git environment"):
            MODULE.validate(
                repository, pin, mode="staged", require_remote_equal=False
            )
        monkeypatch.delenv(name)


def test_wrong_repository_identity_or_origin_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _stage_root(repository, pin, origin)
    monkeypatch.setattr(MODULE, "EXPECTED_GIT_DIR", tmp_path / "forged.git")
    with pytest.raises(MODULE.EvidenceRefError, match="administrative identity"):
        MODULE.validate(repository, pin, mode="staged", require_remote_equal=False)
    monkeypatch.setattr(
        MODULE,
        "EXPECTED_GIT_DIR",
        Path(_git(repository, "rev-parse", "--path-format=absolute", "--git-dir")),
    )
    _git(repository, "remote", "set-url", "origin", (tmp_path / "wrong.git").as_uri())
    with pytest.raises(MODULE.EvidenceRefError, match="local Git config mismatch"):
        MODULE.validate(repository, pin, mode="staged", require_remote_equal=False)


def test_local_config_redirect_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _stage_root(repository, pin, origin)
    _git(repository, "config", "url.file:///tmp/forged.insteadOf", origin)
    with pytest.raises(MODULE.EvidenceRefError, match="unexpected local Git config"):
        MODULE.validate(repository, pin, mode="staged", require_remote_equal=False)


def test_staged_mode_rejects_prior_executable_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _commit_root(repository, pin, origin)
    (repository / "scripts/runtime.py").write_text("VALUE = 2\n")
    _git(repository, "add", "scripts/runtime.py")
    _git(repository, "commit", "-q", "-m", "hostile executable")
    _stage_artifact(repository)
    with pytest.raises(MODULE.EvidenceRefError, match="non-evidence path"):
        MODULE.validate(repository, pin, mode="staged", require_remote_equal=False)


def test_staged_mode_rejects_prior_artifact_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _commit_root(repository, pin, origin)
    _stage_artifact(repository)
    _git(repository, "commit", "-q", "-m", "evidence")
    (repository / "docs/artifacts/gate-d-run.json").write_text('{"changed":true}\n')
    _git(repository, "add", "docs/artifacts/gate-d-run.json")
    _git(repository, "commit", "-q", "-m", "rewrite evidence")
    _stage_artifact(repository, "gate-d-other.json")
    with pytest.raises(MODULE.EvidenceRefError, match="add-only"):
        MODULE.validate(repository, pin, mode="staged", require_remote_equal=False)


def test_merge_commit_is_rejected_in_staged_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _commit_root(repository, pin, origin)
    root_head = _git(repository, "rev-parse", "HEAD")
    _git(repository, "switch", "-q", "-c", "temporary", root_head)
    _stage_artifact(repository, "gate-d-temporary.json")
    _git(repository, "commit", "-q", "-m", "temporary")
    _git(repository, "switch", "-q", "evidence/gate-d-topology-first")
    _stage_artifact(repository, "gate-d-mainline.json")
    _git(repository, "commit", "-q", "-m", "mainline")
    _git(repository, "merge", "-q", "--no-ff", "temporary", "-m", "merge")
    _stage_artifact(repository, "gate-d-after-merge.json")
    with pytest.raises(MODULE.EvidenceRefError, match="merge/root commit"):
        MODULE.validate(repository, pin, mode="staged", require_remote_equal=False)


def test_intermediate_symlink_mode_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _commit_root(repository, pin, origin)
    path = repository / "docs/artifacts/gate-d-link.json"
    path.symlink_to("../../HANDOFF.md")
    _git(repository, "add", str(path.relative_to(repository)))
    _git(repository, "commit", "-q", "-m", "hostile symlink")
    path.unlink()
    path.write_text("{}\n")
    _git(repository, "add", str(path.relative_to(repository)))
    _git(repository, "commit", "-q", "-m", "hide hostile symlink")
    with pytest.raises(MODULE.EvidenceRefError, match="mode-100644"):
        MODULE.validate(repository, pin, mode="committed", require_remote_equal=False)


def test_actual_remote_non_fast_forward_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    root_head = _commit_root(repository, pin, origin)
    _stage_artifact(repository)
    _git(repository, "commit", "-q", "-m", "remote evidence")
    _push_evidence(repository)
    _git(repository, "reset", "-q", "--hard", root_head)
    _stage_artifact(repository, "gate-d-divergent.json")
    _git(repository, "commit", "-q", "-m", "divergent evidence")
    with pytest.raises(MODULE.EvidenceRefError, match="not an ancestor"):
        MODULE.validate(repository, pin, mode="committed", require_remote_equal=False)


def test_stale_tracking_refs_are_ignored_but_actual_remote_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, remote, pin, origin = _repository(tmp_path, monkeypatch)
    _commit_root(repository, pin, origin)
    evidence_head = _git(repository, "rev-parse", "HEAD")
    _git(
        repository,
        "update-ref",
        "refs/remotes/origin/rewrite/topology-first-decode",
        evidence_head,
    )
    MODULE.validate(repository, pin, mode="committed", require_remote_equal=False)
    _push_evidence(repository)
    _bare_git(
        remote,
        "update-ref",
        "refs/heads/rewrite/topology-first-decode",
        evidence_head,
    )
    with pytest.raises(MODULE.EvidenceRefError, match="remote execution branch drift"):
        MODULE.validate(repository, pin, mode="committed", require_remote_equal=False)


def test_wrong_branch_dirty_and_partial_worktrees_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _remote, pin, origin = _repository(tmp_path, monkeypatch)
    _commit_root(repository, pin, origin)
    _git(repository, "switch", "-q", "rewrite/topology-first-decode")
    with pytest.raises(MODULE.EvidenceRefError, match="dedicated evidence branch"):
        MODULE.validate(repository, pin, mode="committed", require_remote_equal=False)
    _git(repository, "switch", "-q", "evidence/gate-d-topology-first")

    handoff = repository / "HANDOFF.md"
    handoff.write_text("dirty\n")
    with pytest.raises(MODULE.EvidenceRefError, match="clean worktree"):
        MODULE.validate(repository, pin, mode="committed", require_remote_equal=False)
    _git(repository, "restore", "HANDOFF.md")

    _git(repository, "update-index", "--skip-worktree", "HANDOFF.md")
    handoff.unlink()
    with pytest.raises(MODULE.EvidenceRefError, match="nonordinary index flag"):
        MODULE.validate(repository, pin, mode="committed", require_remote_equal=False)
