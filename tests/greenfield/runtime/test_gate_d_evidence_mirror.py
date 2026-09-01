from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/greenfield/mirror_gate_d_evidence.py"


def _load():
    specification = importlib.util.spec_from_file_location(
        "gate_d_evidence_mirror_test", SCRIPT
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


def _receipt(head: str) -> bytes:
    value = {
        "evidence_head": head,
        "mode": "committed",
        "remote_evidence_head": head,
        "status": "GATE_D_EVIDENCE_REF_VALID",
    }
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def test_real_export_matches_git_catalogue_before_append_only_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    (repository / "nested").mkdir()
    (repository / "nested/plain.txt").write_text("plain\n")
    executable = repository / "run.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "source")
    head = _git(repository, "rev-parse", "HEAD")

    output_root = tmp_path / "gate-d-evidence-mirror.abcdefgh"
    output_root.mkdir(mode=0o700)
    monkeypatch.setattr(MODULE, "EXPECTED_REPOSITORY", repository)
    monkeypatch.setattr(MODULE, "EXPECTED_EXPORT_PARENT", tmp_path)
    monkeypatch.setattr(MODULE, "MIN_TRACKED_FILES", 1)
    mirrored: list[tuple[tuple[object, ...], str]] = []

    def publish(local_objects, destination, **_kwargs):
        mirrored.append((local_objects, destination))
        return (
            tuple(
                MODULE.RemoteObject(
                    local.name,
                    f"{destination}/{local.name}",
                    str(index + 1),
                    local.size,
                    local.md5,
                )
                for index, local in enumerate(local_objects)
            ),
            True,
        )

    monkeypatch.setattr(MODULE, "_publish_objects", publish)

    report = MODULE.mirror(repository, output_root, _receipt(head))
    tree = output_root / "tree"
    assert (tree / "nested/plain.txt").read_bytes() == b"plain\n"
    assert stat.S_IMODE((tree / "run.sh").stat().st_mode) == 0o755
    assert (tree / MODULE.RECEIPT_NAME).read_bytes() == _receipt(head)
    assert report["tracked_files"] == 2
    assert report["destination"].endswith("/" + head)
    assert report["published"] is True
    assert len(mirrored) == 1
    assert [local.name for local in mirrored[0][0]] == [
        "repository.tar",
        MODULE.RECEIPT_NAME,
    ]
    assert mirrored[0][1] == report["destination"]


def test_nonzero_archive_after_partial_output_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "repository.tar"

    def partial_failure(*_args, **kwargs):
        kwargs["stdout"].write(b"partial archive")
        return subprocess.CompletedProcess([], 17, stderr=b"producer failed")

    monkeypatch.setattr(MODULE.subprocess, "run", partial_failure)
    with pytest.raises(MODULE.EvidenceMirrorError, match="archive failed"):
        MODULE._write_archive(tmp_path, "a" * 40, archive)
    assert archive.read_bytes() == b"partial archive"


def test_publication_archive_is_reverified_against_git_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    (repository / "payload.txt").write_text("payload\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "source")
    head = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(MODULE, "MIN_TRACKED_FILES", 1)
    catalogue, _raw = MODULE._catalogue(repository, head)
    archive = tmp_path / "repository.tar"
    MODULE._write_archive(repository, head, archive)
    local = MODULE._local_object("repository.tar", archive)
    MODULE._verify_local_archive(local, catalogue)
    archive.write_bytes(b"hostile replacement")
    with pytest.raises(MODULE.EvidenceMirrorError, match="sealed identity"):
        MODULE._verify_local_archive(local, catalogue)


@pytest.mark.parametrize("failure_boundary", ["archive", "receipt"])
def test_export_or_receipt_failure_never_invokes_cloud_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    output_root = tmp_path / "gate-d-evidence-mirror.abcdefgh"
    output_root.mkdir()
    head = "a" * 40
    catalogue = {f"file-{index}": MODULE.Blob(0o100644, "b" * 40) for index in range(50)}
    monkeypatch.setattr(MODULE, "_safe_repository", lambda path: path)
    monkeypatch.setattr(MODULE, "_safe_output_root", lambda path: path)
    monkeypatch.setattr(MODULE, "_catalogue", lambda *_args: (catalogue, b"catalogue"))
    monkeypatch.setattr(MODULE, "_verify_tree", lambda *_args: None)
    monkeypatch.setattr(MODULE, "_fsync_directory", lambda *_args: None)
    publication_calls: list[object] = []
    monkeypatch.setattr(
        MODULE, "_publish_objects", lambda *_args: publication_calls.append(object())
    )

    if failure_boundary == "archive":
        def fail_archive(*_args):
            (output_root / "repository.tar").write_bytes(b"partial")
            raise MODULE.EvidenceMirrorError("partial producer failure")

        monkeypatch.setattr(MODULE, "_write_archive", fail_archive)
    else:
        monkeypatch.setattr(
            MODULE,
            "_write_archive",
            lambda _repo, _head, archive: archive.write_bytes(b"archive"),
        )
        monkeypatch.setattr(
            MODULE,
            "_extract_archive",
            lambda _archive, tree, _catalogue: tree.mkdir(),
        )
        monkeypatch.setattr(
            MODULE,
            "_write_receipt",
            lambda *_args: (_ for _ in ()).throw(OSError("ENOSPC")),
        )

    with pytest.raises((MODULE.EvidenceMirrorError, OSError)):
        MODULE.mirror(repository, output_root, _receipt(head))
    assert publication_calls == []


def test_generation_zero_upload_uses_retained_descriptor_and_remote_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "repository.tar"
    payload = b"authenticated archive bytes"
    path.write_bytes(payload)
    local = MODULE._local_object("repository.tar", path)
    invocations: list[tuple[list[str], object]] = []

    def emulate(arguments, *, stdin=None):
        invocations.append((arguments, stdin))
        if arguments[:2] == ["storage", "cp"] and arguments[-2] == "-":
            backup = tmp_path / "retained-original"
            path.rename(backup)
            path.write_bytes(b"hostile replacement")
            assert stdin.read() == payload
            return subprocess.CompletedProcess(arguments, 0, stdout=b"", stderr=b"")
        if arguments[:3] == ["storage", "objects", "describe"]:
            value = {
                "generation": "123",
                "md5_hash": local.md5,
                "name": "repos/glm-tpu-gate-d-evidence/"
                + "a" * 40
                + "/repository.tar",
                "size": local.size,
            }
            return subprocess.CompletedProcess(
                arguments, 0, stdout=json.dumps(value).encode(), stderr=b""
            )
        assert arguments[:2] == ["storage", "cp"]
        assert "--if-generation-match=123" in arguments
        return subprocess.CompletedProcess(arguments, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(MODULE, "_run_gcloud", emulate)
    destination = MODULE.DESTINATION_ROOT + "/" + "a" * 40
    remote = MODULE._publish_one(local, destination)
    assert remote.generation == "123"
    upload = invocations[0][0]
    assert "--if-generation-match=0" in upload
    assert f"--content-md5={local.md5}" in upload
    assert upload[-2:] == ["-", destination + "/repository.tar"]
    assert len(invocations) == 3


def test_generation_zero_collision_never_describes_or_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "repository.tar"
    path.write_bytes(b"local")
    local = MODULE._local_object("repository.tar", path)
    calls: list[list[str]] = []

    def collide(arguments, *, stdin=None):
        calls.append(arguments)
        assert stdin is not None
        return subprocess.CompletedProcess(arguments, 1, stdout=b"", stderr=b"412")

    monkeypatch.setattr(MODULE, "_run_gcloud", collide)
    with pytest.raises(MODULE.EvidenceMirrorError, match="generation-zero"):
        MODULE._publish_one(local, MODULE.DESTINATION_ROOT + "/" + "a" * 40)
    assert len(calls) == 1
    assert "--if-generation-match=0" in calls[0]


def test_partial_or_differing_prefix_is_preserved_without_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "repository.tar"
    receipt = tmp_path / MODULE.RECEIPT_NAME
    archive.write_bytes(b"archive")
    receipt.write_bytes(b"receipt")
    local_objects = (
        MODULE._local_object("repository.tar", archive),
        MODULE._local_object(MODULE.RECEIPT_NAME, receipt),
    )
    destination = MODULE.DESTINATION_ROOT + "/" + "a" * 40
    archive_url = destination + "/repository.tar"
    partial = {
        archive_url: MODULE.RemoteObject(
            "repository.tar", archive_url, "123", len(b"archive"), "wrong-md5"
        )
    }
    monkeypatch.setattr(
        MODULE,
        "_list_prefix",
        lambda _destination, *, soft_deleted=False: {} if soft_deleted else partial,
    )
    uploads: list[object] = []
    monkeypatch.setattr(MODULE, "_publish_one", lambda *_args: uploads.append(object()))
    with pytest.raises(MODULE.EvidenceMirrorError, match="partial or contains extra"):
        MODULE._publish_objects(local_objects, destination)
    assert uploads == []


def test_complete_but_byte_different_prefix_is_refused_without_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "repository.tar"
    receipt = tmp_path / MODULE.RECEIPT_NAME
    archive.write_bytes(b"archive")
    receipt.write_bytes(b"receipt")
    local_objects = (
        MODULE._local_object("repository.tar", archive),
        MODULE._local_object(MODULE.RECEIPT_NAME, receipt),
    )
    destination = MODULE.DESTINATION_ROOT + "/" + "a" * 40
    records = {
        f"{destination}/{local.name}": MODULE.RemoteObject(
            local.name,
            f"{destination}/{local.name}",
            str(index + 20),
            local.size,
            "hostile-md5" if index == 0 else local.md5,
        )
        for index, local in enumerate(local_objects)
    }
    monkeypatch.setattr(
        MODULE,
        "_list_prefix",
        lambda _destination, *, soft_deleted=False: {} if soft_deleted else records,
    )
    monkeypatch.setattr(
        MODULE,
        "_publish_one",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not upload")),
    )
    with pytest.raises(MODULE.EvidenceMirrorError, match="differs from local bytes"):
        MODULE._publish_objects(local_objects, destination)


def test_complete_existing_prefix_is_generation_replayed_without_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "repository.tar"
    receipt = tmp_path / MODULE.RECEIPT_NAME
    archive.write_bytes(b"archive")
    receipt.write_bytes(b"receipt")
    local_objects = (
        MODULE._local_object("repository.tar", archive),
        MODULE._local_object(MODULE.RECEIPT_NAME, receipt),
    )
    destination = MODULE.DESTINATION_ROOT + "/" + "a" * 40
    records = {
        f"{destination}/{local.name}": MODULE.RemoteObject(
            local.name,
            f"{destination}/{local.name}",
            str(index + 10),
            local.size,
            local.md5,
        )
        for index, local in enumerate(local_objects)
    }
    monkeypatch.setattr(
        MODULE,
        "_list_prefix",
        lambda _destination, *, soft_deleted=False: {} if soft_deleted else records,
    )
    downloads: list[str] = []
    monkeypatch.setattr(
        MODULE,
        "_download_and_verify",
        lambda remote, _local: downloads.append(remote.generation),
    )
    monkeypatch.setattr(
        MODULE,
        "_publish_one",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not upload")),
    )
    verified, published = MODULE._publish_objects(local_objects, destination)
    assert published is False
    assert [item.generation for item in verified] == ["10", "11"]
    assert downloads == ["10", "11"]


def test_soft_deleted_prior_prefix_is_refused_before_live_listing_or_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "repository.tar"
    path.write_bytes(b"archive")
    local = MODULE._local_object("repository.tar", path)
    destination = MODULE.DESTINATION_ROOT + "/" + "a" * 40
    remote = MODULE.RemoteObject(
        local.name,
        destination + "/" + local.name,
        "99",
        local.size,
        local.md5,
    )
    calls: list[bool] = []

    def list_prefix(_destination, *, soft_deleted=False):
        calls.append(soft_deleted)
        return {remote.url: remote} if soft_deleted else {}

    monkeypatch.setattr(MODULE, "_list_prefix", list_prefix)
    monkeypatch.setattr(
        MODULE,
        "_publish_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not upload")
        ),
    )
    with pytest.raises(MODULE.EvidenceMirrorError, match="soft-deleted prior"):
        MODULE._publish_objects((local,), destination)
    assert calls == [True]


def test_prefix_vacancy_requires_exact_no_objects_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absent = subprocess.CompletedProcess(
        [], 1, stdout=b"", stderr=MODULE._VACANT_STDERR
    )
    monkeypatch.setattr(MODULE, "_run_gcloud", lambda *_args, **_kwargs: absent)
    assert MODULE._list_prefix(MODULE.DESTINATION_ROOT + "/" + "a" * 40) == {}
    unavailable = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"timeout\n")
    monkeypatch.setattr(MODULE, "_run_gcloud", lambda *_args, **_kwargs: unavailable)
    with pytest.raises(MODULE.EvidenceMirrorError, match="vacancy check failed"):
        MODULE._list_prefix(MODULE.DESTINATION_ROOT + "/" + "a" * 40)


def test_receipt_writer_refuses_short_write_before_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    monkeypatch.setattr(MODULE.os, "write", lambda *_args: 0)
    with pytest.raises(MODULE.EvidenceMirrorError, match="short authority"):
        MODULE._write_receipt(tree, _receipt("a" * 40))
