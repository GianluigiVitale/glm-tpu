from __future__ import annotations

import importlib.util
import hashlib
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


def _listing(destination: str) -> list[dict[str, object]]:
    return [
        {"type": "prefix", "url": destination + "/"},
        {
            "metadata": {
                "generation": "10",
                "md5Hash": "receipt-md5",
                "name": destination.removeprefix("gs://driftbench-dsv4-uc/")
                + "/"
                + MODULE.RECEIPT_NAME,
                "size": "7",
            },
            "type": "cloud_object",
            "url": destination + "/" + MODULE.RECEIPT_NAME + "#10",
        },
        {
            "metadata": {
                "generation": "11",
                "md5Hash": "archive-md5",
                "name": destination.removeprefix("gs://driftbench-dsv4-uc/")
                + "/repository.tar",
                "size": "9",
            },
            "type": "cloud_object",
            "url": destination + "/repository.tar#11",
        },
    ]


def _tooling_authority(head: str = "a" * 40) -> dict[str, object]:
    return {
        "base_execution_pin": "b" * 40,
        "commits": [head],
        "helper_git_blob": "c" * 40,
        "helper_sha256": "d" * 64,
        "remote_tooling_head": head,
        "root_authority_sha256": "e" * 64,
        "status": "GATE_D_EVIDENCE_MIRROR_TOOLING_VALID",
        "tooling_branch": MODULE.TOOLING_BRANCH,
        "tooling_head": head,
        "tooling_tree": "f" * 40,
        "tracked_files": 840,
    }


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
    verified, published = MODULE._publish_objects(
        local_objects, destination, require_existing_replay=True
    )
    assert published is False
    assert [item.generation for item in verified] == ["10", "11"]
    assert downloads == ["10", "11"]


def test_replay_only_vacancy_refuses_before_any_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "repository.tar"
    path.write_bytes(b"archive")
    local = MODULE._local_object("repository.tar", path)
    calls: list[bool] = []

    def vacant(_destination, *, soft_deleted=False):
        calls.append(soft_deleted)
        return {}

    monkeypatch.setattr(MODULE, "_list_prefix", vacant)
    monkeypatch.setattr(
        MODULE,
        "_publish_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replay-only mode must not upload")
        ),
    )
    with pytest.raises(MODULE.EvidenceMirrorError, match="replay-only.*vacant"):
        MODULE._publish_objects(
            (local,),
            MODULE.DESTINATION_ROOT + "/" + "a" * 40,
            require_existing_replay=True,
        )
    assert calls == [True, False]


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


def test_actual_gcloud_prefix_schema_is_accepted_exactly() -> None:
    destination = MODULE.DESTINATION_ROOT + "/" + "a" * 40
    records = MODULE._parse_listing(
        json.dumps(_listing(destination)).encode(), destination
    )
    assert sorted(records) == [
        destination + "/" + MODULE.RECEIPT_NAME,
        destination + "/repository.tar",
    ]
    assert records[destination + "/repository.tar"].generation == "11"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values, _destination: values.pop(0),
        lambda values, _destination: values.insert(1, dict(values[0])),
        lambda values, _destination: values.__setitem__(
            slice(None), [values[1], values[0], values[2]]
        ),
        lambda values, destination: values[0].__setitem__(
            "url", destination + "/wrong/"
        ),
        lambda values, _destination: values[0].__setitem__("metadata", {}),
        lambda values, _destination: values.append(dict(values[-1])),
    ],
    ids=(
        "missing-prefix",
        "duplicate-prefix",
        "misordered-prefix",
        "wrong-prefix",
        "metadata-bearing-prefix",
        "extra-object",
    ),
)
def test_prefix_schema_variants_fail_closed(mutate) -> None:
    destination = MODULE.DESTINATION_ROOT + "/" + "a" * 40
    values = _listing(destination)
    mutate(values, destination)
    with pytest.raises(MODULE.EvidenceMirrorError, match="prefix marker|object"):
        MODULE._parse_listing(json.dumps(values).encode(), destination)


def test_duplicate_cloud_object_identity_is_rejected() -> None:
    destination = MODULE.DESTINATION_ROOT + "/" + "a" * 40
    values = _listing(destination)
    values[2] = json.loads(json.dumps(values[1]))
    with pytest.raises(MODULE.EvidenceMirrorError, match="duplicate"):
        MODULE._parse_listing(json.dumps(values).encode(), destination)


def test_receipt_writer_refuses_short_write_before_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    monkeypatch.setattr(MODULE.os, "write", lambda *_args: 0)
    with pytest.raises(MODULE.EvidenceMirrorError, match="short authority"):
        MODULE._write_receipt(tree, _receipt("a" * 40))


def test_tooling_source_controller_creates_exact_receipt_and_two_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _tooling_authority()
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(MODULE, "EXPECTED_EXPORT_PARENT", tmp_path)
    monkeypatch.setattr(MODULE, "EXPECTED_TOOLING_REPOSITORY", repository)
    monkeypatch.setattr(MODULE, "_require_tooling_authority", lambda: authority)
    catalogue = {"base.txt": MODULE.Blob(0o100644, "1" * 40)}
    monkeypatch.setattr(
        MODULE, "_catalogue", lambda *_args: (catalogue, b"catalogue\0")
    )
    monkeypatch.setattr(
        MODULE,
        "_write_archive",
        lambda _repository, _head, path: path.write_bytes(b"archive"),
    )
    monkeypatch.setattr(MODULE, "_verify_local_archive", lambda *_args: None)
    captured: list[tuple[tuple[object, ...], str, object]] = []

    def publish(local_objects, destination, *, catalogue=None, **_kwargs):
        captured.append((local_objects, destination, catalogue))
        return (
            tuple(
                MODULE.RemoteObject(
                    local.name,
                    destination + "/" + local.name,
                    str(index + 10),
                    local.size,
                    local.md5,
                )
                for index, local in enumerate(local_objects)
            ),
            True,
        )

    monkeypatch.setattr(MODULE, "_publish_objects", publish)
    output_root = tmp_path / "gate-d-tooling-source-mirror.aaaaaaaa"
    report = MODULE.mirror_tooling_source(output_root)

    assert report["status"] == "GATE_D_TOOLING_SOURCE_MIRRORED"
    assert report["published"] is True
    assert report["destination"] == (
        MODULE.TOOLING_SOURCE_ROOT + "/" + "a" * 40
    )
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o700
    receipt = output_root / MODULE.TOOLING_SOURCE_RECEIPT_NAME
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    raw = receipt.read_bytes()
    value = json.loads(raw)
    assert raw == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert value["tooling_authority"] == authority
    assert value["object_names"] == [
        "repository.tar",
        MODULE.TOOLING_SOURCE_RECEIPT_NAME,
    ]
    assert len(captured) == 1
    assert [item.name for item in captured[0][0]] == [
        "repository.tar",
        MODULE.TOOLING_SOURCE_RECEIPT_NAME,
    ]
    assert captured[0][2] == catalogue


@pytest.mark.parametrize("kind", ["wrong-head", "wrong-tree", "wrong-objects"])
def test_tooling_source_receipt_rejects_authority_rebinding(
    tmp_path: Path, kind: str
) -> None:
    path = tmp_path / "repository.tar"
    path.write_bytes(b"archive")
    archive = MODULE._local_object("repository.tar", path)
    authority = _tooling_authority()
    destination = MODULE.TOOLING_SOURCE_ROOT + "/" + "a" * 40
    catalogue_sha = "9" * 64
    value = MODULE._expected_tooling_source_receipt(
        authority, archive, destination, catalogue_sha
    )
    if kind == "wrong-head":
        value["tooling_authority"]["tooling_head"] = "0" * 40
    elif kind == "wrong-tree":
        value["tooling_authority"]["tooling_tree"] = "0" * 40
    else:
        value["object_names"] = ["repository.tar"]
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(MODULE.EvidenceMirrorError, match="authority contract"):
        MODULE._validate_tooling_source_receipt(
            raw, authority, archive, destination, catalogue_sha
        )


def test_tooling_source_destination_and_output_path_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "TOOLING_SOURCE_ROOT", "gs://wrong/prefix")
    with pytest.raises(MODULE.EvidenceMirrorError, match="destination authority"):
        MODULE._tooling_source_destination("a" * 40)
    monkeypatch.setattr(
        MODULE,
        "TOOLING_SOURCE_ROOT",
        "gs://driftbench-dsv4-uc/repos/glm-tpu-gate-d-mirror-tooling",
    )
    monkeypatch.setattr(MODULE, "EXPECTED_EXPORT_PARENT", tmp_path)
    wrong = tmp_path / "gate-d-tooling-source-mirror.wrong000"
    with pytest.raises(MODULE.EvidenceMirrorError, match="unexpected.*path"):
        MODULE._create_tooling_source_output_root(wrong, "a" * 40)
    existing = tmp_path / "gate-d-tooling-source-mirror.aaaaaaaa"
    existing.mkdir()
    with pytest.raises(MODULE.EvidenceMirrorError, match="exclusively create"):
        MODULE._create_tooling_source_output_root(existing, "a" * 40)


def test_tooling_source_controller_refuses_published_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _tooling_authority()
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(MODULE, "EXPECTED_EXPORT_PARENT", tmp_path)
    monkeypatch.setattr(MODULE, "EXPECTED_TOOLING_REPOSITORY", repository)
    monkeypatch.setattr(MODULE, "_require_tooling_authority", lambda: authority)
    monkeypatch.setattr(
        MODULE,
        "_catalogue",
        lambda *_args: ({"base.txt": MODULE.Blob(0o100644, "1" * 40)}, b"cat"),
    )
    monkeypatch.setattr(
        MODULE,
        "_write_archive",
        lambda _repository, _head, path: path.write_bytes(b"archive"),
    )
    monkeypatch.setattr(MODULE, "_verify_local_archive", lambda *_args: None)

    def replayed(local_objects, destination, **_kwargs):
        return (
            tuple(
                MODULE.RemoteObject(
                    local.name,
                    destination + "/" + local.name,
                    str(index + 20),
                    local.size,
                    local.md5,
                )
                for index, local in enumerate(local_objects)
            ),
            False,
        )

    monkeypatch.setattr(MODULE, "_publish_objects", replayed)
    with pytest.raises(MODULE.EvidenceMirrorError, match="not newly published"):
        MODULE.mirror_tooling_source(
            tmp_path / "gate-d-tooling-source-mirror.aaaaaaaa"
        )


def test_tooling_authority_binds_clean_pushed_branch_and_running_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = tmp_path / "origin.git"
    repository = tmp_path / "tooling"
    origin.mkdir()
    repository.mkdir()
    _git(origin, "init", "--bare", "-q")
    _git(repository, "init", "-q")
    _git(repository, "remote", "add", "origin", str(origin))
    (repository / "base.txt").write_text("base\n")
    _git(repository, "add", "--", "base.txt")
    _git(repository, "commit", "-q", "-m", "base")
    _git(repository, "branch", "-M", "rewrite/topology-first-decode")
    _git(
        repository,
        "push",
        "-q",
        "-u",
        "origin",
        "refs/heads/rewrite/topology-first-decode:refs/heads/rewrite/topology-first-decode",
    )
    base = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "-q", "-b", "tooling/gate-d-evidence-mirror")

    monkeypatch.setattr(MODULE, "EXPECTED_TOOLING_REPOSITORY", repository)
    monkeypatch.setattr(MODULE, "EXPECTED_COMMON_DIR", repository / ".git")
    monkeypatch.setattr(MODULE, "EXPECTED_TOOLING_GIT_DIR", repository / ".git")
    monkeypatch.setattr(MODULE, "EXPECTED_ORIGIN", str(origin))
    monkeypatch.setattr(MODULE, "BASE_EXECUTION_PIN", base)
    monkeypatch.setattr(MODULE, "MIN_TRACKED_FILES", 1)

    helper = repository / MODULE.HELPER_RELATIVE
    test_path = repository / MODULE.TOOLING_TEST_RELATIVE
    root = repository / MODULE.TOOLING_ROOT_AUTHORITY
    helper.parent.mkdir(parents=True)
    test_path.parent.mkdir(parents=True)
    root.parent.mkdir(parents=True)
    helper.write_text("# bound helper\n")
    test_path.write_text("# focused test\n")
    root.write_text(
        json.dumps(MODULE._expected_tooling_root(), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    _git(
        repository,
        "add",
        "--",
        MODULE.HELPER_RELATIVE,
        MODULE.TOOLING_TEST_RELATIVE,
        MODULE.TOOLING_ROOT_AUTHORITY,
    )
    _git(repository, "commit", "-q", "-m", "tooling")
    _git(
        repository,
        "push",
        "-q",
        "-u",
        "origin",
        "refs/heads/tooling/gate-d-evidence-mirror:refs/heads/tooling/gate-d-evidence-mirror",
    )
    monkeypatch.setattr(MODULE, "RUNNING_HELPER", helper)
    for name in tuple(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name, raising=False)

    report = MODULE._require_tooling_authority()
    assert report["status"] == "GATE_D_EVIDENCE_MIRROR_TOOLING_VALID"
    assert report["base_execution_pin"] == base
    assert report["tooling_head"] == _git(repository, "rev-parse", "HEAD")
    assert report["remote_tooling_head"] == report["tooling_head"]
    assert report["helper_sha256"] == hashlib.sha256(helper.read_bytes()).hexdigest()

    helper.write_text("hostile worktree replacement\n")
    with pytest.raises(MODULE.EvidenceMirrorError, match="not clean"):
        MODULE._require_tooling_authority()
