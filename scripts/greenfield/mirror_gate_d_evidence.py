#!/usr/bin/env python3
"""Verify and append-only mirror one authenticated Gate-D evidence commit."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple

EXPECTED_REPOSITORY = Path("/home/gianl/glm-tpu-gate-d-evidence")
EXPECTED_TOOLING_REPOSITORY = Path("/home/gianl/glm-tpu-gate-d-mirror-tooling")
EXPECTED_COMMON_DIR = Path("/home/gianl/glm-tpu/.git")
EXPECTED_TOOLING_GIT_DIR = (
    EXPECTED_COMMON_DIR / "worktrees/glm-tpu-gate-d-mirror-tooling"
)
EXPECTED_EXPORT_PARENT = Path("/home/gianl/glm-run")
EXPORT_NAME = re.compile(r"gate-d-evidence-mirror\.[A-Za-z0-9]{8}")
DESTINATION_ROOT = "gs://driftbench-dsv4-uc/repos/glm-tpu-gate-d-evidence"
TOOLING_SOURCE_ROOT = (
    "gs://driftbench-dsv4-uc/repos/glm-tpu-gate-d-mirror-tooling"
)
GCLOUD = Path("/snap/bin/gcloud")
RECEIPT_NAME = ".gate-d-evidence-mirror-authority.json"
TOOLING_SOURCE_RECEIPT_NAME = ".gate-d-tooling-source-mirror-authority.json"
BASE_EXECUTION_PIN = "87dc6e3370290ac6378ef6c70bdac5f2a5783059"
EXECUTION_BRANCH = "refs/heads/rewrite/topology-first-decode"
TOOLING_BRANCH = "refs/heads/tooling/gate-d-evidence-mirror"
EXPECTED_ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
TOOLING_ROOT_AUTHORITY = (
    "docs/artifacts/gate-d-evidence-mirror-tooling-root.json"
)
HELPER_RELATIVE = "scripts/greenfield/mirror_gate_d_evidence.py"
TOOLING_TEST_RELATIVE = "tests/greenfield/runtime/test_gate_d_evidence_mirror.py"
TOOLING_ALLOWED_PATHS = frozenset(
    {HELPER_RELATIVE, TOOLING_TEST_RELATIVE, TOOLING_ROOT_AUTHORITY}
)
RUNNING_HELPER = Path(__file__)
MIN_TRACKED_FILES = 50
MAX_OBJECT_BYTES = 64 * 1024 * 1024
PIN = re.compile(r"[0-9a-f]{40}")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_GIT_ENV = {
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_ALLOW_PROTOCOL": "ssh:file",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/usr/bin/ssh -o BatchMode=yes",
    "GIT_TERMINAL_PROMPT": "0",
}
_GCLOUD_ENV = {
    "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/snap/bin:/usr/bin:/bin",
    "PYTHONWARNINGS": "ignore",
}
_VACANT_STDERR = (
    b"ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n"
)


class EvidenceMirrorError(RuntimeError):
    pass


class Blob(NamedTuple):
    mode: int
    oid: str


class LocalObject(NamedTuple):
    name: str
    path: Path
    size: int
    sha256: str
    md5: str


class RemoteObject(NamedTuple):
    name: str
    url: str
    generation: str
    size: int
    md5: str


def _safe_repository(repository: Path) -> Path:
    if repository != EXPECTED_REPOSITORY or repository.is_symlink():
        raise EvidenceMirrorError("unexpected evidence repository")
    try:
        value = repository.lstat()
        resolved = repository.resolve(strict=True)
    except OSError as error:
        raise EvidenceMirrorError("evidence repository is unavailable") from error
    if (
        resolved != EXPECTED_REPOSITORY
        or not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.getuid()
    ):
        raise EvidenceMirrorError("unsafe evidence repository")
    return resolved


def _safe_output_root(output_root: Path) -> Path:
    try:
        value = output_root.lstat()
        resolved = output_root.resolve(strict=True)
        parent = output_root.parent.resolve(strict=True)
    except OSError as error:
        raise EvidenceMirrorError("evidence export root is unavailable") from error
    if (
        parent != EXPECTED_EXPORT_PARENT
        or resolved.parent != EXPECTED_EXPORT_PARENT
        or EXPORT_NAME.fullmatch(resolved.name) is None
        or output_root.is_symlink()
        or not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.getuid()
    ):
        raise EvidenceMirrorError("unsafe evidence export root")
    if any(output_root.iterdir()):
        raise EvidenceMirrorError("evidence export root is not empty")
    return resolved


def _canonical_receipt(raw: bytes) -> tuple[dict[str, object], bytes]:
    if len(raw) > 1_048_576 or not raw.endswith(b"\n") or b"\0" in raw:
        raise EvidenceMirrorError("invalid validator receipt framing")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceMirrorError("validator receipt is not JSON") from error
    if type(value) is not dict:
        raise EvidenceMirrorError("validator receipt is not an object")
    canonical = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    if raw != canonical:
        raise EvidenceMirrorError("validator receipt is not canonical JSON")
    head = value.get("evidence_head")
    if (
        value.get("status") != "GATE_D_EVIDENCE_REF_VALID"
        or value.get("mode") != "committed"
        or not isinstance(head, str)
        or PIN.fullmatch(head) is None
        or value.get("remote_evidence_head") != head
    ):
        raise EvidenceMirrorError("validator receipt authority mismatch")
    return value, canonical


def _git(repository: Path, *arguments: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        env=_GIT_ENV,
    )
    if check and result.returncode:
        raise EvidenceMirrorError(
            f"git {' '.join(arguments)} failed: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )
    return result.stdout


def _expected_tooling_root() -> dict[str, object]:
    return {
        "artifact_kind": "gate_d_evidence_mirror_tooling_root_v1",
        "base_execution_pin": BASE_EXECUTION_PIN,
        "evidence_branch": "evidence/gate-d-topology-first",
        "helper_path": HELPER_RELATIVE,
        "hlo_work": False,
        "origin": EXPECTED_ORIGIN,
        "schema_version": 1,
        "tooling_branch": TOOLING_BRANCH.removeprefix("refs/heads/"),
        "tpu_work": False,
    }


def _canonical_git_path(repository: Path, *arguments: str) -> Path:
    raw = _git(repository, *arguments).decode("utf-8").strip()
    if not raw:
        raise EvidenceMirrorError("empty tooling repository identity")
    try:
        return Path(raw).resolve(strict=True)
    except OSError as error:
        raise EvidenceMirrorError("unresolvable tooling repository identity") from error


def _parse_tooling_remote_refs(raw: bytes) -> dict[str, str]:
    if not raw.isascii() or not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise EvidenceMirrorError("malformed tooling remote refs")
    records: dict[str, str] = {}
    for line in raw[:-1].split(b"\n"):
        if not line or line.count(b"\t") != 1:
            raise EvidenceMirrorError("malformed tooling remote ref record")
        pin_raw, ref_raw = line.split(b"\t")
        pin = pin_raw.decode("ascii")
        ref = ref_raw.decode("ascii")
        if (
            ref not in {EXECUTION_BRANCH, TOOLING_BRANCH}
            or PIN.fullmatch(pin) is None
            or ref in records
        ):
            raise EvidenceMirrorError("unexpected tooling remote ref record")
        records[ref] = pin
    if set(records) != {EXECUTION_BRANCH, TOOLING_BRANCH}:
        raise EvidenceMirrorError("tooling remote refs are incomplete")
    return records


def _require_tooling_config(repository: Path) -> None:
    keys = _git(repository, "config", "--local", "--name-only", "--list").decode().split()
    fixed = {
        "core.repositoryformatversion",
        "core.filemode",
        "core.bare",
        "core.logallrefupdates",
        "remote.origin.url",
        "remote.origin.fetch",
    }
    suffixes = (".remote", ".merge", ".vscode-merge-base")
    unexpected = sorted(
        key
        for key in keys
        if key not in fixed
        and not (key.startswith("branch.") and key.endswith(suffixes))
    )
    if unexpected:
        raise EvidenceMirrorError(f"unexpected tooling Git config: {unexpected}")
    expected = {
        "core.repositoryformatversion": ["0"],
        "core.filemode": ["true"],
        "core.bare": ["false"],
        "core.logallrefupdates": ["true"],
        "remote.origin.url": [EXPECTED_ORIGIN],
        "remote.origin.fetch": ["+refs/heads/*:refs/remotes/origin/*"],
        "branch.tooling/gate-d-evidence-mirror.remote": ["origin"],
        "branch.tooling/gate-d-evidence-mirror.merge": [TOOLING_BRANCH],
    }
    for key, wanted in expected.items():
        actual = _git(repository, "config", "--local", "--get-all", key).decode().splitlines()
        if actual != wanted:
            raise EvidenceMirrorError(f"tooling Git config mismatch: {key}")


def _require_complete_tooling_worktree(repository: Path) -> int:
    if _git(repository, "status", "--porcelain=v1"):
        raise EvidenceMirrorError("tooling worktree is not clean")
    if _git(repository, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise EvidenceMirrorError("shallow tooling repository is forbidden")
    if _git(repository, "replace", "-l"):
        raise EvidenceMirrorError("tooling replacement refs are forbidden")
    for key in ("core.sparseCheckout", "core.sparseCheckoutCone"):
        value = _git(
            repository,
            "config",
            "--local",
            "--bool",
            "--get",
            key,
            check=False,
        ).strip()
        if value not in {b"", b"false"}:
            raise EvidenceMirrorError(f"sparse tooling worktree is forbidden: {key}")
    records = _git(repository, "ls-files", "-v", "-z").decode("utf-8").split("\0")
    if records and records[-1] == "":
        records.pop()
    if len(records) < MIN_TRACKED_FILES:
        raise EvidenceMirrorError("tooling tracked-file floor is not met")
    for record in records:
        if not record.startswith("H "):
            raise EvidenceMirrorError("nonordinary tooling index flag is forbidden")
        path = repository / record[2:]
        try:
            value = path.lstat()
        except OSError as error:
            raise EvidenceMirrorError("tracked tooling path is absent") from error
        if not stat.S_ISREG(value.st_mode):
            raise EvidenceMirrorError("tracked tooling path is not regular")
    return len(records)


def _require_tooling_history(repository: Path, head: str) -> tuple[list[str], str]:
    merge_base = _git(repository, "merge-base", BASE_EXECUTION_PIN, head).decode().strip()
    if merge_base != BASE_EXECUTION_PIN:
        raise EvidenceMirrorError("tooling branch is not rooted at the execution pin")
    commits = _git(
        repository, "rev-list", "--reverse", f"{BASE_EXECUTION_PIN}..{head}"
    ).decode().split()
    if not commits:
        raise EvidenceMirrorError("tooling branch has no correction commit")
    changed: set[str] = set()
    root_additions = 0
    for commit in commits:
        parents = _git(repository, "show", "-s", "--format=%P", commit).decode().split()
        if len(parents) != 1:
            raise EvidenceMirrorError("merge/root tooling commits are forbidden")
        fields = _git(
            repository,
            "diff-tree",
            "--no-commit-id",
            "--no-renames",
            "--name-status",
            "-r",
            "-z",
            commit,
        ).decode("utf-8").split("\0")
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) % 2:
            raise EvidenceMirrorError("malformed tooling history")
        for offset in range(0, len(fields), 2):
            status, path = fields[offset : offset + 2]
            if status not in {"A", "M"} or path not in TOOLING_ALLOWED_PATHS:
                raise EvidenceMirrorError(f"forbidden tooling history path: {path}")
            if path == TOOLING_ROOT_AUTHORITY:
                if status != "A":
                    raise EvidenceMirrorError("tooling root authority is immutable")
                root_additions += 1
            mode = _git(repository, "ls-tree", commit, "--", path).decode().split()
            if not mode or mode[0] != "100644":
                raise EvidenceMirrorError("tooling changes must be mode-100644 blobs")
            changed.add(path)
    if changed != TOOLING_ALLOWED_PATHS or root_additions != 1:
        raise EvidenceMirrorError("tooling history path set is incomplete")
    tree = _git(repository, "rev-parse", f"{head}^{{tree}}").decode().strip()
    if PIN.fullmatch(tree) is None:
        raise EvidenceMirrorError("invalid tooling tree identity")
    return commits, tree


def _read_bound_helper(repository: Path, head: str) -> tuple[str, str]:
    expected = (repository / HELPER_RELATIVE).resolve(strict=True)
    try:
        actual = RUNNING_HELPER.resolve(strict=True)
        value = RUNNING_HELPER.lstat()
        descriptor = os.open(RUNNING_HELPER, os.O_RDONLY | _NOFOLLOW)
    except OSError as error:
        raise EvidenceMirrorError("running tooling helper is unavailable") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > 2 * 1024 * 1024:
                raise EvidenceMirrorError("running tooling helper is oversized")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        actual != expected
        or RUNNING_HELPER.is_symlink()
        or not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.getuid()
        or value.st_nlink != 1
        or _descriptor_identity(before) != _descriptor_identity(after)
    ):
        raise EvidenceMirrorError("unsafe running tooling helper identity")
    raw = b"".join(chunks)
    committed = _git(repository, "show", f"{head}:{HELPER_RELATIVE}")
    if raw != committed:
        raise EvidenceMirrorError("running helper differs from tooling commit")
    listing = _git(repository, "ls-tree", head, "--", HELPER_RELATIVE).decode().split()
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    blob = digest.hexdigest()
    if len(listing) < 3 or listing[0] != "100644" or listing[1] != "blob" or listing[2] != blob:
        raise EvidenceMirrorError("running helper Git blob mismatch")
    return blob, hashlib.sha256(raw).hexdigest()


def _require_tooling_authority() -> dict[str, object]:
    hostile = sorted(name for name in os.environ if name.startswith("GIT_"))
    if hostile:
        raise EvidenceMirrorError(f"ambient Git environment is forbidden: {hostile}")
    repository = EXPECTED_TOOLING_REPOSITORY
    try:
        value = repository.lstat()
    except OSError as error:
        raise EvidenceMirrorError("tooling repository is absent") from error
    if (
        repository.is_symlink()
        or not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.getuid()
        or repository.resolve(strict=True) != EXPECTED_TOOLING_REPOSITORY
    ):
        raise EvidenceMirrorError("unsafe tooling repository identity")
    identities = {
        "toplevel": _canonical_git_path(
            repository, "rev-parse", "--path-format=absolute", "--show-toplevel"
        ),
        "git_dir": _canonical_git_path(
            repository, "rev-parse", "--path-format=absolute", "--git-dir"
        ),
        "common_dir": _canonical_git_path(
            repository, "rev-parse", "--path-format=absolute", "--git-common-dir"
        ),
    }
    if identities != {
        "toplevel": EXPECTED_TOOLING_REPOSITORY,
        "git_dir": EXPECTED_TOOLING_GIT_DIR,
        "common_dir": EXPECTED_COMMON_DIR,
    }:
        raise EvidenceMirrorError("tooling repository administrative identity mismatch")
    _require_tooling_config(repository)
    branch = _git(repository, "symbolic-ref", "-q", "HEAD").decode().strip()
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    if branch != TOOLING_BRANCH or PIN.fullmatch(head) is None:
        raise EvidenceMirrorError("tooling branch identity mismatch")
    if _git(repository, "rev-parse", EXECUTION_BRANCH).decode().strip() != BASE_EXECUTION_PIN:
        raise EvidenceMirrorError("local execution ref drift")
    remote_refs = _parse_tooling_remote_refs(
        _git(repository, "ls-remote", "--refs", "origin", EXECUTION_BRANCH, TOOLING_BRANCH)
    )
    if remote_refs[EXECUTION_BRANCH] != BASE_EXECUTION_PIN or remote_refs[TOOLING_BRANCH] != head:
        raise EvidenceMirrorError("remote tooling/execution ref drift")
    commits, tree = _require_tooling_history(repository, head)
    tracked_files = _require_complete_tooling_worktree(repository)
    root_raw = _git(repository, "show", f"{head}:{TOOLING_ROOT_AUTHORITY}")
    try:
        root = json.loads(root_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceMirrorError("tooling root authority is not JSON") from error
    expected_root = _expected_tooling_root()
    canonical_root = (
        json.dumps(expected_root, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    if (
        type(root) is not dict
        or root != expected_root
        or root_raw != canonical_root
        or type(root.get("schema_version")) is not int
    ):
        raise EvidenceMirrorError("tooling root authority mismatch")
    helper_blob, helper_sha256 = _read_bound_helper(repository, head)
    return {
        "base_execution_pin": BASE_EXECUTION_PIN,
        "commits": commits,
        "helper_git_blob": helper_blob,
        "helper_sha256": helper_sha256,
        "remote_tooling_head": remote_refs[TOOLING_BRANCH],
        "root_authority_sha256": hashlib.sha256(root_raw).hexdigest(),
        "status": "GATE_D_EVIDENCE_MIRROR_TOOLING_VALID",
        "tooling_branch": TOOLING_BRANCH,
        "tooling_head": head,
        "tooling_tree": tree,
        "tracked_files": tracked_files,
    }


def _safe_tree_path(raw: bytes) -> str:
    try:
        path = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceMirrorError("non-UTF-8 path in evidence tree") from error
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or str(PurePosixPath(path)) != path
    ):
        raise EvidenceMirrorError("unsafe path in evidence tree")
    return path


def _catalogue(repository: Path, head: str) -> tuple[dict[str, Blob], bytes]:
    raw = _git(repository, "ls-tree", "-r", "-z", "--full-tree", head)
    records = raw.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    catalogue: dict[str, Blob] = {}
    for record in records:
        metadata, separator, path_raw = record.partition(b"\t")
        fields = metadata.split(b" ")
        if separator != b"\t" or len(fields) != 3:
            raise EvidenceMirrorError("malformed Git tree catalogue")
        mode_raw, kind, oid_raw = fields
        if mode_raw not in {b"100644", b"100755"} or kind != b"blob":
            raise EvidenceMirrorError("non-regular object in evidence tree")
        try:
            oid = oid_raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise EvidenceMirrorError("invalid object id in evidence tree") from error
        if PIN.fullmatch(oid) is None:
            raise EvidenceMirrorError("invalid object id in evidence tree")
        path = _safe_tree_path(path_raw)
        if path == RECEIPT_NAME or path in catalogue:
            raise EvidenceMirrorError("duplicate/reserved evidence tree path")
        catalogue[path] = Blob(mode=int(mode_raw, 8), oid=oid)
    if len(catalogue) < MIN_TRACKED_FILES:
        raise EvidenceMirrorError("evidence tree tracked-file floor is not met")
    return catalogue, raw


def _write_archive(repository: Path, head: str, archive: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
    try:
        descriptor = os.open(archive, flags, 0o600)
    except OSError as error:
        raise EvidenceMirrorError("cannot create exclusive evidence archive") from error
    try:
        output = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = -1
        with output:
            result = subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(repository),
                    "archive",
                    "--format=tar",
                    head,
                ],
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
                env=_GIT_ENV,
            )
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if result.returncode:
        raise EvidenceMirrorError(
            "authenticated evidence archive failed: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )


def _mkdirs(root: Path, relative_parent: PurePosixPath) -> None:
    current = root
    for part in relative_parent.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            value = current.lstat()
            if not stat.S_ISDIR(value.st_mode) or current.is_symlink():
                raise EvidenceMirrorError("unsafe directory in evidence export")


def _extract_archive(archive: Path, tree: Path, catalogue: dict[str, Blob]) -> None:
    tree.mkdir(mode=0o700)
    seen_members: set[str] = set()
    with tarfile.open(archive, mode="r:") as stream:
        for member in stream:
            path = _safe_tree_path(member.name.encode("utf-8"))
            if path in seen_members:
                raise EvidenceMirrorError("duplicate member in evidence archive")
            seen_members.add(path)
            relative = PurePosixPath(path)
            target = tree.joinpath(*relative.parts)
            if member.isdir():
                _mkdirs(tree, relative)
                continue
            if not member.isreg() or path not in catalogue:
                raise EvidenceMirrorError("unexpected member in evidence archive")
            expected = catalogue[path]
            _mkdirs(tree, relative.parent)
            source = stream.extractfile(member)
            if source is None:
                raise EvidenceMirrorError("archive member has no payload")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
            try:
                descriptor = os.open(target, flags, expected.mode & 0o777)
            except OSError as error:
                raise EvidenceMirrorError("cannot create exclusive export file") from error
            written = 0
            try:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    offset = 0
                    while offset < len(chunk):
                        count = os.write(descriptor, chunk[offset:])
                        if count <= 0:
                            raise EvidenceMirrorError("short evidence export write")
                        offset += count
                        written += count
                os.fchmod(descriptor, expected.mode & 0o777)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                source.close()
            if written != member.size:
                raise EvidenceMirrorError("evidence archive size mismatch")


def _blob_oid(path: Path) -> str:
    value = path.stat(follow_symlinks=False)
    digest = hashlib.sha1()
    digest.update(f"blob {value.st_size}\0".encode("ascii"))
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_tree(tree: Path, catalogue: dict[str, Blob]) -> None:
    actual: set[str] = set()
    for directory, names, files in os.walk(tree, followlinks=False):
        base = Path(directory)
        for name in names:
            path = base / name
            if path.is_symlink() or not stat.S_ISDIR(path.lstat().st_mode):
                raise EvidenceMirrorError("unsafe directory in verified export")
        for name in files:
            path = base / name
            relative = path.relative_to(tree).as_posix()
            value = path.lstat()
            if not stat.S_ISREG(value.st_mode) or path.is_symlink():
                raise EvidenceMirrorError("non-regular file in verified export")
            actual.add(relative)
    if actual != set(catalogue):
        raise EvidenceMirrorError("export path set differs from Git tree")
    for relative, expected in catalogue.items():
        path = tree.joinpath(*PurePosixPath(relative).parts)
        value = path.stat(follow_symlinks=False)
        if stat.S_IMODE(value.st_mode) != expected.mode & 0o777:
            raise EvidenceMirrorError("export mode differs from Git tree")
        if _blob_oid(path) != expected.oid:
            raise EvidenceMirrorError("export blob differs from Git tree")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        count = os.write(descriptor, payload[offset:])
        if count <= 0:
            raise EvidenceMirrorError("short authority receipt write")
        offset += count


def _write_receipt(tree: Path, canonical: bytes) -> str:
    destination = tree / RECEIPT_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as error:
        raise EvidenceMirrorError("cannot create exclusive authority receipt") from error
    try:
        _write_all(descriptor, canonical)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        descriptor = os.open(destination, os.O_RDONLY | _NOFOLLOW)
        source = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        with source:
            observed = source.read(len(canonical) + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if observed != canonical:
        raise EvidenceMirrorError("authority receipt readback mismatch")
    parsed, recanonical = _canonical_receipt(observed)
    if recanonical != observed or parsed.get("evidence_head") is None:
        raise EvidenceMirrorError("authority receipt verification failed")
    return hashlib.sha256(observed).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_gcloud(arguments: list[str], *, stdin=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(GCLOUD), *arguments],
        check=False,
        stdin=stdin,
        capture_output=True,
        env=_GCLOUD_ENV,
    )


def _hash_descriptor(descriptor: int) -> tuple[int, str, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        sha256.update(chunk)
        md5.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return size, sha256.hexdigest(), base64.b64encode(md5.digest()).decode("ascii")


def _descriptor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _local_object(name: str, path: Path) -> LocalObject:
    try:
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
    except OSError as error:
        raise EvidenceMirrorError("cannot open local publication object") from error
    try:
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_uid != os.getuid()
            or value.st_nlink != 1
            or value.st_size > MAX_OBJECT_BYTES
        ):
            raise EvidenceMirrorError("unsafe local publication object")
        size, sha256, md5 = _hash_descriptor(descriptor)
    finally:
        os.close(descriptor)
    if size != value.st_size:
        raise EvidenceMirrorError("local publication object size drift")
    return LocalObject(name=name, path=path, size=size, sha256=sha256, md5=md5)


def _verify_archive_descriptor(
    descriptor: int, catalogue: dict[str, Blob]
) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    duplicate = os.dup(descriptor)
    seen: set[str] = set()
    try:
        with os.fdopen(duplicate, "rb", closefd=True) as source:
            duplicate = -1
            with tarfile.open(fileobj=source, mode="r:") as stream:
                for member in stream:
                    path = _safe_tree_path(member.name.encode("utf-8"))
                    if member.isdir():
                        continue
                    if not member.isreg() or path not in catalogue or path in seen:
                        raise EvidenceMirrorError(
                            "publication archive contains an unexpected member"
                        )
                    payload = stream.extractfile(member)
                    if payload is None:
                        raise EvidenceMirrorError(
                            "publication archive member has no payload"
                        )
                    digest = hashlib.sha1()
                    digest.update(f"blob {member.size}\0".encode("ascii"))
                    size = 0
                    with payload:
                        for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                            size += len(chunk)
                            digest.update(chunk)
                    if size != member.size or digest.hexdigest() != catalogue[path].oid:
                        raise EvidenceMirrorError(
                            "publication archive member differs from Git tree"
                        )
                    seen.add(path)
    finally:
        if duplicate >= 0:
            os.close(duplicate)
        os.lseek(descriptor, 0, os.SEEK_SET)
    if seen != set(catalogue):
        raise EvidenceMirrorError("publication archive path set differs from Git tree")


def _verify_local_archive(local: LocalObject, catalogue: dict[str, Blob]) -> None:
    descriptor = os.open(local.path, os.O_RDONLY | _NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or _hash_descriptor(descriptor) != (local.size, local.sha256, local.md5)
        ):
            raise EvidenceMirrorError("local archive differs from sealed identity")
        _verify_archive_descriptor(descriptor, catalogue)
        after = os.fstat(descriptor)
        if (
            _descriptor_identity(after) != _descriptor_identity(before)
            or _hash_descriptor(descriptor) != (local.size, local.sha256, local.md5)
        ):
            raise EvidenceMirrorError("local archive changed during verification")
    finally:
        os.close(descriptor)


def _remote_name(url: str) -> str:
    prefix = "gs://driftbench-dsv4-uc/"
    if not url.startswith(prefix):
        raise EvidenceMirrorError("remote object escapes the approved bucket")
    return url.removeprefix(prefix)


def _parse_describe(raw: bytes, expected: LocalObject, url: str) -> RemoteObject:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceMirrorError("remote object description is not JSON") from error
    generation = value.get("generation") if type(value) is dict else None
    size = value.get("size") if type(value) is dict else None
    if (
        type(value) is not dict
        or value.get("name") != _remote_name(url)
        or not isinstance(generation, str)
        or not generation.isdigit()
        or int(generation) <= 0
        or type(size) is not int
        or size != expected.size
        or value.get("md5_hash") != expected.md5
    ):
        raise EvidenceMirrorError("remote object description mismatch")
    return RemoteObject(
        name=expected.name,
        url=url,
        generation=generation,
        size=size,
        md5=expected.md5,
    )


def _describe_remote(local: LocalObject, url: str) -> RemoteObject:
    result = _run_gcloud(
        [
            "storage",
            "objects",
            "describe",
            url,
            "--format=json(name,generation,size,crc32c_hash,md5_hash)",
        ]
    )
    if result.returncode:
        raise EvidenceMirrorError(
            "remote object describe failed: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )
    return _parse_describe(result.stdout, local, url)


def _download_and_verify(remote: RemoteObject, local: LocalObject) -> None:
    result = _run_gcloud(
        [
            "storage",
            "cp",
            f"--if-generation-match={remote.generation}",
            "--quiet",
            remote.url,
            "-",
        ]
    )
    if result.returncode:
        raise EvidenceMirrorError(
            "generation-qualified remote replay failed: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )
    digest = hashlib.sha256(result.stdout).hexdigest()
    md5 = base64.b64encode(
        hashlib.md5(result.stdout, usedforsecurity=False).digest()
    ).decode("ascii")
    if len(result.stdout) != local.size or digest != local.sha256 or md5 != local.md5:
        raise EvidenceMirrorError("generation-qualified remote replay mismatch")


def _publish_one(
    local: LocalObject,
    destination: str,
    *,
    catalogue: dict[str, Blob] | None = None,
) -> RemoteObject:
    url = f"{destination}/{local.name}"
    try:
        descriptor = os.open(local.path, os.O_RDONLY | _NOFOLLOW)
    except OSError as error:
        raise EvidenceMirrorError("cannot open retained publication descriptor") from error
    source = os.fdopen(descriptor, "rb", closefd=True)
    descriptor = -1
    try:
        before = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or _hash_descriptor(source.fileno())
            != (local.size, local.sha256, local.md5)
        ):
            raise EvidenceMirrorError("publication descriptor differs from verified bytes")
        if catalogue is not None:
            _verify_archive_descriptor(source.fileno(), catalogue)
        result = _run_gcloud(
            [
                "storage",
                "cp",
                "--if-generation-match=0",
                f"--content-md5={local.md5}",
                "--quiet",
                "-",
                url,
            ],
            stdin=source,
        )
        if result.returncode:
            raise EvidenceMirrorError(
                "generation-zero remote creation failed: "
                + result.stderr.decode("utf-8", "replace")[-1000:]
            )
        remote = _describe_remote(local, url)
        _download_and_verify(remote, local)
        after = os.fstat(source.fileno())
        if (
            _descriptor_identity(after) != _descriptor_identity(before)
            or _hash_descriptor(source.fileno())
            != (local.size, local.sha256, local.md5)
        ):
            raise EvidenceMirrorError("publication descriptor changed during upload")
        return remote
    finally:
        source.close()


def _parse_listing(raw: bytes, destination: str) -> dict[str, RemoteObject]:
    try:
        values = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceMirrorError("remote prefix listing is not JSON") from error
    if type(values) is not list:
        raise EvidenceMirrorError("remote prefix listing is not an array")
    expected_prefix = {"type": "prefix", "url": destination + "/"}
    if len(values) != 3 or values[0] != expected_prefix:
        raise EvidenceMirrorError("remote prefix marker or object count mismatch")
    records: dict[str, RemoteObject] = {}
    for value in values[1:]:
        metadata = value.get("metadata") if type(value) is dict else None
        url_with_generation = value.get("url") if type(value) is dict else None
        if (
            type(value) is not dict
            or set(value) != {"metadata", "type", "url"}
            or value.get("type") != "cloud_object"
            or type(metadata) is not dict
            or not isinstance(url_with_generation, str)
            or url_with_generation.count("#") != 1
        ):
            raise EvidenceMirrorError("malformed remote prefix record")
        url, generation_suffix = url_with_generation.rsplit("#", 1)
        generation = metadata.get("generation")
        size = metadata.get("size")
        name = metadata.get("name")
        md5 = metadata.get("md5Hash")
        if (
            not isinstance(name, str)
            or not isinstance(generation, str)
            or generation != generation_suffix
            or not generation.isdigit()
            or int(generation) <= 0
            or not isinstance(size, str)
            or not size.isdigit()
            or not isinstance(md5, str)
            or not md5
            or url != f"gs://driftbench-dsv4-uc/{name}"
            or url in records
        ):
            raise EvidenceMirrorError("invalid or duplicate remote prefix record")
        records[url] = RemoteObject(
            name=Path(name).name,
            url=url,
            generation=generation,
            size=int(size),
            md5=md5,
        )
    if len(records) != 2:
        raise EvidenceMirrorError("remote prefix object identities are not unique")
    return records


def _list_prefix(
    destination: str, *, soft_deleted: bool = False
) -> dict[str, RemoteObject]:
    arguments = ["storage", "ls", "--json", "--recursive"]
    arguments.append("--soft-deleted" if soft_deleted else "--all-versions")
    arguments.append(destination + "/")
    result = _run_gcloud(arguments)
    if result.returncode:
        if result.stdout == b"" and result.stderr == _VACANT_STDERR:
            return {}
        raise EvidenceMirrorError(
            "remote prefix vacancy check failed: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )
    return _parse_listing(result.stdout, destination)


def _verify_remote_catalogue(
    records: dict[str, RemoteObject],
    local_objects: tuple[LocalObject, ...],
    destination: str,
) -> tuple[RemoteObject, ...]:
    expected_urls = {f"{destination}/{local.name}" for local in local_objects}
    if set(records) != expected_urls:
        raise EvidenceMirrorError("remote prefix is partial or contains extra objects")
    verified: list[RemoteObject] = []
    for local in local_objects:
        url = f"{destination}/{local.name}"
        remote = records[url]
        if (
            remote.name != local.name
            or remote.size != local.size
            or remote.md5 != local.md5
        ):
            raise EvidenceMirrorError("remote prefix catalogue differs from local bytes")
        _download_and_verify(remote, local)
        verified.append(remote)
    return tuple(verified)


def _publish_objects(
    local_objects: tuple[LocalObject, ...],
    destination: str,
    *,
    catalogue: dict[str, Blob] | None = None,
    require_existing_replay: bool = False,
) -> tuple[tuple[RemoteObject, ...], bool]:
    if catalogue is not None:
        _verify_local_archive(local_objects[0], catalogue)
    if _list_prefix(destination, soft_deleted=True):
        raise EvidenceMirrorError("remote prefix has soft-deleted prior objects")
    existing = _list_prefix(destination)
    if existing:
        verified = _verify_remote_catalogue(existing, local_objects, destination)
        if _list_prefix(destination, soft_deleted=True):
            raise EvidenceMirrorError("remote prefix gained soft-deleted objects")
        if _list_prefix(destination) != existing:
            raise EvidenceMirrorError("remote prefix changed during replay")
        return verified, False
    if require_existing_replay:
        raise EvidenceMirrorError("replay-only remote prefix is vacant")
    for local in local_objects:
        _publish_one(
            local,
            destination,
            catalogue=catalogue if local.name == "repository.tar" else None,
        )
    final = _list_prefix(destination)
    if _list_prefix(destination, soft_deleted=True):
        raise EvidenceMirrorError("remote prefix gained soft-deleted objects")
    return _verify_remote_catalogue(final, local_objects, destination), True


def _remote_report(remote: RemoteObject, local: LocalObject) -> dict[str, object]:
    return {
        "generation": remote.generation,
        "md5": remote.md5,
        "name": remote.name,
        "sha256": local.sha256,
        "size": remote.size,
        "url": remote.url,
    }


def _tooling_source_destination(head: str) -> str:
    if (
        TOOLING_SOURCE_ROOT
        != "gs://driftbench-dsv4-uc/repos/glm-tpu-gate-d-mirror-tooling"
        or PIN.fullmatch(head) is None
    ):
        raise EvidenceMirrorError("tooling source destination authority mismatch")
    return f"{TOOLING_SOURCE_ROOT}/{head}"


def _create_tooling_source_output_root(output_root: Path, head: str) -> Path:
    expected = EXPECTED_EXPORT_PARENT / f"gate-d-tooling-source-mirror.{head[:8]}"
    if output_root != expected or output_root.is_symlink():
        raise EvidenceMirrorError("unexpected tooling source export path")
    try:
        parent = EXPECTED_EXPORT_PARENT.resolve(strict=True)
        parent_value = EXPECTED_EXPORT_PARENT.lstat()
    except OSError as error:
        raise EvidenceMirrorError("tooling source export parent is unavailable") from error
    if (
        parent != EXPECTED_EXPORT_PARENT
        or not stat.S_ISDIR(parent_value.st_mode)
        or parent_value.st_uid != os.getuid()
    ):
        raise EvidenceMirrorError("unsafe tooling source export parent")
    parent_fd = os.open(
        EXPECTED_EXPORT_PARENT,
        os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        try:
            os.mkdir(output_root.name, 0o700, dir_fd=parent_fd)
        except OSError as error:
            raise EvidenceMirrorError(
                "cannot exclusively create tooling source export"
            ) from error
        os.fsync(parent_fd)
        directory_fd = os.open(
            output_root.name,
            os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(directory_fd, 0o700)
            os.fsync(directory_fd)
            value = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(value.st_mode)
                or stat.S_IMODE(value.st_mode) != 0o700
                or value.st_uid != os.getuid()
                or value.st_nlink < 2
            ):
                raise EvidenceMirrorError("unsafe tooling source export identity")
        finally:
            os.close(directory_fd)
    finally:
        os.close(parent_fd)
    if output_root.resolve(strict=True) != expected or any(output_root.iterdir()):
        raise EvidenceMirrorError("tooling source export postcondition failed")
    return output_root


def _write_exclusive_verified(path: Path, raw: bytes) -> None:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise EvidenceMirrorError("cannot create tooling source authority") from error
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            b"".join(chunks) != raw
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or _descriptor_identity(before) != _descriptor_identity(after)
        ):
            raise EvidenceMirrorError("tooling source authority readback mismatch")
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _expected_tooling_source_receipt(
    tooling_authority: dict[str, object],
    archive: LocalObject,
    destination: str,
    catalogue_sha256: str,
) -> dict[str, object]:
    return {
        "archive": {
            "md5": archive.md5,
            "sha256": archive.sha256,
            "size": archive.size,
        },
        "artifact_kind": "gate_d_tooling_source_mirror_authority_v1",
        "bucket": "gs://driftbench-dsv4-uc",
        "catalogue_sha256": catalogue_sha256,
        "destination": destination,
        "object_names": ["repository.tar", TOOLING_SOURCE_RECEIPT_NAME],
        "schema_version": 1,
        "status": "TOOLING_SOURCE_MIRROR_AUTHORITY",
        "tooling_authority": copy.deepcopy(tooling_authority),
        "tpu_work": False,
    }


def _validate_tooling_source_receipt(
    raw: bytes,
    tooling_authority: dict[str, object],
    archive: LocalObject,
    destination: str,
    catalogue_sha256: str,
) -> None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceMirrorError("tooling source authority is not JSON") from error
    expected = _expected_tooling_source_receipt(
        tooling_authority, archive, destination, catalogue_sha256
    )
    canonical = (
        json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    archive_value = value.get("archive") if type(value) is dict else None
    authority_value = (
        value.get("tooling_authority") if type(value) is dict else None
    )
    if (
        type(value) is not dict
        or value != expected
        or raw != canonical
        or type(value.get("schema_version")) is not int
        or type(value.get("tpu_work")) is not bool
        or type(archive_value) is not dict
        or type(archive_value.get("size")) is not int
        or type(authority_value) is not dict
        or value.get("destination") != _tooling_source_destination(
            str(tooling_authority.get("tooling_head"))
        )
        or value.get("object_names")
        != ["repository.tar", TOOLING_SOURCE_RECEIPT_NAME]
    ):
        raise EvidenceMirrorError("tooling source authority contract mismatch")


def mirror_tooling_source(output_root: Path) -> dict[str, object]:
    tooling_authority = _require_tooling_authority()
    head = tooling_authority.get("tooling_head")
    tree = tooling_authority.get("tooling_tree")
    if (
        not isinstance(head, str)
        or PIN.fullmatch(head) is None
        or not isinstance(tree, str)
        or PIN.fullmatch(tree) is None
    ):
        raise EvidenceMirrorError("tooling source commit authority mismatch")
    output_root = _create_tooling_source_output_root(output_root, head)
    destination = _tooling_source_destination(head)
    catalogue, catalogue_raw = _catalogue(EXPECTED_TOOLING_REPOSITORY, head)
    archive_path = output_root / "repository.tar"
    _write_archive(EXPECTED_TOOLING_REPOSITORY, head, archive_path)
    archive = _local_object("repository.tar", archive_path)
    _verify_local_archive(archive, catalogue)
    catalogue_sha256 = hashlib.sha256(catalogue_raw).hexdigest()
    receipt_value = _expected_tooling_source_receipt(
        tooling_authority, archive, destination, catalogue_sha256
    )
    receipt_raw = (
        json.dumps(receipt_value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    _validate_tooling_source_receipt(
        receipt_raw,
        tooling_authority,
        archive,
        destination,
        catalogue_sha256,
    )
    receipt_path = output_root / TOOLING_SOURCE_RECEIPT_NAME
    _write_exclusive_verified(receipt_path, receipt_raw)
    receipt = _local_object(TOOLING_SOURCE_RECEIPT_NAME, receipt_path)
    local_objects = (archive, receipt)
    if [item.name for item in local_objects] != [
        "repository.tar",
        TOOLING_SOURCE_RECEIPT_NAME,
    ]:
        raise EvidenceMirrorError("tooling source object names changed")
    remote_objects, published = _publish_objects(
        local_objects, destination, catalogue=catalogue
    )
    if published is not True:
        raise EvidenceMirrorError("tooling source mirror was not newly published")
    return {
        "archive_sha256": archive.sha256,
        "catalogue_sha256": catalogue_sha256,
        "destination": destination,
        "published": published,
        "receipt_sha256": receipt.sha256,
        "remote_objects": [
            _remote_report(remote, local)
            for remote, local in zip(remote_objects, local_objects, strict=True)
        ],
        "status": "GATE_D_TOOLING_SOURCE_MIRRORED",
        "tooling_authority": tooling_authority,
        "tooling_head": head,
        "tooling_tree": tree,
        "tracked_files": len(catalogue),
    }


def mirror(
    repository: Path,
    output_root: Path,
    receipt_raw: bytes,
    *,
    tooling_authority: dict[str, object] | None = None,
    require_existing_replay: bool = False,
) -> dict[str, object]:
    repository = _safe_repository(repository)
    output_root = _safe_output_root(output_root)
    receipt, canonical = _canonical_receipt(receipt_raw)
    head = receipt["evidence_head"]
    assert isinstance(head, str)
    catalogue, catalogue_raw = _catalogue(repository, head)
    archive = output_root / "repository.tar"
    tree = output_root / "tree"
    _write_archive(repository, head, archive)
    _extract_archive(archive, tree, catalogue)
    _verify_tree(tree, catalogue)
    receipt_sha256 = _write_receipt(tree, canonical)
    _fsync_directory(tree)
    _fsync_directory(output_root)
    destination = f"{DESTINATION_ROOT}/{head}"
    local_objects = (
        _local_object("repository.tar", archive),
        _local_object(RECEIPT_NAME, tree / RECEIPT_NAME),
    )
    if local_objects[1].sha256 != receipt_sha256:
        raise EvidenceMirrorError("authority receipt changed before publication")
    remote_objects, published = _publish_objects(
        local_objects,
        destination,
        catalogue=catalogue,
        require_existing_replay=require_existing_replay,
    )
    if require_existing_replay and published:
        raise EvidenceMirrorError("replay-only mirror unexpectedly published")
    return {
        "archive_sha256": local_objects[0].sha256,
        "catalogue_sha256": hashlib.sha256(catalogue_raw).hexdigest(),
        "destination": destination,
        "evidence_head": head,
        "published": published,
        "replay_only": require_existing_replay,
        "receipt_sha256": receipt_sha256,
        "remote_objects": [
            _remote_report(remote, local)
            for remote, local in zip(remote_objects, local_objects, strict=True)
        ],
        "status": "GATE_D_EVIDENCE_MIRRORED",
        "tooling_authority": tooling_authority,
        "tracked_files": len(catalogue),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-existing-replay", action="store_true")
    parser.add_argument("--mirror-tooling-source", action="store_true")
    arguments = parser.parse_args()
    if arguments.mirror_tooling_source:
        if arguments.repository is not None or arguments.require_existing_replay:
            parser.error(
                "tooling source mode forbids evidence repository/replay arguments"
            )
        print(
            json.dumps(
                mirror_tooling_source(arguments.output_root),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if arguments.repository is None:
        parser.error("evidence mirror mode requires --repository")
    tooling_authority = _require_tooling_authority()
    receipt = sys.stdin.buffer.read(1_048_577)
    report = mirror(
        arguments.repository,
        arguments.output_root,
        receipt,
        tooling_authority=tooling_authority,
        require_existing_replay=arguments.require_existing_replay,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceMirrorError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        raise SystemExit(2) from error
