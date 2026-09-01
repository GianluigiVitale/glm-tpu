#!/usr/bin/env python3
"""Fail-closed authority for Gate-D evidence commits off the execution branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

EXECUTION_BRANCH = "refs/heads/rewrite/topology-first-decode"
REMOTE_EXECUTION_BRANCH = "refs/heads/rewrite/topology-first-decode"
EVIDENCE_BRANCH = "refs/heads/evidence/gate-d-topology-first"
REMOTE_EVIDENCE_BRANCH = "refs/heads/evidence/gate-d-topology-first"
EXPECTED_REPOSITORY = Path("/home/gianl/glm-tpu-gate-d-evidence")
EXPECTED_COMMON_DIR = Path("/home/gianl/glm-tpu/.git")
EXPECTED_GIT_DIR = EXPECTED_COMMON_DIR / "worktrees/glm-tpu-gate-d-evidence"
EXPECTED_ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
ROOT_AUTHORITY = "docs/artifacts/gate-d-evidence-ref-root.json"
ARTIFACT_PREFIX = "docs/artifacts/gate-d-"
MIN_TRACKED_FILES = 50
PIN_PATTERN = re.compile(r"[0-9a-f]{40}")

_GIT_ENV = {
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "GIT_ALLOW_PROTOCOL": "ssh:file",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/usr/bin/ssh -o BatchMode=yes",
    "GIT_TERMINAL_PROMPT": "0",
}
_FORBIDDEN_GIT_ENV_NAMES = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_ALLOW_PROTOCOL",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_GRAFT_FILE",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PROTOCOL_FROM_USER",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_WORK_TREE",
    }
)


class EvidenceRefError(RuntimeError):
    pass


def _reject_ambient_git_environment() -> None:
    hostile = sorted(
        name
        for name in os.environ
        if name in _FORBIDDEN_GIT_ENV_NAMES
        or name.startswith("GIT_CONFIG_KEY_")
        or name.startswith("GIT_CONFIG_VALUE_")
    )
    if hostile:
        raise EvidenceRefError(f"ambient Git environment is forbidden: {hostile}")


def _git(repository: Path, *arguments: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        env=_GIT_ENV,
    )
    if check and result.returncode:
        raise EvidenceRefError(
            f"git {' '.join(arguments)} failed: "
            + result.stderr.decode("utf-8", "replace")[-1000:]
        )
    return result.stdout


def _ref(repository: Path, name: str, *, required: bool) -> str | None:
    raw = _git(repository, "rev-parse", "--verify", name, check=False).decode().strip()
    if not raw:
        if required:
            raise EvidenceRefError(f"required ref is absent: {name}")
        return None
    if PIN_PATTERN.fullmatch(raw) is None:
        raise EvidenceRefError(f"invalid ref object id: {name}")
    return raw


def _canonical_path(raw: bytes, label: str) -> Path:
    text = raw.decode().strip()
    if not text:
        raise EvidenceRefError(f"empty repository identity: {label}")
    try:
        return Path(text).resolve(strict=True)
    except OSError as error:
        raise EvidenceRefError(f"unresolvable repository identity: {label}") from error


def _require_repository_config(repository: Path) -> None:
    keys = _git(repository, "config", "--local", "--name-only", "--list").decode().split()
    fixed = {
        "core.repositoryformatversion",
        "core.filemode",
        "core.bare",
        "core.logallrefupdates",
        "remote.origin.url",
        "remote.origin.fetch",
    }
    branch_suffixes = (".remote", ".merge", ".vscode-merge-base")
    unexpected = sorted(
        key
        for key in keys
        if key not in fixed
        and not (key.startswith("branch.") and key.endswith(branch_suffixes))
    )
    if unexpected:
        raise EvidenceRefError(f"unexpected local Git config: {unexpected}")
    expected_values = {
        "core.repositoryformatversion": ["0"],
        "core.filemode": ["true"],
        "core.bare": ["false"],
        "core.logallrefupdates": ["true"],
        "remote.origin.url": [EXPECTED_ORIGIN],
        "remote.origin.fetch": ["+refs/heads/*:refs/remotes/origin/*"],
    }
    for key, expected in expected_values.items():
        actual = _git(repository, "config", "--local", "--get-all", key).decode().splitlines()
        if actual != expected:
            raise EvidenceRefError(f"local Git config mismatch: {key}")


def _require_repository_identity(repository: Path) -> None:
    if repository != EXPECTED_REPOSITORY or repository.is_symlink():
        raise EvidenceRefError("unexpected evidence repository path")
    try:
        value = repository.lstat()
    except OSError as error:
        raise EvidenceRefError("evidence repository is absent") from error
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid():
        raise EvidenceRefError("unsafe evidence repository identity")
    if _git(repository, "rev-parse", "--is-inside-work-tree").strip() != b"true":
        raise EvidenceRefError("evidence repository is not a worktree")
    identities = {
        "toplevel": _canonical_path(
            _git(repository, "rev-parse", "--path-format=absolute", "--show-toplevel"),
            "toplevel",
        ),
        "git_dir": _canonical_path(
            _git(repository, "rev-parse", "--path-format=absolute", "--git-dir"),
            "git_dir",
        ),
        "common_dir": _canonical_path(
            _git(repository, "rev-parse", "--path-format=absolute", "--git-common-dir"),
            "common_dir",
        ),
    }
    expected = {
        "toplevel": EXPECTED_REPOSITORY,
        "git_dir": EXPECTED_GIT_DIR,
        "common_dir": EXPECTED_COMMON_DIR,
    }
    if identities != expected:
        raise EvidenceRefError("evidence repository administrative identity mismatch")
    _require_repository_config(repository)


def _allowed(path: str) -> str:
    if (
        path.startswith(ARTIFACT_PREFIX)
        and path.endswith(".json")
        and "//" not in path
        and not path.endswith("/.json")
    ):
        return "artifact"
    raise EvidenceRefError(f"non-evidence path is forbidden: {path}")


def _name_status(raw: bytes) -> list[tuple[str, str]]:
    fields = raw.decode("utf-8").split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    if len(fields) % 2:
        raise EvidenceRefError("malformed NUL-delimited name-status stream")
    records: list[tuple[str, str]] = []
    for offset in range(0, len(fields), 2):
        status, path = fields[offset : offset + 2]
        if status not in {"A", "M"} or not path:
            raise EvidenceRefError(f"forbidden evidence change status: {status}")
        records.append((status, path))
    return records


def _validate_change_records(
    records: list[tuple[str, str]], seen_artifacts: set[str]
) -> list[str]:
    changed: list[str] = []
    for status, path in records:
        _allowed(path)
        if status != "A" or path in seen_artifacts:
            raise EvidenceRefError(f"evidence artifacts are add-only: {path}")
        seen_artifacts.add(path)
        changed.append(path)
    return changed


def _validate_modes(
    repository: Path,
    paths: list[str],
    *,
    staged: bool,
    treeish: str = "HEAD",
) -> None:
    if not paths:
        raise EvidenceRefError("evidence change set is empty")
    for path in paths:
        raw = _git(repository, "ls-files", "--stage", "--", path) if staged else _git(
            repository, "ls-tree", treeish, "--", path
        )
        fields = raw.decode().split()
        if not fields or fields[0] != "100644":
            raise EvidenceRefError(
                f"evidence path is not a regular mode-100644 blob: {path}"
            )


def _require_complete_worktree(repository: Path) -> int:
    for key in ("core.sparseCheckout", "core.sparseCheckoutCone"):
        value = _git(
            repository, "config", "--local", "--bool", "--get", key, check=False
        )
        if value.strip() not in {b"", b"false"}:
            raise EvidenceRefError(f"sparse worktree is forbidden: {key}")
    flags = _git(repository, "ls-files", "-v", "-z").decode().split("\0")
    if flags and flags[-1] == "":
        flags.pop()
    if len(flags) < MIN_TRACKED_FILES:
        raise EvidenceRefError("evidence worktree tracked-file floor is not met")
    for record in flags:
        if not record.startswith("H "):
            raise EvidenceRefError(f"nonordinary index flag is forbidden: {record[:1]}")
        path = repository / record[2:]
        try:
            value = path.lstat()
        except OSError as error:
            raise EvidenceRefError(f"tracked path is absent: {record[2:]}") from error
        if not stat.S_ISREG(value.st_mode):
            raise EvidenceRefError(f"tracked path is not regular: {record[2:]}")
    return len(flags)


def _parse_remote_heads(raw: bytes) -> dict[str, str]:
    if not raw.isascii():
        raise EvidenceRefError("ls-remote output is not ASCII")
    if not raw.endswith(b"\n"):
        raise EvidenceRefError("ls-remote output lacks its terminal LF")
    if b"\r" in raw or b"\0" in raw:
        raise EvidenceRefError("ls-remote output contains a forbidden byte")

    heads: dict[str, str] = {}
    for line in raw[:-1].split(b"\n"):
        if not line or line.count(b"\t") != 1:
            raise EvidenceRefError("malformed or unexpected ls-remote record")
        pin_raw, name_raw = line.split(b"\t")
        try:
            pin = pin_raw.decode("ascii")
            name = name_raw.decode("ascii")
        except UnicodeDecodeError as error:  # Defensive after isascii().
            raise EvidenceRefError("ls-remote output is not ASCII") from error
        if name not in {
            REMOTE_EXECUTION_BRANCH,
            REMOTE_EVIDENCE_BRANCH,
        }:
            raise EvidenceRefError("malformed or unexpected ls-remote record")
        if PIN_PATTERN.fullmatch(pin) is None or name in heads:
            raise EvidenceRefError("invalid or duplicate ls-remote record")
        heads[name] = pin
    if REMOTE_EXECUTION_BRANCH not in heads:
        raise EvidenceRefError("remote execution branch is absent")
    return heads


def _remote_heads(repository: Path) -> dict[str, str]:
    raw = _git(
        repository,
        "ls-remote",
        "--refs",
        "origin",
        REMOTE_EXECUTION_BRANCH,
        REMOTE_EVIDENCE_BRANCH,
    )
    return _parse_remote_heads(raw)


def _require_execution_freeze(
    repository: Path, execution_pin: str, remote_heads: dict[str, str]
) -> None:
    if _ref(repository, EXECUTION_BRANCH, required=True) != execution_pin:
        raise EvidenceRefError("local execution branch drift")
    if remote_heads[REMOTE_EXECUTION_BRANCH] != execution_pin:
        raise EvidenceRefError("remote execution branch drift")


def _require_evidence_branch(repository: Path) -> None:
    branch = _git(repository, "symbolic-ref", "-q", "HEAD").decode().strip()
    if branch != EVIDENCE_BRANCH:
        raise EvidenceRefError("validator must run on the dedicated evidence branch")


def _require_remote_fast_forward(
    repository: Path,
    head: str,
    remote_heads: dict[str, str],
    *,
    require_remote_equal: bool,
) -> str | None:
    remote = remote_heads.get(REMOTE_EVIDENCE_BRANCH)
    if remote is None:
        if require_remote_equal:
            raise EvidenceRefError("remote evidence ref is absent")
        return None
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            remote,
            head,
        ],
        check=False,
        capture_output=True,
        env=_GIT_ENV,
    )
    if result.returncode:
        raise EvidenceRefError(
            "remote evidence ref is not an ancestor of the candidate"
        )
    if require_remote_equal and remote != head:
        raise EvidenceRefError("remote evidence ref does not equal the local candidate")
    return remote


def _validate_root_authority(
    repository: Path, execution_pin: str | None, *, staged: bool
) -> str:
    treeish = f":{ROOT_AUTHORITY}" if staged else f"HEAD:{ROOT_AUTHORITY}"
    raw = _git(repository, "show", treeish)
    try:
        artifact = json.loads(raw)
    except json.JSONDecodeError as error:
        raise EvidenceRefError("root authority is not valid JSON") from error
    expected_keys = {
        "artifact_kind",
        "evidence_branch",
        "execution_branch",
        "execution_pin",
        "hlo_work",
        "origin",
        "schema_version",
        "tpu_work",
    }
    if set(artifact) != expected_keys:
        raise EvidenceRefError("root authority schema mismatch")
    bound_pin = artifact.get("execution_pin")
    if (
        artifact.get("artifact_kind") != "gate_d_evidence_ref_root_v1"
        or type(artifact.get("schema_version")) is not int
        or artifact.get("schema_version") != 1
        or artifact.get("execution_branch")
        != EXECUTION_BRANCH.removeprefix("refs/heads/")
        or artifact.get("evidence_branch")
        != EVIDENCE_BRANCH.removeprefix("refs/heads/")
        or artifact.get("origin") != EXPECTED_ORIGIN
        or artifact.get("hlo_work") is not False
        or artifact.get("tpu_work") is not False
        or not isinstance(bound_pin, str)
        or PIN_PATTERN.fullmatch(bound_pin) is None
        or (execution_pin is not None and bound_pin != execution_pin)
    ):
        raise EvidenceRefError("root authority contract mismatch")
    return bound_pin


def _validate_committed_prefix(
    repository: Path, execution_pin: str, current_head: str
) -> tuple[list[str], set[str], list[str]]:
    merge_base = _git(
        repository, "merge-base", execution_pin, current_head
    ).decode().strip()
    if merge_base != execution_pin:
        raise EvidenceRefError("evidence branch is not rooted at the execution pin")
    commits = _git(
        repository, "rev-list", "--reverse", f"{execution_pin}..{current_head}"
    ).decode().split()
    seen_artifacts: set[str] = set()
    paths: list[str] = []
    for commit in commits:
        parents = _git(
            repository, "show", "-s", "--format=%P", commit
        ).decode().split()
        if len(parents) != 1:
            raise EvidenceRefError(
                f"merge/root commit forbidden after execution pin: {commit}"
            )
        records = _name_status(
            _git(
                repository,
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-z",
                commit,
            )
        )
        commit_paths = _validate_change_records(records, seen_artifacts)
        _validate_modes(repository, commit_paths, staged=False, treeish=commit)
        paths.extend(commit_paths)
    return commits, seen_artifacts, paths


def validate(
    repository: Path,
    execution_pin: str | None,
    *,
    mode: str,
    require_remote_equal: bool,
) -> dict[str, object]:
    _reject_ambient_git_environment()
    _require_repository_identity(repository)
    _require_evidence_branch(repository)
    if execution_pin is not None and PIN_PATTERN.fullmatch(execution_pin) is None:
        raise EvidenceRefError("invalid execution pin")
    bound_pin = _validate_root_authority(
        repository, execution_pin, staged=mode == "staged"
    )
    remote_heads = _remote_heads(repository)
    _require_execution_freeze(repository, bound_pin, remote_heads)

    current_head = _ref(repository, "HEAD", required=True)
    assert current_head is not None
    remote_evidence = _require_remote_fast_forward(
        repository,
        current_head,
        remote_heads,
        require_remote_equal=require_remote_equal,
    )
    commits, seen_artifacts, paths = _validate_committed_prefix(
        repository, bound_pin, current_head
    )

    if mode == "staged":
        if require_remote_equal:
            raise EvidenceRefError("staged mode cannot require remote equality")
        if _git(repository, "diff", "--name-only"):
            raise EvidenceRefError("unstaged changes are forbidden")
        if _git(repository, "ls-files", "--others", "--exclude-standard"):
            raise EvidenceRefError("untracked files are forbidden")
        records = _name_status(
            _git(repository, "diff", "--cached", "--name-status", "-z")
        )
        staged_paths = _validate_change_records(records, seen_artifacts)
        _validate_modes(repository, staged_paths, staged=True)
        paths.extend(staged_paths)
        candidate_identity = hashlib.sha256(
            _git(repository, "diff", "--cached", "--binary", "--full-index")
        ).hexdigest()
    elif mode == "committed":
        if _git(repository, "status", "--porcelain=v1"):
            raise EvidenceRefError("committed validation requires a clean worktree")
        if not commits:
            raise EvidenceRefError("evidence branch has no evidence commits")
        candidate_identity = _git(
            repository, "rev-parse", "HEAD^{tree}"
        ).decode().strip()
    else:
        raise EvidenceRefError(f"unsupported mode: {mode}")

    tracked_files = _require_complete_worktree(repository)
    root_treeish = (
        f":{ROOT_AUTHORITY}" if mode == "staged" else f"HEAD:{ROOT_AUTHORITY}"
    )
    return {
        "candidate_identity": candidate_identity,
        "changed_paths": sorted(set(paths)),
        "evidence_branch": EVIDENCE_BRANCH,
        "evidence_head": current_head,
        "execution_branch": EXECUTION_BRANCH,
        "execution_pin": bound_pin,
        "mode": mode,
        "origin": EXPECTED_ORIGIN,
        "remote_evidence_head": remote_evidence,
        "remote_execution_head": remote_heads[REMOTE_EXECUTION_BRANCH],
        "root_authority_sha256": hashlib.sha256(
            _git(repository, "show", root_treeish)
        ).hexdigest(),
        "status": "GATE_D_EVIDENCE_REF_VALID",
        "tracked_files": tracked_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--execution-pin")
    parser.add_argument("--mode", choices=("staged", "committed"), required=True)
    parser.add_argument("--require-remote-equal", action="store_true")
    arguments = parser.parse_args()
    report = validate(
        arguments.repository,
        arguments.execution_pin,
        mode=arguments.mode,
        require_remote_equal=arguments.require_remote_equal,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
