from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
SPEC = importlib.util.spec_from_file_location("gate_d_same_region_git_mirror", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(repo), *arguments], text=True
    ).strip()


def _repository_pair(tmp_path: Path) -> tuple[Path, Path, str]:
    local = tmp_path / "local"
    subprocess.run(
        ["/usr/bin/git", "init", "-q", "-b", MODULE.BRANCH, str(local)], check=True
    )
    _git(local, "config", "user.name", "Gate D Test")
    _git(local, "config", "user.email", "gate-d-test@example.invalid")
    _git(local, "remote", "add", "origin", MODULE.ORIGIN)
    (local / "runtime.txt").write_text("exact runtime bytes\n", encoding="ascii")
    (local / "other.txt").write_text("other tree bytes\n", encoding="ascii")
    _git(local, "add", "runtime.txt", "other.txt")
    _git(local, "commit", "-q", "-m", "fixture")
    pin = _git(local, "rev-parse", "HEAD")
    mirror = tmp_path / "mirror.git"
    shutil.copytree(local / ".git", mirror)
    return local, mirror, pin


def _verify(local: Path, mirror: Path, pin: str) -> dict[str, object]:
    return MODULE.verify_mirror_directory(
        mirror,
        local,
        pin,
        branch=MODULE.BRANCH,
        origin=MODULE.ORIGIN,
        bound_paths=("runtime.txt",),
    )


def test_exact_mirror_proves_ref_checkout_closure_and_bound_blob(
    tmp_path: Path,
) -> None:
    local, mirror, pin = _repository_pair(tmp_path)
    evidence = _verify(local, mirror, pin)
    assert evidence["commit"] == pin
    assert evidence["commit_connectivity_fsck"] is True
    assert evidence["bound_blob_count"] == 1
    assert evidence["bound_blobs"][0]["path"] == "runtime.txt"  # type: ignore[index]


@pytest.mark.parametrize("replacement", ["0" * 40 + "\n", "malformed\n"])
def test_mirror_rejects_stale_or_malformed_ref(
    tmp_path: Path, replacement: str
) -> None:
    local, mirror, pin = _repository_pair(tmp_path)
    ref = mirror / "refs/heads" / Path(MODULE.BRANCH)
    ref.write_text(replacement, encoding="ascii")
    with pytest.raises(RuntimeError, match="ref is stale or malformed"):
        _verify(local, mirror, pin)


@pytest.mark.parametrize("attack", ["missing", "corrupt"])
def test_mirror_rejects_missing_or_corrupt_bound_blob(
    tmp_path: Path, attack: str
) -> None:
    local, mirror, pin = _repository_pair(tmp_path)
    object_id = _git(local, "rev-parse", f"{pin}:runtime.txt")
    object_path = mirror / "objects" / object_id[:2] / object_id[2:]
    if attack == "missing":
        object_path.unlink()
    else:
        raw = bytearray(object_path.read_bytes())
        raw[-1] ^= 1
        object_path.chmod(0o600)
        object_path.write_bytes(raw)
    with pytest.raises(RuntimeError, match="Git mirror verification command failed"):
        _verify(local, mirror, pin)


def test_mirror_rejects_absent_bound_runtime_path(tmp_path: Path) -> None:
    local, mirror, pin = _repository_pair(tmp_path)
    with pytest.raises(RuntimeError, match="bound runtime blob is absent"):
        MODULE.verify_mirror_directory(
            mirror,
            local,
            pin,
            branch=MODULE.BRANCH,
            origin=MODULE.ORIGIN,
            bound_paths=("missing-runtime.py",),
        )


def test_mirror_rejects_origin_and_symlink_rebinding(tmp_path: Path) -> None:
    local, mirror, pin = _repository_pair(tmp_path)
    _git(mirror, "config", "remote.origin.url", "https://example.invalid/wrong.git")
    with pytest.raises(RuntimeError, match="origin drifted"):
        _verify(local, mirror, pin)

    local, mirror, pin = _repository_pair(tmp_path / "second")
    (mirror / "unsafe-link").symlink_to(mirror / "HEAD")
    with pytest.raises(RuntimeError, match="contains symlink"):
        _verify(local, mirror, pin)


def test_origin_remote_pin_parser_requires_one_exact_lf_terminated_record() -> None:
    pin = "a" * 40
    branch = MODULE.BRANCH
    exact = f"{pin}\trefs/heads/{branch}\n".encode("ascii")
    MODULE.validate_remote_pin_output(exact, pin, branch)
    hostile = (
        exact[:-1],
        exact + exact,
        f"{pin}\trefs/heads/wrong\n".encode("ascii"),
        f"{'b' * 40}\trefs/heads/{branch}\n".encode("ascii"),
        exact.replace(b"\t", b" "),
    )
    for raw in hostile:
        with pytest.raises(RuntimeError, match="stale or malformed"):
            MODULE.validate_remote_pin_output(raw, pin, branch)


def test_origin_remote_pin_rejects_transport_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pin = "a" * 40

    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del args, kwargs
        return subprocess.CompletedProcess([], 128, b"", b"transport failed")

    monkeypatch.setattr(MODULE.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="transport failed"):
        MODULE.verify_origin_remote_pin(tmp_path, pin, branch=MODULE.BRANCH)


def test_origin_remote_pin_rejects_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pin = "a" * 40

    def timed_out(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        del args, kwargs
        raise subprocess.TimeoutExpired(["git", "ls-remote"], 60)

    monkeypatch.setattr(MODULE.subprocess, "run", timed_out)
    with pytest.raises(RuntimeError, match="transport failed"):
        MODULE.verify_origin_remote_pin(tmp_path, pin, branch=MODULE.BRANCH)
