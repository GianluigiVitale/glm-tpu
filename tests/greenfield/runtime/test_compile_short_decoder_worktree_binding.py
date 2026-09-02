"""The compile script accepts only the canonical worktree or the runner's detached pin worktree.

The 2026-09-02 protected launch failed closed on all eight workers because the
script bound its location to the canonical path while the hardened runner
executes a detached worktree of the pin under the run directory.  These tests
exercise the binding on real git worktrees and the runner's exact worker
prologue under ``env -i``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "greenfield" / "compile_short_decoder.py"
RUNNER = ROOT / "scripts" / "greenfield" / "run_short_decoder_compile_pp8.sh"


def _load_binding():
    spec = importlib.util.spec_from_file_location("compile_short_decoder_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    spec.loader.exec_module(module)
    return module._worktree_binding


def _git(*arguments: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


@pytest.fixture()
def canonical(tmp_path: Path) -> Path:
    repo = tmp_path / "canonical"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "pin", cwd=repo)
    return repo


def test_canonical_and_detached_pin_worktree_bind_and_everything_else_is_refused(
    canonical: Path, tmp_path: Path
) -> None:
    binding = _load_binding()
    run_root = tmp_path / "glm-run"
    tag = "tag_20260902T000000000000000Z"
    source = run_root / tag / "source"
    source.parent.mkdir(parents=True)
    pin = _git("rev-parse", "HEAD", cwd=canonical)
    _git("worktree", "add", "-q", "--detach", str(source), pin, cwd=canonical)

    assert binding(canonical, None, canonical=canonical, run_root=run_root) == "canonical_worktree"
    assert binding(source, tag, canonical=canonical, run_root=run_root) == "detached_pin_worktree"
    with pytest.raises(RuntimeError, match="wrong greenfield worktree"):
        binding(source, None, canonical=canonical, run_root=run_root)
    with pytest.raises(RuntimeError, match="wrong greenfield worktree"):
        binding(source, "other_tag", canonical=canonical, run_root=run_root)
    with pytest.raises(RuntimeError, match="wrong greenfield worktree"):
        binding(tmp_path / "elsewhere", tag, canonical=canonical, run_root=run_root)

    # A foreign repository parked at the right path is not linked to the canonical one.
    foreign_tag = "foreign_tag"
    foreign = run_root / foreign_tag / "source"
    foreign.mkdir(parents=True)
    _git("init", "-q", cwd=foreign)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x", cwd=foreign)
    with pytest.raises(RuntimeError, match="not linked to the canonical repository"):
        binding(foreign, foreign_tag, canonical=canonical, run_root=run_root)

    # Untracked and ignored files in the detached tree are refused.
    (source / "stray.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError, match="not clean"):
        binding(source, tag, canonical=canonical, run_root=run_root)
    (source / "stray.py").unlink()
    (source / ".gitignore").write_text("*.pyc\n")
    with pytest.raises(RuntimeError, match="not clean"):
        binding(source, tag, canonical=canonical, run_root=run_root)
    (source / ".gitignore").unlink()
    assert binding(source, tag, canonical=canonical, run_root=run_root) == "detached_pin_worktree"


def test_runner_prologue_starts_the_compile_script_from_a_detached_worktree_under_env_i(
    tmp_path: Path,
) -> None:
    """Replay the worker prologue on this repository's HEAD: worktree add, clean check, env -i start."""

    runner = RUNNER.read_text()
    assert 'git -C "$wt" worktree add -q --detach "$src" ' in runner
    env_prefix = re.search(r"/usr/bin/env -i (HOME=/home/gianl .*?) JAX_PLATFORMS=tpu", runner).group(1)
    assert env_prefix.startswith("HOME=/home/gianl PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1")
    pin = _git("rev-parse", "HEAD", cwd=ROOT)
    source = tmp_path / "run" / "source"
    source.parent.mkdir(parents=True)
    _git("worktree", "add", "-q", "--detach", str(source), pin, cwd=ROOT)
    try:
        assert _git("rev-parse", "HEAD", cwd=source) == pin
        assert _git("status", "--porcelain", "--ignored", cwd=source) == ""
        completed = subprocess.run(
            [
                "/usr/bin/env", "-i",
                "HOME=/home/gianl", "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C",
                "PYTHONDONTWRITEBYTECODE=1", "GLM_GREENFIELD_STRATEGY_ND_ATTENTION_PROJECTION=1",
                "JAX_PLATFORMS=cpu", f"PYTHONPATH={source}", "GLM_GREENFIELD_RUN_TAG=test",
                sys.executable, "-u", str(source / "scripts" / "greenfield" / "compile_short_decoder.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "--expected-code-hash" in completed.stdout
        assert _git("status", "--porcelain", "--ignored", cwd=source) == ""
    finally:
        _git("worktree", "remove", "--force", str(source), cwd=ROOT)
