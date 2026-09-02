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


def test_main_binds_the_worktree_right_after_the_exact_head_check() -> None:
    source = SCRIPT.read_text()
    main_body = source[source.index("\ndef main(") :]
    head_check = main_body.index('f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"')
    binding_call = main_body.index("worktree_binding = _worktree_binding(\n        REPO, os.environ.get(\"GLM_GREENFIELD_RUN_TAG\", \"\").strip() or None\n    )")
    assert head_check < binding_call
    assert "if REPO != Path(" not in source
    assert '"worktree_binding": worktree_binding,' in source
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location("csd_constants", SCRIPT))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    module.__spec__.loader.exec_module(module)
    assert module.CANONICAL_WORKTREE == Path("/home/gianl/glm-tpu-topology-rewrite")
    assert module.RUN_ROOT == Path("/home/gianl/glm-run")


def _detached_source(tmp_path: Path, canonical: Path, tag: str) -> tuple[Path, Path, str]:
    run_root = tmp_path / "glm-run"
    source = run_root / tag / "source"
    source.parent.mkdir(parents=True)
    pin = _git("rev-parse", "HEAD", cwd=canonical)
    _git("worktree", "add", "-q", "--detach", str(source), pin, cwd=canonical)
    return run_root, source, pin


def _env_i_binding(source: Path, canonical: Path, run_root: Path, tag: str) -> subprocess.CompletedProcess:
    """The runner's worker prologue, then the detached commit's own module binds its real REPO."""

    program = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('csd_detached', {str(source / 'scripts' / 'greenfield' / 'compile_short_decoder.py')!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"assert module.REPO == __import__('pathlib').Path({str(source)!r}), module.REPO\n"
        f"print(module._worktree_binding(module.REPO, {tag!r}, canonical=__import__('pathlib').Path({str(canonical)!r}), run_root=__import__('pathlib').Path({str(run_root)!r})))\n"
    )
    return subprocess.run(
        [
            "/usr/bin/env", "-i",
            "HOME=/home/gianl", "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C",
            "PYTHONDONTWRITEBYTECODE=1", "GLM_GREENFIELD_STRATEGY_ND_ATTENTION_PROJECTION=1",
            "JAX_PLATFORMS=cpu", f"PYTHONPATH={source}", f"GLM_GREENFIELD_RUN_TAG={tag}",
            sys.executable, "-u", "-c", program,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_env_i_detached_source_of_this_repository_executes_and_passes_the_binding(tmp_path: Path) -> None:
    """Worker-0 layout: the canonical tree is itself a linked worktree of the main repository."""

    runner = RUNNER.read_text()
    assert 'git -C "$wt" worktree add -q --detach "$src" ' in runner
    assert "/usr/bin/env -i HOME=/home/gianl PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1" in runner
    tag = "greenfield_test_tag_20260902T000000000000000Z"
    run_root, source, pin = _detached_source(tmp_path, ROOT, tag)
    try:
        assert _git("rev-parse", "HEAD", cwd=source) == pin
        assert _git("status", "--porcelain", "--ignored", cwd=source) == ""
        completed = _env_i_binding(source, ROOT, run_root, tag)
        assert completed.returncode == 0, completed.stderr[-3000:]
        assert completed.stdout.strip().splitlines()[-1] == "detached_pin_worktree"
        # The same detached tree under a different tag, or bound to a foreign canonical, is refused.
        wrong_tag = _env_i_binding(source, ROOT, run_root, "other_tag")
        assert wrong_tag.returncode != 0 and "wrong greenfield worktree" in wrong_tag.stderr
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        _git("init", "-q", cwd=foreign)
        _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x", cwd=foreign)
        unlinked = _env_i_binding(source, foreign, run_root, tag)
        assert unlinked.returncode != 0 and "not linked to the canonical repository" in unlinked.stderr
        assert _git("status", "--porcelain", "--ignored", cwd=source) == ""
    finally:
        _git("worktree", "remove", "--force", str(source), cwd=ROOT)


def test_env_i_detached_source_of_a_main_repository_passes_the_binding(canonical: Path, tmp_path: Path) -> None:
    """Worker 1-7 layout: the canonical tree is a main clone with a ``.git`` directory."""

    tag = "greenfield_test_tag_20260902T000000000000001Z"
    (canonical / "scripts" / "greenfield").mkdir(parents=True)
    (canonical / "scripts" / "greenfield" / "compile_short_decoder.py").write_bytes(SCRIPT.read_bytes())
    (canonical / "glm_tpu").mkdir()
    _git("add", "-A", cwd=canonical)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "script", cwd=canonical)
    run_root, source, pin = _detached_source(tmp_path, canonical, tag)
    # The detached copy has no glm_tpu package of its own; resolve imports from this repository.
    program_env_source = source
    completed = subprocess.run(
        [
            "/usr/bin/env", "-i",
            "HOME=/home/gianl", "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C",
            "PYTHONDONTWRITEBYTECODE=1", "JAX_PLATFORMS=cpu", f"PYTHONPATH={ROOT}", f"GLM_GREENFIELD_RUN_TAG={tag}",
            sys.executable, "-u", "-c",
            "import importlib.util, pathlib\n"
            f"spec = importlib.util.spec_from_file_location('csd_main_layout', {str(program_env_source / 'scripts' / 'greenfield' / 'compile_short_decoder.py')!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            f"assert module.REPO == pathlib.Path({str(source)!r})\n"
            f"print(module._worktree_binding(module.REPO, {tag!r}, canonical=pathlib.Path({str(canonical)!r}), run_root=pathlib.Path({str(run_root)!r})))\n",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip().splitlines()[-1] == "detached_pin_worktree"
