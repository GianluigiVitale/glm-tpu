"""CPU-only checks for the extracted native host-path guard and its limits."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from glm_tpu.host_paths import _plain_path

REPO = Path(__file__).resolve().parents[2]
BASE = "b667f00f1ae48c8ff37e92500550c1395d74c66d"


@pytest.mark.parametrize("name", ("", ".", "relative", "parent/child"))
def test_relative_paths_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="absolute with no symlink ancestors"):
        _plain_path(Path(name))


def test_plain_files_directories_and_absent_targets_are_allowed(tmp_path: Path) -> None:
    file = tmp_path / "evidence.json"
    file.write_text("{}")
    absent = tmp_path / "not-created" / "result.json"
    for path in (tmp_path, file, absent):
        assert _plain_path(path) is None
    assert not absent.parent.exists()
    assert file.read_text() == "{}"


@pytest.mark.parametrize("dangling", (False, True))
def test_final_symlinks_are_rejected(tmp_path: Path, dangling: bool) -> None:
    target = tmp_path / "target"
    if not dangling:
        target.write_text("original")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="no symlink ancestors"):
        _plain_path(link)
    assert link.is_symlink()


@pytest.mark.parametrize("dangling", (False, True))
def test_symlink_ancestors_are_rejected(tmp_path: Path, dangling: bool) -> None:
    target = tmp_path / "directory"
    if not dangling:
        target.mkdir()
    link = tmp_path / "linked-directory"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="no symlink ancestors"):
        _plain_path(link / "nested" / "result.json")


def test_original_guard_function_is_unchanged() -> None:
    original = subprocess.check_output(
        ["git", "show", BASE + ":scripts/greenfield/ws32_history_preflight.py"],
        cwd=REPO,
        text=True,
    )
    current = (REPO / "glm_tpu/host_paths.py").read_text()

    def function(source: str) -> str:
        return ast.dump(
            next(
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "_plain_path"
            )
        )

    assert function(current) == function(original)


def test_import_does_not_load_accelerator_or_experiment_modules() -> None:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import glm_tpu.host_paths; "
        "assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'torch', 'scripts'} "
        "for name in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(REPO)],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        check=True,
        timeout=10,
    )
