"""Sealed runtime helpers: digests, path installation and import-closure checks."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from glm_tpu.greenfield.benchmarking import sealed_runtime as sr


def test_tree_digest_is_deterministic_and_content_sensitive(tmp_path: Path):
    root = tmp_path / "site"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_bytes(b"x = 1\n")
    (root / "pkg" / "link.py").symlink_to("a.py")
    first = sr.runtime_tree_sha256(root, require_sealed=False)
    assert first == sr.runtime_tree_sha256(root, require_sealed=False)
    (root / "pkg" / "a.py").write_bytes(b"x = 2\n")
    assert sr.runtime_tree_sha256(root, require_sealed=False) != first
    # user-owned trees are refused when sealing is required
    with pytest.raises(RuntimeError):
        sr.runtime_tree_sha256(root, require_sealed=True)
    # escaping symlinks are refused
    (root / "pkg" / "escape.py").symlink_to(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_bytes(b"")
    with pytest.raises(RuntimeError):
        sr.runtime_tree_sha256(root, require_sealed=False)


def test_sha_file_refuses_symlinks_and_hardlinks(tmp_path: Path):
    target = tmp_path / "f"
    target.write_bytes(b"abc")
    assert sr.sha_file(target) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    link = tmp_path / "l"
    link.symlink_to(target)
    with pytest.raises(OSError):
        sr.sha_file(link)
    hard = tmp_path / "h"
    os.link(target, hard)
    with pytest.raises(RuntimeError):
        sr.sha_file(target)


def test_install_sealed_source_path_and_closure(tmp_path: Path, monkeypatch):
    original = list(sys.path)
    try:
        expected = sr.install_sealed_source_path(Path("/nonexistent/source"))
        assert tuple(sys.path) == expected
        assert expected[0] == "/nonexistent/source"
        assert expected[1] == str(sr.JAX_SITE_ROOT) and expected[2] == str(sr.LIBTPU_SITE_ROOT)
        assert expected[3:] == sr.EXPECTED_RUNTIME_PATH
    finally:
        sys.path[:] = original
    # glm_tpu modules imported from this repository escape a foreign source root
    with pytest.raises(RuntimeError):
        sr.verify_import_closure(Path("/nonexistent/source"))
    repo_root = Path(__file__).resolve().parents[3]
    records = sr.verify_import_closure(repo_root)
    assert any(record["module"] == "glm_tpu.greenfield.benchmarking.sealed_runtime" for record in records)


def test_python_runtime_validation_refuses_non_sealed_interpreter():
    if Path(sys.executable) == sr.PYTHON:
        pytest.skip("running under the sealed interpreter")
    with pytest.raises(RuntimeError):
        sr.validate_python_runtime()


@pytest.mark.skipif(not sr.JAX_SITE_ROOT.exists(), reason="sealed sites not installed")
def test_dependency_sites_match_pinned_digests():
    sites = sr.validate_dependency_sites()
    assert sites["jax"]["tree_sha256"] == sr.JAX_SITE_TREE_SHA256
    assert sites["libtpu"]["manifest_sha256"] == sr.LIBTPU_SITE_MANIFEST_SHA256
