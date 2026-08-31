from __future__ import annotations

import ast
import importlib.util
import json
from hashlib import sha256
import os
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).parents[3]
BUILDER = ROOT / "scripts/greenfield/build_gate_d_libtpu_site_capsule.py"
BUILD = ROOT / "docs/artifacts/gate-d-libtpu-site-build.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location("gate_d_libtpu_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_libtpu_supplement_build_is_minimal_and_reproducible() -> None:
    source = BUILDER.read_text()
    tree = ast.parse(source)
    allowlist = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ALLOWLIST"
            for target in node.targets
        )
    )
    assert allowlist == ("libtpu", "libtpu-0.0.41.dist-info")
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module).split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported_roots <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "shutil",
        "stat",
        "struct",
        "typing",
    }
    build = json.loads(BUILD.read_text())
    assert build["builder_sha256"] == sha256(BUILDER.read_bytes()).hexdigest()
    assert build["capsule_tree_sha256"] == (
        "db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca"
    )
    assert build["capsule_manifest_sha256"] == (
        "d34064f4a0dfcdcd9ec13288ce967ae060a4650b3e86e53bc47e49756c0e97fa"
    )
    assert build["reproducible_build_count"] == 2
    assert build["capsule_files"] == 13
    assert build["capsule_bytes"] == 719846262
    assert build["installed"] is False
    assert build["jax_backend_or_tpu_work_performed"] is False
    assert build["future_target"] == (
        "/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3"
    )


def test_tree_hash_rejects_escaping_symlink_and_special_file(tmp_path: Path) -> None:
    builder = _load_builder()
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(SystemExit, match="symlink escapes root"):
        builder._tree_sha256(root)
    (root / "escape").unlink()
    os.mkfifo(root / "fifo")
    with pytest.raises(SystemExit, match="unsupported libtpu capsule entry"):
        builder._tree_sha256(root)
    (root / "fifo").unlink()
    outside = tmp_path / "outside-file"
    outside.write_bytes(b"shared")
    os.link(outside, root / "hardlink")
    with pytest.raises(SystemExit, match="libtpu source file is unsafe"):
        builder._tree_sha256(root)


def test_manifest_write_and_output_creation_do_not_clobber(tmp_path: Path) -> None:
    builder = _load_builder()
    target = tmp_path / "manifest.json"
    target.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        builder._write_exclusive(target, b"replacement")
    assert target.read_bytes() == b"preserve"

    output = tmp_path / "already-exists"
    output.mkdir()
    builder.SOURCE_ROOT = tmp_path / "source"
    for name in builder.ALLOWLIST:
        dependency = builder.SOURCE_ROOT / name
        dependency.mkdir(parents=True)
        (dependency / "payload").write_bytes(name.encode("ascii"))
    builder._arguments = lambda: type("Arguments", (), {"output": output})()
    builder.os.geteuid = lambda: 1000
    with pytest.raises(FileExistsError):
        builder.main()


def test_build_rejects_source_change_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _load_builder()
    source = tmp_path / "source"
    output = tmp_path / "output"
    for name in builder.ALLOWLIST:
        dependency = source / name
        dependency.mkdir(parents=True)
        (dependency / "payload").write_bytes(name.encode("ascii"))

    original_copytree = shutil.copytree
    copy_count = 0

    def copy_then_mutate(src, dst, **kwargs):
        nonlocal copy_count
        result = original_copytree(src, dst, **kwargs)
        copy_count += 1
        if copy_count == len(builder.ALLOWLIST):
            (source / builder.ALLOWLIST[0] / "payload").write_bytes(b"changed")
        return result

    monkeypatch.setattr(builder, "SOURCE_ROOT", source)
    monkeypatch.setattr(builder.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        builder,
        "_arguments",
        lambda: type("Arguments", (), {"output": output})(),
    )
    monkeypatch.setattr(builder.shutil, "copytree", copy_then_mutate)
    with pytest.raises(SystemExit, match="dependencies changed during capsule build"):
        builder.main()
