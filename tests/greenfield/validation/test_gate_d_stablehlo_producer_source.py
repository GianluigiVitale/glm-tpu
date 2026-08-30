from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCER = REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_stablehlo.py"
BUILDER = REPO_ROOT / "scripts/greenfield/build_gate_d_jax_site_capsule.py"


def _load_builder() -> object:
    spec = importlib.util.spec_from_file_location("gate_d_jax_site_builder_test", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_producer() -> object:
    spec = importlib.util.spec_from_file_location("gate_d_stablehlo_producer_test", PRODUCER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_name(node: ast.Call) -> str:
    value = node.func
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return ""


def test_gate_d_stablehlo_producer_is_lowering_only_and_default_off() -> None:
    first_line = PRODUCER.read_text().splitlines()[0]
    assert first_line == (
        "#!/usr/bin/env -S "
        "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12 -I -S"
    )
    tree = ast.parse(PRODUCER.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [_call_name(node) for node in calls]
    assert "lower" in names
    assert "compiler_ir" in names
    assert "ShapeDtypeStruct" in names
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr == "compile"
        for node in calls
    )
    source_compiles = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "compile"
    ]
    assert len(source_compiles) == 1
    assert "block_until_ready" not in names
    assert "device_put" not in names
    assert "device_get" not in names
    assert "pmap" not in names
    assert "make_array_from_callback" not in names
    main_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(main_guards) == 1


def test_gate_d_stablehlo_producer_binds_real_source_and_shape() -> None:
    tree = ast.parse(PRODUCER.read_text())
    text = PRODUCER.read_text()
    assert "fused_add_rms_norm" in text
    assert "fused_add_rms_norm_with_auxiliary" in text
    assert "HIDDEN_SIZE = 6144" in text
    assert "EPSILON = 1e-5" in text
    assert "EXPECTED_SOURCE_CERTIFICATE_SHA256" in text
    assert "EXPECTED_PLAN_AUTHORITY_SHA256" in text
    assert "EXPECTED_SOURCE_FILES" in text
    assert "PYTHON_RUNTIME_ROOT" in text
    assert "EXPECTED_PYTHON_RUNTIME_TREE_SHA256" in text
    assert "EXPECTED_SITE_TREE_SHA256" in text
    assert "RISKY_ENVIRONMENT_PREFIXES" in text
    assert "_F_ADD_SEALS = 1033" in text
    assert "_F_GET_SEALS = 1034" in text
    assert "fcntl.F_ADD_SEALS" not in text
    assert "fcntl.F_GET_SEALS" not in text
    assert "expected-producer-sha256" in text
    assert "expected-code-pin" in text
    assert '"installed_path": str(producer)' in text
    assert '"source_path": PRODUCER_SOURCE.relative_to(REPO_ROOT).as_posix()' in text
    assert "producer.relative_to(REPO_ROOT)" not in text
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "_runtime_tree_sha256",
        "_loaded_dependency_records",
        "_annotate_candidate",
        "_lower",
        "main",
    } <= functions


def test_gate_d_stablehlo_producer_seals_source_without_optional_fcntl_names(
    tmp_path: Path,
) -> None:
    producer = _load_producer()
    source = tmp_path / "source.py"
    payload = b"answer = 42\n"
    source.write_bytes(payload)
    assert producer._sealed_source_snapshot(source, sha256(payload).hexdigest()) == payload


def test_gate_d_stablehlo_producer_is_append_only_and_cpu_forced() -> None:
    tree = ast.parse(PRODUCER.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    open_flags = [
        node
        for node in calls
        if _call_name(node) == "open" and any(
            isinstance(argument, ast.Name) and argument.id == "flags"
            for argument in node.args
        )
    ]
    assert open_flags
    text = PRODUCER.read_text()
    assert "os.O_EXCL" in text
    assert "os.O_NOFOLLOW" in text
    assert 'os.environ["JAX_PLATFORMS"] = "cpu"' in text
    assert 'os.environ["JAX_PLATFORM_NAME"] = "cpu"' in text
    assert 'jax.devices("cpu")' in text
    assert "sys.dont_write_bytecode = True" in text
    assert "exec(" in text
    assert "compile(source_payloads[rms_path]" in text
    assert '"SUCCESS"' in text


def test_gate_d_jax_site_capsule_builder_is_explicit_and_never_imports_jax() -> None:
    tree = ast.parse(BUILDER.read_text())
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "jax" not in imports
    assert "jaxlib" not in imports
    text = BUILDER.read_text()
    assert "ALLOWLIST = (" in text
    assert '"jax"' in text
    assert '"jaxlib"' in text
    assert '"libtpu"' not in text
    assert '"jax_plugins"' not in text
    assert 'shutil.ignore_patterns("__pycache__", "*.pyc")' in text
    assert "0o444" in text
    assert "_verify_future_unprivileged_access(output)" in text
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [_call_name(node) for node in calls]
    assert "lower" not in names
    assert "compiler_ir" not in names
    assert "compile" not in names


def test_gate_d_jax_site_capsule_remains_accessible_after_root_seal(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    root = tmp_path / "capsule"
    root.mkdir(mode=0o755)
    manifest = root / "CAPSULE_MANIFEST.json"
    builder._write_exclusive(manifest, b"{}\n")
    assert manifest.stat().st_mode & 0o777 == 0o444
    builder._verify_future_unprivileged_access(root)

    os.chmod(manifest, 0o400)
    with pytest.raises(SystemExit, match="file will be unreadable"):
        builder._verify_future_unprivileged_access(root)
    os.chmod(manifest, 0o444)

    inaccessible = root / "inaccessible"
    inaccessible.mkdir(mode=0o700)
    with pytest.raises(SystemExit, match="directory will be unreadable"):
        builder._verify_future_unprivileged_access(root)
