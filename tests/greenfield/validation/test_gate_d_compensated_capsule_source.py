from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY = (
    REPO_ROOT
    / "glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py"
)


def _tree() -> ast.Module:
    return ast.parse(REPLAY.read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _attribute_path(node: ast.expr) -> str:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def test_compensated_replay_roots_auxiliary_without_primary_substitution() -> None:
    tree = _tree()
    mapped = _function(tree, "mapped")
    calls = [node for node in ast.walk(mapped) if isinstance(node, ast.Call)]
    names = [_call_name(node) for node in calls]
    assert names.count("fused_add_rms_norm_with_compensated_auxiliary") == 1
    assert "fused_add_rms_norm_with_auxiliary" not in names

    result_calls = [
        node
        for node in calls
        if _call_name(node) == "GateDCompensatedCapsuleReplayResult"
    ]
    assert len(result_calls) == 1
    result = result_calls[0]
    assert len(result.args) == 10
    assert isinstance(result.args[0], ast.Subscript)
    assert _attribute_path(result.args[0].value) == "rms.restored_rms_input_fp32"
    assert isinstance(result.args[1], ast.Subscript)
    assert _attribute_path(result.args[1].value) == "rms.output"

    restored_uses = [
        node
        for node in ast.walk(mapped)
        if isinstance(node, ast.Attribute)
        and _attribute_path(node) == "rms.restored_rms_input_fp32"
    ]
    assert len(restored_uses) == 1
    dsa_calls = [node for node in calls if _call_name(node) == "stage_local_dsa_fp8_mapped"]
    assert len(dsa_calls) == 1
    normalized = [
        keyword.value
        for keyword in dsa_calls[0].keywords
        if keyword.arg == "precomputed_normalized"
    ]
    assert len(normalized) == 1
    assert _attribute_path(normalized[0]) == "rms.output"


def test_compensated_replay_is_distinct_from_tombstoned_tuple_path() -> None:
    text = REPLAY.read_text()
    assert "build_gate_d_compensated_capsule_cpu_replay" in text
    assert "GateDCompensatedCapsuleReplayResult" in text
    assert "fused_add_rms_norm_with_compensated_auxiliary" in text
    assert "build_gate_d_tuple_capsule_cpu_replay" not in text
    assert "GateDTupleCapsuleReplayResult" not in text
    assert "fused_add_rms_norm_with_auxiliary(" not in text


def test_compensated_replay_is_forced_to_one_local_cpu_pair() -> None:
    text = REPLAY.read_text()
    for required in (
        "len(runtime_devices) != 2",
        'device.platform != "cpu"',
        "local_parallel_size=2",
        "groups = ((0, 1),)",
        "axis_index_groups=groups",
        "P(axis_name, None, None, None)",
        "check_vma=False",
    ):
        assert required in text
    for forbidden in (
        "pmap(",
        "process_allgather",
        "multihost_utils",
        "libtpu",
        "jax_plugins",
        "subprocess",
        "import ray",
        "ray.",
    ):
        assert forbidden not in text


def test_compensated_replay_has_no_import_time_execution() -> None:
    tree = _tree()
    allowed = (
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.ClassDef,
        ast.FunctionDef,
    )
    assert all(isinstance(node, allowed) for node in tree.body)
    assert not any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )
