from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCER = (
    REPO_ROOT
    / "scripts/greenfield/produce_gate_d_compensated_auxiliary_stablehlo.py"
)


def _load_producer() -> object:
    spec = importlib.util.spec_from_file_location(
        "gate_d_compensated_stablehlo_producer_test", PRODUCER
    )
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


def test_compensated_stablehlo_producer_is_lowering_only_and_default_off() -> None:
    text = PRODUCER.read_text()
    assert text.splitlines()[0] == (
        "#!/usr/bin/env -S "
        "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12 -I -S"
    )
    tree = ast.parse(text)
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
    for forbidden in (
        "block_until_ready",
        "device_get",
        "device_put",
        "make_array_from_callback",
        "pmap",
    ):
        assert forbidden not in names
    main_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(main_guards) == 1


def test_compensated_stablehlo_producer_binds_distinct_candidate_identity() -> None:
    text = PRODUCER.read_text()
    expected = {
        "produce_gate_d_compensated_auxiliary_stablehlo.py",
        "gate-d-compensated-auxiliary-source-authority.json",
        "gate-d-compensated-auxiliary-pp16-plan-authority.json",
        "compensated_auxiliary_dependency",
        "fused_add_rms_norm_with_compensated_auxiliary",
        "gate_d_compensated_auxiliary_rms",
        "e16d74fcc025f5ffb910d9cbab1c4fa06df887ad",
        "5434644b42325393df94169755ad9bf510b95749e907668006e50375bf73feb5",
        "451f4fe010ffd38046ae5716742cb756ce54853bb46b8d6199af2e854946df2a",
        "ac08b3c776e80c1c0510401dbd6cb58f2fdc8c008df2b8605c707a5e954e1eec",
        "237095c7ac9acdd7a37383b951b061752ee83f591722c3a2e2d3bede59326fb0",
        "7d0a5615ff4744801ac6a6598a52ea80e17d431788e4ef21a9d889e81772dbdf",
    }
    for value in expected:
        assert value in text
    assert '"fused_add_rms_norm_with_auxiliary"' not in text
    assert "gate_d_tuple_auxiliary_rms" not in text
    assert "/produce_gate_d_tuple_auxiliary_stablehlo.py" not in text


def test_compensated_stablehlo_producer_is_append_only_and_cpu_forced() -> None:
    text = PRODUCER.read_text()
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        _call_name(node) == "open"
        and any(
            isinstance(argument, ast.Name) and argument.id == "flags"
            for argument in node.args
        )
        for node in calls
    )
    for required in (
        "os.O_EXCL",
        "os.O_NOFOLLOW",
        'os.environ["JAX_PLATFORMS"] = "cpu"',
        'os.environ["JAX_PLATFORM_NAME"] = "cpu"',
        'jax.devices("cpu")',
        "sys.dont_write_bytecode = True",
        '"SUCCESS"',
    ):
        assert required in text


def test_compensated_stablehlo_producer_seals_source_snapshot(
    tmp_path: Path,
) -> None:
    producer = _load_producer()
    source = tmp_path / "source.py"
    payload = b"answer = 42\n"
    source.write_bytes(payload)
    assert producer._sealed_source_snapshot(source, sha256(payload).hexdigest()) == payload
