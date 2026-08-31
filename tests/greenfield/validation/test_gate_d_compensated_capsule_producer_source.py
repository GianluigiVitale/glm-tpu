from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCER = (
    REPO_ROOT
    / "scripts/greenfield/produce_gate_d_compensated_auxiliary_capsule.py"
)
TUPLE_PRODUCER = (
    REPO_ROOT / "scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py"
)


def _load_producer() -> object:
    spec = importlib.util.spec_from_file_location(
        "gate_d_compensated_capsule_producer_test", PRODUCER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _normalized_dump(node: ast.AST) -> str:
    value = ast.dump(node, include_attributes=False)
    replacements = (
        ("compensated_auxiliary_dependency", "auxiliary_device_tuple_dependency"),
        ("compensated-auxiliary", "tuple-auxiliary"),
        ("compensated_auxiliary", "tuple_auxiliary"),
        ("compensated auxiliary", "tuple auxiliary"),
        ("compensated capsule", "tuple capsule"),
        ("compensated candidate", "tuple candidate"),
        ("gate_d_compensated_capsule", "gate_d_tuple_capsule"),
        (
            "build_gate_d_compensated_capsule_cpu_replay",
            "build_gate_d_tuple_capsule_cpu_replay",
        ),
    )
    for current, historical in replacements:
        value = value.replace(current, historical)
    return value


def test_compensated_capsule_producer_retains_audited_execution_machinery() -> None:
    compensated = _functions(PRODUCER)
    historical = _functions(TUPLE_PRODUCER)
    assert set(compensated) == set(historical)
    for name in sorted(historical):
        assert _normalized_dump(compensated[name]) == ast.dump(
            historical[name], include_attributes=False
        ), name


def test_compensated_capsule_producer_binds_distinct_candidate_authorities() -> None:
    producer = _load_producer()
    assert producer.PRODUCER_INSTALLED == Path(
        "/opt/glm-tpu/bin/produce_gate_d_compensated_auxiliary_capsule.py"
    )
    assert producer.PRODUCER_SOURCE == PRODUCER
    assert producer.SOURCE_CODE_PIN == "e16d74fcc025f5ffb910d9cbab1c4fa06df887ad"
    assert producer.SOURCE_AUTHORITY_FILE_SHA256 == (
        "237095c7ac9acdd7a37383b951b061752ee83f591722c3a2e2d3bede59326fb0"
    )
    assert producer.SOURCE_AUTHORITY_SHA256 == (
        "c38492a0c058273b0e3d14aa464cd5fd77d61f88a8f4f9577d161f07a6b4b4c3"
    )
    assert producer.PLAN_AUTHORITY_FILE_SHA256 == (
        "7d0a5615ff4744801ac6a6598a52ea80e17d431788e4ef21a9d889e81772dbdf"
    )
    assert producer.PLAN_SHA256 == (
        "eb2c050b14a6125c1bfb59a7ce0021d7c6c30714834f324e8a18a1a4db57b6dc"
    )
    assert producer.STABLEHLO_AUTHORITY_FILE_SHA256 == (
        "af109e0f7d4d355373212028fd958af9e8566c302183b2fd5482e96805c8eb90"
    )
    assert producer.STABLEHLO_AUTHORITY_SHA256 == (
        "e1b2e4105f247beb6af46325b76bd359df84666911959a7960b4179947d90185"
    )
    assert producer.CANDIDATE_SOURCE_FILES == {
        "glm_tpu/greenfield/kernels/layer.py": (
            "47c48b20564c0338de2bca70c5e01a9305748010e0f28f94e525176c1b882a0c"
        ),
        "glm_tpu/greenfield/kernels/reference/rmsnorm.py": (
            "b707ddaa4208a58c7a1a999fc0d40b8a2b460d64c15943569c03444a71e3edfd"
        ),
    }


def test_compensated_capsule_producer_cannot_select_tuple_candidate() -> None:
    text = PRODUCER.read_text()
    assert text.splitlines()[0] == (
        "#!/usr/bin/env -S "
        "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12 -I -S"
    )
    assert text.count('"id": "compensated_auxiliary_dependency"') == 1
    assert text.count('"candidate_id": "compensated_auxiliary_dependency"') == 1
    assert "auxiliary_device_tuple_dependency" not in text
    assert "gate_d_tuple_capsule" not in text
    assert "build_gate_d_tuple_capsule_cpu_replay" not in text
    assert "produce_gate_d_tuple_auxiliary_capsule.py" not in text
    assert "build_gate_d_compensated_capsule_cpu_replay" in text


def test_compensated_capsule_producer_is_append_only_forced_cpu_and_default_off() -> None:
    text = PRODUCER.read_text()
    for required in (
        "os.O_EXCL",
        "os.O_NOFOLLOW",
        "_sealed_git_source_archive",
        "_verify_runtime_data_mount",
        "_revalidate_tensor_receipts",
        'os.environ["JAX_PLATFORMS"] = "cpu"',
        'os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"',
        'if "libtpu" in sys.modules',
        '"jax_plugins_loaded": False',
        '"tpus_used": 0',
        '_write_exclusive(output_fd, "SUCCESS", success_payload)',
    ):
        assert required in text
    tree = ast.parse(text)
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(guards) == 1
