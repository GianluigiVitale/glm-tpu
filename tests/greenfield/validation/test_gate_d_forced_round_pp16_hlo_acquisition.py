from __future__ import annotations

import ast
import importlib.util
import json
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
DRIVER = ROOT / "scripts/greenfield/acquire_gate_d_forced_round_pp16_hlo.py"
BASE_DRIVER = ROOT / "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py"
ANALYZER = ROOT / (
    "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_acquisition_source.py"
)
SOURCE = ROOT / "docs/artifacts/gate-d-forced-round-pp16-hlo-source.json"
TOPOLOGY = ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json"
ARTIFACT = ROOT / (
    "docs/artifacts/gate-d-forced-round-pp16-hlo-acquisition-source.json"
)


def _load_driver():
    before = {
        name
        for name in sys.modules
        if name == "jax" or name.startswith(("jax.", "jaxlib", "jax_plugins"))
    }
    spec = importlib.util.spec_from_file_location(
        "gate_d_forced_round_hlo_acquisition_driver", DRIVER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    after = {
        name
        for name in sys.modules
        if name == "jax" or name.startswith(("jax.", "jaxlib", "jax_plugins"))
    }
    assert after == before
    return module


DRIVER_MODULE = _load_driver()


def _load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "gate_d_forced_round_hlo_acquisition_source_analyzer", ANALYZER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER_MODULE = _load_analyzer()


def _authority() -> tuple[dict[str, object], dict[str, object]]:
    return json.loads(SOURCE.read_text()), json.loads(TOPOLOGY.read_text())


def _replace(
    value: dict[str, object], path: tuple[object, ...], replacement: object
) -> None:
    current: object = value
    for key in path[:-1]:
        current = current[key]  # type: ignore[index]
    current[path[-1]] = replacement  # type: ignore[index]


def test_exact_forced_round_source_and_git_blobs_pass() -> None:
    source, topology = _authority()
    authority = DRIVER_MODULE.validate_forced_round_source(source, topology)
    assert authority["candidate_id"] == "forced_normalized_bf16_boundary"
    assert authority["builder_path"].endswith("gate_d_forced_round_pp16_hlo.py")
    assert len(authority["direct_runtime_dependency_sha256s"]) == 7
    DRIVER_MODULE.verify_forced_round_source_git_blobs(
        DRIVER_MODULE._git_head(), authority
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("candidate_id",), "other"),
        (("classification",), "PERSISTENCE_ONLY"),
        (("authorization", "hlo_acquisition"), True),
        (("authorization", "tpu_compile"), True),
        (("authorization", "tpu_execution"), True),
        (("runtime_authority", "topology_hash"), "0" * 64),
        (("runtime_authority", "pp16_stage_zero", "device_ids"), [0, 2]),
        (("source_authority", "builder_sha256"), "0" * 64),
        (("source_audit", "forced_round_call_count"), 2),
        (("source_audit", "normalized_feeds_qkv"), False),
        (("source_audit", "normalized_feeds_dsa"), False),
        (("source_audit", "normalized_is_rooted"), False),
        (("repository_authority", "unexpected_delta_paths"), ["hostile.py"]),
    ),
)
def test_forced_round_source_mutations_fail_closed(
    path: tuple[object, ...], replacement: object
) -> None:
    source, topology = _authority()
    _replace(source, path, replacement)
    with pytest.raises((RuntimeError, TypeError)):
        DRIVER_MODULE.validate_forced_round_source(source, topology)


def test_forced_round_dependency_catalogue_and_blob_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, topology = _authority()
    source = deepcopy(source)
    dependencies = source["repository_authority"]["direct_runtime_dependency_sha256s"]
    dependencies.pop(next(iter(dependencies)))
    with pytest.raises(RuntimeError, match="dependency authority"):
        DRIVER_MODULE.validate_forced_round_source(source, topology)

    source, topology = _authority()
    authority = DRIVER_MODULE.validate_forced_round_source(source, topology)
    monkeypatch.setattr(DRIVER_MODULE, "_git_bytes", lambda *args: b"hostile")
    with pytest.raises(RuntimeError, match="source blob drifted"):
        DRIVER_MODULE.verify_forced_round_source_git_blobs("a" * 40, authority)


def test_exact_one_row_two_owner_specs() -> None:
    inputs = {name: (shape, dtype) for name, shape, dtype in DRIVER_MODULE.INPUT_SPEC}
    outputs = {name: (shape, dtype) for name, shape, dtype in DRIVER_MODULE.OUTPUT_SPEC}
    assert len(inputs) == 16
    assert len(outputs) == 9
    assert inputs["rms_hidden_update_bf16"] == ((1, 6144), "bfloat16")
    assert inputs["rms_residual_bf16"] == ((1, 6144), "bfloat16")
    assert outputs["normalized_hidden_owners"] == ((2, 1, 6144), "bfloat16")
    assert outputs["selected_positions_owners"] == ((2, 1, 2048), "int32")
    assert "rms_input_fp32_owners" not in outputs


def test_driver_binds_source_before_jax_and_never_invokes_executable() -> None:
    source = DRIVER.read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    compile_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "compile"
    ]
    compiled_invocations = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "compiled"
    ]
    main_source = ast.unparse(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
    )
    assert len(compile_calls) == 1
    assert not compiled_invocations
    assert main_source.index("validate_forced_round_source(") < main_source.index(
        "import jax"
    )
    assert main_source.index(
        "verify_forced_round_source_git_blobs("
    ) < main_source.index("import jax")
    assert main_source.index("_open_inherited_run_dir(") < main_source.index(
        "import jax"
    )
    assert "jax.device_put" not in source
    assert "block_until_ready" not in source
    assert '"compiled_executable_invocation_count": 0' in source
    assert '"tpu_numerical_execution_performed": False' in source


def test_driver_retains_hardened_base_except_declared_candidate_delta() -> None:
    base = BASE_DRIVER.read_text()
    forced = DRIVER.read_text()
    required_common_tokens = (
        "_validate_python_runtime_storage",
        "_validate_dependency_sites",
        "_open_inherited_run_dir",
        "_sealed_git_source_archive",
        "_verify_project_imports",
        "_compiler_dependency_records",
        "_require_dependency_prefix_stable",
        "compiled = lowered.compile()",
    )
    for token in required_common_tokens:
        assert base.count(token) == forced.count(token)
    base_functions = {
        node.name for node in ast.parse(base).body if isinstance(node, ast.FunctionDef)
    }
    forced_functions = {
        node.name
        for node in ast.parse(forced).body
        if isinstance(node, ast.FunctionDef)
    }
    assert forced_functions - base_functions == {
        "validate_forced_round_source",
        "verify_forced_round_source_git_blobs",
    }
    assert not base_functions - forced_functions
    assert sha256(BASE_DRIVER.read_bytes()).hexdigest() == (
        "a4599e0d88a1ef8a2df897d1677196e8cd0db5009f12b866e7316ecaadb5dd15"
    )


def test_cli_requires_forced_round_certificate_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(DRIVER),
            "--expected-code-hash",
            "a" * 40,
            "--expected-driver-sha256",
            "b" * 64,
            "--admission-report",
            "/tmp/admission.json",
            "--admission-report-sha256",
            "c" * 64,
            "--topology-authority",
            "/tmp/topology.json",
            "--topology-authority-sha256",
            "d" * 64,
            "--forced-round-source",
            str(SOURCE),
            "--forced-round-source-sha256",
            DRIVER_MODULE.FORCED_ROUND_SOURCE_SHA256,
            "--compile-only",
            "1",
            "--run-dir",
            "/tmp/run",
            "--run-dir-fd",
            "7",
        ],
    )
    args = DRIVER_MODULE.parse_args()
    assert args.forced_round_source == SOURCE
    assert args.forced_round_source_sha256 == DRIVER_MODULE.FORCED_ROUND_SOURCE_SHA256


def test_source_certificate_sha_is_fixed_and_current() -> None:
    assert sha256(SOURCE.read_bytes()).hexdigest() == (
        DRIVER_MODULE.FORCED_ROUND_SOURCE_SHA256
    )


def test_acquisition_source_audit_and_repository_authority_pass() -> None:
    audit = ANALYZER_MODULE._audit_driver(DRIVER.read_bytes(), BASE_DRIVER.read_bytes())
    assert audit["compile_call_count"] == 1
    assert audit["compiled_executable_invocation_count"] == 0
    assert audit["output_spec_count"] == 9
    assert audit["forced_round_source_validation_precedes_jax"] is True
    repository = ANALYZER_MODULE._verify_repository_authority()
    assert repository["base_code_pin"] == ("2a050c1182991d93a7a4355a2820b044aa7a5861")
    assert repository["unexpected_delta_paths"] == []


@pytest.mark.parametrize(
    "hostile",
    (
        lambda raw: raw.replace(
            b'("normalized_hidden_owners", (2, 1, 6144), "bfloat16")',
            b'("normalized_hidden_owners", (2, 32, 6144), "bfloat16")',
            1,
        ),
        lambda raw: raw.replace(b"compiled.as_text()", b"compiled()", 1),
        lambda raw: raw.replace(
            b"return subprocess.check_output(", b"return subprocess.check_call(", 1
        ),
        lambda raw: raw.replace(
            b"build_gate_d_forced_round_pp16_hlo_replay",
            b"build_gate_d_compensated_pp16_hlo_replay",
            1,
        ),
    ),
)
def test_acquisition_source_audit_rejects_hostile_drift(hostile) -> None:
    raw = DRIVER.read_bytes()
    mutated = hostile(raw)
    assert mutated != raw
    with pytest.raises(RuntimeError):
        ANALYZER_MODULE._audit_driver(mutated, BASE_DRIVER.read_bytes())


def test_acquisition_source_certificate_regenerates_exactly() -> None:
    result = __import__("subprocess").run(
        [
            "/home/gianl/vllm-env/bin/python",
            "-I",
            "-S",
            "-B",
            str(ANALYZER),
        ],
        cwd=ROOT,
        env={
            "GLM_GATE_D_FORCED_ROUND_HLO_ACQUISITION_SOURCE": "1",
            "HOME": "/home/gianl",
            "JAX_PLATFORMS": "cpu",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stdout == ARTIFACT.read_bytes()
    assert result.stderr == b""


def test_acquisition_source_audit_is_default_off() -> None:
    result = __import__("subprocess").run(
        [
            "/home/gianl/vllm-env/bin/python",
            "-I",
            "-S",
            "-B",
            str(ANALYZER),
        ],
        cwd=ROOT,
        env={
            "HOME": "/home/gianl",
            "JAX_PLATFORMS": "cpu",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert b"default-off" in result.stderr
