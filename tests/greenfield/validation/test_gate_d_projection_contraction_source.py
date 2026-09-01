from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.gate_d_projection_contraction_pp16 import (
    GateDProjectionContractionPp16Result,
    build_gate_d_projection_contraction_pp16,
)
from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.validation.gate_d_projection_contraction_source import (
    EXPECTED_DIRECT_DEPENDENCY_SHA256S,
    ProjectionContractionSourceError,
    audit_projection_contraction_pp16_source,
)

ROOT = Path(__file__).parents[3]
BUILDER = ROOT / "glm_tpu/greenfield/benchmarking/gate_d_projection_contraction_pp16.py"
SCRIPT = ROOT / "scripts/greenfield/analyze_gate_d_projection_contraction_source.py"
ARTIFACT = ROOT / "docs/artifacts/gate-d-projection-contraction-pp16-source.json"
ARTIFACT_SHA256 = "5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f"
PREDECESSOR = (
    ROOT / "docs/artifacts/gate-d-projection-arithmetic-frontier-analysis.json"
)
TOPOLOGY = ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json"


@dataclass(frozen=True)
class _FakeDevice:
    id: int
    platform: str
    device_kind: str = "TPU v4"
    process_index: int = 0
    coords: tuple[int, int, int] = (0, 0, 0)
    core_on_chip: int = 0


def _evidence() -> tuple[dict[str, object], dict[str, object]]:
    return json.loads(PREDECESSOR.read_bytes()), json.loads(TOPOLOGY.read_bytes())


def _dependencies() -> dict[str, str]:
    return {
        relative: sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in EXPECTED_DIRECT_DEPENDENCY_SHA256S
    }


def _audit(source: bytes | None = None) -> dict[str, object]:
    predecessor, topology = _evidence()
    return audit_projection_contraction_pp16_source(
        BUILDER.read_bytes() if source is None else source,
        dependency_sha256s=_dependencies(),
        predecessor=predecessor,
        topology=topology,
    )


def test_builder_rejects_non_pp16_or_non_tpu_devices_before_mesh() -> None:
    with pytest.raises(PlanValidationError, match="two distinct"):
        build_gate_d_projection_contraction_pp16(devices=(_FakeDevice(0, "tpu"),))
    with pytest.raises(PlanValidationError, match="exact PP16"):
        build_gate_d_projection_contraction_pp16(
            devices=(_FakeDevice(0, "cpu"), _FakeDevice(1, "cpu"))
        )
    stage_zero = (
        _FakeDevice(0, "tpu", coords=(0, 0, 0)),
        _FakeDevice(1, "tpu", coords=(1, 0, 0)),
    )
    with pytest.raises(PlanValidationError, match="axis name"):
        build_gate_d_projection_contraction_pp16(
            devices=stage_zero,
            axis_name="model",
        )


@pytest.mark.parametrize(
    "devices",
    [
        (_FakeDevice(1, "tpu", coords=(1, 0, 0)), _FakeDevice(0, "tpu")),
        (_FakeDevice(0, "tpu"), _FakeDevice(2, "tpu", coords=(0, 1, 0))),
        (
            _FakeDevice(0, "tpu", device_kind="TPU v5"),
            _FakeDevice(1, "tpu", coords=(1, 0, 0)),
        ),
        (
            _FakeDevice(0, "tpu", process_index=1),
            _FakeDevice(1, "tpu", coords=(1, 0, 0)),
        ),
    ],
)
def test_builder_rejects_topology_drift(devices: tuple[_FakeDevice, ...]) -> None:
    with pytest.raises(PlanValidationError, match="exact PP16"):
        build_gate_d_projection_contraction_pp16(devices=devices)


def test_result_contract_roots_only_the_causal_frontier() -> None:
    assert GateDProjectionContractionPp16Result._fields == (
        "normalized_hidden_owners",
        "projected_key_owners",
        "current_key_owners",
    )


def test_two_cpu_device_shard_map_is_abstractly_constructable() -> None:
    code = """
import json
import jax
import jax.numpy as jnp
from glm_tpu.greenfield.benchmarking.gate_d_projection_contraction_pp16 import _build_projection_contraction_shard_map
devices = tuple(jax.devices())
assert jax.default_backend() == 'cpu' and len(devices) == 2
replay = _build_projection_contraction_shard_map(devices=devices, axis_name='feature')
outputs = jax.eval_shape(
    replay,
    jax.ShapeDtypeStruct((2, 1, 6144), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 128, 6144), jnp.float32),
    jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),
)
print(json.dumps([{'shape': list(value.shape), 'dtype': str(value.dtype)} for value in outputs]))
"""
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {"dtype": "bfloat16", "shape": [2, 1, 6144]},
        {"dtype": "float32", "shape": [2, 1, 128]},
        {"dtype": "float32", "shape": [2, 1, 128]},
    ]


def test_static_audit_accepts_exact_minimal_source() -> None:
    result = _audit()
    assert result["one_live_row"] is True
    assert result["projection_call_count"] == 1
    assert result["projection_dtype"] == "float32"
    assert result["projection_precision"] == "highest"
    assert result["suffix_key_norm_mode"] == "divide_sqrt"
    assert result["collective_call_count"] == 0
    assert result["compiled_executable_invocation_count"] == 0
    assert result["tpu_compile_or_execution_performed"] is False
    assert result["pp16_stage_zero"]["device_ids"] == [0, 1]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "normalized = lax.optimization_barrier(normalized_hidden_owner[0])",
            "normalized = normalized_hidden_owner[0]",
            "input barrier",
        ),
        (
            'with jax.default_matmul_precision("highest"):',
            'with jax.default_matmul_precision("default"):',
            "highest precision",
        ),
        (
            "output_dtype=jnp.float32",
            "output_dtype=jnp.bfloat16",
            "projection input/dtype",
        ),
        (
            'key_norm_mode="divide_sqrt"',
            'key_norm_mode="multiply_rsqrt"',
            "key suffix lineage",
        ),
        (
            "check_vma=True",
            "check_vma=False",
            "shard_map contract",
        ),
        (
            "return jax.jit(",
            "jax.devices()\n    return jax.jit(",
            "lowering, compilation, execution",
        ),
        (
            "    return _build_projection_contraction_shard_map(",
            (
                '    run = __import__("jax").devices\n'
                "    run()\n"
                "    return _build_projection_contraction_shard_map("
            ),
            "complete builder module AST",
        ),
        (
            "    return _build_projection_contraction_shard_map(",
            (
                '    run = getattr(__import__("jax"), "devices")\n'
                "    run()\n"
                "    return _build_projection_contraction_shard_map("
            ),
            "complete builder module AST",
        ),
        (
            "import numpy as np",
            'import numpy as np\ngetattr(__import__("os"), "system")("true")',
            "complete builder module AST",
        ),
        (
            "projected_key_owners: Any",
            "query_owners: Any",
            "rooted output contract",
        ),
        (
            "        projected_key[None, ...],\n        current_key[None, ...],",
            "        current_key[None, ...],\n        current_key[None, ...],",
            "rooted output lineage",
        ),
        (
            "normalized = lax.optimization_barrier(normalized_hidden_owner[0])",
            (
                "normalized = lax.optimization_barrier(normalized_hidden_owner[0])\n"
                "    lax.psum(normalized, axis_name)"
            ),
            "contains a collective",
        ),
    ],
)
def test_static_audit_rejects_causal_or_authority_drift(
    old: str, new: str, message: str
) -> None:
    raw = BUILDER.read_text()
    assert raw.count(old) == 1
    with pytest.raises(ProjectionContractionSourceError, match=message):
        _audit(raw.replace(old, new, 1).encode())


def test_predecessor_and_topology_drift_fail_closed() -> None:
    predecessor, topology = _evidence()
    predecessor["variant_count"] = 26
    with pytest.raises(ProjectionContractionSourceError, match="predecessor"):
        audit_projection_contraction_pp16_source(
            BUILDER.read_bytes(),
            dependency_sha256s=_dependencies(),
            predecessor=predecessor,
            topology=topology,
        )
    predecessor, topology = _evidence()
    topology["pp16_lp2"]["groups"][0]["device_ids"] = [0, 2]
    with pytest.raises(ProjectionContractionSourceError, match="topology"):
        audit_projection_contraction_pp16_source(
            BUILDER.read_bytes(),
            dependency_sha256s=_dependencies(),
            predecessor=predecessor,
            topology=topology,
        )


def test_direct_dependency_drift_fails_closed() -> None:
    predecessor, topology = _evidence()
    dependencies = _dependencies()
    path = next(iter(dependencies))
    dependencies[path] = "0" * 64
    with pytest.raises(ProjectionContractionSourceError, match="dependency"):
        audit_projection_contraction_pp16_source(
            BUILDER.read_bytes(),
            dependency_sha256s=dependencies,
            predecessor=predecessor,
            topology=topology,
        )


def test_source_analyzer_forces_authenticated_root_before_shadow_package(
    tmp_path: Path,
) -> None:
    shadow = tmp_path / "shadow"
    package = shadow / "glm_tpu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise RuntimeError("hostile shadow glm_tpu imported")\n'
    )
    code = f"""
import importlib.util
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

script = Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location('projection_source_analyzer', script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module._committed_sources = lambda: (
    '0' * 40,
    {{
        relative: sha256((module.ROOT / relative).read_bytes()).hexdigest()
        for relative in (module.SOURCE_PATH, module.BUILDER_PATH, module.VALIDATOR_PATH)
    }},
    dict(module.DIRECT_DEPENDENCY_SHA256S),
)
os.environ['GLM_GATE_D_PROJECTION_CONTRACTION_SOURCE'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
result = module.analyze()
validator = sys.modules['glm_tpu.greenfield.validation.gate_d_projection_contraction_source']
print(json.dumps({{
    'classification': result['classification'],
    'import_root': sys.path[0],
    'validator_path': str(Path(validator.__file__).resolve(strict=True)),
}}))
"""
    environment = {
        **os.environ,
        "PYTHONPATH": f"{shadow}:{ROOT}",
    }
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed == {
        "classification": (
            "PROJECTION_ONLY_PP16_SOURCE_ACCEPTED;COMPILE_UNPROVEN;"
            "TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN"
        ),
        "import_root": str(ROOT),
        "validator_path": str(
            (
                ROOT / "glm_tpu/greenfield/validation/"
                "gate_d_projection_contraction_source.py"
            ).resolve(strict=True)
        ),
    }


def test_source_analyzer_rejects_preloaded_validator_module(tmp_path: Path) -> None:
    code = f"""
import importlib.util
import os
import sys
import types
from pathlib import Path

script = Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location('projection_source_analyzer', script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
hostile = types.ModuleType('glm_tpu')
hostile.__file__ = str(module.ROOT / 'glm_tpu/__init__.py')
hostile.__path__ = [str(module.ROOT / 'glm_tpu')]
sys.modules['glm_tpu'] = hostile
spoofed = types.ModuleType(
    'glm_tpu.greenfield.validation.gate_d_projection_contraction_source'
)
spoofed.__file__ = str(module.ROOT / module.VALIDATOR_PATH)
sys.modules[spoofed.__name__] = spoofed
os.environ['GLM_GATE_D_PROJECTION_CONTRACTION_SOURCE'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
module.analyze()
"""
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-S", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "rejects preloaded glm_tpu modules" in result.stderr


def test_source_analyzer_is_default_off_cpu_only_and_nonmutating() -> None:
    source = SCRIPT.read_text()
    assert "sys.path[:] = [entry for entry in sys.path if entry != ROOT_TEXT]" in source
    assert "sys.path.insert(0, ROOT_TEXT)" in source
    assert "validator_path != expected_validator_path" in source
    assert source.count("_reject_preloaded_glm_tpu()") == 3
    assert "gcloud" not in source
    assert "jax.distributed" not in source
    assert "libtpu" not in source.lower()
    assert "import ray" not in source
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "default-off" in result.stderr
    environment = {
        **os.environ,
        "GLM_GATE_D_PROJECTION_CONTRACTION_SOURCE": "1",
    }
    environment.pop("JAX_PLATFORMS", None)
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "CPU-pinned" in result.stderr


def test_source_artifact_exactly_replays_committed_analyzer() -> None:
    artifact = ARTIFACT.read_bytes()
    assert len(artifact) == 4176
    assert sha256(artifact).hexdigest() == ARTIFACT_SHA256
    parsed = json.loads(artifact)
    assert parsed["code_hash"] == "e3af2ca776ba5a789c6b2c0cc7bbd42258bbdc32"
    assert parsed["gate_d_closed"] is False
    assert parsed["authorization"] == {
        "full_dsa_or_8k": False,
        "hlo_acquisition": False,
        "persistence_only": True,
        "tpu_compile": False,
        "tpu_execution": False,
    }
    environment = {
        **os.environ,
        "GLM_GATE_D_PROJECTION_CONTRACTION_SOURCE": "1",
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
    }
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == artifact
