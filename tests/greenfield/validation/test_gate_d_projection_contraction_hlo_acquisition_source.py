from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from glm_tpu.greenfield.validation.gate_d_projection_contraction_hlo_acquisition_source import (
    ProjectionContractionHloSourceError,
    audit_projection_contraction_hlo_acquisition_source,
)

ROOT = Path(__file__).parents[3]
ACQUIRER = ROOT / "scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py"
ANALYZER = (
    ROOT / "scripts/greenfield/"
    "analyze_gate_d_projection_contraction_hlo_acquisition_source.py"
)
SOURCE_AUTHORITY = (
    ROOT / "docs/artifacts/gate-d-projection-contraction-pp16-source.json"
)
TOPOLOGY = ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json"
SOURCE_AUTHORITY_SHA256 = (
    "5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f"
)
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"


def _load_acquirer() -> ModuleType:
    name = "gate_d_projection_contraction_hlo_acquirer_test"
    spec = importlib.util.spec_from_file_location(name, ACQUIRER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _authorities() -> tuple[dict[str, object], dict[str, object]]:
    return json.loads(SOURCE_AUTHORITY.read_bytes()), json.loads(TOPOLOGY.read_bytes())


def _audit(source: bytes | None = None) -> dict[str, object]:
    source_authority, topology = _authorities()
    return audit_projection_contraction_hlo_acquisition_source(
        ACQUIRER.read_bytes() if source is None else source,
        source_authority=source_authority,
        source_authority_sha256=SOURCE_AUTHORITY_SHA256,
        topology=topology,
        topology_sha256=TOPOLOGY_SHA256,
    )


def test_static_audit_accepts_exact_compile_only_source() -> None:
    result = _audit()
    assert result["lower_call_count"] == 1
    assert result["compile_call_count"] == 1
    assert result["eval_shape_call_count"] == 1
    assert result["compiled_executable_invocation_count"] == 0
    assert result["tpu_numerical_execution_authorized"] is False
    assert result["pp16_stage_zero"]["device_ids"] == [0, 1]
    assert result["input_spec"] == [
        {
            "dtype": "bfloat16",
            "name": "normalized_hidden_bf16",
            "shape": [2, 1, 6144],
        },
        {
            "dtype": "float32",
            "name": "wk_weight_fp32",
            "shape": [2, 128, 6144],
        },
        {
            "dtype": "bfloat16",
            "name": "key_norm_weight_bf16",
            "shape": [2, 128],
        },
        {
            "dtype": "bfloat16",
            "name": "key_norm_bias_bf16",
            "shape": [2, 128],
        },
    ]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "compiled = lowered.compile()",
            "compiled = lowered.compile()\n    compiled()",
            "executable invocation",
        ),
        (
            "lowered = replay.lower(*arguments)",
            "lowered = replay.lower(*arguments)\n    jax.device_put(arguments[0])",
            "executable invocation",
        ),
        (
            "glm_tpu.greenfield.benchmarking.gate_d_projection_contraction_pp16",
            "glm_tpu.greenfield.benchmarking.gate_d_forced_round_pp16_hlo",
            "projection builder import",
        ),
        (
            '"compiled_executable_invocation_count": 0',
            '"compiled_executable_invocation_count": 1',
            "result authority",
        ),
        (
            '"normalized_hidden_bf16", (2, 1, 6144), "bfloat16"',
            '"normalized_hidden_bf16", (2, 32, 6144), "bfloat16"',
            "projection input spec",
        ),
        (
            "import argparse",
            'import argparse\ngetattr(__import__("os"), "system")("true")',
            "complete acquirer AST",
        ),
        (
            '"status": "HLO_ACQUIRED_UNADJUDICATED"',
            '"status": "SUCCESS"',
            "result authority",
        ),
    ],
)
def test_static_audit_rejects_hostile_source_mutations(
    old: str, new: str, message: str
) -> None:
    raw = ACQUIRER.read_text()
    assert raw.count(old) == 1
    with pytest.raises(ProjectionContractionHloSourceError, match=message):
        _audit(raw.replace(old, new, 1).encode())


def test_static_audit_rejects_source_and_topology_drift() -> None:
    source_authority, topology = _authorities()
    source_authority["authorization"]["hlo_acquisition"] = True
    with pytest.raises(ProjectionContractionHloSourceError, match="source authority"):
        audit_projection_contraction_hlo_acquisition_source(
            ACQUIRER.read_bytes(),
            source_authority=source_authority,
            source_authority_sha256=SOURCE_AUTHORITY_SHA256,
            topology=topology,
            topology_sha256=TOPOLOGY_SHA256,
        )
    source_authority, topology = _authorities()
    topology["pp16_lp2"]["groups"][0]["device_ids"] = [0, 2]
    with pytest.raises(ProjectionContractionHloSourceError, match="topology"):
        audit_projection_contraction_hlo_acquisition_source(
            ACQUIRER.read_bytes(),
            source_authority=source_authority,
            source_authority_sha256=SOURCE_AUTHORITY_SHA256,
            topology=topology,
            topology_sha256=TOPOLOGY_SHA256,
        )


def test_runtime_predecessor_validators_bind_exact_source_and_git_blobs() -> None:
    module = _load_acquirer()
    source_authority, topology = _authorities()
    stage_zero = module.validate_topology(topology)
    assert stage_zero["device_ids"] == [0, 1]
    authority = module.validate_projection_contraction_source(source_authority)
    assert authority["source_pin"] == "e3af2ca776ba5a789c6b2c0cc7bbd42258bbdc32"
    assert authority["builder_sha256"] == (
        "3c6a250e249cb0335d5a92ea5d1c862c9efaf128182c7887e39be8d95d867d4f"
    )
    pin = subprocess.check_output(
        ["/usr/bin/git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    module.verify_projection_contraction_source_git_blobs(pin, authority)


def test_runtime_source_validator_rejects_internal_authority_drift() -> None:
    module = _load_acquirer()
    source_authority, _ = _authorities()
    source_authority["source_audit"]["projection_precision"] = "default"
    with pytest.raises(RuntimeError, match="source authority"):
        module.validate_projection_contraction_source(source_authority)
    source_authority, _ = _authorities()
    path = next(iter(source_authority["direct_dependency_sha256s"]))
    source_authority["direct_dependency_sha256s"][path] = "0" * 64
    with pytest.raises(RuntimeError, match="source authority"):
        module.validate_projection_contraction_source(source_authority)


def test_sealed_archive_rejects_replace_refs_and_ignores_replacement(
    tmp_path: Path,
) -> None:
    module = _load_acquirer()
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q", str(repository)], check=True)
    package = repository / "glm_tpu"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("SAFE = True\n")
    subprocess.run(["/usr/bin/git", "-C", str(repository), "add", "."], check=True)
    commit = [
        "/usr/bin/git",
        "-C",
        str(repository),
        "-c",
        "user.name=Gate D Test",
        "-c",
        "user.email=gate-d@example.invalid",
        "commit",
        "-qm",
    ]
    subprocess.run([*commit, "safe"], check=True)
    safe_pin = subprocess.check_output(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    source.write_text("HOSTILE = True\n")
    subprocess.run(["/usr/bin/git", "-C", str(repository), "add", "."], check=True)
    subprocess.run([*commit, "hostile"], check=True)
    hostile_pin = subprocess.check_output(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "replace", safe_pin, hostile_pin],
        check=True,
    )
    assert (
        module._git_bytes("show", f"{safe_pin}:glm_tpu/__init__.py", repo=repository)
        == b"SAFE = True\n"
    )
    with pytest.raises(RuntimeError, match="replacement refs"):
        module._sealed_git_source_archive(safe_pin, repo=repository)
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "replace", "-d", safe_pin],
        check=True,
        capture_output=True,
    )
    descriptor, archive_path, manifest = module._sealed_git_source_archive(
        safe_pin, repo=repository
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.read("glm_tpu/__init__.py") == b"SAFE = True\n"
        assert manifest["file_manifest_count"] == 1
    finally:
        os.close(descriptor)


def test_abstract_output_and_hlo_surface_helpers_are_fail_closed() -> None:
    module = _load_acquirer()
    jax = SimpleNamespace(
        tree_util=SimpleNamespace(tree_leaves=lambda tree: tree),
    )
    leaves = [
        SimpleNamespace(shape=(2, 1, 6144), dtype="bfloat16"),
        SimpleNamespace(shape=(2, 1, 128), dtype="float32"),
        SimpleNamespace(shape=(2, 1, 128), dtype="float32"),
    ]
    records = module._abstract_record(leaves, jax)
    assert [item["name"] for item in records] == [
        "normalized_hidden_owners",
        "projected_key_owners",
        "current_key_owners",
    ]
    with pytest.raises(RuntimeError, match="abstract output"):
        module._abstract_record(leaves[:-1], jax)
    surface = module._hlo_surface(
        "HloModule projection\n%a = f32[] all-reduce(%x), replica_groups={{0,1}}\n"
    )
    assert surface["collective_count"] == 1
    assert surface["replica_group_ids"] == [0, 1]


def test_acquirer_imports_no_jax_and_requires_explicit_arguments() -> None:
    source = ACQUIRER.read_text()
    assert '"GLM_GATE_D_PROJECTION_CONTRACTION_HLO": "1"' in source
    assert "import ray" not in source
    assert "jax.distributed" not in source
    assert "block_until_ready" not in source
    assert "compiled(" not in source
    code = f"""
import importlib.util
import sys
from pathlib import Path
path = Path({str(ACQUIRER)!r})
spec = importlib.util.spec_from_file_location('projection_acquirer_import_probe', path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(int('jax' in sys.modules))
"""
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0\n"
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(ACQUIRER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "required" in result.stderr


def test_authority_files_have_exact_bytes() -> None:
    assert sha256(SOURCE_AUTHORITY.read_bytes()).hexdigest() == SOURCE_AUTHORITY_SHA256
    assert sha256(TOPOLOGY.read_bytes()).hexdigest() == TOPOLOGY_SHA256


def test_source_analyzer_is_default_off_cpu_only_and_imports_no_jax() -> None:
    source = ANALYZER.read_text()
    assert "import jax" not in source
    assert "libtpu" not in source.lower()
    assert "gcloud" not in source
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(ANALYZER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "requires python -I -S" in result.stderr
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-I", "-S", str(ANALYZER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "default-off" in result.stderr
    environment = {**os.environ, "GLM_GATE_D_PROJECTION_HLO_SOURCE": "1"}
    environment.pop("JAX_PLATFORMS", None)
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-I", "-S", str(ANALYZER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "CPU-pinned" in result.stderr


def test_source_analyzer_rejects_preloaded_spoof(tmp_path: Path) -> None:
    code = f"""
import importlib.util
import os
import sys
import types
from pathlib import Path
path = Path({str(ANALYZER)!r})
spec = importlib.util.spec_from_file_location('projection_hlo_source_analyzer', path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
spoof = types.ModuleType('glm_tpu')
spoof.__file__ = str(module.ROOT / 'glm_tpu/__init__.py')
spoof.__path__ = [str(module.ROOT / 'glm_tpu')]
sys.modules['glm_tpu'] = spoof
os.environ['GLM_GATE_D_PROJECTION_HLO_SOURCE'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
module.analyze()
"""
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-I", "-S", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "rejects preloaded glm_tpu" in result.stderr


def test_source_analyzer_rejects_hostile_import_hook(tmp_path: Path) -> None:
    code = f"""
import importlib.util
import os
import sys
from pathlib import Path
path = Path({str(ANALYZER)!r})
spec = importlib.util.spec_from_file_location('projection_hlo_source_analyzer', path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
class HostileFinder:
    def find_spec(self, fullname, path=None, target=None):
        return None
sys.meta_path.insert(0, HostileFinder())
os.environ['GLM_GATE_D_PROJECTION_HLO_SOURCE'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
module.analyze()
"""
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-I", "-S", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "rejects noncanonical import hooks" in result.stderr
