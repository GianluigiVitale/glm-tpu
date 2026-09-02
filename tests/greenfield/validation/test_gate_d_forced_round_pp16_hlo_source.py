from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from glm_tpu.greenfield.benchmarking.gate_d_forced_round_pp16_hlo import (
    GateDForcedRoundPp16ReplayResult,
    build_gate_d_forced_round_pp16_hlo_replay,
)
from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.validation.gate_d_forced_round_hlo_source import (
    ForcedRoundHloSourceError,
    audit_forced_round_pp16_hlo_source,
)

ROOT = Path(__file__).parents[3]
BUILDER = ROOT / "glm_tpu/greenfield/benchmarking/gate_d_forced_round_pp16_hlo.py"
SCRIPT = ROOT / "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_source.py"
ARTIFACT = ROOT / "docs/artifacts/gate-d-forced-round-pp16-hlo-source.json"
SOURCE_COMMIT = "2a050c1182991d93a7a4355a2820b044aa7a5861"
PREDECESSOR = ROOT / "docs/artifacts/gate-d-forced-normalized-bf16-source-design.json"
TOPOLOGY = ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json"
RMSNORM = ROOT / "glm_tpu/greenfield/kernels/reference/rmsnorm.py"


def _load_analyzer_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_test_gate_d_forced_round_pp16_hlo_source_analyzer", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _FakeDevice:
    id: int
    platform: str
    device_kind: str = "TPU v4"
    process_index: int = 0
    coords: tuple[int, int, int] = (0, 0, 0)
    core_on_chip: int = 0


def test_builder_rejects_non_pp16_or_non_tpu_devices_before_mesh() -> None:
    with pytest.raises(PlanValidationError, match="two distinct devices"):
        build_gate_d_forced_round_pp16_hlo_replay(devices=(_FakeDevice(0, "tpu"),))
    with pytest.raises(PlanValidationError, match="exact PP16 stage-zero"):
        build_gate_d_forced_round_pp16_hlo_replay(
            devices=(_FakeDevice(0, "cpu"), _FakeDevice(1, "cpu"))
        )
    stage_zero = (
        _FakeDevice(0, "tpu", coords=(0, 0, 0)),
        _FakeDevice(1, "tpu", coords=(1, 0, 0)),
    )
    with pytest.raises(PlanValidationError, match="axis name must be exactly"):
        build_gate_d_forced_round_pp16_hlo_replay(
            devices=stage_zero,
            axis_name="",
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
        (
            _FakeDevice(0, "tpu"),
            _FakeDevice(1, "tpu", coords=(0, 0, 1)),
        ),
    ],
)
def test_builder_rejects_non_stage_zero_topology(
    devices: tuple[_FakeDevice, ...],
) -> None:
    with pytest.raises(PlanValidationError, match="exact PP16 stage-zero"):
        build_gate_d_forced_round_pp16_hlo_replay(devices=devices)


def test_builder_result_contract_has_only_primary_rooted_outputs() -> None:
    assert GateDForcedRoundPp16ReplayResult._fields == (
        "normalized_hidden_owners",
        "query_owners",
        "head_weights_owners",
        "current_key_owners",
        "index_cache_owners",
        "selected_positions_owners",
        "valid_counts_owners",
        "selected_scores_owners",
        "contract_valid_owners",
    )


def test_static_audit_proves_forced_round_primary_lineage() -> None:
    result = audit_forced_round_pp16_hlo_source(BUILDER.read_bytes())
    assert result["tpu_device_count_required"] == 2
    assert result["pp16_group"] == [0, 1]
    assert result["forced_round_call_count"] == 1
    assert result["normalized_feeds_qkv"] is True
    assert result["normalized_feeds_dsa"] is True
    assert result["normalized_is_rooted"] is True
    assert result["output_field_count"] == 9
    assert result["hlo_lowering_or_compile_call_count"] == 0
    assert result["in_spec_count"] == 16
    assert result["pp16_stage_id"] == 0
    assert result["pp16_stage_process_index"] == 0
    assert result["pp16_stage_coordinates"] == [[0, 0, 0], [1, 0, 0]]
    assert len(result["module_ast_sha256"]) == 64


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "precomputed_normalized=normalized",
            "precomputed_normalized=residual",
            "DSA input lineage drifted",
        ),
        (
            "normalized[None, ...]",
            "residual[None, ...]",
            "primary output lineage drifted",
        ),
        (
            "return jax.jit(",
            "return jax.jit(jax.jit(lambda: 0).lower(),",
            "lowering, compilation, or execution",
        ),
        (
            '(0, "tpu", "TPU v4", 0, (0, 0, 0), 0)',
            '(0, "tpu", "TPU v5", 0, (0, 0, 0), 0)',
            "platform/group contract drifted",
        ),
        (
            "groups = ((0, 1),)",
            "groups = ((0, 0),)",
            "platform/group contract drifted",
        ),
        (
            "check_vma=False",
            "check_vma=True",
            "JIT/shard_map wrapper drifted",
        ),
        (
            "dsa.internals.query[None, ...]",
            "dsa.internals.current_key[None, ...]",
            "primary output lineage drifted",
        ),
        (
            "axis_index_groups=groups",
            "axis_index_groups=None",
            "DSA input lineage drifted",
        ),
        (
            "in_specs=(\n                P(),",
            "in_specs=(\n                P(axis_name),",
            "JIT/shard_map wrapper drifted",
        ),
        (
            "from __future__ import annotations",
            "from __future__ import annotations\n\njax.devices()",
            "builder module AST drifted",
        ),
        (
            "    owner_matrix = P(axis_name, None, None)",
            "    mapped()\n    owner_matrix = P(axis_name, None, None)",
            "builder module AST drifted",
        ),
        (
            "from __future__ import annotations",
            'from __future__ import annotations\n\ngetattr(object(), "lower")()',
            "builder module AST drifted",
        ),
    ],
)
def test_static_audit_rejects_hostile_source_mutations(
    old: str, new: str, message: str
) -> None:
    source = BUILDER.read_text(encoding="utf-8")
    assert source.count(old) == 1
    hostile = source.replace(old, new).encode("utf-8")
    with pytest.raises(ForcedRoundHloSourceError, match=message):
        audit_forced_round_pp16_hlo_source(hostile)


def test_cli_is_default_off_cpu_only_and_imports_no_jax() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import jax" not in source
    assert '"jaxlib"' in source
    assert '"jax_plugins"' in source
    assert "_loaded_jax_modules()" in source
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "default-off" in result.stderr

    environment = {**os.environ, "GLM_GATE_D_FORCED_ROUND_HLO_SOURCE": "1"}
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


def test_analyzer_binds_predecessor_function_and_rejects_drift() -> None:
    module = _load_analyzer_module()
    predecessor = json.loads(PREDECESSOR.read_bytes())
    rmsnorm_raw = RMSNORM.read_bytes()
    authority = module._verify_predecessor_function_authority(predecessor, rmsnorm_raw)
    assert authority == {
        "forced_function_module_sha256": (
            "d6fca18425fb851c2ba3e1a12dacd45c8dadef43b33d6a1e6da72dbfb2eb6276"
        ),
        "forced_function_source_sha256": (
            "d73eaf1f9c8d485769513abe835bbb5ce72d61f4d495a4dedc65b7cfeb0e7a2b"
        ),
    }
    assert rmsnorm_raw.count(b"mantissa_bits=7") == 1
    with pytest.raises(RuntimeError, match="function authority drifted"):
        module._verify_predecessor_function_authority(
            predecessor, rmsnorm_raw.replace(b"mantissa_bits=7", b"mantissa_bits=6")
        )
    hostile_predecessor = copy.deepcopy(predecessor)
    hostile_predecessor["source_authority"]["function_source_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="function authority drifted"):
        module._verify_predecessor_function_authority(hostile_predecessor, rmsnorm_raw)


def test_analyzer_binds_exact_pp16_stage_zero_topology() -> None:
    module = _load_analyzer_module()
    topology = json.loads(TOPOLOGY.read_bytes())
    assert module._verify_topology_authority(topology) == {
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "device_ids": [0, 1],
        "process_index": 0,
        "stage_id": 0,
    }
    hostile = copy.deepcopy(topology)
    hostile["pp16_lp2"]["groups"][0]["coordinates"][1] = [0, 1, 0]
    with pytest.raises(RuntimeError, match="stage-zero authority drifted"):
        module._verify_topology_authority(hostile)


def test_analyzer_rejects_repository_delta_outside_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_analyzer_module()

    def hostile_git(arguments: list[str], *, expected_returncode: int = 0) -> bytes:
        del expected_returncode
        if arguments[0] == "rev-parse":
            return f"{module.BASE_TREE_ID}\n".encode("ascii")
        if arguments[0] == "merge-base":
            return b""
        if arguments[0] == "diff":
            paths = sorted(module.ALLOWED_DELTA_PATHS | {"unexpected_runtime.py"})
            return "".join(f"{path}\0" for path in paths).encode("utf-8")
        if arguments[0] == "ls-files":
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr(module, "_git", hostile_git)
    with pytest.raises(RuntimeError, match="repository delta drifted"):
        module._verify_repository_authority()


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="tracked certificate absent")
def test_cli_regenerates_tracked_certificate_exactly(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "source-authority"
    clone = subprocess.run(
        [
            "/usr/bin/git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(ROOT),
            str(replay),
        ],
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        check=False,
    )
    assert clone.returncode == 0, clone.stderr.decode("utf-8", errors="replace")
    checkout = subprocess.run(
        ["/usr/bin/git", "checkout", "--quiet", "--detach", SOURCE_COMMIT],
        cwd=replay,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        check=False,
    )
    assert checkout.returncode == 0, checkout.stderr.decode("utf-8", errors="replace")
    probe = f"""
import importlib.util
from pathlib import Path
root = Path({str(replay)!r})
script = root / "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_source.py"
spec = importlib.util.spec_from_file_location("_historical_source_analyzer", script)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.WORKTREE = root
module.ANALYZER = script
module.BUILDER = root / "glm_tpu/greenfield/benchmarking/gate_d_forced_round_pp16_hlo.py"
module.AUDITOR = root / "glm_tpu/greenfield/validation/gate_d_forced_round_hlo_source.py"
module.SOURCE_DESIGN_ARTIFACT = root / "docs/artifacts/gate-d-forced-normalized-bf16-source-design.json"
module.TOPOLOGY_AUTHORITY = root / "docs/artifacts/gate-d-runtime-locality-authority.json"
module.RMSNORM_MODULE = root / "glm_tpu/greenfield/kernels/reference/rmsnorm.py"
raise SystemExit(module.main())
"""
    result = subprocess.run(
        [
            "/home/gianl/vllm-env/bin/python",
            "-I",
            "-S",
            "-B",
            "-c",
            probe,
        ],
        env={
            "GLM_GATE_D_FORCED_ROUND_HLO_SOURCE": "1",
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
