from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from glm_tpu.greenfield.validation.gate_d_projection_contraction_hlo import (
    EXPECTED_OPTIMIZED_HLO_SHA256,
    EXPECTED_REMOTE_REPLAY,
    EXPECTED_REMOTE_REPLAY_SHA256,
    EXPECTED_RUN_DIR,
    EXPECTED_STABLEHLO_SHA256,
    ProjectionContractionHloError,
    _audit_remote_replay,
    adjudicate_diagnostic_run,
    audit_optimized_hlo_structure,
    audit_stablehlo_structure,
)

OPTIMIZED = (
    EXPECTED_RUN_DIR / "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt"
)
STABLE = EXPECTED_RUN_DIR / "hlo/projection_contraction_pp16_stage0.stablehlo.mlir"
MODULE_SOURCE = Path(__file__).resolve().parents[3] / (
    "glm_tpu/greenfield/validation/gate_d_projection_contraction_hlo.py"
)
ARTIFACT = Path(__file__).resolve().parents[3] / (
    "docs/artifacts/gate-d-projection-contraction-pp16-diagnostic-hlo-adjudication.json"
)
SCRIPT = Path(__file__).resolve().parents[3] / (
    "scripts/greenfield/adjudicate_gate_d_projection_contraction_pp16_hlo.py"
)
pytestmark = pytest.mark.skipif(
    not (OPTIMIZED.is_file() and STABLE.is_file()),
    reason="diagnostic projection-contraction HLO is not present",
)


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def test_exact_diagnostic_run_passes_without_promoting_the_failed_tag() -> None:
    report = adjudicate_diagnostic_run()
    assert report["classification"] == (
        "DIAGNOSTIC_HLO_CAUSAL_STRUCTURE_ACCEPTED;"
        "PP16_OWNER_LOCALITY_ACCEPTED;TPU_NUMERICAL_UNPROVEN;"
        "NO_HLO_ACQUIRED_TERMINAL;GATE_D_OPEN"
    )
    assert report["optimized_hlo"]["sha256"] == EXPECTED_OPTIMIZED_HLO_SHA256
    assert report["stablehlo"]["sha256"] == EXPECTED_STABLEHLO_SHA256
    assert report["gate_d_closed"] is False
    assert report["authorization"] == {
        "full_8k": False,
        "numerical_execution": False,
        "performance_claim": False,
        "persistence_only": True,
    }
    structure = report["optimized_hlo"]["structure"]
    assert structure["contraction_accumulation_dtype"] == "f32"
    assert structure["contraction_input_width"] == 6144
    assert structure["live_rows_per_owner"] == 1
    assert structure["owner_count"] == 2
    assert structure["collective_count"] == 0
    assert structure["host_effect_count"] == 0
    assert not (EXPECTED_RUN_DIR / "HLO_ACQUIRED").exists()


def test_committed_adjudication_is_the_exact_offline_report() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", str(SCRIPT)],
        cwd=Path("/"),
        env={
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("ascii")
    assert result.stderr == b""
    artifact = ARTIFACT.read_bytes()
    assert artifact == result.stdout
    assert len(artifact) == 2318
    assert sha256(artifact).hexdigest() == (
        "c36af80ce03f0c26ce72dd548c182f83ebec70ad44a54b982dde2ee287a78150"
    )
    assert json.loads(artifact)["process_contract"] == {
        "accelerator_fds_after": [],
        "accelerator_fds_before": [],
        "forbidden_modules_after": [],
        "forbidden_modules_before": [],
        "package_initializers_executed": [],
        "python_flags": ["-I", "-S", "-B"],
    }


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "dimensions={1}, to_apply=%region_0.0",
            "dimensions={0}, to_apply=%region_0.0",
        ),
        (
            "multiply(%dot_general.10, %bitcast.11)",
            "multiply(%bitcast.11, %bitcast.11)",
        ),
        (
            'metadata={op_name="normalized_hidden_owner"}',
            'metadata={op_name="normalized_hidden_dead_row"}',
        ),
        (
            "ROOT %tuple.11 =",
            "%tuple.11 =",
        ),
        (
            "fusion(%param.6, %bitcast.21), kind=kLoop, calls=%fused_computation.1",
            "fusion(%param.6, %param.6), kind=kLoop, calls=%fused_computation.1",
        ),
        (
            "fusion(%copy-done.2, %copy-done.3, %get-tuple-element.4, %sqrt.4, %div.23)",
            "fusion(%copy-done.2, %copy-done.3, %get-tuple-element.3, %sqrt.4, %div.23)",
        ),
        (
            "tuple(%copy.11, %bitcast.22, %maximum_bitcast_fusion)",
            "tuple(%copy.11, %maximum_bitcast_fusion, %maximum_bitcast_fusion)",
        ),
    ),
)
def test_optimized_hlo_causal_mutations_fail(old: str, new: str) -> None:
    source = OPTIMIZED.read_text(encoding="ascii")
    with pytest.raises(ProjectionContractionHloError):
        audit_optimized_hlo_structure(_replace_once(source, old, new))


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "add(%mul.44, %convert_element_type.15)",
            "add(%convert_element_type.15, %convert_element_type.15)",
        ),
        ("tuple(%slice.16, %slice.17)", "tuple(%slice.16, %slice.16)"),
        (
            "tuple(%bitcast.19, %bitcast.18.clone.1)",
            "tuple(%bitcast.19, %bitcast.19)",
        ),
        ("maximum(%pad.5, %pad.4)", "maximum(%pad.5, %pad.5)"),
        ("maximum(%pad.7, %pad.6)", "maximum(%pad.7, %pad.7)"),
    ),
)
def test_internal_fusion_parameter_to_root_bypass_attacks_fail(
    old: str, new: str
) -> None:
    source = OPTIMIZED.read_text(encoding="ascii")
    with pytest.raises(ProjectionContractionHloError):
        audit_optimized_hlo_structure(_replace_once(source, old, new))


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("calls=%fused_computation.6", "calls=%fused_computation.13"),
        (
            "get-tuple-element(%fusion.5), index=0",
            "get-tuple-element(%fusion.5), index=1",
        ),
        (
            "get-tuple-element(%subtract_bitcast_fusion), index=1",
            "get-tuple-element(%subtract_bitcast_fusion), index=0",
        ),
    ),
)
def test_current_key_fusion_call_and_tuple_index_attacks_fail(
    old: str, new: str
) -> None:
    source = OPTIMIZED.read_text(encoding="ascii")
    with pytest.raises(ProjectionContractionHloError):
        audit_optimized_hlo_structure(_replace_once(source, old, new))


def test_optimized_hlo_collective_and_host_effect_attacks_fail() -> None:
    source = OPTIMIZED.read_text(encoding="ascii")
    collective = source.replace(
        "  ROOT %tuple.11 =",
        "  %hostile = f32[1,128] all-reduce(%fusion.2), "
        "replica_groups={{0,1}}, to_apply=%region_0.0\n  ROOT %tuple.11 =",
        1,
    )
    with pytest.raises(ProjectionContractionHloError):
        audit_optimized_hlo_structure(collective)
    with pytest.raises(ProjectionContractionHloError):
        audit_optimized_hlo_structure(source + "\nhost_callback\n")


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("mhlo.num_partitions = 2", "mhlo.num_partitions = 32"),
        ('sdy.mesh @mesh = <["feature"=2]>', 'sdy.mesh @mesh = <["feature"=32]>'),
        ("contracting_dims = [1] x [1]", "contracting_dims = [0] x [0]"),
        (
            "tensor<1x6144xbf16>) -> tensor<1x6144xf32>",
            "tensor<1x6144xbf16>) -> tensor<1x6144xbf16>",
        ),
    ),
)
def test_stablehlo_contract_mutations_fail(old: str, new: str) -> None:
    source = STABLE.read_text(encoding="ascii")
    with pytest.raises(ProjectionContractionHloError):
        audit_stablehlo_structure(_replace_once(source, old, new))


def test_adjudicator_source_is_offline_only() -> None:
    source = MODULE_SOURCE.read_text(encoding="ascii")
    assert "import jax" not in source
    assert "jax.jit" not in source
    assert ".compile(" not in source
    assert "block_until_ready" not in source
    assert "google.cloud" not in source
    assert "subprocess" not in source


def test_remote_replay_is_exact_generation_qualified_and_fail_closed() -> None:
    raw = EXPECTED_REMOTE_REPLAY.read_bytes()
    assert sha256(raw).hexdigest() == EXPECTED_REMOTE_REPLAY_SHA256
    objects = _audit_remote_replay(raw)
    assert len(objects) == 15
    assert objects["runner.json"] == {
        "crc32c": "ToVXlw==",
        "generation": "1788295390400612",
        "path": "runner.json",
        "sha256": "a8a6dcbcf64ce0cc7bd945a51f92ef08c982d740e6b394317d2819b2d1bdb4e8",
        "size": 6702,
    }
    mutation = raw.replace(
        b'"soft_deleted_query_exhaustive": true',
        b'"soft_deleted_query_exhaustive": false',
        1,
    )
    with pytest.raises(ProjectionContractionHloError):
        _audit_remote_replay(mutation)


def test_isolated_cli_ignores_hostile_package_initializers(tmp_path: Path) -> None:
    hostile = tmp_path / "hostile"
    package = hostile / "glm_tpu"
    package.mkdir(parents=True)
    marker = tmp_path / "package-initializer-ran"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="ascii",
    )
    startup = tmp_path / "startup.py"
    startup.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('startup')\n",
        encoding="ascii",
    )
    result = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", str(SCRIPT)],
        cwd=hostile,
        env={
            **os.environ,
            "PYTHONPATH": str(hostile),
            "PYTHONSTARTUP": str(startup),
        },
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("ascii")
    assert not marker.exists()
    assert (
        json.loads(result.stdout)["process_contract"]["package_initializers_executed"]
        == []
    )
