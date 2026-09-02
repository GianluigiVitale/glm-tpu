from __future__ import annotations

import ast
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.py"


def _module():
    spec = importlib.util.spec_from_file_location("projection_numerical_runner", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _npz(values: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in values.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            archive.writestr(f"{name}.npy", payload.getvalue())
    return output.getvalue()


def test_source_is_default_off_bounded_and_projection_only() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "jax" not in imported_roots
    assert "numpy" not in imported_roots
    assert "GLM_GATE_D_PROJECTION_CONTRACTION_NUMERICAL" in source
    assert "choices=(1,)" in source
    assert "build_gate_d_projection_contraction_pp16" in source
    assert "build_gate_d_forced_round_pp16_hlo_replay" not in source
    assert "run_short_decode" not in source
    assert source.count("result = compiled(*arguments)") == 1
    assert source.count("jax.device_get(result)") == 1
    assert "stablehlo != accepted_stablehlo" in source
    assert "optimized_hlo != derived_optimized_hlo" in source
    assert "optimized_hlo != accepted_optimized_hlo" not in source
    assert '"gate_d_closed": False' in source
    assert '"performance_claim": False' in source
    assert '"root_cause_fix_proven": False' in source


def test_bound_json_rejects_wrong_hash_and_non_object(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "value.json"
    path.write_text("[]\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="bytes drifted"):
        module._load_bound_json(path, "0" * 64, "test")
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TypeError, match="root"):
        module._load_bound_json(path, digest, "test")


def test_bound_json_rejects_symlink(tmp_path: Path) -> None:
    module = _module()
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="ascii")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        module._load_bound_json(link, "0" * 64, "test")


@pytest.mark.filterwarnings("ignore:Duplicate name")
def test_npz_loader_rejects_duplicate_and_extra_members() -> None:
    module = _module()
    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("value.npy", b"first")
        archive.writestr("value.npy", b"second")
    with pytest.raises(RuntimeError, match="catalogue"):
        module._load_npz(duplicate.getvalue(), {"value"}, np)
    with pytest.raises(RuntimeError, match="catalogue"):
        module._load_npz(
            _npz({"value": np.zeros(1), "extra": np.zeros(1)}), {"value"}, np
        )


def test_input_validator_rejects_hash_shape_and_dtype() -> None:
    module = _module()
    value = np.zeros((2,), dtype=np.float32)
    record = {
        "array_sha256": module._array_sha256(value, np),
        "shape": [2],
        "storage_dtype": "<f4",
    }
    module._validate_input_arrays({"value": value}, {"value": record}, np)
    for key, replacement in (
        ("array_sha256", "0" * 64),
        ("shape", [1, 2]),
        ("storage_dtype", "<f8"),
    ):
        hostile = json.loads(json.dumps(record))
        hostile[key] = replacement
        with pytest.raises(RuntimeError, match="array drifted"):
            module._validate_input_arrays({"value": value}, {"value": hostile}, np)


def test_materialized_wk_uses_bf16_round_trip() -> None:
    module = _module()

    class FakeMlDtypes:
        bfloat16 = np.float16

    inputs = {
        "wk_weight_bits": np.zeros((2, 128, 6144), dtype=np.uint8),
        "wk_scale_inv": np.ones((2, 1, 48), dtype=np.float32),
    }
    observed = module._materialize_wk(inputs, np, FakeMlDtypes)
    assert observed.shape == (2, 128, 6144)
    assert observed.dtype == np.float32
    assert np.count_nonzero(observed) == 0


def test_deterministic_npz_is_byte_stable() -> None:
    module = _module()
    values = {
        "b": np.arange(4, dtype=np.int32),
        "a": np.arange(2, dtype=np.float32),
    }
    first = module._deterministic_npz(values, np)
    second = module._deterministic_npz(dict(reversed(list(values.items()))), np)
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["a.npy", "b.npy"]
        assert all(
            item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist()
        )


BRIDGE = ROOT / (
    "docs/artifacts/gate-d-projection-contraction-pp16-hlo-source-location-bridge.json"
)
ACCEPTED_OPTIMIZED_HLO = Path(
    "/home/gianl/gate-d-runs/gate_d_projection_contraction_pp16_hlo_"
    "20260901T213605719107105Z/hlo/projection_contraction_pp16_stage0.optimized_hlo.txt"
)
V1_FAILED_OPTIMIZED_HLO = Path(
    "/home/gianl/gate-d-runs/gate_d_projection_contraction_pp16_numerical_"
    "20260901T233855937688834Z/hlo/projection_contraction_pp16_stage0.optimized_hlo.txt"
)


def test_bridge_call_site_lines_match_driver_source_and_installed_path() -> None:
    module = _module()
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    assert lines[module.NUMERICAL_DRIVER_MODULE_CALL_LINE - 1] == (
        "    raise SystemExit(main())"
    )
    assert lines[module.NUMERICAL_DRIVER_LOWER_CALL_LINE - 1].strip() == (
        "lowered = replay.lower(*abstract_arguments)"
    )
    assert module.NUMERICAL_DRIVER_INSTALL_PATH == (
        "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-numerical-v2/"
        "run_gate_d_projection_contraction_pp16_numerical.py"
    )
    source = SOURCE.read_text(encoding="utf-8")
    assert "Path(__file__) != Path(NUMERICAL_DRIVER_INSTALL_PATH)" in source
    assert source.index("validate_hlo_source_location_bridge(\n        json.loads") < (
        source.index("lowered = replay.lower(*abstract_arguments)")
    )
    assert '"hlo/source_location_bridge.json"' in source


def test_bridge_artifact_bytes_and_schema_are_pinned() -> None:
    module = _module()
    from hashlib import sha256

    raw = BRIDGE.read_bytes()
    assert sha256(raw).hexdigest() == module.HLO_SOURCE_LOCATION_BRIDGE_SHA256
    report = json.loads(raw)
    assert report["derivation"]["replacements"] == (
        module.expected_hlo_source_location_replacements()
    )
    assert report["derived_numerical_hlo"] == {
        "byte_count": module.EXPECTED_NUMERICAL_OPTIMIZED_HLO_BYTES,
        "sha256": module.EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256,
    }
    assert report["gate_d_closed"] is False
    assert report["numerical_claim"] is False
    assert report["tpu_numerical_execution_performed"] is False


def test_bridge_derivation_applies_only_single_occurrence_substitutions() -> None:
    module = _module()
    replacements = [
        {"new": "B", "occurrence_count": 1, "old": "A", "surface": "x"},
        {
            "new": "line=2 end_line=2",
            "occurrence_count": 1,
            "old": "line=1 end_line=1",
            "surface": "y",
        },
    ]
    preimage = b"head A tail line=1 end_line=1 rest"
    assert (
        module.derive_hlo_from_source_location_replacements(preimage, replacements)
        == b"head B tail line=2 end_line=2 rest"
    )
    for hostile in (
        b"A A line=1 end_line=1",
        b"line=1 end_line=1",
        b"A B line=1 end_line=1",
        b"A line=1 end_line=1 line=2 end_line=2",
    ):
        with pytest.raises(RuntimeError, match="occurrence drifted"):
            module.derive_hlo_from_source_location_replacements(hostile, replacements)
    with pytest.raises(RuntimeError, match="occurrence drifted"):
        module.derive_hlo_from_source_location_replacements(
            preimage, [{**replacements[0], "occurrence_count": 2}, replacements[1]]
        )


def test_bridge_validation_reproduces_failed_v1_hlo_and_pins_v2_hlo() -> None:
    if not ACCEPTED_OPTIMIZED_HLO.exists() or not V1_FAILED_OPTIMIZED_HLO.exists():
        pytest.skip("protected HLO run directories are not present on this host")
    module = _module()
    from hashlib import sha256

    accepted = ACCEPTED_OPTIMIZED_HLO.read_bytes()
    assert sha256(accepted).hexdigest() == module.EXPECTED_OPTIMIZED_HLO_SHA256
    derived, binding = module.validate_hlo_source_location_bridge(
        json.loads(BRIDGE.read_bytes()), accepted
    )
    assert sha256(derived).hexdigest() == module.EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256
    assert binding["artifact_sha256"] == module.HLO_SOURCE_LOCATION_BRIDGE_SHA256
    assert derived != accepted
    # Everything but the three metadata surfaces is byte-identical.
    assert derived.count(module.NUMERICAL_DRIVER_INSTALL_PATH.encode()) == 1
    assert accepted.count(module.HLO_ACQUIRER_INSTALL_PATH.encode()) == 1
    # The v1 fail-closed run's HLO is the same derivation with the v1 path/lines.
    v1 = [dict(item) for item in module.expected_hlo_source_location_replacements()]
    v1[0]["new"] = (
        "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-numerical-v1/"
        "run_gate_d_projection_contraction_pp16_numerical.py"
    )
    v1[1]["new"] = "line=657 end_line=657"
    v1[2]["new"] = "line=497 end_line=497"
    assert (
        module.derive_hlo_from_source_location_replacements(accepted, v1)
        == V1_FAILED_OPTIMIZED_HLO.read_bytes()
    )
    hostile = json.loads(BRIDGE.read_bytes())
    hostile["derivation"]["replacements"][1]["new"] = "line=1 end_line=1"
    with pytest.raises(RuntimeError, match="bridge schema drifted"):
        module.validate_hlo_source_location_bridge(hostile, accepted)
    with pytest.raises(RuntimeError, match="preimage drifted"):
        module.validate_hlo_source_location_bridge(
            json.loads(BRIDGE.read_bytes()), accepted + b"\n"
        )
