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
    assert "build_gate_d_projection_host_rope_pp16" in source
    assert "build_gate_d_projection_contraction_pp16(" not in source
    assert "materialize_dsa_rope_row(np)" in source
    assert "audit_host_rope_optimized_hlo(" in source
    assert "audit_host_rope_stablehlo(" in source
    assert "classify_host_rope_outputs(" in source
    assert "build_gate_d_forced_round_pp16_hlo_replay" not in source
    assert "run_short_decode" not in source
    assert source.count("result = compiled(*arguments)") == 1
    assert source.count("jax.device_get(result)") == 1
    assert "accepted_stablehlo" not in source
    assert "source_location_bridge" not in source
    assert "rotary_cos_sin" not in source
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


def test_v3_driver_binds_installed_path_and_host_rope_source_hashes() -> None:
    from hashlib import sha256

    module = _module()
    assert module.NUMERICAL_DRIVER_INSTALL_PATH == (
        "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-numerical-v3/"
        "run_gate_d_projection_contraction_pp16_numerical.py"
    )
    assert set(module.HOST_ROPE_SOURCE_SHA256S) == {
        "glm_tpu/greenfield/benchmarking/gate_d_projection_contraction_pp16_host_rope.py",
        "glm_tpu/greenfield/kernels/reference/dsa_host_rope.py",
        "glm_tpu/greenfield/kernels/reference/rotary_table.py",
        "glm_tpu/greenfield/validation/gate_d_projection_host_rope_numerical.py",
    }
    for relative, expected in module.HOST_ROPE_SOURCE_SHA256S.items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    source = SOURCE.read_text(encoding="utf-8")
    assert "Path(__file__) != Path(NUMERICAL_DRIVER_INSTALL_PATH)" in source
    assert source.index(
        "_verify_host_rope_sources(args.expected_code_hash)"
    ) < source.index("import jax")
    assert source.index("hlo_structure = {") < source.index(
        "result = compiled(*arguments)"
    )
    assert '"schema_version": 2' in source
    assert '"dsa_rope_row.f32le"' in source
    assert "gate_d_projection_host_rope_pp16_numerical_replay" in source
