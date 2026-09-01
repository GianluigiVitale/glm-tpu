from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import ml_dtypes
import numpy as np
import pytest

ROOT = Path(__file__).parents[3]


def _load(relative: str, name: str) -> ModuleType:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load(
    "scripts/greenfield/analyze_gate_d_projection_arithmetic_frontier.py",
    "_gate_d_projection_arithmetic_frontier",
)
MODULE_IMPORTED_BACKEND_NEUTRAL = (
    MODULE.np is None
    and MODULE.ml_dtypes is None
    and MODULE.jax is None
    and MODULE.jnp is None
)
MODULE.np = np
MODULE.ml_dtypes = ml_dtypes


def _npz(values: dict[str, np.ndarray]) -> bytes:
    output = BytesIO()
    np.savez(output, **values)
    return output.getvalue()


def test_module_import_is_backend_neutral() -> None:
    assert MODULE_IMPORTED_BACKEND_NEUTRAL


def test_variant_catalogue_is_exact_and_layout_grounded() -> None:
    names = MODULE._variant_names()
    assert len(names) == 27
    assert len(set(names)) == 27
    assert names[:3] == (
        "linear_k:left",
        "linear_k:reverse",
        "linear_k:balanced",
    )
    assert {
        "lane_first:left:balanced",
        "tile_first:balanced:reverse",
        "lane_hierarchy:8x16:balanced:balanced:left",
        "lane_hierarchy:16x8:balanced:balanced:balanced",
    }.issubset(names)
    classes = [MODULE._variant_class(name) for name in names]
    assert classes.count("generic_linear_control") == 3
    assert classes.count("direct_48x128_tile_lane_association") == 18
    assert classes.count("speculative_within_lane_hierarchy") == 6


def test_reduction_orders_apply_binary32_at_every_add() -> None:
    value = np.asarray([[1e20, -1e20, 3.0]], dtype=np.float32)
    assert MODULE._reduce_last(value, "left").tolist() == [3.0]
    assert MODULE._reduce_last(value, "reverse").tolist() == [0.0]
    assert MODULE._reduce_last(value, "balanced").tolist() == [3.0]
    with pytest.raises(ValueError, match="unknown reduction order"):
        MODULE._reduce_last(value, "illegal")


def test_projection_variants_have_exact_geometry_and_catalogue() -> None:
    normalized = np.linspace(-1, 1, MODULE.CONTRACTION, dtype=np.float32)
    weight = np.linspace(
        -0.5,
        0.5,
        128 * MODULE.CONTRACTION,
        dtype=np.float32,
    ).reshape(128, MODULE.CONTRACTION)
    variants = MODULE._explicit_projection_variants(normalized, weight)
    assert tuple(variants) == MODULE._variant_names()
    assert all(value.shape == (128,) for value in variants.values())
    assert all(value.dtype == np.float32 for value in variants.values())
    products = np.multiply(weight, normalized[None, :], dtype=np.float32)
    assert np.array_equal(
        variants["linear_k:left"], MODULE._reduce_last(products, "left")
    )


def test_hlo_validator_requires_the_exact_multiply_reduce_layout() -> None:
    raw = MODULE._snapshot_regular(MODULE.OPTIMIZED_HLO)
    contract = MODULE._verify_hlo(raw)
    assert contract == {
        "contraction": 6144,
        "contraction_tiles": 48,
        "hlo_f32_multiply_reduce": True,
        "layout": "T(8,128)",
        "projection_slice_sha256": (
            "a625c1a715c3c390c650e77101519efa729f9123a60b1c7f0331f09324200207"
        ),
        "tile_lanes": 128,
        "tile_rows": 8,
    }
    with pytest.raises(MODULE.ProjectionFrontierError, match="signature"):
        MODULE._verify_hlo(
            raw.replace(
                b"f32[128,6144]{1,0:T(8,128)} multiply",
                b"f32[128,6144]{1,0:T(4,128)} multiply",
                1,
            )
        )


def test_npz_loader_rejects_duplicate_members() -> None:
    schema = {"value": ((2,), "<f4")}
    raw = _npz({"value": np.asarray([1.0, 2.0], dtype=np.float32)})
    source = zipfile.ZipFile(BytesIO(raw), "r")
    payload = source.read("value.npy")
    source.close()
    duplicate = BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("value.npy", payload)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("value.npy", payload)
    with pytest.raises(MODULE.ProjectionFrontierError, match="catalogue"):
        MODULE._load_npz(duplicate.getvalue(), schema)


@pytest.mark.parametrize("mutation", ("name", "shape", "dtype"))
def test_npz_loader_rejects_schema_drift(mutation: str) -> None:
    schema = {"value": ((2,), "<f4")}
    if mutation == "name":
        raw = _npz({"other": np.asarray([1.0, 2.0], dtype=np.float32)})
    elif mutation == "shape":
        raw = _npz({"value": np.asarray([1.0], dtype=np.float32)})
    else:
        raw = _npz({"value": np.asarray([1.0, 2.0], dtype=np.float64)})
    with pytest.raises(MODULE.ProjectionFrontierError):
        MODULE._load_npz(raw, schema)


def test_regular_snapshot_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"authority")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        MODULE._snapshot_regular(link)


@pytest.mark.parametrize("name", ("jaxlib", "numpy"))
def test_preloaded_numerical_modules_are_rejected(name: str) -> None:
    with pytest.raises(MODULE.ProjectionFrontierError, match="preloaded"):
        MODULE._reject_preloaded_modules({name: object()})


def test_capsule_manifest_schema_and_top_level_catalogue_are_strict(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    manifest = {
        "allowlist": ["dependency"],
        "claim_scope": "test dependency bytes",
        "schema_version": 1,
        "source_entries": {
            "dependency": {"kind": "directory", "tree_sha256": "1" * 64}
        },
        "source_root": "/sealed/source",
    }
    raw = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (tmp_path / "CAPSULE_MANIFEST.json").write_bytes(raw)
    assert MODULE._parse_capsule_manifest(tmp_path, MODULE._sha256(raw)) == {
        "dependency"
    }
    (tmp_path / "unexpected").write_bytes(b"shadow")
    with pytest.raises(MODULE.ProjectionFrontierError, match="catalogue"):
        MODULE._parse_capsule_manifest(tmp_path, MODULE._sha256(raw))
    (tmp_path / "unexpected").unlink()
    manifest["source_entries"]["dependency"]["tree_sha256"] = "not-a-sha256"
    malformed = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (tmp_path / "CAPSULE_MANIFEST.json").write_bytes(malformed)
    with pytest.raises(MODULE.ProjectionFrontierError, match="source record"):
        MODULE._parse_capsule_manifest(tmp_path, MODULE._sha256(malformed))


def test_loaded_capsule_paths_cannot_escape_manifest_allowlist(
    tmp_path: Path,
) -> None:
    records = [
        {
            "bytes": 1,
            "path": str(tmp_path / "numpy" / "core.py"),
            "sha256": "1" * 64,
        }
    ]
    MODULE._validate_loaded_capsule_paths(records, {tmp_path: frozenset({"numpy"})})
    records[0]["path"] = str(tmp_path / "shadow.py")
    with pytest.raises(MODULE.ProjectionFrontierError, match="allowlist"):
        MODULE._validate_loaded_capsule_paths(records, {tmp_path: frozenset({"numpy"})})


def test_fixed_loaded_byte_manifest_rejects_drift() -> None:
    records = [{"bytes": 1, "path": "/sealed/a", "sha256": "1" * 64}]
    digest = MODULE._sha256(MODULE._canonical_json(records))
    assert (
        MODULE._validate_record_manifest(
            "test", records, expected_count=1, expected_sha256=digest
        )
        == digest
    )
    mutated = [{**records[0], "sha256": "2" * 64}]
    with pytest.raises(MODULE.ProjectionFrontierError, match="byte manifest"):
        MODULE._validate_record_manifest(
            "test", mutated, expected_count=1, expected_sha256=digest
        )


def test_cpu_replay_refuses_non_cpu_platform_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    zeros = np.zeros(128, dtype=np.float32)
    with pytest.raises(MODULE.ProjectionFrontierError, match="JAX_PLATFORMS"):
        MODULE._cpu_suffix_and_control(
            {"one": zeros},
            np.zeros(MODULE.CONTRACTION, dtype=np.float32),
            np.zeros((128, MODULE.CONTRACTION), dtype=np.float32),
            zeros,
            zeros,
        )


def test_cpu_replay_refuses_preopened_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setattr(MODULE, "_accelerator_fds", lambda: ("/dev/accel0",))
    monkeypatch.setattr(MODULE, "jax", object())
    monkeypatch.setattr(MODULE, "jnp", object())
    zeros = np.zeros(128, dtype=np.float32)
    with pytest.raises(MODULE.ProjectionFrontierError, match="already open"):
        MODULE._cpu_suffix_and_control(
            {"one": zeros},
            np.zeros(MODULE.CONTRACTION, dtype=np.float32),
            np.zeros((128, MODULE.CONTRACTION), dtype=np.float32),
            zeros,
            zeros,
        )


def test_metrics_are_bitwise_and_keep_nonrotary_scope() -> None:
    reference = np.zeros(128, dtype=np.float32)
    value = reference.copy()
    value[3] = np.float32(-0.0)
    value[90] = np.float32(1.0)
    metrics = MODULE._metrics(value, reference)
    assert metrics["bit_mismatch_count"] == 2
    assert metrics["first_bit_mismatch"] == 3
    assert metrics["nonrotary_bit_mismatch_count"] == 1
    assert metrics["max_abs_error"] == 1.0


def test_full_offline_analysis_is_cpu_only_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")

    def prepare() -> dict[str, object]:
        import jax
        import jax.numpy as jnp

        MODULE.jax = jax
        MODULE.jnp = jnp
        MODULE.np = np
        MODULE.ml_dtypes = ml_dtypes
        return {
            "native_mapping_before_count": 0,
            "native_mapping_before_sha256": "0" * 64,
            "test_runtime": True,
        }

    monkeypatch.setattr(MODULE, "_prepare_sealed_runtime", prepare)
    monkeypatch.setattr(MODULE, "_finalize_sealed_runtime", lambda before: {})
    monkeypatch.setattr(
        MODULE, "_verify_source_committed", lambda: ("test-pin", "0" * 64)
    )
    report = MODULE.analyze()
    assert report["artifact_kind"] == ("gate_d_projection_arithmetic_frontier_analysis")
    assert report["classification"].endswith("GATE_D_OPEN")
    assert report["classification"].startswith(
        "CPU_F32_DOT_CONTROL_EXACT_ACCEPTED_KEY_CAPTURED_INPUT;"
    )
    assert report["gate_d_closed"] is False
    assert report["root_cause_proven"] is False
    assert report["projection_mechanism_authorized"] is False
    assert report["full_dsa_or_8k_authorized"] is False
    assert report["performance_claim"] is False
    assert report["tpu_compile_or_execution_performed"] is False
    assert report["runtime"]["backend"] == "cpu"
    assert report["runtime"]["accelerator_fds_before"] == []
    assert report["runtime"]["accelerator_fds_after"] == []
    assert report["variant_count"] == 27
    assert report["cpu_jax_dot_control"]["accepted"]["bit_mismatch_count"] == 0
    assert report["cpu_jax_dot_control"]["accepted"]["sha256"] == (
        "5006ad4f7224047652bbc46e6275b4c82bec0dead13f5f2b5f91494a2415329b"
    )
    assert (
        report["best_explicit_reduction_probe"]["accepted"]["bit_mismatch_count"] == 100
    )
    assert len(report["variants_ranked_by_accepted_key"]) == 27
    assert json.loads(MODULE._canonical_json(report)) == report


def test_sealed_runtime_boundary_rejects_ambient_test_interpreter() -> None:
    with pytest.raises(MODULE.ProjectionFrontierError, match="sealed Python"):
        MODULE._prepare_sealed_runtime()


def test_exact_isolated_runtime_and_dependency_capsules_are_usable() -> None:
    code = f"""
import importlib.util, json
spec = importlib.util.spec_from_file_location('sealed_projection', {str(ROOT / MODULE.SOURCE_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._verify_source_committed = lambda: ('TEST_ONLY', '0' * 64)
report = module.analyze()
print(json.dumps(report['runtime'], sort_keys=True))
"""
    result = subprocess.run(
        [str(MODULE.PYTHON), "-I", "-S", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["python_sha256"] == MODULE.PYTHON_SHA256
    assert report["jax_site_installed_tree_authority_sha256"] == (
        MODULE.JAX_SITE_TREE_SHA256
    )
    assert report["libtpu_site_installed_tree_authority_sha256"] == (
        MODULE.LIBTPU_SITE_TREE_SHA256
    )
    assert report["native_mapping_before_count"] == (
        MODULE.EXPECTED_NATIVE_BEFORE_COUNT
    )
    assert report["native_mapping_before_sha256"] == (
        MODULE.EXPECTED_NATIVE_BEFORE_SHA256
    )
    assert report["loaded_sealed_module_count"] == (
        MODULE.EXPECTED_LOADED_SEALED_MODULE_COUNT
    )
    assert report["loaded_sealed_module_manifest_sha256"] == (
        MODULE.EXPECTED_LOADED_SEALED_MODULE_SHA256
    )
    assert report["native_mapping_after_count"] == MODULE.EXPECTED_NATIVE_AFTER_COUNT
    assert report["native_mapping_after_sha256"] == MODULE.EXPECTED_NATIVE_AFTER_SHA256


def test_main_rejects_every_argument() -> None:
    with pytest.raises(MODULE.ProjectionFrontierError, match="no arguments"):
        MODULE.main(["--output", os.devnull])
