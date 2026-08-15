from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking import output_geometry as geometry
from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield.validation import output_geometry as terminal


REPO = Path(__file__).resolve().parents[3]
REAL_SOURCE = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
REAL_FLEET = Path(
    "/home/gianl/glm-run/"
    "greenfield_strategy_nd_integrated_dense_native_m1_pallas_output_"
    "20260815T010951234829154Z"
)
REAL_OUTPUT_GEOMETRY_RUN = Path(
    "/home/gianl/glm-run/"
    "greenfield_strategy_nd_output_pallas_geometry_"
    "20260815T023028410400741Z"
)
REAL_OUTPUT_GEOMETRY_STABLEHLO = (
    REAL_OUTPUT_GEOMETRY_RUN
    / "hlo/strategy_nd_output_geometry_bfloat16_m32_m1_pallas_m8.stablehlo.mlir"
)
REAL_OUTPUT_GEOMETRY_OPTIMIZED_HLO = (
    REAL_OUTPUT_GEOMETRY_RUN
    / "hlo/strategy_nd_output_geometry_bfloat16_m32_m1_pallas_m8.optimized_hlo.txt"
)


def test_output_geometry_pairwise_classification_is_decisive() -> None:
    accepted = np.arange(6144, dtype=np.uint16)
    m32 = np.full((32, 6144), np.uint16(0x7FC0), dtype=np.uint16)
    m32[0] = accepted
    auto = accepted.reshape(1, 6144).copy()
    pallas = auto.copy()
    result = geometry.compare_output_geometry_arrays(
        {
            "m32_control": m32,
            "m1_auto": auto,
            "m1_pallas_m8": pallas,
        },
        accepted,
    )
    assert result["classification"] == (
        "output_geometry_m1_pallas_m8_exact_control"
    )
    assert result["m32_control_exact_accepted"] is True
    assert result["m1_pallas_m8_exact_m32_control"] is True

    pallas[0, 17] ^= np.uint16(1)
    result = geometry.compare_output_geometry_arrays(
        {
            "m32_control": m32,
            "m1_auto": auto,
            "m1_pallas_m8": pallas,
        },
        accepted,
    )
    decisive = result["pairwise"]["m1_pallas_m8_vs_m32_control"]
    assert result["classification"] == (
        "output_geometry_m1_pallas_m8_nonexact_new_result"
    )
    assert decisive["mismatch_count"] == 1
    assert decisive["first_mismatch_index"] == 17


@pytest.mark.skipif(not REAL_SOURCE.is_file(), reason="sealed DB548 source absent")
def test_output_geometry_source_reconstructs_exact_pinned_rows() -> None:
    inputs = geometry.load_output_geometry_inputs(
        REAL_SOURCE,
        geometry._PHYSICAL_DEVICE_BY_MODEL_POSITION,
    )
    assert sha256(inputs.dense_bits.tobytes()).hexdigest() == (
        geometry.OUTPUT_GEOMETRY_DENSE_ROW_RAW_SHA256
    )
    assert sha256(inputs.inverse.tobytes()).hexdigest() == (
        geometry.OUTPUT_GEOMETRY_INVERSE_RAW_SHA256
    )
    with pytest.raises(BenchmarkValidationError, match="mapping drifted"):
        geometry.load_output_geometry_inputs(REAL_SOURCE, tuple(range(32)))


@pytest.mark.skipif(
    not REAL_OUTPUT_GEOMETRY_STABLEHLO.is_file()
    or not REAL_OUTPUT_GEOMETRY_OPTIMIZED_HLO.is_file(),
    reason="protected output-geometry HLO absent",
)
def test_output_geometry_hlo_pins_and_live_mutations_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jaxlib import xla_client

    stablehlo = REAL_OUTPUT_GEOMETRY_STABLEHLO.read_text()
    optimized_hlo = REAL_OUTPUT_GEOMETRY_OPTIMIZED_HLO.read_text()
    assert sha256(stablehlo.encode()).hexdigest() == (
        geometry.OUTPUT_GEOMETRY_STABLEHLO_SHA256
    )
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        geometry.OUTPUT_GEOMETRY_OPTIMIZED_HLO_SHA256
    )
    assert geometry.validate_output_geometry_stablehlo(stablehlo)[
        "pallas_true_m1_io"
    ] is True
    assert geometry.validate_output_geometry_hlo(optimized_hlo)[
        "exact_shared_inverse_fanout"
    ] is True

    stable_mutations = (
        stablehlo.replace(
            "stablehlo.custom_call @tpu_custom_call(%38, %39, %40, %41)",
            "stablehlo.custom_call @tpu_custom_call(%39, %38, %40, %41)",
            1,
        ),
        stablehlo.replace(
            "sdy.return %24, %37, %43 :",
            "sdy.return %24, %37, %37 :",
            1,
        ),
        stablehlo.replace(
            "%43 = stablehlo.bitcast_convert %42 :",
            "%43 = stablehlo.bitcast_convert %36 :",
            1,
        ),
    )
    for mutated in stable_mutations:
        assert mutated != stablehlo
        monkeypatch.setattr(
            geometry,
            "OUTPUT_GEOMETRY_STABLEHLO_SHA256",
            sha256(mutated.encode()).hexdigest(),
        )
        with pytest.raises(BenchmarkValidationError, match="structure drifted"):
            geometry.validate_output_geometry_stablehlo(mutated)

    call = (
        "custom-call(%copy-done.4, %copy-done.3, %bitcast.3, "
        "%broadcast_in_dim.2)"
    )
    inverse_anchor = (
        "  %greenfield_weighted_output_m1_m8_scratch_h6144.1 = "
    )
    optimized_mutations = (
        optimized_hlo.replace(
            call,
            "custom-call(%copy-done.3, %copy-done.4, %bitcast.3, "
            "%broadcast_in_dim.2)",
            1,
        ),
        optimized_hlo.replace(
            call,
            "custom-call(%copy-done.4, %copy-done.3, %bitcast.3, "
            "%copy-done.4)",
            1,
        ),
        optimized_hlo.replace(
            inverse_anchor,
            (
                "  %rogue_inverse = f32[1]{0:T(128)S(3)} "
                "broadcast(%bitcast.2), dimensions={}\n"
                + inverse_anchor
            ),
            1,
        ).replace(
            call,
            "custom-call(%copy-done.4, %copy-done.3, %rogue_inverse, "
            "%broadcast_in_dim.2)",
            1,
        ),
        optimized_hlo.replace(
            (
                "tuple(%fusion.1, %multiply_bitcast-convert_fusion, "
                "%bitcast_convert_type.11)"
            ),
            (
                "tuple(%fusion.1, %multiply_bitcast-convert_fusion, "
                "%multiply_bitcast-convert_fusion)"
            ),
            1,
        ),
        optimized_hlo.replace(
            "copy-start(%param.8)",
            "copy-start(%param.7)",
            1,
        ),
        optimized_hlo.replace(
            (
                "%copy-start.4 = "
                "(bf16[1,6144]{1,0:T(2,128)(2,1)S(3)}, "
                "bf16[1,6144]{1,0:T(2,128)(2,1)}, u32[]{:S(2)})"
            ),
            (
                "%copy-start.4 = "
                "(bf16[1,6144]{1,0:T(4,128)(2,1)S(3)}, "
                "bf16[1,6144]{1,0:T(2,128)(2,1)}, u32[]{:S(2)})"
            ),
            1,
        ).replace(
            (
                "%copy-done.4 = "
                "bf16[1,6144]{1,0:T(2,128)(2,1)S(3)}"
            ),
            (
                "%copy-done.4 = "
                "bf16[1,6144]{1,0:T(4,128)(2,1)S(3)}"
            ),
            1,
        ),
    )
    for mutated in optimized_mutations:
        assert mutated != optimized_hlo
        xla_client._xla.hlo_module_from_text(mutated)
        monkeypatch.setattr(
            geometry,
            "OUTPUT_GEOMETRY_OPTIMIZED_HLO_SHA256",
            sha256(mutated.encode()).hexdigest(),
        )
        with pytest.raises(BenchmarkValidationError, match="structure drifted"):
            geometry.validate_output_geometry_hlo(mutated)


def test_output_geometry_abstract_graph_preserves_true_m1_boundary() -> None:
    import jax
    import jax.numpy as jnp

    shapes = (
        jax.ShapeDtypeStruct((32, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((32, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((6144,), jnp.bfloat16),
    )
    result = jax.eval_shape(
        geometry._output_geometry_program(pallas_interpret=False),
        *shapes,
    )
    assert tuple(value.shape for value in result) == (
        (32, 6144),
        (1, 6144),
        (1, 6144),
    )
    assert all(value.dtype == jnp.uint16 for value in result)
    with pytest.raises(BenchmarkValidationError, match="interpret flag"):
        geometry._output_geometry_program(pallas_interpret=1)  # type: ignore[arg-type]


def test_output_geometry_explicitly_shard_maps_pallas_before_jit() -> None:
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "from glm_tpu.greenfield.benchmarking.output_geometry import build_output_geometry_replay",
                    "compiled = build_output_geometry_replay(tuple(range(32)), validate_hlo=False, pallas_interpret=True)",
                    "assert len(compiled.member_device_ids) == 32",
                    "assert 'sdy.manual_computation' in compiled.stablehlo",
                )
            ),
        ],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_output_geometry_replicated_put_preserves_nan_sentinels() -> None:
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import jax, ml_dtypes, numpy as np",
                    "from jax.sharding import Mesh, NamedSharding, PartitionSpec as P",
                    "from glm_tpu.greenfield.benchmarking.output_geometry import _replicated_device_put",
                    "mesh = Mesh(np.asarray(jax.devices(), dtype=object), ('member',))",
                    "sharding = NamedSharding(mesh, P())",
                    "host = np.full((32, 16), ml_dtypes.bfloat16(np.nan), dtype=ml_dtypes.bfloat16)",
                    "host[0] = np.arange(16, dtype=np.float32).astype(ml_dtypes.bfloat16)",
                    "value = _replicated_device_put(host, sharding)",
                    "assert len(value.addressable_shards) == 4",
                    "expected = host.view(np.uint16)",
                    "assert all(np.array_equal(np.asarray(s.data).view(np.uint16), expected) for s in value.addressable_shards)",
                )
            ),
        ],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_output_geometry_wrapper_is_default_off_and_disjoint() -> None:
    wrapper = (REPO / "scripts/greenfield/run_strategy_nd_dense_replay.sh").read_text()
    runner = (REPO / "scripts/greenfield/microbench_collectives.py").read_text()
    assert "GLM_GREENFIELD_STRATEGY_ND_OUTPUT_GEOMETRY_REPLAY:-0" in wrapper
    assert "GLM_GREENFIELD_STRATEGY_ND_OUTPUT_PALLAS_GEOMETRY_TAG" in wrapper
    assert "--mode strategy_nd_output_geometry" in wrapper
    assert "validate_output_geometry_replay" in wrapper
    assert "RMS_REPLAY + INTEGRATED_REPLAY + OUTPUT_GEOMETRY_REPLAY" in wrapper
    assert '"strategy_nd_output_geometry"' in runner
    assert "weighted_output_m1_m8_scratch" in (
        REPO / "glm_tpu/greenfield/benchmarking/output_geometry.py"
    ).read_text()
    kernel = (
        REPO / "glm_tpu/greenfield/kernels/pallas/rmsnorm.py"
    ).read_text()
    assert "pltpu.VMEM((8, 128), jnp.bfloat16)" in kernel


@pytest.mark.skipif(
    not REAL_FLEET.is_dir()
    or not REAL_SOURCE.is_file()
    or not REAL_OUTPUT_GEOMETRY_STABLEHLO.is_file()
    or not REAL_OUTPUT_GEOMETRY_OPTIMIZED_HLO.is_file(),
    reason="sealed fleet/source fixtures absent",
)
def test_output_geometry_terminal_recomputes_every_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    for name in ("host_records", "hlo", "output_geometry", "source_rms"):
        (run_dir / name).mkdir(parents=True)
    for source in (REAL_FLEET / "source_rms").iterdir():
        if source.is_file():
            shutil.copy2(source, run_dir / "source_rms" / source.name)

    stablehlo = REAL_OUTPUT_GEOMETRY_STABLEHLO.read_text()
    optimized_hlo = REAL_OUTPUT_GEOMETRY_OPTIMIZED_HLO.read_text()
    stable_digest = sha256(stablehlo.encode()).hexdigest()
    optimized_digest = sha256(optimized_hlo.encode()).hexdigest()
    stable_contract = geometry.validate_output_geometry_stablehlo(stablehlo)
    optimized_contract = geometry.validate_output_geometry_hlo(optimized_hlo)
    label = "strategy_nd_output_geometry_bfloat16_m32_m1_pallas_m8"
    (run_dir / "hlo" / f"{label}.stablehlo.mlir").write_text(stablehlo)
    (run_dir / "hlo" / f"{label}.optimized_hlo.txt").write_text(optimized_hlo)
    (run_dir / "hlo" / f"{label}.hlo_prevalidation.json").write_text(
        json.dumps(
            {
                "optimized_hlo_sha256": optimized_digest,
                "performance_claim": False,
                "stablehlo_sha256": stable_digest,
                "validated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (run_dir / "hlo" / f"{label}.hlo_contract.json").write_text(
        json.dumps(
            {
                "optimized": optimized_contract,
                "stablehlo": stable_contract,
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    inputs = geometry.load_output_geometry_inputs(
        REAL_SOURCE,
        geometry._PHYSICAL_DEVICE_BY_MODEL_POSITION,
    )
    m32 = np.full((32, 6144), np.uint16(0x7FC0), dtype=np.uint16)
    m32[0] = inputs.accepted_bits
    outputs = {
        "m32_control": m32,
        "m1_auto": inputs.accepted_bits.reshape(1, 6144).copy(),
        "m1_pallas_m8": inputs.accepted_bits.reshape(1, 6144).copy(),
    }
    artifact_values = {
        "accepted_bits": inputs.accepted_bits,
        "m1_auto_bits": outputs["m1_auto"],
        "m1_pallas_m8_bits": outputs["m1_pallas_m8"],
        "m32_control_bits": outputs["m32_control"],
    }
    manifest = {}
    for name, value in artifact_values.items():
        path = run_dir / "output_geometry" / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        manifest[name] = {
            "array_sha256": geometry.array_sha256(value),
            "dtype": value.dtype.str,
            "file": path.name,
            "file_sha256": sha256(path.read_bytes()).hexdigest(),
            "shape": list(value.shape),
        }
    comparison = geometry.compare_output_geometry_arrays(
        outputs, inputs.accepted_bits
    )
    for name, value in (
        ("manifest.json", manifest),
        ("comparison.json", comparison),
    ):
        (run_dir / "output_geometry" / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )

    output_hashes = {
        name: geometry.array_sha256(value) for name, value in outputs.items()
    }
    capture = {
        "invocation_count": 2,
        "local_replica_sha256": {
            name: [digest] * 4 for name, digest in output_hashes.items()
        },
        "output_sha256": output_hashes,
        "repeated_local_replica_sha256": {
            name: [digest] * 4 for name, digest in output_hashes.items()
        },
        "repeated_output_sha256": output_hashes,
    }
    stable_hashes = {
        "accepted": geometry.array_sha256(inputs.accepted_bits),
        "carried": geometry.array_sha256(inputs.carried_bits),
        "dense": geometry.array_sha256(inputs.dense_bits),
        "inverse": geometry.array_sha256(inputs.inverse),
        "m1_auto": output_hashes["m1_auto"],
        "m1_pallas_m8": output_hashes["m1_pallas_m8"],
        "m32_control": output_hashes["m32_control"],
        "source_file": geometry.DENSE_RMS_SOURCE_NPZ_SHA256,
        "weight": geometry.array_sha256(inputs.weight_bits),
    }
    item = {
        "accepted_model_axis_device_ids": list(
            terminal.EXPECTED_MODEL_AXIS_DEVICE_IDS
        ),
        "accepted_model_axis_recipe": terminal.ACCEPTED_TP32_MODEL_AXIS_RECIPE,
        "artifact_manifest": {},
        "capture": capture,
        "comparison": comparison,
        "diagnostic_only": True,
        "fleet_hashes": {
            name: [digest] * 8 for name, digest in stable_hashes.items()
        },
        "fleet_hlo_hashes": [optimized_digest] * 8,
        "fleet_stablehlo_hashes": [stable_digest] * 8,
        "member_device_ids": list(range(32)),
        "optimized_hlo_contract": optimized_contract,
        "optimized_hlo_sha256": optimized_digest,
        "output_layouts": {
            "m1_auto": geometry.M1_AUTO_BF16_LAYOUT,
            "m1_pallas_m8": geometry.M1_AUTO_BF16_LAYOUT,
            "m32_control": geometry.M32_BF16_LAYOUT,
        },
        "pallas_internal_scratch_shape": list(
            geometry.M1_PALLAS_M8_SCRATCH_SHAPE
        ),
        "performance_claim": False,
        "source": terminal.EXPECTED_SOURCE,
        "stablehlo_contract": stable_contract,
        "stablehlo_sha256": stable_digest,
    }
    code_hash = "1" * 40
    run_tag = "unit_output_geometry"
    source_records = sorted((REAL_FLEET / "host_records").glob("*.json"))
    for launch_index, source in enumerate(source_records):
        record = json.loads(source.read_text())
        record.pop("association_integrated_dense_rms")
        nested = dict(item)
        if record["jax_process_index"] == 0:
            nested["artifact_manifest"] = manifest
        record.update(
            {
                "association_output_geometry": nested,
                "code_hash": code_hash,
                "mode": "strategy_nd_output_geometry",
                "run_tag": run_tag,
                "schema_version": 6,
            }
        )
        (run_dir / "host_records" / f"collective.rank{launch_index}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )

    summary = terminal.validate_output_geometry_replay(
        run_dir,
        expected_code_hash=code_hash,
        expected_run_tag=run_tag,
    )
    assert summary["classification"] == (
        "output_geometry_m1_pallas_m8_exact_control"
    )
    assert summary["mismatch_count"] == 0

    record_paths = sorted((run_dir / "host_records").glob("*.json"))
    for path in record_paths:
        record = json.loads(path.read_text())
        record["association_output_geometry"][
            "pallas_internal_scratch_shape"
        ] = [4, 128]
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="execution identity drifted"):
        terminal.validate_output_geometry_replay(
            run_dir,
            expected_code_hash=code_hash,
            expected_run_tag=run_tag,
        )
    for path in record_paths:
        record = json.loads(path.read_text())
        record["association_output_geometry"][
            "pallas_internal_scratch_shape"
        ] = list(geometry.M1_PALLAS_M8_SCRATCH_SHAPE)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    arrays = np.load(
        run_dir / "output_geometry" / "m1_pallas_m8_bits.npy",
        allow_pickle=False,
    )
    arrays[0, 9] ^= np.uint16(1)
    np.save(
        run_dir / "output_geometry" / "m1_pallas_m8_bits.npy",
        arrays,
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="artifact drifted"):
        terminal.validate_output_geometry_replay(
            run_dir,
            expected_code_hash=code_hash,
            expected_run_tag=run_tag,
        )
