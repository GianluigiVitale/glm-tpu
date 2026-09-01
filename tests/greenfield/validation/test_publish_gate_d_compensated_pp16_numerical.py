from __future__ import annotations

import importlib.util
import warnings
import zipfile
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "scripts/greenfield/publish_gate_d_compensated_pp16_numerical.py"
SPEC = importlib.util.spec_from_file_location("gate_d_pp16_numerical_publisher", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _output_arrays() -> dict[str, object]:
    return {
        name: {
            "array_sha256": sha256(name.encode()).hexdigest(),
            "semantic_dtype": semantic_dtype,
            "shape": shape,
            "storage_dtype": storage_dtype,
        }
        for name, (
            shape,
            semantic_dtype,
            storage_dtype,
        ) in MODULE.EXPECTED_OUTPUT_SPEC.items()
    }


def _runner(*, accepted: bool = True) -> dict[str, object]:
    status = "NUMERICAL_ACCEPTED" if accepted else "NUMERICAL_REJECTED"
    classification = (
        "BOUNDED_TPU_NUMERICAL_ACCEPTED;FULL_DECODER_UNPROVEN;GATE_D_OPEN;"
        "NO_PERFORMANCE_CLAIM"
        if accepted
        else "BOUNDED_TPU_NUMERICAL_REJECTED;COMPENSATED_MECHANISM_TOMBSTONED;"
        "GATE_D_OPEN;NO_PERFORMANCE_CLAIM"
    )
    expected_host = {
        "authority_sha256": MODULE.EXPECTED_HOST_AUTHORITY_SHA256,
        "backend": "cpu",
        "capsule_input_sha256": (
            "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
        ),
        "query_weight": {
            "array_sha256": MODULE.EXPECTED_QUERY_SHA256,
            "semantic_dtype": "float32",
            "shape": [2, 2048, 2048],
        },
        "wk_weight": {
            "array_sha256": MODULE.EXPECTED_WK_SHA256,
            "semantic_dtype": "float32",
            "shape": [2, 128, 6144],
        },
    }
    owners = {
        "current_key_owners": True,
        "head_weights_owners": True,
        "normalized_hidden_owners": True,
        "query_owners": True,
        "rms_input_fp32_owners": True,
        "selected_positions_owners": True,
        "selected_scores_owners": True,
        "valid_counts_owners": True,
    }
    if not accepted:
        owners["query_owners"] = False
    observed = {
        "positions_sha256": MODULE.EXPECTED_POSITIONS_SHA256,
        "scores_sha256": MODULE.EXPECTED_SCORES_SHA256,
        "valid_count": 2048,
    }
    if not accepted:
        observed["positions_sha256"] = "0" * 64
    return {
        "artifact_kind": "gate_d_compensated_pp16_numerical_replay",
        "capsule": {},
        "claim_scope": "bounded",
        "classification": classification,
        "code_hash": "f" * 40,
        "compiler_dependency_manifest": {},
        "compiled_executable_invocation_count": 1,
        "gate_d_closed": False,
        "hlo": {
            "optimized_hlo_sha256": MODULE.EXPECTED_OPTIMIZED_HLO_SHA256,
            "stablehlo_sha256": MODULE.EXPECTED_STABLEHLO_SHA256,
        },
        "host_materialization": {
            **expected_host,
            "observed_derived_weights": {
                "query_weight": expected_host["query_weight"],
                "wk_weight": expected_host["wk_weight"],
            },
        },
        "host_transfer_count": 1,
        "memory_after_compile": [{}, {}],
        "memory_after_execute": [{}, {}],
        "memory_before_compile": [{}, {}],
        "numerical": {
            "accepted_tpu_event_match": accepted,
            "contract_valid": [1, 1],
            "cpu_candidate_watchpoint_matches_diagnostic_only": {
                f"watchpoint-{index}": True for index in range(14)
            },
            "observed_event1": observed,
            "owner_agreement": owners,
            "tie_order_exact": True,
        },
        "numerical_claim_scope": "bounded_event1_only",
        "output_arrays": _output_arrays(),
        "output_artifact": {},
        "performance_claim": False,
        "persistent_compilation_cache_enabled": False,
        "physical_group": {
            "coordinates": [[0, 0, 0], [1, 0, 0]],
            "device_ids": [0, 1],
            "process_index": 0,
            "stage_id": 0,
        },
        "policy": {
            "positions_sha256": MODULE.EXPECTED_POSITIONS_SHA256,
            "scores_sha256": MODULE.EXPECTED_SCORES_SHA256,
            "valid_count": 2048,
        },
        "runtime": {
            "jax": "0.10.1",
            "jaxlib": "0.10.1",
            "libtpu": "0.0.41",
            "ml_dtypes": "0.5.4",
            "numpy": "2.3.5",
        },
        "schema_version": 1,
        "sealed_project_source": {},
        "status": status,
        "tpu_execution_elapsed_ns_diagnostic_only": 1,
        "tpu_numerical_execution_performed": True,
    }


def test_runner_accepts_exact_accepted_and_rejected_boundaries() -> None:
    assert MODULE._validate_runner(_runner(), "f" * 40) == "NUMERICAL_ACCEPTED"
    assert (
        MODULE._validate_runner(_runner(accepted=False), "f" * 40)
        == "NUMERICAL_REJECTED"
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("compiled_executable_invocation_count",), True),
        (("host_transfer_count",), True),
        (("schema_version",), True),
        (("performance_claim",), True),
        (("hlo", "optimized_hlo_sha256"), "0" * 64),
        (("host_materialization", "authority_sha256"), "0" * 64),
        (("numerical", "owner_agreement", "query_owners"), 1),
        (
            (
                "numerical",
                "cpu_candidate_watchpoint_matches_diagnostic_only",
                "watchpoint-0",
            ),
            1,
        ),
        (("policy", "valid_count"), True),
        (("tpu_execution_elapsed_ns_diagnostic_only",), True),
    ],
)
def test_runner_rejects_hostile_aliases_and_rebinding(
    path: tuple[str, ...], replacement: object
) -> None:
    runner = deepcopy(_runner())
    cursor: object = runner
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(RuntimeError, match="boundary|drifted"):
        MODULE._validate_runner(runner, "f" * 40)


def test_runner_rejects_extra_top_level_and_host_authority_fields() -> None:
    runner = _runner()
    runner["unbound_claim"] = False
    with pytest.raises(RuntimeError, match="claim boundary"):
        MODULE._validate_runner(runner, "f" * 40)
    runner = _runner()
    runner["host_materialization"]["unbound"] = False  # type: ignore[index]
    with pytest.raises(RuntimeError, match="claim boundary"):
        MODULE._validate_runner(runner, "f" * 40)


def _npy_bytes(shape: list[int], dtype: str, payload: bytes) -> bytes:
    header = repr({"descr": dtype, "fortran_order": False, "shape": tuple(shape)})
    prefix = b"\x93NUMPY\x01\x00"
    padding = (64 - ((len(prefix) + 2 + len(header) + 1) % 64)) % 64
    encoded = (header + (" " * padding) + "\n").encode("ascii")
    return prefix + len(encoded).to_bytes(2, "little") + encoded + payload


def _output_archive(
    *, mutation: tuple[str, str] | None = None
) -> tuple[bytes, dict[str, object]]:
    stream = BytesIO()
    records = _output_arrays()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, (
            shape,
            _semantic_dtype,
            storage_dtype,
        ) in MODULE.EXPECTED_OUTPUT_SPEC.items():
            item_size = {"|u1": 1, "<f4": 4, "<i4": 4, "<u2": 2}[storage_dtype]
            payload_size = item_size
            for dimension in shape:
                payload_size *= dimension
            payload = bytearray(payload_size)
            npy_shape = shape
            npy_dtype = storage_dtype
            if mutation == (name, "payload"):
                payload[0] = 1
            elif mutation == (name, "shape"):
                npy_shape = [*shape[:-1], shape[-1] + 1]
            elif mutation == (name, "dtype"):
                npy_dtype = "<i4" if storage_dtype != "<i4" else "<f4"
            archive.writestr(
                f"{name}.npy", _npy_bytes(npy_shape, npy_dtype, bytes(payload))
            )
            records[name]["array_sha256"] = sha256(bytes(payload_size)).hexdigest()  # type: ignore[index]
    return stream.getvalue(), records


def test_output_archive_requires_exact_unique_bounded_catalogue() -> None:
    raw, records = _output_archive()
    MODULE._validate_output_archive(raw, records)
    missing = BytesIO()
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("unexpected.npy", b"x")
    with pytest.raises(RuntimeError, match="archive catalogue"):
        MODULE._validate_output_archive(missing.getvalue(), records)
    duplicate = BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as archive:
            for name, (shape, _semantic, dtype) in MODULE.EXPECTED_OUTPUT_SPEC.items():
                item_size = {"|u1": 1, "<f4": 4, "<i4": 4, "<u2": 2}[dtype]
                payload_size = item_size
                for dimension in shape:
                    payload_size *= dimension
                member = _npy_bytes(shape, dtype, bytes(payload_size))
                archive.writestr(f"{name}.npy", member)
                if name == "query_owners":
                    archive.writestr(f"{name}.npy", member)
    with pytest.raises(RuntimeError, match="archive catalogue"):
        MODULE._validate_output_archive(duplicate.getvalue(), records)
    for mutation, message in (
        ("payload", "array bytes"),
        ("shape", "NPY schema"),
        ("dtype", "NPY schema"),
    ):
        attacked, attacked_records = _output_archive(
            mutation=("query_owners", mutation)
        )
        with pytest.raises(RuntimeError, match=message):
            MODULE._validate_output_archive(attacked, attacked_records)

    oversized = BytesIO()
    with zipfile.ZipFile(oversized, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in MODULE.EXPECTED_OUTPUT_SPEC:
            archive.writestr(f"{name}.npy", bytes(2 * 1024 * 1024))
    with pytest.raises(RuntimeError, match="archive catalogue"):
        MODULE._validate_output_archive(oversized.getvalue(), records)


def test_publication_primitives_are_exact_historical_bytes() -> None:
    primitive = ROOT / MODULE.PRIMITIVE_PATH
    assert sha256(primitive.read_bytes()).hexdigest() == MODULE.PRIMITIVE_SHA256
    loaded = MODULE._load_primitives("c0a1770b020060d33ccd8c569ec6405d40285865")
    assert loaded.REPO == MODULE.WORKTREE
    assert loaded.RUN_ROOT == MODULE.RUN_ROOT
    assert loaded.REMOTE_ROOT == MODULE.REMOTE_ROOT
    assert loaded.COMPILER_DRIVER_PATH == MODULE.WORKTREE / MODULE.DRIVER_PATH
