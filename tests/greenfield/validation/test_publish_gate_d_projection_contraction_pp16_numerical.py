from __future__ import annotations

import importlib.util
import re
import zipfile
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
PUBLISHER = (
    ROOT / "scripts/greenfield/publish_gate_d_projection_contraction_pp16_numerical.py"
)
RUNNER = ROOT / "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load(PUBLISHER, "projection_numerical_publisher_test")
RUNNER_MODULE = _load(RUNNER, "projection_numerical_runner_for_publisher_test")


V2_RUN = Path(
    "/home/gianl/gate-d-runs/gate_d_projection_contraction_pp16_numerical_20260901T235818944668679Z"
)
CAPSULE_INPUTS = Path(
    "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z/"
    "candidate-inputs.npz"
)


def _protected_bytes_present() -> bool:
    return (V2_RUN / "outputs.npz").exists() and CAPSULE_INPUTS.exists()


def _fixture(current_key: np.ndarray | None = None) -> tuple[bytes, dict, dict, bytes]:
    import ml_dtypes

    from glm_tpu.greenfield.validation.gate_d_projection_host_rope_numerical import (
        classify_host_rope_outputs,
        dsa_rope_row_identity,
        materialize_dsa_rope_row,
    )

    inputs = {k: v for k, v in np.load(CAPSULE_INPUTS).items()}
    outputs = {
        k: np.ascontiguousarray(v) for k, v in np.load(V2_RUN / "outputs.npz").items()
    }
    if current_key is not None:
        outputs["current_key_owners"] = np.ascontiguousarray(
            current_key, dtype=np.float32
        )
    wk = RUNNER_MODULE._materialize_wk(inputs, np, ml_dtypes)
    row = materialize_dsa_rope_row(np)
    numerical = classify_host_rope_outputs(
        outputs,
        np,
        wk_weight=wk,
        key_norm_weight_bits=inputs["key_norm_weight_bf16_bits"],
        key_norm_bias_bits=inputs["key_norm_bias_bf16_bits"],
        rope_row=row,
        ml_dtypes=ml_dtypes,
    )
    raw = RUNNER_MODULE._deterministic_npz(outputs, np)
    runner = {
        "output_arrays": {
            name: {
                "array_sha256": sha256(value.tobytes(order="C")).hexdigest(),
                "shape": list(value.shape),
                "storage_dtype": value.dtype.str,
            }
            for name, value in sorted(outputs.items())
        },
        "dsa_rope_row": dsa_rope_row_identity(row, np),
    }
    return raw, runner, numerical, np.ascontiguousarray(row[0]).tobytes()


def _fixed_key() -> np.ndarray:
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.kernels.reference.dsa_host_rope import (
        dsa_index_keys_from_projection_host_rope,
    )
    from glm_tpu.greenfield.validation.gate_d_projection_host_rope_numerical import (
        materialize_dsa_rope_row,
    )

    inputs = {k: v for k, v in np.load(CAPSULE_INPUTS).items()}
    outputs = {k: v for k, v in np.load(V2_RUN / "outputs.npz").items()}
    key = dsa_index_keys_from_projection_host_rope(
        jnp.asarray(outputs["projected_key_owners"][:, 0]),
        jnp.asarray(inputs["key_norm_weight_bf16_bits"][0].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["key_norm_bias_bf16_bits"][0].view(ml_dtypes.bfloat16)),
        jnp.asarray([8155, 8155], dtype=jnp.int32),
        jnp.asarray(materialize_dsa_rope_row(np)),
        key_norm_mode="divide_sqrt",
    )
    return np.asarray(key, dtype=np.float32)[:, None, :]


@pytest.mark.skipif(not _protected_bytes_present(), reason="protected run bytes absent")
def test_output_npz_rederives_v2_rejection_and_fixed_acceptance_from_bytes() -> None:
    raw, runner, numerical, rope = _fixture()
    assert numerical["accepted_tpu_host_rope_faithful"] is False
    assert MODULE._validate_output_npz(raw, runner, numerical, rope) is False
    fixed_raw, fixed_runner, fixed_numerical, rope = _fixture(_fixed_key())
    assert fixed_numerical["accepted_tpu_host_rope_faithful"] is True
    assert (
        MODULE._validate_output_npz(fixed_raw, fixed_runner, fixed_numerical, rope)
        is True
    )
    hostile = bytearray(fixed_raw)
    hostile[-20] ^= 1
    with pytest.raises((RuntimeError, zipfile.BadZipFile)):
        MODULE._validate_output_npz(bytes(hostile), fixed_runner, fixed_numerical, rope)
    hostile_row = bytearray(rope)
    hostile_row[0] ^= 1
    with pytest.raises(RuntimeError, match="rotary row bytes"):
        MODULE._validate_output_npz(
            fixed_raw, fixed_runner, fixed_numerical, bytes(hostile_row)
        )


@pytest.mark.skipif(not _protected_bytes_present(), reason="protected run bytes absent")
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda n: n.__setitem__("accepted_tpu_host_rope_faithful", True),
            "acceptance flag|faithfulness flags",
        ),
        (
            lambda n: n["outputs"]["current_key_owners"].__setitem__(
                "within_tolerance", True
            ),
            "faithfulness flags",
        ),
        (
            lambda n: n["outputs"]["current_key_owners"].__setitem__(
                "implied_rotary_within_tolerance", True
            ),
            "faithfulness flags",
        ),
        (
            lambda n: n["outputs"]["projected_key_owners"].__setitem__(
                "max_abs_error_vs_f64_reference", 0.0
            ),
            "faithfulness flags",
        ),
        (
            lambda n: n["outputs"]["normalized_hidden_owners"].__setitem__(
                "witness_exact", False
            ),
            "witness flag",
        ),
        (
            lambda n: n["outputs"]["current_key_owners"].__setitem__(
                "owners_equal", False
            ),
            "output record drifted",
        ),
    ],
)
def test_output_npz_rejects_runner_flag_attacks(mutate, message: str) -> None:
    raw, runner, numerical, rope = _fixture()
    mutate(numerical)
    with pytest.raises(RuntimeError, match=message):
        MODULE._validate_output_npz(raw, runner, numerical, rope)


@pytest.mark.skipif(not _protected_bytes_present(), reason="protected run bytes absent")
def test_output_npz_rejects_rope_row_record_attack() -> None:
    raw, runner, numerical, rope = _fixture()
    runner["dsa_rope_row"]["row_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="rotary row record"):
        MODULE._validate_output_npz(raw, runner, numerical, rope)


def test_pure_python_wk_materialization_matches_numpy() -> None:
    assert "dsa_rope_row.f32le" in MODULE.SUCCESS_PAYLOAD
    assert "_dsa_rope_row_host" not in PUBLISHER.read_text(encoding="utf-8")
    if CAPSULE_INPUTS.exists():
        import ml_dtypes

        inputs = {k: v for k, v in np.load(CAPSULE_INPUTS).items()}
        expected = RUNNER_MODULE._materialize_wk(inputs, np, ml_dtypes)[0]
        rows = MODULE._materialize_wk_owner0(MODULE._capsule_inputs())
        assert np.array_equal(np.asarray(rows, dtype=np.float32), expected)


@pytest.mark.parametrize(
    ("accepted", "status"),
    [(True, "NUMERICAL_ACCEPTED"), (False, "NUMERICAL_REJECTED")],
)
def test_status_is_derived_from_boolean_result(accepted: bool, status: str) -> None:
    observed, observed_status, classification = MODULE._expected_status(
        {"numerical": {"accepted_tpu_host_rope_faithful": accepted}}
    )
    assert observed is accepted
    assert observed_status == status
    assert classification.endswith("GATE_D_OPEN")
    assert ("UNFAITHFUL" in classification) is (not accepted)


def test_status_rejects_non_boolean() -> None:
    with pytest.raises(RuntimeError, match="classification is absent"):
        MODULE._expected_status({"numerical": {"accepted_tpu_host_rope_faithful": 1}})


def test_publisher_binds_wrapper_sync_line_and_v3_payload() -> None:
    source = PUBLISHER.read_text(encoding="utf-8")
    wrapper = (
        ROOT / "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.sh"
    ).read_text(encoding="ascii")
    assert "origin_and_same_region_mirror" in source
    assert "SYNC_OK %s %s origin_and_same_region_mirror" in wrapper
    assert "numerical_host_only" not in source
    assert "hlo/acquired_preimage" not in source
    assert "source_location_bridge" not in source
    assert (
        "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt"
        in MODULE.SUCCESS_PAYLOAD
    )
    assert (
        "hlo/projection_contraction_pp16_stage0.stablehlo.mlir"
        in MODULE.SUCCESS_PAYLOAD
    )
    assert MODULE.HLO_MODULE_NAME == "jit__projection_host_rope_local"
    assert (
        " cosine(" in MODULE.HLO_FORBIDDEN_TEXT
        and " sine(" in MODULE.HLO_FORBIDDEN_TEXT
    )


def test_source_keeps_terminal_upload_last_and_claims_open() -> None:
    source = PUBLISHER.read_text(encoding="utf-8")
    terminal_comment = source.index(
        "# This must remain the final remote mutation in the successful result path."
    )
    receipt = source.index('"terminal_upload_receipt.json"', terminal_comment)
    segment = source[terminal_comment:receipt]
    assert segment.count("base._upload_bound(") == 1
    assert '"gate_d_closed": False' in source
    assert '"performance_claim": False' in source
    assert '"root_cause_fix_proven": False' in source
    assert "diagnostic_objects.json" in source
    assert "--status" in source


def test_exact_hardened_base_loads_from_current_commit() -> None:
    import subprocess

    pin = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    base = MODULE._load_base(pin)
    assert base.REMOTE_ROOT == MODULE.REMOTE_ROOT
    assert base.TAG_PATTERN.pattern == MODULE.TAG_PATTERN_TEXT
    assert base._EXPECTED_COMPILER_ENVIRONMENT == MODULE.EXPECTED_COMPILER_ENVIRONMENT


def test_npy_parser_rejects_non_npy() -> None:
    with pytest.raises(RuntimeError, match="header"):
        MODULE._parse_npy(b"not-npy")


def _terminal_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "crc32c": "AAAAAA==",
        "generation": "1788300000000001",
        "path": "NUMERICAL_RESULT",
        "sha256": "2" * 64,
        "size": 512,
    }
    record.update(overrides)
    return record


def test_result_authority_line_binds_status_marker_and_terminal_identity() -> None:
    for status in ("NUMERICAL_ACCEPTED", "NUMERICAL_REJECTED"):
        line = MODULE._result_authority_line(status, "1" * 64, _terminal_record())
        assert line == (
            f"NUMERICAL_RESULT status={status} marker_sha256={'1' * 64} "
            f"terminal_generation=1788300000000001 terminal_sha256={'2' * 64}"
        )
        assert "\n" not in line
        assert line.isascii()


@pytest.mark.parametrize(
    ("status", "marker_sha256", "record"),
    [
        ("NUMERICAL_UNKNOWN", "1" * 64, _terminal_record()),
        ("", "1" * 64, _terminal_record()),
        ("NUMERICAL_ACCEPTED", "1" * 63, _terminal_record()),
        ("NUMERICAL_ACCEPTED", "A" * 64, _terminal_record()),
        ("NUMERICAL_ACCEPTED", "1" * 64, _terminal_record(generation="")),
        ("NUMERICAL_ACCEPTED", "1" * 64, _terminal_record(generation="12a")),
        ("NUMERICAL_ACCEPTED", "1" * 64, _terminal_record(generation="١٢")),
        ("NUMERICAL_ACCEPTED", "1" * 64, _terminal_record(generation=1788300000000001)),
        ("NUMERICAL_ACCEPTED", "1" * 64, _terminal_record(sha256="g" * 64)),
        ("NUMERICAL_ACCEPTED", "1" * 64, _terminal_record(sha256=None)),
        ("NUMERICAL_ACCEPTED", "1" * 64, {}),
    ],
)
def test_result_authority_line_rejects_unbound_identity(
    status: str, marker_sha256: str, record: dict[str, object]
) -> None:
    with pytest.raises(RuntimeError, match="result authority drifted"):
        MODULE._result_authority_line(status, marker_sha256, record)


def test_success_mode_prints_direct_authority_after_terminal_receipt() -> None:
    source = PUBLISHER.read_text(encoding="utf-8")
    receipt = source.index('"terminal_upload_receipt.json"')
    inventory = source.index(
        "projection numerical terminal receipt inventory drifted", receipt
    )
    authority = source.index("return _result_authority_line(", inventory)
    assert receipt < inventory < authority
    assert source.count("return _result_authority_line(") == 1
    assert re.search(r"print\(\s*_publish_success\(", source) is not None
    assert source.count("_publish_success(") == 2


@pytest.mark.skipif(not _protected_bytes_present(), reason="protected run bytes absent")
def test_output_npz_rejects_metric_tolerance_and_row_identity_attacks() -> None:
    raw, runner, numerical, rope = _fixture()
    for mutate in (
        lambda n: n["outputs"]["current_key_owners"].__setitem__(
            "rotary_max_abs_error", 0.0
        ),
        lambda n: n["outputs"]["current_key_owners"].__setitem__(
            "nonrotary_max_abs_error", 1.0
        ),
        lambda n: n["outputs"]["current_key_owners"]["implied_rotary"].__setitem__(
            "max_abs_cos_error", 0.0
        ),
        lambda n: n["outputs"]["current_key_owners"]["implied_rotary"].__setitem__(
            "extra", 0.0
        ),
        lambda n: n["outputs"]["projected_key_owners"].__setitem__("tolerance", 1e-3),
        lambda n: n["outputs"]["current_key_owners"].__setitem__("tolerance", 1e-2),
    ):
        hostile_raw, hostile_runner, hostile_numerical, hostile_rope = _fixture()
        mutate(hostile_numerical)
        with pytest.raises(RuntimeError, match="faithfulness flags"):
            MODULE._validate_output_npz(
                hostile_raw, hostile_runner, hostile_numerical, hostile_rope
            )
    for mutate in (
        lambda r: r.__setitem__("theta", 1e4),
        lambda r: r.__setitem__("shape", [1, 64]),
        lambda r: r.pop("rotary_dim"),
        lambda r: r.__setitem__("array_sha256", "0" * 64),
    ):
        hostile = {**runner, "dsa_rope_row": dict(runner["dsa_rope_row"])}
        mutate(hostile["dsa_rope_row"])
        with pytest.raises(RuntimeError, match="rotary row record"):
            MODULE._validate_output_npz(raw, hostile, numerical, rope)


def test_publisher_pins_driver_host_rope_source_hashes() -> None:
    assert MODULE.HOST_ROPE_SOURCE_SHA256S == RUNNER_MODULE.HOST_ROPE_SOURCE_SHA256S
    for relative, expected in MODULE.HOST_ROPE_SOURCE_SHA256S.items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    source = PUBLISHER.read_text(encoding="utf-8")
    assert (
        'runner.get("host_rope_source_sha256s") != HOST_ROPE_SOURCE_SHA256S' in source
    )
    assert "_metric_matches(" in source
