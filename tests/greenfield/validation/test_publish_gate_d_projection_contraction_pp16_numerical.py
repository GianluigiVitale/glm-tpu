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


def _accepted_arrays() -> dict[str, np.ndarray]:
    return {
        "current_key_owners": np.zeros((2, 1, 128), dtype=np.float32),
        "normalized_hidden_owners": np.zeros((2, 1, 6144), dtype=np.uint16),
        "projected_key_owners": np.zeros((2, 1, 128), dtype=np.float32),
    }


def _owner_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: sha256(value[0].tobytes(order="C")).hexdigest()
        for name, value in arrays.items()
    }


def _output_fixture(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    expected_owner_hashes: dict[str, str] | None = None,
) -> tuple[bytes, dict, dict, dict[str, str]]:
    arrays = _accepted_arrays() if arrays is None else arrays
    expected_owner_hashes = (
        _owner_hashes(_accepted_arrays())
        if expected_owner_hashes is None
        else expected_owner_hashes
    )
    raw = RUNNER_MODULE._deterministic_npz(arrays, np)
    output_arrays = {}
    numerical_arrays = {}
    all_exact = True
    for name, value in arrays.items():
        hashes = [
            sha256(value[index].tobytes(order="C")).hexdigest() for index in range(2)
        ]
        owners_equal = value[0].tobytes(order="C") == value[1].tobytes(order="C")
        witness_exact = hashes == [
            expected_owner_hashes[name],
            expected_owner_hashes[name],
        ]
        finite = bool(np.all(np.isfinite(value)))
        output_arrays[name] = {
            "array_sha256": sha256(value.tobytes(order="C")).hexdigest(),
            "shape": list(value.shape),
            "storage_dtype": value.dtype.str,
        }
        numerical_arrays[name] = {
            "dtype_exact": True,
            "finite": finite,
            "owner_sha256": hashes,
            "owners_equal": owners_equal,
            "shape_exact": True,
            "witness_exact": witness_exact,
        }
        all_exact = all_exact and finite and owners_equal and witness_exact
    runner = {"output_arrays": output_arrays}
    numerical = {
        "accepted_tpu_projection_match": all_exact,
        "outputs": numerical_arrays,
    }
    return raw, runner, numerical, expected_owner_hashes


def test_output_npz_parser_binds_every_owner_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, runner, numerical, owner_hashes = _output_fixture()
    monkeypatch.setattr(MODULE, "EXPECTED_OWNER_SHA256", owner_hashes)
    assert MODULE._validate_output_npz(raw, runner, numerical) is True
    hostile = bytearray(raw)
    hostile[-20] ^= 1
    with pytest.raises((RuntimeError, zipfile.BadZipFile)):
        MODULE._validate_output_npz(bytes(hostile), runner, numerical)


def test_output_npz_publishes_genuine_value_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _owner_hashes(_accepted_arrays())
    rejected = _accepted_arrays()
    rejected["current_key_owners"][:, 0, 0] = 1.0
    raw, runner, numerical, _ = _output_fixture(
        rejected, expected_owner_hashes=expected
    )
    monkeypatch.setattr(MODULE, "EXPECTED_OWNER_SHA256", expected)
    assert numerical["outputs"]["current_key_owners"]["witness_exact"] is False
    assert MODULE._validate_output_npz(raw, runner, numerical) is False


def test_output_npz_publishes_owner_disagreement_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _owner_hashes(_accepted_arrays())
    rejected = _accepted_arrays()
    rejected["projected_key_owners"][1, 0, 0] = 1.0
    raw, runner, numerical, _ = _output_fixture(
        rejected, expected_owner_hashes=expected
    )
    monkeypatch.setattr(MODULE, "EXPECTED_OWNER_SHA256", expected)
    assert numerical["outputs"]["projected_key_owners"]["owners_equal"] is False
    assert MODULE._validate_output_npz(raw, runner, numerical) is False


@pytest.mark.parametrize(
    "flag",
    ["dtype_exact", "finite", "owners_equal", "shape_exact", "witness_exact"],
)
def test_output_npz_rejects_runner_flag_attack(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    raw, runner, numerical, expected = _output_fixture()
    monkeypatch.setattr(MODULE, "EXPECTED_OWNER_SHA256", expected)
    numerical["outputs"]["current_key_owners"][flag] = False
    numerical["accepted_tpu_projection_match"] = False
    with pytest.raises(RuntimeError, match="output bytes drifted"):
        MODULE._validate_output_npz(raw, runner, numerical)


def test_output_npz_rejects_acceptance_flag_attack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, runner, numerical, expected = _output_fixture()
    monkeypatch.setattr(MODULE, "EXPECTED_OWNER_SHA256", expected)
    numerical["accepted_tpu_projection_match"] = False
    with pytest.raises(RuntimeError, match="acceptance flag"):
        MODULE._validate_output_npz(raw, runner, numerical)


@pytest.mark.parametrize(
    ("accepted", "status"),
    [(True, "NUMERICAL_ACCEPTED"), (False, "NUMERICAL_REJECTED")],
)
def test_status_is_derived_from_boolean_result(accepted: bool, status: str) -> None:
    observed, observed_status, classification = MODULE._expected_status(
        {"numerical": {"accepted_tpu_projection_match": accepted}}
    )
    assert observed is accepted
    assert observed_status == status
    assert classification.endswith("GATE_D_OPEN")


def test_status_rejects_non_boolean() -> None:
    with pytest.raises(RuntimeError, match="classification is absent"):
        MODULE._expected_status({"numerical": {"accepted_tpu_projection_match": 1}})


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
