"""Spec §23.8: rotary long-position diagnostic record is self-verifying and pre-registered."""
from __future__ import annotations

import json

import numpy as np
import pytest

from glm_tpu.greenfield.validation import rotary_diagnostic as rd


def test_rotary_diagnostic_record_is_complete_and_self_consistent() -> None:
    result = rd.run_rotary_diagnostic(positions=9_000)  # bands le_8192 and part of le_32768
    record = result.record
    rd.verify_rotary_diagnostic_record(record, expected_script_sha256=rd.script_sha256(), expected_backend="cpu", pinned=False)
    with pytest.raises(ValueError, match="pre-registered"):
        rd.verify_rotary_diagnostic_record(record, expected_backend="cpu", pinned=True)
    with pytest.raises(ValueError, match="expected 'tpu'"):
        rd.verify_rotary_diagnostic_record(record, pinned=False)
    assert record["platform"]["backend"] == "cpu"
    bands = {cell["band"] for cell in record["cells"]}
    assert bands == {"le_8192", "le_32768"}
    assert len(record["cells"]) == 2 * 32
    assert record["positions"] == 9_000 and record["kappa"] == 2.0
    for cell in record["cells"]:
        assert cell["bound"] == pytest.approx(2.0 * cell["legacy_main_max_abs_error_vs_fp64"])
        assert cell["pass"] == (cell["device_main_vs_legacy_main_max_abs"] <= cell["bound"])
        assert cell["device_indexer_max_abs_error_vs_fp64"] >= cell["device_indexer_p99_abs_error_vs_fp64"] >= 0.0
    # Pair 0 has exact FP32 integer angles: the legacy table is within one BF16 ulp of FP64.
    pair0 = [c for c in record["cells"] if c["pair"] == 0 and c["band"] == "le_8192"][0]
    assert pair0["legacy_main_max_abs_error_vs_fp64"] <= 2 ** -8
    assert json.dumps(record)  # serialisable


def test_rotary_diagnostic_verification_refuses_tampering() -> None:
    record = rd.run_rotary_diagnostic(positions=600).record
    tampered = dict(record)
    tampered["verdict"] = "FAIL"
    with pytest.raises(ValueError, match="checksum drifted"):
        rd.verify_rotary_diagnostic_record(tampered, expected_backend="cpu", pinned=False)
    with pytest.raises(ValueError, match="script identity drifted"):
        rd.verify_rotary_diagnostic_record(record, expected_script_sha256="0" * 64, expected_backend="cpu", pinned=False)
    # A record whose cells contradict its verdict is refused even with a fresh checksum.
    forged = json.loads(json.dumps(record))
    forged["cells"][0]["pass"] = False
    forged.pop("record_sha256")
    from hashlib import sha256
    forged["record_sha256"] = sha256(
        json.dumps(forged, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="verdict does not match"):
        rd.verify_rotary_diagnostic_record(forged, expected_backend="cpu", pinned=False)


def test_fp64_and_legacy_forms_agree_at_small_positions() -> None:
    ids = np.arange(64, dtype=np.int64)
    cos64, sin64 = rd.fp64_cos_sin(ids)
    cos, sin = rd.legacy_main_cos_sin(ids)
    assert np.max(np.abs(cos - cos64)) <= 2 ** -8 and np.max(np.abs(sin - sin64)) <= 2 ** -8
