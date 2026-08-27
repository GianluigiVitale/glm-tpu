from __future__ import annotations

import pytest

from scripts.greenfield.restore_soft_deleted_checkpoint import (
    ObjectRecord,
    _reconcile,
)


def _record(name: str, generation: int = 10) -> ObjectRecord:
    return ObjectRecord(name=name, generation=generation, size=100, crc32c="crc")


def test_reconcile_selects_exact_soft_deleted_generations() -> None:
    payload = _record("prefix/payload")
    success = _record("prefix/SUCCESS", 20)
    expected = {item.name: item for item in (payload, success)}
    soft = {
        payload.name: [_record(payload.name, 9), payload],
        success.name: [success],
    }

    active, pending = _reconcile(expected, {}, soft)

    assert active == []
    assert pending == [payload, success]


def test_reconcile_resumes_matching_nonterminal_payload() -> None:
    payload = _record("prefix/payload")
    success = _record("prefix/SUCCESS", 20)
    restored_payload = _record(payload.name, 30)
    expected = {item.name: item for item in (payload, success)}

    active, pending = _reconcile(
        expected,
        {payload.name: [restored_payload]},
        {success.name: [success]},
    )

    assert active == [restored_payload]
    assert pending == [success]


def test_reconcile_rejects_premature_terminal_marker() -> None:
    payload = _record("prefix/payload")
    success = _record("prefix/SUCCESS", 20)
    expected = {item.name: item for item in (payload, success)}

    with pytest.raises(ValueError, match="terminal SUCCESS"):
        _reconcile(
            expected,
            {success.name: [_record(success.name, 30)]},
            {payload.name: [payload]},
        )


def test_reconcile_rejects_active_metadata_drift() -> None:
    payload = _record("prefix/payload")
    drifted = ObjectRecord(
        name=payload.name, generation=30, size=101, crc32c=payload.crc32c
    )

    with pytest.raises(ValueError, match="does not match capsule"):
        _reconcile({payload.name: payload}, {payload.name: [drifted]}, {})
