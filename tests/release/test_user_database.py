"""Actual SQLite transactions; fixture report does not establish model evidence."""

from copy import deepcopy
import json
import sqlite3

import pytest

from scripts.release import ws32_user_database as database
from scripts.release import ws32_user_evidence as evidence
from tests.release.test_user_evidence import case, TAG, PIN


def test_actual_replayed_user_result_idempotent_no_raw_prompt_or_quality_score(
    case, tmp_path
):
    root, _ = case
    report = evidence.replay_collected(root, TAG, PIN)
    path = tmp_path / "user.sqlite"
    result = database.record_result(database=path, tag=TAG, pin=PIN, report=report)
    assert result == database.record_result(
        database=path, tag=TAG, pin=PIN, report=report
    )
    assert not result["rows"]["items"] and not result["protected_result_sealed"]
    assert all(
        r["card_value"] is None and r["delta"] is None
        for r in result["rows"]["summary"]
    )
    env = json.loads(result["rows"]["runs"][0]["env_json"])
    assert env["benchmark"] is False and env["quality_score"] is None
    assert not {"prompt", "prompt_ids", "raw_output", "gold"}.intersection(env)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone() == (1,)
        conn.execute("UPDATE summary SET value=999 WHERE run_id=1")
    with pytest.raises(ValueError, match="rows changed"):
        database.record_result(database=path, tag=TAG, pin=PIN, report=report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("quality_score", 0.9),
        ("benchmark", True),
        ("trace_physical_coverage_verified", False),
        ("worker_ownership_verified", False),
        ("cold_admission_verified", False),
        ("protected_result_sealed", True),
    ],
)
def test_unadmitted_report_refused_before_database_creation(
    case, tmp_path, field, value
):
    root, _ = case
    report = evidence.replay_collected(root, TAG, PIN)
    report[field] = value
    path = tmp_path / "user.sqlite"
    with pytest.raises(ValueError):
        database.record_result(database=path, tag=TAG, pin=PIN, report=report)
    assert not path.exists()


def test_transaction_failure_rolls_back_and_retry_cannot_duplicate(
    case, tmp_path, monkeypatch
):
    root, _ = case
    report = evidence.replay_collected(root, TAG, PIN)
    path = tmp_path / "user.sqlite"
    original = database.export_rows
    monkeypatch.setattr(
        database,
        "export_rows",
        lambda *a: (_ for _ in ()).throw(RuntimeError("fixture interruption")),
    )
    with pytest.raises(RuntimeError):
        database.record_result(database=path, tag=TAG, pin=PIN, report=report)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone() == (0,)
    monkeypatch.setattr(database, "export_rows", original)
    database.record_result(database=path, tag=TAG, pin=PIN, report=report)
    changed = deepcopy(report)
    changed["token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different originals"):
        database.record_result(database=path, tag=TAG, pin=PIN, report=changed)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone() == (1,)
