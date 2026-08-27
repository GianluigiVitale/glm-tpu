from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.greenfield.recover_short_decoder_pp8 import (
    _parse_results_uri,
    rollback_db,
    restore_db,
)


REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/greenfield/run_short_decoder_compile_pp8.sh"


def _database(path: Path, *, include_run: bool) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE runs (
          run_id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_utc TEXT NOT NULL,
          model TEXT NOT NULL
        );
        CREATE TABLE items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          item_id TEXT NOT NULL
        );
        CREATE TABLE summary (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL,
          metric TEXT NOT NULL
        );
        CREATE TABLE auxiliary (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          value TEXT NOT NULL
        );
        INSERT INTO runs VALUES (1, 'before', 'base');
        INSERT INTO items VALUES (1, 1, 'base-item');
        INSERT INTO summary VALUES (1, 1, 'base-summary');
        INSERT INTO auxiliary VALUES (1, 'same');
        """
    )
    if include_run:
        connection.executescript(
            """
            INSERT INTO runs VALUES (2, 'protected', 'gate-d');
            INSERT INTO items VALUES (2, 2, 'gate-d-item');
            INSERT INTO summary VALUES (2, 2, 'contract_valid');
            """
        )
    connection.commit()
    connection.close()


def test_restore_db_reinstates_only_exact_rolled_back_run(tmp_path: Path) -> None:
    live = tmp_path / "live.db"
    snapshot = tmp_path / "snapshot.db"
    _database(live, include_run=False)
    _database(snapshot, include_run=True)

    restore_db(snapshot, live, 2)

    connection = sqlite3.connect(live)
    assert connection.execute("SELECT * FROM runs ORDER BY run_id").fetchall() == [
        (1, "before", "base"),
        (2, "protected", "gate-d"),
    ]
    assert connection.execute("SELECT * FROM items ORDER BY id").fetchall()[-1] == (
        2,
        2,
        "gate-d-item",
    )
    assert connection.execute("SELECT * FROM summary ORDER BY id").fetchall()[
        -1
    ] == (2, 2, "contract_valid")
    connection.close()

    rollback_db(snapshot, live, 2)
    connection = sqlite3.connect(live)
    assert connection.execute("SELECT run_id FROM runs ORDER BY run_id").fetchall() == [
        (1,)
    ]
    assert connection.execute("SELECT run_id FROM items ORDER BY id").fetchall() == [
        (1,)
    ]
    assert connection.execute("SELECT run_id FROM summary ORDER BY id").fetchall() == [
        (1,)
    ]
    connection.close()


def test_restore_db_refuses_nonidentical_live_prefix(tmp_path: Path) -> None:
    live = tmp_path / "live.db"
    snapshot = tmp_path / "snapshot.db"
    _database(live, include_run=False)
    _database(snapshot, include_run=True)
    connection = sqlite3.connect(live)
    connection.execute("UPDATE auxiliary SET value = 'drifted'")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="exact snapshot prefix"):
        restore_db(snapshot, live, 2)


def test_recovery_uri_is_restricted_to_approved_results() -> None:
    assert _parse_results_uri(
        "gs://driftbench-dsv4-uc/results/recovery/"
    ) == ("driftbench-dsv4-uc", "results/recovery/")
    with pytest.raises(ValueError):
        _parse_results_uri("gs://driftbench-storage/results/recovery/")


def test_pp8_archive_ignores_empty_optional_directories() -> None:
    runner = RUNNER.read_text()
    normalized = " ".join(runner.replace(chr(92) + "\n", " ").split())
    marker = (
        'gcloud storage rsync --recursive --checksums-only "$RUN_DIR" '
        '"$REMOTE_PREFIX"'
    )
    assert marker in normalized
    assert 'gcloud storage cp --recursive --no-clobber "$RUN_DIR"/*' not in runner
    assert normalized.index(marker) < normalized.index('(root / "SUCCESS").write_text')
