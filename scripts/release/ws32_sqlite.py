"""Shared host-only SQLite primitives for original-result provenance.

User-request publication needs these primitives, not benchmark scoring or its
archive/replay imports. Callers own admission, BEGIN/commit/rollback and connection
closure. The adapter deliberately suppresses legacy helpers' per-row commits.
The fixed database path preserves the existing site configuration; it is not a
portable deployment default. No connection or filesystem write occurs on import.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

PRIMARY_DB = Path("/home/gianl/glm-tpu/bench/results.db")


class Transaction:
    """Let old per-row helpers participate in ONE outer SQLite transaction."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def execute(self, *args: Any) -> Any:
        return self.connection.execute(*args)

    def commit(self) -> None:
        pass


def export_rows(connection: sqlite3.Connection, run_id: int) -> dict:
    result = {}
    for table, order in (("runs", "run_id"), ("items", "id"), ("summary", "id")):
        cursor = connection.execute(
            f"SELECT * FROM {table} WHERE run_id=? ORDER BY {order}", (run_id,)
        )
        names = [c[0] for c in cursor.description]
        result[table] = [dict(zip(names, row, strict=True)) for row in cursor]
    return result
