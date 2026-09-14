"""Keep user provenance independent of the historical benchmark recorder."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from scripts.release import ws32_sqlite

REPO = Path(__file__).resolve().parents[2]
BASE = "b667f00f1ae48c8ff37e92500550c1395d74c66d"


def test_shared_primitives_match_original_asts() -> None:
    source = subprocess.check_output(
        ["git", "show", BASE + ":scripts/greenfield/ws32_native_benchmark_database.py"],
        cwd=REPO,
        text=True,
    )
    current = (REPO / "scripts/release/ws32_sqlite.py").read_text()

    def primitives(text):
        result = {}
        for node in ast.parse(text).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                result[node.name] = ast.dump(node)
            elif isinstance(node, ast.Assign) and node.targets[0].id == "PRIMARY_DB":
                result["PRIMARY_DB"] = ast.dump(node)
        return result

    original = primitives(source)
    assert primitives(current) == {
        name: original[name] for name in ("Transaction", "export_rows", "PRIMARY_DB")
    }


def test_adapter_does_not_commit_outer_transaction(tmp_path: Path) -> None:
    path = tmp_path / "transaction.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sample (value INTEGER)")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        transaction = ws32_sqlite.Transaction(connection)
        transaction.execute("INSERT INTO sample VALUES (?)", (7,))
        transaction.commit()
        with sqlite3.connect(path) as observer:
            assert observer.execute("SELECT * FROM sample").fetchall() == []
        connection.rollback()
        assert connection.execute("SELECT * FROM sample").fetchall() == []


def test_export_is_ordered_and_scoped_to_one_run() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE runs (run_id INTEGER, label TEXT)")
        connection.execute("CREATE TABLE items (id INTEGER, run_id INTEGER)")
        connection.execute("CREATE TABLE summary (id INTEGER, run_id INTEGER)")
        connection.executemany(
            "INSERT INTO runs VALUES (?, ?)", [(1, "one"), (2, "two")]
        )
        for table in ("items", "summary"):
            connection.executemany(
                f"INSERT INTO {table} VALUES (?, ?)", [(9, 1), (3, 2), (1, 1)]
            )
        result = ws32_sqlite.export_rows(connection, 1)
        assert result == {
            "runs": [{"run_id": 1, "label": "one"}],
            "items": [{"id": 1, "run_id": 1}, {"id": 9, "run_id": 1}],
            "summary": [{"id": 1, "run_id": 1}, {"id": 9, "run_id": 1}],
        }
        assert ws32_sqlite.export_rows(connection, 99) == {
            "runs": [],
            "items": [],
            "summary": [],
        }


def test_stdlib_primitives_and_isolated_user_publication() -> None:
    shared = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import scripts.release.ws32_sqlite; "
        "assert not any(n.startswith(('jax', 'torch', 'scripts.greenfield', 'bench')) "
        "for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", shared, str(REPO)],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        check=True,
        timeout=10,
    )
    user = (
        "import sys; import scripts.release.ws32_user_archive; "
        "import scripts.release.ws32_user_evidence; "
        "assert 'scripts.greenfield.ws32_native_benchmark_database' not in sys.modules; "
        "assert 'scripts.greenfield.ws32_native_benchmark_archive' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", user],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        check=True,
        timeout=20,
    )
