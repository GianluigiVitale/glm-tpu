"""Coverage/hash checks must not invent semantic review or deletion permission."""

import hashlib
import json
from pathlib import Path

import pytest
from tools import curation_inventory as inventory


@pytest.fixture
def tree(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    original = {"a.py": {"blob": "a" * 40, "bytes": 6, "mode": "100644"}}
    current = {"a.py"}
    monkeypatch.setattr(inventory, "baseline", lambda repo: original)
    monkeypatch.setattr(inventory, "tracked", lambda repo: current)
    return tmp_path, original, current


def reviewed(root, original):
    row = inventory.pending("a.py", original["a.py"])
    row.update(
        category="supported",
        action="keep",
        purpose="fixture implementation",
        consumers=["fixture caller"],
        justification="fully inspected fixture",
        review={
            "kind": "full_read",
            "sha256": hashlib.sha256((root / "a.py").read_bytes()).hexdigest(),
            "notes": "fixture only",
        },
    )
    return {"a.py": row}


def test_initial_inventory_cannot_claim_completion(tree):
    root, _, _ = tree
    rows = inventory.inventory(root)
    result = inventory.check(root, rows, ledger_path="disposition.jsonl")
    assert result["unresolved"] == 1 and not result["complete"]
    assert rows["a.py"]["review"]["kind"] == "unreviewed"


def test_reviewed_fixture_and_stale_bytes(tree):
    root, original, _ = tree
    rows = reviewed(root, original)
    assert inventory.check(root, rows, ledger_path="disposition.jsonl")["complete"]
    (root / "a.py").write_text("x = 2\n")
    assert (
        "reviewed bytes differ: a.py"
        in inventory.check(root, rows, ledger_path="disposition.jsonl")["errors"]
    )


def test_missing_new_file_and_changed_baseline_refuse(tree):
    root, original, current = tree
    rows = reviewed(root, original)
    current.add("new.py")
    rows["a.py"]["baseline"] = None
    errors = inventory.check(root, rows, ledger_path="disposition.jsonl")["errors"]
    assert "missing disposition: new.py" in errors
    assert "baseline identity differs: a.py" in errors


def test_generated_validation_cannot_replace_source_read(tree):
    root, original, _ = tree
    rows = reviewed(root, original)
    rows["a.py"]["review"]["kind"] = "generated_validation"
    assert (
        "implementation/documentation lacks full read: a.py"
        in inventory.check(root, rows, ledger_path="disposition.jsonl")["errors"]
    )


def test_removal_requires_missing_file_and_recovery(tree):
    root, original, _ = tree
    rows = reviewed(root, original)
    rows["a.py"].update(action="remove", category="research_or_superseded")
    errors = inventory.check(root, rows, ledger_path="disposition.jsonl")["errors"]
    assert "removal not implemented: a.py" in errors
    assert "removal lacks classification/recovery: a.py" in errors


def test_research_only_keep_is_not_curated(tree):
    root, original, _ = tree
    rows = reviewed(root, original)
    rows["a.py"]["category"] = "research_or_superseded"
    assert not inventory.check(root, rows, ledger_path="disposition.jsonl")["complete"]


@pytest.mark.parametrize("path", ["../private", "/absolute", "a//b"])
def test_invalid_ledger_paths_refused(tmp_path, path):
    file = tmp_path / "rows.jsonl"
    file.write_text(json.dumps({"path": path}) + "\n")
    with pytest.raises(ValueError):
        inventory.read_ledger(file)


def test_duplicate_rows_refused(tmp_path):
    file = tmp_path / "rows.jsonl"
    file.write_text('{"path":"a"}\n{"path":"a"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        inventory.read_ledger(file)


def test_default_dependency_roots_are_user_path_only():
    from tools.release_inventory import ROOTS

    assert ROOTS == (
        "glm_tpu/cli.py",
        "scripts/release/launch_ws32_user_request.py",
        "scripts/release/ws32_user_worker.py",
    )
