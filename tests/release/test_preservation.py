"""Removed legacy automation stays exactly recoverable in preserved local Git."""
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_removed_legacy_schedulers_have_exact_preserved_blobs():
    ledger = json.loads((REPO / "docs/release/removed-legacy-schedulers.json").read_bytes())
    pin = ledger["preserved_commit"]
    assert len(ledger["files"]) == 7
    assert len({row["path"] for row in ledger["files"]}) == 7
    for row in ledger["files"]:
        relative = Path(row["path"])
        assert relative.parts[0] == "scripts"
        assert not relative.is_absolute() and ".." not in relative.parts
        assert not (REPO / relative).exists()
        blob = subprocess.check_output(
            ["git", "rev-parse", f"{pin}:{row['path']}"], cwd=REPO, text=True
        ).strip()
        assert blob == row["blob"]
        size = subprocess.check_output(["git", "cat-file", "-s", blob], cwd=REPO, text=True)
        assert int(size) == row["bytes"]


def test_dynamic_registry_and_oracle_dependencies_remain_available():
    for relative in ("bench/benchmarks.py", "bench/extract.py", "bench/provenance.py",
                     "scripts/launch_glm_32chip.sh", "scripts/validate_ray_network.sh",
                     "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"):
        assert (REPO / relative).is_file()
