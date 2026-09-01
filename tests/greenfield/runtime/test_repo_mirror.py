from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


def test_repository_mirror_is_fail_closed_in_tpu_region() -> None:
    script = (
        REPO / "scripts/greenfield/sync_repo_mirror_same_region.sh"
    ).read_text(encoding="utf-8")
    assert "BUCKET=gs://driftbench-dsv4-uc" in script
    assert 'if [ "$location" != "US-CENTRAL2" ]' in script
    assert "gsutil -m rsync -r -d" in script
    assert "/home/gianl/.glm-tpu-rsync.lock" not in script
    assert "driftbench-storage" not in script
    assert "/home/gianl/bucket" not in script


def test_repository_mirror_covers_both_recovery_trees() -> None:
    script = (
        REPO / "scripts/greenfield/sync_repo_mirror_same_region.sh"
    ).read_text(encoding="utf-8")
    helper = (
        REPO / "scripts/greenfield/mirror_gate_d_evidence.py"
    ).read_text(encoding="utf-8")
    assert '"/home/gianl/glm-tpu:repos/glm-tpu"' in script
    assert (
        '"/home/gianl/glm-tpu-topology-rewrite:'
        'repos/glm-tpu-topology-rewrite"'
    ) in script
    assert (
        '"/home/gianl/glm-tpu-gate-d-evidence:'
        'repos/glm-tpu-gate-d-evidence"'
    ) in script
    assert "set -o pipefail" in script
    assert "--mode committed --require-remote-equal" in script
    assert "mirror_gate_d_evidence.py" in script
    assert '--repository "$src" --output-root "$evidence_export"' in script
    assert "authenticated append-only evidence mirror failed" in script
    assert 'gsutil -m rsync -r -d -e -x "$EXCLUDE" "$sync_src"' in script
    assert "gate-d-evidence-mirror-authority.json" in helper
    assert "--if-generation-match=0" in helper
    assert "--content-md5=" in helper
    assert "--if-generation-match={remote.generation}" in helper
    assert "gsutil" not in helper
