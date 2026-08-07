from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "configs" / "greenfield-reuse-inventory.json"
GREENFIELD = ROOT / "glm_tpu" / "greenfield"


def _inventory() -> dict:
    return json.loads(INVENTORY.read_text())


def test_reuse_inventory_is_complete_and_unambiguous() -> None:
    document = _inventory()
    assert document["schema_version"] == 1
    dispositions = set(document["policy"])
    assert dispositions == {
        "adapted",
        "direct_reuse",
        "oracle_only",
        "rejected",
        "research_candidate",
    }
    assets = document["assets"]
    ids = [asset["id"] for asset in assets]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 25
    required = {
        "provenance-db",
        "xplane-wall-analysis",
        "protected-run-operations",
        "packed-checkpoint-chain",
        "legacy-state-integrity",
        "legacy-dsa-diff-and-dump",
        "legacy-dsa-kernels",
        "legacy-distributed-q-a-rmsnorm",
        "legacy-fp8-matmul-and-gmm",
        "legacy-pipeline-parallel-helpers",
        "ws32-2d-layout-reference",
        "quantized-2d-matmul-drafts",
        "glm-mtp-and-indexshare",
        "dsv4-parity-methodology",
        "legacy-execution-stack",
    }
    assert required <= set(ids)
    for asset in assets:
        assert asset["disposition"] in dispositions
        assert asset["source_paths"]
        assert asset["reason"].strip()
        assert asset["gate"].strip()


def test_external_model_sources_are_never_direct_dependencies() -> None:
    for asset in _inventory()["assets"]:
        source = asset["source_repo"].lower()
        if "tpu-inference" in source or "vllm" in source:
            assert asset["disposition"] != "direct_reuse", asset["id"]


def test_greenfield_execution_has_no_legacy_or_vllm_imports() -> None:
    forbidden = {"tpu_inference", "vllm"}
    offenders: list[str] = []
    for path in sorted(GREENFIELD.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            if roots & forbidden:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


def test_reuse_inventory_is_linked_from_project_entry_points() -> None:
    target = "docs/greenfield/REUSE_INVENTORY.md"
    assert target in (ROOT / "README.md").read_text()
    assert target in (ROOT / "AGENTS.md").read_text()
