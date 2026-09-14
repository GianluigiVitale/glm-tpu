"""The native greenfield implementation must not import legacy execution."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GREENFIELD = ROOT / "glm_tpu" / "greenfield"


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
