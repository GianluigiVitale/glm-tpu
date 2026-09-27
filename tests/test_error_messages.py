"""Error messages carry no development-phase labels.

Every ``raise`` statement of the ``glm_tpu`` package is scanned: a string constant in it (a literal or an f-string
part) may carry a label (``WS32``, ``D8``, ``challenger``, ``greenfield``, ``research``, in any letter case) only when
the equivalence records hold the message, so it must not change: five checkpoint refusals of G4 (frozen) and one
host-mapping refusal of G9 (changed only by an H-numbered host change). The checkpoint-format identifiers that carry a
label (``greenfield_ws32_runtime_*``, ``WS32_2D``) are data, not messages, and are not scanned.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
LABEL = re.compile(r"ws32|\bd8\b|research|challenger|greenfield", re.IGNORECASE)
# (file, string constant) -> the record that holds the message
RECORDED = {
    ("glm_tpu/distributed/parallel_state.py", "WS32 launch/JAX/topology fleet mapping drifted"): "wire.json",
    ("glm_tpu/model_loader/sharded_state/loader.py", "WS32 tensor checksum drifted while loading "): (
        "checkpoint_identity.json"
    ),
    ("glm_tpu/model_loader/sharded_state/verify.py", "WS32 local slot layout must not contain foreign slot "): (
        "checkpoint_identity.json"
    ),
    ("glm_tpu/model_loader/sharded_state/verify.py", "WS32 runtime file/tensor checksum drifted: "): (
        "checkpoint_identity.json"
    ),
    ("glm_tpu/model_loader/sharded_state/verify.py", "WS32 runtime manifest identity drifted"): (
        "checkpoint_identity.json"
    ),
    ("glm_tpu/model_loader/sharded_state/verify.py", "WS32 runtime SUCCESS identity drifted"): (
        "checkpoint_identity.json"
    ),
}


def labelled_raise_constants() -> set[tuple[str, str]]:
    found = set()
    for path in sorted((REPO / "glm_tpu").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Raise):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str) and LABEL.search(inner.value):
                    found.add((path.relative_to(REPO).as_posix(), inner.value))
    return found


def test_only_the_recorded_messages_keep_a_campaign_label():
    assert labelled_raise_constants() == set(RECORDED)


def test_the_kept_messages_are_the_recorded_ones():
    for (_, text), record in RECORDED.items():
        assert text in (REPO / "tests" / "golden" / "data" / record).read_text(), (record, text)
