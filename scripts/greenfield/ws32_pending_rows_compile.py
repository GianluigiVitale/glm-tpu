"""One pending-row E0 graph; metadata preparation ONLY, no numerical opt-in.

Reuse authenticated production inputs, unchanged model kernels and explicit
state-only ownership. This new source recipe must never widen old source guards.
Protected worker/collector registration remains required before TPU compilation.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_canonical_prefill_compile as canonical
from scripts.greenfield import ws32_rolled_prefill_compile as original
from scripts.greenfield.ws32_prefill_owned_state import consume_state


PROGRAM = "prefill_256k_pending_rows"
MODEL_SOURCE_OVERRIDES = {
    **canonical.MODEL_SOURCE_OVERRIDES,
    "glm_tpu/greenfield/runtime/ws32_batched_prefill.py": "b67a86d4d5336f0926a7b176fae2675a2a007c5ef6e28668493fa219025b48b9",
    "glm_tpu/greenfield/kernels/prefill_pending_rows.py": "bc9a7f837109df6e9ae47981d0399873e186723c1f3f0eb52ea2d70a02e352ab",
}
PREREQUISITES = {
    **canonical.PREREQUISITES,
    "docs/artifacts/prefill-owned-state-db613-sealed-20260912.json": "5893c4eb0d8f064df09ca3d7d8b9626eafc049fe5b7ba993d9796fb2fed5c37b",
}


def require_source(repo: Path) -> None:
    """Bind the two changed model files and original canonical/DB613 basis."""
    for name, expected in {**MODEL_SOURCE_OVERRIDES, **PREREQUISITES}.items():
        path = repo / name
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("pending rows source/prerequisite differs")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            admission.FROZEN_SOURCE_PIN,
            "--",
            *admission.MODEL_SOURCE,
            *(":(exclude)" + name for name in MODEL_SOURCE_OVERRIDES),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("pending rows changed unrelated frozen model source")


def read_metadata(repo: Path) -> Any:
    return original.read_metadata(repo, full_canonical=True, pending_cache_rows=True)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> original.AbstractPrefillPair:
    pair = original.prepare(
        mesh,
        metadata,
        repo=repo,
        full_canonical=True,
        long_context_label="256k_e0",
        pending_cache_rows=True,
    )
    return original.AbstractPrefillPair(
        {PROGRAM: consume_state(pair.programs["prefill_chunk"])},
        {PROGRAM: pair.inputs["prefill_chunk"]},
        pair.manifest_sha256,
        pair.source_inventory_sha256,
    )


# Actual production metadata-only CPU TPU-target lowering; not TPU compilation.
RAW = {
    PROGRAM: (21096032, "1ede24a42a146898af7ea22ffb6d0a0e87178ada33695cbbb540477a9bdea873")
}
# Protected compiler route and numerical/runtime HBM admission remain unwired.
