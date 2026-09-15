"""Source-lineage pins of the DB605 canonical dense two-file correction.

Verbatim (b667f00f) from the retired ws32_dense_canonical.py worker:
``MODEL_SOURCE_OVERRIDES`` and ``require_source``. The rolled prefill compiler
recipe calls this only on the historical canonical-dense path; every other
model source stays under the frozen pin. No digest or pinned file changes.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess

MODEL_SOURCE_OVERRIDES = {
    "glm_tpu/greenfield/kernels/ws32_prefill_window.py": "87f6fcad74aa2bbaa0e1ef7309c768021eb6d9686a5ff767a1776fc81a6c8c5d",
    "glm_tpu/greenfield/kernels/ws32_prefill_dense_canonical.py": "513635cc9c792f94792841041e6b3b89df3dbb0725ae5506c34ba2d1df198436",
}


def require_source(repo: Path) -> None:
    """Fixed two-file correction; every other model source stays frozen.

    Metadata preparation only, not acquired-HLO or full-decoder authorization.
    The protected parent additionally binds a clean published pin. Historical
    numerical profiles retain their original strict source checks.
    """
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        FROZEN_SOURCE_PIN,
        MODEL_SOURCE,
    )

    for name, expected in MODEL_SOURCE_OVERRIDES.items():
        path = repo / name
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("canonical dense source differs from registration")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            FROZEN_SOURCE_PIN,
            "--",
            *MODEL_SOURCE,
            *(":(exclude)" + name for name in MODEL_SOURCE_OVERRIDES),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("canonical dense changed an unrelated frozen model source")
