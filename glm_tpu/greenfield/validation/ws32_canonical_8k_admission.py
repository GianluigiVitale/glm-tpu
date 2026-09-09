"""Prospective corrected8K scope; retained diagnostics predict, never certify.

DB610 supplies the complete corrected2K prerequisite. The independent first-
event record comes from the named live32 diagnostic and uses the existing FP64
row. The unchanged section21 loader/sealer must rederive the new run's checks.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


PROFILE = "ws32_b128_b114_8k_cap8192_canonical_dense_v1"
RECORD = "docs/artifacts/gate-d-canonical8k-live32-event1-registration-v2-20260909.json"
RECORD_SHA256 = "2da2e6db7a7e3540a64579127f60db2e9230ea84f89bebef4f37c1ae7a1f3c69"
PREREQUISITES = {
    "docs/artifacts/prefill-canonical-short-db610-sealed-20260909.json": "952c30cde60768c572d13d8dfe192cc44c0ea20b9974202a687901be235de91c",
    "docs/artifacts/prefill-frozen-live32-diagnostic-20260909.json": "1c597126721ccea76c58e5ce066be82a38fc0975688dd37037c6b9c7793571f0",
}


def require_prerequisites(repo: Path) -> None:
    """Bind original prerequisite bytes, without inheriting a numerical pass."""
    from .ws32_short_context import load_ws32_adjudicated_divergence

    for name, digest in PREREQUISITES.items():
        path = repo / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("canonical8K prerequisite evidence drifted")
    load_ws32_adjudicated_divergence(
        repo / RECORD, expected_sha256=RECORD_SHA256, repository_root=repo
    )


def require_adjudication(repo: Path, record: Path | None, digest: str) -> None:
    """Only this exact prospective record can enter the canonical8K profile."""
    if (
        record is None
        or Path(record).is_symlink()
        or Path(record).resolve() != (repo / RECORD).resolve()
        or digest != RECORD_SHA256
    ):
        raise ValueError("canonical8K requires its own exact preregistered record")
    require_prerequisites(repo)
