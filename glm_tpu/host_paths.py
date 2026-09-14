"""Lightweight host-path preflight, independent of experiment orchestration.

This preserves the original native evidence guard's behavior. It requires an
absolute path without symlink components; it does not require the target to
exist or provide race-free descriptor-based access. Callers remain responsible
for file type, ownership, bounds, hashes and publication semantics.
"""

from __future__ import annotations

from pathlib import Path


def _plain_path(path: Path) -> None:
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(
            "history retained path must be absolute with no symlink ancestors"
        )
