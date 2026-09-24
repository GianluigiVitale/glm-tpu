"""Example site configurations for tests (neutral placeholders: example names, RFC 5737 addresses,
``gs://example-bucket/``). The values are the equivalence harness's synthetic site
(``tools.equivalence.site_fixture``), so tests and gates exercise one example."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from glm_tpu.config.site import SiteConfig, set_current_site
from tools.equivalence.site_fixture import EXAMPLE_BUCKET, EXAMPLE_COORDINATOR, site_mapping, write_site

__all__ = [
    "EXAMPLE_BUCKET",
    "EXAMPLE_COORDINATOR",
    "example_mapping",
    "example_site",
    "installed_site",
    "write_example_site",
]


def example_mapping(base: Path, **overrides: dict[str, Any]) -> dict[str, Any]:
    return site_mapping(Path(base), **overrides)


def example_site(base: Path, **overrides: dict[str, Any]) -> SiteConfig:
    return SiteConfig.from_mapping(example_mapping(base, **overrides))


def write_example_site(path: Path, mapping: dict[str, Any] | None = None) -> Path:
    """An owner-only example site file at ``path`` (paths under its directory by default)."""
    return write_site(Path(path), mapping if mapping is not None else example_mapping(Path(path).parent))


@contextmanager
def installed_site(site: SiteConfig | None) -> Iterator[SiteConfig | None]:
    """``site`` as the process's current site for the block (restores the previous one)."""
    previous = set_current_site(site)
    try:
        yield site
    finally:
        set_current_site(previous)
