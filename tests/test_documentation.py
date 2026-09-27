"""The Markdown documentation names only what exists: every relative link resolves (a ``#fragment`` to a heading of
its target), and every test file a documented command names exists.

The pages are the Markdown files at the repository root and under ``docs/``, ``tests/`` and ``tools/``. The release
records of the GLM-5.3 release (``AGENTS.md``, ``HANDOFF.md``, ``goal.md``, ``docs/release/STATUS.md``) are not
checked: they are kept as written for that release until they are revised together, and ``docs/release/STATUS.md``
still links a configuration file that release had.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[1]
RELEASE_RECORDS = {"AGENTS.md", "HANDOFF.md", "goal.md", "docs/release/STATUS.md"}
PAGES = sorted(
    path.relative_to(REPO).as_posix()
    for path in [*REPO.glob("*.md"), *(p for top in ("docs", "tests", "tools") for p in (REPO / top).rglob("*.md"))]
    if path.relative_to(REPO).as_posix() not in RELEASE_RECORDS
)
LINK = re.compile(r"\]\(([^)\s]+)\)")
TEST_FILE = re.compile(r"(?<![\w/.-])tests/[\w/.-]+\.py")


def _prose_and_code(page: str) -> tuple[list[str], list[str]]:
    """The lines outside and inside fenced code blocks."""
    prose, code, fenced = [], [], False
    for line in (REPO / page).read_text().splitlines():
        if line.startswith("```"):
            fenced = not fenced
        else:
            (code if fenced else prose).append(line)
    return prose, code


def _anchor(heading: str) -> str:
    """The fragment of a Markdown heading: lower case, punctuation other than hyphens dropped, spaces as hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    prose, _ = _prose_and_code(path.relative_to(REPO).as_posix())
    return {_anchor(line.lstrip("#")) for line in prose if line.startswith("#")}


def test_the_documentation_pages_are_found():
    assert {"README.md", "docs/release/TESTING.md", "tools/equivalence/README.md"} <= set(PAGES)
    assert not RELEASE_RECORDS & set(PAGES)


@pytest.mark.parametrize("page", PAGES)
def test_relative_links_resolve(page):
    prose, _ = _prose_and_code(page)
    broken = []
    for line in prose:
        for target in LINK.findall(line):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                continue  # a URL
            file_part, _, fragment = target.partition("#")
            path = (REPO / page).parent / file_part if file_part else REPO / page
            if not path.exists() or (fragment and path.suffix == ".md" and fragment not in _anchors(path.resolve())):
                broken.append(target)
    assert broken == []


@pytest.mark.parametrize("page", PAGES)
def test_documented_test_files_exist(page):
    _, code = _prose_and_code(page)
    missing = sorted({name for line in code for name in TEST_FILE.findall(line) if not (REPO / name).is_file()})
    assert missing == []
