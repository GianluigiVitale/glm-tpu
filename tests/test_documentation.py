"""The Markdown documentation names only what exists: every relative link resolves (a ``#fragment`` to a heading of
its target), and every test file a documented command names exists.

The pages are the tracked Markdown files at the repository root and under ``docs/``, ``tests/`` and ``tools/``, and
a link or a documented test file must name a tracked path (``git ls-files``): an untracked file in a checkout is not
documentation and does not satisfy a link. In a tree without git metadata (a ``git archive`` export, which holds the
tracked files only) the file system stands in. The governance files (``AGENTS.md``, ``docs/development/HANDOFF.md``,
``docs/development/goal.md``, ``docs/release/STATUS.md``) are pages like every other.

The repository map of ``AGENTS.md`` stays current: every tracked file and directory under ``glm_tpu/`` and ``tools/``
has an entry, every entry names a tracked path and says what it does, and no path is listed twice.
"""

from __future__ import annotations

from collections import Counter
import itertools
from pathlib import Path
import re
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
GOVERNANCE = {"AGENTS.md", "docs/development/HANDOFF.md", "docs/development/goal.md", "docs/release/STATUS.md"}


def _tracked() -> frozenset[str] | None:
    """The tracked files (``git ls-files``, the index); None without git metadata (an export: every file counts)."""
    if not (REPO / ".git").exists():
        return None
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    return frozenset(name for name in listed.decode().split("\0") if name)


TRACKED = _tracked()
DIRECTORIES = frozenset(str(parent) for name in TRACKED or () for parent in Path(name).parents if str(parent) != ".")
MARKDOWN = (
    TRACKED
    if TRACKED is not None
    else [
        path.relative_to(REPO).as_posix()
        for path in [*REPO.glob("*.md"), *(p for top in ("docs", "tests", "tools") for p in (REPO / top).rglob("*.md"))]
    ]
)
PAGES = sorted(
    name
    for name in MARKDOWN
    if name.endswith(".md") and ("/" not in name or name.split("/")[0] in ("docs", "tests", "tools"))
)
LINK = re.compile(r"\]\(([^)\s]+)\)")
REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(\S+)")  # a reference-style link definition
TEST_FILE = re.compile(r"(?<![\w/.-])tests/[\w/.{},-]+\.py")
BRACES = re.compile(r"\{([^{}]*)\}")


def _prose_and_code(page: str) -> tuple[list[str], list[str]]:
    """The lines outside and inside fenced code blocks (``` or ~~~)."""
    prose, code, fence = [], [], None
    for line in (REPO / page).read_text().splitlines():
        marker = line[:3] if line[:3] in ("```", "~~~") else None
        if marker and fence in (None, marker):
            fence = None if fence else marker
        else:
            (code if fence else prose).append(line)
    return prose, code


def _anchor(heading: str) -> str:
    """The fragment of a Markdown heading: lower case, punctuation other than hyphens dropped, spaces as hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    """Every heading's fragment; a repeated heading's later fragments get ``-1``, ``-2``, ... as on GitHub."""
    prose, _ = _prose_and_code(path.relative_to(REPO).as_posix())
    seen: Counter[str] = Counter()
    anchors = set()
    for line in prose:
        if line.startswith("#"):
            anchor = _anchor(line.lstrip("#"))
            anchors.add(f"{anchor}-{seen[anchor]}" if seen[anchor] else anchor)
            seen[anchor] += 1
    return anchors


def _exists(path: Path) -> bool:
    """``path`` is a tracked file or a directory holding one (in an export: exists)."""
    if TRACKED is None:
        return path.exists()
    try:
        name = path.resolve().relative_to(REPO).as_posix()
    except ValueError:  # outside the repository
        return False
    return name in TRACKED or name in DIRECTORIES or name == "."


def _expand(name: str) -> list[str]:
    """``tests/a/{b,c}.py`` -> ``tests/a/b.py``, ``tests/a/c.py`` (shell brace expansion)."""
    parts = BRACES.split(name)
    return [
        "".join(p) for p in itertools.product(*(part.split(",") if i % 2 else [part] for i, part in enumerate(parts)))
    ]


def test_the_documentation_pages_are_found():
    assert {"README.md", "docs/release/TESTING.md", "tools/equivalence/README.md"} <= set(PAGES)
    assert set(PAGES) >= GOVERNANCE
    assert all((REPO / page).is_file() for page in PAGES)


@pytest.mark.parametrize("page", PAGES)
def test_relative_links_resolve(page):
    prose, _ = _prose_and_code(page)
    broken = []
    for line in prose:
        for target in [*LINK.findall(line), *REFERENCE.findall(line)]:
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                continue  # a URL
            file_part, _, fragment = target.partition("#")
            path = (REPO / page).parent / file_part if file_part else REPO / page
            if not _exists(path) or (fragment and path.suffix == ".md" and fragment not in _anchors(path.resolve())):
                broken.append(target)
    assert broken == []


@pytest.mark.parametrize("page", PAGES)
def test_documented_test_files_exist(page):
    _, code = _prose_and_code(page)
    named = {path for line in code for name in TEST_FILE.findall(line) for path in _expand(name)}
    missing = sorted(name for name in named if not _exists(REPO / name) or not (REPO / name).is_file())
    assert missing == []


MAP_PAGE = "AGENTS.md"
MAP_SECTION = "## Repository map"
MAPPED_TOPS = ("glm_tpu", "tools")  # every tracked file and directory under these must have an entry
MAP_ENTRY = re.compile(r"(?P<indent> *)(?P<name>\S+?)(?: {2,}(?P<text>\S.*))?")


def _map_entries() -> list[tuple[str, str | None]]:
    """``(path, description)`` of every entry of the repository map: the fenced blocks of the section, each a tree
    whose lines are ``<indent><name>  <what it does>``, two spaces of indentation per level, a directory ending in
    ``/``; a block's first line may name a path from the repository root (``glm_tpu/config/``)."""
    lines = (REPO / MAP_PAGE).read_text().splitlines()
    start = lines.index(MAP_SECTION)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    entries, parents, fenced = [], [], False
    for line in lines[start + 1 : end]:
        if line.startswith(("```", "~~~")):
            fenced, parents = not fenced, []
            continue
        if not fenced or not line.strip():
            continue
        match = MAP_ENTRY.fullmatch(line)
        assert match is not None, f"not a map entry: {line!r}"
        depth, odd = divmod(len(match["indent"]), 2)
        assert not odd and depth <= len(parents), f"indentation of {line!r}"
        parents = parents[:depth]
        entries.append(("".join(parents) + match["name"].rstrip("/"), match["text"]))
        if match["name"].endswith("/"):
            parents.append(match["name"])
    return entries


def _mapped_paths_required() -> set[str]:
    """The tracked files under ``MAPPED_TOPS`` and their directories (in an export: the files on disk)."""
    if TRACKED is not None:
        files = {name for name in TRACKED if name.split("/")[0] in MAPPED_TOPS}
    else:
        files = {
            path.relative_to(REPO).as_posix()
            for top in MAPPED_TOPS
            for path in (REPO / top).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
    return files | {str(parent) for name in files for parent in Path(name).parents if str(parent) != "."}


def test_the_repository_map_is_parsed_and_every_entry_is_described():
    entries = _map_entries()
    assert len(entries) > 250
    assert [path for path, text in entries if not text] == []
    assert [path for path, count in Counter(path for path, _ in entries).items() if count > 1] == []


def test_every_repository_map_entry_exists():
    assert [path for path, _ in _map_entries() if not _exists(REPO / path)] == []


def test_the_repository_map_lists_every_package_and_tool_file():
    mapped = {path for path, _ in _map_entries()}
    assert sorted(_mapped_paths_required() - mapped) == []
