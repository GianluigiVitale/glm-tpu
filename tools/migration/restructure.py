"""Apply the restructure tables of DESIGN 10.3 S3/S4 and check that they are applied.

    python tools/migration/restructure.py --apply [TABLE]   # default move_map.toml (S3)
    python tools/migration/restructure.py --check [TABLE ...]   # default: S3 and the S4.4 table

Tables (``[stage] kind``): ``move_map.toml`` (S3, files, below), ``symbol_moves.toml`` (S4.1,
top-level definitions moved to their final modules), ``renames.toml`` (S4.2, public names) and
``test_merges.toml`` (S4.3, the basename-kept tests merged into their mirrored test modules);
the S4 kinds are applied by ``symbols.py`` (its docstring has the rules). ``--check`` exits 1 when
a table is not fully applied: for S3 as described below, for S4 when a moved or renamed
definition differs from its base-commit original (imports aside), a dissolved module or any
tracked ``_s3_`` path is left, or a Python or TOML file still names a dissolved module. Each S4
table is checked at its own step: a later table renames what an earlier one placed
(``--check symbol_moves.toml`` reports the S4.2 renames as differences), and S4.4 edits
definitions the earlier tables placed (reviewed lint fixes; the public helper names). So without a
TABLE argument ``--check`` runs ``move_map.toml`` and, once it exists, the S4.4 table
``helper_names.toml``; an earlier S4 table is checked at the commit that applied it.

S3, ``move_map.toml``:

``--apply``:

1. ``git mv`` every file of ``[files]`` whose destination differs (parents created), ``git rm``
   the ``[removed]`` initializers of the dissolved packages, and create the ``[packages]``
   initializers (docstring only);
2. rewrite, in every tracked text file of ``[rewrite]``, the references to the moved files:
   * Python (libcst): every import -- relative imports of a moved file become absolute, a module
     imported from a package it no longer belongs to is imported from its new package under its
     old local name (``from glm_tpu.config import _s3_model as model``) --; every dotted module
     name and repository path inside a string or a comment (monkeypatch targets, ``python -m``
     strings, entry-module lists, ``module:qualname`` keys, manifest paths); path joins whose
     string literals spell a moved path (``REPO / "glm_tpu" / "optimized" / "runtime.py"``); and
     repository-root anchors ``Path(__file__).resolve().parents[N]`` of a file whose depth changed;
   * other text (Markdown, TOML, ...): dotted names and paths, the same maps;
   * ``pyproject.toml``: additionally the package-data globs and license files of ``[pyproject]``
     (and the legacy black boundary while ``[tool.black]`` exists: S4.3 replaced it by ruff);
   names listed in ``[baseline_references]`` for a file stay as spelled there;
3. add a ``closure_map.toml`` ``[modules]`` entry (new name = recorded name) for every module this
   run moved (G6/G7 pass through the table until the rename-only re-record clears it).

``--check`` computes the same without writing and also scans the rewritten files for any
remaining reference into the dissolved packages (``glm_tpu.optimized``, ``glm_tpu.greenfield``,
``scripts``, ``tests.greenfield``, ``tests.release``) or to a moved path, except the reviewed
``[baseline_references]`` and, in the ``[provenance]`` files, names of the research tree archived
at S2f; Markdown hits are reported (the S6 documentation rewrite). Standard library plus libcst.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import fnmatch
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any

import libcst as cst

REPO = Path(__file__).resolve().parents[2]
MAP = Path(__file__).with_name("move_map.toml")
SYMBOLS = Path(__file__).with_name("symbol_moves.toml")  # S4.1
RENAMES = Path(__file__).with_name("renames.toml")  # S4.2
MERGES = Path(__file__).with_name("test_merges.toml")  # S4.3
HELPERS = Path(__file__).with_name("helper_names.toml")  # S4.4
CLOSURE_MAP = REPO / "tools" / "equivalence" / "closure_map.toml"
# Any remaining reference into these fails --check (Python, TOML); Markdown hits are reported.
STALE = re.compile(
    r"(?<![\w.])(?:glm_tpu\.(?:optimized|greenfield|web)|glm_tpu\.(?:api|cli|ui|user_request)(?![\w])|"
    r"scripts\.[A-Za-z_]|tests\.(?:greenfield|release)(?![\w]))"
    r"|(?<![\w.-])(?:glm_tpu/(?:optimized|greenfield|web)/|glm_tpu/(?:api|cli|ui|user_request)\.py|"
    r"scripts/[A-Za-z_]|reference/hf-glm53|licenses/GLM-5\.3|tests/(?:greenfield|release)/)"
)
# Names of the research tree archived at S2f; allowed only in the [provenance] files.
ARCHIVED = re.compile(r"(?:glm_tpu|scripts|tests)[./]greenfield")
ANCHOR = re.compile(r"^Path\(__file__\)(?:\.resolve\(\))?\.parents$")


def module_name(path: str) -> str:
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def dotted(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return dotted(node.value) + "." + node.attr.value
    raise TypeError(f"not a dotted name: {node!r}")


def expression(name: str) -> cst.BaseExpression:
    return cst.parse_expression(name)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


@dataclass
class MoveMap:
    files: dict[str, str]
    removed: dict[str, str]
    packages: dict[str, str]
    include: list[str]
    exclude: list[str]
    suffixes: list[str]
    baseline: dict[str, list[str]]
    provenance: dict[str, str]
    pyproject: dict[str, list[str]]
    paths: dict[str, str] = field(default_factory=dict)  # moved file: old path -> new path
    dirs: dict[str, str] = field(default_factory=dict)  # moved directory: old -> new
    modules: dict[str, str] = field(default_factory=dict)  # moved module: old dotted -> new dotted
    packages_removed: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path = MAP) -> MoveMap:
        value = tomllib.loads(path.read_text())
        rewrite = value["rewrite"]
        self = cls(
            files=value["files"],
            removed=value["removed"],
            packages=value["packages"],
            include=rewrite["include"],
            exclude=rewrite["exclude"],
            suffixes=rewrite["suffixes"],
            baseline=value.get("baseline_references", {}),
            provenance=value.get("provenance", {}),
            pyproject=value.get("pyproject", {}),
        )
        for old, new in self.files.items():
            if new == "":
                if old not in self.removed:
                    raise SystemExit(f"move_map: {old} has no destination and no [removed] reason")
                self.packages_removed.add(module_name(old))
            elif new != old:
                self.paths[old] = new
                if old.endswith(".py") != new.endswith(".py"):
                    raise SystemExit(f"move_map: {old} -> {new} changes the file kind")
                if old.endswith(".py"):
                    self.modules[module_name(old)] = module_name(new)
        destinations = [new for new in self.files.values() if new]
        if len(destinations) != len(set(destinations)):
            raise SystemExit("move_map: two files share a destination")
        # a directory whose every file moves under one new directory, keeping its relative path
        for old in self.paths:
            parts = old.split("/")
            for depth in range(1, len(parts)):
                directory = "/".join(parts[:depth])
                bases = set()
                for member in (p for p in self.files if p.startswith(directory + "/")):
                    relative = member[len(directory) :]
                    target = self.paths.get(member, "")
                    bases.add(target[: -len(relative)] if target.endswith(relative) else None)
                if len(bases) == 1 and None not in bases:
                    self.dirs[directory] = bases.pop()
        self._dotted = self._alternation(self.modules, r"(?<![\w.])(", r")(?![\w])")
        names = {**self.paths, **self.dirs}
        self._path = self._alternation(names, r"(?<![\w.-])(", r")(?![\w-]|\.[\w])")
        return self

    @staticmethod
    def _alternation(names: dict[str, str], head: str, tail: str) -> re.Pattern[str] | None:
        if not names:
            return None
        return re.compile(head + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + tail)

    def old_path(self, current: str) -> str:
        inverse = {new: old for old, new in self.paths.items()}
        return inverse.get(current, current)

    def rewrite_text(self, text: str, keep: frozenset[str]) -> str:
        def dotted_name(match: re.Match[str]) -> str:
            name = match.group(1)
            return name if name in keep else self.modules[name]

        def path(match: re.Match[str]) -> str:
            name = match.group(1)
            return name if name in keep else self.paths.get(name) or self.dirs[name]

        if self._dotted is not None:
            text = self._dotted.sub(dotted_name, text)
        if self._path is not None:
            text = self._path.sub(path, text)
        return text

    def in_scope(self, path: str) -> bool:
        if not any(fnmatch.fnmatch(path, pattern) for pattern in self.include):
            return False
        if any(fnmatch.fnmatch(path, pattern) for pattern in self.exclude):
            return False
        suffix = Path(path).suffix
        return suffix in self.suffixes or (suffix == "" and Path(path).name in self.suffixes)


class PythonRewriter(cst.CSTTransformer):
    """The Python rewrite of one file (``old``: its path before the move, ``new``: after)."""

    def __init__(self, moves: MoveMap, old: str, new: str) -> None:
        super().__init__()
        self.moves, self.old, self.new = moves, old, new
        self.module = module_name(old)
        self.package = self.module.split(".") if old.endswith("__init__.py") else self.module.split(".")[:-1]
        self.moved = old != new
        self.keep = frozenset(moves.baseline.get(new, ()))
        self.notes: list[str] = []

    # ------------------------------------------------------------------ imports
    def _absolute(self, node: cst.ImportFrom) -> str:
        module = dotted(node.module) if node.module is not None else ""
        level = len(node.relative)
        if level == 0:
            return module
        base = self.package[: len(self.package) - (level - 1)]
        return ".".join([*base, *([module] if module else [])])

    def _from(self, node: cst.ImportFrom) -> list[cst.BaseSmallStatement]:
        target = self._absolute(node)
        modules = self.moves.modules
        if target in modules:
            return [node.with_changes(module=expression(modules[target]), relative=[])]
        if not isinstance(node.names, cst.ImportStar) and any(
            f"{target}.{a.name.value}" in modules for a in node.names
        ):
            groups: dict[str, list[str]] = {}
            for alias in node.names:
                name = alias.name.value
                local = alias.asname.name.value if alias.asname is not None else name
                old = f"{target}.{name}"
                if old in modules:
                    parent, base = modules[old].rsplit(".", 1)
                else:
                    if target in self.moves.packages_removed:
                        raise SystemExit(f"{self.new}: {name} is imported from the dissolved package {target}")
                    parent, base = target, name
                groups.setdefault(parent, []).append(base if base == local else f"{base} as {local}")
            return [
                cst.parse_statement(f"from {parent} import {', '.join(names)}").body[0]
                for parent, names in groups.items()
            ]
        if node.relative and self.moved:
            if target in self.moves.packages_removed:
                raise SystemExit(f"{self.new}: a relative import names the dissolved package {target}")
            return [node.with_changes(module=expression(target), relative=[])]
        return [node]

    def _import(self, node: cst.Import) -> cst.Import:
        names = []
        for alias in node.names:
            name = dotted(alias.name)
            if name in self.moves.modules:
                if alias.asname is None and "." in name:
                    raise SystemExit(f"{self.new}: `import {name}` binds its top package; import it with `as`")
                alias = alias.with_changes(name=expression(self.moves.modules[name]))
            names.append(alias)
        return node.with_changes(names=names)

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.BaseStatement | cst.FlattenSentinel:
        body: list[cst.BaseSmallStatement] = []
        for statement in updated.body:
            if isinstance(statement, cst.ImportFrom):
                body.extend(self._from(statement))
            elif isinstance(statement, cst.Import):
                body.append(self._import(statement))
            else:
                body.append(statement)
        if len(updated.body) == 1 and len(body) > 1:
            lines = [
                updated.with_changes(
                    body=[statement.with_changes(semicolon=cst.MaybeSentinel.DEFAULT)],
                    leading_lines=updated.leading_lines if index == 0 else (),
                )
                for index, statement in enumerate(body)
            ]
            return cst.FlattenSentinel(lines)
        return updated.with_changes(body=body)

    # ------------------------------------------------------------------ strings and comments
    def leave_SimpleString(self, original: cst.SimpleString, updated: cst.SimpleString) -> cst.SimpleString:
        value = updated.value
        prefix = len(value) - len(value.lstrip("rRbBuU"))
        quote = 3 if value[prefix : prefix + 3] in ('"""', "'''") else 1
        inner = value[prefix + quote : len(value) - quote]
        text = self.moves.rewrite_text(inner, self.keep)
        return (
            updated
            if text == inner
            else updated.with_changes(value=value[: prefix + quote] + text + value[len(value) - quote :])
        )

    def leave_FormattedStringText(
        self, original: cst.FormattedStringText, updated: cst.FormattedStringText
    ) -> cst.FormattedStringText:
        text = self.moves.rewrite_text(updated.value, self.keep)
        return updated if text == updated.value else updated.with_changes(value=text)

    def leave_Comment(self, original: cst.Comment, updated: cst.Comment) -> cst.Comment:
        text = self.moves.rewrite_text(updated.value, self.keep)
        return updated if text == updated.value else updated.with_changes(value=text)

    # ------------------------------------------------------------------ path joins
    def leave_BinaryOperation(self, original: cst.BinaryOperation, updated: cst.BinaryOperation) -> cst.BaseExpression:
        if not isinstance(updated.operator, cst.Divide):
            return updated
        operands: list[cst.BaseExpression] = [updated.right]
        node = updated.left
        while isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Divide) and not node.lpar:
            operands.insert(0, node.right)
            node = node.left
        operands.insert(0, node)
        literal = [isinstance(o, cst.SimpleString) and o.prefix == "" for o in operands]
        end = len(operands)
        start = end
        while start > 0 and literal[start - 1]:
            start -= 1
        if end - start < 2:
            return updated
        parts = [operands[i].evaluated_value for i in range(start, end)]  # type: ignore[union-attr]
        for first in range(len(parts) - 1):
            joined = "/".join(parts[first:])
            new = self.moves.paths.get(joined) or self.moves.dirs.get(joined)
            if new is None or joined in self.keep:
                continue
            quote = operands[start].value[0]  # type: ignore[union-attr]
            replacement = [cst.SimpleString(f"{quote}{part}{quote}") for part in new.split("/")]
            items = operands[: start + first] + replacement
            expr = items[0]
            for item in items[1:]:
                expr = cst.BinaryOperation(left=expr, operator=updated.operator, right=item)
            return expr.with_changes(lpar=updated.lpar, rpar=updated.rpar)
        return updated

    # ------------------------------------------------------------------ repository-root anchors
    def leave_Subscript(self, original: cst.Subscript, updated: cst.Subscript) -> cst.BaseExpression:
        if not self.moved or not isinstance(updated.value, cst.Attribute):
            return updated
        if not ANCHOR.match(cst.Module([]).code_for_node(updated.value)):
            return updated
        if len(updated.slice) != 1 or not isinstance(updated.slice[0].slice, cst.Index):
            return updated
        index = updated.slice[0].slice.value
        if not isinstance(index, cst.Integer):
            return updated
        old_root, new_root = self.old.count("/"), self.new.count("/")
        if int(index.value) == new_root:  # already the root of the new location (an applied move)
            return updated
        if int(index.value) != old_root:
            self.notes.append(
                f"{self.new}: Path(__file__) parents[{index.value}] is not the repository root anchor; review by hand"
            )
            return updated
        if old_root == new_root:
            return updated
        element = updated.slice[0].with_changes(slice=cst.Index(cst.Integer(str(new_root))))
        return updated.with_changes(slice=[element])


def rewrite_pyproject(moves: MoveMap, text: str) -> str:
    def replace(pattern: str, value: str, text: str) -> str:
        new, count = re.subn(pattern, lambda m: m.group(1) + value, text, flags=re.M)
        if count != 1:
            raise SystemExit(f"pyproject.toml: {pattern!r} matched {count} times")
        return new

    def toml_list(values: list[str]) -> str:
        return "[" + ", ".join(f'"{v}"' for v in values) + "]"

    if "package_data" in moves.pyproject:
        text = replace(r"^(glm_tpu = )\[.*\]$", toml_list(moves.pyproject["package_data"]), text)
    if "license_files" in moves.pyproject:
        text = replace(r"^(license-files = )\[.*\]$", toml_list(moves.pyproject["license_files"]), text)
    if "black_include" in moves.pyproject and re.search(r"^\[tool\.black\]$", text, flags=re.M):
        include = moves.pyproject["black_include"]
        for path in sorted(dissolved_later()):  # a file S4.1 dissolved leaves the boundary
            stem = path[len("glm_tpu/") : -3] if path.startswith("glm_tpu/") else None
            if stem is not None:
                include = include.replace(f"|{stem}|", "|")
        text = replace(r"^(include = )'.*'$", "'" + include + "'", text)
    return text


def rewrite_file(moves: MoveMap, path: str, notes: list[str]) -> str | None:
    """The new content of the tracked file ``path`` (current location), or None if unchanged."""
    source = (REPO / path).read_text()
    old = moves.old_path(path)
    if path.endswith(".py"):
        rewriter = PythonRewriter(moves, old, path)
        text = cst.parse_module(source).visit(rewriter).code
        notes.extend(rewriter.notes)
    else:
        text = moves.rewrite_text(source, frozenset(moves.baseline.get(path, ())))
        if path == "pyproject.toml":
            text = rewrite_pyproject(moves, text)
    return None if text == source else text


def tracked() -> list[str]:
    return git("ls-files").split()


def dissolved_later() -> set[str]:
    """Modules a later table dissolved (the S4.1 interim modules, the S4.3 merged tests): S3
    destinations that no longer exist."""
    out: set[str] = set()
    for table in (SYMBOLS, RENAMES, MERGES):
        if table.is_file():
            out |= set(tomllib.loads(table.read_text()).get("dissolve", {}).get("modules", []))
    return out


def plan_moves(moves: MoveMap) -> tuple[list[tuple[str, str]], list[str], list[str], list[str]]:
    """(pending git mv, pending git rm, missing package initializers, problems)."""
    pending, removals, problems = [], [], []
    later = dissolved_later()
    for old, new in moves.paths.items():
        src, dst = REPO / old, REPO / new
        if src.exists() and not dst.exists():
            pending.append((old, new))
        elif not (dst.exists() and not src.exists()) and not (new in later and not src.exists()):
            problems.append(f"{old} -> {new}: expected exactly one of them to exist")
    for old in moves.removed:
        if (REPO / old).exists():
            removals.append(old)
    missing = [p for p in moves.packages if not (REPO / p).exists()]
    return pending, removals, missing, problems


def add_closure_entries(moved: list[tuple[str, str]]) -> None:
    entries = {
        module_name(new): module_name(old) for old, new in moved if old.endswith(".py") and not old.startswith("tests/")
    }
    if not entries:
        return
    lines = CLOSURE_MAP.read_text().splitlines(keepends=True)
    start = lines.index("[modules]\n") + 1
    while start < len(lines) and lines[start].startswith("#"):
        start += 1
    present = set(tomllib.loads(CLOSURE_MAP.read_text()).get("modules", {}))
    added = [f'"{new}" = "{old}"\n' for new, old in sorted(entries.items()) if new not in present]
    if added:
        header = "# S3 (tools/migration/move_map.toml): every moved module, new name = recorded name\n"
        lines[start:start] = [header, *added]
        CLOSURE_MAP.write_text("".join(lines))


def stale_references(moves: MoveMap, files: list[str]) -> tuple[list[str], list[str]]:
    """Remaining references into the dissolved packages: (failures, Markdown reports)."""
    failures, reports = [], []
    for path in files:
        if not moves.in_scope(path) or not (REPO / path).is_file():
            continue
        keep = set(moves.baseline.get(path, ()))
        for number, line in enumerate((REPO / path).read_text().splitlines(), 1):
            spans = [(m.start(), m.end()) for k in keep for m in re.finditer(re.escape(k), line)]
            if path in moves.provenance:
                spans += [(m.start(), m.end()) for m in ARCHIVED.finditer(line)]
            if any(not any(a <= match.start() < b for a, b in spans) for match in STALE.finditer(line)):
                row = f"{path}:{number}: {line.strip()[:160]}"
                (reports if path.endswith(".md") else failures).append(row)
    return failures, reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", nargs="?", const=MAP, type=Path, metavar="TABLE")
    mode.add_argument("--check", nargs="*", type=Path, metavar="TABLE")
    parser.add_argument("--map", type=Path, default=None, help="S3 table (default move_map.toml)")
    args = parser.parse_args(argv)
    if args.apply is not None:
        table = _table(args.apply if args.map is None else args.map)
        return run_table(table, check=False)
    newest_s4 = [t for t in (HELPERS,) if t.is_file()]  # S4.4 on (module docstring)
    tables = [_table(t) for t in args.check] or [t for t in (args.map or MAP, *newest_s4) if t.is_file()]
    return max(run_table(table, check=True) for table in tables)


def _table(path: Path) -> Path:
    return path if path.is_absolute() or path.is_file() else Path(__file__).with_name(path.name)


def run_table(table: Path, *, check: bool) -> int:
    kind = tomllib.loads(table.read_text())["stage"].get("kind", "files")
    if kind == "files":
        return run_files(MoveMap.load(table), check=check)
    import symbols  # the S4 engine, next to this file

    plan = symbols.Plan.load(table)
    if check:
        return check_symbols(symbols, plan)
    result = symbols.run(plan)
    symbols.write(result)
    added = symbols.write_closure_entries(plan.stage, f"tools/migration/{table.name}", result.closure)
    for row in result.notes:
        print("review:", row)
    print(
        f"apply {plan.stage}: {len(result.texts)} files written, {len(result.removed)} modules removed, "
        f"{added} closure_map.toml entries"
    )
    return 0


def check_symbols(symbols: Any, plan: Any) -> int:
    problems = symbols.check(plan)
    resolver = symbols.Resolver({}, {}, {module_name(p) for p in plan.dissolve})
    reports = []
    for path in tracked():
        if "/_s3_" in path:
            problems.append(f"{path}: an interim _s3_ module is tracked")
        if not (REPO / path).is_file() or path.startswith(symbols.NEVER) or path in symbols.TEXT_SKIP:
            continue
        if not (
            (path.endswith((".py", ".toml", ".md")) and path.startswith(("glm_tpu/", "tests/", "tools/", "docs/")))
            or path in ("pyproject.toml",)
        ):
            continue
        for number, line in symbols.stale_mentions((REPO / path).read_text(), resolver, plan.allow_stale.get(path, ())):
            row = f"{path}:{number}: {line[:160]}"
            (reports if path.endswith(".md") else problems).append(row)
    for row in reports:
        print("note (Markdown, S6):", row)
    for row in problems:
        print("problem:", row)
    print(f"check {plan.stage}: {len(problems)} problem(s)")
    return 1 if problems else 0


def run_files(moves: MoveMap, *, check: bool) -> int:
    pending, removals, missing, problems = plan_moves(moves)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    notes: list[str] = []
    if check:
        changes = [f"git mv {o} {n}" for o, n in pending] + [f"git rm {p}" for p in removals]
        changes += [f"create {p}" for p in missing]
        if not pending:
            for path in tracked():
                if moves.in_scope(path) and (REPO / path).is_file() and rewrite_file(moves, path, notes) is not None:
                    changes.append(f"rewrite {path}")
        failures, reports = stale_references(moves, tracked())
        for row in reports:
            print("note (Markdown, S6):", row)
        for row in failures:
            print("stale reference:", row)
        for row in changes:
            print("pending:", row)
        for row in notes:
            print("review:", row)
        print(f"check S3: {len(changes)} pending change(s), {len(failures)} stale reference(s)")
        return 1 if changes or failures else 0
    for old, new in pending:
        (REPO / new).parent.mkdir(parents=True, exist_ok=True)
        git("mv", old, new)
    for old in removals:
        git("rm", "-q", old)
    for path in missing:
        (REPO / path).parent.mkdir(parents=True, exist_ok=True)
        (REPO / path).write_text(f'"""{moves.packages[path]}"""\n')
        git("add", path)
    rewritten = 0
    for path in tracked():
        if moves.in_scope(path) and (REPO / path).is_file():
            text = rewrite_file(moves, path, notes)
            if text is not None:
                (REPO / path).write_text(text)
                rewritten += 1
    add_closure_entries(pending)
    for row in notes:
        print("review:", row)
    print(
        f"apply: {len(pending)} moved, {len(removals)} removed, {len(missing)} package initializers, "
        f"{rewritten} files rewritten"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
