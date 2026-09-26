"""S5 class splits (DESIGN 11.3 S5; the S5 plan's P9): prove from git that a split changed only what it declares.

    python tools/migration/split_check.py [TABLE]   # default splits.toml, next to this file

A work unit that splits a class by method, folds a function into a method or changes the callers of the
split (S5 WU-E1 first) cannot be applied by the S4 engine (``symbols.py`` moves and renames whole top-level
definitions); nor can a move whose old home keeps importing the moved name for its importers (S5 WU-S3), which
the engine would rewrite, and whose ``--check`` compares no whole file and ignores every import, function-local
ones included. Its table (``[stage] kind = "splits"``; ``base``, the commit it was written against; ``files``,
the production files the unit touches; ``created``, those of them that do not exist at ``base``, compared as
empty modules there) declares how every definition of those files at ``base`` relates to the current tree, and
this checker compares them:

* ``[definitions]``: ``"path:qualname" = {from = "path:qualname", rewrite = {...}, diff = [...]}``, a
  definition now at the key that came from ``from`` (default: the key itself, an edit in place). ``rewrite``
  is applied to the origin's AST first: ``"x"`` renames a name (also a parameter or an imported name),
  ``"x.y"`` replaces an exact attribute chain, ``"x.*"`` every one-level attribute of the name ``x``
  (``"self.*" = "self.runner.*"``; an exact key wins); every rule must be used. ``diff`` lists what then
  still differs, as ``difflib.unified_diff`` prints it without context (``-`` origin line, ``+`` current
  line, in order) over the ``ast.unparse`` text; nothing else may differ.
* A definition the table does not name is compared verbatim with the same qualname at ``base``, or, for a
  method of a class whose row names another class as its origin (a renamed class), with the method of the
  same name there.
* ``[added]`` / ``[removed]``: ``"path:qualname" = "<token>: reason"``, a definition with no origin, and a
  ``base`` definition that is nobody's origin. Every ``base`` definition of the files must be an origin
  (declared or implicit) or removed; every current definition must have an origin or be added.
* ``[added_source]``: ``"path:qualname" = [lines]``, the exact ``ast.unparse`` lines of each ``[added]``
  definition (a class as its shell, like every comparison here), so added code is compared too; required for
  every ``[added]`` row (from S5 WU-S1/S2).
* ``[docstrings]``: ``"path" = "<token>: reason"``, a module docstring the unit rewrote (not compared).
* ``[imports."path"]``: ``added`` / ``removed``, the top-level import bindings the file gains and loses
  (``module:name``, ``module`` for ``import module``, `` as alias`` appended); the bindings it keeps must
  keep their order (from S5 WU-C1; where a gained binding goes is not checked).
* ``[statements."path"]``: ``diff``, what differs between the file's other top-level statements (neither
  definitions nor imports, e.g. constants or an ``if TYPE_CHECKING:`` block) at ``base`` and now, as ``diff``
  of ``[definitions]`` prints it over their ``ast.unparse`` text in order (from S5 WU-S3); without an entry
  they must be unchanged, in order.

Definitions are top-level functions and classes and the functions and classes directly in a top-level
class; nested functions, lambdas and comprehensions are part of the definition that contains them. A class
is compared as its shell (name, bases, keywords, decorators, docstring and the statements that are not
definitions). Comments and formatting are not compared (``ast.unparse``), nor is the place of the imports
among the other top-level statements. Standard library only; exit 1 with every problem listed. S7 deletes
``tools/migration/``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TABLE = Path(__file__).with_name("splits.toml")
REASON = re.compile(r"(?:WU-[A-Z][A-Za-z]*|H[1-9][0-9]?): \S")
DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
ROW_KEYS = {"from", "rewrite", "diff"}


def source_at(base: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{base}:{path}"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def exists_at(base: str, path: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{base}:{path}"], cwd=REPO, capture_output=True).returncode == 0


def definitions(path: str, tree: ast.Module) -> dict[str, ast.AST]:
    """``path:qualname`` -> node for the top-level definitions and those directly in a top-level class."""
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, DEFINITION):
            out[f"{path}:{node.name}"] = node
            if isinstance(node, ast.ClassDef):
                for inner in node.body:
                    if isinstance(inner, DEFINITION):
                        out[f"{path}:{node.name}.{inner.name}"] = inner
    return out


def shell(node: ast.AST, name: str) -> str:
    """The unparsed definition under ``name``; a class without its methods and nested classes."""
    node = copy.deepcopy(node)
    node.name = name
    if isinstance(node, ast.ClassDef):
        node.body = [s for s in node.body if not isinstance(s, DEFINITION)] or [ast.Pass()]
    return ast.unparse(node)


def module_parts(tree: ast.Module) -> tuple[str | None, list[str], list[str]]:
    """The module docstring, its top-level import bindings (in order) and its other top-level statements."""
    docstring = ast.get_docstring(tree, clean=False)
    body = tree.body[1:] if docstring is not None else tree.body
    imports: list[str] = []
    other: list[str] = []
    for node in body:
        if isinstance(node, ast.Import):
            imports += [a.name + (f" as {a.asname}" if a.asname else "") for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports += [f"{module}:{a.name}" + (f" as {a.asname}" if a.asname else "") for a in node.names]
        elif not isinstance(node, DEFINITION):
            other.append(ast.unparse(node))
    return docstring, imports, other


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return None if inner is None else f"{inner}.{node.attr}"
    return None


class Rewrite(ast.NodeTransformer):
    """The declared ``rewrite`` rules, applied to an origin definition."""

    def __init__(self, rules: dict[str, str]) -> None:
        self.names: dict[str, str] = {}
        self.chains: dict[str, str] = {}
        self.wild: dict[str, str] = {}
        for key, value in rules.items():
            if key.endswith(".*"):
                if not value.endswith(".*"):
                    raise ValueError(f"rewrite {key!r}: a wildcard maps to a wildcard")
                self.wild[key[:-2]] = value[:-2]
            elif "." in key:
                self.chains[key] = value
            else:
                self.names[key] = value
        self.used: set[str] = set()

    @staticmethod
    def _expression(text: str, ctx: ast.expr_context) -> ast.expr:
        node = ast.parse(text, mode="eval").body
        for inner in ast.walk(node):
            if hasattr(inner, "ctx"):
                inner.ctx = ast.Load()
        if hasattr(node, "ctx"):
            node.ctx = ctx
        return node

    def _identifier(self, key: str) -> str:
        value = self.names[key]
        if not value.isidentifier():
            raise ValueError(f"rewrite {key!r}: a parameter or imported name maps to a name, not {value!r}")
        self.used.add(key)
        return value

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        chain = _dotted(node)
        if chain in self.chains:
            self.used.add(chain)
            return ast.copy_location(self._expression(self.chains[chain], node.ctx), node)
        if isinstance(node.value, ast.Name) and node.value.id in self.wild:
            self.used.add(node.value.id + ".*")
            value = self._expression(self.wild[node.value.id], ast.Load())
            return ast.copy_location(ast.Attribute(value=value, attr=node.attr, ctx=node.ctx), node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.names:
            self.used.add(node.id)
            return ast.copy_location(self._expression(self.names[node.id], node.ctx), node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.names:
            node.arg = self._identifier(node.arg)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> ast.AST:
        if node.asname is None and node.name in self.names:
            node.name = self._identifier(node.name)
        elif node.asname in self.names:
            node.asname = self._identifier(node.asname)
        return node

    def unused(self) -> list[str]:
        keys = set(self.names) | set(self.chains) | {k + ".*" for k in self.wild}
        return sorted(keys - self.used)


def changed_lines(old: str, new: str) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def check(table_path: Path) -> tuple[list[str], list[str]]:
    table = tomllib.loads(table_path.read_text())
    stage = table.get("stage", {})
    problems: list[str] = []
    report: list[str] = []
    if stage.get("kind") != "splits":
        return [f"{table_path.name}: [stage] kind must be 'splits'"], report
    unknown = sorted(
        set(table) - {"stage", "definitions", "added", "added_source", "removed", "docstrings", "imports", "statements"}
    )
    if unknown:
        problems.append(f"unknown tables {unknown}")
    base, files, created = stage["base"], list(stage["files"]), list(stage.get("created", []))
    problems += [f"[stage] created {p}: not a listed file" for p in created if p not in files]
    problems += [f"[stage] created {p}: exists at {base}" for p in created if exists_at(base, p)]
    rows: dict[str, dict[str, Any]] = table.get("definitions", {})
    added: dict[str, str] = table.get("added", {})
    added_source: dict[str, list[str]] = table.get("added_source", {})
    removed: dict[str, str] = table.get("removed", {})
    docstrings: dict[str, str] = table.get("docstrings", {})
    imports: dict[str, dict[str, list[str]]] = table.get("imports", {})
    statements: dict[str, dict[str, list[str]]] = table.get("statements", {})
    for section, entries in (("added", added), ("removed", removed), ("docstrings", docstrings)):
        problems += [
            f"[{section}] {k}: reason must start with a token" for k, v in entries.items() if not REASON.match(v)
        ]
    problems += [f"[docstrings] / [imports] {p}: not a listed file" for p in (*docstrings, *imports) if p not in files]
    problems += [f"[imports] {p}: only added and removed" for p, v in imports.items() if set(v) - {"added", "removed"}]
    problems += [f"[statements] {p}: not a listed file" for p in statements if p not in files]
    problems += [
        f"[statements] {p}: only a non-empty diff" for p, v in statements.items() if set(v) != {"diff"} or not v["diff"]
    ]
    problems += [f"[definitions] {k}: only from, rewrite and diff" for k, v in rows.items() if set(v) - ROW_KEYS]

    problems += [f"[added] {k}: no [added_source] entry" for k in sorted(set(added) - set(added_source))]
    problems += [f"[added_source] {k}: not an [added] row" for k in sorted(set(added_source) - set(added))]
    old_trees = {path: ast.parse("" if path in created else source_at(base, path)) for path in files}
    new_trees = {path: ast.parse((REPO / path).read_text()) for path in files}
    old_defs = {k: v for path, tree in old_trees.items() for k, v in definitions(path, tree).items()}
    new_defs = {k: v for path, tree in new_trees.items() for k, v in definitions(path, tree).items()}

    # Module shells: docstring, import bindings, other statements.
    for path in files:
        old_doc, old_order, old_other = module_parts(old_trees[path])
        new_doc, new_order, new_other = module_parts(new_trees[path])
        old_imports, new_imports = set(old_order), set(new_order)
        if path in docstrings:
            report.append(f"{path}: module docstring rewritten ({docstrings[path]})")
        elif old_doc != new_doc:
            problems.append(f"{path}: module docstring differs (not declared in [docstrings])")
        declared = imports.get(path, {})
        gained, lost = sorted(new_imports - old_imports), sorted(old_imports - new_imports)
        if gained != sorted(declared.get("added", [])) or lost != sorted(declared.get("removed", [])):
            problems.append(f"{path}: import bindings +{gained} -{lost} differ from the declared ones")
        elif gained or lost:
            report.append(f"{path}: imports +{gained} -{lost}")
        kept_old = [binding for binding in old_order if binding in new_imports]
        kept_new = [binding for binding in new_order if binding in old_imports]
        if kept_old != kept_new:
            problems.append(f"{path}: the import bindings it keeps changed order")
            problems += [f"    {line}" for line in changed_lines("\n".join(kept_old), "\n".join(kept_new))]
        actual = changed_lines("\n".join(old_other), "\n".join(new_other))
        expected = list(statements.get(path, {}).get("diff", []))
        if actual != expected or (old_other != new_other and not actual):
            problems.append(f"{path}: top-level statements other than definitions and imports differ")
            problems += [f"    {line}" for line in actual]
            if expected:
                problems.append("    declared:")
                problems += [f"    {line}" for line in expected]
        elif actual:
            report.append(f"{path}: other top-level statements {actual}")

    # Origins: declared rows, then the implicit ones (same qualname, or the same method of a renamed class).
    origins: dict[str, str] = {}
    for key, spec in rows.items():
        if key not in new_defs:
            problems.append(f"[definitions] {key}: no such definition now")
            continue
        origins[key] = spec.get("from", key)
    renamed_classes = {
        key: origin
        for key, origin in origins.items()
        if isinstance(new_defs[key], ast.ClassDef) and origin != key and origin in old_defs
    }
    for key in new_defs:
        if key in origins or key in added:
            continue
        path, _, qualname = key.partition(":")
        parent, _, name = qualname.rpartition(".")
        if parent and f"{path}:{parent}" in renamed_classes:
            origins[key] = f"{renamed_classes[f'{path}:{parent}']}.{name}"
        else:
            origins[key] = key
    for key in sorted(added):
        if key not in new_defs:
            problems.append(f"[added] {key}: no such definition now")
        elif key in rows:
            problems.append(f"[added] {key}: also a [definitions] row")
        elif key in added_source:
            name = key.rpartition(":")[2].rpartition(".")[2]
            actual = shell(new_defs[key], name).splitlines()
            if actual != list(added_source[key]):
                problems.append(f"[added_source] {key}: differs from the definition now")
                problems += [f"    {line}" for line in changed_lines("\n".join(added_source[key]), "\n".join(actual))]
    used: dict[str, str] = {}
    for key, origin in sorted(origins.items()):
        if origin not in old_defs:
            problems.append(f"{key}: origin {origin} does not exist at {base} (declare it in [added])")
            continue
        if origin in used:
            problems.append(f"{key}: origin {origin} is also the origin of {used[origin]}")
        used[origin] = key
        spec = rows.get(key, {})
        rewrite = Rewrite(spec.get("rewrite", {}))
        rewritten = rewrite.visit(copy.deepcopy(old_defs[origin]))
        problems += [f"{key}: rewrite rule {rule!r} is not used" for rule in rewrite.unused()]
        name = key.rpartition(":")[2].rpartition(".")[2]
        actual = changed_lines(shell(rewritten, name), shell(new_defs[key], name))
        expected = list(spec.get("diff", []))
        if actual != expected:
            problems.append(f"{key} (from {origin}): undeclared difference")
            problems += [f"    {line}" for line in actual]
            if expected:
                problems.append("    declared:")
                problems += [f"    {line}" for line in expected]
        rules = len(spec.get("rewrite", {}))
        kind = f"rewrite {rules} rules, diff {len(expected)} lines" if rules or expected else "verbatim"
        report.append(f"{key} <- {origin}: {kind}" if origin != key or spec else f"{key}: verbatim")
    for key in sorted(set(old_defs) - set(used)):
        if key in removed:
            report.append(f"{key}: removed ({removed[key]})")
        else:
            problems.append(f"{key}: at {base} but nobody's origin (declare it in [removed])")
    problems += [
        f"[removed] {k}: not a definition at {base}, or still an origin"
        for k in removed
        if k in used or k not in old_defs
    ]
    report += [f"{key}: added, source compared ({reason})" for key, reason in sorted(added.items())]
    return problems, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("table", nargs="?", default=str(TABLE), help="the split table (default: splits.toml)")
    args = parser.parse_args(argv)
    path = Path(args.table)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).with_name(args.table)
    problems, report = check(path)
    print("\n".join(report + problems))
    print(f"{path.name}: {sum(not line.startswith('    ') for line in problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
