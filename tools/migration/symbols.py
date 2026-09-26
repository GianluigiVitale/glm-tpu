"""S4 of the public restructure (DESIGN 10.3 S4): move definitions, then apply the public names.

The engine behind ``restructure.py --apply symbol_moves.toml`` (S4.1), ``restructure.py --apply
renames.toml`` (S4.2), ``restructure.py --apply test_merges.toml`` (S4.3, moves between test
modules), ``restructure.py --apply helper_names.toml`` (S4.4: renames plus ``[named_scopes]``, the
``jax.named_scope`` names, see F), ``restructure.py --apply test_moves.toml`` (S5 A2) and
``restructure.py --apply owed_tests.toml`` (S5 D), moves between test modules, and ``restructure.py
--apply work_units.toml`` (the S5 work units: renames plus ``[messages]``, see G). The tables describe
one kind of change -- a top-level definition ``(module, name)`` becomes ``(module', name')`` -- and
share one reference pass:

A. *move* (``[moves]``, S4.1): every listed top-level definition is cut from its module and pasted
   verbatim (source text with its leading comments) into its destination: before the first
   destination statement that needs it when the module is imported, else at the end. The
   destination receives the imports its new definitions need (the source's own bindings of their
   free names, resolved to their final homes) and ``from __future__ import annotations`` when the
   source had it. Identical definitions moved to one destination (the same private helper of two
   sources) are kept once. A ``[dissolve]`` module must be left with imports only; it is removed.
B. *rename* (``[renames]``, S4.1 collisions and S4.2): the definition and every reference to it in
   its own module (libcst scope analysis; a local that shadows it is left alone).
C. *references*, in every Python file under ``glm_tpu/``, ``tests/`` and ``tools/`` (never
   ``tools/migration/`` or recorded data): ``from X import a [as n]`` imports the final home of
   ``a`` (dropped when that is the importing module itself); ``alias.a`` through a module alias of
   ``X`` becomes the final home's ``alias'.b`` (a bare name in the home module itself; an alias of
   the new module is reused, or added next to the old import); bare references follow a renamed
   binding; a module whose definitions left it imports what its remaining code still uses;
   aliases of a dissolved module that nothing uses any more are dropped. Patch targets
   (``setattr``-style calls with a string attribute, assignments through a module alias) follow
   renames but are never guessed for a move: they keep naming the module they named (its own
   binding) and are reported for review.
D. ``module.name`` and ``module:qualname`` spellings in strings and comments of those files and in
   Markdown and TOML follow the map; mentions of a dissolved module that no longer resolve are
   reported by ``--check``.
E. imports that the changed modules no longer use (and no other file imports from them) are dropped.
F. ``[named_scopes]`` (S4.4): the name literal of every ``<x>.named_scope(...)`` call under ``glm_tpu/``
   whose text is a key (an f-string spelled as its source template, ``"a/{axis}_b"``) is rewritten
   to the value; ``--check`` requires the scope names of each module to be its base-commit scope
   names mapped through the table, and compares renamed definitions with their base originals
   with the table applied to the originals' scope names.
G. ``[messages."<path>"]`` (S5 WU-R): reviewed rewrites of whole string literals in one file,
   ``"old text" = "new text"`` (e.g. an error message losing a campaign label). ``--apply`` rewrites every
   plain one-line double-quoted literal of the file whose value is a key (each key must occur; any other
   spelling of a key -- an f-string part, an implicit concatenation, escapes -- is refused); ``--check``
   compares renamed or moved definitions with their base originals with the file's messages applied to
   the originals' string constants, compares each listed file whole with its base-commit text through
   the table's renames and messages twice -- by AST (imports aside) and, from S5 WU-Docs, by token (every
   token but the layout ones, so imports and comments included; a string literal compared by its value, an
   f-string part by its text; formatting aside) -- and reports a key left in the file. A key must be one
   whole literal or f-string part (the token comparison maps no implicit concatenation).

Formatting, ruff and docstrings are S4.4 and the work units. Standard library plus libcst.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
import copy
from collections import defaultdict
from dataclasses import dataclass, field
import io
import json
import re
import subprocess
import symtable
import textwrap
import tokenize
import tomllib
from pathlib import Path
from typing import Any

import libcst as cst
from libcst.metadata import (
    BuiltinAssignment,
    GlobalScope,
    ImportAssignment,
    MetadataWrapper,
    ParentNodeProvider,
    ScopeProvider,
)

REPO = Path(__file__).resolve().parents[2]
PY_ROOTS = ("glm_tpu/", "tests/", "tools/")
NEVER = ("tools/migration/", "tests/golden/data/")
TEXT_SKIP = (
    "AGENTS.md",
    "HANDOFF.md",
    "goal.md",
    "docs/release/STATUS.md",
    "tests/reference/VALIDATION.md",
    "tools/equivalence/closure_map.toml",
)
TEXT_SUFFIXES = (".toml", ".md")
PATCH_CALLS = (
    "setattr",
    "getattr",
    "hasattr",
    "delattr",
    "monkeypatch.setattr",
    "monkeypatch.delattr",
    "mock.patch.object",
    "patch.object",
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def tracked() -> list[str]:
    return git("ls-files").split()


def module_name(path: str) -> str:
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def in_python_scope(path: str) -> bool:
    return path.endswith(".py") and path.startswith(PY_ROOTS) and not path.startswith(NEVER)


def in_text_scope(path: str) -> bool:
    return (
        path.endswith(TEXT_SUFFIXES)
        and not path.startswith(NEVER)
        and path not in TEXT_SKIP
        and not path.startswith("glm_tpu/models/glm_moe_dsa/hf_config/")
    )


def dotted(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return dotted(node.value) + "." + node.attr.value
    raise TypeError(node)


def is_repo_module(name: str) -> bool:
    path = REPO / name.replace(".", "/")
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


# ============================================================================== module index
@dataclass(frozen=True)
class Imported:
    """A module-level name bound by an import: ``attr`` of ``module`` (None: the module object)."""

    module: str
    attr: str | None


def defined_names(statement: cst.BaseStatement) -> list[str]:
    """Names a top-level statement defines: def, class, a plain assignment to names."""
    if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
        return [statement.name.value]
    if isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1:
        small = statement.body[0]
        if isinstance(small, cst.Assign) and all(isinstance(t.target, cst.Name) for t in small.targets):
            return [t.target.value for t in small.targets]
        if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name):
            return [small.target.value]
    return []


def module_imports(tree: cst.Module) -> Iterable[cst.Import | cst.ImportFrom]:
    """Import statements of the module scope (also inside top-level ``if``/``try`` blocks)."""

    def walk(statements: Iterable[cst.CSTNode]) -> Iterable[cst.Import | cst.ImportFrom]:
        for statement in statements:
            if isinstance(statement, cst.SimpleStatementLine):
                yield from (s for s in statement.body if isinstance(s, (cst.Import, cst.ImportFrom)))
            elif isinstance(statement, (cst.If, cst.Try, cst.Else, cst.ExceptHandler, cst.Finally)):
                yield from walk(statement.body.body)
                for extra in ("orelse", "handlers", "finalbody"):
                    value = getattr(statement, extra, None)
                    if value is not None:
                        yield from walk(value if isinstance(value, (list, tuple)) else [value])

    yield from walk(tree.body)


def absolute_from(module: str, package: bool, node: cst.ImportFrom) -> str:
    base = module.split(".") if package else module.split(".")[:-1]
    target = dotted(node.module) if node.module is not None else ""
    level = len(node.relative)
    if level == 0:
        return target
    base = base[: len(base) - (level - 1)]
    return ".".join([*base, *([target] if target else [])])


def is_future(node: cst.CSTNode) -> bool:
    return isinstance(node, cst.ImportFrom) and isinstance(node.module, cst.Name) and node.module.value == "__future__"


@dataclass
class Module:
    path: str
    source: str

    def __post_init__(self) -> None:
        self.tree = cst.parse_module(self.source)
        self.name = module_name(self.path)
        self.package = self.path.endswith("__init__.py")
        self.defs: dict[str, cst.BaseStatement] = {}
        for statement in self.tree.body:
            for name in defined_names(statement):
                self.defs.setdefault(name, statement)
        self.imports: dict[str, Imported] = {}
        self.import_text: dict[str, str] = {}  # bound name -> a statement that binds it the same way
        self.future = False
        for node in module_imports(self.tree):
            if is_future(node):
                self.future = True
                continue
            for bound, target, text in import_bindings(self.name, self.package, node):
                self.imports.setdefault(bound, target)
                self.import_text.setdefault(bound, text)


def import_bindings(module: str, package: bool, node: cst.Import | cst.ImportFrom) -> list[tuple[str, Imported, str]]:
    out = []
    if isinstance(node, cst.Import):
        for alias in node.names:
            name = dotted(alias.name)
            if alias.asname is not None:
                bound = alias.asname.name.value
                out.append((bound, Imported(name, None), f"import {name} as {bound}"))
            else:
                out.append((name.split(".")[0], Imported(name.split(".")[0], None), f"import {name}"))
        return out
    if isinstance(node.names, cst.ImportStar):
        return out
    source = absolute_from(module, package, node)
    for alias in node.names:
        name = alias.name.value
        bound = alias.asname.name.value if alias.asname is not None else name
        text = f"from {source} import {name}" + (f" as {bound}" if bound != name else "")
        if is_repo_module(f"{source}.{name}"):
            out.append((bound, Imported(f"{source}.{name}", None), text))
        else:
            out.append((bound, Imported(source, name), text))
    return out


def binding_statement(target: Imported, bound: str) -> str:
    if target.attr is None:
        parent, _, leaf = target.module.rpartition(".")
        if parent:
            return f"from {parent} import {leaf}" + (f" as {bound}" if bound != leaf else "")
        return f"import {target.module}" + (f" as {bound}" if bound != target.module else "")
    return f"from {target.module} import {target.attr}" + (f" as {bound}" if bound != target.attr else "")


# ============================================================================== the table
@dataclass
class Plan:
    stage: str
    kind: str  # "symbols" (S4.1, S4.3) or "renames" (S4.2, S4.4)
    base: str
    path: Path
    moves: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)  # (src path, name) -> (dest path, name)
    renames: dict[tuple[str, str], str] = field(default_factory=dict)  # (path, name) -> new name
    create: dict[str, str] = field(default_factory=dict)  # new module path -> docstring
    dissolve: list[str] = field(default_factory=list)
    added: dict[str, str] = field(default_factory=dict)  # closure_map [added] entries
    allow_stale: dict[str, list[str]] = field(default_factory=dict)  # path -> spellings kept on purpose
    scopes: dict[str, str] = field(default_factory=dict)  # named-scope name (template) -> new name (S4.4)
    messages: dict[str, dict[str, str]] = field(default_factory=dict)  # path -> {old literal: new} (S5 WU-R)

    @classmethod
    def load(cls, path: Path) -> Plan:
        value = tomllib.loads(path.read_text())
        stage = value["stage"]
        plan = cls(
            stage=stage["name"],
            kind=stage["kind"],
            base=stage["base"],
            path=path,
            create=value.get("create", {}),
            dissolve=value.get("dissolve", {}).get("modules", []),
            added=value.get("closure_added", {}),
            allow_stale=value.get("baseline_references", {}),
            scopes=value.get("named_scopes", {}),
            messages=value.get("messages", {}),
        )
        for path_, table in plan.messages.items():
            if not (
                isinstance(table, dict)
                and table
                and all(isinstance(k, str) and isinstance(v, str) and k and k != v for k, v in table.items())
            ):
                raise PlanError(f"{path.name}: [messages.{path_!r}] must map old literals to different new ones")
        for source, table in value.get("moves", {}).items():
            for name, spec in table.items():
                dest, new = (spec, name) if isinstance(spec, str) else (spec["to"], spec.get("as", name))
                plan.moves[(source, name)] = (dest, new)
        for key, new in value.get("renames", {}).items():
            path_, _, name = key.partition(":")
            plan.renames[(path_, name)] = new
        return plan

    def mapping(self) -> dict[tuple[str, str], tuple[str, str]]:
        """``(module, name) -> (module', name')`` (module names) of every changed definition."""
        out = {(module_name(s), n): (module_name(d), new) for (s, n), (d, new) in self.moves.items()}
        for (path, name), new in self.renames.items():
            key = (module_name(path), name)
            if key in out:
                raise SystemExit(f"{self.path.name}: {path}:{name} is both moved and renamed")
            out[key] = (module_name(path), new)
        return out


# ============================================================================== resolution
class Resolver:
    """Final homes of names, answered from the index of the tree before the run."""

    def __init__(
        self, modules: dict[str, Module], mapping: dict[tuple[str, str], tuple[str, str]], dissolved: set[str]
    ):
        self.modules, self.mapping, self.dissolved = modules, mapping, dissolved
        self.changed_modules = {m for m, _ in mapping} | dissolved

    def canonical(self, module: str, name: str, seen: frozenset = frozenset()) -> tuple[str, str]:
        """Where the object ``module.name`` is defined after the run (import chains followed)."""
        key = (module, name)
        if key in self.mapping:
            return self.mapping[key]
        info = self.modules.get(module)
        if info is None or key in seen or name in info.defs:
            return key
        target = info.imports.get(name)
        if target is None or target.attr is None:
            return key
        return self.canonical(target.module, target.attr, seen | {key})

    def home(self, module: str, name: str) -> tuple[str, str]:
        """The spelling of ``module.name`` after the run: unchanged unless the definition moved or was
        renamed, or ``module`` is dissolved (then the object's canonical home)."""
        key = (module, name)
        if key in self.mapping:
            return self.mapping[key]
        if module in self.dissolved:
            return self.canonical(module, name)
        return key


# ============================================================================== free names
def statement_references(code: str, future: bool) -> set[str]:
    """Names a top-level statement reads from its module scope: function and class bodies through
    symtable, annotations, decorators, defaults, bases and module-level values through ast."""
    source = ("from __future__ import annotations\n" if future else "") + code
    tree = ast.parse(source)
    names: set[str] = set()

    def walk(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if symbol.is_referenced() and (symbol.is_global() or table.get_type() == "module"):
                names.add(symbol.get_name())
        for child in table.get_children():
            walk(child)

    walk(symtable.symtable(source, "<moved>", "exec"))
    for node in ast.walk(tree):
        for annotation in annotations_of(node):
            for inner in ast.walk(annotation):
                if isinstance(inner, ast.Name):
                    names.add(inner.id)
                elif isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    names.update(re.findall(r"[A-Za-z_]\w*", inner.value))
    return names


def annotations_of(node: ast.AST) -> list[ast.AST]:
    out: list[ast.AST] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        out += [
            a.annotation
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
            if a is not None and a.annotation is not None
        ]
        if node.returns is not None:
            out.append(node.returns)
    elif isinstance(node, ast.AnnAssign):
        out.append(node.annotation)
    return out


def import_time_names(code: str, annotations: bool = False) -> set[str]:
    """Names a top-level statement evaluates when its module is imported (decorators, defaults,
    bases, class-body statements, the value of a module-level statement; not function bodies;
    ``annotations``: also annotations, for a module without ``from __future__ import annotations``)."""
    tree = ast.parse(textwrap.dedent(code))
    exprs: list[ast.AST] = []
    if annotations:
        for node in ast.walk(tree):
            exprs += annotations_of(node)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            exprs += [*node.decorator_list, *node.args.defaults, *[d for d in node.args.kw_defaults if d]]
        elif isinstance(node, ast.ClassDef):
            exprs += [*node.decorator_list, *node.bases, *[k.value for k in node.keywords]]
            for inner in node.body:
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    exprs += [*inner.decorator_list, *inner.args.defaults, *[d for d in inner.args.kw_defaults if d]]
                elif isinstance(inner, ast.AnnAssign):
                    exprs += [inner.value] if inner.value is not None else []
                elif not isinstance(inner, (ast.Expr, ast.Pass, ast.ClassDef)):
                    exprs.append(inner)
        else:
            exprs.append(node)
    return {inner.id for expr in exprs for inner in ast.walk(expr) if isinstance(inner, ast.Name)}


# ============================================================================== A. the move
class PlanError(SystemExit):
    pass


def _leading(statement: cst.BaseStatement, blank: int) -> cst.BaseStatement:
    """The statement with its comment lines kept and ``blank`` empty lines above them."""
    lines = list(statement.leading_lines)
    while lines and lines[0].comment is None:
        lines.pop(0)
    return statement.with_changes(leading_lines=[cst.EmptyLine()] * blank + lines)


def _is_docstring(statement: cst.BaseStatement) -> bool:
    return (
        isinstance(statement, cst.SimpleStatementLine)
        and len(statement.body) == 1
        and isinstance(statement.body[0], cst.Expr)
        and isinstance(statement.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    )


def _is_import_line(statement: cst.BaseStatement) -> bool:
    return isinstance(statement, cst.SimpleStatementLine) and all(
        isinstance(s, (cst.Import, cst.ImportFrom)) for s in statement.body
    )


def _import_block_end(body: list[cst.BaseStatement]) -> int:
    """Index after the module's leading docstring and import block."""
    index = 1 if body and _is_docstring(body[0]) else 0
    end = index
    for position in range(index, len(body)):
        if _is_import_line(body[position]):
            end = position + 1
        elif isinstance(body[position], cst.If) and all(_is_import_line(s) for s in body[position].body.body):
            end = position + 1  # e.g. ``if TYPE_CHECKING:`` imports
        else:
            break
    return end


def _ast_key(code: str) -> str:
    return ast.dump(ast.parse(textwrap.dedent(code)))


def move_definitions(plan: Plan, modules: dict[str, Module], resolver: Resolver) -> dict[str, str]:
    by_path = {m.path: m for m in modules.values()}
    listed: dict[str, set[str]] = defaultdict(set)
    blocks: dict[str, list[tuple[Module, cst.BaseStatement, str, str]]] = defaultdict(list)
    for (source, name), (dest, new) in plan.moves.items():
        listed[source].add(name)
        module = by_path.get(source)
        if module is None or name not in module.defs:
            done = by_path.get(dest)
            if done is not None and new in done.defs:
                continue  # applied before (idempotent)
            raise PlanError(f"{source}: no top-level definition {name}")
        blocks[dest].append((module, module.defs[name], name, new))
    for path in plan.dissolve:
        module = by_path.get(path)
        if module is not None and set(module.defs) - listed[path]:
            raise PlanError(f"{path} is dissolved but keeps {sorted(set(module.defs) - listed[path])}")
    texts: dict[str, str] = {}
    cut: dict[str, set[int]] = defaultdict(set)
    for items in blocks.values():
        for module, statement, _, _ in items:
            cut[module.path].add(id(statement))
    for path, ids in cut.items():
        module = by_path[path]
        body = [s for s in module.tree.body if id(s) not in ids]
        texts[path] = module.tree.with_changes(body=body).code
    for dest, items in blocks.items():
        texts[dest] = _paste(plan, dest, items, by_path, resolver, texts.get(dest))
    return texts


def _paste(
    plan: Plan,
    dest: str,
    items: list[tuple[Module, cst.BaseStatement, str, str]],
    by_path: dict[str, Module],
    resolver: Resolver,
    current: str | None,
) -> str:
    dest_name = module_name(dest)
    if current is not None:
        target = Module(dest, current)
    elif dest in by_path:
        target = by_path[dest]
    elif dest in plan.create:
        doc = plan.create[dest].strip("\n")
        target = Module(dest, '"""' + doc + ('\n"""\n' if "\n" in doc else '"""\n'))
    else:
        raise PlanError(f"{dest}: a new module needs a [create] docstring")
    provided = {new for _, _, _, new in items}
    statements: list[cst.BaseStatement] = []
    kept: dict[str, str] = {}
    needed: dict[str, str] = {}
    future = target.future
    for module, statement, name, new in items:
        code = module.tree.code_for_node(_leading(statement, 0)).strip("\n")
        if new in target.defs:
            raise PlanError(f"{dest} already defines {new} ({module.path}:{name} moves there)")
        key = _ast_key(code)
        if new in kept:
            if kept[new] != key:
                raise PlanError(f"{dest}: two different definitions named {new}")
            continue
        kept[new] = key
        if new != name:
            statement = _renamed_statement(statement, name, new)
        statements.append(statement if _is_constant(statement) else _leading(statement, 2))
        future = future or module.future
        for free in sorted(statement_references(module.tree.code_for_node(statement), module.future)):
            if free == name or free in provided:
                continue
            text = _binding_text(free, module, dest_name, resolver)
            if text is None:
                continue
            if free in target.defs:
                raise PlanError(f"{dest}: {free} (needed by {module.path}:{name}) is defined differently there")
            if free in target.imports:
                if _same_object(target.imports[free], _object_of(free, module, resolver), resolver):
                    continue
                raise PlanError(f"{dest}: {free} is `{target.import_text[free]}`; {module.path}:{name} needs `{text}`")
            if needed.get(free, text) != text:
                raise PlanError(f"{dest}: {free} needed as `{needed[free]}` and as `{text}`")
            needed[free] = text
    return _assemble(target, statements, needed, future and not target.future, future or target.future)


def _object_of(free: str, module: Module, resolver: Resolver) -> tuple[str, str | None]:
    """The object ``free`` names in ``module``: ``(module, None)`` for a module object."""
    if free in module.defs:
        return resolver.canonical(module.name, free)
    target = module.imports[free]
    if target.attr is None:
        return (target.module, None)
    return resolver.canonical(target.module, target.attr)


def _same_object(present: Imported, wanted: tuple[str, str | None], resolver: Resolver) -> bool:
    if present.attr is None:
        return wanted == (present.module, None)
    return resolver.canonical(present.module, present.attr) == wanted


def _binding_text(free: str, module: Module, dest: str, resolver: Resolver) -> str | None:
    """The statement that binds ``free`` in the destination as ``module`` bound it (None: a builtin,
    or defined in the destination)."""
    if free in module.defs:
        home = resolver.home(module.name, free)
    elif free in module.imports:
        target = module.imports[free]
        if target.attr is None or target.module not in resolver.modules:
            return module.import_text[free]
        home = resolver.home(target.module, target.attr)
        if home[0] == dest:
            return None
        if home == (target.module, target.attr):
            return module.import_text[free]
    else:
        return None
    if home[0] == dest:
        return None
    return f"from {home[0]} import {home[1]}" + (f" as {free}" if home[1] != free else "")


def _renamed_statement(statement: cst.BaseStatement, old: str, new: str) -> cst.BaseStatement:
    if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
        return statement.with_changes(name=cst.Name(new))
    small = statement.body[0]
    if isinstance(small, cst.Assign):
        targets = [t.with_changes(target=cst.Name(new)) if t.target.value == old else t for t in small.targets]
        return statement.with_changes(body=[small.with_changes(targets=targets)])
    return statement.with_changes(body=[small.with_changes(target=cst.Name(new))])


def _is_constant(statement: cst.BaseStatement) -> bool:
    return isinstance(statement, cst.SimpleStatementLine) and bool(defined_names(statement))


def _assemble(
    target: Module,
    statements: list[cst.BaseStatement],
    needed: dict[str, str],
    add_future: bool,
    lazy_annotations: bool,
) -> str:
    """The destination with its new imports (merged into its statements of the same module, else
    in their group of its import block) and its new definitions: constants after the module's own
    leading constants, functions and classes at the end -- each before the first module statement
    that needs it at import time, and after the ones it needs."""
    tree = target.tree
    body = list(tree.body)
    end = _import_block_end(body)
    body, new_imports = _merge_imports(body, end, list(needed.values()))
    doc = 1 if body and _is_docstring(body[0]) else 0
    head = _place_imports(body[:end], new_imports, doc)
    rest = body[end:]
    code = tree.code_for_node
    lazy = not lazy_annotations
    trailer = []
    if rest and isinstance(rest[-1], cst.If) and "__name__" in code(rest[-1].test):
        trailer = [rest.pop()]
    constants = [s for s in statements if _is_constant(s)]
    definitions = [s for s in statements if not _is_constant(s)]
    for group in (constants, definitions):
        if not group:
            continue
        names = {n for s in group for n in defined_names(s)}
        needs = set().union(*(import_time_names(code(s), lazy) for s in group))
        first_user = next((i for i, s in enumerate(rest) if import_time_names(code(s), lazy) & names), len(rest))
        last_provider = max((i + 1 for i, s in enumerate(rest) if set(defined_names(s)) & needs), default=0)
        if group is constants:
            position = 0
            while position < len(rest) and _is_constant(rest[position]):
                position += 1
            position = max(position, last_provider)
        else:
            position = len(rest)
        position = min(position, first_user)
        if position < last_provider:
            raise PlanError(
                f"{target.path}: new definitions {sorted(names)} are needed at import time before "
                "module statements they need"
            )
        block = list(group)
        if group is constants:
            block[0] = _leading(block[0], 1 if position and _is_constant(rest[position - 1]) else 2)
        if position < len(rest):
            rest[position] = _leading(rest[position], 2)
        rest[position:position] = block
    if rest:
        rest[0] = _leading(rest[0], 2)
    body = [*head, *rest, *trailer]
    if add_future:
        body.insert(
            doc,
            cst.parse_statement("from __future__ import annotations\n").with_changes(
                leading_lines=[cst.EmptyLine()] if doc else []
            ),
        )
        if doc + 1 < len(body):
            body[doc + 1] = _leading(body[doc + 1], 1)
    return _squeeze(tree.with_changes(body=body).code)


def _place_imports(head: list[cst.BaseStatement], new: list[cst.BaseStatement], doc: int) -> list[cst.BaseStatement]:
    """New import statements, each after the last import of its group (stdlib, third-party,
    first-party) in the module's import block."""
    head = list(head)
    for statement in new:
        group = _statement_group(statement)
        position = None
        for index in range(len(head) - 1, doc - 1, -1):
            other = _statement_group(head[index])
            if other is not None and other <= group:
                position = index + 1
                break
        if position is None:
            position = doc
            while position < len(head) and _statement_group(head[position]) == -1:
                position += 1  # after ``from __future__``
        before = _statement_group(head[position - 1]) if position > doc else None
        blank = 1 if position > 0 and (before is None or before != group) else 0
        statement = statement.with_changes(leading_lines=[cst.EmptyLine()] * blank)
        if position < len(head) and _statement_group(head[position]) not in (None, group):
            head[position] = head[position].with_changes(
                leading_lines=[
                    cst.EmptyLine(),
                    *[line for line in head[position].leading_lines if line.comment is not None],
                ]
            )
        head.insert(position, statement)
    return head


def _statement_group(statement: cst.BaseStatement) -> int | None:
    if not _is_import_line(statement):
        return None
    small = statement.body[0]
    if is_future(small):
        return -1
    if isinstance(small, cst.ImportFrom):
        if small.relative:
            return 2
        return _import_group(dotted(small.module))
    return _import_group(dotted(small.names[0].name))


_FIRST_PARTY = ("glm_tpu", "tests", "tools")


def _import_group(module: str) -> int:
    import sys

    top = module.split(".")[0]
    return 2 if top in _FIRST_PARTY else 0 if top in sys.stdlib_module_names or top == "__future__" else 1


def _merge_imports(
    body: list[cst.BaseStatement], end: int, texts: list[str]
) -> tuple[list[cst.BaseStatement], list[cst.BaseStatement]]:
    """Names imported ``from`` a module the module already imports from join that statement; the
    rest become new statements (stdlib, third-party, first-party; sorted by module)."""
    body = list(body)
    pending: dict[tuple[str, str], list[str]] = {}
    for text in texts:
        match = re.fullmatch(r"from (\S+) import (\S+)(?: as (\S+))?", text)
        if match:
            module, name, asname = match.groups()
            spelled = name + (f" as {asname}" if asname else "")
            for index in range(end):
                statement = body[index]
                if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
                    continue
                small = statement.body[0]
                if (
                    isinstance(small, cst.ImportFrom)
                    and not small.relative
                    and small.module is not None
                    and not isinstance(small.names, cst.ImportStar)
                    and dotted(small.module) == module
                ):
                    names = [*small.names, cst.parse_statement(f"from x import {spelled}\n").body[0].names[0]]
                    body[index] = statement.with_changes(body=[small.with_changes(names=_commas(names))])
                    break
            else:
                pending.setdefault(("from", module), []).append(spelled)
        else:
            pending.setdefault(("import", text), [])
    statements = []

    def order(item: tuple[tuple[str, str], list[str]]) -> tuple[int, str]:
        (kind, module), _ = item
        name = module.split()[1] if kind == "import" else module
        return (_import_group(name), name)

    group = None
    for (kind, module), names in sorted(pending.items(), key=order):
        text = module if kind == "import" else f"from {module} import {', '.join(sorted(set(names)))}"
        statement = cst.parse_statement(text + "\n")
        if group is not None and order(((kind, module), names))[0] != group:
            statement = statement.with_changes(leading_lines=[cst.EmptyLine()])
        group = order(((kind, module), names))[0]
        statements.append(statement)
    return body, statements


def merge_duplicate_imports(text: str, modules: set[str]) -> str:
    """Module-level ``from X import a`` statements of one module ``X`` of ``modules`` (the run's
    destinations) joined into the first (statements with ``as`` aliases stay as they are)."""
    tree = cst.parse_module(text)
    body = list(tree.body)
    first: dict[str, int] = {}
    drop: set[int] = set()
    for index, statement in enumerate(body):
        if not (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1):
            continue
        small = statement.body[0]
        if (
            not isinstance(small, cst.ImportFrom)
            or small.relative
            or small.module is None
            or is_future(small)
            or isinstance(small.names, cst.ImportStar)
            or statement.trailing_whitespace.comment is not None
        ):
            continue
        module = dotted(small.module)
        if module not in modules or any(a.asname is not None for a in small.names):
            continue
        if module not in first:
            first[module] = index
            continue
        target = body[first[module]]
        names = list(target.body[0].names)
        present = {(a.name.value, a.asname.name.value if a.asname else None) for a in names}
        for alias in small.names:
            if (alias.name.value, alias.asname.name.value if alias.asname else None) not in present:
                names.append(alias)
        target_small = target.body[0]
        many = len(names) > 1
        body[first[module]] = target.with_changes(
            body=[
                target_small.with_changes(
                    names=_commas(names),
                    lpar=target_small.lpar if many else None,
                    rpar=target_small.rpar if many else None,
                )
            ]
        )
        drop.add(index)
    if not drop:
        return text
    return _squeeze(tree.with_changes(body=[s for i, s in enumerate(body) if i not in drop]).code)


def _squeeze(text: str) -> str:
    return re.sub(r"\n{4,}", "\n\n\n", text)


# ============================================================================== B. renames
class _RenameDefinitions(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self, renames: dict[str, str]):
        super().__init__()
        self.renames = renames
        self.nodes: dict[int, str] = {}

    def analyze(self, wrapper: MetadataWrapper) -> None:
        scopes = wrapper.resolve(ScopeProvider)
        glob = next(s for s in set(scopes.values()) if isinstance(s, GlobalScope))
        for old, new in self.renames.items():
            found = False
            for assignment in glob[old]:
                if isinstance(assignment, (ImportAssignment, BuiltinAssignment)):
                    continue
                found = True
                node = assignment.node
                self.nodes[id(node.name if isinstance(node, (cst.FunctionDef, cst.ClassDef)) else node)] = new
                for access in assignment.references:
                    if isinstance(access.node, cst.Name):
                        self.nodes[id(access.node)] = new
            if not found:
                raise PlanError(f"no top-level definition {old} to rename")

    def leave_Name(self, original: cst.Name, updated: cst.Name) -> cst.Name:
        new = self.nodes.get(id(original))
        return updated.with_changes(value=new) if new else updated


def rename_definitions(text: str, renames: dict[str, str]) -> str:
    wrapper = MetadataWrapper(cst.parse_module(text))
    transformer = _RenameDefinitions(renames)
    transformer.analyze(wrapper)
    return wrapper.visit(transformer).code


# ============================================================================== C. references
class ReferenceRewriter(cst.CSTTransformer):
    """The reference pass over one Python file (module docstring, step C)."""

    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(self, path: str, resolver: Resolver, moved_away: dict[str, tuple[str, str]], notes: list[str]):
        super().__init__()
        self.path, self.resolver, self.notes = path, resolver, notes
        self.module = module_name(path)
        self.package = path.endswith("__init__.py")
        self.moved_away = moved_away
        self.names: dict[int, str] = {}  # id(Name) -> new identifier
        self.attributes: dict[int, tuple[str | None, str]] = {}  # id(Attribute) -> (alias | None: bare, attr)
        self.strings: dict[int, str] = {}  # id(SimpleString) -> new literal
        self.from_edits: dict[int, tuple[str, str, str]] = {}  # id(ImportAlias) -> (module, name, bound)
        self.drop: set[int] = set()  # id(ImportAlias) removed
        self.after_line: dict[int, list[str]] = defaultdict(list)  # id(statement line) -> imports added after it
        self.module_imports: list[str] = []  # added after the module's import block
        self.aliases: dict[tuple[int, str], str] = {}  # (id(scope), module) -> alias bound in that scope
        self._alias_module: dict[int, str] = {}  # id(assignment) -> the module its alias names

    # ------------------------------------------------------------------ analysis
    def analyze(self, wrapper: MetadataWrapper) -> None:
        scopes = [s for s in set(wrapper.resolve(ScopeProvider).values()) if s is not None]
        self.parents = wrapper.resolve(ParentNodeProvider)
        self.used = {
            n.value for n in _walk(wrapper.module) if isinstance(n, cst.Name) and not _is_label(n, self.parents)
        }
        imports = []
        self.bare: dict[tuple[str, str], str] = {}  # (module, name) imported at module level -> bound name
        for scope in scopes:
            for assignment in scope.assignments:
                if isinstance(assignment, ImportAssignment) and isinstance(scope, GlobalScope):
                    alias = _alias_node(assignment.node, assignment.name)
                    target = self._target(assignment.node, alias) if alias is not None else None
                    if target is not None and target.attr is not None:
                        self.bare.setdefault((target.module, target.attr), assignment.name)
        for scope in scopes:
            for assignment in scope.assignments:
                if isinstance(assignment, ImportAssignment):
                    alias = _alias_node(assignment.node, assignment.name)
                    if alias is None:
                        continue
                    target = self._target(assignment.node, alias)
                    if target is None:
                        continue
                    if target.attr is None:
                        self.aliases[(id(scope), target.module)] = assignment.name
                    imports.append((scope, assignment, alias, target))
        for scope, assignment, alias, target in imports:
            if target.attr is None:
                self._module_alias(scope, assignment, alias, target.module)
            else:
                self._name_import(assignment, alias, target)
        if self.moved_away:
            for scope in scopes:
                self._moved_away_references(scope)

    def _target(self, node: cst.Import | cst.ImportFrom, alias: cst.ImportAlias) -> Imported | None:
        if isinstance(node, cst.ImportFrom):
            if isinstance(node.names, cst.ImportStar) or is_future(node):
                return None
            source = absolute_from(self.module, self.package, node)
            name = alias.name.value
            if f"{source}.{name}" in self.resolver.modules or is_repo_module(f"{source}.{name}"):
                return Imported(f"{source}.{name}", None)
            return Imported(source, name)
        if alias.asname is None:
            # ``import a.b`` binds the top package ``a`` (its uses spell the full path)
            return Imported(dotted(alias.name).split(".")[0], None)
        return Imported(dotted(alias.name), None)

    def _name_import(self, assignment: ImportAssignment, alias: cst.ImportAlias, target: Imported) -> None:
        if target.module not in self.resolver.modules:
            return
        home = self.resolver.home(target.module, target.attr)
        if home == (target.module, target.attr):
            return
        bound = assignment.name
        if home[0] == self.module:
            self.drop.add(id(alias))
            if home[1] != bound:
                self._rename_accesses(assignment, home[1])
            return
        new_bound = bound if alias.asname is not None else home[1]
        if new_bound != bound:
            self._rename_accesses(assignment, new_bound)
        self.from_edits[id(alias)] = (home[0], home[1], new_bound)

    def _rename_accesses(self, assignment: Any, new: str) -> None:
        for access in assignment.references:
            if isinstance(access.node, cst.Name):
                self.names[id(access.node)] = new

    def _module_alias(self, scope: Any, assignment: ImportAssignment, alias: cst.ImportAlias, module: str) -> None:
        self._alias_module[id(assignment)] = module
        if module not in self.resolver.changed_modules:
            return
        dissolved = module in self.resolver.dissolved
        still_used = rewritten = False
        for access in assignment.references:
            node = access.node
            parent = self.parents.get(node) if isinstance(node, cst.Name) else None
            if isinstance(parent, cst.Attribute) and parent.value is node:
                attr = parent.attr.value
                home = self.resolver.home(module, attr)
                bound = self.resolver.modules[module].imports.get(attr)
                if dissolved and bound is not None and bound.attr is None:
                    # a module object the dissolved module had imported (``batched.jax``): that module
                    self.attributes[id(parent)] = ("\0module", self._alias(scope, assignment, bound.module))
                    rewritten = True
                    continue
                if home == (module, attr):
                    still_used = True
                elif home[0] == module:  # a rename: the attribute follows
                    self.attributes[id(parent)] = (assignment.name, home[1])
                    still_used = True
                elif _is_store(parent, self.parents) and not dissolved:
                    # a patch of the surviving module's own binding (it re-imports what it uses)
                    self.notes.append(
                        f"{self.path}: assignment to {assignment.name}.{attr}: {attr} moved to "
                        f"{home[0]} ({home[1]}); the target is kept: review"
                    )
                    still_used = True
                else:
                    if _is_store(parent, self.parents):
                        self.notes.append(
                            f"{self.path}: assignment to {assignment.name}.{attr} of the dissolved "
                            f"{module} now targets {home[0]}: review"
                        )
                    if home[0] == self.module:
                        self.attributes[id(parent)] = (None, home[1])
                    else:
                        self.attributes[id(parent)] = self._reference(
                            scope, assignment, home, bare=not _is_store(parent, self.parents)
                        )
                    rewritten = True
                continue
            patch = self._patch_attribute(node)
            if patch is None:
                still_used = True
                if dissolved:
                    self.notes.append(
                        f"{self.path}: {assignment.name} (the dissolved {module}) is used as a value: rewrite by hand"
                    )
                continue
            attr, string = patch
            home = self.resolver.home(module, attr)
            if home == (module, attr):
                still_used = True
            elif home[0] == module:  # a rename: the string follows
                self.strings[id(string)] = _string_like(string, home[1])
                still_used = True
            elif dissolved:  # the dissolved module's successor is the target
                self.names[id(node)] = self._alias(scope, assignment, home[0])
                if home[1] != attr:
                    self.strings[id(string)] = _string_like(string, home[1])
                self.notes.append(
                    f"{self.path}: patch target ({assignment.name}, {attr!r}) of the dissolved "
                    f"{module} now names {home[0]}:{home[1]}: review"
                )
                rewritten = True
            else:
                self.notes.append(
                    f"{self.path}: patch target ({assignment.name}, {attr!r}): {attr} moved "
                    f"to {home[0]} ({home[1]}); kept: review"
                )
                still_used = True
        if not still_used and (dissolved or rewritten):
            self.drop.add(id(alias))

    def _patch_attribute(self, node: cst.CSTNode) -> tuple[str, cst.SimpleString] | None:
        """``(attribute, its string node)`` when ``node`` is the object of a setattr-style call."""
        arg = self.parents.get(node)
        call = self.parents.get(arg) if isinstance(arg, cst.Arg) else None
        if not isinstance(call, cst.Call) or len(call.args) < 2 or call.args[0] is not arg:
            return None
        try:
            name = dotted(call.func)
        except TypeError:
            return None
        second = call.args[1].value
        if name in PATCH_CALLS and isinstance(second, cst.SimpleString):
            return second.evaluated_value, second
        return None

    def _reference(
        self, scope: Any, assignment: ImportAssignment, home: tuple[str, str], bare: bool = True
    ) -> tuple[str | None, str]:
        """How this file spells ``home`` where ``assignment`` bound the old alias: a name it
        already imports; an alias of the module it already binds; a new alias named after the
        module; else the name itself, imported next to the old import."""
        module, name = home
        if bare and home in self.bare:
            return (None, self.bare[home])
        for candidate in _scope_chain(scope):
            if (id(candidate), module) in self.aliases:
                return (self.aliases[(id(candidate), module)], name)
        leaf = module.rsplit(".", 1)[-1]
        old_module = self._alias_module.get(id(assignment))
        reusable = leaf == assignment.name and old_module in self.resolver.dissolved
        if leaf in self.used and not reusable and bare and name not in self.used:
            self.used.add(name)
            self.bare[home] = name
            self.after_line[id(self._line(assignment.node))].append(f"from {module} import {name}")
            return (None, name)
        return (self._alias(scope, assignment, module), name)

    def _alias(self, scope: Any, assignment: ImportAssignment, module: str) -> str:
        """A name bound to ``module`` where ``assignment`` binds the old alias: reused, or a new
        import right after the old import statement."""
        for candidate in _scope_chain(scope):
            if (id(candidate), module) in self.aliases:
                return self.aliases[(id(candidate), module)]
        parts = module.split(".")
        name = None
        old_module = self._alias_module.get(id(assignment))
        claimed = {v for (sid, mod), v in self.aliases.items() if sid == id(scope) and mod != old_module}
        candidates = [parts[-1]]
        if old_module in self.resolver.dissolved and assignment.name not in claimed:
            candidates.append(assignment.name)  # the dissolved module's alias may name its successor
        candidates += ["_".join(parts[-2:]), "_".join(parts[1:]), "_".join(parts) + "_module"]
        for candidate in [c for c in candidates if c]:
            if name is None and (
                candidate not in self.used
                or (
                    candidate == assignment.name == parts[-1]
                    and old_module in self.resolver.dissolved
                    and candidate not in claimed
                )
            ):
                name = candidate
        if name is None:
            raise PlanError(f"{self.path}: no free alias for {module}")
        self.used.add(name)
        self.aliases[(id(scope), module)] = name
        text = binding_statement(Imported(module, None), name)
        self.after_line[id(self._line(assignment.node))].append(text)
        return name

    def _line(self, node: cst.CSTNode) -> cst.CSTNode:
        while not isinstance(node, cst.SimpleStatementLine):
            node = self.parents[node]
        return node

    def _moved_away_references(self, scope: Any) -> None:
        for access in scope.accesses:
            node = access.node
            if not isinstance(node, cst.Name) or node.value not in self.moved_away:
                continue
            if any(not isinstance(r, BuiltinAssignment) for r in access.referents):
                continue
            home = self.moved_away[node.value]
            if home[0] == self.module:
                continue
            text = f"from {home[0]} import {home[1]}" + (f" as {node.value}" if home[1] != node.value else "")
            if text not in self.module_imports:
                self.module_imports.append(text)

    # ------------------------------------------------------------------ transformation
    def leave_Name(self, original: cst.Name, updated: cst.Name) -> cst.Name:
        new = self.names.get(id(original))
        return updated.with_changes(value=new) if new else updated

    def leave_SimpleString(self, original: cst.SimpleString, updated: cst.SimpleString) -> cst.SimpleString:
        new = self.strings.get(id(original))
        return updated.with_changes(value=new) if new else updated

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
        rewrite = self.attributes.get(id(original))
        if rewrite is None:
            return updated
        alias, attr = rewrite
        if alias is None or alias == "\0module":
            return cst.Name(attr, lpar=updated.lpar, rpar=updated.rpar)
        return updated.with_changes(value=cst.Name(alias), attr=cst.Name(attr))

    def leave_SimpleStatementLine(self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine) -> Any:
        body: list[cst.BaseSmallStatement] = []
        changed = False
        for before, small in zip(original.body, updated.body, strict=False):
            if isinstance(before, cst.ImportFrom) and not isinstance(before.names, cst.ImportStar):
                statements = self._import_from(before, small)
                if statements is not None:
                    body += statements
                    changed = True
                    continue
            elif isinstance(before, cst.Import):
                names = [new for old, new in zip(before.names, small.names, strict=False) if id(old) not in self.drop]
                if len(names) != len(small.names):
                    changed = True
                    if names:
                        body.append(small.with_changes(names=_commas(names)))
                    continue
            body.append(small)
        extra = [cst.parse_statement(t + "\n") for t in self.after_line.get(id(original), [])]
        if not changed and not extra:
            return updated
        lines = [cst.SimpleStatementLine(body=[s.with_changes(semicolon=cst.MaybeSentinel.DEFAULT)]) for s in body]
        lines += extra
        if not lines:
            return _vanish(updated)
        lines[0] = lines[0].with_changes(
            leading_lines=updated.leading_lines,
            trailing_whitespace=updated.trailing_whitespace if len(lines) == 1 else cst.TrailingWhitespace(),
        )
        return cst.FlattenSentinel(lines)

    def _import_from(self, before: cst.ImportFrom, updated: cst.ImportFrom) -> list[cst.ImportFrom] | None:
        if not any(id(a) in self.from_edits or id(a) in self.drop for a in before.names):
            return None
        groups: dict[str, list[cst.ImportAlias]] = {}
        keep_key = "\0keep"
        for old, new in zip(before.names, updated.names, strict=False):
            if id(old) in self.drop:
                continue
            if id(old) in self.from_edits:
                module, name, bound = self.from_edits[id(old)]
                alias = cst.ImportAlias(
                    name=cst.Name(name), asname=cst.AsName(cst.Name(bound)) if bound != name else None
                )
                groups.setdefault(module, []).append(alias)
            else:
                groups.setdefault(keep_key, []).append(new)
        out = []
        for key, names in groups.items():
            if key == keep_key:
                out.append(
                    updated.with_changes(
                        names=_commas(names),
                        lpar=None if len(names) == 1 else updated.lpar,
                        rpar=None if len(names) == 1 else updated.rpar,
                    )
                )
            else:
                out.append(cst.ImportFrom(module=cst.parse_expression(key), names=_commas(names)))
        return out


def _string_like(node: cst.SimpleString, value: str) -> str:
    quote = node.value[len(node.prefix) :][0]
    return f"{node.prefix}{quote}{value}{quote}"


def _commas(names: list[cst.ImportAlias]) -> list[cst.ImportAlias]:
    return [
        a.with_changes(
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
            if i < len(names) - 1
            else cst.MaybeSentinel.DEFAULT
        )
        for i, a in enumerate(names)
    ]


def _alias_node(node: cst.CSTNode, bound: str) -> cst.ImportAlias | None:
    if not isinstance(node, (cst.Import, cst.ImportFrom)) or isinstance(node.names, cst.ImportStar):
        return None
    for alias in node.names:
        if alias.asname is not None:
            name = alias.asname.name.value
        elif isinstance(node, cst.Import):
            name = dotted(alias.name).split(".")[0]
        else:
            name = alias.name.value
        if name == bound:
            return alias
    return None


def _is_label(node: cst.Name, parents: Any) -> bool:
    """A name that binds or reads nothing in its scope: an attribute name, a keyword."""
    parent = parents.get(node)
    return (isinstance(parent, cst.Attribute) and parent.attr is node) or (
        isinstance(parent, cst.Arg) and parent.keyword is node
    )


def _scope_chain(scope: Any) -> list[Any]:
    out = []
    while scope is not None and scope not in out:
        out.append(scope)
        scope = getattr(scope, "parent", None)
    return out


def _is_store(attribute: cst.Attribute, parents: Any) -> bool:
    parent = parents.get(attribute)
    return (
        isinstance(parent, cst.AssignTarget)
        or (isinstance(parent, cst.AugAssign) and parent.target is attribute)
        or (isinstance(parent, cst.AnnAssign) and parent.target is attribute)
        or isinstance(parent, cst.Del)
    )


def _walk(node: cst.CSTNode) -> Iterable[cst.CSTNode]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.children)


_VANISHED = "#\N{NO-BREAK SPACE}symbols:removed"


def _vanish(line: cst.SimpleStatementLine) -> cst.BaseStatement:
    """A removed statement line: a placeholder (marked by a comment) that ``_carry`` drops, handing
    its leading blank lines to the next statement."""
    if not line.leading_lines:
        return cst.RemoveFromParent()
    return cst.SimpleStatementLine(
        body=[cst.Pass()],
        leading_lines=line.leading_lines,
        trailing_whitespace=cst.TrailingWhitespace(comment=cst.Comment(_VANISHED)),
    )


def _is_vanished(statement: cst.BaseStatement) -> bool:
    return (
        isinstance(statement, cst.SimpleStatementLine)
        and statement.trailing_whitespace.comment is not None
        and statement.trailing_whitespace.comment.value == _VANISHED
    )


def _carry(body: Iterable[cst.BaseStatement]) -> list[cst.BaseStatement]:
    """Drop the placeholders of removed import lines; the blank lines above one move to the next
    statement (its comment lines, about the removed import, go with it)."""
    out: list[cst.BaseStatement] = []
    blank = 0
    for statement in body:
        if _is_vanished(statement):
            lines = list(statement.leading_lines)
            leading = 0
            while leading < len(lines) and lines[leading].comment is None:
                leading += 1
            blank = max(blank, leading)
            continue
        if blank:
            own = list(statement.leading_lines)
            leading = 0
            while leading < len(own) and own[leading].comment is None:
                leading += 1
            statement = statement.with_changes(
                leading_lines=[cst.EmptyLine()] * min(2, max(blank, leading)) + own[leading:]
            )
            blank = 0
        out.append(statement)
    return out


class _Carry(cst.CSTTransformer):
    def leave_Module(self, original: cst.Module, updated: cst.Module) -> cst.Module:
        return updated.with_changes(body=_carry(updated.body))

    def leave_IndentedBlock(self, original: cst.IndentedBlock, updated: cst.IndentedBlock) -> cst.IndentedBlock:
        return updated.with_changes(body=_carry(updated.body))


class _CodeStrings(cst.CSTTransformer):
    """Python programs inside string literals (the code of child processes the tests run): their
    imports and references follow the same rewrite."""

    def __init__(self, path: str, resolver: Resolver, notes: list[str]):
        super().__init__()
        self.path, self.resolver, self.notes = path, resolver, notes

    def leave_SimpleString(self, original: cst.SimpleString, updated: cst.SimpleString) -> cst.SimpleString:
        prefix = updated.prefix
        body = updated.value[len(prefix) :]
        quote = body[:3] if body[:3] in ('"""', "'''") else body[:1]
        inner = body[len(quote) : len(body) - len(quote)]
        if "b" in prefix.lower() or "import " not in inner:
            return updated
        if not any(m in inner for m in self.resolver.changed_modules):
            return updated
        try:
            cst.parse_module(inner)
        except Exception:
            return updated
        pseudo = self.path[:-3] + "__code_string.py"
        new = rewrite_python(pseudo, inner, self.resolver, {}, self.notes, strings=False)
        if new == inner:
            return updated
        return updated.with_changes(value=prefix + quote + new + quote)


def rewrite_python(
    path: str,
    text: str,
    resolver: Resolver,
    moved_away: dict[str, tuple[str, str]],
    notes: list[str],
    strings: bool = True,
) -> str:
    if strings:
        text = cst.parse_module(text).visit(_CodeStrings(path, resolver, notes)).code
    wrapper = MetadataWrapper(cst.parse_module(text))
    rewriter = ReferenceRewriter(path, resolver, moved_away, notes)
    rewriter.analyze(wrapper)
    tree = wrapper.visit(rewriter).visit(_Carry())
    if rewriter.module_imports:
        body = list(tree.body)
        end = _import_block_end(body)
        new = [cst.parse_statement(t + "\n") for t in sorted(rewriter.module_imports)]
        if end < len(body):
            body[end] = _leading(body[end], 2)
        body[end:end] = new
        tree = tree.with_changes(body=body)
    return rewrite_strings(tree.code, resolver)


# ============================================================================== D. strings
def rewrite_strings(text: str, resolver: Resolver) -> str:
    pattern = _pattern(resolver)
    if pattern is None:
        return text

    def replace(match: re.Match[str]) -> str:
        module, sep, name = match.groups()
        home = resolver.home(module, name)
        return match.group(0) if home == (module, name) else f"{home[0]}{sep}{home[1]}"

    return pattern.sub(replace, text)


def _pattern(resolver: Resolver) -> re.Pattern[str] | None:
    if not hasattr(resolver, "_pattern"):
        modules = sorted(resolver.changed_modules, key=len, reverse=True)
        resolver._pattern = (
            re.compile(r"(?<![\w.])(" + "|".join(re.escape(m) for m in modules) + r")([.:])([A-Za-z_]\w*)")
            if modules
            else None
        )
    return resolver._pattern


def stale_mentions(text: str, resolver: Resolver, allowed: Iterable[str] = ()) -> list[tuple[int, str]]:
    """Lines that still name a dissolved module (dotted or as a path)."""
    names = sorted(resolver.dissolved, key=len, reverse=True)
    if not names:
        return []
    paths = [n.replace(".", "/") + ".py" for n in names]
    pattern = re.compile(
        r"(?<![\w.])("
        + "|".join(re.escape(n) for n in names)
        + r")(?![\w])|(?<![\w/.-])("
        + "|".join(re.escape(p) for p in paths)
        + r")"
    )
    out = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line
        for allowed_text in allowed:
            stripped = stripped.replace(allowed_text, "")
        if pattern.search(stripped):
            out.append((number, line.strip()))
    return out


# ============================================================================== E. unused imports
def drop_unused_imports(text: str, keep: set[str]) -> str:
    """Remove module-level import bindings nothing in the module references (scope analysis; string
    annotations and ``__all__`` count as uses; ``keep``: names other files import from it)."""
    tree = ast.parse(text)
    used = set(keep)
    for node in ast.walk(tree):
        for annotation in annotations_of(node):
            for inner in ast.walk(annotation):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    used.update(re.findall(r"[A-Za-z_]\w*", inner.value))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            used.update(ast.literal_eval(node.value))
    wrapper = MetadataWrapper(cst.parse_module(text))
    scopes = wrapper.resolve(ScopeProvider)
    glob = next(s for s in set(scopes.values()) if isinstance(s, GlobalScope))
    unused: set[int] = set()
    for assignment in glob.assignments:
        if isinstance(assignment, ImportAssignment) and not assignment.references and assignment.name not in used:
            alias = _alias_node(assignment.node, assignment.name)
            if alias is not None and not is_future(assignment.node):
                unused.add(id(alias))

    class Drop(cst.CSTTransformer):
        def leave_SimpleStatementLine(self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine) -> Any:
            body, changed = [], False
            for before, small in zip(original.body, updated.body, strict=False):
                if isinstance(before, (cst.Import, cst.ImportFrom)) and not isinstance(before.names, cst.ImportStar):
                    names = [new for old, new in zip(before.names, small.names, strict=False) if id(old) not in unused]
                    if len(names) != len(small.names):
                        changed = True
                        if names:
                            one = len(names) == 1 and isinstance(small, cst.ImportFrom)
                            body.append(
                                small.with_changes(names=_commas(names), **(dict(lpar=None, rpar=None) if one else {}))
                            )
                        continue
                body.append(small)
            if not changed:
                return updated
            if not body:
                return _vanish(updated)
            return updated.with_changes(body=body)

    return _squeeze(wrapper.module.visit(Drop()).visit(_Carry()).code)


def _bound_name(node: cst.Import | cst.ImportFrom, alias: cst.ImportAlias) -> str:
    if alias.asname is not None:
        return alias.asname.name.value
    return dotted(alias.name).split(".")[0] if isinstance(node, cst.Import) else alias.name.value


# ============================================================================== the run
@dataclass
class Result:
    texts: dict[str, str]
    removed: list[str]
    notes: list[str]
    closure: dict[str, dict[str, str]]


def index(paths: Iterable[str]) -> dict[str, Module]:
    out = {}
    for path in paths:
        if in_python_scope(path) and (REPO / path).is_file():
            module = Module(path, (REPO / path).read_text())
            out[module.name] = module
    return out


def reexported(modules: dict[str, Module], sources: set[str]) -> dict[str, set[str]]:
    """Names other files import from each module of ``sources`` (``from S import n``, ``S.n``)."""
    out: dict[str, set[str]] = defaultdict(set)
    for module in modules.values():
        tree = ast.parse(module.source)
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module != "__future__":
                base = _ast_absolute(module, node)
                for alias in node.names:
                    if base in sources:
                        out[base].add(alias.name)
                    elif f"{base}.{alias.name}" in sources:
                        aliases[alias.asname or alias.name] = f"{base}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in sources and alias.asname:
                        aliases[alias.asname] = alias.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                out[aliases[node.value.id]].add(node.attr)
    return out


def _ast_absolute(module: Module, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    base = module.name.split(".") if module.package else module.name.split(".")[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def unused_bindings(text: str) -> set[str]:
    """Module-level import bindings a module does not use (kept as they are: not this run's)."""
    tree = ast.parse(text)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for node in ast.walk(tree):
        for annotation in annotations_of(node):
            for inner in ast.walk(annotation):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    used.update(re.findall(r"[A-Za-z_]\w*", inner.value))
    bound = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            bound |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            bound |= {a.asname or a.name for a in node.names}
    return bound - used


def run(plan: Plan) -> Result:
    files = tracked()
    modules = index(files)
    mapping = plan.mapping()
    dissolved = {module_name(p) for p in plan.dissolve}
    resolver = Resolver(modules, mapping, dissolved)
    notes: list[str] = []
    texts: dict[str, str] = {}
    # B first: a rename can make room for a moved definition of the same name
    by_module: dict[str, dict[str, str]] = defaultdict(dict)
    for (path, name), new in plan.renames.items():
        by_module[path][name] = new
    for path, renames in by_module.items():
        texts[path] = rename_definitions((REPO / path).read_text(), renames)
    if plan.moves:
        current = dict(modules)
        for path, text in texts.items():
            current[module_name(path)] = Module(path, text)
        texts.update(move_definitions(plan, current, resolver))
    moved_away: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for (source, name), (dest, new) in plan.moves.items():
        moved_away[module_name(source)][name] = (module_name(dest), new)
    python = sorted({p for p in files if in_python_scope(p)} | set(texts))
    for path in python:
        if not path.endswith(".py") or not (path in texts or (REPO / path).is_file()):
            continue
        before = texts.get(path) or (REPO / path).read_text()
        after = rewrite_python(path, before, resolver, moved_away.get(module_name(path), {}), notes)
        if after != before or path in texts:
            texts[path] = after
    for path in files:
        if in_text_scope(path) and (REPO / path).is_file():
            before = (REPO / path).read_text()
            after = rewrite_strings(before, resolver)
            if after != before:
                texts[path] = after
    sources = {module_name(s) for s, _ in plan.moves}
    keep = reexported(modules, sources)
    touched = {s for s, _ in plan.moves} | {d for d, _ in plan.moves.values()}
    for path in sorted(touched):
        if path in texts and path not in plan.dissolve:
            original = (REPO / path).read_text() if (REPO / path).is_file() else ""
            moved_names = {n for (s, n) in plan.moves if s == path}
            texts[path] = drop_unused_imports(
                texts[path], (keep.get(module_name(path), set()) - moved_names) | unused_bindings(original)
            )
    destinations = {module_name(d) for d, _ in plan.moves.values()}
    for path in list(texts):
        if path.endswith(".py"):
            texts[path] = merge_duplicate_imports(texts[path], destinations)
    removed = []
    for path in plan.dissolve:
        text = texts.get(path)
        if text is None:
            continue
        leftover = [s for s in cst.parse_module(text).body if not (_is_docstring(s) or _is_import_line(s))]
        if leftover:
            raise PlanError(
                f"{path}: statements left after the moves: "
                + "; ".join(cst.Module([]).code_for_node(s).strip()[:60] for s in leftover)
            )
        removed.append(path)
        del texts[path]
    return Result(texts, removed, notes, closure_entries(plan, modules))


def closure_entries(plan: Plan, modules: dict[str, Module]) -> dict[str, dict[str, str]]:
    """``closure_map.toml`` entries of this run: every moved or renamed function or class under its
    recorded name ([functions]); every new module ([added]). A move between two modules outside
    ``glm_tpu/`` (the S4.3 test merges) needs none: G6 and G7 record ``glm_tpu`` only."""
    functions: dict[str, str] = {}
    removed: dict[str, str] = {}
    by_path = {m.path: m for m in modules.values()}
    for (source, name), (dest, new) in plan.moves.items():
        if not (source.startswith("glm_tpu/") or dest.startswith("glm_tpu/")):
            continue
        statement = by_path[source].defs.get(name) if source in by_path else None
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
            key = f"{module_name(dest)}:{new}"
            if key in functions:  # an identical definition kept once: the other one stops executing
                removed[f"{module_name(source)}:{name}"] = (
                    f"{plan.stage}: identical to {functions[key]}, kept once in {module_name(dest)}"
                )
            else:
                functions[key] = f"{module_name(source)}:{name}"
    for (path, name), new in plan.renames.items():
        statement = by_path[path].defs.get(name) if path in by_path else None
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
            functions[f"{module_name(path)}:{new}"] = f"{module_name(path)}:{name}"
    added = {module_name(path): f"{plan.stage}: {reason}" for path, reason in plan.added.items()}
    return dict(functions=functions, added=added, removed=removed)


def write(result: Result) -> None:
    for path, text in sorted(result.texts.items()):
        target = REPO / path
        target.parent.mkdir(parents=True, exist_ok=True)
        new = not target.exists()
        target.write_text(text)
        if new:
            git("add", path)
    for path in result.removed:
        git("rm", "-q", path)


# ============================================================================== --check
_BASE_MODULES: dict[str, set[str]] = {}


def _was_module(name: str, rev: str | None) -> bool:
    """A module of this repository now or at ``rev``."""
    if is_repo_module(name):
        return True
    if rev is None:
        return False
    if rev not in _BASE_MODULES:
        _BASE_MODULES[rev] = {
            module_name(p) for p in git("ls-tree", "-r", "--name-only", rev).split() if p.endswith(".py")
        }
    return name in _BASE_MODULES[rev]


def _module_aliases(tree: ast.Module, rev: str | None = None) -> set[str]:
    """Names bound to modules of this repository anywhere in a file (module or function imports;
    ``rev``: also the modules that existed at that commit)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.asname for a in node.names if a.asname and a.name.startswith(("glm_tpu", "tests", "tools"))}
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("glm_tpu", "tests", "tools")):
            names |= {a.asname or a.name for a in node.names if _was_module(f"{node.module}.{a.name}", rev)}
    return names


class _Normalize(ast.NodeTransformer):
    """A definition modulo import paths: no function-local imports, ``alias.name`` through a module
    alias of this repository spelled ``name``, and the run's renames undone (``messages``: string constants
    mapped as ``[messages]`` rewrites them)."""

    def __init__(
        self,
        aliases: set[str],
        renames: dict[str, str],
        scopes: dict[str, str] | None = None,
        messages: dict[str, str] | None = None,
    ):
        self.aliases, self.renames, self.scopes, self.messages = aliases, renames, scopes or {}, messages or {}

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, str) and node.value in self.messages:
            node.value = self.messages[node.value]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if node.args and is_named_scope(node):
            template = scope_template(node.args[0])
            if template in self.scopes:
                node.args[0] = scope_literal(self.scopes[template])
        return node

    def visit_Import(self, node: ast.AST) -> None:
        return None

    visit_ImportFrom = visit_Import

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.aliases and not hasattr(node.value, "spelled"):
            name = ast.copy_location(ast.Name(id=self.renames.get(node.attr, node.attr), ctx=node.ctx), node)
            # ``alias.module.name``: the spelled ``module`` is an attribute, never one of the file's aliases (a
            # destination that imports that module under the same name would otherwise spell ``name`` alone).
            name.spelled = True
            return name
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self.renames.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self.renames.get(node.name, node.name)
        self.generic_visit(node)
        node.body = node.body or [ast.Pass()]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self.renames.get(node.name, node.name)
        self.generic_visit(node)
        return node


def _definition(tree: ast.Module, name: str) -> ast.AST | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node
    return None


def normalized(
    tree: ast.Module,
    name: str,
    renames: dict[str, str],
    rev: str | None = None,
    scopes: dict[str, str] | None = None,
    messages: dict[str, str] | None = None,
) -> str | None:
    node = _definition(tree, name)
    if node is None:
        return None
    return ast.dump(_Normalize(_module_aliases(tree, rev), renames, scopes, messages).visit(copy.deepcopy(node)))


def _renames_for(tree: ast.Module, module: str, mapping: dict[tuple[str, str], tuple[str, str]]) -> dict[str, str]:
    """``old -> new`` of the names a module (at the base) binds to renamed definitions: its own, the
    ones it imports without an alias, and attributes of the repository modules it aliases."""
    out = {name: new for (m, name), (_, new) in mapping.items() if m == module and new != name}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                home = mapping.get((node.module, alias.name))
                if home is not None and alias.asname is None and home[1] != alias.name:
                    out[alias.name] = home[1]
                sub = f"{node.module}.{alias.name}"
                for (m, name), (_, new) in mapping.items():
                    if m == sub and new != name:
                        out.setdefault(name, new)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for (m, name), (_, new) in mapping.items():
                    if m == alias.name and new != name:
                        out.setdefault(name, new)
    return out


def check(plan: Plan) -> list[str]:
    """Problems of an applied table: a definition not (verbatim) at its home, a dissolved module
    still present."""
    problems: list[str] = []
    mapping = plan.mapping()
    cache: dict[str, ast.Module | None] = {}

    def tree_of(rev: str | None, path: str) -> ast.Module | None:
        key = f"{rev}:{path}"
        if key not in cache:
            try:
                text = git("show", key) if rev else (REPO / path).read_text()
            except (subprocess.CalledProcessError, FileNotFoundError):
                cache[key] = None
            else:
                cache[key] = ast.parse(text)
        return cache[key]

    def compare(source: str, name: str, dest: str, new: str) -> None:
        base, current = tree_of(plan.base, source), tree_of(None, dest)
        if base is None or current is None:
            problems.append(f"{source}:{name}: the base file or the destination {dest} is missing")
            return
        want = normalized(
            base,
            name,
            _renames_for(base, module_name(source), mapping),
            plan.base,
            plan.scopes,
            plan.messages.get(source),
        )
        have = normalized(current, new, {})
        if have is None:
            problems.append(f"{dest}: {new} missing ({plan.base}:{source}:{name})")
        elif want != have:
            problems.append(f"{dest}: {new} differs from {plan.base}:{source}:{name} (imports aside)")

    for (source, name), (dest, new) in plan.moves.items():
        compare(source, name, dest, new)
        remaining = tree_of(None, source)
        if source != dest and remaining is not None and _definition(remaining, name) is not None:
            problems.append(f"{source}: still defines {name}")
    for (path, name), new in plan.renames.items():
        compare(path, name, path, new)
    for path in plan.dissolve:
        if (REPO / path).exists():
            problems.append(f"{path}: dissolved module still present")
    return problems


# ============================================================================== F. named scopes (S4.4)
def is_named_scope(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "named_scope"


def scope_template(node: ast.expr) -> str | None:
    """The name a ``named_scope`` literal spells: a string, or an f-string as its source template
    (``"prefill_linear/{reduction_axis}_reduce"``); None for any other expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and all(
        isinstance(v, ast.Constant) or (v.conversion == -1 and v.format_spec is None) for v in node.values
    ):
        return "".join(
            v.value if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}" for v in node.values
        )
    return None


def scope_literal(template: str) -> ast.expr:
    """The literal that spells ``template`` (an f-string when it has a ``{...}`` part)."""
    return ast.parse(f"f{template!r}" if "{" in template else repr(template), mode="eval").body


def scope_names(tree: ast.Module) -> list[str | None]:
    """The names of a module's ``named_scope`` calls, in source order."""
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args and is_named_scope(n)]
    return [scope_template(n.args[0]) for n in sorted(calls, key=lambda n: (n.lineno, n.col_offset))]


def apply_named_scopes(plan: Plan) -> list[str]:
    """Rewrite every ``named_scope`` name literal under ``glm_tpu/`` that ``[named_scopes]`` names;
    returns the changed paths."""
    changed = []
    for path in tracked():
        if not (path.startswith("glm_tpu/") and path.endswith(".py")):
            continue
        text = (REPO / path).read_text()
        lines = text.split("\n")
        edits = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call) and node.args and is_named_scope(node):
                literal = node.args[0]
                template = scope_template(literal)
                if template in plan.scopes:
                    if literal.lineno != literal.end_lineno:
                        raise PlanError(f"{path}:{literal.lineno}: a named scope literal spans lines")
                    edits.append((literal.lineno, literal.col_offset, literal.end_col_offset, template))
        for line, start, end, template in sorted(edits, reverse=True):
            source = lines[line - 1]
            segment = source[start:end]  # the column offsets count UTF-8 bytes; the names are ASCII
            if segment.count(template) != 1:
                raise PlanError(f"{path}:{line}: {segment} does not spell {template} once")
            lines[line - 1] = source[:start] + segment.replace(template, plan.scopes[template]) + source[end:]
        if edits:
            (REPO / path).write_text("\n".join(lines))
            changed.append(path)
    return changed


def check_named_scopes(plan: Plan) -> list[str]:
    """Problems of an applied ``[named_scopes]``: a module's scope names differ from its base-commit
    scope names mapped through the table, or a mapped name is left."""
    problems = []
    for path in tracked():
        if not (path.startswith("glm_tpu/") and path.endswith(".py")):
            continue
        current = scope_names(ast.parse((REPO / path).read_text()))
        try:
            base = scope_names(ast.parse(git("show", f"{plan.base}:{path}")))
        except subprocess.CalledProcessError:
            base = []
        if current != [plan.scopes.get(name, name) for name in base]:
            problems.append(f"{path}: named scopes {current} differ from {plan.base} mapped through the table")
        problems += [f"{path}: named scope {name} is left" for name in current if name in plan.scopes]
    return problems


# ============================================================================== G. messages (S5 WU-R)
def _message_literals(tree: ast.Module, table: dict[str, str]) -> list[ast.Constant]:
    """The string constants of a module whose value is a key of ``table``, in source order."""
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in table]
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def apply_messages(plan: Plan) -> list[str]:
    """Rewrite the literals ``[messages]`` names (see G); returns the changed paths."""
    changed = []
    for path, table in sorted(plan.messages.items()):
        text = (REPO / path).read_text()
        lines = text.split("\n")
        literals = _message_literals(ast.parse(text), table)
        missing = sorted(set(table) - {n.value for n in literals})
        if missing:
            raise PlanError(f"{path}: [messages] keys not found: {missing}")
        for node in reversed(literals):
            source = lines[node.lineno - 1].encode()  # the column offsets count UTF-8 bytes
            spelled = source[node.col_offset : node.end_col_offset].decode()
            new = table[node.value]
            if node.lineno != node.end_lineno or spelled != f'"{node.value}"':
                raise PlanError(f"{path}:{node.lineno}: {spelled[:60]} is not a plain one-line literal")
            if '"' in new or "\\" in new or "\n" in new:
                raise PlanError(f"{path}: [messages] value {new!r} needs quoting; spell it without quotes or escapes")
            lines[node.lineno - 1] = (
                source[: node.col_offset] + f'"{new}"'.encode() + source[node.end_col_offset :]
            ).decode()
        (REPO / path).write_text("\n".join(lines))
        changed.append(path)
    return changed


_LAYOUT_TOKENS = frozenset({tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER})


def message_tokens(text: str, table: dict[str, str] | None = None, renames: dict[str, str] | None = None) -> list:
    """The tokens ``--check`` compares for a ``[messages]`` file (S5 WU-Docs; FOLLOWUPS 124, 196): every token but
    the layout ones in order, as ``(type, text)``; a string literal as the repr of its value and an f-string part
    as its text, each mapped through ``table``, and a name through ``renames`` (both given for the base side)."""
    table, renames, tokens = table or {}, renames or {}, []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type in _LAYOUT_TOKENS:
            continue
        value = token.string
        if token.type == tokenize.STRING:
            literal = ast.literal_eval(value)
            value = repr(table.get(literal, literal) if isinstance(literal, str) else literal)
        elif token.type == tokenize.FSTRING_MIDDLE:
            value = table.get(value, value)
        elif token.type == tokenize.NAME:
            value = renames.get(value, value)
        tokens.append((tokenize.tok_name[token.type], value))
    return tokens


def check_messages(plan: Plan) -> list[str]:
    """Problems of an applied ``[messages]``: a listed file differs from its base-commit text other than by
    the table's renames and messages, by AST (imports aside) or by token (layout aside), or a key is left in it."""
    problems = []
    mapping = plan.mapping()
    for path, table in sorted(plan.messages.items()):
        try:
            text = git("show", f"{plan.base}:{path}")
        except subprocess.CalledProcessError:
            problems.append(f"{path}: not in {plan.base}")
            continue
        base = ast.parse(text)
        current_text = (REPO / path).read_text()
        current = ast.parse(current_text)
        renames = _renames_for(base, module_name(path), mapping)
        want = ast.dump(
            _Normalize(_module_aliases(base, plan.base), renames, plan.scopes, table).visit(copy.deepcopy(base))
        )
        have = ast.dump(_Normalize(_module_aliases(current), {}).visit(copy.deepcopy(current)))
        if want != have:
            problems.append(f"{path}: differs from {plan.base} other than by the table's renames and messages")
        if message_tokens(text, table, renames) != message_tokens(current_text):
            problems.append(f"{path}: tokens differ from {plan.base} other than by the table's renames and messages")
        problems += [f"{path}:{n.lineno}: message {n.value!r} is left" for n in _message_literals(current, table)]
    return problems


# ============================================================================== closure_map.toml
CLOSURE_MAP = REPO / "tools" / "equivalence" / "closure_map.toml"


def write_closure_entries(stage: str, table_name: str, entries: dict[str, dict[str, str]]) -> int:
    """Add this run's reviewed G6/G7 entries (``[functions]``, ``[added]``, ``[removed]``) to
    ``closure_map.toml``; existing keys are kept. Returns the number of entries added."""
    text = CLOSURE_MAP.read_text()
    present = tomllib.loads(text)
    lines = text.splitlines(keepends=True)
    added = 0
    for table in ("functions", "added", "removed"):
        new = {k: v for k, v in sorted(entries.get(table, {}).items()) if k not in present.get(table, {})}
        if not new:
            continue
        start = lines.index(f"[{table}]\n") + 1
        while start < len(lines) and lines[start].startswith("#"):
            start += 1
        block = [
            f"# {stage} ({table_name}): "
            f"{'moved or renamed definitions, current = recorded' if table == 'functions' else 'reviewed'}\n"
        ]
        block += [f"{json.dumps(k)} = {json.dumps(v)}\n" for k, v in new.items()]
        lines[start:start] = block
        added += len(new)
    CLOSURE_MAP.write_text("".join(lines))
    tomllib.loads(CLOSURE_MAP.read_text())
    return added
