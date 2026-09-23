"""The reviewed rename table for G6 (import closures) and G7 (executed functions).

``closure_map.toml`` maps *current* names to the names recorded in ``tests/golden/data``
(DESIGN.md 7.5.6: G6/G7 names are "mapped through the current rename tables"). Every entry lands
in the commit that moves or renames the code and is reviewed there:

* ``[modules]``: ``"new.module" = "old.module"``; a key and value that both end in ``.`` map a
  package prefix (longest prefix wins). Applies to G6 module names, to the module part of G7
  entries and to the files in G6's static layering scan.
* ``[functions]``: ``"new.module:New.qualname" = "old.module:Old.qualname"`` (G7; the
  G1-protocol/G2-protocol defaults record also finds a renamed option class or builder through it).
* ``[added]``: ``"module"`` or ``"module:qualname"`` = reason (stage, H number or commit) -- a
  genuinely new module in a G6 closure or a new executed function in G7.
* ``[removed]``: ``"old.module:qualname"`` = reason -- a G7 function that may stop executing
  (G6 closures may shrink without an entry).

After ``record --gates G6,G7 --rename-only --reason ...`` has re-recorded the data under the new
names, the entries are removed again (a stale entry maps a current name to a name the new data no
longer contain, so the gate fails until the table is cleared).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any, Callable

from .common import REPO, digest_json

MAP = Path(__file__).with_name("closure_map.toml")
TABLES = ("modules", "functions", "added", "removed")


@dataclass(frozen=True)
class ClosureMap:
    modules: dict[str, str] = field(default_factory=dict)
    functions: dict[str, str] = field(default_factory=dict)
    added: dict[str, str] = field(default_factory=dict)
    removed: dict[str, str] = field(default_factory=dict)

    def digest(self) -> str:
        return digest_json({name: getattr(self, name) for name in TABLES})

    def module(self, name: str) -> str:
        if name in self.modules:
            return self.modules[name]
        prefixes = [key for key in self.modules if key.endswith(".") and name.startswith(key)]
        if prefixes:
            key = max(prefixes, key=len)
            return self.modules[key] + name[len(key):]
        return name

    def function(self, entry: str) -> str:
        if entry in self.functions:
            return self.functions[entry]
        module, _, qualname = entry.partition(":")
        return f"{self.module(module)}:{qualname}"

    def current_names(self, old_name: str) -> tuple[str, ...]:
        """Attribute names under which a top-level 181c013e definition may exist now."""
        names = [old_name]
        for new, old in self.functions.items():
            if old.partition(":")[2] == old_name:
                names.append(new.partition(":")[2])
        return tuple(dict.fromkeys(names))


def load(path: Path = MAP) -> ClosureMap:
    if not path.is_file():
        return ClosureMap()
    value = tomllib.loads(path.read_text())
    unknown = sorted(set(value) - set(TABLES))
    if unknown:
        raise ValueError(f"closure_map.toml: unknown tables {unknown}")
    tables: dict[str, dict[str, str]] = {}
    for name in TABLES:
        table = value.get(name, {})
        if not isinstance(table, dict) or not all(isinstance(k, str) and isinstance(v, str) and v
                                                  for k, v in table.items()):
            raise ValueError(f"closure_map.toml [{name}] must map strings to non-empty strings")
        tables[name] = table
    for key, old in tables["modules"].items():
        if key.endswith(".") != old.endswith("."):
            raise ValueError(f"closure_map.toml [modules] {key!r}: a package prefix maps to a package prefix")
    for key, old in tables["functions"].items():
        if ":" not in key or ":" not in old:
            raise ValueError(f"closure_map.toml [functions] {key!r}: entries are module:qualname")
    return ClosureMap(**tables)


def _is_package(module: str) -> bool:
    return (REPO / module.replace(".", "/") / "__init__.py").is_file()


def compare_modules(old: list[str], new: list[str], cmap: ClosureMap, *,
                    is_package: Callable[[str], bool] = _is_package) -> dict[str, Any]:
    """A stage closure may only shrink: every current module, mapped to its recorded name, must be
    in the recorded closure, declared in ``[added]``, or a package that only exists as the parent
    of a mapped (moved) module."""
    baseline = set(old)
    mapped = {name: cmap.module(name) for name in new}
    moved_parents = {name.rsplit(".", i)[0] for name, target in mapped.items() if target != name
                     for i in range(1, name.count(".") + 1)}
    added = sorted(name for name, target in mapped.items()
                   if target not in baseline and name not in cmap.added
                   and not (name in moved_parents and is_package(name)))
    removed = sorted(baseline - set(mapped.values()))
    return dict(added=added, removed=removed, renamed=sorted(f"{k} -> {v}" for k, v in mapped.items() if k != v))


def compare_functions(old: list[str], new: list[str], cmap: ClosureMap) -> dict[str, Any]:
    """G7: the executed set, mapped to recorded names, equals the recorded set up to declared
    additions and removals."""
    baseline = set(old)
    mapped = {entry: cmap.function(entry) for entry in new}
    targets = set(mapped.values())
    added = sorted(entry for entry, target in mapped.items() if target not in baseline and entry not in cmap.added)
    removed = sorted(entry for entry in baseline - targets if entry not in cmap.removed)
    return dict(added=added, removed=removed, renamed=sorted(f"{k} -> {v}" for k, v in mapped.items() if k != v),
                declared_removed=sorted(baseline.intersection(cmap.removed) - targets))


def layering_entry(entry: str, cmap: ClosureMap) -> str:
    """``path: imported`` of the static scan, with both sides mapped to recorded module names."""
    path, _, imported = entry.partition(": ")
    module = path[:-3].replace("/", ".")
    module = module[: -len(".__init__")] if module.endswith(".__init__") else module
    return f"{cmap.module(module)}: {cmap.module(imported)}"
