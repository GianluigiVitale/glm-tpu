"""Read-only tracked-source inventory; never imports the model or contacts TPU.

Static reachability is conservative, not proof a file is unused. Dynamic imports,
subprocess entry points, source-hash contracts and generated paths require review.
Output contains paths/counts, never file contents or suspected credential values.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable


ROOTS = (
    "glm_tpu/cli.py",
    "scripts/release/launch_ws32_user_request.py",
    "scripts/greenfield/launch_ws32_native_benchmark.py",
    "scripts/greenfield/run_short_decoder_ws32.py",
)
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "hf_token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    "google_private_key": re.compile(rb'"private_key"\s*:\s*"-----BEGIN'),
}


def module_name(path: str) -> str:
    parts = path.removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports(path: str, source: str) -> tuple[set[str], set[str], list[int]]:
    """Return import candidates, string constants and dynamic-dispatch line hints."""
    tree = ast.parse(source, filename=path)
    names: set[str] = set()
    strings: set[str] = set()
    dynamic: list[int] = []
    own = module_name(path)
    package = own if path.endswith("/__init__.py") else own.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, package)
            if base:
                names.add(base)
                names.update(base + "." + a.name for a in node.names if a.name != "*")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in {"__import__", "import_module", "spec_from_file_location",
                        "Popen", "run", "check_call", "check_output", "exec", "eval"}:
                dynamic.append(node.lineno)
    return names, strings, sorted(set(dynamic))


def closure(roots: Iterable[str], edges: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        path = pending.pop()
        if path not in seen:
            seen.add(path)
            pending.extend(edges.get(path, set()) - seen)
    return seen


def scan(repo: Path, roots: tuple[str, ...] = ROOTS) -> dict:
    repo = repo.resolve()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    paths = {p for p in tracked if p}
    if not set(roots) <= paths:
        raise ValueError("entry points must be tracked files")
    modules = {module_name(p): p for p in paths if p.endswith(".py")}
    edges: dict[str, set[str]] = {}
    external: dict[str, set[str]] = {}
    literal_assets: dict[str, list[str]] = {}
    dynamic: dict[str, list[int]] = {}
    counts: Counter = Counter()
    sizes: Counter = Counter()
    findings: list[dict] = []
    errors: list[dict] = []
    for path in sorted(paths):
        target = repo / path
        # Refuse to follow repository symlinks into private/local files.
        if target.is_symlink():
            findings.append(dict(path=path, kind="symlink_requires_review"))
            continue
        if not target.is_file():
            errors.append(dict(path=path, kind="missing_tracked_file"))
            continue
        data = target.read_bytes()
        group = "docs/artifacts" if path.startswith("docs/artifacts/") else path.split("/")[0]
        counts[group] += 1
        sizes[group] += len(data)
        for kind, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(data):
                findings.append(dict(path=path, kind=kind, line=data[:match.start()].count(b"\n") + 1))
        if target.suffix in {".safetensors", ".db", ".pem", ".key"} or target.name.startswith(".env"):
            findings.append(dict(path=path, kind="sensitive_or_payload_filename"))
        if not path.endswith(".py"):
            continue
        try:
            names, strings, sites = imports(path, data.decode("utf-8"))
        except (SyntaxError, UnicodeError, ImportError, ValueError) as exc:
            errors.append(dict(path=path, kind=type(exc).__name__))
            continue
        linked: set[str] = set()
        ext: set[str] = set()
        for name in names:
            parts = name.split(".")
            matches = {modules[".".join(parts[:i])] for i in range(1, len(parts) + 1)
                       if ".".join(parts[:i]) in modules}
            linked.update(matches)
            if not matches:
                ext.add(parts[0])
        edges[path] = linked
        external[path] = ext
        # Only exact literal paths are established here. Computed paths remain unknown.
        assets = sorted(strings & paths)
        if assets:
            literal_assets[path] = assets
        if sites:
            dynamic[path] = sites
    reachable = closure(roots, edges)
    return dict(
        schema="glm_release_static_inventory_v1",
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        scope="tracked working-tree files; no history scan, runtime import or external asset verification",
        entry_points=list(roots), tracked_files=len(paths), tracked_bytes=sum(sizes.values()),
        groups={p: dict(files=counts[p], bytes=sizes[p]) for p in sorted(counts)},
        static_import_closure=sorted(reachable),
        external_import_roots=sorted(set().union(*(external.get(p, set()) for p in reachable))),
        literal_tracked_assets={p: literal_assets[p] for p in sorted(reachable & literal_assets.keys())},
        dynamic_dispatch_review={p: dynamic[p] for p in sorted(reachable & dynamic.keys())},
        findings=findings, errors=errors,
        deletion_authorized=False,
        limitations=["Includes conditional/test-only imports within reachable files.",
                     "Bare sibling imports and dynamically composed paths may be unresolved.",
                     "Secret pattern scan is not comprehensive security or private-data clearance.",
                     "Files outside static closure are NOT proven unused."],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = scan(args.repo)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] or report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
