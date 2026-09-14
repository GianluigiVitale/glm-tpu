"""Create/check a per-file curation ledger; never import or execute model code.

Git enumeration and hashes are evidence of coverage, not semantic review.
Initialization deliberately leaves every disposition unresolved. This tool
cannot certify that a person/agent actually performed a recorded full read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

BASE = "b667f00f1ae48c8ff37e92500550c1395d74c66d"
CATEGORIES = {
    "supported",
    "dependency_or_evidence",
    "test_or_documentation",
    "research_or_superseded",
    "unresolved",
}


def baseline(repo: Path, pin: str = BASE) -> dict[str, dict]:
    if re.fullmatch(r"[0-9a-f]{40}", pin) is None:
        raise ValueError("baseline must be a full commit ID")
    raw = subprocess.check_output(["git", "ls-tree", "-r", "-l", "-z", pin], cwd=repo)
    result = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, path = record.split(b"\t", 1)
        mode, kind, blob, size = header.decode().split()
        if kind != "blob":
            raise ValueError("non-blob entry needs explicit handling")
        result[path.decode()] = dict(blob=blob, bytes=int(size), mode=mode)
    return result


def tracked(repo: Path) -> set[str]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo)
    return {p.decode() for p in raw.split(b"\0") if p}


def pending(path: str, original: dict | None) -> dict:
    return dict(
        path=path,
        baseline=original,
        category="unresolved",
        action="review",
        purpose=None,
        consumers=[],
        justification="Supported-path role and consumers have not yet been semantically reviewed.",
        review=dict(kind="unreviewed", sha256=None),
    )


def read_ledger(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = row["path"]
        pure = PurePosixPath(name)
        if not name or pure.is_absolute() or ".." in pure.parts or str(pure) != name:
            raise ValueError("invalid ledger path")
        if name in rows:
            raise ValueError("duplicate ledger path: " + name)
        rows[name] = row
    return rows


def inventory(repo: Path, existing: dict[str, dict] | None = None) -> dict[str, dict]:
    originals = baseline(repo)
    rows = dict(existing or {})
    for path in sorted(set(originals) | tracked(repo)):
        if path not in rows:
            rows[path] = pending(path, originals.get(path))
    return rows


def check(repo: Path, rows: dict[str, dict], *, ledger_path: str) -> dict:
    originals, current = baseline(repo), tracked(repo)
    required = set(originals) | current
    errors, unresolved = [], []
    for path in sorted(required - rows.keys()):
        errors.append("missing disposition: " + path)
    for path in sorted(rows.keys() - required):
        errors.append("untracked non-baseline disposition: " + path)
    for path, row in rows.items():
        if row.get("baseline") != originals.get(path):
            errors.append("baseline identity differs: " + path)
        category, action = row.get("category"), row.get("action")
        if category not in CATEGORIES or action not in {"keep", "remove", "review"}:
            errors.append("invalid disposition: " + path)
            continue
        if category == "unresolved" or action == "review":
            unresolved.append(path)
            continue
        if not all(
            isinstance(row.get(k), str) and row[k].strip()
            for k in ("purpose", "justification")
        ):
            errors.append("missing semantic justification: " + path)
        if not isinstance(row.get("consumers"), list) or not row["consumers"]:
            errors.append("missing consumer/recovery explanation: " + path)
        review = row.get("review", {})
        if action == "remove":
            if path in current or (repo / path).exists():
                errors.append("removal not implemented: " + path)
            expected_recovery = (
                dict(commit=BASE, path=path, blob=originals[path]["blob"])
                if path in originals
                else None
            )
            if (
                category != "research_or_superseded"
                or expected_recovery is None
                or row.get("recovery") != expected_recovery
            ):
                errors.append("removal lacks classification/recovery: " + path)
            continue
        if path not in current:
            errors.append("retained file not tracked: " + path)
            continue
        local = repo / path
        if (
            local.is_symlink()
            or not local.is_file()
            or not local.resolve().is_relative_to(repo.resolve())
        ):
            errors.append("retained path unavailable or symlink: " + path)
            continue
        if category == "research_or_superseded":
            errors.append("research-only material retained on candidate main: " + path)
        if path == ledger_path:
            # This generated ledger cannot embed its own hash. Its coverage,
            # baseline entries and disposition invariants are checked here.
            if review.get("kind") != "manifest_validation":
                errors.append("ledger self-validation not recorded")
            continue
        kind = review.get("kind")
        if kind not in {"full_read", "generated_validation"} or not review.get("notes"):
            errors.append("review not established: " + path)
        if (
            local.suffix in {".py", ".sh", ".md", ".toml", ".jinja"}
            and kind != "full_read"
        ):
            errors.append("implementation/documentation lacks full read: " + path)
        if review.get("sha256") != hashlib.sha256(local.read_bytes()).hexdigest():
            errors.append("reviewed bytes differ: " + path)
    return dict(
        baseline_commit=BASE,
        baseline_files=len(originals),
        baseline_bytes=sum(r["bytes"] for r in originals.values()),
        current_tracked_files=len(current),
        dispositions=len(rows),
        unresolved=len(unresolved),
        errors=errors,
        complete=not errors and not unresolved,
        scope="ledger consistency; recorded semantic review is not independently verified",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--ledger", default="docs/curation/disposition.jsonl")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="print updated ledger, never mark files reviewed",
    )
    args = parser.parse_args()
    path = args.repo / args.ledger
    rows = read_ledger(path) if path.exists() else {}
    if args.refresh:
        for _, row in sorted(inventory(args.repo, rows).items()):
            print(json.dumps(row, sort_keys=True, separators=(",", ":")))
        return 0
    report = check(args.repo, rows, ledger_path=args.ledger)
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
