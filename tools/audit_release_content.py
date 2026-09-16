"""Read-only credential-pattern/payload audit, optionally over all local Git blobs.

Reports locations/object IDs, NEVER matched contents. Heuristic detection is not
complete security/privacy clearance. No imports of the model, network or writes.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any

PATTERNS = {
    "private_key": rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
    "github_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "huggingface_token": rb"\bhf_[A-Za-z0-9]{30,}\b",
    "anthropic_token": rb"\bsk-ant-[A-Za-z0-9_-]{30,}\b",
    "openai_token": rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b",
    "google_api_key": rb"\bAIza[A-Za-z0-9_-]{35}\b",
    "google_oauth_token": rb"\bya29\.[A-Za-z0-9_-]{30,}\b",
    "aws_access_key_id": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "slack_token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
}
COMPILED = {name: re.compile(value) for name, value in PATTERNS.items()}
PREFIXES = {
    "private_key": (b"PRIVATE KEY-----",),
    "github_token": (b"gh",),
    "huggingface_token": (b"hf_",),
    "anthropic_token": (b"sk-ant-",),
    "openai_token": (b"sk-",),
    "google_api_key": (b"AIza",),
    "google_oauth_token": (b"ya29.",),
    "aws_access_key_id": (b"AKIA", b"ASIA"),
    "slack_token": (b"xox",),
}
# Tracked files and history blobs above this size are reported as unscanned
# errors rather than passed silently. The per-file curation ledger
# (docs/curation/disposition.jsonl) is about 2.5 MB of JSON lines and must be
# scanned like every other tracked text file.
FILE_CAP = 4 << 20
HISTORY_CAP = 2 << 30
FORBIDDEN_SUFFIXES = {
    ".safetensors",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}


def matches(data: bytes) -> list[dict]:
    return [
        dict(kind=kind, line=data.count(b"\n", 0, match.start()) + 1)
        for kind, pattern in COMPILED.items()
        if any(prefix in data for prefix in PREFIXES[kind])
        for match in pattern.finditer(data)
    ]


def forbidden_name(name: str) -> bool:
    path = Path(name)
    return (
        path.suffix.lower() in FORBIDDEN_SUFFIXES
        or path.name == ".env"
        or path.name.startswith(".env.")
        or path.name
        in {
            "id_rsa",
            "id_ed25519",
            "credentials.json",
            "application_default_credentials.json",
            "request.json",
            "requests.json",
            "tokens.jsonl",
            "answer.txt",
            "answer.txt.gz",
        }
        or name.startswith("bench/data/")
    )


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def scan_tree(repo: Path) -> dict:
    paths = sorted(filter(None, git(repo, "ls-files", "-z").decode().split("\0")))
    findings, errors, records = [], [], []
    for name in paths:
        path = repo / name
        if forbidden_name(name):
            findings.append(
                dict(path=name, kind="forbidden_payload_or_credential_filename")
            )
        if any(p.is_symlink() for p in (path, *path.parents)):
            errors.append(dict(path=name, kind="symlink_not_followed"))
            continue
        if not path.is_file() or path.stat().st_size > FILE_CAP:
            errors.append(dict(path=name, kind="missing_or_oversized_tracked_file"))
            continue
        data = path.read_bytes()
        records.append(
            dict(path=name, bytes=len(data), sha256=sha256(data).hexdigest())
        )
        findings.extend(dict(path=name, **row) for row in matches(data))
    digest = sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return dict(
        files=len(paths),
        bytes=sum(row["bytes"] for row in records),
        content_manifest_sha256=digest,
        findings=findings,
        errors=errors,
    )


def scan_history(repo: Path) -> dict:
    # All local objects, not just reachable current/main history. Read headers
    # first so the total byte budget is known before scanning the blob bodies.
    headers = git(
        repo,
        "cat-file",
        "--batch-all-objects",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
    )
    blobs = [
        (row[0], int(row[2]))
        for line in headers.decode().splitlines()
        if (row := line.split())[1] == "blob"
    ]
    total = sum(size for _, size in blobs)
    if total > HISTORY_CAP:
        raise ValueError("Git blob audit exceeds bounded history budget; not scanned")
    findings, errors = [], []
    paths = {}
    for line in git(repo, "rev-list", "--objects", "--all").decode().splitlines():
        oid, _, name = line.partition(" ")
        if name:
            paths[oid] = name
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    scanned = scanned_bytes = 0
    try:
        for oid, size in blobs:
            name = paths.get(oid)
            if name and forbidden_name(name):
                findings.append(
                    dict(
                        object=oid,
                        path_hint=name,
                        kind="historical_payload_or_credential_filename",
                    )
                )
            if size > FILE_CAP:
                errors.append(
                    dict(object=oid, path_hint=name, kind="oversized_blob_not_scanned")
                )
                continue
            process.stdin.write((oid + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().split()
            if header != [oid, "blob", str(size)]:
                raise ValueError("Git batch object identity differs")
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b"\n":
                raise ValueError("Git batch object is incomplete")
            findings.extend(
                dict(object=oid, path_hint=name, **row) for row in matches(data)
            )
            scanned += 1
            scanned_bytes += size
        process.stdin.close()
        if process.wait(timeout=30) != 0:
            raise ValueError("Git batch reader failed")
    finally:
        if process.poll() is None:
            process.terminate()  # This tool's own Git reader only, never a worker.
            process.wait(timeout=10)
        process.stdout.close()
    return dict(
        total_blobs=len(blobs),
        scanned_blobs=scanned,
        scanned_bytes=scanned_bytes,
        object_set_sha256=sha256(headers).hexdigest(),
        findings=findings,
        errors=errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also inspect every local Git blob within 2 GiB",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    tree = scan_tree(repo)
    history = scan_history(repo) if args.history else None
    report = dict(
        schema="glm_release_content_audit_v1",
        head=git(repo, "rev-parse", "HEAD").decode().strip(),
        worktree_dirty=bool(git(repo, "status", "--porcelain")),
        tree=tree,
        history=history,
        rule_names=list(PATTERNS),
        limits=[
            "Known credential formats and filenames only; does not detect every secret or private dataset",
            "Historical path_hint is one reachable name, not all paths; unreachable objects may have no name",
            "No automatic deletion, credential validity check, revocation or history rewrite",
            "A clean pattern scan is not permission to publish the repository",
        ],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(
        any(part and (part["findings"] or part["errors"]) for part in (tree, history))
    )


if __name__ == "__main__":
    raise SystemExit(main())
