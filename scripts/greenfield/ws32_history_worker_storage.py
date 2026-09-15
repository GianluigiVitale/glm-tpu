"""Pre-write HLO cap for the fixed history worker, not a second compiler."""

from __future__ import annotations

import os
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, BinaryIO, TextIO

from scripts.greenfield import ws32_history_protocol as protocol
from glm_tpu.host_paths import _plain_path

MAX_HLO_BYTES = 64 << 20
FORMS = ("stablehlo.mlir", "optimized_hlo.txt")
METADATA = {"runner.json": 6 << 20, "history_report.json": 2 << 20,
            "compile_journal.jsonl": 1 << 20, "retained_preflight.json": 1 << 20,
            "worker.log": 2 << 20}
PUBLICATION = {"worker_receipts.json": 256 << 10, "publication_omissions.json": 256 << 10,
               "ledger_source.json": 256 << 10}


@contextmanager
def _locked(root: Path):
    _plain_path(root)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def is_metadata(path: Path) -> bool:
    """Route only the fixed numerical-history rank namespace; old paths unchanged."""
    return (protocol.is_tag(path.parent.parent.name)
            and re.fullmatch(r"rank[0-7]", path.parent.name) is not None)


def _reserve(path: Path, size: int, *, append: bool = False) -> None:
    _plain_path(path)
    if not is_metadata(path) or not path.parent.is_dir():
        raise ValueError("history worker metadata root differs")
    table = METADATA if path.name in METADATA else PUBLICATION
    if path.name not in table or type(size) is not int or size < 0:
        raise ValueError("history worker metadata name/size differs")
    used = 0
    for existing in path.parent.iterdir():
        temporary = re.fullmatch(r"\.([^/]+)\.tmp\.[0-9]+", existing.name)
        if existing.name in table or (temporary and temporary[1] in table):
            _plain_path(existing)
            if not stat.S_ISREG(existing.stat().st_mode):
                raise ValueError("history worker metadata is not regular")
            used += existing.stat().st_size
    prior = path.stat().st_size if path.exists() else 0
    if (size + (prior if append else 0) > table[path.name]
            or used + size > ((12 << 20) if table is METADATA else (1 << 20))):
        raise ValueError("history worker metadata exceeds pre-write budget")


def write_json(path: Path, value: Any) -> None:
    """Preserve original atomic replacement, reserving old plus temporary bytes."""
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with _locked(path.parent):
        _reserve(path, len(data))
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)


def write_journal(stream: TextIO, text: str) -> None:
    """Bound the existing journal's exact serialized append, not its schema."""
    path = Path(stream.name)
    if path.name != "compile_journal.jsonl":
        raise ValueError("history journal path differs")
    with _locked(path.parent):
        _reserve(path, len(text.encode()), append=True)
        stream.write(text)
        stream.flush()


def stream_log(root: Path, source: BinaryIO) -> None:
    """Retain at most2MiB; overflow refuses the pipe and original worker command."""
    path = root / "worker.log"
    with _locked(root):
        _reserve(path, 0)
        output = path.open("xb", buffering=0)
    with output:
        while chunk := source.read(64 << 10):
            with _locked(root):
                _reserve(path, len(chunk), append=True)
                output.write(chunk)
        os.fsync(output.fileno())


def write_graph(root: Path, name: str, form: str, text: str) -> None:
    """Refuse oversized new text before disk write; retain all earlier originals."""
    _plain_path(root)
    if (not root.is_dir() or not re.fullmatch(r"rank[0-7]", root.name)
            or not protocol.is_tag(root.parent.name)
            or name not in protocol.PROGRAMS or form not in FORMS or not isinstance(text, str)):
        raise ValueError("history graph writer identity differs")
    data = text.encode()
    total = 0
    for graph in protocol.PROGRAMS:
        for kind in FORMS:
            existing = root / f"{graph}.{kind}"
            _plain_path(existing)
            if existing.exists():
                if not stat.S_ISREG(existing.stat().st_mode):
                    raise ValueError("history graph original is not regular")
                total += existing.stat().st_size
    if total + len(data) > MAX_HLO_BYTES:
        raise ValueError("history compiler originals exceed64MiB before write")
    with (root / f"{name}.{form}").open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    stream_log(parser.parse_args().root, sys.stdin.buffer)
