"""Append-once per-call originals for the bounded331-call history diagnostic.

BudgetedCalls still performs every original pre/post memory check, dispatch and
fleet vote. A completed or refused call is stored once; runner snapshots carry
small SHA-bound references instead of repeatedly serializing all old censuses.
An in-flight call stays expanded in runner.json so the existing failure publisher
preserves it even if execution never returns. This is not performance evidence.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.greenfield.prefill_window_worker import BudgetedCalls

SCHEMA = "ws32_history_call_original_v1"
MAX_CALLS = 331
MAX_CALL_BYTES = 2 << 20
MAX_TOTAL_BYTES = 160 << 20


def _bytes(entry: Mapping[str, Any]) -> bytes:
    return (json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def preserve_call(root: Path, entry: Mapping[str, Any], *, index: int, used_bytes: int) -> dict:
    """Publish a fresh fsynced original; refuse overflow before writing it."""
    if (type(index) is not int or not 0 <= index < MAX_CALLS
            or type(used_bytes) is not int or used_bytes < 0
            or not isinstance(entry, Mapping) or "original" in entry
            or type(entry.get("completed")) is not bool
            or not all(isinstance(entry.get(k), str) and entry[k] for k in ("phase", "graph"))):
        raise ValueError("history call original identity differs")
    raw = _bytes(entry)
    if len(raw) > MAX_CALL_BYTES or used_bytes + len(raw) > MAX_TOTAL_BYTES:
        raise ValueError("history call originals exceed registered byte budget")
    directory = root / "call_records"
    if root.is_symlink() or directory.is_symlink():
        raise ValueError("history call originals path is linked")
    directory.mkdir(exist_ok=True)
    path = directory / f"call{index:03d}.json"
    if any(parent.name.startswith("native.rank") for parent in path.parents):
        from scripts.greenfield.ws32_native_benchmark_transport import native_root, require_write_size
        if native_root(path) is not None:
            require_write_size(path, len(raw))
    # O_EXCL refuses existing regular files and links without truncating either.
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return dict(phase=entry["phase"], graph=entry["graph"], completed=entry["completed"],
        original=dict(schema=SCHEMA, path=f"call_records/{path.name}", bytes=len(raw),
                      sha256=sha256(raw).hexdigest(), index=index))


class HistoryCalls(BudgetedCalls):
    """Keep only the current call expanded; identical parent execution checks."""

    def call(self, phase: str, name: str, values: tuple[Any, ...], *, preserve: Callable[[Any], None]) -> Any:
        def preflight() -> int:
            entries = self.record["call_evidence"]
            if len(entries) >= MAX_CALLS or any("original" not in entry for entry in entries):
                raise ValueError("history prior call unsealed or call budget exhausted")
            return len(entries)
        index = self.phase(phase + "/original_preflight", preflight)
        result, failure = None, None
        try:
            result = super().call(phase, name, values, preserve=preserve)
        except Exception as exc:
            failure = exc

        def retain() -> None:
            entries = self.record["call_evidence"]
            if len(entries) != index + 1:
                raise ValueError("history call append inventory differs")
            used = sum(entry["original"]["bytes"] for entry in entries[:index])
            reference = preserve_call(self.root, entries[index], index=index, used_bytes=used)
            entries[index] = reference
            self.record["call_evidence_layout"] = SCHEMA
            self.record["call_original_bytes"] = used + reference["original"]["bytes"]
        try:
            self.phase(phase + "/original_retained", retain)
        except Exception as exc:
            # Keep the original device/refusal cause, disclosing any independent
            # evidence publication failure. Unreplaced entries remain expanded.
            self.record["call_original_error"] = f"{type(exc).__name__}: {exc}"
            failure = failure or exc
        if failure is not None:
            raise failure
        return result


def load_calls(root: Path, record: Mapping[str, Any], *, expected_calls: int | None = None) -> tuple[dict, ...]:
    """Independently resolve all complete-run originals without trusting claims."""
    count = MAX_CALLS if expected_calls is None else expected_calls
    if type(count) is not int or not 1 <= count <= MAX_CALLS:
        raise ValueError("call reader requires a trusted bounded count")
    entries = record.get("call_evidence")
    if (record.get("call_evidence_layout") != SCHEMA or not isinstance(entries, list)
            or len(entries) != count or root.is_symlink() or (root / "call_records").is_symlink()):
        raise ValueError("history complete call original inventory differs")
    result, total = [], 0
    expected_names = {f"call{i:03d}.json" for i in range(count)}
    if {p.name for p in (root / "call_records").iterdir()} != expected_names:
        raise ValueError("history call files missing or unexpected")
    for index, reference in enumerate(entries):
        if not isinstance(reference, Mapping) or set(reference) != {"phase", "graph", "completed", "original"}:
            raise ValueError("history call reference schema differs")
        original = reference["original"]
        relative = f"call_records/call{index:03d}.json"
        if (not isinstance(original, Mapping) or set(original) != {"schema", "path", "bytes", "sha256", "index"}
                or original["schema"] != SCHEMA or original["path"] != relative
                or type(original["index"]) is not int or original["index"] != index
                or type(original["bytes"]) is not int or not 0 < original["bytes"] <= MAX_CALL_BYTES
                or reference["completed"] is not True):
            raise ValueError("history call original binding differs")
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != original["bytes"]:
            raise ValueError("history call original size/type differs")
        total += original["bytes"]
        if total > MAX_TOTAL_BYTES:
            raise ValueError("history call originals exceed total budget")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != original["sha256"]:
            raise ValueError("history call original hash differs")
        entry = json.loads(raw)
        if (not isinstance(entry, dict) or "original" in entry or entry.get("completed") is not True
                or any(entry.get(key) != reference[key] for key in ("phase", "graph"))):
            raise ValueError("history call original identity differs")
        result.append(entry)
    if type(record.get("call_original_bytes")) is not int or record["call_original_bytes"] != total:
        raise ValueError("history call original total differs")
    return tuple(result)
