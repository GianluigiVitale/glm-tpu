"""Bounded history originals through the existing exact-generation transport.

No launcher, checkpoint loader or numerical validator. The protected parent
supplies the independent fleet validator and separately materializes old branch
references. Local originals are never removed, including publication refusals.
"""

from __future__ import annotations

import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Callable, Mapping

import google_crc32c

from glm_tpu.greenfield.validation.ws32_evidence import _atomic_download, _link_exact, _require_blob_identity
from scripts.greenfield import ws32_history_admission as admission
from scripts.greenfield import ws32_history_call_evidence as calls
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_evidence import same_json
from scripts.greenfield.ws32_history_preflight import RUN_ROOT, _plain_path

MAX_RANK_BYTES = 512 << 20
MAX_REFERENCE_BYTES = 32 << 20
MAX_LOCAL_RANK_BYTES = MAX_RANK_BYTES + MAX_REFERENCE_BYTES
MAX_LEDGER_BYTES = 256 << 10
MAX_CONTROLLER_LEDGER_BYTES = 2 << 20
MAX_RANK_FILES = 384
MAX_ARCHIVE_FILES = 4096
MAX_FLEET_BYTES = 8 * MAX_RANK_BYTES
MAX_ARCHIVE_BYTES = 10 << 30
MAX_CONTROLLER_EXTRA_BYTES = 512 << 20
CONTROLLER_METADATA_RESERVE = 8 << 20
DISK_RESERVE = 1 << 30
LIMITS = dict(materializers=147 << 20, history=128 << 20, calls=160 << 20,
              hlo=64 << 20, metadata=12 << 20, publication=1 << 20)
METADATA_LIMITS = {"runner.json": 6 << 20, "history_report.json": 2 << 20,
                   "compile_journal.jsonl": 1 << 20, "retained_preflight.json": 1 << 20,
                   "worker.log": 2 << 20}
WK_FILES = tuple(f"materializers/layer{layer}_{name}.npz" for layer in protocol.PRODUCERS
                 for name in ("wk_decode", "wk_promote"))
EXACT_FILES = ("materializers/exact_decode.json", "materializers/exact_promote.json")
CALL_FILES = tuple(f"call_records/call{i:03d}.json" for i in range(calls.MAX_CALLS))
HLO_FILES = tuple(f"{name}.{form}" for name in protocol.PROGRAMS
                  for form in ("stablehlo.mlir", "optimized_hlo.txt"))
OBSERVER_FILES = tuple(f"observer_{branch}.npz" for branch in protocol.BRANCHES)
FILES = (*METADATA_LIMITS, *HLO_FILES, *WK_FILES, *EXACT_FILES, *CALL_FILES, *OBSERVER_FILES)
FIRST_DIFFERENCES = tuple(f"first_difference_group{i}.npz" for i in range(64))
REFUSED_HISTORY = (
    *(f"{kind}_step{i}.npz" for i in range(len(protocol.plan()))
      for kind in ("unhealthy", "refused_completed", "refused_replica")),
    *(f"{kind}_observer_{branch}.npz" for branch in protocol.BRANCHES
      for kind in ("unhealthy", "refused_replica")),
)
HISTORY_FILES = (*OBSERVER_FILES, *FIRST_DIFFERENCES, *REFUSED_HISTORY)
PARTIALS = tuple(name + ".pending" for name in HISTORY_FILES)
NOTICE = "publication_omissions.json"
LEDGER = "worker_receipts.json"
PUBLICATION_FILES = (NOTICE, LEDGER, "ledger_source.json")
PUBLISH_FILES = (*FILES, *FIRST_DIFFERENCES, *REFUSED_HISTORY, *PARTIALS,
                 "materializers/refused_exact.npz")
_KNOWN = frozenset((*PUBLISH_FILES, *PUBLICATION_FILES))
Validator = Callable[..., dict]


def is_tag(tag: str) -> bool:
    return protocol.is_tag(tag)


def _identity(tag: str, rank: int) -> None:
    if not is_tag(tag) or type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("history transport tag/rank differs")


def _root(root: Path, expected: Path, *, present: bool) -> None:
    _plain_path(root)
    if root != expected:
        raise ValueError("history transport root differs")
    if present and not root.is_dir():
        raise ValueError("history transport root missing")
    if not present and root.exists():
        raise FileExistsError(root)


def _bucket(client: Any) -> Any:
    bucket = client.bucket(protocol.BUCKET)
    bucket.reload()
    if str(bucket.location).upper() != "US-CENTRAL2":
        raise ValueError("history transport bucket must be US-CENTRAL2")
    return bucket


def _kind(name: str) -> str:
    if name not in _KNOWN:
        raise ValueError("history unregistered publication path")
    if name in PUBLICATION_FILES:
        return "publication"
    if name.startswith("materializers/"):
        return "materializers"
    if name in CALL_FILES:
        return "calls"
    if name in HLO_FILES:
        return "hlo"
    return "metadata" if name in METADATA_LIMITS else "history"


def _file_cap(name: str) -> int:
    if name in METADATA_LIMITS:
        return METADATA_LIMITS[name]
    if name in CALL_FILES:
        return calls.MAX_CALL_BYTES
    if name in EXACT_FILES:
        return 1 << 20
    if name == "materializers/refused_exact.npz":
        return 33 << 20
    if name in PUBLICATION_FILES:
        return MAX_LEDGER_BYTES
    return LIMITS[_kind(name)]


def _totals(rows: list[tuple[str, int]]) -> dict[str, int]:
    totals = dict.fromkeys(LIMITS, 0)
    wk = 0
    for name, size in rows:
        kind = _kind(name)
        if type(size) is not int or not 0 <= size <= _file_cap(name):
            raise ValueError("history original per-file budget exceeded")
        totals[kind] += size
        if name in WK_FILES:
            wk += size
    if (wk > 112 << 20 or any(totals[k] > LIMITS[k] for k in totals)
            or sum(totals.values()) > MAX_RANK_BYTES or len(rows) > MAX_RANK_FILES):
        raise ValueError("history original category/rank budget exceeded")
    return totals


def _regular(path: Path) -> None:
    _plain_path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("history original is not a regular file")


def _reference_bytes(root: Path) -> int:
    reference = root / "retained_reference"
    _plain_path(reference)
    if not reference.exists():
        return 0
    names = {f"{branch}.{form}{suffix}" for branch in protocol.BRANCHES
             for form in protocol.ORIGINAL_FORMS for suffix in ("", ".partial")}
    size = 0
    for path in reference.iterdir():
        if path.name not in names:
            raise ValueError("history unexpected retained reference")
        _regular(path)
        size += path.stat().st_size
    if size > MAX_REFERENCE_BYTES:
        raise ValueError("history retained reference budget exceeded")
    return size


def _record_identity(record: Mapping, *, tag: str, rank: int, pin: str | None = None) -> None:
    if (record.get("tag") != tag or type(record.get("launch_rank")) is not int
            or record["launch_rank"] != rank or record.get("protocol") != protocol.PROTOCOL
            or record.get("kernel") != protocol.KERNEL or record.get("profile") != admission.PROFILE
            or not isinstance(record.get("code_hash"), str)
            or re.fullmatch(r"[0-9a-f]{40}", record["code_hash"]) is None
            or (pin is not None and record["code_hash"] != pin)):
        raise ValueError("history original record identity differs")


def publish_rank(*, tag: str, rank: int, root: Path, client: Any) -> list[dict]:
    """After worker exit, preserve known bounded originals; never delete omissions."""
    _identity(tag, rank)
    _root(root, RUN_ROOT / tag / f"rank{rank}", present=True)
    _reference_bytes(root)  # Old generations remain separately recoverable, never republished.
    runner = root / "runner.json"
    if runner.exists() and not runner.is_symlink() and runner.stat().st_size <= METADATA_LIMITS["runner.json"]:
        _regular(runner)
        _record_identity(json.loads(runner.read_bytes()), tag=tag, rank=rank)
    bucket = _bucket(client)
    prefix = f"results/{tag}/workers/rank{rank}/"
    receipts, omitted, sizes = [], [], []
    for name in PUBLISH_FILES:
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue
        try:
            _regular(path)
            size = path.stat().st_size
            _totals([*sizes, (name, size), (LEDGER, MAX_LEDGER_BYTES), (NOTICE, MAX_LEDGER_BYTES)])
        except ValueError as exc:
            omitted.append(dict(name=name, reason=str(exc), originals_retained_locally=True))
            continue
        facts = digest_file(path)
        if facts["size"] != size:
            raise ValueError("history original changed after size preflight")
        receipts.append(publish_exact(bucket, prefix + name, path, facts, compressed=False))
        sizes.append((name, size))
    if omitted:
        notice = root / NOTICE
        value = dict(omitted=omitted, originals_retained_locally=True)
        if len(json.dumps(value).encode()) > MAX_LEDGER_BYTES:
            raise ValueError("history omissions exceed metadata budget; originals remain local")
        _atomic_json(notice, value)
        facts = digest_file(notice)
        if facts["size"] > MAX_LEDGER_BYTES:
            raise ValueError("history omissions exceed metadata budget; originals remain local")
        receipts.append(publish_exact(bucket, prefix + NOTICE, notice, facts, compressed=False))
        sizes.append((NOTICE, facts["size"]))
    ledger = root / LEDGER
    _atomic_json(ledger, receipts)
    facts = digest_file(ledger)
    _totals([*sizes, (LEDGER, facts["size"])])
    publish_exact(bucket, prefix + LEDGER, ledger, facts, compressed=False)
    return receipts


def _receipts(raw: bytes, *, tag: str, rank: int) -> list[dict]:
    """All mandatory files plus at most one first difference; no partial success."""
    _identity(tag, rank)
    if not 0 < len(raw) <= MAX_LEDGER_BYTES:
        raise ValueError("history original ledger budget exceeded")
    values = json.loads(raw)
    prefix = f"results/{tag}/workers/rank{rank}/"
    if not isinstance(values, list) or any(not isinstance(r, dict) for r in values):
        raise ValueError("history original ledger schema differs")
    names = [r.get("name") for r in values]
    if any(not isinstance(name, str) or not name.startswith(prefix) for name in names):
        raise ValueError("history original receipt prefix differs")
    relative = [name[len(prefix):] for name in names]
    if (len(set(relative)) != len(relative) or not set(FILES) <= set(relative)
            or not set(relative) <= set((*FILES, *FIRST_DIFFERENCES))
            or len(set(relative) & set(FIRST_DIFFERENCES)) > 1):
        raise ValueError("history incomplete/duplicate/extra original inventory")
    for receipt in values:
        if (set(receipt) != {"name", "generation", "size", "crc32c", "original_sha256"}
                or type(receipt["size"]) is not int or receipt["size"] < 0
                or not isinstance(receipt["generation"], str) or not receipt["generation"].isdecimal()
                or int(receipt["generation"]) <= 0
                or not isinstance(receipt["original_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt["original_sha256"]) is None
                or not isinstance(receipt["crc32c"], str)
                or re.fullmatch(r"[A-Za-z0-9+/]{6}==", receipt["crc32c"]) is None):
            raise ValueError("history receipt generation/size/digest invalid")
    _totals([*(zip(relative, (r["size"] for r in values))), (LEDGER, len(raw))])
    return values


def _remote(bucket: Any, receipt: Mapping) -> Any:
    generation = int(receipt["generation"])
    blob = bucket.blob(receipt["name"], generation=generation)
    blob.reload(if_generation_match=generation)
    if (int(blob.generation) != generation or blob.size != receipt["size"]
            or blob.crc32c != receipt["crc32c"]):
        raise ValueError("history remote original generation/size/CRC differs")
    return blob


def _ledger(bucket: Any, *, tag: str, rank: int, source: Mapping | None = None) -> tuple[dict, bytes, list[dict]]:
    name = f"results/{tag}/workers/rank{rank}/{LEDGER}"
    if source is None:
        blob = bucket.get_blob(name)
    else:
        if (set(source) != {"name", "generation", "size", "crc32c", "sha256"}
                or source["name"] != name or not isinstance(source["generation"], str)
                or not source["generation"].isdecimal() or int(source["generation"]) <= 0):
            raise ValueError("history original ledger source differs")
        blob = _remote(bucket, source)
    if (blob is None or not str(blob.generation).isdecimal() or int(blob.generation) <= 0
            or type(blob.size) is not int
            or not 0 < blob.size <= MAX_LEDGER_BYTES):
        raise ValueError("history missing/oversized original ledger")
    raw = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum="crc32c")
    actual = dict(name=name, generation=str(blob.generation), size=len(raw), crc32c=blob.crc32c,
                  sha256=sha256(raw).hexdigest())
    if (len(raw) != blob.size
            or base64.b64encode(google_crc32c.Checksum(raw).digest()).decode() != blob.crc32c
            or (source is not None and actual != source)):
        raise ValueError("history ledger generation/size/CRC/SHA differs")
    return actual, raw, _receipts(raw, tag=tag, rank=rank)


def _path(rankroot: Path, receipt: Mapping, *, tag: str, rank: int) -> Path:
    # _receipts already required the exact registered suffix, not just basename.
    relative = receipt["name"][len(f"results/{tag}/workers/rank{rank}/"):]
    _kind(relative)
    path = rankroot / relative
    _plain_path(path)
    return path


def _local(path: Path, receipt: Mapping) -> None:
    _regular(path)
    if digest_file(path) != dict(size=receipt["size"], sha256=receipt["original_sha256"], crc32c=receipt["crc32c"]):
        raise ValueError("history local original differs from receipt")


def _aggregate(records: list[dict], verdict: dict, *, tag: str, pin: str, original_bytes: int) -> dict:
    if (not isinstance(verdict, dict) or verdict.get("reproduced") is not True
            or verdict.get("numerical_promotion") is not False or verdict.get("performance_claim") is not False
            or len(json.dumps(verdict, allow_nan=False).encode()) > MAX_CONTROLLER_LEDGER_BYTES):
        raise ValueError("history aggregate requires compact original-byte reproduction")
    if len(records) != 8:
        raise ValueError("history aggregate requires all eight original records")
    for rank, record in enumerate(records):
        _record_identity(record, tag=tag, rank=rank, pin=pin)
        entries = record.get("call_evidence")
        if (record.get("call_evidence_layout") != calls.SCHEMA or not isinstance(entries, list)
                or len(entries) != calls.MAX_CALLS
                or any(not isinstance(e, dict) or set(e) != {"phase", "graph", "completed", "original"}
                       for e in entries)):
            raise ValueError("history aggregate refuses expanded/incomplete call evidence")
    result = dict(workers=records, diagnostic=verdict, original_bytes=original_bytes,
        protocol=protocol.PROTOCOL, kernel=protocol.KERNEL, profile=admission.PROFILE,
        tag=tag, code_hash=pin, status="SUCCESS", diagnostic_only=True,
        numerical_promotion=False, performance_claim=False, admission_only=False,
        baseline_only=False, boundary_diagnostic=True, compile_only=False,
        warmup=0, iterations=0, latency=None, profiler_free_timing=False,
        device_kind="TPU v4", selected_route_case=None,
        comparison=dict(passed=None, diagnostic_evidence_complete=True),
        hlo=dict(sha256=records[0]["programs"]["candidate_b128"]["optimized_hlo_sha256"],
                 contract=dict(passed=True, scope="ACTUAL_HISTORY_NINE_GRAPH_DIAGNOSTIC_HLO")),
        checksum=sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest())
    if len((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()) > 64 << 20:
        raise ValueError("history compact aggregate exceeds controller budget")
    return result


def collect(*, tag: str, pin: str, root: Path, repo: Path, original_root: Path,
            client: Any, local_rank0_root: Path, validate_fleet: Validator) -> dict:
    """All eight bounded ledgers first; exact nested payloads and local rank0 reuse."""
    _identity(tag, 0)
    if not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{40}", pin) is None or not callable(validate_fleet):
        raise ValueError("history collector pin/validator invalid")
    _root(root, RUN_ROOT / tag / "fleet", present=False)
    _root(local_rank0_root, RUN_ROOT / tag / "rank0", present=True)
    bucket = _bucket(client)
    ledgers = [_ledger(bucket, tag=tag, rank=rank) for rank in range(8)]
    total = sum(len(raw) + sum(r["size"] for r in receipts) for _, raw, receipts in ledgers)
    if total > MAX_FLEET_BYTES:
        raise ValueError("history fleet original budget exceeded")
    # Reuse is counted as zero new payload bytes only after actual local bytes
    # and the pinned remote generation have both been authenticated.
    for receipt in ledgers[0][2]:
        _remote(bucket, receipt)
        _local(_path(local_rank0_root, receipt, tag=tag, rank=0), receipt)
    reused = sum(r["size"] for r in ledgers[0][2])
    if shutil.disk_usage(root.parent).free < total - reused + 8 * MAX_LEDGER_BYTES + DISK_RESERVE:
        raise ValueError("history collector lacks original-bytes headroom")
    root.mkdir(exist_ok=False)
    records = []
    for rank, (source, raw, receipts) in enumerate(ledgers):
        destination = root / f"rank{rank}"
        destination.mkdir(exist_ok=False)
        with (destination / LEDGER).open("xb") as stream:
            stream.write(raw)
        _atomic_json(destination / "ledger_source.json", source)
        for receipt in receipts:
            blob = _remote(bucket, receipt)
            path = _path(destination, receipt, tag=tag, rank=rank)
            if rank == 0:
                local = _path(local_rank0_root, receipt, tag=tag, rank=0)
                _local(local, receipt)
                _link_exact(local, path)
            else:
                _atomic_download(blob, path)
            _regular(path)
            _require_blob_identity(blob, path, receipt["original_sha256"])
        record = json.loads((destination / "runner.json").read_bytes())
        _record_identity(record, tag=tag, rank=rank, pin=pin)
        records.append(record)
    verdict = validate_fleet(root=root, records=records, repo=repo, original_root=original_root)
    return _aggregate(records, verdict, tag=tag, pin=pin, original_bytes=total)


def _collected(*, tag: str, pin: str, root: Path, bucket: Any = None) -> tuple[list[dict], int]:
    _identity(tag, 0)
    _root(root, RUN_ROOT / tag / "fleet", present=True)
    if not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{40}", pin) is None:
        raise ValueError("history collected code pin invalid")
    records, total = [], 0
    for rank in range(8):
        rankroot = root / f"rank{rank}"
        _plain_path(rankroot)
        for name in (LEDGER, "ledger_source.json"):
            path = rankroot / name
            _regular(path)
            if path.stat().st_size > MAX_LEDGER_BYTES:
                raise ValueError("history collected ledger budget exceeded")
        raw = (rankroot / LEDGER).read_bytes()
        source = json.loads((rankroot / "ledger_source.json").read_bytes())
        receipts = _receipts(raw, tag=tag, rank=rank)
        if (set(source) != {"name", "generation", "size", "crc32c", "sha256"}
                or not isinstance(source.get("generation"), str)
                or not source["generation"].isdecimal() or int(source["generation"]) <= 0
                or type(source.get("size")) is not int
                or source.get("name") != f"results/{tag}/workers/rank{rank}/{LEDGER}"
                or source.get("size") != len(raw) or source.get("sha256") != sha256(raw).hexdigest()
                or source.get("crc32c") != base64.b64encode(google_crc32c.Checksum(raw).digest()).decode()):
            raise ValueError("history collected ledger identity differs")
        if bucket is not None and _ledger(bucket, tag=tag, rank=rank, source=source)[1] != raw:
            raise ValueError("history recovered ledger bytes differ")
        total += len(raw) + sum(r["size"] for r in receipts)
        for receipt in receipts:
            if bucket is not None:
                _remote(bucket, receipt)
            _local(_path(rankroot, receipt, tag=tag, rank=rank), receipt)
        record = json.loads((rankroot / "runner.json").read_bytes())
        _record_identity(record, tag=tag, rank=rank, pin=pin)
        records.append(record)
    if total > MAX_FLEET_BYTES:
        raise ValueError("history collected fleet budget exceeded")
    return records, total


def replay_collected(*, tag: str, pin: str, root: Path, repo: Path, original_root: Path,
                     client: Any, validate_fleet: Validator) -> dict:
    """Reauthenticate retained generations and local bytes, without second copies."""
    records, total = _collected(tag=tag, pin=pin, root=root, bucket=_bucket(client))
    verdict = validate_fleet(root=root, records=records, repo=repo, original_root=original_root)
    return _aggregate(records, verdict, tag=tag, pin=pin, original_bytes=total)


def validate_record(record: dict, pin: str, *, root: Path, repo: Path,
                    original_root: Path, validate_fleet: Validator) -> None:
    """DB validation rehashes local originals and reruns independent collection."""
    tag = record.get("tag")
    records, total = _collected(tag=tag, pin=pin, root=root)
    verdict = validate_fleet(root=root, records=records, repo=repo, original_root=original_root)
    same_json(record, _aggregate(records, verdict, tag=tag, pin=pin, original_bytes=total),
              "history independent aggregate")


def archive_inventory(root: Path) -> list[Path]:
    """Four GiB workers plus bounded controller tree fit the ten GiB whole prefix."""
    _identity(root.name, 0)
    _root(root, RUN_ROOT / root.name, present=True)
    files, total, count, extra, ranks = [], 0, 0, 0, {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError("history archive linked/nonregular path refused")
        if stat.S_ISDIR(mode):
            continue
        count += 1
        total += path.stat().st_size
        if (count > MAX_ARCHIVE_FILES
                or total > MAX_ARCHIVE_BYTES - MAX_FLEET_BYTES - CONTROLLER_METADATA_RESERVE):
            raise ValueError("history controller archive budget exceeded; originals remain local")
        relative = path.relative_to(root)
        if "retained_reference" in relative.parts:
            continue  # The original generation-bound branch objects already exist.
        parts = relative.parts
        rank_at = 1 if parts[0] == "fleet" else 0
        if len(parts) > rank_at + 1 and re.fullmatch(r"rank[0-7]", parts[rank_at]):
            rankroot = root.joinpath(*parts[:rank_at + 1])
            name = "/".join(parts[rank_at + 1:])
            # A collector may have stopped while downloading one bounded file.
            if name.endswith(".partial"):
                name = name.removesuffix(".partial")
            temporary = re.fullmatch(r"\.([^/]+)\.tmp\.[0-9]+", name)
            if temporary and temporary[1] in (*METADATA_LIMITS, *PUBLICATION_FILES):
                name = temporary[1]
            ranks.setdefault(rankroot, []).append((name, path.stat().st_size))
        else:
            extra += path.stat().st_size
            if extra > MAX_CONTROLLER_EXTRA_BYTES - CONTROLLER_METADATA_RESERVE:
                raise ValueError("history controller auxiliary budget exceeded")
        if path.name in ("archive_receipts.json", "failure_archive_receipts.json"):
            if path.stat().st_size > MAX_CONTROLLER_LEDGER_BYTES:
                raise ValueError("history controller ledger budget exceeded")
        if path.name not in ("SUCCESS", "archive_receipts.json", "failure_archive_receipts.json"):
            files.append(path)
    for rankroot, sizes in ranks.items():
        _totals(sizes)
        if sum(size for _, size in sizes) + _reference_bytes(rankroot) > MAX_LOCAL_RANK_BYTES:
            raise ValueError("history local rank budget exceeded")
    return files


def publish_controller_failure(root: Path, *, client: Any) -> None:
    """Reuse normal exact object names; only the evolving log gets a failure name."""
    files = archive_inventory(root)
    bucket, receipts = _bucket(client), []
    for path in files:
        relative = str(path.relative_to(root))
        facts = digest_file(path)
        if relative == "orchestrator.log":
            if facts["size"] > 512 << 10:
                raise ValueError("history failure log exceeds metadata reserve")
            relative = "failure_orchestrator.log"
        receipts.append(publish_exact(bucket, f"results/{root.name}/{relative}", path, facts, compressed=False))
    ledger = root / "failure_archive_receipts.json"
    _atomic_json(ledger, receipts)
    facts = digest_file(ledger)
    if facts["size"] > MAX_CONTROLLER_LEDGER_BYTES:
        raise ValueError("history failure archive ledger exceeds metadata reserve")
    publish_exact(bucket, f"results/{root.name}/{ledger.name}", ledger, facts, compressed=False)
