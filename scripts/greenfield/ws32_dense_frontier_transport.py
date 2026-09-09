"""Bounded exact-generation originals for the protected dense01 diagnostic.

No launcher, model payload, trace or SUCCESS. Old DB604 references are reused,
not uploaded again. The outer wrapper still owns DB, archive and fleet cleanup.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import stat
from typing import Any

from glm_tpu.greenfield.validation.ws32_evidence import (
    _atomic_download,
    _require_blob_identity,
)
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield import ws32_dense_frontier_evidence as evidence
from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol
from scripts.greenfield import ws32_dense_canonical as canonical
from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_evidence import same_json

# Original model capsules <=128MiB, WK capsules <=96MiB, other files <=31MiB;
# reserve the final MiB for bounded omissions and receipt metadata.
MAX_RANK_BYTES = 256 << 20
MAX_LEDGER_BYTES = 64 << 10
MAX_AUX_BYTES = 31 << 20
MAX_FLEET_BYTES = 8 * MAX_RANK_BYTES
# Prospective outer allowance: worker originals + collected/archive copies,
# original-reference reuse, logs and DB snapshot. NOT a new checkpoint budget.
MAX_ARCHIVE_BYTES = 6 << 30
DISK_RESERVE = 1 << 30
CAPSULE_NAMES = ("wide_final", "narrow_32", "narrow_64", "narrow_96", "narrow_128")
WK_NAMES = tuple(
    f"layer{layer}_{name}.npz"
    for layer in (0, 1)
    for name in ("wk_decode", "wk_promote")
)
FILES = (
    "runner.json",
    "retained_preflight.json",
    "compile_journal.jsonl",
    "worker.log",
    *(
        f"{graph}.{form}"
        for graph in ("wk_decode", "wk_promote", "dense01")
        for form in ("stablehlo.mlir", "optimized_hlo.txt")
    ),
    *WK_NAMES,
    *(f"{name}.{form}" for name in CAPSULE_NAMES for form in ("json", "npz")),
    "comparison.json",
)
NOTICE = "publication_omissions.json"
PARTIALS = tuple(name + ".pending" for name in FILES if name.endswith(".npz"))
NORM_CAPSULE_NAMES = (
    *CAPSULE_NAMES,
    *(n + "_norm" for n in CAPSULE_NAMES),
    *("own_" + n for n in CAPSULE_NAMES),
    *(f"cross_{s}" for s in (0, 32, 64, 96)),
)
NORM_FILES = (
    "runner.json",
    "retained_preflight.json",
    "compile_journal.jsonl",
    "worker.log",
    *(
        f"{g}.{f}"
        for g in norm_protocol.PROGRAMS
        for f in ("stablehlo.mlir", "optimized_hlo.txt")
    ),
    *WK_NAMES,
    *(f"{n}.{f}" for n in NORM_CAPSULE_NAMES for f in ("json", "npz")),
    "reproduction.json",
    "own_reproduction.json",
    "cross_comparison.json",
)
CANONICAL_FILES = (
    "runner.json", "retained_preflight.json", "compile_journal.jsonl", "worker.log",
    *(f"{g}.{f}" for g in canonical.PROGRAMS for f in ("stablehlo.mlir", "optimized_hlo.txt")),
    *WK_NAMES,
    *(f"{canonical.CAPSULE}.{f}" for f in ("json", "npz")),
    "comparison.json",
)


def is_tag(tag: str) -> bool:
    return protocol.is_tag(tag) or norm_protocol.is_tag(tag) or canonical.is_tag(tag)


def files_for_tag(tag: str) -> tuple[str, ...]:
    if not is_tag(tag):
        raise ValueError("invalid dense publication tag")
    return CANONICAL_FILES if canonical.is_tag(tag) else NORM_FILES if norm_protocol.is_tag(tag) else FILES


def _reference_kwargs(tag: str, root: Path | None) -> dict:
    if (norm_protocol.is_tag(tag) or canonical.is_tag(tag)) != (root is not None):
        raise ValueError("norm transport requires its own retained-original root only")
    return {} if root is None else dict(norm_original_root=root)


def _identity(tag: str, rank: int) -> None:
    protocol.original_names(rank)
    if not is_tag(tag):
        raise ValueError("invalid dense publication tag")


def _bucket(client: Any) -> Any:
    bucket = client.bucket(protocol.BUCKET)
    bucket.reload()
    if str(bucket.location).upper() != "US-CENTRAL2":
        raise ValueError("dense publication bucket must be US-CENTRAL2")
    return bucket


def _kind(name: str) -> str:
    name = name.removesuffix(".pending")
    if name in WK_NAMES:
        return "wk"
    if name in {f"{n}.npz" for n in (*NORM_CAPSULE_NAMES, canonical.CAPSULE)}:
        return "model"
    return "aux"


LIMITS = dict(wk=96 << 20, model=128 << 20, aux=MAX_AUX_BYTES)
NOTE = (
    "Original dense0/1 plus embedding, same physical B128 with128 versus four32 live rows; "
    "four WK and five model calls per host. All32 owners reproduce both DB604 cache branches. "
    "Diagnostic evidence only: not a root cause, 8K correctness, model or performance promotion."
)
NORM_NOTE = (
    "Original first128 dense0/1 norm capture, retained DB605 and DB604 byte reproduction; "
    "four WK, five captures, five own-input suffixes and four cross placements per host. "
    "Diagnostic only: not a root cause, numerical fix,8K correctness or performance promotion."
)
CANONICAL_NOTE = (
    "Dense0/1 canonical placement, four WK and one candidate call per host; "
    "all32 owners reproduce retained DB605 narrow rows, full health and endpoint caches. "
    "Untimed diagnostic only: not token11 causality,8K correctness or performance promotion."
)


def publish_rank(*, tag: str, rank: int, root: Path, client: Any) -> list[dict]:
    """Preserve known complete/partial originals without growing past rank caps.

    Call only after the worker process ended, via the existing EXIT uploader.
    Incomplete or oversized evidence is archived as diagnostic partial data and
    subsequently refused by collect; local originals are never removed here.
    """
    _identity(tag, rank)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("dense original root missing or symlinked")
    bucket = _bucket(client)
    prefix = f"results/{tag}/workers/rank{rank}/"
    receipts, omitted = [], []
    totals = dict(wk=0, model=0, aux=0)
    # save_arrays may fail before its atomic rename. Preserve only these exact
    # known siblings, not arbitrary files or an unbounded directory scan.
    # Their extra inventory always prevents complete collection/promotion.
    files = files_for_tag(tag)
    partials = tuple(n + ".pending" for n in files if n.endswith(".npz"))
    for name in (*files, *partials):
        path = root / name
        if path.is_symlink():
            omitted.append(
                dict(name=name, reason="symlink", originals_retained_locally=True)
            )
            continue
        if not path.exists():
            continue  # Early failure may not have reached this compiler/call.
        if not path.is_file():
            raise ValueError("dense original is not a regular file")
        facts = digest_file(path)
        kind = _kind(name)
        if totals[kind] + facts["size"] > LIMITS[kind]:
            omitted.append(
                dict(
                    name=name,
                    bytes=facts["size"],
                    reason=f"{kind}_budget",
                    originals_retained_locally=True,
                )
            )
            continue
        receipts.append(
            publish_exact(bucket, prefix + name, path, facts, compressed=False)
        )
        totals[kind] += facts["size"]
    if omitted:
        notice = root / NOTICE
        _atomic_json(notice, dict(omitted=omitted, originals_retained_locally=True))
        facts = digest_file(notice)
        if facts["size"] > MAX_LEDGER_BYTES:
            raise ValueError(
                "dense omissions exceed metadata cap; originals remain local"
            )
        receipts.append(
            publish_exact(bucket, prefix + NOTICE, notice, facts, compressed=False)
        )
    ledger = root / "worker_receipts.json"
    _atomic_json(ledger, receipts)
    facts = digest_file(ledger)
    if (
        facts["size"] > MAX_LEDGER_BYTES
        or sum(r["size"] for r in receipts) + facts["size"] > MAX_RANK_BYTES
    ):
        raise ValueError("dense publication exceeded rank budget")
    publish_exact(bucket, prefix + ledger.name, ledger, facts, compressed=False)
    return receipts


def _receipts(raw: bytes, *, tag: str, rank: int) -> list[dict]:
    """Require the COMPLETE fixed set before downloading any new payloads."""
    values = json.loads(raw)
    files = files_for_tag(tag)
    prefix = f"results/{tag}/workers/rank{rank}/"
    if (
        not isinstance(values, list)
        or len(values) != len(files)
        or any(not isinstance(r, dict) for r in values)
        or {r.get("name") for r in values} != {prefix + n for n in files}
    ):
        raise ValueError("dense incomplete/duplicate/extra original inventory")
    totals = dict(wk=0, model=0, aux=0)
    for receipt in values:
        if (
            set(receipt) != {"name", "generation", "size", "crc32c", "original_sha256"}
            or type(receipt["size"]) is not int
            or not 0 <= receipt["size"] < MAX_RANK_BYTES
            or not isinstance(receipt["generation"], str)
            or not receipt["generation"].isdecimal()
            or int(receipt["generation"]) <= 0
            or not isinstance(receipt["original_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt["original_sha256"]) is None
            or not isinstance(receipt["crc32c"], str)
            or re.fullmatch(r"[A-Za-z0-9+/]{6}==", receipt["crc32c"]) is None
        ):
            raise ValueError("dense receipt generation/size/digest invalid")
        totals[_kind(receipt["name"][len(prefix) :])] += receipt["size"]
    if (
        any(totals[k] > LIMITS[k] for k in totals)
        or sum(totals.values()) + len(raw) > MAX_RANK_BYTES
    ):
        raise ValueError("dense original receipt budget exceeded")
    return values


def collect(
    *,
    tag: str,
    pin: str,
    root: Path,
    repo: Path,
    original_root: Path,
    client: Any,
    norm_original_root: Path | None = None,
) -> dict:
    """Preflight all eight ledgers, fetch exact generations, replay all owners.

    ``root`` is a fresh collection directory, not the active worker directory.
    Missing evidence refuses; existing destinations are never replaced.
    """
    _identity(tag, 0)
    reference_kwargs = _reference_kwargs(tag, norm_original_root)
    if re.fullmatch(r"[0-9a-f]{40}", pin) is None:
        raise ValueError("dense collection code pin invalid")
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    bucket = _bucket(client)
    ledgers, total = [], 0
    for rank in range(8):
        blob = bucket.get_blob(f"results/{tag}/workers/rank{rank}/worker_receipts.json")
        if (
            blob is None
            or not blob.generation
            or blob.size is None
            or not 0 < int(blob.size) <= MAX_LEDGER_BYTES
        ):
            raise ValueError("dense missing/oversized ledger")
        raw = blob.download_as_bytes(
            if_generation_match=int(blob.generation), checksum="crc32c"
        )
        import base64
        import google_crc32c

        if (
            len(raw) != int(blob.size)
            or base64.b64encode(google_crc32c.Checksum(raw).digest()).decode()
            != blob.crc32c
        ):
            raise ValueError("dense ledger generation size/CRC differs")
        receipts = _receipts(raw, tag=tag, rank=rank)
        total += len(raw) + sum(r["size"] for r in receipts)
        ledgers.append((blob, raw, receipts))
    if total > MAX_FLEET_BYTES:
        raise ValueError("dense fleet original budget exceeded")
    if shutil.disk_usage(root.parent).free < total + DISK_RESERVE:
        raise ValueError("dense collector lacks original-bytes headroom")
    root.mkdir(exist_ok=False)
    records = []
    for rank, (ledger_blob, raw, receipts) in enumerate(ledgers):
        destination = root / f"rank{rank}"
        destination.mkdir(exist_ok=False)
        (destination / "worker_receipts.json").write_bytes(raw)
        _atomic_json(
            destination / "ledger_source.json",
            dict(
                name=ledger_blob.name,
                generation=str(ledger_blob.generation),
                size=len(raw),
                crc32c=ledger_blob.crc32c,
                sha256=sha256(raw).hexdigest(),
            ),
        )
        for receipt in receipts:
            generation = int(receipt["generation"])
            blob = bucket.blob(receipt["name"], generation=generation)
            blob.reload(if_generation_match=generation)
            if (
                int(blob.generation) != generation
                or int(blob.size) != receipt["size"]
                or blob.crc32c != receipt["crc32c"]
            ):
                raise ValueError("dense payload generation size/CRC differs")
            path = destination / Path(receipt["name"]).name
            _atomic_download(blob, path)
            _require_blob_identity(blob, path, receipt["original_sha256"])
        records.append(json.loads((destination / "runner.json").read_bytes()))
    verdict = evidence.validate_fleet(
        root,
        records,
        pin=pin,
        tag=tag,
        repo=repo,
        original_root=original_root,
        **reference_kwargs,
    )
    return aggregate(records, verdict, tag=tag, pin=pin, original_bytes=total)


def aggregate(
    records: list[dict], verdict: dict, *, tag: str, pin: str, original_bytes: int
) -> dict:
    """Existing wrapper schema: SUCCESS is diagnostic collection, never Gate D."""
    if verdict.get("reproduced") is not True:
        raise ValueError("dense aggregate requires original-byte reproduction")
    norm_mode = norm_protocol.is_tag(tag)
    canonical_mode = canonical.is_tag(tag)
    graph = canonical.GRAPH if canonical_mode else "dense01_norm" if norm_mode else "dense01"
    return dict(
        workers=records,
        diagnostic=verdict,
        original_bytes=original_bytes,
        protocol=canonical.PROTOCOL if canonical_mode else norm_protocol.PROTOCOL if norm_mode else protocol.PROTOCOL,
        kernel=canonical.KERNEL if canonical_mode else norm_protocol.KERNEL if norm_mode else protocol.KERNEL,
        tag=tag,
        code_hash=pin,
        status="SUCCESS",
        diagnostic_only=True,
        numerical_promotion=False,
        performance_claim=False,
        admission_only=False,
        baseline_only=False,
        boundary_diagnostic=True,
        compile_only=False,
        warmup=0,
        iterations=0,
        latency=None,
        profiler_free_timing=False,
        device_kind="TPU v4",
        selected_route_case=None,
        comparison=dict(passed=None, diagnostic_evidence_complete=True),
        hlo=dict(
            sha256=records[0]["programs"][graph]["optimized_hlo_sha256"],
            contract=dict(
                passed=True,
                scope=(
                    "ACTUAL_CANONICAL_DENSE01_AND_WK_DIAGNOSTIC_HLO"
                    if canonical_mode else
                    "ACTUAL_NORM_CAPTURE_SUFFIX_AND_WK_DIAGNOSTIC_HLO"
                    if norm_mode
                    else "ACTUAL_DENSE01_AND_WK_DIAGNOSTIC_HLO"
                ),
            ),
        ),
        checksum=sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    )


def replay_collected(
    *,
    tag: str,
    pin: str,
    root: Path,
    repo: Path,
    original_root: Path,
    client: Any,
    norm_original_root: Path | None = None,
) -> dict:
    """Recover completed originals after a controller-only refusal, no TPU.

    Reauthenticate exact remote ledgers and every local payload against their
    generation/CRC/SHA bindings. Do not replace files or pay for a second copy.
    Fresh collection retains its existing append-only refusal.
    """
    _identity(tag, 0)
    reference_kwargs = _reference_kwargs(tag, norm_original_root)
    if (
        re.fullmatch(r"[0-9a-f]{40}", pin) is None
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise ValueError("dense recovery root/pin invalid")
    bucket = _bucket(client)
    total, records = 0, []
    for rank in range(8):
        rankroot = root / f"rank{rank}"
        if rankroot.is_symlink() or not rankroot.is_dir():
            raise ValueError("dense recovery rank missing/symlinked")
        ledger = rankroot / "worker_receipts.json"
        source_path = rankroot / "ledger_source.json"
        if any(
            p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_LEDGER_BYTES
            for p in (ledger, source_path)
        ):
            raise ValueError("dense recovery ledger missing/oversized/symlinked")
        source = json.loads(source_path.read_bytes())
        name = f"results/{tag}/workers/rank{rank}/worker_receipts.json"
        if (
            source.get("name") != name
            or not isinstance(source.get("generation"), str)
            or not source["generation"].isdecimal()
            or int(source["generation"]) <= 0
        ):
            raise ValueError("dense recovery ledger source differs")
        generation = int(source["generation"])
        blob = bucket.blob(name, generation=generation)
        blob.reload(if_generation_match=generation)
        if not 0 < int(blob.size) <= MAX_LEDGER_BYTES:
            raise ValueError("dense recovery remote ledger oversized")
        raw = blob.download_as_bytes(if_generation_match=generation, checksum="crc32c")
        if raw != ledger.read_bytes() or len(raw) != source["size"]:
            raise ValueError("dense recovery original ledger bytes differ")
        _require_blob_identity(blob, ledger, source["sha256"])
        if blob.crc32c != source["crc32c"]:
            raise ValueError("dense recovery original ledger CRC differs")
        receipts = _receipts(raw, tag=tag, rank=rank)
        total += len(raw) + sum(r["size"] for r in receipts)
        if total > MAX_FLEET_BYTES:
            raise ValueError("dense recovery original budget exceeded")
        for receipt in receipts:
            generation = int(receipt["generation"])
            blob = bucket.blob(receipt["name"], generation=generation)
            blob.reload(if_generation_match=generation)
            if int(blob.size) != receipt["size"] or blob.crc32c != receipt["crc32c"]:
                raise ValueError("dense recovery remote payload identity differs")
            path = rankroot / Path(receipt["name"]).name
            if path.is_symlink() or not path.is_file():
                raise ValueError("dense recovery original missing/symlinked")
            _require_blob_identity(blob, path, receipt["original_sha256"])
        records.append(json.loads((rankroot / "runner.json").read_bytes()))
    verdict = evidence.validate_fleet(
        root,
        records,
        pin=pin,
        tag=tag,
        repo=repo,
        original_root=original_root,
        **reference_kwargs,
    )
    return aggregate(records, verdict, tag=tag, pin=pin, original_bytes=total)


def validate_record(
    record: dict,
    pin: str,
    *,
    root: Path,
    repo: Path,
    original_root: Path,
    norm_original_root: Path | None = None,
) -> None:
    """Recompute original replay for DB accounting, not a verdict-only check."""
    if record.get("code_hash") != pin or not is_tag(record.get("tag")):
        raise ValueError("dense aggregate code/tag identity differs")
    reference_kwargs = _reference_kwargs(record["tag"], norm_original_root)
    records = [
        json.loads((root / f"rank{rank}/runner.json").read_bytes()) for rank in range(8)
    ]
    total = 0
    for rank in range(8):
        rankroot = root / f"rank{rank}"
        raw = (rankroot / "worker_receipts.json").read_bytes()
        if len(raw) > MAX_LEDGER_BYTES:
            raise ValueError("dense collected ledger oversized")
        receipts = _receipts(raw, tag=record["tag"], rank=rank)
        source = json.loads((rankroot / "ledger_source.json").read_bytes())
        if source["sha256"] != sha256(raw).hexdigest() or source["size"] != len(raw):
            raise ValueError("dense collected source ledger differs")
        total += len(raw) + sum(r["size"] for r in receipts)
        for receipt in receipts:
            facts = digest_file(rankroot / Path(receipt["name"]).name)
            if facts != dict(
                size=receipt["size"],
                sha256=receipt["original_sha256"],
                crc32c=receipt["crc32c"],
            ):
                raise ValueError("dense collected original differs from receipt")
    verdict = evidence.validate_fleet(
        root,
        records,
        pin=pin,
        tag=record["tag"],
        repo=repo,
        original_root=original_root,
        **reference_kwargs,
    )
    same_json(
        record,
        aggregate(records, verdict, tag=record["tag"], pin=pin, original_bytes=total),
        "dense independent aggregate",
    )


def archive_inventory(root: Path) -> list[Path]:
    """Bound the WHOLE controller archive in addition to the workers/ prefix.

    The fixed worst-case 2GiB workers plus <=4GiB controller originals/copies,
    old-reference files and DB snapshot stay within the 6GiB total allowance.
    Never follow a symlink or upload an unchecked whole run directory.
    """
    if root.is_symlink() or not root.is_dir() or not is_tag(root.name):
        raise ValueError("dense archive root invalid")
    files, size = [], 0
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("dense archive symlink refused")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("dense archive nonregular file refused")
        size += path.stat().st_size
        if size > MAX_ARCHIVE_BYTES - MAX_FLEET_BYTES - (1 << 20):
            raise ValueError(
                "dense whole archive budget exceeded; originals remain local"
            )
        if path.name not in (
            "SUCCESS",
            "archive_receipts.json",
            "failure_archive_receipts.json",
        ):
            files.append(path)
    return files


def publish_controller_failure(root: Path, *, client: Any) -> None:
    """Same exact archive names (no full-size failure duplicate), no SUCCESS."""
    files = archive_inventory(root)
    bucket = _bucket(client)
    receipts = []
    for path in files:
        relative = str(path.relative_to(root))
        facts = digest_file(path)
        if relative == "orchestrator.log":
            # on_exit appends FAILED after an earlier normal archive may have
            # published this one evolving log. Do not collide with that object
            # or create a second full payload namespace for immutable originals.
            if facts["size"] > 512 << 10:
                raise ValueError("dense failure log exceeds auxiliary reserve")
            relative = "failure_orchestrator.log"
        receipts.append(
            publish_exact(
                bucket, f"results/{root.name}/{relative}", path, facts, compressed=False
            )
        )
    ledger = root / "failure_archive_receipts.json"
    _atomic_json(ledger, receipts)
    if ledger.stat().st_size > 512 << 10:
        raise ValueError("dense failure archive ledger exceeds reserve")
    publish_exact(
        bucket,
        f"results/{root.name}/{ledger.name}",
        ledger,
        digest_file(ledger),
        compressed=False,
    )
