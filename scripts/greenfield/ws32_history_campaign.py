"""History collection glue inside the existing protected selected-layer campaign.

No launcher, model dispatcher, leases or cleanup. The caller completes the existing
eight-host worker/upload phase first. Both original branch histories and bounded
selected-source capsules remain independently replayable for DB/recovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from glm_tpu.greenfield.validation.ws32_evidence import _link_exact
from scripts.greenfield import ws32_history_evidence as evidence
from scripts.greenfield import ws32_history_preflight as preflight
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield import ws32_history_transport as transport

NOTE = (
    "Original8155-token layers0..6 histories, canonical B128/B114 versus retained "
    "live32 control;331 calls per host including8 WK,2 exact materializers,319 "
    "blocks and2 observers. All32 owners must reproduce both original step0 "
    "events before interpreting retained boundary differences. Untimed diagnostic "
    "only: not token11 causality,8K correctness or performance promotion."
)


def _root(root: Path, tag: str) -> None:
    preflight._plain_path(root)
    if not protocol.is_tag(tag) or root != preflight.RUN_ROOT / tag or not root.is_dir():
        raise ValueError("history campaign fixed root/tag differs")


def _references(*, root: Path, repo: Path, client: Any) -> Path:
    """Reuse rank0 references by authenticated hardlink; download only other ranks."""
    destination = root / "references"
    preflight._plain_path(destination)
    destination.mkdir(exist_ok=True)
    for rank in range(8):
        rankroot = destination / f"rank{rank}"
        preflight._plain_path(rankroot)
        rankroot.mkdir(exist_ok=True)
        originals = rankroot / "retained_reference"
        if rank == 0 and not originals.exists():
            source = root / "rank0/retained_reference"
            preflight.load_originals(source, repo=repo, rank=rank)
            originals.mkdir(exist_ok=False)
            for branch in protocol.BRANCHES:
                for form in protocol.ORIGINAL_FORMS:
                    _link_exact(source / f"{branch}.{form}", originals / f"{branch}.{form}")
        preflight.materialize_originals(originals, repo=repo, rank=rank, client=client)
    return destination


def _validator(*, runroot: Path, fetch: Callable | None) -> Callable:
    """Read sources only after every other original identity/replay has passed."""
    def validate(*, root: Path, records: list[dict], repo: Path, original_root: Path) -> dict:
        from scripts.greenfield import ws32_history_source_reader as sources

        reader = None

        def read_source(slot: int, name: str) -> Any:
            nonlocal reader
            if reader is None:
                if fetch is None:
                    reader = sources.load_reader(root=runroot / "source_tiles", repo=repo, records=records)
                else:
                    reader = sources.prepare_reader(root=runroot / "source_tiles", repo=repo,
                                                    records=records, fetch=fetch)
            return reader(slot, name)

        result = evidence.validate_fleet(root=root, records=records, repo=repo,
                                         original_root=original_root, read_source=read_source)
        if reader is None:
            raise ValueError("history collector did not reconstruct selected-source originals")
        return {**result, "selected_source_receipt": reader.receipt}
    return validate


def collect(*, tag: str, pin: str, root: Path, repo: Path, client: Any,
            fetch: Callable) -> dict:
    """Fresh exact-generation collection, then mandatory fixed independent replay."""
    _root(root, tag)
    if not callable(fetch):
        raise ValueError("history selected-source fetch is required")
    if (root / "fleet").exists():
        raise FileExistsError(root / "fleet")
    originals = _references(root=root, repo=repo, client=client)
    return transport.collect(tag=tag, pin=pin, root=root / "fleet", repo=repo,
        original_root=originals, client=client, local_rank0_root=root / "rank0",
        validate_fleet=_validator(runroot=root, fetch=fetch))


def replay_collected(*, tag: str, pin: str, root: Path, repo: Path, client: Any) -> dict:
    """Recovery from complete originals only; no worker rerun or source download."""
    _root(root, tag)
    return transport.replay_collected(tag=tag, pin=pin, root=root / "fleet", repo=repo,
        original_root=root / "references", client=client,
        validate_fleet=_validator(runroot=root, fetch=None))


def validate_record(record: dict, pin: str, *, root: Path, repo: Path) -> None:
    """DB validation reuses retained selected bytes, with no cloud or SSH fallback."""
    _root(root, record.get("tag"))
    transport.validate_record(record, pin, root=root / "fleet", repo=repo,
        original_root=root / "references", validate_fleet=_validator(runroot=root, fetch=None))
