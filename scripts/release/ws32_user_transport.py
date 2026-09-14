"""User-only evidence transport over the original bounded storage primitives.

No model dispatch, scorer, database success row or SUCCESS creation. The outer
controller holds both leases and authenticates the original fleet; these helpers
retain partial failures as partials. Benchmark namespace/schema defaults remain
unchanged and cannot consume a user manifest without the explicit user policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from glm_tpu import user_request as request
from scripts.release import ws32_user_worker as worker
from scripts.greenfield import ws32_native_benchmark_transport as cold
from scripts.greenfield import ws32_native_benchmark_collect as outputs
from scripts.greenfield.collect_ws32_worker_evidence import require_local_idle


def approved_bucket(client: Any) -> Any:
    bucket = cold._bucket(client)
    if bucket.soft_delete_policy.retention_duration_seconds != 0:
        raise ValueError(
            "user storage requires the existing disabled soft-delete policy; never change it here"
        )
    return bucket


@contextmanager
def private_writes():
    """Single-threaded controller/collector only; restore the caller's mask."""
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def private_directory(root: Path) -> None:
    request._plain(root)
    if (
        not root.is_dir()
        or root.stat().st_uid != os.geteuid()
        or stat.S_IMODE(root.stat().st_mode) & 0o077
        or root.resolve().is_relative_to(worker.REPO.resolve())
    ):
        raise ValueError(
            "user evidence requires an owner-only directory outside source"
        )


def check_ended(
    value: dict,
    *,
    tag: str,
    pin: str,
    rank: int,
    request_file_sha256: str,
    host: str,
    boot_id: str,
) -> None:
    worker.identity(tag, pin, rank)
    if (
        not isinstance(request_file_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", request_file_sha256) is None
    ):
        raise ValueError("user request file digest is not a SHA256")
    expected = dict(
        tag=tag,
        code_hash=pin,
        rank=rank,
        host=host,
        boot_id=boot_id,
        request_file_sha256=request_file_sha256,
    )
    if (
        type(value) is not dict
        or any(
            type(value.get(k)) is not type(v) or value[k] != v
            for k, v in expected.items()
        )
        or type(value.get("worker_exit_code")) is not int
        or type(value.get("supervisor_pid")) is not int
        or value["supervisor_pid"] <= 0
    ):
        raise ValueError("user ended marker differs from original source/request/owner")


def publish(
    *, root: Path, tag: str, pin: str, rank: int, request_file_sha256: str, client: Any
) -> dict:
    """Preserve both channels after actual local idle; never hide answers on cold failure."""
    import socket

    worker.identity(tag, pin, rank)
    require_local_idle()
    private_directory(root)
    if root != worker.RUN_ROOT / tag:
        raise ValueError("user publication root differs from the original run")
    data = request.read_bounded(root / "request.json", request.PAYLOAD_CAP)
    value = json.loads(data)
    request.validate(value)
    if sha256(data).hexdigest() != request_file_sha256:
        raise ValueError("user publication request changed since launch")
    host = socket.gethostname()
    if not host.endswith("-w-" + str(rank)):
        raise ValueError("user publication is not on the original rank")
    ended = json.loads(request.read_bounded(root / f"ended.rank{rank}.json", 16 << 10))
    check_ended(
        ended,
        tag=tag,
        pin=pin,
        rank=rank,
        request_file_sha256=request_file_sha256,
        host=host,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    approved_bucket(client)
    errors, receipts = {}, {}
    with private_writes():
        if (root / f"native.rank{rank}").exists():
            try:
                receipts["cold"] = cold.publish_rank(
                    run_root=root,
                    tag=tag,
                    pin=pin,
                    rank=rank,
                    client=client,
                    user_request=True,
                )
            except Exception as exc:
                errors["cold"] = type(exc).__name__
        try:
            receipts["requests"] = outputs.publish(
                root, tag, pin, rank, client, user_request=True
            )
        except Exception as exc:
            errors["requests"] = type(exc).__name__
    if errors:
        raise RuntimeError(
            "user original publication incomplete: "
            + json.dumps(errors, sort_keys=True)
        )
    return dict(
        schema="glm_ws32_user_publication_v1",
        tag=tag,
        code_hash=pin,
        rank=rank,
        request_file_sha256=request_file_sha256,
        request_sha256=value["request_sha256"],
        cold_present="cold" in receipts,
        request_files=len(receipts["requests"]["files"]),
        benchmark=False,
        protected_result_sealed=False,
    )


def collect(
    *,
    destination: Path,
    tag: str,
    pin: str,
    request_file_sha256: str,
    original_fleet: list[dict],
    client: Any,
    blobs: dict,
) -> dict:
    """Exact-generation recollection with original owner binding; no model retry.

    An absent cold manifest is retained as an explicit incomplete rank, never cold
    success. Already published request originals are collected first, so a damaged
    cold channel cannot suppress them. The caller must validate request/cold/trace
    semantics separately before sealing, and must retain the original private input.
    """
    worker.identity(tag, pin, 0)
    private_directory(destination)
    approved_bucket(client)
    if (
        len(original_fleet) != 8
        or {r["rank"] for r in original_fleet} != set(range(8))
        or len({r["host"] for r in original_fleet}) != 8
    ):
        raise ValueError(
            "user collection needs the authenticated original eight-host fleet"
        )
    fleet = sorted(original_fleet, key=lambda r: r["rank"])
    for rank, owner in enumerate(fleet):
        if (
            type(owner["rank"]) is not int
            or owner.get("tag") != tag
            or owner.get("pin") != pin
            or not owner["host"].endswith("-w-" + str(rank))
            or not owner.get("boot_id")
        ):
            raise ValueError("user original fleet identity differs")
    request_receipts, cold_receipts, missing_cold = [], [], []
    expected_cold = set()
    with private_writes():
        for rank, owner in enumerate(fleet):
            manifest = outputs.collect(
                destination, tag, pin, rank, client, blobs, user_request=True
            )
            ended = json.loads(
                request.read_bounded(destination / f"ended.rank{rank}.json", 16 << 10)
            )
            check_ended(
                ended,
                tag=tag,
                pin=pin,
                rank=rank,
                request_file_sha256=request_file_sha256,
                host=owner["host"],
                boot_id=owner["boot_id"],
            )
            request_receipts.append(manifest)
        for rank in range(8):
            name = cold.prefix(tag, rank) + "manifest.json"
            if name not in blobs:
                missing_cold.append(rank)
                continue
            receipt = cold.collect_rank(
                destination=destination,
                tag=tag,
                pin=pin,
                rank=rank,
                client=client,
                blobs=blobs,
                user_request=True,
            )
            cold_receipts.append(receipt)
            expected_cold.add(name)
            expected_cold.update(r["name"] for r in receipt["manifest"]["files"])
    if {
        name for name in blobs if name.startswith(f"results/{tag}/native_cold/")
    } != expected_cold:
        raise ValueError("user cold fleet contains extra or unmanifested objects")
    expected_requests = {
        outputs.prefix(tag, rank, user_request=True) + "manifest.json"
        for rank in range(8)
    }
    expected_requests.update(
        r["name"] for manifest in request_receipts for r in manifest["files"]
    )
    if {
        name for name in blobs if name.startswith(f"results/{tag}/user_requests/")
    } != expected_requests:
        raise ValueError("user request fleet contains extra or unmanifested objects")
    return dict(
        schema="glm_ws32_user_collection_v1",
        tag=tag,
        code_hash=pin,
        request_file_sha256=request_file_sha256,
        requests=request_receipts,
        cold=cold_receipts,
        missing_cold_ranks=missing_cold,
        benchmark=False,
        protected_result_sealed=False,
    )
