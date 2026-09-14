"""Explicit same-tag upload retry, never model execution or marker replacement.

Called by the user controller under BOTH leases after original-owner observation
and a fresh idle census. Original failed publication markers remain immutable.
Successful upload is transport only; collection/replay/DB/archive still follow.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from glm_tpu import user_request
from scripts.release import ws32_user_transport as transport
from scripts.release.ws32_user_result import same


def recover(root: Path, args, owners: list[dict], publication: list[dict], *, ssh, command) -> dict:
    transport.private_directory(root)
    raw = user_request.read_bounded(root / "request.json", user_request.PAYLOAD_CAP)
    same(sha256(raw).hexdigest(), args.request_file_sha256, "recovery original input")
    request = json.loads(raw)
    user_request.validate(request)
    if len(owners) != 8 or len(publication) != 8:
        raise ValueError("publication recovery requires eight original owners and terminal markers")
    failed = []
    # Validate ALL ranks before retrying any upload, including successful ranks.
    for rank, (owner, state) in enumerate(zip(owners, publication, strict=True)):
        if (type(owner["rank"]) is not int or owner["rank"] != rank
                or owner["tag"] != args.tag or owner["pin"] != args.code_hash
                or len(owner["processes"]) != 1):
            raise ValueError("publication recovery lacks original model-process ownership")
        ended, published = state["ended"], state["published"]
        transport.check_ended(ended, tag=args.tag, pin=args.code_hash, rank=rank,
            request_file_sha256=args.request_file_sha256, host=owner["host"], boot_id=owner["boot_id"])
        same({k: ended[k] for k in ("worker_exit_code", "worker_started", "worker_error_type")},
             dict(worker_exit_code=0, worker_started=True, worker_error_type=None), "recovery worker completion")
        expected = dict(tag=args.tag, code_hash=args.code_hash, rank=rank,
            request_file_sha256=args.request_file_sha256, worker_exit_code=0)
        same({k: published[k] for k in expected}, expected, "recovery publication identity")
        if type(published["publish_exit_code"]) is not int:
            raise ValueError("publication recovery requires an explicit original exit code")
        if published["publish_exit_code"] != 0:
            failed.append(rank)
    if not failed:
        raise ValueError("no failed publication to recover")
    receipts = []
    for rank in failed:
        # This role rechecks local idle/source/input and conditionally uploads
        # the same originals. Ambiguous SSH never dispatches a model worker.
        receipt = json.loads(ssh(command(args, "publish"), workers=str(rank), timeout=1200))
        expected = dict(schema="glm_ws32_user_publication_v1", tag=args.tag,
            code_hash=args.code_hash, rank=rank, request_file_sha256=args.request_file_sha256,
            request_sha256=request["request_sha256"], cold_present=True,
            benchmark=False, protected_result_sealed=False)
        same({k: receipt[k] for k in expected}, expected, "recovered upload receipt")
        if type(receipt["request_files"]) is not int or receipt["request_files"] <= 0:
            raise ValueError("recovered user upload contains no originals")
        receipts.append(receipt)
    result = dict(schema="glm_ws32_user_publication_recovery_v1", tag=args.tag,
        code_hash=args.code_hash, request_file_sha256=args.request_file_sha256,
        failed_ranks=failed, original_publication=publication, recovered=receipts,
        model_rerun=False, original_markers_replaced=False, protected_result_sealed=False)
    with transport.private_writes():
        transport.outputs._write_once(root / "publication_recovery.json", user_request.canonical(result)+b"\n")
    return result
