"""Upload-only recovery with actual private files and explicit fake SSH receipts."""

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from glm_tpu import user_request
from scripts.release import ws32_user_publication_recovery as recovery

TAG = "greenfield_ws32_user_request_20260914T020000000000000Z"
PIN = "a" * 40


@pytest.fixture
def case(tmp_path):
    root = tmp_path / TAG
    root.mkdir(mode=0o700)
    request = user_request.from_token_ids(
        [1, 2], request_id="fixture", seed=1, max_new_tokens=3
    )
    raw = user_request.canonical(request) + b"\n"
    (root / "request.json").write_bytes(raw)
    args = SimpleNamespace(
        tag=TAG, code_hash=PIN, request_file_sha256=sha256(raw).hexdigest()
    )
    owners, publication = [], []
    for rank in range(8):
        owner = dict(
            rank=rank,
            host=f"fixture-w-{rank}",
            boot_id="boot",
            tag=TAG,
            pin=PIN,
            processes=[dict(pid=100 + rank)],
        )
        identity = dict(
            tag=TAG,
            code_hash=PIN,
            rank=rank,
            request_file_sha256=args.request_file_sha256,
            worker_exit_code=0,
        )
        ended = dict(
            identity,
            host=owner["host"],
            boot_id="boot",
            supervisor_pid=200 + rank,
            worker_started=True,
            worker_error_type=None,
        )
        published = dict(
            identity, publish_exit_code=1 if rank in (2, 7) else 0, error="fixture"
        )
        owners.append(owner)
        publication.append(dict(ended=ended, published=published))
    calls = []

    def command(value_args, role):
        assert value_args is args and role == "publish"
        return "FIXTURE_PUBLISH_ORIGINALS_ONLY"

    def ssh(value, *, workers, timeout):
        assert value == "FIXTURE_PUBLISH_ORIGINALS_ONLY" and timeout == 1200
        calls.append(int(workers))
        return json.dumps(
            dict(
                schema="glm_ws32_user_publication_v1",
                tag=TAG,
                code_hash=PIN,
                rank=int(workers),
                request_file_sha256=args.request_file_sha256,
                request_sha256=request["request_sha256"],
                cold_present=True,
                request_files=12,
                benchmark=False,
                protected_result_sealed=False,
            )
        )

    return SimpleNamespace(
        root=root,
        args=args,
        owners=owners,
        publication=publication,
        command=command,
        ssh=ssh,
        calls=calls,
    )


def run(case, **overrides):
    return recovery.recover(
        case.root,
        case.args,
        case.owners,
        case.publication,
        ssh=overrides.get("ssh", case.ssh),
        command=case.command,
    )


def test_only_failed_uploads_retried_original_markers_unchanged(case):
    originals = deepcopy(case.publication)
    result = run(case)
    assert case.calls == [2, 7] and case.publication == originals
    assert result["original_publication"] == originals
    assert result["failed_ranks"] == [2, 7]
    assert not result["model_rerun"] and not result["original_markers_replaced"]
    assert not result["protected_result_sealed"]
    assert (case.root / "publication_recovery.json").stat().st_mode & 0o077 == 0
    assert run(case) == result  # idempotent original retry, not regenerated output
    assert case.publication == originals


@pytest.mark.parametrize(
    "fault",
    [
        "request",
        "owner",
        "missing_pid",
        "worker_failed",
        "never_started",
        "wrong_request",
        "noninteger_exit",
        "different_rank",
        "no_failure",
    ],
)
def test_refuses_before_any_upload(case, fault):
    if fault == "request":
        case.args.request_file_sha256 = "c" * 64
    elif fault == "owner":
        case.owners[7]["boot_id"] = "other"
    elif fault == "missing_pid":
        case.owners[7]["processes"] = []
    elif fault == "worker_failed":
        case.publication[7]["ended"]["worker_exit_code"] = 1
    elif fault == "never_started":
        case.publication[7]["ended"]["worker_started"] = False
    elif fault == "wrong_request":
        case.publication[7]["published"]["request_file_sha256"] = "c" * 64
    elif fault == "noninteger_exit":
        case.publication[7]["published"]["publish_exit_code"] = True
    elif fault == "different_rank":
        case.publication[7]["published"]["rank"] = 2
    else:
        for row in case.publication:
            row["published"]["publish_exit_code"] = 0
    with pytest.raises(ValueError):
        run(case)
    assert not case.calls and not (case.root / "publication_recovery.json").exists()


@pytest.mark.parametrize("fault", ["timeout", "bad_json", "wrong_rank", "missing_cold"])
def test_ambiguous_or_invalid_upload_never_makes_recovery_receipt(case, fault):
    def ssh(*args, **kwargs):
        if fault == "timeout":
            raise TimeoutError("observation lost after possible upload")
        if fault == "bad_json":
            return "not a receipt"
        value = json.loads(case.ssh(*args, **kwargs))
        (
            value.update(rank=0)
            if fault == "wrong_rank"
            else value.update(cold_present=False)
        )
        return json.dumps(value)

    with pytest.raises((TimeoutError, ValueError)):
        run(case, ssh=ssh)
    assert not (case.root / "publication_recovery.json").exists()
    assert not any(case.root.glob("*SEALED*"))
