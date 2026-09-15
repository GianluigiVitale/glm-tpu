"""Actual user SQLite/archive transactions with in-memory conditional GCS.

Request/owner/trace replay uses the CPU fixture. Prior cold collection is
represented explicitly, not claimed as real TPU admission or external storage.
"""

from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace as NS

import pytest

from glm_tpu import user_request
from scripts.release import ws32_user_archive as archive
from scripts.release import ws32_user_evidence as evidence
from tests.release.test_user_evidence import case, TAG, PIN
from tests.greenfield.validation.gcs_fixtures import Bucket


@pytest.fixture
def archived_case(case, tmp_path):
    root, _ = case
    report = evidence.replay_collected(root, TAG, PIN)
    bucket = Bucket()
    bucket.soft_delete_policy = NS(retention_duration_seconds=0)
    cold, requests = [], []
    # These prior-collection fixtures test the archive union/readback, not cold
    # model correctness (the real original cold validator has separate tests).
    for rank in range(8):
        for prefix, schema, target in (
            (
                archive.transport.cold.prefix(TAG, rank),
                archive.transport.cold.USER_SCHEMA,
                cold,
            ),
            (
                archive.transport.outputs.prefix(TAG, rank, user_request=True),
                archive.transport.outputs.USER_SCHEMA,
                requests,
            ),
        ):
            source = root / "fixture.json"
            source.write_bytes(user_request.canonical(dict(rank=rank)))
            child = archive.publish_exact(
                bucket,
                prefix + "original.json",
                source,
                archive.digest_file(source),
                compressed=False,
            )
            manifest = dict(
                schema=schema, tag=TAG, code_hash=PIN, rank=rank, files=[child]
            )
            source.write_bytes(user_request.canonical(manifest))
            archive.publish_exact(
                bucket,
                prefix + "manifest.json",
                source,
                archive.digest_file(source),
                compressed=False,
            )
            target.append(dict(manifest=manifest) if target is cold else manifest)
    input_object = archive.publish_exact(
        bucket,
        f"results/{TAG}/private_input/request.json",
        root / "request.json",
        archive.digest_file(root / "request.json"),
        compressed=False,
    )
    launch = json.loads((root / "launch.json").read_bytes())
    launch["input_object"] = input_object
    (root / "launch.json").write_bytes(user_request.canonical(launch))
    collection = dict(
        schema="glm_ws32_user_collection_v1",
        tag=TAG,
        code_hash=PIN,
        request_file_sha256=report["request_file_sha256"],
        missing_cold_ranks=[],
        benchmark=False,
        protected_result_sealed=False,
        requests=requests,
        cold=cold,
    )
    archive.persist(root / "user_replay.json", report)
    archive.persist(root / "user_collection.json", collection)
    for name in ("sync.txt", "prepared.txt"):
        (root / name).write_bytes(b"fixture eight hosts\n")
    return NS(
        root=root,
        bucket=bucket,
        args=dict(
            root=root,
            tag=TAG,
            pin=PIN,
            report=report,
            collection=collection,
            blobs={name: bucket.get_blob(name) for name in bucket.objects},
            client=bucket.client(),
            db_path=tmp_path / "user.sqlite",
        ),
    )


def test_original_user_archive_repeat_no_quality_claim_or_full_db_copy(archived_case):
    case = archived_case
    result = archive.seal(**case.args)
    assert result["protected_result_sealed"] and result["evidence_archived"]
    assert (
        result["quality_score"] is None
        and not result["benchmark"]
        and not result["project_complete"]
    )
    objects = deepcopy(case.bucket.objects)
    assert archive.seal(**case.args) == result and case.bucket.objects == objects
    assert f"results/{TAG}/USER_RESPONSE_SEALED.json" in objects
    assert not any(
        name.endswith("/SUCCESS") or name.endswith("results.db") for name in objects
    )
    with sqlite3.connect(case.args["db_path"]) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM items").fetchone() == (0,)


def test_archive_interruption_keeps_db_and_retry_reuses_originals(
    archived_case, monkeypatch
):
    case = archived_case
    publisher = archive.publish_exact

    def fail(bucket, name, *args, **kwargs):
        if name.endswith("db_link.json"):
            raise RuntimeError("injected publication interruption")
        return publisher(bucket, name, *args, **kwargs)

    monkeypatch.setattr(archive, "publish_exact", fail)
    with pytest.raises(RuntimeError):
        archive.seal(**case.args)
    assert not any(
        name.endswith("USER_RESPONSE_SEALED.json") for name in case.bucket.objects
    )
    monkeypatch.setattr(archive, "publish_exact", publisher)
    assert archive.seal(**case.args)["run_id"] == 1
    with sqlite3.connect(case.args["db_path"]) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone() == (1,)


def test_upload_recovery_record_is_in_bounded_controller_archive(archived_case):
    case = archived_case
    recovery = dict(
        fixture="transport-only recovery, not model evidence", model_rerun=False
    )
    archive.persist(case.root / "publication_recovery.json", recovery)
    result = archive.seal(**case.args)
    name = f"results/{TAG}/controller/publication_recovery.json"
    assert name in case.bucket.objects
    ledger = json.loads((case.root / "archive_ledger.json").read_bytes())
    assert name in {row["name"] for row in ledger["controller"]}
    assert not result["project_complete"] and result["quality_score"] is None


@pytest.mark.parametrize(
    "fault",
    [
        "generation",
        "extra",
        "region",
        "soft_delete",
        "missing_cold",
        "input_generation",
        "owner",
    ],
)
def test_archive_refusal_before_database_and_terminal(archived_case, fault):
    case = archived_case
    if fault in ("generation", "input_generation"):
        name = (
            f"results/{TAG}/private_input/request.json"
            if fault == "input_generation"
            else next(iter(case.bucket.objects))
        )
        generation, data = case.bucket.objects[name]
        case.bucket.objects[name] = (generation + 1000, data)
    elif fault == "extra":
        name = archive.transport.cold.prefix(TAG, 0) + "extra"
        case.bucket.objects[name] = (1, b"extra")
        case.args["blobs"][name] = case.bucket.get_blob(name)
    elif fault == "region":
        case.bucket.location = "EU"
    elif fault == "soft_delete":
        case.bucket.soft_delete_policy.retention_duration_seconds = 604800
    elif fault == "missing_cold":
        case.args["collection"]["missing_cold_ranks"] = [7]
        (case.root / "user_collection.json").write_bytes(
            user_request.canonical(case.args["collection"])
        )
    else:
        path = case.root / "collected/runner.rank7.json"
        value = json.loads(path.read_bytes())
        value["owner"]["pid"] = 999
        path.write_bytes(user_request.canonical(value))
    with pytest.raises(ValueError):
        archive.seal(**case.args)
    assert not case.args["db_path"].exists()
    assert not any(
        name.endswith("USER_RESPONSE_SEALED.json") for name in case.bucket.objects
    )
