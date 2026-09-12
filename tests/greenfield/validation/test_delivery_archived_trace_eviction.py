"""Only the exact archived local copies may reach the shared eviction engine."""
from copy import deepcopy

import pytest

from scripts.greenfield import evict_delivery_resume_copies as copies


def manifest(profile="archived-traces"):
    pairs = copies.scope(profile)
    return dict(artifact_kind=copies.PROFILE_KINDS[profile], status="VERIFIED_NOT_DELETED",
                cloud_objects_deleted=0, bucket_location="US-CENTRAL2",
                total_bytes=sum(size for _, size in pairs.values()),
                files=[dict(path=p, name=n, size=s, nlink=1, generation="1",
                            restore_uri=f"gs://{copies.engine.BUCKET}/{n}#1")
                       for p, (n, s) in pairs.items()])


def test_fixed_scopes_preserve_historical_compiler_review():
    assert len(copies.scope()) == 31
    assert sum(s for _, s in copies.scope().values()) == 2139085384
    value = manifest()
    assert len(copies.validate_manifest(value)) == 15
    assert value["total_bytes"] == 4379119810
    assert all("checkpoints" not in r["path"] for r in value["files"])
    assert all(not r["path"].endswith("/traces/trace.rank0.xplane.pb") for r in value["files"])
    with pytest.raises(ValueError):
        copies.scope("all")


def test_db617_pp8_scope_retains_latest_rank0_and_uses_real_recovery_prefix():
    value = manifest("db617-pp8-traces")
    assert len(copies.validate_manifest(value)) == 24
    assert value["total_bytes"] == 3570775354
    assert not any(copies.DB617_TAG in r["path"] and
                   r["path"].endswith("trace.rank0.xplane.pb") for r in value["files"])
    old = [r for r in value["files"] if copies.PP8_ORIGINAL in r["path"]]
    assert len(old) == 8
    assert all(copies.PP8_RECOVERY in r["name"] for r in old)
    assert len({r["restore_uri"] for r in value["files"]}) == 16


@pytest.mark.parametrize("profile", ["archived-traces", "db617-pp8-traces", "db618-delivery-copies"])
@pytest.mark.parametrize("key,value", [
    ("path", "/home/gianl/glm-tpu/bench/results.db"),
    ("name", "models/GLM-5.2-FP8/weights.safetensors"),
    ("size", 1), ("nlink", 2), ("restore_uri", "gs://another-bucket/file#1"),
])
def test_modified_target_refused(key, value, profile):
    candidate = manifest(profile)
    candidate["files"][0][key] = value
    with pytest.raises(ValueError):
        copies.validate_manifest(candidate)


@pytest.mark.parametrize("profile", ["archived-traces", "db617-pp8-traces", "db618-delivery-copies"])
@pytest.mark.parametrize("mutation", ["extra", "missing", "kind", "region", "total"])
def test_incomplete_or_widened_scope_refused(mutation, profile):
    value = deepcopy(manifest(profile))
    if mutation == "extra":
        value["files"].append(value["files"][0])
    elif mutation == "missing":
        value["files"].pop()
    elif mutation == "kind":
        value["artifact_kind"] = "unreviewed"
    elif mutation == "region":
        value["bucket_location"] = "EU"
    else:
        value["total_bytes"] += 1
    with pytest.raises(ValueError):
        copies.validate_manifest(value)


def test_db618_scope_preserves_active_weights_primary_db_and_original_partners():
    from pathlib import Path

    value = manifest("db618-delivery-copies")
    assert len(copies.validate_manifest(value)) == 36
    assert value["total_bytes"] == 3914678574
    paths = {r["path"] for r in value["files"]}
    for row in value["files"]:
        assert "checkpoints" not in row["path"]
        assert "/glm-tpu/bench/" not in row["path"]
        partner = copies._retained_copy(Path(row["path"]))
        if partner is not None:
            assert str(partner) not in paths
    assert not any(copies.DB618_TAG in p and p.endswith("trace.rank0.xplane.pb") for p in paths)
    for tag in copies.PHASE_TAGS:
        assert str(copies.engine.LOCAL_ROOT / tag / "rank0/phase_first.npz") not in paths
