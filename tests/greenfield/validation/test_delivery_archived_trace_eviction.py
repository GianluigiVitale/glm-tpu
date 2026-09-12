"""Only the exact archived local copies may reach the shared eviction engine."""
from copy import deepcopy

import pytest

from scripts.greenfield import evict_delivery_resume_copies as copies


def manifest():
    pairs = copies.scope("archived-traces")
    return dict(artifact_kind=copies.TRACE_KIND, status="VERIFIED_NOT_DELETED",
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


@pytest.mark.parametrize("key,value", [
    ("path", "/home/gianl/glm-tpu/bench/results.db"),
    ("name", "models/GLM-5.2-FP8/weights.safetensors"),
    ("size", 1), ("nlink", 2), ("restore_uri", "gs://another-bucket/file#1"),
])
def test_modified_target_refused(key, value):
    candidate = manifest()
    candidate["files"][0][key] = value
    with pytest.raises(ValueError):
        copies.validate_manifest(candidate)


@pytest.mark.parametrize("mutation", ["extra", "missing", "kind", "region", "total"])
def test_incomplete_or_widened_scope_refused(mutation):
    value = deepcopy(manifest())
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
