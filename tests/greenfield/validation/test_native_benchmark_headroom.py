"""Exact scope only; production inode/CRC/holder/unlink engine is reused."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.greenfield import evict_native_benchmark_headroom as scope


def manifest():
    rows=json.loads(scope.SCOPE.read_bytes())
    for row in rows:
        row.update(nlink=1,generation="123",
            restore_uri=f"gs://{scope.engine.BUCKET}/{row['name']}#123")
    return dict(artifact_kind=scope.KIND,status="VERIFIED_NOT_DELETED",cloud_objects_deleted=0,
        bucket_location="US-CENTRAL2",files=rows,total_bytes=sum(r["size"] for r in rows))


def test_only_reviewed_archived_graph_copies_seven_peer_traces_and_nonprimary_db():
    value=manifest()
    assert len(scope.validate_manifest(value))==94
    assert value["total_bytes"]==4730052223
    for row in value["files"]:
        path=Path(row["path"])
        assert path.is_relative_to(scope.engine.LOCAL_ROOT)
        assert path.name != "trace.rank0.xplane.pb"
        assert "/checkpoints/" not in row["name"] and "/models/" not in row["name"]
        if path.name=="results.db":assert path.parent!=Path('/home/gianl/glm-tpu/bench')


@pytest.mark.parametrize("change",["path","size","uri","link","missing","duplicate","region","total"])
def test_changed_scope_refuses(change):
    value=deepcopy(manifest())
    row=value["files"][0]
    if change=="path":row["path"]="/home/gianl/glm-tpu/bench/results.db"
    elif change=="size":row["size"]+=1
    elif change=="uri":row["restore_uri"]+="0"
    elif change=="link":row["nlink"]=2
    elif change=="missing":value["files"].pop()
    elif change=="duplicate":value["files"][-1]=row
    elif change=="region":value["bucket_location"]="EU"
    else:value["total_bytes"]+=1
    with pytest.raises(ValueError):scope.validate_manifest(value)
