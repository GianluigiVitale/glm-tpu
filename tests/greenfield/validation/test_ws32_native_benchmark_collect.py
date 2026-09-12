"""Real bounded request-byte publication/recollection using in-memory GCS."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace as NS

import pytest

from scripts.greenfield import ws32_native_benchmark_collect as collect
from tests.greenfield.validation.test_ws32_dense_frontier_transport import Bucket

TAG = "greenfield_ws32_native_benchmark_20260912T220000000000000Z"
PIN = "a" * 40


@pytest.fixture
def case(tmp_path,monkeypatch):
    monkeypatch.setattr(collect,"require_local_idle",lambda:None)
    monkeypatch.setattr(collect.shutil,"disk_usage",lambda _:NS(free=20<<30))
    root=tmp_path/TAG
    data={"ended.rank0.json":b'{"worker_exit_code":0}',"runner.rank0.log":b"",
          "sessions.rank0/item000/tokens.jsonl":b'{"token_id":5}\n',
          "sessions.rank0/item000/answer.txt.gz":gzip.compress(b"reasoning</think>A"),
          "native_trace.rank0/plugins/profile/run/host.xplane.pb":b"trace fixture"}
    for name,raw in data.items(): collect._write_once(root/name,raw)
    bucket=Bucket()
    manifest=collect.publish(root,TAG,PIN,0,bucket.client())
    destination=tmp_path/"collected"
    destination.mkdir()
    return NS(root=root,data=data,bucket=bucket,manifest=manifest,destination=destination)


def restore(case):
    return collect.collect(case.destination,TAG,PIN,0,case.bucket.client(),
        {name:case.bucket.get_blob(name) for name in case.bucket.objects})


def test_complete_byte_roundtrip_and_idempotence_including_empty_log(case):
    before=deepcopy(case.bucket.objects)
    assert collect.publish(case.root,TAG,PIN,0,case.bucket.client()) == case.manifest
    assert restore(case) == case.manifest and restore(case) == case.manifest
    assert case.bucket.objects == before
    assert all((case.destination/name).read_bytes()==raw for name,raw in case.data.items())
    assert not case.manifest["quality_proven"]


@pytest.mark.parametrize("change",["generation","bytes","hash","traversal","total","extra","rank","local","region"])
def test_damage_refused_without_overwriting_originals(case,change):
    m=deepcopy(case.manifest)
    r=m["files"][0]
    if change=="generation":r["generation"]="0"
    elif change=="bytes":r["original_bytes"]+=1
    elif change=="hash":r["original_sha256"]="0"*64
    elif change=="traversal":r["relative_path"]="../outside"
    elif change=="total":m["original_bytes"]+=1
    elif change=="rank":m["rank"]=True
    elif change=="extra":case.bucket.objects[collect.prefix(TAG,0)+"other"]=(10000,b"extra")
    elif change=="region":case.bucket.location="EU"
    else: collect._write_once(case.destination/r["relative_path"],b"preserve me")
    name=collect.prefix(TAG,0)+"manifest.json"
    generation,_=case.bucket.objects[name]
    case.bucket.objects[name]=(generation,json.dumps(m).encode())
    with pytest.raises((ValueError,RuntimeError)):restore(case)
    assert all((case.root/name).read_bytes()==raw for name,raw in case.data.items())


def test_cold_failure_does_not_suppress_answer_publication(case,monkeypatch):
    (case.root/"native.rank0").mkdir()
    monkeypatch.setattr(collect.cold,"publish_rank",lambda **k:(_ for _ in ()).throw(ValueError("cold failure")))
    calls=[]
    monkeypatch.setattr(collect,"publish",lambda *a:calls.append(a) or case.manifest)
    with pytest.raises(RuntimeError,match="cold failure"):
        collect.publish_all(case.root,TAG,PIN,0,case.bucket.client())
    assert len(calls)==1
