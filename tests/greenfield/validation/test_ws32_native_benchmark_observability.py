"""Original DSA/cache validators in the advancing host loop; fixture math only."""
from types import SimpleNamespace as NS
from copy import deepcopy
import gzip
import json

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_native_benchmark_observability as obs
from scripts.greenfield import ws32_native_benchmark_requests as requests
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DsaObservation, Ws32ObservedDecodeStepResult
from tests.greenfield.validation.test_ws32_native_benchmark_requests import setup, capsule
from tests.greenfield.validation.test_ws32_native_benchmark_memory import record


def run_case(tmp_path,monkeypatch,failure):
    runtime,store,tok,calls,allocations=setup(tmp_path,monkeypatch)
    runtime.decode_config.full_index_slots=tuple(range(21))
    runtime.decode_config.geometry.num_layers=78
    runtime.decode_config.geometry.dsa_indexer_head_dim=128
    runtime.decode_config.packed_cache_width=640
    memory=record()
    previous=runtime.authorize
    def authorize(stage,roots,state):
        # Preserve a fixture census with actual production schema; not real HBM.
        runtime.authorize=previous
        try:
            # Original mock callback also writes; avoid a duplicate original.
            from scripts.greenfield.ws32_native_benchmark_memory import budgets
            value=deepcopy(memory)
            value.update(phase=stage,cache_present=stage!='before_cache')
            value['budgets']=budgets(value)
            store.preserve_memory(stage,value)
        finally:runtime.authorize=authorize
    runtime.authorize=authorize
    positions=np.full((21,1,2048),-1,np.int32)
    scores=np.full((21,1,2048),-np.inf,np.float32)
    positions[:,:,:4]=np.arange(4)
    scores[:,:,:4]=1
    if failure=="dsa":positions[0,0,1]=0
    dsa=Ws32DsaObservation(np.arange(21,dtype=np.int32),positions,np.full((21,1),4,np.int32),scores)
    underlying=runtime.decode_compiled
    def observer(*args):
        return Ws32ObservedDecodeStepResult(underlying(*args),dsa)
    def probe(state):
        return NS(position=state.position-1,
            kv_rows=np.ones((78,640),ml_dtypes.bfloat16),
            index_rows=np.ones((21,128),ml_dtypes.bfloat16),
            contract_valid=np.array([failure!="cache"]))
    counters=[dict(device_id=i,process_index=0,platform="tpu",bytes_in_use=350,
        peak_bytes_in_use=359 if failure=="peak" else 360,bytes_limit=memory["census"]["devices"][i]["memory_stats"]["bytes_limit"]) for i in range(4)]
    monkeypatch.setattr(obs,"capture_identified_device_memory",lambda _:counters)
    monkeypatch.setattr(obs.jax,"device_get",lambda x:x)
    traced=[]
    def start(path):
        root=obs.Path(path)
        root.mkdir()
        (root/"host.xplane.pb").write_bytes(b"fixture not actual XPlane")
        traced.append(path)
    monkeypatch.setattr(obs,"start_device_trace",start)
    monkeypatch.setattr(obs.jax.profiler,"stop_trace",lambda:None)
    loaded=NS(runtime=runtime,compiled=dict(observer=observer,cache_probe=probe),
        record=dict(local_slots=[dict(device_id=i,slot=i) for i in range(4)],jax_process_index=0))
    observations=obs.NativeObservability(loaded,store)
    payload,plan=capsule()
    kwargs=dict(loaded=loaded,payload=payload,plan=plan,store=store,tokenizer=tok,
        deadline=runtime.clock()+120,observations=observations)
    if failure:
        with pytest.raises((ValueError,RuntimeError)):requests.execute_requests(**kwargs)
        assert len(allocations)==1 and (store.root/"item000/dsa.npz").is_file()
        assert not (store.root/"item001").exists()
        assert (store.root/"item000/failure.json").is_file()
    else:
        rows=requests.execute_requests(**kwargs)
        assert len(rows)==228 and len(traced)==1
        assert len([c for c in calls if c[0]=="decode"])==228
        assert all(r["generated_tokens"]==2 and r["correct"] is (True if i<198 else None) for i,r in enumerate(rows))
        assert rows[0]["observations"]["traced_decode_indices"]==[0]
        assert rows[1]["observations"]["traced_decode_indices"]==[]
        assert all(r["observations"]["instrumented_decode_indices"]==[0] for r in rows)
        saved=json.loads(gzip.decompress((store.root/"item000/result.json.gz").read_bytes()))
        assert saved["observations"]["cache"]["position"]==[3]
        assert saved["first_token_delivered_before_decode"]
        return NS(root=tmp_path,store=store,runtime=runtime,payload=payload,plan=plan,
                  tokenizer=tok,rows=rows,memory=memory)


@pytest.mark.parametrize("failure",[None,"dsa","cache","peak"])
def test_advancing_requests_original_witnesses_one_trace_and_no_extra_token(tmp_path,monkeypatch,failure):
    run_case(tmp_path,monkeypatch,failure)


def test_compact_witness_budget_before_compression():
    with pytest.raises(ValueError,match="raw budget"):
        obs.npz_bytes(large=np.zeros(obs.WITNESS_CAP+1,np.uint8))
