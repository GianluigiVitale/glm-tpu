"""Exercise the full prefill-to-batch host path with independent prompt lengths."""
from types import SimpleNamespace

import numpy as np
import pytest

from glm_tpu.models.glm_moe_dsa._s3_batched_decode import BatchedDecodeResult
from glm_tpu.runner._s3_batched_runtime import generate_batch
from glm_tpu.engine import request


def test_prefill_commits_each_history_before_shared_decode(monkeypatch):
    from glm_tpu.runner import _s3_batched_runtime as module
    monkeypatch.setattr(module.jax,'block_until_ready',lambda x:x)
    monkeypatch.setattr('jax.experimental.multihost_utils.process_allgather',lambda x:np.tile(x,(8,1)))
    monkeypatch.setattr(module.pre,'finish_ws32_batched_prefill',lambda out:(out.state.decoder,out.next_token))
    values=[request.from_token_ids([7]*length,request_id=f'lane{i}',max_new_tokens=4,
        context_capacity=32768) for i,length in enumerate((129,257,7))]
    admissions=[];inserts=[];decodes=[];events=[];ticks=[0.]
    def initialize(length):
        return SimpleNamespace(decoder=SimpleNamespace(position=0,contract_valid=np.array([True])))
    def prefill(block,count,fresh,*weights):
        ticks[0]+=.1
        assert np.all(block[int(count):]==-1)
        fresh.decoder.position+=int(count)
        return SimpleNamespace(state=fresh,next_token=np.array([9],np.int32))
    def insert(bank,one,index):
        inserts.append(int(index));bank[int(index)]=one.position
        return bank
    def decode(tokens,bank,*rest):
        active=rest[-1];decodes.append(active.copy());ticks[0]+=1
        assert bank.tolist()==[129,257,7]
        status=np.array([[values[i]['eos_ids'][0],1,int(p)+1,int(p)+2] for i,p in enumerate(bank)],np.int32)
        return BatchedDecodeResult(bank,tokens,status)
    r=SimpleNamespace(concurrent_size=3,capacity=32768,active=False,
        require=lambda ok,message:None if ok else (_ for _ in ()).throw(RuntimeError(message)),
        vote=lambda x:bool(x),put=lambda x:x,admit=admissions.append,
        initialize_batch=lambda lengths:np.zeros(3,np.int32),initialize=initialize,
        prefill={114:prefill,128:prefill},insert_batch=insert,decode_batch=decode,
        weights=None,wk=None,rope=None,phase=lambda name,fn:fn(),stats=lambda:[],
        record={'requests':[]},save=lambda x:None)
    results,aggregate=generate_batch(r,values,deliver=lambda *event:events.append(event),
                                    deadline=100,clock=lambda:ticks[0])
    assert inserts==[0,1,2] and len(decodes)==1 and len(events)==6
    assert all(len(tokens)==2 and report['finish_reason']=='eos' for tokens,report in results)
    assert aggregate['decode_rounds']==1 and aggregate['aggregate_decode_tokens_per_second']==3
    assert admissions.count('cache_init')==3 and admissions.count('batch_cache_init')==1
    assert admissions[-1]=='batch_decode' and not r.active
    r.active=True
    with pytest.raises(RuntimeError,match='active or failed'):
        generate_batch(r,values,deliver=lambda *args:None,deadline=100)
    assert len(decodes)==1
