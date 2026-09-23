"""Memory-bounded preparation and generation for a shared-weight decode batch."""
from hashlib import sha256
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..greenfield.runtime import ws32_batched_prefill as pre
from .batched_session import BatchedSession
from .request import batch


def compile_batch(runtime, initial, programs):
    """Compile the batch programs of ``runner.programs.build_program_set`` (``programs``: its
    ``BatchPrograms``) with the bank arguments derived from one prefill state (``initial``)."""
    r=runtime;n=r.concurrent_size
    abstract=jax.tree.map(lambda x,s:jax.ShapeDtypeStruct((n,*x.shape),x.dtype,sharding=s),
                          initial.decoder,programs.state_shardings)
    spec=programs.cache_init
    r.initialize_batch=r.compile(spec.name,spec.fn,(r.put(np.ones(n,np.int32)),),model=spec.model)
    spec=programs.insert
    r.insert_batch=r.compile(spec.name,spec.fn,(abstract,initial.decoder,r.put(np.int32(0))),model=spec.model)
    spec=programs.decode
    r.decode_batch=r.compile(spec.name,spec.fn,
        (r.put(np.zeros((n,1),np.int32)),abstract,r.weights,r.rope,r.put(np.ones(n,bool))),model=spec.model)


def generate_batch(r, values, *, deliver, deadline, clock=time.perf_counter):
    payload=batch(values,concurrent=True)
    r.require(len(values)==r.concurrent_size and payload['context_capacity']==r.capacity,
              'batch differs from compiled conversation count/capacity')
    if r.active:raise RuntimeError('optimized runtime has an active or failed request')
    r.active=True
    started=clock()
    def budget():r.require(r.vote(clock()<deadline) is True,'concurrent deadline expired')
    budget();r.admit('batch_cache_init')
    lengths=np.asarray([len(v['prompt_ids']) for v in values],np.int32)
    state=jax.block_until_ready(r.initialize_batch(r.put(lengths)))
    first_tokens=[];metadata=[];prefill_times=[]
    for lane,item in enumerate(values):
        budget()
        r.admit('cache_init')
        fresh=jax.block_until_ready(r.initialize(r.put(lengths[lane])))
        # Admit with the batch bank AND this prefill cache resident.
        for name in ('prefill_128','prefill_114','batch_insert'):r.admit(name)
        ids=np.asarray(item['prompt_ids'],np.int32)
        prefill_started=clock()
        for start in range(0,len(ids),128):
            budget()
            part=ids[start:start+128];rows=114 if len(part)<=114 else 128
            out=jax.block_until_ready(r.prefill[rows](
                r.put(np.pad(part,(0,rows-len(part)),constant_values=-1)),
                r.put(np.int32(len(part))),fresh,r.weights,r.wk,r.rope))
            fresh=out.state
            r.require(r.vote(bool(np.asarray(fresh.decoder.contract_valid).all())) is True,
                      'concurrent prefill failed')
        one,token=r.phase('batch_prefill_finish',lambda:pre.finish_ws32_batched_prefill(out))
        prefill_times.append(clock()-prefill_started)
        first_tokens.append(int(np.asarray(token)[0]))
        metadata.append([first_tokens[-1],1,int(lengths[lane]),int(lengths[lane])+1])
        state=jax.block_until_ready(r.insert_batch(state,one,r.put(np.int32(lane))))
        # Only one disposable prefill state, never eight individual cache copies.
        del fresh,out,one,token
        gc.collect()
    r.admit('batch_decode');budget()
    session=BatchedSession(values,
        decode=lambda t,s,a:r.decode_batch(t,s,r.weights,r.rope,a),put=r.put,vote=r.vote,
        deliver=deliver,deadline=deadline,clock=clock)
    session.run(state,r.put(np.asarray(first_tokens,np.int32)[:,None]),np.asarray(metadata,np.int32))
    del state
    gc.collect()
    from jax.experimental import multihost_utils
    results=[]
    for lane,item in enumerate(values):
        tokens=np.asarray([e.token_id for e in session.events[lane]],np.int32)
        digest=sha256(tokens.tobytes()).hexdigest()
        hashes=np.asarray(multihost_utils.process_allgather(np.frombuffer(bytes.fromhex(digest),np.uint8)))
        r.require(bool((hashes==hashes[0]).all()),'concurrent output differs across hosts')
        elapsed=max(0.,session.finished_at[lane]-session.decode_started)
        report=dict(request_sha256=item['request_sha256'],prompt_tokens=int(lengths[lane]),
            emitted=len(tokens),timed_decode_tokens=len(tokens)-1,
            finish_reason=session.events[lane][-1].finish_reason,
            prefill_seconds=prefill_times[lane],prefill_tokens_per_second=lengths[lane]/prefill_times[lane],
            decode_wall_seconds=elapsed,decode_tokens_per_second=(len(tokens)-1)/elapsed if elapsed>0 else None,
            token_sha256=digest,peak_memory=r.stats(),sampling='greedy',speculative=False,
            batch_size=len(values),batch_rounds=session.round,context_capacity=r.capacity)
        results.append((tokens,report))
    aggregate=dict(batch_size=len(values),decode_rounds=session.round,
        prefill_seconds=sum(prefill_times),decode_wall_seconds=session.decode_seconds,
        aggregate_decode_tokens_per_second=sum(len(t)-1 for t,_ in results)/session.decode_seconds
            if session.decode_seconds>0 else None,wall_seconds=clock()-started)
    r.record.setdefault('batches',[]).append(aggregate)
    r.record['requests'].extend(report for _,report in results)
    r.save(r.record);r.active=False
    return results,aggregate
