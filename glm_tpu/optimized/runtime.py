"""Cold loading and private user generation for the retained ordinary profile.

The protected controller owns fleet/source admission and process cleanup. This
runtime verifies real checkpoint bytes and admits each new graph against live
memory before execution. It never loads speculative modules or benchmark cases.
"""
from hashlib import sha256
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from ..greenfield.runtime import ws32_decoder as dec, ws32_batched_prefill as pre
from ..greenfield.runtime.ws32_request_session import RequestPolicy
from .admission import inspect_research_hlo, memory_projection
from .bf16_resident import bf16_resident_weights
from .prefill_challenger import build_ws32_prefill_challenger_program
from .request_loop import build_packed_decoder_program, PackedRequestSession
from .request import validate, CAPACITY
from . import model


class OrdinaryRuntime:
    def __init__(self, *, args, repo, root, mesh, physical, topology, fleet_sha, vote, save,
                 context_capacity=CAPACITY, concurrent_size=0):
        self.args,self.root,self.mesh,self.vote,self.save=args,root,mesh,vote,save
        self.capacity=context_capacity
        from .request import CONCURRENT_CAPACITY
        if type(concurrent_size) is not int or not 0<=concurrent_size<=8:
            raise ValueError('concurrent size must be zero through eight')
        if concurrent_size and context_capacity!=CONCURRENT_CAPACITY:
            raise ValueError('concurrent runtime requires 32K per conversation')
        self.concurrent_size=concurrent_size
        self.config=dec.Ws32DecoderConfig(model.geometry(repo),self.capacity,host_main_rope_table=True)
        self.put=lambda x:jax.device_put(x,NamedSharding(mesh,P()))
        self.record=dict(schema='glm_optimized_runtime_v1',profile='ordinary-greedy',
            programs={},phases={},complete=False,capacity=self.capacity,
            state_ownership='exclusive_donated' if self.capacity>CAPACITY else 'non_donating',
            physical_identity=dict(mesh_sha256=physical.mesh_hash,topology_sha256=topology.topology_hash,
                fleet_sha256=fleet_sha,local_slots=[s for s,d in enumerate(physical.flattened_device_ids)
                    if d in {int(d.id) for d in jax.local_devices()}]),requests=[])
        # Host RAM holds compiler originals; do not exhaust the small root disk.
        from pathlib import Path
        import shutil
        self.hlo=Path('/dev/shm/glm-optimized-hlo')/root.parent.name/root.name
        self.require(shutil.disk_usage('/dev/shm').free>8*1024**3,'insufficient RAM for HLO originals')
        self.hlo.mkdir(parents=True,mode=0o700)
        self.record['hlo_originals']=str(self.hlo)
        self.active=False
        started=time.perf_counter()
        self._load(repo,physical)
        self.record['cold_load_compile_seconds']=time.perf_counter()-started
        self.record['complete']=True
        self.save(self.record)

    def stats(self):
        return [dict(device_id=int(d.id),**d.memory_stats()) for d in jax.local_devices()]

    def phase(self,name,action):
        value=error=None;start=time.perf_counter()
        try:value=action()
        except Exception as exc:error=exc
        agreed=self.vote(error is None)
        self.record['phases'][name]=dict(seconds=time.perf_counter()-start,passed=error is None and agreed is True)
        self.save(self.record)
        if error is not None:raise error
        if agreed is not True:raise RuntimeError('optimized peer phase failed: '+name)
        return value

    @staticmethod
    def require(valid,message):
        if not bool(valid):raise RuntimeError(message)

    def admit_memory(self,name,memory):
        report=memory_projection(self.stats(),memory)
        self.phase('memory_'+name,lambda:self.require(report['passed'],'optimized memory admission failed'))
        return report

    def admit(self,name):
        row=self.record['programs'][name]
        row['memory_admission']=self.admit_memory(name,row['compiled_memory'])

    def compile(self,name,fn,values,*,model=True):
        from scripts.greenfield.ws32_compile_originals import compile_program
        from jax.experimental import multihost_utils
        exe=self.phase('compile_'+name,lambda:compile_program(fn,values,name,self.hlo,self.record))
        row=self.record['programs'][name]
        def consensus():
            value=np.frombuffer(bytes.fromhex(row['stablehlo_sha256']+row['optimized_hlo_sha256']),np.uint8)
            all_hashes=np.asarray(multihost_utils.process_allgather(value))
            self.require((all_hashes==all_hashes[0]).all(),'optimized graph differs across hosts')
        self.phase('graph_consensus_'+name,consensus)
        if model:
            row['hlo_admission']=self.phase('hlo_'+name,lambda:inspect_research_hlo(
                (self.hlo/f'{name}.optimized_hlo.txt').read_text()))
        self.admit(name)
        return exe

    def _load(self,repo,physical):
        from scripts.greenfield.ws32_compile_originals import authenticated_inventory,build_wk_programs
        from scripts.greenfield.ws32_native_benchmark_programs import build_cache_initializer
        from ..greenfield.checkpoint.ws32_runtime_checkpoint import (
            verify_ws32_runtime_checkpoint,load_ws32_runtime_checkpoint)
        args,config=self.args,self.config
        model.require_site(args)
        slots=tuple(self.record['physical_identity']['local_slots'])
        self.require(len(slots)==4,'optimized runtime requires four local checkpoint slots')
        pin=args.source_inventory_sha256
        inventory=self.phase('inventory',lambda:authenticated_inventory(args.source_inventory,pin))
        self.phase('model_identity',lambda:model.require_inventory(inventory))
        checkpoint=self.phase('verify_checkpoint',lambda:verify_ws32_runtime_checkpoint(args.checkpoint_root,
            expected_manifest_sha256=args.checkpoint_manifest_sha256,expected_success_sha256=args.checkpoint_success_sha256,
            expected_mesh_hash=args.mesh_sha256,expected_topology_hash=args.topology_sha256,
            inventory=inventory,geometry=config.geometry,verify_file_hashes=True,
            verify_file_hash_slots=slots,local_slot_layout=True))
        load_memory=dict(output_size_in_bytes=max(p.payload_bytes for p in checkpoint.plans),
            temp_size_in_bytes=2*max(t.byte_count for p in checkpoint.plans for t in p.tensors),
            generated_code_size_in_bytes=0,alias_size_in_bytes=0)
        self.record['checkpoint_load_memory']=self.admit_memory('load_checkpoint',load_memory)
        loaded=self.phase('load_checkpoint',lambda:load_ws32_runtime_checkpoint(checkpoint,mesh=self.mesh,physical_mesh=physical))
        self.record['checkpoint']=dict(manifest_sha256=checkpoint.manifest['manifest_sha256'],
            success_sha256=checkpoint.success['success_sha256'],inventory_sha256=inventory.inventory_sha256,
            verified_slots=slots)
        raw=self.phase('bind_weights',lambda:dec.bind_ws32_decoder_weights(loaded.arrays,config))
        del loaded
        decode,promote=build_wk_programs(self.mesh,P(None,'feature'),P(None,'feature'),contract=config.dsa_contract)
        first=raw.layers[config.full_index_slots[0]].dsa
        decode=self.compile('wk_decode',decode,(first.wk_bits_local,first.wk_scale_local),model=False)
        completed=jax.block_until_ready(decode(first.wk_bits_local,first.wk_scale_local))
        promote=self.compile('wk_promote',promote,(completed,),model=False)
        wk=[]
        for layer_id in config.full_index_slots:
            d=raw.layers[layer_id].dsa
            completed=jax.block_until_ready(decode(d.wk_bits_local,d.wk_scale_local))
            wk.append(jax.block_until_ready(promote(completed)))
        self.wk=tuple(wk)
        del completed,d,first,decode,promote
        tables=[]
        for layer in raw.layers:
            tables.extend(x for x in jax.tree.leaves((layer.qkv_a,layer.attention,layer.dsa,layer.dense)) if x.dtype==jnp.uint8)
            if layer.moe is not None:
                tables.extend((layer.moe.shared_gate_bits_local,layer.moe.shared_up_bits_local,layer.moe.shared_down_bits_local))
        sizes=[x.addressable_shards[0].data.nbytes for x in tables]
        self.record['bf16_preparation_memory']=self.admit_memory('bf16_prepare',dict(
            output_size_in_bytes=2*sum(sizes),temp_size_in_bytes=6*max(sizes),
            generated_code_size_in_bytes=128*1024**2,alias_size_in_bytes=0))
        self.weights=self.phase('bf16_prepare',lambda:jax.block_until_ready(bf16_resident_weights(self.mesh,config,raw)))
        del raw,tables,layer
        gc.collect()
        self.rope=self.put(np.asarray(dec.build_ws32_main_rope_table(config)))
        self.initialize=self.compile('cache_init',build_cache_initializer(self.mesh,config),(self.put(np.int32(2034)),),model=False)
        initial=self.phase('initial_cache',lambda:jax.block_until_ready(self.initialize(self.put(np.int32(2034)))))
        options=dict(key_tile=512,mlp_window=True,rolled_prefix=True,expert_panels=True,
                     paired_position_sort=True,sorted_local_merge=True,canonical_dense=True)
        self.prefill={}
        for rows in (128,114):
            fn=build_ws32_prefill_challenger_program(self.mesh,config,block_rows=rows,**options).execute
            # The long-context release already transfers exclusive cache
            # ownership this way. Reusing the consumed state is forbidden;
            # donation avoids retaining a second multi-GB cache allocation.
            if self.capacity>CAPACITY:fn=jax.jit(fn,donate_argnums=(2,))
            self.prefill[rows]=self.compile('prefill_'+str(rows),fn,
                (self.put(np.zeros(rows,np.int32)),self.put(np.int32(rows)),initial,self.weights,self.wk,self.rope))
        if self.concurrent_size:
            from .batched_runtime import compile_batch
            compile_batch(self,initial)
        else:
            decode_fn=build_packed_decoder_program(self.mesh,config).execute
            if self.capacity>CAPACITY:decode_fn=jax.jit(decode_fn,donate_argnums=(1,))
            self.decode=self.compile('decode',decode_fn,
                (self.put(np.array([0],np.int32)),initial.decoder,self.weights,self.rope))
        del initial
        gc.collect()

    def generate(self,request,*,deliver,deadline,clock=time.perf_counter):
        """One fresh request; ambiguous delivery or fleet failure poisons this runtime."""
        validate(request)
        if self.concurrent_size:raise RuntimeError('use generate_concurrent on a batched runtime')
        self.require(request['context_capacity']==self.capacity,'request capacity differs from loaded model')
        if self.active:raise RuntimeError('optimized runtime has an active or failed request')
        self.active=True
        ids=np.asarray(request['prompt_ids'],np.int32)
        policy=RequestPolicy(request['request_id'],0,len(ids),request['max_new_tokens'],
            self.capacity,request['vocab_size'],tuple(request['eos_ids']))
        def budget():
            self.require(self.vote(clock()<deadline) is True,'optimized request deadline expired')
        for name in ('cache_init','prefill_128','prefill_114','decode'):self.admit(name)
        fresh=jax.block_until_ready(self.initialize(self.put(np.int32(len(ids)))))
        staged=[]
        for start in range(0,len(ids),128):
            block=ids[start:start+128];rows=114 if len(block)<=114 else 128
            staged.append((self.put(np.pad(block,(0,rows-len(block)),constant_values=-1)),self.put(np.int32(len(block)))))
        budget();started=clock()
        for i,(block,count) in enumerate(staged):
            budget()
            result=jax.block_until_ready(self.prefill[block.size](block,count,fresh,self.weights,self.wk,self.rope))
            fresh=result.state
            self.require(self.vote(bool(np.asarray(fresh.decoder.contract_valid).all())) is True,'optimized prefill failed')
        prefill_seconds=clock()-started
        session=PackedRequestSession(policy,decode_step=lambda t,s:self.decode(t,s,self.weights,self.rope),
            replicate_uniform=self.put,fleet_all=lambda valid:self.vote(valid and clock()<deadline),
            deliver=deliver,request_started=started,
            delivery_boundary='rank0 private JSONL token write+flush; no network transport',clock=clock)
        session.accept_prefill(result)
        # Release prefill references before entering sustained decode.
        del fresh,result
        decode_started=clock()
        while not session.finished:
            session.step()
        elapsed=clock()-decode_started
        tokens=np.asarray([event.token_id for event in session.events],np.int32)
        # Deadline admission participates in the existing fleet votes. A local
        # pre-dispatch exception would strand peers in the next collective.
        from jax.experimental import multihost_utils
        token_digest=sha256(tokens.tobytes()).hexdigest()
        def agree_tokens():
            hashes=np.asarray(multihost_utils.process_allgather(
                np.frombuffer(bytes.fromhex(token_digest),np.uint8)))
            self.require(bool((hashes==hashes[0]).all()),'optimized output differs across hosts')
        self.phase('output_consensus',agree_tokens)
        report=dict(request_sha256=request['request_sha256'],prompt_tokens=len(ids),
            emitted=len(tokens),timed_decode_tokens=len(tokens)-1,finish_reason=session.events[-1].finish_reason,
            prefill_seconds=prefill_seconds,prefill_tokens_per_second=len(ids)/prefill_seconds,
            decode_wall_seconds=elapsed,decode_tokens_per_second=(len(tokens)-1)/elapsed if len(tokens)>1 else None,
            ttft_seconds=session.ttft_seconds,token_sha256=token_digest,
            peak_memory=self.stats(),sampling='greedy',speculative=False)
        session.release();self.active=False
        self.record['requests'].append(report);self.save(self.record)
        return tokens,report

    def generate_concurrent(self,requests,*,deliver,deadline,clock=time.perf_counter):
        from .batched_runtime import generate_batch
        return generate_batch(self,requests,deliver=deliver,deadline=deadline,clock=clock)
