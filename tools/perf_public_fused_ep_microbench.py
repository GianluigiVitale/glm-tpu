"""Leased, synthetic real-geometry EP32 versus TP4/EP8 routed-MoE experiment.

The outer controller authenticates source, fleet idle and workload leases.
This script never provisions resources. It does not load trained weights.
Public FP8 input sources must match public_fused_ep.SOURCE_HASHES exactly.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--public-source',type=Path,required=True)
    parser.add_argument('--code-hash',required=True)
    parser.add_argument('--coordinator',default='192.168.0.37:8476')
    parser.add_argument('--iters',type=int,choices=(20,50),default=20)
    args=parser.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from jax.experimental import multihost_utils
    from glm_tpu.perf.public_fused_ep import build_candidate,load_public_candidate,pack_weights,SOURCE_PIN,SOURCE_HASHES
    from glm_tpu.perf.speculative_moe import moe_rows_bf16
    from glm_tpu.perf.bf16_resident import Bf16MoeWeights
    from glm_tpu.perf.real_validation import memory_projection
    from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
    from glm_tpu.greenfield.kernels.ws32_prefill_moe import ws32_prefill_moe_from_routes_mapped
    from glm_tpu.perf.function_bindings import bind_dependencies
    from glm_tpu.perf.prefill_bf16 import resident_matmul,resident_matmul_f32
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
    from tools.perf_tpu_microbench import _ws32_mesh,_sharded_random

    public=load_public_candidate(args.public_source)
    rank=int(socket.gethostname().rsplit('-w-',1)[1])
    jax.distributed.initialize(coordinator_address=args.coordinator,num_processes=8,process_id=rank)
    assert jax.default_backend()=='tpu' and jax.device_count()==32 and len(jax.local_devices())==4
    assert all(d.device_kind=='TPU v4' for d in jax.devices())
    mesh=_ws32_mesh();epmesh=Mesh(mesh.devices.reshape(32),('ep',))
    args.output.mkdir(parents=True,exist_ok=True)
    hlo=args.output/f'hlo.rank{rank}';hlo.mkdir(exist_ok=False)
    record=dict(schema='glm_perf_public_fused_ep_v1',rank=rank,hostname=socket.gethostname(),
        code_hash=args.code_hash,jax=jax.__version__,jax_process_index=int(jax.process_index()),
        devices=32,complete=False,programs={},phases={},cases={},iters=args.iters,
        source_pin=SOURCE_PIN,public_source_sha256=SOURCE_HASHES,
        source_sha256={str(p.relative_to(REPO)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),*(REPO/'glm_tpu/perf').glob('*.py')]},
        synthetic=True,trained_parity=False,model_throughput=False,
        timing_includes_input_and_output_resharding=True,weight_conversion_timed_separately=True,
        shared_expert='zero-weight control in both paths; routed suffix comparison only',
        numerical_boundary='FP32 contraction/reduction order changes; gate .0625 absolute maximum versus retained routed suffix',
        limits=['CPU packed-FP8 DMA interpretation could not complete; this bounded TPU run must establish primitive correctness.',
                'No target layer, ordinary model, verifier or serving path is replaced.'])
    receipt=args.output/f'fused_ep.rank{rank}.json'
    def save():
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');tmp.replace(receipt)
    def all_ok(value):return bool(np.asarray(multihost_utils.process_allgather(np.array([bool(value)],np.int32))).all())
    def stats():return [dict(device_id=int(d.id),**d.memory_stats()) for d in jax.local_devices()]
    def phase(name,action):
        start=time.perf_counter();error=None;result=None
        try:result=action()
        except Exception as exc:error=exc
        agreed=all_ok(error is None)
        record['phases'][name]=dict(seconds=time.perf_counter()-start,passed=agreed and error is None)
        if error:record['phases'][name]['error']=type(error).__name__+': '+str(error)[:2000]
        save();print('FUSED_EP_PHASE '+name,flush=True)
        if error:raise error
        if not agreed:raise RuntimeError('peer failed '+name)
        return result
    def require(value,message):
        if not value:raise RuntimeError(message)
    def admit(name,exe):
        admission=memory_projection(stats(),record['programs'][name]['compiled_memory'])
        record['programs'][name]['memory_admission']=admission
        require(admission['passed'],'HBM admission failed for '+name)
    def compile_program(name,fn,values):
        lowered=fn.lower(*values)
        stable=str(lowered.compiler_ir(dialect='stablehlo'))
        exe=lowered.compile();optimized=exe.as_text()
        memory={k:int(getattr(exe.memory_analysis(),k,0) or 0) for k in
            ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes','generated_code_size_in_bytes')}
        record['programs'][name]=dict(stablehlo_sha256=sha256(stable.encode()).hexdigest(),
            optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),compiled_memory=memory)
        (hlo/(name+'.stablehlo.mlir')).write_text(stable)
        (hlo/(name+'.optimized_hlo.txt')).write_text(optimized)
        return exe
    def graph_check(name,exe):
        p=record['programs'][name]
        hashes=np.frombuffer(bytes.fromhex(p['stablehlo_sha256']+p['optimized_hlo_sha256']),np.uint8)
        gathered=np.asarray(multihost_utils.process_allgather(hashes))
        require(bool((gathered==gathered[0]).all()),'graph hashes differ across hosts')
        module=parse_hlo_module((hlo/(name+'.optimized_hlo.txt')).read_text())
        require(module.num_partitions==32,'primitive requires 32 partitions')
        widths={'pred':1,'s8':1,'u8':1,'bf16':2,'f16':2,'s16':2,'u16':2,'f32':4,'s32':4,'u32':4}
        allowed={frozenset(frozenset(range(r*4,r*4+4)) for r in range(8)),
                 frozenset(frozenset(range(c,32,4)) for c in range(4)),frozenset((frozenset(range(32)),))}
        sizes=[]
        for op in module.collectives:
            require(op.opcode in ('all-reduce','all-gather','reduce-scatter') and op.use_global_device_ids,
                    'unsupported primitive collective')
            require(frozenset(frozenset(g) for g in op.replica_groups) in allowed,'unexpected axis groups')
            for shape in (*op.result_shapes,*op.operand_shapes):
                require(shape.dtype in widths,'unknown exchange dtype')
                sizes.append(shape.element_count*widths[shape.dtype])
        require(max(sizes,default=0)<=16*1024**2,'unexpectedly large routed primitive exchange')
        p['hlo_admission']=dict(passed=True,profile='isolated_routed_ep32_16MiB_v1',
            maximum_collective_bytes=max(sizes,default=0),model_admission=False,
            custom_call_dma_not_counted_by_hlo_parser=True)
        admit(name,exe)
    try:
        # One layer plus both layouts and conversion scratch, not a full model.
        phase('preparation_headroom',lambda:require(memory_projection(stats(),dict(
            output_size_in_bytes=2*1024**3,temp_size_in_bytes=1024**3,alias_size_in_bytes=0,
            generated_code_size_in_bytes=128*1024**2))['passed'],'insufficient layer preparation headroom'))
        contract=GlmMoeNumericalContract(stage_size=8)
        gs,ds=P('expert',None,'feature'),P('expert','feature',None)
        specs=(gs,gs,gs,gs,ds,ds)
        shapes=((256,2048,6144),(256,16,48),(256,2048,6144),(256,16,48),(256,6144,2048),(256,48,16))
        def generate():
            values=tuple(_sharded_random(mesh,s,sp,'fp8' if i%2==0 else 'scale',6200+i)
                         for i,(s,sp) in enumerate(zip(shapes,specs)))
            return jax.block_until_ready(values)
        tables=phase('synthetic_weights',generate)
        start=time.perf_counter()
        weights=phase('pack_and_reshard_weights',lambda:jax.block_until_ready(jax.tree.map(
            lambda a:jax.device_put(a,NamedSharding(epmesh,P('ep'))),pack_weights(*tables))))
        record['weight_conversion_seconds']=time.perf_counter()-start
        record['resident_memory_after_conversion']=stats()
        prefill=bind_dependencies(ws32_prefill_moe_from_routes_mapped,
            fp8_block_matmul_f32=resident_matmul_f32,fp8_block_matmul=resident_matmul)
        def baseline_body(x,ids,rw,t):
            zero=jnp.zeros((2048,1536),jnp.bfloat16)
            shared=(zero,zero,jnp.zeros((1536,2048),jnp.bfloat16))
            if x.shape[0]<=8:
                m=Bf16MoeWeights(None,None,*t,*shared)
                out,health=moe_rows_bf16(x,ids,rw,m,contract=contract,small_expert_tiles=True)
            else:
                out,health=prefill(x,ids,rw,*t,shared[0],None,shared[1],None,shared[2],None,
                    contract=contract,expert_panels=True)
            return out,health[None,None]
        baseline=jax.jit(jax.shard_map(baseline_body,mesh=mesh,
            in_specs=(P(None,'feature'),P(),P(),specs),out_specs=(P(None,'feature'),P('expert','feature')),check_vma=False))
        candidate=build_candidate(mesh,args.public_source,routed_scaling_factor=contract.routed_scaling_factor)
        rep=NamedSharding(mesh,P())
        for rows in (2,3,32,128):
            x=_sharded_random(mesh,(rows,6144),P(None,'feature'),'bf16',6300+rows)
            rw=jax.device_put(np.full((rows,8),.125,np.float32),rep)
            for pattern in ('balanced','concentrated'):
                ids=np.broadcast_to(np.arange(8,dtype=np.int32),(rows,8)).copy()
                if pattern=='balanced':ids=(np.arange(rows,dtype=np.int32)[:,None]+32*ids)%256
                ids=jax.device_put(ids,rep)
                key=f'r{rows}_{pattern}';base_name=f'baseline_r{rows}';candidate_name=f'candidate_r{rows}'
                if pattern=='balanced':
                    b=phase('compile_'+base_name,lambda:compile_program(base_name,baseline,(x,ids,rw,tables)))
                    phase('admit_'+base_name,lambda:graph_check(base_name,b))
                    c=phase('compile_'+candidate_name,lambda:compile_program(candidate_name,candidate,(x,ids,rw,weights)))
                    phase('admit_'+candidate_name,lambda:graph_check(candidate_name,c))
                def run_base():return b(x,ids,rw,tables)
                def run_candidate():return c(x,ids,rw,weights)
                expected=phase('execute_baseline_'+key,lambda:jax.block_until_ready(run_base()))
                actual=phase('execute_candidate_'+key,lambda:jax.block_until_ready(run_candidate()))
                pairs=list(zip(expected[0].addressable_shards,actual.addressable_shards))
                finite=all(np.isfinite(np.asarray(v.data)).all() for pair in pairs for v in pair)
                healthy=all(bool(np.asarray(s.data).all()) for s in expected[1].addressable_shards)
                error=max(float(np.max(np.abs(np.asarray(a.data,np.float32)-np.asarray(b.data,np.float32)))) for a,b in pairs)
                row=dict(rows=rows,padded_rows=((rows+31)//32)*32,pattern=pattern,all_finite=bool(finite),
                    baseline_healthy=healthy,max_abs_error=error,absolute_error_limit=.0625,
                    candidate_numerically_admitted=bool(finite and healthy and error<=.0625),timings={})
                record['cases'][key]=row;save()
                phase('numerical_admission_'+key,lambda:require(row['candidate_numerically_admitted'],'routed primitive numerical gate failed'))
                for _ in range(3):jax.block_until_ready((run_base(),run_candidate()))
                samples={'baseline':[],'candidate':[]}
                for iteration in range(args.iters):
                    order=('baseline','candidate') if iteration%2==0 else ('candidate','baseline')
                    for mode in order:
                        start=time.perf_counter();jax.block_until_ready(run_base() if mode=='baseline' else run_candidate())
                        samples[mode].append((time.perf_counter()-start)*1000)
                row['timings']={mode:dict(samples_ms=values,p50_ms=float(np.median(values)),p90_ms=float(np.percentile(values,90)))
                    for mode,values in samples.items()}
                row['memory_after']=stats();save();print('FUSED_EP_CASE '+key,flush=True)
        record['complete']=True;record['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save()
    except Exception as exc:
        record['failure']=dict(type=type(exc).__name__,message=str(exc)[:2000]);save();raise
    finally:
        jax.distributed.shutdown()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
