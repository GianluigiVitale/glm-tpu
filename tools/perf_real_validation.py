"""Leased research worker: authenticated DB610 prompt, real weights, fresh graphs.

The outer controller owns both workload leases, authenticated idle checks and
cleanup. This tool never provisions resources or writes to the results database.
All raw prompts/tokens and HLO originals stay in the private run directory.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-inputs',type=Path)
    p.add_argument('--summarize',type=Path)
    p.add_argument('--summary-output',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--input-sha256')
    p.add_argument('--identity-sha256')
    p.add_argument('--source-manifest-sha256')
    p.add_argument('--code-hash')
    p.add_argument('--diagnose-layerwise',action='store_true')
    p.add_argument('--diagnose-ablation',action='store_true')
    p.add_argument('--coordinator-address',default='192.168.0.37:8476')
    args = p.parse_args()
    os.umask(0o077)
    import numpy as np
    from glm_tpu.perf.real_validation import db610_inputs, inspect_research_hlo, memory_projection, build_db610_decoder, summarize_real_validation
    if args.summarize is not None:
        if args.summary_output is None:raise ValueError('summary output required')
        write_json(args.summary_output,summarize_real_validation(args.summarize))
        return 0
    if args.prepare_inputs is not None:
        root = args.prepare_inputs.resolve()
        if root.is_relative_to(REPO) or not root.is_dir():
            raise ValueError('private input output must be an existing directory outside Git')
        if (root/'inputs.npz').exists():
            raise ValueError('do not overwrite an existing input bundle')
        seal = json.loads((REPO/'docs/artifacts/prefill-canonical-short-db610-sealed-20260909.json').read_bytes())
        prompt,expected,identity = db610_inputs(REPO,Path('/home/gianl/glm-run')/seal['run_tag'])
        np.savez(root/'inputs.npz',prompt=prompt,expected=expected)
        write_json(root/'input_identity.json',identity)
        print(json.dumps(dict(input_sha256=sha256((root/'inputs.npz').read_bytes()).hexdigest(),
                              identity_sha256=sha256((root/'input_identity.json').read_bytes()).hexdigest())))
        return 0
    if args.output is None or any(x is None or len(x)!=64 for x in
            (args.input_sha256,args.identity_sha256,args.source_manifest_sha256)):
        raise ValueError('worker requires output and pinned input/source digests')
    root = args.output.resolve()
    for name,pin in [('inputs.npz',args.input_sha256),('input_identity.json',args.identity_sha256),
                     ('source_manifest.json',args.source_manifest_sha256)]:
        if sha256((root/name).read_bytes()).hexdigest()!=pin:
            raise ValueError('private run input/source manifest hash differs: '+name)
    manifest=json.loads((root/'source_manifest.json').read_bytes())
    for name,pin in manifest.items():
        path=REPO/name
        if not path.resolve().is_relative_to(REPO) or sha256(path.read_bytes()).hexdigest()!=pin:
            raise ValueError('deployed source bytes differ: '+name)
    identity=json.loads((root/'input_identity.json').read_bytes())
    with np.load(root/'inputs.npz',allow_pickle=False) as f:
        prompt,expected=f['prompt'],f['expected']
    if (prompt.shape!=(2034,) or expected.shape!=(29,) or prompt.dtype!=np.int32 or expected.dtype!=np.int32
            or sha256(prompt.tobytes()).hexdigest()!=identity['prompt_sha256']
            or sha256(expected.tobytes()).hexdigest()!=identity['reference_sha256']):
        raise ValueError('private DB610 input geometry/digests differ')
    args.process_id=int(socket.gethostname().rsplit('-w-',1)[1])
    from scripts.release.ws32_user_worker import site_args
    args=site_args(args)
    args.context_capacity=8192
    from scripts.greenfield import run_short_decoder_ws32 as original
    jax,mesh,physical,topology,fleet_sha=original._initialize_runtime(args)
    import jax.numpy as jnp
    from jax.sharding import NamedSharding,PartitionSpec as P
    from scripts.greenfield.ws32_compile_originals import authenticated_inventory,build_wk_programs,compile_program
    from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import verify_ws32_runtime_checkpoint,load_ws32_runtime_checkpoint
    from glm_tpu.greenfield.runtime import ws32_decoder as dec,ws32_batched_prefill as pre
    from glm_tpu.perf.bf16_resident import bf16_resident_weights
    from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program
    rank=args.process_id
    record=dict(schema='glm_perf_real_validation_rank_v1',rank=rank,jax_process_index=jax.process_index(),hostname=socket.gethostname(),
        code_hash=args.code_hash,source_manifest_sha256=args.source_manifest_sha256,
        input_identity=identity,input_sha256=args.input_sha256,devices=jax.device_count(),
        jax=jax.__version__,capacity=8192,programs={},phases={},complete=False,
        frozen_graph_admission_inherited=False,trained_model_quality_claim=False,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    receipt=root/f'validation.rank{rank}.json'
    hlo_root=root/f'hlo.rank{rank}'
    hlo_root.mkdir(exist_ok=False)
    def save():write_json(receipt,record)
    def stats():return [dict(device_id=int(d.id),**d.memory_stats()) for d in jax.local_devices()]
    def phase(name,action):
        result=error=None
        start=time.perf_counter()
        try:result=action()
        except Exception as exc:error=exc
        agreed=original._batched_fleet_all(error is None)
        record['phases'][name]=dict(seconds=time.perf_counter()-start,passed=error is None and agreed)
        save()
        if error is not None:raise error
        if not agreed:raise RuntimeError('peer failed phase: '+name)
        print('REAL_VALIDATION_PHASE '+name,flush=True)
        return result
    def require(value,message):
        if not bool(value):raise RuntimeError(message)
    def admit(name,compiled):
        mem=record['programs'][name]['compiled_memory']
        row=memory_projection(stats(),mem)
        record['programs'][name]['memory_admission']=row
        phase('memory_'+name,lambda:require(row['passed'],'real-weight memory admission refused'))
    def compile_model(name,fn,values):
        exe=phase('compile_'+name,lambda:compile_program(fn,values,name,hlo_root,record))
        from jax.experimental import multihost_utils
        def same_graphs():
            hashes=record['programs'][name]
            value=np.frombuffer(bytes.fromhex(hashes['stablehlo_sha256']+hashes['optimized_hlo_sha256']),np.uint8)
            gathered=np.asarray(multihost_utils.process_allgather(value))
            require(bool((gathered==gathered[0]).all()),'compiled model graph hashes differ across hosts')
        phase('graph_consensus_'+name,same_graphs)
        report=phase('hlo_'+name,lambda:inspect_research_hlo((hlo_root/f'{name}.optimized_hlo.txt').read_text()))
        record['programs'][name]['hlo_admission']=report
        admit(name,exe)
        return exe
    try:
        local_ids={int(d.id) for d in jax.local_devices()}
        slots=tuple(s for s,d in enumerate(physical.flattened_device_ids) if d in local_ids)
        require(len(slots)==4,'expected four checkpoint slots')
        record['physical_identity']=dict(mesh_sha256=physical.mesh_hash,topology_sha256=topology.topology_hash,
            fleet_sha256=fleet_sha,local_slots=slots)
        config=dec.Ws32DecoderConfig(original._geometry(),8192,host_main_rope_table=True)
        decode_program=phase('decoder_configuration',lambda:build_db610_decoder(mesh,config))
        inventory_pin=json.loads((REPO/'docs/artifacts/prefill-window-layer6-host-admission-20260908.json').read_bytes())['source_inventory_sha256']
        inventory=phase('source_inventory',lambda:authenticated_inventory(args.source_inventory,inventory_pin))
        checkpoint=phase('verify_checkpoint',lambda:verify_ws32_runtime_checkpoint(args.checkpoint_root,
            expected_manifest_sha256=args.checkpoint_manifest_sha256,expected_success_sha256=args.checkpoint_success_sha256,
            expected_mesh_hash=args.mesh_sha256,expected_topology_hash=args.topology_sha256,
            inventory=inventory,geometry=config.geometry,verify_file_hashes=True,
            verify_file_hash_slots=slots,local_slot_layout=True))
        load_mem=dict(output_size_in_bytes=max(p.payload_bytes for p in checkpoint.plans),
            temp_size_in_bytes=2*max(t.byte_count for p in checkpoint.plans for t in p.tensors),
            generated_code_size_in_bytes=0,alias_size_in_bytes=0)
        record['checkpoint_load_memory']=memory_projection(stats(),load_mem)
        phase('checkpoint_load_admission',lambda:require(record['checkpoint_load_memory']['passed'],'checkpoint load does not fit'))
        loaded=phase('load_checkpoint',lambda:load_ws32_runtime_checkpoint(checkpoint,mesh=mesh,physical_mesh=physical))
        record['checkpoint']=dict(manifest_sha256=checkpoint.manifest['manifest_sha256'],success_sha256=checkpoint.success['success_sha256'],
            inventory_sha256=inventory.inventory_sha256,verified_slots=slots,memory_after=stats())
        raw=phase('bind_weights',lambda:dec.bind_ws32_decoder_weights(loaded.arrays,config))
        del loaded
        # Mandatory completed BF16 decode followed by separate FP32 promotion.
        wk_decode,wk_promote=build_wk_programs(mesh,P(None,'feature'),P(None,'feature'),contract=config.dsa_contract)
        first_d=raw.layers[config.full_index_slots[0]].dsa
        wk_decode=phase('compile_wk_decode',lambda:compile_program(wk_decode,(first_d.wk_bits_local,first_d.wk_scale_local),'wk_decode',hlo_root,record))
        admit('wk_decode',wk_decode)
        completed=wk_decode(first_d.wk_bits_local,first_d.wk_scale_local)
        jax.block_until_ready(completed)
        wk_promote=phase('compile_wk_promote',lambda:compile_program(wk_promote,(completed,),'wk_promote',hlo_root,record))
        admit('wk_promote',wk_promote)
        del first_d,completed
        wk=[]
        for layer_id in config.full_index_slots:
            d=raw.layers[layer_id].dsa
            completed=wk_decode(d.wk_bits_local,d.wk_scale_local)
            jax.block_until_ready(completed)
            value=wk_promote(completed)
            jax.block_until_ready(value)
            wk.append(value)
        wk=tuple(wk)
        del completed,value,d,wk_decode,wk_promote
        phase('wk_prepared',lambda:None)
        # Bound resident conversion BEFORE dispatch. Raw tables coexist with
        # every BF16 output; allow six times the largest raw table for scratch.
        tables=[]
        for layer in raw.layers:
            tables.extend(x for x in jax.tree.leaves((layer.qkv_a,layer.attention,layer.dsa,layer.dense)) if x.dtype==jnp.uint8)
            if layer.moe is not None:tables.extend((layer.moe.shared_gate_bits_local,layer.moe.shared_up_bits_local,layer.moe.shared_down_bits_local))
        sizes=[x.addressable_shards[0].data.nbytes for x in tables]
        preparation_memory=dict(output_size_in_bytes=2*sum(sizes),temp_size_in_bytes=6*max(sizes),
                                generated_code_size_in_bytes=128*1024**2,alias_size_in_bytes=0)
        record['bf16_preparation_memory']=memory_projection(stats(),preparation_memory)
        phase('bf16_preparation_admission',lambda:require(record['bf16_preparation_memory']['passed'],'BF16 preparation does not fit'))
        weights=phase('bf16_prepare',lambda:bf16_resident_weights(mesh,config,raw))
        jax.block_until_ready(weights)
        del raw,tables,layer
        gc.collect()
        replicated=NamedSharding(mesh,P())
        def put(x):return jax.device_put(x,replicated)
        rope=put(np.asarray(dec.build_ws32_main_rope_table(config)))
        initial=pre.make_ws32_batched_prefill_state(mesh,config,prompt_length=len(prompt))
        count128,count114=put(np.int32(128)),put(np.int32(114))
        blocks=[put(prompt[i:i+128]) for i in range(0,len(prompt),128)]
        options=dict(key_tile=512,mlp_window=True,rolled_prefix=True,expert_panels=True,
                     paired_position_sort=True,sorted_local_merge=True,canonical_dense=True)
        record['prefill_options']=options
        prefill={}
        for rows,count,block in [(128,count128,blocks[0]),(114,count114,blocks[-1])]:
            program=build_ws32_prefill_challenger_program(mesh,config,lse_attention=True,bf16_resident=True,block_rows=rows,**options)
            prefill[rows]=compile_model('prefill_'+str(rows),program.execute,(block,count,initial,weights,wk,rope))
        decode=compile_model('decode',decode_program.execute,(put(np.array([0],np.int32)),initial.decoder,weights,rope))
        # Re-admit with all three resident model executables present.
        for name,exe in [('prefill_128',prefill[128]),('prefill_114',prefill[114]),('decode',decode)]:admit(name,exe)
        warm=prefill[128](blocks[0],count128,initial,weights,wk,rope)
        jax.block_until_ready(warm)
        phase('warm_health',lambda:require(np.asarray(warm.state.decoder.contract_valid).all(),'warm prefill unhealthy'))
        del warm
        current=initial
        initial=None
        seconds=[]
        started=time.perf_counter()
        for i,block in enumerate(blocks):
            n=block.shape[0]
            before=time.perf_counter()
            result=prefill[n](block,count128 if n==128 else count114,current,weights,wk,rope)
            jax.block_until_ready(result)
            seconds.append(time.perf_counter()-before)
            current=result.state
            phase('prefill_block_'+str(i),lambda:require(np.asarray(current.decoder.contract_valid).all(),'prefill unhealthy'))
        elapsed=time.perf_counter()-started
        phase('prefill_frontier',lambda:require(bool(np.asarray(current.finished)) and int(np.asarray(current.decoder.position)[0])==2034,'prefill frontier differs'))
        record['prefill']=dict(prompt_tokens=2034,wall_seconds=elapsed,prompt_tokens_per_second=2034/elapsed,
            block_seconds=seconds,includes_block_health_votes_and_receipts=True,memory_after=stats())
        def cache_finite(state):
            return all(np.isfinite(np.asarray(s.data)).all()
                for x in (state.kv_cache_local,state.index_cache_local) for s in x.addressable_shards)
        phase('prefill_cache_finite',lambda:require(cache_finite(current.decoder),'prefill cache contains nonfinite values'))
        token=result.next_token
        observed=[int(np.asarray(token)[0])]
        state=current.decoder
        del current,result
        if args.diagnose_layerwise:
            from glm_tpu.perf.decode_diagnostics import diagnose_layerwise,activation_stats
            record['diagnostic_programs']={}
            def diagnostic_compile(name,fn,values):
                exe=compile_model(name,fn,values)
                record['diagnostic_programs'][name]=record['programs'].pop(name)
                save()
                return exe
            def diagnostic_observe(value):
                record['first_decode_diagnostic']=value
                save()
            split,diagnostic=phase('layerwise_diagnostic',lambda:diagnose_layerwise(
                mesh,config,decode_program.options,token,state,weights,rope,
                compile_program=diagnostic_compile,observe=diagnostic_observe))
            whole=decode(token,state,weights,rope)
            jax.block_until_ready(whole)
            diagnostic['whole_final_residual']=activation_stats(whole.final_residual_local)
            diagnostic['whole_healthy']=bool(np.asarray(whole.state.contract_valid).all())
            diagnostic['whole_first_token_matches_db610']=bool(int(np.asarray(whole.next_token)[0])==int(expected[1]))
            diagnostic['split_first_token_matches_db610']=bool(int(np.asarray(split[0])[0])==int(expected[1]))
            diagnostic['split_and_whole_token_equal']=bool(np.array_equal(np.asarray(split[0]),np.asarray(whole.next_token)))
            pairs=[(np.ascontiguousarray(a.data),np.ascontiguousarray(b.data))
                   for a,b in zip(split[2].addressable_shards,whole.final_residual_local.addressable_shards)]
            diagnostic['split_and_whole_residual_bitwise']=all(np.array_equal(a.view(np.uint8),b.view(np.uint8)) for a,b in pairs)
            diagnostic['split_whole_max_abs']=max(float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) for a,b in pairs)
            save()
            del split,whole,diagnostic
        if args.diagnose_ablation:
            from glm_tpu.perf.decode_diagnostics import activation_stats
            record['ablation_programs']={}
            record['first_decode_ablations']={}
            for label,lse,two_stage in [('d1_d8',False,False),('d1_d8_d5',True,False),('d1_d8_d10',False,True)]:
                program=build_db610_decoder(mesh,config,lse_attention=lse,dsa_two_stage=two_stage)
                name='ablation_'+label
                fn=compile_model(name,program.execute,(token,state,weights,rope))
                record['ablation_programs'][name]=record['programs'].pop(name)
                sample=phase(name+'_execute',lambda:jax.block_until_ready(fn(token,state,weights,rope)))
                record['first_decode_ablations'][label]=dict(lse_attention=lse,dsa_two_stage=two_stage,
                    healthy=bool(np.asarray(sample.state.contract_valid).all()),
                    first_token_matches_db610=bool(int(np.asarray(sample.next_token)[0])==int(expected[1])),
                    token_sha256=sha256(np.asarray(sample.next_token).tobytes()).hexdigest(),
                    final_residual=activation_stats(sample.final_residual_local),
                    timing_claim=False,arithmetic_unchanged_from_existing_variant=True)
                save()
                program.execute.clear_cache()
                del sample,fn,program
                gc.collect()
        times=[]
        for index in range(1,29):
            before=time.perf_counter()
            result=decode(token,state,weights,rope)
            jax.block_until_ready(result)
            times.append(time.perf_counter()-before)
            state,token=result.state,result.next_token
            finite=all(np.isfinite(np.asarray(s.data)).all() for s in result.final_residual_local.addressable_shards)
            phase('decode_health_'+str(index),lambda:require(np.asarray(state.contract_valid).all() and finite,'decode unhealthy/nonfinite'))
            observed.append(int(np.asarray(token)[0]))
        phase('final_cache_finite',lambda:require(cache_finite(state),'final cache contains nonfinite values'))
        record['finite_cache_checks']=dict(after_prefill=True,after_decode=True,outside_model_timing=True)
        values=np.asarray(observed,np.int32)
        np.savez(root/f'generated.rank{rank}.npz',tokens=values)
        mismatches=np.flatnonzero(values!=expected)
        record['token_comparison']=dict(compared=29,matches=int(np.sum(values==expected)),all_equal=not bool(mismatches.size),
            first_mismatch_index=None if not mismatches.size else int(mismatches[0]),observed_sha256=sha256(values.tobytes()).hexdigest())
        record['decode']=dict(samples=28,seconds=times,p50_ms=float(np.median(times)*1e3),p99_ms=float(np.percentile(times,99)*1e3),
            after_five_warm_steps_p50_ms=float(np.median(times[5:])*1e3),
            model_tokens_per_second=28/sum(times),excludes_host_checks_and_delivery=True,memory_after=stats())
        record.update(complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        save()
        print('REAL_VALIDATION_DONE '+json.dumps(record['token_comparison']),flush=True)
    except Exception as exc:
        record['failure']=dict(kind=type(exc).__name__,message=str(exc))
        save()
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
