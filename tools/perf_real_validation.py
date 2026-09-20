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
    p.add_argument('--topology-rebinding-sha256',
        help='authenticate staged host reassignment on the unchanged sealed chip mesh')
    p.add_argument('--diagnose-layerwise',action='store_true')
    p.add_argument('--diagnose-ablation',action='store_true')
    p.add_argument('--diagnose-request-loop',action='store_true')
    p.add_argument('--prefix-replay-sha256',help='short private same-prefix R1/R2/R3 correctness replay')
    p.add_argument('--prefix-replay-trace',action='store_true',help='observe all layers and live final-head margins in the replay')
    p.add_argument('--prefix-replay-timing-iters',type=int,choices=(0,5,20),default=0,
        help='warmed paired target-window latency, not speculative serving throughput')
    p.add_argument('--prefix-replay-unrolled-attention',action='store_true',
        help='diagnostic one-row attention expressions with pooled M8 experts; requires prefix replay')
    p.add_argument('--prefix-replay-global-max-attention',action='store_true',
        help='diagnostic local partial attention in verifier; ordinary reference stays canonical')
    p.add_argument('--question-sha256',help='run private question.json after the DB610 parity gate')
    p.add_argument('--native-pack-index-sha256',help='compare native MTP speculation using a verified private pack index')
    p.add_argument('--native-suite-sha256',help='pinned private prose/code/structured cases and repeat counts')
    p.add_argument('--ordinary-suite-sha256',help='pinned private cases; ordinary paths only, one model load')
    p.add_argument('--ordinary-suite-compare-prefill',action='store_true',
                   help='alternate ordinary and admitted global-max prefill with common ordinary decode')
    p.add_argument('--ordinary-prefill-admission-sha256',help='completed trained DB610 global-max receipt')
    p.add_argument('--native-component-timing', choices=('blocking','none'), default='blocking',
        help='none removes per-component profiling barriers; request votes and delivery remain')
    p.add_argument('--native-order-policy', choices=('ordinary_first','alternating'), default='ordinary_first',
        help='alternating reverses ordinary/R2/R3 order on even suite repeats; every mode gets fresh prefill')
    p.add_argument('--native-acceptance', choices=('host','device'), default='host',
        help='experimental device acceptance fused into verifier; retains fleet plan agreement before commit')
    p.add_argument('--diagnose-speculative-verifier',action='store_true',
        help='teacher-forced two/three-row verifier diagnostics; no native MTP drafting')
    p.add_argument('--verifier-small-expert-tiles',action='store_true')
    p.add_argument('--verifier-rowwise-dsa',action='store_true')
    p.add_argument('--hlo-in-shm',action='store_true',
                   help='retain new HLO artifacts in shared memory through run-directory links')
    p.add_argument('--prefill-block-rows',type=int,choices=(128,512),default=128)
    p.add_argument('--prefill-owned-key-capacity',type=int,choices=(512,))
    p.add_argument('--prefill-wide-indexshare',action='store_true',
                   help='experimental wider sparse shared-indexer prefixes; requires bounded owner buffers')
    p.add_argument('--prefill-global-max-attention',action='store_true',
                   help='isolated DB610 prefill attention experiment; ordinary decode stays canonical')
    p.add_argument('--decode-lse-attention', action='store_true',
                   help='experimental D5 decode; default D1/D8/D10 passed the DB610 ablation')
    p.add_argument('--write-empty-route-slot',action='store_true',default=True,
                   help='explicit compatibility flag; empty route output stores are now required')
    p.add_argument('--coordinator-address',default='192.168.0.37:8476')
    args = p.parse_args()
    os.umask(0o077)
    import numpy as np
    from glm_tpu.perf.real_validation import db610_inputs, inspect_research_hlo, memory_projection, build_db610_decoder, summarize_real_validation, db610_prefill_plan
    prefill_plan=db610_prefill_plan(args.prefill_block_rows,args.prefill_owned_key_capacity,
                                 wide_indexshare=args.prefill_wide_indexshare)
    if args.ordinary_suite_sha256 is not None and (len(args.ordinary_suite_sha256)!=64
            or prefill_plan!=db610_prefill_plan() or args.decode_lse_attention
            or args.prefill_global_max_attention or any((args.question_sha256,
                args.native_pack_index_sha256,args.native_suite_sha256,args.prefix_replay_sha256,
                args.diagnose_layerwise,args.diagnose_ablation,args.diagnose_request_loop,
                args.diagnose_speculative_verifier))):
        raise ValueError('ordinary suite requires the canonical target and no other experiment')
    if args.ordinary_suite_compare_prefill != (args.ordinary_prefill_admission_sha256 is not None):
        raise ValueError('prefill comparison requires its trained admission receipt')
    if args.ordinary_suite_compare_prefill and args.ordinary_suite_sha256 is None:
        raise ValueError('prefill comparison requires a pinned ordinary suite')
    if args.prefill_global_max_attention and (prefill_plan != db610_prefill_plan()
            or args.decode_lse_attention or any((args.prefix_replay_sha256,
                args.diagnose_layerwise,args.diagnose_ablation,args.diagnose_request_loop,
                args.diagnose_speculative_verifier,args.question_sha256,
                args.native_pack_index_sha256,args.native_suite_sha256))):
        raise ValueError('global-max prefill requires an isolated canonical-plan DB610 experiment')
    if args.prefix_replay_trace and args.prefix_replay_sha256 is None:
        raise ValueError('layer trace requires a pinned prefix replay')
    if args.prefix_replay_timing_iters and args.prefix_replay_sha256 is None:
        raise ValueError('prefix timing requires a pinned prefix replay')
    if args.prefix_replay_unrolled_attention and args.prefix_replay_sha256 is None:
        raise ValueError('unrolled attention requires a pinned prefix replay')
    if args.prefix_replay_global_max_attention and (args.prefix_replay_sha256 is None
            or args.prefix_replay_trace or args.prefix_replay_unrolled_attention):
        raise ValueError('global-max attention requires a pinned untraced batched prefix replay')
    if args.native_component_timing != 'blocking' and args.native_pack_index_sha256 is None:
        raise ValueError('native timing option requires a pinned native pack')
    if args.native_acceptance != 'host' and args.native_pack_index_sha256 is None:
        raise ValueError('device acceptance requires a pinned native pack')
    if args.native_order_policy != 'ordinary_first' and args.native_pack_index_sha256 is None:
        raise ValueError('native order option requires a pinned native pack')
    if args.prefix_replay_sha256 is not None and (len(args.prefix_replay_sha256) != 64
            or args.decode_lse_attention or prefill_plan != db610_prefill_plan()
            or any((args.diagnose_speculative_verifier,args.diagnose_layerwise,
                    args.diagnose_ablation,args.diagnose_request_loop,args.question_sha256,
                    args.native_pack_index_sha256,args.native_suite_sha256))):
        raise ValueError('prefix replay requires canonical target and no other diagnostics')
    if args.diagnose_speculative_verifier and (args.decode_lse_attention or prefill_plan != db610_prefill_plan()):
        raise ValueError('verifier diagnostics require the canonical D1/D8/D10 and B128/B114 baseline')
    if (args.verifier_small_expert_tiles or args.verifier_rowwise_dsa) and not args.diagnose_speculative_verifier:
        raise ValueError('verifier options require --diagnose-speculative-verifier')
    if args.question_sha256 is not None and (len(args.question_sha256) != 64
            or args.decode_lse_attention or prefill_plan != db610_prefill_plan()
            or any((args.diagnose_speculative_verifier,args.diagnose_layerwise,
                    args.diagnose_ablation,args.diagnose_request_loop))):
        raise ValueError('fresh question requires the canonical ordinary path without extra diagnostics')
    if args.native_pack_index_sha256 is not None and (len(args.native_pack_index_sha256)!=64
            or args.decode_lse_attention or prefill_plan!=db610_prefill_plan()
            or any((args.diagnose_speculative_verifier,args.diagnose_layerwise,
                    args.diagnose_ablation,args.diagnose_request_loop))):
        raise ValueError('native comparison requires the canonical target and no other diagnostics')
    if args.native_suite_sha256 is not None and (len(args.native_suite_sha256)!=64
            or args.native_pack_index_sha256 is None or args.question_sha256 is not None):
        raise ValueError('native suite requires a native pack and replaces the single-question input')
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
    question = None
    suite_cases = ()
    native_pack_index = None
    replay_cases = ()
    ordinary_cases = ()
    prefill_admission = None
    if args.ordinary_suite_sha256 is not None:
        from glm_tpu.perf.native_suite import load_native_suite
        model_config=json.loads((REPO/'configs/glm-5.2-fp8-config.json').read_bytes())
        eos=model_config['eos_token_id']
        ordinary_cases=load_native_suite(root,args.ordinary_suite_sha256,capacity=8192,
            vocab_size=model_config['vocab_size'],eos_ids=(eos,) if type(eos) is int else tuple(eos))
    if args.ordinary_prefill_admission_sha256 is not None:
        raw=(root/'ordinary_prefill_admission.json').read_bytes()
        if sha256(raw).hexdigest()!=args.ordinary_prefill_admission_sha256:
            raise ValueError('ordinary prefill admission receipt digest differs')
        prefill_admission=json.loads(raw)
        if (prefill_admission.get('schema')!='glm_perf_real_globalmax_prefill_v1'
                or prefill_admission.get('all_hosts_idle_after') is not True
                or prefill_admission.get('fleet_summary_passed') is not True
                or prefill_admission['summary'].get('db610_token_check_passed') is not True
                or prefill_admission['summary'].get('prefill_global_max_attention') is not True):
            raise ValueError('global-max prefill did not pass the trained DB610 gate')
    if args.prefix_replay_sha256 is not None:
        from glm_tpu.perf.prefix_replay import load_prefix_replay
        model_config=json.loads((REPO/'configs/glm-5.2-fp8-config.json').read_bytes())
        eos=model_config['eos_token_id']
        replay_cases=load_prefix_replay(root,args.prefix_replay_sha256,capacity=8192,
            vocab_size=model_config['vocab_size'],eos_ids=(eos,) if type(eos) is int else tuple(eos))
    if args.native_pack_index_sha256 is not None:
        raw_index=(root/'native_pack_index.json').read_bytes()
        if sha256(raw_index).hexdigest()!=args.native_pack_index_sha256:
            raise ValueError('native pack index hash differs')
        native_pack_index=json.loads(raw_index)
    if args.question_sha256 is not None:
        from glm_tpu.perf.long_question import load_question
        model_config=json.loads((REPO/'configs/glm-5.2-fp8-config.json').read_bytes())
        eos=model_config['eos_token_id']
        question=load_question(root/'question.json',args.question_sha256,capacity=8192,
            vocab_size=model_config['vocab_size'],eos_ids=(eos,) if type(eos) is int else tuple(eos))
    if args.native_suite_sha256 is not None:
        from glm_tpu.perf.native_suite import load_native_suite
        model_config=json.loads((REPO/'configs/glm-5.2-fp8-config.json').read_bytes())
        eos=model_config['eos_token_id']
        suite_cases=load_native_suite(root,args.native_suite_sha256,capacity=8192,
            vocab_size=model_config['vocab_size'],eos_ids=(eos,) if type(eos) is int else tuple(eos))
    args.process_id=int(socket.gethostname().rsplit('-w-',1)[1])
    from scripts.release.ws32_user_worker import site_args
    args=site_args(args)
    from glm_tpu.perf.topology_binding import apply_topology_binding
    topology_binding=apply_topology_binding(args,root,args.topology_rebinding_sha256)
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
        write_empty_route_slot=args.write_empty_route_slot,
        decode_lse_attention=args.decode_lse_attention,
        diagnose_speculative_verifier=args.diagnose_speculative_verifier,
        hlo_storage='shm' if args.hlo_in_shm else 'root',
        prefill_plan=prefill_plan,
        prefill_global_max_attention=args.prefill_global_max_attention,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    receipt=root/f'validation.rank{rank}.json'
    if topology_binding is not None:
        record['topology_rebinding_sha256']=topology_binding['sha256']
    hlo_root=root/f'hlo.rank{rank}'
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
        def allocate_hlo_storage():
            if args.hlo_in_shm:
                fs=os.statvfs('/dev/shm')
                require(fs.f_bavail*fs.f_frsize>4*1024**3,'insufficient shared memory for HLO artifacts')
                target=Path('/dev/shm/glm-perf-hlo')/root.name/hlo_root.name
                target.mkdir(parents=True,exist_ok=False)
                hlo_root.symlink_to(target,target_is_directory=True)
            else:
                hlo_root.mkdir(exist_ok=False)
        phase('hlo_storage',allocate_hlo_storage)
        local_ids={int(d.id) for d in jax.local_devices()}
        slots=tuple(s for s,d in enumerate(physical.flattened_device_ids) if d in local_ids)
        require(len(slots)==4,'expected four checkpoint slots')
        record['physical_identity']=dict(mesh_sha256=physical.mesh_hash,topology_sha256=topology.topology_hash,
            fleet_sha256=fleet_sha,local_slots=slots)
        config=dec.Ws32DecoderConfig(original._geometry(),8192,host_main_rope_table=True)
        decode_program=phase('decoder_configuration',lambda:build_db610_decoder(mesh,config,
            lse_attention=args.decode_lse_attention,write_empty_slot=args.write_empty_route_slot))
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
        block_rows=prefill_plan['block_rows']
        blocks=[put(prompt[i:i+block_rows]) for i in range(0,len(prompt),block_rows)]
        counts={n:put(np.int32(n)) for n in (block_rows,prefill_plan['tail_rows'])}
        options=dict(key_tile=512,mlp_window=True,rolled_prefix=True,expert_panels=True,
                     paired_position_sort=True,sorted_local_merge=True,canonical_dense=True)
        record['prefill_options']=options
        prefill={}
        for block in (blocks[0],blocks[-1]):
            rows=block.shape[0]
            program=build_ws32_prefill_challenger_program(mesh,config,
                lse_attention=not args.prefill_global_max_attention,bf16_resident=True,
                global_max_attention=args.prefill_global_max_attention,
                block_rows=rows,pooled_moe=prefill_plan['pooled_moe'],
                owned_key_capacity=prefill_plan['owned_key_capacity'],
                wide_indexshare=prefill_plan.get('wide_indexshare',False),**options)
            prefill[rows]=compile_model('prefill_'+str(rows),program.execute,(block,counts[rows],initial,weights,wk,rope))
        decode=compile_model('decode',decode_program.execute,(put(np.array([0],np.int32)),initial.decoder,weights,rope))
        # Re-admit with all three resident model executables present.
        for name,exe in [*((f'prefill_{n}',exe) for n,exe in prefill.items()),('decode',decode)]:admit(name,exe)
        warm=prefill[block_rows](blocks[0],counts[block_rows],initial,weights,wk,rope)
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
            result=prefill[n](block,counts[n],current,weights,wk,rope)
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
        request_prefill=result if args.diagnose_request_loop else None
        observed=[int(np.asarray(token)[0])]
        state=current.decoder
        verifier_initial=state if args.diagnose_speculative_verifier else None
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
            record['decode_ablations']={}
            for label,lse,two_stage in [('d1_d8',False,False),('d1_d8_d5',True,False),('d1_d8_d10',False,True)]:
                program=build_db610_decoder(mesh,config,lse_attention=lse,dsa_two_stage=two_stage,
                                           write_empty_slot=args.write_empty_route_slot)
                name='ablation_'+label
                fn=compile_model(name,program.execute,(token,state,weights,rope))
                record['ablation_programs'][name]=record['programs'].pop(name)
                from glm_tpu.perf.decode_diagnostics import collect_greedy_trail
                def ablation_step(index,current_token,current_state):
                    return phase(name+'_step_'+str(index),lambda:jax.block_until_ready(fn(current_token,current_state,weights,rope)))
                values,comparison=collect_greedy_trail(ablation_step,token,state,expected,
                    cache_check=lambda s:phase(name+'_cache_check',lambda:cache_finite(s)))
                np.savez(root/f'ablation.{label}.rank{rank}.npz',tokens=values)
                record['decode_ablations'][label]=dict(lse_attention=lse,dsa_two_stage=two_stage,
                    **comparison,arithmetic_unchanged_from_existing_variant=True)
                save()
                program.execute.clear_cache()
                del values,comparison,fn,program
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
        if args.diagnose_request_loop:
            from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
            from glm_tpu.perf.request_loop import build_packed_decoder_program
            from glm_tpu.perf.request_loop_diagnostic import measure_request_trail
            from glm_tpu.perf.decode_diagnostics import activation_stats
            phase('request_loop_source_admission',lambda:require(record['token_comparison']['all_equal'],
                'request loop requires the complete matching model trail'))
            eos=json.loads((REPO/'configs/glm-5.2-fp8-config.json').read_bytes())['eos_token_id']
            policy=RequestPolicy('perf-db610-greedy',0,2034,29,config.context_capacity,
                config.geometry.vocab_size,(eos,) if type(eos) is int else tuple(eos))
            packed_program=build_packed_decoder_program(mesh,config,options=decode_program.options)
            packed=compile_model('request_loop_packed',packed_program.execute,
                (request_prefill.next_token,request_prefill.state.decoder,weights,rope))
            record['request_loop_program']=record['programs'].pop('request_loop_packed')
            record['request_loops']={}
            trails,finals={},{}
            for label in ('legacy','packed'):
                step=(lambda t,s:packed(t,s,weights,rope)) if label=='packed' else (
                    lambda t,s,u:decode(t,s,weights,rope))
                trail,report,final,last=phase('request_loop_'+label,lambda:measure_request_trail(
                    policy,request_prefill,expected,decode_step=step,packed=label=='packed',
                    replicate_uniform=put,fleet_all=original._batched_fleet_all))
                report['final_cache_finite']=phase('request_loop_'+label+'_cache_check',lambda:cache_finite(final))
                report['final_residual']=None if last is None else activation_stats(last.final_residual_local)
                report['legacy_uniform_ignored']=label=='legacy'
                record['request_loops'][label]=report
                np.savez(root/f'request_loop.{label}.rank{rank}.npz',tokens=trail)
                trails[label]=trail
                finals[label]=(final,None if last is None else last.final_residual_local)
                save()
            a,b=finals['legacy'],finals['packed']
            same=jax.tree.structure(a)==jax.tree.structure(b)
            if same:
                for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
                    same &= x.shape==y.shape and x.dtype==y.dtype and all(np.array_equal(
                        np.ascontiguousarray(xs.data).view(np.uint8),np.ascontiguousarray(ys.data).view(np.uint8))
                        for xs,ys in zip(x.addressable_shards,y.addressable_shards))
            record['request_loop_agreement']=dict(all_tokens_equal=np.array_equal(trails['legacy'],trails['packed']),
                final_state_and_residual_bitwise_equal=phase('request_loop_state_agreement',
                    lambda:original._batched_fleet_all(bool(same))))
            packed_program.execute.clear_cache()
        if args.diagnose_speculative_verifier:
            from glm_tpu.perf.speculative_diagnostics import compare_reference_trail
            from glm_tpu.perf.speculative_verify import build_verifier,build_prefix_committer
            from tools.perf_speculative_verify import local_comparison,local_cache_span_comparison
            phase('verifier_reference_admission',lambda:require(record['token_comparison']['all_equal'],
                'verifier comparison requires all 29 ordinary DB610 tokens'))
            record['speculative_programs']={}
            record['speculative_verifier']={}
            verifier_options=dict(canonical_mlp=True,batched_attention=True,
                small_expert_tiles=args.verifier_small_expert_tiles,rowwise_dsa=args.verifier_rowwise_dsa)
            record['speculative_verifier_options']=verifier_options
            record['speculative_cache_comparison_scope']='whole_and_written_span_v1'
            for n in (2,3):
                label='speculative_verify_'+str(n)
                program=build_verifier(mesh,config,**verifier_options)
                verifier=compile_model(label,program,(put(expected[:n]),verifier_initial,weights,rope))
                record['speculative_programs'][label]=record['programs'].pop(label)
                proposal=phase(label+'_warm_first',lambda:jax.block_until_ready(
                    verifier(put(expected[:n]),verifier_initial,weights,rope)))
                commit_program=build_prefix_committer(mesh,config)
                commit_name='speculative_commit_'+str(n)
                committer=compile_model(commit_name,commit_program,(verifier_initial,proposal,put(np.int32(n))))
                record['speculative_programs'][commit_name]=record['programs'].pop(commit_name)
                def warm_graphs():
                    for _ in range(5):
                        warmed=jax.block_until_ready(verifier(put(expected[:n]),verifier_initial,weights,rope))
                        committed=jax.block_until_ready(committer(verifier_initial,warmed,put(np.int32(n))))
                        require(np.asarray(warmed.contract_valid).all() and np.asarray(committed.contract_valid).all(),
                            'verifier warmup health failed')
                phase(label+'_warm',warm_graphs)
                def health(value,name):
                    phase(label+'_'+name,lambda:require(np.asarray(value).all(),'verifier diagnostic health failed'))
                raw,report,candidate,reference=phase(label+'_trail',lambda:compare_reference_trail(
                    expected,verifier_initial,rows=n,
                    verify=lambda t,s:verifier(t,s,weights,rope),
                    commit=committer,ordinary=lambda t,s:decode(t,s,weights,rope),
                    replicate=put,ready=jax.block_until_ready,healthy=health,compare=local_comparison))
                report['written_cache_comparisons']=phase(label+'_written_cache_comparison',lambda:{
                    name:local_cache_span_comparison(getattr(candidate,name),getattr(reference,name),
                        reference.block_tables,int(np.asarray(verifier_initial.position)[0]),
                        int(np.asarray(reference.position)[0]))
                    for name in ('kv_cache_local','index_cache_local')})
                report['memory_after']=stats()
                report['warmup_pairs']=5
                report['verifier_options']=verifier_options
                np.savez(root/f'verifier.{n}.rank{rank}.npz',predictions=raw)
                record['speculative_verifier'][str(n)]=report
                save()
                program.clear_cache();commit_program.clear_cache()
                del proposal,raw,report,candidate,reference,verifier,committer,program,commit_program
                gc.collect()
        if replay_cases:
            from glm_tpu.perf.prefix_replay import run_prefix_replay
            phase('prefix_replay_reference_admission',lambda:require(
                record['token_comparison']['all_equal'],'prefix replay requires ordinary DB610 parity'))
            record['prefix_replay_sha256']=args.prefix_replay_sha256
            record['prefix_replay_trace']=args.prefix_replay_trace
            record['prefix_replay_unrolled_attention']=args.prefix_replay_unrolled_attention
            record['prefix_replay_global_max_attention']=args.prefix_replay_global_max_attention
            record['prefix_replay_timing_iters']=args.prefix_replay_timing_iters
            run_prefix_replay(cases=replay_cases,mesh=mesh,config=config,weights=weights,
                rope=rope,wk=wk,prefill=prefill,decode=decode,compile_model=compile_model,
                phase=phase,require=require,put=put,record=record,save=save,stats=stats,
                trace_layers=args.prefix_replay_trace,ordinary_options=decode_program.options,
                unrolled_attention=args.prefix_replay_unrolled_attention,
                global_max_attention=args.prefix_replay_global_max_attention,
                timing_iters=args.prefix_replay_timing_iters)
        if ordinary_cases:
            from glm_tpu.perf.ordinary_suite import run_ordinary_suite
            if prefill_admission is not None:
                for key in ('manifest_sha256','success_sha256','inventory_sha256'):
                    require(record['checkpoint'][key]==prefill_admission['summary']['checkpoint'][key],
                            'prefill admission weights differ from this execution')
            record['ordinary_suite_sha256']=args.ordinary_suite_sha256
            record['ordinary_suite_compare_prefill']=args.ordinary_suite_compare_prefill
            record['ordinary_prefill_admission_sha256']=args.ordinary_prefill_admission_sha256
            del state,blocks,result
            gc.collect()
            run_ordinary_suite(root=root,cases=ordinary_cases,suite_sha256=args.ordinary_suite_sha256,
                compare_prefill=args.ordinary_suite_compare_prefill,mesh=mesh,config=config,
                weights=weights,wk=wk,rope=rope,prefill=prefill,prefill_options=options,
                decode_options=decode_program.options,rank=rank,record=record,phase=phase,
                require=require,compile_model=compile_model,admit=admit,stats=stats,
                fleet_all=original._batched_fleet_all,put=put,save=save)
        if question is not None and native_pack_index is None:
            from glm_tpu.perf.long_question import question_blocks,measure_question
            from glm_tpu.perf.request_loop import build_packed_decoder_program
            from dataclasses import asdict
            phase('question_reference_admission',lambda:require(record['token_comparison']['all_equal'],
                'fresh question requires all 29 DB610 tokens'))
            ids,policy=question
            record['question_identity']=dict(file_sha256=args.question_sha256,
                prompt_ids_sha256=sha256(ids.tobytes()).hexdigest(),policy=asdict(policy))
            packed_program=build_packed_decoder_program(mesh,config,options=decode_program.options)
            packed=compile_model('question_packed',packed_program.execute,(token,state,weights,rope))
            # Dispose old request cache roots before allocating a fresh prompt.
            del state,blocks,result
            gc.collect()
            fresh=pre.make_ws32_batched_prefill_state(mesh,config,prompt_length=len(ids))
            staged=[(put(block),put(np.int32(count))) for block,count in question_blocks(ids)]
            for rows,exe in prefill.items():admit('prefill_'+str(rows),exe)
            admit('question_packed',packed)
            started=time.perf_counter()
            for i,(block,count) in enumerate(staged):
                result=phase('question_prefill_'+str(i),lambda:jax.block_until_ready(
                    prefill[block.shape[0]](block,count,fresh,weights,wk,rope)))
                fresh=result.state
                phase('question_prefill_health_'+str(i),lambda:require(
                    np.asarray(fresh.decoder.contract_valid).all(),'question prefill unhealthy'))
            record['question_prefill']=dict(prompt_tokens=len(ids),wall_seconds=time.perf_counter()-started)
            record['question_prefill']['prompt_tokens_per_second']=len(ids)/record['question_prefill']['wall_seconds']
            with (root/f'question.tokens.rank{rank}.jsonl').open('x') as sink:
                def deliver(event):
                    if rank==0:
                        sink.write(json.dumps(asdict(event),separators=(',',':'))+'\n')
                        sink.flush()
                values,report,final=phase('question_generation',lambda:measure_question(policy,result,
                    decode_step=lambda t,s:packed(t,s,weights,rope),replicate_uniform=put,
                    fleet_all=original._batched_fleet_all,deliver=deliver,request_started=started))
            np.savez(root/f'question.generated.rank{rank}.npz',tokens=values)
            from jax.experimental import multihost_utils
            def agree():
                hashes=np.asarray(multihost_utils.process_allgather(
                    np.frombuffer(bytes.fromhex(report['token_sha256']),np.uint8)))
                require((hashes==hashes[0]).all(),'fresh question token hashes differ across hosts')
                return True
            report['all_host_token_agreement']=phase('question_token_agreement',agree)
            def check_question_cache():
                require(cache_finite(final),'question cache contains nonfinite values')
                return True
            report['final_cache_finite']=phase('question_cache_check',check_question_cache)
            report['memory_after']=stats()
            record['question']=report
            save()
        if native_pack_index is not None:
            from glm_tpu.perf.native_comparison import run_native_comparison
            phase('native_reference_admission',lambda:require(record['token_comparison']['all_equal'],
                'native comparison requires ordinary DB610 parity'))
            record['native_pack_index_sha256']=args.native_pack_index_sha256
            if args.native_suite_sha256 is not None:
                record['native_suite_sha256']=args.native_suite_sha256
            del state,blocks,result
            gc.collect()
            run_native_comparison(root=root,pack_index=native_pack_index,mesh=mesh,physical=physical,
                config=config,inventory=inventory,weights=weights,wk=wk,rope=rope,prompt=prompt,
                expected=expected,question=question,decode_options=decode_program.options,
                prefill_options=options,rank=rank,record=record,phase=phase,require=require,
                compile_model=compile_model,stats=stats,fleet_all=original._batched_fleet_all,save=save,
                suite_cases=suite_cases,component_timing=args.native_component_timing,
                order_policy=args.native_order_policy,acceptance_mode=args.native_acceptance)
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
