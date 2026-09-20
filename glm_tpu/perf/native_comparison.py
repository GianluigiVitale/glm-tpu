"""Leased worker's real-weight native MTP comparison; no standalone launcher.

The caller supplies admitted target weights, authenticated inputs/physical mesh,
compile/memory/phase guards and leases. All payload outputs remain private.
"""
from dataclasses import asdict
from hashlib import sha256
import gc
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from ..greenfield.runtime import ws32_batched_prefill as pre
from ..greenfield.runtime.ws32_request_session import RequestPolicy
from .long_question import question_blocks, measure_question
from .mtp_checkpoint import bind_mtp_arrays
from .mtp_draft import mtp_config
from .mtp_pack import build_native_pack_plan, load_native_arrays
from .mtp_state import TargetHistory, make_native_state, build_native_refresh, build_native_inputs
from .native_pairing import COMPARISON_PROTOCOL, comparison_order, run_paired_modes
from .prefill_challenger import build_ws32_prefill_challenger_program
from .real_validation import memory_projection
from .request_loop import build_packed_decoder_program
from .speculative_request import SpeculativeRequestSession
from .speculative_plan import build_planned_verifier
from .speculative_verify import build_verifier, build_prefix_committer


def run_native_comparison(*, root, pack_index, mesh, physical, config, inventory,
        weights, wk, rope, prompt, expected, question, decode_options, prefill_options,
        rank, record, phase, require, compile_model, stats, fleet_all, save, suite_cases=(),
        component_timing='blocking', order_policy='ordinary_first', acceptance_mode='host'):
    from jax.experimental import multihost_utils
    if component_timing not in ('blocking', 'none'):
        raise ValueError('native component timing must be blocking or none')
    record['native_component_timing'] = component_timing
    comparison_order('db610', order_policy)
    record['native_order_policy'] = order_policy
    if acceptance_mode not in ('host', 'device'):
        raise ValueError('native acceptance must be host or device')
    record['native_acceptance'] = acceptance_mode
    eos=json.loads((Path(__file__).resolve().parents[2]/'configs/glm-5.2-fp8-config.json').read_text())['eos_token_id']
    eos=(eos,) if type(eos) is int else tuple(eos)
    policies=(() if question is None else (question[1],))+tuple(case[2] for case in suite_cases)
    if acceptance_mode == 'device' and any(policy.eos_ids != eos for policy in policies):
        raise ValueError('compiled device acceptance requires the same model EOS policy in every case')
    report = dict(schema='glm_native_mtp_comparison_v1',complete=False,cases={},
                  sampled=False,independent_native_reference=False,
                  comparison_protocol=COMPARISON_PROTOCOL,acceptance_mode=acceptance_mode)
    record['native_mtp'] = report
    record['native_programs'] = {}
    def compile_native(name, fn, args):
        name='native_'+name
        exe=compile_model(name,fn,args)
        record['native_programs'][name]=record['programs'].pop(name)
        save()
        return exe
    def admission(name):
        memory=record['native_programs']['native_'+name]['compiled_memory']
        result=memory_projection(stats(),memory)
        require(result['passed'],'native live memory admission refused: '+name)
        return result
    def agree(value):
        gathered=np.asarray(multihost_utils.process_allgather(value))
        return bool((gathered==gathered[0]).all())
    def put(x):return jax.device_put(x,NamedSharding(mesh,P()))
    def healthy(state):return bool(np.asarray(state.contract_valid).all())
    def cache_finite(state):
        return all(np.isfinite(np.asarray(s.data)).all()
            for a in (state.kv_cache_local,state.index_cache_local) for s in a.addressable_shards)
    def budget(extra):
        value=memory_projection(stats(),extra)
        require(value['passed'],'native weight allocation exceeds memory bound')
        return value
    plan=build_native_pack_plan(inventory,config)
    require(pack_index['schema']=='glm_native_mtp_pack_index_v1'
        and pack_index['source_inventory_sha256']==inventory.inventory_sha256
        and pack_index['plan_sha256']==plan.plan_sha256
        and pack_index['mesh_sha256']==physical.mesh_hash
        and pack_index['all_hosts_idle_after'] is True,'native pack index identity differs')
    owner=pack_index['rank_manifests'][str(rank)]
    report['pack_index']=dict(plan_sha256=plan.plan_sha256,manifest_sha256=owner['manifest_sha256'])
    report['load_admission']=phase('native_load_admission',lambda:budget(dict(
        output_size_in_bytes=plan.payload_bytes_per_slot,
        temp_size_in_bytes=2*max(t.byte_count for t in plan.tensors),
        generated_code_size_in_bytes=0,alias_size_in_bytes=0)))
    arrays=phase('native_load',lambda:load_native_arrays(owner['root'],owner['manifest_sha256'],plan,
        mesh=mesh,physical_mesh=physical))
    report['resident_admission']=phase('native_resident_admission',lambda:budget(dict(
        output_size_in_bytes=2*plan.payload_bytes_per_slot,
        temp_size_in_bytes=6*max(t.byte_count for t in plan.tensors),
        generated_code_size_in_bytes=128*1024**2,alias_size_in_bytes=0)))
    native_weights=phase('native_bind',lambda:jax.block_until_ready(bind_mtp_arrays(arrays,weights,mesh,config)))
    del arrays
    gc.collect()
    report['weights_memory_after']=stats()
    native_config=mtp_config(config)
    empty=make_native_state(mesh,native_config)
    initial=pre.make_ws32_batched_prefill_state(mesh,config,prompt_length=len(prompt))
    prefill={}
    for rows in (128,114):
        fn=build_ws32_prefill_challenger_program(mesh,config,lse_attention=True,bf16_resident=True,
            block_rows=rows,export_mtp_hidden=True,**prefill_options).execute
        prefill[rows]=compile_native('prefill_'+str(rows),fn,
            (put(np.zeros(rows,np.int32)),put(np.int32(rows)),initial,weights,wk,rope))
    packed_program=build_packed_decoder_program(mesh,config,options=decode_options)
    packed=compile_native('ordinary',packed_program.execute,
                         (put(np.array([0],np.int32)),initial.decoder,weights,rope))
    refresh={};inputs={};verify={};commit={}
    # Bootstrap batches of eight and compile each registered prompt's exact
    # tail without padding or changing its live frontier.
    refresh_sizes=sorted({1,2,3,8,len(prompt)%8,*(() if question is None else (len(question[0])%8,)),
                          *(len(case[1])%8 for case in suite_cases)}-{0})
    for rows in refresh_sizes:
        history=TargetHistory(empty.cache.position,put(np.zeros(rows,np.int32)),
            jax.device_put(jnp.zeros((rows,config.geometry.hidden_size),jnp.bfloat16),
                           NamedSharding(mesh,P(None,'feature'))),put(np.ones(rows,bool)))
        refresh[rows]=compile_native('refresh_'+str(rows),build_native_refresh(mesh,native_config),
            (empty,history,put(np.int32(rows)),native_weights,rope))
    for rows in (1,2,3):
        inputs[rows]=compile_native('inputs_'+str(rows),build_native_inputs(mesh,native_config,rows=rows),
            (put(np.array([0],np.int32)),empty,native_weights,rope))
        options=dict(canonical_mlp=True,batched_attention=True,
                     small_expert_tiles=True,rowwise_dsa=True)
        fn=(build_planned_verifier(mesh,config,eos_ids=eos,**options) if acceptance_mode=='device'
            else build_verifier(mesh,config,**options))
        args=(put(np.zeros(rows,np.int32)),initial.decoder,weights,rope)
        if acceptance_mode=='device':args=(*args,put(np.int32(rows)))
        verify[rows]=compile_native('verify_'+str(rows),fn,args)
        shape=jax.eval_shape(fn,*args)
        if acceptance_mode=='device':shape=shape.proposal
        commit[rows]=compile_native('commit_'+str(rows),build_prefix_committer(mesh,config),
                                   (initial.decoder,shape,put(np.int32(rows))))
    del initial,history,shape,args,fn,empty
    gc.collect()
    report['all_executables_memory']=stats()

    def prompt_export(ids,label):
        current=pre.make_ws32_batched_prefill_state(mesh,config,prompt_length=len(ids))
        hidden=[];started=time.perf_counter()
        for i,(block,count) in enumerate(question_blocks(ids)):
            admission('prefill_'+str(block.size))
            result=phase(label+'_prefill_'+str(i),lambda:jax.block_until_ready(prefill[block.size](
                put(block),put(np.int32(count)),current,weights,wk,rope)))
            current=result.state
            phase(label+'_prefill_health_'+str(i),lambda:require(healthy(current.decoder)
                and np.asarray(result.hidden_valid)[:count].all(),'native target export unhealthy'))
            hidden.append(result.normalized_hidden_local[:count])
        elapsed=time.perf_counter()-started
        return result,jnp.concatenate(hidden,axis=0),dict(prompt_tokens=len(ids),wall_seconds=elapsed,
             prompt_tokens_per_second=len(ids)/elapsed)

    def bootstrap(ids,prefilled,hidden,label):
        native=make_native_state(mesh,native_config)
        shifted=jnp.concatenate((put(ids[1:]),prefilled.next_token))
        started=time.perf_counter()
        for start in range(0,len(ids),8):
            rows=min(8,len(ids)-start)
            history=TargetHistory(put(np.array([start],np.int32)),shifted[start:start+rows],
                                 hidden[start:start+rows],put(np.ones(rows,bool)))
            admission('refresh_'+str(rows))
            native=jax.block_until_ready(refresh[rows](native,history,put(np.int32(rows)),native_weights,rope))
            require(fleet_all(healthy(native.cache)),'native bootstrap health failed')
        require(int(np.asarray(native.cache.position)[0])==len(ids),'native bootstrap frontier differs')
        return native,time.perf_counter()-started

    components={'draft':[],'verify':[],'commit':[],'refresh':[]}
    def timed(name, action):
        if component_timing == 'none':
            # Executables retain their data dependencies. The request session
            # still waits for proposal and commit/refresh before votes/delivery.
            # Do not misreport asynchronous dispatch duration as device time.
            return action()
        started=time.perf_counter()
        result=jax.block_until_ready(action())
        components[name].append(time.perf_counter()-started)
        return result
    def propose(pending,target,native,rows):
        ids=timed('draft',lambda:inputs[rows](pending,native,native_weights,rope))
        proposal=timed('verify',lambda:verify[rows](ids,target,weights,rope))
        return ids,proposal
    def propose_planned(pending,target,native,rows,remaining):
        ids=timed('draft',lambda:inputs[rows](pending,native,native_weights,rope))
        return timed('verify',lambda:verify[rows](ids,target,weights,rope,put(np.int32(remaining))))
    def commit_target(old,proposal,count):
        return timed('commit',lambda:commit[proposal.predictions.size](old,proposal,count))
    def refresh_target(old,history,count):
        return timed('refresh',lambda:refresh[history.shifted_tokens.size](old,history,count,native_weights,rope))
    def sink(stream):
        def deliver(event):
            if rank==0:
                stream.write(json.dumps(asdict(event),separators=(',',':'))+'\n');stream.flush()
        return deliver
    def speculative(policy,prefilled,native,rows,label,started):
        votes=[];agreements=[]
        for values in components.values():values.clear()
        def vote(value):
            before=time.perf_counter();result=fleet_all(value);votes.append(time.perf_counter()-before);return result
        def consensus(value):
            before=time.perf_counter();result=agree(value);agreements.append(time.perf_counter()-before);return result
        with (root/f'{label}.tokens.rank{rank}.jsonl').open('x') as stream:
            session=SpeculativeRequestSession(policy,native_state=native,propose=propose,commit=commit_target,
                propose_planned=propose_planned if acceptance_mode=='device' else None,
                refresh=refresh_target,replicate_count=put,fleet_agree=consensus,rows=rows,
                decode_step=None,replicate_uniform=put,fleet_all=vote,deliver=sink(stream),
                delivery_boundary='rank0 private JSONL token write+flush; no network transport',request_started=started)
            # Use the ordinary two-field request interface for first delivery.
            ordinary=pre.Ws32BatchedPrefillResult(prefilled.state,prefilled.next_token)
            session.accept_prefill(ordinary)
            first=time.perf_counter();votes.clear()
            while not session.finished:session.step()
            elapsed=time.perf_counter()-first
        tokens=np.asarray([e.token_id for e in session.events],np.int32)
        counts=[r['emitted'] for r in session.rounds]
        result=dict(rows=rows,healthy=not session.failed,emitted=len(tokens),decode_tokens=len(tokens)-1,
            wall_seconds=elapsed,tokens_per_second=(len(tokens)-1)/elapsed,
            rounds=len(counts),accepted_per_round=float(np.mean(counts)) if counts else None,
            accepted_drafts=sum(r['accepted_drafts'] for r in session.rounds),
            proposed_drafts=sum(r['proposed_drafts'] for r in session.rounds),
            accepted_by_position=[sum(r['accepted_drafts']>i for r in session.rounds) for i in range(rows-1)],
            proposals_by_position=[sum(r['proposed_drafts']>i for r in session.rounds) for i in range(rows-1)],
            proposal_seconds=sum(r['proposal_seconds'] for r in session.rounds),
            refresh_commit_seconds=sum(r['refresh_commit_seconds'] for r in session.rounds),
            host_vote_seconds=sum(votes),host_agreement_seconds=sum(agreements),
            p50_round_ms=float(np.median([r['wall_seconds'] for r in session.rounds])*1000) if counts else None,
            ttft_seconds=first-started,request_wall_seconds=time.perf_counter()-started,
            finish_reason=session.events[-1].finish_reason,token_sha256=sha256(tokens.tobytes()).hexdigest(),
            warm_steps_excluded=0,delivery_boundary=session.delivery_boundary,
            includes_draft_verify_rejections_refresh_commit_votes_delivery=True,
            excludes_cold_load_compile=True,decode_rate_excludes_prefill=True)
        result['component_seconds']=({name:sum(values) for name,values in components.items()}
                                     if component_timing == 'blocking' else None)
        result['component_timing_scope']=('synchronized device calls inside measured wall time'
            if component_timing == 'blocking' else 'disabled; request synchronization and wall timing retained')
        result['all_host_token_agreement']=phase(label+'_agreement',lambda:agree(np.frombuffer(bytes.fromhex(result['token_sha256']),np.uint8)))
        require(result['all_host_token_agreement'],'native accepted tokens differ across hosts')
        result['finite_caches']=phase(label+'_finite',lambda:cache_finite(session._state) and cache_finite(session._native.cache))
        require(result['finite_caches'],'native final caches contain nonfinite values')
        result['memory_after']=stats()
        np.savez(root/f'{label}.generated.rank{rank}.npz',tokens=tokens)
        session.release()
        return tokens,result

    cases=[('db610',prompt,RequestPolicy('native-db610',0,len(prompt),len(expected),config.context_capacity,
                                        config.geometry.vocab_size,eos),expected)]
    if question is not None:cases.append(('question',question[0],question[1],None))
    cases.extend(suite_cases)
    for label,ids,policy,reference in cases:
        # All modes warm before any measured continuation. Every measured mode
        # then builds a fresh prefill root, including ordinary when it runs last.
        order=comparison_order(label,order_policy)
        pref,hidden,warm_prefill=prompt_export(ids,label+'_warmup')
        native,bootstrap_seconds=phase(label+'_bootstrap',lambda:bootstrap(ids,pref,hidden,label))
        del hidden
        # Re-admit all graphs at the actual live prompt/native-cache frontier.
        memory=phase(label+'_resident_admission',lambda:{name:admission(name.removeprefix('native_'))
            for name in record['native_programs']})
        case=dict(policy=asdict(policy),prompt_sha256=sha256(ids.tobytes()).hexdigest(),
            bootstrap_seconds=bootstrap_seconds,warmup_prefill=warm_prefill,
            execution_order=[],planned_order=list(order),fresh_prefill_per_mode=True,
            export_db610_parity=reference is not None,live_memory_admission=memory,speculative={})
        report['cases'][label]=case;save()
        # Warm each graph once on a disposable state. Do not consume output or
        # carry warm roots into either measured request.
        warmed_ordinary=jax.block_until_ready(packed(pref.next_token,pref.state.decoder,weights,rope))
        phase(label+'_warm_ordinary',lambda:require(healthy(warmed_ordinary.decoded.state),'ordinary warm health failed'))
        del warmed_ordinary
        for rows in (1,2,3):
            if acceptance_mode=='device':
                planned=propose_planned(pref.next_token,pref.state.decoder,native,rows,rows)
                inputs0,proposal=planned.metadata,planned.proposal
                del planned
            else:
                inputs0,proposal=propose(pref.next_token,pref.state.decoder,native,rows)
            target=commit_target(pref.state.decoder,proposal,put(np.int32(rows)))
            history=TargetHistory(pref.state.decoder.position,proposal.predictions,proposal.normalized_hidden_local,proposal.contract_valid)
            warmed=refresh_target(native,history,put(np.int32(rows)))
            jax.block_until_ready((target,warmed))
            phase(label+'_warm_'+str(rows),lambda:require(healthy(target) and healthy(warmed.cache),'native warm health failed'))
            del inputs0,proposal,target,history,warmed
        del pref,native
        gc.collect()

        def ordinary_request():
            started=time.perf_counter()
            pref,hidden,pref_report=prompt_export(ids,label)
            del hidden
            with (root/f'native.{label}.ordinary.tokens.rank{rank}.jsonl').open('x') as stream:
                ordinary=pre.Ws32BatchedPrefillResult(pref.state,pref.next_token)
                baseline,base_report,base_final=phase(label+'_ordinary',lambda:measure_question(policy,ordinary,
                    decode_step=lambda t,s:packed(t,s,weights,rope),replicate_uniform=put,
                    fleet_all=fleet_all,deliver=sink(stream),request_started=started))
            np.savez(root/f'native.{label}.ordinary.generated.rank{rank}.npz',tokens=baseline)
            phase(label+'_ordinary_agreement',lambda:require(agree(np.frombuffer(bytes.fromhex(base_report['token_sha256']),np.uint8)),
                'ordinary comparison tokens differ across hosts'))
            phase(label+'_ordinary_finite',lambda:require(cache_finite(base_final),'ordinary comparison cache is nonfinite'))
            if reference is not None:
                phase(label+'_export_parity',lambda:require(np.array_equal(baseline,reference),
                    'target prefill hidden export changed DB610 tokens'))
            case['prefill']=pref_report
            del base_final,ordinary,pref
            gc.collect()
            return baseline,base_report

        def speculative_request(rows):
            tag=f'native.{label}.r{rows}'
            request_started=time.perf_counter()
            pref,hidden,variant_prefill=prompt_export(ids,tag)
            native,variant_bootstrap=bootstrap(ids,pref,hidden,tag)
            del hidden
            tokens,result=phase(tag+'_generation',lambda:speculative(policy,pref,native,rows,tag,request_started))
            result['prefill']=variant_prefill
            result['bootstrap_seconds']=variant_bootstrap
            result['ttft_scope']='live warmed request: fresh target prefill, native bootstrap, first delivery'
            del pref,native
            gc.collect()
            return tokens,result

        def observe_mode(mode,result,completed):
            if mode=='ordinary':case['ordinary']=result
            else:case['speculative'][mode[1:]]=result
            case['execution_order']=list(completed)
            save()

        paired=run_paired_modes(label,order_policy=order_policy,ordinary=ordinary_request,
            speculative=speculative_request,observe=observe_mode)
        case['ordinary']=paired['ordinary']
        case['speculative']={key:paired['r'+key] for key in ('2','3')}
        save()
        del paired
        gc.collect()
    report['complete']=True
    save()
    return report
