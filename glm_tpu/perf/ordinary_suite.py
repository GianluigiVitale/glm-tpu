"""Fresh-state ordinary answer pairs; one model load, no speculative execution."""
from dataclasses import asdict
from hashlib import sha256
import gc
import json
import re
import time

import numpy as np

from .long_question import question_blocks, measure_question, summarize_question_rows


def ordinary_order(label, compare_prefill):
    match = re.fullmatch(r'[a-z][a-z0-9_]{0,23}_repeat([1-3])', label)
    if match is None or type(compare_prefill) is not bool:
        raise ValueError('ordinary suite needs an authenticated repeat label and explicit mode')
    modes = ('ordinary', 'optimized_ordinary') if compare_prefill else ('ordinary',)
    return modes if int(match[1]) % 2 else modes[::-1]


def output_agreement(tokens, baseline):
    common = min(len(tokens), len(baseline))
    mismatch = np.flatnonzero(tokens[:common] != baseline[:common])
    return dict(all_equal=bool(np.array_equal(tokens, baseline)),
        first_mismatch=int(mismatch[0]) if len(mismatch) else
            (common if len(tokens) != len(baseline) else None),
        baseline_token_sha256=sha256(baseline.tobytes()).hexdigest())


def run_ordinary_suite(*, root, cases, suite_sha256, compare_prefill, mesh, config,
        weights, wk, rope, prefill, prefill_options, decode_options, rank, record,
        phase, require, compile_model, admit, stats, fleet_all, put, save):
    """Reuse admitted ordinary decode; an optional variant changes prefill only.

    The caller must pass the ordinary DB610 gate before entry. Candidate results
    remain experimental until full paired output and answer checks are collected.
    All modes share request synchronization, rank0 write/flush and greedy policy.
    """
    import jax
    from jax.experimental import multihost_utils
    from ..greenfield.runtime import ws32_batched_prefill as pre
    from .prefill_challenger import build_ws32_prefill_challenger_program
    from .request_loop import build_packed_decoder_program

    require(record['token_comparison']['all_equal'], 'ordinary suite requires DB610 parity')
    report = dict(schema='glm_ordinary_suite_v1', complete=False, cases={},
        suite_sha256=suite_sha256, compare_prefill=compare_prefill,
        profiling='no per-component blocking timers',
        memory_scope='allocator process high-water mark through each completed mode',
        same_decode_program=True, fresh_state_per_mode=True)
    record['ordinary_suite'] = report
    initial = pre.make_ws32_batched_prefill_state(mesh, config, prompt_length=len(cases[0][1]))
    packed = compile_model('ordinary_suite_packed',
        build_packed_decoder_program(mesh, config, options=decode_options).execute,
        (put(np.array([0], np.int32)), initial.decoder, weights, rope))
    variants = {'ordinary': prefill}
    names = {'ordinary': {n: 'prefill_'+str(n) for n in prefill}}
    if compare_prefill:
        variants['optimized_ordinary'] = {}
        names['optimized_ordinary'] = {}
        for rows in (128, 114):
            name = 'ordinary_opt_prefill_'+str(rows)
            fn = build_ws32_prefill_challenger_program(mesh, config,
                bf16_resident=True, lse_attention=False, global_max_attention=True,
                block_rows=rows, **prefill_options).execute
            variants['optimized_ordinary'][rows] = compile_model(name, fn,
                (put(np.zeros(rows, np.int32)), put(np.int32(rows)), initial, weights, wk, rope))
            names['optimized_ordinary'][rows] = name
    del initial
    gc.collect()

    for label, ids, policy, _ in cases:
        order = ordinary_order(label, compare_prefill)
        question_name = 'case-'+label.rsplit('_repeat', 1)[0]+'.json'
        identity = dict(file_sha256=sha256((root/question_name).read_bytes()).hexdigest(),
            prompt_ids_sha256=sha256(ids.tobytes()).hexdigest(), policy=asdict(policy))
        case = dict(identity=identity, planned_order=list(order), execution_order=[], modes={})
        report['cases'][label] = case
        staged = [(put(block), put(np.int32(count))) for block, count in question_blocks(ids)]
        outputs = {}
        # Warm all modes before the pair, on disposable roots. Compilation and
        # warming stay outside measured TTFT; every measured mode starts fresh.
        for mode in order:
            # Finish this disposable prefix so it yields a valid decode token;
            # an unfinished full-prompt state deliberately returns token -1.
            fresh = pre.make_ws32_batched_prefill_state(mesh, config, prompt_length=min(len(ids),128))
            for n, exe in variants[mode].items(): admit(names[mode][n], exe)
            admit('ordinary_suite_packed', packed)
            block, count = staged[0]
            warm = phase(f'suite_{label}_{mode}_warm_prefill', lambda:jax.block_until_ready(
                variants[mode][block.size](block, count, fresh, weights, wk, rope)))
            decoded = phase(f'suite_{label}_{mode}_warm_decode', lambda:jax.block_until_ready(
                packed(warm.next_token, warm.state.decoder, weights, rope)))
            phase(f'suite_{label}_{mode}_warm_health', lambda:require(
                np.asarray(decoded.decoded.state.contract_valid).all(), 'ordinary suite warm health'))
            del fresh, warm, decoded
        gc.collect()
        for mode in order:
            tag = f'suite_{label}_{mode}'
            entry = dict(question_identity=identity, phases={})
            case['modes'][mode] = entry
            def checked(name, action):
                value = phase(tag+'_'+name, action)
                entry['phases'][name] = dict(record['phases'][tag+'_'+name])
                return value
            checked('question_reference_admission', lambda:require(
                record['token_comparison']['all_equal'], 'ordinary reference drift'))
            fresh = pre.make_ws32_batched_prefill_state(mesh, config, prompt_length=len(ids))
            for n, exe in variants[mode].items(): admit(names[mode][n], exe)
            admit('ordinary_suite_packed', packed)
            started = time.perf_counter()
            for i, (block, count) in enumerate(staged):
                result = checked('question_prefill_'+str(i), lambda:jax.block_until_ready(
                    variants[mode][block.size](block, count, fresh, weights, wk, rope)))
                fresh = result.state
                checked('question_prefill_health_'+str(i), lambda:require(
                    np.asarray(fresh.decoder.contract_valid).all(), 'ordinary suite prefill unhealthy'))
            elapsed = time.perf_counter()-started
            entry['question_prefill'] = dict(prompt_tokens=len(ids), wall_seconds=elapsed,
                prompt_tokens_per_second=len(ids)/elapsed)
            with (root/f'{tag}.tokens.rank{rank}.jsonl').open('x') as stream:
                def deliver(event):
                    if rank == 0:
                        stream.write(json.dumps(asdict(event), separators=(',', ':'))+'\n')
                        stream.flush()
                tokens, measured, final = checked('question_generation', lambda:measure_question(
                    policy, result, decode_step=lambda t,s:packed(t,s,weights,rope),
                    replicate_uniform=put, fleet_all=fleet_all, deliver=deliver, request_started=started))
            def agreement():
                hashes = np.asarray(multihost_utils.process_allgather(
                    np.frombuffer(bytes.fromhex(measured['token_sha256']), np.uint8)))
                require((hashes == hashes[0]).all(), 'ordinary suite token disagreement')
                return True
            measured['all_host_token_agreement'] = checked('question_token_agreement', agreement)
            def cache_check():
                require(all(np.isfinite(np.asarray(s.data)).all()
                    for a in (final.kv_cache_local, final.index_cache_local)
                    for s in a.addressable_shards), 'ordinary suite nonfinite cache')
                return True
            measured['final_cache_finite'] = checked('question_cache_check', cache_check)
            measured['memory_after'] = stats()
            entry['question'] = measured
            outputs[mode] = tokens.copy()
            np.savez(root/f'{tag}.generated.rank{rank}.npz', tokens=tokens)
            case['execution_order'].append(mode)
            save()
            del fresh, result, final, tokens
            gc.collect()
        for mode in order:
            entry = case['modes'][mode]
            entry['ordinary_agreement'] = output_agreement(outputs[mode], outputs['ordinary'])
            base_rate = case['modes']['ordinary']['question']['tokens_per_second']
            rate = entry['question']['tokens_per_second']
            entry['paired_wall_speedup'] = rate/base_rate if base_rate and rate else None
        save()
        del outputs, staged
    report['complete'] = True
    save()


def summarize_ordinary_suite(rows, controller, cases, case_digests, *, root):
    """Reuse the strict ordinary request validator for every host/case/mode."""
    compare = controller.get('ordinary_suite_compare_prefill')
    expected = [case[0] for case in cases]
    if type(compare) is not bool or len(rows) != 8:
        raise ValueError('ordinary suite needs eight hosts and an explicit comparison mode')
    for row in rows:
        report = row.get('ordinary_suite', {})
        if (report.get('schema') != 'glm_ordinary_suite_v1' or report.get('complete') is not True
                or report.get('suite_sha256') != controller.get('ordinary_suite_sha256')
                or report.get('compare_prefill') is not compare or set(report.get('cases', {})) != set(expected)
                or report.get('same_decode_program') is not True or report.get('fresh_state_per_mode') is not True
                or report.get('profiling')!='no per-component blocking timers'
                or report.get('memory_scope')!='allocator process high-water mark through each completed mode'
                or row.get('ordinary_suite_sha256')!=controller.get('ordinary_suite_sha256')
                or row.get('ordinary_suite_compare_prefill') is not compare
                or row.get('ordinary_prefill_admission_sha256')!=controller.get('ordinary_prefill_admission_sha256')):
            raise ValueError('ordinary suite completion, input or execution scope differs')
    admission_pin=controller.get('ordinary_prefill_admission_sha256')
    if compare != (admission_pin is not None):
        raise ValueError('ordinary suite prefill comparison lacks trained admission')
    if compare:
        raw=(root/'ordinary_prefill_admission.json').read_bytes()
        admission=json.loads(raw)
        if (sha256(raw).hexdigest()!=admission_pin or admission.get('schema')!='glm_perf_real_globalmax_prefill_v1'
                or admission.get('all_hosts_idle_after') is not True or admission.get('fleet_summary_passed') is not True
                or admission['summary'].get('db610_token_check_passed') is not True
                or admission['summary'].get('prefill_global_max_attention') is not True
                or any(any(row['checkpoint'][k]!=admission['summary']['checkpoint'][k]
                    for k in ('manifest_sha256','success_sha256','inventory_sha256')) for row in rows)):
            raise ValueError('ordinary suite trained prefill admission differs')
    result = dict(schema='glm_ordinary_suite_summary_v1', cases={}, compare_prefill=compare,
        serving_admitted=False, answer_correctness='requires separate answer assessment',
        memory_scope=rows[0]['ordinary_suite']['memory_scope'])
    for label, ids, policy, _ in cases:
        order = ordinary_order(label, compare)
        expected_identity = dict(file_sha256=case_digests[label],
            prompt_ids_sha256=sha256(ids.tobytes()).hexdigest(), policy=json.loads(json.dumps(asdict(policy))))
        output = {}
        actual = {}
        for row in rows:
            case = row['ordinary_suite']['cases'][label]
            if (case['identity'] != expected_identity or case['planned_order'] != list(order)
                    or case['execution_order'] != list(order) or set(case['modes']) != set(order)):
                raise ValueError('ordinary suite policy, order or coverage differs')
            for mode in order:
                if any(row['phases'].get(f'suite_{label}_{mode}_warm_{step}',{}).get('passed') is not True
                       for step in ('prefill','decode','health')):
                    raise ValueError('ordinary suite missing completed warmup')
        for mode in order:
            with np.load(root/f'suite_{label}_{mode}.generated.rank0.npz',allow_pickle=False) as saved:
                tokens=saved['tokens']
            if tokens.dtype!=np.int32 or tokens.ndim!=1:
                raise ValueError('ordinary suite private tokens have invalid geometry')
            actual[mode]=tokens
            requests = []
            for row in rows:
                entry = row['ordinary_suite']['cases'][label]['modes'][mode]
                if entry['question_identity'] != expected_identity:
                    raise ValueError('ordinary suite request identity differs')
                phases = dict(entry['phases'])
                # These are the actual common compiled program's admitted phases,
                # reused for each request, not invented per-case compilations.
                for prefix in ('compile_', 'graph_consensus_', 'hlo_', 'memory_'):
                    phases[prefix+'question_packed'] = row['phases'][prefix+'ordinary_suite_packed']
                requests.append(dict(question=entry['question'], question_identity=entry['question_identity'],
                    question_prefill=entry['question_prefill'], phases=phases, decode=row['decode']))
            output[mode] = summarize_question_rows(requests, dict(question_sha256=case_digests[label]))
            agreements = [row['ordinary_suite']['cases'][label]['modes'][mode]['ordinary_agreement'] for row in rows]
            if any(a != agreements[0] for a in agreements):
                raise ValueError('ordinary suite output agreement differs across hosts')
            if (len(tokens)!=output[mode]['emitted'] or sha256(tokens.tobytes()).hexdigest()!=output[mode]['token_sha256']):
                raise ValueError('ordinary suite private tokens differ from authenticated reports')
            output[mode]['ordinary_agreement'] = agreements[0]
        base = output['ordinary']
        base_rate = base['decode_steps']/base['timings']['wall_seconds']['max'] if base['decode_steps'] else None
        for mode in order:
            value = output[mode]
            if value['ordinary_agreement']!=output_agreement(actual[mode],actual['ordinary']):
                raise ValueError('ordinary suite claimed token agreement differs from private output')
            value['paired_wall_speedup'] = (value['decode_steps']/value['timings']['wall_seconds']['max'])/base_rate if base_rate and value['decode_steps'] else None
        result['cases'][label] = dict(execution_order=list(order), modes=output)
    return result
