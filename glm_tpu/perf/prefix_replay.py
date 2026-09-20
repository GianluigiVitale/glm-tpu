"""Private short-input replay for diagnosing trained verifier disagreement."""
from hashlib import sha256
import json

import numpy as np

from ..user_request import read_bounded
from .long_question import load_question


def load_prefix_replay(root, digest, *, capacity, vocab_size, eos_ids):
    raw = read_bounded(root/'prefix_replay.json', 1 << 20)
    if sha256(raw).hexdigest() != digest:
        raise ValueError('prefix replay digest differs')
    value = json.loads(raw)
    if (type(value) is not dict or set(value) != {'schema', 'cases'}
            or value['schema'] != 'glm_perf_prefix_replay_inputs_v1'
            or type(value['cases']) is not list or not 1 <= len(value['cases']) <= 3):
        raise ValueError('prefix replay schema differs')
    cases = []
    for case in value['cases']:
        if (type(case) is not dict or set(case) != {'name', 'question_sha256',
                'reference_ids', 'reference_sha256', 'source_execution', 'offsets'}
                or case['name'] not in ('prose', 'code', 'structured')
                or case['name'] in {x[0] for x in cases}
                or type(case['reference_ids']) is not list
                or not 5 <= len(case['reference_ids']) <= 32
                or any(type(t) is not int or not 0 <= t < vocab_size for t in case['reference_ids'])
                or type(case['offsets']) is not list or not case['offsets']
                or any(type(i) is not int or i < 0 or i+3 >= len(case['reference_ids'])
                       for i in case['offsets'])
                or sorted(set(case['offsets'])) != case['offsets']
                or type(case['source_execution']) is not str
                or len(case['source_execution']) != 40
                or any(c not in '0123456789abcdef' for c in case['source_execution'])):
            raise ValueError('prefix replay case identity/geometry differs')
        ids, policy = load_question(root/(case['name']+'.question.json'),
            case['question_sha256'], capacity=capacity, vocab_size=vocab_size, eos_ids=eos_ids)
        expected = np.asarray(case['reference_ids'], np.int32)
        if (sha256(expected.tobytes()).hexdigest() != case['reference_sha256']
                or len(ids)+len(expected) > capacity):
            raise ValueError('prefix replay reference identity/capacity differs')
        cases.append((case['name'], ids, expected, tuple(case['offsets']),
                      {k:v for k,v in case.items() if k != 'reference_ids'}))
    return cases


def run_prefix_replay(*, cases, mesh, config, weights, rope, wk, prefill, decode,
                     compile_model, phase, require, put, record, save, stats,
                     trace_layers=False, ordinary_options=None):
    """Caller holds fleet leases and enforces source/HLO/memory admission."""
    import gc
    import jax
    import jax.numpy as jnp
    from ..greenfield.runtime import ws32_batched_prefill as pre
    from .long_question import question_blocks
    from .speculative_diagnostics import compare_same_prefix
    from .speculative_verify import build_verifier, build_prefix_committer
    from tools.perf_speculative_verify import local_comparison, local_cache_span_comparison

    def compare_committed(candidate, reference, root, count):
        if not count:
            return {}
        start = int(np.asarray(root.position)[0])
        return dict(written_cache={name: local_cache_span_comparison(
            getattr(candidate, name), getattr(reference, name), root.block_tables,
            start, start+count) for name in ('kv_cache_local', 'index_cache_local')})

    def compare(a, b):
        result = local_comparison(a, b)
        # Cache arrays are [layer/slot, physical page, logical row, width].
        # Retain per-layer evidence without publishing private tensor values.
        if a.ndim == 4:
            pairs = [(np.asarray(x.data), np.asarray(y.data))
                     for x, y in zip(a.addressable_shards, b.addressable_shards, strict=True)]
            result['differing_layers_or_slots'] = [i for i in range(a.shape[0])
                if any(not np.array_equal(np.ascontiguousarray(x[i]).view(np.uint8),
                                          np.ascontiguousarray(y[i]).view(np.uint8))
                       for x, y in pairs)]
        return result

    record['prefix_replay'] = dict(schema='glm_perf_prefix_replay_rank_v1',
        measured_speculative_throughput=False, cases={},
        trace_layers=trace_layers,
        full_index_slot_by_layer=list(config.full_index_slot_by_layer),
        limits=['Cache divergence locates affected layers, not the first differing arithmetic operation.',
                'No independent trained-native drafter parity or answer-quality claim.'])
    # Keep the same jitted function objects across prompts. Shapes/configuration
    # are identical; tokens, caches and weights remain dynamic arguments. Each
    # case still calls compile_model for graph consensus and live admission.
    # Recreating/clearing these objects forced a second compilation of
    # byte-identical verifier graphs in the first trained replay.
    verify_programs = {n:build_verifier(mesh, config, canonical_mlp=True,
        batched_attention=True, small_expert_tiles=True, rowwise_dsa=True) for n in (1,2,3)}
    commit_program = build_prefix_committer(mesh, config)
    trace_programs = {}
    trace_ordinary_program = None
    if trace_layers:
        if ordinary_options is None:
            raise ValueError('traced replay requires explicit ordinary options')
        from .verifier_trace import build_target_trace, compare_trace_window
        trace_ordinary_program = build_target_trace(mesh, config, ordinary_options=ordinary_options)
        trace_programs = {n:build_target_trace(mesh, config) for n in (1,2,3)}
    for name, ids, expected, offsets, identity in cases:
        current = pre.make_ws32_batched_prefill_state(mesh, config, prompt_length=len(ids))
        for i, (block, count) in enumerate(question_blocks(ids)):
            result = phase(f'replay_{name}_prefill_{i}', lambda: jax.block_until_ready(
                prefill[block.size](put(block), put(np.int32(count)), current, weights, wk, rope)))
            current = result.state
            phase(f'replay_{name}_prefill_health_{i}', lambda: require(
                np.asarray(current.decoder.contract_valid).all(), 'replay prefill unhealthy'))
        initial, token = pre.finish_ws32_batched_prefill(result)
        phase(f'replay_{name}_first_token', lambda: require(
            np.array_equal(np.asarray(token), expected[:1]), 'historical first token differs'))
        case_report = dict(identity=identity, variants={})
        record['prefix_replay']['cases'][name] = case_report
        traced_ordinary = None
        if trace_layers:
            traced_ordinary = compile_model(f'replay_{name}_ordinary_trace', trace_ordinary_program,
                                           (put(expected[:1]), initial, weights, rope))
        for rows in (1, 2, 3):
            label = f'replay_{name}_r{rows}'
            program = verify_programs[rows]
            values = (put(expected[:rows]), initial, weights, rope)
            verifier = compile_model(label+'_verify', program, values)
            shape = jax.eval_shape(program, *values)
            committer = compile_model(label+'_commit', commit_program,
                                      (initial, shape, put(np.int32(rows))))
            trace_program = traced_verifier = None
            if trace_layers:
                trace_program = trace_programs[rows]
                traced_verifier = compile_model(label+'_trace', trace_program, values)

            def health(value, step):
                phase(label+'_'+step, lambda: require(np.asarray(value).all(), 'replay unhealthy'))

            def observe(report):
                case_report['variants'][str(rows)] = report
                save()

            def diagnose(start, tokens, states, results, proposal):
                return compare_trace_window(tokens, states, results, proposal,
                    traced_ordinary=lambda t,s: traced_ordinary(t,s,weights,rope),
                    traced_verifier=lambda t,s: traced_verifier(t,s,weights,rope),
                    ready=jax.block_until_ready, put=put, compare=compare,
                    healthy=lambda value,step: health(value,f'trace_{start}_{step}'))

            report = phase(label+'_comparison', lambda: compare_same_prefix(
                expected, initial, rows=rows, offsets=offsets,
                verify=lambda t,s: verifier(t,s,weights,rope), commit=committer,
                ordinary=lambda t,s: decode(t,s,weights,rope), replicate=put,
                ready=jax.block_until_ready, healthy=health, compare=compare,
                concatenate=jnp.concatenate, observe=observe, compare_committed=compare_committed,
                diagnose_window=diagnose if trace_layers else None))
            report['memory_after'] = stats()
            phase(label+'_historical_reference', lambda: require(
                report['ordinary_reference_equal'], 'ordinary teacher-forced history differs'))
            save()
            del verifier, committer, shape, values, program
            del trace_program, traced_verifier
            gc.collect()
        del traced_ordinary
        del current, result, initial
        gc.collect()
    for program in (*verify_programs.values(), commit_program, *trace_programs.values()):
        program.clear_cache()
    if trace_ordinary_program is not None:
        trace_ordinary_program.clear_cache()
