"""Teacher-forced trained-target verifier diagnostic, not speculative serving.

Reference tokens supply perfect proposals. Numerical comparisons are deliberately
outside the verification timer. No draft quality, accepted throughput or general
model-quality claim follows from this diagnostic.
"""
from hashlib import sha256
import time

import numpy as np


def compare_reference_trail(expected, initial_state, *, rows, verify, commit,
                            ordinary, replicate, ready, healthy, compare):
    """Consume all reference inputs, checking every predicted successor.

    Callables close over already-admitted weights and executables. ``expected``
    includes the pending prefill output; the last element is only an expected
    successor. The short final block is padded with zero IDs, then commits only
    its live prefix. Raw predictions are returned for private storage only.
    ``healthy`` must perform the caller's all-host health/refusal check.
    """
    if (type(rows) is not int or rows not in (2, 3, 5)
            or not isinstance(expected, np.ndarray)
            or expected.ndim != 1 or expected.dtype != np.int32
            or expected.size < 2 or np.any(expected < 0)):
        raise ValueError('diagnostic requires nonnegative int32 reference IDs and 2/3/5 rows')
    candidate = reference = initial_state
    observed, reference_observed, blocks = [], [], []
    for start in range(0, expected.size - 1, rows):
        count = min(rows, expected.size - 1 - start)
        values = np.zeros(rows, np.int32)
        values[:count] = expected[start:start + count]
        inputs = replicate(values)
        accepted_count = replicate(np.int32(count))
        begin = time.perf_counter()
        proposal = ready(verify(inputs, candidate))
        verify_seconds = time.perf_counter() - begin
        healthy(proposal.contract_valid, f'verifier_{start}')
        begin = time.perf_counter()
        candidate = ready(commit(candidate, proposal, accepted_count))
        commit_seconds = time.perf_counter() - begin
        healthy(candidate.contract_valid, f'commit_{start}')
        predictions = np.asarray(proposal.predictions)
        if predictions.shape != (rows,) or predictions.dtype != np.int32:
            raise ValueError('verifier prediction geometry differs')
        predicted = predictions[:count]
        observed.extend(predicted.tolist())
        reference_seconds = 0.
        for i in range(count):
            token = replicate(expected[start+i:start+i+1])
            begin = time.perf_counter()
            result = ready(ordinary(token, reference))
            reference_seconds += time.perf_counter() - begin
            reference = result.state
            healthy(reference.contract_valid, f'ordinary_{start+i}')
            reference_observed.append(int(np.asarray(result.next_token)[0]))
        # Cache/frontier comparison happens only after the complete trail;
        # per-block predictions identify the first divergence without copying
        # full caches between every timed call.
        correct = expected[start+1:start+count+1]
        blocks.append(dict(input_offset=start,live_rows=count,padded_rows=rows-count,
            predictions_equal=bool(np.array_equal(predicted, correct)),
            compared=count,matches=int(np.count_nonzero(predicted == correct)),
            verify_seconds=verify_seconds,commit_seconds=commit_seconds,
            ordinary_seconds=reference_seconds))
    actual = np.asarray(observed, np.int32)
    baseline = np.asarray(reference_observed, np.int32)
    correct = expected[1:]
    mismatch = np.flatnonzero(actual != correct)
    reference_mismatch = np.flatnonzero(baseline != correct)
    comparisons = {name:compare(getattr(candidate,name),getattr(reference,name))
                   for name in candidate._fields}
    report = dict(schema='glm_perf_teacher_forced_verifier_v1',target_rows=rows,
        compared=int(correct.size),matches=int(np.count_nonzero(actual == correct)),
        target_predictions_equal=not bool(mismatch.size),
        first_mismatch_index=None if not mismatch.size else int(mismatch[0])+1,
        ordinary_predictions_equal=not bool(reference_mismatch.size),
        ordinary_observed_sha256=sha256(baseline.tobytes()).hexdigest(),
        observed_sha256=sha256(actual.tobytes()).hexdigest(),
        full_reference_sha256=sha256(expected.tobytes()).hexdigest(),
        reference_sha256=sha256(correct.tobytes()).hexdigest(),blocks=blocks,
        final_state_comparisons=comparisons,
        measured_speculative_throughput=False,draft_cost_included=False,
        timing_scope='diagnostic calls only; comparisons and health votes excluded; no warm timing qualification',
        teacher_forced=True,real_drafter=False)
    return actual, report, candidate, reference


def summarize_reference_trails(ranks):
    """Validate the diagnostic part of an already authenticated DB610 fleet."""
    import math
    import re

    fields = ('kv_cache_local', 'index_cache_local', 'selected_positions',
        'selected_valid_counts', 'selected_scores', 'position', 'block_tables',
        'context_lengths', 'contract_valid')
    comparison_keys = ('bitwise_equal', 'finite', 'differing_elements',
        'local_replica_elements', 'max_abs', 'relative_l2')
    block_keys = ('input_offset', 'live_rows', 'padded_rows', 'predictions_equal',
        'compared', 'matches', 'verify_seconds', 'commit_seconds', 'ordinary_seconds')
    report_keys = ('target_rows', 'compared', 'matches', 'target_predictions_equal',
        'first_mismatch_index', 'ordinary_predictions_equal', 'observed_sha256',
        'ordinary_observed_sha256', 'reference_sha256', 'full_reference_sha256')
    graph_names = {f'speculative_{kind}_{n}' for kind in ('verify', 'commit') for n in (2,3)}
    if len(ranks) != 8 or {r.get('rank') for r in ranks} != set(range(8)):
        raise ValueError('verifier diagnostic requires eight distinct ranks')
    cache_scope = ranks[0].get('speculative_cache_comparison_scope','whole_cache_v1')
    if (cache_scope not in ('whole_cache_v1','whole_and_written_span_v1')
            or any(r.get('speculative_cache_comparison_scope','whole_cache_v1') != cache_scope for r in ranks)):
        raise ValueError('cache comparison scope differs across hosts')
    written_keys = (*comparison_keys,'logical_start','logical_stop','local_replica_rows',
                    'global_unique_elements','expected_feature_replicas')
    legacy_options = dict(canonical_mlp=True,batched_attention=True)
    options = ranks[0].get('speculative_verifier_options',legacy_options)
    if (not isinstance(options,dict)
            or set(options) not in (set(legacy_options),set(legacy_options)|{'small_expert_tiles','rowwise_dsa'})
            or any(type(v) is not bool for v in options.values())
            or any(options[k] is not True for k in legacy_options)
            or any(r.get('speculative_verifier_options',legacy_options)!=options for r in ranks)):
        raise ValueError('verifier options differ across hosts or from supported diagnostic geometry')
    variants = {}
    for r in ranks:
        if (r.get('diagnose_speculative_verifier') is not True
                or r.get('decode_lse_attention') is not False
                or r.get('token_comparison', {}).get('all_equal') is not True
                or set(r.get('speculative_verifier', {})) != {'2', '3'}
                or set(r.get('speculative_programs', {})) != graph_names):
            raise ValueError('missing verifier diagnostic or matching ordinary baseline')
        required = {'verifier_reference_admission'}
        for name in graph_names:
            p = r['speculative_programs'][name]
            if (p.get('hlo_admission', {}).get('passed') is not True
                    or p.get('memory_admission', {}).get('passed') is not True):
                raise ValueError('verifier graph or memory admission missing')
            for key in ('stablehlo_sha256', 'optimized_hlo_sha256'):
                if (re.fullmatch('[0-9a-f]{64}', str(p.get(key))) is None
                        or p[key] != ranks[0]['speculative_programs'][name][key]):
                    raise ValueError('verifier graph identity differs')
            required.update(prefix+name for prefix in ('compile_', 'graph_consensus_', 'hlo_', 'memory_'))
        for n in (2,3):
            d = r['speculative_verifier'][str(n)]
            label = f'speculative_verify_{n}'
            required.update(label+suffix for suffix in ('_warm_first', '_warm', '_trail'))
            if (d.get('schema') != 'glm_perf_teacher_forced_verifier_v1'
                    or d.get('target_rows') != n or d.get('compared') != 28
                    or d.get('teacher_forced') is not True or d.get('real_drafter') is not False
                    or d.get('measured_speculative_throughput') is not False
                    or d.get('draft_cost_included') is not False or d.get('warmup_pairs') != 5
                    or d.get('verifier_options') != options
                    or d.get('full_reference_sha256') != r['input_identity']['reference_sha256']
                    or set(d.get('final_state_comparisons', {})) != set(fields)):
                raise ValueError('verifier diagnostic scope differs')
            for key in ('observed_sha256','ordinary_observed_sha256','reference_sha256','full_reference_sha256'):
                if re.fullmatch('[0-9a-f]{64}', str(d.get(key))) is None:
                    raise ValueError('missing verifier token identity')
            if (type(d.get('matches')) is not int or not 0 <= d['matches'] <= 28
                    or type(d.get('target_predictions_equal')) is not bool
                    or d['target_predictions_equal'] != (d['matches'] == 28)
                    or d['target_predictions_equal'] != (d['observed_sha256'] == d['reference_sha256'])
                    or type(d.get('ordinary_predictions_equal')) is not bool
                    or d['ordinary_predictions_equal'] != (d['ordinary_observed_sha256'] == d['reference_sha256'])):
                raise ValueError('inconsistent verifier prediction comparison')
            mismatch = d.get('first_mismatch_index')
            if ((mismatch is None) != d['target_predictions_equal']
                    or (mismatch is not None and (type(mismatch) is not int or not 1 <= mismatch <= 28))):
                raise ValueError('invalid first verifier mismatch')
            starts = list(range(0,28,n))
            blocks = d.get('blocks', [])
            if [b.get('input_offset') for b in blocks] != starts:
                raise ValueError('incomplete verifier blocks')
            for start,b in zip(starts,blocks):
                live = min(n,28-start)
                if (b.get('live_rows') != live or b.get('compared') != live
                        or b.get('padded_rows') != n-live or type(b.get('matches')) is not int
                        or not 0 <= b['matches'] <= live
                        or type(b.get('predictions_equal')) is not bool
                        or b['predictions_equal'] != (b['matches'] == live)
                        or any(type(b.get(k)) not in (int,float) or not math.isfinite(b[k]) or b[k] <= 0
                               for k in ('verify_seconds','commit_seconds','ordinary_seconds'))):
                    raise ValueError('invalid verifier block count or timing')
                required.update(f'{label}_{kind}_{start}' for kind in ('verifier','commit'))
            required.update(f'{label}_ordinary_{i}' for i in range(28))
            if sum(b['matches'] for b in blocks) != d['matches']:
                raise ValueError('verifier match total differs')
            if mismatch is not None:
                first = next(b for b in blocks if not b['predictions_equal'])
                if not first['input_offset'] < mismatch <= first['input_offset']+first['live_rows']:
                    raise ValueError('first mismatch is outside first failing block')
            for name in fields:
                c = d['final_state_comparisons'][name]
                if (any(type(c.get(k)) is not bool for k in ('bitwise_equal','finite'))
                        or any(type(c.get(k)) is not int for k in ('differing_elements','local_replica_elements'))
                        or not 0 <= c['differing_elements'] <= c['local_replica_elements']
                        or c['local_replica_elements'] <= 0
                        or (c['bitwise_equal'] and c['differing_elements'] != 0)
                        or any(type(c.get(k)) not in (int,float) or not math.isfinite(c[k]) or c[k] < 0
                               for k in ('max_abs','relative_l2'))):
                    raise ValueError('invalid verifier state comparison')
            if cache_scope == 'whole_and_written_span_v1':
                written = d.get('written_cache_comparisons',{})
                if set(written) != {'kv_cache_local','index_cache_local'}:
                    raise ValueError('missing written-cache comparisons')
                required.add(label+'_written_cache_comparison')
                for c in written.values():
                    if (set(c) != set(written_keys)
                            or any(type(c[k]) is not int for k in written_keys if k not in comparison_keys)
                            or c['logical_start'] != 2034 or c['logical_stop'] != 2062
                            or c['expected_feature_replicas'] != 4
                            or not 0 <= c['local_replica_rows'] <= 112
                            or c['global_unique_elements'] <= 0 or c['global_unique_elements'] % 28
                            or type(c['local_replica_elements']) is not int
                            or c['local_replica_elements'] != c['local_replica_rows']*(c['global_unique_elements']//28)
                            or type(c['differing_elements']) is not int
                            or not 0 <= c['differing_elements'] <= c['local_replica_elements']
                            or any(type(c[k]) is not bool for k in ('bitwise_equal','finite'))
                            or (c['bitwise_equal'] and c['differing_elements'] != 0)
                            or any(type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k] < 0
                                   for k in ('max_abs','relative_l2'))
                            or (c['local_replica_rows'] == 0 and
                                (not c['bitwise_equal'] or not c['finite'] or c['max_abs'] != 0 or c['relative_l2'] != 0))):
                        raise ValueError('invalid written-cache comparison')
            elif 'written_cache_comparisons' in d:
                raise ValueError('written-cache comparison requires explicit scope')
            memory = d.get('memory_after', [])
            if (len(memory) != 4 or len({m.get('device_id') for m in memory}) != 4
                    or any(not 0 <= m['bytes_in_use'] <= m['peak_bytes_in_use'] <= m['bytes_limit'] for m in memory)):
                raise ValueError('missing measured verifier memory')
        if any(r.get('phases', {}).get(name, {}).get('passed') is not True for name in required):
            raise ValueError('missing or failed verifier phase')
    for n in (2,3):
        ds = [r['speculative_verifier'][str(n)] for r in ranks]
        chips = [m['device_id'] for d in ds for m in d['memory_after']]
        if len(chips) != 32 or set(chips) != set(range(32)):
            raise ValueError('verifier memory does not cover all 32 chips')
        if len({d['reference_sha256'] for d in ds}) != 1:
            raise ValueError('verifier continuation reference differs across hosts')
        if cache_scope == 'whole_and_written_span_v1':
            for name in ('kv_cache_local','index_cache_local'):
                written = [d['written_cache_comparisons'][name] for d in ds]
                if (len({c['global_unique_elements'] for c in written}) != 1
                        or sum(c['local_replica_rows'] for c in written) != 112
                        or sum(c['local_replica_elements'] for c in written) != 4*written[0]['global_unique_elements']):
                    raise ValueError('written-cache fleet coverage differs')
        variants[str(n)] = dict(
            target_predictions_equal=all(d['target_predictions_equal'] for d in ds),
            ordinary_predictions_equal=all(d['ordinary_predictions_equal'] for d in ds),
            fleet_predictions_agree=len({d['observed_sha256'] for d in ds}) == 1,
            maximum_peak_hbm_bytes=max(m['peak_bytes_in_use'] for d in ds for m in d['memory_after']),
            ranks=[dict(rank=r['rank'],**{k:d[k] for k in report_keys},
                blocks=[{k:b[k] for k in block_keys} for b in d['blocks']],
                final_state_comparisons={name:{k:d['final_state_comparisons'][name][k]
                    for k in comparison_keys} for name in fields},
                **({'written_cache_comparisons':{name:{k:c[k] for k in written_keys}
                    for name,c in d['written_cache_comparisons'].items()}}
                   if cache_scope == 'whole_and_written_span_v1' else {})) for r,d in zip(ranks,ds)])
    return dict(variants=variants,verifier_options=dict(options),teacher_forced=True,measured_speculative_throughput=False,
        cache_comparison_scope=cache_scope,
        timing_scope='diagnostic model calls; health votes, comparisons and drafting excluded',
        serving_admitted=False,programs={name:{k:ranks[0]['speculative_programs'][name][k]
            for k in ('stablehlo_sha256','optimized_hlo_sha256')} for name in sorted(graph_names)})
