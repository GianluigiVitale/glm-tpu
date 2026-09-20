"""Strict, payload-free aggregation of short same-prefix replay evidence."""
import math
from hashlib import sha256


FIELDS = ('kv_cache_local', 'index_cache_local', 'selected_positions',
          'selected_valid_counts', 'selected_scores', 'position', 'block_tables',
          'context_lengths', 'contract_valid')


def _comparison(value):
    if (type(value) is not dict
            or any(type(value.get(k)) is not bool for k in ('bitwise_equal', 'finite'))
            or any(type(value.get(k)) is not int or value[k] < 0
                   for k in ('differing_elements', 'local_replica_elements'))
            or value['differing_elements'] > value['local_replica_elements']
            or (value['bitwise_equal'] and value['differing_elements'] != 0)
            or any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k])
                   or value[k] < 0 for k in ('max_abs', 'relative_l2'))):
        raise ValueError('invalid replay numerical comparison')
    return value


def _trace(value, rows):
    """Validate trace shape; instrumentation changes remain explicit negatives."""
    layer_fields = {'normalized_inputs', 'hidden_updates', 'carried_residuals',
                    'selected_positions', 'selected_counts', 'selected_scores'}
    proposal_fields = {'kv_cache_local', 'index_cache_local', 'selected_positions',
        'selected_valid_counts', 'selected_scores', 'predictions', 'final_residual_local',
        'normalized_hidden_local', 'contract_valid'}
    if (type(value) is not dict or value.get('compiler_outputs_changed') is not True
            or [x.get('layer') for x in value.get('layers', [])] != list(range(78))
            or len(value.get('ordinary_instrumentation', [])) != rows
            or any(type(value.get(k)) is not bool for k in
                   ('verifier_head_matches_prediction', 'ordinary_head_matches_prediction'))):
        raise ValueError('missing layer/head trace')
    for layer in value['layers']:
        if set(layer.get('comparisons', {})) != layer_fields:
            raise ValueError('missing trace activation field')
        for comparison in layer['comparisons'].values(): _comparison(comparison)
    _comparison(value['final_normalized'])
    vi = value.get('verifier_instrumentation', {})
    if type(vi.get('predictions_equal')) is not bool or set(vi.get('fields', {})) != proposal_fields:
        raise ValueError('missing verifier instrumentation comparison')
    for comparison in vi['fields'].values(): _comparison(comparison)
    for ordinary in value['ordinary_instrumentation']:
        if (type(ordinary.get('prediction_equal')) is not bool
                or type(ordinary.get('head_matches_prediction')) is not bool
                or set(ordinary.get('state', {})) != set(FIELDS)):
            raise ValueError('missing ordinary instrumentation comparison')
        _comparison(ordinary['residual'])
        for comparison in ordinary['state'].values(): _comparison(comparison)
    if value['ordinary_head_matches_prediction'] != all(
            x['head_matches_prediction'] for x in value['ordinary_instrumentation']):
        raise ValueError('inconsistent ordinary head probe agreement')
    flags = value.get('top_two_ids_equal_by_row')
    if type(flags) is not list or len(flags) != rows or any(type(x) is not bool for x in flags):
        raise ValueError('missing head identity comparisons')
    for name in ('verifier_top_score', 'ordinary_top_score', 'verifier_logit_margin', 'ordinary_logit_margin'):
        values = value.get(name)
        if (type(values) is not list or len(values) != rows
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
                or ('margin' in name and any(x < 0 for x in values))):
            raise ValueError('invalid trace head scores')
    stable = vi['predictions_equal'] and all(x['bitwise_equal'] for x in vi['fields'].values())
    stable &= all(x['prediction_equal'] and x['residual']['bitwise_equal'] and
                  all(c['bitwise_equal'] for c in x['state'].values()) for x in value['ordinary_instrumentation'])
    differing = [x['layer'] for x in value['layers']
                 if any(not c['bitwise_equal'] for c in x['comparisons'].values())]
    return dict(instrumentation_bitwise_stable=bool(stable),
        first_differing_layer=None if not differing else differing[0],
        head_predictions_consistent=value['verifier_head_matches_prediction'] and value['ordinary_head_matches_prediction'],
        verifier_logit_margin=value['verifier_logit_margin'], ordinary_logit_margin=value['ordinary_logit_margin'])


def summarize_prefix_replay(ranks, cases, digest, *, trace_layers=False):
    """Base DB610 summary separately authenticates fleet/source/graph admission.

    This routine requires all replay windows, prefixes, phases and live memory
    records. Differences are valid negative results; missing evidence is refused.
    No arbitrary worker fields or private prompt/reference IDs are published.
    """
    if type(trace_layers) is not bool or len(ranks) != 8 or {r.get('rank') for r in ranks} != set(range(8)):
        raise ValueError('same-prefix summary requires eight distinct ranks')
    names = {c[0] for c in cases}
    if not names:
        raise ValueError('same-prefix summary requires authenticated cases')
    slots = ranks[0].get('prefix_replay', {}).get('full_index_slot_by_layer')
    if (type(slots) is not list or len(slots) != 78
            or any(x is not None and (type(x) is not int or not 0 <= x < 78) for x in slots)):
        raise ValueError('missing GLM cache layer/slot mapping')
    full_slots = [x for x in slots if x is not None]
    if full_slots != list(range(len(full_slots))) or not full_slots:
        raise ValueError('invalid full-index slot mapping')
    output = dict(schema='glm_perf_prefix_replay_fleet_v1', input_sha256=digest,
        measured_speculative_throughput=False, serving_admitted=False, cases={}, trace_layers=trace_layers,
        limits=['Cache layer differences do not isolate the first differing arithmetic operation.',
                ('Traced replays validate original prediction hashes; instrumented drift is reported separately.'
                 if trace_layers else 'Fleet agreement covers reported mismatch patterns, not exported prediction-ID hashes.')])
    for rank in ranks:
        replay = rank.get('prefix_replay', {})
        if (rank.get('prefix_replay_sha256') != digest or rank.get('complete') is not True
                or replay.get('schema') != 'glm_perf_prefix_replay_rank_v1'
                or replay.get('measured_speculative_throughput') is not False
                or replay.get('full_index_slot_by_layer') != slots
                or replay.get('trace_layers', False) is not trace_layers
                or rank.get('prefix_replay_trace', False) is not trace_layers
                or set(replay.get('cases', {})) != names
                or rank.get('phases', {}).get('prefix_replay_reference_admission', {}).get('passed') is not True):
            raise ValueError('missing or differently scoped replay')
    for name, ids, expected, offsets, identity in cases:
        case_out = dict(prompt_tokens=len(ids), reference_tokens=len(expected),
                        reference_sha256=identity['reference_sha256'], variants={})
        output['cases'][name] = case_out
        for rows in (1, 2, 3):
            reports = []
            label = f'replay_{name}_r{rows}'
            for rank in ranks:
                case = rank['prefix_replay']['cases'][name]
                if case.get('identity') != identity or set(case.get('variants', {})) != {'1', '2', '3'}:
                    raise ValueError('replay case identity or variants differ')
                report = case['variants'][str(rows)]
                if (report.get('schema') != 'glm_perf_same_prefix_v1'
                        or report.get('target_rows') != rows
                        or report.get('reference_sha256') != identity['reference_sha256']
                        or report.get('reset_to_ordinary_state') is not True
                        or report.get('ordinary_reference_equal') is not True
                        or report.get('measured_speculative_throughput') is not False
                        or [w.get('input_offset') for w in report.get('windows', [])] != list(offsets)):
                    raise ValueError('incomplete same-prefix windows or reference mismatch')
                required = {f'replay_{name}_first_token', label+'_comparison', label+'_historical_reference'}
                required.update(label+f'_advance_{i}' for i in range(max(offsets)))
                for suffix in ('verify', 'commit'):
                    required.update(prefix+label+'_'+suffix
                                    for prefix in ('compile_', 'graph_consensus_', 'hlo_', 'memory_'))
                if trace_layers:
                    for graph in (f'replay_{name}_ordinary_trace', label+'_trace'):
                        required.update(p+graph for p in ('compile_', 'graph_consensus_', 'hlo_', 'memory_'))
                first_mismatch = None
                for window in report['windows']:
                    start = window['input_offset']
                    diff = window.get('differing_prediction_rows')
                    if (type(diff) is not list or any(type(i) is not int or not 0 <= i < rows for i in diff)
                            or diff != sorted(set(diff))
                            or type(window.get('predictions_equal')) is not bool
                            or window['predictions_equal'] != (not diff)
                            or [p.get('consumed') for p in window.get('prefixes', [])] != list(range(rows+1))):
                        raise ValueError('invalid replay prediction or prefix accounting')
                    if diff and first_mismatch is None:
                        first_mismatch = start + diff[0] + 1
                    _comparison(window['residual_comparison'])
                    if trace_layers:
                        reference_hash=sha256(expected[start+1:start+rows+1].tobytes()).hexdigest()
                        prediction_hash=window.get('prediction_sha256')
                        if (window.get('ordinary_prediction_sha256') != reference_hash
                                or type(prediction_hash) is not str or len(prediction_hash)!=64
                                or any(c not in '0123456789abcdef' for c in prediction_hash)
                                or (prediction_hash==reference_hash) != window['predictions_equal']):
                            raise ValueError('missing or inconsistent replay prediction hashes')
                        _trace(window.get('trace'), rows)
                        required.add(label+f'_trace_{start}_traced_verify')
                        required.update(label+f'_trace_{start}_traced_ordinary_{i}' for i in range(rows))
                    elif 'trace' in window:
                        raise ValueError('unregistered layer trace')
                    required.add(label+f'_verify_{start}')
                    required.update(label+f'_reference_{start}_{i}' for i in range(rows))
                    for prefix in window['prefixes']:
                        count = prefix['consumed']
                        required.add(label+f'_commit_{start}_{count}')
                        comparisons = prefix.get('state_comparisons', {})
                        if set(comparisons) != set(FIELDS):
                            raise ValueError('missing replay state field')
                        for field, value in comparisons.items():
                            _comparison(value)
                            if field in ('kv_cache_local', 'index_cache_local'):
                                layers = value.get('differing_layers_or_slots')
                                bound = len(slots) if field == 'kv_cache_local' else len(full_slots)
                                if (type(layers) is not list or layers != sorted(set(layers))
                                        or any(type(i) is not int or not 0 <= i < bound for i in layers)
                                        or value['bitwise_equal'] != (not layers)):
                                    raise ValueError('invalid differing cache-layer evidence')
                        written = prefix.get('details', {}).get('written_cache', {})
                        if set(written) != (set() if count == 0 else {'kv_cache_local', 'index_cache_local'}):
                            raise ValueError('missing written-span cache evidence')
                        for value in written.values():
                            _comparison(value)
                            if (value.get('logical_start') != len(ids)+start
                                    or value.get('logical_stop') != len(ids)+start+count
                                    or value.get('expected_feature_replicas') != 4):
                                raise ValueError('wrong written cache span')
                if report.get('first_prediction_mismatch') != first_mismatch:
                    raise ValueError('inconsistent first mismatch')
                if any(rank['phases'].get(k, {}).get('passed') is not True for k in required):
                    raise ValueError('missing replay execution phase')
                memory = report.get('memory_after', [])
                if (len(memory) != 4 or len({d.get('device_id') for d in memory}) != 4
                        or {d.get('device_id') for d in memory} !=
                           {d['device_id'] for d in rank['decode']['memory_after']}
                        or any(not 0 <= d.get('bytes_in_use', -1) <= d.get('peak_bytes_in_use', -1)
                               <= d.get('bytes_limit', -1) for d in memory)):
                    raise ValueError('missing replay memory evidence')
                reports.append(report)
            windows = []
            for i, start in enumerate(offsets):
                values = [r['windows'][i] for r in reports]
                prefixes = []
                for count in range(rows+1):
                    states = [w['prefixes'][count]['state_comparisons'] for w in values]
                    prefixes.append(dict(consumed=count,
                        differing_state_fields=[f for f in FIELDS if any(not s[f]['bitwise_equal'] for s in states)],
                        cache_layers=sorted({k for s in states for k in s['kv_cache_local']['differing_layers_or_slots']}),
                        index_slots=sorted({k for s in states for k in s['index_cache_local']['differing_layers_or_slots']})))
                windows.append(dict(input_offset=start,
                    predictions_equal=all(w['predictions_equal'] for w in values),
                    fleet_mismatch_pattern_agrees=all(w['differing_prediction_rows'] == values[0]['differing_prediction_rows'] for w in values),
                    differing_prediction_rows=sorted({k for w in values for k in w['differing_prediction_rows']}),
                    residual_max_abs=max(w['residual_comparison']['max_abs'] for w in values),
                    all_finite=all(w['residual_comparison']['finite'] and
                        all(v['finite'] for p in w['prefixes'] for v in p['state_comparisons'].values()) for w in values),
                    prefixes=prefixes))
                if trace_layers:
                    windows[-1]['fleet_prediction_hashes_agree'] = len({w['prediction_sha256'] for w in values})==1
                    windows[-1]['trace_by_rank'] = [_trace(w['trace'], rows) for w in values]
            case_out['variants'][str(rows)] = dict(windows=windows,
                first_prediction_mismatch_by_rank=[r['first_prediction_mismatch'] for r in reports],
                peak_bytes_per_chip=max(d['peak_bytes_in_use'] for r in reports for d in r['memory_after']))
    return output
