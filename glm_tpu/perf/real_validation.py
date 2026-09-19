"""Research-only DB610 input authentication and fresh graph admission.

This does not inherit frozen HLO admission or claim universal numerical parity.
Private prompt/reference arrays are returned to the caller, never serialized in
public receipts. The sealed DB610 ledger authenticates the original runner.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def build_db610_decoder(mesh, config, *, lse_attention=True, dsa_two_stage=True, write_empty_slot=True):
    """DB610 compares greedy IDs; the general challenger defaults to sampling."""
    from .ws32_decoder_challenger import Ws32PerfOptions, build_ws32_challenger_decoder_program
    from .fp8_routed_experts import RoutedProjectionConfig

    return build_ws32_challenger_decoder_program(mesh, config, options=Ws32PerfOptions(
        sampler='greedy', bf16_resident=True, lse_attention=lse_attention, dsa_two_stage=dsa_two_stage,
        routed_projection=RoutedProjectionConfig(output_tile=256, contraction_tile=256,write_empty_slot=write_empty_slot)))


def db610_inputs(repo: Path, original_root: Path):
    from ..greenfield.validation.ws32_short_context import load_ws32_short_context_oracle

    seal = json.loads((repo / 'docs/artifacts/prefill-canonical-short-db610-sealed-20260909.json').read_bytes())
    if original_root.name != seal['run_tag'] or seal['results_db_run_id'] != 610:
        raise ValueError('original DB610 identity differs')
    ledger_bytes = (original_root / 'remote_objects.json').read_bytes()
    ledger_pin = next(r for r in seal['remote_readbacks'] if r['name'] == 'remote_objects.json')
    if len(ledger_bytes) != ledger_pin['size'] or sha256(ledger_bytes).hexdigest() != ledger_pin['sha256']:
        raise ValueError('DB610 original ledger hash differs')
    ledger = json.loads(ledger_bytes)
    runner_pin = next(r for r in ledger['objects'] if r['name'] == 'host_records/runner.rank0.json')
    runner_bytes = (original_root / 'runner.rank0.json').read_bytes()
    if len(runner_bytes) != runner_pin['size'] or sha256(runner_bytes).hexdigest() != runner_pin['sha256']:
        raise ValueError('DB610 original runner hash differs')
    runner = json.loads(runner_bytes)
    base = Path('/home/gianl/gcs-models/oracles/greenfield/glm52')
    oracle = load_ws32_short_context_oracle(
        base/'short_context/2k/greenfield_short_context_oracle_20260806T202544155912103Z/oracle',
        base/'short_context_dsa/2k/greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z/oracle',
        **{'expected_'+k.replace('_oracle',''): runner[k] for k in ('token_oracle_manifest_sha256',
             'dsa_oracle_manifest_sha256','token_oracle_success_sha256','dsa_oracle_success_sha256')})
    prompt = np.asarray(oracle.prompt_token_ids, np.int32)
    expected = np.asarray(runner['observed_generated_token_ids'], np.int32)
    if prompt.shape != (2034,) or expected.shape != (29,) or runner['correctness_passed'] is not True:
        raise ValueError('DB610 prompt/reference geometry or original verdict differs')
    if not np.array_equal(expected[:len(oracle.generated_token_ids)], oracle.generated_token_ids):
        raise ValueError('DB610 reference does not match its sealed legacy prefix')
    identity = dict(db_run_id=610, run_tag=seal['run_tag'], runner_sha256=runner_pin['sha256'],
        ledger_sha256=ledger_pin['sha256'], prompt_tokens=2034, reference_tokens=29,
        prompt_sha256=sha256(prompt.tobytes()).hexdigest(),
        reference_sha256=sha256(expected.tobytes()).hexdigest(),
        token_oracle_manifest_sha256=runner['token_oracle_manifest_sha256'])
    return prompt, expected, identity


def inspect_research_hlo(text: str) -> dict:
    """Fresh WS32 graph check: physical axis groups and bounded exchanges.

    This is a deliberately scoped structural check, not the frozen graph's
    exact opcode/identity proof. CPU/TPU numerical evidence remains separate.
    Both operands and results are checked, including async collective forms
    normalized by the retained parser. Full-pod consensus is limited to4KiB.
    """
    from ..greenfield.sharding.hlo_contract import parse_hlo_module

    module = parse_hlo_module(text)
    if module.num_partitions != 32 or module.num_replicas not in (None, 1):
        raise ValueError('research model graph requires exactly32 partitions/one replica')
    widths = {'pred':1,'s8':1,'u8':1,'bf16':2,'f16':2,'s16':2,'u16':2,
              'f32':4,'s32':4,'u32':4,'f64':8,'s64':8,'u64':8}
    allowed = {
        frozenset(frozenset(range(r*4,r*4+4)) for r in range(8)),
        frozenset(frozenset(range(c,32,4)) for c in range(4)),
        frozenset((frozenset(range(32)),)),
    }
    maximum = 0
    for op in module.collectives:
        if op.opcode not in ('all-reduce','all-gather','reduce-scatter'):
            raise ValueError('unreviewed collective kind in research graph: '+op.opcode)
        groups = frozenset(frozenset(g) for g in op.replica_groups)
        if (not op.use_global_device_ids or groups not in allowed
                or sum(map(len, op.replica_groups)) != 32):
            raise ValueError('research collective does not follow physical expert8/feature4 axes')
        sizes = []
        for shape in (*op.result_shapes,*op.operand_shapes):
            if shape.dtype not in widths:
                raise ValueError('unknown collective dtype')
            sizes.append(shape.element_count * widths[shape.dtype])
        payload = max(sizes, default=0)
        maximum = max(maximum,payload)
        if not sizes or payload > 128*1024**2 or (op.maximum_group_size == 32 and payload > 4096):
            raise ValueError('oversized or unparsed collective in research graph')
    if not module.collectives:
        raise ValueError('model graph has no parsed collectives')
    return dict(passed=True, profile='research_ws32_axis_payload_v1',
        num_partitions=32, instructions=len(module.instructions),
        collectives=dict(Counter(op.opcode for op in module.collectives)),
        maximum_collective_payload_bytes=maximum, frozen_graph_admission_inherited=False)


def memory_projection(stats, memory, *, reserve_bytes=512*1024**2):
    fields = ('output_size_in_bytes','temp_size_in_bytes','generated_code_size_in_bytes','alias_size_in_bytes')
    if any(type(memory.get(k)) is not int or memory[k] < 0 for k in fields) or reserve_bytes < 0:
        raise ValueError('invalid compiler memory accounting')
    extra = max(0, memory[fields[0]]+memory[fields[1]]+memory[fields[2]]-memory[fields[3]])
    if len(stats) != 4 or len({s['device_id'] for s in stats}) != 4:
        raise ValueError('memory admission needs four distinct local chips')
    rows = []
    for s in stats:
        if not 0 <= s['bytes_in_use'] <= s['bytes_limit']:
            raise ValueError('invalid live allocator accounting')
        predicted = s['bytes_in_use'] + extra + reserve_bytes
        rows.append(dict(device_id=s['device_id'], predicted_bytes=predicted,
                         limit=s['bytes_limit'], fits=predicted < s['bytes_limit']))
    return dict(passed=all(r['fits'] for r in rows), reserve_bytes=reserve_bytes, chips=rows)


def summarize_real_validation(root: Path) -> dict:
    """All-eight-rank completion evidence; a token mismatch stays a failed check.

    Only named, non-payload fields are published. No prompt, generated IDs,
    exception text or arbitrary worker dictionary is copied into the summary.
    """
    rows=[json.loads((root/f'validation.rank{i}.json').read_bytes()) for i in range(8)]
    controller=json.loads((root/'controller_identity.json').read_bytes())
    programs=('wk_decode','wk_promote','prefill_128','prefill_114','decode')
    model_programs=programs[2:]
    if {r.get('rank') for r in rows} != set(range(8)) or {r.get('jax_process_index') for r in rows} != set(range(8)):
        raise ValueError('all eight distinct launch and JAX ranks are required')
    slots=[]
    measured_devices=[]
    for rank,r in enumerate(rows):
        if (r.get('schema')!='glm_perf_real_validation_rank_v1' or r.get('rank')!=rank
                or not r.get('hostname','').endswith('-w-'+str(rank)) or r.get('complete') is not True
                or r.get('devices')!=32 or r.get('capacity')!=8192 or 'failure' in r):
            raise ValueError('incomplete or differently scoped real-weight acquisition')
        for key in ('code_hash','source_manifest_sha256','input_sha256'):
            if r[key]!=controller[key]:raise ValueError('worker/controller identity disagreement')
        for key in ('input_identity','checkpoint','physical_identity'):
            if key not in r:raise ValueError('missing authenticated real-weight identity')
        if r['input_identity']!=rows[0]['input_identity'] or r['jax']!=rows[0]['jax']:
            raise ValueError('different input/runtime identities across ranks')
        if (r['input_identity']['db_run_id']!=610 or r['input_identity']['prompt_tokens']!=2034
                or r['input_identity']['reference_tokens']!=29):
            raise ValueError('not the registered DB610 comparison')
        for key in ('manifest_sha256','success_sha256','inventory_sha256'):
            if r['checkpoint'][key]!=rows[0]['checkpoint'][key]:raise ValueError('checkpoint identity drift across ranks')
        physical=r['physical_identity']
        for key in ('mesh_sha256','topology_sha256','fleet_sha256'):
            if physical[key]!=rows[0]['physical_identity'][key]:raise ValueError('physical identity drift across ranks')
        if len(physical['local_slots'])!=4 or r['checkpoint']['verified_slots']!=physical['local_slots']:
            raise ValueError('verified checkpoint slots differ from physical owners')
        slots.extend(physical['local_slots'])
        required=('verify_checkpoint','load_checkpoint','prefill_frontier','prefill_cache_finite','final_cache_finite',
                  *(f'prefill_block_{i}' for i in range(16)),*(f'decode_health_{i}' for i in range(1,29)))
        if any(name not in r['phases'] for name in required) or not all(p['passed'] is True for p in r['phases'].values()):
            raise ValueError('missing or failed execution phase')
        if not all(r['finite_cache_checks'].get(k) is True for k in ('after_prefill','after_decode')):
            raise ValueError('missing numerical finiteness checks')
        if set(r['programs'])!=set(programs):raise ValueError('missing compiled program evidence')
        for name in programs:
            p=r['programs'][name]
            if p['memory_admission']['passed'] is not True:raise ValueError('memory admission failed')
            for key in ('stablehlo_sha256','optimized_hlo_sha256'):
                if p[key]!=rows[0]['programs'][name][key]:raise ValueError('compiled graph disagreement')
            if name in model_programs and p['hlo_admission']['passed'] is not True:
                raise ValueError('model HLO admission failed')
        c=r['token_comparison']
        if (c['compared']!=29 or not 0<=c['matches']<=29 or c['all_equal']!=(c['matches']==29)
                or (c['first_mismatch_index'] is None)!=c['all_equal']):
            raise ValueError('inconsistent token comparison')
        if r['prefill']['prompt_tokens']!=2034 or r['decode']['samples']!=28:
            raise ValueError('incomplete timing scope')
        if len(r['decode']['memory_after'])!=4:
            raise ValueError('missing per-chip measured memory')
        measured_devices.extend(d['device_id'] for d in r['decode']['memory_after'])
        times=[r['prefill']['wall_seconds'],r['prefill']['prompt_tokens_per_second'],
               *(r['decode'][k] for k in ('p50_ms','p99_ms','after_five_warm_steps_p50_ms','model_tokens_per_second'))]
        if not all(np.isfinite(t) and t>0 for t in times):raise ValueError('invalid measured timings')
    if len(slots)!=32 or set(slots)!=set(range(32)):
        raise ValueError('checkpoint verification does not cover all32 physical slots exactly once')
    if len(measured_devices)!=32 or set(measured_devices)!=set(range(32)):
        raise ValueError('measured memory does not cover all32 chips exactly once')
    def span(values):return dict(min=min(values),max=max(values))
    comparison_keys=('compared','matches','all_equal','first_mismatch_index','observed_sha256')
    comparisons=[{k:r['token_comparison'][k] for k in comparison_keys} for r in rows]
    same=len({c['observed_sha256'] for c in comparisons})==1
    result = dict(schema='glm_perf_real_validation_fleet_v1',run_dir=root.name,
        code_hash=controller['code_hash'],source_manifest_sha256=controller['source_manifest_sha256'],
        input_sha256=controller['input_sha256'],input_identity={k:rows[0]['input_identity'][k] for k in
            ('db_run_id','run_tag','runner_sha256','ledger_sha256','prompt_tokens','reference_tokens',
             'prompt_sha256','reference_sha256','token_oracle_manifest_sha256')},
        checkpoint={k:rows[0]['checkpoint'][k] for k in ('manifest_sha256','success_sha256','inventory_sha256')},
        physical_identity={k:rows[0]['physical_identity'][k] for k in ('mesh_sha256','topology_sha256','fleet_sha256')},
        checkpoint_slots_verified=32,
        devices=32,hosts=[r['hostname'] for r in rows],jax=rows[0]['jax'],capacity=8192,
        complete_experiment=True,db610_token_check_passed=same and all(c['all_equal'] for c in comparisons),
        fleet_tokens_agree=same,token_comparison=comparisons,
        frozen_graph_admission_inherited=False,trained_model_quality_claim=False,
        programs={name:{k:rows[0]['programs'][name][k] for k in
            ('stablehlo_sha256','optimized_hlo_sha256','compiled_memory')} for name in programs},
        model_hlo_checks={name:rows[0]['programs'][name]['hlo_admission'] for name in model_programs},
        prefill=dict(prompt_tokens=2034,wall_seconds=span([r['prefill']['wall_seconds'] for r in rows]),
            prompt_tokens_per_second=span([r['prefill']['prompt_tokens_per_second'] for r in rows]),
            includes_block_health_votes_and_receipts=True),
        decode={k:span([r['decode'][k] for r in rows]) for k in
            ('p50_ms','p99_ms','after_five_warm_steps_p50_ms','model_tokens_per_second')},
        decode_samples=28,decode_excludes_host_checks_and_delivery=True,
        maximum_peak_hbm_bytes=max(d['peak_bytes_in_use'] for r in rows for d in r['decode']['memory_after']),
        minimum_hbm_headroom_bytes=min(d['bytes_limit']-d['peak_bytes_in_use'] for r in rows for d in r['decode']['memory_after']),
        originals_sha256=[sha256((root/f'validation.rank{i}.json').read_bytes()).hexdigest() for i in range(8)])
    if any('first_decode_diagnostic' in r for r in rows):
        result['first_decode_diagnostic'] = _summarize_layerwise(rows)
    if any('write_empty_route_slot' in r for r in rows):
        flags=[r.get('write_empty_route_slot') for r in rows]
        if any(type(f) is not bool or f!=flags[0] for f in flags):
            raise ValueError('empty route store option differs across ranks')
        result['write_empty_route_slot']=flags[0]
    if any('decode_ablations' in r for r in rows):
        variants={'d1_d8':(False,False),'d1_d8_d5':(True,False),'d1_d8_d10':(False,True)}
        for r in rows:
            if set(r.get('decode_ablations',{}))!=set(variants):
                raise ValueError('incomplete decode ablation')
            for label,(lse,two_stage) in variants.items():
                a=r['decode_ablations'][label]
                name='ablation_'+label
                p=r.get('ablation_programs',{}).get(name,{})
                if (a['lse_attention'] is not lse or a['dsa_two_stage'] is not two_stage
                        or any(type(a.get(k)) is not bool for k in ('healthy','all_steps_finite','final_cache_finite'))
                        or r['phases'].get(name+'_cache_check',{}).get('passed') is not True
                        or a.get('steps')!=28 or any(r['phases'].get(name+'_step_'+str(i),{}).get('passed') is not True for i in range(1,29))
                        or p.get('hlo_admission',{}).get('passed') is not True
                        or p.get('memory_admission',{}).get('passed') is not True):
                    raise ValueError('missing ablation execution/admission')
                if any(p[k]!=rows[0]['ablation_programs'][name][k] for k in ('stablehlo_sha256','optimized_hlo_sha256')):
                    raise ValueError('ablation graph disagreement')
                c=a['token_comparison']
                if (c['compared']!=29 or not 0<=c['matches']<=29 or c['all_equal']!=(c['matches']==29)
                        or (c['first_mismatch_index'] is None)!=c['all_equal']):
                    raise ValueError('inconsistent ablation token comparison')
        result['decode_ablations']={label:dict(
            lse_attention=lse,dsa_two_stage=two_stage,timing_claim=False,
            db610_trail_passed=all(r['decode_ablations'][label]['healthy'] and r['decode_ablations'][label]['all_steps_finite'] and r['decode_ablations'][label]['final_cache_finite'] and r['decode_ablations'][label]['token_comparison']['all_equal'] for r in rows)
                and len({r['decode_ablations'][label]['token_comparison']['observed_sha256'] for r in rows})==1,
            ranks=[dict(healthy=r['decode_ablations'][label]['healthy'],
                        all_steps_finite=r['decode_ablations'][label]['all_steps_finite'],
                        final_cache_finite=r['decode_ablations'][label]['final_cache_finite'],
                        token_comparison={k:r['decode_ablations'][label]['token_comparison'][k] for k in
                            ('compared','matches','all_equal','first_mismatch_index','observed_sha256')}) for r in rows],
            residual_all_finite=all(r['decode_ablations'][label]['final_residual']['finite'] for r in rows),
            residual_nonzero_counts=[r['decode_ablations'][label]['final_residual']['nonzero'] for r in rows],
            program={k:rows[0]['ablation_programs']['ablation_'+label][k] for k in
                     ('stablehlo_sha256','optimized_hlo_sha256','compiled_memory')})
            for label,(lse,two_stage) in variants.items()}
    return result


def _summarize_layerwise(rows):
    """Require a complete diagnostic fleet and expose only aggregate fields."""
    names = {'diagnostic_embedding','diagnostic_layer_full_dense',
             'diagnostic_layer_full_sparse','diagnostic_layer_shared_sparse','diagnostic_head'}
    flags = ('head_healthy','whole_healthy','whole_first_token_matches_db610',
             'split_first_token_matches_db610','split_and_whole_token_equal',
             'split_and_whole_residual_bitwise')
    reports=[]
    for r in rows:
        d=r.get('first_decode_diagnostic',{})
        p=r.get('diagnostic_programs',{})
        if (d.get('schema')!='glm_perf_layerwise_diagnostic_v1' or d.get('compiler_boundary_changed') is not True
                or [x['layer'] for x in d.get('layers',[])]!=list(range(78)) or set(p)!=names
                or r['phases'].get('layerwise_diagnostic',{}).get('passed') is not True):
            raise ValueError('incomplete first-step diagnostic')
        for name in names:
            if p[name]['memory_admission']['passed'] is not True or p[name]['hlo_admission']['passed'] is not True:
                raise ValueError('diagnostic graph admission failed')
            if any(p[name][k]!=rows[0]['diagnostic_programs'][name][k]
                   for k in ('stablehlo_sha256','optimized_hlo_sha256')):
                raise ValueError('diagnostic graph disagreement')
        if any(type(d.get(k)) is not bool for k in flags):
            raise ValueError('missing diagnostic comparison')
        reports.append(d)
    def statistics(values):
        for v in values:
            if (type(v['finite']) is not bool or not 0<=v['nonzero']<=v['elements'] or v['elements']<=0
                    or (v['finite'] and not all(np.isfinite(v[k]) and v[k]>=0 for k in ('max_abs','rms')))):
                raise ValueError('invalid diagnostic activation statistics')
        return dict(all_finite=all(v['finite'] for v in values),
            min_local_nonzero_fraction=min(v['nonzero']/v['elements'] for v in values),
            max_abs=max(v['max_abs'] for v in values) if all(np.isfinite(v['max_abs']) for v in values) else None,
            max_local_rms=max(v['rms'] for v in values) if all(np.isfinite(v['rms']) for v in values) else None)
    return dict(compiler_boundary_changed=True,
        **{k:[d[k] for d in reports] for k in flags},
        **{k:statistics([d[k] for d in reports]) for k in ('embedding','final_residual','whole_final_residual')},
        layers=[dict(layer=i,all_healthy=all(d['layers'][i]['healthy'] for d in reports),
                     **{k:statistics([d['layers'][i][k] for d in reports])
                        for k in ('update','carried','normalized_input')}) for i in range(78)],
        programs={name:{k:rows[0]['diagnostic_programs'][name][k] for k in
                       ('stablehlo_sha256','optimized_hlo_sha256','compiled_memory')} for name in sorted(names)})
