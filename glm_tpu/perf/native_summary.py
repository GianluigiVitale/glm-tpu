"""Strict aggregate-only native comparison receipt, after base fleet validation."""
import math
from hashlib import sha256

from ..greenfield.runtime.ws32_request_session import RequestPolicy


_BOUNDARY='rank0 private JSONL token write+flush; no network transport'


def summarize_native_rows(rows,controller,pack_index,question=None,*,suite_cases=()):
    def require(ok,message):
        if not ok:raise ValueError(message)
    def finite(value,positive=False):
        return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)
    def span(values):return dict(min=min(values),max=max(values))
    def digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    require(len(rows)==8 and {r['rank'] for r in rows}==set(range(8)),'native summary requires eight distinct ranks')
    first=rows[0]['native_mtp']
    profiles={'db610':dict(prompt_tokens=2034,max_new_tokens=29,
        prompt_sha256=rows[0]['input_identity']['prompt_sha256'])}
    if question is not None:
        profiles['question']=dict(prompt_tokens=len(question['prompt_ids']),
            max_new_tokens=question['max_new_tokens'],prompt_sha256=question['prompt_ids_sha256'])
    for label,ids,policy,_ in suite_cases:
        require(label not in profiles,'duplicate native summary case label')
        profiles[label]=dict(prompt_tokens=len(ids),max_new_tokens=policy.max_new_tokens,
                            prompt_sha256=sha256(ids.tobytes()).hexdigest())
    labels=set(profiles)
    sizes={1,2,3,8,*(p['prompt_tokens']%8 for p in profiles.values())}
    programs={'native_prefill_128','native_prefill_114','native_ordinary',
        *(f'native_refresh_{n}' for n in sizes-{0}),
        *(f'native_{kind}_{n}' for kind in ('inputs','verify','commit') for n in (1,2,3))}
    for r in rows:
        d=r.get('native_mtp',{})
        require(d.get('schema')=='glm_native_mtp_comparison_v1' and d.get('complete') is True
            and d.get('sampled') is False and d.get('independent_native_reference') is False
            and set(d.get('cases',{}))==labels,'incomplete or differently scoped native comparison')
        require(r.get('native_pack_index_sha256')==controller['native_pack_index_sha256']
            and d['pack_index']['plan_sha256']==pack_index['plan_sha256']
            and d['pack_index']['manifest_sha256']==pack_index['rank_manifests'][str(r['rank'])]['manifest_sha256'],
            'native pack identity differs')
        if suite_cases:
            require(r.get('native_suite_sha256')==controller.get('native_suite_sha256')
                and digest(r.get('native_suite_sha256')),'native suite identity differs')
        require(set(r.get('native_programs',{}))==programs,'native graph coverage differs')
        for name in programs:
            p=r['native_programs'][name]
            require(p['hlo_admission']['passed'] is True and p['memory_admission']['passed'] is True,
                    'native graph admission failed')
            require(all(digest(p[k]) and p[k]==rows[0]['native_programs'][name][k]
                for k in ('stablehlo_sha256','optimized_hlo_sha256')),'native graph hashes differ')
            require(all(r['phases'].get(prefix+name,{}).get('passed') is True
                for prefix in ('compile_','graph_consensus_','hlo_','memory_')),'native graph phase is absent')
        require(d['load_admission']['passed'] is True and d['resident_admission']['passed'] is True,
                'native load or residency admission failed')
    result=dict(schema='glm_native_mtp_comparison_fleet_v1',pack_index_sha256=controller['native_pack_index_sha256'],
        sampled=False,independent_native_reference=False,cases={},
        programs={name:{k:rows[0]['native_programs'][name][k] for k in
                       ('stablehlo_sha256','optimized_hlo_sha256','compiled_memory')} for name in sorted(programs)})
    for label in sorted(labels):
        case0=first['cases'][label];policy0=case0['policy']
        prompt_sha=profiles[label]['prompt_sha256']
        for r in rows:
            case=r['native_mtp']['cases'][label]
            require(case['policy']==policy0 and case['prompt_sha256']==prompt_sha,'native case input identity differs')
            policy=RequestPolicy(**dict(case['policy'],eos_ids=tuple(case['policy']['eos_ids'])))
            expected_prompt=profiles[label]['prompt_tokens']
            expected_cap=profiles[label]['max_new_tokens']
            require(policy.prompt_tokens==expected_prompt and policy.max_new_tokens==expected_cap,
                    'native case prompt or output budget differs')
            require(case['prefill']['prompt_tokens']==policy.prompt_tokens
                and finite(case['prefill']['wall_seconds'],True)
                and math.isclose(case['prefill']['prompt_tokens_per_second'],
                    policy.prompt_tokens/case['prefill']['wall_seconds'],rel_tol=1e-9),
                'paired ordinary prefill timing differs')
            require(case['export_db610_parity'] is (label=='db610'),'native target export gate differs')
            require(set(case['live_memory_admission'])==programs and all(
                x['passed'] is True for x in case['live_memory_admission'].values()),'native live graph admission missing')
            base=case['ordinary']
            require(base['healthy'] is True and base['speculative'] is False and base['sampling']=='greedy'
                and base['decode_steps']==base['emitted']-1 and base['decode_steps']>0
                and base['timed_votes']==2*base['decode_steps'] and base['warm_steps_excluded']==0
                and base['delivery_boundary']==_BOUNDARY and base['excludes_cold_load_compile'] is True
                and base['decode_rate_excludes_prefill'] is True,'ordinary paired timing boundary differs')
            require(digest(base['token_sha256']) and base['token_sha256']==case0['ordinary']['token_sha256'],
                    'ordinary paired output hashes differ')
            if label=='db610':
                require(base['emitted']==29 and base['token_sha256']==r['input_identity']['reference_sha256'],
                        'native hidden-export DB610 parity failed')
            require(finite(base['wall_seconds'],True) and math.isclose(base['tokens_per_second'],
                base['decode_steps']/base['wall_seconds'],rel_tol=1e-9),'ordinary paired rate differs')
            required=[label+'_ordinary',label+'_ordinary_agreement',label+'_ordinary_finite',
                      label+'_bootstrap',label+'_resident_admission',*(label+'_warm_'+str(n) for n in (1,2,3))]
            if label=='db610':required.append('db610_export_parity')
            require(all(r['phases'].get(k,{}).get('passed') is True for k in required),'native execution phase missing')
            require(set(case['speculative'])=={'2','3'},'native comparison needs both draft lengths')
            for key in ('2','3'):
                d=case['speculative'][key];n=int(key);ref=case0['speculative'][key]
                integer_keys=('rows','emitted','decode_tokens','rounds','accepted_drafts','proposed_drafts')
                require(all(type(d[k]) is int and d[k]>=0 for k in integer_keys) and d['rows']==n
                    and 1<d['emitted']<=policy.max_new_tokens and d['decode_tokens']==d['emitted']-1
                    and 0<d['rounds']<=d['decode_tokens']<=n*d['rounds']
                    and 0<=d['accepted_drafts']<=d['proposed_drafts']<n*d['rounds'],
                    'native token/round accounting differs')
                require(d['finish_reason'] in ('eos','length') and (d['finish_reason']!='length'
                    or d['emitted']==policy.max_new_tokens),'native terminal boundary differs')
                require(all(d.get(k) is True for k in ('healthy','all_host_token_agreement','finite_caches',
                    'includes_draft_verify_rejections_refresh_commit_votes_delivery','excludes_cold_load_compile',
                    'decode_rate_excludes_prefill')) and d['warm_steps_excluded']==0
                    and d['delivery_boundary']==_BOUNDARY,'native health or timing scope differs')
                for k in ('token_sha256','emitted','decode_tokens','rounds','accepted_drafts','proposed_drafts',
                          'accepted_by_position','proposals_by_position','finish_reason','ordinary_agreement'):
                    require(d[k]==ref[k],'native output/acceptance differs across hosts')
                require(digest(d['token_sha256']),'invalid native token digest')
                require(len(d['accepted_by_position'])==n-1 and len(d['proposals_by_position'])==n-1
                    and all(type(a) is int and type(p) is int and 0<=a<=p<=d['rounds']
                            for a,p in zip(d['accepted_by_position'],d['proposals_by_position']))
                    and sum(d['accepted_by_position'])==d['accepted_drafts']
                    and sum(d['proposals_by_position'])==d['proposed_drafts'],'native draft-position counts differ')
                timing=('wall_seconds','tokens_per_second','proposal_seconds','refresh_commit_seconds',
                    'host_vote_seconds','host_agreement_seconds','ttft_seconds','request_wall_seconds','bootstrap_seconds','p50_round_ms')
                require(all(finite(d[k],k in ('wall_seconds','tokens_per_second')) for k in timing),
                        'invalid native timings')
                require(math.isclose(d['tokens_per_second'],d['decode_tokens']/d['wall_seconds'],rel_tol=1e-9)
                    and math.isclose(d['accepted_per_round'],d['decode_tokens']/d['rounds'],rel_tol=1e-9)
                    and math.isclose(d['paired_wall_speedup'],d['tokens_per_second']/base['tokens_per_second'],rel_tol=1e-9),
                    'native wall rate, acceptance or paired speedup differs')
                components=d['component_seconds']
                require(set(components)=={'draft','verify','commit','refresh'}
                    and all(finite(v) for v in components.values())
                    and sum(components.values())+d['host_vote_seconds']+d['host_agreement_seconds']<=d['wall_seconds']+1e-6,
                    'native component work exceeds measured wall')
                pref=d['prefill']
                require(pref['prompt_tokens']==policy.prompt_tokens and finite(pref['wall_seconds'],True)
                    and math.isclose(pref['prompt_tokens_per_second'],policy.prompt_tokens/pref['wall_seconds'],rel_tol=1e-9)
                    and d['ttft_seconds']+1e-6>=pref['wall_seconds']+d['bootstrap_seconds']
                    and d['request_wall_seconds']+1e-6>=d['ttft_seconds']+d['wall_seconds'],
                    'native prefill/bootstrap/TTFT boundary differs')
                comparison=d['ordinary_agreement']
                require(comparison['baseline_token_sha256']==base['token_sha256']
                    and type(comparison['all_equal']) is bool and comparison['multirow_numerical_boundary'] is True
                    and (comparison['first_mismatch'] is None)==comparison['all_equal'],
                    'native ordinary-output comparison is inconsistent')
                require(not comparison['all_equal'] or (d['token_sha256']==base['token_sha256']
                    and d['emitted']==base['emitted']),'native claimed token equality disagrees with digest')
                memory=d['memory_after']
                require(len(memory)==4 and {v['device_id'] for v in memory}==
                    {v['device_id'] for v in r['decode']['memory_after']}
                    and all(0<v['peak_bytes_in_use']<=v['bytes_limit'] for v in memory),
                    'native per-chip memory evidence differs')
                tag=f'native.{label}.r{n}'
                require(all(r['phases'].get(tag+suffix,{}).get('passed') is True
                    for suffix in ('_generation','_agreement','_finite')),'native measured generation phase missing')
        case_result=dict(prompt_tokens=policy0['prompt_tokens'],max_new_tokens=policy0['max_new_tokens'],
            prompt_sha256=prompt_sha,ordinary=dict(emitted=case0['ordinary']['emitted'],
                token_sha256=case0['ordinary']['token_sha256'],tokens_per_second=span([
                    r['native_mtp']['cases'][label]['ordinary']['tokens_per_second'] for r in rows])),speculative={})
        case_result['ordinary']['prefill']={k:span([r['native_mtp']['cases'][label]['prefill'][k] for r in rows])
            for k in ('wall_seconds','prompt_tokens_per_second')}
        for key in ('2','3'):
            ds=[r['native_mtp']['cases'][label]['speculative'][key] for r in rows]
            d=ds[0]
            case_result['speculative'][key]={k:d[k] for k in ('rows','emitted','decode_tokens','rounds',
                'accepted_drafts','proposed_drafts','accepted_by_position','proposals_by_position','finish_reason','token_sha256')}
            case_result['speculative'][key].update(
                ordinary_agreement={k:d['ordinary_agreement'][k] for k in
                    ('all_equal','first_mismatch','baseline_token_sha256','multirow_numerical_boundary')},
                timings={k:span([v[k] for v in ds]) for k in ('wall_seconds','tokens_per_second','accepted_per_round',
                    'paired_wall_speedup','proposal_seconds','refresh_commit_seconds','host_vote_seconds',
                    'host_agreement_seconds','ttft_seconds','request_wall_seconds','bootstrap_seconds','p50_round_ms')},
                prefill={k:span([v['prefill'][k] for v in ds]) for k in ('wall_seconds','prompt_tokens_per_second')},
                component_seconds={k:span([v['component_seconds'][k] for v in ds]) for k in ('draft','verify','commit','refresh')},
                maximum_peak_hbm_bytes=max(v['peak_bytes_in_use'] for d in ds for v in d['memory_after']))
        result['cases'][label]=case_result
    result.update(delivery_boundary=_BOUNDARY,all_host_tokens_agree=True,
                  prefill_and_startup_excluded_from_decode_rate=True,
                  includes_draft_verify_rejections_refresh_commit_votes_delivery=True,
                  answer_correctness='requires separate assessment; output agreement is not a task-quality score')
    return result
