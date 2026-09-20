"""Bounded final-answer checks against private, separately authenticated oracles.

No model-produced Python is executed. Exact final data is a narrower claim than
correct explanatory prose or an independently tested generated implementation.
"""
import json


def _json_objects(text):
    result=[]
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:raise ValueError('duplicate JSON key')
            value[key]=item
        return value
    decoder=json.JSONDecoder(object_pairs_hook=pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON number')))
    for at,char in enumerate(text):
        if char!='{':continue
        try:value,end=decoder.raw_decode(text[at:])
        except ValueError:continue
        if type(value) is dict:result.append((value,at,at+end))
    return result


def assess_answer(text,finish_reason,oracle):
    if type(text) is not str or finish_reason not in ('eos','length') or type(oracle) is not dict:
        raise ValueError('decoded text, registered terminal reason and private oracle required')
    kind=oracle['kind']
    scopes={
        'manual_explanation_review':'Manual explanation review required; no automatic full-quality score',
        'weighted_interval_schedule':'Final schedule compatibility and exact optimal value; generated Python and explanatory proof not executed or fully graded',
        'exact_json_transform':'Exact final JSON values, types, ordering and requested standalone format',
        'manhattan_tsp':'Final tour validity and exact optimal distance; generated Python and explanatory proof not executed or fully graded',
    }
    if kind not in scopes:raise ValueError('unknown answer oracle kind')
    result=dict(scope=scopes[kind],finish_reason=finish_reason,
        reasoning_closed='</think>' in text,structured_answer_present=False,
        correctness_established=False)
    if not result['reasoning_closed']:
        result['status']='incomplete_reasoning_at_token_cap' if finish_reason=='length' else 'no_closed_reasoning_or_finished_answer'
        return result
    answer=text.rsplit('</think>',1)[1].strip()
    if kind=='manual_explanation_review':
        result.update(status='manual_review_required',nonempty_answer=bool(answer))
        return result
    objects=_json_objects(answer)
    required={'weighted_interval_schedule':{'jobs','value'},'manhattan_tsp':{'tour','distance'}}
    if kind=='exact_json_transform':
        # Exact canonical encoding distinguishes booleans/floats from integers.
        candidates=[(value,start,end) for value,start,end in objects if set(value)==set(oracle['expected'])]
        if not candidates:
            result['status']='no_parseable_final_json';return result
        value,start,end=candidates[-1]
        standalone=start==0 and end==len(answer)
        try:
            equal=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
                   ==json.dumps(oracle['expected'],sort_keys=True,separators=(',',':'),allow_nan=False))
        except ValueError:
            equal=False
        result.update(structured_answer_present=True,values_match_oracle=equal,standalone_json=standalone,
                      correctness_established=equal and standalone)
        result['status']='exact_json_correct' if equal and standalone else ('values_correct_format_failed' if equal else 'final_json_failed_exact_oracle')
        return result
    candidates=[(value,start,end) for value,start,end in objects
                if set(value)==required[kind] and answer[end:].strip() in ('','```')]
    if not candidates:
        result['status']='no_parseable_final_result';return result
    value,_,_=candidates[-1]
    result['structured_answer_present']=True
    if kind=='weighted_interval_schedule':
        jobs={j[0]:j for j in oracle['jobs']};selected=value['jobs']
        valid=(type(selected) is list and all(type(k) is str and k in jobs for k in selected)
               and len(set(selected))==len(selected) and type(value['value']) is int)
        if valid:
            chosen=[jobs[k] for k in selected]
            valid=all(a[2]<=b[1] for a,b in zip(chosen,chosen[1:]))
            valid &= sum(j[3] for j in chosen)==value['value']==oracle['optimal_value']
        result.update(correctness_established=bool(valid),
            status='final_schedule_and_value_correct' if valid else 'final_schedule_or_value_failed_exact_oracle')
    else:
        points=oracle['points'];tour=value['tour']
        valid=(type(tour) is list and len(tour)==len(points)+1 and all(type(k) is str for k in tour)
            and tour[0]==tour[-1]=='A' and set(tour[:-1])==set(points)
            and len(set(tour[:-1]))==len(points) and type(value['distance']) is int)
        if valid:
            cost=sum(abs(points[a][0]-points[b][0])+abs(points[a][1]-points[b][1]) for a,b in zip(tour,tour[1:]))
            valid=cost==value['distance']==oracle['optimal_cost']
        result.update(correctness_established=bool(valid),
            status='final_tour_and_distance_correct' if valid else 'final_tour_or_distance_failed_exact_oracle')
    return result
