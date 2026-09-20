import json
import pytest
from glm_tpu.perf.answer_assessment import assess_answer


def text(value):return 'private reasoning</think>\n'+json.dumps(value)


def test_incomplete_reasoning_is_never_graded_from_intermediate_correct_json():
    oracle=dict(kind='exact_json_transform',expected=dict(total=3))
    value=assess_answer('{"total":3}','length',oracle)
    assert not value['correctness_established'] and value['status']=='incomplete_reasoning_at_token_cap'


def test_exact_json_types_format_and_duplicate_keys():
    oracle=dict(kind='exact_json_transform',expected=dict(total=1,ids=['A','B']))
    assert assess_answer(text(oracle['expected']),'eos',oracle)['correctness_established']
    for payload in ('{"total":true,"ids":["A","B"]}', '{"total":1.0,"ids":["A","B"]}',
                    '{"total":1,"ids":["B","A"]}', '{"total":1e999,"ids":["A","B"]}', '{"total":0,"total":1,"ids":["A","B"]}'):
        assert not assess_answer('</think>'+payload,'eos',oracle)['correctness_established']
    value=assess_answer('</think>Here is the result: '+json.dumps(oracle['expected']),'eos',oracle)
    assert value['values_match_oracle'] and not value['correctness_established'] and not value['standalone_json']


def test_scheduling_allows_tied_optima_and_checks_real_compatibility():
    oracle=dict(kind='weighted_interval_schedule',jobs=[['A',0,2,5],['B',2,4,5],['C',0,4,10]],optimal_value=10)
    for chosen in (['A','B'],['C']):
        assert assess_answer(text(dict(jobs=chosen,value=10)),'eos',oracle)['correctness_established']
    for chosen,value in ((['A','C'],15),(['A','A'],10),(['B','A'],10),(['A'],10),(['Z'],10),(['C'],True)):
        assert not assess_answer(text(dict(jobs=chosen,value=value)),'eos',oracle)['correctness_established']
    assert not assess_answer(text(dict(jobs=['A','B'],value=10))+'\nThis was only an example.','eos',oracle)['correctness_established']


def test_tsp_checks_closed_hamiltonian_tour_cost_and_optimum():
    oracle=dict(kind='manhattan_tsp',points=dict(A=[0,0],B=[1,0],C=[1,1]),optimal_cost=4)
    assert assess_answer(text(dict(tour=['A','B','C','A'],distance=4)),'eos',oracle)['correctness_established']
    for tour,distance in ((['A','B','A'],4),(['A','B','B','A'],4),(['A','B','C','A'],5),(['B','C','A','B'],4)):
        assert not assess_answer(text(dict(tour=tour,distance=distance)),'eos',oracle)['correctness_established']


def test_prose_and_code_execution_never_claimed_as_automatically_verified():
    value=assess_answer('thinking</think>Some prose','eos',dict(kind='manual_explanation_review'))
    assert value['status']=='manual_review_required' and not value['correctness_established']
    assert 'Some prose' not in str(value)
    with pytest.raises(ValueError):assess_answer('text','eos',dict(kind='unknown'))
