from hashlib import sha256
import json

import numpy as np
import pytest

from glm_tpu.perf.native_suite import load_native_suite
from glm_tpu.user_request import TOKENIZER_FILES,TEMPLATE_SHA


def fixture(root):
    ids=np.array([1,2,3,4],np.int32)
    question=dict(schema='glm_perf_question_v1',request_id='suite-test',prompt_ids=ids.tolist(),
        prompt_ids_sha256=sha256(ids.tobytes()).hexdigest(),max_new_tokens=10,tokenizer_files=TOKENIZER_FILES,
        chat_template_sha256=TEMPLATE_SHA,thinking='on/max',decode_policy='greedy')
    case=root/'case-prose.json';case.write_text(json.dumps(question))
    suite=dict(schema='glm_native_mtp_suite_v1',cases=[dict(label='prose',repeats=2,question_sha256=sha256(case.read_bytes()).hexdigest())])
    return suite,case


def load(root,suite):
    path=root/'native_suite.json';path.write_text(json.dumps(suite))
    return load_native_suite(root,sha256(path.read_bytes()).hexdigest(),capacity=20,vocab_size=256,eos_ids=(255,))


def test_repeat_inputs_are_identical_but_have_distinct_measurement_labels(tmp_path):
    suite,path=fixture(tmp_path);cases=load(tmp_path,suite)
    assert [c[0] for c in cases]==['prose_repeat1','prose_repeat2']
    np.testing.assert_array_equal(cases[0][1],cases[1][1])
    assert cases[0][2]==cases[1][2] and cases[0][2].max_new_tokens==10
    assert all(c[3] is None for c in cases)


@pytest.mark.parametrize('fault',['label','reserved','duplicate','repeat','boolean_repeat','digest','tamper','symlink','empty'])
def test_invalid_suite_refused_before_any_device_work(tmp_path,fault):
    suite,path=fixture(tmp_path)
    if fault=='label':suite['cases'][0]['label']='../outside'
    if fault=='reserved':suite['cases'][0]['label']='db610'
    if fault=='duplicate':suite['cases'].append(suite['cases'][0].copy())
    if fault=='repeat':suite['cases'][0]['repeats']=4
    if fault=='boolean_repeat':suite['cases'][0]['repeats']=True
    if fault=='digest':suite['cases'][0]['question_sha256']='0'*64
    if fault=='tamper':path.write_text('{}')
    if fault=='symlink':
        target=tmp_path/'original.json';path.rename(target);path.symlink_to(target)
    if fault=='empty':suite['cases']=[]
    with pytest.raises(ValueError):load(tmp_path,suite)
