from hashlib import sha256
import json

import numpy as np
import pytest

from glm_tpu.perf.prefix_replay import load_prefix_replay
from glm_tpu.user_request import TOKENIZER_FILES, TEMPLATE_SHA


def bundle(root):
    ids = np.array([10, 11, 12], np.int32)
    question = dict(schema='glm_perf_question_v1', request_id='replay-test',
        prompt_ids=ids.tolist(), prompt_ids_sha256=sha256(ids.tobytes()).hexdigest(),
        max_new_tokens=16, tokenizer_files=TOKENIZER_FILES, chat_template_sha256=TEMPLATE_SHA,
        thinking='on/max', decode_policy='greedy')
    raw = json.dumps(question).encode()
    (root/'code.question.json').write_bytes(raw)
    reference = np.arange(20, 36, dtype=np.int32)
    return dict(schema='glm_perf_prefix_replay_inputs_v1', cases=[dict(name='code',
        question_sha256=sha256(raw).hexdigest(), reference_ids=reference.tolist(),
        reference_sha256=sha256(reference.tobytes()).hexdigest(), source_execution='a'*40,
        offsets=[0, 2, 4, 6])])


def load(root, value, digest=None):
    raw = json.dumps(value).encode()
    (root/'prefix_replay.json').write_bytes(raw)
    return load_prefix_replay(root, digest or sha256(raw).hexdigest(),
                              capacity=64, vocab_size=256, eos_ids=(2,))


def test_bounded_authenticated_private_replay(tmp_path):
    cases = load(tmp_path, bundle(tmp_path))
    name, ids, expected, offsets, identity = cases[0]
    assert name == 'code' and offsets == (0, 2, 4, 6)
    assert ids.dtype == expected.dtype == np.int32
    assert 'reference_ids' not in identity


@pytest.mark.parametrize('bad', ['digest', 'reference', 'question', 'offset', 'duplicate',
                               'path', 'token', 'source', 'bool', 'extra'])
def test_refuse_malformed_or_tampered_inputs(tmp_path, bad):
    value = bundle(tmp_path)
    case = value['cases'][0]
    if bad == 'reference': case['reference_ids'][0] += 1
    if bad == 'question': case['question_sha256'] = '0'*64
    if bad == 'offset': case['offsets'] = [13]
    if bad == 'duplicate': value['cases'].append(dict(case))
    if bad == 'path': case['name'] = '../code'
    if bad == 'token': case['reference_ids'][0] = 256
    if bad == 'source': case['source_execution'] = 'not-a-pin'
    if bad == 'bool': case['offsets'] = [True]
    if bad == 'extra': case['raw_prompt'] = 'must not be accepted'
    with pytest.raises(ValueError):
        load(tmp_path, value, '0'*64 if bad == 'digest' else None)
