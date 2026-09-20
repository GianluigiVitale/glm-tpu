"""Authenticated private representative inputs, with bounded repeat labels."""
from hashlib import sha256
import json
import re

from ..user_request import read_bounded
from .long_question import load_question


def load_native_suite(root, digest, *, capacity, vocab_size, eos_ids):
    raw=read_bounded(root/'native_suite.json',1<<20)
    if sha256(raw).hexdigest()!=digest:raise ValueError('native suite identity differs')
    suite=json.loads(raw)
    if (type(suite) is not dict or set(suite)!={'schema','cases'}
            or suite['schema']!='glm_native_mtp_suite_v1' or type(suite['cases']) is not list
            or not 1<=len(suite['cases'])<=8):
        raise ValueError('native suite schema or case count differs')
    cases=[];seen=set()
    for entry in suite['cases']:
        if (type(entry) is not dict or set(entry)!={'label','question_sha256','repeats'}
                or type(entry['label']) is not str or not re.fullmatch('[a-z][a-z0-9_]{0,23}',entry['label'])
                or entry['label'] in {'db610','question'} or entry['label'] in seen
                or type(entry['repeats']) is not int or not 1<=entry['repeats']<=3):
            raise ValueError('native suite label or repeat count differs')
        seen.add(entry['label'])
        path=root/('case-'+entry['label']+'.json')
        if path.is_symlink():raise ValueError('native suite case must be a regular staged file')
        ids,policy=load_question(path,entry['question_sha256'],capacity=capacity,
                               vocab_size=vocab_size,eos_ids=eos_ids)
        for i in range(entry['repeats']):
            label=entry['label']+'_repeat'+str(i+1)
            cases.append((label,ids,policy,None))
    return cases
