"""Private greedy requests for the fixed optimized ordinary release profile."""
from hashlib import sha256
import json
import os
from pathlib import Path

from .. import user_request as legacy

SCHEMA = 'glm_ws32_optimized_request_v1'
CAPACITY = 8192
LONG_CAPACITY = 166912
MAX_PROMPT = 131072
BATCH_SCHEMA = 'glm_ws32_optimized_batch_v1'


def from_token_ids(ids, *, request_id, max_new_tokens, context_capacity=CAPACITY):
    # Reuse strict token/type/identity validation, then bind the narrower profile.
    original = legacy.from_token_ids(ids, request_id=request_id, seed=0,
                                     max_new_tokens=max_new_tokens)
    if type(context_capacity) is not int or context_capacity not in (CAPACITY,LONG_CAPACITY):
        raise ValueError('unsupported ordinary context capacity')
    if len(ids) > MAX_PROMPT or len(ids) + max_new_tokens > context_capacity:
        raise ValueError('prompt and complete output budget exceed the selected capacity')
    body = dict(schema=SCHEMA, request_id=request_id, prompt_ids=original['prompt_ids'],
        prompt_ids_sha256=original['prompt_ids_sha256'], max_new_tokens=max_new_tokens,
        context_capacity=context_capacity, vocab_size=legacy.VOCAB, eos_ids=list(legacy.EOS),
        decode_policy='greedy', thinking='on/max', tokenizer_files=dict(legacy.TOKENIZER_FILES),
        chat_template_sha256=legacy.TEMPLATE_SHA)
    return dict(body, request_sha256=sha256(legacy.canonical(body)).hexdigest())


def validate(value):
    if type(value) is not dict:
        raise ValueError('optimized request must be an object')
    try:
        expected = from_token_ids(value['prompt_ids'], request_id=value['request_id'],
                                  max_new_tokens=value['max_new_tokens'],
                                  context_capacity=value['context_capacity'])
    except KeyError as exc:
        raise ValueError('missing optimized request fields') from exc
    if legacy.canonical(value) != legacy.canonical(expected):
        raise ValueError('optimized request identity or fixed profile differs')


def batch(values):
    """Submit up to ten isolated questions to one resident, sequential model."""
    if type(values) is not list or not 1 <= len(values) <= 10:
        raise ValueError('submit between one and ten requests')
    for value in values:validate(value)
    if len({v['request_id'] for v in values}) != len(values):
        raise ValueError('request IDs must be unique')
    capacities={v['context_capacity'] for v in values}
    if len(capacities)!=1:raise ValueError('one loaded model requires one context capacity')
    body=dict(schema=BATCH_SCHEMA,requests=values,context_capacity=capacities.pop())
    return dict(body,request_sha256=sha256(legacy.canonical(body)).hexdigest())


def validate_payload(value):
    if type(value) is dict and value.get('schema')==BATCH_SCHEMA:
        if legacy.canonical(value)!=legacy.canonical(batch(value.get('requests'))):
            raise ValueError('batch identity differs')
    else:validate(value)


def requests(value):
    validate_payload(value)
    return value['requests'] if value.get('schema')==BATCH_SCHEMA else [value]


def read(path, *, expected_sha256=None):
    raw = legacy.read_bounded(Path(path), legacy.PAYLOAD_CAP)
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('optimized request file digest differs')
    value = json.loads(raw)
    validate_payload(value)
    return value


def prepare_file(*, messages_path, output, repo, tokenizer_root, request_id,
                 max_new_tokens, context_capacity=CAPACITY):
    messages = json.loads(legacy.read_bounded(messages_path, legacy.MESSAGES_CAP))
    template = legacy.read_bounded(repo/'reference/hf-repo/chat_template.jinja', 64 << 10).decode()
    for name, digest in legacy.TOKENIZER_FILES.items():
        if sha256(legacy.read_bounded(tokenizer_root/name, 32 << 20)).hexdigest() != digest:
            raise ValueError('tokenizer differs from the fixed profile')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True,
                                               trust_remote_code=False)
    original = legacy.from_messages(messages, tokenizer=tokenizer, chat_template=template,
        request_id=request_id, seed=0,
        max_new_tokens=1 if max_new_tokens is None else max_new_tokens)
    # Tokenize first so long inputs retain their full content while shorter
    # questions can use the existing output allowance instead of a 32K cutoff.
    if max_new_tokens is None:
        max_new_tokens=min(legacy.MAX_NEW,context_capacity-len(original['prompt_ids']))
    value = from_token_ids(original['prompt_ids'], request_id=request_id,
                           max_new_tokens=max_new_tokens,context_capacity=context_capacity)
    output = legacy._plain(output)
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError('private requests must be outside the source checkout')
    raw = legacy.canonical(value) + b'\n'
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(request_sha256=value['request_sha256'], file_sha256=sha256(raw).hexdigest(),
        prompt_tokens=len(value['prompt_ids']), max_new_tokens=max_new_tokens,
        context_capacity=context_capacity, decode_policy='greedy', model_executions=0)
