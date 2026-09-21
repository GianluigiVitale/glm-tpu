"""Private greedy requests for the fixed optimized ordinary release profile."""
from hashlib import sha256
import json
import os
from pathlib import Path

from .. import user_request as legacy
from . import model

SCHEMA = 'glm_ws32_optimized_request_v2'
CAPACITY = 8192
LONG_CAPACITY = 166912
CONCURRENT_CAPACITY = 32768
CONCURRENT_LIMIT = 4
MAX_PROMPT = 131072
BATCH_SCHEMA = 'glm_ws32_optimized_batch_v1'
CONCURRENT_SCHEMA = 'glm_ws32_concurrent_batch_v1'


def from_token_ids(ids, *, request_id, max_new_tokens, context_capacity=CAPACITY):
    # Reuse strict token/type/identity validation, then bind the narrower profile.
    original = legacy.from_token_ids(ids, request_id=request_id, seed=0,
                                     max_new_tokens=max_new_tokens)
    if type(context_capacity) is not int or context_capacity not in (CAPACITY,LONG_CAPACITY,CONCURRENT_CAPACITY):
        raise ValueError('unsupported ordinary context capacity')
    if len(ids) > MAX_PROMPT or len(ids) + max_new_tokens > context_capacity:
        raise ValueError('prompt and complete output budget exceed the selected capacity')
    body = dict(schema=SCHEMA, request_id=request_id, prompt_ids=original['prompt_ids'],
        prompt_ids_sha256=original['prompt_ids_sha256'], max_new_tokens=max_new_tokens,
        context_capacity=context_capacity, vocab_size=legacy.VOCAB, eos_ids=list(legacy.EOS),
        decode_policy='greedy', thinking='on/max', tokenizer_files=dict(model.TOKENIZER_FILES),
        chat_template_sha256=model.TEMPLATE_SHA, model_id=model.MODEL_ID,
        model_revision=model.REVISION)
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


def batch(values, *, concurrent=False):
    """Bind either a sequential queue or an explicitly concurrent 32K batch."""
    if type(concurrent) is not bool:
        raise ValueError('concurrent must be boolean')
    limit=CONCURRENT_LIMIT if concurrent else 10
    if type(values) is not list or not 1 <= len(values) <= limit:
        raise ValueError(f'submit between one and {limit} requests')
    for value in values:validate(value)
    if len({v['request_id'] for v in values}) != len(values):
        raise ValueError('request IDs must be unique')
    capacities={v['context_capacity'] for v in values}
    if len(capacities)!=1:raise ValueError('one loaded model requires one context capacity')
    capacity=capacities.pop()
    if concurrent and capacity!=CONCURRENT_CAPACITY:
        raise ValueError('concurrent conversations require 32K total slots each')
    body=dict(schema=CONCURRENT_SCHEMA if concurrent else BATCH_SCHEMA,
              requests=values,context_capacity=capacity)
    return dict(body,request_sha256=sha256(legacy.canonical(body)).hexdigest())


def validate_payload(value):
    if type(value) is dict and value.get('schema') in (BATCH_SCHEMA,CONCURRENT_SCHEMA):
        if legacy.canonical(value)!=legacy.canonical(batch(value.get('requests'),
                concurrent=value['schema']==CONCURRENT_SCHEMA)):
            raise ValueError('batch identity differs')
    else:validate(value)


def requests(value):
    validate_payload(value)
    return value['requests'] if value.get('schema') in (BATCH_SCHEMA,CONCURRENT_SCHEMA) else [value]


def stop_cause(value, finish_reason):
    """Distinguish normal EOS, context exhaustion and an explicit shorter cap."""
    if finish_reason == 'eos':
        return 'eos'
    if finish_reason == 'length':
        return ('context_exhausted' if len(value['prompt_ids']) + value['max_new_tokens']
                == value['context_capacity'] else 'output_cap')
    raise ValueError('request did not reach a terminal token condition')


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
    template = model.verified_template(repo, tokenizer_root)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True,
                                               trust_remote_code=False)
    value = from_messages(messages, tokenizer=tokenizer, chat_template=template,
        request_id=request_id, max_new_tokens=max_new_tokens,
        context_capacity=context_capacity)
    output = legacy._plain(output)
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError('private requests must be outside the source checkout')
    raw = legacy.canonical(value) + b'\n'
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(request_sha256=value['request_sha256'], file_sha256=sha256(raw).hexdigest(),
        prompt_tokens=len(value['prompt_ids']), max_new_tokens=value['max_new_tokens'],
        context_capacity=context_capacity, decode_policy='greedy', model_executions=0)


def from_messages(messages, *, tokenizer, chat_template, request_id,
                  max_new_tokens=None, context_capacity=CAPACITY):
    """Tokenize the complete chat at maximum thinking effort, without truncation."""
    if (type(messages) is not list or not messages
            or len(legacy.canonical(messages)) > legacy.MESSAGES_CAP
            or any(type(m) is not dict or set(m) != {'role', 'content'}
                   or m['role'] not in ('system', 'user', 'assistant')
                   or type(m['content']) is not str for m in messages)
            or messages[-1]['role'] != 'user'
            or any(m['role'] == 'system' for m in messages[1:])):
        raise ValueError('expected bounded text-only chat ending with a user message')
    if sha256(chat_template.encode()).hexdigest() != model.TEMPLATE_SHA:
        raise ValueError('chat template differs from pinned GLM-5.3')
    if type(context_capacity) is not int or context_capacity not in (CAPACITY, LONG_CAPACITY, CONCURRENT_CAPACITY):
        raise ValueError('unsupported ordinary context capacity')
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=False, chat_template=chat_template,
        enable_thinking=True, reasoning_effort='max')
    if max_new_tokens is None:
        max_new_tokens = min(legacy.MAX_NEW, context_capacity - len(ids))
    return from_token_ids(ids, request_id=request_id, max_new_tokens=max_new_tokens,
                          context_capacity=context_capacity)
