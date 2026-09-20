"""Private greedy requests for the fixed optimized ordinary release profile."""
from hashlib import sha256
import json
import os
from pathlib import Path

from .. import user_request as legacy

SCHEMA = 'glm_ws32_optimized_request_v1'
CAPACITY = 8192


def from_token_ids(ids, *, request_id, max_new_tokens):
    # Reuse strict token/type/identity validation, then bind the narrower profile.
    original = legacy.from_token_ids(ids, request_id=request_id, seed=0,
                                     max_new_tokens=max_new_tokens)
    if len(ids) + max_new_tokens > CAPACITY:
        raise ValueError('optimized prompt plus output budget must fit 8192 tokens')
    body = dict(schema=SCHEMA, request_id=request_id, prompt_ids=original['prompt_ids'],
        prompt_ids_sha256=original['prompt_ids_sha256'], max_new_tokens=max_new_tokens,
        context_capacity=CAPACITY, vocab_size=legacy.VOCAB, eos_ids=list(legacy.EOS),
        decode_policy='greedy', thinking='on/max', tokenizer_files=dict(legacy.TOKENIZER_FILES),
        chat_template_sha256=legacy.TEMPLATE_SHA)
    return dict(body, request_sha256=sha256(legacy.canonical(body)).hexdigest())


def validate(value):
    if type(value) is not dict:
        raise ValueError('optimized request must be an object')
    try:
        expected = from_token_ids(value['prompt_ids'], request_id=value['request_id'],
                                  max_new_tokens=value['max_new_tokens'])
    except KeyError as exc:
        raise ValueError('missing optimized request fields') from exc
    if legacy.canonical(value) != legacy.canonical(expected):
        raise ValueError('optimized request identity or fixed profile differs')


def read(path, *, expected_sha256=None):
    raw = legacy.read_bounded(Path(path), legacy.PAYLOAD_CAP)
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('optimized request file digest differs')
    value = json.loads(raw)
    validate(value)
    return value


def prepare_file(*, messages_path, output, repo, tokenizer_root, request_id,
                 max_new_tokens):
    messages = json.loads(legacy.read_bounded(messages_path, legacy.MESSAGES_CAP))
    template = legacy.read_bounded(repo/'reference/hf-repo/chat_template.jinja', 64 << 10).decode()
    for name, digest in legacy.TOKENIZER_FILES.items():
        if sha256(legacy.read_bounded(tokenizer_root/name, 32 << 20)).hexdigest() != digest:
            raise ValueError('tokenizer differs from the fixed profile')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True,
                                               trust_remote_code=False)
    original = legacy.from_messages(messages, tokenizer=tokenizer, chat_template=template,
        request_id=request_id, seed=0, max_new_tokens=max_new_tokens)
    value = from_token_ids(original['prompt_ids'], request_id=request_id,
                           max_new_tokens=max_new_tokens)
    output = legacy._plain(output)
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError('private requests must be outside the source checkout')
    raw = legacy.canonical(value) + b'\n'
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(request_sha256=value['request_sha256'], file_sha256=sha256(raw).hexdigest(),
        prompt_tokens=len(value['prompt_ids']), max_new_tokens=max_new_tokens,
        context_capacity=CAPACITY, decode_policy='greedy', model_executions=0)
