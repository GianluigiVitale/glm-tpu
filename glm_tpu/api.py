"""Stateless OpenAI-compatible surface over the existing resident GLM session.

This module owns no model and no device. It renders a complete chat with the
pinned GLM-5.3 template, hands the resulting token ids to the same sequential
resident queue the chat UI uses, and shapes the reply as `/v1/chat/completions`.
Conversation state is never stored: every request carries its own messages.
"""
import json
import re
import time
import uuid

from .optimized.request import PROMPT_LIMITS
from .user_request import MAX_NEW

MODEL_ID = 'glm-5.3'
EFFORTS = ('low', 'high', 'max')
ROLES = ('system', 'user', 'assistant', 'tool')
MESSAGES_CAP = 1 << 20
TOOLS_CAP = 1 << 18
# GLM emits <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
CALL = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
ARGUMENT = re.compile(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', re.S)


class ApiError(ValueError):
    def __init__(self, message, *, status=400, kind='invalid_request_error'):
        super().__init__(message)
        self.status = status
        self.kind = kind

    def body(self):
        return dict(error=dict(message=str(self), type=self.kind, param=None, code=None))


def _text(content):
    """Accept OpenAI string or multi-part content; reject non-text parts."""
    if content is None:
        return ''
    if type(content) is str:
        return content
    if type(content) is list:
        parts = []
        for part in content:
            if type(part) is not dict or part.get('type') != 'text' or type(part.get('text')) is not str:
                raise ApiError('only text content parts are supported')
            parts.append(part['text'])
        return ''.join(parts)
    raise ApiError('message content must be a string or a list of text parts')


def _arguments(value):
    """OpenAI sends arguments as a JSON string; the pinned template wants a mapping."""
    if type(value) is dict:
        return value
    if type(value) is not str:
        raise ApiError('tool call arguments must be a JSON object string')
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except ValueError as exc:
        raise ApiError('tool call arguments are not valid JSON') from exc
    if type(parsed) is not dict:
        raise ApiError('tool call arguments must decode to an object')
    return parsed


def convert(messages):
    """Map an OpenAI message array onto what the pinned GLM-5.3 template renders."""
    if type(messages) is not list or not messages:
        raise ApiError('messages must be a non-empty array')
    if len(json.dumps(messages).encode()) > MESSAGES_CAP:
        raise ApiError('messages exceed the supported size')
    out = []
    for message in messages:
        if type(message) is not dict or message.get('role') not in ROLES:
            raise ApiError('each message needs a role of system, user, assistant or tool')
        role = message['role']
        rendered = dict(role=role, content=_text(message.get('content')))
        if role == 'assistant':
            if type(message.get('reasoning_content')) is str:
                rendered['reasoning_content'] = message['reasoning_content']
            calls = message.get('tool_calls')
            if calls is not None:
                if type(calls) is not list or not calls:
                    raise ApiError('tool_calls must be a non-empty array')
                rendered['tool_calls'] = []
                for call in calls:
                    if type(call) is not dict:
                        raise ApiError('each tool call must be an object')
                    function = call.get('function')
                    if type(function) is not dict or type(function.get('name')) is not str:
                        raise ApiError('each tool call needs function.name')
                    rendered['tool_calls'].append(dict(
                        id=call.get('id') or '', type='function',
                        function=dict(name=function['name'],
                                      arguments=_arguments(function.get('arguments', '{}')))))
        elif role == 'tool':
            if type(message.get('tool_call_id')) is not str or not message['tool_call_id']:
                raise ApiError('tool messages need tool_call_id')
            rendered['tool_call_id'] = message['tool_call_id']
        if role == 'assistant' and not rendered['content'] and 'tool_calls' not in rendered:
            raise ApiError('assistant messages need content or tool_calls')
        if role in ('system', 'user') and not rendered['content'].strip():
            raise ApiError(role + ' messages need text content')
        out.append(rendered)
    return out


def check_tools(tools):
    if tools is None:
        return None
    if type(tools) is not list or not tools:
        raise ApiError('tools must be a non-empty array when present')
    if len(json.dumps(tools).encode()) > TOOLS_CAP:
        raise ApiError('tool definitions exceed the supported size')
    for tool in tools:
        if type(tool) is not dict or tool.get('type') not in (None, 'function'):
            raise ApiError('only function tools are supported')
        function = tool.get('function', tool)
        if type(function) is not dict or type(function.get('name')) is not str:
            raise ApiError('each tool needs function.name')
        if function.get('parameters') is not None and type(function['parameters']) is not dict:
            raise ApiError('tool parameters must be a JSON schema object')
    return tools


def instruct(messages, tool_choice, tools):
    """The pinned template has no tool_choice; state the constraint in-band instead."""
    if tool_choice in (None, 'auto') or not tools:
        return messages
    if tool_choice == 'none':
        note = 'Do not call any function for this turn. Answer directly in plain text.'
    elif tool_choice == 'required':
        note = 'You must answer this turn by calling one of the provided functions.'
    elif type(tool_choice) is dict and type(tool_choice.get('function')) is dict:
        name = tool_choice['function'].get('name')
        if type(name) is not str:
            raise ApiError('tool_choice.function.name must be a string')
        note = 'You must answer this turn by calling the function named ' + name + '.'
    else:
        raise ApiError('tool_choice must be auto, none, required or a function selector')
    messages = list(messages)
    first = messages[0]
    if first['role'] == 'system':
        messages[0] = dict(first, content=first['content'].rstrip() + '\n\n' + note)
    else:
        messages.insert(0, dict(role='system', content=note))
    return messages


def parse(answer):
    """Split GLM's final channel into visible content and OpenAI tool calls."""
    calls = []
    for index, block in enumerate(CALL.findall(answer)):
        name = block.split('<arg_key>', 1)[0].strip()
        if not name:
            continue
        arguments = {}
        for key, raw in ARGUMENT.findall(block):
            value = raw.strip()
            try:
                arguments[key.strip()] = json.loads(value)
            except ValueError:
                arguments[key.strip()] = value
        calls.append(dict(id='call_' + uuid.uuid4().hex[:24], type='function', index=index,
                          function=dict(name=name, arguments=json.dumps(arguments, ensure_ascii=False))))
    content = CALL.sub('', answer).strip()
    return content, calls


def finish(job, calls):
    if calls:
        return 'tool_calls'
    if job['status'] == 'complete':
        return 'stop'
    return 'length' if job.get('stop_cause') in ('context_exhausted', 'output_cap') else 'stop'


def usage(job):
    return dict(prompt_tokens=job['prompt_tokens'], completion_tokens=job['output_tokens'],
                total_tokens=job['prompt_tokens'] + job['output_tokens'])


class Api:
    """Stateless request shaping in front of the shared sequential resident queue."""

    def __init__(self, chats, backend, *, capacity, wait_seconds=1800):
        self.chats = chats
        self.backend = backend
        self.capacity = capacity
        self.wait_seconds = wait_seconds

    def models(self):
        # Report the loaded session's real window so a client sizes its own
        # compaction correctly instead of assuming a default.
        # max_input_tokens is what a client must compact against: this profile's
        # prompt ceiling can be lower than its total capacity.
        return dict(object='list', data=[dict(
            id=MODEL_ID, object='model', owned_by='local', created=0,
            context_window=self.capacity,
            max_input_tokens=min(PROMPT_LIMITS[self.capacity], self.capacity - 1),
            max_output_tokens=min(MAX_NEW, self.capacity - 1),
            supports=dict(tools=True, streaming=True, reasoning_effort=list(EFFORTS),
                          parallel_requests=False, sampling=False))])

    def prepare(self, data):
        if type(data) is not dict:
            raise ApiError('expected a JSON object')
        if data.get('model') not in (None, MODEL_ID):
            raise ApiError('unknown model: ' + str(data.get('model')), status=404,
                           kind='model_not_found')
        if data.get('n') not in (None, 1):
            raise ApiError('only one choice per request is supported')
        effort = data.get('reasoning_effort', 'max')
        if effort not in EFFORTS:
            raise ApiError('reasoning_effort must be low, high or max')
        budget = data.get('max_tokens')
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise ApiError('max_tokens must be a positive integer')
        tools = check_tools(data.get('tools'))
        messages = instruct(convert(data.get('messages')), data.get('tool_choice'), tools)
        key = uuid.uuid4().hex
        payload = self.backend.prepare_api(messages, key, tools=tools, effort=effort,
                                           max_new_tokens=budget,
                                           context_capacity=self.capacity)
        return payload, key

    def wait(self, key, *, deadline):
        while True:
            job = self.chats.job(key)
            if job is None:
                raise ApiError('request was not retained', status=500, kind='server_error')
            if job['status'] in ('complete', 'incomplete'):
                return job
            if job.get('error'):
                raise ApiError(job['error'], status=503, kind='server_error')
            if time.time() > deadline:
                raise ApiError('the resident model did not finish within the server deadline; '
                               'use stream=true for long answers', status=504, kind='timeout')
            time.sleep(0.2)

    def completion(self, data):
        payload, key = self.prepare(data)
        self.chats.submit(payload, key, label='api')
        job = self.wait(key, deadline=time.time() + self.wait_seconds)
        content, calls = parse(job['answer'])
        message = dict(role='assistant', content=content or None)
        if job['thinking']:
            message['reasoning_content'] = job['thinking']
        if calls:
            message['tool_calls'] = [dict(id=c['id'], type='function', function=c['function'])
                                     for c in calls]
        return dict(id='chatcmpl-' + key, object='chat.completion', created=int(job['created']),
                    model=MODEL_ID, choices=[dict(index=0, message=message, logprobs=None,
                                                  finish_reason=finish(job, calls))],
                    usage=usage(job))

    def stream(self, data):
        """Yield SSE chunks. Tool calls are emitted once the final channel is known."""
        payload, key = self.prepare(data)
        self.chats.submit(payload, key, label='api')
        identity = dict(id='chatcmpl-' + key, object='chat.completion.chunk', model=MODEL_ID)
        created = int(time.time())
        yield self.chunk(dict(identity, created=created),
                         dict(role='assistant', content=''), None)
        deadline = time.time() + self.wait_seconds
        sent_thinking = sent_answer = 0
        while True:
            job = self.chats.job(key)
            if job is None:
                raise ApiError('request was not retained', status=500, kind='server_error')
            if job.get('error'):
                raise ApiError(job['error'], status=503, kind='server_error')
            terminal = job['status'] in ('complete', 'incomplete')
            thinking, answer = job['thinking'], job['answer']
            if len(thinking) > sent_thinking:
                yield self.chunk(dict(identity, created=created),
                                 dict(reasoning_content=thinking[sent_thinking:]), None)
                sent_thinking = len(thinking)
            # Hold back text until the end when a tool call may still be forming.
            visible = answer if terminal else answer.split('<tool_call>', 1)[0]
            if not terminal and len(visible) > sent_answer:
                yield self.chunk(dict(identity, created=created),
                                 dict(content=visible[sent_answer:]), None)
                sent_answer = len(visible)
            if terminal:
                break
            if time.time() > deadline:
                raise ApiError('the resident model did not finish within the server deadline',
                               status=504, kind='timeout')
            time.sleep(0.2)
        content, calls = parse(job['answer'])
        if len(content) > sent_answer:
            yield self.chunk(dict(identity, created=created),
                             dict(content=content[sent_answer:]), None)
        for call in calls:
            yield self.chunk(dict(identity, created=created), dict(tool_calls=[dict(
                index=call['index'], id=call['id'], type='function',
                function=dict(name=call['function']['name'],
                              arguments=call['function']['arguments']))]), None)
        yield self.chunk(dict(identity, created=created), {}, finish(job, calls),
                         extra=dict(usage=usage(job)))
        yield b'data: [DONE]\n\n'

    @staticmethod
    def chunk(identity, delta, finish_reason, *, extra=None):
        body = dict(identity, choices=[dict(index=0, delta=delta, logprobs=None,
                                            finish_reason=finish_reason)])
        if extra:
            body.update(extra)
        return b'data: ' + json.dumps(body, ensure_ascii=False).encode() + b'\n\n'
