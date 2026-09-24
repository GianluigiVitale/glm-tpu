"""OpenAI chat messages and tools, mapped onto what the pinned GLM-5.3 chat template renders."""

import json

from glm_tpu.entrypoints.openai.protocol import ApiError, MESSAGES_CAP, TOOLS_CAP


ROLES = ('system', 'user', 'assistant', 'tool')


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
