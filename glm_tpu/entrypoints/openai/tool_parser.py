"""Generated text: GLM ``<tool_call>`` parsing and the split of reasoning from the final answer."""

import json
import re
import uuid


# GLM emits <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
CALL = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
ARGUMENT = re.compile(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', re.S)


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


def final_channel(text):
    if '</think>' not in text:
        return text.removeprefix('<think>').strip(), ''
    thinking, answer = text.split('</think>', 1)
    answer = re.split(r'<\|(?:user|endoftext|observation)\|>', answer, maxsplit=1)[0]
    return thinking.removeprefix('<think>').strip(), answer.strip()
