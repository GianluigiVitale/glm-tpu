"""CPU-only tests of the OpenAI-compatible surface; no tokenizer, weights or TPU."""

import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from glm_tpu.entrypoints.openai.serving_chat import Api
from glm_tpu.entrypoints.openai.protocol import ApiError
from glm_tpu.entrypoints.openai.chat_utils import convert, check_tools, instruct
from glm_tpu.entrypoints.openai.tool_parser import parse
from glm_tpu.entrypoints.serve.job_queue import Chats
from glm_tpu.entrypoints.serve.server import ThreadingHTTPServer
from glm_tpu.entrypoints.serve.http_handler import handler


TOOLS = [
    dict(
        type="function",
        function=dict(
            name="read_file",
            description="Read a file",
            parameters=dict(type="object", properties=dict(path=dict(type="string")), required=["path"]),
        ),
    )
]


class FakeResident:
    """Records what the template would receive and returns the pinned request shape."""

    def __init__(self, answer="42"):
        self.prepared = []
        self.published = {}
        self.sequence = 770
        self.answer = answer
        self.thinking = "weighing it up"
        self.complete = True

    def check(self):
        pass

    def prepare_api(self, messages, key, *, tools, effort, max_new_tokens, context_capacity):
        ids = [1] * (len(json.dumps(messages)) // 10 + 1)
        remaining = context_capacity - len(ids)
        if remaining <= 0:
            raise ApiError("this conversation fills the context")
        budget = remaining if max_new_tokens is None else min(max_new_tokens, remaining)
        self.prepared.append(dict(messages=messages, tools=tools, effort=effort, budget=budget))
        return dict(prompt_ids=ids, max_new_tokens=budget, request_id="api-" + key, context_capacity=context_capacity)

    def next_sequence(self):
        return self.sequence + 1

    def publish(self, job, root):
        self.published.setdefault(job["sequence"], job["payload"])

    def observe(self, job):
        self.sequence = job["sequence"]
        return dict(
            status="complete" if self.complete else "generating",
            answer=self.answer,
            thinking=self.thinking,
            output_tokens=7,
            decode_tps=13.5,
            prefill_seconds=1.0,
            stop_cause="eos",
        )


def service(tmp_path, backend=None):
    backend = backend or FakeResident()
    store = Chats(tmp_path, backend)
    return Api(store, backend, capacity=32768), store, backend


def pump(store):
    """Run the shared sequential worker for the duration of one test."""
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            store.step()
            time.sleep(0.01)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop


def chat(**extra):
    body = dict(model="glm-5.3", messages=[dict(role="user", content="hello")])
    body.update(extra)
    return body


def test_openai_tool_calls_round_trip_into_the_template_shape():
    messages = convert(
        [
            dict(role="system", content="be brief"),
            dict(role="user", content="read a.txt"),
            dict(
                role="assistant",
                content=None,
                tool_calls=[
                    dict(id="call_1", type="function", function=dict(name="read_file", arguments='{"path": "a.txt"}'))
                ],
            ),
            dict(role="tool", tool_call_id="call_1", content="hello world"),
        ]
    )
    # The pinned template iterates arguments.items(), so the JSON string must be decoded.
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == dict(path="a.txt")
    assert messages[3] == dict(role="tool", content="hello world", tool_call_id="call_1")
    assert messages[1]["content"] == "read a.txt"


def test_multipart_text_and_rejected_shapes():
    assert (
        convert([dict(role="user", content=[dict(type="text", text="a"), dict(type="text", text="b")])])[0]["content"]
        == "ab"
    )
    for bad in (
        [],
        [dict(role="nobody", content="x")],
        [dict(role="user", content="  ")],
        [dict(role="user", content=[dict(type="image_url", image_url={})])],
        [dict(role="tool", content="x")],
        [dict(role="assistant", content="")],
    ):
        with pytest.raises(ApiError):
            convert(bad)
    with pytest.raises(ApiError):
        convert(
            [dict(role="assistant", content=None, tool_calls=[dict(function=dict(name="f", arguments="not json"))])]
        )
    with pytest.raises(ApiError):
        check_tools([dict(type="function", function=dict(description="no name"))])


def test_glm_tool_call_output_becomes_openai_tool_calls():
    content, calls = parse(
        "I will look.<tool_call>read_file<arg_key>path</arg_key><arg_value>a.txt</arg_value>"
        "<arg_key>lines</arg_key><arg_value>12</arg_value></tool_call>"
    )
    assert content == "I will look."
    assert len(calls) == 1 and calls[0]["function"]["name"] == "read_file"
    # OpenAI clients parse arguments as a JSON string, with non-string values preserved.
    assert json.loads(calls[0]["function"]["arguments"]) == dict(path="a.txt", lines=12)
    assert calls[0]["id"].startswith("call_") and calls[0]["index"] == 0
    assert parse("just text") == ("just text", [])


def test_tool_choice_is_expressed_in_band():
    forced = instruct(convert([dict(role="user", content="hi")]), "required", TOOLS)
    assert forced[0]["role"] == "system" and "must answer this turn by calling" in forced[0]["content"]
    named = instruct(
        convert([dict(role="system", content="be brief"), dict(role="user", content="hi")]),
        dict(type="function", function=dict(name="read_file")),
        TOOLS,
    )
    assert named[0]["content"].startswith("be brief") and "read_file" in named[0]["content"]
    plain = convert([dict(role="user", content="hi")])
    assert instruct(plain, "auto", TOOLS) == plain and instruct(plain, "required", None) == plain
    with pytest.raises(ApiError):
        instruct(plain, "whatever", TOOLS)


def test_completion_is_stateless_and_reports_usage(tmp_path):
    api, store, backend = service(tmp_path)
    stop = pump(store)
    try:
        first = api.completion(chat(messages=[dict(role="user", content="one")]))
        second = api.completion(chat(messages=[dict(role="user", content="two")]))
    finally:
        stop.set()
    # No conversation is created and no history leaks between requests.
    assert store.db["chats"] == []
    assert [m[-1]["content"] for m in (p["messages"] for p in backend.prepared)] == ["one", "two"]
    assert all(len(p["messages"]) == 1 for p in backend.prepared)
    assert first["object"] == "chat.completion" and first["model"] == "glm-5.3"
    choice = first["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["message"]["content"] == "42"
    # Reasoning is never mixed into content.
    assert choice["message"]["reasoning_content"] == "weighing it up"
    assert first["usage"]["completion_tokens"] == 7
    assert first["usage"]["total_tokens"] == first["usage"]["prompt_tokens"] + 7
    assert first["id"] != second["id"]


def test_tool_call_answer_sets_tool_calls_finish_reason(tmp_path):
    backend = FakeResident("<tool_call>read_file<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>")
    api, store, _ = service(tmp_path, backend)
    stop = pump(store)
    try:
        result = api.completion(chat(tools=TOOLS, tool_choice="auto"))
    finally:
        stop.set()
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == dict(path="a.txt")
    assert backend.prepared[0]["tools"] == TOOLS


def test_stream_separates_reasoning_content_and_tool_calls(tmp_path):
    backend = FakeResident("done<tool_call>read_file<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>")
    api, store, _ = service(tmp_path, backend)
    stop = pump(store)
    try:
        raw = b"".join(api.stream(chat(tools=TOOLS, stream=True)))
    finally:
        stop.set()
    assert raw.endswith(b"data: [DONE]\n\n")
    events = [
        json.loads(line[6:]) for line in raw.splitlines() if line.startswith(b"data: ") and line != b"data: [DONE]"
    ]
    deltas = [e["choices"][0]["delta"] for e in events]
    assert deltas[0]["role"] == "assistant"
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "weighing it up"
    # The tool-call markup never reaches content.
    assert "<tool_call>" not in "".join(d.get("content", "") for d in deltas)
    assert "".join(d.get("content", "") for d in deltas) == "done"
    calls = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert calls[0]["index"] == 0 and calls[0]["function"]["name"] == "read_file"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1]["usage"]["completion_tokens"] == 7
    assert all(e["object"] == "chat.completion.chunk" for e in events)


def test_capacity_and_request_validation(tmp_path):
    api, store, backend = service(tmp_path)
    with pytest.raises(ApiError, match="fills the context"):
        api.prepare(chat(messages=[dict(role="user", content="x" * 400000)]))
    with pytest.raises(ApiError, match="reasoning_effort"):
        api.prepare(chat(reasoning_effort="medium"))
    with pytest.raises(ApiError, match="unknown model"):
        api.prepare(chat(model="gpt-4"))
    with pytest.raises(ApiError, match="max_tokens"):
        api.prepare(chat(max_tokens=0))
    payload, _, _ = api.prepare(chat(max_tokens=64))
    assert payload["max_new_tokens"] == 64
    payload, _, _ = api.prepare(chat())
    assert payload["max_new_tokens"] == 32768 - len(payload["prompt_ids"])
    assert backend.prepared[-1]["effort"] == "max"
    assert api.prepare(chat(reasoning_effort="low"))[0] and backend.prepared[-1]["effort"] == "low"


def test_api_traffic_stays_out_of_the_browser_workspace(tmp_path):
    api, store, _ = service(tmp_path)
    stop = pump(store)
    try:
        api.completion(chat(messages=[dict(role="user", content="private")]))
    finally:
        stop.set()
    snapshot = store.snapshot()
    assert snapshot["chats"] == [] and snapshot["jobs"] == []
    assert len(store.db["jobs"]) == 1 and store.db["jobs"][0]["api"]


def test_http_key_boundary_and_model_list(tmp_path):
    api, store, _ = service(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(store, api, "secret-key"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    stop = pump(store)
    try:
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "secret-key"}):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/v1/models", headers=headers))
            assert error.value.code == 401
        good = {"Authorization": "Bearer secret-key"}
        with urlopen(Request(base + "/v1/models", headers=good)) as response:
            listed = json.loads(response.read())["data"][0]
            assert listed["id"] == "glm-5.3"
            # A client must be able to read the real window rather than guess it.
            assert listed["context_window"] == 32768 and listed["max_output_tokens"] == 32767
            assert listed["max_input_tokens"] == 32767
            assert listed["request_deadline_seconds"] == 1800
            assert listed["supports"]["tools"] and not listed["supports"]["parallel_requests"]
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/v1/embeddings", headers=good))
        assert error.value.code == 404
        # The browser gate does not apply to /v1, and the API key does not open the UI.
        body = json.dumps(chat()).encode()
        with urlopen(Request(base + "/v1/chat/completions", data=body, headers=good)) as response:
            assert json.loads(response.read())["choices"][0]["message"]["content"] == "42"
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/chat", data=b'{"action":"create"}', headers=good))
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/v1/chat/completions", data=b'{"messages":[]}', headers=good))
        assert error.value.code == 400
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_backend_failure_reaches_the_waiting_client(tmp_path):
    backend = FakeResident()
    api, store, _ = service(tmp_path, backend)
    backend.observe = lambda _: (_ for _ in ()).throw(OSError("receipt unreadable"))
    stop = pump(store)
    try:
        with pytest.raises(ApiError, match="receipt unreadable") as error:
            api.completion(chat())
        assert error.value.status == 503
        # The queue stays paused rather than resubmitting under a fresh sequence.
        with pytest.raises(ApiError, match="receipt unreadable"):
            api.completion(chat())
    finally:
        stop.set()
    assert len(backend.published) == 1


def test_effort_alias_forces_cheap_side_calls(tmp_path):
    """A client that cannot send reasoning_effort selects it by model id."""
    api, store, backend = service(tmp_path)
    listed = {row["id"]: row for row in api.models()["data"]}
    assert set(listed) == {"glm-5.3", "glm-5.3-low", "glm-5.3-high"}
    assert listed["glm-5.3-low"]["reasoning_effort"] == "low"
    api.prepare(chat(model="glm-5.3-low"))
    assert backend.prepared[-1]["effort"] == "low"
    # The alias wins over a conflicting field so the cheap path stays cheap.
    api.prepare(chat(model="glm-5.3-low", reasoning_effort="max"))
    assert backend.prepared[-1]["effort"] == "low"
    api.prepare(chat(model="glm-5.3", reasoning_effort="high"))
    assert backend.prepared[-1]["effort"] == "high"
    stop = pump(store)
    try:
        assert api.completion(chat(model="glm-5.3-low"))["model"] == "glm-5.3-low"
    finally:
        stop.set()
    with pytest.raises(ApiError, match="unknown model"):
        api.prepare(chat(model="glm-5.3-medium"))
