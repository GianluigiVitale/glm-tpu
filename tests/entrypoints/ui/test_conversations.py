"""Tests of :mod:`glm_tpu.entrypoints.ui.conversations`: the saved conversations of the chat UI.

CPU only, with the synthetic resident of ``tests/fixtures/serving.py``; no tokenizer, checkpoint, cloud or
TPU.
"""

import pytest

from tests.fixtures.serving import FakeResident, chat, open_store, send


def test_history_isolation_and_full_allowance(tmp_path):
    backend = FakeResident()
    store = open_store(tmp_path, backend)
    a, b = chat(store), chat(store)
    send(store, a)
    store.queue.step()
    assert store.snapshot()["jobs"][0]["status"] == "generating"
    with pytest.raises(ValueError, match="completed answer"):
        send(store, a, key="b" * 32)
    backend.complete = True
    store.queue.step()
    send(store, a, "follow up", "b" * 32)
    assert backend.messages[-1] == [
        dict(role="user", content="question"),
        dict(role="assistant", content="42"),
        dict(role="user", content="follow up"),
    ]
    send(store, b, "separate", "c" * 32)
    assert backend.messages[-1] == [dict(role="user", content="separate")]
    assert all(j["payload"]["max_new_tokens"] == 32766 for j in store.queue.db["jobs"])
    assert all("payload" not in j for j in store.snapshot()["jobs"])


def test_overflow_and_delete_cannot_corrupt_active_chat(tmp_path):
    store = open_store(tmp_path, FakeResident())
    c = chat(store)
    with pytest.raises(ValueError, match="capacity"):
        send(store, c, "x" * 101)
    assert store.queue.db["chats"][0]["messages"] == []
    send(store, c)
    with pytest.raises(ValueError, match="finish"):
        store.change(dict(action="delete", chat=c))
    store.change(dict(action="rename", chat=c, title="Renamed"))
    assert store.queue.db["chats"][0]["title"] == "Renamed"
