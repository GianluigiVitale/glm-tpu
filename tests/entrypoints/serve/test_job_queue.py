"""Tests of :mod:`glm_tpu.entrypoints.serve.job_queue`: the persistent queue of chat jobs, run one at a time.

CPU only, over the synthetic resident of ``tests/fixtures/serving.py``; no tokenizer, checkpoint, cloud or TPU.
"""

import json

import pytest

from tests.fixtures.serving import FakeResident, chat, open_store, send


def test_idempotence_and_restart_preserve_admitted_identity(tmp_path):
    backend = FakeResident()
    store = open_store(tmp_path, backend)
    c = chat(store)
    send(store, c)
    send(store, c)
    assert len(store.queue.db["jobs"]) == 1
    with pytest.raises(ValueError, match="already used"):
        send(store, c, "different")
    store.queue.step()
    restarted = open_store(tmp_path, backend)
    backend.complete = True
    restarted.queue.step()
    assert len(backend.published) == 1
    assert restarted.snapshot()["jobs"][0]["status"] == "complete"
    assert restarted.snapshot()["jobs"][0]["sequence"] == 771


def test_crash_after_publish_reconciles_same_sequence(tmp_path):
    backend = FakeResident()
    store = open_store(tmp_path, backend)
    c = chat(store)
    send(store, c)
    original = backend.observe
    backend.observe = lambda _: (_ for _ in ()).throw(OSError("receipt unreadable"))
    store.queue.step()
    assert store.queue.error and len(backend.published) == 1
    assert json.loads((tmp_path / "chats.json").read_text())["jobs"][0]["sequence"] == 771
    backend.observe = original
    backend.complete = True
    restarted = open_store(tmp_path, backend)
    restarted.queue.step()
    assert len(backend.published) == 1


def test_queue_runs_one_at_a_time(tmp_path):
    backend = FakeResident()
    store = open_store(tmp_path, backend)
    a, b = chat(store), chat(store)
    send(store, a)
    send(store, b, key="b" * 32)
    store.queue.step()
    store.queue.step()
    assert list(backend.published) == [771]
    backend.complete = True
    store.queue.step()
    store.queue.step()
    assert list(backend.published) == [771, 772]


def test_a_finished_conversation_job_is_on_disk_in_its_message(tmp_path):
    # step() hands the job's answer and status to its conversation, then saves the one document: read back from
    # disk, the assistant message carries them.
    backend = FakeResident()
    store = open_store(tmp_path, backend)
    c = chat(store)
    send(store, c)
    backend.complete = True
    store.queue.step()
    saved = json.loads((tmp_path / "chats.json").read_text())
    [job] = saved["jobs"]
    assert (job["status"], job["answer"]) == ("complete", "42")
    [conversation] = saved["chats"]
    assert conversation["messages"][-1] == dict(role="assistant", content="42", status="complete", job=job["id"])
