"""The serving layer's test doubles: a synthetic resident controller and the browser workspace composed over it.

``FakeResident`` answers the job queue as the resident controller does (``prepare``, ``next_sequence``, ``publish``,
``observe``), without a tokenizer, checkpoint, cloud or TPU. ``open_store`` composes the workspace as the server
does: the job queue first, then the conversation store over it, which the queue hands each conversation job's
progress to (``JobQueue.step``); a queue stepped without its store cannot finish a conversation job.
"""

from glm_tpu.entrypoints.serve.job_queue import JobQueue
from glm_tpu.entrypoints.ui.conversations import ConversationStore


class FakeResident:
    def __init__(self, capacity=32768):
        self.capacity = capacity
        self.messages = []
        self.published = {}
        self.sequence = 770
        self.complete = False

    def check(self):
        pass

    def prepare(self, messages, key):
        if len(messages[-1]["content"]) > 100:
            raise ValueError("context capacity exceeded")
        self.messages.append(messages)
        return dict(prompt_ids=[1, 2], max_new_tokens=32766, request_id=key)

    def next_sequence(self):
        return self.sequence + 1

    def publish(self, job, root):
        old = self.published.setdefault(job["sequence"], job["payload"])
        assert old == job["payload"]

    def observe(self, job):
        self.sequence = job["sequence"]
        return dict(
            status="complete" if self.complete else "generating",
            answer="42" if self.complete else "",
            thinking="working",
            output_tokens=3,
            decode_tps=13.5,
            stop_cause="eos",
        )


def open_store(root, backend):
    """The browser workspace as the server composes it: a conversation store over the job queue."""
    return ConversationStore(JobQueue(root, backend))


def chat(store):
    return store.change(dict(action="create"))["id"]


def send(store, c, text="question", key="a" * 32):
    return store.change(dict(action="send", chat=c, text=text, id=key))
