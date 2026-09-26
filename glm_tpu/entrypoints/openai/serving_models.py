"""The ``/v1/models`` listing of the OpenAI-compatible surface.

One resident model is listed under its id and its reasoning-effort aliases, each with the loaded session's
window, its prompt and output ceilings and the server's request deadline, so a client sizes its requests
against the real session.
"""

from glm_tpu.engine.request import PROMPT_LIMITS, MAX_NEW
from glm_tpu.entrypoints.openai.protocol import ALIASES, EFFORTS


class OpenAIServingModels:
    """The model list of one loaded resident session (``capacity``) served with ``wait_seconds`` per request."""

    def __init__(self, *, capacity, wait_seconds):
        self.capacity = capacity
        self.wait_seconds = wait_seconds

    def models(self):
        # Report the loaded session's real window so a client sizes its own
        # compaction correctly instead of assuming a default.
        # max_input_tokens is what a client must compact against: this profile's
        # prompt ceiling can be lower than its total capacity.
        return dict(
            object="list",
            data=[
                dict(
                    id=name,
                    object="model",
                    owned_by="local",
                    created=0,
                    context_window=self.capacity,
                    max_input_tokens=min(PROMPT_LIMITS[self.capacity], self.capacity - 1),
                    max_output_tokens=min(MAX_NEW, self.capacity - 1),
                    # A request is also bounded by this server deadline; a client should
                    # size its own output expectation against observed throughput.
                    request_deadline_seconds=self.wait_seconds,
                    reasoning_effort=forced or "max",
                    supports=dict(
                        tools=True,
                        streaming=True,
                        reasoning_effort=list(EFFORTS),
                        parallel_requests=False,
                        sampling=False,
                    ),
                )
                for name, forced in ALIASES.items()
            ],
        )
