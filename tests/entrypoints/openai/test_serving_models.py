"""Tests of :mod:`glm_tpu.entrypoints.openai.serving_models`: the ``/v1/models`` listing (CPU only)."""

import pytest

from glm_tpu.entrypoints.openai.serving_models import OpenAIServingModels


def _row(name, effort):
    """One listed model of a 32K session with a 60 s request deadline."""
    return dict(
        id=name,
        object="model",
        owned_by="local",
        created=0,
        context_window=32768,
        max_input_tokens=32767,
        max_output_tokens=32767,
        request_deadline_seconds=60,
        reasoning_effort=effort,
        supports=dict(
            tools=True,
            streaming=True,
            reasoning_effort=["low", "high", "max"],
            parallel_requests=False,
            sampling=False,
        ),
    )


def test_the_listing_names_the_model_and_its_effort_aliases():
    # The G9-http record's /v1/models body is this listing (32K, 60 s), as JSON.
    assert OpenAIServingModels(capacity=32768, wait_seconds=60).models() == dict(
        object="list",
        data=[_row("glm-5.3", "max"), _row("glm-5.3-low", "low"), _row("glm-5.3-high", "high")],
    )


@pytest.mark.parametrize(
    ("capacity", "max_input", "max_output"),
    [
        (8192, 8191, 8191),
        (32768, 32767, 32767),
        # The 128K-input profile: its prompt ceiling is below its window, its output cap is the global one.
        (166912, 131072, 163840),
        (262144, 262143, 163840),
    ],
)
def test_the_limits_follow_the_loaded_profile(capacity, max_input, max_output):
    listed = OpenAIServingModels(capacity=capacity, wait_seconds=1800).models()["data"]
    assert {row["context_window"] for row in listed} == {capacity}
    assert {row["max_input_tokens"] for row in listed} == {max_input}
    assert {row["max_output_tokens"] for row in listed} == {max_output}
    assert {row["request_deadline_seconds"] for row in listed} == {1800}
