"""Markers and the CPU budget rule for the reference-model tests.

``cpu32`` tests run a child process on 32 forced CPU devices; like the golden
gates they skip while a TPU run is live on this host (DESIGN 7.5.10, D25).
"""

from __future__ import annotations

import pytest

from tools.equivalence.budget import REFUSAL, live_tpu_run


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "cpu32: runs a child process on 32 forced CPU devices",
        "slow: takes more than ~30 s on a 4-core host",
    ):
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    live = None
    for item in items:
        if (
            item.get_closest_marker("cpu32") is not None
            and "tests/reference/" in item.nodeid
        ):
            live = live_tpu_run() if live is None else live
            if live:
                item.add_marker(pytest.mark.skip(reason=REFUSAL))
