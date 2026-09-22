"""Golden-gate glue: markers, the recorded-version skip and the CPU budget rule.

Golden data are bound to the jax/jaxlib versions recorded in ``tests/golden/data/*.json``; on
any other version every golden test skips loudly with the reason. While a TPU run is live on
this host, the heavy (``cpu32``) goldens skip instead of competing with the run (D25).
"""
from __future__ import annotations

import json
from typing import Any, Callable

import pytest

from tools.equivalence.budget import REFUSAL, live_tpu_run
from tools.equivalence.common import DATA, read_json, version_mismatch


def pytest_configure(config: pytest.Config) -> None:
    for line in ("golden: compares against tests/golden/data (bound to the recorded jax/jaxlib versions)",
                 "golden_data(name): the data file a golden test compares against",
                 "cpu32: runs a child process on 32 forced CPU devices",
                 "slow: takes more than ~30 s on a 4-core host"):
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live = None
    for item in items:
        data = item.get_closest_marker("golden_data")
        if data is not None:
            path = DATA / data.args[0]
            if not path.exists():
                item.add_marker(pytest.mark.skip(reason=f"no baseline {data.args[0]}"))
            else:
                reason = version_mismatch(read_json(path).get("environment") or {})
                if reason:
                    item.add_marker(pytest.mark.skip(reason=reason))
        if item.get_closest_marker("cpu32") is not None:
            live = live_tpu_run() if live is None else live
            if live:
                item.add_marker(pytest.mark.skip(reason=REFUSAL))


@pytest.fixture
def gate_check() -> Callable[[str], dict[str, Any]]:
    """Run one gate against its baseline; fail with the compact report."""
    from tools.equivalence import gates

    def run(gate: str) -> dict[str, Any]:
        report = gates.check(gate)
        if report["status"] == "skip":
            pytest.skip(report.get("reason", "skipped"))
        assert report["status"] == "pass", json.dumps(report, indent=1, sort_keys=True)[:6000]
        return report

    return run
