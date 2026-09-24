"""Golden-gate glue: markers, the recorded-version rule and the CPU budget rule.

Golden data are bound to the package versions recorded in ``tests/golden/data/*.json`` (per gate:
``tools.equivalence.gates.BOUND_PACKAGES``). On any other version a golden test skips loudly with
the reason -- unless ``GLM_EQUIVALENCE_STRICT=1`` (CI, integrator), where a mismatch fails. A
missing data file is always a failure, never a skip. While a TPU run is live on this host, the
heavy (``cpu32``) goldens skip instead of competing with the run (D25).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import pytest

from tools.equivalence.budget import live_tpu_run, refusal_reason
from tools.equivalence.common import DATA, read_json, version_mismatch
from tools.equivalence.gates import BOUND_PACKAGES, DATA_FILES

STRICT = os.environ.get("GLM_EQUIVALENCE_STRICT") == "1"
_GATE_OF_FILE = {name: gate for gate, name in DATA_FILES.items()}


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "golden: compares against tests/golden/data (bound to the recorded package versions)",
        "golden_data(name): the data file a golden test compares against",
        "cpu32: runs a child process on 32 forced CPU devices",
        "slow: takes more than ~30 s on a 4-core host",
    ):
        config.addinivalue_line("markers", line)


def mismatch_reason(name: str) -> str | None:
    """The version-mismatch reason for one data file (None when bound versions match)."""
    gate = _GATE_OF_FILE.get(name, "default")
    names = BOUND_PACKAGES.get("G9" if gate == "G9-http" else gate, BOUND_PACKAGES["default"])
    return version_mismatch(read_json(DATA / name).get("environment") or {}, names)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live = None
    for item in items:
        data = item.get_closest_marker("golden_data")
        # A missing data file is not skipped: the test itself fails on it.
        if data is not None and (DATA / data.args[0]).is_file() and not STRICT:
            reason = mismatch_reason(data.args[0])
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason + " (GLM_EQUIVALENCE_STRICT=1 makes this fail)"))
        if item.get_closest_marker("cpu32") is not None:
            live = live_tpu_run() if live is None else live
            if live:
                item.add_marker(pytest.mark.skip(reason=refusal_reason()))


@pytest.fixture
def gate_check() -> Callable[[str], dict[str, Any]]:
    """Run one gate against its baseline; fail with the compact report (a skip fails in strict mode)."""
    from tools.equivalence import gates

    def run(gate: str) -> dict[str, Any]:
        report = gates.check(gate)
        if report["status"] == "skip" and not STRICT:
            pytest.skip(report.get("reason", "skipped"))
        assert report["status"] == "pass", json.dumps(report, indent=1, sort_keys=True)[:6000]
        return report

    return run
