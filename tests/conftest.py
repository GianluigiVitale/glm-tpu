"""Test-session guard: never open the local TPU from pytest.

This controller VM is pod worker 0.  JAX defaults to the TPU backend here, so a
jitting test run without ``JAX_PLATFORMS=cpu`` would open the TPU outside the
fleet lease (HANDOFF 2026-08: a review pytest did exactly that).  Protected TPU
work never runs through pytest; every test runs on the CPU platform.  Tests that
spawn subprocesses set their own environment and inherit this default.

The CPU budget rule (DESIGN 7.5.10, D25) for every test: a ``cpu32`` test runs a child process on 32 forced
CPU devices and is skipped while a TPU run is live on this host (``tools.equivalence.budget``), like the
heavy golden gates.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError(
        f"pytest must run with JAX_PLATFORMS=cpu on the pod controller; got {os.environ.get('JAX_PLATFORMS')!r}"
    )

# Never read the operator's site file from a test: the default site location points at a
# directory that does not exist, and the controller-side overrides are cleared. Tests build
# example sites (tests/fixtures/site.py); site-bound tests name a real one with GLM_TPU_TEST_SITE.
os.environ["GLM_TPU_CONFIG_ROOT"] = "/nonexistent/glm-tpu-test-config"
for _name in ("GLM_TPU_SITE_CONFIG", "GLM_TPU_RUN_ROOT", "GLM_TPU_MODEL_PATH", "GLM_TPU_HLO_DUMP_ROOT"):
    os.environ.pop(_name, None)


def pytest_configure(config) -> None:
    for line in (
        "site: needs the operator's real site file (GLM_TPU_TEST_SITE) on a fleet "
        "host; skips without it (excluded from G10/G11)",
        "cpu32: runs a child process on 32 forced CPU devices",
        "slow: takes more than ~30 s on a 4-core host",
    ):
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config, items) -> None:
    from tools.equivalence.budget import live_tpu_run, refusal_reason

    live = None
    for item in items:
        if item.get_closest_marker("cpu32") is not None:
            live = live_tpu_run() if live is None else live
            if live:
                item.add_marker(pytest.mark.skip(reason=refusal_reason()))
