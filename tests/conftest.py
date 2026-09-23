"""Test-session guard: never open the local TPU from pytest.

This controller VM is pod worker 0.  JAX defaults to the TPU backend here, so a
jitting test run without ``JAX_PLATFORMS=cpu`` would open the TPU outside the
fleet lease (HANDOFF 2026-08: a review pytest did exactly that).  Protected TPU
work never runs through pytest; every test runs on the CPU platform.  Tests that
spawn subprocesses set their own environment and inherit this default.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError(
        "pytest must run with JAX_PLATFORMS=cpu on the pod controller; "
        f"got {os.environ.get('JAX_PLATFORMS')!r}"
    )

# Never read the operator's site file from a test: the default site location points at a
# directory that does not exist, and the controller-side overrides are cleared. Tests build
# example sites (tests/fixtures/site.py); site-bound tests name a real one with GLM_TPU_TEST_SITE.
os.environ["GLM_TPU_CONFIG_ROOT"] = "/nonexistent/glm-tpu-test-config"
for _name in ("GLM_TPU_SITE_CONFIG", "GLM_TPU_RUN_ROOT", "GLM_TPU_MODEL_PATH", "GLM_TPU_HLO_DUMP_ROOT"):
    os.environ.pop(_name, None)


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "site: needs the operator's real site file (GLM_TPU_TEST_SITE) on a fleet "
                                       "host; skips without it (excluded from G10/G11)")
