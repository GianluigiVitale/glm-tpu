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
