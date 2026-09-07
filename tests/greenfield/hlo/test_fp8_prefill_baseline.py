"""Admission tests only: never initialize the TPU from pytest."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.greenfield import fp8_baseline_guard as guard
from scripts.greenfield.microbench_fp8_matmul import (
    F32_OUTPUT_KERNELS,
    _compiled_memory,
    _validate_sampling_contract,
    _validate_shape_contract,
)


@pytest.mark.parametrize("rows", [8, 32, 128, 256])
def test_baseline_has_real_local_shape(rows):
    _validate_shape_contract(
        kernel="ws32_prefill_baseline", rows=rows, contraction=1536, output_width=2048
    )
    assert "ws32_prefill_baseline" in F32_OUTPUT_KERNELS


@pytest.mark.parametrize(
    "rows,k,n",
    [
        (1, 1536, 2048),
        (17, 1536, 2048),
        (512, 1536, 2048),
        (32, 6144, 2048),
        (32, 1536, 1536),
    ],
)
def test_baseline_rejects_unregistered_shape(rows, k, n):
    with pytest.raises(ValueError, match="WS32 baseline"):
        _validate_shape_contract(
            kernel="ws32_prefill_baseline", rows=rows, contraction=k, output_width=n
        )


def test_old_shape_contract_not_widened():
    _validate_shape_contract(
        kernel="single_up", rows=8, contraction=6144, output_width=2048
    )
    with pytest.raises(ValueError, match="production decode shape"):
        _validate_shape_contract(
            kernel="single_up", rows=32, contraction=1536, output_width=2048
        )
    _validate_sampling_contract(
        kernel="ws32_prefill_baseline",
        diagnostic_reference_timing=False,
        warmup=200,
        iterations=1000,
    )
    for warmup, count, reference in (
        (1, 3, False),
        (200, 1001, False),
        (200, 1000, True),
    ):
        with pytest.raises(ValueError, match="exactly 200/1000"):
            _validate_sampling_contract(
                kernel="ws32_prefill_baseline",
                diagnostic_reference_timing=reference,
                warmup=warmup,
                iterations=count,
            )


def test_compiled_memory_is_explicit_and_required():
    with pytest.raises(RuntimeError, match="unavailable"):
        _compiled_memory(SimpleNamespace(memory_analysis=lambda: None))
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    values = dict(zip(names, range(5)))
    assert (
        _compiled_memory(
            SimpleNamespace(memory_analysis=lambda: SimpleNamespace(**values))
        )
        == values
    )


def test_remote_guard_reuses_observer_and_is_source_only():
    argv = shlex.split(guard.census_command())
    assert argv[:4] == ["sudo", "-n", "/home/gianl/vllm-env/bin/python", "-c"]
    # Compile the deployed source, without executing any observation here.
    captured = {}
    exec(argv[4], {"exec": lambda value: captured.update(source=value)})
    source = captured["source"].decode()
    compile(source, "remote_guard", "exec")
    assert "Specified filename /tmp/libtpu_lockfile does not exist." in source
    assert "require_accelerators_idle()" in source
    assert "import jax" not in source


def test_fleet_requires_all_hosts_and_devices():
    records = [
        dict(host=f"pod-w-{i}", boot_id="boot", devices={str(j): [] for j in range(4)})
        for i in range(8)
    ]

    def encode(rows):
        return "\n".join("FP8_IDLE " + json.dumps(row) for row in rows)

    guard.validate_fleet(encode(records))
    for rows in (records[:-1], records + records[:1], records[:-1] + records[:1]):
        with pytest.raises(ValueError):
            guard.validate_fleet(encode(rows))


@pytest.mark.parametrize(
    "rc,out,err",
    [
        (0, "123", "/dev/accel0:"),
        (1, "", "permission denied"),
        (2, "", ""),
        (1, "123", ""),
    ],
)
def test_device_observation_fails_closed(monkeypatch, rc, out, err):
    monkeypatch.setattr(guard.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        guard, "accelerator_snapshot", lambda: {"/dev/accel0": (1, 2, 3, 4)}
    )
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=rc, stdout=out, stderr=err),
    )
    with pytest.raises(RuntimeError, match="busy or holder observation unknown"):
        guard.require_accelerators_idle()


def test_changed_device_is_not_idle(monkeypatch):
    monkeypatch.setattr(guard.os, "geteuid", lambda: 0)
    snapshots = iter(({"/dev/accel0": (1, 2, 3, 4)}, {"/dev/accel0": (1, 9, 3, 4)}))
    monkeypatch.setattr(guard, "accelerator_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        guard.require_accelerators_idle()


def test_wrapper_baseline_is_bounded_distinct_and_generation_verified():
    source = (
        Path(__file__).resolve().parents[3]
        / "scripts/greenfield/run_fp8_matmul_microbench.sh"
    ).read_text()
    assert "timeout --kill-after=30s 600s" in source
    assert source.index("strict_census post ||") < source.index("pv.start_run(")
    assert 'item_id = f"baseline_m{m}_k{k}_n{n}"' in source
    assert "publish_exact" in source and "archive_receipts_sha256" in source
    assert "GLM_GREENFIELD_FP8_MATMUL_KERNEL:-single_up" in source
