#!/usr/bin/env -S /usr/bin/python3 -I -S -B
"""Isolated offline adjudication of successful projection-contraction HLO."""

from __future__ import annotations

import _frozen_importlib
import _frozen_importlib_external
import importlib.util
import json
import os
import stat
import sys
import types
from hashlib import sha256
from pathlib import Path
from types import ModuleType

ROOT = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
SOURCES = {
    "glm_tpu.greenfield.errors": (
        ROOT / "glm_tpu/greenfield/errors.py",
        "f7b8a079ba16f6114714124598d03ae871a2c21d365de4417b934f00cddaae64",
    ),
    "glm_tpu.greenfield.sharding.hlo_contract": (
        ROOT / "glm_tpu/greenfield/sharding/hlo_contract.py",
        "86f08f7e06c96c291ccbd9883031773cf30202ad6ac9591122d5395e479ef7ac",
    ),
    "glm_tpu.greenfield.validation.gate_d_projection_contraction_hlo": (
        ROOT / "glm_tpu/greenfield/validation/gate_d_projection_contraction_hlo.py",
        "5a664dddf454f572fd2db95c6628f6cb37ced37bcd97035d5c94feff72ae47e8",
    ),
    "glm_tpu.greenfield.validation.gate_d_projection_contraction_hlo_success": (
        ROOT
        / "glm_tpu/greenfield/validation/gate_d_projection_contraction_hlo_success.py",
        "cd410a5e7ac3e8e895d45d9eb85e33f4a240da238e38b201298eebb4388fdba3",
    ),
}
PACKAGES = {
    "glm_tpu": ROOT / "glm_tpu",
    "glm_tpu.greenfield": ROOT / "glm_tpu/greenfield",
    "glm_tpu.greenfield.sharding": ROOT / "glm_tpu/greenfield/sharding",
    "glm_tpu.greenfield.validation": ROOT / "glm_tpu/greenfield/validation",
}
FORBIDDEN_MODULE_PREFIXES = (
    "jax",
    "jaxlib",
    "jax_plugins",
    "numpy",
    "tensorflow",
    "torch",
)
FORBIDDEN_DEVICE_PREFIXES = ("/dev/accel", "/dev/tpu", "/dev/vfio")


def _require_isolated_stdlib() -> None:
    flags = sys.flags
    if not (
        flags.isolated == 1
        and flags.no_site == 1
        and flags.ignore_environment == 1
        and flags.no_user_site == 1
        and flags.dont_write_bytecode == 1
        and getattr(flags, "safe_path", flags.isolated == 1)
    ):
        raise RuntimeError("successful HLO adjudicator requires python -I -S -B")
    expected = (
        _frozen_importlib.BuiltinImporter,
        _frozen_importlib.FrozenImporter,
        _frozen_importlib_external.PathFinder,
    )
    if tuple(sys.meta_path) != expected:
        raise RuntimeError("successful HLO adjudicator rejects import hooks")


def _read_source(path: Path, expected_sha256: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe adjudicator source: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
        if (
            len(raw) != before.st_size
            or (before.st_dev, before.st_ino, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_mtime_ns)
            or sha256(raw).hexdigest() != expected_sha256
        ):
            raise RuntimeError(f"adjudicator source drifted: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _forbidden_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_MODULE_PREFIXES
        )
    )


def _accelerator_fds() -> list[str]:
    observed: list[str] = []
    for child in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(child)
        except FileNotFoundError:
            continue
        if target.startswith(FORBIDDEN_DEVICE_PREFIXES):
            observed.append(target)
    return sorted(observed)


def _install_synthetic_packages() -> None:
    for name, path in PACKAGES.items():
        package = types.ModuleType(name)
        package.__package__ = name
        package.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = package


def _load_exact_module(name: str, path: Path, raw: bytes) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None:
        raise RuntimeError(f"cannot create exact module specification: {name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    code = compile(raw, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)  # noqa: S102 - authenticated source bytes only
    return module


def main() -> int:
    if sys.argv[1:]:
        raise RuntimeError("successful HLO adjudicator accepts no arguments")
    _require_isolated_stdlib()
    if _forbidden_modules() or _accelerator_fds():
        raise RuntimeError("successful HLO adjudicator started with accelerator state")
    snapshots = {
        name: _read_source(path, expected) for name, (path, expected) in SOURCES.items()
    }
    _install_synthetic_packages()
    for name in SOURCES:
        if name.endswith("gate_d_projection_contraction_hlo_success"):
            continue
        _load_exact_module(name, SOURCES[name][0], snapshots[name])
    validator_name = (
        "glm_tpu.greenfield.validation.gate_d_projection_contraction_hlo_success"
    )
    validator = _load_exact_module(
        validator_name,
        SOURCES[validator_name][0],
        snapshots[validator_name],
    )
    if _forbidden_modules() or _accelerator_fds():
        raise RuntimeError("successful HLO adjudicator initialized accelerator state")
    report = validator.adjudicate_success_run()
    report["process_contract"] = {
        "accelerator_fds_after": [],
        "accelerator_fds_before": [],
        "forbidden_modules_after": [],
        "forbidden_modules_before": [],
        "package_initializers_executed": [],
        "python_flags": ["-I", "-S", "-B"],
    }
    sys.stdout.write(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
