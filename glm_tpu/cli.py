"""Offline release/environment information. Never imports JAX or opens devices."""
from __future__ import annotations

import argparse
from importlib import metadata, resources
import json
import sys
from typing import Callable, Sequence


def environment_manifest() -> dict:
    return json.loads(resources.files("glm_tpu").joinpath("environment.json").read_text())


def environment_report(
    profile: str, *, version: Callable[[str], str] = metadata.version,
    python_version: tuple[int, int] | None = None,
) -> dict:
    """Compare distribution metadata only; a match is NOT hardware admission."""
    manifest = environment_manifest()
    groups = {
        "core": ("core",),
        "runtime": ("core", "runtime"),
        "tpu": ("core", "runtime", "tpu"),
        "benchmark": ("core", "runtime", "tpu", "benchmark"),
        "dev": ("core", "dev"),
    }
    if profile not in groups:
        raise ValueError("unknown environment profile")
    expected = {name: pin for group in groups[profile]
                for name, pin in manifest["profiles"][group].items()}
    rows = []
    for name, pin in sorted(expected.items()):
        try:
            actual = version(name)
        except metadata.PackageNotFoundError:
            actual = None
        rows.append(dict(package=name, expected=pin, installed=actual,
                         status="match" if actual == pin else "missing" if actual is None else "mismatch"))
    major, minor = python_version if python_version is not None else sys.version_info[:2]
    python = f"{major}.{minor}"
    return dict(
        schema="glm_tpu_environment_report_v1", profile=profile,
        python=python, expected_python=manifest["python"], packages=rows,
        passed=python == manifest["python"] and all(row["status"] == "match" for row in rows),
        scope="distribution metadata only; no dependency solving, payload hashes, model imports, TPU access or launch authorization",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show supported scope and release limitations")
    doctor = sub.add_parser("doctor", help="check installed version metadata without initializing TPU")
    doctor.add_argument("--profile", choices=("core", "runtime", "tpu", "benchmark", "dev"), default="core")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = environment_report(args.profile)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    try:
        version = metadata.version("glm-tpu")
    except metadata.PackageNotFoundError:
        version = "source checkout (not installed)"
    print(json.dumps(dict(
        project="glm-tpu", version=version, release_status="alpha; release checks incomplete",
        engine="native JAX WS32_2D", hardware="8 hosts / 32 TPU v4 chips",
        concurrent_requests=1, resume="same live session only",
        serving="site-specific protected request harness; no supported HTTP endpoint",
        installation="wheel contains Python components only; full deployment requires source checkout and external assets",
        quality="full benchmark/model-card parity not established",
    ), indent=2, sort_keys=True))
    return 0
