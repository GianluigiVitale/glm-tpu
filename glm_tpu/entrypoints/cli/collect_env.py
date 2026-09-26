"""The release environment report: installed versions against the declared requirements."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import metadata, resources
import json
import sys

from glm_tpu.entrypoints.cli.types import CLISubcommand


def environment_manifest() -> dict:
    return json.loads(resources.files("glm_tpu").joinpath("environment.json").read_text())


def environment_report(
    profile: str,
    *,
    version: Callable[[str], str] = metadata.version,
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
    expected = {name: pin for group in groups[profile] for name, pin in manifest["profiles"][group].items()}
    rows = []
    for name, pin in sorted(expected.items()):
        try:
            actual = version(name)
        except metadata.PackageNotFoundError:
            actual = None
        rows.append(
            dict(
                package=name,
                expected=pin,
                installed=actual,
                status=("match" if actual == pin else "missing" if actual is None else "mismatch"),
            )
        )
    major, minor = python_version if python_version is not None else sys.version_info[:2]
    python = f"{major}.{minor}"
    return dict(
        schema="glm_tpu_environment_report_v1",
        profile=profile,
        python=python,
        expected_python=manifest["python"],
        packages=rows,
        passed=python == manifest["python"] and all(row["status"] == "match" for row in rows),
        scope=(
            "distribution metadata only; no dependency solving, payload hashes, model imports, TPU access or "
            "launch authorization"
        ),
    )


class CollectEnvSubcommand(CLISubcommand):
    """``glm-tpu doctor``: the environment report of one profile; exit 0 only when it passed."""

    name = "doctor"

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        report = environment_report(args.profile)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        doctor = subparsers.add_parser("doctor", help="check installed version metadata without initializing TPU")
        doctor.add_argument(
            "--profile",
            choices=("core", "runtime", "tpu", "benchmark", "dev"),
            default="core",
        )
        return doctor
