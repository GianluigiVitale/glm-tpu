"""The release environment report: installed versions against the declared requirements."""

from __future__ import annotations

from importlib import metadata, resources
import json
import sys
from typing import Callable


def environment_manifest() -> dict:
    return json.loads(
        resources.files("glm_tpu").joinpath("environment.json").read_text()
    )


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
    expected = {
        name: pin
        for group in groups[profile]
        for name, pin in manifest["profiles"][group].items()
    }
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
                status=(
                    "match"
                    if actual == pin
                    else "missing" if actual is None else "mismatch"
                ),
            )
        )
    major, minor = (
        python_version if python_version is not None else sys.version_info[:2]
    )
    python = f"{major}.{minor}"
    return dict(
        schema="glm_tpu_environment_report_v1",
        profile=profile,
        python=python,
        expected_python=manifest["python"],
        packages=rows,
        passed=python == manifest["python"]
        and all(row["status"] == "match" for row in rows),
        scope="distribution metadata only; no dependency solving, payload hashes, model imports, TPU access or launch authorization",
    )
