"""``glm-tpu collect-env`` (alias ``doctor``): installed versions against the requirements glm-tpu declares.

The declared requirements are read from the installed glm-tpu distribution's metadata, else from the
``pyproject.toml`` of the source checkout this module runs from. Each is one exact ``name==version`` pin of the core
dependencies or of one extra; a profile is the core dependencies plus the extras it installs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import metadata
import json
import operator
from pathlib import Path
import re
import sys
import tomllib

from glm_tpu.entrypoints.cli.types import CLISubcommand

# The source checkout's project file: glm_tpu/entrypoints/cli/collect_env.py -> the repository root.
PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"
# Each profile: the core dependencies and the extras it installs.
PROFILES = {
    "core": ("core",),
    "runtime": ("core", "runtime"),
    "tpu": ("core", "runtime", "tpu"),
    "dev": ("core", "dev"),
}
_PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([A-Za-z0-9][A-Za-z0-9.+!_-]*)")
_EXTRA = re.compile(r"""extra\s*==\s*(["'])([A-Za-z0-9._-]+)\1""")
_CLAUSE = re.compile(r"(>=|<=|==|!=|>|<)\s*([0-9]+(?:\.[0-9]+)*)")
_COMPARE = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def declared_requirements(
    *,
    distribution: Callable[[str], metadata.Distribution] = metadata.distribution,
    pyproject: Path = PYPROJECT,
) -> dict:
    """The requirements glm-tpu declares: its installed distribution's metadata, else the checkout's pyproject.toml.

    Returns ``declared_by`` (where they were read), ``python`` (the requires-python specifier) and ``groups``
    (``core`` and each extra: package -> pinned version). Refused with ``ValueError``: glm-tpu is not installed and
    ``pyproject`` does not exist or declares another project; a requirement that is not one exact ``name==version``
    pin, of the core dependencies or conditioned only on ``extra == "<name>"``; two pins of one package in one group;
    no requires-python.
    """
    try:
        installed = distribution("glm-tpu")
    except metadata.PackageNotFoundError:
        installed = None
    if installed is not None:
        declared_by = f"glm-tpu {installed.version} distribution metadata"
        python = installed.metadata.get("Requires-Python")
        rows = []
        for requirement in installed.requires or ():
            pin, _, marker = requirement.partition(";")
            extra = _EXTRA.fullmatch(marker.strip())
            if marker.strip() and extra is None:
                raise ValueError(f"unsupported requirement marker: {requirement}")
            rows.append(("core" if extra is None else extra.group(2), pin))
    else:
        if not pyproject.is_file():
            raise ValueError(f"glm-tpu is not installed and {pyproject} does not exist: no declared requirements")
        project = tomllib.loads(pyproject.read_text()).get("project", {})
        if project.get("name") != "glm-tpu":
            raise ValueError(f"{pyproject} does not declare the glm-tpu project")
        declared_by = str(pyproject)
        python = project.get("requires-python")
        rows = [("core", pin) for pin in project.get("dependencies", [])]
        rows += [(extra, pin) for extra, pins in project.get("optional-dependencies", {}).items() for pin in pins]
    if not python:
        raise ValueError("glm-tpu declares no requires-python")
    groups: dict[str, dict[str, str]] = {}
    for group, requirement in rows:
        pin = _PIN.fullmatch(requirement.strip())
        if pin is None:
            raise ValueError(f"not an exact pin: {requirement}")
        if groups.setdefault(group, {}).setdefault(pin.group(1), pin.group(2)) != pin.group(2):
            raise ValueError(f"conflicting pins for {pin.group(1)}")
    return dict(declared_by=declared_by, python=python, groups=groups)


def _matches(installed: str, pin: str) -> bool:
    """PEP 440 ``==pin`` for exact release pins: a pin without a local label also matches it with any local label."""
    return installed == pin or ("+" not in pin and installed.split("+", 1)[0] == pin)


def _satisfies(version: tuple[int, ...], specifier: str) -> bool:
    """Whether the release ``version`` satisfies every comma-separated clause of ``specifier``, release numbers
    compared zero-padded (``3.12`` is ``3.12.0``). Any other clause (``~=``, a wildcard, a pre-release) is refused."""
    results = []
    for clause in specifier.split(","):
        match = _CLAUSE.fullmatch(clause.strip())
        if match is None:
            raise ValueError(f"unsupported requires-python clause: {clause.strip()!r}")
        bound = tuple(int(part) for part in match.group(2).split("."))
        width = max(len(version), len(bound))
        padded = version + (0,) * (width - len(version)), bound + (0,) * (width - len(bound))
        results.append(_COMPARE[match.group(1)](*padded))
    return all(results)


def environment_report(
    profile: str,
    *,
    version: Callable[[str], str] = metadata.version,
    python_version: tuple[int, ...] | None = None,
) -> dict:
    """Compare distribution metadata only; a match is NOT hardware admission.

    The expected versions are the declared pins of the profile's groups (``declared_requirements``). A package
    matches when its installed version equals the pin or, for a pin without a local label, when its public version
    (before ``+``) does: ``torch 2.10.0+cpu`` matches ``torch==2.10.0``. The running Python (major.minor.micro) must
    satisfy the declared requires-python specifier.
    """
    if profile not in PROFILES:
        raise ValueError("unknown environment profile")
    declared = declared_requirements()
    expected: dict[str, str] = {}
    for group in PROFILES[profile]:
        if not declared["groups"].get(group):
            raise ValueError(f"glm-tpu declares no {group} requirements")
        for name, pin in declared["groups"][group].items():
            if expected.setdefault(name, pin) != pin:
                raise ValueError(f"conflicting pins for {name}")
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
                status=("missing" if actual is None else "match" if _matches(actual, pin) else "mismatch"),
            )
        )
    running = tuple(python_version if python_version is not None else sys.version_info[:3])
    return dict(
        schema="glm_tpu_environment_report_v2",
        profile=profile,
        declared_by=declared["declared_by"],
        python=".".join(str(part) for part in running),
        expected_python=declared["python"],
        packages=rows,
        passed=_satisfies(running, declared["python"]) and all(row["status"] == "match" for row in rows),
        scope=(
            "distribution metadata only; no dependency solving, payload hashes, model imports, TPU access or "
            "launch authorization"
        ),
    )


class CollectEnvSubcommand(CLISubcommand):
    """``glm-tpu collect-env`` (alias ``doctor``): the environment report of one profile; exit 0 only when it passed.

    A refusal (no declared requirements, or one the report cannot read) prints its type and message as JSON on
    standard error and exits 1.
    """

    name = "collect-env"

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        try:
            report = environment_report(args.profile)
        except (ValueError, OSError) as exc:
            refusal = dict(error=type(exc).__name__, message=str(exc), status="environment report refused")
            print(json.dumps(refusal), file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        collect_env = subparsers.add_parser(
            "collect-env", aliases=["doctor"], help="check installed version metadata without initializing TPU"
        )
        collect_env.add_argument("--profile", choices=tuple(PROFILES), default="core")
        return collect_env
