"""The environment report reads the requirements glm-tpu declares: installed metadata, else the checkout's pyproject."""

from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest

from glm_tpu.entrypoints.cli import collect_env
from glm_tpu.entrypoints.cli.collect_env import declared_requirements, environment_report
from glm_tpu.entrypoints.cli.main import main

REPO = Path(__file__).resolve().parents[3]
PROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]


def uninstalled(name):
    raise metadata.PackageNotFoundError(name)


def pins_of(requirements: list[str]) -> dict[str, str]:
    return dict(requirement.split("==") for requirement in requirements)


def dist_info(root: Path, *, version: str, python: str, requires: list[str]) -> Path:
    """An installed distribution's metadata as setuptools 78.1.0 writes it (Requires-Python reordered, one
    ``extra == "<name>"`` marker per optional pin)."""
    path = root / f"glm_tpu-{version}.dist-info"
    path.mkdir()
    lines = ["Metadata-Version: 2.4", "Name: glm-tpu", f"Version: {version}", f"Requires-Python: {python}"]
    lines += [f"Requires-Dist: {requirement}" for requirement in requires]
    (path / "METADATA").write_text("\n".join(lines) + "\n")
    return path


def setuptools_requires(project: dict) -> list[str]:
    rows = list(project["dependencies"])
    for extra, pins in project.get("optional-dependencies", {}).items():
        rows += [f'{pin}; extra == "{extra}"' for pin in pins]
    return rows


def pyproject(root: Path, text: str) -> Path:
    path = root / "pyproject.toml"
    path.write_text(text)
    return path


SMALL = """\
[project]
name = "glm-tpu"
requires-python = ">=3.12,<3.13"
dependencies = ["jax==1.0", "torch==2.10.0"]

[project.optional-dependencies]
runtime = ["safetensors==0.7.0"]
tpu = ["libtpu==0.0.41", "local==1.0+cpu"]
dev = ["pytest==9.0.3"]
"""


@pytest.fixture
def source(monkeypatch):
    """Point the report at ``distribution`` and ``pyproject`` (the checkout's own when not given)."""

    def point(*, distribution=uninstalled, path=collect_env.PYPROJECT):
        monkeypatch.setitem(declared_requirements.__kwdefaults__, "distribution", distribution)
        monkeypatch.setitem(declared_requirements.__kwdefaults__, "pyproject", path)

    return point


def test_the_checkout_declares_one_exact_pin_per_package_for_every_profile(source):
    source()
    declared = declared_requirements()
    assert declared == dict(
        declared_by=str(REPO / "pyproject.toml"),
        python=PROJECT["requires-python"],
        groups=dict(
            core=pins_of(PROJECT["dependencies"]),
            **{extra: pins_of(pins) for extra, pins in PROJECT["optional-dependencies"].items()},
        ),
    )
    assert set(declared["groups"]) == {"core", "runtime", "tpu", "dev"}
    assert collect_env.PROFILES == {
        "core": ("core",),
        "runtime": ("core", "runtime"),
        "tpu": ("core", "runtime", "tpu"),
        "dev": ("core", "dev"),
    }
    with pytest.raises(ValueError, match="unknown environment profile"):
        environment_report("benchmark")


def test_the_installed_metadata_is_read_first(tmp_path, source):
    """The installed distribution's metadata wins; the pyproject named here does not even exist."""
    installed = dist_info(tmp_path, version="0.1.0a1", python="<3.13,>=3.12", requires=setuptools_requires(PROJECT))
    source(distribution=lambda name: metadata.PathDistribution(installed), path=tmp_path / "missing.toml")
    declared = declared_requirements()
    source()
    checkout = declared_requirements()
    assert declared == dict(checkout, declared_by="glm-tpu 0.1.0a1 distribution metadata", python="<3.13,>=3.12")


def test_the_installed_metadata_decides_the_report(tmp_path, source):
    installed = dist_info(
        tmp_path,
        version="9.9.9",
        python=">=3.12",
        requires=["jax==1.0", 'torch==2.10.0; extra == "runtime"', "pytest==9.0.3; extra == 'dev'"],
    )
    source(distribution=lambda name: metadata.PathDistribution(installed))
    report = environment_report("dev", version={"jax": "1.0", "pytest": "9.0.3"}.__getitem__, python_version=(3, 12, 1))
    assert report["declared_by"] == "glm-tpu 9.9.9 distribution metadata"
    assert report["expected_python"] == ">=3.12"
    assert [(row["package"], row["status"]) for row in report["packages"]] == [("jax", "match"), ("pytest", "match")]
    assert report["passed"]


def test_the_checkout_pyproject_is_read_when_not_installed(tmp_path, source):
    path = pyproject(tmp_path, SMALL)
    source(path=path)
    installed = {"jax": "1.0", "torch": "2.10.0+cpu", "safetensors": "0.7.0", "libtpu": "0.0.41", "local": "1.0+cpu"}
    report = environment_report("tpu", version=installed.__getitem__, python_version=(3, 12, 13))
    assert report == dict(
        schema="glm_tpu_environment_report_v2",
        profile="tpu",
        declared_by=str(path),
        python="3.12.13",
        expected_python=">=3.12,<3.13",
        packages=[
            dict(package=name, expected=pin, installed=installed[name], status="match")
            for name, pin in sorted(
                {
                    "jax": "1.0",
                    "torch": "2.10.0",
                    "safetensors": "0.7.0",
                    "libtpu": "0.0.41",
                    "local": "1.0+cpu",
                }.items()
            )
        ],
        passed=True,
        scope=(
            "distribution metadata only; no dependency solving, payload hashes, model imports, TPU access or "
            "launch authorization"
        ),
    )
    assert environment_report("core", version=installed.__getitem__)["python"] == ".".join(
        str(part) for part in sys.version_info[:3]
    )


@pytest.mark.parametrize(
    ("installed", "pin", "status"),
    [
        ("2.10.0+cpu", "2.10.0", "match"),
        ("2.10.0+cu128", "2.10.0", "match"),
        ("2.10.0", "2.10.0", "match"),
        ("2.10.1+cpu", "2.10.0", "mismatch"),
        ("2.10.0.post1", "2.10.0", "mismatch"),
        ("1.0+cpu", "1.0+cpu", "match"),
        ("1.0", "1.0+cpu", "mismatch"),
        ("1.0+cu128", "1.0+cpu", "mismatch"),
    ],
)
def test_a_local_version_label_matches_its_public_version(tmp_path, source, installed, pin, status):
    text = f'[project]\nname = "glm-tpu"\nrequires-python = ">=3.12"\ndependencies = ["torch=={pin}"]\n'
    source(path=pyproject(tmp_path, text))
    (row,) = environment_report("core", version=lambda name: installed, python_version=(3, 12))["packages"]
    assert row == dict(package="torch", expected=pin, installed=installed, status=status)


@pytest.mark.parametrize(
    ("python", "specifier", "passed"),
    [
        ((3, 12, 13), ">=3.12,<3.13", True),
        ((3, 12), ">=3.12,<3.13", True),
        ((3, 12, 13), "<3.13,>=3.12", True),
        ((3, 13, 0), ">=3.12,<3.13", False),
        ((3, 11, 9), ">=3.12,<3.13", False),
        ((3, 12, 0), "==3.12", True),
        ((3, 12, 1), "==3.12", False),
        ((3, 12, 1), "!=3.12.1", False),
        ((3, 12, 1), ">3.12", True),
        ((3, 12, 0), ">3.12", False),
        ((3, 12, 5), "<=3.12.5", True),
    ],
)
def test_requires_python_is_a_specifier(tmp_path, source, python, specifier, passed):
    source(path=pyproject(tmp_path, SMALL.replace('">=3.12,<3.13"', f'"{specifier}"')))
    installed = {"jax": "1.0", "torch": "2.10.0"}
    report = environment_report("core", version=installed.__getitem__, python_version=python)
    assert (report["python"], report["expected_python"], report["passed"]) == (
        ".".join(map(str, python)),
        specifier,
        passed,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (('">=3.12,<3.13"', '"~=3.12"'), "unsupported requires-python clause: '~=3.12'"),
        (('">=3.12,<3.13"', '"==3.12.*"'), "unsupported requires-python clause"),
        (('">=3.12,<3.13"', '">=3.12,"'), "unsupported requires-python clause: ''"),
        (('requires-python = ">=3.12,<3.13"\n', ""), "glm-tpu declares no requires-python"),
        (('"jax==1.0"', '"jax>=1.0"'), "not an exact pin: jax>=1.0"),
        (('"jax==1.0"', '"jax[tpu]==1.0"'), "not an exact pin"),
        (('"jax==1.0"', "\"jax==1.0; python_version < '3.13'\""), "not an exact pin"),
        (('"torch==2.10.0"', '"jax==1.1"'), "conflicting pins for jax"),
        (('name = "glm-tpu"', 'name = "other"'), "does not declare the glm-tpu project"),
        (('dev = ["pytest==9.0.3"]', "dev = []"), "glm-tpu declares no dev requirements"),
    ],
)
def test_unreadable_requirements_are_refused(tmp_path, source, change, message):
    old, new = change
    assert old in SMALL
    source(path=pyproject(tmp_path, SMALL.replace(old, new)))
    with pytest.raises(ValueError, match=re.escape(message)):
        environment_report("dev", version=lambda name: "1.0", python_version=(3, 12))


def test_pins_that_differ_between_groups_are_refused(tmp_path, source):
    source(path=pyproject(tmp_path, SMALL.replace('runtime = ["safetensors==0.7.0"]', 'runtime = ["jax==1.1"]')))
    assert environment_report("dev", version=lambda name: "1.0", python_version=(3, 12))["passed"] is False
    with pytest.raises(ValueError, match="conflicting pins for jax"):
        environment_report("runtime", version=lambda name: "1.0", python_version=(3, 12))


def test_a_marker_other_than_an_extra_is_refused(tmp_path, source):
    installed = dist_info(
        tmp_path, version="9.9.9", python=">=3.12", requires=["jax==1.0", 'torch==2.10.0; python_version < "3.13"']
    )
    source(distribution=lambda name: metadata.PathDistribution(installed))
    with pytest.raises(ValueError, match=re.escape("unsupported requirement marker: torch==2.10.0; python_version")):
        environment_report("core", version=lambda name: "1.0", python_version=(3, 12))


@pytest.mark.parametrize("command", ["collect-env", "doctor"])
def test_without_declared_requirements_the_command_refuses(tmp_path, source, capsys, command):
    missing = tmp_path / "pyproject.toml"
    source(path=missing)
    with pytest.raises(ValueError, match=r"glm-tpu is not installed and .* does not exist"):
        environment_report("core")
    capsys.readouterr()
    assert main([command]) == 1
    refusal = dict(
        error="ValueError",
        message=f"glm-tpu is not installed and {missing} does not exist: no declared requirements",
        status="environment report refused",
    )
    assert capsys.readouterr() == ("", json.dumps(refusal) + "\n")


def test_an_installed_distribution_is_found_on_the_path(tmp_path):
    """End to end: ``python -m glm_tpu collect-env`` in a fresh process finds a distribution on its path (run from
    that directory, first on the path, so no metadata of the checkout itself can shadow it)."""
    dist_info(tmp_path, version="9.9.9", python=">=3.12", requires=["jax==0.0.1", 'pytest==0.0.2; extra == "dev"'])
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(REPO)]))
    result = subprocess.run(
        [sys.executable, "-m", "glm_tpu", "collect-env", "--profile", "dev"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert (result.returncode, result.stderr) == (1, ""), result.stderr
    assert report["declared_by"] == "glm-tpu 9.9.9 distribution metadata"
    assert [(row["package"], row["expected"], row["status"]) for row in report["packages"]] == [
        ("jax", "0.0.1", "mismatch"),
        ("pytest", "0.0.2", "mismatch"),
    ]
