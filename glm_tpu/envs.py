"""The ``GLM_TPU_*`` environment variables: one registry, read lazily (vLLM ``envs.py`` shape).

``import glm_tpu.envs as envs; envs.GLM_TPU_RUN_ROOT`` evaluates the variable at attribute access
(module ``__getattr__``), so a test's ``monkeypatch.setenv`` takes effect without re-importing.
Precedence (DESIGN 5.6): command-line flag > ``GLM_TPU_*`` variable > site file > built-in
default. Fleet values and the launch policy are site-file only; no variable overrides them.

Each entry is an :class:`EnvVar` with its getter, one line of documentation and a ``secret`` flag
(a secret's value is only ever reported as ``<set>``/``<unset>``). Raw upstream variables
(``JAX_PLATFORMS``, ``XLA_FLAGS``, ``TPU_*``, ``HF_TOKEN``) are read directly where needed; the
worker handshake flags stay with the resident protocol until the S3 rename.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    GLM_TPU_CONFIG_ROOT: Path
    GLM_TPU_SITE_CONFIG: Path
    GLM_TPU_RUN_ROOT: Path | None
    GLM_TPU_MODEL_PATH: Path | None
    GLM_TPU_HLO_DUMP_ROOT: Path | None
    GLM_TPU_TEST_SITE: Path | None
    GLM_TPU_TEST_TIMEOUT_SCALE: float
    GLM_TPU_TEST_HELPER_PYTHON: str


@dataclass(frozen=True)
class EnvVar:
    getter: Callable[[], Any]
    doc: str
    secret: bool = False


def _optional_path(name: str) -> Path | None:
    value = os.environ.get(name, "")
    return Path(os.path.expanduser(value)) if value else None


def _config_root() -> Path:
    explicit = _optional_path("GLM_TPU_CONFIG_ROOT")
    if explicit is not None:
        return explicit
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "glm-tpu"


def _positive_float(name: str, default: float) -> float:
    value = os.environ.get(name, "")
    if not value:
        return default
    number = float(value)
    if not number > 0:
        raise ValueError(f"{name} must be a positive number")
    return number


# --8<-- [start:env-vars-definition]
environment_variables: dict[str, EnvVar] = {
    "GLM_TPU_CONFIG_ROOT": EnvVar(
        _config_root,
        "Directory of site.toml and the untracked equivalence records "
        "(default $XDG_CONFIG_HOME/glm-tpu, else ~/.config/glm-tpu)."),
    "GLM_TPU_SITE_CONFIG": EnvVar(
        lambda: _optional_path("GLM_TPU_SITE_CONFIG") or _config_root() / "site.toml",
        "Explicit site file (default $GLM_TPU_CONFIG_ROOT/site.toml)."),
    "GLM_TPU_RUN_ROOT": EnvVar(
        lambda: _optional_path("GLM_TPU_RUN_ROOT"),
        "Controller-side override of the site's paths.run_root (resolved into the staged site.json)."),
    "GLM_TPU_MODEL_PATH": EnvVar(
        lambda: _optional_path("GLM_TPU_MODEL_PATH"),
        "Controller-side override of paths.model_path (tokenizer files and SOURCE_COMPLETE.json)."),
    "GLM_TPU_HLO_DUMP_ROOT": EnvVar(
        lambda: _optional_path("GLM_TPU_HLO_DUMP_ROOT"),
        "Controller-side override of paths.hlo_dump_root (per-run compiler originals)."),
    "GLM_TPU_TEST_SITE": EnvVar(
        lambda: _optional_path("GLM_TPU_TEST_SITE"),
        "Tests: path of a real site file; enables the site-bound tests."),
    "GLM_TPU_TEST_TIMEOUT_SCALE": EnvVar(
        lambda: _positive_float("GLM_TPU_TEST_TIMEOUT_SCALE", 1.0),
        "Tests: multiplier for subprocess timeouts (slow CI runners)."),
    "GLM_TPU_TEST_HELPER_PYTHON": EnvVar(
        lambda: os.environ.get("GLM_TPU_TEST_HELPER_PYTHON") or "python3",
        "Tests: interpreter for the remote-helper loopback test (CI sets python3.10)."),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name].getter()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)


def is_set(name: str) -> bool:
    """Whether a registered variable is present in the environment (``KeyError`` for others)."""
    if name not in environment_variables:
        raise KeyError(name)
    return name in os.environ


def describe() -> dict[str, str]:
    """``name: value`` for every registered variable; secrets as ``<set>``/``<unset>`` only."""
    out = {}
    for name, var in environment_variables.items():
        if var.secret:
            out[name] = "<set>" if name in os.environ else "<unset>"
        else:
            out[name] = str(var.getter()) if name in os.environ else "<unset>"
    return out
