"""The GLM_TPU_ environment registry: lazy, documented, one definition per variable."""
from __future__ import annotations

from pathlib import Path

import pytest

import glm_tpu.envs as envs


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)


def test_every_variable_is_prefixed_and_documented():
    assert envs.environment_variables
    for name, var in envs.environment_variables.items():
        assert name.startswith("GLM_TPU_") and var.doc.strip()
    assert sorted(dir(envs)) == sorted(envs.environment_variables)


def test_the_definition_block_is_marked_for_the_docs():
    text = Path(envs.__file__).read_text()
    start = text.index("# --8<-- [start:env-vars-definition]")
    end = text.index("# --8<-- [end:env-vars-definition]")
    assert all(f'"{name}"' in text[start:end] for name in envs.environment_variables)


def test_config_root_and_site_file_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert envs.GLM_TPU_CONFIG_ROOT == tmp_path / ".config" / "glm-tpu"
    assert envs.GLM_TPU_SITE_CONFIG == tmp_path / ".config" / "glm-tpu" / "site.toml"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert envs.GLM_TPU_CONFIG_ROOT == tmp_path / "xdg" / "glm-tpu"
    monkeypatch.setenv("GLM_TPU_CONFIG_ROOT", str(tmp_path / "root"))
    assert envs.GLM_TPU_SITE_CONFIG == tmp_path / "root" / "site.toml"
    monkeypatch.setenv("GLM_TPU_SITE_CONFIG", "~/explicit.toml")
    assert envs.GLM_TPU_SITE_CONFIG == tmp_path / "explicit.toml"


def test_values_are_read_at_access_time(monkeypatch):
    assert envs.GLM_TPU_RUN_ROOT is None and envs.GLM_TPU_MODEL_PATH is None and envs.GLM_TPU_HLO_DUMP_ROOT is None
    monkeypatch.setenv("GLM_TPU_RUN_ROOT", "/runs")
    assert envs.GLM_TPU_RUN_ROOT == Path("/runs")
    monkeypatch.setenv("GLM_TPU_RUN_ROOT", "")
    assert envs.GLM_TPU_RUN_ROOT is None
    assert envs.GLM_TPU_TEST_HELPER_PYTHON == "python3"
    monkeypatch.setenv("GLM_TPU_TEST_HELPER_PYTHON", "python3.10")
    assert envs.GLM_TPU_TEST_HELPER_PYTHON == "python3.10"


def test_timeout_scale_is_a_positive_number(monkeypatch):
    assert envs.GLM_TPU_TEST_TIMEOUT_SCALE == 1.0
    monkeypatch.setenv("GLM_TPU_TEST_TIMEOUT_SCALE", "2.5")
    assert envs.GLM_TPU_TEST_TIMEOUT_SCALE == 2.5
    for bad in ("0", "-1"):
        monkeypatch.setenv("GLM_TPU_TEST_TIMEOUT_SCALE", bad)
        with pytest.raises(ValueError):
            envs.GLM_TPU_TEST_TIMEOUT_SCALE  # noqa: B018


def test_unknown_names_are_errors():
    with pytest.raises(AttributeError):
        envs.GLM_TPU_NOT_A_VARIABLE  # noqa: B018
    with pytest.raises(KeyError):
        envs.is_set("GLM_TPU_NOT_A_VARIABLE")


def test_describe_never_reveals_a_secret(monkeypatch):
    monkeypatch.setitem(envs.environment_variables, "GLM_TPU_EXAMPLE_TOKEN",
                        envs.EnvVar(lambda: "value", "example secret", secret=True))
    monkeypatch.setenv("GLM_TPU_EXAMPLE_TOKEN", "do-not-print")
    monkeypatch.setenv("GLM_TPU_RUN_ROOT", "/runs")
    described = envs.describe()
    assert described["GLM_TPU_EXAMPLE_TOKEN"] == "<set>" and "do-not-print" not in repr(described)
    assert described["GLM_TPU_RUN_ROOT"] == "/runs" and described["GLM_TPU_MODEL_PATH"] == "<unset>"
    assert envs.is_set("GLM_TPU_RUN_ROOT") and not envs.is_set("GLM_TPU_MODEL_PATH")
