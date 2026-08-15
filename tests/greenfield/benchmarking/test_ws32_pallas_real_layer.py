from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[3]


def test_ws32_pallas_runner_and_acquisition_are_distinct_and_default_off() -> None:
    runner = REPO / "scripts/greenfield/run_real_one_layer_ws32_pallas.py"
    wrapper = REPO / "scripts/greenfield/run_real_one_layer_ws32_pallas.sh"
    mapped = (
        REPO
        / "glm_tpu/greenfield/benchmarking/ws32_pallas_one_layer.py"
    )
    compile(runner.read_text(), str(runner), "exec")
    compile(mapped.read_text(), str(mapped), "exec")
    syntax = subprocess.run(
        ["bash", "-n", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stdout + syntax.stderr
    source = wrapper.read_text()
    assert "GLM_GREENFIELD_WS32_PALLAS_REAL_LAYER:-0" in source
    assert "GLM_GREENFIELD_WS32_PALLAS_REAL_LAYER_MODE:-off} == acquire" in source
    assert "run_real_one_layer_ws32_pallas.py" in source
    assert "--compile-only 1" in source
    assert "WS32_PALLAS_ACQUIRE_OK" in source
    assert "no arithmetic, DB row, performance claim, or terminal SUCCESS" in source
    assert "gcloud storage" in source
    assert "gs://driftbench-dsv4-uc" in source
    assert "SUCCESS" not in {line.strip() for line in source.splitlines()}
    heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.DOTALL)
    assert len(heredocs) == 1
    compile(heredocs[0], f"{wrapper}:terminal", "exec")

    environment = dict(os.environ)
    environment.pop("GLM_GREENFIELD_WS32_PALLAS_REAL_LAYER", None)
    environment.pop("GLM_GREENFIELD_WS32_PALLAS_REAL_LAYER_MODE", None)
    refused = subprocess.run(
        [str(wrapper)],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2
    assert "default-off" in refused.stderr


def test_ws32_reference_protected_source_locations_remain_frozen() -> None:
    runner = REPO / "scripts/greenfield/run_real_one_layer_ws32.py"
    mapped = REPO / "glm_tpu/greenfield/benchmarking/ws32_one_layer.py"
    runner_lines = runner.read_text().splitlines()
    mapped_lines = mapped.read_text().splitlines()
    assert "lowered = jax.jit(mapped).lower(*normal_inputs)" in runner_lines[374]
    assert "raise SystemExit(main())" in runner_lines[503]
    assert "return ws32_moe_fp8_from_routes_mapped(" in mapped_lines[217]
    assert "GLM_GREENFIELD_WS32_ARTIFACT_KIND" in runner_lines[433]
