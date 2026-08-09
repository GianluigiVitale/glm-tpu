from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.greenfield.inspect_accepted_prompt_projection_lowering import (
    TARGET_HLO_OP,
    TARGET_INVOCATIONS_PER_CORE,
    _inspect_hlo_text,
    _link_profiles,
    _projection_signature,
)


def _scheduled_hlo(*, emitter: str = "EmitterA", line: int = 1122) -> str:
    header = """HloModule jit_step_fun_impl, is_scheduled=true

FileNames
1 "/home/gianl/tpu-inference/tpu_inference/layers/vllm/custom_ops/glm_dsa_indexer.py"

FileLocations
1 {file_name_id=1 function_name_id=1 line=%d end_line=%d column=1 end_column=2}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
""" % (line, line)
    backend = json.dumps({
        "convolution_algorithm_config": {"emitter": emitter},
        "megacore_config": {"megacore_split_dim": "1"},
        "window_config": {"iteration_bounds": ["1", "4", "1"]},
    }, separators=(",", ":"))
    body = []
    for index in range(TARGET_INVOCATIONS_PER_CORE):
        body.extend([
            f"  %lhs.{index} = f32[2048,6144]{{1,0:T(8,128)}} parameter(0)",
            f"  %rhs.{index} = f32[128,6144]{{1,0:T(8,128)}} parameter(1)",
            f"  %conv.{index} = f32[2048,128]{{0,1:T(8,128)S(3)}} "
            f"convolution(%lhs.{index}, %rhs.{index}), dim_labels=bf_oi->bf, "
            f'metadata={{{TARGET_HLO_OP} stack_frame_id=1}}',
            f"  %fusion.{index} = (f32[2048]{{0:T(1024)S(3)}}, "
            f"f32[2048,128]{{0,1:T(8,128)S(3)}}) fusion(%lhs.{index}), "
            f'kind=kOutput, metadata={{{TARGET_HLO_OP} stack_frame_id=1}}, '
            f"backend_config={backend}",
        ])
    return header + "\n".join(body) + "\n"


def test_projection_signature_requires_current_source_and_m2048() -> None:
    valid = {
        "source": "/tmp/glm_dsa_indexer.py:1122",
        "tf_op": "jit(step_fun_impl)/dot_general:",
        "shape_with_layout": "(f32[2048], f32[2048,128]{0,1})",
        "hlo_category": "convolution fusion",
    }
    assert _projection_signature(valid)
    assert not _projection_signature({**valid, "source": "/tmp/glm_dsa_indexer.py:1125"})
    assert not _projection_signature({**valid, "shape_with_layout": "f32[32,128]{1,0}"})


def test_hlo_inspector_extracts_uniform_source_backed_lowering() -> None:
    result = _inspect_hlo_text(_scheduled_hlo())
    assert result is not None
    assert result["convolution_count"] == TARGET_INVOCATIONS_PER_CORE
    assert result["emitter"] == "EmitterA"
    assert result["convolution"]["lhs_shape"].startswith("f32[2048,6144]")
    assert result["convolution"]["rhs_shape"].startswith("f32[128,6144]")


def test_hlo_inspector_rejects_wrong_source_line() -> None:
    assert _inspect_hlo_text(_scheduled_hlo(line=1121)) is None


def test_hlo_inspector_rejects_nonuniform_emitter() -> None:
    text = _scheduled_hlo().replace(
        '"emitter":"EmitterA"',
        '"emitter":"EmitterB"',
        1,
    )
    with pytest.raises(ValueError, match="not uniform"):
        _inspect_hlo_text(text)


def test_profile_sealer_uses_hard_links(tmp_path) -> None:
    trace_root = tmp_path / "source"
    output = tmp_path / "sealed"
    output.mkdir()
    for worker in range(8):
        worker_root = trace_root / f"w{worker}"
        worker_root.mkdir(parents=True)
        (worker_root / f"worker{worker}.xplane.pb").write_bytes(
            f"xplane-{worker}".encode()
        )
        (worker_root / f"worker{worker}.trace.json.gz").write_bytes(
            f"trace-{worker}".encode()
        )

    records = _link_profiles(trace_root, output)

    assert len(records) == 16
    for worker in range(8):
        source = trace_root / f"w{worker}" / f"worker{worker}.xplane.pb"
        sealed = output / f"trace.worker{worker}.xplane.pb"
        assert source.stat().st_ino == sealed.stat().st_ino


def test_protected_entrypoint_reuses_plain_accepted_oracle_stack() -> None:
    repo = Path(__file__).resolve().parents[3]
    entrypoint = (
        repo
        / "scripts/greenfield/run_capture_accepted_prompt_projection_lowering.sh"
    ).read_text()
    shared = (
        repo / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    inspector = (
        repo
        / "scripts/greenfield/inspect_accepted_prompt_projection_lowering.py"
    ).read_text()

    for required in (
        "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k",
        "GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=0",
        "GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=0",
        "GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE=1",
        "run_capture_short_context_dsa_oracle.sh",
    ):
        assert required in entrypoint
    for required in (
        "PHASED_PROFILING_DIR=$PREFILL_PROFILE_PREFIX",
        "PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=1",
        "PYTHON_TRACER_LEVEL=0",
        "--xla_dump_hlo_module_re=jit_step_fun_impl",
        "PROFILE_ENV_OK",
        "PROFILE_OK",
        "HLO_OWNER",
        "dsa_exact_comparison.json",
        "strict_census post",
    ):
        assert required in shared
    for forbidden in (
        "import tpu_inference",
        "from tpu_inference",
        "import vllm",
        "from vllm",
    ):
        assert forbidden not in inspector
