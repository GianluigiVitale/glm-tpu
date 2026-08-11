from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import tempfile

import pytest

from scripts.greenfield.inspect_accepted_decode_projection_lowering import (
    TARGET_COUNTS,
    TARGET_GROUP,
    TARGET_OP,
    _inspect_hlo_text,
    inspect_capture,
)


def _scheduled_decode_hlo(
    *,
    group: tuple[int, ...] = TARGET_GROUP,
    emitter: str = "RotatedPincerEmitter",
) -> str:
    header = """HloModule jit_step_fun_impl, is_scheduled=true, num_partitions=32

FileNames
1 "/oracle/tpu_inference/layers/common/linear.py"
2 "/oracle/vllm/model_executor/models/deepseek_v2.py"
3 "/oracle/vllm/model_executor/layers/fused_moe/runner/moe_runner.py"

FunctionNames
1 "sharded_quantized_matmul.<locals>.wrapper"
2 "DeepseekV2DecoderLayer.forward"
3 "MoERunner._forward_impl"

FileLocations
1 {file_name_id=1 function_name_id=1 line=234 end_line=234 column=1 end_column=2}
2 {file_name_id=2 function_name_id=2 line=1218 end_line=1218 column=1 end_column=2}
3 {file_name_id=3 function_name_id=3 line=810 end_line=810 column=1 end_column=2}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=1 parent_frame_id=3}
3 {file_location_id=2 parent_frame_id=3}
4 {file_location_id=1 parent_frame_id=5}
5 {file_location_id=3 parent_frame_id=5}

%reduce_bf16 (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  %lhs = bf16[]{:T(256)} parameter(0)
  %rhs = bf16[]{:T(256)} parameter(1)
  ROOT %add = bf16[]{:T(256)} add(%lhs, %rhs)
}

ENTRY %main () -> bf16[32,6144] {
"""
    algorithm = {
        "debug": "\nStrategyND{colors:3 phases:3 dim_sizes:4,2,4}",
        "emitter": emitter,
        "strategy": "StrategyND",
    }
    group_text = ",".join(str(value) for value in group)
    lines: list[str] = []
    index = 0
    for category, count in TARGET_COUNTS.items():
        for _ in range(count):
            if category == "attention":
                shape = "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}"
                operands = f"%partial.{index}"
                frame = 1
            elif category == "dense_mlp":
                shape = "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}"
                operands = f"%partial.{index}"
                frame = 2
            else:
                component = "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}"
                shape = f"({component}, {component})"
                operands = f"%partial.{index}, %shared.{index}"
                frame = 4
            backend = json.dumps(
                {
                    "collective_algorithm_config": algorithm,
                    "used_scoped_memory_configs": [
                        {"size": "2" if category == "moe_tuple" else "1"}
                    ],
                },
                separators=(",", ":"),
            )
            lines.append(
                f"  %psum.{index} = {shape} all-reduce({operands}), channel_id=1, "
                f"replica_groups={{{{{group_text}}}}}, use_global_device_ids=true, "
                f"to_apply=%reduce_bf16, metadata={{{TARGET_OP} "
                f"stack_frame_id={frame}}}, backend_config={backend}"
            )
            index += 1
    return header + "\n".join(lines) + "\n}\n"


def _write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(text.encode())


def _short_scheduled_decode_hlo() -> str:
    text = _scheduled_decode_hlo().replace(
        "%reduce_bf16 (lhs: bf16[], rhs: bf16[]) -> bf16[] {",
        "reduce_bf16 {",
    )
    text = text.replace("ENTRY %main () -> bf16[32,6144] {", "ENTRY main {")
    return text.replace("%", "")


def test_inspector_seals_exact_decode_collective_contract() -> None:
    result = _inspect_hlo_text(_scheduled_decode_hlo())
    assert result is not None
    assert result["category_counts"] == TARGET_COUNTS
    assert result["collective_count"] == 156
    assert result["compile_bucket_rows"] == 32
    assert result["partition_count"] == 32
    assert result["replica_group"] == list(range(32))
    assert result["reduction_dtype"] == "bf16"
    assert result["collective_algorithm_config"]["strategy"] == "StrategyND"


def test_inspector_accepts_short_parsable_hlo_without_percent_names() -> None:
    result = _inspect_hlo_text(_short_scheduled_decode_hlo())
    assert result is not None
    assert result["category_counts"] == TARGET_COUNTS
    assert result["collective_count"] == 156
    assert result["reduction_dtype"] == "bf16"


def test_inspector_rejects_same_op_shape_drift() -> None:
    drifted = _scheduled_decode_hlo().replace(
        "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}",
        "bf16[32,4096]{1,0:T(8,128)(2,1)S(3)}",
        1,
    )
    with pytest.raises(ValueError, match="target-op result shape drifted"):
        _inspect_hlo_text(drifted)


def test_inspector_rejects_group_escape_and_nonuniform_algorithm() -> None:
    with pytest.raises(ValueError, match="escaped"):
        _inspect_hlo_text(_scheduled_decode_hlo(group=tuple(range(31))))
    text = _scheduled_decode_hlo().replace(
        '"emitter":"RotatedPincerEmitter"', '"emitter":"OtherEmitter"', 1
    )
    with pytest.raises(ValueError, match="nonuniform"):
        _inspect_hlo_text(text)


def test_capture_requires_identical_compile_owner_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "sealed"
    first = source / "w0" / "decode_projection_hlo" / "accepted_decode.after_codegen.txt.gz"
    second = source / "w1" / "decode_projection_hlo" / "accepted_decode.after_codegen.txt.gz"
    _write_gzip(first, _scheduled_decode_hlo())
    _write_gzip(second, _scheduled_decode_hlo(emitter="OtherEmitter"))
    with pytest.raises(ValueError, match="bytes differ"):
        inspect_capture(
            source_dump_dir=source,
            output=output,
            code_hash="code",
            legacy_code_hash="legacy",
            run_tag="tag",
        )


def test_capture_hard_links_one_reproducible_hlo(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "sealed"
    hlo = source / "w0" / "decode_projection_hlo" / "accepted_decode.after_codegen.txt.gz"
    _write_gzip(hlo, _scheduled_decode_hlo())
    result = inspect_capture(
        source_dump_dir=source,
        output=output,
        code_hash="code",
        legacy_code_hash="legacy",
        run_tag="tag",
    )
    sealed = output / "jit_step_fun_impl.m32.after_codegen_hlo.txt.gz"
    assert result["artifact_kind"] == "accepted_decode_projection_lowering_v1"
    assert result["compile_owner_workers"] == [0]
    assert result["diagnostic_only"] is True
    assert result["performance_claim"] is False
    assert hlo.stat().st_ino == sealed.stat().st_ino
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["manifest_sha256"] == result["manifest_sha256"]


def test_decode_inspector_has_no_legacy_execution_import() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "scripts/greenfield/inspect_accepted_decode_projection_lowering.py"
    )
    text = source.read_text()
    for forbidden in (
        "import tpu_inference",
        "from tpu_inference",
        "import vllm",
        "from vllm",
    ):
        assert forbidden not in text


def test_protected_entrypoint_reuses_and_compacts_plain_oracle_stack() -> None:
    repo = Path(__file__).resolve().parents[3]
    entrypoint = (
        repo
        / "scripts/greenfield/run_capture_accepted_decode_projection_lowering.sh"
    ).read_text()
    shared = (
        repo / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    for required in (
        "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k",
        "GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=0",
        "GLM_GREENFIELD_DSA_INTERNALS_MODE=scorer",
        "GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=0",
        "GLM_GREENFIELD_DSA_INTERNALS_POSITION=",
        "GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=0",
        "GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE=0",
        "GLM_GREENFIELD_MAIN_CACHE_CAPTURE=0",
        "GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_CAPTURE=1",
        "run_capture_short_context_dsa_oracle.sh",
    ):
        assert required in entrypoint
    for required in (
        "--xla_dump_hlo_as_long_text=false",
        "--xla_dump_hlo_module_re=jit_step_fun_impl",
        "--xla_dump_hlo_pass_re=after_codegen",
        "DECODE_HLO_ENV_OK",
        "DECODE_HLO_OWNER",
        "accepted_decode.after_codegen.txt.gz",
        "compact_accepted_decode_hlo.sh",
        "decode_hlo_compactor_sha",
        "inspect_accepted_decode_projection_lowering.py",
        "dsa_exact_comparison.json",
        "strict_census post",
        "accepted_decode_projection_manifest_sha256",
    ):
        assert required in shared


def test_compactor_seals_before_reclaiming_exact_run_root() -> None:
    repo = Path(__file__).resolve().parents[3]
    helper = repo / "scripts/greenfield/compact_accepted_decode_hlo.sh"
    helper_text = helper.read_text()
    assert helper_text.index('gzip -n -c "$CANDIDATE"') < helper_text.rindex(
        'find "$RAW_ROOT" -depth -delete',
    )
    with tempfile.TemporaryDirectory(prefix="greenfield_", dir="/tmp") as root:
        run_root = Path(root)
        raw = run_root / "decode_projection_hlo_raw"
        compact = run_root / "decode_projection_hlo"
        raw.mkdir(parents=True)
        (raw / "module.after_codegen.txt").write_text(_scheduled_decode_hlo())
        result = subprocess.run(
            ["bash", str(helper), str(raw), str(compact)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "DECODE_HLO_OWNER" in result.stdout
        assert not raw.exists()
        output = compact / "accepted_decode.after_codegen.txt.gz"
        assert output.is_file()
        inventory = compact / "raw_hlo_inventory.txt"
        assert inventory.read_text() == (
            f"module.after_codegen.txt\t{len(_scheduled_decode_hlo().encode())}\n"
        )
        with gzip.open(output, "rt") as stream:
            assert stream.read() == _scheduled_decode_hlo()


def test_compactor_accepts_binary_sharing_nonowner_and_rejects_unsafe_root() -> None:
    helper = (
        Path(__file__).resolve().parents[3]
        / "scripts/greenfield/compact_accepted_decode_hlo.sh"
    )
    with tempfile.TemporaryDirectory(prefix="greenfield_", dir="/tmp") as root:
        run_root = Path(root)
        result = subprocess.run(
            [
                "bash",
                str(helper),
                str(run_root / "decode_projection_hlo_raw"),
                str(run_root / "decode_projection_hlo"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "DECODE_HLO_NONOWNER" in result.stdout
    unsafe = subprocess.run(
        [
            "bash",
            str(helper),
            "/var/tmp/greenfield_bad/decode_projection_hlo_raw",
            "/var/tmp/greenfield_bad/decode_projection_hlo",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert unsafe.returncode != 0
    assert "unsafe_run_root" in unsafe.stdout


def test_compactor_preserves_unmatched_compile_diagnostics() -> None:
    helper = (
        Path(__file__).resolve().parents[3]
        / "scripts/greenfield/compact_accepted_decode_hlo.sh"
    )
    with tempfile.TemporaryDirectory(prefix="greenfield_", dir="/tmp") as root:
        run_root = Path(root)
        raw = run_root / "decode_projection_hlo_raw"
        compact = run_root / "decode_projection_hlo"
        unmatched = raw / "unexpected.after_codegen.txt"
        unmatched.parent.mkdir(parents=True)
        unmatched.write_text("HloModule jit_step_fun_impl, is_scheduled=true\n")
        result = subprocess.run(
            ["bash", str(helper), str(raw), str(compact)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "candidates=0 raw_files=1" in result.stdout
        assert unmatched.is_file()
        assert not compact.exists()
