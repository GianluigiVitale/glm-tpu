from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import numpy as np

from glm_tpu.greenfield.validation.prompt_index_cache import (
    LegacyPromptIndexCacheConfig,
    capture_legacy_prompt_index_cache,
    compare_prompt_index_key_bits,
    inspect_legacy_prompt_index_cache,
    validate_prompt_index_key_association_hlo,
    validate_prompt_index_key_probe_hlo,
)


def _write_dump(
    path: Path,
    *,
    process_index: int,
    global_bits: np.ndarray,
    corrupt_replica: bool = False,
) -> None:
    shard = global_bits.copy()
    duplicate = shard.copy()
    if corrupt_replica:
        duplicate.flat[0] ^= np.uint16(1)
    index = (
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
    )
    np.savez(
        path,
        step_index=np.asarray(2, dtype=np.int64),
        phase=np.asarray("postfwd"),
        num_scheduled_tokens=np.asarray(2, dtype=np.int64),
        process_index=np.asarray(process_index, dtype=np.int64),
        process_count=np.asarray(2, dtype=np.int64),
        layer_indices=np.asarray([0], dtype=np.int64),
        mesh_shape=np.asarray(
            "{'data': 1, 'attn_dp': 1, 'attn_dp_expert': 1, "
            "'expert': 1, 'model': 4, 'dcp': 1}"
        ),
        meta__block_tables=np.asarray([1, 2, 0, 0], dtype=np.int32),
        meta__seq_lens=np.asarray([66], dtype=np.int32),
        layer0__sharding=np.asarray("P(None, 'dcp')"),
        layer0__shape=np.asarray(global_bits.shape, dtype=np.int64),
        layer0__dtype=np.asarray("bfloat16"),
        layer0__nshards=np.asarray(2, dtype=np.int64),
        layer0__shard0__data=shard,
        layer0__shard0__index=np.asarray(str(index)),
        layer0__shard0__device=np.asarray(
            f"TPU_{process_index * 2}(process={process_index},(0,0,0,0))"
        ),
        layer0__shard1__data=duplicate,
        layer0__shard1__index=np.asarray(str(index)),
        layer0__shard1__device=np.asarray(
            f"TPU_{process_index * 2 + 1}(process={process_index},(1,0,0,0))"
        ),
    )


def _config(source: Path, output: Path) -> LegacyPromptIndexCacheConfig:
    return LegacyPromptIndexCacheConfig(
        source_dump_dir=source,
        output_dir=output,
        capture_code_hash="a" * 40,
        legacy_repository_pin="b" * 40,
        run_tag="unit",
        source_run_id=1,
        source_item_row_id=2,
        layer0_input_manifest_sha256="c" * 64,
        prompt_token_ids_sha256="d" * 64,
        expected_process_count=2,
        expected_local_replication=2,
        expected_physical_replication=4,
        expected_mesh_model_size=4,
        expected_mesh_dcp_size=1,
        expected_step_index=2,
        expected_last_chunk_tokens=2,
        expected_prompt_tokens=66,
        expected_physical_pages=3,
        expected_logical_page_size=64,
        expected_head_dim=4,
    )


def test_reconstructs_replicated_dcp_cache_in_logical_order(tmp_path: Path) -> None:
    import ml_dtypes

    source = tmp_path / "source"
    source.mkdir()
    # The portable dump keeps a fixed packing of 32; page size 64 -> dim1=2.
    values = np.arange(3 * 64 * 4, dtype=np.float32).reshape(3, 64, 4)
    values = (values / 100).astype(ml_dtypes.bfloat16)
    global_bits = values.view(np.uint16).reshape(3, 2, 32, 4)
    _write_dump(
        source / "index_cache.postfwd.step0002.proc0.npz",
        process_index=0,
        global_bits=global_bits,
    )
    _write_dump(
        source / "index_cache.postfwd.step0002.proc1.npz",
        process_index=1,
        global_bits=global_bits,
    )
    manifest = capture_legacy_prompt_index_cache(
        _config(source, tmp_path / "artifact")
    )
    inspected, bits = inspect_legacy_prompt_index_cache(
        tmp_path / "artifact",
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    expected = np.concatenate((global_bits[1].reshape(64, 4),
                               global_bits[2].reshape(64, 4)[:2]))
    np.testing.assert_array_equal(bits, expected)
    assert inspected["source_layout"]["physical_replication"] == 4
    assert inspected["source_layout"]["local_replication_per_process"] == 2
    assert inspected["source_layout"]["live_block_table"] == [1, 2]
    assert inspected["prompt_index_key_bfloat16_sha256"] == sha256(
        expected.tobytes()
    ).hexdigest()


def test_rejects_disagreeing_model_replicas(tmp_path: Path) -> None:
    import pytest

    source = tmp_path / "source"
    source.mkdir()
    global_bits = np.zeros((3, 2, 32, 4), dtype=np.uint16)
    _write_dump(
        source / "index_cache.postfwd.step0002.proc0.npz",
        process_index=0,
        global_bits=global_bits,
        corrupt_replica=True,
    )
    _write_dump(
        source / "index_cache.postfwd.step0002.proc1.npz",
        process_index=1,
        global_bits=global_bits,
    )
    with pytest.raises(ValueError, match="replicas disagree"):
        capture_legacy_prompt_index_cache(
            _config(source, tmp_path / "artifact")
        )


def test_exact_bit_comparison_reports_first_position() -> None:
    expected = np.zeros((3, 4), dtype=np.uint16)
    observed = expected.copy()
    observed[1, 2] = np.uint16(0x3F80)
    result = compare_prompt_index_key_bits(expected, observed)
    assert result["elementwise_exact"] is False
    assert result["first_mismatch_position"] == 1
    assert result["mismatch_count"] == 1
    assert result["mismatched_position_count"] == 1


def test_prompt_key_probe_hlo_requires_one_row_scan_and_raw_wk() -> None:
    hlo = """
ENTRY main {
  %embeddings = bf16[37,6144]{1,0} parameter(0)
  %rows = s32[8155]{0} parameter(1)
  %wk = u8[128,6144]{1,0} parameter(2)
  %scale = f32[1,48]{1,0} parameter(3)
  %row = bf16[1,6144]{1,0} dynamic-slice(%embeddings)
  %keys = bf16[8155,128]{1,0} while(%row)
  ROOT %call = f32[8,128]{1,0} custom-call(%row, %wk, %scale), custom_call_target="tpu_custom_call", backend_config="greenfield_fp8_block_matmul_f32_m8_k6144_n128"
}
"""
    result = validate_prompt_index_key_probe_hlo(hlo)
    assert result["passed"] is True
    assert result["key_kernel_count"] == 1
    assert result["outer_scan_while_count"] == 1


def test_prompt_key_probe_hlo_rejects_collective_and_dead_rows() -> None:
    hlo = """
ENTRY main {
  %embeddings = bf16[37,6144]{1,0} parameter(0)
  %rows = s32[8155]{0} parameter(1)
  %wk = u8[128,6144]{1,0} parameter(2)
  %scale = f32[1,48]{1,0} parameter(3)
  %dead = bf16[32,6144]{1,0} all-gather(%embeddings)
  %keys = bf16[8155,128]{1,0} while(%dead)
  ROOT %call = f32[8,128]{1,0} custom-call(%dead, %wk, %scale), custom_call_target="tpu_custom_call", backend_config="greenfield_fp8_block_matmul_f32_m8_k6144_n128"
}
"""
    result = validate_prompt_index_key_probe_hlo(hlo)
    assert result["passed"] is False
    assert result["forbidden_operations"]["all-gather"] == 1
    assert "bf16[32,6144]" in result["forbidden_shapes"]


def test_prompt_key_association_hlo_accepts_pallas_divide_sqrt() -> None:
    hlo = """
ENTRY main {
  %embeddings = bf16[37,6144]{1,0} parameter(0)
  %rows = s32[8155]{0} parameter(1)
  %wk = u8[128,6144]{1,0} parameter(2)
  %scale = f32[1,48]{1,0} parameter(3)
  %row = bf16[1,6144]{1,0} dynamic-slice(%embeddings)
  %root = f32[1]{0} sqrt(%scale)
  %normalized = f32[1]{0} divide(%scale, %root)
  %keys = bf16[8155,128]{1,0} while(%row)
  ROOT %call = f32[8,128]{1,0} custom-call(%row, %wk, %scale), custom_call_target="tpu_custom_call", backend_config="greenfield_fp8_block_matmul_f32_m8_k6144_n128"
}
"""
    result = validate_prompt_index_key_association_hlo(
        hlo, candidate="production_pallas_m1_divide_sqrt"
    )
    assert result["passed"] is True
    assert result["association"]["mode"] == "divide_sqrt"


def test_prompt_key_association_hlo_accepts_xla_modes() -> None:
    common = """
ENTRY main {
  %embeddings = bf16[37,6144]{1,0} parameter(0)
  %rows = s32[8155]{0} parameter(1)
  %wk = f32[128,6144]{1,0} parameter(2)
  %chunk = bf16[2048,6144]{1,0} dynamic-slice(%embeddings)
  %projection = f32[2048,128]{1,0} convolution(%chunk, %wk), dim_labels=bf_oi->bf
  %keys = bf16[8155,128]{1,0} while(%projection)
  ASSOCIATION
}
"""
    divide = common.replace(
        "ASSOCIATION",
        "%root = f32[1]{0} sqrt(%projection)\n"
        "  ROOT %normalized = f32[1]{0} divide(%projection, %root)",
    )
    result = validate_prompt_index_key_association_hlo(
        divide, candidate="accepted_xla_m2048_divide_sqrt"
    )
    assert result["passed"] is True
    multiplied = common.replace(
        "ASSOCIATION",
        "%input_norm = f32[1]{0} rsqrt(%projection)\n"
        "  ROOT %key_norm = f32[1]{0} rsqrt(%projection)",
    )
    result = validate_prompt_index_key_association_hlo(
        multiplied, candidate="accepted_xla_m2048_multiply_rsqrt"
    )
    assert result["passed"] is True


def test_prompt_key_association_hlo_rejects_full_prompt_hidden() -> None:
    hlo = """
ENTRY main {
  %embeddings = bf16[37,6144]{1,0} parameter(0)
  %rows = s32[8155]{0} parameter(1)
  %wk = f32[128,6144]{1,0} parameter(2)
  %chunk = bf16[2048,6144]{1,0} parameter(3)
  %dead = bf16[4,2048,6144]{2,1,0} parameter(4)
  %projection = f32[2048,128]{1,0} convolution(%chunk, %wk), dim_labels=bf_oi->bf
  %keys = bf16[8155,128]{1,0} while(%projection)
  %root = f32[1]{0} sqrt(%projection)
  ROOT %normalized = f32[1]{0} divide(%projection, %root)
}
"""
    result = validate_prompt_index_key_association_hlo(
        hlo, candidate="accepted_xla_m2048_divide_sqrt"
    )
    assert result["passed"] is False
    assert "bf16[4,2048,6144]" in result["forbidden_shapes"]


def test_prompt_key_chunk_parameter_hlo_has_no_loop_or_full_prompt() -> None:
    hlo = """
ENTRY main {
  %hidden = bf16[2048,6144]{1,0} parameter(0)
  %positions = s32[2048]{0} parameter(1)
  %wk = f32[128,6144]{1,0} parameter(2)
  %projection = f32[2048,128]{1,0} convolution(%hidden, %wk), dim_labels=bf_oi->bf
  %root = f32[1]{0} sqrt(%projection)
  %normalized = f32[1]{0} divide(%projection, %root)
  ROOT %keys = bf16[2048,128]{1,0} convert(%normalized)
}
"""
    result = validate_prompt_index_key_association_hlo(
        hlo,
        candidate="accepted_xla_m2048_chunk_parameter_divide_sqrt",
    )
    assert result["passed"] is True
    assert result["loop_count"] == 0
    assert result["bf16_wk_conversion_count"] == 0
    assert result["convolution_weight_bf16"] is False


def test_prompt_key_chunk_bf16_weight_hlo_pins_conversion() -> None:
    hlo = """
ENTRY main {
  %hidden = bf16[2048,6144]{1,0} parameter(0)
  %positions = s32[2048]{0} parameter(1)
  %wk_weight.1 = f32[128,6144]{1,0} parameter(2)
  %wk_copy = f32[128,6144]{1,0} copy(%wk_weight.1)
  %wk_bf16 = bf16[128,6144]{1,0} convert(%wk_copy)
  %projection = f32[2048,128]{1,0} convolution(%hidden, %wk_bf16), dim_labels=bf_oi->bf
  %root = f32[1]{0} sqrt(%projection)
  %normalized = f32[1]{0} divide(%projection, %root)
  ROOT %keys = bf16[2048,128]{1,0} convert(%normalized)
}
"""
    result = validate_prompt_index_key_association_hlo(
        hlo,
        candidate=(
            "accepted_xla_m2048_chunk_bf16_weight_divide_sqrt"
        ),
    )
    assert result["passed"] is True
    assert result["loop_count"] == 0
    assert result["bf16_wk_conversion_count"] == 1
    assert result["convolution_weight_bf16"] is True

    disconnected = hlo.replace(
        "convolution(%hidden, %wk_bf16)",
        "convolution(%hidden, %wk_weight.1)",
    )
    rejected = validate_prompt_index_key_association_hlo(
        disconnected,
        candidate=(
            "accepted_xla_m2048_chunk_bf16_weight_divide_sqrt"
        ),
    )
    assert rejected["passed"] is False
    assert "M2048 convolution does not consume a BF16 wk operand" in (
        rejected["violations"]
    )


def test_prompt_key_gather_chunk_hlo_pins_input_rms_producer() -> None:
    hlo = """
ENTRY main {
  %unique = bf16[37,6144]{1,0} parameter(0)
  %rows = s32[2048]{0} parameter(1)
  %positions = s32[2048]{0} parameter(2)
  %wk_weight = f32[128,6144]{1,0} parameter(3)
  %wk_bf16 = bf16[128,6144]{1,0} convert(%wk_weight)
  %gathered = bf16[2048,6144]{1,0} fusion(%unique, %rows), kind=kCustom, metadata={op_name="jit(probe)/jit(_take)/gather"}
  %input_rms = f32[2048]{0} fusion(%gathered), kind=kLoop, metadata={op_name="jit(probe)/reduce_sum"}
  %projection = f32[2048,128]{1,0} convolution(%gathered, %wk_bf16), dim_labels=bf_oi->bf
  %root = f32[1]{0} sqrt(%projection)
  %normalized = f32[1]{0} divide(%projection, %root)
  ROOT %keys = bf16[2048,128]{1,0} convert(%normalized)
}
"""
    candidate = (
        "accepted_xla_m2048_gather_chunk_bf16_weight_divide_sqrt"
    )
    result = validate_prompt_index_key_association_hlo(
        hlo,
        candidate=candidate,
    )
    assert result["passed"] is True
    assert result["loop_count"] == 0
    assert result["physical_embedding_gather_count"] == 1
    assert result["gather_coupled_input_rms"] is True

    disconnected = hlo.replace(
        "%input_rms = f32[2048]{0} fusion(%gathered)",
        "%input_rms = f32[2048]{0} fusion(%unique)",
    )
    rejected = validate_prompt_index_key_association_hlo(
        disconnected,
        candidate=candidate,
    )
    assert rejected["passed"] is False
    assert "input RMS reduction does not consume the gather producer" in (
        rejected["violations"]
    )


def test_prompt_key_gather_chunk_hlo_classifies_wk_feature_slices() -> None:
    hlo = """
ENTRY main {
  %unique = bf16[37,6144]{1,0} parameter(0)
  %rows = s32[2048]{0} parameter(1)
  %positions = s32[2048]{0} parameter(2)
  %wk_weight.1 = f32[128,6144]{1,0} parameter(3), metadata={op_name="wk_weight"}
  %wk_bf16 = bf16[128,6144]{1,0} convert(%wk_weight.1)
  %gathered = bf16[2048,6144]{1,0} fusion(%unique, %rows), kind=kCustom, metadata={op_name="jit(probe)/jit(_take)/gather"}
  %input_rms = f32[2048]{0} fusion(%gathered), kind=kLoop, metadata={op_name="jit(probe)/reduce_sum"}
  %slice-start = ((f32[128,6144]{1,0}), f32[32,6144]{1,0}, s32[]) slice-start(%wk_weight.1), slice={[0:32], [0:6144]}
  %slice-start.1 = ((f32[128,6144]{1,0}), f32[32,6144]{1,0}, s32[]) slice-start(%wk_weight.1), slice={[32:64], [0:6144]}
  %slice-start.2 = ((f32[128,6144]{1,0}), f32[32,6144]{1,0}, s32[]) slice-start(%wk_weight.1), slice={[64:96], [0:6144]}
  %slice-start.3 = ((f32[128,6144]{1,0}), f32[32,6144]{1,0}, s32[]) slice-start(%wk_weight.1), slice={[96:128], [0:6144]}
  %slice-done = f32[32,6144]{1,0} slice-done(%slice-start)
  %slice-done.1 = f32[32,6144]{1,0} slice-done(%slice-start.1)
  %slice-done.2 = f32[32,6144]{1,0} slice-done(%slice-start.2)
  %slice-done.3 = f32[32,6144]{1,0} slice-done(%slice-start.3)
  %wk = f32[128,6144]{1,0} custom-call(%slice-done, %slice-done.1, %slice-done.2, %slice-done.3), custom_call_target="ConcatBitcast"
  %projection = f32[2048,128]{1,0} convolution(%gathered, %wk_bf16), dim_labels=bf_oi->bf
  %root = f32[1]{0} sqrt(%projection)
  %normalized = f32[1]{0} divide(%projection, %root)
  ROOT %keys = bf16[2048,128]{1,0} convert(%normalized)
}
"""
    candidate = (
        "accepted_xla_m2048_gather_chunk_bf16_weight_divide_sqrt"
    )
    result = validate_prompt_index_key_association_hlo(
        hlo,
        candidate=candidate,
    )
    assert result["passed"] is True
    assert result["forbidden_shapes"] == []
    assert result["wk_feature_slices"] == {
        "done_count": 4,
        "shape_line_count": 8,
        "slice_count": 4,
        "slice_spans": [[0, 32], [32, 64], [64, 96], [96, 128]],
        "unclassified_line_count": 0,
        "valid": True,
        "wk_parameter_count": 1,
    }

    dead_parameter = hlo.replace(
        "%positions = s32[2048]{0} parameter(2)",
        "%positions = s32[2048]{0} parameter(2)\n"
        "  %dead = f32[32,6144]{1,0} parameter(7)",
    )
    rejected = validate_prompt_index_key_association_hlo(
        dead_parameter,
        candidate=candidate,
    )
    assert rejected["passed"] is False
    assert "f32[32,6144]" in rejected["forbidden_shapes"]
    assert rejected["wk_feature_slices"]["unclassified_line_count"] == 1


def test_protected_prompt_cache_probe_reuses_capture_and_production_path() -> None:
    repo = Path(__file__).resolve().parents[3]
    probe = repo / "scripts/greenfield/probe_layer0_prompt_index_cache.py"
    wrapper = repo / (
        "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    )
    entrypoint = repo / (
        "scripts/greenfield/run_capture_legacy_prompt_index_cache.sh"
    )
    resume = repo / (
        "scripts/greenfield/run_prompt_index_cache_comparison.sh"
    )
    association_probe = repo / (
        "scripts/greenfield/probe_layer0_prompt_index_cache_association.py"
    )
    association_wrapper = repo / (
        "scripts/greenfield/run_prompt_index_cache_association_probe.sh"
    )
    probe_source = probe.read_text()
    wrapper_source = wrapper.read_text()
    entrypoint_source = entrypoint.read_text()
    resume_source = resume.read_text()
    association_probe_source = association_probe.read_text()
    association_wrapper_source = association_wrapper.read_text()
    for required in (
        "fp8_block_matmul_f32",
        "dsa_index_keys_from_projection",
        "lax.scan",
        "compare_prompt_index_key_bits",
        "validate_prompt_index_key_probe_hlo",
        '"one_live_row": True',
        '"performance_claim": False',
    ):
        assert required in probe_source
    for forbidden in (
        "import tpu_inference",
        "from tpu_inference",
        "import vllm",
        "from vllm",
    ):
        assert forbidden not in probe_source
    for required in (
        "GLM_DCP_CACHE_DUMP=$PROMPT_CACHE_DUMP_PREFIX",
        "GLM_DCP_CACHE_DUMP_LAYERS=0",
        "prompt_cache_source_count -eq 32",
        "capture_legacy_prompt_index_cache.py",
        "probe_layer0_prompt_index_cache.py",
        "strict_census post",
        "prompt_index_cache_production_elementwise_exact",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
    ):
        assert required in wrapper_source
    for required in (
        "GLM_GREENFIELD_PROMPT_CACHE_CAPTURE=1",
        "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k",
        "run_capture_short_context_dsa_oracle.sh",
    ):
        assert required in entrypoint_source
    for required in (
        "SOURCE_RUN_ID=505",
        "SOURCE_ITEM_ROW_ID=1788",
        "SOURCE_GLOBAL_CACHE_SHA=c65552a6",
        "SOURCE_BLOCK_TABLE_SHA=eedb3f92",
        "SOURCE_LOGICAL_CACHE_SHA=3808d502",
        "source_cache_files.sha256",
        "remote_source_final_files.sha256",
        "capture_legacy_prompt_index_cache.py",
        "probe_layer0_prompt_index_cache.py",
        "greenfield_layer0_prompt_index_cache_comparison",
        "strict_census post",
        '"performance_claim": "false"',
        "TPU_VISIBLE_DEVICES=0,1,2,3",
    ):
        assert required in resume_source
    for forbidden in (
        "import tpu_inference",
        "from tpu_inference",
        "import vllm",
        "from vllm",
        'local label=$1 out=',
    ):
        assert forbidden not in resume_source
    for required in (
        "production_pallas_m1_divide_sqrt",
        "accepted_xla_m2048_divide_sqrt",
        "accepted_xla_m2048_multiply_rsqrt",
        "accepted_xla_m2048_chunk_parameter_divide_sqrt",
        "accepted_xla_m2048_chunk_bf16_weight_divide_sqrt",
        "accepted_xla_m2048_gather_chunk_bf16_weight_divide_sqrt",
        "--candidate-set",
        "layer0_prompt_index_key_gather_chunk",
        "layer0_prompt_index_keys_chunked",
        "validate_prompt_index_key_association_hlo",
        '"performance_claim": False',
    ):
        assert required in association_probe_source
    for required in (
        "SOURCE_RUN_ID=506",
        "SOURCE_ITEM_ROW_ID=1789",
        "SOURCE_CACHE_MANIFEST_SHA=d869f6cf",
        "SOURCE_COMPARISON_MANIFEST_SHA=b1822e71",
        "MATRIX_RUN_ID=507",
        "MATRIX_ASSOCIATION_MANIFEST_SHA=7216756c",
        "CHUNK_RUN_ID=508",
        "CHUNK_ITEM_ROW_ID=1793",
        "CHUNK_ASSOCIATION_MANIFEST_SHA=8539a81d",
        "BF16_RUN_ID=509",
        "BF16_ITEM_ROW_ID=1794",
        "BF16_ASSOCIATION_MANIFEST_SHA=df0b901e",
        "matrix_validation.json",
        "chunk_parameter_validation.json",
        "bf16_weight_validation.json",
        "chunk_bf16_weight",
        "chunk_gather_bf16_weight",
        "GLM_GREENFIELD_PROMPT_CACHE_ASSOCIATION_PROFILE",
        '--candidate-set "$PROFILE"',
        "probe_layer0_prompt_index_cache_association.py",
        "strict_census post",
        '"performance_claim": "false"',
    ):
        assert required in association_wrapper_source
    for forbidden in (
        "import tpu_inference",
        "from tpu_inference",
        "import vllm",
        "from vllm",
    ):
        assert forbidden not in association_probe_source
        assert forbidden not in association_wrapper_source
    assert 'local label=$1 out=' not in association_wrapper_source
    for script in (probe, association_probe):
        completed = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
