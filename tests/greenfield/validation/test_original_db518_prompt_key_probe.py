from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "scripts/greenfield/probe_original_db518_prompt_key_chunk0.py"
KERNEL = REPO / "glm_tpu/greenfield/kernels/reference/dsa_association.py"
ORIGINAL_PIN = "86243115452920fe4244bb77a9bbf4c44110aeab"
PRODUCER_SYMBOLS = (
    "Layer0DsaProbeGeometry",
    "_project_key_states_f32",
    "affine_key_layer_norm",
    "layer0_prompt_index_key_gather_cache_chunk",
    "layer0_prompt_index_key_gather_cache_states_chunk",
)


def _definitions(source: str) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _ast_sha256(node: ast.AST) -> str:
    return sha256(ast.dump(node,
                           include_attributes=False).encode()).hexdigest()


def test_current_producer_symbols_are_original_db518_source_exact():
    current = _definitions(KERNEL.read_text())
    historical = _definitions(
        subprocess.check_output(
            ["git", "show", f"{ORIGINAL_PIN}:{KERNEL.relative_to(REPO)}"],
            cwd=REPO,
            text=True,
        ))
    for name in PRODUCER_SYMBOLS:
        assert _ast_sha256(current[name]) == _ast_sha256(historical[name])


def test_probe_has_one_completed_producer_and_one_post_completion_transfer():
    source = PROBE.read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    compiled_invocations = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "compiled"
    ]
    device_gets = [
        node for node in calls if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "jax" and node.func.attr == "device_get"
    ]
    assert len(compiled_invocations) == 1
    assert len(device_gets) == 1
    assert source.index(
        "candidate_cache_device.block_until_ready()") < source.index(
            "candidate_cache_host, wk_host = jax.device_get(")
    assert source.index(
        "candidate_cache_device = compiled(*arguments)") < source.index(
            "candidate_cache_device.block_until_ready()")


def test_probe_binds_exact_full_chunk_original_db518_configuration():
    source = PROBE.read_text()
    for required in (
            "CHUNK_ROWS = 2048",
            "UNIQUE_ROWS = 37",
            "np.searchsorted(",
            'key_norm_mode="divide_sqrt"',
            'rotary_mode="accepted_source"',
            'projection_weight_mode="adapted_fp32"',
            'projection_mapping_mode="physical_m64_projection_keynorm_lax_map"',
            "EXPECTED_ADAPTED_WK_SHA256",
            "EXPECTED_CHUNK_BITS_SHA256",
            "validate_original_db518_prompt_key_hlo(optimized_hlo)",
            "validate_original_db518_prompt_key_stablehlo(",
    ):
        assert required in source
    assert "legacy_geometry_chunk_pipeline" not in source
    assert "layer1_keys_from_normalized" not in source
    assert "decode_batch1(" not in source


def test_probe_writes_both_hlo_forms_before_invocation():
    source = PROBE.read_text()
    invocation = source.index("candidate_cache_device = compiled(*arguments)")
    assert source.index("original_db518_chunk0.optimized_hlo.txt") < invocation
    assert source.index("original_db518_chunk0.stablehlo.mlir") < invocation
    assert source.index('"hlo/contract_failure.json"') < invocation
