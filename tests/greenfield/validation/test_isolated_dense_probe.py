from __future__ import annotations

import importlib.util
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import google_crc32c


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/greenfield/probe_layer0_isolated_dense.py"
WRAPPER = REPO / "scripts/greenfield/run_layer0_projection_reduction_probe.sh"
REAL_CAPTURE = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
REAL_DB548 = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_envelope_cross_layer_"
    "20260813T120703034434907Z/dense_envelope_cross_layer.npz"
)
SPEC = importlib.util.spec_from_file_location("isolated_dense_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _stablehlo(*, bit_dtype: str = "jnp.uint8") -> str:
    program = f'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scripts.greenfield.probe_layer0_isolated_dense import _build_isolated_partial
mesh = Mesh(np.asarray(jax.devices()), ("lp4",))
rep = NamedSharding(mesh, P())
slot = NamedSharding(mesh, P("lp4", None, None, None))
contracts = (
    ((1,6144), jnp.bfloat16, rep),
    ((1,6144), jnp.bfloat16, rep),
    ((6144,), jnp.bfloat16, rep),
    ((4,1,6144,768), {bit_dtype}, slot),
    ((4,1,48,768), jnp.float32, slot),
    ((4,1,384,6144), {bit_dtype}, slot),
    ((4,1,3,6144), jnp.float32, slot),
)
arguments = tuple(
    jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    for shape, dtype, sharding in contracts
)
print(jax.jit(_build_isolated_partial(mesh)).lower(*arguments).as_text())
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _config(label: str) -> str:
    expected = (
        {
            "kernel_window_bounds": ["384", "6"],
            "input_window_bounds": ["4", "24"],
            "output_window_bounds": ["4", "6"],
            "iteration_bounds": ["1", "1", "2"],
            "megacore_split_dim": "2",
            "megacore_allreduce_bytes": "98304",
        }
        if label == "gate"
        else {
            "kernel_window_bounds": ["48", "6"],
            "input_window_bounds": ["4", "3"],
            "output_window_bounds": ["4", "6"],
            "iteration_bounds": ["8", "1", "1"],
            "megacore_split_dim": "0",
            "megacore_allreduce_bytes": None,
        }
    )
    return json.dumps(
        {
            "convolution_algorithm_config": {
                "emitter": "EmitAllBatchInSublanes"
            },
            "window_config": {
                "kernel_window_bounds": expected["kernel_window_bounds"],
                "input_window_bounds": expected["input_window_bounds"],
                "output_window_bounds": expected["output_window_bounds"],
                "iteration_bounds": expected["iteration_bounds"],
                "cost_model_type": "COST_MODEL_TYPE_CLASSIC",
                "is_mask": False,
                "pad_input_on_minor_dim": "0",
                "pad_output_on_minor_dim": "0",
            },
            "megacore_config": {
                "megacore_split_dim": expected["megacore_split_dim"],
                "megacore_allreduce_bytes": expected[
                    "megacore_allreduce_bytes"
                ],
            },
        },
        separators=(",", ":"),
    )


def _optimized_hlo() -> str:
    return f'''HloModule isolated_dense, is_scheduled=true, replica_count=1, num_partitions=4

ENTRY main {{
  %attention = bf16[1,6144]{{1,0}} parameter(0)
  %residual = bf16[1,6144]{{1,0}} parameter(1)
  %norm = bf16[6144]{{0}} parameter(2)
  %zero = bf16[] constant(0)
  %attention_pad = bf16[32,6144]{{1,0}} pad(%attention, %zero), padding=0_31x0_0
  %residual_pad = bf16[32,6144]{{1,0}} pad(%residual, %zero), padding=0_31x0_0
  %attention_wide = f32[32,6144]{{1,0}} convert(%attention_pad)
  %residual_wide = f32[32,6144]{{1,0}} convert(%residual_pad)
  %combined = f32[32,6144]{{1,0}} add(%attention_wide, %residual_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_predense_rmsnorm/add"}}
  %carried = bf16[32,6144]{{1,0}} convert(%combined), metadata={{op_name="jit/local/greenfield_dense_convolution_predense_rmsnorm/convert_element_type"}}
  %gate_bits = f8e4m3fn[1,1,6144,768]{{3,2,1,0}} parameter(3)
  %gate_scale = f32[1,1,48,768]{{3,2,1,0}} parameter(4)
  %down_bits = f8e4m3fn[1,1,384,6144]{{3,2,1,0}} parameter(5)
  %down_scale = f32[1,1,3,6144]{{3,2,1,0}} parameter(6)
  %gate_bits_slice = f8e4m3fn[1,1,6144,768]{{3,2,1,0}} slice(%gate_bits), slice={{[0:1],[0:1],[0:6144],[0:768]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %gate_bits_flat = f8e4m3fn[6144,768]{{1,0}} reshape(%gate_bits_slice), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %gate_decoded = f32[6144,768]{{1,0}} convert(%gate_bits_flat), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}}
  %gate_scale_slice = f32[1,1,48,768]{{3,2,1,0}} slice(%gate_scale), slice={{[0:1],[0:1],[0:48],[0:768]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %gate_scale_seed = f32[48,768]{{1,0}} reshape(%gate_scale_slice), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %gate_scale_expanded = f32[48,128,768]{{2,1,0}} broadcast(%gate_scale_seed), dimensions={{0,2}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/broadcast_in_dim"}}
  %gate_scale_wide = f32[6144,768]{{1,0}} reshape(%gate_scale_expanded), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %gate_scaled = f32[6144,768]{{1,0}} multiply(%gate_decoded, %gate_scale_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}}
  %gate_rhs = bf16[6144,768]{{1,0}} convert(%gate_scaled), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}}
  %gate = f32[32,768]{{1,0}} convolution(%carried, %gate_rhs), dim_labels=bf_io->bf, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/conv_general_dilated"}}, backend_config={_config("gate")}
  %gate_round = bf16[32,768]{{1,0}} convert(%gate)
  %gate_slice = bf16[32,384]{{1,0}} slice(%gate_round), slice={{[0:32],[0:384]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %up_slice = bf16[32,384]{{1,0}} slice(%gate_round), slice={{[0:32],[384:768]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %gate_wide = f32[32,384]{{1,0}} convert(%gate_slice)
  %up_wide = f32[32,384]{{1,0}} convert(%up_slice)
  %one = bf16[] constant(1)
  %one_broadcast = bf16[32,384]{{1,0}} broadcast(%one), dimensions={{}}
  %one_wide = f32[32,384]{{1,0}} convert(%one_broadcast)
  %negate = f32[32,384]{{1,0}} negate(%gate_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/neg"}}
  %exponential = f32[32,384]{{1,0}} exponential(%negate), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/exp"}}
  %denominator = f32[32,384]{{1,0}} add(%exponential, %one_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/add"}}
  %sigmoid = f32[32,384]{{1,0}} divide(%one_wide, %denominator), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/div"}}
  %silu = f32[32,384]{{1,0}} multiply(%gate_wide, %sigmoid), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}}, backend_config={{"float_type_correction_info":{{"original_type":"BF16"}}}}
  %activated = f32[32,384]{{1,0}} multiply(%silu, %up_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}}, backend_config={{"float_type_correction_info":{{"original_type":"BF16"}}}}
  %activated_round = bf16[32,384]{{1,0}} convert(%activated)
  %down_bits_slice = f8e4m3fn[1,1,384,6144]{{3,2,1,0}} slice(%down_bits), slice={{[0:1],[0:1],[0:384],[0:6144]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %down_bits_flat = f8e4m3fn[384,6144]{{1,0}} reshape(%down_bits_slice), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %down_decoded = f32[384,6144]{{1,0}} convert(%down_bits_flat), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}}
  %down_scale_slice = f32[1,1,3,6144]{{3,2,1,0}} slice(%down_scale), slice={{[0:1],[0:1],[0:3],[0:6144]}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/slice"}}
  %down_scale_seed = f32[3,6144]{{1,0}} reshape(%down_scale_slice), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %down_scale_expanded = f32[3,128,6144]{{2,1,0}} broadcast(%down_scale_seed), dimensions={{0,2}}, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/broadcast_in_dim"}}
  %down_scale_wide = f32[384,6144]{{1,0}} reshape(%down_scale_expanded), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/reshape"}}
  %down_scaled = f32[384,6144]{{1,0}} multiply(%down_decoded, %down_scale_wide), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/mul"}}
  %down_rhs = bf16[384,6144]{{1,0}} convert(%down_scaled), metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/convert_element_type"}}
  %down = f32[32,6144]{{1,0}} convolution(%activated_round, %down_rhs), dim_labels=bf_io->bf, metadata={{op_name="jit/local/greenfield_dense_convolution_virtual_rank_00/conv_general_dilated"}}, backend_config={_config("down")}
  %partial = bf16[32,6144]{{1,0}} convert(%down)
  %stack = bf16[1,32,6144]{{2,1,0}} reshape(%partial)
  %barrier = bf16[1,32,6144]{{2,1,0}} optimization-barrier(%stack)
  %live = bf16[1,1,6144]{{2,1,0}} slice(%barrier), slice={{[0:1],[0:1],[0:6144]}}
  ROOT %result = (bf16[1,1,6144]{{2,1,0}}, bf16[32,6144]{{1,0}}) tuple(%live, %carried)
}}
'''


def test_isolated_dense_stablehlo_contract_accepts_u8_and_fp8() -> None:
    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_isolated_dense_partial_stablehlo,
    )

    for bit_dtype, expected_bitcasts in (
        ("jnp.uint8", 2),
        ("jnp.float8_e4m3fn", 0),
    ):
        stablehlo = _stablehlo(bit_dtype=bit_dtype)
        result = validate_isolated_dense_partial_stablehlo(stablehlo)
        assert result["passed"], result
        assert result["convolution_count"] == 2
        assert result["matched_virtual_shards"] == [0]
        assert result["runtime_u8_bitcast_count"] == expected_bitcasts
        assert result["collective_counts"] == {
            "all_gather": 0,
            "all_reduce": 0,
            "all_to_all": 0,
            "collective_broadcast": 0,
            "collective_permute": 0,
            "reduce_scatter": 0,
        }


def test_isolated_dense_stablehlo_contract_refuses_wrong_arithmetic() -> None:
    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_isolated_dense_partial_stablehlo,
    )

    stablehlo = _stablehlo()
    mutations = (
        stablehlo.replace("applies stablehlo.add", "applies stablehlo.maximum", 1),
        stablehlo.replace("[0:1, 0:1, 0:6144]", "[0:1, 1:2, 0:6144]", 1),
        stablehlo.replace("stablehlo.multiply %46, %39", "stablehlo.add %46, %39", 1),
        stablehlo.replace("sdy.return %59, %6", "sdy.return %59, %20", 1),
    )
    assert all(
        not validate_isolated_dense_partial_stablehlo(value)["passed"]
        for value in mutations
    )


def test_isolated_dense_optimized_contract_pins_schedule_and_liveness() -> None:
    accepted = _optimized_hlo()
    result = MODULE._validate_isolated_optimized_hlo(accepted)
    assert result["passed"], result
    assert result["exact_activation_graph"]
    assert result["exact_carried_residual_binding"]
    assert result["exact_packed_weight_lineage"]
    assert result["exact_result_binding"]
    mutations = (
        accepted.replace('"iteration_bounds":["1","1","2"]', '"iteration_bounds":["1","1","3"]', 1),
        accepted.replace(
            "  %down = f32[32,6144]",
            "  %rogue_activation = bf16[32,384]{1,0} add(%activated_round, %activated_round)\n"
            "  %down = f32[32,6144]",
            1,
        ).replace("convolution(%activated_round, %down_rhs)", "convolution(%rogue_activation, %down_rhs)", 1),
        accepted.replace(
            "  %partial = bf16[32,6144]",
            "  %rogue_down = f32[32,6144]{1,0} add(%down, %down)\n"
            "  %partial = bf16[32,6144]",
            1,
        ).replace("convert(%down)", "convert(%rogue_down)", 1),
        accepted.replace(
            "  ROOT %result",
            "  %rogue_live = bf16[1,1,6144]{2,1,0} add(%live, %live)\n"
            "  ROOT %result",
            1,
        ).replace("tuple(%live, %carried)", "tuple(%rogue_live, %carried)", 1),
        accepted.replace(
            "(bf16[1,1,6144]{2,1,0}, bf16[32,6144]{1,0}) tuple(%live, %carried)",
            "(bf16[1,1,6144]{2,1,0}, bf16[32,6144]{1,0}, bf16[32,6144]{1,0}) tuple(%live, %carried, %carried)",
            1,
        ),
        accepted.replace(
            "%gate_rhs = bf16[6144,768]{1,0} convert",
            "%gate_rhs = bf16[6144,768]{0,1} convert",
            1,
        ),
        accepted.replace(
            "  %gate = f32[32,768]",
            "  %rogue_gate_rhs = bf16[6144,768]{1,0} add(%gate_rhs, %gate_rhs)\n"
            "  %gate = f32[32,768]",
            1,
        ).replace("convolution(%carried, %gate_rhs)", "convolution(%carried, %rogue_gate_rhs)", 1),
        accepted.replace(
            "%down_rhs = bf16[384,6144]{1,0} convert",
            "%down_rhs = bf16[384,6144]{0,1} convert",
            1,
        ),
        accepted.replace(
            "  %down = f32[32,6144]",
            "  %rogue_down_rhs = bf16[384,6144]{1,0} add(%down_rhs, %down_rhs)\n"
            "  %down = f32[32,6144]",
            1,
        ).replace("convolution(%activated_round, %down_rhs)", "convolution(%activated_round, %rogue_down_rhs)", 1),
        accepted.replace(
            "%live = bf16[1,1,6144]{2,1,0} slice(%barrier), slice={[0:1],[0:1],[0:6144]}",
            "%live = bf16[1,1,6144]{2,1,0} slice(%barrier), slice={[0:1],[1:2],[0:6144]}",
            1,
        ),
        accepted.replace(
            "tuple(%live, %carried)",
            "tuple(%live, %attention_pad)",
            1,
        ),
        accepted.replace(
            "%zero = bf16[] constant(0)",
            '%zero = bf16[] constant(1), metadata={op_name="constant(0)"}',
            1,
        ),
        accepted.replace(
            "%one = bf16[] constant(1)",
            '%one = bf16[] constant(0), metadata={op_name="constant(1)"}',
            1,
        ),
        accepted.replace(
            "  ROOT %result",
            "  %rogue_carried = bf16[32,6144]{1,0} add(%carried, %carried)\n"
            "  ROOT %result",
            1,
        ).replace(
            "tuple(%live, %carried)",
            "tuple(%live, %rogue_carried)",
            1,
        ),
        accepted.replace(
            "  ROOT %result",
            "  %extra = bf16[32,6144]{1,0} all-reduce(%partial), replica_groups={{0,1,2,3}}, to_apply=%add\n"
            "  ROOT %result",
            1,
        ),
    )
    assert all(
        not MODULE._validate_isolated_optimized_hlo(value)["passed"]
        for value in mutations
    )


def test_isolated_partial_comparison_reports_each_virtual_rank() -> None:
    reference = np.zeros((4, 8, 1, 6144), dtype=np.uint16)
    observed = reference.copy()
    observed[2, 5, 0, 2795] = 1
    result = MODULE._partial_comparison(reference, observed)
    assert not result["elementwise_exact"]
    assert result["mismatch_count"] == 1
    assert result["first_mismatch_index"] == [2, 5, 0, 2795]
    assert result["per_virtual_rank"][5] == {
        "mismatch_count": 1,
        "virtual_rank": 5,
    }


def test_isolated_dense_wrapper_is_default_off_and_no_db() -> None:
    wrapper = WRAPPER.read_text()
    assert "GLM_GREENFIELD_ISOLATED_DENSE_REPLAY:-0" in wrapper
    assert "probe_layer0_isolated_dense.py" in wrapper
    assert "bounded replays are exclusive" in wrapper
    assert wrapper.index("validate_isolated_dense_replay()") < wrapper.index(
        'if [[ $ISOLATED_DENSE_REPLAY == 1 ]]; then\n'
        "  validate_isolated_dense_replay"
    )
    rollback = wrapper[wrapper.index("rollback_provisional_db()") :]
    assert (
        'if [[ $CAPTURED_RMS_REPLAY == 1 || '
        '$ISOLATED_DENSE_REPLAY == 1 ]]; then' in rollback
    )
    assert "NO_PROVISIONAL_DB_RUN" in rollback


@pytest.mark.skipif(
    not REAL_CAPTURE.exists() or not REAL_DB548.exists(),
    reason="sealed isolated-dense sources are unavailable",
)
def test_isolated_dense_wrapper_validates_exact_synthetic_result(
    tmp_path: Path,
) -> None:
    import re

    programs = re.findall(
        r"<<'PY'\n(.*?)\nPY\n", WRAPPER.read_text(), re.DOTALL
    )
    record_program = next(
        value for value in programs if "ISOLATED_DENSE_REPLAY_VALID" in value
    )
    run_dir = tmp_path / "isolated"
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir(parents=True)
    with np.load(REAL_CAPTURE, allow_pickle=False) as payload:
        accepted = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
        captured = np.ascontiguousarray(
            payload["dense_virtual_partials_bfloat16_bits"]
        )
    with np.load(REAL_DB548, allow_pickle=False) as payload:
        control = np.ascontiguousarray(
            payload["layer1_normalized_bfloat16_bits"]
        )
    for name in ("isolated_dense", "layer1_replay"):
        (hlo_dir / f"{name}.stablehlo.mlir").write_text(f"{name} stable\n")
        (hlo_dir / f"{name}.optimized_hlo.txt").write_text(
            f"{name} optimized\n"
        )
    records = MODULE._FINAL_DENSE_LAYOUT_RECORDS
    zero_collectives = {
        "all_gather": 0,
        "all_reduce": 0,
        "all_to_all": 0,
        "collective_broadcast": 0,
        "collective_permute": 0,
        "reduce_scatter": 0,
    }
    isolated_stable = {
        "collective_counts": zero_collectives,
        "convolution_count": 2,
        "exact_result_binding": True,
        "gate_up_layout_constraint_count": 1,
        "matched_virtual_shards": [0],
        "runtime_u8_bitcast_count": 0,
        "virtual_contractions_per_chip": 1,
        "passed": True,
        "violations": [],
    }
    isolated_optimized = {
        "accepted_down_schedule": True,
        "accepted_gate_up_schedule": True,
        "async_collectives": [],
        "collective_count": 0,
        "compile_rows": 32,
        "convolution_count": 2,
        "dense_envelope": True,
        "down_convolution_count": 1,
        "exact_accepted_kernel_geometry": True,
        "exact_accepted_weight_layout": True,
        "exact_activation_graph": True,
        "exact_carried_residual_binding": True,
        "exact_packed_weight_lineage": True,
        "exact_result_binding": True,
        "final_dense_layout": True,
        "gate_up_convolution_count": 1,
        "isolated_dense": True,
        "live_rows": 1,
        "num_partitions": 4,
        "num_replicas": 1,
        "performance_claim": False,
        "passed": True,
        "violations": [],
    }
    rms_stable = {
        "collective_counts": {**zero_collectives, "all_gather": 1},
        "exact_result_binding": True,
        "live_rows": 1,
        "split_layer1_rms": False,
        "passed": True,
        "violations": [],
    }
    rms_optimized = {
        "accepted_scheduled_reduction_values": [],
        "async_collectives": [],
        "collective_count": 1,
        "exact_output_fusion": True,
        "exact_result_binding": True,
        "exact_scheduled_reduction_binding": True,
        "live_rows": 1,
        "num_partitions": 4,
        "num_replicas": 1,
        "performance_claim": False,
        "split_layer1_rms": False,
        "passed": True,
        "violations": [],
    }
    source_names = (
        "capture_tensor_sha256",
        "capture_runner_sha256",
        "capture_summary_sha256",
        "capture_success_sha256",
        "db548_tensor_sha256",
        "db548_runner_sha256",
        "db548_summary_sha256",
        "db548_success_sha256",
        "db548_hlo_sha256",
        "db549_tensor_sha256",
        "db549_runner_sha256",
        "db549_summary_sha256",
        "db549_success_sha256",
        "db549_hlo_sha256",
        "checkpoint_manifest_sha256",
    )
    source_hashes = [f"{index:064x}" for index in range(1, 16)]
    ordered_values = dict(zip(source_names, source_hashes, strict=True))
    source = {
        "capture_runner_sha256": ordered_values["capture_runner_sha256"],
        "capture_summary_sha256": ordered_values["capture_summary_sha256"],
        "capture_success_sha256": ordered_values["capture_success_sha256"],
        "capture_tensor_sha256": ordered_values["capture_tensor_sha256"],
        "checkpoint_manifest_sha256": ordered_values[
            "checkpoint_manifest_sha256"
        ],
        "db548_hlo_sha256": ordered_values["db548_hlo_sha256"],
        "db548_runner_sha256": ordered_values["db548_runner_sha256"],
        "db548_summary_sha256": ordered_values["db548_summary_sha256"],
        "db548_success_sha256": ordered_values["db548_success_sha256"],
        "db548_tensor_sha256": ordered_values["db548_tensor_sha256"],
        "db549_hlo_sha256": ordered_values["db549_hlo_sha256"],
        "db549_runner_sha256": ordered_values["db549_runner_sha256"],
        "db549_summary_sha256": ordered_values["db549_summary_sha256"],
        "db549_success_sha256": ordered_values["db549_success_sha256"],
        "db549_tensor_sha256": ordered_values["db549_tensor_sha256"],
    }
    partial_comparison = MODULE._partial_comparison(captured, captured)
    comparison = MODULE._compare_bits(accepted, accepted)
    pin = "a" * 40
    runner = {
        "artifact_kind": "glm52_layer0_isolated_dense_replay",
        "classification": "isolated_virtual_contractions_exact",
        "code_hash": pin,
        "control_admissible": True,
        "exact": True,
        "exact_arms": ["isolated_virtual_contractions"],
        "final_layout_records": records,
        "hlo": {},
        "isolated_layer1_comparison": comparison,
        "isolated_layer1_sha256": MODULE._array_sha256(accepted),
        "partial_comparison": partial_comparison,
        "performance_claim": False,
        "position": 8155,
        "source": source,
        "status": "SUCCESS",
        "virtual_contractions_per_chip": 1,
        "virtual_rank_batches": 8,
    }
    for name, stable, optimized in (
        ("isolated_dense", isolated_stable, isolated_optimized),
        ("layer1_replay", rms_stable, rms_optimized),
    ):
        runner["hlo"][name] = {
            "optimized_contract": optimized,
            "optimized_sha256": MODULE._array_sha256(
                np.frombuffer(
                    (hlo_dir / f"{name}.optimized_hlo.txt").read_bytes(),
                    dtype=np.uint8,
                )
            ),
            "stablehlo_contract": stable,
            "stablehlo_sha256": MODULE._array_sha256(
                np.frombuffer(
                    (hlo_dir / f"{name}.stablehlo.mlir").read_bytes(),
                    dtype=np.uint8,
                )
            ),
        }
    (run_dir / "runner.json").write_text(json.dumps(runner))
    np.savez(
        run_dir / "isolated_dense_replay.npz",
        accepted_layer1_normalized_bfloat16_bits=accepted,
        captured_dense_virtual_partials_bfloat16_bits=captured,
        control_layer1_normalized_bfloat16_bits=control,
        isolated_dense_virtual_partials_bfloat16_bits=captured,
        isolated_layer1_normalized_bfloat16_bits=accepted,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(run_dir),
            pin,
            "5",
            "isolated_dense_test",
            *source_hashes,
        ],
        input=record_program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["results_db_run_id"] is None
    assert summary["classification"] == "isolated_virtual_contractions_exact"

    success_program = next(
        value
        for value in programs
        if "isolated dense replay summary drifted before SUCCESS" in value
    )
    evidence_paths = [
        run_dir / "runner.json",
        run_dir / "isolated_dense_replay.npz",
        run_dir / "summary.json",
        *(hlo_dir / f"{name}.{suffix}"
          for name in ("isolated_dense", "layer1_replay")
          for suffix in ("optimized_hlo.txt", "stablehlo.mlir")),
    ]
    (run_dir / "evidence.sha256").write_text(
        "".join(
            f"{sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(run_dir).as_posix()}\n"
            for path in evidence_paths
        )
    )

    def crc32c(path: Path) -> str:
        checksum = google_crc32c.Checksum()
        checksum.update(path.read_bytes())
        return base64.b64encode(checksum.digest()).decode()

    ledger_paths = [
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name not in {"SUCCESS", "remote_objects.json"}
    ]
    (run_dir / "remote_objects.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "crc32c": crc32c(path),
                        "generation": str(index + 1),
                        "path": path.relative_to(run_dir).as_posix(),
                        "size": path.stat().st_size,
                    }
                    for index, path in enumerate(sorted(ledger_paths))
                ]
            }
        )
    )
    sealed = subprocess.run(
        [
            sys.executable,
            "-",
            str(run_dir),
            "gs://driftbench-dsv4-uc/results/isolated_dense_test",
            pin,
        ],
        input=success_program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert sealed.returncode == 0, sealed.stderr
    success = (run_dir / "SUCCESS").read_text()
    assert "artifact_kind=glm52_layer0_isolated_dense_replay\n" in success
    assert "results_db_run_id=none\n" in success
