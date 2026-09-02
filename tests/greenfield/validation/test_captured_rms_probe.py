from __future__ import annotations

import base64
import importlib.util
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import google_crc32c


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/greenfield/probe_layer0_captured_rms.py"
WRAPPER = REPO / "scripts/greenfield/run_layer0_projection_reduction_probe.sh"
REAL_CAPTURE = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
REAL_FAILED_CONTROL_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_CAPTURED_RMS_CONTROL_HLO",
        "/home/gianl/glm-run/"
        "greenfield_layer0_captured_rms_replay_20260813T210310956272256Z/"
        "hlo/control.optimized_hlo.txt",
    )
)
REAL_FAILED_CONTROL_HLO_SHA256 = (
    "fc208e238305cf112a69214944e0af47ab39ff51651d5b603cb9b88d7c45cecd"
)
REAL_ACCEPTED_SPLIT_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_CAPTURED_RMS_SPLIT_HLO",
        "/home/gianl/glm-run/"
        "greenfield_layer0_captured_rms_replay_20260813T211240092773251Z/"
        "hlo/accepted_split.optimized_hlo.txt",
    )
)
REAL_ACCEPTED_SPLIT_HLO_SHA256 = (
    "c8fdc9d63ff87644d4b6cd7525bf23c475b081c87624865e1f53179c3a168d55"
)
SPEC = importlib.util.spec_from_file_location("captured_rms_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _captured_rms_optimized_hlo() -> str:
    config = json.dumps(
        {
            "window_config": {
                "kernel_window_bounds": [],
                "output_window_bounds": ["2", "48"],
                "input_window_bounds": [],
                "iteration_bounds": ["2", "1"],
                "cost_model_type": "COST_MODEL_TYPE_INVALID",
                "is_mask": False,
                "pad_input_on_minor_dim": "0",
                "pad_output_on_minor_dim": "0",
            },
            "megacore_config": {
                "megacore_split_dim": "0",
                "megacore_allreduce_bytes": "4096",
            },
        },
        separators=(",", ":"),
    )
    return f"""HloModule captured, num_partitions=4, num_replicas=1

%rms (a: f32[32], b: f32[32]) -> f32[32] {{
  %a = f32[32] parameter(0)
  %b = f32[32] parameter(1)
  ROOT %r = f32[32] add(%a, %b)
}}

%rsqrt (a: f32[32]) -> f32[32] {{
  %a = f32[32] parameter(0)
  ROOT %r = f32[32] copy(%a)
}}

%output (a: bf16[1,6144], b: bf16[1,6144], c: bf16[1,6144], d: bf16[1,6144], e: bf16[1,6144]) -> bf16[1,6144] {{
  %a = bf16[1,6144] parameter(0)
  %b = bf16[1,6144] parameter(1)
  %c = bf16[1,6144] parameter(2)
  %d = bf16[1,6144] parameter(3)
  %e = bf16[1,6144] parameter(4)
  %x = bf16[1,6144] add(%a, %b)
  %y = bf16[1,6144] add(%x, %c)
  %z = bf16[1,6144] add(%y, %d)
  ROOT %zz = bf16[1,6144] add(%z, %e)
}}

ENTRY %main (p0: bf16[1,8,1,6144], p1: bf16[1,6144], p2: bf16[1,6144], p3: bf16[6144]) -> bf16[1,6144] {{
  %p0 = bf16[1,8,1,6144] parameter(0)
  %g = bf16[4,8,1,6144] all-gather(%p0), channel_id=1, replica_groups={{{{0,1,2,3}}}}, dimensions={{0}}, use_global_device_ids=true, metadata={{op_name="jit/local/greenfield_captured_rms_strategy_gather/all_gather"}}
  %p1 = bf16[1,6144] parameter(1)
  %p2 = bf16[1,6144] parameter(2)
  %s = f32[32] fusion(%g, %p1), kind=kLoop, calls=%rms, metadata={{op_name="jit/local/greenfield_captured_rms_layer1/split_reduction/reduce_sum"}}, backend_config={config}
  %r = f32[32]{{0:T(128)S(3)}} fusion(%s), kind=kLoop, calls=%rsqrt, metadata={{op_name="jit/local/greenfield_captured_rms_layer1/split_reduction/rsqrt"}}
  %rb = f32[1]{{0:T(128)S(3)}} bitcast(%r)
  %p3 = bf16[6144] parameter(3)
  ROOT %out = bf16[1,6144] fusion(%g, %p1, %p2, %p3, %rb), kind=kLoop, calls=%output
}}
"""


def test_captured_rms_source_builds_two_exact_stablehlo_arms() -> None:
    code = f"""
import json
import re
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, {str(REPO)!r})
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scripts.greenfield.probe_layer0_captured_rms import _build_arm
from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import validate_captured_dense_rms_stablehlo
mesh = Mesh(np.asarray(jax.devices()), ('lp4',))
partials = jax.device_put(np.zeros((4,8,1,6144), dtype=np.uint16).view(jax.numpy.bfloat16), NamedSharding(mesh, P('lp4',None,None,None)))
residual = jax.device_put(np.zeros((32,6144), dtype=np.uint16).view(jax.numpy.bfloat16), NamedSharding(mesh, P()))
combined = jax.device_put(np.zeros((1,6144), dtype=np.uint16).view(jax.numpy.bfloat16), NamedSharding(mesh, P()))
norm = jax.device_put(np.ones((6144,), dtype=np.float32).astype(jax.numpy.bfloat16), NamedSharding(mesh, P()))
result = {{}}
for name, split in (('control', False), ('accepted_split', True)):
    text = jax.jit(_build_arm(mesh, split_layer1_rms=split)).lower(partials, residual[:1], combined, norm).as_text()
    result[name] = validate_captured_dense_rms_stablehlo(text, split_layer1_rms=split)
    if split:
        gather_line = next(line for line in text.splitlines() if '"stablehlo.all_gather"' in line)
        residual_add_line = next(
            line
            for line in text.splitlines()
            if "stablehlo.add" in line
            and "tensor<32x6144xf32>" in line
        )
        residual_operands = re.findall(r"%[0-9]+", residual_add_line)[:3]
        assert len(residual_operands) == 3
        barrier_line = next(
            line
            for line in text.splitlines()
            if "stablehlo.optimization_barrier" in line
            and "tensor<32x6144xf32>" in line
        )
        barrier_operand = re.findall(r"%[0-9]+", barrier_line)[1]
        mutations = {{
            'wrong_reducer': text.replace('applies stablehlo.add', 'applies stablehlo.maximum', 1),
            'wrong_group': text.replace(
                'replica_groups = dense<[[0, 1, 2, 3]]>',
                'replica_groups = /* replica_groups = dense<[[0, 1, 2, 3]]> */ dense<[[0, 1, 3, 2]]>',
                1,
            ),
            'rogue_gather_input': text.replace(
                gather_line,
                '      %rogue = stablehlo.add %2, %2 : tensor<1x8x1x6144xbf16>\\n' + gather_line.replace('(%2)', '(%rogue)'),
                1,
            ),
            'wrong_outer_return': text.replace(
                '    return %0 : tensor<1x6144xbf16>',
                '    return %arg0 : tensor<1x6144xbf16>',
                1,
            ),
            'rounded_carry': text.replace(
                barrier_line,
                f'      %rc_b = stablehlo.convert {{barrier_operand}} : (tensor<32x6144xf32>) -> tensor<32x6144xbf16>\\n'
                f'      %rc_r = stablehlo.convert %rc_b : (tensor<32x6144xbf16>) -> tensor<32x6144xf32>\\n'
                + barrier_line.replace(f'{{barrier_operand}} :', '%rc_r :', 1),
                1,
            ),
            'duplicate_residual_source': text.replace(
                residual_add_line,
                residual_add_line.replace(
                    f'{{residual_operands[1]}}, {{residual_operands[2]}}',
                    f'{{residual_operands[1]}}, {{residual_operands[1]}}',
                ),
                1,
            ),
        }}
        result['refusals'] = {{
            key: validate_captured_dense_rms_stablehlo(value, split_layer1_rms=True)
            for key, value in mutations.items()
        }}
print(json.dumps(result))
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["control"]["passed"], result["control"]["violations"]
    assert result["accepted_split"]["passed"], (
        result["accepted_split"]["violations"]
    )
    assert all(
        not contract["passed"] for contract in result["refusals"].values()
    ), result["refusals"]


def test_captured_rms_mode_is_default_off() -> None:
    wrapper = WRAPPER.read_text()
    assert "CAPTURED_RMS_REPLAY=${GLM_GREENFIELD_CAPTURED_RMS_REPLAY:-0}" in wrapper


def test_captured_rms_mode_carries_canonical_protections() -> None:
    wrapper = WRAPPER.read_text()
    # Literal reviewed tag only: no auto-generated tag in this mode.
    assert "TAG=${GLM_GREENFIELD_CAPTURED_RMS_TAG:-}" in wrapper
    assert (
        "^greenfield_layer0_captured_rms_replay_[0-9]{8}T[0-9]{15}Z$" in wrapper
    )
    assert "GLM_GREENFIELD_CAPTURED_RMS_TAG:-greenfield_layer0_captured_rms_replay_$(date" not in wrapper
    # Sanitized Git authority and mirror replay.
    for marker in (
        "CAPTURED_RMS_GIT_AUTHORITY_VERIFIER",
        'git(Path("/"), "ls-remote", "--refs", origin, expected_ref)',
        "captured_rms_git_local status --porcelain --untracked-files=all",
        "verify_gate_d_same_region_git_mirror.py",
        "091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b",
    ):
        assert marker in wrapper, marker
    # Three-scope vacancy.
    for marker in (
        "gcloud storage ls --all-versions",
        "gcloud storage ls --soft-deleted --exhaustive",
        "append-only remote-prefix history is not canonically vacant",
    ):
        assert marker in wrapper, marker
    # Four leases held simultaneously.
    for marker in (
        "exec 9>/home/gianl/glm-run/.glm_pod_workload.lock",
        "exec 8>/home/gianl/.glm-tpu-rsync.lock",
        "exec 11>/opt/glm-tpu/locks/glm_pod_workload.lock",
        "exec 12>/opt/glm-tpu/locks/glm_tpu_rsync.lock",
        "CAPTURED_RMS_IMMUTABLE_LOCK_VERIFIER",
    ):
        assert marker in wrapper, marker
    # Runs from the tooling worktree the immutable mirror verifier is bound to.
    for marker in (
        "readonly CAPTURED_RMS_WORKTREE=/home/gianl/glm-tpu-gate-d-pp16-numerical",
        "readonly CAPTURED_RMS_BRANCH=tooling/gate-d-compensated-pp16-numerical",
        '--expected-worktree "$CAPTURED_RMS_WORKTREE"',
    ):
        assert marker in wrapper, marker
    probe = SCRIPT.read_text()
    assert '"--expected-worktree", type=Path, required=True' in probe
    assert "probe_layer0_captured_rms.py" in wrapper
    assert "NO_PROVISIONAL_DB_RUN" in wrapper
    assert "glm52_layer0_captured_rms_replay" in wrapper
    assert wrapper.index("validate_captured_rms_replay()") < wrapper.index(
        "started=$(date +%s)"
    )
    assert wrapper.index("  validate_captured_rms_replay\n") > wrapper.index(
        "post_census_done=1"
    )
    assert "performance_claim\": False" in SCRIPT.read_text()
    assert SCRIPT.read_text().index("arithmetic_diagnostic.json") < (
        SCRIPT.read_text().index("if not control_admissible:")
    )


def test_captured_rms_optimized_contract_refuses_decoys() -> None:
    accepted = _captured_rms_optimized_hlo()
    result = MODULE._validate_captured_rms_optimized_hlo(
        accepted, split_layer1_rms=True
    )
    assert result["passed"], result
    narrow_output = """%output (a: bf16[1,6144], b: bf16[1,6144], c: bf16[1,6144], d: bf16[1,6144], e: bf16[1,6144]) -> bf16[1,6144] {
  %a = bf16[1,6144] parameter(0)
  %b = bf16[1,6144] parameter(1)
  %c = bf16[1,6144] parameter(2)
  %d = bf16[1,6144] parameter(3)
  %e = bf16[1,6144] parameter(4)
  %x = bf16[1,6144] add(%a, %b)
  %y = bf16[1,6144] add(%x, %c)
  %z = bf16[1,6144] add(%y, %d)
  ROOT %zz = bf16[1,6144] add(%z, %e)
}"""
    wide_output = """%output (a: bf16[4,8,1,6144], b: bf16[1,6144], c: bf16[1,6144], d: bf16[6144], e: f32[1]) -> bf16[32,6144] {
  %a = bf16[4,8,1,6144] parameter(0)
  %flat = bf16[32,6144] reshape(%a)
  %b = bf16[1,6144] parameter(1)
  %bb = bf16[32,6144] broadcast(%b), dimensions={0,1}
  %c = bf16[1,6144] parameter(2)
  %cb = bf16[32,6144] broadcast(%c), dimensions={0,1}
  %d = bf16[6144] parameter(3)
  %db = bf16[32,6144] broadcast(%d), dimensions={1}
  %e = f32[1] parameter(4)
  %eb = f32[32,6144] broadcast(%e), dimensions={0}
  %ebc = bf16[32,6144] convert(%eb)
  %x = bf16[32,6144] add(%flat, %bb)
  %y = bf16[32,6144] add(%x, %cb)
  %z = bf16[32,6144] add(%y, %db)
  ROOT %zz = bf16[32,6144] add(%z, %ebc)
}"""
    wide = accepted.replace(narrow_output, wide_output, 1).replace(
        "  ROOT %out = bf16[1,6144] fusion(%g, %p1, %p2, %p3, %rb), kind=kLoop, calls=%output",
        "  %wide = bf16[32,6144] fusion(%g, %p1, %p2, %p3, %rb), kind=kLoop, calls=%output\n"
        "  ROOT %row0 = bf16[1,6144] slice(%wide), slice={[0:1],[0:6144]}",
        1,
    )
    row_zero = MODULE._validate_captured_rms_optimized_hlo(
        wide, split_layer1_rms=True
    )
    assert row_zero["passed"], row_zero
    mutations = (
        accepted.replace(
            "replica_groups={{0,1,2,3}}",
            'replica_groups={{0,1,3,2}}, metadata={op_name="replica_groups={{0,1,2,3}}"}',
            1,
        ),
        accepted.replace(
            "dimensions={0}",
            '/* dimensions={0} */ dimensions={1}, metadata={op_name="dimensions={0}"}',
            1,
        ),
        accepted.replace(
            "%rb = f32[1]{0:T(128)S(3)} bitcast(%r)",
            '%rb = f32[1]{0} bitcast(%r), metadata={op_name="= f32[1]{0:T(128)S(3)} bitcast("}',
            1,
        ),
        accepted.replace(
            "bitcast(%r)",
            "bitcast(%s)",
            1,
        ),
        accepted.replace(
            "fusion(%s), kind=kLoop, calls=%rsqrt",
            "fusion(%s, %s), kind=kLoop, calls=%rsqrt",
            1,
        ),
        accepted.replace(
            "%g = bf16[4,8,1,6144] all-gather(%p0)",
            "%slot = bf16[8,1,6144] slice(%p0), slice={[0:1],[0:8],[0:1],[0:6144]}\n"
            "  %g = bf16[4,8,1,6144] all-gather(%slot)",
            1,
        ),
        accepted.replace(
            "  %p3 = bf16[6144] parameter(3)",
            "  %dead_start = bf16[4,8,1,6144] all-reduce-start(%g), replica_groups={{0,1,2,3}}, to_apply=%output\n"
            "  %dead_done = bf16[4,8,1,6144] all-reduce-done(%dead_start)\n"
            "  %p3 = bf16[6144] parameter(3)",
            1,
        ),
        accepted.replace(
            "greenfield_captured_rms_layer1/split_reduction/reduce_sum",
            "not_the_live_rms/split_reduction/reduce_sum",
            1,
        ).replace(
            "  %p2 = bf16[6144] parameter(2)",
            "  %decoy = f32[32] fusion(%g, %p1), kind=kLoop, calls=%rms, metadata={op_name=\"greenfield_captured_rms_layer1/split_reduction/reduce_sum\"}, backend_config="
            + json.dumps(
                {
                    "window_config": {
                        "kernel_window_bounds": [],
                        "output_window_bounds": ["2", "48"],
                        "input_window_bounds": [],
                        "iteration_bounds": ["2", "1"],
                        "cost_model_type": "COST_MODEL_TYPE_INVALID",
                        "is_mask": False,
                        "pad_input_on_minor_dim": "0",
                        "pad_output_on_minor_dim": "0",
                    },
                    "megacore_config": {
                        "megacore_split_dim": "0",
                        "megacore_allreduce_bytes": "4096",
                    },
                },
                separators=(",", ":"),
            )
            + "\n  %p3 = bf16[6144] parameter(3)",
            1,
        ),
        accepted.replace(
            "fusion(%g, %p1, %p2, %p3, %rb)",
            "fusion(%g, %p1, %p1, %p3, %rb)",
            1,
        ),
        accepted.replace(
            "  ROOT %out = bf16[1,6144] fusion(%g, %p1, %p2, %p3, %rb), kind=kLoop, calls=%output",
            "  %out = bf16[1,6144] fusion(%g, %p1, %p2, %p3, %rb), kind=kLoop, calls=%output\n"
            "  ROOT %rogue = bf16[1,6144] add(%out, %out)",
            1,
        ),
        accepted.replace(
            "  %p3 = bf16[6144] parameter(3)",
            "  %rogue_r = f32[32] add(%r, %r)\n"
            "  %rogue_rb = f32[1]{0:T(128)S(3)} bitcast(%rogue_r)\n"
            "  %p3 = bf16[6144] parameter(3)",
            1,
        ).replace(
            "fusion(%g, %p1, %p2, %p3, %rb)",
            "fusion(%g, %p1, %p2, %p3, %rogue_rb)",
            1,
        ),
        wide.replace(
            "ROOT %row0 = bf16[1,6144] slice(%wide), slice={[0:1],[0:6144]}",
            "ROOT %row1 = bf16[1,6144] slice(%wide), "
            "/* slice={[0:1],[0:6144]} */ slice={[1:2],[0:6144]}, "
            'metadata={op_name="slice={[0:1],[0:6144]}"}',
            1,
        ),
    )
    assert all(
        not MODULE._validate_captured_rms_optimized_hlo(
            mutation, split_layer1_rms=True
        )["passed"]
        for mutation in mutations
    )


@pytest.mark.skipif(
    not REAL_FAILED_CONTROL_HLO.exists(),
    reason="protected captured-RMS control HLO is unavailable",
)
def test_retired_single_residual_control_hlo_refuses() -> None:
    hlo = REAL_FAILED_CONTROL_HLO.read_text()
    assert sha256(hlo.encode()).hexdigest() == REAL_FAILED_CONTROL_HLO_SHA256
    contract = MODULE._validate_captured_rms_optimized_hlo(
        hlo, split_layer1_rms=False
    )
    assert not contract["passed"], contract


@pytest.mark.skipif(
    not REAL_ACCEPTED_SPLIT_HLO.exists(),
    reason="protected captured-RMS accepted-split HLO is unavailable",
)
def test_retired_single_residual_split_hlo_refuses() -> None:
    hlo = REAL_ACCEPTED_SPLIT_HLO.read_text()
    assert sha256(hlo.encode()).hexdigest() == REAL_ACCEPTED_SPLIT_HLO_SHA256
    contract = MODULE._validate_captured_rms_optimized_hlo(
        hlo, split_layer1_rms=True
    )
    assert not contract["passed"], contract


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(), reason="sealed dense-partial capture is unavailable"
)
def test_captured_partials_reproduce_pinned_strategy_nd_update() -> None:
    with np.load(REAL_CAPTURE, allow_pickle=False) as payload:
        partials = np.ascontiguousarray(
            payload["dense_virtual_partials_bfloat16_bits"]
        )
    reduced = MODULE._reproduce_dense_update(partials)
    assert reduced.shape == (1, 6144)
    assert sha256(reduced.tobytes()).hexdigest() == MODULE._DENSE_UPDATE_SHA


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(), reason="sealed dense-partial capture is unavailable"
)
def test_captured_rms_wrapper_seals_without_a_db_row(tmp_path: Path) -> None:
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", WRAPPER.read_text(), re.DOTALL)
    record_program = next(
        value for value in programs if "CAPTURED_RMS_REPLAY_VALID" in value
    )
    success_program = next(
        value
        for value in programs
        if "captured RMS replay summary drifted before SUCCESS" in value
    )
    run_dir = tmp_path / "replay"
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir(parents=True)
    with np.load(REAL_CAPTURE, allow_pickle=False) as payload:
        accepted = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
    db548_path = Path(
        "/home/gianl/glm-run/"
        "greenfield_layer0_dense_envelope_cross_layer_"
        "20260813T120703034434907Z/dense_envelope_cross_layer.npz"
    )
    with np.load(db548_path, allow_pickle=False) as payload:
        control = np.ascontiguousarray(
            payload["layer1_normalized_bfloat16_bits"]
        )
    source_hashes = [f"{index:064x}" for index in range(1, 15)]
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
    )
    source = dict(zip(source_names, source_hashes, strict=True))
    pin = "a" * 40
    arms = {}
    for name, split, output, reference in (
        ("control", False, control, control),
        ("accepted_split", True, accepted, accepted),
    ):
        stable_path = hlo_dir / f"{name}.stablehlo.mlir"
        optimized_path = hlo_dir / f"{name}.optimized_hlo.txt"
        stable_path.write_text(f"{name} stable\n")
        optimized_path.write_text(f"{name} optimized\n")
        arms[name] = {
            "comparison": MODULE._compare_bits(reference, output),
            "hlo": {
                "optimized_contract": {
                    "accepted_scheduled_reduction_values": (
                        ["%accepted"] if split else []
                    ),
                    "async_collectives": [],
                    "collective_count": 1,
                    "exact_output_fusion": True,
                    "exact_result_binding": True,
                    "exact_scheduled_reduction_binding": True,
                    "live_rows": 1,
                    "num_partitions": 4,
                    "num_replicas": 1,
                    "passed": True,
                    "performance_claim": False,
                    "split_layer1_rms": split,
                    "violations": [],
                },
                "optimized_sha256": sha256(
                    optimized_path.read_bytes()
                ).hexdigest(),
                "stablehlo_contract": {
                    "collective_counts": {
                        "all_gather": 1,
                        "all_reduce": 0,
                        "all_to_all": 0,
                        "collective_broadcast": 0,
                        "collective_permute": 0,
                        "reduce_scatter": 0,
                    },
                    "exact_result_binding": True,
                    "live_rows": 1,
                    "passed": True,
                    "split_layer1_rms": split,
                    "violations": [],
                },
                "stablehlo_sha256": sha256(
                    stable_path.read_bytes()
                ).hexdigest(),
            },
            "output_sha256": sha256(output.tobytes()).hexdigest(),
            "split_layer1_rms": split,
        }
    runner = {
        "artifact_kind": "glm52_layer0_captured_rms_replay",
        "arms": arms,
        "classification": "captured_partials_split_rms_exact",
        "code_hash": pin,
        "control_admissible": True,
        "dense_update_sha256": MODULE._DENSE_UPDATE_SHA,
        "exact": True,
        "exact_arms": ["accepted_split"],
        "live_rows": 1,
        "performance_claim": False,
        "position": 8155,
        "source": source,
        "status": "SUCCESS",
    }
    (run_dir / "runner.json").write_text(json.dumps(runner))
    np.savez(
        run_dir / "captured_rms_replay.npz",
        accepted_layer1_normalized_bfloat16_bits=accepted,
        control_layer1_normalized_bfloat16_bits=control,
        db548_layer1_normalized_bfloat16_bits=control,
        split_layer1_normalized_bfloat16_bits=accepted,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(run_dir),
            pin,
            "7",
            "captured_rms_test",
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
    assert summary["exact_arms"] == ["accepted_split"]

    original_runner_text = (run_dir / "runner.json").read_text()
    runner["arms"]["control"]["comparison"].update(
        mismatch_count=False,
        max_abs_error=False,
        mean_abs_error=False,
    )
    (run_dir / "runner.json").write_text(json.dumps(runner))
    refused = subprocess.run(
        [
            sys.executable,
            "-",
            str(run_dir),
            pin,
            "7",
            "captured_rms_test",
            *source_hashes,
        ],
        input=record_program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    (run_dir / "runner.json").write_text(original_runner_text)

    evidence_paths = [
        *sorted(hlo_dir.iterdir()),
        run_dir / "captured_rms_replay.npz",
        run_dir / "runner.json",
        run_dir / "summary.json",
    ]
    (run_dir / "evidence.sha256").write_text(
        "".join(
            f"{sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(run_dir).as_posix()}\n"
            for path in evidence_paths
        )
    )

    def crc32c(path: Path) -> str:
        checksum = google_crc32c.Checksum(path.read_bytes())
        return base64.b64encode(checksum.digest()).decode()

    ledger_paths = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name not in {"SUCCESS", "remote_objects.json"}
    )
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
                    for index, path in enumerate(ledger_paths)
                ]
            }
        )
    )
    original_summary_text = (run_dir / "summary.json").read_text()
    forged_summary = json.loads(original_summary_text)
    forged_summary["source"] = {"forged": "0" * 64}
    forged_summary["arms"]["control"]["hlo"]["optimized_sha256"] = (
        "f" * 64
    )
    (run_dir / "summary.json").write_text(json.dumps(forged_summary))
    refused = subprocess.run(
        [sys.executable, "-", str(run_dir), "gs://test/replay", pin],
        input=success_program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    (run_dir / "summary.json").write_text(original_summary_text)
    completed = subprocess.run(
        [sys.executable, "-", str(run_dir), "gs://test/replay", pin],
        input=success_program,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    terminal = dict(
        line.split("=", 1)
        for line in (run_dir / "SUCCESS").read_text().splitlines()
    )
    assert terminal["results_db_run_id"] == "none"
    assert terminal["exact"] == "true"
    assert terminal["exact_arms"] == "accepted_split"
