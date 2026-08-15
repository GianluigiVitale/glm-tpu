from __future__ import annotations

from hashlib import sha256
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[3]
REAL_TOPOLOGY_ROOT = Path(
    "/home/gianl/gcs-models/results/"
    "greenfield_topology_20260805T125842425591441Z/host_records"
)
REAL_TOPOLOGY_SHA256 = (
    "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
)
REAL_TOPOLOGY_FLEET_SHA256 = (
    "50de0729c9e5080c5ddb5ae4f5cd948317c53ce8a8f6c9f3f6064e7afc6515a0"
)
REAL_WS32_HLO_ROOT = Path(
    os.environ.get(
        "GLM_TEST_WS32_REAL_LAYER_HLO_ROOT",
        "/home/gianl/glm-run/"
        "greenfield_ws32_real_layer_hlo_20260815T081128291986777Z/fleet_hlo",
    )
)
REAL_WS32_STABLEHLO_SHA256 = (
    "dea1384e0b6548c9ab81b5bafb84aceb3353cce1db7f5e21282742990295c01f"
)
REAL_WS32_OPTIMIZED_HLO_SHA256 = (
    "6a6acf94b92a0ad355fc824d3f62d58b562d9bb252e7bbcbafba1bead2f3e157"
)
REAL_WS32_ACQUISITION_ROOT = REAL_WS32_HLO_ROOT.parent
REAL_WS32_ARTIFACT_ROOT = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/layer3/WS32_2D/"
    "greenfield_ws32_one_layer_pack_20260815T070628458699950Z"
)
REAL_WS32_ARTIFACT_MANIFEST_SHA256 = (
    "4bf8679de10ebbba055e9d0be991495080388c6355fa449f393c28f4751e1f40"
)
REAL_WS32_ORACLE_ROOT = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/layer3/"
    "greenfield_one_layer_oracle_20260805T162210370718434Z"
)
REAL_WS32_ORACLE_MANIFEST_SHA256 = (
    "c63ffa19820d5c2c39865ac8611fb313ffc6ebcd2f893c3507745a356bfdebff"
)
REAL_WS32_MESH_SHA256 = (
    "de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88"
)


def test_ws32_runtime_device_record_uses_sealed_observed_local_order() -> None:
    import importlib.util

    path = REPO / "scripts/greenfield/run_real_one_layer_ws32.py"
    spec = importlib.util.spec_from_file_location("ws32_real_layer_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = SimpleNamespace(
        coords=(0, 2, 0),
        core_on_chip=0,
        id=4,
        device_kind="TPU v4",
        local_hardware_id=None,
        platform="tpu",
        process_index=1,
    )
    assert module._device_record(device, observed_local_device_id=0) == {
        "coordinates": [0, 2, 0],
        "core_on_chip": 0,
        "device_id": 4,
        "device_kind": "TPU v4",
        "local_device_id": 0,
        "platform": "tpu",
        "process_index": 1,
    }
    device.local_hardware_id = 1
    with pytest.raises(ValueError, match="local device ids disagree"):
        module._device_record(device, observed_local_device_id=0)


@pytest.mark.skipif(
    not (REAL_WS32_HLO_ROOT / "layer.rank0.optimized_hlo.txt").is_file(),
    reason="protected WS32 TPU HLO unavailable",
)
def test_ws32_real_tpu_hlo_is_pinned_and_fail_closed() -> None:
    from jaxlib.xla_client import _xla

    from glm_tpu.greenfield.benchmarking.ws32_one_layer import (
        validate_ws32_one_layer_hlo,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    stable = (REAL_WS32_HLO_ROOT / "layer.rank0.stablehlo.mlir").read_text()
    optimized = (REAL_WS32_HLO_ROOT / "layer.rank0.optimized_hlo.txt").read_text()

    def validate(optimized_text: str, *, result_dtype: str = "bf16"):
        return validate_ws32_one_layer_hlo(
            stable,
            optimized_text,
            expected_stablehlo_sha256=REAL_WS32_STABLEHLO_SHA256,
            expected_optimized_hlo_sha256=sha256(
                optimized_text.encode("utf-8")
            ).hexdigest(),
            expected_optimized_collective_result_dtype=result_dtype,
        )

    report = validate(optimized)
    assert report.passed
    assert report.f32_operand_reduce_count == 10
    assert report.bf16_result_reduce_count == 10
    assert report.f32_result_reduce_count == 0
    assert not validate(optimized, result_dtype="f32").passed

    module = parse_hlo_module(optimized)
    first = module.collectives[0]
    reducer_name = re.search(r"\bto_apply=([^,\s]+)", first.raw_line)
    assert reducer_name is not None
    reducer_root = next(
        item
        for item in module.instructions
        if item.computation.split(" ", 1)[0] == reducer_name.group(1)
        and item.raw_line.lstrip().startswith("ROOT ")
    )
    wrong_reducer = optimized.replace(
        reducer_root.raw_line,
        reducer_root.raw_line.replace(" add(", " maximum(", 1),
        1,
    )
    _xla.hlo_module_from_text(wrong_reducer)
    refused_reducer = validate(wrong_reducer)
    assert not refused_reducer.passed
    assert any("reducer is not exact scalar add" in item for item in refused_reducer.violations)

    wrong_group = optimized.replace(
        "replica_groups={{0,1,2,3},{4,5,6,7}",
        "replica_groups={{0,1,2,4},{3,5,6,7}",
        1,
    )
    _xla.hlo_module_from_text(wrong_group)
    refused_group = validate(wrong_group)
    assert not refused_group.passed
    assert any("unknown replica group" in item for item in refused_group.violations)


def test_ws32_production_shape_hlo_is_live_and_fail_closed() -> None:
    program = r'''
from hashlib import sha256
import json
import re
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from glm_tpu.greenfield.benchmarking.ws32_one_layer import build_ws32_one_layer_mapped,validate_ws32_one_layer_hlo
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

mesh=Mesh(np.asarray(jax.devices(),dtype=object).reshape(8,4),('expert','feature'))
specs=(
((1,6144),jnp.bfloat16,P(None,'feature')),((1,8),jnp.int32,P()),((1,8),jnp.float32,P()),
((256,2048,6144),jnp.uint8,P('expert',None,'feature')),((256,16,48),jnp.float32,P('expert',None,'feature')),
((256,2048,6144),jnp.uint8,P('expert',None,'feature')),((256,16,48),jnp.float32,P('expert',None,'feature')),
((256,6144,2048),jnp.uint8,P('expert','feature',None)),((256,48,16),jnp.float32,P('expert','feature',None)),
((2048,6144),jnp.uint8,P(None,'feature')),((16,48),jnp.float32,P(None,'feature')),
((2048,6144),jnp.uint8,P(None,'feature')),((16,48),jnp.float32,P(None,'feature')),
((6144,2048),jnp.uint8,P('feature',None)),((48,16),jnp.float32,P('feature',None)))
args=[jax.ShapeDtypeStruct(shape,dtype,sharding=NamedSharding(mesh,spec)) for shape,dtype,spec in specs]
lowered=jax.jit(build_ws32_one_layer_mapped(mesh,contract=GlmMoeNumericalContract(stage_size=8))).lower(*args)
stable=str(lowered.compiler_ir(dialect='stablehlo')); optimized=lowered.compile().as_text()
def validate(stable_text,optimized_text):
  return validate_ws32_one_layer_hlo(stable_text,optimized_text,expected_stablehlo_sha256=sha256(stable_text.encode()).hexdigest(),expected_optimized_hlo_sha256=sha256(optimized_text.encode()).hexdigest(),expected_optimized_collective_result_dtype='f32')
base=validate(stable,optimized)

wrong_group=optimized.replace('replica_groups={{0,1,2,3},{4,5,6,7}', 'replica_groups={{0,1,2,4},{3,5,6,7}',1)
dead_input=re.sub(r'(?<!^)(%param\.61)(?!\s*=)', '%param.63', optimized, flags=re.MULTILINE)

module=parse_hlo_module(optimized); first=next(item for item in module.collectives if not item.computation.startswith('ENTRY '))
reducer_name=re.search(r'to_apply=([^,\s]+)',first.raw_line).group(1)
reducer_root=next(item for item in module.instructions if item.computation.split(' ',1)[0]==reducer_name and item.raw_line.lstrip().startswith('ROOT '))
wrong_reducer=optimized.replace(reducer_root.raw_line,reducer_root.raw_line.replace(' add(', ' maximum(',1),1)
computation=[item for item in module.instructions if item.computation==first.computation]
root=next(item for item in computation if item.raw_line.lstrip().startswith('ROOT '))
candidates=[item for item in computation if item is not root and len(item.result_shapes)==1 and item.result_shapes[0].dtype=='bf16' and item.result_shapes[0].dimensions==(1,1536) and not any(first.name==name for name in item.operand_names)]
replacement=next(item for item in candidates if item.name != root.operand_names[0])
dead_collective=optimized.replace(root.raw_line,root.raw_line.replace(root.operand_names[0],replacement.name,1),1)

entry_root=next(item for item in module.instructions if item.computation.startswith('ENTRY ') and item.raw_line.lstrip().startswith('ROOT '))
callee=re.search(r'calls=([^,\s]+)',entry_root.raw_line).group(1)
dead_fusion_input=re.sub(r'(?<!^)(%param\.62)(?!\s*=)', '%param.64', optimized, flags=re.MULTILINE)
mutated_root=entry_root.raw_line.replace('), kind=', ', %param.62), kind=',1)
dead_fusion_input=dead_fusion_input.replace(entry_root.raw_line,mutated_root,1)
header=re.search(r'^'+re.escape(callee)+r' \([^\n]*\) ->',dead_fusion_input,flags=re.MULTILINE).group(0)
dead_fusion_input=dead_fusion_input.replace(header,header.replace(') ->',', dead_scale: f32[32,16,12]) ->',1),1)

vacant=validate_ws32_one_layer_hlo(stable,optimized,expected_stablehlo_sha256='0'*64,expected_optimized_hlo_sha256='0'*64,expected_optimized_collective_result_dtype=None)
print(json.dumps({'base':base.to_dict(),'wrong_group':validate(stable,wrong_group).passed,'wrong_reducer':validate(stable,wrong_reducer).passed,'dead_input':validate(stable,dead_input).passed,'dead_collective':validate(stable,dead_collective).passed,'dead_fusion_input':validate(stable,dead_fusion_input).passed,'vacant':vacant.to_dict()},sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["base"]["passed"]
    assert result["base"]["feature_reduce_count"] == 9
    assert result["base"]["expert_reduce_count"] == 1
    assert result["base"]["f32_operand_reduce_count"] == 10
    assert result["base"]["f32_result_reduce_count"] == 10
    assert result["base"]["bf16_result_reduce_count"] == 0
    assert result["base"]["live_collective_count"] == 10
    assert result["base"]["live_entry_parameter_count"] == 15
    assert not result["wrong_group"]
    assert not result["wrong_reducer"]
    assert not result["dead_input"]
    assert not result["dead_collective"]
    assert not result["dead_fusion_input"]
    assert set(result["vacant"]["violations"]) == {
        "StableHLO pin is intentionally vacant",
        "optimized HLO pin is intentionally vacant",
    }


def test_ws32_runner_and_protected_wrapper_are_default_off() -> None:
    runner = REPO / "scripts/greenfield/run_real_one_layer_ws32.py"
    mapped = REPO / "glm_tpu/greenfield/benchmarking/ws32_one_layer.py"
    wrapper = REPO / "scripts/greenfield/run_real_one_layer_ws32.sh"
    compile(runner.read_text(), str(runner), "exec")
    syntax = subprocess.run(
        ["bash", "-n", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stdout + syntax.stderr
    source = wrapper.read_text()
    assert "GLM_GREENFIELD_WS32_REAL_LAYER:-0" in source
    assert "GLM_GREENFIELD_WS32_REAL_LAYER_MODE:-off} == numerical" in source
    assert "--compile-only 0 --warmup 2 --iterations 5" in source
    assert "WS32_LAYER_OK" in source
    assert "protected WS32 layer correctness/HBM diagnostic" in source
    assert "run_real_one_layer_ws32.py" in source
    assert REAL_WS32_STABLEHLO_SHA256 in source
    assert REAL_WS32_OPTIMIZED_HLO_SHA256 in source
    assert "local label=$1\n  local out=" in source
    assert "local label=$1 out=" not in source
    assert '"$REMOTE_PREFIX/SUCCESS"' in source
    assert "performance_claim':False" in source
    assert "diagnostic_only':True" in source
    assert "expected_optimized_collective_result_dtype='bf16'" in source
    assert "observed_bf16_bits" in source
    assert "remote_objects.json" in source
    assert "nonterminal_object_count']!=50" in source
    assert "blob.upload_from_filename" in source
    assert "if_generation_match=0" in source
    heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.DOTALL)
    assert len(heredocs) == 4
    for index, heredoc in enumerate(heredocs):
        compile(heredoc, f"{wrapper}:heredoc{index}", "exec")
    runner_lines = runner.read_text().splitlines()
    mapped_lines = mapped.read_text().splitlines()
    assert "lowered = jax.jit(mapped).lower(*normal_inputs)" in runner_lines[374]
    assert "raise SystemExit(main())" in runner_lines[503]
    assert "return ws32_moe_fp8_from_routes_mapped(" in mapped_lines[217]


@pytest.mark.skipif(
    not (
        (REAL_WS32_ACQUISITION_ROOT / "fleet_prevalidation").is_dir()
        and REAL_WS32_ARTIFACT_ROOT.is_dir()
        and REAL_WS32_ORACLE_ROOT.is_dir()
        and REAL_TOPOLOGY_ROOT.is_dir()
    ),
    reason="protected WS32 acquisition inputs unavailable",
)
def test_ws32_numerical_terminal_recomputes_the_sealed_shards(
    tmp_path: Path,
) -> None:
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    import torch

    from glm_tpu.greenfield.benchmarking import (
        REAL_LAYER_OUTPUT_TOLERANCE,
        compare_bounded_tensor,
        latency_distribution,
        validate_ws32_one_layer_hlo,
    )

    run = tmp_path / "run"
    fleet = run / "fleet"
    fleet_hlo = run / "fleet_hlo"
    fleet.mkdir(parents=True)
    fleet_hlo.mkdir()
    pin = "a" * 40
    tag = "unit_ws32_numerical"
    oracle_outputs: dict[str, np.ndarray] = {}
    with safe_open(
        REAL_WS32_ORACLE_ROOT / "oracle.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        for case in ("normal", "concentrated"):
            oracle_outputs[case] = np.ascontiguousarray(
                handle.get_tensor(f"{case}_output").view(torch.uint16).numpy()
            )

    for rank in range(8):
        stable_source = REAL_WS32_HLO_ROOT / f"layer.rank{rank}.stablehlo.mlir"
        optimized_source = (
            REAL_WS32_HLO_ROOT / f"layer.rank{rank}.optimized_hlo.txt"
        )
        stable_target = fleet_hlo / stable_source.name
        optimized_target = fleet_hlo / optimized_source.name
        shutil.copyfile(stable_source, stable_target)
        shutil.copyfile(optimized_source, optimized_target)
        report = validate_ws32_one_layer_hlo(
            stable_target.read_text(),
            optimized_target.read_text(),
            expected_stablehlo_sha256=REAL_WS32_STABLEHLO_SHA256,
            expected_optimized_hlo_sha256=REAL_WS32_OPTIMIZED_HLO_SHA256,
            expected_optimized_collective_result_dtype="bf16",
        )
        assert report.passed
        pre = json.loads(
            (
                REAL_WS32_ACQUISITION_ROOT
                / "fleet_prevalidation"
                / f"prevalidation.rank{rank}.json"
            ).read_text()
        )
        pre["code_hash"] = pin
        pre["hlo"] = report.to_dict()
        correctness = {}
        for case in ("normal", "concentrated"):
            shards = []
            for owner in pre["local_device_slots"]:
                slot = owner["device_slot"]
                expert_coordinate, feature_coordinate = divmod(slot, 4)
                start = feature_coordinate * 1536
                bits = np.ascontiguousarray(
                    oracle_outputs[case][:, start : start + 1536]
                )
                comparison = compare_bounded_tensor(
                    bits.view(ml_dtypes.bfloat16),
                    bits.view(ml_dtypes.bfloat16),
                    REAL_LAYER_OUTPUT_TOLERANCE,
                )
                shards.append(
                    {
                        "bitwise_mismatch_count": 0,
                        "comparison": comparison,
                        "device_id": owner["device_id"],
                        "device_slot": slot,
                        "expert_coordinate": expert_coordinate,
                        "feature_coordinate": feature_coordinate,
                        "observed_bf16_bits": bits.reshape(-1).tolist(),
                        "observed_bf16_sha256": sha256(bits.tobytes()).hexdigest(),
                        "oracle_bf16_sha256": sha256(bits.tobytes()).hexdigest(),
                    }
                )
            correctness[case] = {
                "case": case,
                "local_shards": shards,
                "passed": True,
            }
        timing = {}
        for case_index, case in enumerate(("normal", "concentrated")):
            samples = [float(rank + case_index + 1) + offset / 10 for offset in range(5)]
            timing[case] = {
                "iterations": 5,
                "latency": latency_distribution(samples).to_dict(),
                "performance_claim": False,
                "profiler_active": False,
                "samples_ms": samples,
                "warmup": 2,
            }
        runner = {
            **pre,
            "artifact_kind": "greenfield_ws32_real_layer3",
            "correctness": correctness,
            "device_memory_after_execute": pre["device_memory_after_load"],
            "performance_claim": False,
            "schema_version": 1,
            "status": "SUCCESS",
            "timing": timing,
        }
        (fleet / f"prevalidation.rank{rank}.json").write_text(
            json.dumps(pre, indent=2, sort_keys=True) + "\n"
        )
        (fleet / f"runner.rank{rank}.json").write_text(
            json.dumps(runner, indent=2, sort_keys=True) + "\n"
        )

    wrapper = REPO / "scripts/greenfield/run_real_one_layer_ws32.sh"
    terminal = re.findall(r"<<'PY'\n(.*?)\nPY", wrapper.read_text(), re.DOTALL)[0]
    command = [
        sys.executable,
        "-c",
        terminal,
        str(run),
        pin,
        REAL_WS32_ARTIFACT_MANIFEST_SHA256,
        REAL_WS32_ORACLE_MANIFEST_SHA256,
        REAL_TOPOLOGY_SHA256,
        REAL_TOPOLOGY_FLEET_SHA256,
        REAL_WS32_MESH_SHA256,
        str(REAL_WS32_ARTIFACT_ROOT / "manifest.json"),
        str(REAL_TOPOLOGY_ROOT),
        str(REAL_WS32_ORACLE_ROOT),
        REAL_WS32_STABLEHLO_SHA256,
        REAL_WS32_OPTIMIZED_HLO_SHA256,
        tag,
    ]
    environment = {**os.environ, "PYTHONPATH": str(REPO)}
    accepted = subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "SUCCESS"
    assert summary["correctness"]["normal"]["replicated_shard_count"] == 32
    assert summary["correctness"]["normal"]["total_bitwise_mismatch_count"] == 0

    (run / "summary.json").unlink()
    forged_path = fleet / "runner.rank0.json"
    forged = json.loads(forged_path.read_text())
    forged_bits = np.full((1, 1536), ml_dtypes.bfloat16(1000)).view(np.uint16)
    forged_shard = forged["correctness"]["normal"]["local_shards"][0]
    forged_shard["observed_bf16_bits"] = forged_bits.reshape(-1).tolist()
    forged_shard["observed_bf16_sha256"] = sha256(
        forged_bits.tobytes()
    ).hexdigest()
    forged_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")
    refused = subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )
    assert refused.returncode != 0
    assert "independent oracle recomputation mismatch" in (
        refused.stdout + refused.stderr
    )


def test_ws32_numerical_archive_requires_exact_remote_set(
    tmp_path: Path,
) -> None:
    import base64

    import google_crc32c

    wrapper = REPO / "scripts/greenfield/run_real_one_layer_ws32.sh"
    heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", wrapper.read_text(), re.DOTALL)
    run = tmp_path / "run"
    run.mkdir()
    remote = "gs://driftbench-dsv4-uc/results/unit_ws32_numerical"
    pin, tag = "a" * 40, "unit_ws32_numerical"
    artifact, oracle = "b" * 64, "c" * 64
    topology, topology_fleet, mesh = "d" * 64, "e" * 64, "f" * 64
    stable, optimized = "1" * 64, "2" * 64
    summary = {
        "code_hash": pin,
        "correctness": {"normal": {}, "concentrated": {}},
        "diagnostic_only": True,
        "hlo": {
            "stablehlo_sha256": stable,
            "optimized_hlo_sha256": optimized,
        },
        "measured_memory": {"records": 32},
        "mesh_sha256": mesh,
        "oracle_manifest_sha256": oracle,
        "packed_manifest_sha256": artifact,
        "performance_claim": False,
        "status": "SUCCESS",
        "tag": tag,
        "topology_fleet_sha256": topology_fleet,
        "topology_sha256": topology,
    }
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "fleet").mkdir()
    (run / "fleet_hlo").mkdir()
    local_objects = [
        ("summary.json", run / "summary.json"),
        *[
            (name, run / name)
            for name in (
                "census_pre.txt",
                "census_post.txt",
                "sync.txt",
                "launch.txt",
                "remote_vacancy.stdout",
                "remote_vacancy.stderr",
                "evidence.sha256",
                "orchestrator.log",
            )
        ],
    ]
    for rank in range(8):
        local_objects.extend(
            [
                (
                    f"host_records/runner.rank{rank}.json",
                    run / "fleet" / f"runner.rank{rank}.json",
                ),
                (
                    f"host_records/prevalidation.rank{rank}.json",
                    run / "fleet" / f"prevalidation.rank{rank}.json",
                ),
                (
                    f"host_records/runner.rank{rank}.log",
                    run / "fleet" / f"runner.rank{rank}.log",
                ),
                (
                    f"hlo/layer.rank{rank}.stablehlo.mlir",
                    run / "fleet_hlo" / f"layer.rank{rank}.stablehlo.mlir",
                ),
                (
                    f"hlo/layer.rank{rank}.optimized_hlo.txt",
                    run / "fleet_hlo" / f"layer.rank{rank}.optimized_hlo.txt",
                ),
            ]
        )
    for name, path in local_objects:
        if not path.exists():
            path.write_text(f"unit object {name}\n")

    def crc(path: Path) -> str:
        return base64.b64encode(
            google_crc32c.Checksum(path.read_bytes()).digest()
        ).decode("ascii")

    valid_records = [
        {
            "crc32c": crc(path),
            "generation": "123",
            "name": name,
            "size": path.stat().st_size,
        }
        for name, path in local_objects
    ]
    ledger = run / "remote_objects.json"
    ledger.write_text(json.dumps({"objects": valid_records}))
    expected = {name for name, _ in local_objects} | {"remote_objects.json"}
    descriptions = {
        name: {
            "crc32c_hash": crc(path),
            "generation": "123",
            "size": path.stat().st_size,
        }
        for name, path in local_objects
    }
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gcloud = fake_bin / "gcloud"
    fake_gcloud.write_text(
        """#!/usr/bin/env python3
import json,os
from pathlib import Path
import sys
args=sys.argv[1:]
if args[:3]==['storage','objects','describe']:
    name=args[3].removeprefix(os.environ['FAKE_REMOTE'].rstrip('/')+'/')
    print(json.dumps(json.loads(os.environ['FAKE_DESCRIPTIONS'])[name]))
elif args[:3]==['storage','objects','list']:
    prefix=os.environ['FAKE_BUCKET_PREFIX'].rstrip('/')+'/'
    for name in json.loads(os.environ['FAKE_OBJECTS']): print(prefix+name)
else: raise SystemExit(f'unexpected fake gcloud arguments: {args}')
"""
    )
    fake_gcloud.chmod(0o755)

    command = [
        sys.executable,
        "-c",
        heredocs[1],
        str(run),
        remote,
        pin,
        tag,
        artifact,
        oracle,
        topology,
        topology_fleet,
        mesh,
        stable,
        optimized,
    ]
    base_environment = {
        **os.environ,
        "FAKE_BUCKET_PREFIX": remote.removeprefix("gs://").split("/", 1)[1],
        "FAKE_REMOTE": remote,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    def invoke(objects: set[str]) -> subprocess.CompletedProcess[str]:
        descriptions["remote_objects.json"] = {
            "crc32c_hash": crc(ledger),
            "generation": "123",
            "size": ledger.stat().st_size,
        }
        return subprocess.run(
            command,
            env={
                **base_environment,
                "FAKE_DESCRIPTIONS": json.dumps(descriptions),
                "FAKE_OBJECTS": json.dumps(sorted(objects)),
            },
            check=False,
            capture_output=True,
            text=True,
        )

    ledger.write_text(json.dumps({"objects": []}))
    empty = invoke(expected)
    assert empty.returncode != 0
    assert "ledger schema/count drifted" in empty.stderr
    ledger.write_text(json.dumps({"objects": valid_records[:-1]}))
    missing = invoke(expected)
    assert missing.returncode != 0
    assert "ledger schema/count drifted" in missing.stderr
    duplicate_records = deepcopy(valid_records)
    duplicate_records[-1] = deepcopy(duplicate_records[-2])
    ledger.write_text(json.dumps({"objects": duplicate_records}))
    duplicate = invoke(expected)
    assert duplicate.returncode != 0
    assert "ledger names/order drifted" in duplicate.stderr
    ledger.write_text(json.dumps({"objects": valid_records}))
    planted = subprocess.run(
        command, env={
            **base_environment,
            "FAKE_DESCRIPTIONS": json.dumps({
                **descriptions,
                "remote_objects.json": {
                    "crc32c_hash": crc(ledger),
                    "generation": "123",
                    "size": ledger.stat().st_size,
                },
            }),
            "FAKE_OBJECTS": json.dumps(sorted(expected | {"rogue"})),
        }, check=False, capture_output=True, text=True,
    )
    assert planted.returncode != 0
    assert "object set drifted" in planted.stderr
    assert not (run / "SUCCESS").exists()
    accepted = invoke(expected)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    success = json.loads((run / "SUCCESS").read_text())
    assert success["nonterminal_object_count"] == 50

    fake_python = tmp_path / "fake_python"
    fake_storage = fake_python / "google" / "cloud"
    fake_storage.mkdir(parents=True)
    (fake_python / "google" / "__init__.py").write_text("")
    (fake_storage / "__init__.py").write_text("")
    (fake_storage / "storage.py").write_text(
        """import base64
from pathlib import Path
import google_crc32c
class Blob:
    def __init__(self,name): self.name=name; self.generation=None; self.size=None; self.crc32c=None
    def upload_from_filename(self,path,**kwargs):
        assert kwargs['if_generation_match']==0 and kwargs['checksum']=='crc32c'
        data=Path(path).read_bytes(); self.generation=123; self.size=len(data)
        self.crc32c=base64.b64encode(google_crc32c.Checksum(data).digest()).decode('ascii')
    def reload(self,**kwargs): assert kwargs['if_generation_match']==123
class Bucket:
    def blob(self,name): return Blob(name)
class Client:
    def bucket(self,name): return Bucket()
"""
    )
    ownership = run / "success_upload.json"
    published = subprocess.run(
        [
            sys.executable,
            "-c",
            heredocs[2],
            str(run / "SUCCESS"),
            f"{remote}/SUCCESS",
            str(ownership),
        ],
        env={**os.environ, "PYTHONPATH": str(fake_python)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert published.returncode == 0, published.stdout + published.stderr
    assert json.loads(ownership.read_text()) == {
        "crc32c": crc(run / "SUCCESS"),
        "generation": "123",
        "remote": f"{remote}/SUCCESS",
        "size": (run / "SUCCESS").stat().st_size,
    }

    verified = subprocess.run(
        [
            sys.executable,
            "-c",
            heredocs[3],
            str(run / "SUCCESS"),
            str(ownership),
            remote,
        ],
        env={
            **base_environment,
            "FAKE_DESCRIPTIONS": json.dumps({
                "SUCCESS": {
                    "crc32c_hash": crc(run / "SUCCESS"),
                    "generation": "123",
                    "size": (run / "SUCCESS").stat().st_size,
                },
            }),
            "FAKE_OBJECTS": "[]",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert json.loads(verified.stdout)["terminal_object_count"] == 51


def test_ws32_success_verification_failure_rolls_back_without_diagnostic(
    tmp_path: Path,
) -> None:
    import base64

    import google_crc32c

    source = (REPO / "scripts/greenfield/run_real_one_layer_ws32.sh").read_text()
    rollback = re.search(
        r"rollback_remote_success\(\) \{\n.*?^\}",
        source,
        re.DOTALL | re.MULTILINE,
    )
    on_exit = re.search(
        r"on_exit\(\) \{\n.*?^\}", source, re.DOTALL | re.MULTILINE
    )
    assert rollback is not None and on_exit is not None
    publication = source.index("terminal_publication_started=1")
    upload = source.index("blob.upload_from_filename")
    verification = source.index("remote WS32 numerical SUCCESS identity drifted")
    verified = source.index("terminal_success_verified=1")
    assert publication < upload < verification < verified

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "gcloud.log"
    fake = fake_bin / "gcloud"
    fake.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >>"$FAKE_GCLOUD_LOG"
if [[ $* == 'storage objects describe '* ]]; then
  [[ $FAKE_DESCRIBE_STATUS == 0 ]] || exit "$FAKE_DESCRIBE_STATUS"
  printf '%s\n' "$FAKE_DESCRIBE_JSON"
  exit 0
fi
if [[ $* == 'storage rm '* ]]; then exit 0; fi
exit 99
"""
    )
    fake.chmod(0o755)
    run = tmp_path / "run"
    run.mkdir()
    remote = "gs://driftbench-dsv4-uc/results/unit_ws32_numerical"
    success_path = run / "SUCCESS"
    success_path.write_text('{"status":"SUCCESS"}\n')
    checksum = google_crc32c.Checksum(success_path.read_bytes())
    local_crc = base64.b64encode(checksum.digest()).decode("ascii")
    (run / "success_upload.json").write_text(
        json.dumps(
            {
                "crc32c": local_crc,
                "generation": "123",
                "remote": f"{remote}/SUCCESS",
                "size": success_path.stat().st_size,
            }
        )
    )
    script = (
        "set +e\n"
        f"RUN_DIR={run}\n"
        f"REMOTE_PREFIX={remote}\n"
        "post_census_done=1\n"
        "terminal_publication_started=1\n"
        "terminal_success_verified=0\n"
        "say() { :; }\n"
        + rollback.group(0)
        + "\n"
        + on_exit.group(0)
        + "\nfalse\non_exit\n"
    )

    def invoke(status: int, generation: str) -> list[str]:
        log.unlink(missing_ok=True)
        completed = subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "FAKE_DESCRIBE_JSON": json.dumps(
                    {
                        "crc32c_hash": local_crc,
                        "generation": generation,
                        "size": success_path.stat().st_size,
                    }
                ),
                "FAKE_DESCRIBE_STATUS": str(status),
                "FAKE_GCLOUD_LOG": str(log),
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
            },
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return log.read_text().splitlines()

    describe_failed = invoke(17, "123")
    assert not any(call.startswith("storage rm ") for call in describe_failed)
    foreign = invoke(0, "124")
    assert not any(call.startswith("storage rm ") for call in foreign)
    owned = invoke(0, "123")
    assert owned[-1] == (
        f"storage rm --if-generation-match=123 {remote}/SUCCESS"
    )
    assert not any(call.startswith("storage cp ") for call in owned)


@pytest.mark.skipif(
    not REAL_TOPOLOGY_ROOT.is_dir(), reason="protected topology fleet unavailable"
)
def test_ws32_real_topology_launch_to_jax_permutation_is_pinned() -> None:
    from glm_tpu.greenfield.benchmarking.ws32_one_layer import (
        validate_ws32_topology_fleet,
    )

    captures = tuple(
        json.loads((REAL_TOPOLOGY_ROOT / f"topology.rank{rank}.json").read_text())
        for rank in range(8)
    )
    topology, ordered, observed = validate_ws32_topology_fleet(
        captures,
        expected_topology_sha256=REAL_TOPOLOGY_SHA256,
        expected_fleet_sha256=REAL_TOPOLOGY_FLEET_SHA256,
        slice_name="db-v4-64-od",
    )
    assert topology.topology_hash == REAL_TOPOLOGY_SHA256
    assert observed == REAL_TOPOLOGY_FLEET_SHA256
    assert [item["jax_process_index"] for item in ordered] == [1, 6, 0, 7, 2, 4, 3, 5]
    assert [item["hostname"] for item in ordered] == [
        f"t1v-n-6c15e171-w-{rank}" for rank in range(8)
    ]

    mutated = list(deepcopy(captures))
    mutated[0]["jax_process_index"] = 0
    with pytest.raises(ValueError, match="JAX process identities"):
        validate_ws32_topology_fleet(
            tuple(mutated),
            expected_topology_sha256=REAL_TOPOLOGY_SHA256,
            expected_fleet_sha256=REAL_TOPOLOGY_FLEET_SHA256,
            slice_name="db-v4-64-od",
        )


def test_ws32_remote_vacancy_is_fail_closed(
    tmp_path: Path,
) -> None:
    source = (REPO / "scripts/greenfield/run_real_one_layer_ws32.sh").read_text()
    match = re.search(
        r"remote_prefix_is_vacant\(\) \{\n.*?^\}",
        source,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    function = match.group(0)

    def invoke(stdout: str, stderr: str, status: int) -> int:
        case = tmp_path / f"case_{status}_{len(stdout)}_{len(stderr)}"
        binary = case / "bin"
        run = case / "run"
        binary.mkdir(parents=True)
        run.mkdir()
        fake = binary / "gcloud"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%b' {json.dumps(stdout)}\n"
            f"printf '%b' {json.dumps(stderr)} >&2\n"
            f"exit {status}\n"
        )
        fake.chmod(0o755)
        completed = subprocess.run(
            [
                "bash",
                "-c",
                "set -u\n"
                + f"RUN_DIR={run!s}\n"
                + "REMOTE_PREFIX=gs://driftbench-dsv4-uc/results/unit\n"
                + function
                + "\nremote_prefix_is_vacant\n",
            ],
            env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode

    no_objects = (
        "ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n"
    )
    assert invoke("", no_objects, 1) == 0
    assert invoke("gs://driftbench-dsv4-uc/results/unit/stale\n", "", 0) != 0
    assert invoke("", "ERROR: authentication failed\n", 17) != 0
