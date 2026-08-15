from __future__ import annotations

from hashlib import sha256
from copy import deepcopy
import json
import os
from pathlib import Path
import re
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
  return validate_ws32_one_layer_hlo(stable_text,optimized_text,expected_stablehlo_sha256=sha256(stable_text.encode()).hexdigest(),expected_optimized_hlo_sha256=sha256(optimized_text.encode()).hexdigest())
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

vacant=validate_ws32_one_layer_hlo(stable,optimized,expected_stablehlo_sha256='0'*64,expected_optimized_hlo_sha256='0'*64)
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
    assert "--compile-only 1" in source
    assert "no arithmetic, DB row, performance claim, or terminal SUCCESS" in source
    assert "run_real_one_layer_ws32.py" in source
    assert "EXPECTED_STABLEHLO_SHA" in source
    assert "EXPECTED_OPTIMIZED_HLO_SHA" in source
    assert "local label=$1\n  local out=" in source
    assert "local label=$1 out=" not in source
    assert '"$REMOTE_PREFIX/SUCCESS"' not in source
    heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.DOTALL)
    assert len(heredocs) == 1
    compile(heredocs[0], f"{wrapper}:terminal", "exec")


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
