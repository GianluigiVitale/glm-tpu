"""Retained actual WK graphs and structural refusal fixtures; CPU only.

The synthetic composed instruction inventory is NOT a compiled rolled graph.
Actual candidate TPU HLO and numerical allocations remain a launch requirement.
"""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from scripts.greenfield import prefill_rolled_admission as admission
from scripts.greenfield import prefill_rolled_window as protocol
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.hlo.test_prefill_rolled_worker import ROOT


@pytest.mark.parametrize("name", protocol.PROGRAMS[:2])
def test_actual_retained_wk_raw_structure_memory_replay(name):
    root = ROOT / "fleet/rank0"
    record = json.loads((root / "runner.json").read_text())
    args = (
        name,
        (root / f"{name}.stablehlo.mlir").read_text(),
        (root / f"{name}.optimized_hlo.txt").read_text(),
        record["programs"][name]["compiled_memory"],
    )
    report = admission.inspect_program(*args)
    admission.validate_program_report(json.loads(json.dumps(report)), *args)
    with pytest.raises(ValueError, match="raw graph"):
        admission.inspect_program(name, args[1] + "\n", args[2], args[3])
    bad = dict(args[3])
    bad["alias_size_in_bytes"] = True
    with pytest.raises(ValueError):
        admission.inspect_program(name, args[1], args[2], bad)


def test_candidate_caps_refuse_and_composed_inventory_mutations(monkeypatch):
    caps = dict(admission.MEMORY_CAPS)
    admission.validate_memory("candidate", caps)
    for key in caps:
        wrong = dict(caps)
        wrong[key] += 1
        with pytest.raises(ValueError):
            admission.validate_memory("candidate", wrong)
    root = ROOT / "fleet/rank0"
    # Reuse real observed prefix/suffix collectives; only the one outer loop
    # and precision verdict here are fixtures. Actual precision function is
    # already tested on the original suffix, and will run on candidate TPU HLO.
    pieces = [
        parse_hlo_module((root / f"{n}.optimized_hlo.txt").read_text())
        for n in ("prefix", "candidate")
    ]
    instructions = [op for m in pieces for op in m.instructions if op.is_collective]
    loop = replace(
        instructions[0],
        opcode="while",
        raw_opcode="while",
        op_name="jit/greenfield_ws32_prefill_rolled_prefix/while",
        raw_line="synthetic outer-loop inventory fixture",
    )
    instructions.append(loop)
    monkeypatch.setattr(
        admission,
        "parse_hlo_module",
        lambda text: SimpleNamespace(instructions=instructions),
    )
    monkeypatch.setattr(
        admission, "check_fp32_route_sum", lambda *a, **kw: dict(passed=True)
    )
    assert admission.inspect_structure("fixture")["dynamic_prefix_iterations"] == 4
    original = list(instructions)
    for mode in ("missing", "extra_loop", "global_group", "host_callback"):
        instructions[:] = original
        if mode == "missing":
            instructions.pop(0)
        elif mode == "extra_loop":
            instructions.append(loop)
        elif mode == "global_group":
            instructions[0] = replace(
                instructions[0], replica_groups=(tuple(range(32)),)
            )
        else:
            instructions.append(
                replace(
                    loop,
                    opcode="custom-call",
                    raw_line='custom_call_target="xla_python_cpu_callback"',
                    op_name=None,
                )
            )
        with pytest.raises(ValueError, match="actual structure"):
            admission.inspect_structure("fixture")


def test_generation_bound_reference_download_and_cached_mutation(tmp_path, monkeypatch):
    from google.cloud import storage

    seal = protocol.originals._json_bound(protocol.SEAL, protocol.SEAL_SHA)
    rows = json.loads((ROOT / "archive_receipts.json").read_bytes())
    pins = {row["name"]: row for row in rows}
    for name in ("SUCCESS", "archive_receipts.json"):
        pins[f"results/{seal['tag']}/{name}"] = seal["source_objects"][name]
    downloaded = []

    class Blob:
        def __init__(self, name, generation):
            self.name, self.generation = name, generation
            pin = pins[name]
            self.size, self.crc32c = pin["size"], pin["crc32c"]
            assert generation == int(pin["generation"])

        def reload(self, *, if_generation_match):
            assert if_generation_match == self.generation

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            downloaded.append(self.name)
            return (
                ROOT / self.name.split(f"results/{seal['tag']}/", 1)[1]
            ).read_bytes()

    def bucket(name):
        assert name == "driftbench-dsv4-uc"
        return SimpleNamespace(blob=lambda name, generation: Blob(name, generation))

    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=bucket))
    ref = protocol.materialize_reference(tmp_path, rank=0)
    assert len(downloaded) == 4 and len(ref.slots) == 4
    protocol.materialize_reference(tmp_path, rank=0)
    assert len(downloaded) == 4
    (tmp_path / "SUCCESS").write_bytes(b"mutated")
    with pytest.raises(ValueError):
        protocol.materialize_reference(tmp_path, rank=0)
    assert len(downloaded) == 4  # Existing mismatched bytes are never overwritten.
