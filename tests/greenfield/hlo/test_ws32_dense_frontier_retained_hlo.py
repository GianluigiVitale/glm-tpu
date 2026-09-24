"""Reuse DB602's real first-two-layer loop and scratch mechanisms, CPU only.

The enclosing graph is the historical full model, NOT the unacquired reduced
diagnostic. Only loop selection is filtered to exercise the two-layer checker.
"""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.greenfield import ws32_dense_frontier_admission as admission
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import _target
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_helper_hlo import (
    _merge_scratch,
)
from glm_tpu.optimized.hlo_contract import parse_hlo_module


def test_actual_dense_loop_and_scratch_mechanisms():
    receipt = json.loads(
        Path(
            "docs/artifacts/prefill-rolled-model-compile-db602-sealed-20260909.json"
        ).read_text()
    )
    path = (
        Path("/home/gianl/glm-run")
        / receipt["tag"]
        / "fleet/rank0/prefill_chunk.optimized_hlo.txt"
    )
    raw = path.read_bytes()
    assert (
        sha256(raw).hexdigest()
        == receipt["programs"]["prefill_chunk"]["optimized_hlo_sha256"]
    )
    module = parse_hlo_module(raw.decode())
    live = _live_instruction_closure(module.instructions)
    index = PrefillHloIndex(module)

    def relevant(op):
        return admission.LAYER.findall(op.op_name or "") in (["0"], ["1"])

    # Keep every original operand/body unchanged. Hide only other outer-loop
    # names from selection, not their dataflow or helpers.
    filtered = replace(
        module,
        instructions=tuple(
            (
                replace(op, op_name="outside_test_scope")
                if op.opcode == "while"
                and (op.op_name or "").endswith(admission.LOOP)
                and not relevant(op)
                else op
            )
            for op in module.instructions
        ),
    )
    bodies = admission.prefix_bodies(PrefillHloIndex(filtered), live)
    assert sorted(bodies.values()) == [0, 1]
    allocations = [
        op
        for op in module.instructions
        if op.raw_opcode == "custom-call"
        and _target(op) == "AllocateBuffer"
        and op.result_shapes[0].dtype == "s32"
    ]
    # Default historical 42-buffer proof and the new four-buffer scope share
    # the SAME actual half-initialization/completion checker.
    alive = {(op.computation, op.name) for op in live}
    historical = _merge_scratch(index, allocations, alive)
    names = {r["allocation"] for r in historical if r["layer"] in (0, 1)}
    selected = [op for op in allocations if op.name in names]
    assert len(selected) == 4
    small = _merge_scratch(index, selected, alive, layer_ids=(0, 1))
    assert small == [r for r in historical if r["layer"] in (0, 1)]
    for bad in (selected[:-1], selected + [selected[0]]):
        with pytest.raises(ValueError, match="allocation count"):
            _merge_scratch(index, bad, alive, layer_ids=(0, 1))
    with pytest.raises(ValueError, match="layer scope"):
        _merge_scratch(index, selected, alive, layer_ids=(0, 2))
