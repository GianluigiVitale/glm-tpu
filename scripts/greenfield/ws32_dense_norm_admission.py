"""Fixed norm diagnostic compiler checks, never numerical or launch admission.

Raw pins bind the existing four builders. Reuse the dense capture inventory;
the isolated suffix has only three original raw matmuls, two local reductions
and bounded row selection. The first actual realization is preserved under
greenfield_fp8_ws32_dense_norm_d01_20260909T173035450074269Z.
All four originals must be preserved before this inspector can refuse a graph.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from glm_tpu.greenfield.benchmarking.ws32_batched_kernel_hlo import (
    _check_kernel_schedule,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import (
    _check_helper_schedule,
)
from scripts.greenfield import ws32_dense_frontier_admission as original
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from scripts.greenfield.ws32_dense_norm_helpers import capture_helpers, padded_gather

SUFFIX_KERNEL = "greenfield_fp8_block_matmul_f32_m128_k1536_n1536"
SUFFIX_SCOPE = f"jit(body)/shard_map/{SUFFIX_KERNEL}/pallas_call"


def suffix_layer(op: Any) -> int:
    original._require(
        op.op_name == SUFFIX_SCOPE
        and original._computation_base(op.computation) == "ENTRY",
        "norm suffix requires exact unscoped ENTRY kernel",
    )
    return -1


def suffix_collectives(index: Any, live: tuple) -> dict:
    records, votes = original._physical_records(index, live)
    original._require(not votes, "norm suffix has no scalar vote collective")
    expected = Counter()
    for family, dims in (("feature", (2, 128, 1536)), ("expert", (128, 1536))):
        expected[
            (-1, "all-reduce", family, "add", -1, (("f32", dims),), (("bf16", dims),))
        ] = 1
    for op, key in records:
        suffix = (
            "feature_gate_up_reduce" if key[2] == "feature" else "expert_down_reduce"
        )
        original._require(
            original._computation_base(op.computation) == "ENTRY"
            and op.op_name
            == f"jit(body)/shard_map/greenfield_ws32_prefill_dense/{suffix}/psum",
            "norm suffix reduction scope/placement differs",
        )
    original._require(
        Counter(key for _, key in records) == expected,
        "norm suffix physical paired reductions differ",
    )
    return dict(
        static_instruction_count=len(records),
        physical_groups=dict(feature=original.FEATURE, expert=original.EXPERT),
    )


def suffix_kernels(index: Any, live: tuple) -> dict:
    expected = Counter(
        {
            (
                -1,
                "raw",
                SUFFIX_KERNEL,
                (("bf16", (128, 1536)), ("u8", (1536, 1536)), ("f32", (16, 128))),
                (("f32", (128, 1536)),),
                -1,
            ): 3
        }
    )
    report = _check_kernel_schedule(
        index,
        live_instructions=live,
        expected=expected,
        families=("raw",),
        layer_resolver=suffix_layer,
    )
    original._require(
        report["passed"], f"norm suffix kernel inventory differs:{report}"
    )
    return report


def suffix_helpers(index: Any, live: tuple) -> dict:
    # Three dense U8 weights; one bounded row-take index vector. Compiler may
    # eliminate copies/annotations, never add unknown helpers or scratch here.
    copies = {("ConcatBitcast", "u8", (1536, 1536)): 3}
    annotations = {("AssumeGatherIndicesInBound", "s32", (1024,)): 1}

    def no_scratch(index: Any, allocations: list, rows: int, live: set) -> list:
        original._require(not allocations, "norm suffix scratch forbidden")
        return []

    report = _check_helper_schedule(
        index,
        block_rows=128,
        live_instructions=live,
        expected=Counter({**copies, **annotations}),
        copy_limits=copies,
        count_limits=annotations,
        scratch_check=no_scratch,
    )
    original._require(
        report["passed"], f"norm suffix helper inventory differs:{report}"
    )
    report["bounded_row_gathers"] = [
        padded_gather(index, op)
        for op in index.module.instructions
        if op.raw_opcode == "custom-call"
        and original._target(op) == "AssumeGatherIndicesInBound"
    ]
    return report


def inspect_suffix(optimized: str) -> dict:
    module = original.parse_hlo_module(optimized)
    index = original.PrefillHloIndex(module)
    live = original._live_instruction_closure(module.instructions)
    original._require(module.num_partitions == 32, "norm suffix requires32 partitions")
    original._require(
        not any(
            op.opcode in ("infeed", "outfeed", "send", "recv")
            or re.search(
                r'custom_call_target="[^"]*(?:host|io|python)[^"]*callback', op.raw_line
            )
            for op in module.instructions
        ),
        "norm host transport forbidden",
    )
    # Full-floating-weight expansion, actual kernel operands, aliases and
    # liveness are checked by the shared kernel checker over ALL instructions.
    return dict(
        collectives=suffix_collectives(index, live),
        kernels=suffix_kernels(index, live),
        helpers=suffix_helpers(index, live),
    )


def inspect_program(
    name: str, stable: str, optimized: str, memory: Mapping[str, Any]
) -> dict:
    original._require(
        name in protocol.RAW
        and (len(stable.encode()), sha256(stable.encode()).hexdigest())
        == protocol.RAW[name],
        "norm preregistered raw graph differs",
    )
    # Retain the original tight maxima. Suffix/capture are still selected-layer
    # diagnostics, not new full-model allocation or alias policies.
    original.validate_memory(
        "dense01" if name in protocol.PROGRAMS[2:] else name, memory
    )
    structure = (
        inspect_suffix(optimized)
        if name == "dense_suffix"
        else original.inspect_structure(
            "dense01" if name == "dense01_norm" else name,
            optimized,
            helper_check=capture_helpers if name == "dense01_norm" else None,
        )
    )
    return json.loads(
        json.dumps(
            dict(
                profile=protocol.PROFILE,
                graph=name,
                passed=True,
                stablehlo_sha256=protocol.RAW[name][1],
                optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
                compiled_memory=dict(memory),
                structure=structure,
                scope="NORM_DIAGNOSTIC_REQUIRES_LIVE_MEMORY_DB605_AND_OWN_SUFFIX_REPRODUCTION",
                numerical_promotion=False,
                performance_claim=False,
            ),
            allow_nan=False,
        )
    )
