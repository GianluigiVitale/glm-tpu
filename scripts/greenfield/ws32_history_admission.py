"""DB611-bound layers0..6 diagnostic admission, never full-model promotion.

Bind the seven exact acquired graphs and frozen model recipe; reuse physical,
kernel, helper-completion and FP32-combine checks at the explicit reduced scope.
Actual all-live memory and reproduction of BOTH original events remain separate
mandatory execution gates. No new model math, precision or cache proof is claimed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import (
    PrefillHloIndex, _check_moe_route_sums,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_commit_hlo import _require
from glm_tpu.greenfield.benchmarking.ws32_batched_collective_hlo import _physical_records
from glm_tpu.greenfield.benchmarking.ws32_batched_kernel_hlo import _check_kernel_schedule
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _computation_base, _live_instruction_closure, _callee_attribute_text,
)
from glm_tpu.greenfield.benchmarking.ws32_prefill_fixed_loops import fixed_loop_bodies
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_collective_hlo import _expected as rolled_collectives
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_kernel_hlo import _expected as rolled_kernels
from scripts.greenfield import ws32_history_compile as compiler
from scripts.greenfield import ws32_dense_frontier_admission as dense
from scripts.greenfield.prefill_layer_hlo import FEATURE, EXPERT

PROFILE = "ws32-history-l06-db611-original-reproduction-v1"
RECEIPT = "docs/artifacts/prefill-history-seven-graph-db611-sealed-20260911.json"
RECEIPT_SHA256 = "39ed445241ebfb56332bfe6966fec190659fe7841172cb3b2c35dcaf304ae287"
NUMERICAL_RECEIPT = "docs/artifacts/prefill-history-numerical-hlo-identity-20260911.json"
NUMERICAL_RECEIPT_SHA256 = "012a710a6c7a26723f4283b4c4100bb7534267c7af9330f0785a213c4a1dede2"
PROGRAMS = ("wk_decode", "wk_promote", *compiler.PROGRAMS)
LAYERS = tuple(range(7))
MOE_LAYERS = (3, 4, 5, 6)
_LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")


def registration(repo: Path) -> dict[str, Any]:
    """Require a byte-exact sealed receipt and the existing frozen source recipe."""
    compiler.require_source(repo)
    path = repo / RECEIPT
    _require(path.is_file() and not path.is_symlink(), "history receipt is missing or linked")
    raw = path.read_bytes()
    _require(sha256(raw).hexdigest() == RECEIPT_SHA256, "history DB611 receipt changed")
    result = json.loads(raw)
    _require(result["results_db_run_id"] == 611 and set(result["graphs"]) == set(compiler.PROGRAMS),
             "history DB611 graph registration differs")
    return result


def expected_collectives(rows: int, canonical: bool) -> Counter:
    """Original rolled schedule without a final head or nonexistent layer7 bias."""
    full = rolled_collectives(rows, canonical_dense=canonical)
    selected = Counter()
    for key, count in full.items():
        place, layer, opcode, family, reducer, axis, inputs, outputs = key
        keep_outer = layer == -1 and (
            reducer == "minimum"
            or (family == "expert" and inputs == (("bf16", (rows, 1536)),))
            or (canonical and family == "expert" and inputs == (("f32", (256,)),))
        )
        if layer not in LAYERS and not keep_outer:
            continue
        for source, dest in zip(inputs, outputs, strict=True):
            if layer == 6 and source == ("f32", (256,)):
                continue
            selected[(place, layer, opcode, family, reducer, axis, source, dest)] += count
    return selected


def check_frontier(index: PrefillHloIndex, live: tuple, *, name: str) -> dict[str, Any]:
    _require(name in compiler.protocol.FRONTIER_PROGRAMS, "unregistered history frontier")
    rows = 128 if name.endswith("128") else 114
    canonical = name.startswith("candidate_")
    bodies = fixed_loop_bodies(index, live, loop_suffix=dense.LOOP, expected_layers=LAYERS)
    suffixes = fixed_loop_bodies(index, live,
        loop_suffix="greenfield_ws32_prefill_dense_canonical/while", expected_layers=(0, 1, 2)) if canonical else {}
    if not canonical:
        _require(not any(op.opcode == "while" and (op.op_name or "").endswith(
            "greenfield_ws32_prefill_dense_canonical/while") for op in index.module.instructions),
            "control must not contain canonical dense loops")
    records, votes = _physical_records(index, live)
    observed: Counter = Counter()
    for op, key in records:
        comp, layer = _computation_base(op.computation), key[0]
        if comp in bodies:
            _require(bodies[comp] == layer, "history collective wrong prefix layer")
            place = "prefix"
        elif comp in suffixes:
            _require(suffixes[comp] == layer, "history collective wrong dense layer")
            place = "canonical_dense"
        else:
            _require(comp == "ENTRY", "history outer collective not ENTRY")
            place = "outer" if layer == -1 else "suffix"
        observed[(place, *key)] += 1
    flattened = dense._flatten_placed(observed)
    _require(flattened == expected_collectives(rows, canonical), "history physical leaf schedule differs")
    _require(len(votes) == 2, "history requires two local scalar health votes")

    all_expected, placements = rolled_kernels(rows, canonical_dense=canonical)
    expected = Counter({key: n for key, n in all_expected.items() if key[0] in LAYERS})
    callers: dict[str, list[tuple[Any, int]]] = defaultdict(list)
    for op in index.module.instructions:
        if op.opcode != "conditional":
            continue
        groups = re.findall(r"\bbranch_computations=\{([^}]*)\}", _callee_attribute_text(op.raw_line))
        _require(len(groups) == 1, "history ambiguous conditional branches")
        branches = [name.strip() for name in groups[0].split(",")]
        _require(len(branches) == 2 and len(op.operand_names) == 3, "history conditional arity")
        for branch, target in enumerate(branches):
            callers[target].append((op, branch))
    live_ids = {op.index for op in live}

    def placement(op: Any, key: tuple) -> None:
        _require(key in expected, "unregistered history Pallas interface")
        place, comp = placements[key], _computation_base(op.computation)
        if place == "prefix":
            _require(bodies.get(comp) == key[0], "history Pallas wrong prefix owner")
        elif place == "canonical_dense":
            _require(suffixes.get(comp) == key[0], "history Pallas wrong canonical owner")
        elif key[1] == "panels":
            owners = callers.get(comp, [])
            _require(len(owners) == 1, "history panel requires one conditional caller")
            owner, branch = owners[0]
            _require(branch == 1 and owner.index in live_ids
                and _computation_base(owner.computation) == "ENTRY"
                and _LAYER.findall(owner.op_name or "") == [str(key[0])],
                "history panel wrong wide-suffix owner")
        else:
            _require(comp == "ENTRY", "history wide Pallas not ENTRY")

    kernels = _check_kernel_schedule(index, live_instructions=live, expected=expected,
        placement_check=placement, families=("raw", "panels", "structured", "sparse"))
    _require(kernels["passed"], f"history kernel inventory differs:{kernels}")
    moe = _check_moe_route_sums(index, block_rows=rows, live_instructions=live,
        router_bias_tuple=True, layer_ids=MOE_LAYERS)
    _require(moe["passed"], f"history FP32 route/combine differs:{moe}")
    return dict(prefix_bodies=bodies, canonical_dense_bodies=suffixes,
        static_collective_count=len(records), static_collective_leaf_pairs=sum(flattened.values()),
        kernels=kernels, fp32_moe=moe, measured_dynamic_count_claim=False)


def inspect_structure(name: str, optimized: str) -> dict[str, Any]:
    """Inspect actual originals once; arithmetic is covered separately by reproduction."""
    from scripts.greenfield.ws32_history_helpers import check_history_helpers

    _require(name in compiler.PROGRAMS, "history structural graph is unregistered")
    module = parse_hlo_module(optimized)
    index = PrefillHloIndex(module)
    live = _live_instruction_closure(module.instructions)
    _require(module.num_partitions == 32, "history requires32 physical partitions")
    _require(not any(op.opcode in ("infeed", "outfeed", "send", "recv")
        or re.search(r'custom_call_target="[^"]*(?:host|io|python)[^"]*callback', op.raw_line)
        for op in module.instructions), "history host transport forbidden")
    _require(not any(s.dtype in ("bf16", "f32", "f64") and s.element_count >= 32 * 2048 * 1536
        for op in module.instructions for s in op.result_shapes), "history full floating expert expansion forbidden")
    records, _ = _physical_records(index, live)
    result = dict(physical_groups=dict(feature=FEATURE, expert=EXPERT),
        static_collective_count=len(records),
        static_collective_leaf_pairs=sum(len(key[-1]) for _, key in records))
    if name in compiler.protocol.FRONTIER_PROGRAMS:
        result["frontier"] = check_frontier(index, live, name=name)
    # Observer is the original production StrategyND/exact implementation,
    # registered by all bytes plus source. Do not invent a second math proof.
    helpers = check_history_helpers(index, name=name, live_instructions=live)
    _require(helpers["passed"], f"history helper completion differs:{helpers}")
    result["helpers"] = helpers
    return result


def inspect_program(name: str, stable: str, optimized: str, memory: Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    """Require exact DB611 identities/caps BEFORE structural or runtime admission."""
    _require(name in PROGRAMS, "unregistered history graph")
    receipt = registration(repo)
    if name in PROGRAMS[:2]:
        result = dense.inspect_program(name, stable, optimized, memory)
        result.update(profile=PROFILE, acquisition_sha256=RECEIPT_SHA256,
            scope="HISTORY_WK_REUSE_REQUIRES_RUNTIME_MEMORY_AND_BOTH_ORIGINAL_EVENTS")
        return result
    saved = receipt["graphs"][name]
    _require((len(stable.encode()), sha256(stable.encode()).hexdigest())
        == (saved["stablehlo_bytes"], saved["stablehlo_sha256"]) == compiler.RAW[name],
        "history raw identity differs from DB611")
    optimized_identity = (len(optimized.encode()), sha256(optimized.encode()).hexdigest())
    if optimized_identity != (saved["optimized_hlo_bytes"], saved["optimized_hlo_sha256"]):
        # Compiler-only and numerical parents have different host debug stacks.
        # Admit only the exact reviewed originals: never mask/normalize live HLO.
        path = repo / NUMERICAL_RECEIPT
        _require(path.is_file() and not path.is_symlink(), "history numerical identity receipt missing or linked")
        raw = path.read_bytes()
        _require(sha256(raw).hexdigest() == NUMERICAL_RECEIPT_SHA256,
                 "history numerical identity receipt changed")
        numerical = json.loads(raw)["graphs"][name]
        _require(optimized_identity == (numerical["optimized_hlo_bytes"], numerical["optimized_hlo_sha256"]),
                 "history optimized identity differs from both exact originals")
    caps = compiler.memory_caps(name)
    _require(isinstance(memory, Mapping) and set(memory) == set(caps)
        and all(type(v) is int and 0 <= v <= caps[k] for k, v in memory.items())
        and dict(memory) == saved["compiled_memory"], "history compiled allocation identity/caps differ")
    structure = inspect_structure(name, optimized)
    if name in ("exact_decode", "exact_promote"):
        from glm_tpu.greenfield.benchmarking.ws32_decoder import validate_ws32_exact_dsa_materializer_hlo
        materializer = validate_ws32_exact_dsa_materializer_hlo(stable, optimized,
            expected_stablehlo_sha256=saved["stablehlo_sha256"],
            expected_optimized_hlo_sha256=optimized_identity[1],
            kind="exact_materialize" if name == "exact_decode" else "exact_promote", full_indexer_count=4)
        _require(materializer.passed, f"history exact materializer differs:{materializer}")
        structure["materializer"] = materializer.to_dict()
    inventory = saved["independent_inventory"]
    _require(structure["static_collective_count"] == inventory["static_collectives"]
        and structure["static_collective_leaf_pairs"] == inventory["collective_leaf_pairs"],
        "history collective inventory differs from DB611")
    return json.loads(json.dumps(dict(profile=PROFILE, graph=name, passed=True,
        acquisition_sha256=RECEIPT_SHA256, stablehlo_sha256=saved["stablehlo_sha256"],
        optimized_hlo_sha256=optimized_identity[1], compiled_memory=dict(memory),
        structure=structure, numerical_promotion=False, performance_claim=False,
        scope="BOUNDED_HISTORY_HLO_REQUIRES_LIVE_MEMORY_AND_BOTH_ORIGINAL_EVENTS"), allow_nan=False))
