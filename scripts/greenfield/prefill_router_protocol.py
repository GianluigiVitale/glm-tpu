"""Original-array and structural contracts for an untimed router diagnostic.

This is NOT full-layer numerical admission. Reproduction refers only to the
archived ordered routes; originals did not observe router inputs or logits.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from scripts.greenfield.prefill_router_boundary import FIELDS
from scripts.greenfield.prefill_layer_evidence import (
    BF16,
    INPUT_FIELDS,
    decode_arrays,
    equal_bytes,
    host_case,
    input_hashes,
)
from scripts.greenfield.prefill_layer_hlo import FEATURE, EXPERT
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

KERNEL = "ws32_prefill_router_boundary_diagnostic"
PROTOCOL = "ws32-prefill-router-boundary-v1"
PROGRAMS = ("candidate", "reference", "router_batch", "router_scalar")
KINDS = (
    "actual",
    "reference",
    "actual_batch",
    "actual_scalar",
    "reference_batch",
    "reference_scalar",
)
ORIGINAL = (
    Path(__file__).resolve().parents[2]
    / "docs/artifacts/prefill-router-original-routes-20260908.json"
)
SHAPES = {
    "router_input": (17, 1536),
    "partial_logits": (17, 32),
    "logits": (17, 256),
    "bias": (256,),
    "biased_scores": (17, 256),
    "routes": (17, 8),
    "route_weights": (17, 8),
    "attention_update": (17, 1536),
    "combined_residual": (17, 1536),
    "post_residual": (17, 1536),
    "kv": (2, 64, 640),
    "prefix_health": (17,),
    "router_weight": (32, 1536),
    "correction_bias": (32,),
}
BF16_FIELDS = {
    "router_input",
    "attention_update",
    "combined_residual",
    "post_residual",
    "kv",
    "router_weight",
}


def is_router_tag(tag: str) -> bool:
    return bool(
        re.fullmatch(
            r"greenfield_fp8_ws32_prefill_router_boundary_diagnostic_l3_[a-zA-Z0-9_]+",
            tag,
        )
    )


def fields_for(kind: str) -> tuple[str, ...]:
    if kind not in KINDS:
        raise ValueError("unknown router diagnostic capture")
    return FIELDS if kind in ("actual", "reference") else FIELDS[:7]


def read_capture(arrays: Any, prefix: str, fields: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for name in fields:
        a = arrays[f"{prefix}__{name}"]
        dtype = (
            np.uint16
            if name in BF16_FIELDS
            else (
                np.int32
                if name == "routes"
                else np.bool_ if name == "prefix_health" else np.float32
            )
        )
        if a.dtype != dtype or a.shape != SHAPES[name]:
            raise ValueError(f"router original shape/dtype differs: {prefix}/{name}")
        a = a.view(BF16) if name in BF16_FIELDS else a
        if not np.isfinite(a.astype(np.float32)).all():
            raise ValueError(f"nonfinite router capture: {prefix}/{name}")
        result[name] = a
    return result


def _own_routes(value: dict[str, Any]) -> None:
    scores = value["biased_scores"]
    ids = np.broadcast_to(np.arange(256), scores.shape)
    canonical = np.lexsort((ids, -scores), axis=1)[:, :8].astype(np.int32)
    if not np.array_equal(value["routes"], canonical):
        raise ValueError("router does not select canonical own-score IDs/ties")


def verify_prefix(
    arrays: Any, slots: dict[int, int], ledger: dict[int, Any]
) -> dict[str, Any]:
    """Also used before replay, so failed reproduction retains prefix bytes."""
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host

    original = json.loads(ORIGINAL.read_text())
    if original["protocol"] != PROTOCOL:
        raise ValueError("original router protocol differs")
    host = decode_arrays(arrays, "input", INPUT_FIELDS)
    fixed = host_case(
        "boundary", build_rotary_table_host(1024, rotary_dim=64, theta=8e6)
    )
    if any(not equal_bytes(host[n], fixed[n]) for n in INPUT_FIELDS):
        raise ValueError(
            "router prefix inputs differ from original fixed boundary case"
        )
    result = {}
    for device, slot in slots.items():
        w = read_capture(
            arrays, f"weights_{device}", ("router_weight", "correction_bias")
        )
        for field, leaf in (
            ("router_weight", "weight"),
            ("correction_bias", "e_score_correction_bias"),
        ):
            expected = ledger[slot]["selected"][f"model.layers.3.mlp.gate.{leaf}"]
            if sha256(w[field].tobytes()).hexdigest() != expected:
                raise ValueError(
                    "observed router weight is not original checkpoint shard"
                )
        values = {
            k: read_capture(arrays, f"{k}_{device}", FIELDS)
            for k in ("actual", "reference")
        }
        for kind, v in values.items():
            _own_routes(v)
            if not v["prefix_health"].all():
                raise ValueError("prefix incoming/attention health failed")
            if not np.array_equal(v["routes"], original["original_routes"][kind]):
                raise ValueError(
                    f"original full17-row routes not reproduced: {kind}/{device}"
                )
        a, r = values["actual"], values["reference"]
        result[str(device)] = dict(
            slot=slot,
            original_routes_reproduced=True,
            router_input_equal=equal_bytes(a["router_input"], r["router_input"]),
            router_input_unequal_elements=int(
                np.count_nonzero(a["router_input"] != r["router_input"])
            ),
            row4_input_unequal_elements=int(
                np.count_nonzero(a["router_input"][4] != r["router_input"][4])
            ),
            row4_biased_41_minus_98={
                k: float(v["biased_scores"][4, 41] - v["biased_scores"][4, 98])
                for k, v in values.items()
            },
        )
    return dict(
        owners=result,
        input_sha256=input_hashes(host),
        original_fixture_sha256=sha256(ORIGINAL.read_bytes()).hexdigest(),
    )


def replay_file(
    path: Any, slots: dict[int, int], ledger: dict[int, Any]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        result = verify_prefix(arrays, slots, ledger)
        expected = {f"input__{n}" for n in INPUT_FIELDS}
        for d in slots:
            expected.update(
                f"weights_{d}__{n}" for n in ("router_weight", "correction_bias")
            )
            for kind in KINDS:
                expected.update(f"{kind}_{d}__{n}" for n in fields_for(kind))
                v = read_capture(arrays, f"{kind}_{d}", fields_for(kind))
                _own_routes(v)
            differences = {}
            for source in ("actual", "reference"):
                prefix = read_capture(arrays, f"{source}_{d}", FIELDS)
                b = read_capture(arrays, f"{source}_batch_{d}", FIELDS[:7])
                s = read_capture(arrays, f"{source}_scalar_{d}", FIELDS[:7])
                if not all(
                    equal_bytes(v["router_input"], prefix["router_input"])
                    for v in (b, s)
                ):
                    raise ValueError(
                        "router replay did not consume identical captured input"
                    )
                if not all(equal_bytes(v["bias"], prefix["bias"]) for v in (b, s)):
                    raise ValueError("router replay bias differs")
                differences[source] = dict(
                    partial_max_abs=float(
                        np.max(np.abs(b["partial_logits"] - s["partial_logits"]))
                    ),
                    logits_max_abs=float(np.max(np.abs(b["logits"] - s["logits"]))),
                    ordered_id_mismatches=int(
                        np.count_nonzero(b["routes"] != s["routes"])
                    ),
                    row4_routes_batch=b["routes"][4].tolist(),
                    row4_routes_scalar=s["routes"][4].tolist(),
                    row4_margin_batch=float(
                        b["biased_scores"][4, 41] - b["biased_scores"][4, 98]
                    ),
                    row4_margin_scalar=float(
                        s["biased_scores"][4, 41] - s["biased_scores"][4, 98]
                    ),
                )
            result["owners"][str(d)]["same_input_replays"] = differences
        if set(arrays.files) != expected:
            raise ValueError("router diagnostic original file inventory differs")
        result.update(
            evidence_complete=True, numerical_admission=False, performance_claim=False
        )
        return result


def verify_fleet_replicas(root: Path, records: list[dict[str, Any]]) -> None:
    """Compare original replica bytes without treating local partials as replicas."""
    seen: dict[tuple[str, str, int], bytes] = {}
    owners: set[int] = set()
    replicated = {"logits", "bias", "biased_scores", "routes", "route_weights"}
    feature = {"router_input", "attention_update", "combined_residual", "post_residual"}
    for record in records:
        with np.load(
            root / f"rank{record['launch_rank']}" / "boundary.npz", allow_pickle=False
        ) as arrays:
            for owner in record["local_device_slots"]:
                d, slot = owner["device_id"], owner["device_slot"]
                if slot in owners:
                    raise ValueError("duplicate fleet router slot")
                owners.add(slot)
                for kind in KINDS:
                    value = read_capture(arrays, f"{kind}_{d}", fields_for(kind))
                    for name, a in value.items():
                        if name in replicated:
                            group = 0
                        elif name in feature:
                            group = slot % 4
                        elif name == "kv":
                            group = slot // 4
                        else:
                            continue  # partial logits and health have distinct owners
                        key = kind, name, group
                        raw = a.tobytes()
                        if key in seen and seen[key] != raw:
                            raise ValueError(
                                f"router fleet replica disagreement: {key}"
                            )
                        seen[key] = raw
    if owners != set(range(32)):
        raise ValueError("router fleet requires32 unique slots")


def check_hlo(hlo: str, kind: str) -> dict[str, Any]:
    """Source-shaped prefix/replay guard; exact observed inventory is retained.

    Compiler layout/index annotations are bounded known local mechanisms, not
    model Pallas calls. This diagnostic guard cannot promote a full layer.
    """
    if kind not in PROGRAMS:
        raise ValueError("unknown router program")
    rows = 17 if kind in ("candidate", "router_batch") else 1
    prefix = kind in ("candidate", "reference")
    m = parse_hlo_module(hlo)
    collectives = [o for o in m.instructions if o.is_collective]
    payload = Counter()
    for o in collectives:
        shapes = o.operand_shapes if o.opcode == "all-reduce" else o.result_shapes
        for s in shapes:
            payload[o.opcode, o.maximum_group_size, s.dtype, s.element_count] += 1
    expected = Counter({("all-reduce", 4, "f32", rows * 32): 1})
    if prefix:
        expected.update(
            {
                ("all-reduce", 4, "f32", rows): 2,
                ("all-reduce", 4, "f32", rows * 2048): 1,
                ("all-reduce", 4, "f32", rows * 576): 1,
                ("all-reduce", 8, "bf16", rows * 2048 * 640): 1,
                ("all-reduce", 8, "f32", rows * 1536): 1,
            }
        )
    # Both scalar gathers lowered to tuple-merged disjoint-insert sums in the
    # original cf99c6a8 reference HLO. Matrix logits retained a gather there.
    variants = [
        expected + Counter([(logit_op, 8, "f32", rows * 256), (bias_op, 8, "f32", 256)])
        for logit_op in ("all-gather", "all-reduce")
        for bias_op in ("all-gather", "all-reduce")
    ]
    calls = [o for o in m.instructions if o.opcode == "custom-call"]
    pallas, helpers = [], []
    helper_valid = True
    for o in calls:
        match = re.search(r'custom_call_target="([^"]+)"', o.raw_line)
        target = match[1] if match else "missing"
        if target == "tpu_custom_call":
            pallas.append(o)
            continue
        helpers.append(o)
        if (
            len(o.result_shapes) != 1
            or "custom_call_has_side_effect=true" in o.raw_line
        ):
            helper_valid = False
            continue
        s = o.result_shapes[0]
        if target in ("AssumeGatherIndicesInBound", "GatherScatterIndicesBitpacked"):
            helper_valid &= (
                s.dtype == "s32"
                and s.element_count <= 17 * 2048 * 2
                and len(o.operand_names) == 1
                and o.operand_shapes == o.result_shapes
                and bool(o.op_name and o.op_name.endswith("/gather"))
            )
        elif target == "ConcatBitcast":
            helper_valid &= (
                s.dtype == "u8"
                and len(s.dimensions) == 2
                and s.element_count <= 2048 * 2048
                and len(o.operand_shapes) == len(o.operand_names) == 4
                and all(
                    t.dtype == "u8"
                    and t.dimensions == (s.dimensions[0] // 4, s.dimensions[1])
                    for t in o.operand_shapes
                )
            )
        else:
            helper_valid = False
    checks = dict(
        physical_groups=all(
            o.replica_groups in (FEATURE, EXPERT)
            and o.opcode in ("all-reduce", "all-gather")
            for o in collectives
        ),
        payload_inventory=payload in variants,
        pallas_inventory=(
            len(pallas) == (7 if prefix else 0)
            and sum("greenfield_fp8_block_matmul" in o.raw_line for o in pallas)
            == (4 if prefix else 0)
            and sum("greenfield_fp8_structured_kv_b" in o.raw_line for o in pallas)
            == (2 if prefix else 0)
            and sum("greenfield_pregathered_sparse_mla" in o.raw_line for o in pallas)
            == (1 if prefix else 0)
        ),
        local_helpers=helper_valid and len(helpers) <= 32,
        no_host_transport=not any(
            o.opcode in ("infeed", "outfeed", "send", "recv") for o in m.instructions
        ),
        no_expert_mlp=not any(
            "greenfield_prefill_grouped_raw_fp8" in o.raw_line for o in pallas
        ),
        bounded_arrays=all(
            s.element_count < 32 * 2048 * 1536
            for o in m.instructions
            for s in o.result_shapes
        ),
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        collectives=[o.to_dict() for o in collectives],
        helper_inventory=[
            dict(
                name=o.name,
                result=[s.to_dict() for s in o.result_shapes],
                target=re.search(r'custom_call_target="([^"]+)"', o.raw_line)[1],
            )
            for o in helpers
        ],
        diagnostic_only=True,
        performance_claim=False,
    )
