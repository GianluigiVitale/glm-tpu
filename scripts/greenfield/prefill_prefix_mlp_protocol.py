"""DB585 observable-prefix reproduction followed by completed-input scalar MLP.

Diagnostic only. The graph called candidate is the UNCHANGED scalar prefix,
not a batched layer candidate. No full-layer or pre-attention-norm claim.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from scripts.greenfield import prefill_router_protocol as router
from scripts.greenfield.prefill_layer_evidence import INPUT_FIELDS, input_hashes

KERNEL = "ws32_prefill_prefix_mlp_diagnostic"
PROTOCOL = "ws32-prefill-db585-prefix-completed-mlp-v1"
PROGRAMS = ("candidate", "reference")
REFERENCE_SCOPE = "DB585_SCALAR_PREFIX_COMPLETED_MLP_BOUNDARY_NOT_FULL_LAYER"
REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "docs/artifacts/prefill-router-captured-input-fp64-20260908.json"
ANALYSIS_SHA = "9bb941d6caa28845987f31bd2132f9e5fa9758942b7427ac012832d25d8113f0"
FIXTURE = REPO / "docs/artifacts/prefill-db585-prefix-fingerprints-20260908.json"


def is_prefix_mlp_tag(tag: str) -> bool:
    return bool(
        re.fullmatch(
            r"greenfield_fp8_ws32_prefill_prefix_mlp_diagnostic_l3_[a-zA-Z0-9_]+", tag
        )
    )


def fingerprints(value: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: sha256(a.tobytes()).hexdigest() for name, a in value.items()}


def build_fixture(root: Path) -> dict[str, Any]:
    """Derive compact expectations from the already generation-verified DB585 bytes."""
    raw = ANALYSIS.read_bytes()
    if sha256(raw).hexdigest() != ANALYSIS_SHA:
        raise ValueError("DB585 analysis pin differs")
    analysis = json.loads(raw)
    if root.name != analysis["tag"]:
        raise ValueError("DB585 source tag differs")
    receipts = {
        r["name"]: r for r in analysis["authentication"]["verified_original_receipts"]
    }
    owners = {}
    sources = []
    for rank in range(8):
        data = {}
        for name in ("runner.json", "boundary.npz"):
            path = root / "fleet" / f"rank{rank}" / name
            receipt = receipts[f"results/{root.name}/fleet/rank{rank}/{name}"]
            raw = path.read_bytes()
            if (
                len(raw) != receipt["size"]
                or sha256(raw).hexdigest() != receipt["original_sha256"]
            ):
                raise ValueError("DB585 original receipt binding differs")
            data[name] = path
            sources.append(receipt)
        worker = json.loads(data["runner.json"].read_text())
        if (
            worker["launch_rank"] != rank
            or worker["code_hash"] != analysis["code_hash"]
        ):
            raise ValueError("DB585 worker identity differs")
        with np.load(data["boundary.npz"], allow_pickle=False) as arrays:
            for item in worker["local_device_slots"]:
                d, slot = item["device_id"], item["device_slot"]
                if str(slot) in owners:
                    raise ValueError("duplicate DB585 slot")
                prefix = router.read_capture(arrays, f"reference_{d}", router.FIELDS)
                scalar = router.read_capture(
                    arrays, f"reference_scalar_{d}", router.FIELDS[:7]
                )
                weights = router.read_capture(
                    arrays, f"weights_{d}", ("router_weight", "correction_bias")
                )
                if not router.equal_bytes(
                    prefix["router_input"], scalar["router_input"]
                ):
                    raise ValueError("DB585 completed-input replay binding differs")
                owners[str(slot)] = dict(
                    prefix=fingerprints(prefix),
                    weights=fingerprints(weights),
                    suffix_routes=scalar["routes"].tolist(),
                    source_device_id=d,
                    prefix_routes=prefix["routes"].tolist(),
                )
    if set(owners) != {str(s) for s in range(32)}:
        raise ValueError("DB585 source must cover32 owners")
    return dict(
        protocol=PROTOCOL,
        source_analysis_sha256=ANALYSIS_SHA,
        source_tag=root.name,
        source_code_hash=analysis["code_hash"],
        source_receipts=sources,
        owners=owners,
        numerical_admission=False,
        performance_claim=False,
    )


def load_fixture() -> dict[str, Any]:
    value = json.loads(FIXTURE.read_text())
    if (
        value["protocol"] != PROTOCOL
        or value["source_analysis_sha256"] != ANALYSIS_SHA
        or set(value["owners"]) != {str(s) for s in range(32)}
    ):
        raise ValueError("DB585 fingerprint fixture identity differs")
    return value


def verify_prefix(
    arrays: Any, slots: dict[int, int], ledger: dict[int, Any]
) -> dict[str, Any]:
    """Must pass on EVERY host before the first MLP execution, not only afterwards."""
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host

    fixture = load_fixture()
    if (
        len(slots) != 4
        or len(set(slots.values())) != 4
        or any(s not in range(32) for s in slots.values())
    ):
        raise ValueError("prefix needs four distinct local slots")
    host = router.decode_arrays(arrays, "input", INPUT_FIELDS)
    fixed = router.host_case(
        "boundary", build_rotary_table_host(1024, rotary_dim=64, theta=8e6)
    )
    if any(not router.equal_bytes(host[n], fixed[n]) for n in INPUT_FIELDS):
        raise ValueError("prefix initial inputs differ")
    owners = {}
    for d, slot in slots.items():
        expected = fixture["owners"][str(slot)]
        prefix = router.read_capture(arrays, f"prefix_{d}", router.FIELDS)
        weights = router.read_capture(
            arrays, f"weights_{d}", ("router_weight", "correction_bias")
        )
        if fingerprints(prefix) != expected["prefix"] or not np.array_equal(
            prefix["routes"], expected["prefix_routes"]
        ):
            raise ValueError(f"complete DB585 prefix fingerprint differs: slot{slot}")
        if fingerprints(weights) != expected["weights"]:
            raise ValueError("DB585 observed weights differ")
        for field, leaf in (
            ("router_weight", "weight"),
            ("correction_bias", "e_score_correction_bias"),
        ):
            if (
                sha256(weights[field].tobytes()).hexdigest()
                != ledger[slot]["selected"][f"model.layers.3.mlp.gate.{leaf}"]
            ):
                raise ValueError("prefix weights differ from selected checkpoint")
        router._own_routes(prefix)
        if not prefix["prefix_health"].all():
            raise ValueError("prefix health failed")
        owners[str(d)] = dict(
            slot=slot, all12_prefix_fields_exact=True, original_routes_reproduced=True
        )
    return dict(
        owners=owners,
        input_sha256=input_hashes(host),
        fingerprint_fixture_sha256=sha256(FIXTURE.read_bytes()).hexdigest(),
    )


def replay_file(
    path: Path, slots: dict[int, int], ledger: dict[int, Any]
) -> dict[str, Any]:
    fixture = load_fixture()
    with np.load(path, allow_pickle=False) as arrays:
        result = verify_prefix(arrays, slots, ledger)
        expected = {f"input__{n}" for n in INPUT_FIELDS}
        for d, slot in slots.items():
            expected.update(f"prefix_{d}__{n}" for n in router.FIELDS)
            expected.update(
                f"weights_{d}__{n}" for n in ("router_weight", "correction_bias")
            )
            expected.update(
                f"suffix_{d}__{n}"
                for n in ("router_input", "output", "routes", "route_weights")
            )
            source = router.read_capture(arrays, f"prefix_{d}", ("router_input",))
            suffix = router.read_capture(
                arrays, f"suffix_{d}", ("router_input", "routes", "route_weights")
            )
            output = arrays[f"suffix_{d}__output"]
            if (
                output.dtype != np.uint16
                or output.shape != (17, 1536)
                or not np.isfinite(output.view(router.BF16).astype(np.float32)).all()
            ):
                raise ValueError("MLP output shape/dtype/finiteness differs")
            if not router.equal_bytes(source["router_input"], suffix["router_input"]):
                raise ValueError("MLP did not consume exact completed prefix input")
            if not np.array_equal(
                suffix["routes"], fixture["owners"][str(slot)]["suffix_routes"]
            ):
                raise ValueError(
                    "MLP routes differ from DB585 same-input scalar replay"
                )
            result["owners"][str(d)]["suffix_routes_reproduced"] = True
        if set(arrays.files) != expected:
            raise ValueError("prefix/MLP original array inventory differs")
    return dict(
        **result,
        evidence_complete=True,
        numerical_admission=False,
        performance_claim=False,
    )


def check_hlo(hlo: str, name: str) -> dict[str, Any]:
    from scripts.greenfield.prefill_materialized_reference import check_reference_hlo

    if name == "candidate":
        return router.check_hlo(hlo, "reference")
    if name == "reference":
        return check_reference_hlo(hlo, "reference")
    raise ValueError("unknown prefix/MLP graph")


def verify_fleet_replicas(root: Path, records: list[dict[str, Any]]) -> None:
    """Prefix replicas are bound to DB585; check new MLP feature/output replicas."""
    outputs, routes, weights = {}, [], []
    for r in records:
        with np.load(
            root / f"rank{r['launch_rank']}" / "boundary.npz", allow_pickle=False
        ) as a:
            for s in r["local_device_slots"]:
                d, slot = s["device_id"], s["device_slot"]
                out = a[f"suffix_{d}__output"]
                feature = slot % 4
                if feature in outputs and not router.equal_bytes(out, outputs[feature]):
                    raise ValueError("MLP feature replicas disagree")
                outputs[feature] = out.copy()
                routes.append(a[f"suffix_{d}__routes"].copy())
                weights.append(a[f"suffix_{d}__route_weights"].copy())
    if len(routes) != 32 or any(
        not router.equal_bytes(v, values[0])
        for values in (routes, weights)
        for v in values
    ):
        raise ValueError("MLP routing replicas disagree")


if __name__ == "__main__":
    import argparse
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _atomic_json(args.output, build_fixture(args.source_root))
