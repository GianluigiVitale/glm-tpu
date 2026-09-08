"""Generation-bound v2 refusal analysis; no TPU and no admission promotion."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from scripts.greenfield.prefill_layer_evidence import BF16, replay_case
from scripts.greenfield.prefill_layer_hlo import check_layer_hlo
from scripts.greenfield.prefill_materialized_reference import (
    PROTOCOL,
    check_reference_hlo,
)
from scripts.greenfield.ws32_prefill_layer_campaign import checkpoint_ledger

TAG = "greenfield_fp8_ws32_prefill_layer_materialized_admission_l3_20260908T005509603049814Z"
PIN = "e7ba4a9eca914d80fb12254dff283521544d39d5"
BASE = "greenfield_fp8_ws32_prefill_router_boundary_diagnostic_l3_20260908T003156848804906Z"


def main() -> None:
    from google.cloud import storage

    output = Path(
        "docs/artifacts/prefill-materialized-v2-refusal-fp64-replicas-20260908.json"
    )
    if output.exists():
        raise FileExistsError(output)
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    _, tensors = checkpoint_ledger(3)
    original = json.loads(
        Path(
            "docs/artifacts/prefill-router-captured-input-fp64-20260908.json"
        ).read_text()
    )
    prior_sha = {x["path"]: x["sha256"] for x in original["source_npz"]}
    receipt_rows, owners, hostnames, programs = [], {}, set(), {}
    for rank in range(8):
        prefix = f"results/{TAG}/workers/rank{rank}/"
        ledger_blob = bucket.get_blob(prefix + "worker_receipts.json")
        raw = ledger_blob.download_as_bytes(if_generation_match=ledger_blob.generation)
        receipts = json.loads(raw)
        row = dict(
            rank=rank,
            ledger_generation=str(ledger_blob.generation),
            ledger_sha256=sha256(raw).hexdigest(),
            originals=[],
        )
        files = {}
        for receipt in receipts:
            name = receipt["name"].removeprefix(prefix)
            if name not in {
                "runner.json",
                "empty.npz",
                "boundary.npz",
                "boundary.reference_input.npz",
                *(
                    f"{n}.optimized_hlo.txt"
                    for n in ("candidate", "reference_prefix", "reference")
                ),
            }:
                continue
            generation = int(receipt["generation"])
            blob = bucket.blob(receipt["name"], generation=generation)
            blob.reload(if_generation_match=generation)
            data = blob.download_as_bytes(if_generation_match=generation)
            if (
                int(blob.size) != receipt["size"]
                or blob.crc32c != receipt["crc32c"]
                or sha256(data).hexdigest() != receipt["original_sha256"]
            ):
                raise ValueError("original generation/size/CRC/SHA differs")
            files[name] = data
            row["originals"].append(receipt)
        if len(files) != 7:
            raise ValueError("expected seven decisive originals per rank")
        record = json.loads(files["runner.json"])
        if (
            record["status"] != "FAILED"
            or record["protocol"] != PROTOCOL
            or record["code_hash"] != PIN
            or record["launch_rank"] != rank
        ):
            raise ValueError("failed worker identity differs")
        hostnames.add(record["hostname"])
        slots = {r["device_id"]: r["device_slot"] for r in record["local_device_slots"]}
        for r in record["local_device_slots"]:
            if (
                r["observed_selected_tensor_sha256"]
                != tensors[r["device_slot"]]["selected"]
            ):
                raise ValueError("original selected checkpoint hashes differ")
        for name in ("candidate", "reference_prefix", "reference"):
            hlo = files[f"{name}.optimized_hlo.txt"]
            digest = sha256(hlo).hexdigest()
            if digest != record["programs"][name]["optimized_hlo_sha256"] or (
                name in programs and programs[name] != digest
            ):
                raise ValueError("original fleet HLO identities differ")
            programs[name] = digest
            proof = (
                check_layer_hlo(hlo.decode(), layer=3)
                if name == "candidate"
                else check_reference_hlo(hlo.decode(), name)
            )
            if not proof["passed"]:
                raise ValueError("original HLO replay fails")
        empty = replay_case(
            BytesIO(files["empty.npz"]), layer=3, case="empty", slots_by_device=slots
        )
        if not empty["passed"]:
            raise ValueError("empty original replay fails")
        prior_path = (
            Path("/home/gianl/glm-run") / BASE / f"fleet/rank{rank}/boundary.npz"
        )
        if (
            sha256(prior_path.read_bytes()).hexdigest()
            != prior_sha[f"fleet/rank{rank}/boundary.npz"]
        ):
            raise ValueError("DB585 captured bytes drifted")
        with np.load(
            BytesIO(files["boundary.npz"]), allow_pickle=False
        ) as boundary, np.load(
            BytesIO(files["boundary.reference_input.npz"]), allow_pickle=False
        ) as current, np.load(
            prior_path, allow_pickle=False
        ) as prior:
            for d, slot in slots.items():
                h = current[f"device_{d}"]
                if h.shape != (17, 1536) or h.dtype != np.uint16:
                    raise ValueError("v2 BF16 capture schema differs")
                old = prior[f"reference_{d}__router_input"]
                owners[slot] = dict(
                    hidden=h.view(BF16).astype(np.float64),
                    weight=prior[f"weights_{d}__router_weight"]
                    .view(BF16)
                    .astype(np.float64),
                    bias=prior[f"weights_{d}__correction_bias"].astype(np.float64),
                    actual=boundary[f"actual_{d}__routes"].copy(),
                    reference=boundary[f"reference_{d}__routes"].copy(),
                    input_diff=int(np.count_nonzero(h != old)),
                    row4_input_diff=int(np.count_nonzero(h[4] != old[4])),
                )
        receipt_rows.append(row)
    if set(owners) != set(range(32)) or len(hostnames) != 8:
        raise ValueError("fleet evidence incomplete")
    for slot, owner in owners.items():
        if not all(np.isfinite(owner[n]).all() for n in ("hidden", "weight", "bias")):
            raise ValueError("nonfinite captured FP64 inputs/weights")
        if owner["hidden"].tobytes() != owners[slot % 4]["hidden"].tobytes():
            raise ValueError("v2 BF16 input feature replicas disagree")
    hidden = np.concatenate([owners[f]["hidden"] for f in range(4)], axis=1)
    weight = np.concatenate(
        [
            np.concatenate([owners[e * 4 + f]["weight"] for f in range(4)], axis=1)
            for e in range(8)
        ],
        axis=0,
    )
    bias = np.concatenate([owners[e * 4]["bias"] for e in range(8)])
    logits = hidden @ weight.T
    scores = 1 / (1 + np.exp(-logits)) + bias
    ids = np.lexsort((np.broadcast_to(np.arange(256), scores.shape), -scores), axis=1)[
        :, :8
    ]
    result = dict(
        tag=TAG,
        pin=PIN,
        diagnostic_only=True,
        numerical_admission=False,
        performance_claim=False,
        analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        verified_original_receipts=receipt_rows,
        empty_pass32=True,
        hlo_sha256=programs,
        fp64_scope="CAPTURED_BF16_INPUT_AND_CHECKPOINT_ROUTER_WEIGHTS_NOT_FULL_FORWARD",
        input_feature_replicas_exact32=True,
        captured_inputs_weights_finite=True,
        prior_unverified_replica_analysis_sha256=sha256(
            Path(
                "docs/artifacts/prefill-materialized-v2-refusal-fp64-20260908.json"
            ).read_bytes()
        ).hexdigest(),
        fp64_row4_routes=ids[4].tolist(),
        fp64_row4_margin41_minus98=float(scores[4, 41] - scores[4, 98]),
        owners={
            str(s): dict(
                input_diff_vs_db585=v["input_diff"],
                row4_input_diff=v["row4_input_diff"],
                candidate_reference_route_differences=np.argwhere(
                    v["actual"] != v["reference"]
                ).tolist(),
                reference_fp64_route_differences=np.argwhere(
                    v["reference"] != ids
                ).tolist(),
                row4_reference=v["reference"][4].tolist(),
            )
            for s, v in owners.items()
        },
    )
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                fp64_row4=result["fp64_row4_routes"],
                margin=result["fp64_row4_margin41_minus98"],
                reference_fp64_differences=sum(
                    len(x["reference_fp64_route_differences"])
                    for x in result["owners"].values()
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
