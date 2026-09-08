"""FP64 diagnostic of sealed router captures, never numerical admission.

Uses captured BF16 inputs, NOT a full-forward model reference. Validates the
existing campaign records and generation-bound originals before analysis.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.greenfield import prefill_router_protocol as protocol
from scripts.greenfield.ws32_prefill_layer_campaign import (
    validate_files,
    validate_record,
)


def analyze(root: Path) -> dict[str, Any]:
    """Recompute captured-input FP64 dots from the32 original owner shards."""
    runner = json.loads((root / "runner.json").read_text())
    validate_record(runner, runner["code_hash"], diagnostic=True)
    protocol.verify_fleet_replicas(root / "fleet", runner["workers"])
    owners = {}
    source_npz = []
    for worker in runner["workers"]:
        folder = root / "fleet" / f"rank{worker['launch_rank']}"
        validate_files(folder, worker, diagnostic=True)
        path = folder / "boundary.npz"
        source_npz.append(
            dict(
                path=str(path.relative_to(root)),
                sha256=sha256(path.read_bytes()).hexdigest(),
            )
        )
        with np.load(path, allow_pickle=False) as data:
            for item in worker["local_device_slots"]:
                device, slot = item["device_id"], item["device_slot"]
                owners[slot] = {
                    kind: protocol.read_capture(
                        data, f"{kind}_{device}", protocol.fields_for(kind)
                    )
                    for kind in protocol.KINDS
                }
                owners[slot]["weights"] = protocol.read_capture(
                    data, f"weights_{device}", ("router_weight", "correction_bias")
                )
    weights = np.concatenate(
        [
            np.concatenate(
                [
                    owners[e * 4 + f]["weights"]["router_weight"].astype(np.float64)
                    for f in range(4)
                ],
                axis=1,
            )
            for e in range(8)
        ],
        axis=0,
    )
    bias = np.concatenate(
        [
            owners[e * 4]["weights"]["correction_bias"].astype(np.float64)
            for e in range(8)
        ]
    )
    result = dict(
        tag=root.name,
        code_hash=runner["code_hash"],
        analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__,
        source_npz=source_npz,
        diagnostic_only=True,
        numerical_admission=False,
        performance_claim=False,
        reference_scope="FP64_DOT_OF_CAPTURED_BF16_ROUTER_INPUTS_NOT_FULL_MODEL_FORWARD",
        original_routes_reproduced_on32_owners=True,
        feature_input_differences={},
        sources={},
        peak_hbm_including_reference=max(
            s["stats"]["peak_bytes_in_use"]
            for w in runner["workers"]
            for s in w["device_memory_stats_including_reference"]
        ),
    )
    for f in range(4):
        a, r = owners[f]["actual"], owners[f]["reference"]
        result["feature_input_differences"][str(f)] = np.argwhere(
            a["router_input"] != r["router_input"]
        ).tolist()
    for source in ("actual", "reference"):
        hidden = np.concatenate(
            [owners[f][source]["router_input"].astype(np.float64) for f in range(4)],
            axis=1,
        )
        logits = hidden @ weights.T
        score = 1 / (1 + np.exp(-logits)) + bias
        ids = np.lexsort(
            (np.broadcast_to(np.arange(256), score.shape), -score), axis=1
        )[:, :8]
        stats = dict(
            fp64_row4_routes=ids[4].tolist(),
            fp64_row4_margin41_minus98=float(score[4, 41] - score[4, 98]),
            implementations={},
        )
        for kind in (source, source + "_batch", source + "_scalar"):
            captured = owners[0][kind]
            delta = captured["logits"].astype(np.float64) - logits
            partial_error = max(
                float(
                    np.max(
                        np.abs(
                            owners[e * 4 + f][kind]["partial_logits"].astype(np.float64)
                            - hidden[:, f * 1536 : (f + 1) * 1536]
                            @ weights[
                                e * 32 : (e + 1) * 32, f * 1536 : (f + 1) * 1536
                            ].T
                        )
                    )
                )
                for e in range(8)
                for f in range(4)
            )
            stats["implementations"][kind] = dict(
                logits_max_abs=float(np.max(np.abs(delta))),
                partial_max_abs=partial_error,
                row4_logit_error41_98=delta[4, [41, 98]].tolist(),
                row4_routes=captured["routes"][4].tolist(),
                row4_score_margin41_minus98=float(
                    captured["biased_scores"][4, 41] - captured["biased_scores"][4, 98]
                ),
            )
        result["sources"][source] = stats
    return result


def authenticate(root: Path) -> dict[str, Any]:
    """Read exact cloud generations of compact terminal and decisive originals."""
    from google.cloud import storage

    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    prefix = f"results/{root.name}/"
    terminal = bucket.get_blob(prefix + "SUCCESS")
    if terminal is None:
        raise ValueError("missing remote diagnostic SUCCESS")
    terminal_bytes = terminal.download_as_bytes(if_generation_match=terminal.generation)
    if terminal_bytes != (root / "SUCCESS").read_bytes():
        raise ValueError("terminal local/cloud bytes differ")
    success = json.loads(terminal_bytes)
    ledger_blob = bucket.get_blob(prefix + "archive_receipts.json")
    raw = ledger_blob.download_as_bytes(if_generation_match=ledger_blob.generation)
    if sha256(raw).hexdigest() != success["archive_receipts_sha256"]:
        raise ValueError("terminal archive-ledger binding differs")
    receipts = json.loads(raw)
    selected = []
    for receipt in receipts:
        relative = receipt["name"].removeprefix(prefix)
        path = root / relative
        if relative in (
            "summary.json",
            "runner.json",
            "census_post.txt",
            "devices_post.txt",
        ) or (
            relative.startswith("fleet/")
            and relative.endswith(("/boundary.npz", "/runner.json"))
        ):
            if (
                path.is_file()
                and sha256(path.read_bytes()).hexdigest() != receipt["original_sha256"]
            ):
                raise ValueError(f"local decisive sealed artifact differs: {relative}")
            generation = int(receipt["generation"])
            blob = bucket.blob(receipt["name"], generation=generation)
            blob.reload(if_generation_match=generation)
            data = blob.download_as_bytes(if_generation_match=generation)
            if (
                int(blob.size) != receipt["size"]
                or blob.crc32c != receipt["crc32c"]
                or sha256(data).hexdigest() != receipt["original_sha256"]
            ):
                raise ValueError("original cloud generation differs")
            selected.append(receipt)
    if len(selected) != 20:
        raise ValueError("expected terminal4 and16 original worker objects")
    summary = json.loads((root / "summary.json").read_text())
    if (
        sha256((root / "summary.json").read_bytes()).hexdigest()
        != success["summary_sha256"]
    ):
        raise ValueError("terminal summary binding differs")
    return dict(
        terminal_generation=str(terminal.generation),
        terminal_sha256=sha256(terminal_bytes).hexdigest(),
        archive_ledger_generation=str(ledger_blob.generation),
        archive_ledger_sha256=sha256(raw).hexdigest(),
        results_db_run_id=summary["results_db_run_id"],
        verified_original_receipts=selected,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    authentication = authenticate(args.root)
    result = analyze(args.root)
    result["authentication"] = authentication
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                sha256=sha256(args.output.read_bytes()).hexdigest(),
            )
        )
    )
