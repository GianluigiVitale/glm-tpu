"""Offline own-input diagnostics for DB592; never admission or original causality.

Reuses the original-array replay and physical-owner binding. FP64 arithmetic
starts at captured operands, NOT tokens/checkpoint full-forward R of section21.
No TPU initialization, checkpoint payload read, or new capture is required.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from scripts.greenfield.analyze_prefill_window_failure import validate_mapping
from scripts.greenfield.prefill_layer_evidence import decode_arrays, equal_bytes
from scripts.greenfield.prefill_layer_numerical import FIELDS
from scripts.greenfield.prefill_window_boundary_evidence import replay_arrays
from scripts.greenfield.prefill_window_protocol import BF16
from scripts.greenfield.ws32_prefill_layer_campaign import checkpoint_ledger


TAG = "greenfield_fp8_ws32_prefill_window_boundary_diagnostic_l6_20260908T155153878825613Z"
PIN = "988818433e002049e21034a114efae9ed2e9aa81"
SUCCESS_SHA = "5e8f72b449d5da5742b304f61e8213cad78d8ff6af8325abcb98b5449fdeaafd"


def ordered_ids(scores: np.ndarray, count: int = 8) -> np.ndarray:
    """Canonical descending-score, lower-index ties for finite dense rows."""
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("finite rank-two score rows required")
    if not 0 < count <= scores.shape[1]:
        raise ValueError("invalid selection count")
    return np.argsort(-scores, axis=1, kind="stable")[:, :count]


def delta(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Diagnostic statistics only, with no invented numerical threshold."""
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("aligned finite operands required")
    d = a.astype(np.float64) - b.astype(np.float64)
    return dict(
        max_abs=float(np.max(np.abs(d))),
        mean=float(np.mean(d)),
        rms=float(np.sqrt(np.mean(d * d))),
        changed_elements=int(np.count_nonzero(d)),
    )


def row_changes(a: np.ndarray, b: np.ndarray) -> list[int]:
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("aligned rows required")
    return np.flatnonzero(np.any(a != b, axis=tuple(range(1, a.ndim)))).tolist()


def dsa_fp64(query: np.ndarray, keys: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """One row: signed head sum after scaled per-head dot and ReLU.

    Preserve captured F32 query/head values and BF16 keys as exact FP64 inputs.
    This does NOT emulate DEFAULT TPU contraction or silently round the query.
    """
    if query.ndim != 2 or keys.ndim != 2 or weights.shape != query.shape[:1]:
        raise ValueError("invalid one-row DSA geometry")
    dots = query.astype(np.float64) @ keys.astype(np.float64).T
    return weights.astype(np.float64) @ np.maximum(dots / np.sqrt(query.shape[1]), 0)


def align_scores(positions: np.ndarray, scores: np.ndarray, length: int) -> np.ndarray:
    """Below K, require every causal position exactly once before alignment."""
    live = positions >= 0
    p = positions[live]
    if not np.array_equal(np.sort(p), np.arange(length)):
        raise ValueError("selected row lacks exact complete causal coverage")
    return scores[live][np.argsort(p)]


def authenticate(root: Path) -> dict[str, Any]:
    """Read compact remote terminal/ledger; SHA-check existing local originals.

    No repeated half-GB download: the terminal authenticates the generation/SHA
    ledger, and local bytes must match it. This is not fresh payload readback.
    """
    from google.cloud import storage

    if root.name != TAG:
        raise ValueError("this analysis is fixed to DB592")
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    prefix = f"results/{TAG}/"
    remote = {}
    for name in ("SUCCESS", "archive_receipts.json"):
        blob = bucket.get_blob(prefix + name)
        if blob is None:
            raise ValueError("missing sealed terminal or archive ledger")
        raw = blob.download_as_bytes(
            if_generation_match=blob.generation, checksum="crc32c"
        )
        if raw != (root / name).read_bytes():
            raise ValueError("local/remote compact evidence differs")
        remote[name] = dict(
            generation=str(blob.generation),
            size=len(raw),
            crc32c=blob.crc32c,
            sha256=sha256(raw).hexdigest(),
        )
    if remote["SUCCESS"]["sha256"] != SUCCESS_SHA:
        raise ValueError("fixed DB592 terminal differs")
    terminal = json.loads((root / "SUCCESS").read_text())
    if terminal["archive_receipts_sha256"] != remote["archive_receipts.json"]["sha256"]:
        raise ValueError("terminal ledger binding differs")
    if (
        terminal["summary_sha256"]
        != sha256((root / "summary.json").read_bytes()).hexdigest()
    ):
        raise ValueError("terminal summary binding differs")
    receipts = json.loads((root / "archive_receipts.json").read_text())
    by_name = {r["name"].removeprefix(prefix): r for r in receipts}
    needed = ["summary.json", "census_post.txt", "devices_post.txt", "results_ckpt.db"]
    needed += [
        f"fleet/rank{r}/{n}" for r in range(8) for n in ("runner.json", "boundary.npz")
    ]
    for name in needed:
        raw = (root / name).read_bytes()
        entry = by_name[name]
        if (
            len(raw) != entry["size"]
            or sha256(raw).hexdigest() != entry["original_sha256"]
        ):
            raise ValueError(f"sealed local original differs: {name}")
    return dict(
        remote_compact_readback=remote,
        local_generation_bound_originals=[by_name[n] for n in needed],
        archive_objects=len(receipts),
        archive_bytes=sum(r["size"] for r in receipts),
        verification_scope="fresh_remote_terminal_ledger_and_local_payload_SHA_not_fresh_remote_payload_download",
    )


def load_owners(root: Path) -> tuple[dict, list]:
    """Validate all32 physical owners; retain only needed host operands."""
    _, ledger = checkpoint_ledger(6)
    owners, records, processes, mesh = {}, [], set(), None
    for rank in range(8):
        folder = root / "fleet" / f"rank{rank}"
        record = json.loads((folder / "runner.json").read_text())
        if (
            record["code_hash"] != PIN
            or record["launch_rank"] != rank
            or record["status"] != "SUCCESS"
        ):
            raise ValueError("fixed successful diagnostic identity differs")
        mesh = validate_mapping(record, mesh, processes)
        slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
        replay_arrays(folder, record, slots)
        with np.load(folder / "boundary.npz", allow_pickle=False) as z:
            for device, slot in slots.items():
                if slot in owners:
                    raise ValueError("duplicate owner")

                def capture(kind: str) -> dict:
                    schema = record["programs"][
                        "candidate" if kind == "actual" else "control"
                    ]["compiler_output_schema"]["captures"]
                    out = {}
                    for name, spec in schema.items():
                        value = z[f"capture_{kind}_{device}__{name}"]
                        if spec["dtype"] == "bfloat16":
                            value = value.view(BF16)
                        out[name] = value
                    return out

                a = capture("actual")
                tiles = [capture(f"tile{i}") for i in range(4)]
                control = {
                    n: (
                        np.concatenate([t[n] for t in tiles], axis=0)
                        if n not in ("router/weight", "router/bias")
                        else tiles[0][n]
                    )
                    for n in a
                    if n.startswith(("router/", "router_selection/"))
                }
                for t in tiles:
                    for n in ("router/weight", "router/bias"):
                        if not equal_bytes(t[n], a[n]):
                            raise ValueError(
                                "candidate/control static router operands differ"
                            )
                expected = ledger[slot]["selected"]
                for name, value in (
                    ("weight", a["router/weight"]),
                    (
                        "e_score_correction_bias",
                        a["router/bias"][slot // 4 * 32 : (slot // 4 + 1) * 32],
                    ),
                ):
                    if (
                        sha256(value.tobytes()).hexdigest()
                        != expected[f"model.layers.6.mlp.gate.{name}"]
                    ):
                        raise ValueError(
                            "captured router weight/bias differs from checkpoint"
                        )
                owners[slot] = dict(
                    actual=a,
                    control=control,
                    tiles=tiles,
                    original={
                        k: decode_arrays(z, f"{k}_{device}", FIELDS)
                        for k in ("actual", "control")
                    },
                )
        records.append(record)
        print(f"verified original captures rank{rank}", flush=True)
    if set(owners) != set(range(32)) or len({r["hostname"] for r in records}) != 8:
        raise ValueError("incomplete physical fleet")
    return owners, records


def analyze_router(owners: dict) -> dict:
    """Own-input FP64 projection plus exact selection of captured biased scores."""
    result = {}
    for kind in ("actual", "control"):
        # Hidden input replicates over expert, globals over both axes; partials
        # retain unique expert/feature ownership. Validate BEFORE deduplicating.
        for slot in range(32):
            v = owners[slot][kind]
            for n in ("router/input", "router/clean", "router/live"):
                if not equal_bytes(v[n], owners[slot % 4][kind][n]):
                    raise ValueError("same-feature router input replicas differ")
            for n in (
                "router/logits",
                "router/bias",
                "router_selection/biased_scores",
                "router_selection/indices",
                "router_selection/scores",
                "router_selection/weights",
            ):
                if not equal_bytes(v[n], owners[0][kind][n]):
                    raise ValueError("global router replicas differ")
            if not equal_bytes(
                v["router/local_logits"],
                owners[slot // 4 * 4][kind]["router/local_logits"],
            ):
                raise ValueError("feature-reduced logits replicas differ")
            expected = np.where(v["router/live"][:, None], v["router/input"], 0)
            if not equal_bytes(v["router/clean"], expected):
                raise ValueError("captured clean input differs from live mask")
        partials = [
            owners[s][kind]["router/clean"].astype(np.float64)
            @ owners[s][kind]["router/weight"].astype(np.float64).T
            for s in range(32)
        ]
        math_logits = np.concatenate(
            [sum(partials[e * 4 : (e + 1) * 4]) for e in range(8)], axis=1
        )
        v = owners[0][kind]
        captured_logits = np.concatenate(
            [owners[e * 4][kind]["router/local_logits"] for e in range(8)], axis=1
        )
        if not equal_bytes(captured_logits, v["router/logits"]):
            raise ValueError("expert gathered logits are not concatenated local blocks")
        own = ordered_ids(v["router_selection/biased_scores"])
        ids = v["router_selection/indices"]
        if not np.array_equal(own, ids) or not np.array_equal(
            ids, owners[0]["original"][kind]["routes"]
        ):
            raise ValueError("captured selection is not canonical own-score order")
        math_score = 1 / (1 + np.exp(-math_logits)) + v["router/bias"].astype(
            np.float64
        )
        math_ids = ordered_ids(math_score)
        captured_logit_sigmoid = 1 / (
            1 + np.exp(-v["router/logits"].astype(np.float64))
        )
        result[kind] = dict(
            own_score_order_exact_all32=True,
            partial_max_abs=max(
                delta(owners[s][kind]["router/partial"], partials[s])["max_abs"]
                for s in range(32)
            ),
            logits=delta(v["router/logits"], math_logits),
            biased_scores=delta(v["router_selection/biased_scores"], math_score),
            sigmoid_vs_fp64_of_captured_logits=delta(
                v["router_selection/scores"], captured_logit_sigmoid
            ),
            fp64_order_mismatch_rows=row_changes(ids, math_ids),
            fp64_set_mismatch_rows=row_changes(
                np.sort(ids, axis=1), np.sort(math_ids, axis=1)
            ),
            fp64_routes_at_row2=math_ids[2].tolist(),
            captured_routes_at_row2=ids[2].tolist(),
        )
    result["candidate_control"] = dict(
        input_by_feature={
            str(f): delta(
                owners[f]["actual"]["router/input"],
                owners[f]["control"]["router/input"],
            )
            for f in range(4)
        },
        input_changed_rows=sorted(
            set().union(
                *(
                    row_changes(
                        owners[f]["actual"]["router/input"],
                        owners[f]["control"]["router/input"],
                    )
                    for f in range(4)
                )
            )
        ),
        route_changed_rows=row_changes(
            owners[0]["actual"]["router_selection/indices"],
            owners[0]["control"]["router_selection/indices"],
        ),
    )
    return result


def analyze_dsa_row2(owners: dict) -> dict:
    """Position-align all508 causal scores at first differing row, all owners."""
    out, gathered = {}, {}
    for kind in ("actual", "control"):
        ds = [
            {
                n.removeprefix("tile0/dsa/" if kind == "actual" else "dsa/"): v
                for n, v in (
                    owners[s]["actual"] if kind == "actual" else owners[s]["tiles"][0]
                ).items()
                if n.startswith("tile0/dsa/" if kind == "actual" else "dsa/")
            }
            for s in range(32)
        ]
        for s in range(32):
            for n in ("query", "head_weights", "causal_lengths", "positions", "live"):
                if not equal_bytes(ds[s][n], ds[0][n]):
                    raise ValueError("replicated DSA query metadata differs")
            for n in ("keys", "logical_positions", "current_keys"):
                if not equal_bytes(ds[s][n], ds[s // 4 * 4][n]):
                    raise ValueError("feature-replicated DSA keys differ")
        length = int(ds[0]["causal_lengths"][2])
        positions = np.concatenate([ds[e * 4]["logical_positions"] for e in range(8)])
        keys = np.concatenate([ds[e * 4]["keys"] for e in range(8)])
        valid = (positions >= 0) & (positions < length)
        order = np.argsort(positions[valid])
        if not np.array_equal(positions[valid][order], np.arange(length)):
            raise ValueError("owner keys lack unique complete causal coverage")
        keys = keys[valid][order]
        q, h = ds[0]["query"][2], ds[0]["head_weights"][2]
        reference = dsa_fp64(q, keys, h)
        # This second calculation is an explicit diagnostic model, not an
        # assertion about an unobserved internal TPU contraction boundary.
        rounded_query = dsa_fp64(q.astype(BF16), keys, h)
        original = owners[0]["original"][kind]
        observed = align_scores(original["positions"][2], original["scores"][2], length)
        for s in range(32):
            other = owners[s]["original"][kind]
            if not equal_bytes(
                observed,
                align_scores(other["positions"][2], other["scores"][2], length),
            ):
                raise ValueError("DSA score replicas differ")
        out[kind] = dict(
            causal_length=length,
            captured_fp32_input_reference=delta(observed, reference),
            hypothetical_bf16_query_reference=delta(observed, rounded_query),
            own_score_order_exact=bool(
                np.array_equal(
                    ordered_ids(observed[None], length)[0],
                    original["positions"][2, :length],
                )
            ),
        )
        gathered[kind] = dict(
            query=q, head_weights=h, keys=keys, observed=observed, reference=reference
        )
    out["candidate_control"] = {
        n: delta(gathered["actual"][n], gathered["control"][n])
        for n in gathered["actual"]
    }
    return out


def analyze(root: Path) -> dict:
    auth = authenticate(root)
    owners, records = load_owners(root)
    peaks = {}
    for r in records:
        if (
            len(r["call_evidence"]) != 7
            or r["model_executable_calls"] != 5
            or r["wk_executable_calls"] != 2
        ):
            raise ValueError("fixed diagnostic executable count differs")
        for call in r["call_evidence"]:
            for d in call["post_memory"]:
                key = str(d["device_id"])
                peaks[key] = max(peaks.get(key, 0), d["peak_bytes_in_use"])
    if set(peaks) != {str(d) for d in range(32)}:
        raise ValueError("incomplete per-device measured peaks")
    code_bytes = {
        sum(
            p["compiled_memory"]["generated_code_size_in_bytes"]
            for p in r["programs"].values()
        )
        for r in records
    }
    if len(code_bytes) != 1:
        raise ValueError("fleet resident compiled code totals differ")
    expected_hosts = {r["hostname"] for r in records}
    normal = {
        line.split()[1]
        for line in (root / "census_post.txt").read_text().splitlines()
        if line.startswith("CENSUS_OK ")
    }
    device = [
        json.loads(line.removeprefix("FP8_IDLE "))
        for line in (root / "devices_post.txt").read_text().splitlines()
        if line.startswith("FP8_IDLE ")
    ]
    if (
        normal != expected_hosts
        or len(device) != 8
        or {d["host"] for d in device} != expected_hosts
    ):
        raise ValueError("sealed normal/root cleanup incomplete")
    with sqlite3.connect(f"file:{root/'results_ckpt.db'}?mode=ro", uri=True) as db:
        rows = db.execute(
            "select correct,score,latency_ms from items where run_id=592"
        ).fetchall()
    if rows != [(None, None, None)]:
        raise ValueError("diagnostic DB correctness/score/latency must remain NULL")
    return dict(
        tag=TAG,
        code_hash=PIN,
        db_run=592,
        diagnostic_only=True,
        numerical_admission=False,
        performance_claim=False,
        original_cause_established=False,
        reference_scope="CAPTURED_OWN_INPUT_FP64_NOT_SECTION21_FULL_FORWARD_R",
        classification=sorted({r["classification"] for r in records}),
        original_reproduction={
            str(s["device_slot"]): {
                k: {
                    n: v[n]
                    for n in (
                        "signature_reproduced",
                        "all_outputs_reproduced",
                        "fields_byte_identical",
                    )
                }
                for k, v in r["original_reproduction"][str(s["device_id"])].items()
            }
            for r in records
            for s in r["local_device_slots"]
        },
        post_call_allocator_peak_bytes_by_device=peaks,
        resident_compiled_code_bytes_separately_budgeted=code_bytes.pop(),
        memory_scope="selected_layer_plus_diagnostic_not_full_model_HBM; allocator_counter_not_complete_physical_HBM",
        cleanup=dict(normal_hosts=len(normal), root_device_hosts=len(device)),
        database=dict(correct=None, score=None, latency_ms=None),
        prior_analysis=dict(
            path="docs/artifacts/prefill-window-captured-input-fp64-20260908.json",
            sha256=sha256(
                Path(
                    "docs/artifacts/prefill-window-captured-input-fp64-20260908.json"
                ).read_bytes()
            ).hexdigest(),
            limitation="Initial offline analysis before separate sigmoid, DB and cleanup checks; not a protected model result",
        ),
        router=analyze_router(owners),
        dsa_row2=analyze_dsa_row2(owners),
        authentication=auth,
        numpy_version=np.__version__,
        analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("router", "dsa_row2")}))
