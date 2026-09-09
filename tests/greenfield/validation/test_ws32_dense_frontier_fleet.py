"""Fleet join mutations over actual DB604 ownership; compute replays are fixtures.

The separate evidence tests exercise real producer/consumer arrays and journals.
This file isolates join policy, not hardware or full outer transport composition.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.greenfield import ws32_dense_frontier_evidence as evidence
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield.ws32_dense_frontier_execution import (
    DSA_PINS,
    DSA_PIN_SOURCE_SHA,
)
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_layer_numerical import FIELDS
from tests.greenfield.validation.test_ws32_dense_frontier_execution import ORIGINAL


def test_actual32_owner_join_and_scope_mutations(tmp_path, monkeypatch):
    pin = "a" * 40
    tag = "greenfield_fp8_ws32_dense_frontier_d01_20260909T150000000Z"
    priors = [
        json.loads((ORIGINAL / protocol.original_names(i)[0]).read_bytes())
        for i in range(8)
    ]
    # Actual saved non-identity placement; ledger validation tested separately.
    monkeypatch.setattr(
        protocol, "load_reference", lambda root, rank: (priors[rank], {})
    )
    bindings = {
        o["device_slot"]: dict(
            full_file_sha256=o["file_sha256"], selected={"fixture_leaf": "b" * 64}
        )
        for prior in priors
        for o in prior["local_device_slots"]
    }
    monkeypatch.setattr(evidence, "checkpoint_bindings", lambda repo: bindings)
    ids = {
        o["device_slot"]: o["device_id"]
        for p in priors
        for o in p["local_device_slots"]
    }
    mesh = [[ids[e * 4 + f] for f in range(4)] for e in range(8)]
    records = []
    for rank, prior in enumerate(priors):
        record = dict(
            status="DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION",
            tag=tag,
            code_hash=pin,
            launch_rank=rank,
            protocol=protocol.PROTOCOL,
            kernel=protocol.KERNEL,
            profile=evidence.admission.PROFILE,
            diagnostic_only=True,
            compile_only=False,
            performance_claim=False,
            numerical_promotion=False,
            current_phase="dense/comparison",
            selected_layer_ids=[0, 1],
            include_embedding=True,
            payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
            integrity_scope="selected_layer_tensors_only_not_complete_checkpoint",
            original_tag=protocol.ORIGINAL_TAG,
            original_ledger_sha256=protocol.LEDGER_SHA,
            original_runner_sha256=sha256(
                (ORIGINAL / protocol.original_names(rank)[0]).read_bytes()
            ).hexdigest(),
            dsa_oracle_pins=DSA_PINS,
            dsa_pin_source_sha256=DSA_PIN_SOURCE_SHA,
            versions={"jax": "0.10.1", "libtpu": "0.0.41"},
            pid=1000 + rank,
            start_ticks=100,
            boot_id="fixture-boot",
            physical_device_ids=mesh,
            checkpoint_pins={"fixture": "no-payload-read"},
        )
        for key in (
            "hostname",
            "jax_process_index",
            "mesh_sha256",
            "topology_sha256",
            "topology_fleet_sha256",
            "checkpoint_manifest_sha256",
            "checkpoint_success_sha256",
            "source_inventory_sha256",
            "main_rope_table",
            "prompt_ids_sha256",
        ):
            record[key] = prior[key]
        record["local_device_slots"] = [
            dict(
                device_id=o["device_id"],
                device_slot=o["device_slot"],
                expected_full_file_sha256_not_verified=o["file_sha256"],
                observed_selected_tensor_sha256=bindings[o["device_slot"]]["selected"],
                selected_payload_bytes=protocol.PAYLOAD_BYTES,
            )
            for o in prior["local_device_slots"]
        ]
        record["programs"] = {
            n: dict(
                stablehlo_sha256="c" * 64,
                optimized_hlo_sha256="d" * 64,
                compiled_memory={"fixture": 1},
            )
            for n in evidence.worker.PROGRAMS
        }
        folder = tmp_path / f"rank{rank}"
        folder.mkdir()
        preflight = {
            k: record[k]
            for k in (
                "tag",
                "code_hash",
                "launch_rank",
                "hostname",
                "original_runner_sha256",
                "checkpoint_pins",
            )
        }
        _atomic_json(folder / "retained_preflight.json", preflight)
        record["retained_preflight_sha256"] = sha256(
            (folder / "retained_preflight.json").read_bytes()
        ).hexdigest()
        records.append(record)
    controls = dict(wk_bad=False, output_bad=False)
    calls = []
    monkeypatch.setattr(
        evidence,
        "replay_execution",
        lambda root, record, slots, **kw: calls.append(record["launch_rank"]),
    )

    def wk(root, record, slots, bindings):
        return [
            dict(
                layer=l,
                slot=s,
                bf16_sha256="bad" if controls["wk_bad"] and s == 31 else str(l),
                fp32_sha256=str(l),
            )
            for l in (0, 1)
            for s in slots.values()
        ]

    monkeypatch.setattr(evidence, "replay_wk", wk)

    def outputs(root, record, slots, witness):
        hashes = {
            (name, f"slot{s}_layer{l}__{field}"): (
                "bad"
                if controls["output_bad"] and s == 31 and field == "output"
                else "same"
            )
            for name, _, end in evidence.CAPSULES
            for s in slots.values()
            for l in (0, 1)
            for field in FIELDS
            if end or field not in ("kv", "index", "repair")
        }
        return (
            dict(
                owners=[
                    dict(branch=b, slot=s, reproduced=True)
                    for b in ("wide_final", "narrow_128")
                    for s in slots.values()
                ]
            ),
            hashes,
        )

    monkeypatch.setattr(evidence, "replay_outputs", outputs)

    def check(rows):
        return evidence.validate_fleet(
            tmp_path, rows, pin=pin, tag=tag, repo=Path.cwd(), original_root=ORIGINAL
        )

    report = check(records)
    assert (
        report["reproduced"]
        and report["owners"] == 32
        and len(report["comparisons"]) == 64
        and calls == list(range(8))
    )
    for mode in (
        "missing_rank",
        "duplicate_rank",
        "owner",
        "selected_hash",
        "process",
        "mesh",
        "graph",
        "promotion",
    ):
        changed = deepcopy(records)
        if mode == "missing_rank":
            changed.pop()
        elif mode == "duplicate_rank":
            changed[7]["launch_rank"] = 0
        elif mode == "owner":
            changed[0]["local_device_slots"][0]["device_slot"] = 0
        elif mode == "selected_hash":
            changed[0]["local_device_slots"][0]["observed_selected_tensor_sha256"][
                "fixture_leaf"
            ] = ("f" * 64)
        elif mode == "process":
            changed[0]["jax_process_index"] = 0
        elif mode == "mesh":
            changed[0]["physical_device_ids"][0][0] = -1
        elif mode == "graph":
            changed[7]["programs"]["dense01"]["optimized_hlo_sha256"] = "e" * 64
        else:
            changed[0]["numerical_promotion"] = True
        with pytest.raises(ValueError):
            check(changed)
    for kind in ("wk_bad", "output_bad"):
        controls[kind] = True
        with pytest.raises(ValueError, match="replicas disagree"):
            check(records)
        controls[kind] = False
