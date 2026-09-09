"""Real DB604/605 source/32-owner joins, fixture compute replay clearly isolated."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.greenfield import ws32_dense_frontier_evidence as base
from scripts.greenfield import ws32_dense_frontier_protocol as prior_protocol
from scripts.greenfield import ws32_dense_norm_evidence as evidence
from scripts.greenfield import ws32_dense_norm_originals as originals_module
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_layer_numerical import FIELDS
from tests.greenfield.validation.test_ws32_dense_frontier_execution import ORIGINAL


def test_all8_original_bindings_norm_fleet_and_mutations(tmp_path, monkeypatch):
    repo = Path.cwd()
    source = Path("/home/gianl/glm-run") / originals_module.TAG / "fleet"
    tag = "greenfield_fp8_ws32_dense_norm_d01_20260909T180000000Z"
    pin = "a" * 40
    records = []
    bindings = {}
    immutable = []
    calls = []
    for rank in range(8):
        old, arrays, identity = originals_module.load_bundle(
            source / f"rank{rank}", repo=repo, rank=rank
        )
        del arrays
        immutable.append(identity)
        record = deepcopy(old)
        record.update(
            tag=tag,
            code_hash=pin,
            protocol=protocol.PROTOCOL,
            kernel=protocol.KERNEL,
            profile=protocol.PROFILE,
            current_phase="norm/comparison",
            norm_originals=identity,
        )
        record["programs"] = {
            name: dict(
                stablehlo_sha256="c" * 64,
                optimized_hlo_sha256="d" * 64,
                compiled_memory={"fixture": 1},
            )
            for name in protocol.PROGRAMS
        }
        for owner in record["local_device_slots"]:
            bindings[owner["device_slot"]] = dict(
                full_file_sha256=owner["expected_full_file_sha256_not_verified"],
                selected=owner["observed_selected_tensor_sha256"],
            )
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
                "protocol",
                "norm_originals",
            )
        }
        preflight["combined_reference_bytes"] = (
            prior_protocol.LEDGER_PIN["size"]
            + sum(
                p["size"]
                for p in prior_protocol.reference_pins(ORIGINAL, rank).values()
            )
            + identity["bytes"]
        )
        _atomic_json(folder / "retained_preflight.json", preflight)
        record["retained_preflight_sha256"] = sha256(
            (folder / "retained_preflight.json").read_bytes()
        ).hexdigest()
        records.append(record)
    monkeypatch.setattr(base, "checkpoint_bindings", lambda repo: bindings)
    monkeypatch.setattr(
        base,
        "replay_execution",
        lambda root, record, slots, **kw: calls.append(record["launch_rank"]),
    )
    monkeypatch.setattr(
        base,
        "replay_wk",
        lambda root, record, slots, bindings: [
            dict(layer=l, slot=s, bf16_sha256=str(l), fp32_sha256=str(l))
            for l in (0, 1)
            for s in slots.values()
        ],
    )

    def output(root, record, slots, witness, originals, bindings):
        assert set(originals) == set(evidence.CAPTURE_NAMES)
        hashes = {
            (name, f"slot{s}_layer{l}__{f}"): "replica_fixture"
            for name, _, end in base.CAPSULES
            for s in slots.values()
            for l in (0, 1)
            for f in FIELDS
            if end or f not in ("kv", "index", "repair")
        }
        return (
            dict(
                owners=[
                    dict(branch=b, slot=s, reproduced=True)
                    for b in ("wide_final", "narrow_128")
                    for s in slots.values()
                ],
                reproduced=True,
                cause_claim=False,
            ),
            hashes,
        )

    monkeypatch.setattr(evidence, "replay_outputs", output)

    def replay(rows, norm_root=source):
        return base.validate_fleet(
            tmp_path,
            rows,
            pin=pin,
            tag=tag,
            repo=repo,
            original_root=ORIGINAL,
            norm_original_root=norm_root,
        )

    report = replay(records)
    assert (
        report["hosts"] == 8
        and report["owners"] == 32
        and report["model_calls_per_host"] == 14
    )
    assert len(report["norm_replays"]) == 8 and len(report["comparisons"]) == 64
    assert calls == list(range(8)) and not report["cause_claim"]
    for change in (
        "source",
        "generation",
        "selected",
        "protocol",
        "mesh",
        "missing_root",
        "preflight_bytes",
    ):
        changed = deepcopy(records)
        if change == "source":
            changed[0]["norm_originals"]["code_hash"] = "e" * 40
        elif change == "generation":
            changed[0]["norm_originals"]["original_pins"]["runner.json"][
                "generation"
            ] = "1"
        elif change == "selected":
            changed[0]["local_device_slots"][0]["observed_selected_tensor_sha256"][
                evidence.NORM_WEIGHT
            ] = ("f" * 64)
        elif change == "protocol":
            changed[0]["protocol"] = prior_protocol.PROTOCOL
        elif change == "mesh":
            changed[0]["physical_device_ids"][0][0] = 999
        elif change == "preflight_bytes":
            path = tmp_path / "rank0/retained_preflight.json"
            original = path.read_bytes()
            value = json.loads(original)
            value["combined_reference_bytes"] += 1
            _atomic_json(path, value)
            changed[0]["retained_preflight_sha256"] = sha256(
                path.read_bytes()
            ).hexdigest()
        try:
            with pytest.raises(ValueError):
                replay(changed, None if change == "missing_root" else source)
        finally:
            if change == "preflight_bytes":
                path.write_bytes(original)
