"""Actual collector/DB; panel mode uses real DB596 originals on all owners.

Historical modes retain real rank0 and synthetic other-owner capsules.
No deleted rank1..7 cache restoration and no distributed model execution. The
paired variant uses actual CPU-preregistered StableHLO but the old optimized
graph as a structural compiler fixture, NOT evidence of paired TPU compilation.
"""

from copy import deepcopy
from hashlib import sha256
import gzip
import json
import os
import ast
import subprocess
import shutil
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest
import numpy as np

from scripts.greenfield import prefill_phase_baseline as phase
from scripts.greenfield import prefill_phase_evidence as evidence
from scripts.greenfield import prefill_phase_originals as originals
from scripts.greenfield import prefill_completed_window_assembly as assembly
from scripts.greenfield import prefill_phase_variant as variants
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from tests.greenfield.hlo.test_prefill_phase_evidence import collected


def production_stablehlo(*, panels=False, graph="prefix"):
    """Lower production abstract inputs on CPU; no payload or TPU backend."""
    source = Path("tests/greenfield/hlo/test_prefill_completed_window.py")
    tree = ast.parse(source.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "test_completed_window_production_shapes_without_payloads"
    )
    setup = next(
        n.value.value
        for n in fn.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    )
    code = (
        setup.split("programs=prepare_programs", 1)[0]
        + """
from unittest.mock import patch
programs=prepare_programs(mesh=mesh,config=config,weights=w,completed_window=True,paired_position_sort=True,expert_panels=PANEL_FLAG)
name,fn,args=next(p for p in programs if p[0]=='GRAPH_NAME')
with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
 text=str(fn.trace(*args).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo'))
print(text,end='')
"""
    )
    code = code.replace("PANEL_FLAG", repr(panels)).replace("GRAPH_NAME", graph)
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=32",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    from scripts.greenfield import prefill_paired_sort_admission as paired

    from scripts.greenfield import prefill_panel_admission as panel

    assert sha256(result.stdout.encode()).hexdigest() == (
        panel.CANDIDATE_SHA if panels else paired.PREFIX_SHA
    )
    return result.stdout


@pytest.fixture(scope="module")
def paired_stablehlo():
    return production_stablehlo()


@pytest.fixture(scope="module")
def panel_stablehlo():
    return production_stablehlo(panels=True, graph="candidate")


@pytest.mark.parametrize(
    "paired", [False, True, "panel"], ids=["historical", "paired", "panel"]
)
def test_phase_fleet_publication_collection_and_real_db(
    collected, tmp_path, monkeypatch, paired, request
):
    from google.cloud import storage
    from scripts.greenfield import collect_ws32_worker_evidence as publication
    from scripts.analysis import parse_xplane
    from scripts.analysis.test_parse_xplane import fake_core

    base, template, old_slots = collected
    template = deepcopy(template)
    panel = paired == "panel"
    variant = variants.variants()[2 if panel else int(paired)]
    capsule = deepcopy(originals.load_capsule())
    archive = Path("/home/gianl/glm-run") / capsule["source_tag"] / "fleet"
    if panel:
        # Rebinding rank0 bytes to another owner is not a valid cache fixture:
        # untouched cache rows depend on the physical context shard. Reuse
        # retained DB596 originals, already exact against DB594, with no download.
        archive = Path(
            "/home/gianl/glm-run/greenfield_fp8_ws32_prefill_paired_sort_phase_l6_"
            "20260908T205932416058403Z/fleet"
        )
    src = archive / "rank0"
    original_rank0 = json.loads((src / "runner.json").read_text())
    _, selected_ledger = campaign.checkpoint_ledger(6)
    # Explicit synthetic replay capsules reuse rank0 bytes under other physical
    # owners. Checkpoint identities still come from the retained32-owner ledger.
    source_devices = list(old_slots)
    for rank in () if panel else range(1, 8):
        owners = [
            (s, o) for s, o in capsule["owners"].items() if o["launch_rank"] == rank
        ]
        for (_, owner), device in zip(owners, source_devices, strict=True):
            source_owner = capsule["owners"][str(old_slots[device])]
            for field in ("components", "wk_decode", "wk_promote"):
                owner[field] = deepcopy(source_owner[field])
    monkeypatch.setattr(originals, "load_capsule", lambda: deepcopy(capsule))
    template.update(protocol=variant.protocol, profile=variant.admission.PROFILE)
    changed_stable = request.getfixturevalue("paired_stablehlo") if paired else None
    changed_panel = request.getfixturevalue("panel_stablehlo") if panel else None
    if paired:
        for name, program in template["programs"].items():
            if name in assembly.PROGRAMS:
                continue
            stable = (
                changed_stable
                if name == "prefix"
                else (base / f"{name}.stablehlo.mlir").read_text()
            )
            if panel and name == "candidate":
                stable = changed_panel
            program["stablehlo_sha256"] = sha256(stable.encode()).hexdigest()
            program["admission"] = variant.admission.inspect_program(
                name,
                stable,
                (base / f"{name}.optimized_hlo.txt").read_text(),
                program["compiled_memory"],
            )
    tag = f"greenfield_fp8_{variant.kernel}_l6_cpu_fixture"
    records, ledger = [], {}
    programs = {n: (base / f"{n}.optimized_hlo.txt").read_text() for n in phase.COUNTS}
    groups = {g["module_regex"]: g for g in phase.trace_groups(programs).values()}
    for rank in range(8):
        root = tmp_path / f"rank{rank}"
        root.mkdir()
        old = deepcopy(original_rank0)
        if panel:
            old = json.loads((archive / f"rank{rank}/runner.json").read_text())
        elif rank:
            owner_rows = [
                (int(s), o)
                for s, o in capsule["owners"].items()
                if o["launch_rank"] == rank
            ]
            old.update(
                launch_rank=rank,
                jax_process_index=owner_rows[0][1]["jax_process_index"],
                hostname=f"synthetic-phase-worker-{rank}",
                pid=90000 + rank,
                start_ticks=80000 + rank,
                boot_id=f"synthetic-boot-{rank}",
            )
            old["local_device_slots"] = [
                dict(
                    device_id=o["device_id"],
                    device_slot=s,
                    observed_selected_tensor_sha256=selected_ledger[s]["selected"],
                    expected_full_file_sha256_not_verified=selected_ledger[s][
                        "full_sha256"
                    ],
                    selected_payload_bytes=old["payload_bytes_per_chip"],
                )
                for s, o in owner_rows
            ]
        record = deepcopy(template)
        for k in (
            "launch_rank",
            "jax_process_index",
            "local_device_slots",
            "hostname",
            "pid",
            "start_ticks",
            "boot_id",
            "checkpoint_pins",
            "topology_sha256",
            "topology_fleet_sha256",
            "versions",
            "integrity_scope",
            "payload_bytes_per_chip",
        ):
            record[k] = old[k]
        record.update(
            status="SUCCESS",
            layer=6,
            selected_layer_ids=[6],
            rows=128,
            control_rows=32,
            context_capacity=4096,
            key_tile=512,
            iterations=0,
            latency=None,
            admission_only=False,
            diagnostic_only=True,
            numerical_execution_authorized=True,
            state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
            hlo=dict(
                sha256=template["programs"]["candidate"]["optimized_hlo_sha256"],
                contract=dict(passed=True, profile=variant.admission.PROFILE),
            ),
        )
        slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
        for slot in record["local_device_slots"]:
            ledger[slot["device_slot"]] = dict(
                selected=slot["observed_selected_tensor_sha256"],
                full_sha256=slot["expected_full_file_sha256_not_verified"],
            )
        for n in evidence.PROGRAMS:
            for form in ("stablehlo.mlir", "optimized_hlo.txt"):
                if panel and n == "candidate" and form == "stablehlo.mlir":
                    (root / f"{n}.{form}").write_text(changed_panel)
                elif paired and n == "prefix" and form == "stablehlo.mlir":
                    (root / f"{n}.{form}").write_text(changed_stable)
                else:
                    os.link(base / f"{n}.{form}", root / f"{n}.{form}")
        for original_name, output_name in (
            ("competitive", "phase_first"),
            ("wk_decode", "wk_decode"),
            ("wk_promote", "wk_promote"),
            ("wk_boundary", "wk_boundary"),
        ):
            target = root / f"{output_name}.npz"
            if panel:
                retained_name = (
                    "phase_first" if original_name == "competitive" else original_name
                )
                os.link(archive / f"rank{rank}/{retained_name}.npz", target)
            elif rank == 0:
                os.link(src / f"{original_name}.npz", target)
            else:
                with np.load(src / f"{original_name}.npz", allow_pickle=False) as saved:
                    renamed = {}
                    mapping = dict(zip(source_devices, slots, strict=True))
                    for key in saved.files:
                        if key.startswith("input__"):
                            new_key = key
                        elif original_name in ("wk_decode", "wk_promote"):
                            new_key = str(mapping[int(key)])
                        elif original_name == "wk_boundary":
                            kind, device = key.rsplit("_", 1)
                            new_key = f"{kind}_{mapping[int(device)]}"
                        else:
                            owner, field = key.split("__", 1)
                            kind, device = owner.rsplit("_", 1)
                            new_key = f"{kind}_{mapping[int(device)]}__{field}"
                        renamed[new_key] = saved[key]
                    np.savez_compressed(target, **renamed)
            digest = sha256(target.read_bytes()).hexdigest()
            if output_name in ("wk_decode", "wk_promote"):
                record["wk_originals"][output_name] = digest
            elif output_name == "wk_boundary":
                record["wk_boundary_sha256"] = digest
        record["original_binding"] = originals.bind_originals(record, slots, capsule)
        record["original_authentication"].update(
            binding=record["original_binding"],
            first_npz_sha256=sha256(
                (root / "phase_first.npz").read_bytes()
            ).hexdigest(),
        )
        if panel:
            from scripts.greenfield.prefill_panel_originals import bounded_receipt

            record["original_authentication"]["panel_bounded_reference"] = (
                bounded_receipt(root / "phase_first.npz", slots)
            )
        journal = [
            json.loads(s)
            for s in (base / "compile_journal.jsonl").read_text().splitlines()
        ]
        journal[0]["identity"]["launch_rank"] = rank
        journal[0]["identity"].update(
            protocol=variant.protocol, profile=variant.admission.PROFILE
        )
        for entry in journal:
            program = record["programs"].get(entry["graph"])
            if program and entry["stage"] == "raw_written":
                entry["stablehlo_sha256"] = program["stablehlo_sha256"]
            if program and entry["stage"] == "inspected":
                entry["report"] = program["admission"]
        raw = ("\n".join(json.dumps(r) for r in journal) + "\n").encode()
        (root / "compile_journal.jsonl").write_bytes(raw)
        record["compile_journal_sha256"] = sha256(raw).hexdigest()
        index = []
        with (root / "phase_calls.jsonl.gz").open("wb") as stream:
            for entry in phase.read_call_witnesses(
                base / "phase_calls.jsonl.gz", template["phase_call_index"]
            ):
                for rows in (entry["census"]["devices"], entry["post_memory"]):
                    for row, device in zip(rows, slots, strict=True):
                        row.update(
                            device_id=device, process_index=record["jax_process_index"]
                        )
                entry["budget"] = variant.budgeter(
                    entry["census"],
                    entry["compiled_memory"],
                    active_graph=entry["graph"],
                )
                raw = (
                    json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                packed = gzip.compress(raw, mtime=0)
                compact = {
                    k: entry[k]
                    for k in ("phase", "graph", "completed", "completed_call_seconds")
                }
                compact.update(
                    offset=stream.tell(),
                    compressed_bytes=len(packed),
                    raw_sha256=sha256(raw).hexdigest(),
                )
                index.append(compact)
                stream.write(packed)
        record["phase_call_index"] = index
        xs = parse_xplane.build_xplane_classes()["XSpace"]()
        xs.hostnames.append(record["hostname"])
        raw = xs.SerializeToString()
        (root / "phase.xplane.pb").write_bytes(raw)
        record["phase_trace_file"].update(
            bytes=len(raw), sha256=sha256(raw).hexdigest()
        )
        preflight = dict(
            code_hash=record["code_hash"],
            layer=6,
            launch_rank=rank,
            hostname=record["hostname"],
            headers=[dict(device_slot=s) for s in slots.values()],
        )
        raw = json.dumps(preflight).encode()
        (root / "retained_preflight.json").write_bytes(raw)
        record["retained_preflight_sha256"] = sha256(raw).hexdigest()
        (root / "runner.json").write_text(json.dumps(record))
        (root / "worker.log").write_text(
            "CPU fixture: original tensors, synthetic counters/trace events\n"
        )
        records.append(record)
    pin = template["code_hash"]
    monkeypatch.setattr(
        campaign,
        "checkpoint_ledger",
        lambda layer: (records[0]["checkpoint_pins"], ledger),
    )
    blobs = {}

    class Blob:
        generation, crc32c = 23, "fixture-crc"

        def __init__(self, name, path):
            self.name, self.path, self.size = name, path, path.stat().st_size

        def reload(self, *, if_generation_match):
            assert if_generation_match == self.generation

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            return self.path.read_bytes()

    bucket = SimpleNamespace(
        get_blob=lambda n: blobs[n], blob=lambda n, generation: blobs[n]
    )
    monkeypatch.setattr(
        storage, "Client", lambda: SimpleNamespace(bucket=lambda n: bucket)
    )

    def publish(bucket, name, path, digest, *, compressed):
        assert not compressed
        b = blobs[name] = Blob(name, path)
        return dict(
            name=name,
            generation=str(b.generation),
            size=b.size,
            crc32c=b.crc32c,
            original_sha256=digest["sha256"],
        )

    monkeypatch.setattr(publication, "publish_exact", publish)
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path)
    for rank in range(8):
        campaign.publish_rank(tag, rank)

    # Parse real protobuf hostnames; only TPU event contents are fixture data.
    seen = set()

    def aggregate_host(path, regex):
        host = list(parse_xplane.load_xspace(path).hostnames)[0]
        seen.add(path)
        return [
            fake_core(host, i, steps=groups[regex]["expected_calls_per_core"])
            for i in range(8)
        ]

    monkeypatch.setattr(parse_xplane, "aggregate_host", aggregate_host)
    destination = tmp_path / "collected"
    destination.mkdir()
    monkeypatch.setattr(campaign, "run_root", lambda tag: destination)
    result = campaign.collect(tag, pin)
    assert len(seen) == 8 and all("/collected/fleet/" in p for p in seen)
    campaign.validate_record(result, pin)
    assert result["phase_wall"]["scope"] == phase.SCOPE
    for change in ("wall", "owner", "scope"):
        bad = deepcopy(result)
        if change == "wall":
            bad["phase_wall"]["wall"]["shared_prefix_seconds"]["p50"] = -1
        elif change == "owner":
            bad["workers"][0]["local_device_slots"][0]["device_slot"] = 31
        else:
            bad["reference_scope"] = "END_TO_END"
        with pytest.raises(ValueError):
            campaign.validate_record(bad, pin)
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        b.split("\nPY\n", 1)[0]
        for b in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in b
    )
    (destination / "runner.json").write_text(json.dumps(result))
    db = tmp_path / "phase.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accounting",
            str(destination),
            pin,
            str(db),
            str(Path.cwd()),
            "1",
            variant.kernel,
        ],
    )
    exec(
        compile(accounting, "<actual-phase-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "select item_id, correct, score, latency_ms from items"
        ).fetchall() == [
            (
                (
                    "layer6_panel_b128_original_b32_bounded_phase_sum_287calls_v1"
                    if panel
                    else (
                        "layer6_db594_paired_sort_phase_sum_estimate_287calls_v1"
                        if paired
                        else "layer6_db594_b128_four_b32_phase_sum_estimate_287calls_v1"
                    )
                ),
                None,
                None,
                None,
            )
        ]
    assert (
        "NOT full-layer latency"
        in json.loads((destination / "summary.json").read_text())["claim_scope"]
    )
    # These are reproducible temporary test copies, not archived run evidence.
    shutil.rmtree(tmp_path)
