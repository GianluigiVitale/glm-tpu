"""Actual bounded download/inflate/CRC path; fixture cloud and model replay."""

import base64
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import google_crc32c
import pytest

from scripts.greenfield import ws32_prefill_frontier_collect as collector

TAG = "greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_first128_20260909T120000000000000Z"
GENERATION = 9007199254740993


def cloud():
    events, objects = [], {}
    class Blob:
        def __init__(self, name, raw):
            self.name, self.raw = f"results/{TAG}/"+name, raw
            self.generation, self.size = GENERATION, len(raw)
            self.crc32c = base64.b64encode(google_crc32c.Checksum(raw).digest()).decode()
        def download_to_filename(self, filename, *, if_generation_match):
            assert if_generation_match == GENERATION
            events.append(self.name)
            Path(filename).write_bytes(self.raw)
    reports = {g: {"stablehlo_sha256": sha256((g+" stablehlo.mlir").encode()).hexdigest(),
                   "optimized_hlo_sha256": sha256((g+" optimized_hlo.txt").encode()).hexdigest()}
               for g in collector.GRAPHS}
    for name in collector.expected_names(TAG):
        if name.startswith("hlo/"):
            graph, form = name[4:-3].split(".", 1)
            raw = gzip.compress((graph+" "+form).encode())
        elif name.startswith("host_records/"):
            raw = json.dumps(dict(graphs=reports, compiled_memory_analysis={}, rank=int(name.split("rank")[1][0]))).encode()
        else:
            raw = b"fixture original"
        objects[name] = Blob(name, raw)
    def bucket(name):
        assert name == collector.BUCKET
        return NS(name=name)
    def listing(b, *, prefix):
        assert prefix == f"results/{TAG}/"
        return list(objects.values())
    return NS(bucket=bucket, list_blobs=listing), objects, events


@pytest.mark.parametrize("mutation", [None, "missing", "extra_graph", "extra_partial", "rank_budget", "bad_crc", "wrong_size"])
def test_actual_generation_download_and_inflation(tmp_path, monkeypatch, mutation):
    client, objects, events = cloud()
    name = f"diagnostic_local/{TAG}/first_window.rank0/runner.json"
    if mutation == "missing":
        objects.pop(name)
    if mutation in ("extra_graph", "extra_partial"):
        extra = "hlo/decode.stablehlo.mlir.gz" if mutation == "extra_graph" else f"diagnostic_local/{TAG}/first_window.rank0/wide_final.npz.pending"
        objects[extra] = deepcopy(objects[name])
        objects[extra].name = f"results/{TAG}/"+extra
    if mutation == "rank_budget":
        for key, blob in objects.items():
            if "/first_window.rank0/" in key:
                blob.size = collector.LIMIT // 2
    if mutation == "bad_crc":
        objects[name].crc32c = "AAAAAA=="
    if mutation == "wrong_size":
        objects[name].size += 1
    target = tmp_path / "collected"
    if mutation:
        with pytest.raises((ValueError, SystemExit)):
            collector.materialize(TAG, target, client=client)
        if mutation in ("missing", "extra_graph", "extra_partial", "rank_budget"):
            assert not events and not target.exists()
    else:
        sources = collector.materialize(TAG, target, client=client)
        assert len(events) == len(collector.expected_names(TAG)) == 150
        assert all(v["generation"] == str(GENERATION) for v in sources["objects"])
        assert (target / "hlo/prefill_chunk.stablehlo.mlir").read_text() == "prefill_chunk stablehlo.mlir"
        with pytest.raises(ValueError, match="replace"):
            collector.materialize(TAG, target, client=client)


def test_downloads_join_actual_outer_aggregate(tmp_path, monkeypatch):
    client, objects, events = cloud()
    target = tmp_path / "collected"
    collector.materialize(TAG, target, client=client)
    identities = [dict(jax_process_index=(r+3)%8, hostname=f"host{r}") for r in range(8)]
    slots = [{r+i*8: r+i*8 for i in range(4)} for r in range(8)]
    calls = []
    monkeypatch.setattr(collector.evidence, "replay_graphs", lambda *a, **k: calls.append("graphs"))
    def envelope(root, record, journal, *, local_slots, expected_identity):
        rank = record["rank"]
        assert expected_identity == {**identities[rank], "code_hash": "a"*40, "launch_process_id": rank,
            "artifact_kind": "greenfield_ws32_first_window_diagnostic", "numerical_promotion": False, "performance_claim": False}
        assert local_slots == slots[rank]
        assert journal.name == f"numerical_journal.rank{rank}.jsonl"
        calls.append(rank)
        return {"fixture": True}
    monkeypatch.setattr(collector.evidence, "replay_envelope", envelope)
    monkeypatch.setattr(collector.evidence, "replay_host", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(collector.evidence, "replay_replicas", lambda hosts: {"hosts": len(hosts)})
    def run():
        return collector.aggregate(target, tag=TAG, pin="a"*40, identities=identities, owner_maps=slots, args=NS())
    result = run()
    assert calls == ["graphs", *range(8)] and result["hosts"] == 8
    assert result["cleanup_claim"] is False and result["complete_8k_claim"] is False
    file = target / f"diagnostic_local/{TAG}/first_window.rank0/runner.json"
    file.write_bytes(b"altered original")
    with pytest.raises(ValueError, match="generation bytes"):
        run()


def test_saved_production_three_graph_replay():
    """Original TPU texts; only expected profile label changes, no new compile."""
    run = Path("/home/gianl/glm-run/greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_20260909T093335570726655Z")
    record = json.loads((run / "runner.rank0.json").read_text())
    record["graphs"] = {g: record["graphs"][g] for g in collector.GRAPHS}
    record["compiled_memory_analysis"] = {g: record["compiled_memory_analysis"][g] for g in collector.GRAPHS}
    profile = collector.admission.FROZEN_FIRST_WINDOW_PROFILE
    for graph, report in record["graphs"].items():
        report["source_location_identity"]["profile"] = profile
        if graph == "prefill_chunk":
            report["profile_name"] = profile
    pins = collector.admission.short_acquisition(collector.REPO, profile=profile)
    args = NS(batched_prefill_profile=profile,
        **{f"expected_{g}_{form}": v for g, row in pins["graphs"].items() for form, v in row.items()})
    texts = {g: tuple((run / "hlo" / f"{g}.{form}").read_text() for form in collector.FORMS) for g in collector.GRAPHS}
    collector.evidence.replay_graphs(record, texts, args=args)
