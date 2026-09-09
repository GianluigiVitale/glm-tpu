"""Collect bounded original first128 evidence; no launcher, seal or promotion.

The existing protected wrapper still owns leases, archives and terminal census.
This collector downloads only a fixed diagnostic inventory from the same-region
bucket and independently replays it. It never creates SUCCESS or a performance DB row.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from glm_tpu.greenfield.validation.ws32_evidence import (
    _atomic_download, _inflate_gzip, _require_blob_identity, _sha256_file, _crc32c_file,
)
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_evidence import same_json
from scripts.greenfield.ws32_prefill_frontier_publish import FILES, LIMIT
from scripts.greenfield import ws32_prefill_frontier_evidence as evidence

REPO = Path(__file__).resolve().parents[2]
BUCKET = "driftbench-dsv4-uc"
GRAPHS = ("exact_materialize", "exact_promote", "prefill_chunk")
FORMS = ("stablehlo.mlir", "optimized_hlo.txt")
HLO_LIMIT = 512 << 20  # All six inflated texts together; no per-rank copies.
EXTRA_LIMIT = 16 << 20  # Primary runner + journal per rank, outside128MiB originals.


def expected_names(tag: str) -> set[str]:
    if not re.fullmatch(r"greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_first128_[0-9]{8}T[0-9]+Z", tag):
        raise ValueError("first-window collector requires its exact tag family")
    names = {f"hlo/{g}.{f}.gz" for g in GRAPHS for f in FORMS}
    for rank in range(8):
        names.add(f"host_records/runner.rank{rank}.json")
        names.add(f"diagnostic_local/{tag}/numerical_journal.rank{rank}.jsonl")
        names.update(f"diagnostic_local/{tag}/first_window.rank{rank}/{name}" for name in FILES)
    return names


def materialize(tag: str, destination: Path, *, client: Any) -> dict[str, Any]:
    """Resolve and budget exact generations BEFORE any original payload download."""
    names = expected_names(tag)
    prefix = f"results/{tag}/"
    bucket = client.bucket(BUCKET)
    all_blobs = {b.name[len(prefix):]: b for b in client.list_blobs(bucket, prefix=prefix)}
    if not names <= set(all_blobs):
        raise ValueError("first-window original inventory incomplete")
    watched = {n for n in all_blobs if n.startswith("hlo/") or
               re.match(re.escape(f"diagnostic_local/{tag}/") + r"first_window\.rank[0-7]/", n)}
    if watched != {n for n in names if n.startswith("hlo/") or "/first_window.rank" in n}:
        raise ValueError("first-window unexpected graph or capsule original")
    blobs = {n: all_blobs[n] for n in names}
    if any(not b.generation or not str(b.generation).isdecimal() or int(b.generation) <= 0
           or b.size is None or not 0 < int(b.size) <= LIMIT or not b.crc32c for b in blobs.values()):
        raise ValueError("first-window original metadata invalid/oversized")
    for rank in range(8):
        if sum(int(b.size) for n, b in blobs.items() if f"/first_window.rank{rank}/" in n) > LIMIT:
            raise ValueError("first-window capsule rank budget exceeded")
        extras = (f"host_records/runner.rank{rank}.json", f"diagnostic_local/{tag}/numerical_journal.rank{rank}.jsonl")
        if sum(int(blobs[n].size) for n in extras) > EXTRA_LIMIT:
            raise ValueError("first-window runner/journal rank budget exceeded")
    compressed = sum(int(b.size) for n, b in blobs.items() if n.startswith("hlo/"))
    if compressed > HLO_LIMIT:
        raise ValueError("first-window compressed HLO budget exceeded")
    total = sum(int(b.size) for b in blobs.values())
    if destination.exists() or destination.is_symlink():
        raise ValueError("refusing to replace first-window collected evidence")
    if shutil.disk_usage(destination.parent).free < total + HLO_LIMIT + (1 << 30):
        raise ValueError("first-window collection requires bounded local headroom")
    destination.mkdir()
    ledger = []
    for name in sorted(names):
        blob, path = blobs[name], destination / name
        _atomic_download(blob, path)
        digest = _sha256_file(path)
        _require_blob_identity(blob, path, digest)
        ledger.append(dict(name=prefix+name, generation=str(blob.generation),
                           size=int(blob.size), crc32c=blob.crc32c, sha256=digest))
    # Write before inflation/replay, preserving sources on subsequent refusal.
    sources = dict(bucket=BUCKET, tag=tag, objects=ledger, downloaded_bytes=total)
    _atomic_json(destination / "sources.json", sources)
    remaining = HLO_LIMIT
    for graph in GRAPHS:
        for form in FORMS:
            path = destination / "hlo" / f"{graph}.{form}"
            _inflate_gzip(path.with_name(path.name+".gz"), path, maximum_bytes=remaining)
            remaining -= path.stat().st_size
    return sources


def aggregate(
    destination: Path, *, tag: str, pin: str, identities: list[dict[str, Any]],
    owner_maps: list[dict[int, int]], args: Any,
) -> dict[str, Any]:
    """Replay authenticated files with independently derived workload/owner pins.

    Caller obtains identities from verified checkpoint/oracle/topology metadata,
    not from the new worker records. No inference of numerical or fleet cleanup
    success from a terminal diagnostic envelope is permitted.
    """
    expected_names(tag)
    if len(identities) != 8 or len(owner_maps) != 8 or not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError("first-window collector needs eight independent identities")
    sources = json.loads((destination / "sources.json").read_text())
    prefix = f"results/{tag}/"
    if (sources["bucket"] != BUCKET or sources["tag"] != tag
            or len(sources["objects"]) != len(expected_names(tag))
            or {v["name"] for v in sources["objects"]} != {prefix+n for n in expected_names(tag)}):
        raise ValueError("first-window source ledger inventory differs")
    for source in sources["objects"]:
        path = destination / source["name"][len(prefix):]
        if (type(source["generation"]) is not str or not source["generation"].isdecimal()
                or int(source["generation"]) <= 0 or path.is_symlink()
                or path.stat().st_size != source["size"]
                or _sha256_file(path) != source["sha256"] or _crc32c_file(path) != source["crc32c"]):
            raise ValueError("first-window downloaded generation bytes differ")
    texts = {g: tuple((destination / "hlo" / f"{g}.{f}").read_text() for f in FORMS) for g in GRAPHS}
    records = [json.loads((destination / "host_records" / f"runner.rank{r}.json").read_text()) for r in range(8)]
    for graph, (stable, optimized) in texts.items():
        for text, key in ((stable, "stablehlo_sha256"), (optimized, "optimized_hlo_sha256")):
            if sha256(text.encode()).hexdigest() != records[0]["graphs"][graph][key]:
                raise ValueError("first-window inflated HLO differs from raw runner hash")
    evidence.replay_graphs(records[0], texts, args=args)
    hosts, envelopes = [], []
    for rank, (record, identity, slots) in enumerate(zip(records, identities, owner_maps, strict=True)):
        same_json(record["graphs"], records[0]["graphs"], "fleet actual compiler reports")
        same_json(record["compiled_memory_analysis"], records[0]["compiled_memory_analysis"], "fleet compiled memory")
        expected = {**identity, "code_hash": pin, "launch_process_id": rank,
                    "artifact_kind": "greenfield_ws32_first_window_diagnostic",
                    "numerical_promotion": False, "performance_claim": False}
        root = destination / "diagnostic_local" / tag / f"first_window.rank{rank}"
        journal = root.parent / f"numerical_journal.rank{rank}.jsonl"
        envelopes.append(evidence.replay_envelope(root, record, journal, local_slots=slots, expected_identity=expected))
        hosts.append(evidence.replay_host(root, record, local_slots=slots, expected_identity=expected))
    result = evidence.replay_replicas(hosts)
    return dict(**result, tag=tag, code_hash=pin, envelopes=envelopes,
                source_ledger_sha256=_sha256_file(destination / "sources.json"),
                cleanup_claim=False, complete_8k_claim=False)


def authenticated_inputs(reference: Path) -> tuple[list[dict[str, Any]], list[dict[int, int]], Any]:
    """Reuse pinned failed-run workload metadata, NOT its numerical verdict.

    The reference JSON is byte-bound by the committed refusal receipt. Current
    checkpoint metadata and complete token/DSA oracle are independently loaded
    through their existing validators. No checkpoint payload is read or copied.
    """
    from scripts.greenfield.ws32_prefill_budget_campaign import topology_bindings
    from scripts.greenfield.ws32_rolled_prefill_compile import read_metadata
    from glm_tpu.greenfield.validation.ws32_short_context import load_ws32_short_context_oracle

    pins = admission.short_acquisition(REPO, profile=admission.FROZEN_FIRST_WINDOW_PROFILE)
    receipt = json.loads((REPO / admission.FROZEN_FAILURE_RECEIPT).read_text())
    source = receipt["hosts"][0]["originals"]["json"]
    raw = reference.read_bytes()
    if len(raw) != source["size"] or sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("first-window prior workload original differs from fixed receipt")
    prior = json.loads(raw)
    physical, captures = topology_bindings()
    metadata = read_metadata(REPO)
    if (metadata.manifest["manifest_sha256"] != prior["checkpoint_manifest_sha256"]
            or metadata.manifest["source"]["inventory_sha256"] != prior["source_inventory_sha256"]):
        raise ValueError("first-window current checkpoint differs from frozen workload")
    root = Path("/home/gianl/gcs-models/oracles/greenfield/glm52")
    oracle = load_ws32_short_context_oracle(
        root / "short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle",
        root / "short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle",
        **{f"expected_{kind}_{suffix}": prior[f"{kind}_oracle_{suffix}"]
           for kind in ("token", "dsa") for suffix in ("manifest_sha256", "success_sha256")},
    )
    if oracle.prompt_token_ids.shape != (8155,):
        raise ValueError("first-window original prompt length differs")
    common = {key: prior[key] for key in (
        "checkpoint_manifest_sha256", "checkpoint_success_sha256", "source_inventory_sha256",
        "token_oracle_manifest_sha256", "token_oracle_success_sha256", "main_rope_table",
        "topology_sha256", "topology_fleet_sha256", "mesh_sha256", "evidence_layout")}
    common["prompt_ids_sha256"] = sha256(oracle.prompt_token_ids.tobytes()).hexdigest()
    identities, owner_maps = [], []
    for rank, capture in enumerate(captures):
        slots = {int(d): slot for slot, d in enumerate(physical.flattened_device_ids)
                 if d in capture["local_device_ids"]}
        owners = [dict(device_id=d, device_slot=s, expert_coordinate=s//4, feature_coordinate=s%4,
                       file_sha256=metadata.records_by_slot[s]["sha256"]) for d, s in slots.items()]
        identities.append(dict(**common, hostname=capture["hostname"], jax_process_index=capture["jax_process_index"],
                               checkpoint_verified_device_slots=sorted(slots.values()), local_device_slots=owners))
        owner_maps.append(slots)
    args = SimpleNamespace(batched_prefill_profile=admission.FROZEN_FIRST_WINDOW_PROFILE,
        **{f"expected_{g}_{form}": value for g, values in pins["graphs"].items() for form, value in values.items()})
    return identities, owner_maps, args


def main() -> None:
    from google.cloud import storage
    from scripts.greenfield.seal_short_decoder_ws32 import _require_clean_worktree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--reference-runner", type=Path, required=True)
    args = parser.parse_args()
    expected_names(args.tag)
    _require_clean_worktree(REPO)
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise ValueError("first-window collector requires a clean worktree")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if head != args.code_hash:
        raise ValueError("first-window collector requires the original run pin")
    subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", head,
                    "origin/rewrite/topology-first-decode"], check=True)
    identities, slots, graph_args = authenticated_inputs(args.reference_runner)
    destination = Path("/home/gianl/glm-run") / args.tag / "first_window_collected"
    materialize(args.tag, destination, client=storage.Client())
    result = aggregate(destination, tag=args.tag, pin=args.code_hash, identities=identities,
                       owner_maps=slots, args=graph_args)
    _atomic_json(destination / "diagnostic.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
