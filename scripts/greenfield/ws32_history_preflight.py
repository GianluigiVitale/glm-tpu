"""Header/original-only history preflight; never numerical execution admission.

Adapt the dense retained preflight and generation-qualified materializers. Only
four bounded original result files are fetched; checkpoint/overlay payloads,
live owners, HLO admission and runtime HBM remain separate unchecked gates.
"""

from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from pathlib import Path
import re
import shutil
import socket
from typing import Any, Mapping

import numpy as np

from glm_tpu.greenfield.checkpoint.ws32_layer_subset import (
    Ws32LayerSubsetMetadata, read_ws32_layer_subset_metadata,
)
from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import (
    verify_ws32_strategy_nd_dense_overlay,
)
from glm_tpu.greenfield.types import ModelGeometry
from glm_tpu.greenfield.validation.ws32_evidence import _atomic_download, _require_blob_identity
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.ws32_dense_frontier_evidence import read_npz
from scripts.greenfield.ws32_dense_frontier_execution import ORACLE_ROOT
from scripts.greenfield.ws32_dense_frontier_preflight import RUN_ROOT
from scripts.greenfield.ws32_dense_frontier_protocol import _read_bound
from scripts.greenfield.ws32_history_compile import require_source

ORACLE_KEYS = tuple(f"{kind}_oracle_{suffix}" for kind in ("token", "dsa")
                    for suffix in ("manifest_sha256", "success_sha256"))
CONTEXT_KEYS = ("hostname", "jax_process_index", "checkpoint_manifest_sha256",
                "checkpoint_success_sha256", "source_inventory_sha256", "mesh_sha256",
                "topology_sha256", "topology_fleet_sha256", "prompt_length",
                "context_capacity", "main_rope_table", *ORACLE_KEYS)


def _plain_path(path: Path) -> None:
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("history retained path must be absolute with no symlink ancestors")


def selected_metadata(repo: Path, slots: tuple[int, ...]) -> tuple[dict, Ws32LayerSubsetMetadata]:
    """Dense preflight schema/source guards, selecting seven layers + embedding."""
    from scripts.greenfield.probe_ws32_prefill_layer import authenticated_inventory

    require_source(repo)
    pins = json.loads((repo / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())
    inventory = authenticated_inventory(Path(pins["source_inventory"]), pins["source_inventory_sha256"])
    checkpoint = Path(pins["checkpoint_root"])
    for name, key in (("manifest.json", "manifest_file_sha256"), ("SUCCESS", "success_file_sha256")):
        if sha256((checkpoint / name).read_bytes()).hexdigest() != pins[key]:
            raise ValueError("history retained checkpoint metadata file differs")
    geometry = ModelGeometry.from_hf_config(json.loads((repo / "configs/glm-5.2-fp8-config.json").read_text()))
    subset = read_ws32_layer_subset_metadata(
        checkpoint, layer_ids=protocol.LAYERS, include_embedding=True, local_slots=slots,
        max_payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
        expected_manifest_sha256=pins["expected_manifest_sha256"],
        expected_success_sha256=pins["expected_success_sha256"],
        expected_mesh_hash=pins["expected_mesh_sha256"],
        expected_topology_hash=pins["expected_topology_sha256"], inventory=inventory, geometry=geometry,
    )
    if (len(subset.tensor_indices) != protocol.SELECTED_LEAVES
            or subset.payload_bytes_per_chip != protocol.PAYLOAD_BYTES):
        raise ValueError("history selected inventory or exact byte count differs")
    return pins, subset


def _owners(runner: Mapping[str, Any], rank: int) -> dict[int, dict]:
    rows = runner["local_device_slots"]
    if (runner["launch_process_id"] != rank or type(runner["launch_process_id"]) is not int
            or type(runner["jax_process_index"]) is not int or not 0 <= runner["jax_process_index"] < 8
            or len(rows) != 4):
        raise ValueError("history original launch/process/owner identity differs")
    for row in rows:
        if (any(type(row[k]) is not int for k in
                ("device_id", "device_slot", "expert_coordinate", "feature_coordinate"))
                or not 0 <= row["device_id"] < 32 or not 0 <= row["device_slot"] < 32
                or (row["expert_coordinate"], row["feature_coordinate"]) != divmod(row["device_slot"], 4)):
            raise ValueError("history original owner coordinates differ")
    owners = {v["device_slot"]: v for v in rows}
    if len(owners) != 4 or len({v["device_id"] for v in rows}) != 4:
        raise ValueError("history original duplicate owners")
    return owners


def load_originals(root: Path, *, repo: Path, rank: int) -> tuple[dict, dict, dict]:
    """Authenticate both complete originals, returning only their first-event rows."""
    _plain_path(root)
    pins = protocol.original_pins(repo, rank)
    if sum(v["size"] for branch in pins.values() for v in branch.values()) > protocol.ORIGINALS_LIMIT:
        raise ValueError("history combined original budget exceeded")
    runners, observations = {}, {}
    for branch in protocol.BRANCHES:
        runner = json.loads(_read_bound(root / f"{branch}.json", pins[branch]["json"]))
        _owners(runner, rank)
        path = root / f"{branch}.npz"
        _read_bound(path, pins[branch]["npz"])
        report = runner["numerical_tensors"]
        if (report["filename"] != f"runner.rank{rank}.npz"
                or report["sha256"] != pins[branch]["npz"]["sha256"]
                or report["byte_count"] != pins[branch]["npz"]["size"]):
            raise ValueError("history runner/NPZ binding differs")
        declared = report["arrays"]
        raw_bytes = sum(prod(v["shape"]) * np.dtype(v["dtype"]).itemsize for v in declared.values())
        arrays = read_npz(path, dict(bytes=report["byte_count"], npz_sha256=report["sha256"],
                                   raw_array_bytes=raw_bytes), limit=protocol.MAX_ORIGINAL_BYTES)
        if set(arrays) != set(declared) or any(
            list(value.shape) != declared[name]["shape"] or str(value.dtype) != declared[name]["dtype"]
            or sha256(value.tobytes()).hexdigest() != declared[name]["sha256"]
            for name, value in arrays.items()
        ):
            raise ValueError("history original array declaration differs")
        observations[branch] = protocol.reproduction_rows(arrays)
        if observations[branch]["positions"].shape != (4, 1, 2048):
            raise ValueError("history original first-event width differs")
        runners[branch] = runner
    candidate, control = (runners[b] for b in protocol.BRANCHES)
    if ({k: candidate[k] for k in CONTEXT_KEYS} != {k: control[k] for k in CONTEXT_KEYS}
            or _owners(candidate, rank) != _owners(control, rank)):
        raise ValueError("history original branches disagree on context/owners")
    return runners, observations, dict(
        original_tags=protocol.ORIGINAL_TAGS, original_pins=pins,
        receipt_sha256={b: protocol.RECEIPTS[b][1] for b in protocol.BRANCHES},
        bytes=sum(v["size"] for branch in pins.values() for v in branch.values()),
        reproduction_row_sha256={b: {k: sha256(v.tobytes()).hexdigest() for k, v in rows.items()}
                                 for b, rows in observations.items()},
    )


def materialize_originals(root: Path, *, repo: Path, rank: int, client: Any) -> tuple[dict, dict, dict]:
    """Four exact-generation same-region result downloads; never overwrite evidence."""
    _plain_path(root)
    pins = protocol.original_pins(repo, rank)
    if sum(v["size"] for branch in pins.values() for v in branch.values()) > protocol.ORIGINALS_LIMIT:
        raise ValueError("history combined original budget exceeded")
    if shutil.disk_usage(root if root.exists() else root.parent).free < protocol.ORIGINALS_LIMIT + protocol.RESERVE:
        raise ValueError("history original materialization needs bounded disk headroom")
    bucket = client.bucket(protocol.BUCKET)
    bucket.reload()
    if str(bucket.location).upper() != "US-CENTRAL2":
        raise ValueError("history original bucket must be US-CENTRAL2")
    for branch in protocol.BRANCHES:
        for form in protocol.ORIGINAL_FORMS:
            pin, path = pins[branch][form], root / f"{branch}.{form}"
            partial = path.with_name(path.name + ".partial")
            if partial.exists() or partial.is_symlink():
                raise FileExistsError(partial)
            if path.exists() or path.is_symlink():
                _read_bound(path, pin)
                continue
            generation = int(pin["generation"])
            blob = bucket.blob(pin["name"], generation=generation)
            blob.reload(if_generation_match=generation)
            if (int(blob.generation) != generation or blob.size != pin["size"] or blob.crc32c != pin["crc32c"]):
                raise ValueError("history remote original generation/size/CRC differs")
            _atomic_download(blob, path)
            _require_blob_identity(blob, path, pin["sha256"])
    return load_originals(root, repo=repo, rank=rank)


def host_inputs(runners: Mapping[str, Mapping], config: Any) -> tuple[np.ndarray, np.ndarray]:
    """Derive missing historical prompt SHA from the bound original oracle IDs."""
    from glm_tpu.greenfield.validation.ws32_short_context import load_ws32_short_context_oracle
    from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
    from scripts.greenfield.ws32_prefill_frontier_state import require_config

    require_config(config)
    if set(runners) != set(protocol.BRANCHES):
        raise ValueError("history requires both original branches")
    prior = runners["candidate"]
    if any({k: r[k] for k in ORACLE_KEYS} != {k: prior[k] for k in ORACLE_KEYS} for r in runners.values()):
        raise ValueError("history original oracle identities differ")
    oracle = load_ws32_short_context_oracle(
        ORACLE_ROOT / "short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle",
        ORACLE_ROOT / "short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle",
        **{f"expected_{kind}_{suffix}": prior[f"{kind}_oracle_{suffix}"]
           for kind in ("token", "dsa") for suffix in ("manifest_sha256", "success_sha256")},
    )
    tokens, rope = oracle.prompt_token_ids, np.asarray(build_ws32_main_rope_table(config))
    if (tokens.dtype != np.int32 or tokens.shape != (protocol.PROMPT_LENGTH,)
            or sha256(tokens.tobytes()).hexdigest() != protocol.PROMPT_SHA
            or int(oracle.generated_token_ids[0]) != protocol.WITNESS["token"]
            or int(oracle.decode_positions[0]) != protocol.WITNESS["position"]
            or rope.shape != (protocol.CAPACITY, 64) or str(rope.dtype) != "bfloat16"):
        raise ValueError("history original oracle prompt/witness/RoPE geometry differs")
    expected_rope = dict(rows=protocol.CAPACITY, rotary_dim=64, theta=8000000.0,
                         bytes_per_device=rope.nbytes, sha256=sha256(rope.tobytes()).hexdigest())
    for runner in runners.values():
        # Original runners do not carry this field. The mandatory original oracle
        # above supplies the proof; a present-but-conflicting field is rejected.
        if (runner["prompt_length"] != protocol.PROMPT_LENGTH or runner["context_capacity"] != protocol.CAPACITY
                or ("prompt_ids_sha256" in runner and runner["prompt_ids_sha256"] != protocol.PROMPT_SHA)
                or runner["main_rope_table"] != expected_rope):
            raise ValueError("history original prompt/RoPE binding differs")
    return tokens, rope


def bind_metadata(repo: Path, runners: Mapping[str, Mapping], pins: dict, subset: Ws32LayerSubsetMetadata) -> Any:
    """Bind selected file ledgers and all original local overlay owner records."""
    env = json.loads((repo / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    prefix = "GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_"
    overlay = verify_ws32_strategy_nd_dense_overlay(Path(env[prefix + "ROOT"]),
        expected_manifest_sha256=env[prefix + "MANIFEST_SHA"],
        expected_manifest_file_sha256=env[prefix + "MANIFEST_FILE_SHA"],
        expected_success_file_sha256=env[prefix + "SUCCESS_FILE_SHA"])
    byte_counts = {sum(v["byte_count"] for i in range(3)
                      for v in overlay.records[(i, e, f)]["tensors"].values())
                   for e in range(8) for f in range(4)}
    if byte_counts != {protocol.OVERLAY_BYTES} or any(len(r["tensors"]) != 4 for r in overlay.records.values()):
        raise ValueError("history overlay exact inventory/bytes differ")
    metadata = subset.metadata
    for runner in runners.values():
        expected = dict(checkpoint_manifest_sha256=metadata.manifest["manifest_sha256"],
                        source_inventory_sha256=metadata.manifest["source"]["inventory_sha256"],
                        checkpoint_success_sha256=pins["expected_success_sha256"],
                        mesh_sha256=pins["expected_mesh_sha256"], topology_sha256=pins["expected_topology_sha256"])
        if any(runner[k] != v for k, v in expected.items()):
            raise ValueError("history original checkpoint/topology differs")
        original = runner["strategy_nd_dense_overlay"]
        if (original["manifest_sha256"] != overlay.manifest["manifest_sha256"]
                or original["manifest_file_sha256"] != overlay.manifest_file_sha256
                or original["success_file_sha256"] != overlay.success_file_sha256):
            raise ValueError("history original overlay identity differs")
        expected_rows = []
        for owner in runner["local_device_slots"]:
            slot = owner["device_slot"]
            if owner["file_sha256"] != metadata.records_by_slot[slot]["sha256"]:
                raise ValueError("history original selected owner-file ledger differs")
            e, f = divmod(slot, 4)
            expected_rows.extend(dict(device_id=owner["device_id"], expert_coordinate=e,
                                     feature_coordinate=f, layer_id=i,
                                     file_sha256=overlay.records[(i, e, f)]["sha256"]) for i in range(3))
        key = lambda r: (r["device_id"], r["layer_id"])
        if sorted(original["local_records"], key=key) != sorted(expected_rows, key=key):
            raise ValueError("history original overlay owner ledger differs")
    return overlay


def retained_preflight(*, tag: str, rank: int, pin: str, root: Path, repo: Path, client: Any) -> None:
    """Dense-compatible entry API, intentionally not yet wired to any launcher."""
    _plain_path(root)
    if (type(rank) is not int or not 0 <= rank < 8 or not protocol.is_tag(tag)
            or not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{40}", pin) is None
            or root != RUN_ROOT / tag / f"rank{rank}"):
        raise ValueError("history retained tag/rank/path differs")
    output = root / "retained_preflight.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require_source(repo)
    root.mkdir(parents=True, exist_ok=True)
    runners, _, identity = materialize_originals(root / "retained_reference", repo=repo, rank=rank, client=client)
    prior = runners["candidate"]
    if prior["hostname"] != socket.gethostname():
        raise ValueError("history retained source belongs to another host")
    slots = tuple(sorted(_owners(prior, rank)))
    pins, subset = selected_metadata(repo, slots)
    overlay = bind_metadata(repo, runners, pins, subset)
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
    config = Ws32DecoderConfig(ModelGeometry.from_dict(subset.metadata.manifest["geometry"]),
                               protocol.CAPACITY, host_main_rope_table=True)
    host_inputs(runners, config)
    _atomic_json(output, dict(
        protocol=protocol.PROTOCOL, tag=tag, code_hash=pin, launch_rank=rank, hostname=socket.gethostname(),
        selected_layer_ids=list(protocol.LAYERS), include_embedding=True, context_capacity=protocol.CAPACITY,
        host_main_rope_table=True, selected_leaf_count=protocol.SELECTED_LEAVES,
        payload_bytes_per_chip=protocol.PAYLOAD_BYTES, checkpoint_pins=pins,
        local_device_slots=prior["local_device_slots"],
        headers=[{key: subset.metadata.records_by_slot[slot][key]
                  for key in ("device_slot", "filename", "file_bytes", "header_sha256")} for slot in slots],
        overlay_tensor_count=protocol.OVERLAY_TENSORS, overlay_bytes_per_chip=protocol.OVERLAY_BYTES,
        overlay_manifest_sha256=overlay.manifest["manifest_sha256"],
        strategy_nd_dense_overlay=prior["strategy_nd_dense_overlay"], **identity,
        prompt_ids_sha256=protocol.PROMPT_SHA, prompt_hash_source="AUTHENTICATED_ORIGINAL_TOKEN_ORACLE",
        original_oracle_pins={key: prior[key] for key in ORACLE_KEYS}, main_rope_table=prior["main_rope_table"],
        scope="HEADERS_ORIGINALS_AND_HOST_INPUTS_ONLY_NOT_SELECTED_PAYLOAD_OR_LIVE_TOPOLOGY",
        numerical_execution_available=False, numerical_admission=False,
        numerical_promotion=False, performance_claim=False,
    ))
    print(f"PREFILL_RETAINED_OK {socket.gethostname()}", flush=True)
