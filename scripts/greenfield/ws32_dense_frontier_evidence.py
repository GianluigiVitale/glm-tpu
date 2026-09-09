"""Independent local-file replay for the nine-call dense01 diagnostic.

No launcher, download, arithmetic change or promotion. The parent still binds
generation-qualified publication, fleet/process/source identity and cleanup.
An exact DB604 reproduction is attribution eligibility, not model correctness.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping
import zipfile

import ml_dtypes
import numpy as np

from scripts.greenfield import ws32_dense_frontier_admission as admission
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield import ws32_dense_frontier_worker as worker
from scripts.greenfield.ws32_dense_frontier_execution import (
    DenseJournal,
    WK_ORIGINALS_LIMIT,
)
from scripts.greenfield.ws32_dense_frontier_capture import cache_bits
from scripts.greenfield.ws32_dense_frontier_witness import compare_owner
from scripts.greenfield.prefill_window_evidence import (
    same_json,
    validate_call_sequence,
    validate_graph_journal,
)

CALLS = tuple(
    (f"layer{layer}/{name}", name) for layer in (0, 1) for name in worker.PROGRAMS[:2]
) + (("dense/wide", "dense01"), *((f"dense/narrow{i}", "dense01") for i in range(4)))
CAPSULES = (
    ("wide_final", 128, True),
    ("narrow_32", 32, False),
    ("narrow_64", 32, False),
    ("narrow_96", 32, False),
    ("narrow_128", 32, True),
)


def digest(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def comparison_by_owner(value: Mapping[str, Any]) -> dict:
    """Canonicalize only owner-record order, never numerical values or DSA order.

    Runtime addressable-device order and original checkpoint slot order differ.
    Every complete branch/slot record still compares exactly after this join.
    """
    rows = value.get("owners")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or row.get("branch") not in ("wide_final", "narrow_128")
        or type(row.get("slot")) is not int
        or not 0 <= row["slot"] < 32
        for row in rows
    ):
        raise ValueError("dense comparison owner identity invalid")
    keys = [(r["branch"], r["slot"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("dense duplicate comparison owner")
    return {**value, "owners": sorted(rows, key=lambda r: (r["branch"], r["slot"]))}


def read_npz(
    path: Path, report: Mapping[str, Any], *, limit: int, size_key: str = "bytes"
) -> dict[str, np.ndarray]:
    """Bound compressed AND expanded originals before materializing arrays."""
    size = report.get(size_key)
    if (
        type(size) is not int
        or not 0 < size <= limit
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != size
    ):
        raise ValueError("dense capsule path/size differs")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != report["npz_sha256"]:
        raise ValueError("dense capsule SHA differs")
    with zipfile.ZipFile(BytesIO(raw)) as z:
        infos = z.infolist()
        names = [v.filename for v in infos]
        if (
            len(set(names)) != len(names)
            or not names
            or len(names) > 128
            or sum(v.file_size for v in infos) > limit
            or any("/" in n or not n.endswith(".npy") for n in names)
        ):
            raise ValueError("dense expanded capsule inventory/budget differs")
        declared_bytes = 0
        for member in infos:
            with z.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("dense unsupported NPY header version")
                if dtype not in tuple(
                    map(np.dtype, (np.uint8, np.uint16, np.int32, np.float32, np.bool_))
                ) or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError("dense NPY dtype/shape differs")
                payload = math.prod(shape) * dtype.itemsize
                declared_bytes += payload
                if (
                    declared_bytes > limit
                    or stream.tell() + payload != member.file_size
                ):
                    raise ValueError("dense NPY declared payload exceeds file/budget")
    with np.load(BytesIO(raw), allow_pickle=False) as saved:
        arrays = dict(saved)
    if sum(v.nbytes for v in arrays.values()) != report["raw_array_bytes"]:
        raise ValueError("dense capsule raw bytes differ")
    return arrays


def checkpoint_bindings(repo: Path) -> dict[int, dict]:
    """Expected selected/WK hashes from authenticated existing metadata only."""
    from scripts.greenfield.ws32_rolled_prefill_compile import read_metadata
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DecoderConfig,
        ws32_decoder_weight_names,
    )
    from glm_tpu.greenfield.types import ModelGeometry
    from glm_tpu.greenfield.validation import ws32_prefill_admission
    import jax

    ws32_prefill_admission.require_acquired_model_source(
        repo, profile=ws32_prefill_admission.ROLLED_SHORT_PROFILE
    )
    metadata = read_metadata(repo)
    config = Ws32DecoderConfig(
        ModelGeometry.from_dict(metadata.manifest["geometry"]),
        8192,
        host_main_rope_table=True,
    )
    names = ws32_decoder_weight_names(config)
    selected = set(jax.tree.leaves((names.embedding_local, names.layers[:2])))
    schema = metadata.plans[0].tensors
    indices = [i for i, t in enumerate(schema) if t.name in selected]
    by_name = {t.name: i for i, t in enumerate(schema)}
    if (
        len(indices) != 55
        or sum(schema[i].byte_count for i in indices) != protocol.PAYLOAD_BYTES
    ):
        raise ValueError("dense checkpoint selected inventory differs")
    return {
        slot: dict(
            full_file_sha256=row["sha256"],
            selected={schema[i].name: row["tensor_sha256"][i] for i in indices},
            wk={
                (layer, role): row["tensor_sha256"][
                    by_name[getattr(names.layers[layer].dsa, field)]
                ]
                for layer in (0, 1)
                for role, field in (
                    ("input0", "wk_bits_local"),
                    ("input1", "wk_scale_local"),
                )
            },
        )
        for slot, row in metadata.records_by_slot.items()
    }


def replay_wk(
    root: Path,
    record: Mapping[str, Any],
    slots: Mapping[int, int],
    bindings: Mapping[int, dict],
) -> list[dict]:
    """Check each owned FP8 tile and full completed BF16/FP32 boundaries.

    Fleet equality of full results plus all four feature slices establishes
    complete reconstruction, without retaining 32 full WK matrices in memory.
    """
    expected = {f"layer{l}/{n}" for l in (0, 1) for n in worker.PROGRAMS[:2]}
    if set(record["wk_originals"]) != expected:
        raise ValueError("dense WK capsule inventory differs")
    result, used = [], 0
    for layer in (0, 1):
        values = {}
        for name in worker.PROGRAMS[:2]:
            report = record["wk_originals"][f"layer{layer}/{name}"]
            arrays = read_npz(
                root / f"layer{layer}_{name}.npz", report, limit=WK_ORIGINALS_LIMIT
            )
            used += report["raw_array_bytes"] + (1 << 20)
            roles = (
                ("input0", "input1", "result")
                if name == "wk_decode"
                else ("input0", "result")
            )
            if used > WK_ORIGINALS_LIMIT or set(arrays) != {
                f"slot{s}_{r}" for s in slots.values() for r in roles
            }:
                raise ValueError("dense WK owner/budget differs")
            same_json(
                report["shapes"],
                {k: list(v.shape) for k, v in arrays.items()},
                "WK shapes",
            )
            if report["valid"] is not True:
                raise ValueError("dense WK worker reports invalid boundary")
            values[name] = arrays
        for slot in slots.values():
            d, p = values["wk_decode"], values["wk_promote"]
            bits, scales, decoded = (
                d[f"slot{slot}_{r}"] for r in ("input0", "input1", "result")
            )
            source, promoted = (p[f"slot{slot}_{r}"] for r in ("input0", "result"))
            for array, shape, dtype in (
                (bits, (128, 1536), np.uint8),
                (scales, (1, 12), np.float32),
                (decoded, (128, 6144), np.uint16),
                (source, (128, 6144), np.uint16),
                (promoted, (128, 6144), np.float32),
            ):
                if array.shape != shape or array.dtype != dtype:
                    raise ValueError("dense WK dtype/shape differs")
            if any(
                digest(v) != bindings[slot]["wk"][(layer, role)]
                for role, v in (("input0", bits), ("input1", scales))
            ):
                raise ValueError("dense WK source is not checkpoint leaf")
            raw32 = bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
            local = (raw32 * np.repeat(scales, 128, axis=1)).astype(ml_dtypes.bfloat16)
            feature = slot % 4
            decoded_values = decoded.view(ml_dtypes.bfloat16)
            if (
                not np.isfinite(local).all()
                or not np.isfinite(decoded_values).all()
                or not np.isfinite(promoted).all()
                or digest(local.view(np.uint16))
                != digest(decoded[:, feature * 1536 : (feature + 1) * 1536])
                or digest(source) != digest(decoded)
                or digest(promoted) != digest(decoded_values.astype(np.float32))
            ):
                raise ValueError("dense WK reconstruction/completed promotion differs")
            result.append(
                dict(
                    layer=layer,
                    slot=slot,
                    bf16_sha256=digest(decoded),
                    fp32_sha256=digest(promoted),
                )
            )
    return result


def read_model_capsules(
    root: Path, reports: Mapping[str, Any], slots: Mapping[int, int], witness: Mapping
) -> tuple[list, dict, dict]:
    """Shared original five-capsule schema/cache reader, not run admission.

    Return the actual arrays for a caller's additional retained-byte comparison;
    callers must independently validate their protocol, completion and journal.
    """
    if set(reports) != {n for n, _, _ in CAPSULES}:
        raise ValueError("dense model capsule inventory differs")
    owners, fingerprints, used = [], {}, 0
    saved = {}
    for name, count, endpoint in CAPSULES:
        report = reports[name]
        same_json(
            json.loads((root / f"{name}.json").read_bytes()),
            report,
            "dense original report",
        )
        same_json(
            {
                k: report.get(k)
                for k in (
                    "count",
                    "layers",
                    "local_slots",
                    "keep_caches",
                    "valid",
                    "errors",
                    "numerical_promotion",
                    "performance_claim",
                )
            },
            dict(
                count=count,
                layers=[0, 1],
                local_slots={str(d): s for d, s in slots.items()},
                keep_caches=endpoint,
                valid=True,
                errors=[],
                numerical_promotion=False,
                performance_claim=False,
            ),
            "dense capture schema",
        )
        arrays = read_npz(
            root / f"{name}.npz",
            report,
            limit=worker.ORIGINALS_LIMIT,
            size_key="npz_bytes",
        )
        used += report["raw_array_bytes"] + (1 << 20)
        if used > worker.ORIGINALS_LIMIT:
            raise ValueError("dense model originals exceed rank budget")
        specs = dict(
            output=((count, 1536), np.uint16),
            residual=((count, 1536), np.uint16),
            normalized=((count, 1536), np.uint16),
            health=((128,), np.bool_),
            positions=((count, 2048), np.int32),
            scores=((count, 2048), np.float32),
            counts=((count,), np.int32),
            routes=((count, 8), np.int32),
            route_weights=((count, 8), np.float32),
        )
        if endpoint:
            specs.update(
                {
                    f: ((16, 64, w), np.uint16)
                    for f, w in (("kv", 640), ("index", 128), ("repair", 128))
                }
            )
        expected = {
            f"slot{s}_layer{l}__{f}": (shape, dtype)
            for s in slots.values()
            for l in (0, 1)
            for f, (shape, dtype) in specs.items()
        }
        if set(arrays) != set(expected):
            raise ValueError("dense model array inventory differs")
        for key, (shape, dtype) in expected.items():
            v = arrays[key]
            field = key.split("__")[1]
            if v.shape != shape or v.dtype != dtype:
                raise ValueError("dense model array shape/dtype differs")
            if field == "health" and not v.all():
                raise ValueError("dense model recorded unhealthy rows")
            finite = v.view(ml_dtypes.bfloat16) if dtype == np.uint16 else v
            if field != "scores" and not np.isfinite(finite).all():
                raise ValueError("dense model nonfinite recorded operands")
            if field == "scores" and (np.isnan(v).any() or np.isposinf(v).any()):
                raise ValueError("dense model invalid scores")
            fingerprints[(name, key)] = digest(v)
        if endpoint:
            owners.extend(
                compare_owner(
                    witness, branch=name, slot=s, caches=cache_bits(arrays, s)
                )
                for s in slots.values()
            )
        saved[name] = arrays
    return owners, fingerprints, saved


def replay_outputs(
    root: Path, record: Mapping[str, Any], slots: Mapping[int, int], witness: Mapping
) -> tuple[dict, dict]:
    """Replay exact endpoint caches and all recorded live row/health geometry."""
    original = record["dense_frontier"]
    same_json(
        {
            k: original.get(k)
            for k in ("complete", "numerical_promotion", "performance_claim")
        },
        dict(complete=True, numerical_promotion=False, performance_claim=False),
        "dense output scope",
    )
    owners, fingerprints, _ = read_model_capsules(
        root, original["originals"], slots, witness
    )
    comparison = dict(
        owners=owners,
        reproduced=all(v["reproduced"] for v in owners),
        model_calls=5,
        wk_calls=4,
        numerical_promotion=False,
        performance_claim=False,
    )
    same_json(
        comparison_by_owner(original["comparison"]),
        comparison_by_owner(comparison),
        "dense DB604 comparison",
    )
    same_json(
        comparison_by_owner(json.loads((root / "comparison.json").read_bytes())),
        comparison_by_owner(comparison),
        "dense comparison file",
    )
    if comparison["reproduced"] is not True:
        raise ValueError("dense realization does not reproduce DB604")
    return comparison, fingerprints


def replay_execution(
    root: Path,
    record: Mapping[str, Any],
    slots: Mapping[int, int],
    *,
    graph_cache: dict | None = None,
) -> None:
    """Recompute original9-call or fixed norm18-call graph/journal/live budget."""
    from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol

    norm_mode = record.get("protocol") == norm_protocol.PROTOCOL
    selected_admission, selected_worker = admission, worker
    protocol_id, names, calls, journal_type = (
        protocol.PROTOCOL,
        worker.PROGRAMS,
        CALLS,
        DenseJournal,
    )
    if norm_mode:
        from scripts.greenfield import ws32_dense_norm_admission as selected_admission
        from scripts.greenfield import ws32_dense_norm_worker as selected_worker
        from scripts.greenfield.ws32_dense_frontier_execution import NormJournal

        protocol_id, names, calls, journal_type = (
            norm_protocol.PROTOCOL,
            norm_protocol.PROGRAMS,
            norm_protocol.CALLS,
            NormJournal,
        )

    def inspect(name, p, stable, optimized):
        key = (
            protocol_id,
            name,
            sha256(stable.encode()).hexdigest(),
            sha256(optimized.encode()).hexdigest(),
            json.dumps(p["compiled_memory"], sort_keys=True),
        )
        if graph_cache is None:
            checked = selected_admission.inspect_program(
                name, stable, optimized, p["compiled_memory"]
            )
        else:
            if key not in graph_cache:
                graph_cache[key] = selected_admission.inspect_program(
                    name, stable, optimized, p["compiled_memory"]
                )
            checked = graph_cache[key]
        same_json(p["admission"], checked, "dense actual HLO admission")

    journal = validate_graph_journal(
        root,
        record,
        protocol_id=protocol_id,
        profile=selected_admission.PROFILE if not norm_mode else norm_protocol.PROFILE,
        names=names,
        journal_type=journal_type,
        inspect=inspect,
        identity_fields=dict(
            protocol=protocol_id,
            compile_only=False,
            diagnostic_only=True,
            code_hash=record["code_hash"],
            launch_rank=record["launch_rank"],
        ),
        capture_report=dict(scope="ORIGINAL_CAPTURE_ONLY_NOT_ADMISSION"),
    )
    expected = ["identity"] + [
        stage
        for _ in worker.PROGRAMS
        for stage in ("lower_compile_started", "compiled", "raw_written", "inspected")
    ]
    expected += ["dense/admission"]
    for phase, _ in CALLS[:4]:
        expected += [phase + "/" + s for s in ("memory", "execute", "memory_after")]
    expected += [
        "dense/preflight",
        "dense/wide_initial",
        "dense/narrow_initial",
        "dense/independent",
        "dense/wide_inputs",
    ]
    for i, (phase, _) in enumerate(CALLS[4:]):
        if i:
            expected += [f"dense/narrow{i-1}_inputs"]
        expected += [phase + "/" + s for s in ("memory", "execute", "memory_after")]
    expected += ["dense/comparison"]
    if norm_mode:
        from scripts.greenfield.ws32_dense_norm_evidence import expected_stages

        expected = expected_stages()
    same_json([r["stage"] for r in journal], expected, "dense journal phase order")
    times = [r["monotonic_seconds"] for r in journal]
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in times
    ) or times != sorted(times):
        raise ValueError("dense journal time order differs")
    validate_call_sequence(
        record,
        record["call_evidence"],
        expected=calls,
        local_slots=slots,
        names=names,
        budgeter=selected_worker.memory_budget,
    )


def validate_fleet(
    root: Path,
    records: list[dict],
    *,
    pin: str,
    tag: str,
    repo: Path,
    original_root: Path,
    norm_original_root: Path | None = None,
) -> dict:
    """Read all eight rank directories; no worker verdict substitutes for replay.

    The protected publisher/controller must authenticate each local file's cloud
    generation and preserve failures separately before invoking this consumer.
    """
    import re
    from scripts.greenfield.ws32_dense_frontier_execution import (
        DSA_PINS,
        DSA_PIN_SOURCE_SHA,
    )
    from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol

    norm_mode = norm_protocol.is_tag(tag)
    if (norm_original_root is not None) != norm_mode:
        raise ValueError("norm fleet requires its own retained-original root only")
    protocol_id = norm_protocol.PROTOCOL if norm_mode else protocol.PROTOCOL
    kernel = norm_protocol.KERNEL if norm_mode else protocol.KERNEL
    profile = norm_protocol.PROFILE if norm_mode else admission.PROFILE
    names = norm_protocol.PROGRAMS if norm_mode else worker.PROGRAMS

    if (
        not (norm_mode or protocol.is_tag(tag))
        or not re.fullmatch(r"[0-9a-f]{40}", pin)
        or len(records) != 8
        or [r["launch_rank"] for r in records] != list(range(8))
    ):
        raise ValueError("dense fleet tag/pin/rank inventory differs")
    bindings = checkpoint_bindings(repo)
    if set(bindings) != set(range(32)):
        raise ValueError("dense checkpoint ownership incomplete")
    ids, processes, hosts, wk_rows, row_hashes, comparisons = (
        {},
        set(),
        set(),
        [],
        {},
        [],
    )
    graph_cache, graph_signature = {}, None
    norm_replays = []
    for rank, record in enumerate(records):
        prior, witness = protocol.load_reference(original_root, rank=rank)
        slots = {o["device_id"]: o["device_slot"] for o in prior["local_device_slots"]}
        fixed = dict(
            status="DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION",
            tag=tag,
            code_hash=pin,
            launch_rank=rank,
            protocol=protocol_id,
            kernel=kernel,
            profile=profile,
            diagnostic_only=True,
            compile_only=False,
            performance_claim=False,
            numerical_promotion=False,
            current_phase="norm/comparison" if norm_mode else "dense/comparison",
            selected_layer_ids=[0, 1],
            include_embedding=True,
            payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
            integrity_scope="selected_layers_and_embedding_only_not_complete_checkpoint",
            original_tag=protocol.ORIGINAL_TAG,
            original_ledger_sha256=protocol.LEDGER_SHA,
            original_runner_sha256=sha256(
                (original_root / protocol.original_names(rank)[0]).read_bytes()
            ).hexdigest(),
            dsa_oracle_pins=DSA_PINS,
            dsa_pin_source_sha256=DSA_PIN_SOURCE_SHA,
            versions={"jax": "0.10.1", "libtpu": "0.0.41"},
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
            fixed[key] = prior[key]
        same_json({k: record.get(k) for k in fixed}, fixed, "dense fleet identity")
        if (
            record["jax_process_index"] in processes
            or record["hostname"] in hosts
            or not all(
                type(record.get(k)) is int and record[k] > 0
                for k in ("pid", "start_ticks")
            )
            or not record.get("boot_id")
        ):
            raise ValueError("dense process identity incomplete/duplicated")
        processes.add(record["jax_process_index"])
        hosts.add(record["hostname"])
        own = record["local_device_slots"]
        if len(own) != 4 or {o["device_id"]: o["device_slot"] for o in own} != slots:
            raise ValueError("dense current physical owners differ from original")
        for owner in own:
            device, slot = owner["device_id"], owner["device_slot"]
            if slot in ids or device in ids.values():
                raise ValueError("dense duplicate physical owner")
            ids[slot] = device
            same_json(
                owner,
                dict(
                    device_id=device,
                    device_slot=slot,
                    expected_full_file_sha256_not_verified=bindings[slot][
                        "full_file_sha256"
                    ],
                    observed_selected_tensor_sha256=bindings[slot]["selected"],
                    selected_payload_bytes=protocol.PAYLOAD_BYTES,
                ),
                "dense selected leaf identity",
            )
            original_owner = next(
                o for o in prior["local_device_slots"] if o["device_slot"] == slot
            )
            if bindings[slot]["full_file_sha256"] != original_owner["file_sha256"]:
                raise ValueError("dense original checkpoint file differs")
        rankroot = root / f"rank{rank}"
        preflight_raw = (rankroot / "retained_preflight.json").read_bytes()
        if sha256(preflight_raw).hexdigest() != record["retained_preflight_sha256"]:
            raise ValueError("dense retained preflight SHA differs")
        preflight = json.loads(preflight_raw)
        same_json(
            {
                k: preflight[k]
                for k in (
                    "tag",
                    "code_hash",
                    "launch_rank",
                    "hostname",
                    "original_runner_sha256",
                )
            },
            {
                k: fixed[k]
                for k in (
                    "tag",
                    "code_hash",
                    "launch_rank",
                    "hostname",
                    "original_runner_sha256",
                )
            },
            "dense saved preflight identity",
        )
        same_json(
            preflight["checkpoint_pins"],
            record["checkpoint_pins"],
            "dense checkpoint pins",
        )
        norm_originals = None
        if norm_mode:
            from scripts.greenfield import ws32_dense_norm_originals as norm_source

            prior_norm, norm_originals, identity = norm_source.load_bundle(
                norm_original_root / f"rank{rank}", repo=repo, rank=rank
            )
            norm_source.bind_prior(
                prior_norm, prior, prior_sha256=fixed["original_runner_sha256"]
            )
            same_json(record["norm_originals"], identity, "norm fleet source originals")
            same_json(
                preflight["norm_originals"], identity, "norm fleet preflight originals"
            )
            same_json(
                preflight["protocol"], protocol_id, "norm fleet preflight protocol"
            )
            combined_bytes = (
                protocol.LEDGER_PIN["size"]
                + sum(
                    p["size"]
                    for p in protocol.reference_pins(original_root, rank).values()
                )
                + identity["bytes"]
            )
            if combined_bytes > norm_source.MAX_REFERENCE_BYTES:
                raise ValueError("norm fleet combined reference budget exceeded")
            same_json(
                preflight["combined_reference_bytes"],
                combined_bytes,
                "norm combined reference bytes",
            )
            same_json(
                prior_norm["physical_device_ids"],
                record["physical_device_ids"],
                "norm original physical mesh",
            )
        signature = {
            n: {
                k: record["programs"][n][k]
                for k in ("stablehlo_sha256", "optimized_hlo_sha256", "compiled_memory")
            }
            for n in names
        }
        if graph_signature is None:
            graph_signature = signature
        same_json(signature, graph_signature, "dense cross-host actual graphs")
        replay_execution(rankroot, record, slots, graph_cache=graph_cache)
        wk_rows.extend(replay_wk(rankroot, record, slots, bindings))
        if norm_mode:
            from scripts.greenfield import ws32_dense_norm_evidence as norm_evidence

            comparison, fingerprints = norm_evidence.replay_outputs(
                rankroot, record, slots, witness, norm_originals, bindings
            )
            norm_replays.append(dict(rank=rank, **comparison))
        else:
            comparison, fingerprints = replay_outputs(rankroot, record, slots, witness)
        comparisons.extend(comparison["owners"])
        if row_hashes.keys() & fingerprints.keys():
            raise ValueError("dense duplicate row evidence")
        row_hashes.update(fingerprints)
    if set(ids) != set(range(32)) or processes != set(range(8)):
        raise ValueError("dense fleet does not cover all physical owners/processes")
    expected_mesh = [[ids[e * 4 + f] for f in range(4)] for e in range(8)]
    for record in records:
        same_json(
            record["physical_device_ids"], expected_mesh, "dense recorded global mesh"
        )
    for layer in (0, 1):
        rows = [r for r in wk_rows if r["layer"] == layer]
        if len(rows) != 32 or any(
            len({r[k] for r in rows}) != 1 for k in ("bf16_sha256", "fp32_sha256")
        ):
            raise ValueError("dense completed WK replicas disagree")
    # Completed model outputs are expert-replicated/feature-sharded; caches are
    # feature-replicated/expert-sharded. Metadata and selected scores are global.
    from scripts.greenfield.prefill_layer_numerical import FIELDS

    for name, _, endpoint in CAPSULES:
        for layer in (0, 1):
            for field in FIELDS:
                if field in ("kv", "index", "repair"):
                    if not endpoint:
                        continue
                    groups = [range(e * 4, e * 4 + 4) for e in range(8)]
                elif field in ("output", "residual", "normalized"):
                    groups = [range(f, 32, 4) for f in range(4)]
                else:
                    groups = [range(32)]
                for group in groups:
                    if (
                        len(
                            {
                                row_hashes[(name, f"slot{s}_layer{layer}__{field}")]
                                for s in group
                            }
                        )
                        != 1
                    ):
                        raise ValueError("dense recorded output replicas disagree")
    return dict(
        protocol=protocol_id,
        owners=32,
        hosts=8,
        model_calls_per_host=14 if norm_mode else 5,
        wk_calls_per_host=4,
        reproduced=True,
        comparisons=comparisons,
        numerical_promotion=False,
        performance_claim=False,
        scope=(
            "NORM_DB605_OWN_SUFFIX_REPRODUCTION_NOT_8K_CORRECTNESS_OR_CAUSE"
            if norm_mode
            else "DENSE01_DB604_REPRODUCTION_NOT_8K_CORRECTNESS_OR_CAUSE"
        ),
        **(dict(norm_replays=norm_replays, cause_claim=False) if norm_mode else {}),
    )
