"""Independent saved-materializer replay, with bounded selected-source reads.

Use replay_rank on each authenticated local original directory, then pass those
fresh results to replay_fleet. The latter REQUIRES read_source(slot, tensor_name)
to return one original selected-checkpoint ndarray. Shape, dtype, exact bytes
and ledger SHA are checked here, regardless of the transport's own assertions.
Only 256 unique non-WK tiles / 99,639,040 bytes are requested across the fleet;
WK reconstruction uses the already-saved source/result NPZs. No whole model,
overlay payload, device mesh or new exact-output artifact is materialized.

The callback is intentionally a parent dependency: this controller's four
retained owner files cannot supply all feature slices. No absent-reader or
worker-success fallback is permitted. This module opens no network connection,
dispatches no model and never establishes history/reproduction/HBM admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import ml_dtypes
import numpy as np

from scripts.greenfield import ws32_history_materializers as capture
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.prefill_window_evidence import same_json
from scripts.greenfield.ws32_dense_frontier_evidence import digest, read_npz

SOURCE_READ_LIMIT = 100_000_000
SOURCE_READ_BYTES = 99_639_040
SOURCE_READ_TILES = 256
SCHEMA = "ws32_history_materializer_replay_v1"


@dataclass(frozen=True)
class CheckpointBindings:
    """Build only from the existing authenticated metadata verifier."""

    config: Any
    context: Mapping[str, str]
    owners: Mapping[int, Mapping]
    overlay: Any


@dataclass(frozen=True)
class RankReplay:
    """Transient collector result, not an accepted worker assertion/cache."""

    rank: int
    process_index: int
    local_slots: Mapping[int, int]
    sources: Mapping[str, Mapping[int, str]]
    decoded: Mapping[str, Mapping[int, str]]
    promoted: Mapping[str, Mapping[int, str]]
    wk: Mapping[int, Mapping[int, tuple[str, str]]]
    original_bytes: int


def checkpoint_bindings(repo: Path) -> CheckpointBindings:
    """Authenticate raw/overlay metadata, reading zero tensor payload bytes."""
    from scripts.greenfield.ws32_history_compile import require_source
    from scripts.greenfield.ws32_canonical_prefill_compile import read_metadata
    from glm_tpu.greenfield.checkpoint.ws32_layer_subset import ws32_expected_layer_names
    from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import verify_ws32_strategy_nd_dense_overlay
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
    from glm_tpu.greenfield.types import ModelGeometry

    require_source(repo)
    metadata = read_metadata(repo)
    geometry = ModelGeometry.from_dict(metadata.manifest["geometry"])
    config = Ws32DecoderConfig(geometry, protocol.CAPACITY, exact_dsa=True, host_main_rope_table=True)
    names = frozenset().union(*(ws32_expected_layer_names(geometry, layer) for layer in protocol.LAYERS))
    names |= {"model.embed_tokens.weight"}
    schema = metadata.plans[0].tensors
    selected = {tensor.name: index for index, tensor in enumerate(schema) if tensor.name in names}
    if (set(selected) != names or len(selected) != protocol.SELECTED_LEAVES
            or sum(schema[index].byte_count for index in selected.values()) != protocol.PAYLOAD_BYTES
            or set(metadata.records_by_slot) != set(range(32))):
        raise ValueError("history materializer selected metadata inventory differs")
    for layer in protocol.PRODUCERS:
        for field, shape, dtype, axes in capture.RAW:
            tensor = schema[selected[capture.source_names(layer)[field]]]
            expected_dtype = {"uint8": "U8", "float32": "F32", "bfloat16": "BF16"}[dtype]
            expected_shape = _shape((field, shape, dtype, axes), 0)
            if (tensor.global_shape != shape or tensor.local_shape != expected_shape
                    or tensor.dtype != expected_dtype or tensor.partition_spec != axes
                    or tensor.byte_count != prod(expected_shape) * np.dtype(dtype).itemsize):
                raise ValueError("history materializer raw metadata contract differs")
    owners = {slot: dict(full_file_sha256=row["sha256"], selected={name: row["tensor_sha256"][index]
              for name, index in selected.items()}, header={key: row[key] for key in
              ("device_slot", "filename", "file_bytes", "header_sha256")})
              for slot, row in metadata.records_by_slot.items()}
    env = json.loads((repo / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    prefix = "GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_"
    overlay = verify_ws32_strategy_nd_dense_overlay(Path(env[prefix + "ROOT"]),
        expected_manifest_sha256=env[prefix + "MANIFEST_SHA"],
        expected_manifest_file_sha256=env[prefix + "MANIFEST_FILE_SHA"],
        expected_success_file_sha256=env[prefix + "SUCCESS_FILE_SHA"])
    context = dict(checkpoint_manifest_sha256=metadata.manifest["manifest_sha256"],
        checkpoint_success_sha256=metadata.success["success_sha256"],
        source_inventory_sha256=metadata.manifest["source"]["inventory_sha256"],
        mesh_sha256=metadata.manifest["mesh_hash"], topology_sha256=metadata.success["topology_hash"])
    return CheckpointBindings(config, context, owners, overlay)


def _shape(spec: tuple, slot: int) -> tuple:
    _, shape, _, axes = spec
    return tuple(len(range(*part.indices(size))) for part, size in
                 zip(capture._index(shape, axes, slot), shape, strict=True))


def _expected_row(spec: tuple, slot: int, device: int) -> dict:
    _, shape, dtype, axes = spec
    return dict(device_id=device, slot=slot,
        index=[list(part.indices(size)) for part, size in zip(capture._index(shape, axes, slot), shape, strict=True)],
        shape=list(_shape(spec, slot)), dtype=dtype,
        bytes=prod(_shape(spec, slot)) * np.dtype(dtype).itemsize, finite=True)


def _source(bindings: CheckpointBindings, layer: int, field: str, slot: int) -> dict:
    name = capture.source_names(layer)[field]
    row = bindings.owners[slot]
    return dict(tensor=name, selected_tensor_sha256=row["selected"][name],
                expected_full_file_sha256_not_verified=row["full_file_sha256"])


def _rows(rows: Any, spec: tuple, slots: Mapping[int, int], *, extras: Mapping[int, Mapping] | None = None) -> dict:
    if (not isinstance(rows, list) or len(rows) != len(slots)
            or any(not isinstance(row, dict) or type(row.get("slot")) is not int for row in rows)):
        raise ValueError("history materializer row inventory differs")
    by_slot = {row["slot"]: row for row in rows}
    if len(by_slot) != len(rows) or set(by_slot) != set(slots.values()):
        raise ValueError("history materializer missing/duplicate row owner")
    for device, slot in slots.items():
        row = by_slot[slot]
        expected = {**_expected_row(spec, slot, device), **({} if extras is None else extras[slot])}
        if set(row) != {*expected, "sha256"} or not isinstance(row["sha256"], str) or not capture._HEX.fullmatch(row["sha256"]):
            raise ValueError("history materializer row schema/digest differs")
        same_json({key: row[key] for key in expected}, expected, "history materializer actual owner geometry")
    return by_slot


def _base(report: Mapping, name: str, slots: Mapping, record: Mapping) -> None:
    expected = capture._base(name, slots, record)
    expected.update(valid=True)
    for key in ("leaves",):
        expected.pop(key)
    same_json({key: report.get(key) for key in expected}, expected, "history materializer completed receipt")


def _same_hashes(actual: Mapping[int, str], expected: Mapping[int, str], label: str) -> None:
    # Existing same_json correctly rejects integer-key JSON maps. Preserve the
    # exact owner join as sorted explicit pairs instead of changing that guard.
    same_json([[slot, value] for slot, value in sorted(actual.items())],
              [[slot, value] for slot, value in sorted(expected.items())], label)


def _record_owners(record: Mapping, slots: Mapping, bindings: CheckpointBindings) -> None:
    if (type(record.get("launch_rank")) is not int or not 0 <= record["launch_rank"] < 8
            or type(record.get("jax_process_index")) is not int or not 0 <= record["jax_process_index"] < 8
            or len(slots) != 4 or len(set(slots.values())) != 4
            or any(type(device) is not int or not 0 <= device < 32 or type(slot) is not int or not 0 <= slot < 32
                   for device, slot in slots.items())):
        raise ValueError("history materializer rank/process/physical owners differ")
    same_json({key: record.get(key) for key in bindings.context}, bindings.context, "history materializer checkpoint context")
    expected = [dict(device_id=device, device_slot=slot,
        expected_full_file_sha256_not_verified=bindings.owners[slot]["full_file_sha256"],
        observed_selected_tensor_sha256=bindings.owners[slot]["selected"], selected_payload_bytes=protocol.PAYLOAD_BYTES)
        for device, slot in slots.items()]
    rows = record.get("local_device_slots", ())
    if not isinstance(rows, (tuple, list)) or len(rows) != 4:
        raise ValueError("history materializer selected owner ledger differs")
    same_json(sorted(rows, key=lambda row: row["device_slot"]), sorted(expected, key=lambda row: row["device_slot"]),
              "history materializer selected source ledger")
    overlay = bindings.overlay
    expected_overlay = dict(manifest_sha256=overlay.manifest["manifest_sha256"],
        manifest_file_sha256=overlay.manifest_file_sha256, success_file_sha256=overlay.success_file_sha256,
        local_records=[dict(device_id=device, expert_coordinate=slot // 4, feature_coordinate=slot % 4,
                           layer_id=layer, file_sha256=overlay.records[layer, slot // 4, slot % 4]["sha256"])
                       for device, slot in slots.items() for layer in range(3)])
    observed = record.get("strategy_nd_dense_overlay", {})
    key = lambda row: (row["device_id"], row["layer_id"])
    same_json({**observed, "local_records": sorted(observed.get("local_records", ()), key=key)},
              {**expected_overlay, "local_records": sorted(expected_overlay["local_records"], key=key)},
              "history materializer original overlay context")


def _wk(root: Path, record: Mapping, slots: Mapping, bindings: CheckpointBindings) -> dict:
    results = {}
    for layer in protocol.PRODUCERS:
        actual = {}
        for name in ("wk_decode", "wk_promote"):
            key = f"layer{layer}/{name}"
            report = record["materializer_originals"][key]
            _base(report, key, slots, record)
            original = report["original"]
            expected_path = f"materializers/{capture._FILENAMES[key]}"
            if original.get("path") != expected_path or original.get("sha256") != original.get("npz_sha256"):
                raise ValueError("history WK original path/hash binding differs")
            arrays = read_npz(root / expected_path, original, limit=capture.WK_ORIGINALS_LIMIT)
            specs = (("result", *capture.DECODED[3][1:]), ("input0", *capture.RAW[6][1:]),
                     ("input1", *capture.RAW[7][1:])) if name == "wk_decode" else (
                     ("result", (128, 6144), "float32", (None, None)), ("input0", *capture.DECODED[3][1:]))
            if set(report["leaves"]) != {spec[0] for spec in specs} or set(arrays) != {
                f"slot{slot}_{spec[0]}" for spec in specs for slot in slots.values()}:
                raise ValueError("history WK leaf/array inventory differs")
            for spec in specs:
                role, _, dtype, _ = spec
                extra = {slot: dict(checkpoint_source=_source(bindings, layer,
                    "wk_bits_local" if role == "input0" else "wk_scale_local", slot)) for slot in slots.values()}
                rows = _rows(report["leaves"][role], spec, slots,
                    extras=extra if name == "wk_decode" and role != "result" else None)
                for slot in slots.values():
                    value = arrays[f"slot{slot}_{role}"]
                    storage_dtype = "uint16" if dtype == "bfloat16" else dtype
                    if tuple(value.shape) != _shape(spec, slot) or str(value.dtype) != storage_dtype or digest(value) != rows[slot]["sha256"]:
                        raise ValueError("history WK original actual shape/dtype/hash differs")
                    host = value.view(ml_dtypes.bfloat16) if dtype == "bfloat16" else value
                    if not np.isfinite(host).all():
                        raise ValueError("history WK original actual values nonfinite")
                    if name == "wk_decode" and role != "result" and digest(value) != extra[slot]["checkpoint_source"]["selected_tensor_sha256"]:
                        raise ValueError("history WK original differs from selected checkpoint")
            actual[name] = arrays
        results[layer] = {}
        for slot in slots.values():
            decoded = actual["wk_decode"][f"slot{slot}_result"]
            source = actual["wk_promote"][f"slot{slot}_input0"]
            promoted = actual["wk_promote"][f"slot{slot}_result"]
            bits = actual["wk_decode"][f"slot{slot}_input0"]
            scales = actual["wk_decode"][f"slot{slot}_input1"]
            # Existing dense replay arithmetic, including the completed BF16 round.
            local = (bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
                     * np.repeat(scales, 128, axis=1)).astype(ml_dtypes.bfloat16)
            feature = slot % 4
            if (not np.isfinite(local).all() or digest(local) != digest(decoded[:, feature * 1536:(feature + 1) * 1536])
                    or digest(source) != digest(decoded)
                    or digest(promoted) != digest(decoded.view(ml_dtypes.bfloat16).astype(np.float32))):
                raise ValueError("history WK reconstruction/completed promotion differs")
            results[layer][slot] = (digest(decoded), digest(promoted))
    return results


def replay_rank(root: Path, record: Mapping, local_slots: Mapping[int, int], bindings: CheckpointBindings) -> RankReplay:
    """Verify eight WK capsules and two exact receipts; return hashes, not arrays."""
    _record_owners(record, local_slots, bindings)
    keys = {*capture._WK_KEYS, "exact_decode", "exact_promote"}
    if set(record.get("materializer_originals", {})) != keys:
        raise ValueError("history materializer complete original inventory differs")
    directory = capture._directory(root, create=False)
    expected_files = {capture._FILENAMES[key] for key in keys}
    if {path.name for path in directory.iterdir()} != expected_files:
        raise ValueError("history materializer unexpected/missing complete-run files")
    sizes = {path.name: path.stat().st_size for path in directory.iterdir()}
    if (sum(sizes.values()) > capture.TOTAL_LIMIT
            or sum(sizes[capture._FILENAMES[key]] for key in capture._WK_KEYS) > capture.WK_ORIGINALS_LIMIT):
        raise ValueError("history materializer original aggregate budget exceeded")
    wk = _wk(root, record, local_slots, bindings)
    decoded = capture.load_exact_receipt(root, record, "exact_decode")
    promoted = capture.load_exact_receipt(root, record, "exact_promote")
    sources, decoded_hashes, promoted_hashes = {}, {}, {}
    for name, report, input_specs, output_specs in (
        ("exact_decode", decoded, capture.RAW, capture.DECODED),
        ("exact_promote", promoted, capture.DECODED, capture.PROMOTED)):
        _base(report, name, local_slots, record)
        if (report.get("full_exact_output_originals_preserved") is not False
                or report.get("exact_decode_reconstruction_checked") is not False or "refused_original" in report):
            raise ValueError("history exact receipt evidence scope differs")
        for category, specs in (("source_leaves", input_specs), ("leaves", output_specs)):
            expected_names = {f"layer{layer}/{spec[0]}" for layer in protocol.PRODUCERS for spec in specs}
            if set(report.get(category, {})) != expected_names:
                raise ValueError("history exact leaf inventory differs")
            for layer in protocol.PRODUCERS:
                for spec in specs:
                    field, context = spec[0], f"layer{layer}/{spec[0]}"
                    extras = {}
                    for slot in local_slots.values():
                        extra = {}
                        if category == "source_leaves":
                            if name == "exact_decode":
                                extra["checkpoint_source"] = _source(bindings, layer, field, slot)
                            else:
                                extra["exact_decode_original_sha256"] = decoded_hashes[context][slot]
                        else:
                            if field in ("wk_weight_bf16", "wk_weight"):
                                extra["wk_original_sha256"] = wk[layer][slot][name == "exact_promote"]
                            if name == "exact_promote":
                                source_field = "wq_b_weight_local" if field.startswith("wq_b_weight_aliases/") else field
                                extra["promotion_input_sha256"] = (wk[layer][slot][1] if field == "wk_weight" else
                                    decoded_hashes[f"layer{layer}/{source_field}"][slot])
                        extras[slot] = extra
                    rows = _rows(report[category][context], spec, local_slots, extras=extras)
                    hashes = {slot: row["sha256"] for slot, row in rows.items()}
                    if category == "source_leaves":
                        expected = {slot: (_source(bindings, layer, field, slot)["selected_tensor_sha256"]
                            if name == "exact_decode" else decoded_hashes[context][slot]) for slot in local_slots.values()}
                        _same_hashes(hashes, expected, "history exact original input/source bytes")
                        if name == "exact_decode":
                            sources[context] = hashes
                    elif name == "exact_decode":
                        decoded_hashes[context] = hashes
                        if field == "wk_weight_bf16":
                            _same_hashes(hashes, {slot: wk[layer][slot][0] for slot in local_slots.values()}, "history exact decoded WK bytes")
                    else:
                        promoted_hashes[context] = hashes
                        _same_hashes(hashes, {slot: extras[slot]["promotion_input_sha256"] for slot in local_slots.values()},
                                  "history exact completed promotion bytes")
    return RankReplay(record["launch_rank"], record["jax_process_index"], dict(local_slots),
                      sources, decoded_hashes, promoted_hashes, wk, sum(sizes.values()))


def _cpu_functions() -> tuple[Callable, Callable]:
    # Never let a collector reference accidentally initialize or dispatch TPU.
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("history materializer reconstruction requires JAX_PLATFORMS=cpu")
    import jax
    from glm_tpu.greenfield.runtime.ws32_decoder import _pack_ws32_fused_qkv_a
    from glm_tpu.greenfield.kernels.reference.fp8 import dequantize_fp8_bits_block_weight
    if jax.default_backend() != "cpu":
        raise ValueError("history materializer reconstruction backend differs")
    return _pack_ws32_fused_qkv_a, dequantize_fp8_bits_block_weight


def replay_fleet(reports: Sequence[RankReplay], *, bindings: CheckpointBindings,
                 read_source: Callable[[int, str], np.ndarray]) -> dict:
    """Reconstruct hashes from authenticated selected bytes, never success flags.

    The parent must supply fresh replay_rank results, authenticated original
    rank/physical identity, and a bounded selected source transport. This function
    deliberately cannot accept a missing reader as partial numerical admission.
    """
    if not callable(read_source):
        raise ValueError("history exact reconstruction requires selected source reader")
    if (len(reports) != 8 or any(not isinstance(report, RankReplay) or type(report.rank) is not int
            or type(report.process_index) is not int for report in reports)
            or {report.rank for report in reports} != set(range(8))
            or {report.process_index for report in reports} != set(range(8))):
        raise ValueError("history materializer eight-rank/process inventory differs")
    slots = [slot for report in reports for slot in report.local_slots.values()]
    devices = [device for report in reports for device in report.local_slots]
    if len(slots) != 32 or set(slots) != set(range(32)) or len(devices) != 32 or set(devices) != set(range(32)):
        raise ValueError("history materializer complete physical owner join differs")
    sources, decoded, promoted, wk = {}, {}, {}, {}
    for target, attribute, specs in ((sources, "sources", capture.RAW), (decoded, "decoded", capture.DECODED),
                                     (promoted, "promoted", capture.PROMOTED)):
        for layer in protocol.PRODUCERS:
            for spec in specs:
                context = f"layer{layer}/{spec[0]}"
                if any(set(getattr(report, attribute)[context]) != set(report.local_slots.values()) for report in reports):
                    raise ValueError("history materializer leaf/rank owner association differs")
                target[context] = {slot: sha for report in reports for slot, sha in getattr(report, attribute)[context].items()}
                if set(target[context]) != set(range(32)) or sum(len(getattr(report, attribute)[context]) for report in reports) != 32:
                    raise ValueError("history materializer leaf owner join differs")
                groups = {}
                for slot, value in target[context].items():
                    index = tuple(tuple(part) for part in _expected_row(spec, slot, 0)["index"])
                    groups.setdefault(index, set()).add(value)
                if any(len(group) != 1 for group in groups.values()):
                    raise ValueError("history materializer fleet replica hashes differ")
    for layer in protocol.PRODUCERS:
        if any(set(report.wk[layer]) != set(report.local_slots.values()) for report in reports):
            raise ValueError("history materializer WK/rank owner association differs")
        wk[layer] = {slot: value for report in reports for slot, value in report.wk[layer].items()}
        if (set(wk[layer]) != set(range(32)) or len(set(wk[layer].values())) != 1
                or any(decoded[f"layer{layer}/wk_weight_bf16"][slot] != wk[layer][slot][0]
                       or promoted[f"layer{layer}/wk_weight"][slot] != wk[layer][slot][1] for slot in range(32))):
            raise ValueError("history materializer fleet completed WK replicas differ")
        for field, *_ in capture.RAW:
            if any(sources[f"layer{layer}/{field}"][slot] != _source(bindings, layer, field, slot)["selected_tensor_sha256"]
                   for slot in range(32)):
                raise ValueError("history exact fleet source differs from authenticated selected ledger")
        for field, *_ in capture.PROMOTED:
            source_field = "wq_b_weight_local" if field.startswith("wq_b_weight_aliases/") else field
            expected = {slot: wk[layer][slot][1] if field == "wk_weight" else decoded[f"layer{layer}/{source_field}"][slot]
                        for slot in range(32)}
            _same_hashes(promoted[f"layer{layer}/{field}"], expected, "history exact fleet promotion bytes")
    pack, dequantize = _cpu_functions()
    used, tile_count = 0, 0

    def read_global(layer: int, field: str) -> np.ndarray:
        nonlocal used, tile_count
        spec = next(spec for spec in capture.RAW if spec[0] == field)
        _, shape, dtype, axes = spec
        result = np.empty(shape, dtype=dtype)
        seen = set()
        for slot in range(32):
            expected_sha = _source(bindings, layer, field, slot)["selected_tensor_sha256"]
            if sources[f"layer{layer}/{field}"][slot] != expected_sha:
                raise ValueError("history exact fleet source differs from authenticated selected ledger")
            index = capture._index(shape, axes, slot)
            key = tuple(part.indices(size) for part, size in zip(index, shape, strict=True))
            if key in seen:
                continue
            seen.add(key)
            requested = prod(_shape(spec, slot)) * np.dtype(dtype).itemsize
            if used + requested > SOURCE_READ_LIMIT:
                raise ValueError("history exact selected source read budget exceeded")
            value = read_source(slot, capture.source_names(layer)[field])
            if not isinstance(value, np.ndarray) or value.shape != _shape(spec, slot) or str(value.dtype) != dtype:
                raise ValueError("history exact selected source shape/dtype differs")
            raw = value.tobytes(order="C")
            if len(raw) != requested or sha256(raw).hexdigest() != expected_sha:
                raise ValueError("history exact selected source byte count/SHA differs")
            from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import _validate_finite_chunk
            _validate_finite_chunk(raw, {"uint8": "U8", "float32": "F32", "bfloat16": "BF16"}[dtype])
            result[index] = np.frombuffer(raw, dtype=dtype).reshape(value.shape)
            used += requested
            tile_count += 1
        return result

    def check(layer: int, field: str, value: Any, *, feature: int | None = None) -> None:
        host = np.asarray(value)
        if not np.isfinite(host).all():
            raise ValueError("history exact reconstructed output nonfinite")
        observed = digest(host)
        if any(observed != decoded[f"layer{layer}/{field}"][slot]
               for slot in range(32) if feature is None or slot % 4 == feature):
            raise ValueError(f"history exact decoded reconstruction differs: layer{layer}/{field}")

    for layer in protocol.PRODUCERS:
        q_bits, q_scale, kv_bits, kv_scale = (read_global(layer, field) for field in
            ("q_a_bits_local", "q_a_scale_local", "kv_a_bits_local", "kv_a_scale_local"))
        packed_bits, packed_scale = pack(q_bits, q_scale, kv_bits, kv_scale, config=bindings.config)
        check(layer, "qkv_a_bits", packed_bits)
        check(layer, "qkv_a_scale", packed_scale)
        del q_bits, q_scale, kv_bits, kv_scale, packed_bits, packed_scale
        bits, scales = read_global(layer, "wq_b_bits_local"), read_global(layer, "wq_b_scale_local")
        for feature in range(4):
            value = dequantize(bits[feature * 1024:(feature + 1) * 1024],
                scales[feature * 8:(feature + 1) * 8], output_dtype=np.float32)
            check(layer, "wq_b_weight_local", value, feature=feature)
            del value
        del bits, scales
        head = read_global(layer, "head_weight_local")
        for feature in range(4):
            check(layer, "head_weight_local", head[feature * 8:(feature + 1) * 8], feature=feature)
        del head
    if used != SOURCE_READ_BYTES or tile_count != SOURCE_READ_TILES:
        raise ValueError("history exact selected read inventory differs")
    return dict(schema=SCHEMA, physical_owners=32, rank_count=8, wk_originals_replayed=64,
        exact_receipts_replayed=16, selected_source_read_bytes=used, selected_source_read_tiles=tile_count,
        original_bytes=sum(report.original_bytes for report in reports),
        exact_decode_reconstruction_checked=True, exact_promote_reconstruction_checked=True,
        overlay_payload_bytes_read=0, full_checkpoint_verified=False, numerical_promotion=False,
        history_or_original_event_reproduction=False, runtime_hbm_admission=False, performance_claim=False)
