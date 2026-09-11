"""Bounded, addressable-owner materializer originals; no executable dispatch.

WK capsules retain the same source/result boundaries as the dense diagnostic,
extended to producers 0/1/2/6. Exact trees are *not* dumped: each actual local
leaf is streamed through SHA/finite/replica checks. Their source bytes remain in
the selected checkpoint owners named by the receipt, not in this directory.
An independent collector must authenticate those ledgers and reconstruct the
frozen decode operations before claiming numerical materializer correctness.

Completed numerically refused calls retain their receipt and, for exact trees,
at most one offending local array/pair. Invalid owner geometry is not trusted
and can refuse before payload preservation. These callbacks confer no launch,
HLO, all-live HBM, legacy bit-identity or performance authority.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import ml_dtypes
import numpy as np

from scripts.greenfield import ws32_history_capture as owner_capture
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.ws32_prefill_frontier_state import _index_key

SCHEMA = "ws32_history_materializer_original_v1"
WK_ORIGINALS_LIMIT = 112 << 20
EXACT_RECEIPT_LIMIT = 1 << 20
REFUSAL_ORIGINAL_LIMIT = 33 << 20
TOTAL_LIMIT = WK_ORIGINALS_LIMIT + 2 * EXACT_RECEIPT_LIMIT + REFUSAL_ORIGINAL_LIMIT
WK_RAW_BYTES_FOUR_OWNERS = 103_809_792
MAX_REFUSAL_RAW_BYTES = 32_243_712
_HEX = re.compile(r"[0-9a-f]{64}")
_WK_KEYS = tuple(f"layer{layer}/{name}" for layer in protocol.PRODUCERS
                 for name in ("wk_decode", "wk_promote"))
_FILENAMES = {key: key.replace("/", "_") + ".npz" for key in _WK_KEYS}
_FILENAMES.update(exact_decode="exact_decode.json", exact_promote="exact_promote.json",
                  refused_exact="refused_exact.npz")

# Global shape, dtype and explicit partition axes. Never infer output sharding
# from eval_shape (it drops it), labels, local shard order or full-host gathers.
RAW = (
    ("q_a_bits_local", (2048, 6144), "uint8", (None, "feature")),
    ("q_a_scale_local", (16, 48), "float32", (None, "feature")),
    ("kv_a_bits_local", (576, 6144), "uint8", (None, "feature")),
    ("kv_a_scale_local", (5, 48), "float32", (None, "feature")),
    ("wq_b_bits_local", (4096, 2048), "uint8", ("expert", None)),
    ("wq_b_scale_local", (32, 16), "float32", ("expert", None)),
    ("wk_bits_local", (128, 6144), "uint8", (None, "feature")),
    ("wk_scale_local", (1, 48), "float32", (None, "feature")),
    ("head_weight_local", (32, 6144), "bfloat16", ("expert", "feature")),
)
DECODED = (
    ("qkv_a_bits", (32, 6144, 82), "uint8", (None, None, None)),
    ("qkv_a_scale", (32, 48, 82), "float32", (None, None, None)),
    ("wq_b_weight_local", (4096, 2048), "float32", ("feature", None)),
    ("wk_weight_bf16", (128, 6144), "bfloat16", (None, None)),
    ("head_weight_local", (32, 6144), "bfloat16", ("feature", None)),
)
PROMOTED = (
    *DECODED[:2],
    *((f"wq_b_weight_aliases/{i}", (4096, 2048), "float32", ("feature", None)) for i in range(4)),
    ("wk_weight", (128, 6144), "float32", (None, None)),
    DECODED[-1],
)


def source_names(layer: int) -> dict[str, str]:
    """Names from the frozen ws32_decoder_weight_names/select raw views."""
    prefix = f"model.layers.{layer}.self_attn"
    paths = ("q_a_proj.weight_bits", "q_a_proj.scale_inv",
             "kv_a_proj_with_mqa.weight_bits", "kv_a_proj_with_mqa.scale_inv",
             "indexer.wq_b.weight_bits", "indexer.wq_b.scale_inv",
             "indexer.wk.weight_bits", "indexer.wk.scale_inv", "indexer.weights_proj.weight")
    return {entry[0]: f"{prefix}.{path}" for entry, path in zip(RAW, paths, strict=True)}


def _index(shape: tuple, axes: tuple, slot: int) -> tuple:
    parts = []
    for size, axis in zip(shape, axes, strict=True):
        if axis is None:
            parts.append(slice(None))
        else:
            coordinate, width = (slot % 4, size // 4) if axis == "feature" else (slot // 4, size // 8)
            parts.append(slice(coordinate * width, (coordinate + 1) * width))
    return tuple(parts)


def _owners(value: Any, spec: tuple, slots: Mapping, record: Mapping, platform: str) -> dict:
    _, shape, dtype, axes = spec
    return owner_capture._owner_shards(value, shape=shape, dtype=dtype,
        index_for_slot=lambda slot: _index(shape, axes, slot), local_slots=slots,
        process_index=record["jax_process_index"], platform=platform)


def _get(layer: Any, field: str) -> Any:
    parts = field.split("/")
    value = getattr(layer, parts[0])
    return value if len(parts) == 1 else value[int(parts[1])]


def _tree(tree: Any, specs: tuple, slots: Mapping, record: Mapping, platform: str) -> dict:
    expected_fields = tuple(dict.fromkeys(spec[0].split("/")[0] for spec in specs))
    if not isinstance(tree, tuple) or len(tree) != len(protocol.PRODUCERS):
        raise ValueError("history exact producer tree differs")
    result = {}
    for layer_id, layer in zip(protocol.PRODUCERS, tree, strict=True):
        if (getattr(layer, "_fields", None) != expected_fields
                or (specs is PROMOTED and (not isinstance(layer.wq_b_weight_aliases, tuple)
                    or len(layer.wq_b_weight_aliases) != 4))):
            raise ValueError("history exact leaf schema differs")
        for spec in specs:
            result[f"layer{layer_id}/{spec[0]}"] = (spec, _owners(_get(layer, spec[0]), spec, slots, record, platform))
    return result


def _directory(root: Path, *, create: bool = True) -> Path:
    directory = root / "materializers"
    if any(path.is_symlink() for path in (root, *root.parents)) or not root.is_dir() or directory.is_symlink():
        raise ValueError("history materializer original path is linked or absent")
    if create:
        directory.mkdir(exist_ok=True)
    elif not directory.is_dir():
        raise ValueError("history materializer original directory is absent")
    for path in directory.iterdir():
        if path.name not in _FILENAMES.values() or path.is_symlink() or not path.is_file():
            raise ValueError("history materializer unexpected original path")
    return directory


def _path(root: Path, key: str, maximum: int) -> Path:
    directory = _directory(root)
    path = directory / _FILENAMES[key]
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"history refuses to replace original: {path.name}")
    current = {p.name: p.stat().st_size for p in directory.iterdir()}
    wk_used = sum(current.get(_FILENAMES[k], 0) for k in _WK_KEYS)
    if ((key in _WK_KEYS and wk_used + maximum > WK_ORIGINALS_LIMIT)
            or (key.startswith("exact_") and maximum > EXACT_RECEIPT_LIMIT)
            or (key == "refused_exact" and maximum > REFUSAL_ORIGINAL_LIMIT)
            or sum(current.values()) + maximum > TOTAL_LIMIT):
        raise ValueError("history materializer originals exceed registered byte budget")
    return path


def _sync(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _descriptor(path: Path, *, raw_bytes: int | None = None) -> dict:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    result = dict(path=f"materializers/{path.name}", bytes=path.stat().st_size,
                  sha256=digest.hexdigest())
    if raw_bytes is not None:
        result["raw_array_bytes"] = raw_bytes
        result["npz_sha256"] = result["sha256"]  # existing bounded NPZ reader interface
    return result


class _CappedWriter:
    def __init__(self, stream: Any, maximum: int):
        self.stream, self.maximum = stream, maximum

    def write(self, value: Any) -> int:
        if self.stream.tell() + len(value) > self.maximum:
            raise ValueError("history materializer writer exceeded its reserved bytes")
        return self.stream.write(value)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.stream, key)


def _npz(root: Path, key: str, arrays: Mapping[str, np.ndarray]) -> dict:
    stored = {name: value.view(np.uint16) if str(value.dtype) == "bfloat16" else value
              for name, value in arrays.items()}
    raw_bytes = sum(value.nbytes for value in stored.values())
    if key == "refused_exact" and raw_bytes > MAX_REFUSAL_RAW_BYTES:
        raise ValueError("history exact refusal pair exceeds registered raw bytes")
    # Uncompressed NPZ: 1024 bytes/member plus footer is above the fixed NPY/ZIP
    # headers here. The capped writer also bounds incomplete terminal originals.
    maximum = raw_bytes + 1024 * (len(stored) + 1)
    path = _path(root, key, maximum)
    with path.open("xb") as stream:
        np.savez(_CappedWriter(stream, maximum), **stored)
        stream.flush()
        os.fsync(stream.fileno())
    _sync(path.parent)
    return _descriptor(path, raw_bytes=raw_bytes)


def _json(root: Path, key: str, report: Mapping) -> dict:
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path = _path(root, key, len(raw))
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _sync(path.parent)
    return _descriptor(path)


def _binding(record: Mapping, slots: Mapping, layer: int, field: str, device: int) -> dict:
    rows = record.get("local_device_slots", ())
    if (not isinstance(rows, (tuple, list)) or len(rows) != len(slots)
            or any(not isinstance(row, Mapping) for row in rows)
            or {(row.get("device_id"), row.get("device_slot")) for row in rows} != set(slots.items())):
        raise ValueError("history materializer selected-source owner ledger differs")
    row = next(row for row in rows if row["device_id"] == device)
    name = source_names(layer)[field]
    digest = row.get("observed_selected_tensor_sha256", {}).get(name)
    file_digest = row.get("expected_full_file_sha256_not_verified")
    if not all(isinstance(v, str) and _HEX.fullmatch(v) for v in (digest, file_digest)):
        raise ValueError("history materializer selected-source SHA binding differs")
    return dict(tensor=name, selected_tensor_sha256=digest,
                expected_full_file_sha256_not_verified=file_digest)


def _leaf(value: np.ndarray, shard: Any, shape: tuple, slot: int) -> dict:
    return dict(device_id=shard.device.id, slot=slot,
        index=[list(part) for part in _index_key(shard.index, shape)],
        shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
        sha256=sha256(value.tobytes()).hexdigest(), finite=bool(np.isfinite(value).all()))


def _identity(record: dict, slots: Mapping, platform: str, key: str) -> None:
    owner_capture._require_owners(slots, record.get("jax_process_index"), platform)
    originals = record.get("materializer_originals", {})
    if (not isinstance(originals, dict) or key in originals
            or set(originals) - set(_FILENAMES)):
        raise ValueError("history materializer original identity differs")


def _base(name: str, slots: Mapping, record: Mapping) -> dict:
    return dict(schema=SCHEMA, name=name, completed=True, valid=False, errors=[],
        process_index=record["jax_process_index"],
        local_slots={str(device): slot for device, slot in sorted(slots.items())},
        numerical_admission=False, legacy_bit_identity=False,
        reconstruction_source="SELECTED_CHECKPOINT_OWNER_BYTES_PLUS_WK_ORIGINALS",
        replica_scope="SAME_GLOBAL_SLICE_ADDRESSABLE_OWNERS_ONLY", leaves={})


def capture_wk(root: Path, record: dict, local_slots: Mapping[int, int], *,
               layer: int, name: str, value: Any, inputs: tuple = (), platform: str = "tpu") -> None:
    """Preservation callback for each of the eight completed WK calls.

    All local source/result arrays are retained, including nonfinite values and
    inconsistent replicas. Full feature reconstruction remains a collector duty.
    """
    if (type(layer) is not int or layer not in protocol.PRODUCERS
            or name not in ("wk_decode", "wk_promote") or not isinstance(inputs, tuple)
            or len(inputs) != (2 if name == "wk_decode" else 1)):
        raise ValueError("history WK role differs")
    key = f"layer{layer}/{name}"
    _identity(record, local_slots, platform, key)
    specs = (("result", (128, 6144), "bfloat16", (None, None)),
             ("input0", *RAW[6][1:]), ("input1", *RAW[7][1:])) if name == "wk_decode" else (
             ("result", (128, 6144), "float32", (None, None)), ("input0", *DECODED[3][1:]))
    values = (value, *inputs)
    owners = [_owners(v, spec, local_slots, record, platform) for v, spec in zip(values, specs, strict=True)]
    bindings = {(role, device): _binding(record, local_slots, layer,
                "wk_bits_local" if role == "input0" else "wk_scale_local", device)
                for role in ("input0", "input1") for device in local_slots} if name == "wk_decode" else {}
    prior = {device: _prior_wk(record, layer, "wk_decode", device) for device in local_slots} if name == "wk_promote" else {}
    report, arrays = _base(key, local_slots, record), {}
    for spec, shards in zip(specs, owners, strict=True):
        role, shape, _, _ = spec
        groups = {}
        report["leaves"][role] = rows = []
        for device in sorted(shards):
            shard = shards[device]
            host = owner_capture._read(shard)
            arrays[f"slot{local_slots[device]}_{role}"] = host
            row = _leaf(host, shard, shape, local_slots[device])
            rows.append(row)
            if (role, device) in bindings:
                row["checkpoint_source"] = bindings[role, device]
                if row["sha256"] != row["checkpoint_source"]["selected_tensor_sha256"]:
                    report["errors"].append(f"{role}/slot{row['slot']}/checkpoint_sha")
            if not row["finite"]:
                report["errors"].append(f"{role}/slot{row['slot']}/nonfinite")
            index = tuple(map(tuple, row["index"]))
            if index in groups and not owner_capture._same_bytes(groups[index], host):
                report["errors"].append(f"{role}/slot{row['slot']}/replica_bytes")
            groups.setdefault(index, host)
    if name == "wk_promote":
        for device, slot in local_slots.items():
            if sha256(arrays[f"slot{slot}_input0"].tobytes()).hexdigest() != prior[device]:
                report["errors"].append(f"input0/slot{slot}/wk_decode_original_bytes")
            if not owner_capture._same_bytes(arrays[f"slot{slot}_input0"].astype(np.float32), arrays[f"slot{slot}_result"]):
                report["errors"].append(f"result/slot{slot}/promotion_bytes")
    report["valid"] = not report["errors"]
    report["original"] = _npz(root, key, arrays)
    record.setdefault("materializer_originals", {})[key] = report
    if report["errors"]:
        raise ValueError("history WK boundary refused; completed originals preserved")


def _prior_wk(record: Mapping, layer: int, name: str, device: int) -> str:
    report = record.get("materializer_originals", {}).get(f"layer{layer}/{name}", {})
    rows = report.get("leaves", {}).get("result", ())
    found = [row for row in rows if row.get("device_id") == device]
    if report.get("valid") is not True or len(found) != 1 or not _HEX.fullmatch(str(found[0].get("sha256"))):
        raise ValueError("history exact WK original linkage missing")
    return found[0]["sha256"]


def capture_exact(root: Path, record: dict, bound: Any, *, name: str, value: Any,
                  inputs: tuple, platform: str = "tpu") -> None:
    """Capture decoded/promoted exact trees without materializing a global leaf.

    Decode takes ``(bound.raw_exact,)``; promote takes ``(decoded_exact,)``.
    The executor keeps those input objects live through this preservation call.
    The two JSON originals carry every source/output owner's hash and geometry.
    Hashes are reconstruction targets, not a substitute for checkpoint replay.
    """
    if (name not in ("exact_decode", "exact_promote") or not isinstance(inputs, tuple)
            or len(inputs) != 1 or (name == "exact_decode" and inputs[0] is not bound.raw_exact)):
        raise ValueError("history exact role/input binding differs")
    slots = bound.local_slots
    _identity(record, slots, platform, name)
    source = _tree(inputs[0], RAW if name == "exact_decode" else DECODED, slots, record, platform)
    output = _tree(value, DECODED if name == "exact_decode" else PROMOTED, slots, record, platform)
    bindings = {(context, device): _binding(record, slots, int(context.split("/")[0][5:]),
                 context.split("/")[1], device) for context in source for device in slots} if name == "exact_decode" else {}
    wk_name = "wk_decode" if name == "exact_decode" else "wk_promote"
    wk_hashes = {(layer, device): _prior_wk(record, layer, wk_name, device)
                 for layer in protocol.PRODUCERS for device in slots}
    decoded_rows = {}
    if name == "exact_promote":
        decoded_receipt = load_exact_receipt(root, record, "exact_decode")
        if decoded_receipt.get("valid") is not True:
            raise ValueError("history exact decode original is refused")
        decoded_rows = {(context, row["device_id"]): row["sha256"]
                        for context, rows in decoded_receipt["leaves"].items() for row in rows}
        if set(decoded_rows) != {(context, device) for context in source for device in slots}:
            raise ValueError("history exact decode original owner inventory differs")
    report = _base(name, slots, record)
    report.update(full_exact_output_originals_preserved=False,
                  exact_decode_reconstruction_checked=False, source_leaves={})
    first_failure = None
    expected = {}

    def retain_failure(context: str, samples: tuple[tuple[Any, np.ndarray], ...], shape: tuple) -> None:
        nonlocal first_failure
        if first_failure is None or (len(samples) == 2 and len(first_failure["samples"]) == 1):
            first_failure = dict(context=context, global_shape=list(shape),
                samples=[_leaf(host, shard, shape, slots[shard.device.id]) for shard, host in samples],
                arrays={f"sample{i}": np.array(host, copy=True, order="C")
                        for i, (_, host) in enumerate(samples)})

    for category, inventory in (("source_leaves", source), ("leaves", output)):
        for context, (spec, shards) in inventory.items():
            _, shape, _, _ = spec
            groups = {}
            report[category][context] = rows = []
            layer = int(context.split("/")[0][5:])
            field = context.split("/", 1)[1]
            for device in sorted(shards):
                shard = shards[device]
                host = owner_capture._read(shard)
                row = _leaf(host, shard, shape, slots[device])
                rows.append(row)
                label = f"{category}/{context}/slot{slots[device]}"
                if not row["finite"]:
                    report["errors"].append(label + "/nonfinite")
                    retain_failure(label, ((shard, host),), shape)
                index = tuple(map(tuple, row["index"]))
                if index in groups and not owner_capture._same_bytes(groups[index][1], host):
                    report["errors"].append(label + "/replica_bytes")
                    retain_failure(label, (groups[index], (shard, host)), shape)
                # One first owner per distinct slice, never all local payloads.
                groups.setdefault(index, (shard, host))
                if category == "source_leaves":
                    if name == "exact_decode":
                        row["checkpoint_source"] = bindings[context, device]
                        if row["sha256"] != row["checkpoint_source"]["selected_tensor_sha256"]:
                            report["errors"].append(label + "/checkpoint_sha")
                            retain_failure(label, ((shard, host),), shape)
                    else:
                        row["exact_decode_original_sha256"] = decoded_rows[context, device]
                        if row["sha256"] != decoded_rows[context, device]:
                            report["errors"].append(label + "/exact_decode_original_bytes")
                            retain_failure(label, ((shard, host),), shape)
                        expected[context, device] = row["sha256"]
                        if field == "wk_weight_bf16":
                            expected[f"layer{layer}/wk_weight", device] = sha256(host.astype(np.float32).tobytes()).hexdigest()
                    continue
                if field in ("wk_weight_bf16", "wk_weight"):
                    row["wk_original_sha256"] = wk_hashes[layer, device]
                    if row["sha256"] != wk_hashes[layer, device]:
                        report["errors"].append(label + "/wk_original_bytes")
                        retain_failure(label, ((shard, host),), shape)
                if name == "exact_promote":
                    input_field = "wq_b_weight_local" if field.startswith("wq_b_weight_aliases/") else field
                    expected_sha = expected[f"layer{layer}/{input_field}", device]
                    row["promotion_input_sha256"] = expected_sha
                    if row["sha256"] != expected_sha:
                        report["errors"].append(label + "/promotion_bytes")
                        retain_failure(label, ((shard, host),), shape)
    if first_failure is not None:
        arrays = first_failure.pop("arrays")
        first_failure["original"] = _npz(root, "refused_exact", arrays)
        report["refused_original"] = first_failure
    report["valid"] = not report["errors"]
    original = _json(root, name, report)
    record.setdefault("materializer_originals", {})[name] = dict(
        schema=SCHEMA, name=name, completed=True, valid=report["valid"],
        errors=report["errors"], original=original,
        **({"refused_original": report["refused_original"]} if first_failure is not None else {}))
    if report["errors"]:
        raise ValueError("history exact boundary refused; bounded completed originals preserved")


def load_exact_receipt(root: Path, record: Mapping, name: str) -> dict:
    """Resolve one SHA-bound receipt; full fleet/semantic replay belongs outside."""
    if name not in ("exact_decode", "exact_promote"):
        raise ValueError("history exact receipt name differs")
    reference = record["materializer_originals"][name]
    original = reference["original"]
    path = root / "materializers" / _FILENAMES[name]
    _directory(root, create=False)
    if (original.get("path") != f"materializers/{_FILENAMES[name]}"
            or type(original.get("bytes")) is not int or not 0 < original["bytes"] <= EXACT_RECEIPT_LIMIT
            or not path.is_file() or path.stat().st_size != original["bytes"]):
        raise ValueError("history exact receipt path/size differs")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != original.get("sha256"):
        raise ValueError("history exact receipt hash differs")
    report = json.loads(raw)
    if (report.get("schema") != SCHEMA or report.get("name") != name
            or report.get("completed") is not True
            or any(report.get(key) != reference.get(key) for key in ("valid", "errors"))):
        raise ValueError("history exact receipt identity differs")
    return report
