"""One-layer-only GLM MoE checkpoint artifact.

This is the bounded precursor to the full plan-aware checkpoint format.  It
reads only the source safetensor files containing one sparse layer, assigns
complete routed experts contiguously to the chips of one stage, shards the
shared expert's intermediate dimension over those chips, and replicates only
the small router state.  It never imports or constructs a model.

The artifact is written append-only: the destination must not exist and the
manifest is committed last.  Every source leaf, destination tensor, byte
count, dtype, shape, owner, and SHA-256 is recorded and re-inspectable without
loading tensor payloads into memory.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT_VERSION = 1
ARTIFACT_KIND = "greenfield_one_layer_moe"
MODEL_ID = "zai-org/GLM-5.2-FP8"
PLAN_ID = "PP8_LP4"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _manifest_hash(value: Mapping[str, Any]) -> str:
    without_hash = dict(value)
    without_hash.pop("manifest_sha256", None)
    return sha256(_canonical_json(without_hash).encode("utf-8")).hexdigest()


def _positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class OneLayerPackConfig:
    source_root: Path
    output_dir: Path
    source_uri: str
    source_revision: str
    code_hash: str
    topology_hash: str
    plan_group_hash: str
    layer: int = 3
    hidden_size: int = 6144
    intermediate_size: int = 2048
    num_experts: int = 256
    top_k: int = 8
    stage_size: int = 4
    fp8_block_shape: tuple[int, int] = (128, 128)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "fp8_block_shape", tuple(self.fp8_block_shape))
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "stage_size",
        ):
            _positive_integer(getattr(self, name), name)
        if not isinstance(self.layer, int) or isinstance(self.layer, bool) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if self.num_experts % self.stage_size:
            raise ValueError("num_experts must divide evenly over the stage")
        if self.intermediate_size % self.stage_size:
            raise ValueError("intermediate_size must divide evenly over the stage")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if len(self.fp8_block_shape) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in self.fp8_block_shape
        ):
            raise ValueError("fp8_block_shape must contain two positive integers")
        if any(self.hidden_size % block for block in self.fp8_block_shape):
            raise ValueError("hidden_size must be divisible by both FP8 block dimensions")
        if any(self.intermediate_size % block for block in self.fp8_block_shape):
            raise ValueError(
                "intermediate_size must be divisible by both FP8 block dimensions"
            )
        if not self.source_uri.startswith("gs://driftbench-dsv4-uc/"):
            raise ValueError("source_uri must use the approved driftbench-dsv4-uc bucket")
        if not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        for name in ("code_hash", "topology_hash", "plan_group_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.stage_size

    @property
    def local_intermediate(self) -> int:
        return self.intermediate_size // self.stage_size

    @property
    def layer_prefix(self) -> str:
        return f"model.layers.{self.layer}.mlp"


def expected_source_names(config: OneLayerPackConfig) -> tuple[str, ...]:
    prefix = config.layer_prefix
    names = []
    for expert in range(config.num_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"{prefix}.experts.{expert}.{projection}"
            names.extend((f"{base}.weight", f"{base}.weight_scale_inv"))
    names.extend(
        (
            f"{prefix}.gate.weight",
            f"{prefix}.gate.e_score_correction_bias",
        )
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"{prefix}.shared_experts.{projection}"
        names.extend((f"{base}.weight", f"{base}.weight_scale_inv"))
    return tuple(sorted(names))


def _source_shape_dtype(
    config: OneLayerPackConfig, source_name: str
) -> tuple[tuple[int, ...], str]:
    block_out, block_in = config.fp8_block_shape
    prefix = config.layer_prefix
    if source_name == f"{prefix}.gate.weight":
        return (config.num_experts, config.hidden_size), "BF16"
    if source_name == f"{prefix}.gate.e_score_correction_bias":
        return (config.num_experts,), "F32"
    is_scale = source_name.endswith(".weight_scale_inv")
    if ".down_proj." in source_name:
        shape = (
            (config.hidden_size // block_out, config.intermediate_size // block_in)
            if is_scale
            else (config.hidden_size, config.intermediate_size)
        )
    elif ".gate_proj." in source_name or ".up_proj." in source_name:
        shape = (
            (config.intermediate_size // block_out, config.hidden_size // block_in)
            if is_scale
            else (config.intermediate_size, config.hidden_size)
        )
    else:
        raise ValueError(f"unrecognized one-layer source tensor {source_name!r}")
    return shape, "F32" if is_scale else "F8_E4M3"


def _tensor_nbytes(shape: Sequence[int], dtype: str) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    item_sizes = {"BF16": 2, "F32": 4, "F8_E4M3": 1}
    try:
        return elements * item_sizes[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported manifest dtype {dtype!r}") from exc


def _load_source_index(config: OneLayerPackConfig) -> tuple[dict[str, str], str]:
    path = config.source_root / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing source index {path}")
    value = json.loads(path.read_text())
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise ValueError("source index has no valid weight_map")
    expected = set(expected_source_names(config))
    actual = {name for name in weight_map if name.startswith(config.layer_prefix + ".")}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "one-layer source leaf set mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return weight_map, _sha256_file(path)


def _open_sources(
    stack: ExitStack,
    config: OneLayerPackConfig,
    weight_map: Mapping[str, str],
):
    from safetensors import safe_open

    filenames = sorted({weight_map[name] for name in expected_source_names(config)})
    handles = {}
    for filename in filenames:
        path = config.source_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing source checkpoint shard {path}")
        handles[filename] = stack.enter_context(
            safe_open(path, framework="pt", device="cpu")
        )
    return handles


def _validate_source_leaves(
    config: OneLayerPackConfig,
    weight_map: Mapping[str, str],
    handles: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    leaves = []
    total_bytes = 0
    for name in expected_source_names(config):
        expected_shape, expected_dtype = _source_shape_dtype(config, name)
        filename = weight_map[name]
        handle = handles[filename]
        if name not in handle.keys():
            raise ValueError(f"index points to absent tensor {name!r} in {filename}")
        tensor_slice = handle.get_slice(name)
        shape = tuple(tensor_slice.get_shape())
        dtype = tensor_slice.get_dtype()
        if shape != expected_shape or dtype != expected_dtype:
            raise ValueError(
                f"source tensor {name!r} expected {expected_dtype}{expected_shape}, "
                f"got {dtype}{shape}"
            )
        byte_count = _tensor_nbytes(shape, dtype)
        total_bytes += byte_count
        leaves.append(
            {
                "byte_count": byte_count,
                "dtype": dtype,
                "name": name,
                "shape": list(shape),
                "source_shard": filename,
            }
        )
    return leaves, total_bytes


def _load_checked_tensor(
    handles: Mapping[str, Any],
    weight_map: Mapping[str, str],
    name: str,
):
    import torch

    tensor = handles[weight_map[name]].get_tensor(name)
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"source tensor {name!r} contains non-finite values")
    return tensor


def _stack_experts(
    config: OneLayerPackConfig,
    slot: int,
    projection: str,
    suffix: str,
    handles: Mapping[str, Any],
    weight_map: Mapping[str, str],
):
    import torch

    expert_start = slot * config.local_experts
    names = [
        f"{config.layer_prefix}.experts.{expert}.{projection}.{suffix}"
        for expert in range(expert_start, expert_start + config.local_experts)
    ]
    first = _load_checked_tensor(handles, weight_map, names[0])
    destination = torch.empty(
        (config.local_experts, *first.shape), dtype=first.dtype, device="cpu"
    )
    destination[0].copy_(first)
    for local_index, name in enumerate(names[1:], start=1):
        source = _load_checked_tensor(handles, weight_map, name)
        if source.shape != first.shape or source.dtype != first.dtype:
            raise ValueError(f"expert stack member {name!r} changed shape or dtype")
        destination[local_index].copy_(source)
    return destination, names


def _destination_record(
    name: str,
    tensor: Any,
    source_names: Sequence[str],
    ownership: Mapping[str, Any],
) -> dict[str, Any]:
    dtype_names = {
        "torch.bfloat16": "BF16",
        "torch.float32": "F32",
        "torch.float8_e4m3fn": "F8_E4M3",
    }
    dtype = dtype_names.get(str(tensor.dtype))
    if dtype is None:
        raise ValueError(f"unsupported packed torch dtype {tensor.dtype}")
    shape = tuple(tensor.shape)
    return {
        "byte_count": _tensor_nbytes(shape, dtype),
        "dtype": dtype,
        "name": name,
        "ownership": dict(ownership),
        "shape": list(shape),
        "source_names": list(source_names),
    }


def _pack_slot(
    config: OneLayerPackConfig,
    slot: int,
    handles: Mapping[str, Any],
    weight_map: Mapping[str, str],
    source_index_sha256: str,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    prefix = config.layer_prefix
    tensors = {}
    records = []
    expert_start = slot * config.local_experts
    expert_end = expert_start + config.local_experts
    expert_ownership = {
        "device_slot": slot,
        "expert_end_exclusive": expert_end,
        "expert_start": expert_start,
        "kind": "complete_experts",
    }
    for projection, output_name in (
        ("gate_proj", "expert_gate"),
        ("up_proj", "expert_up"),
        ("down_proj", "expert_down"),
    ):
        weight, weight_sources = _stack_experts(
            config, slot, projection, "weight", handles, weight_map
        )
        scale, scale_sources = _stack_experts(
            config, slot, projection, "weight_scale_inv", handles, weight_map
        )
        tensors[output_name] = weight
        tensors[f"{output_name}_scale"] = scale
        records.append(
            _destination_record(output_name, weight, weight_sources, expert_ownership)
        )
        records.append(
            _destination_record(
                f"{output_name}_scale", scale, scale_sources, expert_ownership
            )
        )

    intermediate_start = slot * config.local_intermediate
    intermediate_end = intermediate_start + config.local_intermediate
    output_block_start = intermediate_start // config.fp8_block_shape[0]
    output_block_end = intermediate_end // config.fp8_block_shape[0]
    input_block_start = intermediate_start // config.fp8_block_shape[1]
    input_block_end = intermediate_end // config.fp8_block_shape[1]
    shared_ownership = {
        "device_slot": slot,
        "intermediate_end_exclusive": intermediate_end,
        "intermediate_start": intermediate_start,
        "kind": "shared_intermediate_shard",
    }
    for projection, output_name in (
        ("gate_proj", "shared_gate"),
        ("up_proj", "shared_up"),
        ("down_proj", "shared_down"),
    ):
        weight_name = f"{prefix}.shared_experts.{projection}.weight"
        scale_name = f"{weight_name}_scale_inv"
        weight = _load_checked_tensor(handles, weight_map, weight_name)
        scale = _load_checked_tensor(handles, weight_map, scale_name)
        if projection == "down_proj":
            weight = weight[:, intermediate_start:intermediate_end].contiguous()
            scale = scale[:, input_block_start:input_block_end].contiguous()
        else:
            weight = weight[intermediate_start:intermediate_end, :].contiguous()
            scale = scale[output_block_start:output_block_end, :].contiguous()
        tensors[output_name] = weight
        tensors[f"{output_name}_scale"] = scale
        records.append(
            _destination_record(output_name, weight, (weight_name,), shared_ownership)
        )
        records.append(
            _destination_record(
                f"{output_name}_scale", scale, (scale_name,), shared_ownership
            )
        )

    replicated_ownership = {"device_slot": slot, "kind": "stage_replicated"}
    for source_name, output_name in (
        (f"{prefix}.gate.weight", "router_weight"),
        (f"{prefix}.gate.e_score_correction_bias", "correction_bias"),
    ):
        tensor = _load_checked_tensor(handles, weight_map, source_name).contiguous()
        tensors[output_name] = tensor
        records.append(
            _destination_record(
                output_name, tensor, (source_name,), replicated_ownership
            )
        )

    filename = f"device_slot_{slot:02d}.safetensors"
    path = config.output_dir / filename
    partial = path.with_suffix(path.suffix + ".partial")
    save_file(
        tensors,
        partial,
        metadata={
            "artifact_kind": ARTIFACT_KIND,
            "device_slot": str(slot),
            "format_version": str(FORMAT_VERSION),
            "layer": str(config.layer),
            "model_id": MODEL_ID,
            "plan_id": PLAN_ID,
            "source_index_sha256": source_index_sha256,
        },
    )
    partial.replace(path)
    stat = path.stat()
    result = {
        "device_slot": slot,
        "file_byte_count": stat.st_size,
        "filename": filename,
        "payload_byte_count": sum(record["byte_count"] for record in records),
        "sha256": _sha256_file(path),
        "tensors": sorted(records, key=lambda record: record["name"]),
    }
    del tensors
    gc.collect()
    return result


def pack_one_layer_moe(config: OneLayerPackConfig) -> dict[str, Any]:
    """Write and fully inspect one real layer in final PP8 stage ownership."""

    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only one-layer destination already exists: {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True)
    weight_map, source_index_sha256 = _load_source_index(config)
    with ExitStack() as stack:
        handles = _open_sources(stack, config, weight_map)
        source_leaves, source_payload_bytes = _validate_source_leaves(
            config, weight_map, handles
        )
        source_shards = []
        for filename in sorted(handles):
            path = config.source_root / filename
            source_shards.append(
                {"byte_count": path.stat().st_size, "filename": filename}
            )
        files = [
            _pack_slot(
                config,
                slot,
                handles,
                weight_map,
                source_index_sha256,
            )
            for slot in range(config.stage_size)
        ]

    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "code_hash": config.code_hash,
        "files": files,
        "format_version": FORMAT_VERSION,
        "geometry": {
            "fp8_block_shape": list(config.fp8_block_shape),
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_experts": config.num_experts,
            "stage_size": config.stage_size,
            "top_k": config.top_k,
        },
        "layer": config.layer,
        "model_id": MODEL_ID,
        "packed_payload_byte_count": sum(
            file_record["payload_byte_count"] for file_record in files
        ),
        "plan_group_hash": config.plan_group_hash,
        "plan_id": PLAN_ID,
        "source_index_sha256": source_index_sha256,
        "source_leaves": source_leaves,
        "source_payload_byte_count": source_payload_bytes,
        "source_revision": config.source_revision,
        "source_shards": source_shards,
        "source_uri": config.source_uri.rstrip("/"),
        "topology_hash": config.topology_hash,
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    inspect_one_layer_artifact(config.output_dir)
    return manifest


def inspect_one_layer_artifact(output_dir: Path) -> dict[str, Any]:
    """Verify hashes and safetensor metadata without reading tensor payloads."""

    from safetensors import safe_open

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing one-layer manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported one-layer manifest version")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("not a greenfield one-layer MoE artifact")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("one-layer manifest checksum mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != manifest["geometry"]["stage_size"]:
        raise ValueError("one-layer manifest has incomplete device files")
    payload_total = 0
    for file_record in files:
        path = output_dir / file_record["filename"]
        if path.stat().st_size != file_record["file_byte_count"]:
            raise ValueError(f"packed file size mismatch for {path.name}")
        if _sha256_file(path) != file_record["sha256"]:
            raise ValueError(f"packed file checksum mismatch for {path.name}")
        expected = {record["name"]: record for record in file_record["tensors"]}
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected):
                raise ValueError(f"packed tensor key mismatch for {path.name}")
            for name, record in expected.items():
                tensor_slice = handle.get_slice(name)
                if (
                    list(tensor_slice.get_shape()) != record["shape"]
                    or tensor_slice.get_dtype() != record["dtype"]
                ):
                    raise ValueError(
                        f"packed tensor metadata mismatch for {path.name}:{name}"
                    )
                if _tensor_nbytes(record["shape"], record["dtype"]) != record["byte_count"]:
                    raise ValueError(
                        f"packed tensor byte mismatch for {path.name}:{name}"
                    )
        payload = sum(record["byte_count"] for record in expected.values())
        if payload != file_record["payload_byte_count"]:
            raise ValueError(f"packed payload total mismatch for {path.name}")
        payload_total += payload
    if payload_total != manifest["packed_payload_byte_count"]:
        raise ValueError("manifest packed payload total mismatch")
    source_total = sum(record["byte_count"] for record in manifest["source_leaves"])
    if source_total != manifest["source_payload_byte_count"]:
        raise ValueError("manifest source payload total mismatch")
    return manifest
