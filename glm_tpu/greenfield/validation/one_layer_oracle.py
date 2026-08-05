"""Standalone raw-checkpoint oracle for one GLM sparse MoE layer.

This module deliberately uses PyTorch CPU arithmetic and raw source tensors.
It does not import the greenfield JAX kernels, the packed checkpoint, a model
class, or the legacy execution path.  Exact source and legacy implementation
pins make the accepted-path arithmetic transcription auditable while keeping
the comparison independent of the system under test.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT_VERSION = 1
ARTIFACT_KIND = "greenfield_one_layer_oracle"
MODEL_ID = "zai-org/GLM-5.2-FP8"


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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str, name: str, lengths: tuple[int, ...]) -> None:
    if len(value) not in lengths or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")


def _positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class OneLayerOracleConfig:
    source_root: Path
    output_dir: Path
    source_uri: str
    source_revision: str
    code_hash: str
    legacy_code_hash: str
    legacy_source_hashes: tuple[tuple[str, str], ...]
    vllm_code_hash: str
    vllm_source_hashes: tuple[tuple[str, str], ...]
    allowed_source_shards: tuple[str, ...] = ()
    layer: int = 3
    hidden_size: int = 6144
    intermediate_size: int = 2048
    num_experts: int = 256
    top_k: int = 8
    stage_size: int = 4
    concentrated_slot: int = 2
    min_normal_stage_slots: int = 2
    routed_scaling_factor: float = 2.5
    fp8_block_shape: tuple[int, int] = (128, 128)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "fp8_block_shape", tuple(self.fp8_block_shape))
        object.__setattr__(
            self, "legacy_source_hashes", tuple(self.legacy_source_hashes)
        )
        object.__setattr__(self, "vllm_source_hashes", tuple(self.vllm_source_hashes))
        object.__setattr__(
            self, "allowed_source_shards", tuple(self.allowed_source_shards)
        )
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "stage_size",
            "min_normal_stage_slots",
        ):
            _positive_integer(getattr(self, name), name)
        if not isinstance(self.layer, int) or isinstance(self.layer, bool) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if self.num_experts % self.stage_size:
            raise ValueError("num_experts must divide evenly over stage_size")
        if self.top_k > self.num_experts // self.stage_size:
            raise ValueError("one stage slot must contain at least top_k experts")
        if not 0 <= self.concentrated_slot < self.stage_size:
            raise ValueError("concentrated_slot must identify one stage slot")
        if self.min_normal_stage_slots > self.stage_size:
            raise ValueError("min_normal_stage_slots cannot exceed stage_size")
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        if len(self.fp8_block_shape) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in self.fp8_block_shape
        ):
            raise ValueError("fp8_block_shape must contain two positive integers")
        if not self.source_uri.startswith("gs://driftbench-dsv4-uc/"):
            raise ValueError("source_uri must use the approved driftbench-dsv4-uc bucket")
        if not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        _validate_digest(self.code_hash, "code_hash", (40, 64))
        _validate_digest(self.legacy_code_hash, "legacy_code_hash", (40, 64))
        _validate_digest(self.vllm_code_hash, "vllm_code_hash", (40, 64))
        for collection_name in ("legacy_source_hashes", "vllm_source_hashes"):
            collection = getattr(self, collection_name)
            if not collection or len({name for name, _ in collection}) != len(collection):
                raise ValueError(f"{collection_name} must contain unique source names")
            for source_name, digest in collection:
                if not source_name:
                    raise ValueError(f"{collection_name} contains an empty source name")
                _validate_digest(digest, collection_name, (64,))

    @property
    def layer_prefix(self) -> str:
        return f"model.layers.{self.layer}.mlp"

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.stage_size


def _dtype_name(tensor: Any) -> str:
    names = {
        "torch.bfloat16": "BF16",
        "torch.float32": "F32",
        "torch.float8_e4m3fn": "F8_E4M3",
        "torch.int32": "I32",
    }
    try:
        return names[str(tensor.dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported oracle tensor dtype {tensor.dtype}") from exc


def _tensor_sha256(tensor: Any) -> str:
    import torch

    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    digest = sha256()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "byte_count": tensor.numel() * tensor.element_size(),
        "dtype": _dtype_name(tensor),
        "name": name,
        "sha256": _tensor_sha256(tensor),
        "shape": list(tensor.shape),
    }


def _expected_shape_dtype(
    config: OneLayerOracleConfig, name: str
) -> tuple[tuple[int, ...], str]:
    prefix = config.layer_prefix
    if name == f"{prefix}.gate.weight":
        return (config.num_experts, config.hidden_size), "BF16"
    if name == f"{prefix}.gate.e_score_correction_bias":
        return (config.num_experts,), "F32"
    is_scale = name.endswith(".weight_scale_inv")
    block_out, block_in = config.fp8_block_shape
    if ".down_proj." in name:
        shape = (
            (
                (config.hidden_size + block_out - 1) // block_out,
                (config.intermediate_size + block_in - 1) // block_in,
            )
            if is_scale
            else (config.hidden_size, config.intermediate_size)
        )
    elif ".gate_proj." in name or ".up_proj." in name:
        shape = (
            (
                (config.intermediate_size + block_out - 1) // block_out,
                (config.hidden_size + block_in - 1) // block_in,
            )
            if is_scale
            else (config.intermediate_size, config.hidden_size)
        )
    else:
        raise ValueError(f"unrecognized one-layer oracle tensor {name!r}")
    return shape, "F32" if is_scale else "F8_E4M3"


def _dequantize_fp8(weight: Any, scale: Any, block_shape: tuple[int, int]) -> Any:
    import torch

    expected = tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(weight.shape, block_shape, strict=True)
    )
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"FP8 scale shape {tuple(scale.shape)} does not match {expected}"
        )
    expanded = scale.float().repeat_interleave(block_shape[0], dim=0)
    expanded = expanded.repeat_interleave(block_shape[1], dim=1)
    expanded = expanded[: weight.shape[0], : weight.shape[1]]
    return (weight.float() * expanded).to(torch.bfloat16)


def _stable_topk(scores: Any, top_k: int) -> Any:
    import torch

    if tuple(scores.shape)[0] != 1:
        raise ValueError("oracle routing supports exactly one live token")
    order = sorted(
        range(scores.shape[1]),
        key=lambda expert: (-float(scores[0, expert]), expert),
    )[:top_k]
    return torch.tensor([order], dtype=torch.int32)


def _route(logits: Any, correction_bias: Any, top_k: int) -> tuple[Any, Any]:
    import torch

    scores = torch.sigmoid(logits.float())
    indices = _stable_topk(scores + correction_bias.float()[None, :], top_k)
    weights = torch.gather(scores, 1, indices.long())
    weights = weights / weights.sum(dim=-1, keepdim=True, dtype=torch.float32)
    return indices, weights.float()


def _deterministic_hidden(hidden_size: int) -> Any:
    import torch

    values = (torch.arange(hidden_size, dtype=torch.int64) * 37 + 11) % 257
    return ((values.float() - 128.0) / 128.0).to(torch.bfloat16)[None, :]


def capture_one_layer_oracle(config: OneLayerOracleConfig) -> dict[str, Any]:
    """Capture two independent raw-source layer-3 oracle cases append-only."""

    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open
    from safetensors.torch import save_file

    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only one-layer oracle destination exists: {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True)
    index_path = config.source_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("source index has no weight_map")

    source_records: dict[str, dict[str, Any]] = {}
    handles: dict[str, Any] = {}
    allowed = set(config.allowed_source_shards)

    with ExitStack() as stack:

        def load(name: str) -> Any:
            filename = weight_map.get(name)
            if not isinstance(filename, str):
                raise ValueError(f"source index has no tensor {name!r}")
            if allowed and filename not in allowed:
                raise ValueError(f"oracle tensor {name!r} escaped allowed shards: {filename}")
            if filename not in handles:
                path = config.source_root / filename
                if not path.is_file():
                    raise FileNotFoundError(f"missing source shard {path}")
                handles[filename] = stack.enter_context(
                    safe_open(path, framework="pt", device="cpu")
                )
            tensor = handles[filename].get_tensor(name)
            expected_shape, expected_dtype = _expected_shape_dtype(config, name)
            if tuple(tensor.shape) != expected_shape or _dtype_name(tensor) != expected_dtype:
                raise ValueError(
                    f"source {name!r} expected {expected_dtype}{expected_shape}, "
                    f"got {_dtype_name(tensor)}{tuple(tensor.shape)}"
                )
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor.float()).all()):
                raise ValueError(f"source tensor {name!r} contains non-finite values")
            record = _tensor_record(name, tensor)
            record["source_shard"] = filename
            prior = source_records.setdefault(name, record)
            if prior != record:
                raise ValueError(f"source tensor {name!r} changed during capture")
            return tensor

        prefix = config.layer_prefix
        router_weight = load(f"{prefix}.gate.weight")
        correction_bias = load(f"{prefix}.gate.e_score_correction_bias")
        hidden = _deterministic_hidden(config.hidden_size)
        router_logits = hidden.float() @ router_weight.float().T
        normal_indices, normal_weights = _route(
            router_logits, correction_bias, config.top_k
        )
        normal_slots = sorted(
            {int(value) // config.local_experts for value in normal_indices[0]}
        )
        if len(normal_slots) < config.min_normal_stage_slots:
            raise ValueError(
                "normal routing does not span the required number of stage slots: "
                f"required={config.min_normal_stage_slots} actual={normal_slots}"
            )

        concentrated_bias = correction_bias.clone()
        concentrated_bias -= 1_000_000.0
        expert_start = config.concentrated_slot * config.local_experts
        expert_end = expert_start + config.top_k
        concentrated_bias[expert_start:expert_end] += 2_000_000.0
        concentrated_indices, concentrated_weights = _route(
            router_logits, concentrated_bias, config.top_k
        )
        concentrated_ids = [int(value) for value in concentrated_indices[0]]
        if not all(expert_start <= value < expert_end for value in concentrated_ids):
            raise ValueError("adversarial routing did not remain on one stage chip")

        def load_dequantized(base: str) -> Any:
            weight = load(f"{base}.weight")
            scale = load(f"{base}.weight_scale_inv")
            return _dequantize_fp8(weight, scale, config.fp8_block_shape)

        def expert_output(expert: int) -> Any:
            base = f"{prefix}.experts.{expert}"
            gate_weight = load_dequantized(f"{base}.gate_proj")
            gate = hidden @ gate_weight.T
            del gate_weight
            up_weight = load_dequantized(f"{base}.up_proj")
            up = hidden @ up_weight.T
            del up_weight
            activated = functional.silu(gate) * up
            down_weight = load_dequantized(f"{base}.down_proj")
            result = activated @ down_weight.T
            return result.to(torch.bfloat16)

        selected = sorted(
            set(int(value) for value in normal_indices[0])
            | set(concentrated_ids)
        )
        outputs_by_expert = {expert: expert_output(expert) for expert in selected}

        shared_base = f"{prefix}.shared_experts"
        shared_gate_weight = load_dequantized(f"{shared_base}.gate_proj")
        shared_gate = hidden @ shared_gate_weight.T
        del shared_gate_weight
        shared_up_weight = load_dequantized(f"{shared_base}.up_proj")
        shared_up = hidden @ shared_up_weight.T
        del shared_up_weight
        shared_activated = functional.silu(shared_gate) * shared_up
        shared_down_weight = load_dequantized(f"{shared_base}.down_proj")
        shared_output = (shared_activated @ shared_down_weight.T).to(torch.bfloat16)

    def combine(indices: Any, weights: Any) -> tuple[Any, Any, Any]:
        ordered = torch.cat(
            [outputs_by_expert[int(expert)] for expert in indices[0]], dim=0
        )
        weighted = ordered * weights[0].to(torch.bfloat16)[:, None]
        routed = weighted.sum(dim=0, dtype=torch.bfloat16)[None, :]
        scale = torch.tensor(config.routed_scaling_factor, dtype=torch.bfloat16)
        final = (routed * scale + shared_output).to(torch.bfloat16)
        return ordered, routed, final

    normal_experts, normal_routed, normal_output = combine(
        normal_indices, normal_weights
    )
    concentrated_experts, concentrated_routed, concentrated_output = combine(
        concentrated_indices, concentrated_weights
    )
    tensors = {
        "concentrated_correction_bias": concentrated_bias,
        "concentrated_expert_outputs": concentrated_experts,
        "concentrated_output": concentrated_output,
        "concentrated_route_indices": concentrated_indices,
        "concentrated_route_weights": concentrated_weights,
        "concentrated_routed_unscaled": concentrated_routed,
        "correction_bias": correction_bias,
        "hidden_states": hidden,
        "normal_expert_outputs": normal_experts,
        "normal_output": normal_output,
        "normal_route_indices": normal_indices,
        "normal_route_weights": normal_weights,
        "normal_routed_unscaled": normal_routed,
        "router_logits": router_logits,
        "shared_output": shared_output,
    }
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"oracle output {name!r} contains non-finite values")

    filename = "oracle.safetensors"
    output_path = config.output_dir / filename
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    save_file(
        tensors,
        partial,
        metadata={
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
            "layer": str(config.layer),
            "model_id": MODEL_ID,
        },
    )
    partial.replace(output_path)
    output_records = sorted(
        (_tensor_record(name, tensor) for name, tensor in tensors.items()),
        key=lambda record: record["name"],
    )
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "cases": {
            "concentrated": {
                "expert_end_exclusive": expert_end,
                "expert_start": expert_start,
                "route_indices": concentrated_ids,
                "stage_slot": config.concentrated_slot,
            },
            "normal": {
                "route_indices": [int(value) for value in normal_indices[0]],
                "stage_slots": normal_slots,
            },
        },
        "code_hash": config.code_hash,
        "file": {
            "byte_count": output_path.stat().st_size,
            "filename": filename,
            "sha256": _sha256_file(output_path),
            "tensors": output_records,
        },
        "format_version": FORMAT_VERSION,
        "geometry": {
            "fp8_block_shape": list(config.fp8_block_shape),
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_experts": config.num_experts,
            "routed_scaling_factor": config.routed_scaling_factor,
            "stage_size": config.stage_size,
            "top_k": config.top_k,
        },
        "input_generator": {
            "formula": "bf16((((i*37+11)%257)-128)/128)",
            "live_rows": 1,
        },
        "layer": config.layer,
        "legacy_code_hash": config.legacy_code_hash,
        "legacy_source_hashes": dict(sorted(config.legacy_source_hashes)),
        "model_id": MODEL_ID,
        "source_index_sha256": _sha256_file(index_path),
        "source_revision": config.source_revision,
        "source_tensors": sorted(source_records.values(), key=lambda item: item["name"]),
        "source_uri": config.source_uri.rstrip("/"),
        "vllm_code_hash": config.vllm_code_hash,
        "vllm_source_hashes": dict(sorted(config.vllm_source_hashes)),
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    inspect_one_layer_oracle(config.output_dir)
    return manifest


def inspect_one_layer_oracle(output_dir: Path) -> dict[str, Any]:
    """Verify the standalone oracle file and manifest without source weights."""

    from safetensors import safe_open

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported one-layer oracle format")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("not a greenfield one-layer oracle")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("one-layer oracle manifest checksum mismatch")
    file_record = manifest["file"]
    path = output_dir / file_record["filename"]
    if path.stat().st_size != file_record["byte_count"]:
        raise ValueError("one-layer oracle file size mismatch")
    if _sha256_file(path) != file_record["sha256"]:
        raise ValueError("one-layer oracle file checksum mismatch")
    expected = {record["name"]: record for record in file_record["tensors"]}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("one-layer oracle tensor key mismatch")
        for name, record in expected.items():
            tensor_slice = handle.get_slice(name)
            if (
                list(tensor_slice.get_shape()) != record["shape"]
                or tensor_slice.get_dtype() != record["dtype"]
            ):
                raise ValueError(f"one-layer oracle tensor metadata mismatch: {name}")
    return manifest
