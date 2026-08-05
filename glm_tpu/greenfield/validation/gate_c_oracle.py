"""Independent raw-checkpoint oracle for Gate C dense/DSA/IndexShare cases.

The artifact producer deliberately uses only PyTorch CPU arithmetic and raw
safetensor leaves.  It never imports a greenfield JAX kernel, packed weights,
a model class, or the legacy execution path.  The captured tensors cover a
real dense MLP, a full DSA producer, and the next layer's IndexShare reuse
attention while keeping the 2,048-position payload score ordered.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


FORMAT_VERSION = 1
ARTIFACT_KIND = "greenfield_gate_c_oracle"
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
class GateCOracleConfig:
    """Immutable raw-source capture contract for the three missing cases."""

    source_root: Path
    output_dir: Path
    source_uri: str
    source_revision: str
    code_hash: str
    legacy_code_hash: str
    vllm_code_hash: str
    reference_source_hashes: tuple[tuple[str, str], ...]
    allowed_source_shards: tuple[str, ...]
    producer_layer: int = 2
    consumer_layer: int = 3
    hidden_size: int = 6144
    dense_intermediate_size: int = 12288
    q_lora_rank: int = 2048
    num_attention_heads: int = 64
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    kv_lora_rank: int = 512
    v_head_dim: int = 256
    indexer_heads: int = 32
    indexer_head_dim: int = 128
    indexer_rotary_dim: int = 64
    top_k: int = 2048
    context_length: int = 2304
    packed_cache_width: int = 640
    fp8_block_shape: tuple[int, int] = (128, 128)
    rms_norm_epsilon: float = 1e-5
    indexer_norm_epsilon: float = 1e-6
    rope_theta: float = 8_000_000.0
    require_current_selected: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self, "reference_source_hashes", tuple(self.reference_source_hashes)
        )
        object.__setattr__(
            self, "allowed_source_shards", tuple(self.allowed_source_shards)
        )
        object.__setattr__(self, "fp8_block_shape", tuple(self.fp8_block_shape))
        for name in (
            "hidden_size",
            "dense_intermediate_size",
            "q_lora_rank",
            "num_attention_heads",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "qk_head_dim",
            "kv_lora_rank",
            "v_head_dim",
            "indexer_heads",
            "indexer_head_dim",
            "indexer_rotary_dim",
            "top_k",
            "context_length",
            "packed_cache_width",
        ):
            _positive_integer(getattr(self, name), name)
        for name in ("producer_layer", "consumer_layer"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.consumer_layer != self.producer_layer + 1:
            raise ValueError("the IndexShare consumer must immediately follow producer")
        if self.qk_nope_head_dim + self.qk_rope_head_dim != self.qk_head_dim:
            raise ValueError("qk_head_dim must equal nope plus RoPE widths")
        if self.indexer_rotary_dim > self.indexer_head_dim:
            raise ValueError("indexer rotary width exceeds its head width")
        if self.indexer_rotary_dim % 2 or self.qk_rope_head_dim % 2:
            raise ValueError("rotary widths must be even")
        if self.top_k >= self.context_length:
            raise ValueError(
                "real DSA oracle requires context_length greater than top_k"
            )
        if self.kv_lora_rank + self.qk_rope_head_dim > self.packed_cache_width:
            raise ValueError("packed cache width truncates latent or RoPE state")
        if len(self.fp8_block_shape) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in self.fp8_block_shape
        ):
            raise ValueError("fp8_block_shape must contain two positive integers")
        if any(
            isinstance(value, bool) or value <= 0
            for value in (
                self.rms_norm_epsilon,
                self.indexer_norm_epsilon,
                self.rope_theta,
            )
        ):
            raise ValueError("normalization epsilons and RoPE theta must be positive")
        if not isinstance(self.require_current_selected, bool):
            raise ValueError("require_current_selected must be boolean")
        if not self.source_uri.startswith("gs://driftbench-dsv4-uc/"):
            raise ValueError(
                "source_uri must use the approved driftbench-dsv4-uc bucket"
            )
        if not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if not self.allowed_source_shards:
            raise ValueError("allowed_source_shards must be non-empty")
        _validate_digest(self.code_hash, "code_hash", (40, 64))
        _validate_digest(self.legacy_code_hash, "legacy_code_hash", (40, 64))
        _validate_digest(self.vllm_code_hash, "vllm_code_hash", (40, 64))
        if not self.reference_source_hashes or len(
            {name for name, _ in self.reference_source_hashes}
        ) != len(self.reference_source_hashes):
            raise ValueError("reference_source_hashes must contain unique names")
        for source_name, digest in self.reference_source_hashes:
            if not source_name:
                raise ValueError("reference_source_hashes contains an empty name")
            _validate_digest(digest, "reference_source_hashes", (64,))

    @property
    def producer_prefix(self) -> str:
        return f"model.layers.{self.producer_layer}"

    @property
    def consumer_prefix(self) -> str:
        return f"model.layers.{self.consumer_layer}"


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
        raise ValueError(f"unsupported Gate C oracle dtype {tensor.dtype}") from exc


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


def _scale_shape(
    shape: tuple[int, int], block_shape: tuple[int, int]
) -> tuple[int, int]:
    return tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(shape, block_shape, strict=True)
    )


def _expected_source_tensors(
    config: GateCOracleConfig,
) -> dict[str, tuple[tuple[int, ...], str]]:
    producer = config.producer_prefix
    consumer = config.consumer_prefix
    result: dict[str, tuple[tuple[int, ...], str]] = {}

    def bf16(name: str, shape: tuple[int, ...]) -> None:
        result[name] = (shape, "BF16")

    def fp8_pair(name: str, shape: tuple[int, int]) -> None:
        result[f"{name}.weight"] = (shape, "F8_E4M3")
        result[f"{name}.weight_scale_inv"] = (
            _scale_shape(shape, config.fp8_block_shape),
            "F32",
        )

    bf16(f"{producer}.input_layernorm.weight", (config.hidden_size,))
    bf16(f"{producer}.post_attention_layernorm.weight", (config.hidden_size,))
    fp8_pair(
        f"{producer}.mlp.gate_proj",
        (config.dense_intermediate_size, config.hidden_size),
    )
    fp8_pair(
        f"{producer}.mlp.up_proj",
        (config.dense_intermediate_size, config.hidden_size),
    )
    fp8_pair(
        f"{producer}.mlp.down_proj",
        (config.hidden_size, config.dense_intermediate_size),
    )
    fp8_pair(
        f"{producer}.self_attn.q_a_proj",
        (config.q_lora_rank, config.hidden_size),
    )
    bf16(
        f"{producer}.self_attn.q_a_layernorm.weight",
        (config.q_lora_rank,),
    )
    indexer = f"{producer}.self_attn.indexer"
    fp8_pair(
        f"{indexer}.wq_b",
        (config.indexer_heads * config.indexer_head_dim, config.q_lora_rank),
    )
    fp8_pair(f"{indexer}.wk", (config.indexer_head_dim, config.hidden_size))
    bf16(f"{indexer}.k_norm.weight", (config.indexer_head_dim,))
    bf16(f"{indexer}.k_norm.bias", (config.indexer_head_dim,))
    bf16(f"{indexer}.weights_proj.weight", (config.indexer_heads, config.hidden_size))

    bf16(f"{consumer}.input_layernorm.weight", (config.hidden_size,))
    attention = f"{consumer}.self_attn"
    fp8_pair(f"{attention}.q_a_proj", (config.q_lora_rank, config.hidden_size))
    bf16(f"{attention}.q_a_layernorm.weight", (config.q_lora_rank,))
    fp8_pair(
        f"{attention}.q_b_proj",
        (
            config.num_attention_heads * config.qk_head_dim,
            config.q_lora_rank,
        ),
    )
    fp8_pair(
        f"{attention}.kv_a_proj_with_mqa",
        (
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
        ),
    )
    bf16(f"{attention}.kv_a_layernorm.weight", (config.kv_lora_rank,))
    fp8_pair(
        f"{attention}.kv_b_proj",
        (
            config.num_attention_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
        ),
    )
    fp8_pair(
        f"{attention}.o_proj",
        (config.hidden_size, config.num_attention_heads * config.v_head_dim),
    )
    return result


def _dequantize_fp8(weight: Any, scale: Any, block_shape: tuple[int, int]) -> Any:
    import torch

    expected = _scale_shape(tuple(weight.shape), block_shape)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"FP8 scale shape {tuple(scale.shape)} does not match {expected}"
        )
    expanded = scale.float().repeat_interleave(block_shape[0], dim=0)
    expanded = expanded.repeat_interleave(block_shape[1], dim=1)
    expanded = expanded[: weight.shape[0], : weight.shape[1]]
    return (weight.float() * expanded).to(torch.bfloat16)


def _deterministic_bf16(rows: int, width: int, salt: int) -> Any:
    import torch

    row = torch.arange(rows, dtype=torch.int64)[:, None]
    column = torch.arange(width, dtype=torch.int64)[None, :]
    values = ((row * 37 + column * 17 + salt) % 257).float()
    return ((values - 128.0) / 128.0).to(torch.bfloat16)


def _rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    import torch

    value_f32 = value.float()
    variance = torch.mean(value_f32 * value_f32, dim=-1, keepdim=True)
    normalized = value_f32 * torch.rsqrt(variance + epsilon)
    return normalized.to(value.dtype) * weight.to(value.dtype)


def _layer_norm(value: Any, weight: Any, bias: Any, epsilon: float) -> Any:
    import torch

    value_f32 = value.float()
    mean = torch.mean(value_f32, dim=-1, keepdim=True)
    variance = torch.mean((value_f32 - mean) ** 2, dim=-1, keepdim=True)
    return (
        (value_f32 - mean) * torch.rsqrt(variance + epsilon)
    ) * weight.float() + bias.float()


def _rotary_interleaved(
    value: Any,
    positions: Any,
    *,
    rotary_dim: int,
    theta: float,
) -> Any:
    import torch

    frequencies = torch.pow(
        torch.tensor(theta, dtype=torch.float32),
        -torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim,
    )
    angles = positions.float()[:, None] * frequencies[None, :]
    cosine = torch.cos(angles).to(value.dtype)[:, None, :]
    sine = torch.sin(angles).to(value.dtype)[:, None, :]
    first = value[..., :rotary_dim:2]
    second = value[..., 1:rotary_dim:2]
    rotated = torch.stack(
        (
            first * cosine - second * sine,
            second * cosine + first * sine,
        ),
        dim=-1,
    ).reshape(*value.shape[:-1], rotary_dim)
    return torch.cat((rotated, value[..., rotary_dim:]), dim=-1)


def _stable_topk(scores: Any, top_k: int) -> Any:
    import torch

    if tuple(scores.shape[:1]) != (1,):
        raise ValueError("Gate C DSA oracle supports one live query row")
    order = sorted(
        range(scores.shape[1]),
        key=lambda position: (-float(scores[0, position]), position),
    )[:top_k]
    return torch.tensor([order], dtype=torch.int32)


def capture_gate_c_oracle(config: GateCOracleConfig) -> dict[str, Any]:
    """Capture the independent real dense/full-DSA/IndexShare oracle."""

    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open
    from safetensors.torch import save_file

    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only Gate C oracle destination exists: {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True)
    index_path = config.source_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("source index has no weight_map")
    expected = _expected_source_tensors(config)
    allowed = set(config.allowed_source_shards)
    source_records: dict[str, dict[str, Any]] = {}
    handles: dict[str, Any] = {}

    with ExitStack() as stack:

        def load(name: str) -> Any:
            if name not in expected:
                raise ValueError(f"unrecognized Gate C source tensor {name!r}")
            filename = weight_map.get(name)
            if not isinstance(filename, str):
                raise ValueError(f"source index has no tensor {name!r}")
            if filename not in allowed:
                raise ValueError(
                    f"Gate C tensor {name!r} escaped allowed shards: {filename}"
                )
            if filename not in handles:
                path = config.source_root / filename
                if not path.is_file():
                    raise FileNotFoundError(f"missing source shard {path}")
                handles[filename] = stack.enter_context(
                    safe_open(path, framework="pt", device="cpu")
                )
            tensor = handles[filename].get_tensor(name)
            expected_shape, expected_dtype = expected[name]
            if (
                tuple(tensor.shape) != expected_shape
                or _dtype_name(tensor) != expected_dtype
            ):
                raise ValueError(
                    f"source {name!r} expected {expected_dtype}{expected_shape}, "
                    f"got {_dtype_name(tensor)}{tuple(tensor.shape)}"
                )
            if tensor.is_floating_point() and not bool(
                torch.isfinite(tensor.float()).all()
            ):
                raise ValueError(f"source tensor {name!r} contains non-finite values")
            record = _tensor_record(name, tensor)
            record["source_shard"] = filename
            prior = source_records.setdefault(name, record)
            if prior != record:
                raise ValueError(f"source tensor {name!r} changed during capture")
            return tensor

        def dequantize(base: str) -> Any:
            return _dequantize_fp8(
                load(f"{base}.weight"),
                load(f"{base}.weight_scale_inv"),
                config.fp8_block_shape,
            )

        producer = config.producer_prefix
        consumer = config.consumer_prefix
        positions = torch.arange(config.context_length, dtype=torch.int32)
        decode_position = positions[-1:]

        layer2_decode_residual = _deterministic_bf16(
            1, config.hidden_size, 0
        )
        layer2_input_normalized = _rms_norm(
            layer2_decode_residual,
            load(f"{producer}.input_layernorm.weight"),
            config.rms_norm_epsilon,
        )
        layer2_history_hidden = _deterministic_bf16(
            config.context_length, config.hidden_size, 23
        )
        layer2_history_hidden[-1:] = layer2_input_normalized

        q_a_weight = dequantize(f"{producer}.self_attn.q_a_proj")
        layer2_q_residual = _rms_norm(
            layer2_input_normalized @ q_a_weight.T,
            load(f"{producer}.self_attn.q_a_layernorm.weight"),
            config.rms_norm_epsilon,
        )
        del q_a_weight
        indexer = f"{producer}.self_attn.indexer"
        wq_weight = dequantize(f"{indexer}.wq_b")
        layer2_index_query = (
            layer2_q_residual.float() @ wq_weight.float().T
        ).reshape(1, config.indexer_heads, config.indexer_head_dim)
        del wq_weight
        layer2_index_query = _rotary_interleaved(
            layer2_index_query,
            decode_position,
            rotary_dim=config.indexer_rotary_dim,
            theta=config.rope_theta,
        )
        wk_weight = dequantize(f"{indexer}.wk")
        projected_keys = layer2_history_hidden.float() @ wk_weight.float().T
        del wk_weight
        layer2_index_keys = _layer_norm(
            projected_keys,
            load(f"{indexer}.k_norm.weight"),
            load(f"{indexer}.k_norm.bias"),
            config.indexer_norm_epsilon,
        )
        del projected_keys
        layer2_index_keys = _rotary_interleaved(
            layer2_index_keys[:, None, :],
            positions,
            rotary_dim=config.indexer_rotary_dim,
            theta=config.rope_theta,
        )[:, 0, :]
        layer2_index_head_weights = (
            layer2_input_normalized.float()
            @ load(f"{indexer}.weights_proj.weight").float().T
        ) * (config.indexer_heads**-0.5)
        per_head_scores = torch.einsum(
            "rhd,sd->rhs", layer2_index_query, layer2_index_keys
        ) * (config.indexer_head_dim**-0.5)
        layer2_index_scores = torch.einsum(
            "rh,rhs->rs",
            layer2_index_head_weights,
            torch.relu(per_head_scores),
        ).float()
        layer2_selected_positions = _stable_topk(
            layer2_index_scores, config.top_k
        )
        layer2_selected_scores = torch.gather(
            layer2_index_scores,
            1,
            layer2_selected_positions.long(),
        )
        current_position = config.context_length - 1
        current_selected = current_position in {
            int(value) for value in layer2_selected_positions[0]
        }
        if config.require_current_selected and not current_selected:
            raise ValueError("real DSA oracle did not select the current position")

        layer2_dense_residual = _deterministic_bf16(
            1, config.hidden_size, 41
        )
        layer2_dense_normalized = _rms_norm(
            layer2_dense_residual,
            load(f"{producer}.post_attention_layernorm.weight"),
            config.rms_norm_epsilon,
        )
        gate_weight = dequantize(f"{producer}.mlp.gate_proj")
        layer2_dense_gate = layer2_dense_normalized @ gate_weight.T
        del gate_weight
        up_weight = dequantize(f"{producer}.mlp.up_proj")
        layer2_dense_up = layer2_dense_normalized @ up_weight.T
        del up_weight
        layer2_dense_activated = (
            functional.silu(layer2_dense_gate) * layer2_dense_up
        ).to(torch.bfloat16)
        down_weight = dequantize(f"{producer}.mlp.down_proj")
        layer2_dense_update = layer2_dense_activated @ down_weight.T
        del down_weight
        layer2_dense_output = (
            layer2_dense_residual + layer2_dense_update
        ).to(torch.bfloat16)

        layer3_decode_residual = _deterministic_bf16(
            1, config.hidden_size, 53
        )
        layer3_input_normalized = _rms_norm(
            layer3_decode_residual,
            load(f"{consumer}.input_layernorm.weight"),
            config.rms_norm_epsilon,
        )
        attention = f"{consumer}.self_attn"
        q_a_weight = dequantize(f"{attention}.q_a_proj")
        layer3_q_residual = _rms_norm(
            layer3_input_normalized @ q_a_weight.T,
            load(f"{attention}.q_a_layernorm.weight"),
            config.rms_norm_epsilon,
        )
        del q_a_weight
        q_b_weight = dequantize(f"{attention}.q_b_proj")
        q_states = (layer3_q_residual @ q_b_weight.T).reshape(
            1, config.num_attention_heads, config.qk_head_dim
        )
        del q_b_weight
        layer3_q_nope = q_states[..., : config.qk_nope_head_dim]
        layer3_q_rope = _rotary_interleaved(
            q_states[..., config.qk_nope_head_dim :],
            decode_position,
            rotary_dim=config.qk_rope_head_dim,
            theta=config.rope_theta,
        )

        kv_a_weight = dequantize(f"{attention}.kv_a_proj_with_mqa")
        current_kv = layer3_input_normalized @ kv_a_weight.T
        del kv_a_weight
        current_latent = _rms_norm(
            current_kv[..., : config.kv_lora_rank],
            load(f"{attention}.kv_a_layernorm.weight"),
            config.rms_norm_epsilon,
        )
        current_k_rope = _rotary_interleaved(
            current_kv[..., config.kv_lora_rank :][:, None, :],
            decode_position,
            rotary_dim=config.qk_rope_head_dim,
            theta=config.rope_theta,
        )[:, 0, :]
        cache_padding = config.packed_cache_width - (
            config.kv_lora_rank + config.qk_rope_head_dim
        )
        layer3_current_cache_row = torch.cat(
            (
                current_latent,
                current_k_rope,
                torch.zeros((1, cache_padding), dtype=torch.bfloat16),
            ),
            dim=-1,
        )
        layer3_cache = _deterministic_bf16(
            config.context_length, config.packed_cache_width, 67
        )
        layer3_cache[-1:] = layer3_current_cache_row

        kv_b_weight = dequantize(f"{attention}.kv_b_proj").reshape(
            config.num_attention_heads,
            config.qk_nope_head_dim + config.v_head_dim,
            config.kv_lora_rank,
        )
        weight_uk = kv_b_weight[:, : config.qk_nope_head_dim, :]
        weight_uv = kv_b_weight[:, config.qk_nope_head_dim :, :].transpose(1, 2)
        layer3_q_absorbed = torch.einsum(
            "rhp,hpl->rhl", layer3_q_nope.float(), weight_uk.float()
        ).to(torch.bfloat16)
        del weight_uk

        layer3_selected_positions = layer2_selected_positions.clone()
        layer3_attention_positions = torch.sort(
            layer3_selected_positions, dim=-1
        ).values
        layer3_selected_cache = layer3_cache[
            layer3_attention_positions[0].long()
        ][None, ...]
        selected_latent = layer3_selected_cache[..., : config.kv_lora_rank]
        selected_rope = layer3_selected_cache[
            ...,
            config.kv_lora_rank : config.kv_lora_rank
            + config.qk_rope_head_dim,
        ]
        layer3_attention_scores = (
            torch.einsum(
                "rhd,rkd->rhk",
                layer3_q_absorbed.float(),
                selected_latent.float(),
            )
            + torch.einsum(
                "rhd,rkd->rhk",
                layer3_q_rope.float(),
                selected_rope.float(),
            )
        ) * (config.qk_head_dim**-0.5)
        score_maximum = torch.max(
            layer3_attention_scores, dim=-1, keepdim=True
        ).values
        unnormalized = torch.exp(layer3_attention_scores - score_maximum)
        denominator = torch.sum(unnormalized, dim=-1, keepdim=True)
        weighted_values = torch.einsum(
            "rhk,rkd->rhd",
            unnormalized.to(torch.bfloat16).float(),
            selected_latent.float(),
        )
        layer3_attended_latent = (
            weighted_values / denominator
        ).to(torch.bfloat16)
        layer3_attention_lse = (
            score_maximum + torch.log(denominator)
        )[..., 0].float()
        layer3_value_states = torch.einsum(
            "rhl,hlv->rhv",
            layer3_attended_latent.float(),
            weight_uv.float(),
        ).to(torch.bfloat16)
        del kv_b_weight, weight_uv
        o_weight = dequantize(f"{attention}.o_proj")
        layer3_attention_update = (
            layer3_value_states.reshape(1, -1) @ o_weight.T
        )
        del o_weight
        layer3_attention_output = (
            layer3_decode_residual + layer3_attention_update
        ).to(torch.bfloat16)

    tensors = {
        "consumer_attended_latent": layer3_attended_latent,
        "consumer_attention_lse": layer3_attention_lse,
        "consumer_attention_output": layer3_attention_output,
        "consumer_attention_positions": layer3_attention_positions,
        "consumer_attention_scores": layer3_attention_scores,
        "consumer_attention_update": layer3_attention_update,
        "consumer_cache": layer3_cache,
        "consumer_current_cache_row": layer3_current_cache_row,
        "consumer_decode_residual": layer3_decode_residual,
        "consumer_input_normalized": layer3_input_normalized,
        "consumer_q_absorbed": layer3_q_absorbed,
        "consumer_q_nope": layer3_q_nope,
        "consumer_q_residual": layer3_q_residual,
        "consumer_q_rope": layer3_q_rope,
        "consumer_selected_cache": layer3_selected_cache,
        "consumer_selected_positions": layer3_selected_positions,
        "consumer_value_states": layer3_value_states,
        "positions": positions,
        "producer_decode_residual": layer2_decode_residual,
        "producer_dense_activated": layer2_dense_activated,
        "producer_dense_gate": layer2_dense_gate,
        "producer_dense_normalized": layer2_dense_normalized,
        "producer_dense_output": layer2_dense_output,
        "producer_dense_residual": layer2_dense_residual,
        "producer_dense_up": layer2_dense_up,
        "producer_dense_update": layer2_dense_update,
        "producer_history_hidden": layer2_history_hidden,
        "producer_index_head_weights": layer2_index_head_weights,
        "producer_index_keys": layer2_index_keys,
        "producer_index_query": layer2_index_query,
        "producer_index_scores": layer2_index_scores,
        "producer_input_normalized": layer2_input_normalized,
        "producer_q_residual": layer2_q_residual,
        "producer_selected_positions": layer2_selected_positions,
        "producer_selected_scores": layer2_selected_scores,
    }
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in tensors.items()
    }
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not bool(
            torch.isfinite(tensor.float()).all()
        ):
            raise ValueError(f"Gate C oracle output {name!r} is non-finite")
    if not torch.equal(
        tensors["producer_selected_positions"],
        tensors["consumer_selected_positions"],
    ):
        raise ValueError("IndexShare consumer changed producer positions")
    if not torch.equal(
        tensors["consumer_attention_positions"],
        torch.sort(tensors["producer_selected_positions"], dim=-1).values,
    ):
        raise ValueError("attention position copy is not canonical")
    if not torch.equal(
        tensors["consumer_cache"][-1:],
        tensors["consumer_current_cache_row"],
    ):
        raise ValueError("IndexShare attention cache is not write-before-attend")

    filename = "oracle.safetensors"
    output_path = config.output_dir / filename
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    save_file(
        tensors,
        partial,
        metadata={
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
            "producer_layer": str(config.producer_layer),
            "consumer_layer": str(config.consumer_layer),
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
            "dense": {
                "layer": config.producer_layer,
                "subsystem": "post_attention_norm_dense_swiglu_residual",
            },
            "full_dsa": {
                "context_length": config.context_length,
                "current_position": current_position,
                "current_selected": current_selected,
                "layer": config.producer_layer,
                "selected_count": config.top_k,
                "tie_policy": "descending_score_then_lowest_global_position",
            },
            "index_share": {
                "attention_order": "ascending_global_position",
                "consumer_layer": config.consumer_layer,
                "current_cache_written_before_attend": True,
                "payload_byte_count": config.top_k * 4,
                "payload_dtype": "I32",
                "payload_shape": [1, config.top_k],
                "producer_layer": config.producer_layer,
                "state_order": "descending_score",
            },
        },
        "code_hash": config.code_hash,
        "consumer_layer": config.consumer_layer,
        "file": {
            "byte_count": output_path.stat().st_size,
            "filename": filename,
            "sha256": _sha256_file(output_path),
            "tensors": output_records,
        },
        "format_version": FORMAT_VERSION,
        "geometry": {
            "context_length": config.context_length,
            "dense_intermediate_size": config.dense_intermediate_size,
            "fp8_block_shape": list(config.fp8_block_shape),
            "hidden_size": config.hidden_size,
            "indexer_head_dim": config.indexer_head_dim,
            "indexer_heads": config.indexer_heads,
            "indexer_rotary_dim": config.indexer_rotary_dim,
            "kv_lora_rank": config.kv_lora_rank,
            "num_attention_heads": config.num_attention_heads,
            "packed_cache_width": config.packed_cache_width,
            "q_lora_rank": config.q_lora_rank,
            "qk_head_dim": config.qk_head_dim,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "top_k": config.top_k,
            "v_head_dim": config.v_head_dim,
        },
        "input_generator": {
            "formula": "bf16((((row*37+column*17+salt)%257)-128)/128)",
            "salts": {
                "consumer_cache": 67,
                "consumer_decode_residual": 53,
                "producer_decode_residual": 0,
                "producer_dense_residual": 41,
                "producer_history_hidden": 23,
            },
        },
        "legacy_code_hash": config.legacy_code_hash,
        "model_id": MODEL_ID,
        "numerical_contract": {
            "activation_dtype": "BF16",
            "dsa_score_dtype": "F32",
            "indexer_rope_interleave": True,
            "main_rope_interleave": True,
            "probability_rounding": "unnormalized_BF16_before_PV",
            "rope_theta": config.rope_theta,
        },
        "producer_layer": config.producer_layer,
        "reference_source_hashes": dict(sorted(config.reference_source_hashes)),
        "source_index_sha256": _sha256_file(index_path),
        "source_revision": config.source_revision,
        "source_tensors": sorted(
            source_records.values(), key=lambda item: item["name"]
        ),
        "source_uri": config.source_uri.rstrip("/"),
        "vllm_code_hash": config.vllm_code_hash,
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    inspect_gate_c_oracle(config.output_dir)
    return manifest


def inspect_gate_c_oracle(output_dir: Path) -> dict[str, Any]:
    """Verify the Gate C oracle manifest and every captured tensor."""

    from safetensors import safe_open

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported Gate C oracle format")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("not a greenfield Gate C oracle")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("Gate C oracle manifest checksum mismatch")
    file_record = manifest["file"]
    path = output_dir / file_record["filename"]
    if path.stat().st_size != file_record["byte_count"]:
        raise ValueError("Gate C oracle file size mismatch")
    if _sha256_file(path) != file_record["sha256"]:
        raise ValueError("Gate C oracle file checksum mismatch")
    expected = {record["name"]: record for record in file_record["tensors"]}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("Gate C oracle tensor key mismatch")
        for name, record in expected.items():
            tensor_slice = handle.get_slice(name)
            if (
                list(tensor_slice.get_shape()) != record["shape"]
                or tensor_slice.get_dtype() != record["dtype"]
            ):
                raise ValueError(f"Gate C oracle tensor metadata mismatch: {name}")
    index_share = manifest["cases"]["index_share"]
    expected_shape = [1, manifest["geometry"]["top_k"]]
    if index_share["payload_shape"] != expected_shape:
        raise ValueError("Gate C IndexShare payload shape drifted")
    if index_share["payload_byte_count"] != expected_shape[1] * 4:
        raise ValueError("Gate C IndexShare payload byte count drifted")
    return manifest
