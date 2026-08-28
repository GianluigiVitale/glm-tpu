"""Causal, evidence-bound prefill graph for the PP16 feature2 discriminator.

This module describes data dependencies only. It authorizes neither a TPU
compile nor numerical execution. The graph is deliberately unrolled across
four distinct chunks so every cache version, position range and tail mask is
explicit. Accepted/oracle model intermediates cannot enter the graph; only the
authenticated token sequence is an external workload input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..errors import BenchmarkValidationError
from ..kernels.reference.rotary import (
    build_rotary_table_host,
    rotary_table_sha256,
)
from ..validation.short_context_oracle import inspect_short_context_oracle
from .pp16_feature_sharded_state import (
    PP16_FEATURE2_CONTEXT_LENGTH,
    PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
    Feature2TensorRead,
)


PP16_FEATURE2_PREFILL_CHUNK = 2048
PP16_FEATURE2_PREFILL_VALID_ROWS = (2048, 2048, 2048, 2012)
PP16_FEATURE2_CURRENT_POSITION = 8155
PP16_FEATURE2_CURRENT_TOKEN_ID = 220
PP16_FEATURE2_LOGICAL_PAGE_SIZE = 512
PP16_FEATURE2_LOCAL_ROWS_PER_PAGE = 256
PP16_FEATURE2_LOGICAL_PAGES = 16
PP16_FEATURE2_DSA_TOP_K = 2048
PP16_FEATURE2_MAIN_ROPE_CAPACITY = 8192
PP16_FEATURE2_MAIN_ROPE_WIDTH = 64
PP16_FEATURE2_MAIN_ROPE_THETA = 8_000_000.0
PP16_FEATURE2_MAIN_ROPE_TABLE_SHA256 = (
    "6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701"
)
PP16_FEATURE2_LOCAL_GROUP = (0, 1)
PP16_FEATURE2_TIE_POLICY = "descending_score_then_lowest_global_position"
PP16_FEATURE2_ORACLE_MANIFEST_SHA256 = (
    "e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2"
)
PP16_FEATURE2_TOKEN_FILE_SHA256 = (
    "f354e30c6688b8377350da0e28a276c45e5a3bcfce8571512c45fa0dd0328878"
)
PP16_FEATURE2_PROMPT_TOKEN_IDS_SHA256 = (
    "d860b7f4be91608c86e0a629c4096fd7a95036287d0f8e31ea67b01475de0cc0"
)
PP16_FEATURE2_GENERATED_TOKEN_IDS_SHA256 = (
    "909682cb84f03b1570893a3b28435603f7ed617d39e2a685017e016964538173"
)
PP16_FEATURE2_CANDIDATE_TOKEN_IDS_SHA256 = (
    "0ba9602ef9feeb406796eb80398ff36ab8999b9f0e38bace0b9550578a27b4b9"
)
PP16_FEATURE2_POSITIONS_SHA256 = (
    "72605bc1aa8518c8670d42b7916b7dcd0f3c8971000e499ea1503116e547ce3a"
)
PP16_FEATURE2_BLOCK_TABLES_SHA256 = (
    "5d85718ec594b982c252d0279e5966ffca33a5eaf2a455038d3ab331fde70cea"
)
PP16_FEATURE2_CONTEXT_LENGTHS_SHA256 = (
    "1212f21b46dbbe5299cdae8e1b60d67e89842b523409cdbf1ee7d8d0d7918ef8"
)
PP16_FEATURE2_CURRENT_POSITION_SHA256 = (
    "3369abcb97b3bcd3bad56a378c968aaad080314d197e8553628812c397ecf04c"
)
PP16_FEATURE2_SELECTED_READS_SHA256 = (
    "44d2a98eea774f991d021748655bd7ef5f7448797f570235ab538dd4c9208da1"
)

_CHUNK_TOKEN_SHA256 = (
    "3a7505a9ae21c3d8c5cff7d46b8af27a250531c876d8257ecdc1318ef6d1f59c",
    "efad5460717cac90d239324057b22012d6e51dd8cb600abdb3390566e464cb51",
    "010129fbbc67c7b051e5751ff3e6e4502bab269080128ecbe23256a72b93f82b",
    "e1bb8ca52aef779ba7a3b6a403f645142c7aa785ca8cd84a257a51df1f193464",
)
_CHUNK_POSITION_SHA256 = (
    "cc76b029564c7257d6c27e130546ac40603f1e3ae5efc1106b2656294f599ec5",
    "782a79365a05d09b97fac8e5f21cbfcdcc9c2c77c6fb5c0f097ae7cfc204fadb",
    "af05bae6d4e603fc1492b3e28df195a48fec6633c0c8feeeeb8f3975b2bccffb",
    "13907a23dcf30afb3d81292fae913f72d520486ccac42adb0df808b434ca0e5f",
)
_CHUNK_CAUSAL_CONTEXT_SHA256 = (
    "791f77305adabfcc6ed5741c707fbdd16658a735be6412ea97ba1b014ff096f5",
    "14fbca94499d21cbb617f331ab3259f8044d5cce278e6a041072387e00a488eb",
    "e47be16b72f585deab7119f1267811e13d31004586075112eb311f31e7965846",
    "02290d60d03799c0f583fa326043403c1e90e6d7da9b168385c200c863c623a5",
)

_LAYER0_ATTENTION = (
    "attention.slot_00.input_norm",
    "attention.slot_00.kv_a_norm",
    "attention.slot_00.kv_b.scale_inv",
    "attention.slot_00.kv_b.weight_bits",
    "attention.slot_00.o.scale_inv",
    "attention.slot_00.o.weight_bits",
    "attention.slot_00.q_a_norm",
    "attention.slot_00.q_b.scale_inv",
    "attention.slot_00.q_b.weight_bits",
    "attention.slot_00.qkv_a.scale_inv",
    "attention.slot_00.qkv_a.weight_bits",
)
_LAYER0_DENSE = (
    "attention.slot_00.post_norm",
    "dense.slot_00.down.scale_inv",
    "dense.slot_00.down.weight_bits",
    "dense.slot_00.gate.scale_inv",
    "dense.slot_00.gate.weight_bits",
    "dense.slot_00.up.scale_inv",
    "dense.slot_00.up.weight_bits",
)
_LAYER0_INDEXER = (
    "indexer.slot_00.head_weight",
    "indexer.slot_00.key_norm_bias",
    "indexer.slot_00.key_norm_weight",
    "indexer.slot_00.wk.scale_inv",
    "indexer.slot_00.wk.weight_bits",
    "indexer.slot_00.wq_b.scale_inv",
    "indexer.slot_00.wq_b.weight_bits",
)
_LAYER1_BOUNDARY = ("attention.slot_01.input_norm",)
_LAYER1_N82 = (
    "attention.slot_01.q_a_norm",
    "attention.slot_01.qkv_a.scale_inv",
    "attention.slot_01.qkv_a.weight_bits",
)
_LAYER1_ATTENTION_QUERY = (
    "attention.slot_01.q_b.scale_inv",
    "attention.slot_01.q_b.weight_bits",
)
_LAYER1_INDEX_KEYS = (
    "indexer.slot_01.key_norm_bias",
    "indexer.slot_01.key_norm_weight",
    "indexer.slot_01.wk.scale_inv",
    "indexer.slot_01.wk.weight_bits",
)
_LAYER1_DSA_QUERY = (
    "indexer.slot_01.head_weight",
    "indexer.slot_01.wq_b.scale_inv",
    "indexer.slot_01.wq_b.weight_bits",
)
_EXPECTED_WEIGHTS = frozenset(
    ("global.embedding",)
    + _LAYER0_ATTENTION
    + _LAYER0_DENSE
    + _LAYER0_INDEXER
    + _LAYER1_BOUNDARY
    + _LAYER1_N82
    + _LAYER1_ATTENTION_QUERY
    + _LAYER1_INDEX_KEYS
    + _LAYER1_DSA_QUERY
)


@dataclass(frozen=True, slots=True)
class Feature2PrefillSource:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    exact_values: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "exact_values": list(self.exact_values),
            "name": self.name,
            "sha256": self.sha256,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class Feature2PrefillChunk:
    chunk_index: int
    start: int
    stop: int
    padded_stop: int
    valid_rows: int
    token_ids_sha256: str
    positions_sha256: str
    causal_context_sha256: str
    tail_policy: str = "exact_valid_rows_repair_only_pad_and_drop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "causal_context_sha256": self.causal_context_sha256,
            "padded_stop": self.padded_stop,
            "positions_sha256": self.positions_sha256,
            "start": self.start,
            "stop": self.stop,
            "tail_policy": self.tail_policy,
            "token_ids_sha256": self.token_ids_sha256,
            "valid_rows": self.valid_rows,
        }


@dataclass(frozen=True, slots=True)
class Feature2PrefillInputs:
    oracle_manifest_sha256: str
    token_file_sha256: str
    prompt_token_ids_sha256: str
    generated_token_ids_sha256: str
    current_token_id: int
    sources: tuple[Feature2PrefillSource, ...]
    chunks: tuple[Feature2PrefillChunk, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "current_token_id": self.current_token_id,
            "generated_token_ids_sha256": self.generated_token_ids_sha256,
            "oracle_manifest_sha256": self.oracle_manifest_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "sources": [source.to_dict() for source in self.sources],
            "token_file_sha256": self.token_file_sha256,
        }


@dataclass(frozen=True, slots=True)
class Feature2WeightRole:
    name: str
    weights: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": list(self.weights)}


@dataclass(frozen=True, slots=True)
class Feature2PrefillTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    storage: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "name": self.name,
            "shape": list(self.shape),
            "storage": self.storage,
        }


@dataclass(frozen=True, slots=True)
class Feature2PrefillNode:
    name: str
    operation: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    weight_role: str | None = None
    devices: tuple[int, ...] = PP16_FEATURE2_LOCAL_GROUP

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": list(self.devices),
            "inputs": list(self.inputs),
            "name": self.name,
            "operation": self.operation,
            "outputs": list(self.outputs),
            "weight_role": self.weight_role,
        }


@dataclass(frozen=True, slots=True)
class Feature2PrefillGraph:
    context_length: int
    current_position: int
    prompt_chunk: int
    valid_rows_by_chunk: tuple[int, ...]
    logical_page_size: int
    local_rows_per_page: int
    logical_pages: int
    local_group: tuple[int, ...]
    decode_rows: int
    dsa_top_k: int
    score_precision: str
    tie_policy: str
    key_association: str
    query_association: str
    runtime_manifest_sha256: str
    selected_reads_sha256: str
    selected_reads: tuple[Feature2TensorRead, ...]
    inputs: Feature2PrefillInputs
    weight_roles: tuple[Feature2WeightRole, ...]
    tensors: tuple[Feature2PrefillTensor, ...]
    nodes: tuple[Feature2PrefillNode, ...]
    terminal_outputs: tuple[str, ...]
    graph_sha256: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "context_length": self.context_length,
            "current_position": self.current_position,
            "decode_rows": self.decode_rows,
            "dsa_top_k": self.dsa_top_k,
            "inputs": self.inputs.to_dict(),
            "key_association": self.key_association,
            "local_group": list(self.local_group),
            "local_rows_per_page": self.local_rows_per_page,
            "logical_page_size": self.logical_page_size,
            "logical_pages": self.logical_pages,
            "nodes": [node.to_dict() for node in self.nodes],
            "prompt_chunk": self.prompt_chunk,
            "query_association": self.query_association,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "score_precision": self.score_precision,
            "selected_reads": [item.to_dict() for item in self.selected_reads],
            "selected_reads_sha256": self.selected_reads_sha256,
            "terminal_outputs": list(self.terminal_outputs),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "tie_policy": self.tie_policy,
            "valid_rows_by_chunk": list(self.valid_rows_by_chunk),
            "weight_roles": [role.to_dict() for role in self.weight_roles],
        }
        if include_hash:
            value["graph_sha256"] = self.graph_sha256
        return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value, dtype="<i4").tobytes()).hexdigest()


def _selected_reads_sha256(reads: Sequence[Feature2TensorRead]) -> str:
    return sha256(
        _canonical_json([item.to_dict() for item in reads]).encode("utf-8")
    ).hexdigest()


def _graph_sha256(graph: Feature2PrefillGraph) -> str:
    return sha256(
        _canonical_json(graph.to_dict(include_hash=False)).encode("utf-8")
    ).hexdigest()


def load_feature2_prefill_inputs(oracle_dir: Path) -> Feature2PrefillInputs:
    """Authenticate the exact 8K token workload and derive causal inputs."""

    from safetensors import safe_open

    root = Path(oracle_dir)
    manifest = inspect_short_context_oracle(root)
    if manifest["manifest_sha256"] != PP16_FEATURE2_ORACLE_MANIFEST_SHA256:
        raise BenchmarkValidationError("feature2 prefill oracle manifest drifted")
    token_record = manifest["files"]["tokens"]
    if token_record["sha256"] != PP16_FEATURE2_TOKEN_FILE_SHA256:
        raise BenchmarkValidationError("feature2 prefill token file drifted")
    with safe_open(root / token_record["filename"], framework="np") as handle:
        prompt = np.asarray(handle.get_tensor("prompt_token_ids"), dtype=np.int32)
        generated = np.asarray(
            handle.get_tensor("generated_token_ids"), dtype=np.int32
        )
    if _sha256_array(prompt) != PP16_FEATURE2_PROMPT_TOKEN_IDS_SHA256 or (
        _sha256_array(generated) != PP16_FEATURE2_GENERATED_TOKEN_IDS_SHA256
    ):
        raise BenchmarkValidationError("feature2 prefill token arrays drifted")
    candidate = np.ascontiguousarray(
        np.concatenate((prompt, generated[:1])), dtype=np.int32
    )
    positions = np.arange(PP16_FEATURE2_CONTEXT_LENGTH, dtype=np.int32)
    block_tables = np.arange(PP16_FEATURE2_LOGICAL_PAGES, dtype=np.int32)[
        None, :
    ]
    context_lengths = np.asarray(
        [PP16_FEATURE2_CONTEXT_LENGTH], dtype=np.int32
    )
    current_position = np.asarray(
        [PP16_FEATURE2_CURRENT_POSITION], dtype=np.int32
    )
    main_rope_table = build_rotary_table_host(
        PP16_FEATURE2_MAIN_ROPE_CAPACITY,
        rotary_dim=PP16_FEATURE2_MAIN_ROPE_WIDTH,
        theta=PP16_FEATURE2_MAIN_ROPE_THETA,
    )
    if (
        rotary_table_sha256(main_rope_table)
        != PP16_FEATURE2_MAIN_ROPE_TABLE_SHA256
    ):
        raise BenchmarkValidationError("feature2 main-RoPE table drifted")
    chunks: list[Feature2PrefillChunk] = []
    start = 0
    for index, valid_rows in enumerate(PP16_FEATURE2_PREFILL_VALID_ROWS):
        stop = start + valid_rows
        chunks.append(
            Feature2PrefillChunk(
                chunk_index=index,
                start=start,
                stop=stop,
                padded_stop=start + PP16_FEATURE2_PREFILL_CHUNK,
                valid_rows=valid_rows,
                token_ids_sha256=_sha256_array(candidate[start:stop]),
                positions_sha256=_sha256_array(positions[start:stop]),
                causal_context_sha256=_sha256_array(
                    positions[start:stop] + np.int32(1)
                ),
            )
        )
        start = stop
    inputs = Feature2PrefillInputs(
        oracle_manifest_sha256=manifest["manifest_sha256"],
        token_file_sha256=token_record["sha256"],
        prompt_token_ids_sha256=manifest["prompt_token_ids_sha256"],
        generated_token_ids_sha256=manifest["generated_token_ids_sha256"],
        current_token_id=int(candidate[-1]),
        sources=(
            Feature2PrefillSource(
                "candidate_token_ids",
                candidate.shape,
                "i32",
                _sha256_array(candidate),
            ),
            Feature2PrefillSource(
                "candidate_positions",
                positions.shape,
                "i32",
                _sha256_array(positions),
            ),
            Feature2PrefillSource(
                "block_tables",
                block_tables.shape,
                "i32",
                _sha256_array(block_tables),
                tuple(int(item) for item in block_tables.reshape(-1)),
            ),
            Feature2PrefillSource(
                "context_lengths",
                context_lengths.shape,
                "i32",
                _sha256_array(context_lengths),
                (PP16_FEATURE2_CONTEXT_LENGTH,),
            ),
            Feature2PrefillSource(
                "current_position",
                current_position.shape,
                "i32",
                _sha256_array(current_position),
                (PP16_FEATURE2_CURRENT_POSITION,),
            ),
            Feature2PrefillSource(
                "main_rope_table",
                main_rope_table.shape,
                "bf16",
                rotary_table_sha256(main_rope_table),
            ),
        ),
        chunks=tuple(chunks),
    )
    validate_feature2_prefill_inputs(inputs)
    return inputs


def _expected_sources() -> tuple[Feature2PrefillSource, ...]:
    return (
        Feature2PrefillSource(
            "candidate_token_ids",
            (8156,),
            "i32",
            PP16_FEATURE2_CANDIDATE_TOKEN_IDS_SHA256,
        ),
        Feature2PrefillSource(
            "candidate_positions",
            (8156,),
            "i32",
            PP16_FEATURE2_POSITIONS_SHA256,
        ),
        Feature2PrefillSource(
            "block_tables",
            (1, 16),
            "i32",
            PP16_FEATURE2_BLOCK_TABLES_SHA256,
            tuple(range(16)),
        ),
        Feature2PrefillSource(
            "context_lengths",
            (1,),
            "i32",
            PP16_FEATURE2_CONTEXT_LENGTHS_SHA256,
            (8156,),
        ),
        Feature2PrefillSource(
            "current_position",
            (1,),
            "i32",
            PP16_FEATURE2_CURRENT_POSITION_SHA256,
            (8155,),
        ),
        Feature2PrefillSource(
            "main_rope_table",
            (
                PP16_FEATURE2_MAIN_ROPE_CAPACITY,
                PP16_FEATURE2_MAIN_ROPE_WIDTH,
            ),
            "bf16",
            PP16_FEATURE2_MAIN_ROPE_TABLE_SHA256,
        ),
    )


def _expected_chunks() -> tuple[Feature2PrefillChunk, ...]:
    result = []
    start = 0
    for index, valid_rows in enumerate(PP16_FEATURE2_PREFILL_VALID_ROWS):
        stop = start + valid_rows
        result.append(
            Feature2PrefillChunk(
                index,
                start,
                stop,
                start + PP16_FEATURE2_PREFILL_CHUNK,
                valid_rows,
                _CHUNK_TOKEN_SHA256[index],
                _CHUNK_POSITION_SHA256[index],
                _CHUNK_CAUSAL_CONTEXT_SHA256[index],
            )
        )
        start = stop
    return tuple(result)


def validate_feature2_prefill_inputs(
    inputs: Feature2PrefillInputs,
) -> None:
    expected_header = (
        PP16_FEATURE2_ORACLE_MANIFEST_SHA256,
        PP16_FEATURE2_TOKEN_FILE_SHA256,
        PP16_FEATURE2_PROMPT_TOKEN_IDS_SHA256,
        PP16_FEATURE2_GENERATED_TOKEN_IDS_SHA256,
        PP16_FEATURE2_CURRENT_TOKEN_ID,
    )
    observed_header = (
        inputs.oracle_manifest_sha256,
        inputs.token_file_sha256,
        inputs.prompt_token_ids_sha256,
        inputs.generated_token_ids_sha256,
        inputs.current_token_id,
    )
    if observed_header != expected_header:
        raise BenchmarkValidationError("feature2 prefill input identity drifted")
    if inputs.sources != _expected_sources():
        raise BenchmarkValidationError("feature2 prefill source tensors drifted")
    if inputs.chunks != _expected_chunks():
        raise BenchmarkValidationError("feature2 prefill chunk intervals drifted")


def _weight_roles() -> tuple[Feature2WeightRole, ...]:
    return (
        Feature2WeightRole("embedding", ("global.embedding",)),
        Feature2WeightRole(
            "layer0_attention_indexer", _LAYER0_ATTENTION + _LAYER0_INDEXER
        ),
        Feature2WeightRole("layer0_dense", _LAYER0_DENSE),
        Feature2WeightRole("layer1_boundary", _LAYER1_BOUNDARY),
        Feature2WeightRole("layer1_n82", _LAYER1_N82),
        Feature2WeightRole("layer1_attention_query", _LAYER1_ATTENTION_QUERY),
        Feature2WeightRole("layer1_index_keys", _LAYER1_INDEX_KEYS),
        Feature2WeightRole("layer1_dsa_query", _LAYER1_DSA_QUERY),
    )


def _tensor(
    name: str, shape: tuple[int, ...], dtype: str, storage: str
) -> Feature2PrefillTensor:
    return Feature2PrefillTensor(name, shape, dtype, storage)


def _tensors(inputs: Feature2PrefillInputs) -> tuple[Feature2PrefillTensor, ...]:
    values = [
        _tensor(source.name, source.shape, source.dtype, "authenticated_source")
        for source in inputs.sources
    ]
    values.extend(
        (
            _tensor(
                "layer0_kv_cache_0", (16, 256, 640), "bf16", "zero_owner_local"
            ),
            _tensor(
                "layer0_index_cache_0",
                (16, 256, 128),
                "bf16",
                "zero_owner_local",
            ),
            _tensor(
                "layer1_index_cache_0",
                (16, 256, 128),
                "bf16",
                "zero_owner_local",
            ),
            _tensor(
                "layer1_carried_liveness_digest_0",
                (2,),
                "u32",
                "zero_owner_local",
            ),
        )
    )
    for index in range(4):
        suffix = str(index)
        next_suffix = str(index + 1)
        rows = inputs.chunks[index].valid_rows
        values.extend(
            (
                _tensor(f"token_chunk_{suffix}", (rows,), "i32", "ephemeral"),
                _tensor(
                    f"position_chunk_{suffix}", (rows,), "i32", "ephemeral"
                ),
                _tensor(
                    f"causal_context_chunk_{suffix}",
                    (rows,),
                    "i32",
                    "ephemeral",
                ),
                _tensor(
                    f"embedding_chunk_{suffix}",
                    (rows, 6144),
                    "bf16",
                    "ephemeral",
                ),
                _tensor(
                    f"layer0_attention_update_{suffix}",
                    (rows, 6144),
                    "bf16",
                    "ephemeral",
                ),
                _tensor(
                    f"layer0_post_attention_carried_{suffix}",
                    (rows, 6144),
                    "bf16",
                    "ephemeral",
                ),
                _tensor(
                    f"layer0_dense_input_{suffix}",
                    (rows, 6144),
                    "bf16",
                    "ephemeral",
                ),
                _tensor(
                    f"layer0_dense_update_{suffix}",
                    (rows, 6144),
                    "bf16",
                    "ephemeral",
                ),
                _tensor(
                    f"layer1_carried_{suffix}",
                    (rows, 6144),
                    "bf16",
                    "ephemeral",
                ),
                _tensor(
                    f"layer1_carried_liveness_digest_{next_suffix}",
                    (2,),
                    "u32",
                    "owner_local",
                ),
                _tensor(
                    f"layer1_normalized_{suffix}",
                    (rows, 6144),
                    "bf16",
                    "ephemeral",
                ),
                _tensor(
                    f"layer0_kv_cache_{next_suffix}",
                    (16, 256, 640),
                    "bf16",
                    "owner_local",
                ),
                _tensor(
                    f"layer0_index_cache_{next_suffix}",
                    (16, 256, 128),
                    "bf16",
                    "owner_local",
                ),
                _tensor(
                    f"layer1_index_cache_{next_suffix}",
                    (16, 256, 128),
                    "bf16",
                    "owner_local",
                ),
            )
        )
    values.extend(
        (
            _tensor(
                "candidate_current_carried", (1, 6144), "bf16", "terminal"
            ),
            _tensor(
                "candidate_current_normalized", (1, 6144), "bf16", "ephemeral"
            ),
            _tensor(
                "candidate_layer1_q_residual", (1, 2048), "bf16", "ephemeral"
            ),
            _tensor(
                "candidate_layer1_kv_a", (1, 576), "bf16", "diagnostic"
            ),
            _tensor(
                "candidate_layer1_attention_query",
                (1, 32, 256),
                "bf16",
                "diagnostic",
            ),
            _tensor(
                "candidate_layer1_dsa_query", (1, 32, 128), "f32", "ephemeral"
            ),
            _tensor(
                "candidate_layer1_head_weights", (1, 32), "f32", "ephemeral"
            ),
            _tensor(
                "candidate_event1_positions", (1, 2048), "i32", "terminal"
            ),
            _tensor(
                "candidate_event1_valid_counts", (1,), "i32", "terminal"
            ),
            _tensor(
                "candidate_event1_scores", (1, 2048), "f32", "terminal"
            ),
        )
    )
    return tuple(values)


def _nodes(inputs: Feature2PrefillInputs) -> tuple[Feature2PrefillNode, ...]:
    nodes = [
        Feature2PrefillNode(
            "initialize_candidate_caches",
            "zero_initialize_lp2_layer0_kv_index_and_layer1_index",
            (),
            (
                "layer0_kv_cache_0",
                "layer0_index_cache_0",
                "layer1_index_cache_0",
                "layer1_carried_liveness_digest_0",
            ),
        )
    ]
    for chunk in inputs.chunks:
        index = chunk.chunk_index
        suffix = str(index)
        next_suffix = str(index + 1)
        nodes.extend(
            (
                Feature2PrefillNode(
                    f"slice_chunk_{suffix}",
                    (
                        f"slice_exact_valid_[{chunk.start},{chunk.stop})_"
                        f"repair_only_pad_to_{chunk.padded_stop}_and_drop"
                    ),
                    ("candidate_token_ids", "candidate_positions"),
                    (
                        f"token_chunk_{suffix}",
                        f"position_chunk_{suffix}",
                        f"causal_context_chunk_{suffix}",
                    ),
                ),
                Feature2PrefillNode(
                    f"prompt_embedding_{suffix}",
                    "owner_local_embedding_exact_valid_chunk",
                    (f"token_chunk_{suffix}",),
                    (f"embedding_chunk_{suffix}",),
                    "embedding",
                ),
                Feature2PrefillNode(
                    f"layer0_attention_indexer_{suffix}",
                    (
                        "complete_layer0_with_loop_carried_kv_and_index_"
                        "accepted_main_rope_table"
                    ),
                    (
                        f"embedding_chunk_{suffix}",
                        f"position_chunk_{suffix}",
                        f"causal_context_chunk_{suffix}",
                        f"layer0_kv_cache_{suffix}",
                        f"layer0_index_cache_{suffix}",
                        "block_tables",
                        "main_rope_table",
                    ),
                    (
                        f"layer0_attention_update_{suffix}",
                        f"layer0_kv_cache_{next_suffix}",
                        f"layer0_index_cache_{next_suffix}",
                    ),
                    "layer0_attention_indexer",
                ),
                Feature2PrefillNode(
                    f"layer0_post_attention_boundary_{suffix}",
                    "feature_sharded_attention_add_then_dense_rms_exact_valid",
                    (
                        f"embedding_chunk_{suffix}",
                        f"layer0_attention_update_{suffix}",
                    ),
                    (
                        f"layer0_post_attention_carried_{suffix}",
                        f"layer0_dense_input_{suffix}",
                    ),
                    "layer0_dense",
                ),
                Feature2PrefillNode(
                    f"layer0_dense_{suffix}",
                    "complete_candidate_layer0_dense_exact_valid_chunk",
                    (f"layer0_dense_input_{suffix}",),
                    (f"layer0_dense_update_{suffix}",),
                    "layer0_dense",
                ),
                Feature2PrefillNode(
                    f"layer1_feature2_boundary_{suffix}",
                    "feature_sharded_add_then_layer1_rms_exact_valid_chunk",
                    (
                        f"layer0_post_attention_carried_{suffix}",
                        f"layer0_dense_update_{suffix}",
                    ),
                    (
                        f"layer1_carried_{suffix}",
                        f"layer1_normalized_{suffix}",
                    ),
                    "layer1_boundary",
                ),
                Feature2PrefillNode(
                    f"layer1_carried_liveness_{suffix}",
                    (
                        "owner_local_ordered_dual_u32_digest_all_valid_"
                        f"bf16_values_[{chunk.start},{chunk.stop})"
                    ),
                    (
                        f"layer1_carried_{suffix}",
                        f"position_chunk_{suffix}",
                        f"layer1_carried_liveness_digest_{suffix}",
                    ),
                    (f"layer1_carried_liveness_digest_{next_suffix}",),
                ),
                Feature2PrefillNode(
                    f"layer1_index_key_prefill_{suffix}",
                    "physical_m64_divide_sqrt_keys_loop_carried_lp2_pages",
                    (
                        f"layer1_normalized_{suffix}",
                        f"position_chunk_{suffix}",
                        f"layer1_index_cache_{suffix}",
                        "block_tables",
                    ),
                    (f"layer1_index_cache_{next_suffix}",),
                    "layer1_index_keys",
                ),
            )
        )
    nodes.extend(
        (
            Feature2PrefillNode(
                "layer1_current_row",
                "slice_chunk_3_local_row_2011_position_8155",
                (
                    "layer1_carried_3",
                    "layer1_normalized_3",
                    "position_chunk_3",
                    "current_position",
                ),
                ("candidate_current_carried", "candidate_current_normalized"),
            ),
            Feature2PrefillNode(
                "layer1_n82_qkv_a",
                "full_n82_qkv_a_then_q_a_norm",
                ("candidate_current_normalized",),
                ("candidate_layer1_q_residual", "candidate_layer1_kv_a"),
                "layer1_n82",
            ),
            Feature2PrefillNode(
                "layer1_attention_query",
                (
                    "candidate_attention_q_b_projection_and_position_8155_"
                    "accepted_main_rope_table"
                ),
                (
                    "candidate_layer1_q_residual",
                    "current_position",
                    "main_rope_table",
                ),
                ("candidate_layer1_attention_query",),
                "layer1_attention_query",
            ),
            Feature2PrefillNode(
                "layer1_dsa_query_head",
                "tuple4_exact_dsa_query_head_at_position_8155",
                (
                    "candidate_layer1_q_residual",
                    "candidate_current_normalized",
                    "current_position",
                ),
                (
                    "candidate_layer1_dsa_query",
                    "candidate_layer1_head_weights",
                ),
                "layer1_dsa_query",
            ),
            Feature2PrefillNode(
                "layer1_event1_scorer",
                "default_precision_exact_distributed_topk",
                (
                    "candidate_layer1_dsa_query",
                    "candidate_layer1_head_weights",
                    "layer1_index_cache_4",
                    "block_tables",
                    "context_lengths",
                ),
                (
                    "candidate_event1_positions",
                    "candidate_event1_valid_counts",
                    "candidate_event1_scores",
                ),
            ),
        )
    )
    return tuple(nodes)


def _selected_names(reads: Sequence[Feature2TensorRead]) -> frozenset[str]:
    by_slot: dict[int, set[str]] = {0: set(), 1: set()}
    for tensor in reads:
        if tensor.device_slot not in by_slot:
            raise BenchmarkValidationError(
                "feature2 prefill reads contain an out-of-stage owner"
            )
        by_slot[tensor.device_slot].add(tensor.name)
    if not by_slot[0] or by_slot[0] != by_slot[1]:
        raise BenchmarkValidationError(
            "feature2 prefill owners have different selected weights"
        )
    return frozenset(by_slot[0])


def build_feature2_prefill_graph(
    allowed: Sequence[Feature2TensorRead],
    inputs: Feature2PrefillInputs,
) -> Feature2PrefillGraph:
    """Build the only admitted candidate-coherent feature2 prefill graph."""

    validate_feature2_prefill_inputs(inputs)
    selected_reads = tuple(allowed)
    selected = _selected_names(selected_reads)
    if selected != _EXPECTED_WEIGHTS:
        missing = sorted(_EXPECTED_WEIGHTS - selected)
        extra = sorted(selected - _EXPECTED_WEIGHTS)
        raise BenchmarkValidationError(
            f"feature2 prefill weights drifted: missing={missing}, extra={extra}"
        )
    reads_sha = _selected_reads_sha256(selected_reads)
    if reads_sha != PP16_FEATURE2_SELECTED_READS_SHA256:
        raise BenchmarkValidationError("feature2 prefill selected ranges drifted")
    graph = Feature2PrefillGraph(
        context_length=PP16_FEATURE2_CONTEXT_LENGTH,
        current_position=PP16_FEATURE2_CURRENT_POSITION,
        prompt_chunk=PP16_FEATURE2_PREFILL_CHUNK,
        valid_rows_by_chunk=PP16_FEATURE2_PREFILL_VALID_ROWS,
        logical_page_size=PP16_FEATURE2_LOGICAL_PAGE_SIZE,
        local_rows_per_page=PP16_FEATURE2_LOCAL_ROWS_PER_PAGE,
        logical_pages=PP16_FEATURE2_LOGICAL_PAGES,
        local_group=PP16_FEATURE2_LOCAL_GROUP,
        decode_rows=1,
        dsa_top_k=PP16_FEATURE2_DSA_TOP_K,
        score_precision="default",
        tie_policy=PP16_FEATURE2_TIE_POLICY,
        key_association="physical_m64_divide_sqrt",
        query_association="tuple4_exact_from_candidate_current_row",
        runtime_manifest_sha256=PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
        selected_reads_sha256=reads_sha,
        selected_reads=selected_reads,
        inputs=inputs,
        weight_roles=_weight_roles(),
        tensors=_tensors(inputs),
        nodes=_nodes(inputs),
        terminal_outputs=(
            "candidate_event1_positions",
            "candidate_event1_valid_counts",
            "candidate_event1_scores",
            "candidate_current_carried",
            "candidate_layer1_attention_query",
            "candidate_layer1_kv_a",
            "layer0_kv_cache_4",
            "layer0_index_cache_4",
            "layer1_index_cache_4",
            "layer1_carried_liveness_digest_4",
        ),
        graph_sha256="",
    )
    graph = replace(graph, graph_sha256=_graph_sha256(graph))
    validate_feature2_prefill_graph(graph)
    return graph


def validate_feature2_prefill_graph(
    graph: Feature2PrefillGraph,
) -> dict[str, Any]:
    """Reject incomplete history, unbound inputs, dead rows and broad groups."""

    violations: list[str] = []
    exact_scalars = {
        "context_length": PP16_FEATURE2_CONTEXT_LENGTH,
        "current_position": PP16_FEATURE2_CURRENT_POSITION,
        "prompt_chunk": PP16_FEATURE2_PREFILL_CHUNK,
        "valid_rows_by_chunk": PP16_FEATURE2_PREFILL_VALID_ROWS,
        "logical_page_size": PP16_FEATURE2_LOGICAL_PAGE_SIZE,
        "local_rows_per_page": PP16_FEATURE2_LOCAL_ROWS_PER_PAGE,
        "logical_pages": PP16_FEATURE2_LOGICAL_PAGES,
        "local_group": PP16_FEATURE2_LOCAL_GROUP,
        "decode_rows": 1,
        "dsa_top_k": PP16_FEATURE2_DSA_TOP_K,
        "score_precision": "default",
        "tie_policy": PP16_FEATURE2_TIE_POLICY,
        "key_association": "physical_m64_divide_sqrt",
        "query_association": "tuple4_exact_from_candidate_current_row",
        "runtime_manifest_sha256": PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
        "selected_reads_sha256": PP16_FEATURE2_SELECTED_READS_SHA256,
    }
    for field, expected in exact_scalars.items():
        if getattr(graph, field) != expected:
            violations.append(f"feature2 prefill {field} drifted")
    try:
        validate_feature2_prefill_inputs(graph.inputs)
    except BenchmarkValidationError as error:
        violations.append(str(error))
    if graph.inputs.chunks != _expected_chunks():
        violations.append("feature2 prefill chunk intervals/tails drifted")
    if sum(graph.valid_rows_by_chunk) != graph.context_length:
        violations.append("feature2 prefill chunks do not cover all candidate rows")
    if len(graph.selected_reads) != 78 or (
        _selected_reads_sha256(graph.selected_reads)
        != PP16_FEATURE2_SELECTED_READS_SHA256
    ):
        violations.append("feature2 prefill selected range/hash binding drifted")
    try:
        selected = _selected_names(graph.selected_reads)
    except BenchmarkValidationError as error:
        violations.append(str(error))
        selected = frozenset()
    if selected != _EXPECTED_WEIGHTS:
        violations.append("feature2 prefill selected weight names drifted")

    expected_roles = _weight_roles()
    if graph.weight_roles != expected_roles:
        violations.append("feature2 prefill weight-role contract drifted")
    role_names = {role.name for role in graph.weight_roles}
    role_weights = [name for role in graph.weight_roles for name in role.weights]
    if frozenset(role_weights) != _EXPECTED_WEIGHTS or (
        len(role_weights) != len(set(role_weights))
    ):
        violations.append("feature2 prefill does not assign 39 weights once")

    tensors = {tensor.name: tensor for tensor in graph.tensors}
    if len(tensors) != len(graph.tensors) or graph.tensors != _tensors(graph.inputs):
        violations.append("feature2 prefill tensor/version contract drifted")
    forbidden_names = (
        "accepted_topk",
        "oracle_topk",
        "legacy_topk",
        "expected_topk",
    )
    for tensor in graph.tensors:
        lowered = tensor.name.lower()
        if any(marker in lowered for marker in forbidden_names):
            violations.append("feature2 prefill consumes a model oracle intermediate")
        if tensor.shape == (PP16_FEATURE2_CONTEXT_LENGTH, 6144):
            violations.append("feature2 prefill materializes full hidden history")
        if tensor.shape and tensor.shape[0] == 32 and 6144 in tensor.shape:
            violations.append("feature2 prefill contains batch-32 hidden rows")
        if tensor.storage == "ephemeral" and 6144 in tensor.shape and (
            tensor.shape[0] > PP16_FEATURE2_PREFILL_CHUNK
        ):
            violations.append("feature2 prefill hidden chunk exceeds 2,048 rows")

    if graph.nodes != _nodes(graph.inputs):
        violations.append("feature2 prefill causal node/cache contract drifted")
    produced = {
        tensor.name
        for tensor in graph.tensors
        if tensor.storage == "authenticated_source"
    }
    consumers: dict[str, list[str]] = {}
    for node in graph.nodes:
        if node.devices != PP16_FEATURE2_LOCAL_GROUP:
            violations.append(f"feature2 node {node.name} leaves LP2 group")
        if node.weight_role is not None and node.weight_role not in role_names:
            violations.append(f"feature2 node {node.name} has unknown weight role")
        for input_name in node.inputs:
            consumers.setdefault(input_name, []).append(node.name)
            if input_name not in produced:
                violations.append(
                    f"feature2 node {node.name} lacks causal input {input_name}"
                )
        for output_name in node.outputs:
            if output_name in produced or output_name not in tensors:
                violations.append(
                    f"feature2 node {node.name} output ownership drifted"
                )
            produced.add(output_name)
        lowered = f"{node.name} {node.operation}".lower()
        if any(marker in lowered for marker in forbidden_names):
            violations.append("feature2 prefill has a model oracle operation")
        if any(
            marker in lowered
            for marker in (
                "layer1_attention_output",
                "layer1_mlp",
                "layer1_dense",
                "layer2",
                "logits",
                "sampling",
            )
        ):
            violations.append("feature2 prefill crosses the admitted stop boundary")
    for tensor_name in produced:
        if tensor_name not in consumers and tensor_name not in graph.terminal_outputs:
            violations.append(
                f"feature2 prefill produced nonterminal {tensor_name!r} is dead"
            )
    node_by_name = {node.name: node for node in graph.nodes}
    for index, chunk in enumerate(graph.inputs.chunks):
        suffix = str(index)
        carried_name = f"layer1_carried_{suffix}"
        liveness_name = f"layer1_carried_liveness_{suffix}"
        expected_consumers = [liveness_name]
        if index == 3:
            expected_consumers.append("layer1_current_row")
        if sorted(consumers.get(carried_name, ())) != sorted(expected_consumers):
            violations.append(
                f"feature2 carried liveness for chunk {index} is partial or dead"
            )
        liveness_node = node_by_name.get(liveness_name)
        expected_inputs = (
            carried_name,
            f"position_chunk_{suffix}",
            f"layer1_carried_liveness_digest_{suffix}",
        )
        expected_operation = (
            "owner_local_ordered_dual_u32_digest_all_valid_"
            f"bf16_values_[{chunk.start},{chunk.stop})"
        )
        if liveness_node is None or (
            liveness_node.inputs != expected_inputs
            or liveness_node.operation != expected_operation
            or liveness_node.outputs
            != (f"layer1_carried_liveness_digest_{index + 1}",)
        ):
            violations.append(
                f"feature2 carried liveness witness for chunk {index} drifted"
            )
    if any(output not in produced for output in graph.terminal_outputs):
        violations.append("feature2 prefill terminal output is not candidate-produced")
    expected_terminal = (
        "candidate_event1_positions",
        "candidate_event1_valid_counts",
        "candidate_event1_scores",
        "candidate_current_carried",
        "candidate_layer1_attention_query",
        "candidate_layer1_kv_a",
        "layer0_kv_cache_4",
        "layer0_index_cache_4",
        "layer1_index_cache_4",
        "layer1_carried_liveness_digest_4",
    )
    if graph.terminal_outputs != expected_terminal:
        violations.append("feature2 prefill terminal/state boundary drifted")
    if _graph_sha256(graph) != graph.graph_sha256:
        violations.append("feature2 prefill graph SHA-256 drifted")
    if violations:
        raise BenchmarkValidationError(
            f"feature2 prefill graph rejected: {violations}"
        )
    return {
        "context_length": graph.context_length,
        "graph_sha256": graph.graph_sha256,
        "local_group": list(graph.local_group),
        "node_count": len(graph.nodes),
        "selected_read_count": len(graph.selected_reads),
        "selected_weight_count": len(_EXPECTED_WEIGHTS),
        "valid_rows_by_chunk": list(graph.valid_rows_by_chunk),
    }
