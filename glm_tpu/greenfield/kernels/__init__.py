"""Greenfield model kernels.

The package is intentionally independent of ``tpu_inference``.  The legacy
repository may produce comparison artifacts, but it is never an execution
dependency of this tree.
"""

from .layer import (
    AttentionFp8Weights,
    AttentionProjectionBackend,
    DenseFp8Weights,
    DsaFp8Weights,
    MoeFp8Weights,
    StageLocalLayerFp8Result,
    stage_local_transformer_layer_fp8_mapped,
)
from .ws32_layer import (
    Ws32AttentionLayerResult,
    Ws32AttentionResult,
    Ws32AttentionWeights,
    Ws32DsaResult,
    Ws32DsaWeights,
    Ws32DenseWeights,
    Ws32MoeWeights,
    Ws32MlpResult,
    Ws32PreparedAttention,
    Ws32QkvAWeights,
    Ws32TransformerLayerResult,
    ws32_attention_layer_mapped,
    ws32_dsa_mapped,
    ws32_index_share_attention_mapped,
    ws32_prepare_attention_mapped,
    ws32_mlp_mapped,
    ws32_transformer_layer_mapped,
)
from .ws32_io import (
    Ws32EmbeddingResult,
    Ws32GreedySampleResult,
    ws32_embedding_mapped,
    ws32_final_sample_mapped,
    ws32_greedy_sample_mapped,
    ws32_logits_mapped,
)

__all__ = (
    "AttentionFp8Weights",
    "AttentionProjectionBackend",
    "DenseFp8Weights",
    "DsaFp8Weights",
    "MoeFp8Weights",
    "StageLocalLayerFp8Result",
    "Ws32AttentionLayerResult",
    "Ws32AttentionResult",
    "Ws32AttentionWeights",
    "Ws32DsaResult",
    "Ws32DsaWeights",
    "Ws32DenseWeights",
    "Ws32MoeWeights",
    "Ws32MlpResult",
    "Ws32PreparedAttention",
    "Ws32QkvAWeights",
    "Ws32TransformerLayerResult",
    "Ws32EmbeddingResult",
    "Ws32GreedySampleResult",
    "stage_local_transformer_layer_fp8_mapped",
    "ws32_attention_layer_mapped",
    "ws32_dsa_mapped",
    "ws32_embedding_mapped",
    "ws32_final_sample_mapped",
    "ws32_greedy_sample_mapped",
    "ws32_index_share_attention_mapped",
    "ws32_prepare_attention_mapped",
    "ws32_logits_mapped",
    "ws32_mlp_mapped",
    "ws32_transformer_layer_mapped",
)
