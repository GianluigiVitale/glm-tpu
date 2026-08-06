"""Compact Pallas kernels for the greenfield TPU execution path."""

from .dsa import DsaScoreConfig, dsa_scores_kernel, dsa_scores_pallas
from .fp8_matmul import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_block_up_gate,
    fp8_fused_block_swiglu,
    fp8_fused_selected_moe,
    fp8_selected_swiglu_down,
    fp8_selected_up_gate,
)
from .sparse_attention import (
    SparseMlaConfig,
    stage_local_sparse_mla_kernel,
    stage_local_sparse_mla_pallas,
)
from .topk import (
    DsaTopKConfig,
    local_topk_candidates_kernel,
    local_topk_candidates_pallas,
    merge_topk_candidates_kernel,
    merge_topk_candidates_pallas,
)

__all__ = [
    "DsaScoreConfig",
    "DsaTopKConfig",
    "Fp8BlockMatmulConfig",
    "SparseMlaConfig",
    "dsa_scores_kernel",
    "dsa_scores_pallas",
    "fp8_block_matmul",
    "fp8_block_up_gate",
    "fp8_fused_block_swiglu",
    "fp8_fused_selected_moe",
    "fp8_selected_swiglu_down",
    "fp8_selected_up_gate",
    "local_topk_candidates_kernel",
    "local_topk_candidates_pallas",
    "merge_topk_candidates_kernel",
    "merge_topk_candidates_pallas",
    "stage_local_sparse_mla_kernel",
    "stage_local_sparse_mla_pallas",
]
