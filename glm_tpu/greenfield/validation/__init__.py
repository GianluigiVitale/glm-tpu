"""Independent greenfield correctness artifacts and comparisons."""

from .attention_update import (
    AcceptedAttentionUpdateCaptureConfig,
    AttentionUpdateComparisonConfig,
    capture_accepted_attention_update,
    compare_attention_update_candidates,
)
from .dense_boundary import (
    AcceptedDenseBoundaryCaptureConfig,
    DenseBoundaryComparisonConfig,
    capture_accepted_dense_boundary,
    compare_dense_boundary_candidate,
)
from .dense_input import (
    AcceptedDenseInputCaptureConfig,
    DenseInputComparisonConfig,
    capture_accepted_dense_input,
    compare_dense_input_candidate,
)
from .attention_output_operand import (
    AcceptedAttentionProjectionCaptureConfig,
    AcceptedAttentionOutputCaptureConfig,
    AttentionProjectionComparisonConfig,
    AttentionOutputComparisonConfig,
    capture_accepted_attention_projection_operands,
    capture_accepted_attention_output_operand,
    compare_attention_projection_operands,
    compare_attention_output_operands,
)

from .gate_c_oracle import (
    GateCOracleConfig,
    capture_gate_c_oracle,
    inspect_gate_c_oracle,
)
from .one_layer_oracle import (
    OneLayerOracleConfig,
    capture_one_layer_oracle,
    inspect_one_layer_oracle,
)
from .legacy_residuals import (
    LegacyResidualComparisonConfig,
    compare_legacy_residuals,
)
from .legacy_main_cache import (
    LegacyMainCacheComparisonConfig,
    compare_legacy_layer0_main_cache,
)
from .legacy_dsa_internals import (
    AcceptedGreenfieldDsaInternalComparisonConfig,
    LegacyDsaInternalCaptureConfig,
    LegacyDsaInternalComparisonConfig,
    compare_accepted_greenfield_dsa_internal_observation,
    compare_legacy_dsa_internals,
    inspect_legacy_dsa_internal_capture,
)
from .layer0_dsa_association import (
    compare_dsa_association_scores,
    inspect_distributed_q_a_norm_artifact,
    inspect_greenfield_layer0_dsa_internal_observation,
    inspect_greenfield_layer0_dsa_selected_observation,
    inspect_layer0_dsa_association_input,
    pack_stage_local_index_keys,
    stitch_stage_local_scores,
)
from .prompt_index_cache import (
    LegacyPromptKeyInternalConfig,
    LegacyPromptIndexCacheConfig,
    capture_legacy_prompt_index_cache,
    compare_prompt_key_internal_states,
    compare_prompt_index_key_bits,
    compare_prompt_projection_input,
    inspect_legacy_prompt_key_internal_capture,
    inspect_legacy_prompt_index_cache,
    inspect_prompt_key_internal_capture_artifact,
    validate_prompt_index_key_association_hlo,
    validate_prompt_index_key_probe_hlo,
    validate_prompt_projection_input_hlo,
)
from .short_context_oracle import (
    ShortContextOracleConfig,
    capture_short_context_oracle,
    inspect_short_context_oracle,
)
from .short_context_dsa_oracle import (
    ShortContextDsaOracleConfig,
    capture_short_context_dsa_oracle,
    compare_short_context_dsa_oracles,
    inspect_short_context_dsa_oracle,
)
from .short_context_logprob_oracle import (
    ShortContextLogprobOracleConfig,
    capture_short_context_logprob_oracle,
    inspect_short_context_logprob_oracle,
    normalize_sample_logprobs,
)
from .strategy_nd_dense_replay import validate_strategy_nd_dense_replay

__all__ = (
    "AcceptedAttentionUpdateCaptureConfig",
    "AcceptedAttentionOutputCaptureConfig",
    "AcceptedAttentionProjectionCaptureConfig",
    "AttentionOutputComparisonConfig",
    "AttentionProjectionComparisonConfig",
    "AttentionUpdateComparisonConfig",
    "AcceptedDenseBoundaryCaptureConfig",
    "AcceptedDenseInputCaptureConfig",
    "DenseBoundaryComparisonConfig",
    "DenseInputComparisonConfig",
    "GateCOracleConfig",
    "OneLayerOracleConfig",
    "AcceptedGreenfieldDsaInternalComparisonConfig",
    "LegacyResidualComparisonConfig",
    "LegacyMainCacheComparisonConfig",
    "LegacyDsaInternalCaptureConfig",
    "LegacyDsaInternalComparisonConfig",
    "LegacyPromptKeyInternalConfig",
    "LegacyPromptIndexCacheConfig",
    "ShortContextOracleConfig",
    "ShortContextDsaOracleConfig",
    "ShortContextLogprobOracleConfig",
    "capture_gate_c_oracle",
    "capture_accepted_attention_output_operand",
    "capture_accepted_attention_projection_operands",
    "capture_accepted_attention_update",
    "capture_accepted_dense_boundary",
    "capture_accepted_dense_input",
    "capture_one_layer_oracle",
    "capture_legacy_prompt_index_cache",
    "compare_prompt_key_internal_states",
    "compare_prompt_index_key_bits",
    "compare_prompt_projection_input",
    "compare_attention_output_operands",
    "compare_attention_projection_operands",
    "compare_attention_update_candidates",
    "compare_dense_boundary_candidate",
    "compare_dense_input_candidate",
    "compare_accepted_greenfield_dsa_internal_observation",
    "compare_legacy_residuals",
    "compare_legacy_layer0_main_cache",
    "compare_legacy_dsa_internals",
    "inspect_legacy_dsa_internal_capture",
    "inspect_legacy_prompt_key_internal_capture",
    "inspect_prompt_key_internal_capture_artifact",
    "capture_short_context_oracle",
    "capture_short_context_dsa_oracle",
    "compare_short_context_dsa_oracles",
    "capture_short_context_logprob_oracle",
    "compare_dsa_association_scores",
    "inspect_gate_c_oracle",
    "inspect_distributed_q_a_norm_artifact",
    "inspect_greenfield_layer0_dsa_internal_observation",
    "inspect_greenfield_layer0_dsa_selected_observation",
    "inspect_layer0_dsa_association_input",
    "inspect_one_layer_oracle",
    "inspect_legacy_prompt_index_cache",
    "validate_prompt_index_key_association_hlo",
    "validate_prompt_index_key_probe_hlo",
    "validate_prompt_projection_input_hlo",
    "inspect_short_context_oracle",
    "inspect_short_context_dsa_oracle",
    "inspect_short_context_logprob_oracle",
    "normalize_sample_logprobs",
    "pack_stage_local_index_keys",
    "stitch_stage_local_scores",
    "validate_strategy_nd_dense_replay",
)
