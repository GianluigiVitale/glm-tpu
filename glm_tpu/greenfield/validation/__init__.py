"""Independent greenfield correctness artifacts and comparisons."""

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
    inspect_layer0_dsa_association_input,
)
from .prompt_index_cache import (
    LegacyPromptKeyInternalConfig,
    LegacyPromptIndexCacheConfig,
    capture_legacy_prompt_index_cache,
    compare_prompt_key_internal_states,
    compare_prompt_index_key_bits,
    inspect_legacy_prompt_key_internal_capture,
    inspect_legacy_prompt_index_cache,
    inspect_prompt_key_internal_capture_artifact,
    validate_prompt_index_key_association_hlo,
    validate_prompt_index_key_probe_hlo,
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

__all__ = (
    "GateCOracleConfig",
    "OneLayerOracleConfig",
    "AcceptedGreenfieldDsaInternalComparisonConfig",
    "LegacyResidualComparisonConfig",
    "LegacyDsaInternalCaptureConfig",
    "LegacyDsaInternalComparisonConfig",
    "LegacyPromptKeyInternalConfig",
    "LegacyPromptIndexCacheConfig",
    "ShortContextOracleConfig",
    "ShortContextDsaOracleConfig",
    "ShortContextLogprobOracleConfig",
    "capture_gate_c_oracle",
    "capture_one_layer_oracle",
    "capture_legacy_prompt_index_cache",
    "compare_prompt_key_internal_states",
    "compare_prompt_index_key_bits",
    "compare_accepted_greenfield_dsa_internal_observation",
    "compare_legacy_residuals",
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
    "inspect_layer0_dsa_association_input",
    "inspect_one_layer_oracle",
    "inspect_legacy_prompt_index_cache",
    "validate_prompt_index_key_association_hlo",
    "validate_prompt_index_key_probe_hlo",
    "inspect_short_context_oracle",
    "inspect_short_context_dsa_oracle",
    "inspect_short_context_logprob_oracle",
    "normalize_sample_logprobs",
)
