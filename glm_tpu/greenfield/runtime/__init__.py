"""Device-resident pipeline execution contracts."""

from .pipeline import (
    CompiledPipelineSkeleton,
    PipelineSkeletonConfig,
    build_pipeline_skeleton,
    validate_pipeline_skeleton_hlo,
)
from .decoder import (
    DecoderStepConfig,
    DecoderStepProgram,
    build_decoder_step_program,
    validate_decoder_step_hlo,
)
from .prefill import (
    TeacherForcedPrefillProgram,
    build_teacher_forced_prefill_program,
    validate_teacher_forced_prefill_hlo,
)

__all__ = [
    "CompiledPipelineSkeleton",
    "PipelineSkeletonConfig",
    "DecoderStepConfig",
    "DecoderStepProgram",
    "TeacherForcedPrefillProgram",
    "build_decoder_step_program",
    "build_teacher_forced_prefill_program",
    "validate_teacher_forced_prefill_hlo",
    "validate_decoder_step_hlo",
    "build_pipeline_skeleton",
    "validate_pipeline_skeleton_hlo",
]
