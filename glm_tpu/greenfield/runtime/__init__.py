"""Device-resident pipeline execution contracts."""

from .pipeline import (
    CompiledPipelineSkeleton,
    PipelineSkeletonConfig,
    build_pipeline_skeleton,
    validate_pipeline_skeleton_hlo,
)

__all__ = [
    "CompiledPipelineSkeleton",
    "PipelineSkeletonConfig",
    "build_pipeline_skeleton",
    "validate_pipeline_skeleton_hlo",
]
