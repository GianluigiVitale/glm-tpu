from __future__ import annotations

from tests.greenfield.model.test_schedule import _plan

from glm_tpu.greenfield.model import (
    build_decoder_state_layout,
    build_pipeline_schedule,
)


def test_short_context_state_has_one_global_padded_shape() -> None:
    plan = _plan()
    schedule = build_pipeline_schedule(plan)
    state = build_decoder_state_layout(
        plan, schedule, context_capacity=8192
    )
    assert state.padded_state_shape_per_device == {
        "kv": (12, 16, 128, 640),
        "index_keys": (5, 16, 128, 128),
        "selected_positions": (1, 2048),
    }
    assert [stage.full_indexer_count for stage in state.stages] == [5, 2, 3, 2, 2, 3, 2, 2]
    assert state.stages[0].padding_bytes == 0
    assert state.stages[1].padding_bytes > 0
    assert len(state.state_layout_hash) == 64


def test_target_context_state_reconciles_every_plan_assignment() -> None:
    plan = _plan()
    # The synthetic plan fixture uses placeholder memory fields. Replace it
    # with the checked-in production arithmetic before exercising the strict
    # target-context reconciliation.
    from dataclasses import replace
    from glm_tpu.greenfield.partitioning.memory_model import (
        MemoryPolicy,
        stage_runtime_memory,
    )

    policy = MemoryPolicy(target_context_length=plan.target_context_length)
    assignments = []
    for assignment in plan.stage_assignments:
        runtime = stage_runtime_memory(
            plan.geometry,
            layer_start=assignment.layer_start,
            layer_end_exclusive=assignment.layer_end_exclusive,
            local_parallel_size=plan.local_parallel_size,
            policy=policy,
        )
        assignments.append(
            replace(
                assignment,
                kv_bytes_at_target_context=runtime.kv_cache_bytes,
                dsa_state_bytes=runtime.dsa_state_bytes,
            )
        )
    plan = replace(plan, stage_assignments=tuple(assignments))
    schedule = build_pipeline_schedule(plan)
    state = build_decoder_state_layout(
        plan, schedule, context_capacity=plan.target_context_length
    )
    for assignment, layout in zip(plan.stage_assignments, state.stages, strict=True):
        assert layout.kv_cache_bytes == assignment.kv_bytes_at_target_context
        assert (
            layout.indexer_cache_bytes
            + layout.selected_index_bytes
            + layout.transport_bytes
            == assignment.dsa_state_bytes
        )
    assert state.maximum_padded_bytes_per_chip == 1_090_555_904
