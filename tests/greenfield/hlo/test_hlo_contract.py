from __future__ import annotations

from collections import Counter

from glm_tpu.optimized.hlo_contract import parse_hlo_module


GOOD_HLO = r'''HloModule jit_pipeline, replica_count=1, num_partitions=8

add_region {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT sum = bf16[] add(a, b)
}

decode_repeated_layer {
  x = bf16[1,6144]{1,0} parameter(0)
  local = bf16[1,6144]{1,0} all-reduce(x), channel_id=1, replica_groups={{0,1,2,3},{4,5,6,7}}, use_global_device_ids=true, to_apply=add_region, metadata={op_name="decode/layer/moe_combine" source_file="model/moe.py" source_line=91 source_stack="decoder.layer"}
  routed = bf16[1,2048]{1,0} slice(local), slice={[0:1],[0:2048]}, metadata={op_name="decode/layer/routed" source_file="model/moe.py" source_line=92}
  ROOT moved = bf16[1,6144]{1,0} collective-permute(local), channel_id=2, source_target_pairs={{0,4},{4,0},{1,5},{5,1},{2,6},{6,2},{3,7},{7,3}}, metadata={op_name="decode/pipeline/transport" source_file="runtime/transport.py" source_line=20}
}

ENTRY main {
  input = bf16[1,6144]{1,0} parameter(0)
  ROOT call = bf16[1,6144]{1,0} call(input), to_apply=decode_repeated_layer
}
'''


def test_parser_extracts_physical_collective_contract() -> None:
    module = parse_hlo_module(GOOD_HLO)
    assert module.name == "jit_pipeline"
    assert module.num_partitions == 8
    assert len(module.collectives) == 2
    reduction, permute = module.collectives
    assert reduction.replica_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert reduction.maximum_group_size == 4
    assert reduction.result_shapes[0].dimensions == (1, 6144)
    assert reduction.operand_shapes[0].dimensions == (1, 6144)
    assert reduction.source_file == "model/moe.py"
    assert reduction.source_line == 91
    assert reduction.source_stack == "decoder.layer"
    assert permute.source_target_pairs[0] == (0, 4)


def test_parser_preserves_tuple_operands_after_index_comments() -> None:
    text = r'''HloModule tuple_comments

ENTRY main {
  a = bf16[1] parameter(0)
  b = bf16[1] parameter(1)
  c = bf16[1] parameter(2)
  ROOT result = (bf16[1], bf16[1], bf16[1]) tuple(a, /*index=1*/b, c)
}
'''
    root = parse_hlo_module(text).instructions[-1]
    assert root.operand_names == ("a", "b", "c")


def test_parser_expands_current_jax_mesh_replica_groups() -> None:
    mesh_hlo = GOOD_HLO.replace(
        "replica_groups={{0,1,2,3},{4,5,6,7}}",
        "replica_groups=mesh['stage'=2,'local'=4] {'local'}",
    )
    reduction = parse_hlo_module(mesh_hlo).collectives[0]
    assert reduction.replica_groups == ((0, 1, 2, 3), (4, 5, 6, 7))

    strided_hlo = GOOD_HLO.replace(
        "replica_groups={{0,1,2,3},{4,5,6,7}}",
        "replica_groups=mesh['stage'=2,'local'=4] {'stage'}",
    )
    strided = parse_hlo_module(strided_hlo).collectives[0]
    assert strided.replica_groups == (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )


def test_tuple_collective_shapes_and_operands_are_preserved() -> None:
    text = r'''HloModule tuple_reduce, replica_count=1, num_partitions=4

decode_layer {
  x = bf16[1,6144]{1,0} parameter(0)
  y = f32[1,8]{1,0} parameter(1)
  ROOT reduced = (bf16[1,6144]{1,0}, f32[1,8]{1,0}) all-reduce(x, y), channel_id=3, replica_groups={{0,1,2,3}}, use_global_device_ids=true, metadata={op_name="decode/layer/tuple"}
}
'''
    instruction = parse_hlo_module(text).collectives[0]
    assert [shape.dimensions for shape in instruction.result_shapes] == [
        (1, 6144),
        (1, 8),
    ]
    assert [shape.dtype for shape in instruction.operand_shapes] == ["bf16", "f32"]


def test_operand_shape_resolution_is_computation_local() -> None:
    text = r'''HloModule duplicate_names

first {
  x = f32[99]{0} parameter(0)
  ROOT copy = f32[99]{0} copy(x)
}

decode_layer {
  x = bf16[1,6144]{1,0} parameter(0)
  ROOT reduced = bf16[1,6144]{1,0} all-reduce(x), replica_groups={{0,1,2,3}}, use_global_device_ids=true
}
'''
    instruction = parse_hlo_module(text).collectives[0]
    assert instruction.operand_shapes == instruction.result_shapes


def test_async_collective_start_counts_once_and_done_is_not_a_collective() -> None:
    # The TPU graph admission (optimized.admission.inspect_research_hlo) counts
    # module.collectives by opcode and relies on this normalization of async forms.
    async_hlo = GOOD_HLO.replace(
        "all-reduce(x)", "all-reduce-start(x)"
    ).replace(
        "  routed = bf16[1,2048]{1,0} slice(local),",
        "  completed = bf16[1,6144]{1,0} all-reduce-done(local)\n"
        "  routed = bf16[1,2048]{1,0} slice(completed),",
    )
    assert "all-reduce-start(x)" in async_hlo and "all-reduce-done(local)" in async_hlo
    module = parse_hlo_module(async_hlo)
    reduction, permute = module.collectives
    assert reduction.opcode == "all-reduce"
    assert reduction.raw_opcode == "all-reduce-start"
    assert reduction.replica_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert permute.opcode == "collective-permute"
    (done,) = (op for op in module.instructions if op.raw_opcode == "all-reduce-done")
    assert not done.is_collective
    assert Counter(op.opcode for op in module.collectives) == {
        "all-reduce": 1,
        "collective-permute": 1,
    }
