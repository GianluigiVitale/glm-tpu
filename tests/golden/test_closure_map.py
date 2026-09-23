"""G6/G7 comparison through the reviewed rename table (``tools/equivalence/closure_map.toml``)."""
from tools.equivalence import closure_map
from tools.equivalence.closure_map import ClosureMap, compare_functions, compare_modules, layering_entry


def test_committed_table_parses():
    closure_map.load()


def test_module_moves_map_to_recorded_names_and_closures_may_only_shrink():
    recorded = ["glm_tpu.optimized", "glm_tpu.optimized.fp8_routed_experts", "glm_tpu.optimized.runtime"]
    moved = ["glm_tpu.optimized", "glm_tpu.optimized.routed", "glm_tpu.optimized.routed.fp8_experts"]
    table = ClosureMap(modules={"glm_tpu.optimized.routed.fp8_experts": "glm_tpu.optimized.fp8_routed_experts"})
    result = compare_modules(recorded, moved, table, is_package=lambda name: name == "glm_tpu.optimized.routed")
    assert result["added"] == []  # the new parent package of a mapped module is allowed
    assert result["removed"] == ["glm_tpu.optimized.runtime"]  # shrinking is allowed (reported)
    assert compare_modules(recorded, moved, ClosureMap(), is_package=lambda name: True)["added"] == [
        "glm_tpu.optimized.routed", "glm_tpu.optimized.routed.fp8_experts"]


def test_new_modules_fail_unless_reviewed():
    recorded = ["glm_tpu.optimized.runtime"]
    grown = recorded + ["bench.provenance"]
    assert compare_modules(recorded, grown, ClosureMap())["added"] == ["bench.provenance"]
    assert compare_modules(recorded, grown, ClosureMap(added={"bench.provenance": "S9 reviewed"}))["added"] == []


def test_package_prefix_mapping():
    table = ClosureMap(modules={"glm_tpu.engine.": "glm_tpu.optimized."})
    assert table.module("glm_tpu.engine.request") == "glm_tpu.optimized.request"
    assert table.function("glm_tpu.engine.runtime:Runner.generate") == "glm_tpu.optimized.runtime:Runner.generate"
    assert table.module("glm_tpu.engineering") == "glm_tpu.engineering"


def test_function_renames_additions_and_removals():
    recorded = ["m:_dot_f32", "m:keep", "f:bind_dependencies"]
    current = ["m:_bf16_dot", "m:keep", "p:build_program_set"]
    result = compare_functions(recorded, current, ClosureMap())
    assert result["added"] == ["m:_bf16_dot", "p:build_program_set"]
    assert result["removed"] == ["f:bind_dependencies", "m:_dot_f32"]
    table = ClosureMap(functions={"m:_bf16_dot": "m:_dot_f32"}, added={"p:build_program_set": "S2c"},
                       removed={"f:bind_dependencies": "S2d c3"})
    result = compare_functions(recorded, current, table)
    assert result["added"] == [] and result["removed"] == [] and result["declared_removed"] == ["f:bind_dependencies"]
    assert table.current_names("_dot_f32") == ("_dot_f32", "_bf16_dot")


def test_static_layering_entries_follow_moves():
    table = ClosureMap(modules={"glm_tpu.runner.compile": "glm_tpu.optimized.runtime"})
    entry = "glm_tpu/runner/compile.py: scripts.greenfield.ws32_compile_originals"
    assert layering_entry(entry, table) == "glm_tpu.optimized.runtime: scripts.greenfield.ws32_compile_originals"
    assert layering_entry("glm_tpu/optimized/__init__.py: scripts.x", ClosureMap()) == "glm_tpu.optimized: scripts.x"
