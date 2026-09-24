"""G6/G7 comparison through the reviewed rename table (``tools/equivalence/closure_map.toml``)."""

from tools.equivalence import closure_map
from tools.equivalence.closure_map import ClosureMap, compare_functions, compare_modules, layering_entry


def test_committed_table_parses():
    closure_map.load()


def test_module_moves_map_to_recorded_names_and_closures_may_only_shrink():
    recorded = ["pkg.old", "pkg.old.fp8_routed_experts", "pkg.old.runtime"]
    moved = ["pkg.old", "pkg.old.routed", "pkg.old.routed.fp8_experts"]
    table = ClosureMap(modules={"pkg.old.routed.fp8_experts": "pkg.old.fp8_routed_experts"})
    result = compare_modules(recorded, moved, table, is_package=lambda name: name == "pkg.old.routed")
    assert result["added"] == []  # the new parent package of a mapped module is allowed
    assert result["removed"] == ["pkg.old.runtime"]  # shrinking is allowed (reported)
    assert compare_modules(recorded, moved, ClosureMap(), is_package=lambda name: True)["added"] == [
        "pkg.old.routed",
        "pkg.old.routed.fp8_experts",
    ]


def test_new_modules_fail_unless_reviewed():
    recorded = ["pkg.old.runtime"]
    grown = recorded + ["bench.provenance"]
    assert compare_modules(recorded, grown, ClosureMap())["added"] == ["bench.provenance"]
    assert compare_modules(recorded, grown, ClosureMap(added={"bench.provenance": "S9 reviewed"}))["added"] == []


def test_package_prefix_mapping():
    table = ClosureMap(modules={"pkg.new.": "pkg.old."})
    assert table.module("pkg.new.request") == "pkg.old.request"
    assert table.function("pkg.new.runtime:Runner.generate") == "pkg.old.runtime:Runner.generate"
    assert table.module("pkg.newer") == "pkg.newer"


def test_function_renames_additions_and_removals():
    recorded = ["m:_dot_f32", "m:keep", "f:bind_dependencies"]
    current = ["m:_bf16_dot", "m:keep", "p:build_program_set"]
    result = compare_functions(recorded, current, ClosureMap())
    assert result["added"] == ["m:_bf16_dot", "p:build_program_set"]
    assert result["removed"] == ["f:bind_dependencies", "m:_dot_f32"]
    table = ClosureMap(
        functions={"m:_bf16_dot": "m:_dot_f32"},
        added={"p:build_program_set": "S2c"},
        removed={"f:bind_dependencies": "S2d c3"},
    )
    result = compare_functions(recorded, current, table)
    assert result["added"] == [] and result["removed"] == [] and result["declared_removed"] == ["f:bind_dependencies"]
    assert table.current_names("_dot_f32") == ("_dot_f32", "_bf16_dot")


def test_function_rename_maps_nested_qualnames():
    table = ClosureMap(functions={"m:build_new": "m:build_old"})
    assert table.function("m:build_new.<locals>.decode") == "m:build_old.<locals>.decode"
    assert table.function("m:build_new.<locals>.<lambda>") == "m:build_old.<locals>.<lambda>"
    assert table.function("m:build_newer") == "m:build_newer"  # a prefix of the name, not of the qualname
    recorded = ["m:build_old", "m:build_old.<locals>.decode"]
    current = ["m:build_new", "m:build_new.<locals>.decode"]
    result = compare_functions(recorded, current, table)
    assert result["added"] == [] and result["removed"] == []


def test_static_layering_entries_follow_moves():
    table = ClosureMap(modules={"pkg.new.compile": "pkg.old.runtime"})
    entry = "pkg/new/compile.py: tools.compile_originals"
    assert layering_entry(entry, table) == "pkg.old.runtime: tools.compile_originals"
    assert layering_entry("pkg/old/__init__.py: tools.x", ClosureMap()) == "pkg.old: tools.x"
