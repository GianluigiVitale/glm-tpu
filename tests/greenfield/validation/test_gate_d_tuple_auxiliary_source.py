from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RMS_SOURCE = REPO_ROOT / "glm_tpu/greenfield/kernels/reference/rmsnorm.py"
LAYER_SOURCE = REPO_ROOT / "glm_tpu/greenfield/kernels/layer.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _named_tuple(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(matches) == 1
    result = matches[0]
    assert [ast.unparse(base) for base in result.bases] == ["NamedTuple"]
    return result


def _assignments(function: ast.FunctionDef) -> dict[str, ast.AST]:
    return {
        target.id: statement.value
        for statement in function.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance((target := statement.targets[0]), ast.Name)
    }


class _Rename(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802
        if node.id == self.old:
            return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)
        return node


def _canonical_expression(value: ast.AST, *, rename: tuple[str, str] | None = None) -> str:
    clone = ast.parse(ast.unparse(value), mode="eval").body
    if rename is not None:
        clone = _Rename(*rename).visit(clone)
        ast.fix_missing_locations(clone)
    return ast.dump(clone, annotate_fields=True, include_attributes=False)


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    value: ast.AST = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def test_tuple_auxiliary_source_binds_concrete_operators() -> None:
    tree = _tree(RMS_SOURCE)
    result_type = _named_tuple(tree, "FusedAddRmsNormAuxiliaryResult")
    assert [
        (statement.target.id, ast.unparse(statement.annotation))
        for statement in result_type.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    ] == [
        ("output", "jax.Array"),
        ("carried_residual", "jax.Array"),
        ("rms_input_fp32", "jax.Array"),
    ]
    accepted = _function(tree, "fused_add_rms_norm")
    candidate = _function(tree, "fused_add_rms_norm_with_auxiliary")
    calls = {_call_name(node) for node in ast.walk(candidate) if isinstance(node, ast.Call)}
    assert calls == {
        "FusedAddRmsNormAuxiliaryResult",
        "ValueError",
        "astype",
        "hidden_states.astype",
        "isinstance",
        "jnp.float32",
        "jnp.issubdtype",
        "jnp.mean",
        "lax.rsqrt",
        "lax.square",
        "normalized.astype",
        "residual.astype",
        "rms_input_fp32.astype",
    }
    accepted_assignments = _assignments(accepted)
    candidate_assignments = _assignments(candidate)
    assert set(accepted_assignments) == {
        "activation_dtype",
        "summed",
        "carried_residual",
        "variance",
        "normalized",
        "output",
    }
    assert set(candidate_assignments) == {
        "activation_dtype",
        "rms_input_fp32",
        "carried_residual",
        "variance",
        "normalized",
        "output",
    }
    for accepted_name, candidate_name in (
        ("activation_dtype", "activation_dtype"),
        ("summed", "rms_input_fp32"),
        ("carried_residual", "carried_residual"),
        ("variance", "variance"),
        ("normalized", "normalized"),
        ("output", "output"),
    ):
        assert _canonical_expression(accepted_assignments[accepted_name]) == (
            _canonical_expression(
                candidate_assignments[candidate_name],
                rename=("rms_input_fp32", "summed"),
            )
        )
    returned = candidate.body[-1]
    assert isinstance(returned, ast.Return)
    assert isinstance(returned.value, ast.Call)
    assert _call_name(returned.value) == "FusedAddRmsNormAuxiliaryResult"
    assert [ast.unparse(item) for item in returned.value.args] == [
        "output",
        "carried_residual",
        "rms_input_fp32",
    ]
    source = ast.unparse(candidate)
    assert all(
        marker not in source
        for marker in ("device_get", "host_callback", "np.asarray", "pure_callback")
    )


def test_real_layer_callsite_is_default_off_and_device_returned() -> None:
    tree = _tree(LAYER_SOURCE)
    result_type = _named_tuple(tree, "StageLocalSplitLayerFp8AuxiliaryResult")
    assert [
        (statement.target.id, ast.unparse(statement.annotation))
        for statement in result_type.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    ] == [
        ("result", "StageLocalSplitLayerFp8Result"),
        ("input_rms_fp32", "Any"),
    ]
    layer = _function(tree, "stage_local_transformer_layer_fp8_split_mapped")
    keyword_defaults = dict(
        zip(
            (argument.arg for argument in layer.args.kwonlyargs),
            layer.args.kw_defaults,
            strict=True,
        )
    )
    default = keyword_defaults["retain_input_rms_auxiliary"]
    assert isinstance(default, ast.Constant) and default.value is False
    candidate_calls = [
        node
        for node in ast.walk(layer)
        if isinstance(node, ast.Call)
        and _call_name(node) == "fused_add_rms_norm_with_auxiliary"
    ]
    assert len(candidate_calls) == 1
    call = candidate_calls[0]
    assert [ast.unparse(item) for item in call.args] == [
        "hidden_states",
        "residual",
        "input_norm_weight",
    ]
    assert [(item.arg, ast.unparse(item.value)) for item in call.keywords] == [
        ("epsilon", "rms_norm_epsilon")
    ]
    candidate_branches = [
        node
        for node in ast.walk(layer)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "retain_input_rms_auxiliary"
    ]
    assert len(candidate_branches) == 2
    first_branch = candidate_branches[0]
    expected_enabled = ast.parse(
        """rms_candidate = fused_add_rms_norm_with_auxiliary(
    hidden_states, residual, input_norm_weight, epsilon=rms_norm_epsilon
)
normalized_input = rms_candidate.output
combined_residual = rms_candidate.carried_residual
input_rms_fp32 = rms_candidate.rms_input_fp32
"""
    ).body
    expected_default = ast.parse(
        """normalized_input, combined_residual = fused_add_rms_norm(
    hidden_states, residual, input_norm_weight, epsilon=rms_norm_epsilon
)
input_rms_fp32 = None
"""
    ).body
    assert ast.dump(ast.Module(body=first_branch.body, type_ignores=[])) == ast.dump(
        ast.Module(body=expected_enabled, type_ignores=[])
    )
    assert ast.dump(ast.Module(body=first_branch.orelse, type_ignores=[])) == ast.dump(
        ast.Module(body=expected_default, type_ignores=[])
    )
    wrapper_returns = [
        node
        for node in ast.walk(layer)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value) == "StageLocalSplitLayerFp8AuxiliaryResult"
    ]
    assert len(wrapper_returns) == 1
    assert [ast.unparse(item) for item in wrapper_returns[0].value.args] == [
        "result",
        "input_rms_fp32",
    ]
    return_branch = next(
        node
        for node in candidate_branches
        if any(item is wrapper_returns[0] for item in node.body)
    )
    assert [type(item) for item in return_branch.body] == [ast.Assert, ast.Return]
    assert ast.unparse(return_branch.body[0].test) == "input_rms_fp32 is not None"
    assert all(
        marker not in ast.unparse(layer)
        for marker in ("device_get", "host_callback", "np.asarray", "pure_callback")
    )
