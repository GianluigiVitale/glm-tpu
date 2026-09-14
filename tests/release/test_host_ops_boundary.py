"""Host extraction preserves guards without importing the campaign controller."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
BASE = "b667f00f1ae48c8ff37e92500550c1395d74c66d"
OLD = "scripts/greenfield/launch_ws32_native_benchmark.py"
NEW = "scripts/release/ws32_host_ops.py"
NAMES = (
    "ssh",
    "persist",
    "reviewed_branch",
    "sync_command",
    "source_preflight",
    "validate_attach",
    "markers",
    "publication_state",
    "observe_originals",
)


@pytest.fixture(scope="module")
def source_trees():
    before = subprocess.check_output(
        ["git", "show", BASE + ":" + OLD], cwd=REPO, text=True
    )
    return ast.parse(before), ast.parse((REPO / NEW).read_text())


@pytest.mark.parametrize("name", NAMES)
def test_extracted_function_ast_is_original(source_trees, name):
    def function(tree):
        return next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        )

    assert ast.dump(function(source_trees[0])) == ast.dump(function(source_trees[1]))


@pytest.mark.parametrize("name", ("PYTHON", "BRANCH", "PUBLICATION_REMOTE"))
def test_host_constants_and_remote_program_are_original(source_trees, name):
    def value(tree):
        return ast.literal_eval(
            next(
                n.value
                for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            )
        )

    assert value(source_trees[0]) == value(source_trees[1])


def test_campaign_launcher_is_not_a_release_entrypoint():
    assert not (REPO / OLD).exists()
    assert (REPO / NEW).is_file()


def test_user_controller_does_not_import_campaign_launcher():
    program = """
import json, sys
from scripts.release import launch_ws32_user_request as user
print(json.dumps({'shared': user.shared.__name__,
                  'campaign_loaded': 'scripts.greenfield.launch_ws32_native_benchmark' in sys.modules}))
"""
    result = subprocess.check_output(
        [sys.executable, "-c", program],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        text=True,
    )
    assert json.loads(result) == {
        "shared": "scripts.release.ws32_host_ops",
        "campaign_loaded": False,
    }
