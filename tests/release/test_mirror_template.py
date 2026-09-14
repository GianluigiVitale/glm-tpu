"""The staged mirror adds one worktree without changing installed behavior."""

from hashlib import sha256
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "scripts/release/sync_glm_repositories.sh"
ORIGINAL_SHA = "8303e37d4f3732b66fa896ac0cf8df0903640a00738166838011f1120610cf6b"
RELEASE_PAIR = b'  "/home/gianl/glm-tpu-release:repos/glm-tpu-release"\n'


def test_template_is_only_reviewed_release_pair_addition():
    data = TEMPLATE.read_bytes()
    assert data.count(RELEASE_PAIR) == 1
    assert sha256(data.replace(RELEASE_PAIR, b"")).hexdigest() == ORIGINAL_SHA
    subprocess.run(["bash", "-n", str(TEMPLATE)], check=True)
