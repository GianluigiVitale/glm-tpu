"""Static contract of the chunk-0 legacy-geometry probe runner and probe script."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/greenfield/run_probe_layer1_prompt_chunk0_geometry.sh"
PROBE = REPO / "scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py"
DIGESTS = REPO / "docs/artifacts/gate-d-chunk0-probe-weight-digests.json"


def test_runner_pins_inputs_and_digests_and_executes_committed_bytes():
    runner = RUNNER.read_text()
    assert "readonly CHECKPOINT_INDEX_SHA=e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf" in runner
    digest_sha = sha256(DIGESTS.read_bytes()).hexdigest()
    assert f"readonly WEIGHT_DIGESTS_SHA={digest_sha}" in runner
    assert "readonly WEIGHT_DIGESTS_RELATIVE=docs/artifacts/gate-d-chunk0-probe-weight-digests.json" in runner
    # tag must be an approved composed tag; no self-derived default
    assert 'TAG=${GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG:-}' in runner
    assert 'must be an approved composed tag' in runner
    # origin authentication and detached worktree execution
    assert 'git -C "$WORKTREE" fetch -q origin "$BRANCH"' in runner
    assert '[[ $(git -C "$WORKTREE" rev-parse "origin/$BRANCH") == "$PIN" ]]' in runner
    assert 'git -C "$WORKTREE" worktree add -q --detach "$SOURCE" "$PIN"' in runner
    assert 'status --porcelain --ignored' in runner
    assert '"$PYTHON" "$SOURCE/scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py"' in runner
    assert 'PYTHONPATH="$SOURCE"' in runner
    assert '/usr/bin/env -i HOME=/home/gianl PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1' in runner
    assert '--weight-digests "$SOURCE/$WEIGHT_DIGESTS_RELATIVE"' in runner


def test_runner_vacancy_ledger_and_census_contract():
    runner = RUNNER.read_text()
    assert '"$GCLOUD" storage ls "$REMOTE_PREFIX/**" >"$RUN_DIR/remote_vacancy.txt"' in runner
    assert 'remote prefix is not vacant or vacancy could not be proven' in runner
    assert 'objects", "describe"' in runner
    assert '"generation": remote["generation"]' in runner
    assert 'validate_exact_remote_object_set(root, prefix, listing)' in runner
    assert 'terminal ledger checksum mismatch' in runner
    assert 'touch "$RUN_DIR/CENSUS_UNVERIFIED"' in runner
    assert 'strict_census failure_exit || true' not in runner
    # SUCCESS only after post census and probe success, before upload
    assert runner.index('strict_census post ||') < runner.index('touch "$RUN_DIR/SUCCESS"') < runner.index('storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/"')


def test_probe_binds_worktree_digests_and_m64_one_row_arm():
    probe = PROBE.read_text()
    assert "def _worktree_binding(" in probe and "rev-parse\", \"--git-common-dir\"" in probe
    assert 'if repo.resolve() != (RUN_ROOT / run_tag / "source").resolve():' in probe
    assert "def _verify_weight_digests(" in probe
    assert 'raise RuntimeError("checkpoint shard bytes drifted from the pinned digest")' in probe
    assert 'raise RuntimeError(f"checkpoint tensor drifted from the pinned digest: {key}")' in probe
    assert "layer1_keys_from_normalized" in probe
    assert '"one_row_block_m64_keys_vs_legacy_lanes"' in probe
    assert '"control_layer0_keys_vs_db518_mismatched_rows"' in probe
    record = json.loads(DIGESTS.read_text())
    assert record["artifact_kind"] == "gate_d_chunk0_probe_weight_digests"
    assert record["checkpoint_index_sha256"] == "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf"
    assert record["shard"]["filename"] == "model-00001-of-00141.safetensors"
    assert len(record["tensors"]) == 19
    for entry in record["tensors"].values():
        assert len(entry["sha256"]) == 64 and entry["shard"] == "model-00001-of-00141.safetensors"
