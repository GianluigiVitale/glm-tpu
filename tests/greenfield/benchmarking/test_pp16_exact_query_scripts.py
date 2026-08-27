from __future__ import annotations

from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[3]


def test_pp16_exact_query_wrapper_is_bounded_and_default_off() -> None:
    wrapper = REPO / "scripts/greenfield/run_pp16_lp2_exact_query.sh"
    text = wrapper.read_text()
    completed = subprocess.run(
        ["bash", str(wrapper)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    assert "greenfield_ws32_layer0_dsa_association_20260816" in text
    assert "source_db=554" in text
    assert "--warmup 1 --iterations 3" in text
    assert "TPU_VISIBLE_DEVICES=0,1,2,3" in text
    assert "probe_pp16_lp2_exact_query.py" in text
    assert "runtime_feature" not in text
    assert "compile_short_decoder.py --" not in text
    assert "strict_census pre" in text
    assert "strict_census post" in text
    assert "results_ckpt.db" in text
    assert "gcloud storage cp --no-clobber \"$RUN_DIR/SUCCESS\"" in text


def test_pp16_exact_query_runner_pins_two_local_chunks() -> None:
    runner = REPO / "scripts/greenfield/probe_pp16_lp2_exact_query.py"
    text = runner.read_text()
    assert "jax.local_devices()[:2]" in text
    assert '"device_ids": [device.id for device in devices]' in text
    assert "_local_dsa_query_tuple4_exact" in text
    assert "local_output_width=2048" in text
    assert "full_indexer_layers=1" in text
    assert '"exact_chunks_per_local_owner"' not in text
    assert '"expected_dot_general_count": 9' in text
    assert '"expected_optimization_barrier_count": 3' in text
    assert "elementwise_exact" in text
