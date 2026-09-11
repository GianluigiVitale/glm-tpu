"""Production metadata and abstract observer shapes; no payload, placement or TPU."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def test_prepared_observer_binds_original_overlay_and_exact_owners_abstractly():
    source = r'''
from pathlib import Path
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.ws32_history_frontier_prepare import prepare as prepare_frontier
from scripts.greenfield.ws32_history_observer_prepare import prepare as prepare_observer
assert jax.default_backend() == 'cpu'
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
opened = Path.open
def checked(path, *args, **kwargs):
    assert path.suffix not in ('.safetensors', '.bin'), path
    return opened(path, *args, **kwargs)
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ('expert', 'feature'))
with patch.object(Path, 'open', checked), patch('jax.device_put', side_effect=AssertionError('concrete placement')):
    raw = prepare_frontier(mesh, repo=Path.cwd())
    p = prepare_observer(mesh, raw=raw, repo=Path.cwd())
    assert all(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(p.inputs))
    # The observer keeps the ORIGINAL representation: exact DSA plus StrategyND.
    assert p.config.geometry.num_layers == 7 and p.config.exact_dsa and p.config.strategy_nd_dense
    assert p.config.full_index_slots == (0, 1, 2, 6)
    assert len(p.overlay_tensor_names) == 12 and p.overlay_bytes_per_chip == 65691648
    assert len(p.raw_exact) == len(p.decoded_exact) == len(p.exact) == 4
    for value in p.exact:
        # Four distinct promoted owners per slot; nothing is budgeted as shared.
        assert len(value.wq_b_weight_aliases) == 4
        assert all(v.shape == (4096, 2048) for v in value.wq_b_weight_aliases)
    out = jax.eval_shape(p.program, *p.inputs)
    assert len(out.boundaries) == 7 and out.healthy.shape == ()
    for boundary in out.boundaries:
        assert boundary.update.shape == boundary.residual.shape == boundary.normalized_input.shape == (1, 6144)
        assert boundary.route_ids.shape == boundary.route_weights.shape == (1, 8)
        assert boundary.health.shape == (8, 4, 1)
    assert out.dsa.producer_layer_ids.shape == (4,)
    assert out.dsa.selected_positions.shape == out.dsa.selected_scores.shape == (4, 1, 2048)
    assert out.dsa.selected_valid_counts.shape == (4, 1)
    print('OBSERVER_ABSTRACT_PASS', flush=True)
'''
    flags = f"{os.environ.get('XLA_FLAGS', '').strip()} --xla_force_host_platform_device_count=32".strip()
    result = subprocess.run([sys.executable, "-c", source], text=True, cwd=ROOT,
        capture_output=True, timeout=600,
        env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OBSERVER_ABSTRACT_PASS" in result.stdout
